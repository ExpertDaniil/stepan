from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import MethodType, SimpleNamespace
import io
import json
import os
import tempfile
import time
import unittest
from contextlib import redirect_stderr
from unittest.mock import patch

from agent.budget import Budget
from agent.catalog import Catalog
from agent.config import Config
from agent.duration import DurationModel
from agent.errors import AgentError
from agent.graph.controller import research_batch, train
from agent.local_search import LocalSearch
from agent.orchestrator import Orchestrator
from agent.runtime import main
from agent.llm.gateway import MockTransport
from scripts.make_demo import make_demo
from tests.graph_helpers import GraphHarness


def cost_plan(family="catboost", **params):
    recipe = {"catboost": "catboost-regressor-v1", "linear": "linear-regressor-v1",
              "hist_gradient": "hist-gradient-regressor-v1", "random_forest": "random-forest-regressor-v1"}[family]
    return {"recipe_id": recipe, "model": {"family": family, "hyperparameters": params},
            "resources": {"cpu_threads": 3}}


class DurationTests(unittest.TestCase):
    def model(self):
        model = DurationModel(Config(time_budget_seconds=90))
        model.rows = {"train": 800, "refit": 1000}
        model.features = 10
        return model

    def test_iterations_depth_and_full_refit_scale_from_real_family_history(self):
        model = self.model()
        small = cost_plan(iterations=100, depth=4)
        model.observe(small, "train", 12, fit_seconds=10)
        self.assertGreater(model.estimate(cost_plan(iterations=200, depth=4)), model.estimate(small))
        self.assertGreater(model.estimate(cost_plan(iterations=100, depth=5)), model.estimate(small))
        self.assertGreater(model.estimate(small, "refit"), model.estimate(small, "train"))
        self.assertLess(model.estimate(cost_plan("linear", alpha=1)), model.estimate(small))

    def test_hgb_leaves_and_forest_trees_change_the_estimate(self):
        model = self.model()
        for family, params, heavier in [
            ("hist_gradient", {"max_iter": 100, "max_leaf_nodes": 15}, {"max_iter": 100, "max_leaf_nodes": 31}),
            ("random_forest", {"n_estimators": 100, "max_depth": 4}, {"n_estimators": 200, "max_depth": 8}),
        ]:
            a, b = cost_plan(family, **params), cost_plan(family, **heavier)
            model.observe(a, "train", 20, fit_seconds=18)
            self.assertGreater(model.estimate(b), model.estimate(a))

    def test_fixed_overhead_does_not_disappear_for_a_small_trial(self):
        model = self.model()
        model.observe(cost_plan(iterations=100, depth=4), "train", 12, fit_seconds=2)
        self.assertGreaterEqual(model.estimate(cost_plan(iterations=1, depth=4)), 10)

    def test_small_unseen_family_still_reserves_process_startup(self):
        model = self.model()
        model.defaults["random-forest-regressor-v1"] = {"n_estimators": 100, "max_depth": 16}
        self.assertGreater(model.estimate(cost_plan("random_forest", n_estimators=1, max_depth=2)), 2)

    def test_unseen_family_uses_measured_baseline_startup_instead_of_two_seconds(self):
        model=self.model()
        dummy={'recipe_id':'dummy-regressor-v1','model':{'family':'dummy','hyperparameters':{}},'resources':{'cpu_threads':3}}
        model.observe(dummy,'train',12,fit_seconds=2)
        self.assertGreaterEqual(model.estimate(cost_plan('linear',alpha=1)),13.5)

    def test_timeout_is_a_lower_bound_not_a_completed_fast_fit(self):
        model = self.model()
        plan = cost_plan(iterations=100, depth=4)
        model.observe(plan, "train", 20, state="timed_out")
        self.assertGreater(model.estimate(plan), 20)
        self.assertLess(model.estimate(cost_plan(iterations=25, depth=4)), model.estimate(plan))

    def test_new_observations_update_estimates_and_feature_cost(self):
        model = self.model()
        plan = cost_plan(iterations=100, depth=4)
        model.observe(plan, "train", 5, fit_seconds=4)
        before = model.estimate(plan)
        model.observe(plan, "train", 20, fit_seconds=19)
        self.assertGreater(model.estimate(plan), before)
        self.assertGreater(model.estimate(cost_plan(iterations=100, depth=4, numeric_interactions=True)), model.estimate(plan))


class TimeAdmissionTests(unittest.TestCase):
    def test_half_second_gap_is_rejected_consistently(self):
        budget = Budget(Config(time_budget_seconds=90), time.monotonic(), None)
        budget.deadline = time.monotonic() + budget.reserve_seconds() + .5
        self.assertFalse(budget.snapshot()["search_allowed"])
        with self.assertRaises(AgentError):
            budget.lease({"operation": "acquire", "owner_id": "test", "phase": "train",
                          "requested": {"cpu_threads": 1, "memory_limit_mb": 256, "timeout_seconds": 1},
                          "lease_id": None, "fencing_token": None})

    def test_candidate_must_leave_time_for_its_full_refit_and_output(self):
        budget = Budget(Config(time_budget_seconds=90), time.monotonic(), None)
        plan = cost_plan(iterations=100, depth=4)
        budget.duration.rows = {"train": 80, "refit": 100}
        budget.duration.observe(plan, "train", 10, fit_seconds=10)
        budget.deadline = time.monotonic() + 20
        self.assertFalse(budget.can_train(plan)["allowed"])
        self.assertTrue(budget.can_train(cost_plan("linear", alpha=1))["allowed"])

    def test_heavy_final_model_reserve_is_not_truncated_to_35_percent(self):
        budget = Budget(Config(time_budget_seconds=90), time.monotonic(), None)
        plan = cost_plan(iterations=100, depth=4)
        budget.duration.observe(plan, "train", 40, fit_seconds=40)
        budget.set_final_plan(plan)
        self.assertGreater(budget.reserve_seconds(), 90 * .35)

    def test_admission_rechecks_time_after_dispatch_work(self):
        budget = Budget(Config(time_budget_seconds=90), time.monotonic(), None)
        plan = cost_plan("linear", alpha=1)
        self.assertTrue(budget.admit("trial", plan)["allowed"])
        budget.deadline = time.monotonic() + budget.candidate_reserve(plan) + .5
        with self.assertRaises(AgentError):
            budget.lease({"operation": "acquire", "owner_id": "trial", "phase": "train",
                          "requested": {"cpu_threads": 1, "memory_limit_mb": 256, "timeout_seconds": 1},
                          "lease_id": None, "fencing_token": None})
        self.assertFalse(budget.leases)

    def test_race_at_graph_lease_is_cancelled_without_degrading_the_run(self):
        h = GraphHarness()
        try:
            controller = h.controller
            controller.graph = h.g
            controller.degraded = False
            controller.admit_training = lambda plan, eid: controller.budget.admit(eid, plan)["allowed"]
            controller.lease = lambda *args: (_ for _ in ()).throw(AgentError("BUDGET_EXHAUSTED", "Time changed"))
            controller.wait_jobs = lambda jobs: {}
            ticket = h.issue()
            plan = h.proposal(ticket)["proposal"]["plan"]
            self.assertEqual(train(controller, [plan], [ticket]), 0)
            self.assertFalse(controller.degraded)
            row = h.g.db.execute("SELECT state FROM experiments").fetchone()
            self.assertEqual(row[0], "cancelled")
            self.assertNotEqual(h.g.order(ticket["work_order"]["work_order_id"])["state"], "submitted")
        finally:
            h.close()


class LocalSearchTests(unittest.TestCase):
    def setUp(self):
        self.h = GraphHarness()
        self.c = self.h.controller
        self.c.resources = lambda: {"cpu_threads": 1, "memory_limit_mb": 1024, "timeout_seconds": 30}
        self.search = LocalSearch(self.c)

    def tearDown(self):
        self.h.close()

    def test_failed_reserved_configuration_is_not_retried(self):
        ticket = self.h.issue()
        reply, payload = self.h.submit(ticket, {"alpha": 1})
        self.c.store.fail_start(reply["lead_submission"]["experiment_id"], AgentError("FAILED", "Synthetic failure"))
        plan = payload["proposal"]["plan"]
        self.search.start_series("researcher-1", self.h.root, plan)
        recipe, params = self.search.next_choice("researcher-1", self.h.root, plan)
        self.assertNotEqual(self.search.key(recipe, params), self.search.key(plan["recipe_id"], plan["model"]["hyperparameters"]))

    def test_required_and_forbidden_components_survive_local_variation(self):
        cap = "feature.target_transform"
        for branch, required, forbidden, expected in [("accept", [cap], [], "shifted_log1p"), ("reject", [], [cap], "identity")]:
            recipe, params = self.search.next_choice("researcher-1", branch, constraints={"required_capability_ids": required, "forbidden_capability_ids": forbidden})
            self.assertEqual(params.get("target_transform", "identity"), expected)
            self.c.catalog.normalize(recipe, params)

    def test_series_are_small_and_do_not_cross_branches(self):
        ticket = self.h.issue()
        plan = self.h.proposal(ticket)["proposal"]["plan"]
        self.search.start_series("researcher-1", "a", plan)
        self.assertTrue(self.search.has_series("researcher-1", "a"))
        self.assertFalse(self.search.has_series("researcher-1", "b"))
        self.assertEqual(self.search.series[("a", "researcher-1")][0], self.c.config.research_series_size - 1)

    def test_no_candidate_is_returned_when_finalization_has_consumed_the_time(self):
        self.c.budget.deadline = time.monotonic() + self.c.budget.output_reserve()
        self.assertIsNone(self.search.next_choice("researcher-1", self.h.root))


class ProposalReplacementTests(unittest.TestCase):
    def test_invalid_and_stop_replies_get_a_checked_local_replacement(self):
        for invalid in (False, True):
            with self.subTest(invalid=invalid):
                h = GraphHarness()
                try:
                    c = h.controller
                    c.graph, c.degraded = h.g, False
                    c.resources = lambda: {"cpu_threads": 1, "memory_limit_mb": 1024, "timeout_seconds": 30}
                    c.call = lambda cid, payload, fn, key=None: fn(payload)
                    c.warmup_plan = MethodType(Orchestrator.warmup_plan, c)
                    c.local_plan = MethodType(Orchestrator.local_plan, c)
                    c.local_search = LocalSearch(c)
                    trained = []

                    def propose(payload):
                        if invalid:
                            raise AgentError("INVALID_LLM_OUTPUT", "Malformed synthetic reply")
                        return {"decision": "stop", "plan": None}

                    def capture(plans, tickets):
                        for plan, ticket in zip(plans, tickets):
                            c.catalog.validate_plan(plan, c.spec, c.validation["validation_id"], c.artifacts)
                            self.assertEqual(plan["parent_ids"], [])
                            self.assertEqual(plan["researcher_id"], ticket["work_order"]["researcher_id"])
                            trained.append((plan, ticket))
                            h.g.abandon(ticket["work_order"]["work_order_id"], "Fixture captured the checked plan")
                        return len(plans)

                    c.researcher = SimpleNamespace(propose=propose)
                    c.run_training = capture
                    ticket = h.issue()
                    accepted, had_plans = research_batch(c, [], [], [ticket])
                    self.assertEqual(accepted, 1)
                    self.assertTrue(had_plans)
                    self.assertEqual(len(trained), 1)
                    self.assertEqual(c.degraded, invalid)
                    if invalid:
                        self.assertNotEqual(ticket["work_order"]["work_order_id"], trained[0][1]["work_order"]["work_order_id"])
                finally:
                    h.close()


class ContinuedSearchTests(unittest.TestCase):
    def run_case(self, graph, calls, duplicate=False, fail_once=False):
        from agent.runner import Runner
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            task = make_demo(root / "task", "regression", rows=90, test_rows=12)
            original_update = Orchestrator.update_final_reserve
            original_catalog = Catalog.__init__
            original_api = MockTransport.complete
            original_start = Runner.start_train
            failures = []

            def small_catalog(catalog, *args):
                original_catalog(catalog, *args)
                catalog.recipes = {k: v for k, v in catalog.recipes.items() if v["family"] in {"dummy", "linear"}}

            def stop_after_evidence(controller):
                original_update(controller)
                n = controller.store.db.execute("SELECT count(*) FROM experiments WHERE attempt_no>0 AND json_extract(plan,'$.model.family')<>'dummy'").fetchone()[0]
                if n >= 14 + len(failures):
                    controller.budget.deadline = time.monotonic() + controller.budget.reserve_seconds() + .5

            def repeat(transport, request, timeout, context):
                if context["metadata"]["schema_id"] == "ResearchChoice-v1" and duplicate:
                    payload = {"recipe_id": "linear-regressor-v1", "hyperparameters": {"alpha": 1}, "hypothesis": "Synthetic repeated choice", "expected_effect": "Measure"}
                    return {"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": json.dumps(payload)}}]}
                return original_api(transport, request, timeout, context)

            def start(runner, payload, spec, experiment):
                if fail_once and experiment["plan"]["model"]["family"] == "linear" and not failures:
                    failures.append(experiment["fingerprint"])
                    raise AgentError("SPAWN_FAILED", "One injected start failure")
                return original_start(runner, payload, spec, experiment)

            # Stop after fourteen measured fits above; this ceiling must leave
            # enough room for process startup on slower CI machines.
            settings = {"LLM_MODE": "mock", "GRAPH_ENABLED": str(int(graph)), "TIME_BUDGET_SECONDS": "600",
                        "RESEARCH_MODE": "catalog",
                        "LLM_MAX_CALLS": str(calls), "CATALOG_WARMUP_EXPERIMENTS": "0", "MAX_RESEARCH_EXPERIMENTS": "1",
                        "RESEARCH_SERIES_SIZE": "1" if duplicate else "4", "MAX_WORKERS": "1", "GRAPH_PEERS": "1",
                        "GRAPH_MIN_FREE_DISK_BYTES": "0", "LOG_LEVEL": "ERROR"}
            with patch.dict(os.environ, settings), patch.object(Catalog, "__init__", small_catalog), \
                    patch.object(Orchestrator, "update_final_reserve", stop_after_evidence), patch.object(MockTransport, "complete", repeat), \
                    patch.object(Runner, "start_train", start), redirect_stderr(io.StringIO()) as stderr:
                code = main([str(task), "--work-dir", str(root / "runs")])
            self.assertEqual(code, 0, stderr.getvalue()[-3000:])
            run = next((root / "runs").glob("run-*"))
            events = [json.loads(line) for line in (run / "events.jsonl").read_text().splitlines()]
            self.assertTrue((task.parent / "submission.csv").is_file())
            fits = [e for e in events if e["event"] == "job.started" and e["phase"] == "train" and e["recipe_id"].startswith("linear-")]
            self.assertGreaterEqual(len(fits), 14)
            finish = next(e for e in events if e["event"] == "search.finished")
            self.assertNotIn(finish["stop_reason"], {"experiment_limit", "round_limit", "llm_call_limit", "llm_output_budget"})
            self.assertTrue(any(e["event"] == "search.local_trial" for e in events))
            if calls == 2:
                self.assertEqual(sum(e["event"] == "llm.call_started" for e in events), 2)
            if duplicate and graph:
                self.assertTrue(any(e["event"] == "search.proposal_replaced" and e.get("reason") == "duplicate" for e in events))
            if fail_once:
                self.assertEqual(len(failures), 1)
            if not fail_once:
                self.assertEqual(events[-1]["state"], "completed",
                    json.dumps([e for e in events if e['level'] in {'ERROR', 'WARNING'}])[-12000:])

    def test_more_than_twelve_real_fits_continue_after_llm_quota(self):
        self.run_case(True, 2)

    def test_duplicate_llm_proposals_are_replaced_without_retraining_the_same_plan(self):
        self.run_case(True, 32, duplicate=True)

    def test_failed_training_is_replaced_in_the_next_related_trial(self):
        self.run_case(True, 2, fail_once=True)

    def test_legacy_controller_also_uses_time_and_local_trials(self):
        self.run_case(False, 2)
