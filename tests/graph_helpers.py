from copy import deepcopy
from dataclasses import replace
import time
from types import SimpleNamespace
import uuid

from agent.budget import Budget
from agent.catalog import Catalog, factory_source
from agent.context import ContextBuilder, demo_metadata
from agent.evaluation import Evaluator
from agent.graph.controller import lineage_for
from agent.graph.service import GraphService
from tests.helpers import Harness
from tools.code import write_code


class GraphHarness:
    """State-machine fixture: scores are explicitly injected, never advertised as ML validation."""
    def __init__(self, kind="regression", **config):
        self.h = Harness(kind)
        h = self.h
        defaults = dict(graph_min_free_disk_bytes=0, graph_plateau_seconds=0, graph_refinement_enabled=False,
                        feature_research_enabled=False)
        defaults.update(config)
        cfg = replace(h.config, **defaults)
        budget = Budget(cfg, time.monotonic(), h.log)
        catalog = Catalog(h.spec["task_kind"], cfg)
        evaluator = Evaluator(h.spec, h.artifacts, h.store, h.log)
        validation = evaluator.create_protocol({"dataset_id": h.spec["dataset_id"], "task_spec_id": h.spec["task_spec_id"],
            "proposal": {"strategy": "random_holdout" if kind == "regression" else "stratified_holdout", "parameters": {"valid_fraction": .2}, "seed": 42, "reason": "Synthetic graph state fixture"}})
        context = ContextBuilder(h.spec, h.task.read_text(), h.profile["summary"], catalog, h.artifacts, h.log, demo_metadata(h.task.read_text()))
        context.validation_id = validation["validation_id"]
        context.validation_sha256 = validation["protocol_sha256"]
        context.store = h.store
        controller = SimpleNamespace(config=cfg, bus=h.bus, budget=budget, artifacts=h.artifacts, store=h.store,
            runner=SimpleNamespace(active=lambda: [], jobs={}), evaluator=evaluator, catalog=catalog,
            spec=h.spec, validation=validation, log=h.log, context=context)
        self.controller = controller
        self.g = GraphService(controller, h.task.parent)
        self.g.initialize_defaults()
        self.root = self.g.metadata("root_branch_id")
        self.token = 0

    def issue(self, peer="researcher-1", branch=None, donors=None, action="improve"):
        branch = branch or self.root;b = self.g.branch(branch)
        return self.g.invoke("G03", {"branch_id": branch, "expected_branch_revision": b["branch_revision"], "researcher_id": peer,
            "action": action, "allowed_donor_experiment_ids": donors or [], "tools_catalog_version": "tools-v1", "ttl_seconds": 60}, key=uuid.uuid4().hex)

    def proposal(self, ticket, params=None, recipe=None, seed=42, source=None):
        g = self.g;work = ticket["work_order"]
        recipe = recipe or ("linear-regressor-v1" if g.spec["task_kind"] == "regression" else "linear-classifier-v1")
        params = g.catalog.normalize(recipe, params or {})
        code = write_code({"researcher_id": work["researcher_id"], "filename": "graph.py", "content": source or factory_source(recipe), "declared_entrypoint": "make_pipeline"}, g.artifacts)["artifact"]
        lineage = lineage_for(work)
        plan = {"plan_id": "plan-"+uuid.uuid4().hex[:12], "researcher_id": work["researcher_id"],
            "parent_ids": list(dict.fromkeys(([work["base_experiment_id"]] if work["base_experiment_id"] else [])+lineage["idea_donor_experiment_ids"])),
            "hypothesis": "Synthetic state-machine proposal", "expected_effect": "Verify graph contracts, not model quality",
            "recipe_id": recipe, "task_spec_id": g.spec["task_spec_id"], "dataset_id": g.spec["dataset_id"], "validation_id": g.validation["validation_id"],
            "seed": seed, "fidelity": "full", "model": {"family": g.catalog.recipes[recipe]["family"], "hyperparameters": params},
            "feature_steps": g.catalog.feature_steps(params), "code_artifact_id": code["artifact_id"],
            "resources": {"cpu_threads": 1, "memory_limit_mb": 1024, "timeout_seconds": 30}}
        return {"work_order_id": work["work_order_id"], "proposal": {"proposal_id": "proposal-"+uuid.uuid4().hex[:12],
            "researcher_id": work["researcher_id"], "plan": plan, "decision": "propose", "stop_reason": None},
            "manifest": g.constructor.build(plan), "lineage": lineage}

    def submit(self, ticket, params=None, **kw):
        payload = self.proposal(ticket, params, **kw)
        return self.g.invoke("G04", payload, key=uuid.uuid4().hex), payload

    def train_event(self, reply, state="succeeded", fence=None):
        g = self.g;eid = reply["lead_submission"]["experiment_id"]
        exp = g.store.get(eid)
        if exp["state"] == "queued":
            self.token += 1
            g.store.start(eid, {"lease_id": "lease-"+str(self.token), "fencing_token": self.token})
        exp = g.store.get(eid)
        return {"event_id": "event-"+uuid.uuid4().hex, "job_id": "job-"+eid, "experiment_id": eid, "attempt_no": exp["attempt_no"],
            "lease_id": exp["lease_id"], "fencing_token": exp["fencing_token"] if fence is None else fence, "state": state,
            "output_artifact_ids": [], "duration_seconds": 1.0, "peak_memory_mb": 10.0, "error": None}

    def evaluate(self, reply, score):
        g = self.g;eid = reply["lead_submission"]["experiment_id"]
        if g.store.get(eid)["state"] == "evaluated":
            return g.store.experiment_evaluation(eid)
        event = self.train_event(reply)
        g.store.apply_event(event, "train")
        diag = g.artifacts.write_json("diagnostics/"+eid+".json", {"rows": 18, "metric_id": g.spec["metric"]["metric_id"], "score": score}, "diagnostics")
        ev = {"evaluation_id": "eval-"+uuid.uuid4().hex[:12], "experiment_id": eid, "valid": True,
            "primary_metric": {"metric_id": g.spec["metric"]["metric_id"], "direction": g.spec["metric"]["direction"], "value": score,
                "validation_id": g.validation["validation_id"], "dataset_id": g.spec["dataset_id"], "fidelity": "full"},
            "diagnostics_artifact_id": diag["artifact_id"], "violations": []}
        g.store.save_evaluation(ev)
        return ev

    def commit(self, reply, score):
        ev = self.evaluate(reply, score)
        return self.g.invoke("G05", {"experiment_id": ev["experiment_id"], "evaluation_id": ev["evaluation_id"], "selection_policy_id": "same-protocol-primary-v1"}, key=uuid.uuid4().hex)

    def result(self, peer, params, score, **kw):
        ticket = self.issue(peer, **kw)
        reply, payload = self.submit(ticket, params)
        result = self.commit(reply, score)
        return ticket, reply, payload, result

    def make_fork(self):
        g = self.g
        self.result("researcher-1", {"alpha": 1}, 10)
        self.result("researcher-2", {"alpha": 2}, 12)
        ticket, source, payload, _ = self.result("researcher-1", {"alpha": 1, "target_transform": "shifted_log1p"}, 8)
        source_id = source["lead_submission"]["experiment_id"]
        cap = "feature.target_transform"
        c = next(c for c in payload["manifest"]["core"]["components"] if c["capability_id"] == cap)
        patch = {"patch_version": "transfer-1.0", "idea_id": "idea-"+uuid.uuid4().hex[:16],
            "supported_recipe_ids": [r for r, v in g.catalog.recipes.items() if "target_transform" in v["parameters"]],
            "required_base_capability_ids": [], "operations": [{"operation": "add_component", "target_component_id": None, "component": g.constructor.external_component(c)}],
            "expected_added_capability_ids": [cap], "expected_removed_capability_ids": []}
        _, pilot, _, _ = self.result("researcher-2", {"alpha": 2, "target_transform": "shifted_log1p"}, 9, donors=[source_id], action="transfer")
        art = g.files.put_json(patch, "transfer_patch")
        evidence = {"title": "Synthetic controlled feature", "hypothesis": "State machine evidence fixture", "source_experiment_id": source_id,
            "capability_ids": [cap], "fork_checkpoint_id": ticket["work_order"]["base_team_checkpoint_id"],
            "baseline_experiment_ids": [ticket["work_order"]["base_experiment_id"]], "candidate_experiment_ids": [source_id],
            "transfer_pilot_experiment_ids": [pilot["lead_submission"]["experiment_id"]], "transfer_patch_artifact_id": art["artifact_id"]}
        idea = g.invoke("G06", evidence, key=uuid.uuid4().hex)["idea"]
        fork = {"parent_branch_id": self.root, "expected_branch_revision": g.branch(self.root)["branch_revision"],
                "checkpoint_id": evidence["fork_checkpoint_id"], "idea_id": idea["idea_id"]}
        return fork, evidence, idea

    def close(self):
        self.h.close()
