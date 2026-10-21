from copy import deepcopy
import unittest

from agent.errors import AgentError
from agent.store import is_better
from tests.helpers import Harness


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.h = Harness()

    def tearDown(self):
        self.h.close()

    def test_duplicate_plan_ignores_researcher_and_plan_id(self):
        first = self.h.store.reserve(self.h.plan(), self.h.code["sha256"])
        second = self.h.store.reserve(self.h.plan(plan_id="other", researcher_id="researcher-2"), self.h.code["sha256"])
        self.assertEqual(second["disposition"], "duplicate")
        self.assertEqual(first["experiment_id"], second["experiment_id"])

    def test_new_seed_is_distinct(self):
        first = self.h.store.reserve(self.h.plan(seed=42), self.h.code["sha256"])
        second = self.h.store.reserve(self.h.plan(seed=43), self.h.code["sha256"])
        self.assertNotEqual(first["experiment_id"], second["experiment_id"])

    def test_missing_parent_rejected(self):
        with self.assertRaises(AgentError) as caught:
            self.h.store.reserve(self.h.plan(parent_ids=["missing"]), self.h.code["sha256"])
        self.assertEqual(caught.exception.code, "PARENT_NOT_FOUND")

    def test_duplicate_and_stale_job_events(self):
        ident = self.h.store.reserve(self.h.plan(), self.h.code["sha256"])["experiment_id"]
        self.h.store.start(ident, {"lease_id": "lease-unit", "fencing_token": 7})
        event = {"event_id": "evt-unit", "job_id": "job-unit", "experiment_id": ident, "attempt_no": 1,
                 "lease_id": "lease-unit", "fencing_token": 7, "state": "succeeded", "output_artifact_ids": [],
                 "duration_seconds": 1, "peak_memory_mb": 10, "error": None}
        self.assertEqual(self.h.store.apply_event(event, "train")["experiment_state"], "trained")
        self.assertEqual(self.h.store.apply_event(event, "train")["disposition"], "already_applied")
        stale = {**event, "event_id": "evt-late", "fencing_token": 6}
        with self.assertRaises(AgentError) as caught:
            self.h.store.apply_event(stale, "train")
        self.assertEqual(caught.exception.code, "STALE_LEASE")

    def test_maximize_selection_does_not_regress(self):
        first = self.h.evaluation(0.8)
        self.h.evaluation(0.6)
        self.assertEqual(self.h.store.best(), first["experiment_id"])

    def test_comparison_rejects_different_protocol(self):
        metric = self.h.evaluation(0.5)["primary_metric"]
        with self.assertRaises(AgentError):
            is_better({**metric, "validation_id": "another"}, metric)

    def test_minimize_and_tie_policy(self):
        a = {"dataset_id": "d", "validation_id": "v", "metric_id": "rmse", "direction": "minimize", "fidelity": "full", "value": 2.0}
        self.assertTrue(is_better({**a, "value": 1.0}, a))
        self.assertFalse(is_better(a, a))
