from unittest.mock import patch
import unittest

from agent.errors import AgentError
from agent.submission import Submission
from tests.helpers import Harness


class SubmissionTests(unittest.TestCase):
    def setUp(self):
        self.h = Harness()
        self.submission = Submission(self.h.spec, self.h.artifacts, self.h.store, self.h.log)

    def tearDown(self):
        self.h.close()

    def payload(self, evaluation, label="A", reverse=False):
        artifact = self.h.candidate(label, reverse)
        report = self.submission.validate_output({"submission_job_id": "unit-subjob", "task_spec_id": self.h.spec["task_spec_id"],
                                                 "candidate_artifact_id": artifact["artifact_id"],
                                                 "reference_test_artifact_id": self.h.spec["test_artifact_id"]})
        champion = self.h.store.champion()
        return {"experiment_id": evaluation["experiment_id"], "evaluation_id": evaluation["evaluation_id"],
                "candidate_artifact_id": artifact["artifact_id"], "validation_report_id": report["validation_report_id"],
                "candidate_sha256": report["candidate_sha256"], "expected_champion_version": champion["champion_version"] if champion else 0}

    def test_leading_zero_ids_and_valid_output(self):
        ev = self.h.evaluation(0.5)
        result = self.submission.publish(self.payload(ev))
        self.assertEqual(result["disposition"], "published")
        text = (self.h.task.parent / "submission.csv").read_text()
        self.assertIn("000090,A", text)
        self.assertEqual(self.submission.require_output()["champion_version"], 1)

    def test_output_order_is_checked(self):
        with self.assertRaises(AgentError) as caught:
            self.payload(self.h.evaluation(0.5), reverse=True)
        self.assertEqual(caught.exception.code, "ID_ORDER_MISMATCH")

    def test_changed_candidate_cannot_be_published(self):
        payload = self.payload(self.h.evaluation(0.5))
        path = self.h.artifacts.resolve(payload["candidate_artifact_id"])
        path.write_text(path.read_text().replace(",A", ",B"))
        with self.assertRaises(AgentError) as caught:
            self.submission.publish(payload)
        self.assertEqual(caught.exception.code, "CANDIDATE_CHANGED")

    def test_weaker_late_candidate_does_not_replace_file(self):
        first = self.submission.publish(self.payload(self.h.evaluation(0.8), "A"))
        result = self.submission.publish(self.payload(self.h.evaluation(0.6), "B"))
        self.assertEqual(result["disposition"], "skipped_not_better")
        self.assertEqual(result["champion"], first["champion"])

    def test_better_candidate_with_identical_test_predictions_updates_metadata(self):
        first = self.submission.publish(self.payload(self.h.evaluation(0.5), "A"))
        second = self.submission.publish(self.payload(self.h.evaluation(0.8), "A"))
        self.assertEqual(first["champion"]["output_sha256"], second["champion"]["output_sha256"])
        self.assertEqual(second["champion"]["champion_version"], 2)
        self.assertEqual(second["champion"]["primary_metric"]["value"], 0.8)

    def test_version_conflict(self):
        first = self.payload(self.h.evaluation(0.5))
        self.submission.publish(first)
        second = self.payload(self.h.evaluation(0.8), "B")
        second["expected_champion_version"] = 0
        with self.assertRaises(AgentError) as caught:
            self.submission.publish(second)
        self.assertEqual(caught.exception.code, "VERSION_CONFLICT")

    def test_journal_recovers_replace_before_database_commit(self):
        payload = self.payload(self.h.evaluation(0.8))
        original = self.h.store.finish_publication
        with patch.object(self.h.store, "finish_publication", side_effect=OSError("simulated crash after replace")):
            with self.assertRaises(AgentError):
                self.submission.publish(payload)
        self.assertIsNone(self.h.store.champion())
        self.assertTrue((self.h.task.parent / "submission.csv").exists())
        self.submission.recover()
        self.assertEqual(self.h.store.champion()["primary_metric"]["value"], 0.8)
