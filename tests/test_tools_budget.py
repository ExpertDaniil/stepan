from pathlib import Path
from dataclasses import replace
import json
import os
import tempfile
import time
import unittest
from unittest.mock import patch

from agent.budget import Budget
from agent.config import Config
from agent.errors import AgentError
from tools.code import validate_code, write_code
from tools.data import read_table
from tools.data import profile_data
from tools.files import resolve_inputs
from tests.helpers import Harness


class ToolsBudgetTests(unittest.TestCase):
    def test_no_guessing_ambiguous_dataset_paths(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "task.md").write_text("Read train.csv and test.csv")
            for folder in ["a", "b"]:
                (root / folder).mkdir()
                (root / folder / "train.csv").write_text("x,y\n1,2\n")
            with self.assertRaises(AgentError) as caught:
                resolve_inputs(root / "task.md")
            self.assertEqual(caught.exception.code, "AMBIGUOUS_PATH")

    def test_archive_style_wrong_paths_fall_back_only_when_unique(self):
        h = Harness()
        try:
            h.task.write_text("Files: `not-present/participants/train.csv`, `not-present/participants/test.csv`.")
            _, paths = resolve_inputs(h.task)
            self.assertEqual(paths["train"], str(h.task.parent / "train.csv"))
        finally:
            h.close()

    def test_na_is_a_category_and_duplicate_header_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "x.csv"
            path.write_text("x,y\nNA,\nblue,2\n")
            frame = read_table(path)
            self.assertEqual(frame.iloc[0]["x"], "NA")
            self.assertTrue(frame["y"].isna().iloc[0])
            path.write_text("x,x\n1,2\n")
            with self.assertRaises(AgentError):
                read_table(path)

    def test_llm_profile_is_bounded_and_full_report_is_retained(self):
        h = Harness()
        try:
            result = profile_data({"candidate_paths": {"train": str(h.task.parent / "train.csv"), "test": str(h.task.parent / "test.csv")}, "max_report_bytes": 512}, h.artifacts)
            self.assertLessEqual(len(json.dumps(result["summary"], ensure_ascii=False).encode()), 512)
            self.assertIn("train_dtypes", h.artifacts.read_json(result["profile_artifact_id"]))
        finally:
            h.close()

    def test_generated_code_policy_and_path(self):
        with self.assertRaises(AgentError):
            validate_code("import os\ndef make_pipeline(task_spec, params):\n    return None\n")
        with self.assertRaises(AgentError):
            validate_code("def make_pipeline(task_spec, params):\n    return eval('1')\n")
        h = Harness()
        try:
            with self.assertRaises(AgentError):
                write_code({**h.code_payload, "filename": "../solution.py"}, h.artifacts)
        finally:
            h.close()

    def test_budget_caps_and_fencing(self):
        h = Harness()
        try:
            budget = Budget(h.config, time.monotonic(), h.log)
            req = {"cpu_threads": 3, "memory_limit_mb": 10240, "timeout_seconds": 60}
            def acquire(owner):
                return budget.lease({"operation": "acquire", "owner_id": owner, "phase": "train", "requested": req, "lease_id": None, "fencing_token": None})
            one, two = acquire("one"), acquire("two")
            self.assertTrue(one["granted"] and two["granted"])
            self.assertFalse(acquire("three")["granted"])
            self.assertEqual(budget.snapshot()["cpu_threads_available"], 0)
            bad = {"operation": "renew", "owner_id": "one", "phase": "train", "requested": req,
                   "lease_id": one["lease"]["lease_id"], "fencing_token": 999}
            with self.assertRaises(AgentError):
                budget.lease(bad)
        finally:
            h.close()

    def test_expired_budget_cannot_start_training(self):
        h = Harness()
        try:
            budget = Budget(h.config, time.monotonic() - 100, h.log)
            with self.assertRaises(AgentError) as caught:
                budget.lease({"operation": "acquire", "owner_id": "one", "phase": "train",
                              "requested": {"cpu_threads": 1, "memory_limit_mb": 512, "timeout_seconds": 2}, "lease_id": None, "fencing_token": None})
            self.assertEqual(caught.exception.code, "BUDGET_EXHAUSTED")
        finally:
            h.close()

    def test_invalid_config_is_not_silently_clipped(self):
        with patch.dict(os.environ, {"MAX_WORKERS": "9"}):
            with self.assertRaises(AgentError):
                Config.from_env()
        with patch.dict(os.environ, {"LOG_MODEL_IO": "perhaps"}):
            with self.assertRaises(AgentError):
                Config.from_env()
