import unittest

from agent.evaluation import Evaluator
from tests.helpers import Harness
from tools.data import read_table


class ValidationTests(unittest.TestCase):
    def test_temporal_holdout_preserves_complete_periods(self):
        h = Harness(with_periods=True)
        try:
            evaluator = Evaluator(h.spec, h.artifacts, h.store, h.log)
            result = evaluator.create_protocol({"task_spec_id": h.spec["task_spec_id"], "dataset_id": h.spec["dataset_id"],
                                                 "proposal": {"strategy": "temporal_holdout", "parameters": {"column": "period", "valid_fraction": 0.3}, "seed": 42, "reason": "synthetic ordered periods"}})
            protocol = evaluator.protocols[result["validation_id"]]
            frame = read_table(h.task.parent / "train.csv")
            train = frame.iloc[protocol["train_indices"]]["period"]
            valid = frame.iloc[protocol["valid_indices"]]["period"]
            self.assertLess(train.max(), valid.min())
            self.assertFalse(set(train) & set(valid))
        finally:
            h.close()

    def test_group_holdout_keeps_groups_disjoint(self):
        h = Harness()
        try:
            evaluator = Evaluator(h.spec, h.artifacts, h.store, h.log)
            result = evaluator.create_protocol({"task_spec_id": h.spec["task_spec_id"], "dataset_id": h.spec["dataset_id"],
                                                 "proposal": {"strategy": "group_holdout", "parameters": {"column": "category", "valid_fraction": 0.3}, "seed": 42, "reason": "synthetic groups"}})
            protocol = evaluator.protocols[result["validation_id"]]
            frame = read_table(h.task.parent / "train.csv").fillna("__MISSING__")
            self.assertFalse(set(frame.iloc[protocol["train_indices"]]["category"]) & set(frame.iloc[protocol["valid_indices"]]["category"]))
        finally:
            h.close()

    def test_duplicate_groups_never_straddle_the_holdout(self):
        import pandas as pd
        for strategy in ("stratified_holdout", "random_holdout"):
            h = Harness(with_duplicates=True)
            try:
                evaluator = Evaluator(h.spec, h.artifacts, h.store, h.log)
                result = evaluator.create_protocol({"task_spec_id": h.spec["task_spec_id"], "dataset_id": h.spec["dataset_id"],
                                                     "proposal": {"strategy": strategy, "parameters": {"valid_fraction": 0.2, "duplicate_groups": True},
                                                                  "seed": 42, "reason": "synthetic duplicates"}})
                protocol = evaluator.protocols[result["validation_id"]]
                frame = read_table(h.task.parent / "train.csv")
                key = pd.util.hash_pandas_object(frame[["x_numeric", "z_numeric", "category"]].astype(str), index=False)
                self.assertFalse(set(key.iloc[protocol["train_indices"]]) & set(key.iloc[protocol["valid_indices"]]), strategy)
                self.assertTrue(0.1 <= len(protocol["valid_indices"])/len(frame) <= 0.35, strategy)
            finally:
                h.close()

