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

    def test_temporal_holdout_scores_rolling_blocks_on_earlier_rows_only(self):
        import json
        h = Harness(with_periods=True)
        try:
            evaluator = Evaluator(h.spec, h.artifacts, h.store, h.log)
            result = evaluator.create_protocol({"task_spec_id": h.spec["task_spec_id"], "dataset_id": h.spec["dataset_id"],
                                                 "proposal": {"strategy": "temporal_holdout", "parameters": {"column": "period", "valid_fraction": 0.3}, "seed": 42, "reason": "synthetic ordered periods"}})
            protocol = evaluator.protocols[result["validation_id"]]
            h.schemas.validate(protocol, 'RProtocol')
            self.assertEqual(protocol['fold_plan'], [
                {'valid_periods': {'first': '4', 'last': '4', 'count': 1}, 'fit_rows': 60, 'valid_rows': 15},
                {'valid_periods': {'first': '5', 'last': '5', 'count': 1}, 'fit_rows': 75, 'valid_rows': 15}])
            self.assertEqual(protocol['confirmation_indices'], [])  # 15-row periods are too small to split
            folds = json.loads(h.artifacts.safe_path('validation/'+result['validation_id']+'.folds.json').read_text())
            period = read_table(h.task.parent / "train.csv")["period"]
            for fold in folds:
                self.assertLess(period.iloc[fold['fit']].max(), period.iloc[fold['valid']].min())
            self.assertEqual(sorted(i for fold in folds for i in fold['valid']), protocol['valid_indices'])
        finally:
            h.close()

    def test_rolling_confirmation_holds_whole_latest_periods_out_of_every_fit(self):
        import numpy as np
        import pandas as pd
        from agent.predictions import validation_columns
        h = Harness('regression', with_periods=True, rows=480)
        try:
            evaluator = Evaluator(h.spec, h.artifacts, h.store, h.log)
            result = evaluator.create_protocol({"task_spec_id": h.spec["task_spec_id"], "dataset_id": h.spec["dataset_id"],
                                                 "proposal": {"strategy": "temporal_holdout", "parameters": {"column": "period", "valid_fraction": 0.3}, "seed": 42, "reason": "synthetic ordered periods"}})
            protocol = evaluator.protocols[result["validation_id"]]
            h.schemas.validate(protocol, 'RProtocol')
            frame = read_table(h.task.parent / "train.csv")
            selection, confirmation = protocol['selection_indices'], protocol['confirmation_indices']
            self.assertIsNone(protocol['confirmation_unavailable_reason'])
            self.assertFalse(set(selection) & set(confirmation))
            self.assertEqual(sorted(selection+confirmation), protocol['valid_indices'])
            self.assertLess(frame.loc[selection, 'period'].max(), frame.loc[confirmation, 'period'].min())
            import json
            folds = json.loads(h.artifacts.safe_path('validation/'+result['validation_id']+'.folds.json').read_text())
            self.assertTrue(all(not set(confirmation).intersection(fold['fit']) for fold in folds))
            # Perturbing only withheld labels cannot influence any rolling fit,
            # selection prediction or selection diagnostic used by the search.
            target = frame[h.spec['target_column']].to_numpy(float)
            changed = target.copy();changed[confirmation] += 10000
            for fold in folds:
                np.testing.assert_array_equal(target[fold['fit']], changed[fold['fit']])
            np.testing.assert_array_equal(target[selection], changed[selection])
            # Research scores read selection rows only: wrong withheld predictions cannot move them.
            plan = h.plan(task_spec_id=h.spec['task_spec_id'], dataset_id=h.spec['dataset_id'], validation_id=result['validation_id'],
                          recipe_id='linear-regressor-v1', model={'family': 'linear', 'hyperparameters': {'alpha': 1}})
            eid = h.store.reserve(plan, h.code['sha256'])['experiment_id']
            with h.store.db:
                h.store.db.execute("UPDATE experiments SET state='trained' WHERE experiment_id=?", (eid,))
            truth = frame.loc[protocol['valid_indices'], h.spec['target_column']].to_numpy(float)
            values = np.where(np.isin(protocol['valid_indices'], confirmation), truth+1000, truth)
            path = h.run/'rolling-predictions.csv'
            pd.DataFrame({'__row_index': protocol['valid_indices'], validation_columns(h.spec)[0]: values}).to_csv(path, index=False)
            art = h.artifacts.register(path, 'validation_predictions', 'text/csv')
            ev = evaluator.evaluate({'experiment_id': eid, 'task_spec_id': h.spec['task_spec_id'], 'dataset_id': h.spec['dataset_id'],
                'validation_id': result['validation_id'], 'predictions_artifact_id': art['artifact_id'], 'fidelity': 'full'})['evaluation']
            self.assertAlmostEqual(ev['primary_metric']['value'], 0.0, places=9)
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

    def test_temporal_confirmation_labels_cannot_change_selection_fits_in_either_task(self):
        import json
        import numpy as np
        from sklearn.dummy import DummyClassifier, DummyRegressor
        for kind in ['regression', 'classification']:
            h = Harness(kind, with_periods=True, rows=480)
            try:
                evaluator = Evaluator(h.spec, h.artifacts, h.store, h.log)
                result = evaluator.create_protocol({'task_spec_id': h.spec['task_spec_id'], 'dataset_id': h.spec['dataset_id'],
                    'proposal': {'strategy': 'temporal_holdout', 'parameters': {'column': 'period', 'valid_fraction': .3},
                                 'seed': 42, 'reason': 'Synthetic forward prediction'}})
                protocol = evaluator.protocols[result['validation_id']]
                folds = json.loads(h.artifacts.safe_path('validation/'+result['validation_id']+'.folds.json').read_text())
                held = protocol['confirmation_indices'];self.assertTrue(held)
                data = read_table(h.task.parent/'train.csv')
                original = data[h.spec['target_column']].to_numpy()
                changed = original.copy()
                if kind == 'regression':changed[held] += 10000
                else:
                    labels = sorted(set(original))
                    changed[held] = [labels[(labels.index(label)+1)%len(labels)] for label in original[held]]
                self.assertFalse(np.array_equal(original[held], changed[held]))
                model_type = DummyRegressor if kind == 'regression' else DummyClassifier
                for fold in folds:
                    self.assertFalse(set(held).intersection(fold['fit']))
                    x = data[['period']]
                    before = model_type().fit(x.iloc[fold['fit']], original[fold['fit']]).predict(x.iloc[fold['valid']])
                    after = model_type().fit(x.iloc[fold['fit']], changed[fold['fit']]).predict(x.iloc[fold['valid']])
                    np.testing.assert_array_equal(before, after)
                self.assertEqual(len(folds), 2)  # Existing predictions are reused, no extra confirmation fit.
            finally:h.close()

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

