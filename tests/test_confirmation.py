"""Withheld-label and one-shot final selection checks; scores are synthetic fixtures."""
from copy import deepcopy
from dataclasses import replace
import threading
import time
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd

from agent.budget import Budget
from agent.confirmation import finalize, partition
from agent.evaluation import Evaluator
from agent.submission import Submission
from tests.helpers import Harness


class FinalConfirmationTests(unittest.TestCase):
    def test_confirmation_errors_do_not_enter_research_scores_and_choose_finalist_only_once(self):
        h = Harness('regression')
        try:
            evaluator = Evaluator(h.spec, h.artifacts, h.store, h.log)
            split = evaluator.create_protocol({'task_spec_id': h.spec['task_spec_id'], 'dataset_id': h.spec['dataset_id'],
                'proposal': {'strategy': 'random_holdout', 'parameters': {'valid_fraction': .5}, 'seed': 42, 'reason': 'Synthetic final comparison'}})
            protocol = evaluator.protocols[split['validation_id']]
            h.schemas.validate(protocol, 'RProtocol')
            self.assertTrue(protocol['confirmation_indices'])
            self.assertFalse(set(protocol['selection_indices']) & set(protocol['confirmation_indices']))
            self.assertEqual(set(protocol['selection_indices']) | set(protocol['confirmation_indices']), set(protocol['valid_indices']))
            data = pd.read_csv(h.task.parent/'train.csv')
            ids = []
            for index in range(2):
                plan = h.plan(validation_id=split['validation_id'], recipe_id='linear-regressor-v1',
                              model={'family': 'linear', 'hyperparameters': {'alpha': index+1}})
                eid = h.store.reserve(plan, h.code['sha256'])['experiment_id'];ids.append(eid)
                rows = protocol['valid_indices']
                predicted = data.loc[rows, 'label'].to_numpy().copy()
                for n, row in enumerate(rows):
                    predicted[n] += (100 if index == 0 else 0) if row in protocol['confirmation_indices'] else index
                art = h.artifacts.write_text('predictions/'+eid+'.csv', pd.DataFrame({'__row_index': rows,
                    'prediction': predicted}).to_csv(index=False), 'validation_predictions', 'text/csv')
                lease = {'lease_id': 'lease-'+str(index), 'fencing_token': index+1}
                attempt = h.store.start(eid, lease)
                h.store.apply_event({'event_id': 'event-'+str(index), 'job_id': 'job-'+str(index), 'experiment_id': eid,
                    'attempt_no': attempt, 'lease_id': lease['lease_id'], 'fencing_token': lease['fencing_token'],
                    'state': 'succeeded', 'output_artifact_ids': [art['artifact_id']], 'duration_seconds': 1,
                    'peak_memory_mb': 1, 'error': None}, 'train')
                ev = evaluator.evaluate({'experiment_id': eid, 'task_spec_id': h.spec['task_spec_id'], 'dataset_id': h.spec['dataset_id'],
                    'validation_id': split['validation_id'], 'predictions_artifact_id': art['artifact_id'], 'fidelity': 'full'})['evaluation']
                self.assertAlmostEqual(ev['primary_metric']['value'], index)
                h.store.commit_evaluation({'experiment_id': eid, 'evaluation_id': ev['evaluation_id'], 'selection_policy_id': 'same-protocol-primary-v1'})
            self.assertEqual(h.store.best(), ids[0])
            budget = Budget(replace(h.config, time_budget_seconds=10800), time.monotonic(), h.log)
            gateway = SimpleNamespace(cancelled=threading.Event())
            c = SimpleNamespace(artifacts=h.artifacts, store=h.store, spec=h.spec, validation=split, evaluator=evaluator,
                failed_refits=set(), budget=budget, log=h.log, final_candidate=h.store.best,
                researcher=SimpleNamespace(gateway=gateway), submission=Submission(h.spec, h.artifacts, h.store, h.log))
            # Normal search exits inside its finalization reserve. Confirmation
            # must consume its budget there, rather than require another reserve.
            budget.set_final_plan(h.store.get(ids[0])['plan'])
            budget.deadline = time.monotonic()+budget.reserve_seconds()+.5
            finalize(c)
            self.assertEqual(c.confirmed_candidate, ids[1])
            self.assertTrue(gateway.cancelled.is_set())
            self.assertTrue(budget.search_stopped)
            report = deepcopy(c.submission.final_confirmation)
            finalize(c)
            self.assertEqual(c.submission.final_confirmation, report)
        finally:
            h.close()

    def test_temporal_and_group_confirmation_keep_units_whole(self):
        data = pd.DataFrame({'period': np.repeat(np.arange(8), 20), 'group': np.repeat(np.arange(16), 10), 'y': 0})
        spec = {'task_kind': 'regression', 'target_column': 'y'}
        valid = list(range(80, 160))
        for strategy, column in [('temporal_holdout', 'period'), ('group_holdout', 'group')]:
            a, b, reason = partition(data, valid, {'strategy': strategy, 'parameters': {'column': column}, 'seed': 42}, spec)
            self.assertIsNone(reason)
            self.assertFalse(set(data.loc[a, column]) & set(data.loc[b, column]))
            if strategy == 'temporal_holdout':
                self.assertLess(data.loc[a, column].max(), data.loc[b, column].min())

    def test_tiny_or_single_period_validation_is_explicitly_not_confirmed(self):
        data = pd.DataFrame({'period': [1]*80, 'y': np.arange(80)})
        spec = {'task_kind': 'regression', 'target_column': 'y'}
        _, b, reason = partition(data, list(range(80)), {'strategy': 'temporal_holdout', 'parameters': {'column': 'period'}, 'seed': 1}, spec)
        self.assertEqual(b, [])
        self.assertEqual(reason, 'one_validation_period')
