"""Label-free deployment facts and deterministic metric recovery."""
import unittest

import numpy as np
import pandas as pd

from agent.data_facts import apply_duplicate_groups, deployment_facts, future_period_column, scenario_proposal
from agent.task_rules import infer_metric


class FuturePeriodTests(unittest.TestCase):
    def test_test_after_every_training_period_is_temporal(self):
        train = pd.DataFrame({'period': np.repeat([1, 2, 3, 4], 50), 'x': np.arange(200)})
        test = pd.DataFrame({'period': np.repeat([5], 40), 'x': np.arange(40)})
        found = future_period_column(train, test)
        self.assertEqual(found['column'], 'period')
        proposal = scenario_proposal({'future_period': found}, 7)
        self.assertEqual(proposal['strategy'], 'temporal_holdout')
        self.assertEqual(proposal['parameters']['column'], 'period')

    def test_row_counter_and_random_test_are_not_periods(self):
        train = pd.DataFrame({'counter': np.arange(200), 'period': np.repeat([1, 2, 3, 4], 50)})
        test = pd.DataFrame({'counter': np.arange(200, 240), 'period': np.repeat([2, 3], 20)})
        self.assertIsNone(future_period_column(train, test))

    def test_iso_dates_are_orderable_periods(self):
        days = pd.date_range('2024-01-01', periods=10, freq='D').strftime('%Y-%m-%d')
        train = pd.DataFrame({'day': np.repeat(days[:8], 20)})
        test = pd.DataFrame({'day': np.repeat(days[8:], 10)})
        self.assertEqual(future_period_column(train, test)['column'], 'day')

    def test_facts_never_raise(self):
        facts = deployment_facts(pd.DataFrame({'a': [1, 2]}), pd.DataFrame({'b': [1]}), ['y'])
        self.assertIsNone(facts['future_period'])
        self.assertEqual(facts['duplicates'], {'train_duplicate_share': 0.0, 'test_seen_in_train_share': 0.0})


class DuplicateGroupTests(unittest.TestCase):
    STRATIFIED = {'strategy': 'stratified_holdout', 'seed': 0, 'parameters': {'valid_fraction': .2}, 'reason': 'r.'}

    def test_repeats_absent_from_test_group_the_holdout(self):
        facts = {'duplicates': {'train_duplicate_share': .4, 'test_seen_in_train_share': .05}}
        grouped = apply_duplicate_groups(self.STRATIFIED, facts)
        self.assertTrue(grouped['parameters']['duplicate_groups'])
        self.assertEqual(grouped['parameters']['valid_fraction'], .2)
        self.assertNotIn('duplicate_groups', self.STRATIFIED['parameters'])

    def test_repeats_shared_with_test_or_rare_keep_the_holdout(self):
        for duplicates in ({'train_duplicate_share': .4, 'test_seen_in_train_share': .35},
                           {'train_duplicate_share': .02, 'test_seen_in_train_share': 0.0}, None):
            self.assertIs(apply_duplicate_groups(self.STRATIFIED, {'duplicates': duplicates}), self.STRATIFIED)
        temporal = {**self.STRATIFIED, 'strategy': 'temporal_holdout'}
        facts = {'duplicates': {'train_duplicate_share': .4, 'test_seen_in_train_share': 0.0}}
        self.assertIs(apply_duplicate_groups(temporal, facts), temporal)

    def test_profile_counts_repeats_by_features_only(self):
        train = pd.DataFrame({'a': [1, 1, 2, 3], 'b': ['x', 'x', 'y', 'z'], 'y': [0, 1, 0, 1]})
        test = pd.DataFrame({'a': [1, 9], 'b': ['x', 'q']})
        facts = deployment_facts(train, test, ['y'])
        self.assertEqual(facts['duplicates'], {'train_duplicate_share': .5, 'test_seen_in_train_share': .5})


class MetricInferenceTests(unittest.TestCase):
    def test_primary_metric_from_prose(self):
        cases = {'Метрика оценки — **RMSE** на исходной шкале таргета.': 'rmse',
                 'Основная метрика — **Macro F1**. Accuracy и Weighted F1 справочные.': 'f1_macro',
                 'Macro F1 выбрана основной, потому что Accuracy не отражает редкие классы.': 'f1_macro',
                 'Metric: MAE': 'mae', 'Оценка по accuracy и macro F1': None}
        for text, expected in cases.items():
            self.assertEqual(infer_metric(text), expected, text)


if __name__ == '__main__':
    unittest.main()
