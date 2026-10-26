"""Measured usefulness and independence of the completed slice research path."""
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd

from agent.fit_diagnostics import inner_folds, execute_oof, diagnostic_analysis
from agent.ml import build_pipeline
from agent.significance import paired_bootstrap, admits_update
from tests.test_error_slices import regression_spec, panel, KindMean
from tests.test_significance import spec


class DiagnosticBoundaryTests(unittest.TestCase):
    def test_held_labels_cannot_change_discovery_models_for_any_protocol(self):
        rng = np.random.default_rng(5)
        X = pd.DataFrame({'x': rng.normal(size=600), 'period': np.repeat(np.arange(6), 100),
                          'group': np.arange(600) % 15})
        for kind in ['regression', 'multiclass_classification']:
            y = pd.Series(X.x*4 if kind == 'regression' else np.where(X.x > .5, 'a', np.where(X.x < -.5, 'b', 'c')))
            task = {**spec('rmse' if kind == 'regression' else 'f1_macro', kind, [] if kind == 'regression' else ['a', 'b', 'c']),
                    'target_column': 'y', 'excluded_features': []}
            for strategy in ['random_holdout', 'group_holdout', 'temporal_holdout']:
                proposal = {'strategy': strategy, 'parameters': {'column': 'period' if strategy == 'temporal_holdout' else 'group'}}
                folds = inner_folds(X.assign(y=y), task, proposal, 42)
                held = folds[-1]['predict']
                self.assertTrue(all(not set(held).intersection(f['fit']) for f in folds))
                changed = y.copy()
                changed.iloc[held] = changed.iloc[held]+10000 if kind == 'regression' else changed.iloc[held].map({'a': 'b', 'b': 'c', 'c': 'a'})
                source = SimpleNamespace(make_pipeline=lambda *_: build_pipeline(
                    'linear-regressor-v1' if kind == 'regression' else 'linear-classifier-v1', task, {}))
                with tempfile.TemporaryDirectory() as directory:
                    matrices = []
                    for index, labels in enumerate([y, changed]):
                        job = {'cpu_threads': 1, 'diagnostic': {'folds': folds},
                               'prediction_path': str(Path(directory)/f'p{index}.csv')}
                        execute_oof(source, task, {}, X, labels, job, None, ExitStack())
                        matrices.append(pd.read_csv(job['prediction_path']))
                    pd.testing.assert_frame_equal(*matrices)

    def test_history_discovery_excludes_check_targets(self):
        frame = panel(entities=180)
        task = regression_spec()
        X, y = frame.drop(columns='y'), frame.y
        folds = inner_folds(frame, task, {'strategy': 'temporal_holdout', 'parameters': {'column': 'period'}}, 42)
        parts = []
        for index, fold in enumerate(folds):
            values = KindMean().fit(X.iloc[fold['fit']], y.iloc[fold['fit']]).predict(X.iloc[fold['predict']])
            parts.append(pd.DataFrame({'__row_index': X.index.to_numpy()[fold['predict']], '__fold': index, 'prediction': values}))
        output = pd.concat(parts).sort_values('__row_index')
        changed = y.copy()
        changed.iloc[folds[-1]['predict']] = np.random.default_rng(11).normal(size=len(folds[-1]['predict']))*1e6
        reports = [diagnostic_analysis(X, labels, output, folds, task,
                   {'strategy': 'temporal_holdout', 'column': 'period'}, 42, 1, float('inf')) for labels in [y, changed]]
        self.assertTrue(reports[0]['error_slices']['entity'])
        self.assertEqual(reports[0]['error_slices']['entity'], reports[1]['error_slices']['entity'])
        self.assertFalse(set(reports[0]['check_row_ids']) & set(reports[0]['discovery_fit_row_ids']))


class ClusterSupportTests(unittest.TestCase):
    def test_repeated_rows_do_not_invent_independent_evidence(self):
        # Three groups win a little; one loses more. Repeating each row makes an
        # IID test look certain while the four independent groups remain uncertain.
        delta = np.repeat([.2, .2, .2, -.4], 600)
        truth, base = np.zeros(len(delta)), np.ones(len(delta))
        candidate = np.sqrt(1-delta)
        row = paired_bootstrap(spec('rmse'), truth, base, candidate)
        cluster = paired_bootstrap(spec('rmse'), truth, base, candidate,
                                   clusters=np.repeat(np.arange(4), 600), unit='group')
        self.assertTrue(row['significant'])
        self.assertFalse(cluster['significant'])
        self.assertEqual(cluster['independent_units'], 4)
        repeated = paired_bootstrap(spec('rmse'), np.tile(truth, 2), np.tile(base, 2), np.tile(candidate, 2),
                                    clusters=np.tile(np.repeat(np.arange(4), 600), 2), unit='group')
        self.assertEqual(cluster['bootstrap_win_fraction'], repeated['bootstrap_win_fraction'])

    def test_one_period_is_unconfirmed_but_does_not_freeze_measured_progress(self):
        truth, base, better = np.zeros(100), np.ones(100), np.full(100, .9)
        verdict = paired_bootstrap(spec('rmse'), truth, base, better, clusters=np.zeros(100), unit='period')
        self.assertIsNone(verdict['bootstrap_win_fraction'])
        self.assertFalse(verdict['significant'])
        self.assertTrue(admits_update(verdict))
        self.assertEqual(verdict['unavailable'], 'insufficient_independent_units')

    def test_nonadditive_metric_resamples_whole_groups(self):
        truth = np.tile(['a', 'b'], 80)
        base = np.tile(['a', 'a'], 80)
        task = spec('f1_macro', 'binary_classification', ['a', 'b'])
        verdict = paired_bootstrap(task, truth, base, truth, clusters=np.repeat(np.arange(8), 20), unit='group')
        self.assertTrue(verdict['significant'])
        self.assertEqual(verdict['independent_units'], 8)
        self.assertEqual(verdict['method'], 'recomputed_metric')


class StableUtilityTests(unittest.TestCase):
    def test_publication_gate_does_not_erase_raw_finalist_refit_reserve(self):
        from agent.orchestrator import Orchestrator
        from agent.duration import refit_seconds
        c = Orchestrator.__new__(Orchestrator)
        final_plans = []
        raw = {'plan_id': 'raw'}
        c.final_candidate = lambda:'published'
        c.store = SimpleNamespace(champion=lambda:{'experiment_id':'published'}, best=lambda:'raw', get=lambda _: {'plan':raw})
        c.submission = SimpleNamespace(ready_candidate=lambda _:None)
        c.failed_refits = set()
        c.budget = SimpleNamespace(set_final_plan=final_plans.append)
        c.update_final_reserve()
        self.assertEqual(final_plans, [raw])
        self.assertEqual(refit_seconds(SimpleNamespace(estimate=lambda *_:1.4), raw), 3)

    def test_raw_trial_progress_does_not_wake_a_waiting_researcher(self):
        from agent.pipeline_search import VersionSearch
        search = VersionSearch.__new__(VersionSearch)
        search.c = SimpleNamespace(science=None, research_leaders={'b': 'base'}, research_feedback=[])
        version = {'version_id': 'v', 'initial_experiment_id': 'base', 'initial_metric': {'value': 10},
                   'paused': None, 'ticket': {'work_order': {'branch_id': 'b'}}, 'best_experiment_id': 'base'}
        search.versions = {'v':version}
        before = search.research_generation('b')
        version.update(best_experiment_id='jitter', best_metric={'value':9.999})
        self.assertEqual(before, search.research_generation('b'))
        search.c.research_leaders['b'] = 'material_gain'
        self.assertNotEqual(before, search.research_generation('b'))

    def test_noise_does_not_change_utility_but_a_supported_gain_does(self):
        from agent.orchestrator import Orchestrator
        c = Orchestrator.__new__(Orchestrator)
        c.config = SimpleNamespace(noise_aware_publication=True)
        events = []
        c.log = SimpleNamespace(event=lambda name, **kw: events.append((name, kw)))
        supported = False
        c.evaluator = SimpleNamespace(paired_significance=lambda *_: {'significant': supported})
        def row(eid, value):
            return {'experiment_id': eid, 'primary_metric': {'value': value, 'direction': 'minimize'}}
        base, jitter, gain = row('base', 10), row('jitter', 9.999), row('gain', 9)
        self.assertEqual(c.research_reference([base], 'branch')['experiment_id'], 'base')
        self.assertEqual(c.research_reference([base, jitter], 'branch')['experiment_id'], 'base')
        supported = True
        self.assertEqual(c.research_reference([base, jitter, gain], 'branch')['experiment_id'], 'gain')
        self.assertEqual(len(events), 2)
        # A sibling branch sees only its own measurements.
        self.assertEqual(c.research_reference([jitter], 'sibling')['experiment_id'], 'jitter')

    def test_best_solution_read_honors_delivered_reference_and_keeps_raw_trial_readable(self):
        from tests.graph_helpers import GraphHarness
        from agent.context_details import read_solution, solution_record
        h = GraphHarness(research_mode='adaptive')
        try:
            _, a, _, _ = h.result('researcher-1', {'alpha': 1}, 10)
            _, b, _, _ = h.result('researcher-1', {'alpha': 2}, 9.999)
            base, trial = a['lead_submission']['experiment_id'], b['lead_submission']['experiment_id']
            ticket = h.issue()
            context = h.h.artifacts.read_json(ticket['work_order']['context_id'])
            scope = context['metadata']['detail_scope']
            update = h.h.artifacts.write_json('research/stable-test.json', {
                **{k:scope[k] for k in ('branch_id','dataset_id','task_spec_id','validation_id')},
                'best_solution': {'experiment_id': base},
                'solution_catalog': {eid:solution_record(h.h.store,h.h.artifacts,eid) for eid in [base,trial]}}, 'research_updates')
            context['metadata']['authorized_feedback_ids'] = [update['artifact_id']]
            self.assertEqual(read_solution({'experiment_id':'best'}, context,h.h.artifacts,h.h.log)['experiment_id'], base)
            self.assertEqual(read_solution({'experiment_id':trial}, context,h.h.artifacts,h.h.log)['experiment_id'], trial)
        finally:h.close()
