"""Error slices: rule semantics, cross-fitted remedies, executable corrections and research wiring."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import time
import unittest

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from agent import error_slices, slice_actions as actions
from agent.errors import AgentError
from agent.strong import EntityHistory


def regression_spec(metric='rmse'):
    from agent import metrics
    definition = metrics.get(metric)
    return {'task_spec_id': 'task-unit', 'task_kind': 'regression', 'target_column': 'y', 'target_columns': ['y'],
            'excluded_features': ['row_id'], 'allowed_labels': [],
            'metric': {'metric_id': metric, 'direction': definition.direction, 'prediction_kind': 'values', 'parameters': {}}}


def panel(seed=0, entities=900):
    """Entities observed over periods; their level persists, a kind-mean model ignores it."""
    rng = np.random.default_rng(seed)
    level = rng.lognormal(4, .6, entities)
    kind = rng.choice(['a', 'b', 'c'], entities)
    rows = []
    for e in range(entities):
        for period in range(int(rng.integers(1, 4)), 5):
            rows.append({'row_id': f'r{len(rows)}', 'area': float(1000+e), 'kind': kind[e], 'period': period,
                         'noise': rng.normal(), 'y': level[e]*(1+.1*rng.normal())})
    return pd.DataFrame(rows)


class KindMean(BaseEstimator):
    """A weak learner that ignores entity history: per-kind training mean."""
    def fit(self, X, y):
        self.means_ = pd.Series(np.asarray(y, dtype=float), index=X.index).groupby(X['kind']).mean()
        self.default_ = float(np.mean(y))
        return self

    def predict(self, X):
        return X['kind'].map(self.means_).fillna(self.default_).to_numpy(dtype=float)


def oof(frame):
    """Rolling OOF of KindMean: each period predicted from earlier periods."""
    parts, folds = [], []
    for number, period in enumerate([2, 3, 4]):
        fit, test = frame[frame.period < period], frame[frame.period == period]
        parts.append(pd.Series(KindMean().fit(fit, fit.y).predict(test), index=test.index))
        folds.append(pd.Series(number, index=test.index))
    rows = frame.period >= 2
    return frame[rows].reset_index(drop=True), pd.concat(parts).loc[frame.index[rows]].to_numpy(), pd.concat(folds).loc[frame.index[rows]].to_numpy()


class RuleTests(unittest.TestCase):
    def test_numeric_and_set_conditions_with_missing_semantics(self):
        frame = pd.DataFrame({'x': [1., np.nan, 5.], 'k': ['a', None, 'b']})
        np.testing.assert_array_equal(actions.condition_mask(frame, {'column': 'x', 'op': 'le', 'value': 2.}), [True, False, False])
        np.testing.assert_array_equal(actions.condition_mask(frame, {'column': 'x', 'op': 'le', 'value': 2., 'missing': True}), [True, True, False])
        np.testing.assert_array_equal(actions.condition_mask(frame, {'column': 'x', 'op': 'gt', 'value': 2.}), [False, False, True])
        np.testing.assert_array_equal(actions.condition_mask(frame, {'column': 'k', 'op': 'in', 'value': ['a', None]}), [True, True, False])
        np.testing.assert_array_equal(actions.condition_mask(frame, {'column': 'k', 'op': 'not_in', 'value': ['a']}), [False, True, True])

    def test_simplify_merges_bounds_and_drops_redundant_missing(self):
        merged = error_slices.simplify([{'column': 'h', 'op': 'le', 'value': 9., 'missing': True},
                                        {'column': 'h', 'op': 'gt', 'value': 3.}, {'column': 'h', 'op': 'le', 'value': 7.}])
        self.assertEqual(merged, [{'column': 'h', 'op': 'gt', 'value': 3.}, {'column': 'h', 'op': 'le', 'value': 7.}])

    def test_corrections_are_validated_against_task_and_columns(self):
        spec = regression_spec()
        good = [{'slice_id': 'S1', 'conditions': [{'column': '__prediction', 'op': 'gt', 'value': 3.}], 'action': 'shift',
                 'params': {'value': 1.5}, 'scale': 'identity', 'entity': None}]
        self.assertEqual(actions.validate_corrections(good, spec, ['x']), good)
        for change, message in ((lambda c: c[0].update(action='remap'), 'action'),
                                (lambda c: c[0].update(scale='log1p'), 'scale'),
                                (lambda c: c[0]['conditions'][0].update(column='__truth'), 'meta'),
                                (lambda c: c[0]['conditions'][0].update(column='absent'), 'absent'),
                                (lambda c: c[0].update(params={'value': float('nan')}), 'params'),
                                (lambda c: c[0].update(action='blend_history', params={'weight': .5}), 'entity')):
            broken = deepcopy(good)
            change(broken)
            with self.subTest(message=message), self.assertRaises(ValueError):
                actions.validate_corrections(broken, spec, ['x'])

    def test_regression_actions_in_log_scale(self):
        p = np.array([10., 100.])
        shifted = actions.apply_regression(p, np.array([True, False]), 'shift', {'value': np.log(2)}, 'log1p')
        self.assertAlmostEqual(shifted[0], 2*11-1)
        self.assertEqual(shifted[1], 100.)
        blended = actions.apply_regression(p, np.array([True, True]), 'blend_history', {'weight': 1.}, 'identity', np.array([5., np.nan]))
        np.testing.assert_allclose(blended, [5., 100.])


class LabTests(unittest.TestCase):
    def test_lab_finds_a_robust_history_blend_and_ranks_by_excess(self):
        frame = panel()
        rows, prediction, fold = oof(frame)
        history = EntityHistory(['area', 'kind'], 'period', prefix='__history_').fit(frame, frame.y).transform(rows)
        report = error_slices.analyze(rows, rows.y, prediction.reshape(-1, 1), fold, regression_spec(),
                                      meta_history=history.reset_index(drop=True),
                                      entity={'columns': ['area', 'kind'], 'period_column': 'period'})
        self.assertTrue(report['available'])
        fixable = [s for s in report['slices'] if s['verdict'] == 'fixable']
        self.assertTrue(fixable)
        best = fixable[0]['remedies'][0]
        self.assertEqual(best['action'], 'blend_history')
        self.assertTrue(best['robust'])
        self.assertLess(best['oof_metric_after'], best['oof_metric_before'])
        self.assertEqual(best['folds_improved'], best['folds'])
        for item in report['slices']:
            self.assertTrue(all(not c['column'] == 'y' for c in item['conditions']), 'truth never defines a slice')
            self.assertGreater(item['loss_share'], 0)
            self.assertIn(item['slice_id'], item['journal'])
        json.dumps(report)  # immutable artifact must be plain JSON

    def test_single_column_population_is_not_reported_as_a_hard_region(self):
        rng = np.random.default_rng(1)
        n = 3000
        data = pd.DataFrame({'state': np.where(rng.random(n) < .8, 'big', 'small'), 'x': rng.normal(size=n), 'row_id': np.arange(n)})
        y = rng.normal(size=n)*10
        prediction = y+rng.normal(size=n)*np.where(data.x > 1.5, 30, 3)  # the hard region is x>1.5, not the big state
        report = error_slices.analyze(data, y, prediction.reshape(-1, 1), np.arange(n) % 3, regression_spec())
        top = report['slices'][0]
        # The hard region shows up through x or through the predictions it distorts; never as the big state.
        self.assertTrue(any(c['column'] in {'x', '__prediction'} for c in top['conditions']), top['rule'])
        self.assertFalse(any(c['column'] == 'state' for s in report['slices'] for c in s['conditions'] if len(s['conditions']) == 1))
        self.assertGreater(top['lift'], 3)

    def test_unexplained_slices_are_unresolved_with_a_reason_not_noise(self):
        rng = np.random.default_rng(2)
        n = 2400
        data = pd.DataFrame({'x': rng.normal(size=n), 'row_id': np.arange(n)})
        y = rng.normal(size=n)
        noise = np.where(data.x > 1.0, rng.standard_t(2, size=n)*5, rng.normal(size=n))
        report = error_slices.analyze(data, y, (y+noise).reshape(-1, 1), np.arange(n) % 3, regression_spec())
        self.assertNotIn('noise', {item['verdict'] for item in report['slices']})
        for item in report['slices']:
            if item['verdict'] == 'unresolved':
                self.assertTrue(item['unresolved_reason'])
                self.assertIn(item['unresolved_reason'], item['journal'])

    def test_slices_are_chosen_without_the_holdout_fold(self):
        rng = np.random.default_rng(4)
        n = 3000
        data = pd.DataFrame({'x': rng.normal(size=n), 'row_id': np.arange(n)})
        y = rng.normal(size=n)
        fold = np.arange(n) % 3
        # Large errors only in the holdout fold (2): discovery folds show no hard region there.
        prediction = y+np.where((fold == 2) & (data.x > 1).to_numpy(), 20, 0)+rng.normal(size=n)*.1
        report = error_slices.analyze(data, y, prediction.reshape(-1, 1), fold, regression_spec())
        self.assertEqual(report['holdout_rows'], 1000)
        # Random discovery errors can themselves select an x threshold. The
        # boundary is invariance to held targets/predictions, not absence of x.
        changed_y, changed_prediction = y.copy(), prediction.copy()
        changed_y[fold == 2] += 100
        changed_prediction[fold == 2] -= 100
        changed = error_slices.analyze(data, changed_y, changed_prediction.reshape(-1, 1), fold, regression_spec())
        def rules(value):
            return sorted(json.dumps(s['conditions'], sort_keys=True) for s in value['slices'])
        self.assertEqual(rules(report), rules(changed))

    def test_label_remap_for_a_systematic_confusion(self):
        from agent import metrics
        rng = np.random.default_rng(3)
        n = 1800
        data = pd.DataFrame({'region': rng.choice(['north', 'south'], n), 'x': rng.normal(size=n), 'row_id': np.arange(n)})
        truth = np.where(rng.random(n) < .5, 'alpha', 'beta').astype(object)
        truth[(data.region == 'north').to_numpy() & (rng.random(n) < .8)] = 'gamma'
        predicted = truth.copy()
        predicted[truth == 'gamma'] = 'beta'  # base model never predicts gamma in the north
        spec = {'task_spec_id': 'task-unit', 'task_kind': 'multiclass_classification', 'target_column': 'y', 'excluded_features': ['row_id'],
                'allowed_labels': ['alpha', 'beta', 'gamma'],
                'metric': metrics.resolve({'metric_id': 'f1_macro'}, 'multiclass_classification', ['alpha', 'beta', 'gamma'])}
        report = error_slices.analyze(data, truth.astype(str), predicted.astype(str).reshape(-1, 1), np.arange(n) % 3, spec)
        remedies = [r for s in report['slices'] for r in s['remedies'] if r['robust']]
        self.assertTrue(remedies)
        self.assertEqual(remedies[0]['action'], 'remap')
        self.assertEqual(remedies[0]['correction']['params']['mapping'], {'beta': 'gamma'})


class ExecutionTests(unittest.TestCase):
    def test_executed_correction_reproduces_the_lab_mask_and_values(self):
        frame = panel(seed=4)
        rows, prediction, fold = oof(frame)
        entity = {'columns': ['area', 'kind'], 'period_column': 'period'}
        history = EntityHistory(entity['columns'], 'period', prefix='__history_').fit(frame, frame.y).transform(rows)
        report = error_slices.analyze(rows, rows.y, prediction.reshape(-1, 1), fold, regression_spec(),
                                      meta_history=history.reset_index(drop=True), entity=entity)
        remedy = next(r for s in report['slices'] for r in s['remedies'] if r['robust'])
        correction = remedy['correction']
        train, test = frame[frame.period < 4], frame[frame.period == 4]
        features = [c for c in frame.columns if c != 'y']
        wrapped = actions.SliceCorrectionModel(KindMean(), [correction], regression_spec()).fit(train[features], train.y)
        base = KindMean().fit(train[features], train.y).predict(test[features])
        corrected = wrapped.predict(test[features])
        history = EntityHistory(entity['columns'], 'period', prefix='__history_').fit(train[features], train.y).transform(test[features])
        meta = actions.meta_frame(base, regression=True, history=history)
        mask = actions.rule_mask(actions.rule_frame(test[features], meta), correction['conditions'])
        expected = actions.apply_regression(base, mask, correction['action'], correction['params'], correction['scale'],
                                            meta['__history_last'].to_numpy())
        np.testing.assert_allclose(corrected, expected)
        np.testing.assert_allclose(corrected[~mask], base[~mask])
        rmse = lambda values: float(np.sqrt(np.mean((test.y.to_numpy()-values)**2)))
        self.assertLess(rmse(corrected), rmse(base))

    def test_feature_pipeline_wraps_owned_recipes_with_corrections(self):
        from agent.feature_engineering import make_feature_pipeline
        spec = regression_spec()
        correction = {'slice_id': 'S1', 'conditions': [{'column': '__prediction', 'op': 'gt', 'value': 0.}], 'action': 'shift',
                      'params': {'value': 2.}, 'scale': 'identity', 'entity': None}
        model = make_feature_pipeline(spec, {'base_recipe_id': 'linear-regressor-v1', 'model_parameters': {'alpha': 1.0},
                                             'feature_program': [], 'slice_corrections': [correction]})
        self.assertIsInstance(model, actions.SliceCorrectionModel)
        X = pd.DataFrame({'x': np.arange(20, dtype=float)})
        y = X.x*2
        model.fit(X, y)
        raw = model.estimator.predict(X)
        np.testing.assert_allclose(model.predict(X), np.where(raw > 0, raw+2, raw))

    def test_remedy_reference_resolves_only_from_a_trusted_report(self):
        from agent.artifacts import Artifacts
        from agent.observability.logging import RunLogger
        with tempfile.TemporaryDirectory() as temp:
            log = RunLogger(Path(temp), 'unit', level='ERROR')
            artifacts = Artifacts(Path(temp), 'unit', log)
            correction = {'slice_id': 'S1', 'conditions': [{'column': 'x', 'op': 'gt', 'value': 0.}], 'action': 'shift',
                          'params': {'value': 2.}, 'scale': 'identity', 'entity': None}
            report = {'task_spec_id': 'task-unit', 'error_slices': {'slices': [{'slice_id': 'S1', 'remedies': [
                {'remedy_id': 'S1/shift', 'correction': correction}]}]}}
            art = artifacts.write_json('diagnostics/unit.json', report, 'fit_diagnostics')
            other = artifacts.write_json('research/other.json', report, 'research_note')
            spec = regression_spec()
            resolved = error_slices.resolve_remedy(art['artifact_id']+'#S1/shift', artifacts, spec, ['x'])
            self.assertEqual(resolved['correction'], {**correction, 'slice_id': art['artifact_id']+'#S1/shift'})
            self.assertEqual(resolved['base']['experiment_id'], None)  # this synthetic report names no base
            for reference in ('S1/shift', art['artifact_id']+'#S9/shift', other['artifact_id']+'#S1/shift'):
                with self.subTest(reference=reference), self.assertRaises(AgentError):
                    error_slices.resolve_remedy(reference, artifacts, spec, ['x'])
            with self.assertRaises(AgentError):
                error_slices.resolve_remedy(art['artifact_id']+'#S1/shift', artifacts, {**spec, 'task_spec_id': 'task-other'}, ['x'])


class ResearchWiringTests(unittest.TestCase):
    def test_slice_problem_question_and_compact_report(self):
        from agent.error_research import slice_problem, slice_question
        from agent.fit_diagnostics import compact_report
        frame = panel(seed=5)
        rows, prediction, fold = oof(frame)
        history = EntityHistory(['area', 'kind'], 'period', prefix='__history_').fit(frame, frame.y).transform(rows)
        slices = error_slices.analyze(rows, rows.y, prediction.reshape(-1, 1), fold, regression_spec(),
                                      meta_history=history.reset_index(drop=True),
                                      entity={'columns': ['area', 'kind'], 'period_column': 'period'})
        problem = slice_problem(slices['slices'][0], 'art-0123456789abcdef')
        self.assertEqual(problem['definition']['kind'], 'rule')
        self.assertTrue(problem['card']['remedies'][0]['remedy_ref'].startswith('art-0123456789abcdef#'))
        self.assertIn('slice_remedies', slice_question(problem['card']))
        compact = compact_report({'hard_groups': [{'rows': 1}]*6, 'error_slices': slices}, source_artifact_id='art-0123456789abcdef')
        self.assertNotIn('hard_groups', compact)
        self.assertEqual(compact['omitted_hard_groups'], 6)
        self.assertTrue(all('remedy_ref' in r for s in compact['error_slices']['slices'] for r in s['remedies']))

    def test_rule_membership_on_selection_rows_uses_base_meta_and_fit_history(self):
        from agent.error_research import rule_membership
        frame = panel(seed=6)
        fit, selection = frame[frame.period < 4], frame[frame.period == 4].reset_index(drop=True)
        definition = {'kind': 'rule', 'slice_id': 'S1', 'entity': {'columns': ['area', 'kind'], 'period_column': 'period'},
                      'conditions': [{'column': '__history_count', 'op': 'gt', 'value': .5}, {'column': '__prediction', 'op': 'gt', 'value': 0.}]}
        mask = rule_membership(definition, selection, np.ones((len(selection), 1)), regression_spec(),
                               (fit[['area', 'kind', 'period']], fit.y.to_numpy()))
        seen = set(zip(fit.area, fit.kind))
        np.testing.assert_array_equal(mask, [(a, k) in seen for a, k in zip(selection.area, selection.kind)])


class DiagnosticSliceTests(unittest.TestCase):
    """A real fit-only OOF diagnostic job produces the slice report and research problems."""
    from tests.test_error_driven_research import DiagnosticIntegrationTests as _Base
    setUp, tearDown, complete = _Base.setUp, _Base.tearDown, _Base.complete

    def test_diagnostic_report_contains_slices_and_feeds_error_research(self):
        record = self.complete()
        report = record['report']
        self.assertIn('error_slices', report)
        self.assertTrue(report['error_slices']['available'], report['error_slices'])
        self.assertEqual(report['task_spec_id'], self.c.spec['task_spec_id'])
        stored = self.c.artifacts.read_json(record['artifact_id'])
        self.assertEqual(stored['error_slices'], report['error_slices'])
        self.assertEqual(report['analysis_execution'], 'leased_oof_worker')
        self.assertFalse(set(report['check_row_ids']) & set(report['discovery_fit_row_ids']))
        problems = list(self.c.error_research.problems.values())
        self.assertTrue(problems)
        if report['error_slices']['slices']:
            self.assertTrue(all(p['definition']['kind'] == 'rule' for p in problems if p['source_artifact_id'] == record['artifact_id']))
            events = [json.loads(line) for line in (self.c.artifacts.root/'events.jsonl').read_text().splitlines()]
            journal = [e for e in events if e['event'] == 'research.journal' and e.get('kind') == 'error_slice']
            self.assertEqual(len(journal), len(report['error_slices']['slices']))


class SliceEndToEndTests(unittest.TestCase):
    """Mock adaptive run: OOF slices -> remedy bound to its base -> executed -> paired journal line."""
    def test_remedy_is_bound_executed_and_journalled(self):
        import io
        import os
        from contextlib import redirect_stderr
        from unittest.mock import patch
        from agent.runtime import main
        from scripts.make_demo import make_demo
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            task = make_demo(root/'task', 'regression', rows=2400)
            def target(frame):
                x = pd.to_numeric(frame.x_numeric, errors='coerce').fillna(0)
                return 2*x+3*frame.z_numeric+12*(x > 2.4)+.2*np.sin(frame.z_numeric)
            train = pd.read_csv(task.parent/'train.csv', dtype={'id': str})
            train['label'] = target(train)
            train.to_csv(task.parent/'train.csv', index=False)
            test = pd.read_csv(task.parent/'test.csv', dtype={'id': str})
            truth = target(test).to_numpy()
            # Controlled weak reference: the discontinuity produces a stable
            # correctable bias. Scripted decisions exercise production tools;
            # they do not invent metrics or force a remedy to pass validation.
            from agent.llm.gateway import MockTransport
            original = MockTransport.complete
            issued = {'base': False, 'remedy': False}
            def complete(transport, request, timeout, context):
                if context['metadata']['schema_id'] != 'ResearchAction-v1':
                    return original(transport, request, timeout, context)
                user = json.loads(context['messages'][1]['content'])
                advice = user.get('profile_summary', user).get('error_research') or {}
                remedies = [r for r in (advice.get('observation') or {}).get('remedies', []) if r.get('robust')]
                # A continuing dialogue receives diagnostics through feedback,
                # not by mutating its frozen first user message.
                for message in request['messages']:
                    if message['role'] != 'user':continue
                    try:updates = json.loads(message['content']).get('research_updates') or {}
                    except (ValueError, TypeError):continue
                    for record in updates.get('fit_diagnostics', []):
                        for card in (record.get('report', {}).get('error_slices') or {}).get('slices', []):
                            remedies.extend(r for r in card.get('remedies', []) if r.get('robust'))
                value = {'action': 'research', 'next_step': 'wait', 'hypothesis': 'Wait for the measured diagnostic.'}
                if not issued['base']:
                    issued['base'] = True
                    value = {'action': 'catalog', 'recipe_id': 'linear-regressor-v1', 'hyperparameters': {'alpha': 1.},
                             'hypothesis': 'Measure a weak linear reference on a synthetic discontinuity.'}
                elif remedies and not issued['remedy']:
                    issued['remedy'] = True
                    value = {'action': 'features', 'slice_remedies': [remedies[0]['remedy_ref']],
                             'hypothesis': 'Execute the independently checked correction of the discontinuity.'}
                return {'choices': [{'message': {'content': json.dumps(value)}, 'finish_reason': 'stop'}]}
            settings = {'LLM_MODE': 'mock', 'RESEARCH_MODE': 'adaptive', 'TIME_BUDGET_SECONDS': '120',
                        'RESEARCH_OOF_BUDGET_FRACTION': '.10', 'RESEARCH_OOF_ROWS': '2400',
                        'FINAL_ENSEMBLE_SIZE': '0', 'LLM_MAX_CALLS': '0',
                        'DATA_RESEARCH_MIN_SECONDS': '0', 'CATALOG_WARMUP_EXPERIMENTS': '0', 'LOG_LEVEL': 'ERROR',
                        'MODEL_SNAPSHOT_MAX_BYTES': '0', 'GRAPH_MIN_FREE_DISK_BYTES': '0'}
            with patch.dict(os.environ, settings), patch.object(MockTransport, 'complete', complete), redirect_stderr(io.StringIO()) as stderr:
                code = main([str(task), '--llm-mode', 'mock', '--work-dir', str(root/'runs')])
            self.assertEqual(code, 0, stderr.getvalue()[-5000:])
            events = [json.loads(line) for line in next((root/'runs').glob('run-*/events.jsonl')).read_text().splitlines()]
            self.assertFalse([e for e in events if e['event'] in {'research.proposal_failed', 'diagnostics.slices_unavailable'}])
            slices = [e for e in events if e['event'] == 'research.journal' and e['kind'] == 'error_slice']
            self.assertTrue(slices, 'the fit-only OOF diagnostic must report slices on this task')
            bound = [e for e in events if e['event'] == 'research.slice_remedies_bound']
            self.assertTrue(bound)
            executed = [e for e in events if e['event'] == 'research.journal' and e['kind'] == 'candidate' and 'remedies' in e['line']]
            self.assertTrue(executed, 'a remedy candidate must be trained, evaluated and journalled')
            self.assertIn(executed[0]['status'], {'overall_improved', 'unconfirmed_gain', 'within_noise', 'partial_group_improvement', 'no_gain'})
            self.assertEqual(events[-1]['event'], 'run.finished')
            from agent.ml import build_pipeline
            from agent import metrics
            spec = json.loads(next((root/'runs').glob('run-*/task/task-spec.json')).read_text())
            reference = build_pipeline('linear-regressor-v1', spec, {'alpha': 1.})
            features = [c for c in train if c not in {'id', 'label'}]
            reference.fit(train[features], train.label)
            baseline = metrics.score(spec, truth, reference.predict(test[features]))
            delivered = pd.read_csv(task.parent/'submission.csv', dtype={'id': str})
            self.assertEqual(delivered.id.tolist(), test.id.tolist())
            final = metrics.score(spec, truth, delivered.label.to_numpy())
            self.assertLess(final, baseline)
            print('SYNTHETIC_SLICE_DELIVERY '+json.dumps({'baseline_rmse': baseline, 'delivered_rmse': final,
                'scope': 'scripted decisions, real diagnostic/fit/selection/confirmation/refit; independent synthetic test'}), flush=True)


class ActionSchemaTests(unittest.TestCase):
    def test_features_action_accepts_remedies_without_a_program(self):
        from types import SimpleNamespace
        from agent.research_actions import normalize_action, validate_action
        catalog = SimpleNamespace(recipes={'lightgbm-regressor-v1': {'family': 'lightgbm'}})
        action = normalize_action({'action': 'features', 'recipe_id': 'lightgbm-regressor-v1', 'hypothesis': 'h', 'evidence_ids': [],
                                   'slice_remedies': ['art-1#S1/shift']})
        validate_action(action, catalog)
        with self.assertRaises(AgentError):
            validate_action({**action, 'slice_remedies': ['art-1#S1/shift']*2}, catalog)
        with self.assertRaises(AgentError):
            validate_action(normalize_action({'action': 'catalog', 'recipe_id': 'lightgbm-regressor-v1', 'hypothesis': 'h',
                                              'evidence_ids': [], 'slice_remedies': ['art-1#S1/shift']}), catalog)


if __name__ == '__main__':
    unittest.main()
