"""Executable research freedom, sealed analysis, explicit tuning and exact refit."""
from contextlib import redirect_stderr
from dataclasses import replace
import io
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from agent.catalog import Catalog
from agent.config import Config, ROOT
from agent.context import output_schemas
from agent.errors import AgentError
from agent.llm.gateway import Gateway, MockTransport
from agent.local_search import LocalSearch
from agent.research_actions import ACTION_SCHEMA, CUSTOM_RECIPE, validate_action
from agent.researcher import Researcher
from agent.runtime import main
from scripts.make_demo import make_demo
from scripts.reconstruct_solution import reconstruct
from tests.graph_helpers import GraphHarness
from tools.analysis import ANALYSIS_SCHEMA, AnalysisService
from tools.code import validate_code


SOURCE = '''from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression
def make_pipeline(task_spec, params):
    numerical = Pipeline([("impute", SimpleImputer()), ("polynomial", PolynomialFeatures(2)), ("scale", StandardScaler())])
    prepare = ColumnTransformer([("numeric", numerical, ["x_numeric", "z_numeric"])])
    model = Ridge(alpha=params["alpha"]) if task_spec["task_kind"] == "regression" else LogisticRegression(C=1.0, max_iter=200, random_state=params["_seed"])
    return Pipeline([("features", prepare), ("model", model)])
'''


def action(kind='pipeline', **changes):
    value = dict(action=kind, recipe_id=CUSTOM_RECIPE if kind == 'pipeline' else 'linear-regressor-v1',
                 hyperparameters={'alpha': .03}, hypothesis='Measure polynomial features and a regularized model together.',
                 expected_effect='Compare the whole pipeline on the frozen split.',
                 feature_steps=[{'step': 'selected_numeric_polynomial', 'degree': 2}] if kind == 'pipeline' else [],
                 code=SOURCE if kind == 'pipeline' else None, tuning=None, evidence_ids=[])
    value.update(changes)
    if 'investigation' not in value:
        value['investigation'] = {
            'question': 'Does this complete synthetic configuration improve the measured reference?',
            'basis_ids': [], 'columns': ['x_numeric', 'z_numeric'] if kind == 'pipeline' else
                list(dict.fromkeys(c for op in value.get('feature_program', []) for c in op.get('columns', []))),
            'interpretation': 'exploratory', 'mechanism': 'Compare the explicitly declared synthetic inputs and estimator.',
            'assumptions': ['Synthetic regression fixture; no physical interpretation is claimed.'],
            'expected_observation': 'The independent evaluator measures the complete proposed configuration.',
            'alternative_explanation': 'The proposed representation may add noise or duplicate the baseline signal.',
            'decision_rule': 'Use the fixed evaluation protocol and retain the better valid output.'}
    if 'research_notes' not in value:
        from agent.scientific_research import synthetic_notes
        value['research_notes'] = synthetic_notes(value, {})
        value['research_notes']['test_plan']['switch_reason'] = 'This fixture explicitly exercises the proposed alternative configuration.'
    return value


class AdaptiveResearchTests(unittest.TestCase):
    def setUp(self):
        self.h = GraphHarness('regression', research_mode='adaptive')
        self.g, self.c = self.h.g, self.h.controller

    def tearDown(self):
        self.h.close()

    def service(self):
        return AnalysisService(self.g.spec, self.g.validation, self.g.artifacts, self.g.budget, self.g.log)

    def test_schema_and_missingness_survive_pressure_and_request_compaction(self):
        self.g.context.profile['data_dictionary'] = [{'description': 'large redundant description '*2000}]
        ticket = self.h.issue()
        context = self.g.artifacts.read_json(ticket['work_order']['context_id'])
        user = json.loads(context['messages'][1]['content'])
        core = user['profile_summary']['data']['core_profile']
        self.assertIn('train_missing', core['column_fields'])
        self.assertTrue(any(row[0] == 'x_numeric' and row[2] > 0 for row in core['columns']))
        request = {'messages': [{'role': 'user', 'content': json.dumps(user)}]}
        Gateway._compact_research_request(type('Fixture', (), {'log': self.g.log})(), request, 1)
        compacted = json.loads(request['messages'][0]['content'])
        self.assertEqual(compacted['profile_summary']['data']['core_profile'], core)
        self.assertEqual(compacted['base_code'], user['base_code'])

    def test_analysis_is_fit_only_and_rejects_raw_target_column_access(self):
        import pandas as pd
        service = self.service()
        indices = self.g.artifacts.read_json(self.g.validation['train_rows_artifact_id'])
        train = pd.read_csv(self.g.artifacts.resolve(self.g.spec['train_artifact_id']))
        query = dict(kind='summary', columns=['x_numeric'], group_by=None, include_target=True)
        result = service.analyze(query)
        target = next(c for c in result['report']['columns'] if c['column'] == 'label')
        self.assertEqual(target['quantiles']['0.0'], float(train.iloc[indices]['label'].min()))
        self.assertEqual(result['report']['fit_rows'], len(indices))
        self.assertNotIn('valid_indices', json.dumps(result))
        self.assertEqual(service.analyze(query), result)
        self.assertEqual(service.analyze(dict(kind='report', report_id=result['artifact_id'], columns=[],
                                             group_by=None, include_target=False)), result)
        self.assertEqual(len(service.recent()), 1)
        with self.assertRaises(AgentError):
            service.analyze({**query, 'columns': ['label']})
        comparison = service.analyze({**query, 'kind': 'compare', 'include_target': False})
        self.assertEqual(comparison['report']['scope'], 'fit_and_unlabelled_test_features')
        self.assertTrue(all(c['column'] != 'label' for c in comparison['report']['test']))

    def test_tool_roundtrip_then_custom_pipeline_passes_graph_contracts(self):
        service = self.service()
        class Transport:
            def complete(transport, request, timeout, context):
                if request['messages'][-1]['role'] != 'tool':
                    call = {'id': 'analysis-call', 'type': 'function', 'function': {'name': 'analyze_data',
                        'arguments': json.dumps(dict(kind='summary', columns=['x_numeric'], group_by=None, include_target=True))}}
                    message = {'role': 'assistant', 'content': None, 'tool_calls': [call]}
                else:
                    report = json.loads(request['messages'][-1]['content'])
                    message = {'role': 'assistant', 'content': json.dumps(action(evidence_ids=[report['artifact_id']]))}
                return {'choices': [{'message': message, 'finish_reason': 'stop'}],
                        'usage': {'prompt_tokens': 100, 'completion_tokens': 100}}
            def close(transport):
                pass
        gateway = Gateway(self.g.config, self.g.budget, self.g.artifacts, self.g.bus, self.g.log, output_schemas(self.g.bus), Transport())
        gateway.analysis = service
        researcher = Researcher(self.g.bus, gateway, self.g.artifacts, self.g.spec, self.g.catalog, self.g.config, self.g.log)
        researcher.validation_id = self.g.validation['validation_id']
        ticket = self.h.issue()
        proposal = researcher.propose(ticket['researcher_request']['payload'])
        plan = proposal['plan']
        self.assertEqual(plan['model']['family'], 'custom')
        self.assertEqual(plan['model']['hyperparameters'], {'alpha': .03})
        manifest = self.g.constructor.build(plan)
        self.assertEqual(manifest['verification_level'], 'declared_only')
        from agent.graph.controller import lineage_for
        submitted = self.g.invoke('G04', {'work_order_id': ticket['work_order']['work_order_id'],
            'proposal': proposal, 'manifest': manifest, 'lineage': lineage_for(ticket['work_order'])})
        self.assertEqual(submitted['lead_submission']['disposition'], 'created')
        intent = self.g.artifacts.read_json('intent-'+plan['plan_id'])
        self.assertEqual(intent['evidence_ids'], [service.recent()[0]['artifact_id']])
        self.assertEqual(self.g.budget.calls, 2)

    def test_local_tuning_requires_authorization_and_respects_axes_count_and_time(self):
        self.c.llm_available = lambda: True
        self.c.resources = lambda: {'cpu_threads': 1, 'memory_limit_mb': 1024, 'timeout_seconds': 30}
        local = LocalSearch(self.c)
        ticket = self.h.issue()
        plan = self.h.proposal(ticket)['proposal']['plan']
        local.start_series('researcher-1', self.h.root, plan)
        self.assertFalse(local.has_series('researcher-1', self.h.root))
        request = {'parameters': ['alpha'], 'max_trials': 3, 'budget_seconds': 30}
        local.start_series('researcher-1', self.h.root, plan, request=request)
        for _ in range(3):
            recipe, params = local.next_choice('researcher-1', self.h.root, plan)
            self.assertEqual(recipe, plan['recipe_id'])
            self.assertEqual([k for k in params if params[k] != plan['model']['hyperparameters'][k]], ['alpha'])
        self.assertIsNone(local.next_choice('researcher-1', self.h.root, plan))
        local.start_series('researcher-1', self.h.root, plan, request=request)
        local.authorizations[(self.h.root, 'researcher-1')]['deadline'] = time.monotonic()-1
        self.assertFalse(local.has_series('researcher-1', self.h.root))

    def test_measured_feature_does_not_force_an_automatic_branch(self):
        fork, _, _ = self.h.make_fork()
        cp = self.g.invoke('G01', {'branch_id': self.h.root, 'expected_branch_revision': self.g.branch(self.h.root)['branch_revision'], 'reason': 'manual'})['checkpoint']
        decision = self.g.invoke('G14', {'checkpoint_id': cp['checkpoint_id'], 'budget': self.g.budget.snapshot(), 'plateau_assessment_id': None})['decision']
        self.assertEqual(decision['action'], 'CONTINUE')
        self.assertEqual(self.g.branch(self.h.root)['constraints'], {'required_capability_ids': [], 'forbidden_capability_ids': []})
        # Deliberately requested controlled branches still enforce their contract.
        out = self.g.invoke('G02', fork)
        self.assertIn('feature.target_transform', out['accept_branch']['constraints']['required_capability_ids'])

    def test_api_unavailability_uses_owned_fallback_without_authorizing_a_series(self):
        self.c.llm_available = lambda: False
        self.c.resources = lambda: {'cpu_threads': 1, 'memory_limit_mb': 1024, 'timeout_seconds': 30}
        local = LocalSearch(self.c)
        recipe, params = local.next_choice('researcher-1', self.h.root)
        self.assertNotEqual(recipe, CUSTOM_RECIPE)
        self.g.catalog.normalize(recipe, params)
        self.assertFalse(local.has_series('researcher-1', self.h.root))

    def test_custom_source_is_checked_and_parameters_are_not_limited_to_a_recipe(self):
        self.assertEqual(self.g.catalog.normalize(CUSTOM_RECIPE, {'degree': 3, 'columns': ['a', 'b']}), {'degree': 3, 'columns': ['a', 'b']})
        with self.assertRaises(AgentError):
            self.g.catalog.normalize(CUSTOM_RECIPE, {'_seed': 0})
        with self.assertRaises(AgentError):
            self.g.catalog.normalize(CUSTOM_RECIPE, {'x': float('nan')})
        with self.assertRaises(AgentError):
            validate_code(SOURCE.replace('return Pipeline', 'pd.read_csv("train.csv")\n    return Pipeline'))
        validate_code(SOURCE)
        with self.assertRaises(AgentError):
            validate_action(action('tune', tuning={'parameters': ['invented'], 'max_trials': 3, 'budget_seconds': 10}), self.g.catalog)
        self.assertEqual(self.g.schemas.schema['$defs']['ResearchAction'], ACTION_SCHEMA)
        self.assertEqual(self.g.schemas.schema['$defs']['AnalysisQuery'], ANALYSIS_SCHEMA)


class AdaptiveIntegrationTests(unittest.TestCase):
    def test_custom_pipeline_train_publish_and_reconstruct_refit(self):
        self.check_custom_pipeline('regression')

    def test_custom_classifier_train_publish_and_reconstruct_refit(self):
        self.check_custom_pipeline('classification')

    def check_custom_pipeline(self, kind):
        original = MockTransport.complete
        def scripted(transport, request, timeout, context):
            if context['metadata']['schema_id'] != 'ResearchAction-v1':
                return original(transport, request, timeout, context)
            peer = context['metadata']['researcher_id']
            payload = (action(hyperparameters={'alpha': .03 if peer.endswith('1') else .05})
                       if context['metadata']['round_no'] == 0 else action('stop', hyperparameters={}))
            return {'choices': [{'message': {'role': 'assistant', 'content': json.dumps(payload)}, 'finish_reason': 'stop'}],
                    'usage': {'prompt_tokens': 100, 'completion_tokens': 100}}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp);task = make_demo(root/'task', kind, rows=120, test_rows=16)
            settings = {'LLM_MODE': 'mock', 'RESEARCH_MODE': 'adaptive', 'TIME_BUDGET_SECONDS': '120',
                # This fixture tests reconstruction and explicit stop, independently of timing policy.
                'DATA_RESEARCH_MIN_SECONDS': '0',
                'GRAPH_ENABLED': '1', 'GRAPH_REFINEMENT_ENABLED': '0', 'CATALOG_WARMUP_EXPERIMENTS': '0',
                'LLM_MAX_CALLS': '12', 'LOG_LEVEL': 'ERROR', 'MODEL_SNAPSHOT_MAX_BYTES': '0', 'GRAPH_MIN_FREE_DISK_BYTES': '0'}
            with patch.dict(os.environ, settings), patch.object(MockTransport, 'complete', scripted), redirect_stderr(io.StringIO()) as stderr:
                result = main([str(task), '--llm-mode', 'mock', '--work-dir', str(root/'runs')])
            self.assertEqual(result, 0, stderr.getvalue()[-7000:])
            run = next((root/'runs').glob('run-*'))
            events = [json.loads(l) for l in (run/'events.jsonl').read_text().splitlines()]
            self.assertTrue(any(e['event'] == 'research.action_admitted' and e['action'] == 'pipeline' for e in events))
            self.assertFalse(any(e['event'] == 'search.local_trial' for e in events))
            self.assertFalse(any(e['event'] == 'research.proposal_failed' for e in events))
            db = sqlite3.connect(run/'state.sqlite')
            champion = json.loads(db.execute('SELECT record FROM publications').fetchone()[0])['experiment_id']
            plan = json.loads(db.execute('SELECT plan FROM experiments WHERE experiment_id=?', (champion,)).fetchone()[0]);db.close()
            errors = '\n'.join(p.read_text()[-3000:] for p in run.glob('experiments/*/*/stderr.log'))
            self.assertEqual(plan['model']['family'], 'custom', errors[-7000:])
            rebuilt = root/'rebuilt'
            reconstruct(run, champion, rebuilt, task.parent)
            job = json.loads((rebuilt/'refit-job.json').read_text())
            self.assertTrue(job['guard_generated'])
            self.assertEqual((rebuilt/'candidate.py').read_text(), SOURCE)
            done = subprocess.run([sys.executable, str(rebuilt/'run_refit.py')], capture_output=True, text=True, timeout=40)
            self.assertEqual(done.returncode, 0, done.stderr[-4000:])
            import pandas as pd
            original_frame = pd.read_csv(task.parent/'submission.csv')
            rebuilt_frame = pd.read_csv(rebuilt/'candidate.csv')
            pd.testing.assert_frame_equal(original_frame, rebuilt_frame)
            self.assertEqual((task.parent/'submission.csv').read_bytes(), (rebuilt/'candidate.csv').read_bytes())

    def test_io_guard_denies_dataset_access_but_allows_installed_imports(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp);source = root/'candidate.py';source.write_text('')
            secret = root/'train.csv';secret.write_text('private validation labels')
            code = '''from tools.runtime_guard import PipelineGuard
from pathlib import Path
import sys
guard=PipelineGuard(sys.argv[1]);guard.active=True
try:
    import sklearn.preprocessing
    print('imported')
    Path(sys.argv[2]).read_text()
except PermissionError:
    guard.active=False
    print('blocked')
else:
    guard.active=False
    raise RuntimeError('dataset access was allowed')
'''
            done = subprocess.run([sys.executable, '-c', code, str(source), str(secret)], cwd=ROOT, capture_output=True, text=True, timeout=30)
            self.assertEqual(done.returncode, 0, done.stderr[-3000:])
            self.assertIn('blocked', done.stdout)
            self.assertIn('imported', done.stdout)


if __name__ == '__main__':
    unittest.main()
