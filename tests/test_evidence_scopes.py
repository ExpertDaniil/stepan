from copy import deepcopy
import hashlib
from types import SimpleNamespace
import unittest

from agent.catalog import factory_source
from agent.evidence import assess_claims, compare


class EvidenceScopeTests(unittest.TestCase):
    def fixture(self):
        plan = {'recipe_id': 'hist-gradient-regressor-v1', 'task_spec_id': 'task', 'dataset_id': 'data',
                'validation_id': 'split', 'seed': 42, 'fidelity': 'full', 'code_artifact_id': 'code',
                'feature_steps': [], 'model': {'family': 'hist_gradient', 'hyperparameters': {'max_iter': 300, 'learning_rate': .1}}}
        plans = {'left': plan, 'right': deepcopy(plan)}
        metrics = {eid: {'dataset_id': 'data', 'validation_id': 'split', 'fidelity': 'full',
                        'metric_id': 'rmse', 'direction': 'minimize', 'value': value}
                   for eid, value in [('left', 42.803634807122776), ('right', 44.679219416371964)]}
        store = SimpleNamespace(get=lambda eid: {'state': 'evaluated', 'plan': plans[eid]},
            experiment_evaluation=lambda eid: {'valid': True, 'violations': [], 'primary_metric': metrics[eid]})
        artifacts = SimpleNamespace(entries={'execution-left': {}, 'execution-right': {}},
            read_json=lambda eid: {'runtime_sha256': {'worker': 'same'}},
            meta=lambda eid: {'sha256': hashlib.sha256(factory_source(plan['recipe_id']).encode()).hexdigest() if eid == 'code' else 'changed'})
        return store, artifacts, plans

    def test_joint_changes_cannot_identify_a_component(self):
        store, artifacts, plans = self.fixture()
        plans['right']['model']['hyperparameters'].update(max_iter=900, learning_rate=.03)
        result = compare(store, artifacts, 'left', 'right')
        self.assertEqual(result['supported_scope'], 'complete_configuration_comparison')
        self.assertAlmostEqual(result['right_improvement'], -1.875584609249188)
        claims = [{'scope': 'component', 'component': 'parameter:max_iter',
                   'statement': 'This parameter always harms.', 'evidence_ids': ['left', 'right']}]
        assessed = assess_claims(store, artifacts, claims, ['left', 'right'])[0]
        self.assertEqual(assessed['status'], 'unverified')
        self.assertFalse(assessed['model_statement_verified'])

    def test_single_parameter_contrast_has_explicit_local_scope(self):
        store, artifacts, plans = self.fixture()
        plans['right']['model']['hyperparameters']['max_iter'] = 900
        result = compare(store, artifacts, 'left', 'right')
        self.assertEqual(result['supported_scope'], 'single_parameter_contrast')
        self.assertFalse(result['component_effect_identified'])
        claim = {'scope': 'component', 'component': 'parameter:max_iter',
                 'statement': 'Local observation.', 'evidence_ids': ['left', 'right']}
        self.assertEqual(assess_claims(store, artifacts, [claim], ['left', 'right'])[0]['status'], 'limited_parameter_contrast')
        artifacts.entries.clear()
        self.assertEqual(compare(store, artifacts, 'left', 'right')['supported_scope'], 'complete_configuration_comparison')

    def test_source_and_protocol_changes_limit_comparisons(self):
        store, artifacts, plans = self.fixture()
        plans['right']['code_artifact_id'] = 'different-source'
        self.assertTrue(compare(store, artifacts, 'left', 'right')['source_changed'])
        plans['right']['seed'] = 99
        self.assertFalse(compare(store, artifacts, 'left', 'right')['comparable'])

    def test_family_exhaustion_and_invisible_evidence_never_become_facts(self):
        store, artifacts, _ = self.fixture()
        claim = {'scope': 'family', 'component': None, 'statement': 'Twelve trials exhausted the family.',
                 'evidence_ids': ['left', 'right']}
        result = assess_claims(store, artifacts, [claim], ['left', 'right'])[0]
        self.assertEqual(result['status'], 'unverified')
        self.assertIsNone(result['authoritative_statement'])
        result = assess_claims(store, artifacts, [{**claim, 'scope': 'configuration'}], ['left'])[0]
        self.assertEqual(result['status'], 'unverified')
