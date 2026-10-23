"""Contract/freeze integration with a synthetic transport; no estimator fitting."""
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from agent.context import output_schemas
from agent.errors import AgentError
from agent.feature_schema import FACTORY_SOURCE
from agent.llm.gateway import Gateway
from agent.researcher import Researcher
from agent.runner import frozen_execution
from tests.graph_helpers import GraphHarness
from tests.test_adaptive_research import action
from tools.analysis import AnalysisService
from tools.runtime_guard import prepare_pipeline_imports


class FeatureAdmissionTests(unittest.TestCase):
    def test_owned_backend_is_preloaded_without_executing_the_feature_source(self):
        with TemporaryDirectory() as directory:
            path = Path(directory)/'features.py'
            path.write_text(FACTORY_SOURCE)
            imported = []
            with patch('importlib.import_module', side_effect=lambda name: imported.append(name)):
                prepare_pipeline_imports(path, 'catboost-regressor-v1')
            self.assertIn('catboost', imported)
            self.assertIn('agent.feature_engineering', imported)

    def setUp(self):
        self.h = GraphHarness('regression', research_mode='adaptive', feature_research_enabled=True,
                              time_budget_seconds=10800, llm_max_calls=None, llm_max_total_output_tokens=None)

    def tearDown(self):
        self.h.close()

    def propose(self, column):
        g = self.h.g
        proposed = action('features', feature_program=[{'op': 'missing', 'name': 'missing_feature', 'columns': [column]}])
        class Transport:
            def complete(self, request, timeout, context):
                return {'choices': [{'message': {'content': json.dumps(proposed)}, 'finish_reason': 'stop'}],
                        'usage': {'completion_tokens': 100}}
        gateway = Gateway(g.config, g.budget, g.artifacts, g.bus, g.log, output_schemas(g.bus), Transport())
        gateway.analysis = AnalysisService(g.spec, g.validation, g.artifacts, g.budget, g.log)
        researcher = Researcher(g.bus, gateway, g.artifacts, g.spec, g.catalog, g.config, g.log)
        researcher.validation_id = g.validation['validation_id']
        ticket = self.h.issue()
        return researcher.propose(ticket['researcher_request']['payload'])['plan']

    def test_feature_action_compiles_the_exact_owned_source_and_frozen_dependencies(self):
        g = self.h.g
        plan = self.propose('x_numeric')
        self.assertEqual(plan['recipe_id'], 'python-pipeline-v1')
        self.assertEqual(plan['model']['hyperparameters']['base_recipe_id'], 'linear-regressor-v1')
        self.assertEqual(plan['model']['hyperparameters']['model_parameters']['alpha'], .03)
        self.assertEqual(g.artifacts.resolve(plan['code_artifact_id']).read_text(), FACTORY_SOURCE)
        record = g.store.reserve(plan, g.artifacts.meta(plan['code_artifact_id'])['sha256'])
        exp = g.store.get(record['experiment_id'])
        frozen = frozen_execution(g.artifacts, exp, g.spec, 'train')
        self.assertIn('agent/feature_engineering.py', frozen['runtime_sha256'])
        self.assertIn('agent/feature_schema.py', frozen['runtime_sha256'])
        self.assertEqual(frozen_execution(g.artifacts, exp, g.spec, 'refit'), frozen)

    def test_feature_action_rejects_target_input_before_training(self):
        with self.assertRaises(AgentError) as error:
            self.propose(self.h.g.spec['target_column'])
        self.assertEqual(error.exception.code, 'INVALID_FEATURE_PROGRAM')
