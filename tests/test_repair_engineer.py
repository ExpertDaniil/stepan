"""Protect research intent while isolating interface-only repair."""
from copy import deepcopy
import json
import unittest

from agent.llm.repair_engineer import prepare, accept, messages
from tests.helpers import Harness


class EngineerContractTests(unittest.TestCase):
    def setUp(self):
        self.h = Harness()
        self.original = {'action': 'features', 'recipe_id': 'python-pipeline-v1',
                         'hyperparameters': {'alpha': 2},
                         'feature_program': [{'op': 'target_mean', 'name': 'f', 'columns': ['x']}],
                         'slice_remedies': ['report#one'], 'hypothesis': 'Keep the chosen experiment',
                         'code': None, 'tuning': None, 'feature_steps': [], 'evidence_ids': ['e1']}
        self.base = {'experiment_id': 'pinned-base', 'recipe_id': 'python-pipeline-v1',
                     'hyperparameters': {'base_recipe_id': 'linear-regressor-v1'}}

    def tearDown(self):
        self.h.close()

    def packet(self):
        return prepare(json.dumps(self.original), 'INVALID_RESEARCH_ACTION: features needs an owned recipe', {}, self.base)

    def reply(self, candidate):
        return json.dumps({'status': 'repaired', 'candidate': candidate, 'reason': 'Fix recipe interface only'})

    def test_repairs_owned_recipe_without_changing_experiment(self):
        p = self.packet()
        candidate = {**self.original, 'recipe_id': 'linear-regressor-v1'}
        self.assertEqual(accept(self.reply(candidate), p, self.h.bus.schemas._check), candidate)
        self.assertEqual(p['base_experiment_id'], 'pinned-base')
        self.assertEqual(self.original['recipe_id'], 'python-pipeline-v1')
        self.assertEqual(len(messages(p)), 2)

    def test_rejects_changes_to_every_scientific_field_and_omissions(self):
        p = self.packet()
        changes = {'action': 'catalog', 'hyperparameters': {'alpha': 3}, 'feature_program': [],
                   'slice_remedies': [], 'hypothesis': 'Another experiment', 'evidence_ids': [],
                   'code': 'new source', 'unknown': 1, 'recipe_id': 'catboost-regressor-v1'}
        for key, value in changes.items():
            with self.subTest(key=key), self.assertRaises(ValueError):
                accept(self.reply({**self.original, key: value}), p, self.h.bus.schemas._check)
        candidate = {**self.original, 'recipe_id': 'linear-regressor-v1'}
        del candidate['hyperparameters']
        with self.assertRaises(ValueError):
            accept(self.reply(candidate), p, self.h.bus.schemas._check)

    def test_conflicting_remedies_and_unreadable_json_use_existing_path(self):
        self.assertIsNone(prepare(json.dumps(self.original), 'INVALID_SLICE_REMEDY: 1..4 corrections', {}, self.base))
        self.assertIsNone(prepare('{broken', 'bad json', {}, self.base))
        without_base = {**self.original, 'code': None, 'tuning': None, 'feature_steps': []}
        self.assertIsNone(prepare(json.dumps(without_base), 'bad recipe', {}, None))

    def test_nonempty_code_and_parameter_search_are_never_discarded(self):
        original = {**self.original, 'code': 'def make_pipeline(): pass', 'tuning': {'search_space': {'a': 1}}}
        p = prepare(json.dumps(original), 'incompatible fields', {}, self.base)
        self.assertNotIn('code', p['allowed_edits'])
        self.assertNotIn('tuning', p['allowed_edits'])

    def test_large_source_is_skipped_without_truncation(self):
        value = {**self.original, 'code': 'x'*40000}
        self.assertIsNone(prepare(json.dumps(value), 'bad interface', {}, self.base))

    def test_refusal_is_explicit_and_does_not_invent_a_candidate(self):
        p = self.packet()
        content = json.dumps({'status': 'needs_researcher', 'candidate': None, 'reason': 'Ambiguous intention'})
        self.assertIsNone(accept(content, p, self.h.bus.schemas._check))
        with self.assertRaises(ValueError):
            accept(self.reply(self.original), p, self.h.bus.schemas._check)


class EngineerGatewayTests(unittest.TestCase):
    def run_case(self, mode='repair', enabled=True, calls=8):
        from types import SimpleNamespace
        from agent.context import output_schemas
        from agent.errors import AgentError
        from agent.llm.gateway import Gateway
        from agent.researcher import Researcher
        from tests.graph_helpers import GraphHarness
        h = GraphHarness('regression', research_mode='adaptive', background_tuning_enabled=True,
                         graph_hard_decisions=False, time_budget_seconds=10800, llm_max_calls=calls,
                         llm_repair_engineer_enabled=enabled)
        self.addCleanup(h.close)
        h.result('researcher-1', {'alpha': 1}, 10)
        g = h.g
        bad = {'action': 'features', 'recipe_id': 'python-pipeline-v1', 'hyperparameters': {'alpha': 1},
               'feature_program': [{'op': 'product', 'name': 'joint_signal', 'columns': ['x_numeric', 'z_numeric']}],
               'code': None, 'tuning': None, 'feature_steps': [], 'evidence_ids': [],
               'hypothesis': 'Measure the product feature on the chosen linear base.'}
        class Transport:
            def __init__(self): self.requests = []; self.timeouts = []
            def complete(self, request, timeout, context):
                self.requests.append(deepcopy(request)); self.timeouts.append(timeout)
                if context['metadata']['schema_id'] == 'RepairEngineer-v1':
                    if mode == 'timeout':
                        raise AgentError('LLM_UNAVAILABLE', 'simulated engineer timeout', retryable=True)
                    packet = json.loads(request['messages'][1]['content'])
                    candidate = {**packet['original'], 'recipe_id': packet['allowed_edits']['recipe_id']}
                    if mode == 'mutate': candidate['hyperparameters'] = {'alpha': 999}
                    value = ({'status': 'needs_researcher', 'candidate': None, 'reason': 'Ambiguous'} if mode == 'decline'
                             else {'status': 'repaired', 'candidate': candidate, 'reason': 'Use the pinned owned recipe'})
                elif len(self.requests) == 1:
                    value = ({**bad, 'recipe_id': 'linear-regressor-v1'} if mode == 'valid' else bad)
                else:
                    value = {'action': 'research', 'hypothesis': 'Continue with the saved observation.', 'evidence_ids': []}
                return {'choices': [{'message': {'content': json.dumps(value)}, 'finish_reason': 'stop'}],
                        'usage': {'prompt_tokens': 100, 'completion_tokens': 50}}
        transport = Transport()
        gateway = Gateway(g.config, g.budget, g.artifacts, g.bus, g.log, output_schemas(g.bus), transport)
        gateway.analysis = SimpleNamespace()
        researcher = Researcher(g.bus, gateway, g.artifacts, g.spec, g.catalog, g.config, g.log)
        ticket = h.issue()
        result = researcher.propose(ticket['researcher_request']['payload'])
        events = [json.loads(line) for line in (h.h.run/'events.jsonl').read_text().splitlines()]
        self.assertFalse(g.budget.llm_dialogues)
        self.assertEqual(g.budget.token_snapshot()['output_reserved'], 0)
        return result, transport, events, g, bad

    def test_real_validator_accepts_repair_in_existing_reserved_attempt(self):
        result, transport, events, g, bad = self.run_case(calls=3)
        self.assertEqual(result['decision'], 'propose')
        self.assertEqual(len(transport.requests), 2)
        packet = json.loads(transport.requests[1]['messages'][1]['content'])
        self.assertEqual(packet['original']['feature_program'], bad['feature_program'])
        self.assertEqual(result['plan']['model']['hyperparameters']['feature_program'], bad['feature_program'])
        self.assertEqual(g.budget.calls, 2)
        self.assertNotIn('tools', transport.requests[1])
        self.assertEqual(len(transport.requests[1]['messages']), 2)
        self.assertLessEqual(transport.timeouts[1], 90)
        self.assertIn('llm.repair_engineer_accepted', [e['event'] for e in events])

    def test_refusal_mutation_and_transport_failure_resume_original_research(self):
        for mode in ('decline', 'mutate', 'timeout'):
            with self.subTest(mode=mode):
                result, transport, events, g, _ = self.run_case(mode)
                self.assertEqual(result['decision'], 'research')
                self.assertEqual(len(transport.requests), 3)
                self.assertEqual(g.budget.calls, 3)
                self.assertIn('llm.repair_engineer_fallback', [e['event'] for e in events])
                self.assertNotIn('llm.repair_engineer_accepted', [e['event'] for e in events])
                self.assertGreater(len(transport.requests[2]['messages']), 2)

    def test_last_funded_repair_turn_stays_with_the_researcher(self):
        result, transport, events, g, _ = self.run_case('decline', calls=2)
        self.assertEqual(result['decision'], 'research')
        self.assertEqual(result['stop_reason'], 'Continue with the saved observation.')
        self.assertEqual(len(transport.requests), 2)
        self.assertNotIn('llm.repair_engineer_started', [e['event'] for e in events])

    def test_valid_actions_and_disabled_feature_have_no_extra_engineer(self):
        for mode, enabled in (('valid', True), ('repair', False)):
            with self.subTest(mode=mode):
                result, transport, events, _, _ = self.run_case(mode, enabled)
                self.assertEqual(len(transport.requests), 1 if mode == 'valid' else 2)
                self.assertNotIn('llm.repair_engineer_started', [e['event'] for e in events])
