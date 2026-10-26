"""Real-sized diagnostic audits must not block the first research API call."""
from copy import deepcopy
import json
from types import SimpleNamespace
import unittest

from agent.context import output_schemas
from agent.context_budget import request_bytes
from agent.context_details import read_details
from agent.errors import AgentError
from agent.fit_diagnostics import compact_report
from agent.llm.gateway import Gateway
from agent.pipeline_search import VersionSearch
from agent.research_execution import PeerRecovery
from agent.research_feedback import FeedbackMailbox, compact_updates
from agent.researcher import Researcher
from tests.graph_helpers import GraphHarness


def large_report():
    return {'analysis_execution': 'leased_oof_worker', 'scope': 'fit_only',
            'discovery_fit_row_ids': list(range(100000000, 100005133)),
            'check_row_ids': list(range(100005133, 100010000)),
            'rows': 8998, 'hard_groups': [], 'diagnostic_groups': [],
            'error_slices': {'metric_id': 'rmse', 'oof_metric': 59.799,
                'slices': [{'slice_id': 'H1', 'rule': '__history_count > 0.5',
                    'rows': 1422, 'row_share': .158, 'loss_share': .103,
                    'verdict': 'fixable', 'remedies': [{'remedy_ref': 'report#H1/blend_history',
                        'action': 'blend_history', 'oof_metric_before': 59.799,
                        'oof_metric_after': 58.6272, 'robust': True}], 'examples': []}]}}


class LiveContextRegressionTests(unittest.TestCase):
    def test_projection_is_constant_size_and_preserves_cards_and_audit_source(self):
        full = large_report()
        original = deepcopy(full)
        short = compact_report(full, source_artifact_id='report')
        self.assertLess(request_bytes(short), 2500)
        self.assertEqual(short['discovery_fit_row_count'], 5133)
        self.assertEqual(short['check_row_count'], 4867)
        self.assertEqual(short['error_slices']['slices'][0]['remedies'][0]['remedy_ref'], 'report#H1/blend_history')
        self.assertEqual(short, compact_report(short, source_artifact_id='report'))
        self.assertEqual(full, original)
        self.assertGreater(request_bytes(full), 100000)

    def test_first_research_call_with_two_full_diagnostics_reaches_transport_for_both_tasks(self):
        for kind in ('regression', 'classification'):
            with self.subTest(kind=kind):
                h = GraphHarness(kind, research_mode='adaptive', background_tuning_enabled=True,
                    time_budget_seconds=10800, graph_hard_decisions=False,
                    graph_context_bytes=196608, llm_request_max_bytes=196608,
                    llm_max_output_tokens=32768, llm_max_output_tokens_limit=65536)
                try:
                    g = h.g
                    h.result('researcher-1', {}, 10 if kind == 'regression' else .6)
                    full = large_report()
                    art = g.artifacts.write_json('diagnostics/real-sized.json', full, 'fit_diagnostics')
                    record = {'artifact_id': art['artifact_id'], 'report': full}
                    g.context.science = SimpleNamespace(context=lambda *_: {'fit_diagnostics': [record, record]})
                    recipe = 'linear-regressor-v1' if kind == 'regression' else 'linear-classifier-v1'
                    class Transport:
                        def __init__(self): self.requests = []
                        def complete(self, request, timeout, context):
                            self.requests.append(deepcopy(request))
                            action = {'action': 'features', 'recipe_id': recipe, 'hyperparameters': {},
                                'feature_program': [{'op': 'product', 'name': 'joint', 'columns': ['x_numeric', 'z_numeric']}],
                                'hypothesis': 'Measure a coordinated representation change.', 'evidence_ids': []}
                            return {'choices': [{'message': {'content': json.dumps(action)}, 'finish_reason': 'stop'}]}
                    transport = Transport()
                    gateway = Gateway(g.config, g.budget, g.artifacts, g.bus, g.log, output_schemas(g.bus), transport)
                    gateway.analysis = SimpleNamespace()
                    gateway.feedback = FeedbackMailbox(g.artifacts)
                    update = gateway.feedback.publish(h.root, {
                        'branch_id': h.root, 'dataset_id': g.spec['dataset_id'], 'task_spec_id': g.spec['task_spec_id'],
                        'validation_id': g.validation['validation_id'], 'fit_diagnostics': [record, record], 'results': [], 'series': []})
                    self.assertGreater(request_bytes(update), 220000)
                    self.assertLess(request_bytes(compact_updates(update)), 7000)
                    researcher = Researcher(g.bus, gateway, g.artifacts, g.spec, g.catalog, g.config, g.log)
                    researcher.validation_id = g.validation['validation_id']
                    ticket = h.issue()
                    context = g.artifacts.read_json(ticket['work_order']['context_id'])
                    # Exact audit rows remain available under the same scope.
                    page = read_details({'reference_id': context['metadata']['detail_artifact_id'],
                        'section': 'fit_diagnostics/0', 'limit_bytes': 6000}, context, g.artifacts, g.log)
                    self.assertIn('check_row_ids', page['text'])
                    snapshot = g.artifacts.read_json(context['metadata']['detail_artifact_id'])
                    self.assertEqual(snapshot['sections']['fit_diagnostics/0']['report'], full)
                    proposal = g.bus.call('C09', ticket['researcher_request']['payload'], researcher.propose)['data']
                    self.assertEqual(proposal['decision'], 'propose')
                    self.assertEqual(len(transport.requests), 1)
                    request = transport.requests[0]
                    self.assertLess(request_bytes(request), 180224)
                    self.assertNotIn('discovery_fit_row_ids', json.dumps(request))
                    self.assertIn('report#H1/blend_history', json.dumps(request))
                    self.assertFalse(g.budget.llm_dialogues)
                    self.assertEqual(g.artifacts.read_json(art['artifact_id']), full)
                finally:
                    h.close()

    def test_initial_irreducible_context_is_quarantined_without_fake_model_checkpoint(self):
        h = GraphHarness('regression', research_mode='adaptive', background_tuning_enabled=True,
                         time_budget_seconds=10800)
        try:
            g = h.g
            gateway = Gateway(g.config, g.budget, g.artifacts, g.bus, g.log, output_schemas(g.bus), SimpleNamespace())
            gateway.analysis = SimpleNamespace()
            researcher = Researcher(g.bus, gateway, g.artifacts, g.spec, g.catalog, g.config, g.log)
            def fail(*args, **kwargs):
                raise AgentError('CONTEXT_TOO_LARGE', 'Mandatory input cannot fit')
            gateway._request = fail
            with self.assertRaises(AgentError) as caught:
                researcher.propose(h.issue()['researcher_request']['payload'])
            self.assertEqual(caught.exception.code, 'CONTEXT_TOO_LARGE')
            self.assertEqual(g.budget.calls, 0)
            self.assertFalse(g.budget.llm_dialogues)
            self.assertFalse(any(v['meta']['kind'] == 'controller_research_checkpoint' for v in g.artifacts.entries.values()))
        finally:
            h.close()

    def test_packing_failure_cannot_wake_itself_but_new_evidence_can(self):
        now = [0.]
        recovery = PeerRecovery(clock=lambda: now[0])
        c = SimpleNamespace(science=None, research_leaders={'b': 'base'}, research_feedback=[])
        service = VersionSearch.__new__(VersionSearch)
        service.c, service.versions = c, {}
        generation = service.research_generation('b')
        recovery.failure('one', AgentError('CONTEXT_TOO_LARGE', 'Unchanged packing'), generation)
        c.research_feedback.append({'branch_id': 'b', 'code': 'CONTEXT_TOO_LARGE', 'message': 'Unchanged packing'})
        self.assertEqual(service.research_generation('b'), generation)
        now[0] = 100000
        self.assertFalse(recovery.ready('one', generation))
        self.assertTrue(recovery.ready('other', generation))
        c.research_leaders['b'] = 'new-base'
        self.assertTrue(recovery.ready('one', service.research_generation('b')))

    def test_controller_checkpoint_preserves_question_across_repeated_tool_failures(self):
        h = GraphHarness('regression', research_mode='adaptive', background_tuning_enabled=True)
        try:
            g = h.g
            gateway = Gateway(g.config, g.budget, g.artifacts, g.bus, g.log, output_schemas(g.bus), SimpleNamespace())
            question = 'Can the frozen history correction improve this measured base?'
            for _ in range(20):
                context = {'messages': [{}, {'content': json.dumps({'profile_summary': {
                    'error_research': {'recommendation': {'question': question}}}})}]}
                answer = gateway._research_checkpoint(context, [], AgentError('CONTEXT_TOO_LARGE', 'pressure'), 1)
                self.assertEqual(answer['parsed_payload']['hypothesis'], question)
                self.assertEqual(answer['parsed_payload']['next_step'], 'wait')
                question = answer['parsed_payload']['hypothesis']
        finally:
            h.close()
