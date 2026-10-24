"""Synthetic regressions, written but NOT executed during the implementation session."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from agent.context import output_schemas
from agent.context_budget import effective_limit, request_bytes
from agent.context_details import read_details, scope_for
from agent.contracts import canonical, Schemas
from agent.errors import AgentError
from agent.evaluation import error_diagnostics
from agent.feature_engineering import FeatureProgram
from agent.feature_schema import operation_facts
from agent.llm.gateway import Gateway, OpenAITransport
from agent.ml import NumericInteractions
from agent.research_feedback import compact_memory
from agent.research_memory import semantic_projection, validate_semantic_updates
from tests.graph_helpers import GraphHarness
from tests.test_adaptive_research import action
from tests import test_analysis_access as analysis_fixture
from tests.test_analysis_access import query


def update(statement='Scale is unknown', **changes):
    value = {'topic': 'quantity scale', 'kind': 'hypothesis', 'statement': statement,
             'columns': ['quantity'], 'evidence_ids': [], 'properties': {'units': 'unknown'}}
    value.update(changes)
    return value


def large_memory():
    return {'decisions': [{'decision_artifact_id': 'decision-'+str(i), 'action': 'catalog',
        'question': 'quantity response '+('explanation '*5000), 'decision_reason': 'words '*10000,
        'observations': [{'experiment_id': 'base' if i == 0 else 'e-'+str(i), 'state': 'evaluated',
                          'primary_metric': {'value': i}, 'diagnostics_artifact_id': None}],
        'evidence_ids': [], 'tuning_series': [], 'idea_id': None} for i in range(10)],
        'open_ideas': []}


class SemanticProjectionTests(unittest.TestCase):
    def test_long_prose_is_bounded_and_base_relevance_beats_recency(self):
        memory = large_memory()
        projected = compact_memory(memory, base_experiment_id='base', max_bytes=6000)
        self.assertLessEqual(len(canonical(projected).encode()), 6000)
        self.assertEqual(projected['decisions'][0]['decision_artifact_id'], 'decision-0')
        self.assertGreater(len(memory['decisions'][0]['decision_reason']), 10000)

    def test_model_cannot_verify_prose_by_attaching_an_existing_link(self):
        scope = {'run_id': 'r', 'dataset_id': 'd', 'validation_id': 'v'}
        entries = semantic_projection([('a', {'semantic_scope': scope,
            'semantic_updates': [update(kind='observation', evidence_ids=['actual-analysis'])]})])
        self.assertEqual(entries[0]['verification'], 'model_statement_unverified')
        self.assertEqual(entries[0]['predictive_usefulness'], 'not_established')
        with self.assertRaises(AgentError):
            validate_semantic_updates([update(evidence_ids=['invented'])], {}, set(), ['quantity'], scope)

    def test_stale_correction_marks_conflict_without_overwriting_fresh_content(self):
        scope = {'run_id': 'r', 'dataset_id': 'd', 'validation_id': 'v'}
        first = ('a', {'semantic_scope': scope, 'semantic_updates': [update()]})
        old_id = semantic_projection([first])[0]['record_id']
        fresh = ('b', {'semantic_scope': scope, 'semantic_updates': [update('New evidence', supersedes=old_id)]})
        stale = ('c', {'semantic_scope': scope, 'semantic_updates': [update('Old response', supersedes=old_id)]})
        records = semantic_projection([first, fresh, stale])
        self.assertEqual(records[0]['status'], 'superseded')
        self.assertEqual(records[1]['content']['statement'], 'New evidence')
        self.assertEqual(records[1]['status'], 'conflicted')
        self.assertEqual(records[2]['conflicts_with'], records[1]['record_id'])

    def test_task_validation_and_legacy_records_do_not_mix(self):
        records = [('legacy', {}), ('a', {'semantic_scope': {'dataset_id': 'other', 'validation_id': 'v'},
                    'semantic_updates': [update()]})]
        self.assertEqual(semantic_projection(records, dataset_id='current', validation_id='v'), [])


class FrozenContextTests(unittest.TestCase):
    def setUp(self):
        self.h = GraphHarness('regression', research_mode='adaptive', time_budget_seconds=10800,
                              llm_max_calls=None, llm_max_total_output_tokens=None)

    def tearDown(self):
        self.h.close()

    def context(self):
        ticket = self.h.issue()
        return self.h.g.artifacts.read_json(ticket['work_order']['context_id'])

    def test_large_definition_survives_as_excerpt_with_reachable_full_source(self):
        g = self.h.g
        description = 'Размер/unknown ' * 8000
        g.context.profile['data_dictionary'] = [{'feature': 'quantity', 'description': description, 'units': 'unknown'}]
        context = self.context()
        user = json.loads(context['messages'][1]['content'])
        data = user['profile_summary']['data']
        self.assertEqual(data['relevant_definitions'][0]['definition']['units'], 'unknown')
        self.assertTrue(data['relevant_definitions'][0]['details_truncated'])
        ref = context['metadata']['detail_artifact_id']
        text, offset = '', 0
        while True:
            page = read_details({'reference_id': ref, 'section': 'dictionary/0', 'offset': offset,
                                 'limit_bytes': 1024}, context, g.artifacts, g.log)
            self.assertLessEqual(len(page['text'].encode()), 1024)
            text += page['text']
            if page['next_offset'] is None:
                break
            offset = page['next_offset']
        self.assertEqual(json.loads(text)['description'], description)

    def test_unlisted_reference_and_changed_identity_are_denied(self):
        g = self.h.g
        context = self.context()
        request = {'reference_id': 'some-unrelated-artifact', 'section': 'decisions'}
        with self.assertRaises(AgentError):
            read_details(request, context, g.artifacts, g.log)
        request['reference_id'] = context['metadata']['detail_artifact_id']
        context['metadata']['detail_scope']['validation_id'] = 'different-protocol'
        with self.assertRaises(AgentError):
            read_details(request, context, g.artifacts, g.log)

    def test_frozen_dictionary_does_not_read_later_mutations(self):
        g = self.h.g
        g.context.profile['data_dictionary'] = [{'feature': 'quantity', 'description': 'before fork'}]
        context = self.context()
        g.context.profile['data_dictionary'][0]['description'] = 'later ancestor information'
        page = read_details({'reference_id': context['metadata']['detail_artifact_id'], 'section': 'dictionary/0'},
                            context, g.artifacts, g.log)
        self.assertEqual(json.loads(page['text'])['description'], 'before fork')

    def test_catalogue_details_removed_from_display_remain_in_the_frozen_snapshot(self):
        g = self.h.g
        context = self.context()
        user = json.loads(context['messages'][1]['content'])
        page = read_details({'reference_id': context['metadata']['detail_artifact_id'],
                             'section': 'recipes', 'limit_bytes': 6000}, context, g.artifacts, g.log)
        snapshot = g.artifacts.read_json(context['metadata']['detail_artifact_id'])
        self.assertEqual(snapshot['sections']['recipes'], user['profile_summary']['recipes'])
        self.assertIn('parameters', page['text'])

    def test_serialized_budget_survives_tools_feedback_final_and_repair(self):
        g = self.h.g
        context = self.context()
        user = json.loads(context['messages'][1]['content'])
        user['profile_summary']['research_memory'] = large_memory()
        context['messages'][1]['content'] = json.dumps(user, ensure_ascii=False)
        art = g.artifacts.write_json('contexts/pressure.json', context, 'llm_context')
        requests = []
        class Analysis:
            def analyze(self, args, context=None):
                return {'artifact_id': 'analysis-pressure', 'report': {'scope': 'fit_only', 'description': 'текст '*300}}
        class Transport:
            def complete(self, request, timeout, context):
                requests.append(deepcopy(request))
                if len(requests) == 1:
                    message = {'tool_calls': [{'id': 'one', 'function': {'name': 'analyze_data',
                        'arguments': json.dumps(query(columns=['x_numeric']))}}]}
                elif len(requests) == 2:
                    message = {'content': '{invalid json'}
                else:
                    message = {'content': json.dumps(action('catalog'))}
                return {'choices': [{'message': message, 'finish_reason': 'stop'}], 'usage': {'completion_tokens': 100}}
        gateway = Gateway(g.config, g.budget, g.artifacts, g.bus, g.log, output_schemas(g.bus), Transport())
        gateway.analysis = Analysis()
        from agent.research_feedback import FeedbackMailbox
        gateway.feedback = FeedbackMailbox(g.artifacts)
        gateway.feedback.publish(self.h.root, {'branch_id': self.h.root, 'results': [], 'series': [],
            **{k: context['metadata']['detail_scope'][k] for k in ('dataset_id', 'task_spec_id', 'validation_id')}})
        gateway.complete({'purpose': 'research_proposal', 'context_artifact_id': art['artifact_id'],
            'response_schema_id': 'ResearchAction-v1', 'tool_schema_ids': ['analyze_data', 'read_memory'],
            'max_output_tokens': 1024, 'deadline_seconds': 300, 'attempt_limit': 6})
        self.assertGreaterEqual(len(requests), 3)
        for request in requests:
            limit, _ = effective_limit(g.config, context['metadata'], request['max_tokens'])
            self.assertLessEqual(request_bytes(request), limit)
            calls = [c['id'] for m in request['messages'] for c in m.get('tool_calls', [])]
            replies = [m['tool_call_id'] for m in request['messages'] if m['role'] == 'tool']
            self.assertEqual(calls, replies)
        self.assertNotIn('tools', requests[-1])

    def test_reader_cannot_follow_a_new_siblings_source(self):
        g = self.h.g
        fork, _, _ = self.h.make_fork()
        children = g.invoke('G02', fork)
        reject, accept = (children[k]['branch_id'] for k in ('reject_branch', 'accept_branch'))
        g.invoke('G08', {'branch_id': reject, 'expected_branch_revision': 0,
                        'action': 'resume', 'reason': 'Synthetic sibling visibility regression'})
        ticket = self.h.issue(branch=reject)
        reply = g.invoke('G04', self.h.proposal(ticket, {'alpha': .123}))
        self.h.commit(reply, 55)
        sibling = reply['lead_submission']['experiment_id']
        ticket = self.h.issue(branch=accept)
        context = g.artifacts.read_json(ticket['work_order']['context_id'])
        ref = context['metadata']['detail_artifact_id']
        with self.assertRaises(AgentError):
            read_details({'reference_id': ref, 'section': 'code/'+sibling}, context, g.artifacts, g.log)

    def test_pending_trial_reference_does_not_authorize_a_later_diagnostic(self):
        from tools.analysis import AnalysisService
        g = self.h.g
        context = self.context()
        document = g.artifacts.read_json(context['metadata']['detail_artifact_id'])
        document['sections']['decisions'] = [{'trials': [{'experiment_id': 'pending-later'}]}]
        art = g.artifacts.write_json('research/pending-fixture.json', document, 'research_context_details')
        context['metadata']['detail_artifact_id'] = art['artifact_id']
        service = AnalysisService(g.spec, g.validation, g.artifacts, g.budget, g.log)
        service.query_context = context
        service.error_provider = lambda *_: self.fail('Out-of-cut diagnostic source was consulted')
        with self.assertRaises(AgentError):
            service._errors(query('errors', experiment_id='pending-later'), ['x_numeric'], 0)

    def test_analysis_reservation_reduces_the_same_training_capacity(self):
        budget = self.h.g.budget
        before = budget.snapshot()
        with budget.analysis_slot():
            self.assertEqual(budget.snapshot()['cpu_threads_available'], before['cpu_threads_available']-1)
            self.assertEqual(budget.snapshot()['memory_mb_available'], before['memory_mb_available']-1024)
        self.assertEqual(budget.snapshot()['cpu_threads_available'], before['cpu_threads_available'])
        budget.leases = {'busy': {'lease': {'cpu_threads': 6, 'memory_limit_mb': 20480}}}
        try:
            with self.assertRaises(AgentError), budget.analysis_slot():
                pass
        finally:
            budget.leases.clear()

    def test_busy_analysis_defers_feature_seeds_without_stopping_search(self):
        from agent.feature_search import FeatureSearch
        g = self.h.g
        self.h.result('researcher-2', {'alpha': .1}, 3)
        def busy():
            raise AgentError('ANALYSIS_RESOURCES_BUSY', 'Synthetic active train leases')
        controller = self.h.controller
        controller.graph = g
        controller.config = replace(g.config, feature_research_enabled=True, feature_seed_budget_fraction=.15)
        controller.context.analysis = SimpleNamespace(feature_frame=busy)
        engine = FeatureSearch(controller)
        self.assertIsNone(engine.next_plan('researcher-1', self.h.root))
        self.assertNotIn(self.h.root, engine.banks)
        self.assertFalse(g.budget.search_stopped)

    def test_custom_feature_manifest_is_a_declaration_after_a_valid_score(self):
        from agent.feature_memory import archive
        from agent.catalog import factory_source
        g = self.h.g
        ticket = self.h.issue()
        payload = self.h.proposal(ticket, {'alpha': 1}, recipe='python-pipeline-v1',
                                  source=factory_source('linear-regressor-v1'))
        payload['proposal']['plan']['feature_steps'] = [{'step': 'model_declares_physical_meaning'}]
        payload['manifest'] = g.constructor.build(payload['proposal']['plan'])
        reply = g.invoke('G04', payload)
        self.h.commit(reply, 55)  # State-machine fixture; no fit or ML-quality claim.
        result = archive(g.store, g.artifacts, [reply['lead_submission']['experiment_id']])
        self.assertFalse(result['representations'][0]['feature_declaration_verified'])

    def test_diagnostic_service_uses_selection_and_never_confirmation_labels(self):
        from agent.predictions import validation_columns
        g = self.h.g
        evaluator = self.h.controller.evaluator
        target = g.spec['target_column']
        frame = pd.DataFrame({'synthetic_zone': ['ordinary']*30, target: [2.]*20+[999999.]*10})
        validation_id = 'synthetic-selection-only'
        evaluator.protocols[validation_id] = {'train_indices': list(range(10, 20)),
            'valid_indices': list(range(10))+list(range(20, 30)), 'selection_indices': list(range(10)),
            'confirmation_indices': list(range(20, 30))}
        columns = validation_columns(g.spec)
        predictions = pd.DataFrame({'__row_index': evaluator.protocols[validation_id]['valid_indices'],
                                    columns[0]: [0.]*20})
        path = g.artifacts.safe_path('predictions/synthetic-selection.csv')
        path.parent.mkdir(parents=True, exist_ok=True);predictions.to_csv(path, index=False)
        art = g.artifacts.register(path, 'validation_predictions', 'text/csv')
        evaluator.diagnostic_sources['synthetic-exp'] = {'predictions_artifact_id': art['artifact_id'],
            'validation_id': validation_id, 'diagnostics_artifact_id': 'synthetic-diagnostics', 'selection_sse': 40.}
        tables = SimpleNamespace(read=lambda *args, columns=None, **kwargs: frame[columns].copy())
        with patch('agent.evaluation.transport', return_value=tables):
            result = evaluator.diagnose(query('errors', columns=['synthetic_zone'], experiment_id='synthetic-exp'), lambda: None)
        self.assertEqual(result['sse'], 40.)
        self.assertEqual(result['rows'], 10)
        self.assertFalse(result['confirmation_labels_used'])
        self.assertNotIn('999999', canonical(result))

    def test_http_hook_rejects_actual_bytes_before_network(self):
        transport = OpenAITransport.__new__(OpenAITransport)
        transport.wire_context = SimpleNamespace(limit=10)
        transport.log = None
        with self.assertRaises(AgentError):
            transport._check_wire_request(SimpleNamespace(content=b'01234567890'))


class ExecutedFeatureTests(unittest.TestCase):
    def test_explicit_interaction_and_program_ignore_csv_column_order(self):
        frame = pd.DataFrame({'zeta': [2., 4.], 'quantity': [3., 5.], 'unused': [9., 8.]})
        interaction = NumericInteractions(columns=['quantity', 'zeta']).fit(frame)
        self.assertEqual(interaction.report_['inputs'], ['quantity', 'zeta'])
        self.assertEqual(interaction.pairs_[1][:2], ('quantity', 'zeta'))
        program = [{'op': 'product', 'name': 'qz', 'columns': ['quantity', 'zeta']}]
        model = FeatureProgram(program).fit(frame)
        self.assertEqual(model.transform(frame[['unused', 'quantity', 'zeta']])['qz'].tolist(), [6., 20.])
        self.assertEqual(operation_facts(program)['operations'][0]['inputs'], ['quantity', 'zeta'])

    def test_automatic_interactions_record_deterministic_rule(self):
        frame = pd.DataFrame({k: [1., 2.] for k in reversed(['a', 'b', 'c', 'd'])})
        first = NumericInteractions(max_numeric_columns=2).fit(frame)
        second = NumericInteractions(max_numeric_columns=2).fit(frame[list(reversed(frame.columns))])
        self.assertEqual(first.selected_columns_, second.selected_columns_)
        self.assertEqual(first.report_['selection_rule'], 'sorted_numeric_names_then_limit')

    def test_unknown_units_and_custom_declarations_are_not_certified(self):
        program = [{'op': 'difference', 'name': 'd', 'columns': ['code', 'calendar'],
                    'input_units': {'code': 'anonymous ordinal', 'calendar': 'calendar unit'}}]
        facts = operation_facts(program)
        self.assertTrue(facts['operations'][0]['warnings'])
        self.assertEqual(facts['predictive_usefulness'], 'not_established')
        self.assertIsNone(facts['operations'][0]['units']['declared_output'])

    def test_learned_groups_refit_and_predict_use_only_the_current_fit(self):
        program = [{'op': 'group_mean', 'name': 'm', 'columns': ['quantity'], 'group_by': 'category'}]
        model = FeatureProgram(program).fit(pd.DataFrame({'category': ['a', 'a'], 'quantity': [2., 4.]}))
        test = pd.DataFrame({'category': ['a', 'new'], 'quantity': [900., 800.]})
        self.assertEqual(model.transform(test)['m'].iloc[0], 3.)
        self.assertTrue(np.isnan(model.transform(test)['m'].iloc[1]))
        model.fit(pd.DataFrame({'category': ['a', 'a'], 'quantity': [10., 20.]}))
        self.assertEqual(model.transform(test)['m'].iloc[0], 15.)

    def test_rmse_sse_and_overlapping_support_are_exact(self):
        data = pd.DataFrame({'zone': ['a']*5+['b']*5, 'mode': ['same']*10, 'target': [0.]*10})
        truth = np.zeros(10)
        prediction = np.array([1.]*5+[3.]*5)
        task = {'task_kind': 'regression', 'target_column': 'target', 'excluded_features': []}
        report = error_diagnostics(data, task, list(range(10)), list(range(10)), truth, prediction, ['zone', 'mode'])
        self.assertEqual(report['sse'], 50.)
        group = next(g for g in report['hard_groups'] if g['feature'] == 'zone' and g['group'] == 'b')
        self.assertEqual(group['rows'], 5)
        self.assertEqual(group['sse_fraction'], .9)
        self.assertTrue(report['groups_overlap'])
        self.assertNotIn('truth', report)

    def test_macro_f1_is_not_mae_or_error_rate(self):
        truth = np.array(['a']*9+['b'])
        prediction = np.array(['a']*10)
        data = pd.DataFrame({'zone': ['x']*10, 'target': truth})
        task = {'task_kind': 'multiclass_classification', 'target_column': 'target',
                'allowed_labels': ['a', 'b'], 'excluded_features': []}
        result = error_diagnostics(data, task, list(range(10)), list(range(10)), truth, prediction)
        self.assertAlmostEqual(result['class_summary']['macro_f1'], 9/19)
        self.assertNotEqual(result['class_summary']['macro_f1'], .9)


class ComposableAnalysisTests(analysis_fixture.AnalysisAccessTests):
    def test_derived_composite_filter_precedes_sampling_and_excludes_holdout(self):
        service = self.service()
        report = service.analyze(query(columns=['delta'], derived=[{
            'op': 'difference', 'name': 'delta', 'columns': ['x', 'x']}],
            filters={'mode': 'all', 'clauses': [{'column': 'group', 'op': 'eq', 'value': 'rare'},
                                               {'column': 'delta', 'op': 'eq', 'value': 0}]}))['report']
        self.assertEqual(report['matched_fit_rows'], 1)
        self.assertEqual(report['columns'][0]['mean'], 0.)
        with self.assertRaises(AgentError):
            service.analyze(query(columns=['delta'], derived=[{'op': 'ratio', 'name': 'delta', 'columns': ['target', 'x']}]))

    def test_error_queries_require_visible_experiment_and_never_target_summary(self):
        service = self.service()
        with self.assertRaises(AgentError):
            service.analyze(query('errors', experiment_id='unseen', include_target=True))

    def test_full_saved_report_supports_utf8_pagination(self):
        service = self.service()
        first = service.analyze(query())
        page = service.analyze(query('report', columns=[], report_id=first['artifact_id'], offset=0, limit_bytes=256))['report']
        self.assertLessEqual(len(page['text'].encode()), 256)
        self.assertEqual(page['offset'], 0)
        self.assertIsNotNone(page['next_offset'])
