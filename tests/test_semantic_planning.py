"""Synthetic regressions. Written for later execution; no quality claims."""
from copy import deepcopy
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from agent.config import Config, ROOT
from agent.contracts import Schemas, canonical
from agent.errors import AgentError
from agent.feature_engineering import FeatureProgram
from agent.feature_schema import operation_facts, validate_program
from agent.local_search import LocalSearch
from agent.researcher import Researcher
from agent.semantic_planning import (build_task_model, unknown_model, semantic_view,
    effective_model, relation_checks, validate_investigation, check_declared_semantics, load_task_model)
from tests.graph_helpers import GraphHarness
from tests.test_adaptive_research import action
from tests import test_analysis_access as analysis_fixture
from tests.test_analysis_access import query


def fixture():
    return json.loads((ROOT/'contracts/examples/semantic-planning.json').read_text())


def investigation(**changes):
    result = deepcopy(fixture()['valid']['Investigation'])
    result.update(changes)
    return result


def model():
    data = fixture()
    return build_task_model(data['valid']['TaskModel'], data['task_text'], data['profile'], Schemas(),
        {'task_spec_id': 'synthetic-task', 'dataset_id': 'synthetic-dataset', 'target_column': 'response'})


def context(view=None, *, required=True):
    return {'metadata': {'semantic_contract': 'semantic-planning-1'} if required else {},
        'messages': [{'role': 'system', 'content': ''}, {'role': 'user',
            'content': json.dumps({'profile_summary': {'semantic_task': view or {}}})}]}


class TaskModelTests(unittest.TestCase):
    def test_quotes_are_checked_without_promoting_the_interpretation(self):
        result = model()
        self.assertIn('interpretation_unverified', result['verification'])
        self.assertIn('quotation_matched', result['quantities'][0]['source_verification'])
        self.assertEqual(result['predictive_usefulness'], 'not_established')
        data = fixture()
        data['valid']['TaskModel']['quantities'][0]['source']['quote'] = 'Invented source text'
        with self.assertRaises(AgentError):
            build_task_model(data['valid']['TaskModel'], data['task_text'], data['profile'], Schemas(),
                {'task_spec_id': 's', 'dataset_id': 'd', 'target_column': 'response'})

    def test_unknown_metadata_does_not_fabricate_units_or_semantic_nodes(self):
        data = fixture()
        result = build_task_model(None, '', data['profile'], Schemas(),
            {'task_spec_id': 's', 'dataset_id': 'd', 'target_column': 'response'})
        self.assertEqual(result['context']['prediction_time'], 'unknown')
        self.assertEqual(result['quantities'], [])
        self.assertTrue(result['unknowns'])

    def test_hypothetical_units_and_absent_columns_are_rejected(self):
        data = fixture()
        for change in ({'column': 'not_a_real_column'}, {'source': {'source': 'hypothesis', 'record': None, 'quote': ''}}):
            raw = deepcopy(data['valid']['TaskModel'])
            raw['quantities'][0].update(change)
            with self.assertRaises(AgentError):
                build_task_model(raw, data['task_text'], data['profile'], Schemas(),
                    {'task_spec_id': 's', 'dataset_id': 'd', 'target_column': 'response'})

    def test_source_model_is_independent_of_csv_column_order(self):
        data = fixture()
        original = model()
        data['profile']['train_columns'].reverse()
        data['profile']['test_columns'].reverse()
        reordered = build_task_model(data['valid']['TaskModel'], data['task_text'], data['profile'], Schemas(),
            {'task_spec_id': 'synthetic-task', 'dataset_id': 'synthetic-dataset', 'target_column': 'response'})
        self.assertEqual(original, reordered)

    def test_task_identity_is_part_of_the_model_identity(self):
        data = fixture()
        other = build_task_model(data['valid']['TaskModel'], data['task_text'], data['profile'], Schemas(),
            {'task_spec_id': 'other-task', 'dataset_id': 'synthetic-dataset', 'target_column': 'response'})
        self.assertNotEqual(model()['model_id'], other['model_id'])


class InvestigationTests(unittest.TestCase):
    def test_active_contract_requires_a_testable_commitment_and_legacy_remains_readable(self):
        with self.assertRaises(AgentError):
            validate_investigation(None, context(), set(), [], Schemas())
        result = validate_investigation(None, context(required=False), set(), [], Schemas())
        self.assertEqual(result['verification'], 'legacy_no_investigation')

    def test_unread_semantic_record_cannot_be_cited(self):
        view = semantic_view(model(), {})
        valid = investigation(basis_ids=[view['relations'][0]['record_id']])
        validate_investigation(valid, context(view), set(), fixture()['profile']['train_columns'], Schemas())
        with self.assertRaises(AgentError):
            validate_investigation({**valid, 'basis_ids': ['unread-sibling-record']}, context(view), set(),
                fixture()['profile']['train_columns'], Schemas())

    def test_feature_dependency_chain_must_match_the_research_question(self):
        program = [{'op': 'difference', 'name': 'delta', 'columns': ['major', 'minor']},
                   {'op': 'square', 'name': 'delta_squared', 'columns': ['delta']}]
        validate_investigation(investigation(), context(), set(), ['major', 'minor'], Schemas(), program=program)
        with self.assertRaises(AgentError):
            validate_investigation(investigation(columns=['major']), context(), set(), ['major', 'minor'], Schemas(), program=program)

    def test_anonymous_period_cannot_be_presented_as_physical_age(self):
        program = [{'op': 'difference', 'name': 'age', 'columns': ['period_tag', 'calendar_stamp']}]
        with self.assertRaises(AgentError):
            check_declared_semantics(program, investigation(interpretation='physical'), model())
        self.assertEqual(check_declared_semantics(program, investigation(interpretation='exploratory'), model()), [])

    def test_incompatible_declared_units_reject_physical_sum_but_allow_products(self):
        source = model()
        program = [{'op': 'sum', 'name': 'mixed', 'columns': ['major', 'calendar_stamp']}]
        with self.assertRaises(AgentError):
            check_declared_semantics(program, investigation(interpretation='physical'), source)
        program[0]['op'] = 'product'
        self.assertEqual(check_declared_semantics(program, investigation(interpretation='physical'), source), [])

    def test_prediction_time_and_cyclic_period_contradictions_are_rejected(self):
        source = model()
        program = [{'op': 'cyclic_sin', 'name': 'bearing_sin', 'columns': ['bearing'], 'period': 24}]
        with self.assertRaises(AgentError):
            check_declared_semantics(program, investigation(), source)
        program[0]['period'] = 360
        bearing = next(q for q in source['quantities'] if q['column'] == 'bearing')
        bearing['availability'] = 'unavailable'
        with self.assertRaises(AgentError):
            check_declared_semantics(program, investigation(), source)

    def test_supported_relation_suggests_a_named_exact_check_without_target_leakage(self):
        relation = model()['relations'][0]
        check = relation_checks([relation])[0]
        self.assertEqual(check['query']['checks'][0]['violation'], {'column': 'minor', 'op': 'gt', 'other_column': 'major'})
        normalized = {**relation, 'kind': 'normalized_by', 'columns': ['response', 'duration']}
        check = relation_checks([normalized], ['response'])[0]
        self.assertEqual(check['query']['columns'], ['duration'])

    def test_typed_revision_changes_planning_and_conflicts_remove_certainty(self):
        source = model()
        record = {'record_id': 'revision-1', 'status': 'proposed', 'verification': 'model_statement_unverified',
            'content': {'columns': ['period_tag'], 'statement': 'The mapping remains unknown.',
                        'properties': {'quantity_kind': 'unknown', 'units': 'unknown'}}}
        revised = effective_model(source, [record])
        value = next(q for q in revised['quantities'] if q['column'] == 'period_tag')
        self.assertEqual(value['kind'], 'unknown')
        self.assertEqual(value['record_id'], 'revision-1')
        self.assertEqual(next(q for q in source['quantities'] if q['column'] == 'period_tag')['kind'], 'ordinal_code')
        a = {**record, 'status': 'conflicted'}
        b = {**record, 'record_id': 'revision-2', 'content': {**record['content'], 'properties': {'quantity_kind': 'calendar'}}}
        value = next(q for q in effective_model(source, [a, b])['quantities'] if q['column'] == 'period_tag')
        self.assertEqual(value['kind'], 'unknown')

    def test_current_question_retrieves_its_relation_instead_of_first_columns(self):
        source = model()
        relation = source['relations'][0]
        memory = {'decisions': [{'decision_artifact_id': 'decision', 'investigation': investigation(
            basis_ids=[relation['record_id']]), 'observations': []}]}
        view = semantic_view(source, memory)
        self.assertEqual(view['relations'][0]['record_id'], relation['record_id'])
        self.assertTrue({'minor', 'major'} <= {q['column'] for q in view['quantities']})
        self.assertLessEqual(len(canonical(view).encode()), 10000)


class CyclicFeatureTests(unittest.TestCase):
    def test_wraparound_period_shift_and_refit_preserve_geometry(self):
        program = [{'op': op, 'name': op, 'columns': ['bearing'], 'period': 360}
                   for op in ('cyclic_sin', 'cyclic_cos')]
        transform = FeatureProgram(program).fit(pd.DataFrame({'bearing': [0., 90.]}))
        values = transform.transform(pd.DataFrame({'bearing': [359., 1., 361., np.nan]}))
        self.assertAlmostEqual(values['cyclic_sin'].iloc[1], values['cyclic_sin'].iloc[2])
        points = values[['cyclic_sin', 'cyclic_cos']].to_numpy()
        self.assertLess(np.linalg.norm(points[0]-points[1]), .04)
        self.assertTrue(np.isnan(points[3]).all())
        facts = operation_facts(program)['operations'][0]
        self.assertEqual(facts['period'], 360)
        self.assertEqual(facts['origin'], 0)

    def test_invalid_periods_and_unrelated_period_fields_fail(self):
        for period in (0, -1, True, float('inf')):
            with self.assertRaises(ValueError):
                validate_program([{'op': 'cyclic_cos', 'name': 'c', 'columns': ['a'], 'period': period}])
        with self.assertRaises(ValueError):
            validate_program([{'op': 'square', 'name': 's', 'columns': ['a'], 'period': 360}])


class ExactSemanticChecks(analysis_fixture.AnalysisAccessTests):
    def test_rare_violation_is_counted_over_all_fit_rows_and_missing_support_is_explicit(self):
        train = pd.DataFrame({'x': [float(i) for i in range(30000)], 'group': ['a']*30000, 'target': [999999]*30000})
        train.loc[1, 'x'] = np.nan
        report = self.service(train, train.drop(columns='target')).analyze(query('check',
            checks=[{'name': 'rare', 'violation': {'column': 'x', 'op': 'eq', 'value': 29998}},
                    {'name': 'holdout', 'violation': {'column': 'x', 'op': 'eq', 'value': 29999}}]))['report']
        self.assertFalse(report['sampled'])
        self.assertEqual(report['analyzed_rows'], 29999)
        self.assertEqual(report['checks'][0]['violation_rows'], 1)
        self.assertEqual(report['checks'][0]['evaluable_rows'], 29998)
        self.assertEqual(report['checks'][0]['missing_rows'], 1)
        self.assertEqual(report['checks'][1]['violation_rows'], 0)
        def numeric_values(value):
            if isinstance(value, dict):
                return [v for item in value.values() for v in numeric_values(item)]
            if isinstance(value, list):
                return [v for item in value for v in numeric_values(item)]
            return [value] if isinstance(value, (int, float)) else []
        self.assertNotIn(999999, numeric_values(report))

    def test_check_filters_before_counting_and_supports_named_derived_expression(self):
        report = self.service().analyze(query('check', columns=['delta'],
            derived=[{'op': 'difference', 'name': 'delta', 'columns': ['x', 'x']}],
            filter={'column': 'group', 'value': 'rare'},
            checks=[{'name': 'negative_delta', 'violation': {'column': 'delta', 'op': 'lt', 'value': 0}}]))['report']
        self.assertEqual(report['matched_fit_rows'], 1)
        self.assertEqual(report['checks'][0]['evaluable_rows'], 1)
        self.assertEqual(report['checks'][0]['violation_rows'], 0)

    def test_raw_target_checks_are_rejected_even_when_not_in_requested_columns(self):
        with self.assertRaises(AgentError):
            self.service().analyze(query('check', checks=[{'name': 'leak',
                'violation': {'column': 'target', 'op': 'gt', 'value': 0}}]))

    def test_computation_requires_investigation_before_any_scan_under_new_contract(self):
        service = self.service()
        ctx = context()
        with patch.object(service, '_sample', side_effect=AssertionError('Should not scan')):
            with self.assertRaises(AgentError) as caught:
                service.analyze(query(), context=ctx)
        self.assertEqual(caught.exception.code, 'INVALID_INVESTIGATION')


class SemanticPersistenceTests(unittest.TestCase):
    def setUp(self):
        self.h = GraphHarness('regression', research_mode='adaptive',research_protocol='legacy',background_tuning_enabled=False)
        g = self.h.g
        self.source = build_task_model(unknown_model(), '', self.h.h.profile['summary'], g.schemas, g.spec)
        art = g.artifacts.write_json('task/synthetic-semantic-model.json', self.source, 'task_semantic_model')
        g.spec.update(semantic_model_artifact_id=art['artifact_id'], semantic_model_id=self.source['model_id'])

    def tearDown(self):
        self.h.close()

    def test_model_details_are_frozen_and_identity_checks_reject_other_tasks(self):
        from agent.context_details import read_details
        g = self.h.g
        ctx = g.artifacts.read_json(self.h.issue()['work_order']['context_id'])
        self.assertEqual(ctx['metadata']['semantic_contract'], 'semantic-planning-1')
        page = read_details({'reference_id': ctx['metadata']['detail_artifact_id'], 'section': 'semantic_model'},
                            ctx, g.artifacts, g.log)
        self.assertEqual(json.loads(page['text'])['model_id'], self.source['model_id'])
        with self.assertRaises(AgentError):
            load_task_model({**g.spec, 'dataset_id': 'different'}, g.artifacts)

    def test_independent_configuration_result_returns_to_its_question_without_proving_mechanism(self):
        g = self.h.g
        ticket = self.h.issue()
        payload = self.h.proposal(ticket, {'alpha': 1})
        ctx = g.artifacts.read_json(ticket['work_order']['context_id'])
        proposal = action('catalog', investigation=investigation(columns=[], interpretation='model_behavior'))
        researcher = Researcher(g.bus, None, g.artifacts, g.spec, g.catalog, g.config, g.log)
        researcher.save_decision(proposal, payload['proposal']['plan'], ticket['researcher_request']['payload'], ctx,
                                {'output_artifact_id': 'synthetic-output'})
        reply = g.invoke('G04', payload)
        self.h.commit(reply, 12.5)  # Injected state-machine score, never an ML result.
        memory = g.research_memory(self.h.root, [])
        view = semantic_view(self.source, memory)
        result = view['recent_investigations'][0]
        self.assertEqual(result['investigation']['question'], proposal['investigation']['question'])
        self.assertEqual(result['observations'][0]['primary_metric']['value'], 12.5)
        self.assertEqual(result['outcome'], 'measured_configuration; mechanism_unresolved')

    def test_semantic_revisions_obey_existing_fork_visibility(self):
        g = self.h.g
        fork, _, _ = self.h.make_fork()
        # Write the parent's interpretation *after* the checkpoint chosen for
        # this fork. It must not appear merely because it predates child issue.
        parent_ticket = self.h.issue()
        parent_context = g.artifacts.read_json(parent_ticket['work_order']['context_id'])
        parent_proposal = action('stop', semantic_updates=[{
            'topic': 'late-parent-interpretation', 'kind': 'hypothesis',
            'statement': 'A late parent interpretation must stay outside the child checkpoint.',
            'columns': ['x_numeric'], 'evidence_ids': [],
            'properties': {'quantity_kind': 'unknown'}}])
        researcher = Researcher(g.bus, None, g.artifacts, g.spec, g.catalog, g.config, g.log)
        researcher.save_decision(parent_proposal, None, parent_ticket['researcher_request']['payload'],
                                parent_context, {'output_artifact_id': 'synthetic-late-output'})
        g.invoke('G04', {'work_order_id': parent_ticket['work_order']['work_order_id'],
            'proposal': {'proposal_id': 'synthetic-stop', 'researcher_id': 'researcher-1',
                         'decision': 'stop', 'plan': None, 'stop_reason': 'Synthetic branch visibility fixture'},
            'manifest': None, 'lineage': None})
        fork['expected_branch_revision'] = g.branch(self.h.root)['branch_revision']
        children = g.invoke('G02', fork)
        accept = children['accept_branch']['branch_id']
        ctx = g.artifacts.read_json(self.h.issue(branch=accept)['work_order']['context_id'])
        details = g.artifacts.read_json(ctx['metadata']['detail_artifact_id'])
        self.assertEqual(details['scope']['branch_id'], accept)
        self.assertIn('visibility', details['scope'])
        self.assertEqual(details['sections']['semantic_model']['model_id'], self.source['model_id'])
        self.assertNotIn('late-parent-interpretation', canonical(details['sections']['semantics']))

    def test_bootstrap_source_error_uses_existing_repair_before_spec_resolution(self):
        from agent.context import demo_metadata, output_schemas
        from agent.llm.gateway import Gateway
        g = self.h.g
        draft = deepcopy(demo_metadata(self.h.h.task.read_text())['draft'])
        draft['task_model'] = unknown_model()
        broken = deepcopy(draft)
        broken['task_model']['quantities'] = [{'column': 'invented-column', 'meaning': 'unknown',
            'kind': 'unknown', 'unit': None, 'period': None, 'availability': 'unknown',
            'source': {'source': 'hypothesis', 'record': None, 'quote': ''}}]
        requests = []
        class Transport:
            def complete(self, request, timeout, context):
                requests.append(deepcopy(request))
                value = broken if len(requests) == 1 else draft
                return {'choices': [{'message': {'content': json.dumps(value)}, 'finish_reason': 'stop'}],
                        'usage': {'completion_tokens': 100}}
        ctx = {'metadata': {'schema_id': 'TaskDraft-v1',
                           'bootstrap_profile_artifact_id': self.h.h.profile['profile_artifact_id']},
            'messages': [{'role': 'system', 'content': 'Synthetic bootstrap source validation fixture.'},
                         {'role': 'user', 'content': json.dumps({'task_markdown': self.h.h.task.read_text()})}]}
        art = g.artifacts.write_json('contexts/synthetic-bootstrap-source.json', ctx, 'llm_context')
        gateway = Gateway(g.config, g.budget, g.artifacts, g.bus, g.log, output_schemas(g.bus), Transport())
        result = gateway.complete({'purpose': 'task_interpretation', 'context_artifact_id': art['artifact_id'],
            'response_schema_id': 'TaskDraft-v1', 'tool_schema_ids': [], 'max_output_tokens': 1024,
            'deadline_seconds': 300, 'attempt_limit': 3})
        self.assertEqual(result['attempts_used'], 2)
        self.assertEqual(result['parsed_payload']['task_model']['quantities'], [])


class SearchPolicyTests(unittest.TestCase):
    def test_uninterpreted_background_seeds_are_opt_in(self):
        self.assertEqual(Config().feature_seed_budget_fraction, 0)
        self.assertTrue(Config().feature_research_enabled)

    def test_emergency_variants_preserve_interactions_until_explicit_authorization(self):
        search = LocalSearch.__new__(LocalSearch)
        search.c = SimpleNamespace(config=SimpleNamespace(research_mode='adaptive'), graph=None,
            catalog=SimpleNamespace(recipes={'r': {'parameters': {
                'numeric_interactions': {'enum': [False, True]}, 'alpha': {'type': 'float', 'min': .01, 'max': 10}}}}))
        variants = list(search.variants('r', {'numeric_interactions': False, 'alpha': 1.}))
        self.assertTrue(variants)
        self.assertTrue(all(not v['numeric_interactions'] for v in variants))
        explicit = list(search.variants('r', {'numeric_interactions': False, 'alpha': 1.}, parameter_names=['numeric_interactions']))
        self.assertTrue(explicit[0]['numeric_interactions'])
