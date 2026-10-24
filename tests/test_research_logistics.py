"""Synthetic regressions for pipeline/tuning, scoped reads and final time reserve."""
from copy import deepcopy
import json
import time
import threading
import unittest
from unittest.mock import patch

from agent.catalog import factory_source
from agent.context import output_schemas
from agent.context_details import read_details, read_solution, solution_record, SOLUTION_SCHEMA
from agent.errors import AgentError
from agent.llm.gateway import Gateway
from agent.pipeline_candidates import normalize_candidate, automatic_space
from agent.research_actions import validate_action
from agent.research_feedback import FeedbackMailbox, compact_updates
from tests.graph_helpers import GraphHarness


class ResearchLogisticsTests(unittest.TestCase):
    def setUp(self):
        self.h = GraphHarness(graph_hard_decisions=False, research_mode='adaptive')
        self.c = self.h.controller

    def tearDown(self):
        self.h.close()

    def measured(self, peer='researcher-1', value=10):
        _, reply, _, _ = self.h.result(peer, {'alpha': 1 if value == 10 else 2}, value)
        return reply['lead_submission']['experiment_id']

    def context(self):
        ticket = self.h.issue()
        return self.c.artifacts.read_json(ticket['work_order']['context_id'])

    def read(self, context, **query):
        self.h.g.schemas._check(query, SOLUTION_SCHEMA, '$')
        return read_solution(query, context, self.c.artifacts, self.c.log)

    def test_previous_tuning_wire_format_becomes_a_version_domain(self):
        space = {'alpha': {'type': 'float', 'low': .01, 'high': 10, 'log': True}}
        source = 'from __future__ import annotations\nDEFAULT_PARAMS={"alpha":1}\ndef make_pipeline(task_spec,params):\n    return None\n'
        action = normalize_candidate({'action': 'pipeline', 'code': source,
            'tuning': {'search_space': space, 'max_trials': 12, 'budget_seconds': 1200}})
        validate_action(action, self.c.catalog)
        compile(action['code'], 'candidate.py', 'exec')
        domains = automatic_space(self.c.catalog, {'recipe_id':action['recipe_id'],
            'model':{'hyperparameters':action['hyperparameters']}}, action['code'])
        self.assertEqual(domains['alpha'][1]['high'], 10)
        self.assertIsNone(action['tuning'])
        self.assertEqual(normalize_candidate(action), action)

    def test_conflicting_domains_are_explained_not_silently_replaced(self):
        source = "DEFAULT_PARAMS={'alpha':1}\nSEARCH_SPACE={'alpha':{'type':'float','low':1,'high':2}}\n"
        with self.assertRaises(AgentError) as caught:
            normalize_candidate({'action':'pipeline', 'code':source,
                'tuning':{'search_space':{'alpha':{'type':'float','low':1,'high':3}}}})
        self.assertEqual(caught.exception.code, 'INVALID_TUNING_SPACE')
        self.assertIn('Conflicting', str(caught.exception))

    def test_custom_parameter_names_do_not_select_the_owned_feature_factory(self):
        source = "DEFAULT_PARAMS={'base_recipe_id':'anything','alpha':1}\nSEARCH_SPACE={'alpha':{'type':'float','low':1,'high':2}}\ndef make_pipeline(task_spec,params):\n    return None\n"
        a = normalize_candidate({'action':'pipeline','code':source})
        space = automatic_space(self.c.catalog, {'recipe_id':a['recipe_id'],
            'model':{'hyperparameters':a['hyperparameters']}}, a['code'])
        self.assertEqual(list(space), ['alpha'])

    def test_best_solution_is_measured_and_readable_in_one_call(self):
        self.measured(value=10)
        best = self.measured('researcher-2', value=9)
        context = self.context()
        result = self.read(context, experiment_id='best')
        value = json.loads(result['text'])
        self.assertEqual(result['experiment_id'], best)
        self.assertEqual(value['metric']['value'], 9)
        self.assertIn('make_pipeline', value['code'])
        self.assertEqual(value['hyperparameters']['alpha'], 2)
        user = json.loads(context['messages'][1]['content'])
        self.assertEqual(user['profile_summary']['research_incumbent']['experiment_id'], best)

    def test_delivered_solution_overrides_frozen_base_without_sql_reads(self):
        first = self.measured()
        context = self.context()
        second = self.measured('researcher-2', value=8)
        mailbox = FeedbackMailbox(self.c.artifacts)
        update = mailbox.publish(self.h.root, {'branch_id': self.h.root,
            'dataset_id':self.c.spec['dataset_id'], 'task_spec_id':self.c.spec['task_spec_id'],
            'validation_id':self.c.validation['validation_id'],
            'solution_catalog':{second:solution_record(self.c.store,self.c.artifacts,second)},
            'best_solution':{'experiment_id':second}, 'results':[]})
        context['metadata']['authorized_feedback_ids'] = [update['artifact_id']]
        with patch.object(self.c.store, 'get', side_effect=AssertionError('API thread read SQLite')):
            result = self.read(context, experiment_id='best', view='result')
        self.assertEqual(result['experiment_id'], second)
        self.assertNotEqual(result['experiment_id'], first)
        compact = compact_updates(update)
        self.assertNotIn('solution_catalog', compact)
        self.assertEqual(compact['best_solution']['experiment_id'], second)

    def test_unknown_and_cross_branch_reads_are_denied(self):
        self.measured()
        context = self.context()
        with self.assertRaises(AgentError) as caught:
            self.read(context, experiment_id='exp-not-visible')
        self.assertEqual(caught.exception.code, 'MEMORY_SCOPE_DENIED')
        update = self.c.artifacts.write_json('research/wrong-branch.json', {'branch_id':'sibling'})
        context['metadata']['authorized_feedback_ids'] = [update['artifact_id']]
        with self.assertRaises(AgentError) as caught:
            self.read(context, experiment_id='best')
        self.assertEqual(caught.exception.code, 'MEMORY_SCOPE_DENIED')

    def test_utf8_source_pages_pin_the_experiment_and_check_immutable_hash(self):
        eid = self.measured()
        context = self.context()
        aid = context['metadata']['detail_artifact_id']
        snapshot = self.c.artifacts.read_json(aid)
        source = '# Кириллица\n'*700+factory_source('linear-regressor-v1')
        code = self.c.artifacts.write_text('code/long-source.py',source,'python_source','text/x-python')
        snapshot['sections']['experiment/'+eid]['documents']['code'] = code
        art = self.c.artifacts.write_json('research/paged-context.json', snapshot)
        context['metadata']['detail_artifact_id'] = art['artifact_id']
        query = {'experiment_id':'best','view':'code','limit_bytes':1000}
        pieces = []
        while query:
            reply = self.read(context, **query)
            pieces.append(reply['text'])
            query = reply['next_read']
            if query:self.assertEqual(query['experiment_id'], eid)
        self.assertEqual(''.join(pieces), source)
        self.c.artifacts.resolve(code['artifact_id']).write_text('changed')
        with self.assertRaises(AgentError) as caught:self.read(context, experiment_id=eid,view='code')
        self.assertEqual(caught.exception.code,'MEMORY_SOURCE_CHANGED')

    def test_memory_errors_offer_real_sections_and_index_is_readable(self):
        self.measured()
        context = self.context()
        query = {'reference_id':context['metadata']['detail_artifact_id'],'section':'detail_index'}
        result = read_details(query,context,self.c.artifacts,self.c.log)
        self.assertIn('base_code', json.loads(result['text'])['sections'])
        query['section'] = 'imaginary-section'
        with self.assertRaises(AgentError) as caught:read_details(query,context,self.c.artifacts,self.c.log)
        self.assertIn('base_code',caught.exception.details['available_sections'])

    def test_tool_timeout_leaves_a_real_final_response_window(self):
        self.measured()
        context = self.context()
        art = self.c.artifacts.write_json('contexts/clock-test.json',context)
        clock = [time.monotonic()]
        requests, timeouts = [], []
        class Transport:
            def complete(inner,request,timeout,ctx):
                requests.append(deepcopy(request));timeouts.append(timeout)
                if len(requests)==1:
                    clock[0] += timeout+.01
                    raise TimeoutError('Request timed out')
                clock[0] += .1
                return {'choices':[{'message':{'content':json.dumps({'action':'research',
                    'hypothesis':'Continue from the saved reports.'})},'finish_reason':'stop'}],
                    'usage':{'completion_tokens':10}}
        gateway = Gateway(self.c.config,self.c.budget,self.c.artifacts,self.c.bus,self.c.log,
                          output_schemas(self.c.bus),Transport())
        self.c.budget.reasoning_estimate = lambda: 1
        with patch('agent.llm.gateway.time.monotonic', side_effect=lambda:clock[0]):
            answer = gateway.complete({'purpose':'research_proposal','context_artifact_id':art['artifact_id'],
                'response_schema_id':'ResearchAction-v1','tool_schema_ids':['read_solution'],
                'max_output_tokens':1024,'deadline_seconds':10,'attempt_limit':4})
        self.assertEqual(answer['parsed_payload']['action'],'research')
        self.assertEqual(len(requests),2)
        self.assertIn('tools',requests[0]);self.assertNotIn('tools',requests[1])
        self.assertGreater(timeouts[1],4)
        self.assertLess(timeouts[0],6)

    def test_gateway_exposes_and_executes_the_direct_solution_tool(self):
        from tests.test_gateway_logging import FakeTransport
        self.measured()
        art = self.c.artifacts.write_json('contexts/native-read.json',self.context())
        transport = FakeTransport([
            {'choices':[{'message':{'tool_calls':[{'id':'native','type':'function','function':{
                'name':'read_solution','arguments':json.dumps({'experiment_id':'best','view':'code'})}}]},'finish_reason':'tool_calls'}]},
            {'choices':[{'message':{'content':json.dumps({'action':'research','hypothesis':'Keep the measured solution and test a new representation.'})},'finish_reason':'stop'}]},
        ])
        gateway = Gateway(self.c.config,self.c.budget,self.c.artifacts,self.c.bus,self.c.log,
                          output_schemas(self.c.bus),transport)
        self.c.budget.reasoning_estimate = lambda: 1
        with patch.object(self.c.store,'get',side_effect=AssertionError('API thread read SQLite')):
            answer = gateway.complete({'purpose':'research_proposal','context_artifact_id':art['artifact_id'],
                'response_schema_id':'ResearchAction-v1','tool_schema_ids':['read_solution'],
                'max_output_tokens':1024,'deadline_seconds':30,'attempt_limit':4})
        self.assertEqual(answer['tool_calls'][0]['name'],'read_solution')
        reply = next(m for m in transport.requests[1]['messages'] if m['role']=='tool')
        self.assertIn('make_pipeline',json.loads(reply['content'])['text'])

    def test_analysis_deadline_does_not_leak_into_later_queries(self):
        from tools.analysis import AnalysisService
        service = AnalysisService.__new__(AnalysisService)
        service.lock = threading.RLock()
        service.query_context = None
        service.budget = self.c.budget
        with self.assertRaises(AgentError) as caught:
            with service.query_slot({'_tool_deadline':time.monotonic()-1}):
                service._check_time(time.monotonic())
        self.assertEqual(caught.exception.code,'ANALYSIS_TIMEOUT')
        self.assertIsNone(service.query_context)
        with service.query_slot():service._check_time(time.monotonic())

    def test_compacted_solution_page_resumes_at_the_bytes_actually_delivered(self):
        from agent.context_budget import compact_tool_reply
        original = {'experiment_id':'exp-pinned','view':'code','reference_id':'context-a',
            'section':'solution/exp-pinned/code','scope':{},'offset':0,'total_bytes':20000,
            'text':'# Пример\n'*700,'next_offset':10000,'evidence_ids':[],'idea_ids':[],
            'next_read':{'experiment_id':'exp-pinned','view':'code','offset':10000,'limit_bytes':6000}}
        reply = compact_tool_reply(original,1600)
        self.assertEqual(reply['next_read']['offset'],len(reply['text'].encode('utf-8')))
        self.assertEqual(reply['next_read']['experiment_id'],'exp-pinned')
        self.assertEqual(original['next_read']['offset'],10000)


if __name__ == '__main__':
    unittest.main()
