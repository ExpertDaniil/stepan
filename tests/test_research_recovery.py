"""Deterministic provider recovery and immutable, scoped error observations."""
from copy import deepcopy
from datetime import datetime, timezone
from types import SimpleNamespace
import json
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from agent.context import output_schemas
from agent.error_research import ErrorResearch
from agent.errors import AgentError
from agent.fit_diagnostics import compact_report, error_profile
from agent.llm.gateway import Gateway
from agent.research_feedback import compact_updates
from agent.scientific_research import compact_protocol
from tests.graph_helpers import GraphHarness


class ProviderRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.h = GraphHarness(time_budget_seconds=3600)
        self.c = self.h.controller
        self.now = [time.monotonic()]
        self.clock = patch('agent.llm.gateway.time.monotonic', side_effect=lambda:self.now[0])
        self.clock.start()
        self.context = {'metadata': {'schema_id':'ResearchProposal-v1'}}
        self.request = {'messages':[], 'max_tokens':128, 'model':'scripted'}
        self.payload = {'purpose':'research_proposal', 'context_artifact_id':'synthetic'}

    def tearDown(self):
        self.clock.stop()
        self.h.close()

    def gateway(self, responses):
        def complete(*args):
            value = responses.pop(0)
            if isinstance(value, Exception):raise value
            return value
        return Gateway(self.c.config, self.c.budget, self.c.artifacts, self.c.bus, self.c.log,
                       output_schemas(self.c.bus), SimpleNamespace(complete=complete))

    def call(self, gateway):
        return gateway._request(deepcopy(self.request), self.context, self.payload, 1, self.now[0]+120)

    @staticmethod
    def failure(message='Budget has been exceeded', status=429, body=None):
        error = RuntimeError(message)
        error.status_code, error.body = status, body
        return error

    def test_exhaustion_probe_502_then_success_resumes_without_paid_ping(self):
        gateway = self.gateway([self.failure(), self.failure('temporary gateway',502), {'choices':[]}])
        with self.assertRaises(AgentError):self.call(gateway)
        self.assertFalse(gateway.daily_reset_known)
        first = gateway.daily_retry_at
        self.assertGreater(first, self.now[0])
        self.now[0] = first
        with self.assertRaises(AgentError) as caught:self.call(gateway)
        self.assertEqual(caught.exception.code, 'LLM_UNAVAILABLE')
        self.assertTrue(caught.exception.retryable)
        self.assertTrue(gateway.daily_budget_exhausted.is_set())
        self.assertFalse(gateway.daily_probe_active)
        self.assertGreater(gateway.daily_retry_at, first)
        with self.assertRaises(AgentError):gateway.check_availability()
        self.now[0] = gateway.daily_retry_at
        self.call(gateway)
        gateway.check_availability()
        self.assertFalse(gateway.daily_budget_exhausted.is_set())
        self.assertIsNone(gateway.daily_retry_at)
        self.assertEqual(self.c.budget.calls, 3)

    def test_unknown_reset_uses_increasing_intervals_and_preserves_reserve(self):
        gateway = self.gateway([self.failure() for _ in range(5)])
        delays = []
        for _ in range(3):
            with self.assertRaises(AgentError):self.call(gateway)
            delays.append(gateway.daily_retry_at-self.now[0])
            self.now[0] = gateway.daily_retry_at
        self.assertEqual(delays, [60,120,240])
        self.c.budget.deadline = self.now[0]+self.c.budget.reserve_seconds()+3
        with self.assertRaises(AgentError):self.call(gateway)
        self.assertIsNone(gateway.daily_retry_at)
        self.assertTrue(gateway.daily_budget_exhausted.is_set())

    def test_explicit_reset_replaces_previous_schedule(self):
        class Clock(datetime):
            @classmethod
            def now(cls, tz=None):return cls(2026,10,25,tzinfo=timezone.utc)
        gateway = self.gateway([self.failure(body={'error':{'reset_at':'2026-10-25T00:00:20Z'}}),
                                self.failure(body={'budget_reset_at':'2026-10-25T00:00:40Z'})])
        with patch('datetime.datetime', Clock):
            with self.assertRaises(AgentError):self.call(gateway)
            self.assertEqual(gateway.daily_retry_at-self.now[0],20)
            self.now[0] = gateway.daily_retry_at
            with self.assertRaises(AgentError):self.call(gateway)
            self.assertEqual(gateway.daily_retry_at-self.now[0],40)
        self.assertTrue(gateway.daily_reset_known)

    def test_only_one_due_probe_can_reach_transport(self):
        gateway = self.gateway([])
        gateway.daily_budget_exhausted.set()
        gateway.daily_retry_at = self.now[0]
        entered, release, errors = threading.Event(), threading.Event(), []
        def complete(*args):
            entered.set()
            if not release.wait(2):raise AssertionError('Probe did not finish')
            return {'choices':[]}
        gateway.transport.complete = complete
        def run():
            try:self.call(gateway)
            except BaseException as error:errors.append(error)
        thread = threading.Thread(target=run)
        thread.start()
        try:
            self.assertTrue(entered.wait(2))
            with self.assertRaises(AgentError):self.call(gateway)
        finally:
            release.set();thread.join(3)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(self.c.budget.calls,1)

    def test_ordinary_502_does_not_activate_budget_probe(self):
        gateway = self.gateway([self.failure('upstream 502',502), {'choices':[]}])
        with self.assertRaises(AgentError):self.call(gateway)
        self.assertFalse(gateway.daily_budget_exhausted.is_set())
        self.assertIsNone(gateway.daily_retry_at)
        self.call(gateway)
        self.assertEqual(self.c.budget.calls,2)


class GroupSummaryTests(unittest.TestCase):
    def test_duplicate_missing_groups_do_not_hide_nested_rare_sse(self):
        frame = pd.DataFrame({'category':['rare']*10+['ordinary']*990})
        for i in range(12):frame['sensor_'+str(i)] = np.r_[np.full(750,np.nan), np.arange(250)]
        task = {'task_kind':'regression','target_column':'answer','excluded_features':[], 'metric':{'metric_id':'rmse'}}
        report = error_profile(frame,np.zeros(1000),np.r_[np.full(10,30),np.ones(990)],np.arange(1000)%3,task)
        missing = [g for g in report['diagnostic_groups'] if g['definition']['kind']=='missing']
        self.assertEqual(len(missing),1)
        self.assertEqual(len(missing[0]['related_groups']),12)
        self.assertAlmostEqual(missing[0]['sse'],9740)
        rare = next(g for g in report['hard_groups'] if g['group']=='rare')
        self.assertAlmostEqual(rare['sse_fraction'],9000/9990)
        self.assertTrue(rare['support_uncertain'])
        self.assertLessEqual(len(report['hard_groups']),12)
        self.assertGreaterEqual(report['duplicate_groups_merged'],11)
        self.assertGreaterEqual(len(report['diagnostic_groups']),len(report['hard_groups']))
        self.assertIn(rare['definition'],[g['definition'] for g in compact_report(report)['hard_groups']])
        short_missing = next(g for g in compact_report(report)['hard_groups'] if g['definition']['kind']=='missing')
        self.assertEqual(short_missing['omitted_related_groups'],6)
        twice = compact_report(compact_report(report))
        self.assertEqual(next(g for g in twice['hard_groups'] if g['definition']['kind']=='missing')['omitted_related_groups'],6)
        full = {'fit_diagnostics':[{'report':report}]}
        short = compact_protocol(full)
        self.assertNotIn('diagnostic_groups',short['fit_diagnostics'][0]['report'])
        self.assertIn('diagnostic_groups',full['fit_diagnostics'][0]['report'])
        update = compact_updates({'artifact_id':'snapshot',**full})
        self.assertNotIn('diagnostic_groups',update['fit_diagnostics'][0]['report'])

    def test_full_results_survive_twelve_group_summary(self):
        task = {'task_kind':'regression','target_column':'answer','excluded_features':[], 'metric':{'metric_id':'rmse'}}
        report = error_profile(pd.DataFrame({'category':[str(i%25) for i in range(1000)]}),
                               np.zeros(1000),np.arange(1000)%25,np.arange(1000)%3,task)
        self.assertEqual(len(report['diagnostic_groups']),25)
        self.assertEqual(len(report['hard_groups']),12)
        self.assertEqual(report['omitted_groups'],13)


class ObservationScopeTests(unittest.TestCase):
    def setUp(self):
        self.h = GraphHarness(research_mode='adaptive',graph_hard_decisions=False)
        self.c = self.h.controller
        self.c.graph = self.h.g
        self.errors = ErrorResearch(self.c)
        self.serial = 0
        self.a = self.experiment(1,10)

    def tearDown(self):self.h.close()

    def experiment(self, alpha, score, branch=None):
        _,reply,_,_ = self.h.result('researcher-1',{'alpha':alpha},score,branch=branch)
        return reply['lead_submission']['experiment_id']

    def record(self, eid, *, cost=10, definition=None, membership='same-rows'):
        self.serial += 1
        identity = {**self.errors.identity(),'experiment_id':eid}
        provenance = self.c.artifacts.write_json(f'diagnostics/identity-{self.serial}.json',identity)
        definition = definition or {'kind':'missing','column':'unfamiliar'}
        group = {'definition':definition,'priority_cost':cost,'row_membership_sha256':membership,
                 'rows':10,'support_fraction':.1,'feature':definition.get('column','x'),'group':'missing'}
        report = {'sampled':False,'hard_groups':[group]}
        art = self.c.artifacts.write_json(f'diagnostics/report-{self.serial}.json',report)
        return {'experiment_id':eid,'identity_artifact_id':provenance['artifact_id'],
                'artifact_id':art['artifact_id'],'report':report,'completed_monotonic':self.serial}

    def test_b_cannot_overwrite_a_or_change_already_bound_ticket(self):
        first = self.record(self.a)
        self.errors.ingest(first)
        advice = self.errors.advice([self.a],'researcher-1',self.h.root)
        assignment = deepcopy(self.errors.assignments[self.h.root+'/researcher-1'])
        ticket = {'work_order':{'branch_id':self.h.root,'work_order_id':'frozen'},'error_assignment':assignment}
        b = self.experiment(2,9)
        self.errors.ingest(self.record(b,cost=900))
        again = self.errors.advice([self.a],'researcher-1',self.h.root)
        self.assertEqual(again,advice)
        self.assertEqual(again['source_experiment_id'],self.a)
        binding = self.errors.bind({'plan_id':'frozen-plan','researcher_id':'researcher-1'},ticket)
        self.assertEqual(binding['base_experiment_id'],self.a)
        self.assertEqual(binding['source_artifact_id'],first['artifact_id'])
        self.assertEqual(binding['work_order_id'],'frozen')
        self.assertEqual(self.errors.advice([self.a,b],'researcher-1',self.h.root)['leader']['experiment_id'],b)
        self.assertEqual(binding['observation']['priority_cost'],10)

    def test_refresh_of_same_model_keeps_old_observation_until_assignment_finishes(self):
        old, new = self.record(self.a,cost=10), self.record(self.a,cost=90)
        self.errors.ingest(old)
        before = self.errors.advice([self.a],'researcher-1',self.h.root)
        self.errors.ingest(new)
        self.assertEqual(self.errors.advice([self.a],'researcher-1',self.h.root),before)
        self.errors.assignments[self.h.root+'/researcher-1']['status'] = 'evaluated'
        after = self.errors.advice([self.a],'researcher-1',self.h.root)
        self.assertEqual(after['source_artifact_id'],new['artifact_id'])
        self.assertEqual(after['observation']['priority_cost'],90)
        self.assertEqual(len(self.errors.problems),2)
        self.errors.ingest(new)
        self.assertEqual(len(self.errors.problems),2)

    def test_equal_rows_with_different_definitions_are_one_peer_focus(self):
        self.errors.ingest(self.record(self.a))
        one = self.errors.advice([self.a],'researcher-1',self.h.root)
        self.errors.ingest(self.record(self.a,definition={'kind':'equals','column':'alias','value':'rare'}))
        two = self.errors.advice([self.a],'researcher-2',self.h.root)
        self.assertTrue(two['shared_focus'])
        self.assertNotEqual(one['recommendation']['question'],two['recommendation']['question'])

    def test_new_report_after_fork_is_not_inherited_with_visible_parent_model(self):
        source = self.a
        old = self.record(source)
        self.errors.ingest(old)
        problem = next(iter(self.errors.problems.values()))
        early = {'candidate_experiment_id':source,'proposal_base_experiment_id':source,
                 'scope':self.errors.publication(source),'status':'no_gain'}
        problem['attempts'].append(early)
        fork, _, _ = self.h.make_fork()
        children = self.h.g.invoke('G02',fork,key='fork-observations')
        child = children['accept_branch']['branch_id']
        visible = self.h.g.visible_history(child)
        self.assertIn(source,visible)
        self.errors.ingest(self.record(source,cost=999))
        problem['attempts'].append({**early,'scope':self.errors.publication(source)})
        advice = self.errors.advice(visible,'researcher-1',child)
        self.assertEqual(advice['source_artifact_id'],old['artifact_id'])
        self.assertEqual(advice['relevant_attempts'],[early])
        sibling = children['reject_branch']['branch_id']
        self.h.g.invoke('G08',{'branch_id':sibling,'expected_branch_revision':0,
                            'action':'resume','reason':'Test sibling visibility'})
        b = self.experiment(77,1,branch=sibling)
        self.errors.ingest(self.record(b,cost=9999))
        self.assertNotIn(b,self.h.g.visible_history(child))
        again = self.errors.advice(self.h.g.visible_history(child),'researcher-1',child)
        self.assertEqual(again['source_experiment_id'],source)

    def test_attempts_obey_source_visibility_and_fork_cut(self):
        self.errors.ingest(self.record(self.a))
        self.errors.advice([self.a],'researcher-1',self.h.root)
        problem = next(iter(self.errors.problems.values()))
        b = self.experiment(2,11)
        attempt = {'candidate_experiment_id':b,'proposal_base_experiment_id':self.a,
                   'scope':self.errors.publication(b),'status':'no_gain'}
        problem['attempts'].append(attempt)
        self.assertEqual(self.errors.advice([self.a],'researcher-1',self.h.root)['relevant_attempts'],[])
        self.assertEqual(len(self.errors.advice([self.a,b],'researcher-1',self.h.root)['relevant_attempts']),1)
        attempt['scope']['protocol_sha256'] = 'other'
        self.assertEqual(self.errors.advice([self.a,b],'researcher-1',self.h.root)['relevant_attempts'],[])

    def test_task_or_protocol_mismatch_cannot_be_ingested(self):
        record = self.record(self.a)
        self.c.validation = {**self.c.validation,'protocol_sha256':'other-protocol'}
        with self.assertRaises(AgentError) as caught:self.errors.ingest(record)
        self.assertEqual(caught.exception.code,'MEMORY_SCOPE_DENIED')

    def test_issued_context_is_frozen_and_full_report_is_only_in_scoped_details(self):
        record = self.record(self.a)
        record['report']['diagnostic_groups'] = deepcopy(record['report']['hard_groups'])*25
        art = self.c.artifacts.write_json('diagnostics/full-report.json',record['report'])
        record['artifact_id'] = art['artifact_id']
        self.errors.ingest(record)
        self.c.context.error_research = self.errors
        self.c.context.science = SimpleNamespace(context=lambda *_:{'fit_diagnostics':[record]})
        ticket = self.h.issue()
        context_id = ticket['work_order']['context_id']
        before = self.c.artifacts.read_json(context_id)
        user = json.loads(before['messages'][1]['content'])
        self.assertNotIn('diagnostic_groups',user['profile_summary']['scientific_protocol']['fit_diagnostics'][0]['report'])
        details = self.c.artifacts.read_json(before['metadata']['detail_artifact_id'])
        self.assertEqual(len(details['sections']['fit_diagnostics/0']['report']['diagnostic_groups']),25)
        self.assertEqual(details['sections']['error_observation']['source_artifact_id'],record['artifact_id'])
        _,reply,_,_ = self.h.result('researcher-2',{'alpha':2},9)
        b = reply['lead_submission']['experiment_id']
        self.errors.ingest(self.record(b,cost=900))
        self.errors.advice([self.a,b],'researcher-1',self.h.root)
        self.assertEqual(self.c.artifacts.read_json(context_id),before)


if __name__ == '__main__':unittest.main()
