from dataclasses import replace
import json
import unittest

from agent.catalog import Catalog
from agent.context import output_schemas
from agent.errors import AgentError
from agent.llm.gateway import Gateway
from agent.ml import NumericInteractions, build_pipeline
from agent.graph.controller import train
from tests.graph_helpers import GraphHarness


class GraphFeatureTests(unittest.TestCase):
    def test_numeric_interactions_help_a_quadratic_and_are_bounded(self):
        import numpy as np
        import pandas as pd
        from threadpoolctl import threadpool_limits
        spec={'task_kind':'regression'}
        X=pd.DataFrame({'x':np.linspace(-3,3,100),'category':['a']*100});y=5*X['x'].to_numpy()**2
        with threadpool_limits(limits=1):
            plain=build_pipeline('linear-regressor-v1',spec,{'alpha':.01,'numeric_interactions':False}).fit(X,y)
            feature=build_pipeline('linear-regressor-v1',spec,{'alpha':.01,'numeric_interactions':True}).fit(X,y)
        self.assertLess(np.mean((feature.predict(X)-y)**2),np.mean((plain.predict(X)-y)**2)/100)
        many=pd.DataFrame({str(i):np.arange(4,dtype=float)for i in range(20)})
        transform=NumericInteractions().fit(many.iloc[:2]);self.assertEqual(len(transform.pairs_),36)
        self.assertEqual(transform.transform(many.iloc[2:]).shape[1],56)

    def test_full_request_guard_includes_added_schema(self):
        h=GraphHarness()
        try:
            ticket=h.issue();context=h.g.artifacts.read_json(ticket['work_order']['context_id'])
            class NeverTransport:
                calls=0
                def complete(self,*args):self.calls+=1;raise AssertionError('Must reject locally before transport')
            transport=NeverTransport();cfg=replace(h.g.config,graph_context_bytes=1)
            gateway=Gateway(cfg,h.g.budget,h.g.artifacts,h.g.bus,h.g.log,output_schemas(h.g.bus),transport)
            before=h.g.budget.calls
            with self.assertRaises(AgentError)as caught:
                gateway.complete({'purpose':'research_proposal','context_artifact_id':ticket['work_order']['context_id'],'response_schema_id':'ResearchChoice-v1',
                    'tool_schema_ids':[],'max_output_tokens':1024,'deadline_seconds':30,'attempt_limit':2})
            self.assertEqual(caught.exception.code,'CONTEXT_TOO_LARGE')
            self.assertEqual(transport.calls,0);self.assertEqual(h.g.budget.calls,before)
        finally:h.close()

    def test_oversized_mandatory_context_is_an_explicit_error_and_releases_peer(self):
        h=GraphHarness()
        try:
            original=h.g.context.task_text;h.g.context.task_text='x'*100000
            with self.assertRaises(AgentError)as caught:h.issue()
            self.assertEqual(caught.exception.code,'CONTEXT_TOO_LARGE')
            self.assertEqual(h.g.db.execute("SELECT count(*) FROM g_work_orders WHERE state IN ('issued','preparing','submitted')").fetchone()[0],0)
            h.g.context.task_text=original;h.issue()
        finally:h.close()

    def test_repeated_initialization_does_not_rewrite_pinned_files(self):
        h=GraphHarness()
        try:
            before=h.g.files.resolve(h.g.metadata('environment_artifact_id'))
            deadline=h.g.budget.deadline;revision=h.g.revision()
            h.g.initialize_defaults()
            self.assertEqual(before,h.g.files.resolve(before['artifact_id']))
            self.assertEqual(h.g.budget.deadline,deadline);self.assertEqual(h.g.revision(),revision)
        finally:h.close()

    def test_plateau_backtracking_is_selected_from_persisted_evidence(self):
        h=GraphHarness()
        try:
            fork,_,_=h.make_fork();out=h.g.invoke('G02',fork);accept=out['accept_branch']['branch_id']
            # First establish an eligible measured reference inside the required-feature branch.
            h.result('researcher-1',{'alpha':1,'target_transform':'shifted_log1p'},8,branch=accept)
            for alpha in [3,4,5]:h.result('researcher-1',{'alpha':alpha,'target_transform':'shifted_log1p'},20+alpha,branch=accept)
            assessment=h.g.invoke('G07',{'branch_id':accept,'expected_branch_revision':0})['assessment']
            self.assertEqual(assessment['status'],'plateau')
            cp=h.g.invoke('G01',{'branch_id':accept,'expected_branch_revision':0,'reason':'manual'})['checkpoint']
            decision=h.g.invoke('G14',{'checkpoint_id':cp['checkpoint_id'],'budget':h.g.budget.snapshot(),'plateau_assessment_id':assessment['assessment_id']})['decision']
            self.assertEqual(decision['action'],'BACKTRACK');self.assertEqual(decision['parameters']['target_branch_id'],out['reject_branch']['branch_id'])
        finally:h.close()

    def test_catalog_rejects_integer_disguised_as_boolean(self):
        h=GraphHarness()
        try:
            with self.assertRaises(AgentError):h.g.catalog.normalize('hist-gradient-regressor-v1',{'early_stopping':0})
        finally:h.close()

    def test_nested_fork_inherits_the_previous_ban(self):
        import uuid
        h=GraphHarness()
        try:
            fork,_,_=h.make_fork();first=h.g.invoke('G02',fork);reject=first['reject_branch']['branch_id']
            h.g.invoke('G08',{'branch_id':reject,'expected_branch_revision':0,'action':'resume','reason':'counterfactual'})
            source_ticket,source,payload,_=h.result('researcher-1',{'alpha':1,'numeric_interactions':True},9,branch=reject)
            source_id=source['lead_submission']['experiment_id']
            _,pilot,_,_=h.result('researcher-2',{'alpha':2,'numeric_interactions':True},11,branch=reject,action='transfer',donors=[source_id])
            component=next(c for c in payload['manifest']['core']['components']if c['capability_id']=='feature.numeric_interactions')
            patch_body={'patch_version':'transfer-1.0','idea_id':'idea-'+uuid.uuid4().hex,'supported_recipe_ids':list(h.g.catalog.recipes),
                'required_base_capability_ids':[],'operations':[{'operation':'add_component','target_component_id':None,'component':h.g.constructor.external_component(component)}],
                'expected_added_capability_ids':['feature.numeric_interactions'],'expected_removed_capability_ids':[]}
            patch_art=h.g.files.put_json(patch_body,'transfer_patch')
            p={'title':'Nested feature fixture','hypothesis':'State test','source_experiment_id':source_id,'capability_ids':['feature.numeric_interactions'],
                'fork_checkpoint_id':source_ticket['work_order']['base_team_checkpoint_id'],'baseline_experiment_ids':[source_ticket['work_order']['base_experiment_id']],
                'candidate_experiment_ids':[source_id],'transfer_pilot_experiment_ids':[pilot['lead_submission']['experiment_id']],'transfer_patch_artifact_id':patch_art['artifact_id']}
            idea=h.g.invoke('G06',p)['idea'];self.assertEqual(idea['status'],'confirmed')
            second=h.g.invoke('G02',{'parent_branch_id':reject,'expected_branch_revision':1,'checkpoint_id':p['fork_checkpoint_id'],'idea_id':idea['idea_id']})
            for name in ['accept_branch','reject_branch']:
                self.assertIn('feature.target_transform',second[name]['constraints']['forbidden_capability_ids'])
            self.assertIn('feature.numeric_interactions',second['accept_branch']['constraints']['required_capability_ids'])
            self.assertIn('feature.numeric_interactions',second['reject_branch']['constraints']['forbidden_capability_ids'])
            self.assertEqual(h.g.db.execute('SELECT count(*) FROM g_branches').fetchone()[0],5)
        finally:h.close()

    def test_recovery_applies_a_trusted_evaluation_committed_before_graph_projection(self):
        h=GraphHarness()
        try:
            ticket=h.issue();reply,_=h.submit(ticket);ev=h.evaluate(reply,10)
            out=h.g.invoke('G12',{'mode':'repair','trusted_wall_seconds_remaining':60})
            self.assertEqual(out['state'],'recovered')
            self.assertEqual(h.g.store.best(),ev['experiment_id'])
            self.assertEqual(h.g.order(ticket['work_order']['work_order_id'])['state'],'completed')
        finally:h.close()

    def test_recovery_closes_unstarted_submitted_work_without_retraining(self):
        h=GraphHarness()
        try:
            ticket=h.issue();reply,_=h.submit(ticket)
            out=h.g.invoke('G12',{'mode':'repair','trusted_wall_seconds_remaining':60})
            self.assertEqual(out['state'],'recovered')
            self.assertEqual(h.g.store.get(reply['lead_submission']['experiment_id'])['state'],'cancelled')
            self.assertEqual(h.g.order(ticket['work_order']['work_order_id'])['state'],'completed')
        finally:h.close()

    def test_replayed_submission_uses_current_state_instead_of_its_historical_queued_reply(self):
        h=GraphHarness()
        try:
            ticket=h.issue();payload=h.proposal(ticket)
            reply=h.g.invoke('G04',payload,key='submit/'+ticket['work_order']['work_order_id']);h.commit(reply,10)
            ticket['_proposal']=payload['proposal'];h.controller.graph=h.g;h.controller.degraded=False
            attempted=train(h.controller,[payload['proposal']['plan']],[ticket])
            self.assertEqual(attempted,0);self.assertFalse(h.controller.degraded)
            self.assertEqual(h.g.store.get(reply['lead_submission']['experiment_id'])['state'],'evaluated')
        finally:h.close()

    def test_explicit_agent_stop_is_persisted_as_a_proposal(self):
        h=GraphHarness()
        try:
            ticket=h.issue();work=ticket['work_order']
            data=h.g.invoke('G04',{'work_order_id':work['work_order_id'],'proposal':{'proposal_id':'stop-test','researcher_id':work['researcher_id'],
                'plan':None,'decision':'stop','stop_reason':'No useful candidate left'},'manifest':None,'lineage':None})
            self.assertEqual(data['disposition'],'stopped')
            stored=json.loads(h.g.db.execute("SELECT record_json FROM g_proposals WHERE proposal_id='stop-test'").fetchone()[0])
            self.assertEqual(stored['proposal']['stop_reason'],'No useful candidate left')
        finally:h.close()
