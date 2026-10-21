from copy import deepcopy
import json
import sqlite3
import threading
import unittest
from unittest.mock import patch
import uuid

from agent.artifacts import Artifacts
from agent.contracts import ContractBus
from agent.errors import AgentError
from agent.graph.identity import dependency_closure
from agent.graph.service import GraphService
from tests.graph_helpers import GraphHarness


class GraphTests(unittest.TestCase):
    def setUp(self):
        self.h = GraphHarness()
        self.g = self.h.g

    def tearDown(self):
        self.h.close()

    def test_reverse_completion_keeps_independent_heads(self):
        a, b = self.h.issue("researcher-1"), self.h.issue("researcher-2")
        ra, _ = self.h.submit(a, {"alpha": 1});rb, _ = self.h.submit(b, {"alpha": 2})
        self.h.commit(rb, 12);result = self.h.commit(ra, 10)
        heads = {h["researcher_id"]: h for h in self.g.heads(self.h.root)}
        self.assertEqual(heads['researcher-1']['experiment_id'],ra['lead_submission']['experiment_id'])
        self.assertEqual(heads['researcher-2']['experiment_id'],rb['lead_submission']['experiment_id'])
        self.assertEqual(result['head_results'][0]['disposition'],'advanced')
        self.assertNotEqual(result['graph_revision'],result['lead_commit']['store_revision'])

    def test_cached_experiment_adds_a_binding_without_training(self):
        a, b = self.h.issue("researcher-1"), self.h.issue("researcher-2")
        ra, _ = self.h.submit(a);self.h.commit(ra, 10)
        rb, _ = self.h.submit(b)
        self.assertEqual(rb['lead_submission']['disposition'],'duplicate')
        self.assertIsNotNone(rb['reused_evaluation_id'])
        self.h.commit(rb,10)
        self.assertEqual(self.g.db.execute('SELECT count(*) FROM experiments').fetchone()[0],1)
        self.assertEqual(len({h['experiment_id'] for h in self.g.heads(self.h.root)}),1)

    def test_seed_replica_is_same_solution_different_experiment(self):
        a=self.h.issue();ra,pa=self.h.submit(a,seed=42);self.h.commit(ra,10)
        b=self.h.issue();rb,pb=self.h.submit(b,seed=43)
        self.assertEqual(pa['manifest']['solution_id'],pb['manifest']['solution_id'])
        self.assertNotEqual(ra['lead_submission']['fingerprint'],rb['lead_submission']['fingerprint'])

    def test_proposal_cannot_change_its_peer_or_primary_parent(self):
        a=self.h.issue();p=self.h.proposal(a)
        bad=deepcopy(p);bad['proposal']['researcher_id']='researcher-2'
        with self.assertRaises(AgentError):self.g.invoke('G04',bad)
        bad=deepcopy(p);bad['proposal']['plan']['parent_ids']=['invented']
        with self.assertRaises(AgentError):self.g.invoke('G04',bad)
        self.assertEqual(self.g.db.execute('SELECT count(*) FROM experiments').fetchone()[0],0)

    def test_expired_order_frees_peer(self):
        a=self.h.issue();p=self.h.proposal(a)
        with patch.object(self.g,'elapsed',return_value=a['work_order']['expires_elapsed_ms']):
            with self.assertRaises(AgentError)as caught:self.g.invoke('G04',p)
        self.assertEqual(caught.exception.code,'WORK_ORDER_EXPIRED')
        self.assertEqual(self.g.order(a['work_order']['work_order_id'])['state'],'expired')
        self.h.issue()

    def test_only_dispatcher_can_write_graph(self):
        failures=[]
        def worker():
            try:self.g.invoke('G13',{'branch_id':None,'after_event_seq':0,'limit':1})
            except AgentError as e:failures.append(e.code)
        t=threading.Thread(target=worker);t.start();t.join()
        self.assertEqual(failures,['WRITER_THREAD_VIOLATION'])

    def test_fork_restores_each_peer_and_keeps_global_best(self):
        fork,_,idea=self.h.make_fork();self.assertEqual(idea['status'],'confirmed')
        old_best=self.g.store.best();cp=self.g.checkpoint_record(fork['checkpoint_id'])
        out=self.g.invoke('G02',fork,key='fork')
        for child in ['accept_branch','reject_branch']:
            self.assertEqual(self.g.heads(out[child]['branch_id']),cp['heads'])
        self.assertNotEqual(cp['heads'][0]['solution_id'],cp['heads'][1]['solution_id'])
        self.assertEqual(self.g.store.best(),old_best)
        self.assertEqual(out['parent_branch']['state'],'sealed')
        self.assertIn('feature.target_transform',out['reject_branch']['constraints']['forbidden_capability_ids'])

    def test_fork_is_binary_and_rejects_another_split(self):
        fork,_,_=self.h.make_fork();self.g.invoke('G02',fork,key='fork')
        with self.assertRaises(AgentError):self.g.invoke('G02',fork,key='second-fork')
        self.assertEqual(self.g.db.execute('SELECT count(*) FROM g_branches WHERE parent_branch_id=?',(self.h.root,)).fetchone()[0],2)

    def test_branch_limit_rolls_back_both_children(self):
        fork,_,_=self.h.make_fork();self.g.policy['max_total_branches']=2
        with self.assertRaises(AgentError)as caught:self.g.invoke('G02',fork)
        self.assertEqual(caught.exception.code,'BRANCH_LIMIT')
        self.assertEqual(self.g.branch(self.h.root)['state'],'active')
        self.assertEqual(self.g.db.execute('SELECT count(*) FROM g_branches').fetchone()[0],1)

    def test_issued_proposal_is_rejected_after_fork(self):
        fork,_,_=self.h.make_fork();a=self.h.issue();p=self.h.proposal(a,{'alpha':3})
        count=self.g.db.execute('SELECT count(*) FROM experiments').fetchone()[0]
        self.g.invoke('G02',fork)
        with self.assertRaises(AgentError):self.g.invoke('G04',p)
        self.assertEqual(self.g.db.execute('SELECT count(*) FROM experiments').fetchone()[0],count)

    def test_late_result_keeps_origin_but_can_win_global_selection(self):
        fork,_,_=self.h.make_fork();a=self.h.issue();late,_=self.h.submit(a,{'alpha':3,'target_transform':'shifted_log1p'})
        out=self.g.invoke('G02',fork);initial=self.g.heads(out['accept_branch']['branch_id'])
        committed=self.h.commit(late,7)
        self.assertEqual(committed['head_results'][0]['disposition'],'archived_branch_closed')
        self.assertEqual(self.g.heads(out['accept_branch']['branch_id']),initial)
        self.assertEqual(self.g.store.best(),late['lead_submission']['experiment_id'])

    def test_backtrack_context_is_from_own_reject_line(self):
        fork,_,_=self.h.make_fork();out=self.g.invoke('G02',fork)
        accept,reject=out['accept_branch'],out['reject_branch']
        self.g.invoke('G08',{'branch_id':accept['branch_id'],'expected_branch_revision':0,'action':'pause','reason':'plateau'})
        self.g.invoke('G08',{'branch_id':reject['branch_id'],'expected_branch_revision':0,'action':'resume','reason':'return'})
        a=self.h.issue(branch=reject['branch_id']);user=json.loads(self.g.artifacts.read_json(a['work_order']['context_id'])['messages'][1]['content'])
        cp=self.g.checkpoint_record(fork['checkpoint_id'])
        self.assertEqual(user['base_experiment_id'],cp['heads'][0]['experiment_id'])
        self.assertTrue(all(f['source_id'] in {h['experiment_id'] for h in cp['heads']} for f in user['facts']))
        self.assertIn('feature.target_transform',user['restrictions']['forbidden_capability_ids'])
        bad=self.h.proposal(a,{'alpha':1,'target_transform':'shifted_log1p'})
        with self.assertRaises(AgentError):self.g.invoke('G04',bad)

    def test_free_code_cannot_claim_a_hard_ban_is_verified(self):
        fork,_,_=self.h.make_fork();out=self.g.invoke('G02',fork);r=out['reject_branch']
        self.g.invoke('G08',{'branch_id':r['branch_id'],'expected_branch_revision':0,'action':'resume','reason':'test'})
        a=self.h.issue(branch=r['branch_id'])
        p=self.h.proposal(a,source='from agent.ml import build_pipeline\ndef make_pipeline(task_spec, params):\n    return build_pipeline("linear-regressor-v1", task_spec, params)\n')
        self.assertEqual(p['manifest']['verification_level'],'declared_only')
        with self.assertRaises(AgentError)as caught:self.g.invoke('G04',p)
        self.assertEqual(caught.exception.code,'UNVERIFIABLE_CONSTRAINT')

    def test_mixed_hyperparameters_do_not_prove_a_feature_effect(self):
        fork,evidence,_=self.h.make_fork();g=self.g
        # The supplied before/after pair changes alpha too; a title/score cannot make it controlled.
        bad=deepcopy(evidence);bad['baseline_experiment_ids']=[g.checkpoint_record(fork['checkpoint_id'])['heads'][1]['experiment_id']]
        patch_body=g.artifacts.read_json(bad['transfer_patch_artifact_id']);patch_body['idea_id']='new-invalid-idea'
        bad['transfer_patch_artifact_id']=g.files.put_json(patch_body,'transfer_patch')['artifact_id']
        with self.assertRaises(AgentError)as caught:g.invoke('G06',bad)
        self.assertEqual(caught.exception.code,'INVALID_COMPONENT_DIFF')

    def test_t4_failure_rolls_back_native_reservation_and_graph(self):
        a=self.h.issue();p=self.h.proposal(a);real=self.g.event
        def fail(operation,*args):
            if operation=='G04':raise RuntimeError('injected T4 crash')
            return real(operation,*args)
        with patch.object(self.g,'event',side_effect=fail):
            with self.assertRaises(AgentError):self.g.invoke('G04',p,key='atomic')
        for table in ['experiments','g_solutions','g_bindings']:
            self.assertEqual(self.g.db.execute('SELECT count(*) FROM '+table).fetchone()[0],0)
        self.assertEqual(self.g.order(a['work_order']['work_order_id'])['state'],'issued')
        self.g.invoke('G04',p,key='atomic')

    def test_t5_failure_rolls_back_selection_and_peer_cas(self):
        a=self.h.issue();r,_=self.h.submit(a);ev=self.h.evaluate(r,10)
        p={'experiment_id':ev['experiment_id'],'evaluation_id':ev['evaluation_id'],'selection_policy_id':'same-protocol-primary-v1'}
        real=self.g.event
        def fail(operation,*args):
            if operation=='G05':raise RuntimeError('injected T5 crash')
            return real(operation,*args)
        with patch.object(self.g,'event',side_effect=fail):
            with self.assertRaises(AgentError):self.g.invoke('G05',p,key='atomic-eval')
        self.assertIsNone(self.g.store.best())
        self.assertIsNone(self.g.heads(self.h.root)[0]['experiment_id'])
        self.assertEqual(self.g.store.get(ev['experiment_id'])['state'],'trained')
        self.g.invoke('G05',p,key='atomic-eval')

    def test_durable_idempotency_survives_a_new_contract_bus(self):
        a=self.h.issue();p=self.h.proposal(a);first=self.g.invoke('G04',p,key='durable')
        revision=self.g.revision();self.g.bus=ContractBus(self.g.run_id,self.g.log,self.g.schemas)
        replay=self.g.invoke('G04',p,key='durable')
        self.assertEqual(replay,first);self.assertEqual(self.g.revision(),revision)
        changed=deepcopy(p);changed['proposal']['proposal_id']='changed'
        with self.assertRaises(AgentError)as caught:self.g.invoke('G04',changed,key='durable')
        self.assertEqual(caught.exception.code,'IDEMPOTENCY_CONFLICT')

    def test_failed_and_invalid_training_release_the_peer(self):
        a=self.h.issue();r,_=self.h.submit(a);event=self.h.train_event(r,state='failed')
        self.g.store.apply_event(event,'train')
        self.assertEqual(self.g.order(a['work_order']['work_order_id'])['state'],'completed')
        b=self.h.issue();rb,_=self.h.submit(b,{'alpha':2});self.g.store.invalidate(rb['lead_submission']['experiment_id'],'bad predictions')
        self.assertEqual(self.g.order(b['work_order']['work_order_id'])['state'],'completed')
        self.h.issue()

    def test_obsolete_fencing_token_never_produces_evidence(self):
        a=self.h.issue();r,_=self.h.submit(a);bad=self.h.train_event(r,fence=999)
        with self.assertRaises(AgentError)as caught:self.g.store.apply_event(bad,'train')
        self.assertEqual(caught.exception.code,'STALE_LEASE')
        self.assertEqual(self.g.db.execute('SELECT count(*) FROM g_attempts').fetchone()[0],0)
        self.assertIsNone(self.g.store.best())

    def test_blob_recovery_finishes_a_durable_intent(self):
        import agent.graph.artifacts as blobs
        real=blobs.os.replace
        def fail(src,dst):
            if '/objects/' in str(dst):raise OSError('injected rename crash')
            return real(src,dst)
        with patch.object(blobs.os,'replace',side_effect=fail):
            with self.assertRaises(AgentError):self.g.files.put_json({'value':'new-byte-content'},'test_blob')
        self.assertGreater(self.g.db.execute("SELECT count(*) FROM g_artifact_intents WHERE state='prepared'").fetchone()[0],0)
        out=self.g.invoke('G12',{'mode':'repair','trusted_wall_seconds_remaining':60})
        self.assertEqual(out['state'],'recovered');self.assertTrue(out['repaired_intent_ids'])
        self.assertEqual(self.g.db.execute("SELECT count(*) FROM g_artifact_intents WHERE state='prepared'").fetchone()[0],0)

    def test_corrupt_pin_blocks_recovery_without_resetting_budget(self):
        art=self.g.files.put_json({'test':'immutable'},'test_blob');self.g.artifacts.resolve(art['artifact_id']).write_text('corrupt')
        deadline=self.g.budget.deadline
        out=self.g.invoke('G12',{'mode':'repair','trusted_wall_seconds_remaining':60})
        self.assertEqual(out['state'],'blocked');self.assertFalse(out['search_allowed'])
        self.assertEqual(self.g.budget.deadline,deadline)

    def test_recovery_rehydrates_artifacts_from_sql_locations(self):
        self.h.result('researcher-1',{'alpha':1},10)
        self.g.artifacts.entries={}
        out=self.g.invoke('G12',{'mode':'inspect','trusted_wall_seconds_remaining':60})
        self.assertEqual(out['state'],'healthy')
        self.h.issue()

    def test_plateau_uses_distinct_solutions_and_old_best(self):
        self.h.result('researcher-1',{'alpha':1},10)
        for a in [2,3,4]:self.h.result('researcher-1',{'alpha':a},11+a)
        out=self.g.invoke('G07',{'branch_id':self.h.root,'expected_branch_revision':0})['assessment']
        self.assertEqual(out['status'],'plateau');self.assertEqual(out['utility_gain'],0)
        self.assertEqual(out['distinct_solution_count'],3)

    def test_final_ranking_keeps_ancestors_and_paused_branches(self):
        fork,_,_=self.h.make_fork();best=self.g.store.best();self.g.invoke('G02',fork)
        out=self.g.invoke('G11',{'task_spec_id':self.g.spec['task_spec_id'],'validation_id':self.g.validation['validation_id'],'selection_policy_id':'same-protocol-primary-v1','scope':'final'})
        ranking=self.g.artifacts.read_json(out['ranking_artifact']['artifact_id'])
        self.assertEqual(ranking['rows'][0]['experiment_id'],best)
        graph=self.g.artifacts.read_json(out['graph_artifact']['artifact_id'])
        self.assertEqual(len(graph['branches']),3)
        self.assertEqual(len(graph['experiments']),4)
        self.g.schemas.validate(graph,'GGraphExport')
        self.assertEqual(self.g.db.execute('PRAGMA foreign_key_check').fetchall(),[])

    def test_dependency_closure_detects_renamed_wrappers(self):
        base={'component_id':'f','capability_id':'feature.forbidden','dependency_component_ids':[]}
        wrapper={'component_id':'renamed','capability_id':'feature.wrapper','dependency_component_ids':['f']}
        caps,_=dependency_closure([base,wrapper])
        with self.assertRaises(AgentError):self.g.invoke('I01',{'constraints':{'required_capability_ids':[],'forbidden_capability_ids':['feature.forbidden']},'effective_capability_ids':caps})

    def test_history_and_dag_cycles_are_rejected_by_sql(self):
        self.h.result('researcher-1',{'alpha':1},10);self.h.result('researcher-1',{'alpha':2},9)
        sols=[r[0]for r in self.g.db.execute('SELECT solution_id FROM g_solutions ORDER BY rowid')]
        with self.assertRaises(sqlite3.IntegrityError),self.g.store.transaction():
            self.g.db.execute('INSERT INTO g_solution_edges VALUES(?,?,?,?)',(self.g.run_id,sols[1],sols[0],'idea_donor'))
        with self.assertRaises(sqlite3.IntegrityError),self.g.store.transaction():
            self.g.db.execute("UPDATE g_solutions SET core_json='{}'")
