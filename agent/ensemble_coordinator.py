"""Dispatcher-owned ensemble proposals and version-bound complementarity reports."""
from copy import deepcopy
import hashlib
import json
import time
from itertools import combinations
from agent.contracts import canonical
from agent.artifacts import file_sha256
from agent.errors import AgentError
from agent.specialists import member
from agent.ensemble import FACTORY_SOURCE


class EnsembleCoordinator:
    def __init__(self,controller):
        self.c=controller;self.attempted=set();self.last_review=0.;self.last_signature=None

    def candidates(self,service):
        c=self.c;choices={}
        for v in service.versions.values():
            for eid in dict.fromkeys([v.get('best_experiment_id'),v.get('initial_experiment_id')]):
                if not eid:continue
                exp=c.store.get(eid);plan=exp['plan'];ev=c.store.experiment_evaluation(eid)
                if (exp['state']!='evaluated' or not ev or not ev['valid'] or ev['violations'] or
                    plan['validation_id']!=c.validation['validation_id'] or any(
                        plan[k]!=c.spec[k] for k in ('task_spec_id','dataset_id'))):continue
                code=c.artifacts.resolve(plan['code_artifact_id'])
                if file_sha256(code)!=c.artifacts.meta(plan['code_artifact_id'])['sha256']:continue
                representation=member(plan,code.read_text(encoding='utf-8'))
                if not representation:continue
                key=representation['base_recipe_id'];quality=service.quality(ev['primary_metric'])
                choices.setdefault(key,{})[eid]={'experiment_id':eid,'representation':representation,'quality':quality,
                    'metric':ev['primary_metric'],'version':v,
                    'estimated_cost':sum(c.budget.duration.estimate(plan,p) for p in ('train','refit'))}
        # A bounded current frontier, not a lifetime exclusion: retain both the
        # strongest and cheapest measured configuration of each architecture.
        selected={}
        for rows in choices.values():
            for row in (max(rows.values(),key=lambda r:r['quality']),min(rows.values(),key=lambda r:r['estimated_cost'])):
                selected[row['experiment_id']]=row
        return sorted(selected.values(),key=lambda r:r['quality'],reverse=True)

    def prediction(self,eid):
        import pandas as pd
        c=self.c;protocol=c.evaluator.protocols[c.validation['validation_id']]
        wanted='validation_predictions' if c.spec['task_kind']=='regression' else 'validation_probabilities'
        for row in c.store.db.execute('SELECT record FROM job_events ORDER BY rowid DESC'):
            event=json.loads(row[0])
            if event['experiment_id']!=eid or event['state']!='succeeded':continue
            for aid in event['output_artifact_ids']:
                meta=c.artifacts.meta(aid)
                if meta['kind']!=wanted:continue
                path=c.artifacts.resolve(aid)
                if file_sha256(path)!=meta['sha256']:raise AgentError('CANDIDATE_CHANGED','Ensemble prediction source changed')
                frame=pd.read_csv(path)
                if frame['__row_index'].tolist()!=protocol['valid_indices']:
                    raise AgentError('INVALID_PREDICTIONS','Ensemble members use different validation rows')
                expected=c.spec['allowed_labels'] if wanted=='validation_probabilities' else [c.spec['target_column']]
                if list(frame.columns)!=['__row_index',*expected]:
                    raise AgentError('INVALID_PREDICTIONS','Ensemble class/target columns differ')
                return frame.set_index('__row_index').loc[protocol['selection_indices'],expected].to_numpy(),aid
        return None,None

    def context(self,rows):
        import numpy as np
        c=self.c;records=[];predictions=[]
        for row in rows:
            values,aid=self.prediction(row['experiment_id'])
            records.append({'experiment_id':row['experiment_id'],'recipe_id':row['representation']['base_recipe_id'],
                            'metric':row['metric'],'prediction_artifact_id':aid})
            predictions.append(values)
        report={'validation_id':c.validation['validation_id'],'protocol_sha256':c.validation['protocol_sha256'],
            'members':records,'scope':'selection_validation_only; no confirmation labels',
            'instruction':'Correlation/disagreement is descriptive, not proof of useful marginal gain. Read the actual evaluated ensemble result.'}
        if c.spec['task_kind']=='regression' and all(p is not None for p in predictions):
            from agent.data_transport import transport
            protocol=c.evaluator.protocols[c.validation['validation_id']]
            truth=transport(c.artifacts).read(c.spec['train_artifact_id'],c.spec,columns=[c.spec['target_column']],rows=protocol['selection_indices'])[c.spec['target_column']].to_numpy()
            errors=np.column_stack([p.reshape(-1)-truth for p in predictions])
            if np.isfinite(errors).all():
                with np.errstate(invalid='ignore',divide='ignore'):corr=np.corrcoef(errors,rowvar=False)
                report['residual_correlation']=np.where(np.isfinite(corr),corr,0.).tolist()
                report['error_second_moments']=(errors.T@errors/max(1,len(errors))).tolist()
        elif all(p is not None for p in predictions):
            report['prediction_disagreement']=[[float(np.mean(a.argmax(axis=1)!=b.argmax(axis=1))) for b in predictions] for a in predictions]
        return report

    def review(self,service,root):
        c=self.c
        if len(c.spec.get('target_columns',[c.spec['target_column']]))>1:return
        if time.monotonic()-self.last_review<c.config.ensemble_review_seconds:return
        rows=self.candidates(service)
        if len(rows)<2:return
        signature=hashlib.sha256(canonical([[r['experiment_id'],r['representation']] for r in rows]).encode()).hexdigest()
        if signature==self.last_signature:return
        self.last_review=time.monotonic()
        # Proposed pipelines stay optional; no score is fabricated from a fitted
        # weight search on the same selection rows. Worker performs fresh OOF.
        params={'members':[r['representation'] for r in rows],'folds':c.config.ensemble_folds}
        plan=c.warmup_plan('python-pipeline-v1',params)
        plan['model']['family']='custom'
        from tools.code import write_code
        source=write_code({'researcher_id':'ensemble-coordinator','filename':'ensemble.py',
            'content':FACTORY_SOURCE,'declared_entrypoint':'make_pipeline'},c.artifacts)['artifact']
        plan.update(code_artifact_id=source['artifact_id'],hypothesis='Test complementary architecture representations with fit-only OOF weights.',
            expected_effect='Measure the complete ensemble on the frozen outer protocol; standalone fallback stays available.',
            feature_steps=[{'step':'cross_fitted_ensemble','members':[r['experiment_id'] for r in rows]}])
        affordable=[]
        for size in range(min(5,len(rows)),1,-1):
            for committee in combinations(rows,size):
                if len({r['representation']['base_recipe_id'] for r in committee})!=size:continue
                plan['model']['hyperparameters']['members']=[r['representation'] for r in committee]
                decision=c.budget.can_train(plan)
                if decision['allowed']:
                    affordable.append((sum(r['quality'] for r in committee)/size,-decision['estimated_seconds'],committee,decision))
            if affordable:break
        if not affordable:
            c.log.event('ensemble.review_deferred',reason='complete_oof_refit_output_do_not_fit',members=len(rows));return
        _,_,committee,admission=max(affordable,key=lambda r:r[:2])
        rows=list(committee)
        plan['model']['hyperparameters']['members']=[r['representation'] for r in rows]
        candidate_hash=hashlib.sha256(canonical(plan['model']['hyperparameters']).encode()).hexdigest()
        if candidate_hash in self.attempted:
            self.last_signature=signature
            return
        report=self.context(rows)
        art=c.artifacts.write_json('ensembles/complementarity-'+signature+'.json',report,'ensemble_analysis')
        c.context.ensemble_status={'artifact_id':art['artifact_id'],**report}
        plan['feature_steps'][0]['members']=[r['experiment_id'] for r in rows]
        from agent.pipeline_candidates import execution_ticket
        from agent.graph.controller import bind_trusted
        ticket=execution_ticket(c,rows[0]['version']['ticket'],rows[0]['experiment_id'],
            donors=[r['experiment_id'] for r in rows[1:]])
        plan=bind_trusted(plan,ticket)
        service.add(plan,ticket)
        self.last_signature=signature
        self.attempted.add(candidate_hash)
        c.log.event('ensemble.candidate_queued',members=len(plan['model']['hyperparameters']['members']),
            representation_signature=candidate_hash,estimated_seconds=admission['estimated_seconds'],
            selection_labels_used_for_weights=False)
