"""Fit-only cross-fitted blending, followed by an ordinary outer evaluation."""
from copy import deepcopy
import numpy as np
from sklearn.base import BaseEstimator
from agent.learned_features import inner_splits
from agent.specialists import make_specialist_pipeline

FACTORY_SOURCE=('from agent.ensemble import make_ensemble_pipeline\n\n'
    'def make_pipeline(task_spec, params):\n    return make_ensemble_pipeline(task_spec, params)\n')


def aligned_probabilities(model,X,classes):
    raw=np.asarray(model.predict_proba(X),dtype=float)
    order=[str(c) for c in model.classes_]
    if raw.shape!=(len(X),len(order)) or len(set(order))!=len(order) or not set(order)<=set(classes):
        raise ValueError('Ensemble probability class order/shape mismatch')
    result=np.zeros((len(X),len(classes)))
    for i,c in enumerate(order):result[:,classes.index(c)]=raw[:,i]
    if not np.isfinite(result).all() or (result<0).any() or not np.allclose(result.sum(axis=1),1.,atol=1e-6):
        raise ValueError('Invalid ensemble member probabilities')
    return result


def select_weights(predictions, truth, spec):
    """Nonnegative simplex; standalone candidates always remain feasible."""
    from scipy.optimize import minimize
    from agent.evaluation import score_predictions
    count=predictions.shape[1]
    def metric(weights):
        blended=np.tensordot(predictions,weights,axes=(1,0))
        if spec['task_kind']!='regression' and spec['metric']['prediction_kind']=='labels':
            blended=np.asarray(spec['allowed_labels'])[blended.argmax(axis=1)]
        return float(score_predictions(spec,truth,blended))
    sign=1 if spec['metric']['direction']=='minimize' else -1
    singles=[metric(np.eye(count)[i]) for i in range(count)]
    best=int(np.argmin(np.asarray(singles)*sign));weights=np.eye(count)[best]
    best_score=singles[best]
    if spec['metric']['metric_id']=='rmse':
        errors=predictions-np.asarray(truth)[:,None]
        covariance=errors.T@errors/max(1,len(truth))
        result=minimize(lambda w:float(w@covariance@w),weights,jac=lambda w:2*covariance@w,
            bounds=[(0.,1.)]*count,constraints={'type':'eq','fun':lambda w:w.sum()-1.,'jac':lambda w:np.ones_like(w)},
            method='SLSQP',options={'maxiter':100,'ftol':1e-8})
        if result.success and np.isfinite(result.x).all():
            candidate=np.maximum(result.x,0);candidate/=candidate.sum()
            score=metric(candidate)
            if score<best_score:weights,best_score=candidate,score
    else:
        for _ in range(3):
            changed=False
            for i in range(count):
                for fraction in (.1,.25,.5):
                    candidate=(1-fraction)*weights+fraction*np.eye(count)[i]
                    score=metric(candidate)
                    if score*sign<best_score*sign-1e-12:
                        weights,best_score,changed=candidate,score,True
            if not changed:break
    return weights,{'best_single_index':best,'best_single_score':singles[best],'blend_score':best_score,
                    'weights':weights.tolist(),'selection_scope':'inner_fit_predictions_only'}


class CrossFittedEnsemble(BaseEstimator):
    def __init__(self, task_spec, members, folds=3, cpu_threads=1, seed=42):
        self.task_spec,self.members,self.folds,self.cpu_threads,self.seed=task_spec,members,folds,cpu_threads,seed

    def _model(self, member):
        params=deepcopy(member)
        params.update(_cpu_threads=self.cpu_threads,_seed=self.seed)
        return make_specialist_pipeline(self.task_spec,params)

    def fit(self,X,y):
        from agent.evaluation import score_predictions
        spec=self.task_spec;regression=spec['task_kind']=='regression'
        if not 2<=len(self.members)<=5 or not 2<=self.folds<=5:
            raise ValueError('Ensemble requires two to five members and two to five inner folds')
        context=spec.get('_fit_context')
        if not context:raise ValueError('Ensemble requires a frozen executor-provided fit context')
        if not regression:self.classes_=np.asarray(spec['allowed_labels'])
        truth=np.asarray(y,dtype=float if regression else str)
        splits=list(inner_splits(X,truth,classification=not regression,folds=self.folds,seed=self.seed,
            time_column=context.get('column') if context['strategy']=='temporal_holdout' else None,
            group_column=context.get('column') if context['strategy']=='group_holdout' else None))
        if not splits:raise ValueError('Insufficient independent fit units for cross-fitted blending')
        shape=(len(X),len(self.members)) if regression else (len(X),len(self.members),len(self.classes_))
        if np.prod(shape)>5000000:raise ValueError('Ensemble prediction memory limit exceeded')
        predictions=np.full(shape,np.nan);covered=np.zeros(len(X),dtype=bool);fits=0
        for a,b in splits:
            if not regression and len(set(truth[a]))<2:continue
            if covered[b].any() or len(set(a)&set(b)):raise ValueError('Invalid cross-fitting partition')
            for i,member in enumerate(self.members):
                model=self._model(member);model.fit(X.iloc[a],truth[a]);fits+=1
                values=model.predict(X.iloc[b]) if regression else aligned_probabilities(model,X.iloc[b],list(self.classes_))
                predictions[b,i]=values
                del model
            covered[b]=True
        rows=np.flatnonzero(covered)
        if len(rows)<20 or not np.isfinite(predictions[rows]).all():raise ValueError('Insufficient valid OOF coverage')
        # Reuse the same scenario for the small meta-level check. Temporal first
        # periods remain unavailable instead of being filled with in-sample data.
        meta=list(inner_splits(X.iloc[rows],truth[rows],classification=not regression,folds=3,seed=self.seed+31,
            time_column=context.get('column') if context['strategy']=='temporal_holdout' else None,
            group_column=context.get('column') if context['strategy']=='group_holdout' else None))
        tune,check=meta[-1] if meta else (np.arange(len(rows)),np.array([],dtype=int))
        weights,report=select_weights(predictions[rows][tune],truth[rows][tune],spec)
        baseline=np.eye(len(self.members))[report['best_single_index']]
        def metric(w,indices):
            blended=np.tensordot(predictions[rows][indices],w,axes=(1,0))
            if not regression and spec['metric']['prediction_kind']=='labels':blended=self.classes_[blended.argmax(axis=1)]
            return float(score_predictions(spec,truth[rows][indices],blended))
        if len(check):
            before,after=metric(baseline,check),metric(weights,check)
            improvement=before-after if spec['metric']['direction']=='minimize' else after-before
            if improvement<=0:weights=baseline
            report.update(meta_check_rows=len(check),meta_check_before=before,meta_check_after=after,
                blend_retained=bool(improvement>0))
        else:
            weights=baseline
            report.update(meta_check_rows=0,blend_retained=False,reason='insufficient independent meta-check units')
        if not regression:
            self.class_adjustments_=np.ones(len(self.classes_))
            if spec['metric']['metric_id']=='f1_macro' and spec['metric']['prediction_kind']=='labels' and len(check):
                from sklearn.metrics import f1_score
                probabilities=np.tensordot(predictions[rows],weights,axes=(1,0))
                def decision_score(adjustments,indices):
                    labels=self.classes_[(probabilities[indices]*adjustments).argmax(axis=1)]
                    return float(f1_score(truth[rows][indices],labels,labels=self.classes_,average='macro',zero_division=0))
                adjustments=self.class_adjustments_.copy()
                for _ in range(2):
                    for i,label in enumerate(self.classes_):
                        if min(np.sum(truth[rows][tune]==label),np.sum(truth[rows][check]==label))<3:continue
                        best=decision_score(adjustments,tune)
                        for factor in (.5,.75,1.,1.5,2.):
                            candidate=adjustments.copy();candidate[i]=factor
                            value=decision_score(candidate,tune)
                            if value>best+1e-12:adjustments,best=candidate,value
                before,after=decision_score(self.class_adjustments_,check),decision_score(adjustments,check)
                if after>before:self.class_adjustments_=adjustments
                report['class_decision']={'adjustments':self.class_adjustments_.tolist(),'inner_check_before':before,
                    'inner_check_after':after,'accepted':bool(after>before),'probability_calibration':False}
        self.weights_=weights;self.models_=[]
        for weight,member in zip(weights,self.members):
            model=None
            if weight>1e-8:model=self._model(member).fit(X,truth);fits+=1
            self.models_.append(model)
        self.report_={'operator':'cross_fitted_ensemble','fit_rows':len(X),'oof_rows':len(rows),
            'folds':len(splits),'member_count':len(self.members),'member_fits':fits,'weights':weights.tolist(),
            'member_recipes':[m['base_recipe_id'] for m in self.members], 'meta_selection':report,
            'protocol_sha256':context.get('protocol_sha256'), 'training_scope':'current_fit_only; full pipeline refit per inner fold',
            'confirmation_used':False,'limitations':'One internal meta-check; outer evaluation still required. Different architectures do not prove error independence.'}
        return self

    def predict_proba(self,X):
        if self.task_spec['task_kind']=='regression':raise TypeError('Regression has no probabilities')
        return sum(w*aligned_probabilities(m,X,list(self.classes_)) for w,m in zip(self.weights_,self.models_) if m is not None)

    def predict(self,X):
        if self.task_spec['task_kind']!='regression':return self.classes_[(self.predict_proba(X)*self.class_adjustments_).argmax(axis=1)]
        return sum(w*np.asarray(m.predict(X)).reshape(-1) for w,m in zip(self.weights_,self.models_) if m is not None)


def make_ensemble_pipeline(task_spec,params):
    options=deepcopy(params)
    members=options.pop('members');folds=options.pop('folds',3)
    threads=options.pop('_cpu_threads',1);seed=options.pop('_seed',42)
    if options:raise ValueError('Unknown ensemble parameters')
    return CrossFittedEnsemble(task_spec,members,folds,threads,seed)
