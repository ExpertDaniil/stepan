"""Protocol-bound, fit-only target history and optional Macro F1 decisions."""
from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold


def folds(X, y, policy, *, count=5, classification=False):
    if not policy or policy.get('strategy') not in {
            'random_holdout','stratified_holdout','group_holdout','temporal_holdout'}:
        raise ValueError('Supervised operations require the controller validation policy')
    strategy=policy['strategy'];seed=policy.get('seed',42)
    count=min(count,len(X))
    if count<2:raise ValueError('Too few training rows for cross fitting')
    if strategy in {'group_holdout','temporal_holdout'}:
        column=policy.get('column')
        if column not in X:raise ValueError('Validation column is unavailable to the supervised operator')
        values=X[column]
        if strategy=='group_holdout':
            from agent.feature_engineering import _keys
            groups=_keys(values);count=min(count,groups.nunique())
            if count<2:raise ValueError('Too few independent training groups')
            return list(GroupKFold(count).split(X,groups=groups))
        if values.isna().any():raise ValueError('Temporal values must be complete')
        if not pd.api.types.is_numeric_dtype(values):
            values=pd.to_datetime(values,format='ISO8601',utc=True,errors='raise')
        periods=sorted(values.unique())
        blocks=np.array_split(np.arange(len(periods)),min(count,len(periods)))
        result=[];past=[]
        for block in blocks:
            current=[periods[i] for i in block]
            result.append((np.flatnonzero(values.isin(past)),np.flatnonzero(values.isin(current))))
            past.extend(current)
        return result  # The first block has no history, never future labels.
    if classification:
        count=min(count,int(pd.Series(y).value_counts().min()))
        if count<2:return []
        return list(StratifiedKFold(count,shuffle=True,random_state=seed).split(X,y))
    return list(KFold(count,shuffle=True,random_state=seed).split(X))


def bind_validation_policy(model, policy):
    """Worker binds the frozen policy before fitting; sklearn clones retain it."""
    objects=[model]
    if hasattr(model,'get_params'):objects.extend(model.get_params(deep=True).values())
    seen=set()
    for obj in objects:
        if id(obj) in seen:continue
        seen.add(id(obj))
        if isinstance(obj,(CrossFitHistory,MacroF1DecisionClassifier)):
            obj.set_params(validation_policy=deepcopy(policy))
        else:
            # Keep the feature language independent of this module's imports.
            from agent.feature_engineering import FeatureProgram
            if isinstance(obj,FeatureProgram):obj.set_params(validation_policy=deepcopy(policy))
    return model


class CrossFitHistory(TransformerMixin,BaseEstimator):
    def __init__(self,columns,alpha=10.,validation_policy=None,max_groups=100000):
        self.columns,self.alpha,self.validation_policy,self.max_groups=columns,alpha,validation_policy,max_groups

    def _keys(self,X):
        from agent.feature_engineering import _keys
        parts=[_keys(X[c]) for c in self.columns]
        key=parts[0].str.len().astype(str)+':'+parts[0]
        for part in parts[1:]:key=key+part.str.len().astype(str)+':'+part
        return key

    def _stats(self,keys,y):
        if keys.nunique()>self.max_groups:raise ValueError('Target history exceeds fitted group limit')
        stats=pd.DataFrame({'key':keys.to_numpy(),'y':y}).groupby('key',sort=False).y.agg(['sum','count'])
        return stats,float(np.mean(y)) if len(y) else np.nan

    def _encode(self,keys,stats,prior):
        count=keys.map(stats['count']).fillna(0).to_numpy(float)
        sums=keys.map(stats['sum']).fillna(0).to_numpy(float)
        mean=np.full(len(keys),prior,dtype=float)
        known=count>0
        mean[known]=(sums[known]+self.alpha*prior)/(count[known]+self.alpha)
        return pd.DataFrame({'mean':mean,'count':count},index=keys.index)

    def fit(self,X,y):
        target=np.asarray(y,dtype=float)
        if target.ndim!=1 or len(target)!=len(X) or not np.isfinite(target).all():
            raise ValueError('Target history needs one finite numeric regression target')
        if not np.isfinite(self.alpha) or self.alpha<0:raise ValueError('Invalid target history smoothing')
        if not 1<=len(self.columns)<=4 or any(c not in X for c in self.columns):
            raise ValueError('Target history needs 1..4 available input columns')
        self.statistics_,self.prior_=self._stats(self._keys(X),target)
        self.report_={'fit_rows':len(X),'groups':len(self.statistics_),'columns':list(self.columns),
            'alpha':float(self.alpha),'scope':'fit_only_cross_fitted_training; full_fit_mapping_at_predict'}
        return self

    def fit_transform(self,X,y=None,**fit_params):
        self.fit(X,y)
        target=np.asarray(y,dtype=float);keys=self._keys(X)
        result=pd.DataFrame({'mean':np.nan,'count':0.},index=X.index)
        partitions=folds(X,target,self.validation_policy)
        for train,valid in partitions:
            if not len(train):continue
            stats,prior=self._stats(keys.iloc[train],target[train])
            result.iloc[valid]=self._encode(keys.iloc[valid],stats,prior).to_numpy()
        self.report_.update(folds=len(partitions),history_free_rows=int(result['mean'].isna().sum()))
        return result

    def transform(self,X):
        return self._encode(self._keys(X),self.statistics_,self.prior_)


class MacroF1DecisionClassifier(ClassifierMixin,BaseEstimator):
    """Choose label offsets on inner OOF only; outer labels never enter fit."""
    def __init__(self,estimator,validation_policy=None):
        self.estimator,self.validation_policy=estimator,validation_policy

    def _probabilities(self,model,X):
        values=model.predict_proba(X)
        result=np.zeros((len(X),len(self.classes_)))
        for j,label in enumerate(model.classes_):result[:,np.flatnonzero(self.classes_==label)[0]]=values[:,j]
        return result

    def fit(self,X,y):
        target=np.asarray(y)
        if target.ndim!=1:raise ValueError('Macro F1 decisions need one classification target')
        self.classes_=np.unique(target);self.offsets_=np.zeros(len(self.classes_))
        predictions=np.full((len(X),len(self.classes_)),np.nan)
        for train,valid in folds(X,target,self.validation_policy,count=3,classification=True):
            if not len(train) or len(np.unique(target[train]))!=len(self.classes_):continue
            inner=bind_validation_policy(clone(self.estimator),self.validation_policy)
            inner.fit(X.iloc[train],target[train])
            predictions[valid]=self._probabilities(inner,X.iloc[valid])
        available=np.isfinite(predictions).all(axis=1)
        self.report_={'inner_oof_rows':int(available.sum()),'fit_rows':len(X),
            'status':'insufficient_inner_class_support','score_scope':'inner_fit_only; not independent validation'}
        if available.sum()>=2*len(self.classes_) and len(np.unique(target[available]))==len(self.classes_):
            logp=np.log(np.clip(predictions[available],1e-15,1))
            def objective(offset):
                labels=self.classes_[(logp+offset).argmax(axis=1)]
                return float(f1_score(target[available],labels,labels=self.classes_,average='macro',zero_division=0))
            initial=best=objective(self.offsets_)
            for _ in range(2):
                for j in range(len(self.classes_)):
                    current=self.offsets_.copy()
                    for value in (-.5,0.,.5,1.):
                        trial=current.copy();trial[j]=value;score=objective(trial)
                        if score>best+1e-12:best,self.offsets_=score,trial
            self.report_.update(status='fitted',inner_raw_macro_f1=initial,inner_adjusted_macro_f1=best)
        self.estimator_=bind_validation_policy(clone(self.estimator),self.validation_policy)
        self.estimator_.fit(X,y)
        self.report_['offsets']=[float(v) for v in self.offsets_]
        return self

    def predict_proba(self,X):
        # Label decisions do not pretend to calibrate the underlying probabilities.
        return self._probabilities(self.estimator_,X)

    def predict(self,X):
        return self.classes_[(np.log(np.clip(self.predict_proba(X),1e-15,1))+self.offsets_).argmax(axis=1)]
