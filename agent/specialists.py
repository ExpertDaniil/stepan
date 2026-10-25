"""Architecture-owned representation research; model axes belong to the dispatcher."""
from copy import deepcopy
import ast
import hashlib
import json
from sklearn.base import BaseEstimator,TransformerMixin

from agent.contracts import canonical
from agent.errors import AgentError

FACTORY_SOURCE = ('from agent.specialists import make_specialist_pipeline\n\n'
    'def make_pipeline(task_spec, params):\n    return make_specialist_pipeline(task_spec, params)\n')

ROLES = [
    ('native_categories','catboost','Category meaning, missingness mechanisms, object linkage hypotheses and conditional effects.'),
    ('numeric_boosting','lightgbm','Numerical scales, normalization, relations, train-only aggregates and safe target statistics.'),
    ('randomized_trees','extra-trees','Alternative partitions, robust representations and errors complementary to boosting.'),
    ('structured_response','linear','Units, ratios, log scales, smooth basis functions and understandable systematic bias.'),
    ('neural_representation','tabular-net','Learned feature interactions and categorical embeddings; compare small Transformer/MLP by evidence.')]


def enabled(config):
    return bool(getattr(config,'ensemble_research_enabled',False))


def role_for(peer, catalog):
    index=int(peer.rsplit('-',1)[-1])-1
    if not 0<=index<len(ROLES):raise AgentError('INVALID_SPECIALIST','Unknown specialist')
    ident,prefix,question=ROLES[index]
    suffix='regressor-v1' if catalog.kind=='regression' else 'classifier-v1'
    recipe=prefix+'-'+suffix
    if recipe not in catalog.recipes:
        fallback='hist-native-' if prefix=='lightgbm' else 'linear-'
        recipe=fallback+suffix
    return {'role_id':ident,'researcher_id':peer,'recipe_id':recipe,'focus':question,
            'availability':'assigned' if recipe.startswith(prefix+'-') else 'fallback_optional_package_absent'}


def startup(catalog):
    result=[]
    for i in range(len(ROLES)):
        role=role_for('researcher-'+str(i+1),catalog)
        recipe=role['recipe_id'];params=catalog.normalize(recipe,{})
        for k,v in {'iterations':96,'n_estimators':64,'max_iter':64,'depth':6,'epochs':6}.items():
            domain=catalog.recipes[recipe]['parameters'].get(k)
            if domain and 'min' in domain:params[k]=max(domain['min'],min(domain['max'],v))
        result.append((role['researcher_id'],recipe,params))
    return result


def member(plan, source):
    params=deepcopy(plan['model']['hyperparameters'])
    if source.strip()==FACTORY_SOURCE.strip():
        return params
    from agent.catalog import factory_source
    if source.strip()==factory_source(plan['recipe_id']).strip() and not plan['recipe_id'].startswith(('dummy','python')):
        return {'base_recipe_id':plan['recipe_id'],'model_parameters':params}
    from agent.feature_schema import FACTORY_SOURCE as FEATURE_FACTORY
    if source.strip()==FEATURE_FACTORY.strip():return params
    return None


def representation_source(owned):
    """Exact editable feature source; never expose the predictor as feature code."""
    if owned.get('representation_source'):
        return owned['representation_source']
    if owned.get('feature_program'):
        return ('from agent.feature_engineering import FeatureProgram\n\n'
            'def make_features(task_spec):\n    return FeatureProgram('+repr(owned['feature_program'])+')\n')
    return ('from sklearn.preprocessing import FunctionTransformer\n\n'
        'def make_features(task_spec):\n    return FunctionTransformer(validate=False)\n')


def validate_feature_source(source):
    from tools.code import ALLOWED_IMPORTS,BLOCKED_CALLS,BLOCKED_ATTRIBUTES
    try:tree=ast.parse(source)
    except SyntaxError as error:raise AgentError('INVALID_FEATURE_SOURCE',str(error)) from error
    if len(source.encode())>64000:raise AgentError('INVALID_FEATURE_SOURCE','Feature source exceeds 64 KiB')
    functions={n.name for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
    if 'make_features' not in functions or functions & {'make_pipeline','analyze'}:
        raise AgentError('INVALID_FEATURE_SOURCE','Export make_features(task_spec); the executor supplies the assigned estimator')
    for node in ast.walk(tree):
        if isinstance(node,(ast.Import,ast.ImportFrom)):
            if isinstance(node,ast.ImportFrom) and node.level:
                raise AgentError('INVALID_FEATURE_SOURCE','Relative imports are not allowed')
            modules=[a.name for a in node.names] if isinstance(node,ast.Import) else [node.module or '']
            if any(not any(m==a or m.startswith(a+'.') for a in ALLOWED_IMPORTS) for m in modules):
                raise AgentError('INVALID_FEATURE_SOURCE','Import outside approved feature libraries')
        if isinstance(node,ast.Name) and (node.id in BLOCKED_CALLS or node.id in {'SEARCH_SPACE','DEFAULT_PARAMS'}):
            raise AgentError('INVALID_FEATURE_SOURCE','Search and executable model factories belong to the controller')
        if isinstance(node,ast.Attribute) and (node.attr in BLOCKED_ATTRIBUTES or node.attr.startswith(('__','read_')) or
                node.attr in {'to_csv','to_json','to_excel','to_parquet','to_feather','to_sql','to_hdf','to_clipboard'}):
            raise AgentError('INVALID_FEATURE_SOURCE','Unsafe feature access')
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr in {'fit_predict','predict_proba'}:
            raise AgentError('INVALID_FEATURE_SOURCE','Submit a fit/transform representation, not a separate predictor')
    return tree


def normalize_proposal(proposed, context, catalog, artifacts):
    """No extra model-authored forms; attach architecture/provenance in code."""
    role=context['metadata'].get('specialist')
    if not role:return
    if proposed['action'] in {'research','stop'}:return
    if proposed['action']=='tune' or proposed.get('tuning') or 'SEARCH_SPACE' in (proposed.get('code') or ''):
        raise AgentError('OPTIMIZER_OWNED','Optuna runs in your architecture cluster; submit a representation or investigate errors')
    user=json.loads(context['messages'][1]['content'])
    reference_id=role.get('reference_artifact_id')
    reference=(artifacts.read_json(reference_id) if reference_id else
        user.get('profile_summary',{}).get('specialist',{}).get('reference')) or {}
    previous=deepcopy(reference.get('representation',{}))
    # Model parameters are copied from the last measured assigned estimator.
    # Numeric changes in LLM output cannot silently become tuning experiments.
    params={'base_recipe_id':role['recipe_id'],
            'model_parameters':deepcopy(previous.get('model_parameters',catalog.normalize(role['recipe_id'],{})))}
    if previous.get('base_recipe_id')!=role['recipe_id']:params['model_parameters']=catalog.normalize(role['recipe_id'],{})
    for key in ('feature_program','representation_source'):
        if key in previous:params[key]=previous[key]
    if proposed['action']=='catalog' and any(params['model_parameters'].get(k)!=v for k,v in proposed.get('hyperparameters',{}).items()):
        raise AgentError('OPTIMIZER_OWNED','Do not request numerical model changes; the background study owns these axes')
    if (proposed.get('code') or '').strip()==FACTORY_SOURCE.strip():
        normalized=proposed['hyperparameters']
        if normalized.get('base_recipe_id')!=role['recipe_id'] or normalized.get('model_parameters')!=params['model_parameters']:
            raise AgentError('OPTIMIZER_OWNED','Assigned model parameters must match the frozen measured reference')
        if set(normalized)-{'base_recipe_id','model_parameters','feature_program','representation_source'}:
            raise AgentError('INVALID_FEATURE_SOURCE','Unknown specialist representation fields')
        if normalized.get('representation_source'):validate_feature_source(normalized['representation_source'])
        if normalized.get('feature_program'):
            from agent.feature_schema import validate_program
            validate_program(normalized['feature_program'])
        return
    if proposed['action']=='features':
        from agent.feature_schema import validate_program
        params['feature_program']=validate_program(proposed['feature_program'])
        params.pop('representation_source',None)
    elif proposed['action']=='pipeline':
        source=proposed.get('code')
        if not source:raise AgentError('INVALID_FEATURE_SOURCE','Supply make_features source; read the current representation through read_solution')
        validate_feature_source(source)
        params['representation_source']=source;params.pop('feature_program',None)
    proposed.update(action='pipeline',recipe_id='python-pipeline-v1',hyperparameters=params,
        code=FACTORY_SOURCE,tuning=None,feature_steps=[{'step':'specialist_representation','role_id':role['role_id'],
            'verification':'owned_estimator; representation_semantics_unverified'}])
    proposed.pop('feature_program',None)


def attach_context(builder,user,graph,peer):
    role=role_for(peer,builder.catalog)
    candidates=[]
    for eid in graph.get('visible_experiment_ids',[]):
        exp=builder.store.get(eid);ev=builder.store.experiment_evaluation(eid)
        if exp['state']!='evaluated' or not ev or not ev['valid'] or ev['violations']:continue
        source=builder.artifacts.resolve(exp['plan']['code_artifact_id']).read_text(encoding='utf-8')
        representation=member(exp['plan'],source)
        if representation and representation['base_recipe_id']==role['recipe_id']:
            candidates.append((ev['primary_metric']['value']*(1 if ev['primary_metric']['direction']=='maximize' else -1),
                {'experiment_id':eid,'metric':ev['primary_metric'],'representation':representation}))
    reference=max(candidates,key=lambda x:x[0])[1] if candidates else None
    if reference is None:
        _,recipe,params=startup(builder.catalog)[int(peer.rsplit('-',1)[-1])-1]
        reference={'experiment_id':None,'metric':None,'status':'unmeasured_startup_configuration',
            'representation':{'base_recipe_id':recipe,'model_parameters':params}}
    if reference:
        art=builder.artifacts.write_json('contexts/specialist-'+__import__('uuid').uuid4().hex+'.json',reference,'research_context_details')
        role['reference_artifact_id']=art['artifact_id']
    user['profile_summary']['specialist']={**role,'reference':reference,
        'objective':'Improve useful data representations and marginal ensemble contribution; standalone score alone is insufficient.',
        'optimizer':'Independent version studies in this architecture cluster; model axes are executor-owned.',
        'source_detail':'read_solution(experiment_id=reference.experiment_id, view=representation) for editable feature code; view=pipeline for the complete model.'}
    user['profile_summary']['current_pipeline_facts']=builder.catalog.pipeline_facts({
        'recipe_id':role['recipe_id'],'hyperparameters':(reference or {}).get('representation',{}).get('model_parameters',{})})
    user['profile_summary']['feature_engineering']['semantics'] = (
        'Use named inputs with evidence-backed meaning; several coherent changes are allowed. '
        'Learned operations fit inside every enclosing fit. No mandatory investigation form.')
    if reference and reference['representation'].get('representation_source'):
        source=reference['representation']['representation_source']
        if len(source.encode())>4000:
            reference['representation']['representation_source_sha256']=hashlib.sha256(source.encode()).hexdigest()
            reference['representation'].pop('representation_source')
            # The complete reference is frozen separately by prepare_details.
    user['assignment']=role['focus']+' Investigate a concrete error/data mechanism. Do not search estimator parameters or switch architecture. '
    user['assignment']+='Share source-backed meanings, uncertainty and negative findings. Several coherent feature changes are allowed.'
    user['profile_summary']['recipes']=[r for r in user['profile_summary']['recipes'] if r['id'] in {role['recipe_id'],'python-pipeline-v1'}]
    return role


def compact_context(user, log):
    """Project duplicated execution logistics after the exact snapshot is frozen.

    Semantic records, contradictory claims and the inference contract use their
    existing memory projection; this helper never rewrites their epistemic status.
    """
    before=len(canonical(user).encode());profile=user['profile_summary']
    detail=profile['detail_index']['reference_id']
    for i,record in enumerate(profile.get('analysis_history',[])):
        query=record.get('query',{})
        if isinstance(query,dict):
            record['query']={k:v for k,v in query.items() if k in {'kind','columns','group_by','question','artifact_id'}}
        record['exact_read']={'reference_id':detail,'section':'analyses/'+str(i)}
    for recipe in profile.get('recipes',[]):
        for field in ('hyperparameters','parameters','parameter_space','defaults'):
            recipe.pop(field,None)
        recipe['parameters_owner']='executor Optuna; exact catalogue in section=recipes'
    interface=profile.get('feature_engineering',{})
    profile['feature_engineering']={k:v for k,v in interface.items() if k in {
        'action','compiler','operations','geometry','semantics','output'}}
    profile['feature_engineering']['exact_read']={'reference_id':detail,'section':'feature_engineering'}
    profile['optimizer_facts']={'owner':'dispatcher','instruction':'Work on representations; no model parameter forms.'}
    # Repeated complete configurations need not crowd out data semantics. IDs,
    # metrics and the frozen full_context remain available without reconstruction.
    measured=profile.get('measured_search',{})
    if isinstance(measured,dict):
        for record in measured.get('entries',[]):
            for key in ('hyperparameters','effective_parameters','code','feature_steps'):
                record.pop(key,None)
    for i,fact in enumerate(user.get('facts',[])):
        if len(canonical(fact).encode())>1600:
            user['facts'][i]={**{k:v for k,v in fact.items() if k in {'source_id','kind','confidence','experiment_id','metric'}},
                'excerpt':canonical(fact)[:600],'excerpt_only':True,
                'exact_read':{'reference_id':detail,'section':'facts/'+str(i)}}
    log.event('research.specialist_context_projected',before_bytes=before,after_bytes=len(canonical(user).encode()),
        exact_reference_id=detail,semantics_rewritten=False)


def make_specialist_pipeline(task_spec, params):
    from sklearn.pipeline import Pipeline
    from agent.ml import build_pipeline
    from agent.feature_engineering import FeatureProgram
    from agent.feature_schema import validate_program
    from agent.predictions import targets
    options=deepcopy(params)
    base=options.pop('base_recipe_id');model=dict(options.pop('model_parameters'))
    program=options.pop('feature_program',None);source=options.pop('representation_source',None)
    for key in ('_cpu_threads','_seed'):
        if key in options:model[key]=options.pop(key)
    if options or base.startswith(('python','dummy')) or (program and source):
        raise ValueError('Invalid owned specialist representation')
    estimator=build_pipeline(base,task_spec,model)
    if source:
        validate_feature_source(source)
        transformer=SourceFeatures(source,task_spec)
    elif program:
        transformer=FeatureProgram(validate_program(program,excluded=[*targets(task_spec),*task_spec['excluded_features']]))
    else:return estimator
    return Pipeline([('representation',transformer),('model',estimator)])


class SourceFeatures(TransformerMixin,BaseEstimator):
    """Serializable source-owned transformer, including its true fit_transform."""
    def __init__(self,source,task_spec):
        self.source,self.task_spec=source,task_spec

    def _create(self):
        validate_feature_source(self.source)
        namespace={'__name__':'_owned_features_'+hashlib.sha256(self.source.encode()).hexdigest()[:16]}
        exec(compile(self.source,'<specialist-features>','exec'),namespace)
        transformer=namespace['make_features'](self.task_spec)
        if not callable(getattr(transformer,'fit',None)) or not callable(getattr(transformer,'transform',None)):
            raise ValueError('make_features must return a fit/transform estimator')
        return transformer

    def _checked(self,X,result):
        import pandas as pd
        if not isinstance(result,pd.DataFrame) or not result.index.equals(X.index) or not result.columns.is_unique:
            raise ValueError('Feature transformer must return a DataFrame preserving row order/index and unique columns')
        return result

    def fit(self,X,y=None):
        self._fitted=self._create().fit(X,y)
        return self

    def fit_transform(self,X,y=None,**options):
        self._fitted=self._create()
        if callable(getattr(self._fitted,'fit_transform',None)):
            result=self._fitted.fit_transform(X,y,**options)
        else:
            result=self._fitted.fit(X,y).transform(X)
        return self._checked(X,result)

    def transform(self,X):
        return self._checked(X,self._fitted.transform(X))

    def __getstate__(self):
        import cloudpickle
        state=dict(self.__dict__)
        if '_fitted' in state:state['_fitted_bytes']=cloudpickle.dumps(state.pop('_fitted'))
        return state

    def __setstate__(self,state):
        import cloudpickle
        self.__dict__.update(state)
        if '_fitted_bytes' in self.__dict__:self._fitted=cloudpickle.loads(self.__dict__.pop('_fitted_bytes'))
