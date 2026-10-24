"""Dispatcher-owned ask/tell optimization; Runner/Evaluator remain authoritative."""
from copy import deepcopy
import hashlib
import time
import uuid

from agent.contracts import canonical
from agent.errors import AgentError
from agent.tuning_space import contains, normalize_space


class OptunaSearch:
    def __init__(self, controller):
        self.c = controller
        self.active, self.sessions, self.studies, self.plans = {}, {}, {}, {}
        self.library = None
        self.disabled_error = None

    def _library(self):
        if self.library is None:
            try:
                import optuna
            except ImportError as error:
                raise AgentError('OPTUNA_UNAVAILABLE', 'Optuna 4.2.1 and its dependencies must be installed before the run; use requirements.runtime.txt. No runtime installation is attempted.') from error
            self.library = optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            self.c.log.event('optuna.loaded', version=optuna.__version__, expected_version='4.2.1')
        return self.library

    def _visible_ids(self, branch, anchor):
        graph = getattr(self.c, 'graph', None)
        if graph:
            return list(dict.fromkeys([*graph.visible_history(branch), *anchor['parent_ids']]))
        return [r[0] for r in self.c.store.db.execute('SELECT experiment_id FROM experiments WHERE run_id=?', (self.c.store.run_id,))]

    def _core(self, plan):
        return self.c.store.fingerprint_core(plan, self.c.artifacts.meta(plan['code_artifact_id'])['sha256'])

    def _scope(self, plan, space, branch):
        core = deepcopy(self._core(plan))
        core['model']['hyperparameters'] = {k: v for k, v in core['model']['hyperparameters'].items() if k not in space}
        if plan['model']['family'] != 'custom':
            core['feature_steps'] = 'derived_by_owned_constructor_from_parameters'
        return {'run_id': self.c.store.run_id, 'branch_id': branch, 'pipeline': core,
                'metric': deepcopy(self.c.spec['metric']), 'search_space': deepcopy(space)}

    def _compatible(self, session, exp):
        plan = exp['plan']
        return self._scope(plan, session['space'], session['branch']) == session['scope']

    @staticmethod
    def _parameter_key(session, values):
        return canonical({name: float(value) if session['space'][name]['type'] == 'float' else value
                          for name, value in values.items()})

    def _warm(self, session):
        record = session['study_record']
        for eid in self._visible_ids(session['branch'], session['anchor']):
            if eid in record['imported'] or eid in record['bound_eids']:
                continue
            exp = self.c.store.get(eid)
            if exp['state'] != 'evaluated' or not self._compatible(session, exp):
                continue
            ev = self.c.store.experiment_evaluation(eid)
            if not self._trusted(session, ev):
                continue
            params = exp['plan']['model']['hyperparameters']
            values = {name: params[name] for name in session['space'] if name in params}
            if set(values) != set(session['space']) or not all(contains(session['space'][k], v) for k, v in values.items()):
                continue
            # Arbitrary custom source can branch on Python numeric types.
            if exp['plan']['model']['family'] == 'custom' and any(d['type'] == 'float' and not isinstance(values[n], float) for n, d in session['space'].items()):
                continue
            key = self._parameter_key(session, values)
            if key not in record['known']:
                trial = self._library().trial.create_trial(params=values, distributions=record['distributions'],
                    value=ev['primary_metric']['value'], user_attrs={'experiment_id': eid, 'origin': 'verified_history'})
                record['study'].add_trial(trial)
                record['known'][key] = {'experiment_id': eid, 'primary_metric': deepcopy(ev['primary_metric'])}
            record['imported'].add(eid)

    def _trusted(self, session, ev):
        if not ev or not ev['valid'] or ev['violations']:
            return False
        metric = ev['primary_metric']
        anchor = session['anchor']
        return (metric['dataset_id'] == anchor['dataset_id'] and metric['validation_id'] == anchor['validation_id']
                and metric['fidelity'] == anchor['fidelity'] and metric['metric_id'] == self.c.spec['metric']['metric_id']
                and metric['direction'] == self.c.spec['metric']['direction'])

    def start_series(self, peer, branch, plan, request):
        self.c.store._writer()
        from agent.data_research import require_tuning
        require_tuning(self.c.budget)
        if plan['researcher_id'] != peer:
            raise AgentError('INVALID_TUNING_SPACE', 'Tuning authorization must keep the proposal owner')
        if self.disabled_error is not None:
            raise AgentError('OPTUNA_UNAVAILABLE', 'Optimizer was disabled after an error: '+self.disabled_error)
        self.stop(peer, branch, 'replaced_by_new_authorization')
        if request['budget_seconds'] <= 0:
            return
        constraints = (self.c.graph.branch(branch)['constraints'] if getattr(self.c, 'graph', None) else None)
        space = normalize_space(self.c.catalog, plan, request, constraints)
        library = self._library()
        scope = self._scope(deepcopy(plan), space, branch)
        study_id = 'study-'+hashlib.sha256(canonical(scope).encode()).hexdigest()[:24]
        record = self.studies.get(study_id)
        if record is None:
            distributions = {}
            for name, domain in space.items():
                if domain['type'] == 'categorical':
                    dist = library.distributions.CategoricalDistribution(domain['choices'])
                elif domain['type'] == 'int':
                    dist = library.distributions.IntDistribution(domain['low'], domain['high'], log=domain['log'], step=domain.get('step', 1))
                else:
                    dist = library.distributions.FloatDistribution(domain['low'], domain['high'], log=domain['log'], step=domain.get('step'))
                distributions[name] = dist
            record = {'study': library.create_study(direction=self.c.spec['metric']['direction'],
                sampler=library.samplers.TPESampler(seed=self.c.config.seed)),
                'distributions': distributions, 'known': {}, 'imported': set(), 'bound_eids': set(), 'claimed': set()}
            self.studies[study_id] = record
        ident = 'series-'+uuid.uuid4().hex[:16]
        session = {'series_id': ident, 'study_id': study_id, 'study_record': record, 'source_plan_id': plan['plan_id'],
            'anchor': deepcopy(plan), 'scope': scope, 'space': space, 'branch': branch, 'researcher_id': peer,
            'fixed_parameters': {k: v for k, v in plan['model']['hyperparameters'].items() if k not in space},
            'deadline': time.monotonic()+request['budget_seconds'], 'started': time.monotonic(),
            'max_trials': request['max_trials'], 'budget_seconds': request['budget_seconds'],
            'trials': [], 'status': 'running', 'stop_reason': None, 'revision': 0}
        self.sessions[ident] = session
        self.active[(branch, peer)] = session
        self._warm(session)
        session['initial_best'] = self._best(list(record['known'].values()))
        session['warm_start_trials'] = len(record['known'])
        self._snapshot(session)
        self.c.log.event('optuna.series_started', series_id=ident, study_id=study_id, source_plan_id=plan['plan_id'],
            branch_id=branch, researcher_id=peer, search_space=space, max_trials=request['max_trials'],
            budget_seconds=request['budget_seconds'], warm_start_trials=len(record['known']),
            space_origin='explicit' if request.get('search_space') is not None else 'legacy_catalog_domains')

    def stop(self, peer, branch, reason='research_action_changed'):
        session = self.active.pop((branch, peer), None)
        if session:
            session['status'], session['stop_reason'] = ('completed' if reason == 'trial_limit' else 'stopped'), reason
            for item in session['trials']:
                if not item['finished'] and item['experiment_id'] is None:
                    self._tell(session, item, error='Unsubmitted trial: '+reason)
            self._snapshot(session)
            self.c.log.event('optuna.series_finished', series_id=session['series_id'], reason=reason,
                             proposed_trials=len(session['trials']))

    def clear_all(self, reason):
        for branch, peer in list(self.active):
            self.stop(peer, branch, reason)

    def has_series(self, peer, branch):
        self.sync()
        if self.c.budget.data_research_snapshot()['active']:
            return False
        session = self.active.get((branch, peer))
        if session is None:
            return False
        reason = ('time_limit' if time.monotonic() >= session['deadline'] else
                  'trial_limit' if len(session['trials']) >= session['max_trials'] else None)
        finite_size = self._finite_size(session['space'])
        record = session['study_record']
        if reason is None and finite_size is not None and len(record['known'])+len(record['claimed']) >= finite_size:
            reason = 'declared_space_already_evaluated_or_pending'
        if reason:
            self.stop(peer, branch, reason)
            return False
        return True

    @staticmethod
    def _finite_size(space):
        size = 1
        for domain in space.values():
            if domain['type'] == 'categorical':
                count = len(domain['choices'])
            elif domain['low'] == domain['high']:
                count = 1
            elif domain['type'] == 'int':
                count = (domain['high']-domain['low'])//domain.get('step', 1)+1
            elif domain.get('step') is not None:
                count = round((domain['high']-domain['low'])/domain['step'])+1
            else:
                return None
            size *= count
        return size

    def next_plan(self, peer, branch, constraints=None):
        self.c.store._writer()
        if not self.has_series(peer, branch):
            return None
        session = self.active[(branch, peer)]
        self._warm(session)
        record = session['study_record']
        finite_size = self._finite_size(session['space'])
        if finite_size is not None and len(record['known'])+len(record['claimed']) >= finite_size:
            self.stop(peer, branch, 'declared_space_already_evaluated_or_pending')
            return None
        while len(session['trials']) < session['max_trials'] and time.monotonic() < session['deadline']:
            trial = record['study'].ask(fixed_distributions=record['distributions'])
            trial.set_user_attr('series_id', session['series_id'])
            trial.set_user_attr('source_plan_id', session['source_plan_id'])
            item = {'number': trial.number, 'trial': trial, 'parameters': dict(trial.params), 'plan_id': None,
                    'experiment_id': None, 'work_order_id': None, 'primary_metric': None, 'cached': False,
                    'experiment_state': None, 'duration_seconds': None, 'experiment_error': None,
                    'error': None, 'finished': False, 'fingerprint': None, 'claimed': False}
            session['trials'].append(item)
            known = record['known'].get(self._parameter_key(session, trial.params))
            if known:
                item.update(experiment_id=known['experiment_id'], cached=True)
                self._tell(session, item, known['primary_metric'])
                continue
            plan = deepcopy(session['anchor'])
            plan['plan_id'] = 'plan-'+uuid.uuid4().hex[:12]
            plan['model']['hyperparameters'] = self.c.catalog.normalize(plan['recipe_id'], {**session['fixed_parameters'], **trial.params})
            if constraints and self.c.catalog.constrain_params(plan['recipe_id'], plan['model']['hyperparameters'], constraints) != plan['model']['hyperparameters']:
                self._tell(session, item, error='Current branch constraints reject the sampled configuration')
                continue
            if plan['model']['family'] != 'custom':
                plan['feature_steps'] = self.c.catalog.feature_steps(plan['model']['hyperparameters'])
            left = int(min(session['deadline']-time.monotonic(), self.c.budget.phase_remaining(), self.c.budget.remaining()-self.c.budget.reserve_seconds()))
            if left < 1:
                self._tell(session, item, error='No search time remains for this trial')
                break
            plan['resources']['timeout_seconds'] = min(plan['resources']['timeout_seconds'], left)
            plan['hypothesis'] = 'Optuna trial for '+session['source_plan_id']+': '+session['anchor']['hypothesis']
            plan['expected_effect'] = 'Measure the entire sampled pipeline; improvement is not assumed.'
            fingerprint = canonical(self._core(plan))
            item['fingerprint'] = fingerprint
            if fingerprint in record['claimed']:
                self._tell(session, item, error='Identical configuration is already pending in this study')
                continue
            if not self.c.budget.can_train(plan)['allowed']:
                self._tell(session, item, error='Sampled configuration is not affordable under current training/finalization reserves')
                continue
            item['plan_id'] = plan['plan_id']
            self.plans[plan['plan_id']] = (session, item)
            record['claimed'].add(fingerprint)
            item['claimed'] = True
            self._snapshot(session)
            self.c.log.event('optuna.trial_proposed', series_id=session['series_id'], study_id=session['study_id'],
                trial_number=trial.number, plan_id=plan['plan_id'], hyperparameters=plan['model']['hyperparameters'])
            return plan
        self.stop(peer, branch, 'trial_limit' if len(session['trials']) >= session['max_trials'] else 'time_limit')
        return None

    def bind(self, plan, experiment_id, work_order_id=None):
        self.c.store._writer()
        pair = self.plans.get(plan['plan_id'])
        if pair:
            session, item = pair
            item.update(experiment_id=experiment_id, work_order_id=work_order_id)
            session['study_record']['bound_eids'].add(experiment_id)
            if not item['finished']:
                item['trial'].set_user_attr('experiment_id', experiment_id)
            if canonical(self._core(self.c.store.get(experiment_id)['plan'])) != item['fingerprint']:
                self._tell(session, item, error='Reserved configuration differs from the Optuna proposal')
            self._snapshot(session)

    def abandon(self, plan, reason):
        pair = self.plans.get(plan['plan_id'])
        if pair and not pair[1]['finished']:
            self._tell(*pair, error=reason)
            self._snapshot(pair[0])

    def _tell(self, session, item, metric=None, error=None):
        if item['finished']:
            return
        if item['experiment_id']:
            exp = self.c.store.get(item['experiment_id'])
            item.update(experiment_state=exp['state'], duration_seconds=exp['duration'],
                        experiment_error=self.c.log.redactor.clean(exp['error'])[:1200] if exp['error'] else None)
        record = session['study_record']
        if metric is not None:
            record['study'].tell(item['trial'], metric['value'])
            item['primary_metric'] = deepcopy(metric)
            record['known'][self._parameter_key(session, item['parameters'])] = {'experiment_id': item['experiment_id'], 'primary_metric': deepcopy(metric)}
            if item['experiment_id']:
                record['imported'].add(item['experiment_id'])
        else:
            record['study'].tell(item['trial'], state=self._library().trial.TrialState.FAIL)
            item['error'] = str(error)
        if item['claimed']:
            record['claimed'].discard(item['fingerprint'])
            item['claimed'] = False
        item['finished'] = True
        self.c.log.event('optuna.trial_finished', series_id=session['series_id'], trial_number=item['number'],
            experiment_id=item['experiment_id'], cached=item['cached'], primary_metric=item['primary_metric'], error=item['error'])

    def sync(self):
        self.c.store._writer()
        if self.disabled_error is not None:
            return
        try:
            self._sync()
        except Exception as error:
            # A valid ML evaluation must survive an optimizer bookkeeping error.
            # Disable further asks; never synthesize a success/failure score.
            self.disabled_error = str(error)
            self.c.log.exception('optuna.sync_failed', error)
            for session in list(self.active.values()):
                session['status'], session['stop_reason'] = 'failed', 'optimizer_error: '+str(error)
                try:
                    self._snapshot(session)
                except Exception as snapshot_error:
                    self.c.log.exception('optuna.snapshot_failed', snapshot_error, series_id=session['series_id'])
            self.active.clear()

    def _sync(self):
        for session in self.sessions.values():
            changed = False
            for item in session['trials']:
                if item['finished'] or not item['experiment_id']:
                    continue
                exp = self.c.store.get(item['experiment_id'])
                if exp['state'] == 'evaluated':
                    ev = self.c.store.experiment_evaluation(item['experiment_id'])
                    self._tell(session, item, ev['primary_metric'] if self._trusted(session, ev) else None,
                               error='Evaluation is not a committed valid observation of this objective')
                    changed = True
                elif exp['state'] in {'failed', 'invalid', 'timed_out', 'cancelled'}:
                    self._tell(session, item, error='Experiment ended as '+exp['state'])
                    changed = True
            if changed:
                if session['status'] == 'running' and all(i['finished'] for i in session['trials']) and (
                        len(session['trials']) >= session['max_trials'] or time.monotonic() >= session['deadline']):
                    self.stop(session['researcher_id'], session['branch'],
                              'trial_limit' if len(session['trials']) >= session['max_trials'] else 'time_limit')
                else:
                    self._snapshot(session)

    def _snapshot(self, session):
        session['revision'] += 1
        rows = [{k: deepcopy(v) for k, v in item.items() if k not in {'trial', 'fingerprint', 'claimed'}} for item in session['trials']]
        observations = [r for r in rows if r['primary_metric'] is not None]
        best = self._best([*session['study_record']['known'].values(), *observations])
        graph = getattr(self.c, 'graph', None)
        payload = {'series_id': session['series_id'], 'study_id': session['study_id'], 'revision': session['revision'],
            'source_plan_id': session['source_plan_id'], 'branch_id': session['branch'], 'researcher_id': session['researcher_id'],
            'graph_event_seq': graph.cut() if graph else None, 'status': session['status'], 'stop_reason': session['stop_reason'],
            'optimizer': 'optuna', 'optuna_version': self._library().__version__, 'sampler': 'TPESampler', 'sampler_seed': self.c.config.seed,
            'search_space': session['space'], 'scope': session['scope'], 'max_trials': session['max_trials'],
            'budget_seconds': session['budget_seconds'], 'elapsed_seconds': round(time.monotonic()-session['started'], 3),
            'proposed_trials': len(rows), 'completed_trials': len(observations),
            'failed_trials': sum(r['finished'] and r['error'] is not None for r in rows),
            'cached_trials': sum(r['cached'] for r in rows), 'warm_start_trials': session['warm_start_trials'],
            'best_before': session.get('initial_best'),
            'best_experiment_id': best['experiment_id'] if best else None,
            'best_metric': best['primary_metric'] if best else None, 'trials': rows,
            'study_history': [{'number': t.number, 'parameters': t.params, 'value': t.value, 'state': t.state.name,
                               'user_attrs': t.user_attrs} for t in session['study_record']['study'].trials]}
        art = self.c.artifacts.write_json('optuna/'+session['series_id']+'-'+str(session['revision'])+'.json', payload, 'optuna_series')
        self.c.log.event('optuna.series_snapshot', series_id=session['series_id'], artifact_id=art['artifact_id'],
                         proposed_trials=len(rows), completed_trials=len(observations), status=session['status'])

    def _best(self, observations):
        direction = self.c.spec['metric']['direction']
        return min(observations, key=lambda r: r['primary_metric']['value']*(1 if direction == 'minimize' else -1)) if observations else None
