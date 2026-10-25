"""Version-owned background Optuna. All SQL, admissions and evaluation stay on dispatcher."""
from agent.research_execution import ResearchExecutor, PeerRecovery
from copy import deepcopy
import hashlib
import json
import math
import time
import uuid

from agent.contracts import canonical
from agent.errors import AgentError
from agent.pipeline_candidates import automatic_space, execution_ticket, version_id


class VersionSearch:
    def __init__(self, controller):
        self.c = controller
        self.versions = {}
        self.queue = []
        self.jobs = {}
        self.attempted = 0
        self.publish_due = False
        self.revision = 0
        self.library = None
        self.measurement_revision = 0
        self.recovery = PeerRecovery()

    def quality(self, metric):
        value = metric['value']
        return value if metric['direction'] == 'maximize' else -value

    def save(self):
        self.revision += 1
        rows = [{k: deepcopy(v[k]) for k in ('version_id', 'initial_experiment_id', 'initial_metric',
            'best_experiment_id', 'best_metric', 'trials', 'paused', 'last_scheduled', 'errors','pending','retry_at')} for v in self.versions.values()]
        histories = {}
        for version in self.versions.values():
            histories[version['version_id']] = {
                'code_artifact_id': version['anchor']['code_artifact_id'],
                'search_space': deepcopy(version['space']),
                'research_binding': deepcopy(version.get('research_binding')),
                'batch_limit': version['batch_limit'], 'retry_at': version['retry_at'],
                'blocked_inputs': deepcopy(version['blocked_inputs']),
                'trials': [{'number': t.number, 'state': t.state.name, 'parameters': t.params,
                            'value': t.value, 'metadata': t.user_attrs}
                           for t in version['study'].get_trials()] if version['study'] is not None else []}
        art = self.c.artifacts.write_json(f'pipelines/search-{self.revision:05d}.json',
            {'revision': self.revision, 'versions': rows, 'studies': histories,
             'queued_initial_versions': [v['version_id'] for v in self.queue],
             'running': [{'version_id': v['version_id'], 'experiment_id': job.experiment_id, 'initial': initial}
                         for job, v, _, initial in self.jobs.values()]}, 'pipeline_search')
        clusters={}
        for version in self.versions.values():
            key=self.cluster_key(version)
            cluster=clusters.setdefault(key,{'cluster_id':key,'versions':0,'trials':0,'running':0,'best_metric':None})
            cluster['versions']+=1;cluster['trials']+=version['trials'];cluster['running']+=int(version['pending'])
            if version['best_metric'] and (cluster['best_metric'] is None or self.quality(version['best_metric'])>self.quality(cluster['best_metric'])):
                cluster['best_metric']=version['best_metric']
        self.c.context.pipeline_status = {'artifact_id': art['artifact_id'], 'versions': rows[-8:], 'clusters':list(clusters.values()),
            'queued_versions':[{'version_id':v['version_id'],'cluster_id':self.cluster_key(v),
                'researcher_id':v['anchor']['researcher_id']} for v in self.queue[:8]],
            'omitted_queued_versions':max(0,len(self.queue)-8),
            'instruction': 'Tuning runs independently. Submit improved executable representations; no need to wait for trials.'}

    @staticmethod
    def cluster_key(version):
        params=version['anchor']['model']['hyperparameters']
        return 'ensemble' if params.get('members') else params.get('base_recipe_id',version['anchor']['recipe_id'])

    def add(self, plan, ticket, *, evaluated=None):
        c = self.c
        plan = deepcopy(plan)
        source = c.artifacts.resolve(plan['code_artifact_id']).read_text(encoding='utf-8')
        space = automatic_space(c.catalog, plan, source)
        ident = version_id(plan, c.artifacts.meta(plan['code_artifact_id'])['sha256'], space)
        if ident in self.versions:
            c.log.event('pipeline.version_reused', version_id=ident)
            return self.versions[ident]
        version = {'version_id': ident, 'anchor': plan, 'ticket': ticket, 'space': space,
            'initial_experiment_id': evaluated, 'initial_metric': None, 'best_experiment_id': None,
            'best_metric': None, 'trials': 0, 'paused': None, 'last_scheduled': 0,
            'errors': 0, 'study': None, 'pending': False, 'retry_at': 0,
            'batch_limit': c.config.background_max_trials, 'batch_best': None, 'blocked_inputs': {}}
        if getattr(c, 'error_research', None):
            version['research_binding'] = c.error_research.bind(plan, ticket)
        self.versions[ident] = version
        if evaluated:
            ev = c.store.experiment_evaluation(evaluated)
            if ev and ev['valid'] and not ev['violations']:
                version.update(initial_metric=ev['primary_metric'], best_metric=ev['primary_metric'], best_experiment_id=evaluated)
        else:
            plan['plan_id'] = 'candidate-'+uuid.uuid4().hex[:16]
            self.queue.append(version)
        c.log.event('pipeline.version_registered', version_id=ident, recipe_id=plan['recipe_id'],
                    code_artifact_id=plan['code_artifact_id'], tuning_parameters=list(space), measured=bool(evaluated))
        self.save()
        return version

    def study(self, version):
        if version['study'] is not None:
            return version['study']
        if self.library is None:
            import optuna
            self.library = optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        lib = self.library
        study = lib.create_study(direction=self.c.spec['metric']['direction'],
            sampler=lib.samplers.TPESampler(seed=self.c.config.seed, n_startup_trials=4))
        version['study'] = study
        # Import the independently measured initial configuration; never refit it merely to seed TPE.
        params, distributions = {}, {}
        from agent.tuning_space import contains
        for name, (path, domain) in version['space'].items():
            value = version['anchor']
            for key in path:
                value = value[key]
            if not contains(domain, value):
                return study
            params[name] = value
            if domain['type'] == 'categorical':
                distributions[name] = lib.distributions.CategoricalDistribution(domain['choices'])
            elif domain['type'] == 'int':
                distributions[name] = lib.distributions.IntDistribution(domain['low'], domain['high'],
                    log=domain.get('log', False), step=domain.get('step', 1))
            else:
                distributions[name] = lib.distributions.FloatDistribution(domain['low'], domain['high'],
                    log=domain.get('log', False), step=domain.get('step'))
        if version['initial_metric']:
            study.add_trial(lib.trial.create_trial(params=params, distributions=distributions,
                value=version['initial_metric']['value'], user_attrs={'experiment_id': version['initial_experiment_id']}))
        return study

    def eligible(self):
        rows = []
        for v in self.versions.values():
            if not v['initial_metric'] or not v['space'] or v['paused'] or v['pending'] or v['retry_at'] > time.monotonic():
                continue
            if v['trials'] >= v['batch_limit']:
                admission = self.c.budget.can_train(v['anchor'])
                if not admission['allowed']:continue
                previous = v['batch_best']
                current = self.quality(v['best_metric'])
                v['batch_limit'] += self.c.config.background_max_trials
                v['batch_best'] = current
                self.c.log.event('pipeline.batch_extended', version_id=v['version_id'],
                    trials=v['trials'], next_review=v['batch_limit'], measured_gain=None if previous is None else current-previous,
                    admission=admission, study_preserved=True)
            rows.append(v)
        return rows

    def choose(self):
        candidates = self.eligible()
        if not candidates:
            return None
        leader = max(candidates, key=lambda v: self.quality(v['best_metric']))
        scale = max(1., abs(self.quality(leader['best_metric'])))
        costs = {v['version_id']: max(.1, self.c.budget.duration.estimate(v['anchor'], 'train')) for v in candidates}
        reference_cost = sorted(costs.values())[len(costs)//2]
        def priority(v):
            gap = (self.quality(v['best_metric'])-self.quality(leader['best_metric']))/scale
            uncertainty = 1/math.sqrt(1+v['trials'])
            gain = max(0., self.quality(v['best_metric'])-self.quality(v['initial_metric']))/scale
            age = min(.2, .02*math.log1p(max(0, self.attempted-v['last_scheduled'])))
            cluster_trials=sum(other['trials'] for other in self.versions.values() if self.cluster_key(other)==self.cluster_key(v))
            coverage=.25/math.sqrt(1+cluster_trials) if self.c.config.ensemble_research_enabled else 0.
            return (gap+uncertainty+gain+age+coverage)/math.sqrt(1+costs[v['version_id']]/reference_cost)
        return max(candidates, key=priority)

    def ask(self, version):
        trial = self.study(version).ask()
        plan = deepcopy(version['anchor'])
        plan.update(plan_id='background-trial-'+uuid.uuid4().hex[:16], parent_ids=[version['initial_experiment_id']])
        for name, (path, domain) in version['space'].items():
            if domain['type'] == 'categorical':
                value = trial.suggest_categorical(name, domain['choices'])
            elif domain['type'] == 'int':
                value = trial.suggest_int(name, domain['low'], domain['high'],
                    step=domain.get('step', 1), log=domain.get('log', False))
            else:
                value = trial.suggest_float(name, domain['low'], domain['high'],
                    step=domain.get('step'), log=domain.get('log', False))
            target = plan
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = value
        version['trials'] += 1
        version['last_scheduled'] = self.attempted
        return trial, plan

    def start(self, version, *, initial=False):
        c = self.c
        trial = None
        plan = deepcopy(version['anchor'])
        try:
            snapshot = c.budget.snapshot()
            available_cpu = snapshot['cpu_threads_available']-int(not c.budget.analysis_active and c.budget.cpu_capacity > 1)
            available_memory = snapshot['memory_mb_available']-(c.budget.analysis_memory if not c.budget.analysis_active else 0)
            if available_cpu < 1 or available_memory < 256:
                return False
            total_cpu = c.budget.cpu_capacity
            # Reserve a fair CPU share even when the second candidate arrives
            # later. A running estimator cannot safely surrender its threads.
            cpu = max(1, min(available_cpu, c.config.background_cpu_threads,
                max(1, (total_cpu-int(total_cpu > 1))//max(1, c.config.max_workers))))
            memory = max(256, min(available_memory, c.config.train_memory_mb,
                (c.budget.memory_capacity-c.budget.analysis_memory)//max(1, c.config.max_workers)))
            if not initial:
                trial, plan = self.ask(version)
            input_hash = hashlib.sha256(canonical(plan['model']['hyperparameters']).encode()).hexdigest()
            if version['blocked_inputs'].get(input_hash, 0) > time.monotonic():
                if trial is not None:self.study(version).tell(trial, state=self.library.trial.TrialState.FAIL)
                return False
            plan['resources'] = {'cpu_threads': cpu, 'memory_limit_mb': memory,
                'timeout_seconds': min(c.config.experiment_timeout_seconds, c.config.background_trial_seconds)
                                   if not initial else c.config.experiment_timeout_seconds}
            c.catalog.validate_plan(plan, c.spec, c.validation['validation_id'], c.artifacts)
            if not c.budget.can_train(plan)['allowed']:
                # A costly sampled configuration is not proof that the complete
                # representation has no affordable configurations left.
                version['paused'] = None
                version['retry_at'] = time.monotonic()+1
                if trial is not None:
                    trial.set_user_attr('not_started', 'candidate_and_finalization_do_not_fit')
                    self.study(version).tell(trial, state=self.library.trial.TrialState.PRUNED)
                return False
            ticket = execution_ticket(c, version['ticket'], None if initial else version['initial_experiment_id'])
            from agent.graph.controller import bind_trusted, lineage_for
            # The execution order is authoritative for both typed lineage and
            # C12 parents. A copied warmup/anchor can name another peer's leader.
            plan = bind_trusted(plan, ticket)
            reply = c.graph.invoke('G04', {'work_order_id': ticket['work_order']['work_order_id'],
                'proposal': {'proposal_id': 'execution-'+uuid.uuid4().hex[:16], 'researcher_id': plan['researcher_id'],
                    'decision': 'propose', 'plan': plan, 'stop_reason': None},
                'manifest': c.graph.constructor.build(plan), 'lineage': lineage_for(ticket['work_order'])},
                key='execution/'+ticket['work_order']['work_order_id'])
            eid = reply['lead_submission']['experiment_id']
            if trial is not None:
                trial.set_user_attr('experiment_id', eid)
            exp = c.store.get(eid)
            if exp['state'] == 'evaluated':
                ev = c.store.experiment_evaluation(eid)
                c.graph.invoke('G05', {'experiment_id': eid, 'evaluation_id': ev['evaluation_id'],
                    'selection_policy_id': 'same-protocol-primary-v1'}, key='cached/'+ticket['work_order']['work_order_id'])
                self.measured(version, eid, ev['primary_metric'], trial, initial)
                return True
            if exp['state'] != 'queued':
                if trial is not None:
                    self.study(version).tell(trial, state=self.library.trial.TrialState.FAIL)
                version['errors'] += 1
                return False
            if not c.admit_training(plan, eid):
                from agent.graph.controller import cancel_queued
                cancel_queued(c, eid, 'Training reserve no longer fits')
                if trial is not None:
                    self.study(version).tell(trial, state=self.library.trial.TrialState.FAIL)
                version['paused'] = 'initial_unaffordable' if initial else None
                version['retry_at'] = time.monotonic()+1
                return False
            allocation = c.lease(eid, 'train', plan['resources'])
            if not allocation['granted']:
                from agent.graph.controller import cancel_queued
                cancel_queued(c, eid, 'Retry candidate after an execution slot is free')
                if trial is not None:
                    self.study(version).tell(trial, state=self.library.trial.TrialState.FAIL)
                return False
            lease = allocation['lease']
            try:
                attempt = c.store.start(eid, lease)
                value = c.call('C14', {'experiment_id': eid, 'attempt_no': attempt, 'plan': plan, 'lease': lease,
                    'train_rows_artifact_id': c.validation['train_rows_artifact_id'],
                    'valid_rows_artifact_id': c.validation['valid_rows_artifact_id'],
                    'output_kinds': ['validation_predictions', 'fitted_pipeline', 'execution_log']},
                    lambda p: c.runner.start_train(p, c.spec, c.store.get(eid)), key=eid+'/'+str(attempt))
            except Exception as error:
                c.store.fail_start(eid, error)
                c.call('C13', {'operation': 'release', 'owner_id': eid, 'phase': 'train',
                    'requested': plan['resources'], 'lease_id': lease['lease_id'],
                    'fencing_token': lease['fencing_token']}, c.budget.lease)
                raise
            job = c.runner.jobs[value['job_id']]
            c.graph.track_job(job)
            version['pending'] = True
            self.jobs[job.job_id] = (job, version, trial, initial)
            self.attempted += 1
            c.log.event('pipeline.execution_started', version_id=version['version_id'],
                        experiment_id=eid, initial=initial, cpu_threads=cpu, memory_mb=memory)
            self.save()
            c.publish_research_updates()
            return True
        except Exception as error:
            version['errors'] += 1
            _, failed = self.recovery.quarantine(plan['code_artifact_id'], plan['model']['hyperparameters'],
                {'code': getattr(error, 'code', type(error).__name__), 'message': str(error)[:600]})
            version['retry_at'] = failed['retry_at']
            version['blocked_inputs'][hashlib.sha256(canonical(plan['model']['hyperparameters']).encode()).hexdigest()] = failed['retry_at']
            if trial is not None and trial.number in {t.number for t in self.study(version).get_trials() if not t.state.is_finished()}:
                self.study(version).tell(trial, state=self.library.trial.TrialState.FAIL)
            c.record_research_feedback(plan['researcher_id'], version['ticket']['work_order']['branch_id'],
                'pipeline_rejected', str(error), code=getattr(error, 'code', 'PIPELINE_ERROR'))
            c.log.exception('pipeline.execution_failed', error, version_id=version['version_id'])
            return False

    def measured(self, version, eid, metric, trial, initial):
        self.measurement_revision += 1
        if trial is not None:
            self.study(version).tell(trial, metric['value'])
        old_best = max((self.quality(v['best_metric']) for v in self.versions.values()
                        if v is not version and v['best_metric']), default=-math.inf)
        if version['best_metric'] is None or self.quality(metric) > self.quality(version['best_metric']):
            version.update(best_metric=metric, best_experiment_id=eid)
        if initial:
            version.update(initial_experiment_id=eid, initial_metric=metric)
            if self.quality(metric) > old_best+self.c.config.background_switch_margin:
                for other in self.versions.values():
                    if other is not version and other['best_metric'] and self.quality(other['best_metric']) < self.quality(metric):
                        other['priority_reduced_by'] = version['version_id']
                self.c.log.event('pipeline.leader_replaced', version_id=version['version_id'],
                    experiment_id=eid, metric=metric, reason='New initial score exceeds previous tuned best')
        self.c.update_final_reserve()
        version['errors'] = 0
        if getattr(self.c, 'error_research', None):
            self.c.error_research.compare(eid, version.get('research_binding', {}))
        if getattr(self.c, 'science', None):
            self.c.science.diagnostics.ensure(self.c.graph.visible_history(version['ticket']['work_order']['branch_id']))
        champion = self.c.store.champion()
        best = self.c.store.best()
        self.publish_due |= bool(best and (not champion or best != champion['experiment_id']))
        self.save()

    def poll(self):
        c = self.c
        changed = False
        for key, (job, version, trial, initial) in list(self.jobs.items()):
            if job.process.poll() is None and not job.event:
                c._heartbeat_and_limits(job)
            if job.process.poll() is None and not job.event:
                continue
            event = c.wait_jobs([job])[job.job_id]
            changed = True
            del self.jobs[key]
            version['pending'] = False
            if event['state'] == 'succeeded':
                try:
                    prediction = next(a for a in event['output_artifact_ids']
                                      if c.artifacts.meta(a)['kind'] == 'validation_predictions')
                    ev = c.call('C16', {'experiment_id': job.experiment_id, 'task_spec_id': c.spec['task_spec_id'],
                        'dataset_id': c.spec['dataset_id'], 'validation_id': c.validation['validation_id'],
                        'predictions_artifact_id': prediction, 'fidelity': 'full'}, c.evaluator.evaluate)['evaluation']
                    c.graph.invoke('G05', {'experiment_id': job.experiment_id, 'evaluation_id': ev['evaluation_id'],
                        'selection_policy_id': 'same-protocol-primary-v1'}, key='eval/'+ev['evaluation_id'])
                    self.measured(version, job.experiment_id, ev['primary_metric'], trial, initial)
                    continue
                except Exception as error:
                    failure = error if isinstance(error, AgentError) else AgentError('PIPELINE_EVALUATION_FAILED', str(error))
                    c.store.invalidate(job.experiment_id, failure)
                    c.log.exception('pipeline.evaluation_failed', error, experiment_id=job.experiment_id)
            if trial is not None:
                self.study(version).tell(trial, state=self.library.trial.TrialState.FAIL)
            if event['state'] != 'cancelled':
                version['errors'] += 1
                signature, failure = self.recovery.quarantine(version['anchor']['code_artifact_id'],
                    c.store.get(job.experiment_id)['plan']['model']['hyperparameters'], event.get('error'))
                version['retry_at'] = failure['retry_at']
                version['blocked_inputs'][hashlib.sha256(canonical(c.store.get(job.experiment_id)['plan']['model']['hyperparameters']).encode()).hexdigest()] = failure['retry_at']
                c.record_research_feedback(version['anchor']['researcher_id'], version['ticket']['work_order']['branch_id'],
                    'pipeline_execution_failed', canonical(event.get('error')), code='PIPELINE_EXECUTION_FAILED')
            if initial:
                version['paused'] = None if event['state'] in {'timed_out','cancelled'} or (event.get('error') or {}).get('code') in {'JOB_MEMORY_LIMIT','JOB_TIMEOUT'} else 'deterministic_initial_failure'
                if version['paused'] is None:self.queue.append(version)
            self.save()
        if changed:
            c.publish_research_updates()

    def fill(self):
        if not self.c.config.ensemble_research_enabled and self.queue and self.jobs and (len(self.jobs) >= self.c.config.max_workers
                or self.c.budget.snapshot()['cpu_threads_available'] <= int(not self.c.budget.analysis_active)):
            trials = [(job, v) for job, v, _, initial in self.jobs.values() if not initial and not job.event]
            if trials:
                job, version = min(trials, key=lambda pair: self.quality(pair[1]['best_metric']))
                self.c.cancel_job(job, 'Priority first evaluation of a new executable version')
                self.c.log.event('pipeline.execution_preempted', version_id=version['version_id'],
                                 experiment_id=job.experiment_id, reason='new_initial_version')
                return
        if self.publish_due:
            if self.jobs or (getattr(self.c, 'science', None) and self.c.science.diagnostics.running):
                return
            self.c.finalize_best()
            self.publish_due = False
        dispatched = 0
        while len(self.jobs) < self.c.config.max_workers and self.c.budget.snapshot()['search_allowed'] and dispatched < self.c.config.max_workers*2:
            dispatched += 1
            initial = bool(self.queue)
            if initial and self.queue[0]['retry_at'] > time.monotonic():break
            version = self.queue.pop(0) if initial else self.choose()
            if version is None:
                break
            started = self.start(version, initial=initial)
            if not started:
                if initial and not version['paused']:
                    self.queue.append(version)
                    continue
                break


def search(controller, root, accepted):
    """Research futures and version-owned train processes share one dispatcher loop."""
    from agent.graph.controller import issue, stop_ticket, publish_activity, finish
    c = controller
    service = VersionSearch(c)
    c.pipeline_search = service
    coordinator=None
    if c.config.ensemble_research_enabled:
        from agent.ensemble_coordinator import EnsembleCoordinator
        coordinator=EnsembleCoordinator(c)
    c.budget.train_worker_limit = c.config.max_workers
    # Seed already evaluated startup versions, without new fits or LLM authorisations.
    for eid in c.graph.visible_history(root):
        exp = c.store.get(eid)
        if exp['state'] != 'evaluated' or exp['plan']['model']['family'] == 'dummy':
            continue
        row = c.store.db.execute('SELECT w.work_order_id FROM g_work_orders w JOIN g_bindings b USING(run_id,work_order_id) '
            'WHERE b.run_id=? AND b.experiment_id=? ORDER BY w.rowid LIMIT 1', (c.store.run_id, eid)).fetchone()
        if row:
            service.add(exp['plan'], {'work_order': c.graph.order(row[0])}, evaluated=eid)
    peers = c.graph.metadata('researcher_ids')
    pending = {}
    recovery = PeerRecovery()
    pool = ResearchExecutor(max_workers=len(peers), thread_name_prefix='researcher')
    reason = 'time_reserve'
    try:
        while c.budget.snapshot()['search_allowed']:
            service.poll()
            if c.science:
                c.science.diagnostics.ensure(c.graph.visible_history(root))
                c.science.diagnostics.poll()
            generation = (service.measurement_revision, len(c.science.diagnostics.records) if c.science else 0)
            for future, ticket in list(pending.items()):
                if not future.done():
                    continue
                del pending[future]
                publish_activity(c, ticket, started=False)
                work = ticket['work_order'];peer = work['researcher_id']
                try:
                    proposal = future.result()
                    if proposal['decision'] == 'propose':
                        service.add(proposal['plan'], ticket)
                        stop_ticket(c, ticket, 'Executable version queued; researcher continues independently')
                        recovery.peers.pop(peer, None)
                    else:
                        stop_ticket(c, ticket, proposal.get('stop_reason') or 'Research checkpoint')
                        c.error_research.checkpoint(peer, proposal.get('stop_reason') or '', branch=root)
                        signature = hashlib.sha256(canonical([proposal.get('stop_reason'),
                            sorted({r['artifact_id'] for r in c.context.analysis.records})]).encode()).hexdigest()
                        kind = ('explicit_stop' if proposal['decision'] == 'stop' else
                            'checkpoint' if proposal.get('checkpoint_kind') == 'wait' else 'continuation')
                        old = recovery.peers.get(peer, {})
                        if kind == 'continuation' and old.get('signature') == signature and old.get('repetitions', 0) >= 2:
                            kind = 'checkpoint'
                        state = recovery.defer(peer, kind, generation, signature, proposal.get('stop_reason') or '')
                        c.log.event('research.peer_recovery', **state)
                except Exception as error:
                    c.graph.abandon(work['work_order_id'], str(error))
                    c.record_research_feedback(peer, root, 'proposal_failed', str(error), code=getattr(error, 'code', 'PIPELINE_ERROR'))
                    c.log.exception('research.proposal_failed', error, work_order_id=work['work_order_id'])
                    state = recovery.failure(peer, error, generation)
                    c.error_research.checkpoint(peer, '', error, branch=root)
                    c.log.event('research.peer_recovery', **state)
            if coordinator:
                try:coordinator.review(service,root)
                except (AgentError,ValueError,OSError,KeyError) as error:
                    c.log.exception('ensemble.review_failed',error,retained_output=bool(c.store.champion()))
            for peer in peers:
                if any(t['work_order']['researcher_id'] == peer for t in pending.values()):
                    continue
                if not recovery.ready(peer, generation) or not c.llm_available(len(pending)):
                    continue
                try:
                    c.context.round_no += 1
                    ticket = issue(c, root, peer)
                    ticket['error_assignment'] = deepcopy(c.error_research.assignments.get(root+'/'+peer, {}))
                    publish_activity(c, ticket, started=True)
                    future = pool.submit(c.call, 'C09', ticket['researcher_request']['payload'], c.researcher.propose)
                    pending[future] = ticket
                    c.log.event('research.dialogue_dispatched', work_order_id=ticket['work_order']['work_order_id'],
                        researcher_id=peer, branch_id=root)
                except AgentError as error:
                    recovery.failure(peer, error, generation)
                    c.record_research_feedback(peer, root, 'context_unavailable', str(error), code=error.code)
            service.fill()
            diagnostic_pending = bool(c.science and (c.science.diagnostics.running or c.science.diagnostics.pending))
            renewable = any(state['kind'] != 'explicit_stop' for state in recovery.peers.values())
            provider_reason = c.llm_unavailable_reason(len(pending))
            if provider_reason in {'llm_call_limit', 'llm_output_budget', 'llm_provider_credits_exhausted', 'llm_budget_exhausted'}:
                renewable = False  # Internal quota is a pause; prepared work still executes.
                retry_at = c.researcher.gateway.daily_retry_at
                if provider_reason == 'llm_budget_exhausted' and retry_at is not None:
                    renewable = retry_at < c.budget.deadline-c.budget.reserve_seconds()-1
            cooling_versions = any(v.get('retry_at', 0) > time.monotonic() and not v['paused'] for v in service.versions.values())
            if not pending and not service.jobs and not service.queue and not service.eligible() and not diagnostic_pending and not renewable and not cooling_versions:
                reason = 'prepared_work_exhausted_llm_paused' if provider_reason else 'research_and_background_exhausted'
                break
            time.sleep(.1)
    finally:
        c.researcher.gateway.stop_research()
        for future, ticket in pending.items():
            future.cancel()
            c.graph.abandon(ticket['work_order']['work_order_id'], 'Research deadline reached')
        pool.shutdown(wait=False, cancel_futures=True)
        if c.science:c.science.diagnostics.shutdown()
        c.error_research.save()
        for job, _, _, _ in list(service.jobs.values()):
            c.cancel_job(job, 'Publication reserve reached')
        if service.jobs:
            service.poll()
        service.save()
    return finish(c, reason, accepted+service.attempted)
