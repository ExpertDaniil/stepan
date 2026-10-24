"""Version-owned background Optuna. All SQL, admissions and evaluation stay on dispatcher."""
from concurrent.futures import ThreadPoolExecutor
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

    def quality(self, metric):
        value = metric['value']
        return value if metric['direction'] == 'maximize' else -value

    def save(self):
        self.revision += 1
        rows = [{k: deepcopy(v[k]) for k in ('version_id', 'initial_experiment_id', 'initial_metric',
            'best_experiment_id', 'best_metric', 'trials', 'paused', 'last_scheduled', 'errors')} for v in self.versions.values()]
        histories = {}
        for version in self.versions.values():
            histories[version['version_id']] = {
                'code_artifact_id': version['anchor']['code_artifact_id'],
                'search_space': deepcopy(version['space']),
                'trials': [{'number': t.number, 'state': t.state.name, 'parameters': t.params,
                            'value': t.value, 'metadata': t.user_attrs}
                           for t in version['study'].get_trials()] if version['study'] is not None else []}
        art = self.c.artifacts.write_json(f'pipelines/search-{self.revision:05d}.json',
            {'revision': self.revision, 'versions': rows, 'studies': histories,
             'queued_initial_versions': [v['version_id'] for v in self.queue],
             'running': [{'version_id': v['version_id'], 'experiment_id': job.experiment_id, 'initial': initial}
                         for job, v, _, initial in self.jobs.values()]}, 'pipeline_search')
        self.c.context.pipeline_status = {'artifact_id': art['artifact_id'], 'versions': rows[-8:],
            'instruction': 'Tuning runs independently. Submit improved executable representations; no need to wait for trials.'}

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
            'errors': 0, 'study': None, 'pending': False}
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
        return [v for v in self.versions.values() if v['initial_metric'] and v['space'] and not v['paused']
                and not v['pending'] and v['errors'] < 4 and v['trials'] < self.c.config.background_max_trials]

    def choose(self):
        candidates = self.eligible()
        if not candidates:
            return None
        leader = max(candidates, key=lambda v: self.quality(v['best_metric']))
        # A busy leader does not monopolise the second slot. Give every challenger
        # a small probe before preferring the strongest measured version.
        untested = [v for v in candidates if v['trials'] < 3]
        if untested:
            return min(untested, key=lambda v: (v['trials'], -self.quality(v['initial_metric'])))
        if self.attempted % 5 == 4:
            return min(candidates, key=lambda v: v['last_scheduled'])
        return leader

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
            plan['resources'] = {'cpu_threads': cpu, 'memory_limit_mb': memory,
                'timeout_seconds': min(c.config.experiment_timeout_seconds, c.config.background_trial_seconds)
                                   if not initial else c.config.experiment_timeout_seconds}
            c.catalog.validate_plan(plan, c.spec, c.validation['validation_id'], c.artifacts)
            if not c.budget.can_train(plan)['allowed']:
                version['paused'] = 'no_affordable_trial'
                if trial is not None:
                    self.study(version).tell(trial, state=self.library.trial.TrialState.FAIL)
                return False
            ticket = execution_ticket(c, version['ticket'], None if initial else version['initial_experiment_id'])
            from agent.graph.controller import lineage_for
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
                version['paused'] = 'no_affordable_trial'
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
            if trial is not None and trial.number in {t.number for t in self.study(version).get_trials() if not t.state.is_finished()}:
                self.study(version).tell(trial, state=self.library.trial.TrialState.FAIL)
            c.record_research_feedback(plan['researcher_id'], version['ticket']['work_order']['branch_id'],
                'pipeline_rejected', str(error), code=getattr(error, 'code', 'PIPELINE_ERROR'))
            c.log.exception('pipeline.execution_failed', error, version_id=version['version_id'])
            return False

    def measured(self, version, eid, metric, trial, initial):
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
                        other['paused'] = 'superseded_by_initial:'+version['version_id']
                self.c.log.event('pipeline.leader_replaced', version_id=version['version_id'],
                    experiment_id=eid, metric=metric, reason='New initial score exceeds previous tuned best')
        self.c.update_final_reserve()
        champion = self.c.store.champion()
        best = self.c.store.best()
        self.publish_due |= bool(best and (not champion or best != champion['experiment_id']))
        self.save()

    def poll(self):
        c = self.c
        changed = False
        for key, (job, version, trial, initial) in list(self.jobs.items()):
            if version['paused'] and version['paused'].startswith('superseded_by_initial:') and not initial and not job.event and job.process.poll() is None:
                c.cancel_job(job, 'Superseded by a stronger initial pipeline version')
                c.log.event('pipeline.execution_preempted', version_id=version['version_id'], experiment_id=job.experiment_id)
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
                c.record_research_feedback(version['anchor']['researcher_id'], version['ticket']['work_order']['branch_id'],
                    'pipeline_execution_failed', canonical(event.get('error')), code='PIPELINE_EXECUTION_FAILED')
            if initial:
                version['paused'] = 'initial_execution_failed'
            self.save()
        if changed:
            c.publish_research_updates()

    def fill(self):
        if self.queue and self.jobs and (len(self.jobs) >= self.c.config.max_workers
                or self.c.budget.snapshot()['cpu_threads_available'] <= int(not self.c.budget.analysis_active)):
            trials = [(job, v) for job, v, _, initial in self.jobs.values() if not initial and not job.event]
            if trials:
                job, version = min(trials, key=lambda pair: self.quality(pair[1]['best_metric']))
                self.c.cancel_job(job, 'Priority first evaluation of a new executable version')
                self.c.log.event('pipeline.execution_preempted', version_id=version['version_id'],
                                 experiment_id=job.experiment_id, reason='new_initial_version')
                return
        if self.publish_due:
            if self.jobs:
                return
            self.c.finalize_best()
            self.publish_due = False
        while len(self.jobs) < self.c.config.max_workers and self.c.budget.snapshot()['search_allowed']:
            initial = bool(self.queue)
            version = self.queue.pop(0) if initial else self.choose()
            if version is None:
                break
            started = self.start(version, initial=initial)
            if not started:
                if initial and not version['paused'] and version['errors'] < 4:
                    self.queue.insert(0, version)
                break


def search(controller, root, accepted):
    """Research futures and version-owned train processes share one dispatcher loop."""
    from agent.graph.controller import issue, stop_ticket, publish_activity, finish
    c = controller
    service = VersionSearch(c)
    c.pipeline_search = service
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
    pending, deferred = {}, {}
    repeats = {}
    pool = ThreadPoolExecutor(max_workers=len(peers), thread_name_prefix='researcher')
    reason = 'time_reserve'
    try:
        while c.budget.snapshot()['search_allowed']:
            service.poll()
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
                        repeats.pop(peer, None)
                    else:
                        stop_ticket(c, ticket, proposal.get('stop_reason') or 'Research checkpoint')
                        signature = hashlib.sha256(canonical([proposal.get('stop_reason'),
                            sorted({r['artifact_id'] for r in c.context.analysis.records})]).encode()).hexdigest()
                        previous, count = repeats.get(peer, (None, 0))
                        repeats[peer] = (signature, count+1 if signature == previous else 1)
                        if proposal['decision'] == 'stop' or repeats[peer][1] >= 4:
                            deferred[peer] = service.revision
                except Exception as error:
                    c.graph.abandon(work['work_order_id'], str(error))
                    c.record_research_feedback(peer, root, 'proposal_failed', str(error), code=getattr(error, 'code', 'PIPELINE_ERROR'))
                    c.log.exception('research.proposal_failed', error, work_order_id=work['work_order_id'])
                    # A failed dialogue does not revoke independently completed tool reports.
                    previous, count = repeats.get(peer, (None, 0))
                    repeats[peer] = ('error', count+1 if previous == 'error' else 1)
                    if repeats[peer][1] >= 3:
                        deferred[peer] = service.revision
            for peer in peers:
                if any(t['work_order']['researcher_id'] == peer for t in pending.values()):
                    continue
                if deferred.get(peer) == service.revision or not c.llm_available(len(pending)):
                    continue
                try:
                    c.context.round_no += 1
                    ticket = issue(c, root, peer)
                    publish_activity(c, ticket, started=True)
                    future = pool.submit(c.call, 'C09', ticket['researcher_request']['payload'], c.researcher.propose)
                    pending[future] = ticket
                    c.log.event('research.dialogue_dispatched', work_order_id=ticket['work_order']['work_order_id'],
                        researcher_id=peer, branch_id=root)
                except AgentError as error:
                    deferred[peer] = service.revision
                    c.record_research_feedback(peer, root, 'context_unavailable', str(error), code=error.code)
            service.fill()
            if not pending and not service.jobs and not service.queue and not service.eligible():
                reason = 'research_and_background_exhausted'
                break
            time.sleep(.1)
    finally:
        c.researcher.gateway.cancelled.set()
        for future, ticket in pending.items():
            future.cancel()
            c.graph.abandon(ticket['work_order']['work_order_id'], 'Research deadline reached')
        pool.shutdown(wait=False, cancel_futures=True)
        for job, _, _, _ in list(service.jobs.values()):
            c.cancel_job(job, 'Publication reserve reached')
        if service.jobs:
            service.poll()
        service.save()
    return finish(c, reason, accepted+service.attempted)
