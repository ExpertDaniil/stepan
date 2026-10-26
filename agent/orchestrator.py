from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
import time
import uuid

from agent.catalog import factory_source
from agent.context import ContextBuilder
from agent.errors import AgentError
from tools.code import write_code


def budget_trial_timeout(event, job, plan):
    return event['state'] == 'timed_out' and job.lease['timeout_seconds'] < plan['resources']['timeout_seconds']


class Orchestrator:
    def __init__(self, *, config, bus, budget, artifacts, store, runner, evaluator, submission, researcher,
                 spec, validation, catalog, task_text, profile, log, demo=None, task_root=None):
        self.config, self.bus, self.budget, self.artifacts, self.store = config, bus, budget, artifacts, store
        self.runner, self.evaluator, self.submission, self.researcher = runner, evaluator, submission, researcher
        self.spec, self.validation, self.catalog, self.log = spec, validation, catalog, log
        self.context = ContextBuilder(spec, task_text, profile, catalog, artifacts, log, demo)
        self.context.data_research = budget.data_research_snapshot
        self.context.store = store
        self.context.validation_id = validation["validation_id"]
        self.context.validation_sha256 = validation["protocol_sha256"]
        self.researcher.validation_id = validation["validation_id"]
        if config.research_mode == 'adaptive':
            from tools.analysis import AnalysisService
            analysis = AnalysisService(spec, validation, artifacts, budget, log)
            analysis.error_provider = evaluator.diagnose
            self.context.analysis = analysis
            self.researcher.gateway.analysis = analysis
            from tools.knowledge import KnowledgeService
            try:
                knowledge = KnowledgeService(spec, profile, catalog, artifacts, log)
            except AgentError as error:
                log.exception('knowledge.unavailable', error)
            else:
                self.context.knowledge = knowledge
                self.researcher.gateway.knowledge = knowledge
        from agent.predictions import targets
        # Rolling protocols fit once per fold; duration work counts every fitted row.
        fold_plan = evaluator.protocols[validation['validation_id']].get('fold_plan')
        fitted_rows = (sum(fold['fit_rows'] for fold in fold_plan) if fold_plan
                       else len(artifacts.read_json(validation["train_rows_artifact_id"])))
        self.budget.duration.configure(profile, fitted_rows, catalog.recipes, target_count=len(targets(spec)),
                                       predicted_rows=len(artifacts.read_json(validation["valid_rows_artifact_id"])))
        if evaluator.protocols[validation['validation_id']].get('confirmation_indices'):
            self.budget.confirmation_seconds = 1.
            evaluator.timing_observer = lambda seconds: setattr(self.budget, 'confirmation_seconds',
                max(self.budget.confirmation_seconds, 3*seconds))
        from agent.local_search import LocalSearch
        self.local_search = LocalSearch(self)
        from agent.feature_search import FeatureSearch
        self.feature_search = FeatureSearch(self)
        from agent.research_coordination import ResearchCoordination
        self.coordination = ResearchCoordination(self)
        self.no_fit_branches = set()
        self.no_fit_reasons = {}
        self.research_feedback = []
        self.context.research_feedback = self.research_feedback
        self.failed_refits = set()
        from agent.research_feedback import FeedbackMailbox
        self.researcher.gateway.feedback = FeedbackMailbox(artifacts)
        self.fallback_started_at = None
        self.fallback_seconds = 0.0
        self.research_stalls = {}
        self.degraded = False
        self.graph = None
        if config.graph_enabled:
            from agent.graph.service import GraphService
            task_root = task_root or artifacts.resolve(spec["train_artifact_id"]).parent
            self.graph = GraphService(self, task_root)
            self.graph.initialize_defaults()
            recovery = self.graph.invoke("G12", {"mode": "inspect", "trusted_wall_seconds_remaining": int(budget.remaining())})
            if recovery["state"] == "blocked":
                raise AgentError("GRAPH_RECOVERY_BLOCKED", "; ".join(recovery["blocking_reasons"]))
        self.science = None
        if config.research_mode == 'adaptive' and config.scientific_research_enabled:
            from agent.scientific_research import ScientificResearch
            self.science = ScientificResearch(self)
            self.context.science = self.science
        from agent.error_research import ErrorResearch
        self.error_research = ErrorResearch(self)
        self.context.error_research = self.error_research

    def call(self, cid, payload, handler, key=None):
        return self.bus.call(cid, payload, handler, key=key)["data"]

    def lease(self, owner, phase, resources):
        return self.call("C13", {"operation": "acquire", "owner_id": owner, "phase": phase, "requested": resources,
                                  "lease_id": None, "fencing_token": None}, self.budget.lease,
                         key="acquire/" + owner + "/" + phase)

    def release(self, job):
        if self.runner.poll(job) is None:
            raise AgentError("PROCESS_STOP_FAILED", "Refusing to release a running process tree")
        self.call("C13", {"operation": "release", "owner_id": getattr(job, 'lease_owner_id', None) or (job.experiment_id if job.phase == "train" else job.submission_job_id),
                           "phase": job.phase, "requested": self.resources(job.lease["timeout_seconds"]),
                           "lease_id": job.lease["lease_id"], "fencing_token": job.lease["fencing_token"]}, self.budget.lease,
                  key="release/" + job.lease["lease_id"])

    def resources(self, timeout=None):
        return self.budget.sized_resources({"cpu_threads": self.config.train_cpu_threads, "memory_limit_mb": self.config.train_memory_mb,
                "timeout_seconds": timeout or self.config.experiment_timeout_seconds})

    def admit_training(self, plan, owner):
        window = self.budget.data_research_snapshot()
        if window['active'] and not plan.get('feature_steps'):
            self.log.event('research.data_phase_trial_rejected', plan_id=plan['plan_id'],
                experiment_id=owner, recipe_id=plan['recipe_id'], **window,
                message='Обучение без изменений данных отложено до завершения обязательного этапа')
            return False
        decision = self.budget.admit(owner, plan)
        self.log.event("search.admission", experiment_id=owner, recipe_id=plan["recipe_id"],
                       hyperparameters=plan["model"]["hyperparameters"], **decision)
        return decision["allowed"]

    def llm_available(self, pending=0):
        if self.budget.llm_unavailable_reason(pending):
            return False
        try:
            self.researcher.gateway.check_availability()
        except AgentError:
            return False
        return self.budget.snapshot()["search_allowed"]

    def llm_unavailable_reason(self, pending=0):
        try:
            self.researcher.gateway.check_availability()
        except AgentError as error:
            return error.code.lower()
        return self.budget.llm_unavailable_reason(pending)

    def set_fallback(self, reason=None):
        if reason is not None and self.fallback_started_at is None:
            self.fallback_started_at = time.monotonic()
            self.log.event('search.fallback_started', reason=reason, budget=self.budget.snapshot(),
                           token_budget=self.budget.token_snapshot())
        elif reason is None and self.fallback_started_at is not None:
            seconds = time.monotonic()-self.fallback_started_at
            self.fallback_seconds += seconds
            self.fallback_started_at = None
            self.log.event('search.fallback_finished', duration_seconds=seconds, total_seconds=self.fallback_seconds)

    def publish_research_updates(self):
        """Publish only this branch's visible results, after evaluation and Optuna tell."""
        if self.config.research_mode != 'adaptive':
            return
        self.store._writer()
        mailbox = self.researcher.gateway.feedback
        scopes = ([r[0] for r in self.store.db.execute(
            "SELECT branch_id FROM g_branches WHERE run_id=? AND state='active'", (self.store.run_id,))]
            if self.graph else ['legacy'])
        for scope in scopes:
            ids = (list(reversed(self.graph.visible_history(scope))) if self.graph else [r[0] for r in self.store.db.execute(
                'SELECT experiment_id FROM experiments WHERE run_id=? ORDER BY rowid', (self.store.run_id,))])
            measured = []
            for eid in ids:
                ev = self.store.experiment_evaluation(eid)
                if ev and ev['valid'] and not ev['violations']:
                    exp = self.store.get(eid)
                    measured.append({'experiment_id': eid, 'evaluation_id': ev['evaluation_id'],
                        'primary_metric': ev['primary_metric'], 'diagnostics_artifact_id': ev['diagnostics_artifact_id'],
                        'plan_id': exp['plan']['plan_id'], 'recipe_id': exp['plan']['recipe_id'],
                        'execution_kind': ('background_tuning' if exp['plan']['plan_id'].startswith('background-trial-') else 'candidate'),
                        'effective_parameters': exp['plan']['model']['hyperparameters']})
            memory = self.graph.research_memory(scope, []) if self.graph else None
            if memory is None:
                from agent.research_memory import build_research_memory
                memory = build_research_memory(self.store, self.artifacts, ids)
            series = [s for d in memory['decisions'] for s in d.get('tuning_series', [])]
            latest = measured[-8:]
            # Candidate outcomes must survive the fast-moving trial window, even
            # when a researcher's submitted representation does not become best.
            candidate_results = [{k:r[k] for k in ('experiment_id', 'evaluation_id', 'primary_metric',
                'diagnostics_artifact_id', 'plan_id', 'recipe_id', 'execution_kind')}
                for r in measured if r['execution_kind'] == 'candidate'][-4:]
            research = getattr(self, 'error_research', None)
            verdicts = {r['candidate_experiment_id']: r for r in (research.results if research else [])}
            for row in candidate_results:
                verdict = verdicts.get(row['experiment_id'])
                if verdict:
                    # Row-noise verdict of the paired comparison with the research base.
                    row['paired_verdict'] = {'status': verdict['status'], 'base_experiment_id': verdict['proposal_base_experiment_id'],
                                             'improvement': verdict['overall'].get('improvement'),
                                             **{k: verdict.get(k) for k in ('probability_better', 'resampling_unit', 'independent_units', 'unavailable')}}
            incumbent = self.research_reference(measured, scope)
            if incumbent and incumbent not in latest:
                latest.insert(0, incumbent)
            for result in latest[:-2]:
                result.pop('effective_parameters')
                result['parameter_details'] = 'Frozen plan/source in execution-'+result['experiment_id']
            summaries = []
            for s in series[-4:]:
                summaries.append({k: v for k, v in s.items() if k != 'trials'})
            activity = self.coordination.snapshot(scope)
            fit_diagnostics = self.science.diagnostics.visible(ids) if getattr(self, 'science', None) else []
            terminal_states = [(eid, self.store.get(eid)['state']) for eid in ids
                               if self.store.get(eid)['state'] in {'failed', 'invalid', 'timed_out', 'cancelled'}]
            generation = hashlib.sha256(json.dumps([
                [(r['experiment_id'], r['primary_metric']) for r in measured], terminal_states,
                [r['artifact_id'] for r in fit_diagnostics]],
                sort_keys=True, allow_nan=False).encode()).hexdigest()
            pending_experiments = []
            pending_ids = ([r[0] for r in self.store.db.execute(
                "SELECT DISTINCT e.experiment_id FROM experiments e JOIN g_bindings b USING(run_id,experiment_id) "
                "JOIN g_work_orders w USING(run_id,work_order_id) WHERE e.run_id=? AND w.branch_id=? "
                "AND e.state IN ('queued','running','trained')", (self.store.run_id, scope))] if self.graph else ids)
            for eid in pending_ids:
                exp = self.store.get(eid)
                if exp['state'] in {'queued', 'running', 'trained'}:
                    pending_experiments.append({'experiment_id': eid, 'state': exp['state'],
                        'researcher_id': exp['plan']['researcher_id'], 'recipe_id': exp['plan']['recipe_id'],
                        'plan_id': exp['plan']['plan_id'],
                        'execution_kind': ('background_tuning' if exp['plan']['plan_id'].startswith('background-trial-') else 'candidate')})
            from agent.evidence import comparison_context
            from agent.context_details import solution_record
            solution_catalog = {r['experiment_id']: solution_record(self.store, self.artifacts, r['experiment_id'])
                                for r in [*latest, *candidate_results]}
            mailbox.publish(scope, {'branch_id': scope, 'dataset_id': self.spec['dataset_id'],
                'task_spec_id': self.spec['task_spec_id'], 'validation_id': self.validation['validation_id'],
                'comparison_evidence': comparison_context(self.store, self.artifacts, ids),
                'measurement_generation': generation, 'activity': activity,
                'pending_experiments': pending_experiments[-4:],
                'admission_feedback': [r for r in self.research_feedback if r['branch_id'] == scope][-4:],
                'results': latest, 'omitted_result_count': len(measured)-len(latest),
                'candidate_results': candidate_results,
                'solution_catalog': solution_catalog,
                'best_solution': ({'experiment_id': incumbent['experiment_id'], 'recipe_id': incumbent['recipe_id'],
                    'metric': incumbent['primary_metric'], 'read': {'experiment_id': incumbent['experiment_id'], 'view': 'pipeline'}}
                    if incumbent else None),
                'fit_diagnostics': fit_diagnostics,
                'series': summaries, 'omitted_series_count': max(0, len(series)-len(summaries)),
                'instruction': 'Controller measurements at request dispatch. Compare only compatible protocols. '
                    'Metrics here are exact; a result read repeats them. The assignment base is graph lineage, not '
                    'necessarily the utility reference; read_solution(experiment_id="best") returns the stable visible '
                    'reference. Individual trial IDs remain readable; bootstrap support does not prove transfer.'})

    def research_reference(self, measured, scope):
        """Stable branch-local utility reference; raw best/results remain intact."""
        if not measured:return None
        best = max(measured, key=lambda r: r['primary_metric']['value']*(
            1 if r['primary_metric']['direction'] == 'maximize' else -1))
        leaders = getattr(self, 'research_leaders', {})
        old = next((r for r in measured if r['experiment_id'] == leaders.get(scope)), None)
        chosen = best
        if old and old['experiment_id'] != best['experiment_id'] and getattr(self.config, 'noise_aware_publication', True):
            from agent.significance import admits_update
            try:
                verdict = self.evaluator.paired_significance(old['experiment_id'], best['experiment_id'])
                if not admits_update(verdict):chosen = old
            except (AgentError, ValueError, KeyError):
                chosen = old  # Missing evidence is not a reason to interrupt research.
        self.research_leaders = {**leaders, scope: chosen['experiment_id']}
        if leaders.get(scope) != chosen['experiment_id']:
            self.log.event('research.utility_reference_changed', branch_id=scope,
                           previous_experiment_id=leaders.get(scope), experiment_id=chosen['experiment_id'])
        return chosen

    def local_plan(self, peer, branch="legacy", base=None, constraints=None, assignment=None):
        if self.budget.data_research_snapshot()['active']:
            return None  # API exhaustion never opens parameter search before the data window ends.
        optimizer = getattr(self.local_search, 'optuna', None)
        if optimizer is not None and optimizer.has_series(peer, branch):
            return optimizer.next_plan(peer, branch, constraints)
        choice = self.local_search.next_choice(peer, branch, base, constraints, assignment=assignment)
        if choice is None:
            return None
        plan = self.warmup_plan(*choice)
        if self.config.research_mode == 'adaptive':
            authorization = self.local_search.authorizations.get((branch, peer))
            if authorization:
                left = int(authorization['deadline']-time.monotonic())
                if left < 1:
                    return None
                plan['resources']['timeout_seconds'] = min(plan['resources']['timeout_seconds'], left)
        plan.update(researcher_id=peer, parent_ids=[self.store.best()] if self.store.best() else [],
                    hypothesis="Related local catalogue trial: vary checked parameters on the fixed validation.",
                    expected_effect="Measure against the recorded base; improvement is not assumed.")
        self.log.event("search.local_trial", researcher_id=peer, branch_id=branch, recipe_id=plan["recipe_id"],
                       hyperparameters=plan["model"]["hyperparameters"])
        return plan

    def authorize_research(self, plan, branch='legacy'):
        """Dispatcher admits an explicit research action; implicit catalogue series stay legacy-only."""
        if self.config.research_mode != 'adaptive':
            self.local_search.start_series(plan['researcher_id'], branch, plan)
            return
        intent = self.artifacts.read_json('intent-'+plan['plan_id'])
        from agent.data_research import require_action
        require_action({**intent, 'feature_steps': plan['feature_steps']}, self.budget.data_research_snapshot())
        request = None
        if intent['action'] == 'tune':
            request = dict(intent['tuning'])
            science = getattr(self, 'science', None)
            if science is not None and plan['plan_id'] in science.funding:
                request['budget_seconds'] = min(request['budget_seconds'], science.funding[plan['plan_id']])
            request['budget_seconds'] = min(request['budget_seconds'], max(0, int(min(
                self.budget.phase_remaining(), self.budget.remaining()-self.budget.reserve_seconds()))))
            self.local_search.start_series(plan['researcher_id'], branch, plan, request=request)
        else:
            self.local_search.clear_series(plan['researcher_id'], branch)
        self.log.event('research.action_admitted', plan_id=plan['plan_id'], action=intent['action'],
                       branch_id=branch, tuning=intent['tuning'], effective_tuning=request, evidence_ids=intent['evidence_ids'])
        if hasattr(self, 'publish_research_updates'):
            self.publish_research_updates()
        return intent

    def record_research_feedback(self, peer, branch, status, message, *, code=None):
        """Dispatcher-owned feedback, including failures that created no ExperimentPlan."""
        record = {'researcher_id': peer, 'branch_id': branch, 'status': status,
                  'message': self.log.redactor.clean(str(message))[:1200], 'code': code}
        art = self.artifacts.write_json('research/feedback-'+uuid.uuid4().hex[:12]+'.json', record, 'research_feedback')
        self.research_feedback.append({**record, 'artifact_id': art['artifact_id']})
        # This bounds context size, not the number or scientific direction of actions.
        self.research_feedback[:] = self.research_feedback[-8:]
        self.log.event('research.feedback_recorded', **record, artifact_id=art['artifact_id'])

    def prepare_adaptive_proposal(self, plan, branch='legacy', constraints=None, assignment=None):
        """Authorize an action before deduplicating its optional starting fit."""
        candidate_intent = self.artifacts.read_json('intent-'+plan['plan_id'])
        from agent.data_research import require_action, require_feature_parameters
        window = self.budget.data_research_snapshot()
        require_action({**candidate_intent, 'feature_steps': plan['feature_steps']}, window)
        if candidate_intent['action'] == 'features':
            params = plan['model']['hyperparameters']
            require_feature_parameters({'action': 'features', 'recipe_id': params['base_recipe_id'],
                'hyperparameters': params['model_parameters']}, self.catalog, window, constraints)
        science = getattr(self, 'science', None)
        if science is not None and science.review(plan, candidate_intent, branch):
            self.publish_research_updates()
            return None
        from agent.evidence import assess_claims
        assessed = assess_claims(self.store, self.artifacts, candidate_intent.get('claims', []),
                                 candidate_intent.get('visible_evidence_ids', []))
        assessment = self.artifacts.write_json('research/evidence-'+uuid.uuid4().hex[:16]+'.json',
            {'plan_id': plan['plan_id'], 'branch_id': branch, 'assessments': assessed,
             'model_prose_verified': False, 'scope': 'Only controller-computed measurements/contrasts are authoritative.'},
            'evidence_assessment')
        unsupported = sum(a['status'] == 'unverified' and a['model_claim']['scope'] != 'hypothesis' for a in assessed)
        self.log.event('research.evidence_assessed', plan_id=plan['plan_id'], artifact_id=assessment['artifact_id'],
                       claims=len(assessed), unsupported_claims=unsupported, model_prose_verified=False)
        if unsupported:
            self.record_research_feedback(plan['researcher_id'], branch, 'unsupported_claim_scope',
                'The complete experiment can proceed, but '+str(unsupported)+' claims exceed the available evidence. '
                'Read controller comparison_evidence; component mechanisms and family ceilings remain unresolved.')
        coordinator = getattr(self, 'coordination', None)
        conflict = coordinator.review(plan, candidate_intent, branch) if coordinator is not None else None
        if conflict is not None:
            self.record_research_feedback(plan['researcher_id'], branch, 'overlapping_search_deferred',
                json.dumps(conflict, ensure_ascii=False), code='RESEARCH_OVERLAP')
            self.log.event('research.admission_reconsidered', plan_id=plan['plan_id'], branch_id=branch, **conflict)
            self.publish_research_updates()
            return None
        try:
            intent = self.authorize_research(plan, branch)
        except AgentError as error:
            if error.code not in {'OPTUNA_UNAVAILABLE', 'INVALID_TUNING_SPACE'}:
                raise
            self.record_research_feedback(plan['researcher_id'], branch, 'tuning_rejected', str(error), code=error.code)
            return None
        if intent['action'] == 'tune':
            # The anchor is a reference, not an implicit fit outside the declared
            # search space. Historical compatible evaluations warm-start Optuna.
            return self.local_search.optuna.next_plan(plan['researcher_id'], branch, constraints)
        repeated = self.local_search.repeated(plan)
        if repeated:
            # Store/Graph deduplication will return the recorded result; never
            # replace an explicit adaptive choice with an unauthorized local series.
            self.record_research_feedback(plan['researcher_id'], branch, 'duplicate',
                'This configuration is already recorded. Reuse its result and choose a new action.')
            return plan
        if not self.budget.can_train(plan)['allowed']:
            self.record_research_feedback(plan['researcher_id'], branch, 'not_affordable',
                json.dumps(self.budget.can_train(plan), ensure_ascii=False))
            return None
        return plan

    def update_final_reserve(self):
        best = self.final_candidate()
        champion = self.store.champion()
        # Publication may keep a stable champion while raw-best is still a final
        # confirmation contender. Its refit must not become an unreserved cost.
        if not getattr(self, 'finalization_started', False):
            raw = self.store.best()
            if raw and raw not in self.failed_refits and not self.submission.ready_candidate(raw):
                best = raw
        if best and best not in self.failed_refits and (not champion or champion["experiment_id"] != best):
            self.budget.set_final_plan(self.store.get(best)["plan"])
        else:
            self.budget.set_final_plan(None)

    def final_candidate(self):
        """A validation winner that failed refit must not block other candidates."""
        confirmed = getattr(self, 'confirmed_candidate', None)
        if confirmed and confirmed not in self.failed_refits:
            return confirmed
        if getattr(self, 'finalization_started', False):
            frozen = getattr(self, 'finalization_candidate', None)
            if frozen and frozen not in self.failed_refits:
                return frozen
            return self.affordable_final_candidate()
        best = self.store.best()
        if not best or best not in self.failed_refits:
            return self.noise_gate(best)
        from agent.store import comparable, is_better
        reference = self.store.experiment_evaluation(best)['primary_metric']
        chosen, metric = None, None
        for row in self.store.db.execute("SELECT experiment_id FROM experiments WHERE run_id=? AND state='evaluated' ORDER BY rowid", (self.store.run_id,)):
            eid = row[0]
            if eid in self.failed_refits:
                continue
            evaluation = self.store.experiment_evaluation(eid)
            if not evaluation or not evaluation['valid'] or evaluation['violations']:
                continue
            candidate = evaluation['primary_metric']
            if comparable(candidate, reference) and (metric is None or is_better(candidate, metric)):
                chosen, metric = eid, candidate
        return chosen

    def noise_gate(self, best):
        """Keep the published champion while a newer best is within selection-row noise.

        Applies only during search: closed confirmation and finalization choose among
        finalists on their own rows. A raw-best that is not significantly better than
        the champion (paired bootstrap, agent.significance) would cost a refit and
        replace a measured output by a statistically indistinguishable one.
        """
        champion = self.store.champion()
        if (not best or not champion or champion['experiment_id'] == best or champion['experiment_id'] in self.failed_refits
                or getattr(self, 'finalization_started', False) or not getattr(self.config, 'noise_aware_publication', True)):
            return best
        incumbent = champion['experiment_id']
        try:
            verdict = self.evaluator.paired_significance(incumbent, best)
        except (AgentError, ValueError, KeyError) as error:
            self.log.event('submission.noise_gate_unavailable', experiment_id=best, champion_experiment_id=incumbent,
                           reason=str(error)[:300])
            return best
        logged = getattr(self, 'noise_gate_logged', set())
        self.noise_gate_logged = logged
        if (incumbent, best) not in logged:
            logged.add((incumbent, best))
            self.log.event('submission.noise_gate', experiment_id=best, champion_experiment_id=incumbent,
                           significant=verdict['significant'], gain=verdict['gain'], probability_better=verdict['probability_better'],
                           interval=verdict['interval'], method=verdict['method'], rule=verdict.get('rule'),
                           **{k: verdict.get(k) for k in ('resampling_unit', 'independent_units', 'unavailable')})
        from agent.significance import admits_update
        return best if admits_update(verdict) else incumbent

    def affordable_final_candidate(self):
        """Choose an entire deliverable, including refit/predict/validation cost."""
        from agent.store import comparable
        from agent.duration import refit_seconds
        best = self.store.best()
        if not best:return None
        reference = self.store.experiment_evaluation(best)['primary_metric']
        champion = self.store.champion()
        choices = []
        for row in self.store.db.execute("SELECT experiment_id FROM experiments WHERE run_id=? AND state='evaluated'", (self.store.run_id,)):
            eid = row[0]
            if eid in self.failed_refits:continue
            ev = self.store.experiment_evaluation(eid)
            if not ev or not ev['valid'] or ev['violations'] or not comparable(ev['primary_metric'], reference):continue
            ready = (champion and champion['experiment_id'] == eid) or bool(self.submission.ready_candidate(eid))
            cost = 0 if ready else (refit_seconds(self.budget.duration, self.store.get(eid)['plan'])
                                   +self.budget.output_reserve()+self.budget.confirmation_seconds)
            if ready or cost <= self.budget.remaining():
                metric = ev['primary_metric']
                choices.append((metric['value']*(1 if metric['direction']=='maximize' else -1), eid))
        return max(choices)[1] if choices else (champion['experiment_id'] if champion else None)

    def begin_finalization(self):
        if getattr(self, 'finalization_started', False):return
        stop = getattr(self.researcher.gateway, 'stop_research', self.researcher.gateway.cancelled.set)
        stop()
        self.budget.search_stopped = True
        analysis = getattr(self.context, 'analysis', None)
        if analysis is not None:
            analysis.stop()
            until = min(time.monotonic()+1, self.budget.deadline)
            while self.budget.analysis_active and time.monotonic() < until:
                time.sleep(.01)
        if getattr(self, 'science', None):self.science.diagnostics.shutdown()
        active = self.runner.active()
        for job in active:self.cancel_job(job, 'Final confirmation/publication barrier')
        if active:self.wait_jobs(active)
        self.finalization_candidate = self.affordable_final_candidate()
        self.finalization_started = True
        self.update_final_reserve()
        self.log.event('submission.finalization_barrier', experiment_id=self.finalization_candidate,
                       remaining_seconds=self.budget.remaining(), research_closed=True)

    def submit_plan(self, payload):
        plan = self.catalog.validate_plan(payload["plan"], self.spec, self.validation["validation_id"], self.artifacts)
        return self.store.reserve(plan, self.artifacts.meta(plan["code_artifact_id"])["sha256"])

    def cancel_job(self, job, reason, state="cancelled"):
        return self.call("C24", {"job_id": job.job_id, "experiment_id": job.experiment_id,
                                  "attempt_no": job.attempt_no, "fencing_token": job.lease["fencing_token"],
                                  "reason": reason, "terminal_state": state, "grace_seconds": 1}, self.runner.cancel,
                         key="cancel/" + job.job_id + "/" + state)

    def _heartbeat_and_limits(self, job):
        now = time.monotonic()
        if job.process.poll() is not None:
            # The worker can exit between poll() and this check. Runner.poll()
            # owns descendant cleanup and validates the completed result next.
            return
        elapsed = now - job.started
        lease_state = self.budget.leases.get(job.lease["lease_id"])
        if self.budget.remaining() <= 1 or elapsed > job.lease["timeout_seconds"]:
            self.cancel_job(job, "Job or run deadline reached", "timed_out")
            return
        if job.peak_memory_mb > job.lease["memory_limit_mb"]:
            self.cancel_job(job, "Resident memory limit exceeded")
            return
        if lease_state and now > lease_state["heartbeat_deadline"]:
            self.cancel_job(job, "Lease heartbeat expired")
            return
        if now - job.last_heartbeat >= 10:
            self.log.event("job.progress", job_id=job.job_id, experiment_id=job.experiment_id, phase=job.phase,
                           elapsed_seconds=round(elapsed, 1), timeout_seconds=job.lease["timeout_seconds"],
                           peak_memory_mb=round(job.peak_memory_mb, 1), run_seconds_remaining=round(self.budget.remaining(), 1))
            self.call("C13", {"operation": "renew", "owner_id": getattr(job, 'lease_owner_id', None) or (job.experiment_id if job.phase == "train" else job.submission_job_id),
                               "phase": job.phase, "requested": self.resources(job.lease["timeout_seconds"]),
                               "lease_id": job.lease["lease_id"], "fencing_token": job.lease["fencing_token"]}, self.budget.lease)
            job.last_heartbeat = now

    def wait_jobs(self, jobs):
        results = {}
        pending = list(jobs)
        while pending:
            for job in list(pending):
                event = self.runner.poll(job)
                if event is None:
                    self._heartbeat_and_limits(job)
                    event = self.runner.poll(job)
                if event is None:
                    continue
                if getattr(job, 'diagnostic', False):
                    # Diagnostic predictions are discovery evidence, never a
                    # candidate state transition, score, or duration observation.
                    self.release(job)
                    job.reported = True
                    results[job.job_id] = event
                    pending.remove(job)
                    continue
                if getattr(self, "graph", None):
                    try:
                        self.graph.pin_event_outputs(event)
                    except (AgentError, OSError) as exc:
                        self.degraded = True
                        error = exc if isinstance(exc, AgentError) else AgentError("ARTIFACT_REGISTRATION_FAILED", str(exc))
                        event = {**event, "state": "failed", "error": error.as_dict()}
                self.call("C15", {"event": event}, lambda p, phase=job.phase: self.store.apply_event(p["event"], phase), key=event["event_id"])
                self.release(job)
                job.reported = True
                results[job.job_id] = event
                pending.remove(job)
                if job.phase == "train":
                    self.budget.last_train_duration = max(self.budget.last_train_duration, event["duration_seconds"])
                plan = self.store.get(job.experiment_id)["plan"]
                try:
                    result = json.loads((job.directory / "result.json").read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    result = {}
                self.budget.observe(plan, job.phase, event, result.get("fit_seconds"))
                self.log.event("search.duration_observed", experiment_id=job.experiment_id,
                               family=plan["model"]["family"], phase=job.phase, state=event["state"],
                               duration_seconds=event["duration_seconds"], fit_seconds=result.get("fit_seconds"),
                               hyperparameters=plan["model"]["hyperparameters"])
            if pending:
                if getattr(self, 'science', None) and any(not getattr(j, 'diagnostic', False) for j in jobs):
                    self.science.diagnostics.poll()
                time.sleep(0.1)
        return results

    def run_training(self, plans, tickets=None):
        started = time.monotonic()
        try:
            return self._run_training(plans, tickets)
        finally:
            features = getattr(self, 'feature_search', None)
            if features is not None and any(p['plan_id'] in features.plan_ids for p in plans):
                features.spent_seconds += time.monotonic()-started
            optimizer = getattr(getattr(self, 'local_search', None), 'optuna', None)
            if optimizer is not None:
                optimizer.sync()
            self.publish_research_updates()

    def _run_training(self, plans, tickets=None):
        if getattr(self, "graph", None):
            from agent.graph.controller import train
            return train(self, plans, tickets)
        jobs = []
        for plan in plans:
            record = self.call("C12", {"plan": plan}, self.submit_plan)
            optimizer = getattr(self.local_search, 'optuna', None)
            if optimizer is not None:
                optimizer.bind(plan, record['experiment_id'])
            if record["disposition"] == "duplicate":
                self.log.event("experiment.duplicate_skipped", experiment_id=record["experiment_id"])
                continue
            ident = record["experiment_id"]
            if not self.admit_training(plan, ident):
                with self.store.transaction():
                    self.store.db.execute("UPDATE experiments SET state='cancelled',error=? WHERE experiment_id=?", ("Predicted duration exceeds remaining search time", ident))
                continue
            try:
                lease_reply = self.lease(ident, "train", plan["resources"])
            except AgentError as exc:
                if exc.code != "BUDGET_EXHAUSTED":
                    raise
                with self.store.transaction():
                    self.store.db.execute("UPDATE experiments SET state='cancelled',error=? WHERE experiment_id=?", (str(exc), ident))
                continue
            if not lease_reply["granted"]:
                raise AgentError("RESOURCE_BUSY", "Batch exceeds available execution slots")
            lease = lease_reply["lease"]
            attempt = self.store.start(ident, lease)
            args = {"experiment_id": ident, "attempt_no": attempt, "plan": plan, "lease": lease,
                    "train_rows_artifact_id": self.validation["train_rows_artifact_id"],
                    "valid_rows_artifact_id": self.validation["valid_rows_artifact_id"],
                    "output_kinds": ["validation_predictions", "fitted_pipeline", "execution_log"]}
            try:
                reply = self.call("C14", args, lambda p, eid=ident: self.runner.start_train(p, self.spec, self.store.get(eid)), key=ident + "/" + str(attempt))
            except Exception as exc:
                self.store.fail_start(ident, exc)
                if not any(job.lease["lease_id"] == lease["lease_id"] for job in self.runner.jobs.values()):
                    self.call("C13", {"operation": "release", "owner_id": ident, "phase": "train", "requested": plan["resources"],
                                       "lease_id": lease["lease_id"], "fencing_token": lease["fencing_token"]}, self.budget.lease)
                raise
            jobs.append(self.runner.jobs[reply["job_id"]])
        events = self.wait_jobs(jobs)
        for job in jobs:
            event = events[job.job_id]
            if event["state"] != "succeeded":
                if budget_trial_timeout(event, job, self.store.get(job.experiment_id)['plan']) and self.store.champion():
                    self.log.event('search.budget_trial_timed_out', experiment_id=job.experiment_id, retained_output=True)
                else:
                    self.degraded = True
                continue
            predictions = next((a for a in event["output_artifact_ids"] if self.artifacts.meta(a)["kind"] == "validation_predictions"), None)
            if not predictions:
                self.store.invalidate(job.experiment_id, "Missing predictions")
                self.degraded = True
                continue
            try:
                result = self.call("C16", {"experiment_id": job.experiment_id, "task_spec_id": self.spec["task_spec_id"],
                                            "dataset_id": self.spec["dataset_id"], "validation_id": self.validation["validation_id"],
                                            "predictions_artifact_id": predictions, "fidelity": "full"}, self.evaluator.evaluate)
                self.call("C17", {"evaluation_id": result["evaluation"]["evaluation_id"], "experiment_id": job.experiment_id,
                                   "selection_policy_id": "same-protocol-primary-v1"}, self.store.commit_evaluation,
                          key=result["evaluation"]["evaluation_id"])
                self.update_final_reserve()
            except AgentError as exc:
                self.store.invalidate(job.experiment_id, str(exc))
                self.degraded = True
                self.log.exception("experiment.rejected", exc, experiment_id=job.experiment_id)
        return len(jobs)

    def baseline_plan(self):
        recipe_id = next(x for x in self.catalog.recipes if x.startswith("dummy-"))
        source = factory_source(recipe_id)
        args = {"researcher_id": "baseline", "filename": "baseline.py", "content": source, "declared_entrypoint": "make_pipeline"}
        code = self.call("C11", args, lambda p: write_code(p, self.artifacts), key="baseline-code")["artifact"]
        return {"plan_id": "plan-baseline", "researcher_id": "baseline", "parent_ids": [],
                "hypothesis": "Create a fast valid fallback before spending the research budget.",
                "expected_effect": "A valid output file and a measured reference score.", "recipe_id": recipe_id,
                "task_spec_id": self.spec["task_spec_id"], "dataset_id": self.spec["dataset_id"],
                "validation_id": self.validation["validation_id"], "seed": self.config.seed, "fidelity": "full",
                "model": {"family": "dummy", "hyperparameters": self.catalog.normalize(recipe_id, {})},
                "feature_steps": [], "code_artifact_id": code["artifact_id"], "resources": self.resources()}

    def warmup_plan(self, recipe_id, params):
        args = {"researcher_id": "catalog-warmup", "filename": "experiment.py", "content": factory_source(recipe_id), "declared_entrypoint": "make_pipeline"}
        code = self.call("C11", args, lambda p: write_code(p, self.artifacts))["artifact"]
        parent = self.store.best()
        return {"plan_id": "warmup-"+uuid.uuid4().hex[:12], "researcher_id": "catalog-warmup",
                "parent_ids": [parent] if parent else [], "hypothesis": "Measure a verified catalogue configuration before LLM-guided tuning.",
                "expected_effect": "Establish measured alternatives on the fixed validation; improvement is not assumed.",
                "recipe_id": recipe_id, "task_spec_id": self.spec["task_spec_id"], "dataset_id": self.spec["dataset_id"],
                "validation_id": self.validation["validation_id"], "seed": self.config.seed, "fidelity": "full",
                "model": {"family": self.catalog.recipes[recipe_id]["family"], "hyperparameters": params},
                "feature_steps": self.catalog.feature_steps(params), "code_artifact_id": code["artifact_id"], "resources": self.resources()}

    def finalize_best(self):
        while True:
            previous = set(self.submission.jobs)
            candidate = self.final_candidate()
            try:
                result = self._finalize_best()
                if self.graph:
                    self.graph.acknowledge_publications()
                return result
            except Exception as exc:
                for ident in set(self.submission.jobs) - previous:
                    if self.submission.jobs[ident]["state"] not in {"failed", "timed_out", "cancelled", "published", "skipped_not_better"}:
                        self.submission.change_state(ident, "failed")
                recoverable = {'REFIT_FAILED', 'INVALID_OUTPUT_SCHEMA', 'INVALID_OUTPUT_VALUES', 'ID_ORDER_MISMATCH'}
                if isinstance(exc, AgentError) and exc.code in recoverable and self.store.champion() and candidate is not None:
                    self.submission.require_output()
                    self.failed_refits.add(candidate)
                    self.degraded = True
                    self.update_final_reserve()
                    self.log.exception('submission.refit_failed', exc, experiment_id=candidate, retained_output=True)
                    continue
                raise

    def _finalize_best(self):
        if getattr(self, 'science', None) and self.science.diagnostics.running:
            self.science.diagnostics.poll()
            if self.science.diagnostics.running and not getattr(self, 'finalization_started', False):
                return  # Periodic publication waits for its fair resource slot.
        ident = self.final_candidate()
        champion = self.store.champion()
        if not ident or ident in self.failed_refits or (champion and champion["experiment_id"] == ident):
            self.budget.set_final_plan(None)
            return
        experiment = self.store.get(ident)
        ready = self.submission.ready_candidate(ident)
        if ready:
            evaluation = self.store.experiment_evaluation(ident)
            prepared = self.call('C18', {'experiment_id': ident, 'evaluation_id': evaluation['evaluation_id'],
                'task_spec_id': self.spec['task_spec_id'], 'mode': 'reuse_verified_output',
                'expected_champion_version': champion['champion_version'] if champion else 0}, self.submission.prepare)
            subjob = prepared['submission_job_id']
            self.submission.change_state(subjob, 'validating')
            self.publish_candidate(ident, evaluation, ready['submission_artifact_id'], subjob)
            self.budget.set_final_plan(None)
            self.log.event('submission.ready_output_reused', experiment_id=ident,
                candidate_artifact_id=ready['submission_artifact_id'], refit_seconds=0)
            return
        self.budget.set_final_plan(experiment["plan"])
        from agent.duration import refit_seconds
        predicted = refit_seconds(self.budget.duration, experiment["plan"])
        if champion and self.budget.remaining() < predicted + self.budget.output_reserve():
            self.log.event("submission.refit_deferred", experiment_id=ident, estimated_seconds=predicted,
                           remaining_seconds=self.budget.remaining())
            return
        evaluation = self.store.experiment_evaluation(ident)
        prepared = self.call("C18", {"experiment_id": ident, "evaluation_id": evaluation["evaluation_id"],
                                     "task_spec_id": self.spec["task_spec_id"], "mode": "refit_full_train",
                                     "expected_champion_version": champion["champion_version"] if champion else 0}, self.submission.prepare,
                             key="prepare/" + ident + "/" + str(champion["champion_version"] if champion else 0))
        subjob = prepared["submission_job_id"]
        lease_reply = self.lease(subjob, "refit", self.resources(max(self.config.experiment_timeout_seconds, math.ceil(predicted))))
        if not lease_reply["granted"]:
            raise AgentError("RESOURCE_BUSY", "Cannot refit while other jobs hold resources")
        experiment = self.store.get(ident)
        args = {"submission_job_id": subjob, "experiment_id": ident, "task_spec_id": self.spec["task_spec_id"],
                "code_artifact_id": experiment["plan"]["code_artifact_id"], "lease": lease_reply["lease"], "fit_scope": "all_train",
                "predict_artifact_id": self.spec["test_artifact_id"], "output_relative_path": "submissions/" + subjob + "/candidate.csv"}
        try:
            result = self.call("C19", args, lambda p: self.runner.start_refit(p, self.spec, experiment), key=subjob)
        except BaseException:
            lease = lease_reply['lease']
            if not any(j.lease['lease_id'] == lease['lease_id'] for j in self.runner.jobs.values()):
                self.call('C13', {'operation': 'release', 'owner_id': subjob, 'phase': 'refit',
                    'requested': self.resources(lease['timeout_seconds']), 'lease_id': lease['lease_id'],
                    'fencing_token': lease['fencing_token']}, self.budget.lease)
            self.submission.change_state(subjob, 'failed')
            raise
        job = self.runner.jobs[result["job_id"]]
        job.submission_job_id = subjob
        self.submission.change_state(subjob, "refitting")
        event = self.wait_jobs([job])[job.job_id]
        if event["state"] != "succeeded":
            self.submission.change_state(subjob, event["state"])
            raise AgentError("REFIT_FAILED", "Final training failed; previous published file is retained")
        candidate = next((a for a in event["output_artifact_ids"] if self.artifacts.meta(a)["kind"] == "submission"), None)
        if not candidate:
            raise AgentError("REFIT_FAILED", "Refit did not produce a candidate")
        self.submission.change_state(subjob, "validating")
        self.publish_candidate(ident, evaluation, candidate, subjob)

    def publish_candidate(self, ident, evaluation, candidate, subjob):
        """Reused and freshly refitted outputs pass the same C20/C21 checks."""
        validation_started = time.monotonic()
        report = self.call("C20", {"submission_job_id": subjob, "task_spec_id": self.spec["task_spec_id"],
                                   "candidate_artifact_id": candidate, "reference_test_artifact_id": self.spec["test_artifact_id"]}, self.submission.validate_output)
        current = self.store.champion()
        publication_payload = {"experiment_id": ident, "evaluation_id": evaluation["evaluation_id"], "candidate_artifact_id": candidate,
                           "validation_report_id": report["validation_report_id"], "candidate_sha256": report["candidate_sha256"],
                           "expected_champion_version": current["champion_version"] if current else 0}
        confirmation = getattr(self.submission, 'final_confirmation', None)
        if confirmation and confirmation['chosen_experiment_id'] == ident:
            publication_payload['confirmation_report_id'] = confirmation['artifact_id']
        publication = self.call("C21", publication_payload, self.submission.publish,
                  key="publish/" + ident + "/" + report["candidate_sha256"] + "/" + str(current["champion_version"] if current else 0))
        self.submission.change_state(subjob, "skipped_not_better" if publication["disposition"] == "skipped_not_better" else "published")
        self.budget.observe_publication(time.monotonic()-validation_started)
        self.update_final_reserve()

    def control(self, payload):
        self.budget.search_stopped = True
        if self.graph:
            self.graph.close_search(payload["reason"])
        self.store.set_run_state("finalizing")
        if payload["action"] == "cancel_jobs":
            for ident in payload["cancel_job_ids"]:
                self.cancel_job(self.runner.jobs[ident], payload["reason"])
        return {"run_state": "finalizing", "new_experiments_allowed": False,
                "published_champion_present": self.store.champion() is not None,
                "active_job_ids": [job.job_id for job in self.runner.active()]}

    def confirm_finalists(self):
        from agent.confirmation import finalize
        self.begin_finalization()
        try:
            finalize(self)
        except (AgentError, OSError, ValueError) as error:
            self.log.exception('validation.confirmation_failed', error, retained_output=bool(self.store.champion()))
        finally:
            self.budget.confirmation_seconds = 0.

    def run(self):
        if self.graph:
            from agent.graph.controller import run
            return run(self)
        self.store.set_run_state("baseline")
        self.run_training([self.baseline_plan()])
        self.finalize_best()
        self.submission.require_output()
        self.store.set_run_state("searching")
        accepted = 0
        choices = self.catalog.warmup_choices()[:self.config.catalog_warmup_experiments]
        for start in range(0, len(choices), self.config.max_workers):
            if not self.budget.snapshot()["search_allowed"]:
                break
            batch = choices[start:start+self.config.max_workers]
            self.log.event("search.catalog_warmup", configurations=[{"recipe_id": key, "hyperparameters": params} for key, params in batch])
            accepted += self.run_training([self.warmup_plan(key, params) for key, params in batch])
            self.finalize_best()
        stop_reason, round_no = "time_reserve", 0
        self.budget.start_data_research()
        while self.budget.snapshot()["search_allowed"]:
            if self.science:self.science.diagnostics.poll()
            self.log.event("search.round_started", round_no=round_no+1, attempted_experiments=accepted,
                           budget=self.budget.snapshot(),
                           token_budget=self.budget.token_snapshot())
            self.context.round_no = round_no
            round_no += 1
            snapshot = self.call("C07", {"task_spec_id": self.spec["task_spec_id"], "validation_id": self.validation["validation_id"],
                                         "limit_recent": 6, "limit_best": 3}, lambda p: self.store.snapshot(p, self.artifacts))
            requests, plans = [], []
            for index in range(self.config.max_workers):
                researcher_id = "researcher-" + str(index + 1)
                base_id = self.store.best()
                base = self.store.get(base_id)["plan"] if base_id else None
                authorized = self.local_search.has_series(researcher_id, 'legacy')
                api_available = self.llm_available(len(requests))
                if authorized or not api_available:
                    if not authorized and not api_available:
                        self.set_fallback(self.llm_unavailable_reason(len(requests)))
                    plan = self.local_plan(researcher_id, base=base)
                    if plan:
                        plans.append(plan)
                        continue
                    if self.config.research_mode != 'adaptive' or not self.llm_available(len(requests)):
                        continue
                assignment = "exploit_incumbent" if index == 0 else "explore_alternative"
                self.set_fallback()
                context = self.call("C08", {"task_spec_id": self.spec["task_spec_id"], "snapshot_id": snapshot["snapshot_id"],
                                            "tools_catalog_version": "tools-v1", "researcher_id": researcher_id,
                                            "assignment": assignment, "max_input_tokens": self.config.max_context_tokens}, lambda p: self.context.build(p, snapshot))
                requests.append({"researcher_id": researcher_id, "context_id": context["context_id"], "assignment": assignment,
                                 "base_experiment_ids": [snapshot["best_evaluated_experiment_id"]] if snapshot["best_evaluated_experiment_id"] else [],
                                 "forbidden_fingerprints": snapshot["reserved_fingerprints"], "budget": self.budget.snapshot(), "proposal_limit": 1})
            retry_research = False
            outcomes = []
            calls_before = self.budget.calls
            with ThreadPoolExecutor(max_workers=self.config.max_workers, thread_name_prefix="researcher") as pool:
                futures = {pool.submit(self.call, "C09", request, self.researcher.propose): request for request in requests}
                for future in as_completed(futures):
                    try:
                        reply = future.result()
                        outcomes.append(reply['decision'])
                        if reply["decision"] == "stop" and self.config.research_mode != 'adaptive':
                            best = self.store.best()
                            plan = self.local_plan(reply["researcher_id"], base=self.store.get(best)["plan"] if best else None)
                            if plan:
                                plans.append(plan)
                        if reply["decision"] == "propose":
                            plan = reply["plan"]
                            if self.config.research_mode == 'adaptive':
                                plan = self.prepare_adaptive_proposal(plan)
                                retry_research |= self.llm_available()
                            elif self.local_search.repeated(plan) or not self.budget.can_train(plan)["allowed"]:
                                plan = self.local_plan(reply["researcher_id"], base=plan)
                            else:
                                self.authorize_research(plan)
                            if plan:
                                plans.append(plan)
                    except AgentError as exc:
                        self.degraded = True
                        self.log.exception("research.proposal_failed", exc)
                        if self.config.research_mode == 'adaptive':
                            outcomes.append('failed')
                            self.record_research_feedback(futures[future]['researcher_id'], 'legacy', 'proposal_failed', str(exc), code=exc.code)
                            can_reason = self.llm_available()
                            retry_research |= self.budget.calls > calls_before and can_reason
                            if can_reason:
                                continue
                        best = self.store.best()
                        plan = self.local_plan("local-replacement", base=self.store.get(best)["plan"] if best else None)
                        if plan:
                            plans.append(plan)
            if not plans:
                if retry_research and self.budget.snapshot()['search_allowed']:
                    continue
                stop_reason = ('researchers_stopped' if outcomes and all(o == 'stop' for o in outcomes) else
                               'research_unavailable' if 'failed' in outcomes else 'no_affordable_candidates')
                break
            try:
                accepted += self.run_training(plans)
                self.finalize_best()
            except AgentError as exc:
                self.degraded = True
                self.log.exception("research.round_failed", exc)
                if not self.budget.snapshot()["search_allowed"]:
                    break
        self.set_fallback()
        self.budget.end_data_research(self.llm_unavailable_reason() or stop_reason)
        self.confirm_finalists()
        self.finalize_best()
        self.log.event("search.finished", stop_reason=stop_reason, attempted_experiments=accepted,
                       data_research=self.budget.data_research_snapshot(),
                       fallback_seconds=self.fallback_seconds, budget=self.budget.snapshot(), token_budget=self.budget.token_snapshot())
        self.call("C22", {"action": "stop_search", "reason": stop_reason,
                           "wall_seconds_remaining": int(self.budget.remaining()), "cancel_job_ids": []}, self.control)
        self.cleanup()
        return self.submission.require_output()

    def cleanup(self):
        active = self.runner.active()
        for job in active:
            try:
                self.cancel_job(job, "Runtime cleanup")
            except AgentError as exc:
                self.log.exception("job.cleanup_failed", exc, job_id=job.job_id)
                raise
        if active:
            self.wait_jobs(active)
        optimizer = getattr(getattr(self, 'local_search', None), 'optuna', None)
        if optimizer is not None:
            optimizer.sync()
            optimizer.clear_all('runtime_cleanup')
