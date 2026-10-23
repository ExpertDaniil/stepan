"""Rolling adaptive dispatch. Only the dispatcher trains, evaluates and writes SQL."""
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import hashlib

from agent.contracts import canonical
from agent.errors import AgentError


def research_batch(controller, trusted_plans, trusted_tickets, llm_tickets):
    from agent.graph.controller import (advance_authorized_tuning, bind_trusted, issue,
                                        local_for_ticket, stop_ticket, feature_seed_for_ticket, publish_activity)
    graph = controller.graph
    controller.research_round_outcomes = []
    accepted, activity = 0, False
    pending = {}
    released = []
    available = {(t['work_order']['branch_id'], t['work_order']['researcher_id']): t['work_order']
                 for t in trusted_tickets}
    pool = ThreadPoolExecutor(max_workers=graph.policy['peer_count'], thread_name_prefix='researcher')

    def generation(branch):
        view = controller.researcher.gateway.feedback.latest(branch)
        return view.get('measurement_generation', view['revision']) if view else None

    def blocked(branch, peer):
        record = controller.research_stalls.get((branch, peer))
        return record is not None and record['generation'] == generation(branch) and record['count'] >= 2

    def repeated(work, signature):
        key = (work['branch_id'], work['researcher_id'])
        old = controller.research_stalls.get(key, {})
        current = generation(work['branch_id'])
        count = old.get('count', 0)+1 if old.get('generation') == current and old.get('signature') == signature else 1
        controller.research_stalls[key] = {'generation': current, 'signature': signature, 'count': count}
        if count >= 2:
            controller.log.event('research.peer_deferred', branch_id=key[0], researcher_id=key[1],
                reason='Repeated action without a new measurement', signature=signature, repetitions=count,
                resume_when='new_controller_feedback', heuristic=True)
        return count >= 2

    def submit(ticket):
        work = ticket['work_order']
        publish_activity(controller, ticket, started=True)
        future = pool.submit(controller.call, 'C09', ticket['researcher_request']['payload'], controller.researcher.propose)
        pending[future] = ticket
        controller.log.event('research.dialogue_dispatched', work_order_id=work['work_order_id'],
                            researcher_id=work['researcher_id'], branch_id=work['branch_id'])

    def refill(work):
        nonlocal accepted, activity
        branch, peer = work['branch_id'], work['researcher_id']
        if (not controller.budget.snapshot()['search_allowed'] or blocked(branch, peer)
                or graph.refinement_due(len(pending))):
            return
        if any(t['work_order']['researcher_id'] == peer for t in pending.values()):
            return
        if any(f.done() for f in pending):
            return
        try:
            ticket = issue(controller, branch, peer, work['action'], work['allowed_donor_experiment_ids'])
        except AgentError as error:
            if error.code != 'CONTEXT_TOO_LARGE':
                raise
            controller.record_research_feedback(peer, branch, 'context_unavailable', str(error), code=error.code)
            controller.research_round_outcomes.append('failed')
            repeated(work, 'failure:CONTEXT_TOO_LARGE');repeated(work, 'failure:CONTEXT_TOO_LARGE')
            controller.log.exception('research.context_failed', error, researcher_id=peer, branch_id=branch)
            return
        if controller.local_search.has_series(peer, branch):
            plan = local_for_ticket(controller, ticket)
            if plan is not None:
                accepted += controller.run_training([plan], [ticket])
                activity = True
                return
        seed = feature_seed_for_ticket(controller, ticket) if pending else None
        if seed is not None:
            accepted += controller.run_training([seed], [ticket])
            activity = True
            return
        if controller.llm_available():
            controller.set_fallback()
            submit(ticket)
            controller.log.event('research.peer_resumed_while_waiting', researcher_id=peer,
                work_order_id=ticket['work_order']['work_order_id'], waiting_peers=[
                    t['work_order']['researcher_id'] for t in pending.values() if t is not ticket])
            return
        reason = controller.llm_unavailable_reason()
        controller.set_fallback(reason)
        plan = local_for_ticket(controller, ticket)
        if plan is not None:
            accepted += controller.run_training([plan], [ticket])
            activity = True
        else:
            stop_ticket(controller, ticket, 'No affordable fallback: '+str(reason))

    try:
        for ticket in llm_tickets:
            work = ticket['work_order']
            if blocked(work['branch_id'], work['researcher_id']):
                stop_ticket(controller, ticket, 'Deferred until new measured feedback; repeated action had no new information')
                controller.research_round_outcomes.append('stalled')
            else:
                submit(ticket)
        if trusted_plans:
            accepted += controller.run_training(trusted_plans, trusted_tickets)
            activity |= bool(accepted)
        while pending:
            if not controller.budget.snapshot()['search_allowed']:
                controller.researcher.gateway.cancelled.set()
                for future, ticket in list(pending.items()):
                    future.cancel()
                    publish_activity(controller, ticket, started=False)
                    graph.abandon(ticket['work_order']['work_order_id'], 'Finalization reserve reached; late result cannot be admitted')
                controller.log.event('research.dialogues_cancelled', count=len(pending), reason='time_reserve')
                break
            ready, _ = wait(pending, timeout=.1, return_when=FIRST_COMPLETED)
            if not ready:
                branches = {t['work_order']['branch_id'] for t in pending.values()}
                branches.update(t['work_order']['branch_id'] for t in trusted_tickets)
                count, progressed = advance_authorized_tuning(controller, pending, branches)
                accepted += count
                activity |= progressed
                for work in list(available.values()):
                    if not controller.local_search.has_series(work['researcher_id'], work['branch_id']):
                        refill(work)
                continue
            for future in ready:
                ticket = pending.pop(future)
                publish_activity(controller, ticket, started=False)
                work = ticket['work_order']
                try:
                    proposal = future.result()
                except Exception as error:
                    code = getattr(error, 'code', 'UNHANDLED_ERROR')
                    graph.abandon(work['work_order_id'], str(error))
                    controller.record_research_feedback(work['researcher_id'], work['branch_id'],
                        'proposal_failed', str(error), code=code)
                    controller.research_round_outcomes.append('failed')
                    if code not in {'SEARCH_PHASE_BUDGET_EXHAUSTED', 'RESEARCH_CANCELLED'}:
                        controller.degraded = True
                        controller.log.exception('research.proposal_failed', error, work_order_id=work['work_order_id'])
                    else:
                        controller.log.event('research.phase_budget_reached', work_order_id=work['work_order_id'], code=code)
                    deferred = repeated(work, 'failure:'+code)
                    if code in {'SEARCH_PHASE_BUDGET_EXHAUSTED', 'RESEARCH_CANCELLED', 'CONTEXT_TOO_LARGE'}:
                        repeated(work, 'failure:'+code)
                    elif not deferred:
                        activity = True
                        released.append(work)
                    continue
                if proposal['decision'] == 'stop':
                    stop_ticket(controller, ticket, proposal.get('stop_reason') or 'Researcher requested stop')
                    controller.research_round_outcomes.append('stop')
                    # A stop is respected until another peer supplies new evidence.
                    repeated(work, 'stop');repeated(work, 'stop')
                    continue
                try:
                    plan = controller.prepare_adaptive_proposal(proposal['plan'], work['branch_id'],
                        work['constraints'], ticket['researcher_request']['payload']['assignment'])
                    if plan is None:
                        stop_ticket(controller, ticket, 'No affordable untried fit in this action; feedback was retained')
                        controller.research_round_outcomes.append('unaffordable')
                        activity |= not repeated(work, 'unaffordable')
                    else:
                        if plan is proposal['plan']:
                            ticket['_proposal'] = proposal
                        else:
                            plan = bind_trusted(plan, ticket)
                        signature = hashlib.sha256(canonical(controller.store.fingerprint_core(plan,
                            controller.artifacts.meta(plan['code_artifact_id'])['sha256'])).encode()).hexdigest()
                        count = controller.run_training([plan], [ticket])
                        accepted += count
                        controller.research_round_outcomes.append('proposed' if count else 'duplicate')
                        if count:
                            activity = True
                            controller.research_stalls.pop((work['branch_id'], work['researcher_id']), None)
                        else:
                            activity |= not repeated(work, signature)
                    released.append(work)
                except AgentError as error:
                    graph.abandon(work['work_order_id'], str(error))
                    controller.record_research_feedback(work['researcher_id'], work['branch_id'], 'action_rejected', str(error), code=error.code)
                    controller.research_round_outcomes.append('failed')
                    activity |= not repeated(work, 'action_rejected:'+error.code)
                    released.append(work)
            for work in released:
                available[(work['branch_id'], work['researcher_id'])] = work
            # Refill a free peer while another dialogue is still pending. When
            # all replies arrived, return to G14 for normal branch/phase control.
            if pending:
                for work in released:
                    refill(work)
            released.clear()
        return accepted, bool(activity or accepted)
    finally:
        pool.shutdown(wait=not pending, cancel_futures=True)
