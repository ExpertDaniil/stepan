from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from copy import deepcopy
import json
import time
import uuid

from agent.catalog import factory_source
from agent.contracts import canonical
from agent.errors import AgentError
from tools.code import write_code


def lineage_for(work):
    donors = work["allowed_donor_experiment_ids"] if work["action"] == "transfer" or work["constraints"]["required_capability_ids"] else []
    return {"base_experiment_id": work["base_experiment_id"], "idea_donor_experiment_ids": donors,
            "ensemble_member_experiment_ids": [], "source_idea_ids": []}


def issue(controller, branch_id, peer, action="improve", donors=None):
    optimizer = getattr(getattr(controller, 'local_search', None), 'optuna', None)
    if optimizer is not None:
        # Finalize an exhausted authorization before G03 builds the next context.
        optimizer.has_series(peer, branch_id)
    graph = controller.graph;branch = graph.branch(branch_id)
    return graph.invoke("G03", {"branch_id": branch_id, "expected_branch_revision": branch["branch_revision"], "researcher_id": peer,
                               "action": action, "allowed_donor_experiment_ids": donors or [], "tools_catalog_version": "tools-v1",
                               # C10 owns the response timeout. The assignment also
                               # covers dispatcher/training queue time after receipt.
                               "ttl_seconds": max(1, int(controller.budget.remaining()))}, key="assign/"+uuid.uuid4().hex)


def bind_trusted(plan, ticket):
    plan = deepcopy(plan);work = ticket["work_order"]
    plan["researcher_id"] = work["researcher_id"]
    lineage = lineage_for(work)
    plan["parent_ids"] = list(dict.fromkeys(([lineage["base_experiment_id"]] if lineage["base_experiment_id"] else [])+lineage["idea_donor_experiment_ids"]))
    return plan


def stop_ticket(controller, ticket, reason):
    work = ticket["work_order"]
    proposal = {"proposal_id": "stop-" + uuid.uuid4().hex[:16], "researcher_id": work["researcher_id"],
                "decision": "stop", "plan": None, "stop_reason": reason}
    controller.graph.invoke("G04", {"work_order_id": work["work_order_id"], "proposal": proposal,
                                  "manifest": None, "lineage": None}, key="stop/" + work["work_order_id"])


def local_for_ticket(controller, ticket):
    work = ticket["work_order"]
    base = controller.store.get(work["base_experiment_id"])["plan"] if work["base_experiment_id"] else None
    plan = controller.local_plan(work["researcher_id"], work["branch_id"], base, work["constraints"],
                                 ticket["researcher_request"]["payload"]["assignment"])
    ticket.pop("_proposal", None)
    return bind_trusted(plan, ticket) if plan else None


def feature_seed_for_ticket(controller, ticket):
    engine = getattr(controller, 'feature_search', None)
    if engine is None:
        return None
    work = ticket['work_order']
    plan = engine.next_plan(work['researcher_id'], work['branch_id'], work['constraints'])
    ticket.pop('_proposal', None)
    return bind_trusted(plan, ticket) if plan else None


def publish_activity(controller, ticket, *, started):
    coordinator = getattr(controller, 'coordination', None)
    if coordinator is not None:
        (coordinator.dispatched if started else coordinator.released)(ticket)
        controller.publish_research_updates()


def cancel_queued(controller, experiment_id, reason):
    with controller.store.transaction():
        controller.graph.db.execute("UPDATE experiments SET state='cancelled',error=? WHERE experiment_id=? AND state='queued'", (reason, experiment_id))
        controller.graph.terminal_tx(experiment_id, "cancelled", reason, "C22")


def acquire_training_lease(controller, eid, resources):
    """A diagnostic lease is temporary contention, not a failed candidate."""
    logged = False
    while True:
        # A denied C13 response is idempotently cached. Each retry needs a new
        # request key, otherwise a released diagnostic can never unblock it.
        allocation = controller.call("C13", {"operation":"acquire", "owner_id":eid,
            "phase":"train", "requested":resources, "lease_id":None, "fencing_token":None},
            controller.budget.lease, key="acquire/"+eid+"/train/"+uuid.uuid4().hex)
        if allocation["granted"]:
            return allocation
        diagnostics = getattr(getattr(controller, "science", None), "diagnostics", None)
        if not diagnostics or not diagnostics.running:
            return allocation
        if not logged:
            controller.log.event("graph.train_deferred", experiment_id=eid, reason="fit_oof_resources")
            logged = True
        while diagnostics.running:
            diagnostics.poll()
            time.sleep(.05)


def train(controller, plans, tickets):
    graph = controller.graph
    if tickets is None or len(plans) != len(tickets):
        raise AgentError("WORK_ORDER_REQUIRED", "Every graph training plan must keep its explicit assignment ticket")
    accepted, queue, entries = 0, {}, []
    for plan, ticket in zip(plans, tickets):
        work = ticket["work_order"]
        proposal = ticket.get("_proposal") or {"proposal_id": "trusted-"+uuid.uuid4().hex[:16], "researcher_id": work["researcher_id"], "plan": plan, "decision": "propose", "stop_reason": None}
        try:
            reply = graph.invoke("G04", {"work_order_id": work["work_order_id"], "proposal": proposal, "manifest": graph.constructor.build(plan), "lineage": lineage_for(work)}, key="submit/"+work["work_order_id"])
            optimizer = getattr(getattr(controller, 'local_search', None), 'optuna', None)
            if optimizer is not None:
                optimizer.bind(plan, reply['lead_submission']['experiment_id'], work['work_order_id'])
            entries.append((ticket, reply))
            current = controller.store.get(reply["lead_submission"]["experiment_id"])
            if current["state"] == "evaluated":
                evaluation = controller.store.experiment_evaluation(current["experiment_id"])
                graph.invoke("G05", {"evaluation_id": evaluation["evaluation_id"], "experiment_id": current["experiment_id"], "selection_policy_id": "same-protocol-primary-v1"}, key="cached-eval/"+work["work_order_id"])
                graph.finish_pilot(work["work_order_id"], reply["lead_submission"]["experiment_id"])
            elif current["state"] == "queued":
                queue.setdefault(reply["lead_submission"]["experiment_id"], plan)
            elif current["state"] in {"failed", "invalid", "timed_out", "cancelled"}:
                graph.finish_pilot(work["work_order_id"], reply["lead_submission"]["experiment_id"])
        except AgentError as exc:
            optimizer = getattr(getattr(controller, 'local_search', None), 'optuna', None)
            if optimizer is not None:
                optimizer.abandon(plan, str(exc))
            graph.abandon(work["work_order_id"], str(exc))
            if exc.code in {"SEARCH_CLOSED", "WORK_ORDER_EXPIRED"} and not controller.budget.snapshot()["search_allowed"]:
                controller.log.event("search.time_reserve_reached", work_order_id=work["work_order_id"], reason=str(exc))
                continue
            controller.degraded = True
            controller.log.exception("graph.proposal_rejected", exc, work_order_id=work["work_order_id"], branch_id=work["branch_id"])
    if hasattr(controller, 'publish_research_updates'):
        controller.publish_research_updates()
    pending = list(queue.items())
    width = graph.policy["max_train_workers"]
    for offset in range(0, len(pending), width):
        jobs = []
        for eid, plan in pending[offset:offset+width]:
            if not controller.admit_training(plan, eid):
                cancel_queued(controller, eid, "Predicted training and finalization exceed remaining time")
                continue
            lease = None
            try:
                for art_id in (controller.spec["train_artifact_id"], controller.validation["train_rows_artifact_id"], controller.validation["valid_rows_artifact_id"]):
                    graph.files.resolve(art_id, True)
                allocation = acquire_training_lease(controller, eid, plan["resources"])
                if not allocation["granted"]:
                    raise AgentError("RESOURCE_BUSY", "Graph train limit cannot acquire an execution slot")
                lease = allocation["lease"]
                attempt = controller.store.start(eid, lease)
                accepted += 1
                args = {"experiment_id": eid, "attempt_no": attempt, "plan": plan, "lease": lease,
                        "train_rows_artifact_id": controller.validation["train_rows_artifact_id"], "valid_rows_artifact_id": controller.validation["valid_rows_artifact_id"],
                        "output_kinds": ["validation_predictions", "fitted_pipeline", "execution_log"]}
                result = controller.call("C14", args, lambda value, exp_id=eid: controller.runner.start_train(value, controller.spec, controller.store.get(exp_id)), key=eid+"/"+str(attempt))
                job = controller.runner.jobs[result["job_id"]]
                graph.track_job(job)
                if hasattr(controller, 'publish_research_updates'):
                    controller.publish_research_updates()
                jobs.append(job)
            except Exception as exc:
                if isinstance(exc, AgentError) and exc.code == "BUDGET_EXHAUSTED" and lease is None:
                    cancel_queued(controller, eid, str(exc))
                    controller.log.event("search.time_reserve_reached", experiment_id=eid, reason=str(exc))
                    continue
                controller.store.fail_start(eid, exc)
                controller.degraded = True
                controller.log.exception("graph.train_start_failed", exc, experiment_id=eid)
                if lease and not any(j.lease["lease_id"] == lease["lease_id"] for j in controller.runner.jobs.values()):
                    controller.call("C13", {"operation": "release", "owner_id": eid, "phase": "train", "requested": plan["resources"], "lease_id": lease["lease_id"], "fencing_token": lease["fencing_token"]}, controller.budget.lease)
        events = controller.wait_jobs(jobs)
        for job in jobs:
            event = events[job.job_id]
            if event["state"] != "succeeded":
                from agent.orchestrator import budget_trial_timeout
                if budget_trial_timeout(event, job, controller.store.get(job.experiment_id)['plan']) and controller.store.champion():
                    controller.log.event('search.budget_trial_timed_out', experiment_id=job.experiment_id, retained_output=True)
                else:
                    controller.degraded = True
            else:
                prediction = next((a for a in event["output_artifact_ids"] if controller.artifacts.meta(a)["kind"] == "validation_predictions"), None)
                try:
                    if not prediction:
                        raise AgentError("INVALID_PREDICTIONS", "Training produced no validation predictions")
                    graph.files.resolve(prediction, True)
                    graph.files.resolve(controller.spec["train_artifact_id"], True)
                    ev = controller.call("C16", {"experiment_id": job.experiment_id, "task_spec_id": controller.spec["task_spec_id"],
                        "dataset_id": controller.spec["dataset_id"], "validation_id": controller.validation["validation_id"], "predictions_artifact_id": prediction, "fidelity": "full"}, controller.evaluator.evaluate)["evaluation"]
                except AgentError as exc:
                    controller.store.invalidate(job.experiment_id, exc)
                    controller.degraded = True
                    controller.log.exception("experiment.rejected", exc, experiment_id=job.experiment_id)
                else:
                    graph.invoke("G05", {"evaluation_id": ev["evaluation_id"], "experiment_id": job.experiment_id, "selection_policy_id": "same-protocol-primary-v1"}, key="eval/"+ev["evaluation_id"])
                    controller.update_final_reserve()
            # Component reasoning never changes the validity of a completed ML evaluation.
            try:
                for ticket, reply in entries:
                    if reply["lead_submission"]["experiment_id"] == job.experiment_id:
                        graph.finish_pilot(ticket["work_order"]["work_order_id"], job.experiment_id)
                if controller.store.get(job.experiment_id)["state"] == "evaluated":
                    graph.offer_component(job.experiment_id)
            except AgentError as exc:
                controller.degraded = True
                controller.log.exception("graph.component_gate_failed", exc, experiment_id=job.experiment_id)
        # Publish a measured improvement before waiting for another peer/fit.
        # wait_jobs has already released this batch's process leases.
        if jobs:
            controller.finalize_best()
    return accepted


def transfer_ticket(controller, decision):
    graph = controller.graph;p = decision["parameters"]
    branch = graph.branch(p["branch_id"])
    acceptance = None
    if branch["decision"] == "accept" and branch["decision_idea_id"]:
        idea_row = graph.db.execute("SELECT record_json FROM g_ideas WHERE run_id=? AND idea_id=?", (graph.run_id, branch["decision_idea_id"])).fetchone()
        idea = json.loads(idea_row[0]) if idea_row else None
        if idea and idea["source_experiment_id"] == p["source_experiment_id"]:
            acceptance = idea
    offer_row = graph.db.execute("SELECT record_json FROM g_idea_candidates WHERE run_id=? AND source_experiment_id=?", (graph.run_id, p["source_experiment_id"])).fetchone()
    if not offer_row and not acceptance:
        raise AgentError("ENTITY_NOT_FOUND", "Transfer candidate")
    offer = acceptance or json.loads(offer_row[0])
    ticket = issue(controller, p["branch_id"], p["researcher_id"], "transfer", [p["source_experiment_id"]])
    work = ticket["work_order"];base = deepcopy(controller.store.get(work["base_experiment_id"])["plan"])
    patch = controller.artifacts.read_json(offer["transfer_patch_artifact_id"] if acceptance else offer["patch_artifact_id"])
    params = graph.constructor.patch_params(base["recipe_id"], base["model"]["hyperparameters"], patch)
    code = controller.call("C11", {"researcher_id": work["researcher_id"], "filename": "transfer.py", "content": factory_source(base["recipe_id"]), "declared_entrypoint": "make_pipeline"}, lambda args: write_code(args, controller.artifacts))["artifact"]
    base.update(plan_id="pilot-"+uuid.uuid4().hex[:16], hypothesis="Controlled recipient transfer: "+",".join(offer["capability_ids"]),
                expected_effect="Measure the component against this peer's own unchanged base.", model={"family": base["model"]["family"], "hyperparameters": params},
                feature_steps=controller.catalog.feature_steps(params), code_artifact_id=code["artifact_id"])
    base = bind_trusted(base, ticket)
    if not acceptance:
        offer.update(state="pilot_running", pilot_work_order_id=work["work_order_id"])
        with controller.store.transaction():
            graph.db.execute("UPDATE g_idea_candidates SET record_json=? WHERE run_id=? AND source_experiment_id=?", (canonical(offer), graph.run_id, p["source_experiment_id"]))
    controller.log.event("research.component_transfer_prepared", branch_id=branch["branch_id"], researcher_id=work["researcher_id"],
                         purpose="acceptance" if acceptance else "pilot", base_experiment_id=work["base_experiment_id"],
                         source_experiment_id=p["source_experiment_id"], work_order_id=work["work_order_id"])
    return base, ticket


def apply_switch(controller, branch_id, decision):
    graph = controller.graph
    if decision["action"] == "FORK":
        result = graph.invoke("G02", decision["parameters"], key="fork/"+decision["decision_id"])
        return result["accept_branch"]["branch_id"]
    target = graph.branch(decision["parameters"]["target_branch_id"])
    current = graph.branch(branch_id)
    graph.invoke("G08", {"branch_id": branch_id, "expected_branch_revision": current["branch_revision"], "action": "pause", "reason": decision["reason"]}, key="backtrack-from/"+decision["decision_id"])
    graph.invoke("G08", {"branch_id": target["branch_id"], "expected_branch_revision": target["branch_revision"], "action": "resume", "reason": decision["reason"]}, key="backtrack-to/"+decision["decision_id"])
    return target["branch_id"]


def warmup_ticket(controller, branch_id, peer, recipe, params):
    graph = controller.graph
    # An already planned warmup can also be the independent recipient pilot.
    # Preserve the peer's own base and exact configuration; no extra fit or API.
    for row in graph.db.execute("SELECT record_json FROM g_idea_candidates WHERE run_id=? ORDER BY rowid", (graph.run_id,)):
        offer = json.loads(row[0])
        if offer["state"] != "pending" or offer["branch_id"] != branch_id:
            continue
        head = graph._compatible_pilot(offer)
        if not head or head["researcher_id"] != peer:
            continue
        base = controller.store.get(head["experiment_id"])["plan"]
        patch = controller.artifacts.read_json(offer["patch_artifact_id"])
        if base["recipe_id"] == recipe and graph.constructor.patch_params(recipe, base["model"]["hyperparameters"], patch) == params:
            return transfer_ticket(controller, {"parameters": {"branch_id": branch_id, "researcher_id": peer,
                                   "source_experiment_id": offer["source_experiment_id"]}})
    ticket = issue(controller, branch_id, peer, "improve")
    return bind_trusted(controller.warmup_plan(recipe, params), ticket), ticket


def advance_authorized_tuning(controller, futures, branch_ids):
    """Dispatcher-only progress while other peers await API replies; no new research action."""
    optimizer = getattr(getattr(controller, 'local_search', None), 'optuna', None)
    if optimizer is None or not controller.budget.snapshot()['search_allowed']:
        return 0, False
    busy = {ticket['work_order']['researcher_id'] for ticket in futures.values()}
    for branch_id, peer in list(optimizer.active):
        if branch_id not in branch_ids or peer in busy or controller.graph.branch(branch_id)['state'] != 'active':
            continue
        if not optimizer.has_series(peer, branch_id):
            continue
        ticket = issue(controller, branch_id, peer)
        try:
            plan = local_for_ticket(controller, ticket)
        except Exception:
            controller.graph.abandon(ticket['work_order']['work_order_id'], 'Authorized tuning continuation could not be prepared')
            raise
        if plan is None:
            stop_ticket(controller, ticket, 'Authorized tuning segment has no remaining affordable candidate')
            continue
        controller.log.event('search.tuning_while_reasoning', branch_id=branch_id, researcher_id=peer,
                             work_order_id=ticket['work_order']['work_order_id'], waiting_peers=sorted(busy))
        return controller.run_training([plan], [ticket]), True
    return 0, False


def research_batch(controller, trusted_plans, trusted_tickets, llm_tickets):
    """Keep all SQL on the dispatcher while overlapping serial training with other peers' API work."""
    graph = controller.graph
    if (hasattr(controller, 'research_stalls') and controller.config.research_mode == 'adaptive'
            and graph.policy['max_train_workers'] == 1):
        from agent.graph.research_scheduler import research_batch as rolling_batch
        return rolling_batch(controller, trusted_plans, trusted_tickets, llm_tickets)
    plans, tickets = list(trusted_plans), list(trusted_tickets)
    accepted = 0
    activity = False
    controller.research_round_outcomes = []
    adaptive = bool(getattr(controller, 'local_search', None) and controller.config.research_mode == 'adaptive')
    calls_before = controller.budget.calls if adaptive else 0
    streaming = graph.policy["max_train_workers"] == 1
    peers = graph.metadata("researcher_ids")
    with ThreadPoolExecutor(max_workers=len(peers), thread_name_prefix="researcher") as pool:
        for ticket in llm_tickets:
            publish_activity(controller, ticket, started=True)
        futures = {pool.submit(controller.call, "C09", ticket["researcher_request"]["payload"], controller.researcher.propose): ticket for ticket in llm_tickets}
        if streaming and plans:
            accepted += controller.run_training(plans, tickets)
            plans, tickets = [], []
        def completed_replies():
            nonlocal accepted, activity
            if not streaming or not adaptive:
                yield from as_completed(futures)
                return
            pending = dict(futures)
            branches = {ticket['work_order']['branch_id'] for ticket in llm_tickets+trusted_tickets}
            while pending:
                if getattr(controller, 'science', None):controller.science.diagnostics.poll()
                ready, _ = wait(pending, timeout=.1, return_when=FIRST_COMPLETED)
                if ready:
                    for future in ready:
                        del pending[future]
                        yield future
                else:
                    count, progressed = advance_authorized_tuning(controller, pending, branches)
                    accepted += count
                    activity |= progressed
        for future in completed_replies():
            ticket = futures[future]
            publish_activity(controller, ticket, started=False)
            try:
                proposal = future.result()
            except Exception as exc:
                phase_budget = isinstance(exc, AgentError) and exc.code == 'SEARCH_PHASE_BUDGET_EXHAUSTED'
                if not phase_budget:
                    controller.degraded = True
                graph.abandon(ticket["work_order"]["work_order_id"], str(exc))
                if phase_budget:
                    controller.log.event('research.phase_budget_reached', work_order_id=ticket['work_order']['work_order_id'])
                else:
                    controller.log.exception("research.proposal_failed", exc, work_order_id=ticket["work_order"]["work_order_id"])
                if adaptive:
                    work = ticket['work_order']
                    controller.research_round_outcomes.append('failed')
                    controller.record_research_feedback(work['researcher_id'], work['branch_id'], 'proposal_failed',
                        str(exc), code=getattr(exc, 'code', 'UNHANDLED_ERROR'))
                    # A completed bounded API dialogue can be followed by a new
                    # decision using the error. Zero-call deterministic failures
                    # must not become an unbounded retry loop.
                    can_reason = controller.llm_available()
                    activity |= (not phase_budget and controller.budget.calls > calls_before and can_reason)
                    if not can_reason and controller.budget.snapshot()['search_allowed']:
                        replacement = issue(controller, work['branch_id'], work['researcher_id'], work['action'], work['allowed_donor_experiment_ids'])
                        plan = local_for_ticket(controller, replacement)
                        if plan is not None:
                            activity = True
                            controller.log.event('search.fallback_started', reason=getattr(exc, 'code', 'UNHANDLED_ERROR'),
                                                 work_order_id=replacement['work_order']['work_order_id'])
                            if streaming:
                                accepted += controller.run_training([plan], [replacement])
                            else:
                                plans.append(plan);tickets.append(replacement)
                        else:
                            stop_ticket(controller, replacement, 'No affordable catalogue fallback after API failure')
                    continue
                if not getattr(controller, "local_search", None) or not controller.budget.snapshot()["search_allowed"]:
                    continue
                work = ticket["work_order"]
                ticket = issue(controller, work["branch_id"], work["researcher_id"], work["action"], work["allowed_donor_experiment_ids"])
                proposal = {"decision": "stop", "plan": None}
            if getattr(controller, "local_search", None):
                plan = proposal.get("plan")
                if proposal['decision'] == 'stop' and controller.config.research_mode == 'adaptive':
                    controller.research_round_outcomes.append('stop')
                    stop_ticket(controller, ticket, proposal.get('stop_reason') or 'No useful affordable research action')
                    continue
                if adaptive:
                    work = ticket['work_order']
                    plan = controller.prepare_adaptive_proposal(plan, work['branch_id'], work['constraints'],
                        ticket['researcher_request']['payload']['assignment'])
                    controller.research_round_outcomes.append('proposed')
                    activity = True
                    if plan is None:
                        stop_ticket(controller, ticket, 'The proposed action has no affordable untried fit; feedback was saved')
                        continue
                    if plan is proposal['plan']:
                        ticket['_proposal'] = proposal
                    else:
                        plan = bind_trusted(plan, ticket)
                    proposal = {**proposal, 'plan': plan}
                else:
                    reason = ("stopped_proposal" if proposal["decision"] == "stop" else
                              "duplicate" if controller.local_search.repeated(plan) else
                              "predicted_duration" if not controller.budget.can_train(plan)["allowed"] else None)
                    if reason:
                        controller.log.event("search.proposal_replaced", work_order_id=ticket["work_order"]["work_order_id"], reason=reason)
                        plan = local_for_ticket(controller, ticket)
                        if plan is None:
                            stop_ticket(controller, ticket, "No affordable untried catalogue candidate")
                            continue
                        proposal = {"decision": "propose", "plan": plan}
                    else:
                        work = ticket["work_order"]
                        controller.authorize_research(plan, work['branch_id'])
                        ticket["_proposal"] = proposal
            elif proposal["decision"] != "stop":
                ticket["_proposal"] = proposal
            if proposal["decision"] == "stop":
                stop_ticket(controller, ticket, proposal.get("stop_reason") or "No proposal")
                continue
            if proposal["decision"] != "stop":
                ticket["_has_plan"] = True
                if streaming:
                    accepted += controller.run_training([proposal["plan"]], [ticket])
                else:
                    plans.append(proposal["plan"]);tickets.append(ticket)
    if plans:
        accepted += controller.run_training(plans, tickets)
    return accepted, bool(activity or accepted or plans or trusted_plans or any("_has_plan" in ticket for ticket in llm_tickets))


def run(controller):
    try:
        return search(controller)
    except AgentError as exc:
        if (exc.code not in {"SEARCH_CLOSED", "WORK_ORDER_EXPIRED"}
                or controller.budget.snapshot()["search_allowed"] or not controller.store.champion()):
            raise
        controller.log.event("search.time_reserve_reached", reason=str(exc))
        return finish(controller, "time_reserve", controller.graph.attempted_research_count())


def search(controller):
    graph = controller.graph;root = graph.metadata("root_branch_id");peers = graph.metadata("researcher_ids")
    controller.store.set_run_state("baseline")
    ticket = issue(controller, root, peers[0], "baseline")
    plan = bind_trusted(controller.baseline_plan(), ticket)
    controller.run_training([plan], [ticket])
    controller.finalize_best();controller.submission.require_output()
    for peer in peers[1:]:
        if not controller.budget.snapshot()["search_allowed"]:
            break
        ticket = issue(controller, root, peer, "baseline")
        controller.run_training([bind_trusted(plan, ticket)], [ticket])
    controller.store.set_run_state("searching")
    accepted, branch_id = 0, root
    warmups = controller.catalog.warmup_choices()[:controller.config.catalog_warmup_experiments]
    specialist_warmups=None
    if controller.config.ensemble_research_enabled:
        from agent.specialists import startup
        specialist_warmups=startup(controller.catalog)
        warmups=[(recipe,params) for _,recipe,params in specialist_warmups]
        # Startup models share the version queue/leases with research. In
        # particular, a slow CPU network must not block all five dialogues.
        from agent.pipeline_search import search as pipeline_search
        return pipeline_search(controller,root,accepted,warmups=specialist_warmups)
    owners = {}
    feature_parameters = {feature["parameter"] for feature in graph.constructor.registry["features"].values()}
    for warmup_index, (recipe, params) in enumerate(warmups):
        if not controller.budget.snapshot()["search_allowed"]:
            break
        owner_key = canonical([recipe, {name: value for name, value in params.items() if name not in feature_parameters}])
        owner = owners.setdefault(owner_key, peers[len(owners) % len(peers)])
        if specialist_warmups:owner=specialist_warmups[warmup_index][0]
        controller.log.event("search.catalog_warmup", configurations=[{"recipe_id": recipe, "hyperparameters": params}], researcher_id=owner)
        plan, ticket = warmup_ticket(controller, root, owner, recipe, params)
        accepted += controller.run_training([plan], [ticket])
        if (warmup_index+1) % len(peers) == 0 or warmup_index+1 == len(warmups):
            controller.finalize_best()
    from agent.pipeline_candidates import enabled
    if enabled(controller.config):
        from agent.pipeline_search import search as pipeline_search
        return pipeline_search(controller, root, accepted)
    stop_reason, round_no = "time_reserve", 0
    controller.budget.start_data_research()
    while controller.budget.snapshot()["search_allowed"]:
        if getattr(controller, 'science', None):controller.science.diagnostics.poll()
        controller.context.round_no = round_no
        round_no += 1
        branch = graph.branch(branch_id)
        assessment = graph.invoke("G07", {"branch_id": branch_id, "expected_branch_revision": branch["branch_revision"]})["assessment"]
        cp = graph.invoke("G01", {"branch_id": branch_id, "expected_branch_revision": branch["branch_revision"], "reason": "manual"})["checkpoint"]
        controller.log.event("search.round_started", round_no=round_no, branch_id=branch_id, attempted_experiments=accepted,
                             phase='refine' if graph.refinement() else 'search',
                             budget=controller.budget.snapshot(), token_budget=controller.budget.token_snapshot())
        decision = graph.invoke("G14", {"checkpoint_id": cp["checkpoint_id"], "budget": controller.budget.snapshot(), "plateau_assessment_id": assessment["assessment_id"]})["decision"]
        if decision["action"] == "FINALIZE":
            stop_reason = decision["reason"];break
        if decision['action'] == 'REFINE_BEST':
            result = graph.invoke('G15', {'source_experiment_id': decision['parameters']['source_experiment_id'],
                'expected_graph_revision': graph.revision(), 'reason': decision['reason']}, key='start-final-refinement')
            branch_id = result['branch']['branch_id'];continue
        if decision["action"] in {"FORK", "BACKTRACK"}:
            branch_id = apply_switch(controller, branch_id, decision);continue
        llm_tickets, trusted_plans, trusted_tickets = [], [], []
        context_failures = []
        slots = graph.research_slots(branch, max(1, controller.budget.research_capacity()))
        for slot in range(slots):
            if slot:
                decision = graph.invoke("G14", {"checkpoint_id": cp["checkpoint_id"], "budget": controller.budget.snapshot(), "plateau_assessment_id": None})["decision"]
            if decision["action"] == "TRANSFER_COMPONENT":
                plan, ticket = transfer_ticket(controller, decision);trusted_plans.append(plan);trusted_tickets.append(ticket)
            elif decision["action"] == "CONTINUE":
                peer = decision["parameters"]["researcher_id"]
                donors = graph.required_donors(branch)
                try:
                    ticket = issue(controller, branch_id, peer, decision["parameters"]["action"], donors)
                except AgentError as exc:
                    if exc.code != 'CONTEXT_TOO_LARGE':
                        raise
                    controller.degraded = True
                    context_failures.append(peer)
                    controller.log.exception('research.context_failed', exc, researcher_id=peer, branch_id=branch_id)
                    controller.record_research_feedback(peer, branch_id, 'context_unavailable',
                        str(exc)+' '+json.dumps(exc.details), code=exc.code)
                    continue
                authorized = controller.local_search.has_series(peer, branch_id)
                api_available = controller.llm_available(len(llm_tickets))
                if authorized or not api_available:
                    if not authorized and not api_available:
                        controller.set_fallback(controller.llm_unavailable_reason(len(llm_tickets)))
                    plan = local_for_ticket(controller, ticket)
                    if plan:
                        trusted_plans.append(plan);trusted_tickets.append(ticket)
                    elif controller.config.research_mode == 'adaptive' and controller.llm_available(len(llm_tickets)):
                        # A sampler may exhaust/reject its remaining points after
                        # admission. That does not exhaust research on the branch.
                        llm_tickets.append(ticket)
                    else:
                        stop_ticket(controller, ticket, "No affordable untried catalogue candidate")
                else:
                    controller.set_fallback()
                    seed = feature_seed_for_ticket(controller, ticket) if llm_tickets else None
                    if seed is not None:
                        trusted_plans.append(seed);trusted_tickets.append(ticket)
                    else:
                        llm_tickets.append(ticket)
            else:
                break
        attempted, had_plans = research_batch(controller, trusted_plans, trusted_tickets, llm_tickets)
        if not had_plans:
            controller.no_fit_branches.add(branch_id)
            outcomes = getattr(controller, 'research_round_outcomes', [])
            reason = ('context_unavailable' if context_failures else
                      'researchers_stopped' if outcomes and all(o == 'stop' for o in outcomes) else
                      'no_new_information' if 'stalled' in outcomes or 'duplicate' in outcomes else
                      'research_unavailable' if 'failed' in outcomes else 'no_affordable_candidates')
            controller.no_fit_reasons[branch_id] = reason
            decision = graph.invoke("G14", {"checkpoint_id": cp["checkpoint_id"], "budget": controller.budget.snapshot(), "plateau_assessment_id": None})["decision"]
            if decision['action'] == 'REFINE_BEST':
                continue
            if decision["action"] == "BACKTRACK":
                branch_id = apply_switch(controller, branch_id, decision)
                continue
            stop_reason = reason;break
        if attempted:
            controller.no_fit_branches.clear()
            controller.no_fit_reasons.clear()
        accepted += attempted
        controller.finalize_best()
    return finish(controller, stop_reason, accepted)


def finish(controller, stop_reason, accepted):
    graph = controller.graph
    controller.budget.end_data_research(controller.llm_unavailable_reason() or stop_reason)
    controller.set_fallback()
    controller.confirm_finalists()
    controller.finalize_best()
    controller.log.event("search.finished", stop_reason=stop_reason, phase='refine' if graph.refinement() else 'search', attempted_experiments=accepted,
                         data_research=controller.budget.data_research_snapshot(),
                         fallback_seconds=controller.fallback_seconds, budget=controller.budget.snapshot(), token_budget=controller.budget.token_snapshot())
    controller.call("C22", {"action": "stop_search", "reason": stop_reason, "wall_seconds_remaining": int(controller.budget.remaining()), "cancel_job_ids": []}, controller.control)
    controller.cleanup()
    graph.invoke("G11", {"task_spec_id": controller.spec["task_spec_id"], "validation_id": controller.validation["validation_id"], "selection_policy_id": "same-protocol-primary-v1", "scope": "final"}, key="final-export")
    return controller.submission.require_output()
