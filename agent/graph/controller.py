from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import json
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
    graph = controller.graph;branch = graph.branch(branch_id)
    return graph.invoke("G03", {"branch_id": branch_id, "expected_branch_revision": branch["branch_revision"], "researcher_id": peer,
                               "action": action, "allowed_donor_experiment_ids": donors or [], "tools_catalog_version": "tools-v1",
                               "ttl_seconds": max(1, min(controller.config.llm_operation_timeout_seconds+30, int(controller.budget.remaining())))}, key="assign/"+uuid.uuid4().hex)


def bind_trusted(plan, ticket):
    plan = deepcopy(plan);work = ticket["work_order"]
    plan["researcher_id"] = work["researcher_id"]
    lineage = lineage_for(work)
    plan["parent_ids"] = list(dict.fromkeys(([lineage["base_experiment_id"]] if lineage["base_experiment_id"] else [])+lineage["idea_donor_experiment_ids"]))
    return plan


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
            controller.degraded = True
            graph.abandon(work["work_order_id"], str(exc))
            controller.log.exception("graph.proposal_rejected", exc, work_order_id=work["work_order_id"], branch_id=work["branch_id"])
    pending = list(queue.items())
    width = graph.policy["max_train_workers"]
    for offset in range(0, len(pending), width):
        jobs = []
        for eid, plan in pending[offset:offset+width]:
            if not controller.budget.snapshot()["search_allowed"]:
                with controller.store.transaction():
                    graph.db.execute("UPDATE experiments SET state='cancelled',error=? WHERE experiment_id=? AND state='queued'", ("Search time reserve reached before training", eid))
                    graph.terminal_tx(eid, "cancelled", "Search time reserve reached", "C22")
                continue
            lease = None
            try:
                for art_id in (controller.spec["train_artifact_id"], controller.validation["train_rows_artifact_id"], controller.validation["valid_rows_artifact_id"]):
                    graph.files.resolve(art_id, True)
                allocation = controller.lease(eid, "train", plan["resources"])
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
                jobs.append(job)
            except Exception as exc:
                controller.store.fail_start(eid, exc)
                controller.degraded = True
                controller.log.exception("graph.train_start_failed", exc, experiment_id=eid)
                if lease and not any(j.lease["lease_id"] == lease["lease_id"] for j in controller.runner.jobs.values()):
                    controller.call("C13", {"operation": "release", "owner_id": eid, "phase": "train", "requested": plan["resources"], "lease_id": lease["lease_id"], "fencing_token": lease["fencing_token"]}, controller.budget.lease)
        events = controller.wait_jobs(jobs)
        for job in jobs:
            event = events[job.job_id]
            if event["state"] != "succeeded":
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
    live = graph.db.execute("SELECT count(*) FROM g_branches WHERE run_id=? AND state IN ('active','paused')", (graph.run_id,)).fetchone()[0]
    graph.invoke("G08", {"branch_id": branch_id, "expected_branch_revision": current["branch_revision"], "action": "archive" if live >= graph.policy["max_live_branches"] else "pause", "reason": decision["reason"]}, key="backtrack-from/"+decision["decision_id"])
    graph.invoke("G08", {"branch_id": target["branch_id"], "expected_branch_revision": target["branch_revision"], "action": "resume", "reason": decision["reason"]}, key="backtrack-to/"+decision["decision_id"])
    return target["branch_id"]


def research_batch(controller, trusted_plans, trusted_tickets, llm_tickets):
    """Keep all SQL on the dispatcher while overlapping serial training with other peers' API work."""
    graph = controller.graph
    plans, tickets = list(trusted_plans), list(trusted_tickets)
    accepted = 0
    streaming = graph.policy["max_train_workers"] == 1
    peers = graph.metadata("researcher_ids")
    with ThreadPoolExecutor(max_workers=len(peers), thread_name_prefix="researcher") as pool:
        futures = {pool.submit(controller.call, "C09", ticket["researcher_request"]["payload"], controller.researcher.propose): ticket for ticket in llm_tickets}
        if streaming and plans:
            accepted += controller.run_training(plans, tickets)
            plans, tickets = [], []
        for future in as_completed(futures):
            ticket = futures[future]
            try:
                proposal = future.result()
                if proposal["decision"] == "stop":
                    graph.invoke("G04", {"work_order_id": ticket["work_order"]["work_order_id"], "proposal": proposal, "manifest": None, "lineage": None}, key="stop/"+ticket["work_order"]["work_order_id"])
                else:
                    ticket["_proposal"] = proposal
            except Exception as exc:
                controller.degraded = True
                graph.abandon(ticket["work_order"]["work_order_id"], str(exc))
                controller.log.exception("research.proposal_failed", exc, work_order_id=ticket["work_order"]["work_order_id"])
                continue
            if proposal["decision"] != "stop":
                if streaming:
                    accepted += controller.run_training([proposal["plan"]], [ticket])
                else:
                    plans.append(proposal["plan"]);tickets.append(ticket)
    if plans:
        accepted += controller.run_training(plans, tickets)
    return accepted, bool(plans or trusted_plans or any("_proposal" in ticket for ticket in llm_tickets))


def run(controller):
    graph = controller.graph;root = graph.metadata("root_branch_id");peers = graph.metadata("researcher_ids")
    controller.store.set_run_state("baseline")
    ticket = issue(controller, root, peers[0], "baseline")
    plan = bind_trusted(controller.baseline_plan(), ticket)
    controller.run_training([plan], [ticket])
    controller.finalize_best();controller.submission.require_output()
    for peer in peers[1:]:
        ticket = issue(controller, root, peer, "baseline")
        controller.run_training([bind_trusted(plan, ticket)], [ticket])
    controller.store.set_run_state("searching")
    accepted, branch_id = 0, root
    warmups = controller.catalog.warmup_choices()[:min(controller.config.catalog_warmup_experiments, controller.config.max_research_experiments)]
    for offset in range(0, len(warmups), len(peers)):
        if not controller.budget.snapshot()["search_allowed"]:
            break
        choices = warmups[offset:offset+len(peers)]
        controller.log.event("search.catalog_warmup", configurations=[{"recipe_id": r, "hyperparameters": p} for r, p in choices])
        tickets = [issue(controller, root, peers[i], "improve") for i in range(len(choices))]
        plans = [bind_trusted(controller.warmup_plan(recipe, params), ticket) for (recipe, params), ticket in zip(choices, tickets)]
        accepted += controller.run_training(plans, tickets);controller.finalize_best()
    stop_reason = "round_limit"
    for round_no in range(controller.config.max_research_experiments*2+4):
        if accepted >= controller.config.max_research_experiments or not controller.budget.snapshot()["search_allowed"]:
            stop_reason = "experiment_limit" if accepted >= controller.config.max_research_experiments else "time_reserve";break
        controller.context.round_no = round_no
        branch = graph.branch(branch_id)
        assessment = graph.invoke("G07", {"branch_id": branch_id, "expected_branch_revision": branch["branch_revision"]})["assessment"]
        cp = graph.invoke("G01", {"branch_id": branch_id, "expected_branch_revision": branch["branch_revision"], "reason": "manual"})["checkpoint"]
        controller.log.event("search.round_started", round_no=round_no+1, branch_id=branch_id, attempted_experiments=accepted,
                             experiment_limit=controller.config.max_research_experiments, budget=controller.budget.snapshot(), token_budget=controller.budget.token_snapshot())
        decision = graph.invoke("G14", {"checkpoint_id": cp["checkpoint_id"], "budget": controller.budget.snapshot(), "plateau_assessment_id": assessment["assessment_id"]})["decision"]
        if decision["action"] == "FINALIZE":
            stop_reason = decision["reason"];break
        if decision["action"] in {"FORK", "BACKTRACK"}:
            branch_id = apply_switch(controller, branch_id, decision);continue
        llm_tickets, trusted_plans, trusted_tickets = [], [], []
        slots = graph.research_slots(branch, controller.config.max_research_experiments-accepted)
        for slot in range(slots):
            if slot:
                decision = graph.invoke("G14", {"checkpoint_id": cp["checkpoint_id"], "budget": controller.budget.snapshot(), "plateau_assessment_id": None})["decision"]
            if decision["action"] == "TRANSFER_COMPONENT":
                plan, ticket = transfer_ticket(controller, decision);trusted_plans.append(plan);trusted_tickets.append(ticket)
            elif decision["action"] == "CONTINUE":
                peer = decision["parameters"]["researcher_id"]
                donors = []
                if branch["decision_idea_id"] and branch["constraints"]["required_capability_ids"]:
                    idea_row = graph.db.execute("SELECT record_json FROM g_ideas WHERE run_id=? AND idea_id=?", (graph.run_id, branch["decision_idea_id"])).fetchone()
                    donors = [json.loads(idea_row[0])["source_experiment_id"]]
                llm_tickets.append(issue(controller, branch_id, peer, decision["parameters"]["action"], donors))
            else:
                break
        attempted, had_plans = research_batch(controller, trusted_plans, trusted_tickets, llm_tickets)
        if not had_plans:
            stop_reason = "no_valid_proposals";break
        accepted += attempted
        controller.finalize_best()
    controller.log.event("search.finished", stop_reason=stop_reason, attempted_experiments=accepted, experiment_limit=controller.config.max_research_experiments, budget=controller.budget.snapshot())
    controller.call("C22", {"action": "stop_search", "reason": stop_reason, "wall_seconds_remaining": int(controller.budget.remaining()), "cancel_job_ids": []}, controller.control)
    controller.cleanup()
    graph.invoke("G11", {"task_spec_id": controller.spec["task_spec_id"], "validation_id": controller.validation["validation_id"], "selection_policy_id": "same-protocol-primary-v1", "scope": "final"}, key="final-export")
    return controller.submission.require_output()
