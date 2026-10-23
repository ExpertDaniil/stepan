from copy import deepcopy
import json
import uuid

from agent.contracts import canonical
from agent.errors import AgentError
from agent.graph.identity import dependency_closure, digest
from agent.store import comparable
from agent.graph.refinement import RefinementPolicy


def utility(evaluation):
    m = evaluation["primary_metric"]
    return m["value"] * (1 if m["direction"] == "maximize" else -1)


class SearchPolicy(RefinementPolicy):
    def fork_victims(self, parent_branch_id):
        live = self.db.execute("SELECT count(*) FROM g_branches WHERE run_id=? AND state IN ('active','paused')", (self.run_id,)).fetchone()[0]
        needed = max(0, live+1-self.policy['max_live_branches'])
        if not needed:
            return []
        candidates = []
        for row in self.db.execute("SELECT branch_id FROM g_branches WHERE run_id=? AND state='paused' AND branch_id<>? ORDER BY created_event_seq", (self.run_id, parent_branch_id)):
            bid = row[0]
            if (self.branch_trial_count(bid) < self.policy['scheduling']['minimum_alternative_trials']
                    and bid not in getattr(self.controller, 'no_fit_branches', set())):
                continue
            scores = [utility(self.store.experiment_evaluation(h['experiment_id'])) for h in self.heads(bid)
                      if h['experiment_id'] and self.store.experiment_evaluation(h['experiment_id'])]
            owns_best = any(h['experiment_id'] == self.store.best() for h in self.heads(bid))
            candidates.append((owns_best, max(scores, default=float('-inf')), bid))
        return [item[2] for item in sorted(candidates)[:needed]] if len(candidates) >= needed else None

    def attempted_research_count(self):
        return self.db.execute("SELECT count(*) FROM experiments WHERE run_id=? AND attempt_no>0 AND json_extract(plan,'$.model.family')<>'dummy'", (self.run_id,)).fetchone()[0]

    def branch_trial_count(self, branch_id):
        return self.db.execute("SELECT count(DISTINCT b.experiment_id) FROM g_bindings b JOIN g_work_orders w USING(run_id,work_order_id) JOIN experiments e USING(run_id,experiment_id) WHERE b.run_id=? AND w.branch_id=? AND (e.attempt_no>0 OR EXISTS(SELECT 1 FROM g_branch_evaluations v WHERE v.run_id=b.run_id AND v.branch_id=w.branch_id AND v.experiment_id=b.experiment_id))", (self.run_id, branch_id)).fetchone()[0]

    def pending_alternatives(self, branch):
        candidates = []
        for row in self.db.execute("SELECT branch_id FROM g_branches WHERE run_id=? AND state='paused' AND decision='reject' ORDER BY rowid DESC", (self.run_id,)):
            alternative = self.branch(row[0])
            deficit = self.policy["scheduling"]["minimum_alternative_trials"] - self.branch_trial_count(alternative["branch_id"])
            if deficit > 0:
                candidates.append((alternative, deficit))
        return sorted(candidates, key=lambda pair: pair[0]["parent_branch_id"] != branch["parent_branch_id"])

    def research_slots(self, branch, remaining):
        if self.refinement():
            return min(self.policy["peer_count"], max(0, remaining))
        reserve = sum(deficit for _, deficit in self.pending_alternatives(branch))
        return min(self.policy["peer_count"], max(0, remaining), max(1, remaining-reserve))

    def _paired_effect(self, before_id, after_id, patch):
        before, after = self.store.get(before_id), self.store.get(after_id)
        a, b = self.store.experiment_evaluation(before_id), self.store.experiment_evaluation(after_id)
        if not a or not b or not a["valid"] or not b["valid"] or a["violations"] or b["violations"] or before["state"] != "evaluated" or after["state"] != "evaluated":
            raise AgentError("INVALID_IDEA_EVIDENCE", "Controlled pairs need two committed trusted evaluations")
        if not comparable(a["primary_metric"], b["primary_metric"]) or a["primary_metric"]["fidelity"] != "full" or before["plan"]["seed"] != after["plan"]["seed"]:
            raise AgentError("INCOMPARABLE_EVALUATIONS", "A transfer pair must use the same full protocol and seed")
        ma = self.manifest(self.experiment_solution(before_id));mb = self.manifest(self.experiment_solution(after_id))
        if ma["verification_level"] != "catalog_verified" or mb["verification_level"] != "catalog_verified":
            raise AgentError("UNVERIFIABLE_CONSTRAINT", "Only the owned constructor can certify a component transfer")
        ca, cb = ma["core"], mb["core"]
        if ca["recipe_id"] != cb["recipe_id"] or ca["code_sha256"] != cb["code_sha256"] or ca["environment_lock_sha256"] != cb["environment_lock_sha256"]:
            raise AgentError("INVALID_COMPONENT_DIFF", "A feature trial cannot also replace the model or environment")
        expected = self.constructor.patch_params(ca["recipe_id"], ca["effective_hyperparameters"], patch)
        if expected != cb["effective_hyperparameters"] or self.catalog.feature_steps(expected) != cb["effective_feature_steps"]:
            raise AgentError("INVALID_COMPONENT_DIFF", "Other hyperparameters changed alongside the claimed feature")
        ac, _ = dependency_closure(ca["components"]);bc, _ = dependency_closure(cb["components"])
        added, removed = set(bc)-set(ac), set(ac)-set(bc)
        if added != set(patch["expected_added_capability_ids"]) or removed != set(patch["expected_removed_capability_ids"]) or not set(patch["required_base_capability_ids"]) <= set(ac):
            raise AgentError("INVALID_COMPONENT_DIFF", "Patch capabilities disagree with the actual dependency closure")
        return {"baseline_experiment_id": before_id, "candidate_experiment_id": after_id,
                "baseline_evaluation_id": a["evaluation_id"], "candidate_evaluation_id": b["evaluation_id"],
                "seed": before["plan"]["seed"], "delta_utility": utility(b)-utility(a)}, ma, mb

    def register_idea(self, p, key):
        self.files.pin(p["transfer_patch_artifact_id"])
        patch = self.artifacts.read_json(p["transfer_patch_artifact_id"])
        self.schemas.validate(patch, "GTransferPatch")
        caps = p["capability_ids"]
        if len(set(caps)) != len(caps) or set(caps) != set(patch["expected_added_capability_ids"]) or patch["expected_removed_capability_ids"] or not set(caps) <= set(self.constructor.registry["features"]):
            raise AgentError("INVALID_COMPONENT_DIFF", "Graph v1 promotes portable registered feature additions")
        for operation in patch["operations"]:
            if operation["operation"] != "add_component" or operation["target_component_id"] is not None or operation["component"] is None or operation["component"]["capability_id"] not in caps:
                raise AgentError("INVALID_COMPONENT_DIFF", "Only explicit feature additions may be promoted")
        cp = self.checkpoint_record(p["fork_checkpoint_id"])
        self._check_preidea(cp, caps)
        self.branch(cp["branch_id"], cp["branch_revision"])
        if len(p["baseline_experiment_ids"]) != len(p["candidate_experiment_ids"]) or p["source_experiment_id"] not in p["candidate_experiment_ids"]:
            raise AgentError("INVALID_IDEA_EVIDENCE", "Baseline/candidate lists must form controlled pairs including the source")
        source_peer = self.store.get(p["source_experiment_id"])["plan"]["researcher_id"]
        effects, trials = [], []
        for before_id, after_id in zip(p["baseline_experiment_ids"], p["candidate_experiment_ids"]):
            effect, ma, mb = self._paired_effect(before_id, after_id, patch)
            effects.append(effect)
            for cap in caps:
                c = next(c for c in mb["core"]["components"] if c["capability_id"] == cap)
                declared = next(op["component"] for op in patch["operations"] if op["component"]["capability_id"] == cap)
                if self.constructor.external_component(c) != declared:
                    raise AgentError("INVALID_COMPONENT_DIFF", "Transfer component differs from the executed source component")
                trial = self._trial(c["component_id"], source_peer, effect, ma, mb, None)
                trials.append(trial)
        # A checkpoint after a source submission is not a pre-idea checkpoint for that line.
        first = self.db.execute("SELECT MIN(e.event_seq) FROM g_events e JOIN g_bindings b ON json_extract(e.record_json,'$.entity_id')=b.work_order_id AND e.run_id=b.run_id JOIN g_work_orders w ON w.run_id=b.run_id AND w.work_order_id=b.work_order_id WHERE b.run_id=? AND b.experiment_id=? AND w.branch_id=? AND json_extract(e.record_json,'$.operation')='G04'", (self.run_id, p["source_experiment_id"], cp["branch_id"])).fetchone()[0]
        if first is not None and cp["captured_event_seq"] >= first:
            raise AgentError("INVALID_FORK_CHECKPOINT", "Checkpoint was captured after source submission")
        pilot_peers, pilot_effects = set(), []
        for after_id in dict.fromkeys(p["transfer_pilot_experiment_ids"]):
            bindings = self.db.execute("SELECT w.order_json FROM g_bindings b JOIN g_work_orders w USING(run_id,work_order_id) WHERE b.run_id=? AND b.experiment_id=? AND w.researcher_id<>? AND w.order_json IS NOT NULL ORDER BY b.rowid", (self.run_id, after_id, source_peer)).fetchall()
            valid = None
            for row in bindings:
                order = json.loads(row[0])
                if order["base_experiment_id"] and order["branch_id"] == cp["branch_id"] and p["source_experiment_id"] in order["allowed_donor_experiment_ids"]:
                    try:
                        effect, ma, mb = self._paired_effect(order["base_experiment_id"], after_id, patch)
                        valid = effect, ma, mb, order
                        break
                    except AgentError:
                        continue
            if not valid:
                raise AgentError("INVALID_IDEA_EVIDENCE", "Pilot is not a controlled transfer on another assigned peer's own base")
            effect, ma, mb, order = valid
            pilot_peers.add(order["researcher_id"]);pilot_effects.append(effect)
            for cap in caps:
                c = next(c for c in mb["core"]["components"] if c["capability_id"] == cap)
                trials.append(self._trial(c["component_id"], order["researcher_id"], effect, ma, mb, trials[0]["component_trial_id"]))
        reasons = []
        promotion = self.policy["promotion"]
        if len({e["seed"] for e in effects}) < promotion["min_paired_seeds"]:
            reasons.append("insufficient_paired_seeds")
        mean_gain = sum(e["delta_utility"] for e in effects)/len(effects)
        if mean_gain <= 0 or mean_gain < promotion["min_effect"]:
            reasons.append("source_effect_below_gate")
        if promotion["require_transfer_pilot"] and not pilot_peers:
            reasons.append("transfer_pilot_required")
        if any(e["delta_utility"] <= 0 or e["delta_utility"] < promotion["min_effect"] for e in pilot_effects):
            reasons.append("recipient_effect_below_gate")
        for trial in trials:
            self.files.put_json(trial, "component_trial")
        idea = {"idea_id": patch["idea_id"], "title": p["title"], "hypothesis": p["hypothesis"], "source_experiment_id": p["source_experiment_id"],
                "capability_ids": list(caps), "transfer_patch_artifact_id": p["transfer_patch_artifact_id"], "fork_checkpoint_id": cp["checkpoint_id"],
                "effects": effects, "transfer_pilot_experiment_ids": list(dict.fromkeys(p["transfer_pilot_experiment_ids"])),
                "status": "provisional" if reasons else "confirmed", "gate_reasons": reasons, "created_event_seq": self.next_seq(),
                "component_trial_ids": list(dict.fromkeys(t["component_trial_id"] for t in trials))}
        def mutate():
            self.branch(cp["branch_id"], cp["branch_revision"])
            idea["created_event_seq"] = self.next_seq()
            if self.db.execute("SELECT 1 FROM g_ideas WHERE run_id=? AND idea_id=?", (self.run_id, idea["idea_id"])).fetchone():
                raise AgentError("IDEA_IMMUTABLE", "Promotion must create a new idea ID and preserve the provisional record")
            for trial in trials:
                self.db.execute("INSERT OR IGNORE INTO g_component_trials VALUES(?,?,?,?,?,?)", (self.run_id, trial["component_trial_id"], trial["component_id"], trial["before_experiment_id"], trial["after_experiment_id"], canonical(trial)))
            self.db.execute("INSERT INTO g_ideas VALUES(?,?,?,?,?)", (self.run_id, idea["idea_id"], idea["source_experiment_id"], canonical(idea), idea["created_event_seq"]))
            self.event("G06", "idea", idea["idea_id"], "GIdea", idea)
            return {"idea": idea, "graph_revision": 0}
        result = self.commit("G06", p, key, mutate)
        self.files.put_json(result["idea"], "component_idea")
        self.log.event("graph.idea_registered", idea_id=result["idea"]["idea_id"], branch_id=cp["branch_id"],
                       status=result["idea"]["status"], capability_ids=result["idea"]["capability_ids"],
                       source_experiment_id=p["source_experiment_id"], transfer_pilot_experiment_ids=p["transfer_pilot_experiment_ids"],
                       gate_reasons=result["idea"]["gate_reasons"])
        return result

    def _trial(self, component_id, peer_id, effect, before, after, source_trial):
        sha = digest([component_id, peer_id, effect["baseline_experiment_id"], effect["candidate_experiment_id"], source_trial])
        return {"component_trial_id": "trial-"+sha[:24], "component_id": component_id, "researcher_id": peer_id,
                "before_experiment_id": effect["baseline_experiment_id"], "after_experiment_id": effect["candidate_experiment_id"],
                "before_solution_id": before["solution_id"], "after_solution_id": after["solution_id"],
                "before_evaluation_id": effect["baseline_evaluation_id"], "after_evaluation_id": effect["candidate_evaluation_id"],
                "validation_id": self.validation["validation_id"], "protocol_sha256": self.validation["protocol_sha256"],
                "delta_utility": effect["delta_utility"], "outcome": "improved" if effect["delta_utility"] > 0 else ("regressed" if effect["delta_utility"] < 0 else "no_gain"),
                "transfer_source_trial_id": source_trial}

    def assess_plateau(self, p, key):
        branch = self.branch(p["branch_id"], p["expected_branch_revision"])
        refinement = self.refinement()
        final_phase = refinement and refinement['branch_id'] == branch['branch_id']
        records = []
        for row in self.db.execute("SELECT * FROM g_branch_evaluations WHERE run_id=? AND branch_id=? ORDER BY event_seq", (self.run_id, branch["branch_id"])):
            if final_phase and row['event_seq'] <= refinement['started_event_seq']:
                continue
            ev = self.store.evaluation(row["evaluation_id"])
            event = self.db.execute("SELECT record_json FROM g_events WHERE run_id=? AND event_seq=?", (self.run_id, row["event_seq"])).fetchone()
            archived = event and json.loads(event[0])["record"].get("disposition", "").startswith("archived")
            if not archived and ev["valid"] and not ev["violations"] and ev["primary_metric"]["fidelity"] == "full":
                records.append((dict(row), ev))
        params = self.policy["plateau"]
        seed_heads = [json.loads(r[0])["record"]["head"] for r in self.db.execute(
            "SELECT record_json FROM g_events WHERE run_id=? AND json_extract(record_json,'$.record_schema')='GBranchSeed' AND json_extract(record_json,'$.record.branch_id')=? ORDER BY event_seq",
            (self.run_id, branch["branch_id"]))]
        if not seed_heads and branch["fork_checkpoint_id"]:
            seed_heads = self.checkpoint_record(branch["fork_checkpoint_id"])["heads"]
        if final_phase:
            seed_heads = refinement['seed_heads']
        seed = [self.store.experiment_evaluation(h["experiment_id"]) for h in seed_heads if self.head_satisfies(h, branch["constraints"])]
        seen = {h["solution_id"] for h in seed_heads if self.head_satisfies(h, branch["constraints"])}
        distinct = []
        for row, ev in records:
            if row["solution_id"] not in seen:
                seen.add(row["solution_id"]);distinct.append((row, ev))
        window = distinct[-params["window_full_evaluations"]:]
        before = seed + [ev for _, ev in distinct[:-len(window)]] if window else seed
        before_best = max(before, key=utility) if before else None
        all_evals = seed+[ev for _, ev in distinct]
        after_best = max(all_evals, key=utility) if all_evals else None
        def tolerance(value):
            return max(params["absolute_epsilon"], params["relative_epsilon"]*max(abs(value), 1e-12))
        anchor = max((utility(e) for e in seed), default=None)
        best, last_gain, last_gain_seq, streak = anchor, 0, 0, 0
        for row, ev in distinct:
            value = utility(ev)
            if best is None:
                best = anchor = value;last_gain = row["elapsed_ms"];last_gain_seq = row['event_seq']
                continue
            best = max(best, value)
            # Accumulate small improvements against a fixed anchor. Updating the
            # anchor after every tiny gain would hide meaningful gradual progress.
            if best-anchor > tolerance(anchor):
                anchor = best;last_gain = row["elapsed_ms"];last_gain_seq = row['event_seq'];streak = 0
            else:
                streak += 1
        epsilon = tolerance(utility(before_best) if before_best else anchor if anchor is not None else 0)
        active = self.db.execute("SELECT record_json FROM g_events WHERE run_id=? AND json_extract(record_json,'$.entity_id')=? AND json_extract(record_json,'$.record_schema')='GBranch' AND json_extract(record_json,'$.record.state')='active' ORDER BY event_seq DESC LIMIT 1", (self.run_id, branch["branch_id"])).fetchone()
        activated_ms = json.loads(active[0])["elapsed_ms"] if active else 0
        elapsed = max(0, (self.elapsed()-max(last_gain, activated_ms))//1000)
        peer_trials = {peer: 0 for peer in self.metadata('researcher_ids')}
        progress_cut = max(last_gain_seq, json.loads(active[0])['event_seq'] if active else 0,
                           refinement['started_event_seq'] if final_phase else 0)
        for row, _ in distinct:
            if row['event_seq'] <= progress_cut:
                continue
            event = self.db.execute('SELECT record_json FROM g_events WHERE run_id=? AND event_seq=?', (self.run_id, row['event_seq'])).fetchone()
            peer = json.loads(event[0])['record'].get('researcher_id') if event else None
            if peer in peer_trials:
                peer_trials[peer] += 1
        fair_trials = all(count >= params['window_full_evaluations'] for count in peer_trials.values())
        gain = utility(after_best)-utility(before_best) if before_best and after_best else None
        reasons = []
        if not self.budget.snapshot()["search_allowed"]:
            status, action = "budget_stop", "stop_search"
        elif len(window) < params["window_full_evaluations"] or len(distinct) < params["min_distinct_solutions"] or before_best is None:
            status, action = "insufficient_evidence", "continue";reasons.append("need_distinct_full_solutions_and_a_before_reference")
        elif not fair_trials:
            status, action = 'searching', 'continue';reasons.append('each_peer_needs_completed_distinct_trials_since_progress_or_resume')
        elif streak >= params["window_full_evaluations"] and elapsed >= params["min_elapsed_seconds"] and gain <= epsilon:
            status, action = "plateau", "pause_and_resume_sibling"
        else:
            status, action = "searching", "continue"
        if final_phase and status == 'plateau':
            action = 'continue';reasons.append('final_refinement_continues_on_plateau')
        result = {"assessment_id": "plateau-"+uuid.uuid4().hex[:16], "branch_id": branch["branch_id"], "branch_revision": branch["branch_revision"],
                  "status": status, "evaluation_ids": [e["evaluation_id"] for _, e in window], "distinct_solution_count": len(window), "elapsed_seconds": elapsed,
                  "consecutive_no_gain": streak, "required_no_gain": params["window_full_evaluations"],
                  "peer_trials_since_progress": peer_trials,
                  "cumulative_gain_since_reset": best-anchor if best is not None else None,
                  "best_before_evaluation_id": before_best["evaluation_id"] if before_best else None, "best_after_evaluation_id": after_best["evaluation_id"] if after_best else None,
                  "utility_gain": gain, "epsilon": epsilon, "recommended_action": action, "reasons": reasons}
        artifact = self.files.put_json(result, "plateau_report")
        def mutate():
            self.branch(branch["branch_id"], branch["branch_revision"])
            self.db.execute("INSERT INTO g_assessments VALUES(?,?,?,?,?)", (self.run_id, result["assessment_id"], branch["branch_id"], artifact["artifact_id"], canonical(result)))
            self.event("G07", "branch", branch["branch_id"], "GPlateau", result)
            return {"assessment": result, "artifact": artifact}
        result = self.commit("G07", p, key, mutate)
        self.log.event("graph.plateau_assessed", branch_id=branch["branch_id"], status=result["assessment"]["status"],
                       distinct_solution_count=result["assessment"]["distinct_solution_count"],
                       elapsed_seconds=result["assessment"]["elapsed_seconds"], utility_gain=result["assessment"]["utility_gain"], epsilon=epsilon,
                       consecutive_no_gain=streak, required_no_gain=params["window_full_evaluations"], peer_trials_since_progress=peer_trials)
        return result

    def _compatible_pilot(self, offer):
        branch = self.branch(offer["branch_id"])
        if branch["state"] != "active":
            return None
        patch = self.artifacts.read_json(offer["patch_artifact_id"])
        for h in self.heads(branch["branch_id"]):
            if h["researcher_id"] == offer["source_peer_id"] or not h["experiment_id"]:
                continue
            if self.db.execute("SELECT 1 FROM g_work_orders WHERE run_id=? AND researcher_id=? AND state IN ('preparing','issued','submitted')", (self.run_id, h["researcher_id"])).fetchone():
                continue
            base = self.store.get(h["experiment_id"])["plan"]
            try:
                self.constructor.patch_params(base["recipe_id"], base["model"]["hyperparameters"], patch)
                caps, _ = dependency_closure(self.manifest(h["solution_id"])["core"]["components"])
                if not set(caps) & set(offer["capability_ids"]):
                    return h
            except AgentError:
                pass
        return None

    def _acceptance_transfer(self, branch, free):
        if branch["decision"] != "accept" or not branch["decision_idea_id"]:
            return None
        idea = json.loads(self.db.execute("SELECT record_json FROM g_ideas WHERE run_id=? AND idea_id=?", (self.run_id, branch["decision_idea_id"])).fetchone()[0])
        patch = self.artifacts.read_json(idea["transfer_patch_artifact_id"])
        for head in self.heads(branch["branch_id"]):
            if head["researcher_id"] not in free or not head["experiment_id"] or self.head_satisfies(head, branch["constraints"]):
                continue
            attempted = self.db.execute("SELECT 1 FROM g_bindings b JOIN g_work_orders w USING(run_id,work_order_id) WHERE b.run_id=? AND w.branch_id=? AND w.researcher_id=? AND w.state='completed' AND json_extract(w.order_json,'$.action')='transfer' AND EXISTS(SELECT 1 FROM json_each(w.order_json,'$.allowed_donor_experiment_ids') WHERE value=?)", (self.run_id, branch["branch_id"], head["researcher_id"], idea["source_experiment_id"])).fetchone()
            if attempted:
                continue
            base = self.store.get(head["experiment_id"])["plan"]
            try:
                self.constructor.patch_params(base["recipe_id"], base["model"]["hyperparameters"], patch)
            except AgentError:
                continue
            source = self.manifest(self.experiment_solution(idea["source_experiment_id"]))
            components = [c["component_id"] for c in source["core"]["components"] if c["capability_id"] in idea["capability_ids"]]
            return {"branch_id": branch["branch_id"], "expected_branch_revision": branch["branch_revision"],
                    "researcher_id": head["researcher_id"], "source_experiment_id": idea["source_experiment_id"], "component_ids": components}
        return None

    def decide(self, p, key):
        self.expire_work()
        cp = self.checkpoint_record(p["checkpoint_id"])
        branch = self.branch(cp["branch_id"], cp["branch_revision"])
        if branch["state"] != "active":
            raise AgentError("INVALID_BRANCH_STATE", "Decision checkpoint must belong to an active branch")
        decision = {"decision_id": "decision-"+uuid.uuid4().hex[:16], "checkpoint_id": cp["checkpoint_id"], "evidence_ids": [], "reason": "Round-robin work on equal peers", "action": "FINALIZE", "parameters": {"reason": "search_budget_closed"}}
        if p["budget"]["search_allowed"] and self.budget.snapshot()["search_allowed"]:
            final_phase = self.refinement() is not None
            peers = self.metadata("researcher_ids")
            counter = self.db.execute("SELECT value FROM g_metadata WHERE run_id=? AND name='schedule_counter'", (self.run_id,)).fetchone()
            count = json.loads(counter[0]) if counter else 0
            free = [peer for peer in peers if not self.db.execute("SELECT 1 FROM g_work_orders WHERE run_id=? AND researcher_id=? AND state IN ('preparing','issued','submitted')", (self.run_id, peer)).fetchone()]
            if not free:
                raise AgentError("PEER_BUSY", "All equal peers already have work; wait for their existing assignments")
            peer = free[count % len(free)]
            decision.update(action="CONTINUE", parameters={"branch_id": branch["branch_id"], "researcher_id": peer, "action": "counterfactual" if branch["decision"] == "reject" and not final_phase else "improve", "allowed_donor_experiment_ids": []})
            pending = self.db.execute("SELECT count(*) FROM g_work_orders WHERE run_id=? AND state IN ('preparing','issued') AND (order_json IS NULL OR json_extract(order_json,'$.action') NOT IN ('baseline','transfer'))", (self.run_id,)).fetchone()[0]
            phase_reason = self.refinement_due(pending)
            if phase_reason and self.store.best():
                origin = self.best_origin(self.store.best())
                decision.update(action='REFINE_BEST', parameters={'source_experiment_id': self.store.best(), 'source_branch_id': origin['branch_id']},
                                reason=phase_reason, evidence_ids=[self.store.best()])
            if decision['action'] == 'CONTINUE' and not final_phase and p["plateau_assessment_id"]:
                row = self.db.execute("SELECT record_json FROM g_assessments WHERE run_id=? AND assessment_id=?", (self.run_id, p["plateau_assessment_id"])).fetchone()
                if not row:
                    raise AgentError("ENTITY_NOT_FOUND", "Plateau assessment")
                assessment = json.loads(row[0])
                if assessment["branch_id"] != branch["branch_id"] or assessment["branch_revision"] != branch["branch_revision"]:
                    raise AgentError("BRANCH_REVISION_CONFLICT", "Plateau assessment belongs to another branch epoch")
                alternatives = [self.branch(r[0]) for r in self.db.execute("SELECT branch_id FROM g_branches WHERE run_id=? AND state='paused' ORDER BY decision='reject' DESC,rowid DESC", (self.run_id,))]
                if assessment["status"] == "plateau" and alternatives:
                    alt = next((x for x in alternatives if x["parent_branch_id"] == branch["parent_branch_id"]), alternatives[0])
                    decision.update(action="BACKTRACK", parameters={"target_branch_id": alt["branch_id"], "expected_branch_revision": alt["branch_revision"]}, reason="Measured plateau; resume a saved alternative", evidence_ids=[assessment["assessment_id"]])
            attempted = self.attempted_research_count()
            remaining = self.budget.research_capacity()
            if decision["action"] == "CONTINUE" and not final_phase:
                # Share the estimated remaining time with saved alternatives.
                threshold = max(1, int((attempted+remaining)*self.policy["scheduling"]["preferred_branch_share"]))
                current_trials = self.branch_trial_count(branch["branch_id"])
                alternatives = self.pending_alternatives(branch)
                reserve = sum(deficit for _, deficit in alternatives)
                if current_trials and alternatives and ((branch["decision"] == "accept" and attempted >= threshold and current_trials >= self.policy["plateau"]["window_full_evaluations"]) or remaining <= reserve):
                    alt = alternatives[0][0]
                    decision.update(action="BACKTRACK", parameters={"target_branch_id": alt["branch_id"], "expected_branch_revision": alt["branch_revision"]}, reason="Reserve time for a counterfactual trial")
            if decision["action"] == "CONTINUE" and not final_phase and self.config.research_mode != 'adaptive':
                for row in self.db.execute("SELECT record_json FROM g_ideas WHERE run_id=? ORDER BY rowid DESC", (self.run_id,)):
                    idea = json.loads(row[0]);origin = self.checkpoint_record(idea["fork_checkpoint_id"])
                    already = self.db.execute("SELECT 1 FROM g_branches WHERE run_id=? AND decision_idea_id=?", (self.run_id, idea["idea_id"])).fetchone()
                    total = self.db.execute("SELECT count(*) FROM g_branches WHERE run_id=?", (self.run_id,)).fetchone()[0]
                    if idea["status"] == "confirmed" and origin["branch_id"] == branch["branch_id"] and origin["branch_revision"] == branch["branch_revision"] and not already and self.fork_victims(branch['branch_id']) is not None and total+2 <= self.policy["max_total_branches"] and remaining >= 2*self.policy["scheduling"]["minimum_alternative_trials"]:
                        decision.update(action="FORK", parameters={"parent_branch_id": branch["branch_id"], "expected_branch_revision": branch["branch_revision"], "checkpoint_id": idea["fork_checkpoint_id"], "idea_id": idea["idea_id"]}, reason="A controlled component and recipient pilot passed the transfer gate", evidence_ids=[idea["idea_id"]]);break
            if decision["action"] == "CONTINUE" and not final_phase and self.config.research_mode != 'adaptive':
                acceptance = self._acceptance_transfer(branch, free)
                if acceptance:
                    decision.update(action="TRANSFER_COMPONENT", parameters=acceptance,
                                    reason="Apply the confirmed component to this peer's restored base before tuning", evidence_ids=[branch["decision_idea_id"]])
            if decision["action"] == "CONTINUE" and not final_phase and self.config.research_mode != 'adaptive':
                for row in self.db.execute("SELECT record_json FROM g_idea_candidates WHERE run_id=? ORDER BY rowid", (self.run_id,)):
                    offer = json.loads(row[0])
                    if offer["state"] == "pending" and offer["branch_id"] == branch["branch_id"]:
                        pilot = self._compatible_pilot(offer)
                        if pilot:
                            source = self.manifest(self.experiment_solution(offer["source_experiment_id"]))
                            cids = [c["component_id"] for c in source["core"]["components"] if c["capability_id"] in offer["capability_ids"]]
                            decision.update(action="TRANSFER_COMPONENT", parameters={"branch_id": branch["branch_id"], "expected_branch_revision": branch["branch_revision"], "researcher_id": pilot["researcher_id"], "source_experiment_id": offer["source_experiment_id"], "component_ids": cids}, reason="Run one controlled recipient pilot before collective adoption", evidence_ids=[offer["source_experiment_id"]]);break
            if decision["action"] == "CONTINUE":
                # Assignments are issued before the parallel API calls start. Account for them now.
                pending = self.db.execute("SELECT count(*) FROM g_work_orders WHERE run_id=? AND state IN ('preparing','issued') AND (order_json IS NULL OR json_extract(order_json,'$.action') NOT IN ('baseline','transfer'))", (self.run_id,)).fetchone()[0]
                stop_reason = None
                stop_reason = self.budget.llm_unavailable_reason(pending)
                gateway = getattr(getattr(self.controller, "researcher", None), "gateway", None)
                if gateway is not None:
                    try:
                        gateway.check_availability()
                    except AgentError as exc:
                        stop_reason = exc.code.lower()
                if stop_reason:
                    decision.update(reason="Local catalogue search: " + stop_reason)
        if decision['action'] != 'REFINE_BEST' and branch["branch_id"] in getattr(self.controller, "no_fit_branches", set()) and self.budget.snapshot()["search_allowed"]:
            closed = self.controller.no_fit_branches
            alternatives = [self.branch(row[0]) for row in self.db.execute("SELECT branch_id FROM g_branches WHERE run_id=? AND state='paused' ORDER BY rowid DESC", (self.run_id,)) if row[0] not in closed]
            if alternatives and not self.refinement():
                alt = alternatives[0]
                decision.update(action="BACKTRACK", parameters={"target_branch_id": alt["branch_id"], "expected_branch_revision": alt["branch_revision"]}, reason="No affordable candidate in this branch; try a saved alternative")
            else:
                reason = getattr(self.controller, 'no_fit_reasons', {}).get(branch['branch_id'], 'no_affordable_candidates')
                decision.update(action="FINALIZE", parameters={"reason": reason}, reason=reason)
        artifact = self.files.put_json(decision, "search_decision")
        def mutate():
            self.branch(branch["branch_id"], branch["branch_revision"])
            old = self.db.execute("SELECT value FROM g_metadata WHERE run_id=? AND name='schedule_counter'", (self.run_id,)).fetchone()
            if decision['action'] != 'REFINE_BEST':
                self.set_metadata_tx("schedule_counter", (json.loads(old[0]) if old else 0)+1)
            self.db.execute("INSERT INTO g_decisions VALUES(?,?,?,?,?)", (self.run_id, decision["decision_id"], cp["checkpoint_id"], artifact["artifact_id"], canonical(decision)))
            self.event("G14", "branch", branch["branch_id"], "GSearchDecision", decision)
            return {"decision": decision, "decision_artifact_id": artifact["artifact_id"], "graph_revision": 0}
        result = self.commit("G14", p, key, mutate)
        self.log.event("search.decision", decision_id=decision["decision_id"], branch_id=branch["branch_id"],
                       action=decision["action"], reason=decision["reason"], parameters=decision["parameters"], evidence_ids=decision["evidence_ids"])
        return result

    def offer_component(self, experiment_id):
        """Discover only a controlled portable feature change, never infer causality from a mixed improvement."""
        if self.refinement():
            return
        if self.config.research_mode != "catalog" or self.db.execute("SELECT 1 FROM g_idea_candidates WHERE run_id=? AND source_experiment_id=?", (self.run_id, experiment_id)).fetchone():
            return
        exp = self.store.get(experiment_id);ev = self.store.experiment_evaluation(experiment_id)
        if not ev or not ev["valid"]:
            return
        after = self.manifest(self.experiment_solution(experiment_id))
        origin = self.db.execute("SELECT w.order_json FROM g_bindings b JOIN g_work_orders w USING(run_id,work_order_id) WHERE b.run_id=? AND b.experiment_id=? ORDER BY b.rowid LIMIT 1", (self.run_id, experiment_id)).fetchone()
        if not origin:
            return
        order = json.loads(origin[0]);branch = self.branch(order["branch_id"])
        if branch["state"] != "active":
            return
        for cap, feature in self.constructor.registry["features"].items():
            if cap in branch["constraints"]["required_capability_ids"]+branch["constraints"]["forbidden_capability_ids"]:
                continue
            component = next((c for c in after["core"]["components"] if c["capability_id"] == cap), None)
            if component is None:
                continue
            cp = self.checkpoint_record(order["base_team_checkpoint_id"])
            try:
                self._check_preidea(cp, [cap])
            except AgentError:
                continue
            patch = {"patch_version": "transfer-1.0", "idea_id": "idea-"+uuid.uuid4().hex[:16],
                     "supported_recipe_ids": [rid for rid, recipe in self.catalog.recipes.items() if feature["parameter"] in recipe["parameters"]],
                     "required_base_capability_ids": [], "operations": [{"operation": "add_component", "target_component_id": None, "component": self.constructor.external_component(component)}],
                     "expected_added_capability_ids": [cap], "expected_removed_capability_ids": []}
            before_id = None
            for row in self.db.execute("SELECT experiment_id FROM experiments WHERE run_id=? AND state='evaluated' AND experiment_id<>? ORDER BY rowid DESC", (self.run_id, experiment_id)):
                try:
                    effect, _, _ = self._paired_effect(row[0], experiment_id, patch)
                    if effect["delta_utility"] > 0 and effect["delta_utility"] >= self.policy["promotion"]["min_effect"]:
                        before_id = row[0];break
                except AgentError:
                    pass
            if before_id is None:
                continue
            art = self.files.put_json(patch, "transfer_patch")
            payload = {"title": "Controlled "+cap, "hypothesis": exp["plan"]["hypothesis"], "source_experiment_id": experiment_id, "capability_ids": [cap],
                       "fork_checkpoint_id": cp["checkpoint_id"], "baseline_experiment_ids": [before_id], "candidate_experiment_ids": [experiment_id],
                       "transfer_pilot_experiment_ids": [], "transfer_patch_artifact_id": art["artifact_id"]}
            idea = self.invoke("G06", payload, key="offer/"+experiment_id)["idea"]
            offer = {"branch_id": branch["branch_id"], "source_experiment_id": experiment_id, "baseline_experiment_id": before_id,
                     "checkpoint_id": cp["checkpoint_id"], "capability_ids": [cap], "patch_artifact_id": art["artifact_id"],
                     "source_peer_id": exp["plan"]["researcher_id"], "pilot_experiment_ids": [], "state": "confirmed" if idea["status"] == "confirmed" else "pending",
                     "pilot_work_order_id": None, "idea_id": idea["idea_id"]}
            with self.store.transaction():
                self.db.execute("INSERT INTO g_idea_candidates VALUES(?,?,?)", (self.run_id, experiment_id, canonical(offer)))
            return

    def finish_pilot(self, order_id, experiment_id):
        for row in self.db.execute("SELECT source_experiment_id,record_json FROM g_idea_candidates WHERE run_id=?", (self.run_id,)).fetchall():
            offer = json.loads(row[1])
            if offer["pilot_work_order_id"] != order_id or offer["state"] != "pilot_running":
                continue
            ev = self.store.experiment_evaluation(experiment_id)
            if ev:
                patch = self.artifacts.read_json(offer["patch_artifact_id"])
                patch["idea_id"] = "idea-"+uuid.uuid4().hex[:16]
                art = self.files.put_json(patch, "transfer_patch")
                p = {"title": "Transferred "+",".join(offer["capability_ids"]), "hypothesis": "Controlled source and recipient effects on the fixed protocol.",
                     "source_experiment_id": offer["source_experiment_id"], "capability_ids": offer["capability_ids"], "fork_checkpoint_id": offer["checkpoint_id"],
                     "baseline_experiment_ids": [offer["baseline_experiment_id"]], "candidate_experiment_ids": [offer["source_experiment_id"]],
                     "transfer_pilot_experiment_ids": [experiment_id], "transfer_patch_artifact_id": art["artifact_id"]}
                idea = self.invoke("G06", p, key="pilot-result/"+order_id)["idea"]
                offer.update(state="confirmed" if idea["status"] == "confirmed" else "rejected", pilot_experiment_ids=[experiment_id], idea_id=idea["idea_id"], patch_artifact_id=art["artifact_id"])
            else:
                offer["state"] = "pilot_failed"
            with self.store.transaction():
                self.db.execute("UPDATE g_idea_candidates SET record_json=? WHERE run_id=? AND source_experiment_id=?", (canonical(offer), self.run_id, row[0]))
