from copy import deepcopy
import json
import uuid

from agent.contracts import canonical
from agent.errors import AgentError
from agent.graph.identity import dependency_closure
from agent.research_memory import build_research_memory


class WorkService:
    def required_donors(self, branch):
        """Only explicit ancestor sources compatible with every inherited ban."""
        required = set(branch['constraints']['required_capability_ids'])
        forbidden = set(branch['constraints']['forbidden_capability_ids'])
        donors = []
        current = branch
        while current and required:
            if current['decision_idea_id']:
                row = self.db.execute('SELECT record_json FROM g_ideas WHERE run_id=? AND idea_id=?',
                                      (self.run_id, current['decision_idea_id'])).fetchone()
                idea = json.loads(row[0]) if row else None
                if idea and required.intersection(idea['capability_ids']):
                    eid = idea['source_experiment_id']
                    caps, _ = dependency_closure(self.manifest(self.experiment_solution(eid))['core']['components'])
                    if not forbidden.intersection(caps):
                        donors.append(eid)
                        required.difference_update(caps)
            current = self.branch(current['parent_branch_id']) if current['parent_branch_id'] else None
        return list(dict.fromkeys(donors))

    def lineage_cuts(self, branch_id):
        seen = set()
        cut = self.cut()
        while branch_id:
            if branch_id in seen:
                raise AgentError("INVALID_BRANCH_GRAPH", "Branch ancestry contains a cycle")
            seen.add(branch_id)
            yield branch_id, cut
            branch = self.branch(branch_id)
            if branch["fork_checkpoint_id"]:
                cut = min(cut, self.checkpoint_record(branch["fork_checkpoint_id"])["captured_event_seq"])
            branch_id = branch["parent_branch_id"]

    def visible_history(self, branch_id, researcher_id=None):
        """Only observations made on this line before each ancestor's fork cut."""
        history = []
        for branch_id, cut in self.lineage_cuts(branch_id):
            for row in self.db.execute("SELECT event_seq,record_json FROM g_events WHERE run_id=? AND event_seq<=? AND json_extract(record_json,'$.record_schema') IN ('GHeadResult','GBranchSeed','GRefinement','Error') ORDER BY event_seq DESC", (self.run_id, cut)):
                event = json.loads(row["record_json"]);record = event["record"]
                if event['record_schema'] == 'GRefinement':
                    if record['branch_id'] == branch_id:
                        history.append((row['event_seq'], record['source_experiment_id']))
                elif event["record_schema"] == "GBranchSeed":
                    if record["branch_id"] == branch_id and (not researcher_id or record["head"]["researcher_id"] == researcher_id) and record["head"]["experiment_id"]:
                        history.append((row["event_seq"], record["head"]["experiment_id"]))
                elif event["record_schema"] == "GHeadResult":
                    if record["branch_id"] != branch_id or (researcher_id and record["researcher_id"] != researcher_id):
                        continue
                    history.append((row["event_seq"], record["experiment_id"]))
                elif event["entity_type"] == "work_order":
                    binding = self.db.execute("SELECT b.experiment_id,w.branch_id,w.researcher_id FROM g_bindings b JOIN g_work_orders w USING(run_id,work_order_id) WHERE b.run_id=? AND b.work_order_id=?", (self.run_id, event["entity_id"])).fetchone()
                    if binding and binding["branch_id"] == branch_id and (not researcher_id or binding["researcher_id"] == researcher_id):
                        history.append((row["event_seq"], binding["experiment_id"]))
        return list(dict.fromkeys(eid for _, eid in sorted(history, reverse=True)))

    def rejection_facts(self, branch_id, researcher_id):
        errors = []
        for branch_id, cut in self.lineage_cuts(branch_id):
            rows = self.db.execute("SELECT e.event_seq,e.record_json FROM g_events e JOIN g_work_orders w ON w.run_id=e.run_id AND w.work_order_id=json_extract(e.record_json,'$.entity_id') WHERE e.run_id=? AND e.event_seq<=? AND w.branch_id=? AND w.researcher_id=? AND json_extract(e.record_json,'$.record_schema')='Error' AND NOT EXISTS(SELECT 1 FROM g_bindings b WHERE b.run_id=w.run_id AND b.work_order_id=w.work_order_id) ORDER BY e.event_seq DESC", (self.run_id, cut, branch_id, researcher_id))
            for row in rows:
                event = json.loads(row["record_json"]);error = event["record"]
                summary = canonical({"code": error["code"], "message": error["message"][:self.policy["context"]["max_error_excerpt_bytes"]//4], "stage": "proposal_rejected"})
                errors.append((row["event_seq"], {"kind": "failure", "source_id": event["entity_id"], "summary": summary, "artifact_ids": []}))
        return [fact for _, fact in sorted(errors, key=lambda pair: pair[0], reverse=True)]

    def memory_record(self, experiment_id, head=None):
        exp = self.store.get(experiment_id);plan = exp["plan"]
        ev = self.store.experiment_evaluation(experiment_id)
        trusted = ev and exp["state"] == "evaluated" and ev["valid"] and not ev["violations"]
        error = self.log.redactor.clean(str(exp["error"])) if exp["error"] else None
        diagnostics = {}
        if trusted:
            raw = self.artifacts.read_json(ev["diagnostics_artifact_id"])
            diagnostics = {k: raw[k] for k in ("mae", "absolute_error_p50", "absolute_error_p90", "class_summary", "hard_groups") if k in raw}
        return {"experiment_id": experiment_id, "state": exp["state"], "recipe_id": plan["recipe_id"],
                "hypothesis": plan["hypothesis"][:600], "metric": ev["primary_metric"] if trusted else None,
                "hyperparameters": plan["model"]["hyperparameters"], "feature_steps": plan["feature_steps"],
                "seed": plan["seed"], "is_incumbent": bool(head and experiment_id == head["experiment_id"]),
                "error": error[:self.policy["context"]["max_error_excerpt_bytes"]//4] if error else None,
                "diagnostics": diagnostics}

    def research_assignment(self, branch_id, researcher_id):
        refinement = self.refinement()
        if refinement and refinement['branch_id'] == branch_id:
            return 'exploit_incumbent'
        # Count this peer's completed proposals, not loop iterations spent forking
        # or backtracking. Both equal peers get the same exploit/explore ratio.
        trials = 0
        for line_id, cut in self.lineage_cuts(branch_id):
            trials += self.db.execute("SELECT count(*) FROM g_work_orders w WHERE w.run_id=? AND w.branch_id=? AND w.researcher_id=? AND w.state='completed' AND json_extract(w.order_json,'$.action') IN ('improve','counterfactual') AND EXISTS(SELECT 1 FROM g_proposals p WHERE p.run_id=w.run_id AND p.work_order_id=w.work_order_id AND p.proposal_id NOT LIKE 'trusted-%') AND EXISTS(SELECT 1 FROM g_events e WHERE e.run_id=w.run_id AND e.event_seq<=? AND json_extract(e.record_json,'$.operation')='G04' AND json_extract(e.record_json,'$.entity_id')=w.work_order_id)", (self.run_id, line_id, researcher_id, cut)).fetchone()[0]
        exploit = self.policy["scheduling"].get("exploit_steps", 3)
        phase = (trials + self.metadata("researcher_ids").index(researcher_id)*2) % (exploit+1)
        return "explore_alternative" if phase == exploit else "exploit_incumbent"

    def recent_for_context(self, branch_id, head, donors):
        history = self.visible_history(branch_id, head["researcher_id"])
        ids = list(dict.fromkeys(([head["experiment_id"]] if head["experiment_id"] else []) + list(donors) + history))
        return [self.memory_record(eid, head) for eid in ids[:max(1, self.policy["context"]["max_related_records"])]], len(ids)

    def search_memory(self, branch_id, head, limit=24):
        """Keep old model proposals visible after cheap local trials fill recent facts."""
        ids = self.visible_history(branch_id, None if self.config.research_mode == 'adaptive' else head['researcher_id'])
        base = self.store.get(head['experiment_id'])['plan'] if head['experiment_id'] else None
        proposal_ids = set()
        for line, cut in self.lineage_cuts(branch_id):
            rows = self.db.execute("SELECT b.experiment_id FROM g_proposals p JOIN g_work_orders w USING(run_id,work_order_id) JOIN g_bindings b USING(run_id,work_order_id) WHERE p.run_id=? AND w.branch_id=? AND w.researcher_id=? AND p.proposal_id NOT LIKE 'trusted-%' AND EXISTS(SELECT 1 FROM g_events e WHERE e.run_id=w.run_id AND e.event_seq<=? AND json_extract(e.record_json,'$.entity_id')=w.work_order_id AND json_extract(e.record_json,'$.operation')='G04')",
                                   (self.run_id, line, head['researcher_id'], cut))
            proposal_ids.update(row[0] for row in rows)
        best = {}
        records = {}
        for eid in ids:
            exp = self.store.get(eid);plan = exp['plan']
            ev = self.store.experiment_evaluation(eid)
            valid = ev and ev['valid'] and not ev['violations'] and exp['state'] == 'evaluated'
            params = plan['model']['hyperparameters']
            record = {'experiment_id': eid, 'recipe_id': plan['recipe_id'], 'state': exp['state'],
                      'score': ev['primary_metric']['value'] if valid else None}
            if self.config.research_mode == 'adaptive':
                record['hypothesis'] = plan['hypothesis'][:240]
                if plan['model']['family'] == 'custom':
                    record['code_artifact_id'] = plan['code_artifact_id']
                    record['feature_steps'] = plan['feature_steps']
                try:
                    intent = self.artifacts.read_json('intent-'+plan['plan_id'])
                    record['research_action'] = intent['action']
                    record['evidence_ids'] = intent['evidence_ids']
                except AgentError as error:
                    if error.code != 'ARTIFACT_MISSING':
                        raise
            if base and plan['recipe_id'] == base['recipe_id'] and plan['model']['family'] != 'custom':
                record['changes_from_base'] = {k: v for k, v in params.items() if v != base['model']['hyperparameters'].get(k)}
            else:
                record['hyperparameters'] = params
            records[eid] = record
            if valid:
                value = ev['primary_metric']['value'] * (1 if ev['primary_metric']['direction'] == 'maximize' else -1)
                if plan['recipe_id'] not in best or value > best[plan['recipe_id']][0]:
                    best[plan['recipe_id']] = (value, eid)
        priority = list(dict.fromkeys([eid for eid in ids if eid in proposal_ids] + [pair[1] for pair in best.values()] + ids))
        selected = priority[:limit]
        return {'base_experiment_id': head['experiment_id'], 'entries': [records[eid] for eid in selected],
                'omitted_count': len(priority)-len(selected),
                'instruction': 'Measured trials on your own visible lineage. changes_from_base overrides the current base parameters. Do not repeat a tested configuration; use its measured score, including old regressions.'}

    def research_memory(self, branch_id, donors):
        orders = set()
        for line, cut in self.lineage_cuts(branch_id):
            rows = self.db.execute("SELECT DISTINCT w.work_order_id FROM g_work_orders w JOIN g_events e ON e.run_id=w.run_id AND json_extract(e.record_json,'$.entity_id')=w.work_order_id WHERE w.run_id=? AND w.branch_id=? AND e.event_seq<=? AND json_extract(e.record_json,'$.operation')='G04'", (self.run_id, line, cut))
            orders.update(row[0] for row in rows)
        # Donor measurements are explicit, but donor branch decisions are not
        # silently imported into this branch's pre-idea memory.
        ids = [*self.visible_history(branch_id), *donors]
        return build_research_memory(self.store, self.artifacts, ids, work_order_ids=orders,
                                     branch_cuts=dict(self.lineage_cuts(branch_id)))

    def expire_work(self):
        expired = self.db.execute("SELECT work_order_id,order_json FROM g_work_orders WHERE run_id=? AND state IN ('preparing','issued') AND expires_elapsed_ms<=?", (self.run_id, self.elapsed())).fetchall()
        if not expired:
            return
        with self.store.transaction():
            for row in expired:
                self.db.execute("UPDATE g_work_orders SET state='expired' WHERE run_id=? AND work_order_id=?", (self.run_id, row["work_order_id"]))
                if row["order_json"]:
                    self.event("G03", "work_order", row["work_order_id"], "GWorkOrder", self.order(row["work_order_id"]))
            self.db.execute("UPDATE g_runs SET graph_revision=graph_revision+1 WHERE run_id=?", (self.run_id,))

    def issue_work(self, p, key):
        refinement = self.refinement()
        if refinement and p['branch_id'] != refinement['branch_id']:
            raise AgentError('FINAL_REFINEMENT_ONLY', 'All final work belongs to the selected best branch')
        branch = self.branch(p["branch_id"], p["expected_branch_revision"])
        if branch["state"] != "active" or not self.budget.snapshot()["search_allowed"] or not self.db.execute("SELECT search_allowed FROM g_runs WHERE run_id=?", (self.run_id,)).fetchone()[0]:
            raise AgentError("SEARCH_CLOSED", "Work can only be issued to an active branch with remaining search time")
        head = next((h for h in self.heads(branch["branch_id"]) if h["researcher_id"] == p["researcher_id"]), None)
        if not head:
            raise AgentError("ENTITY_NOT_FOUND", "Unknown peer for this branch")
        if p["action"] == "ensemble":
            raise AgentError("UNSUPPORTED_ACTION", "No unverified ensemble compiler in graph v1")
        for donor in p["allowed_donor_experiment_ids"]:
            ev = self.store.experiment_evaluation(donor)
            if not ev or not ev["valid"] or ev["primary_metric"]["fidelity"] != "full":
                raise AgentError("INVALID_DONOR", "Donor must have a trusted full evaluation")
            caps, _ = dependency_closure(self.manifest(self.experiment_solution(donor))["core"]["components"])
            if set(caps) & set(branch["constraints"]["forbidden_capability_ids"]):
                raise AgentError("CAPABILITY_FORBIDDEN", "Donor contains a capability forbidden in this line")
        now = self.elapsed()
        self.expire_work()
        with self.store.transaction():
            if self.db.execute("SELECT 1 FROM g_work_orders WHERE run_id=? AND researcher_id=? AND state IN ('preparing','issued','submitted')", (self.run_id, p["researcher_id"])).fetchone():
                raise AgentError("PEER_BUSY", "This equal peer already has an unfinished work order")
            order_id = "work-"+uuid.uuid4().hex[:16]
            expiry = now + min(p["ttl_seconds"], max(1, int(self.budget.remaining())))*1000
            self.db.execute("INSERT INTO g_work_orders VALUES(?,?,?,?,?,?,?,?,?)", (self.run_id, order_id, branch["branch_id"], p["researcher_id"], branch["branch_revision"], head["head_revision"], "preparing", expiry, None))
        try:
            cp = self.invoke("G01", {"branch_id": branch["branch_id"], "expected_branch_revision": branch["branch_revision"], "reason": "before_work"}, key=order_id+"/checkpoint")["checkpoint"]
            snapshot = self.bus.call("C07", {"task_spec_id": self.spec["task_spec_id"], "validation_id": self.validation["validation_id"], "limit_recent": 8, "limit_best": 3}, lambda args: self.store.snapshot(args, self.artifacts))["data"]
            binding = {"snapshot_id": snapshot["snapshot_id"], "work_order_id": order_id, "branch_id": branch["branch_id"], "branch_revision": branch["branch_revision"],
                       "base_head_revision": head["head_revision"], "checkpoint_id": cp["checkpoint_id"], "allowed_donor_experiment_ids": list(dict.fromkeys(p["allowed_donor_experiment_ids"]))}
            context_id, capsule_id = "art-"+uuid.uuid4().hex, "art-"+uuid.uuid4().hex
            recent, total_recent = self.recent_for_context(branch["branch_id"], head, binding["allowed_donor_experiment_ids"])
            base_exp = self.store.get(head["experiment_id"]) if head["experiment_id"] else None
            base_manifest = self.db.execute("SELECT manifest_artifact_id FROM g_solutions WHERE run_id=? AND solution_id=?", (self.run_id, head["solution_id"])).fetchone() if head["solution_id"] else None
            base_code = None
            if base_exp:
                self.files.resolve(base_exp["plan"]["code_artifact_id"])
                base_code = self.artifacts.resolve(base_exp["plan"]["code_artifact_id"]).read_text(encoding="utf-8")
            task_contract = self.artifacts.read_json(self.metadata("task_contract_artifact_id"))
            # The base has full diagnostics separately. Historical facts retain
            # parameters/scores without duplicating large group summaries.
            facts = [{"kind": "result" if r["metric"] else "failure", "source_id": r["experiment_id"], "summary": canonical({**r, 'diagnostics': {}}),
                      "artifact_ids": [self.store.get(r["experiment_id"])["plan"]["code_artifact_id"]]} for r in recent]
            rejected = self.rejection_facts(branch["branch_id"], p["researcher_id"])
            protected_ids = {head["experiment_id"], *binding["allowed_donor_experiment_ids"]}
            facts = ([f for f in facts if f["source_id"] in protected_ids] + rejected + [f for f in facts if f["source_id"] not in protected_ids])[:max(1, self.policy["context"]["max_related_records"])]
            total_recent += len(rejected)
            assignment = self.research_assignment(branch["branch_id"], p["researcher_id"])
            if p["action"] == "transfer":
                assignment = "exploit_incumbent"
            observed_recipes = [self.store.get(eid)["plan"]["recipe_id"] for eid in self.visible_history(branch["branch_id"])]
            graph_input = {"binding": {"work_order_id": order_id, "branch_id": branch["branch_id"], "branch_revision": branch["branch_revision"],
                                        "head_revision": head["head_revision"], "capsule_artifact_id": capsule_id},
                           "task_contract": task_contract, "head": head, "base_code": base_code, "constraints": branch["constraints"],
                           "phase": 'refine' if self.refinement() else 'search',
                           "facts": facts, "recent": recent, "total_recent": total_recent, "donors": binding["allowed_donor_experiment_ids"],
                           "base_result": self.memory_record(head["experiment_id"], head) if base_exp else None, "observed_recipes": observed_recipes,
                           "search_memory": self.search_memory(branch['branch_id'], head),
                           "context_id": context_id, "action": p["action"], "policy": self.policy, "budget": self.budget.snapshot()}
            if self.config.research_mode == 'adaptive':
                graph_input['research_memory'] = self.research_memory(branch['branch_id'], binding['allowed_donor_experiment_ids'])
            args = {"task_spec_id": self.spec["task_spec_id"], "snapshot_id": snapshot["snapshot_id"], "tools_catalog_version": p["tools_catalog_version"],
                    "researcher_id": p["researcher_id"], "assignment": assignment, "max_input_tokens": self.policy["context"]["max_input_tokens"]}
            context = self.bus.call("C08", args, lambda value: self.context.build(value, snapshot, graph=graph_input))["data"]
            body = self.artifacts.read_json(context_id)
            user = json.loads(body["messages"][1]["content"])
            self.schemas.validate(body, "GResearchContext");self.schemas.validate(user, "GContextUser")
            self.files.pin(context_id)
            capsule = {"capsule_version": "capsule-1.0", "run_id": self.run_id, "work_order_id": order_id, "branch_id": branch["branch_id"], "branch_revision": branch["branch_revision"],
                       "task_spec_id": self.spec["task_spec_id"], "validation_id": self.validation["validation_id"], "protocol_sha256": self.validation["protocol_sha256"],
                       "mandatory_contract_artifact_id": self.metadata("task_contract_artifact_id"), "base_solution_manifest_artifact_id": base_manifest[0] if base_manifest else None,
                       "constraints": branch["constraints"], "facts": user["facts"], "included_record_ids": [f["source_id"] for f in user["facts"]],
                       "omitted_record_count": total_recent-len(user["facts"]), "estimated_input_tokens": context["estimated_input_tokens"],
                       "token_count_mode": "estimated_bytes", "messages_artifact_id": context_id}
            capsule_art = self.files.put_json(capsule, "graph_capsule", ident=capsule_id)
            order = {"work_order_id": order_id, "run_id": self.run_id, "researcher_id": p["researcher_id"], "branch_id": branch["branch_id"],
                     "branch_revision": branch["branch_revision"], "base_head_revision": head["head_revision"], "base_experiment_id": head["experiment_id"],
                     "base_solution_id": head["solution_id"], "base_team_checkpoint_id": cp["checkpoint_id"], "allowed_donor_experiment_ids": binding["allowed_donor_experiment_ids"],
                     "action": p["action"], "constraints": branch["constraints"], "task_spec_id": self.spec["task_spec_id"], "dataset_id": self.spec["dataset_id"],
                     "validation_id": self.validation["validation_id"], "protocol_sha256": self.validation["protocol_sha256"], "context_id": context_id,
                     "context_capsule_artifact_id": capsule_art["artifact_id"], "issued_elapsed_ms": now, "expires_elapsed_ms": expiry, "state": "issued"}
            request = {"schema_version": "1.0", "request_id": "assigned-"+uuid.uuid4().hex[:16], "run_id": self.run_id, "idempotency_key": None,
                       "payload": {"researcher_id": p["researcher_id"], "context_id": context_id, "assignment": assignment,
                                   "base_experiment_ids": [head["experiment_id"]] if head["experiment_id"] else [],
                                   "forbidden_fingerprints": snapshot["reserved_fingerprints"], "budget": self.budget.snapshot(), "proposal_limit": 1}}
            def mutate():
                self.branch(branch["branch_id"], branch["branch_revision"])
                current = next(h for h in self.heads(branch["branch_id"]) if h["researcher_id"] == p["researcher_id"])
                if current != head or self.elapsed() >= expiry:
                    raise AgentError("HEAD_REVISION_CONFLICT", "Assignment changed while its context was being built")
                self.db.execute("INSERT INTO g_snapshot_bindings VALUES(?,?,?,?)", (self.run_id, snapshot["snapshot_id"], order_id, canonical(binding)))
                self.db.execute("INSERT INTO g_contexts VALUES(?,?,?,?,?)", (self.run_id, context_id, order_id, capsule_id, context_id))
                changed = self.db.execute("UPDATE g_work_orders SET state='issued',order_json=? WHERE run_id=? AND work_order_id=? AND state='preparing'", (canonical(order), self.run_id, order_id)).rowcount
                if changed != 1:
                    raise AgentError("STALE_WORK_ORDER", "Preparing assignment was revoked")
                self.event("G03", "work_order", order_id, "GWorkOrder", order)
                return {"work_order": order, "capsule": capsule, "researcher_request": request}
            return self.commit("G03", p, key, mutate)
        except BaseException:
            with self.store.transaction():
                self.db.execute("UPDATE g_work_orders SET state='revoked' WHERE run_id=? AND work_order_id=? AND state='preparing'", (self.run_id, order_id))
            raise
