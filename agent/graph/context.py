from copy import deepcopy
import json
import uuid

from agent.contracts import canonical
from agent.errors import AgentError
from agent.graph.identity import dependency_closure


class WorkService:
    def recent_for_context(self, branch_id, head, donors):
        rows = self.db.execute("SELECT b.experiment_id FROM g_bindings b JOIN g_work_orders w USING(run_id,work_order_id) WHERE w.run_id=? AND w.branch_id=? ORDER BY b.rowid DESC", (self.run_id, branch_id)).fetchall()
        ids = list(dict.fromkeys(([head["experiment_id"]] if head["experiment_id"] else []) + list(donors) + [r[0] for r in rows]))
        recent = []
        for eid in ids[:self.policy["context"]["max_related_records"]]:
            exp = self.store.get(eid);plan = exp["plan"];ev = self.store.experiment_evaluation(eid)
            recent.append({"experiment_id": eid, "state": exp["state"], "recipe_id": plan["recipe_id"], "hypothesis": plan["hypothesis"],
                           "metric": ev["primary_metric"] if ev else None, "hyperparameters": plan["model"]["hyperparameters"],
                           "feature_steps": plan["feature_steps"], "seed": plan["seed"], "is_incumbent": eid == head["experiment_id"]})
        return recent, len(ids)

    def issue_work(self, p, key):
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
        with self.store.transaction():
            self.db.execute("UPDATE g_work_orders SET state='expired' WHERE run_id=? AND state IN ('preparing','issued') AND expires_elapsed_ms<=?", (self.run_id, now))
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
            facts = [{"kind": "result" if r["metric"] else "failure", "source_id": r["experiment_id"], "summary": canonical(r),
                      "artifact_ids": [self.store.get(r["experiment_id"])["plan"]["code_artifact_id"]]} for r in recent]
            assignment = "exploit_incumbent" if (self.context.round_no + self.metadata("researcher_ids").index(p["researcher_id"])) % 2 == 0 else "explore_alternative"
            graph_input = {"binding": {"work_order_id": order_id, "branch_id": branch["branch_id"], "branch_revision": branch["branch_revision"],
                                        "head_revision": head["head_revision"], "capsule_artifact_id": capsule_id},
                           "task_contract": task_contract, "head": head, "base_code": base_code, "constraints": branch["constraints"],
                           "facts": facts, "recent": recent, "total_recent": total_recent, "donors": binding["allowed_donor_experiment_ids"],
                           "context_id": context_id, "action": p["action"], "policy": self.policy, "budget": self.budget.snapshot()}
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
