import json


class GraphRecovery:
    def recover(self, p, key):
        repair = p["mode"] == "repair"
        reasons, revoked, orphans = [], [], []
        check = self.db.execute("PRAGMA quick_check").fetchall()
        if [r[0] for r in check] != ["ok"]:
            reasons.append("SQLite quick_check failed")
        if self.db.execute("PRAGMA foreign_key_check").fetchall():
            reasons.append("SQLite foreign-key violation")
        repaired, intent_errors = self.files.recover_intents(repair)
        reasons.extend(intent_errors)
        from agent.errors import AgentError
        for row in self.db.execute("SELECT artifact_id FROM g_artifacts WHERE run_id=? AND state='ready'", (self.run_id,)).fetchall():
            try:
                self.files.resolve(row[0], True)
            except AgentError as exc:
                reasons.append(str(exc))
        active_ids = {job.job_id for job in self.runner.active()}
        known_exps = set()
        for row in self.db.execute("SELECT experiment_id,record_json FROM g_attempts WHERE run_id=?", (self.run_id,)):
            record = json.loads(row[1])
            if record["state"] in {"accepted", "running"}:
                known_exps.add(row[0])
                if record["job_id"] not in active_ids:
                    orphans.append(record["job_id"])
        for row in self.db.execute("SELECT experiment_id FROM experiments WHERE run_id=? AND state='running'", (self.run_id,)):
            if row[0] not in known_exps:
                reasons.append("Untracked running experiment: "+row[0])
        if orphans:
            reasons.append("Unconfirmed orphan processes; no lease or process is silently treated as stopped")
        if p["trusted_wall_seconds_remaining"] <= 0:
            reasons.append("Run deadline exhausted")
        repaired_work = False
        if repair and not reasons:
            pending = self.db.execute("SELECT DISTINCT b.experiment_id,e.state FROM g_bindings b JOIN experiments e USING(run_id,experiment_id) WHERE b.run_id=? AND b.state='pending'", (self.run_id,)).fetchall()
            for row in pending:
                eid, state = row[0], row[1]
                evaluation = self.store.experiment_evaluation(eid)
                if state in {"evaluated", "trained"} and evaluation:
                    attempt = self.db.execute("SELECT record_json FROM g_attempts WHERE run_id=? AND experiment_id=? ORDER BY attempt_no DESC LIMIT 1", (self.run_id, eid)).fetchone()
                    exp = self.store.get(eid)
                    evidence = json.loads(attempt[0]) if attempt else None
                    if not evidence or evidence["state"] != "succeeded" or evidence["fencing_token"] != exp["fencing_token"] or evidence["attempt_no"] != exp["attempt_no"]:
                        reasons.append("Evaluation lacks matching successful execution evidence: "+eid)
                        continue
                    self.invoke("G05", {"experiment_id": eid, "evaluation_id": evaluation["evaluation_id"], "selection_policy_id": "same-protocol-primary-v1"}, key="recovery-eval/"+evaluation["evaluation_id"]+"/"+str(self.revision()))
                    repaired_work = True
                elif state == "queued":
                    with self.store.transaction():
                        self.db.execute("UPDATE experiments SET state='cancelled',error=? WHERE experiment_id=?", ("Recovery closes an unstarted queued assignment", eid))
                        self.terminal_tx(eid, "cancelled", "Queued assignment was never started", "C22")
                    repaired_work = True
                elif state in {"failed", "timed_out", "cancelled", "invalid"}:
                    with self.store.transaction():
                        self.terminal_tx(eid, state, "Recovered terminal assignment", "C15")
                    repaired_work = True
                elif state == "trained":
                    reasons.append("Trained predictions need trusted C16 evaluation before selection: "+eid)
        def mutate():
            if repair:
                rows = self.db.execute("SELECT work_order_id FROM g_work_orders WHERE run_id=? AND state IN ('preparing','issued')", (self.run_id,)).fetchall()
                revoked.extend(r[0] for r in rows)
                self.db.execute("UPDATE g_work_orders SET state='revoked' WHERE run_id=? AND state IN ('preparing','issued')", (self.run_id,))
                if reasons:
                    self.db.execute("UPDATE g_runs SET search_allowed=0 WHERE run_id=?", (self.run_id,))
            allowed = bool(self.db.execute("SELECT search_allowed FROM g_runs WHERE run_id=?", (self.run_id,)).fetchone()[0])
            return {"state": "blocked" if reasons else ("recovered" if repaired or revoked or repaired_work else "healthy"), "repaired_intent_ids": repaired,
                    "revoked_work_order_ids": revoked, "orphan_job_ids": orphans, "blocking_reasons": reasons,
                    "search_allowed": allowed and not reasons and self.budget.snapshot()["search_allowed"], "graph_revision": self.revision()}
        if repair:
            return self.commit("G12", p, key, mutate)
        return mutate()
