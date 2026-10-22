import json

from agent.contracts import canonical
from agent.errors import AgentError
from agent.graph.identity import digest
from agent.store import comparable


class GraphExport:
    def inspect(self, p, key=None):
        cut = self.cut()
        branches = [self.branch(r[0]) for r in self.db.execute("SELECT branch_id FROM g_branches WHERE run_id=? ORDER BY rowid", (self.run_id,))]
        if p["branch_id"] is not None:
            self.branch(p["branch_id"])
            branches = [b for b in branches if b["branch_id"] == p["branch_id"]]
        events = [json.loads(r[0]) for r in self.db.execute("SELECT record_json FROM g_events WHERE run_id=? AND event_seq>? AND event_seq<=? ORDER BY event_seq", (self.run_id, p["after_event_seq"], cut))]
        if p["branch_id"] is not None:
            branch_id = p["branch_id"]
            events = [e for e in events if e["record"].get("branch_id") == branch_id or e["entity_id"] == branch_id]
        selected = events[:p["limit"]]
        return {"branches": branches, "heads": [{"branch_id": b["branch_id"], "head": h} for b in branches for h in self.heads(b["branch_id"])],
                "events": selected, "next_event_seq": selected[-1]["event_seq"] if len(events) > len(selected) else None,
                "captured_event_seq": cut, "graph_revision": self.revision()}

    def export(self, p, key):
        if p["task_spec_id"] != self.spec["task_spec_id"] or p["validation_id"] != self.validation["validation_id"] or p["selection_policy_id"] != "same-protocol-primary-v1":
            raise AgentError("STALE_VALIDATION", "Export scope does not match the current task and selection policy")
        # One dispatcher-owned read cut. No other writer may interleave this projection.
        with self.store.transaction():
            cut, revision = self.cut(), self.revision()
            branches = [self.branch(r[0]) for r in self.db.execute("SELECT branch_id FROM g_branches WHERE run_id=? ORDER BY rowid", (self.run_id,))]
            heads = [{"branch_id": b["branch_id"], "head": h} for b in branches for h in self.heads(b["branch_id"])]
            checkpoints = [json.loads(r[0]) for r in self.db.execute("SELECT snapshot_json FROM g_checkpoints WHERE run_id=? ORDER BY rowid", (self.run_id,))]
            solution_rows = self.db.execute("SELECT * FROM g_solutions WHERE run_id=? ORDER BY rowid", (self.run_id,)).fetchall()
            exp_rows = self.db.execute("SELECT e.*,l.solution_id FROM experiments e JOIN g_experiment_links l USING(run_id,experiment_id) WHERE e.run_id=? ORDER BY e.rowid", (self.run_id,)).fetchall()
            evaluation_rows = self.db.execute("SELECT v.record FROM evaluations v JOIN experiments e USING(experiment_id) WHERE e.run_id=? ORDER BY v.rowid", (self.run_id,)).fetchall()
            artifacts = [{k: r[k] for k in ("artifact_id", "run_id", "kind", "relative_path", "sha256", "size_bytes", "media_type")} for r in self.db.execute("SELECT * FROM g_artifacts WHERE run_id=? AND state='ready' ORDER BY rowid", (self.run_id,))]
            event_records = [json.loads(r[0]) for r in self.db.execute("SELECT record_json FROM g_events WHERE run_id=? AND event_seq<=? ORDER BY event_seq", (self.run_id, cut))]
            selections_revision = self.db.execute("SELECT revision FROM selections WHERE run_id=?", (self.run_id,)).fetchone()[0]
            ideas = [json.loads(r[0]) for r in self.db.execute("SELECT record_json FROM g_ideas WHERE run_id=? ORDER BY rowid", (self.run_id,))]
            trials = [json.loads(r[0]) for r in self.db.execute("SELECT record_json FROM g_component_trials WHERE run_id=? ORDER BY rowid", (self.run_id,))]
            decisions = [r[0] for r in self.db.execute("SELECT artifact_id FROM g_decisions WHERE run_id=? ORDER BY rowid", (self.run_id,))]
            orders = [self.order(r[0]) for r in self.db.execute("SELECT work_order_id FROM g_work_orders WHERE run_id=? AND order_json IS NOT NULL ORDER BY rowid", (self.run_id,))]
            bindings = [{"work_order_id": r["work_order_id"], "experiment_id": r["experiment_id"], "state": r["state"], "invalidated": bool(r["invalidated"])} for r in self.db.execute("SELECT * FROM g_bindings WHERE run_id=? ORDER BY rowid", (self.run_id,))]
            edges = [{k: r[k] for k in ("parent_solution_id", "child_solution_id", "kind")} for r in self.db.execute("SELECT * FROM g_solution_edges WHERE run_id=? ORDER BY rowid", (self.run_id,))]
            attempts = {}
            for r in self.db.execute("SELECT experiment_id,record_json FROM g_attempts WHERE run_id=? ORDER BY attempt_no", (self.run_id,)):
                attempts.setdefault(r[0], []).append(json.loads(r[1]))
            evaluated_order = {r[0]: r[1] for r in self.db.execute("SELECT experiment_id,MIN(event_seq) FROM g_branch_evaluations WHERE run_id=? GROUP BY experiment_id", (self.run_id,))}
            champion = self.store.champion()
        solutions = [self.manifest(r["solution_id"]) for r in solution_rows]
        evaluations = [json.loads(r[0]) for r in evaluation_rows]
        latest = {e["experiment_id"]: e for e in evaluations}
        experiments, eligible, excluded = [], [], []
        for row in exp_rows:
            plan = json.loads(row["plan"]);ev = latest.get(row["experiment_id"])
            meta = self.files.resolve(plan["code_artifact_id"])
            core = self.store.fingerprint_core(plan, meta["sha256"])
            if digest(core) != row["fingerprint"]:
                raise AgentError("FINGERPRINT_MISMATCH", "Stored experiment cannot be reconstructed from pinned code/params")
            error = None
            if row["error"]:
                try:
                    parsed = json.loads(row["error"])
                    self.schemas.validate(parsed, "Error");error = parsed
                except (ValueError, TypeError, AgentError):
                    error = AgentError("EXECUTION_FAILED", row["error"]).as_dict()
            record = {"experiment_id": row["experiment_id"], "solution_id": row["solution_id"], "fingerprint": row["fingerprint"],
                      "fingerprint_core": core, "plan": plan, "state": row["state"], "attempts": attempts.get(row["experiment_id"], []),
                      "evaluation_id": ev["evaluation_id"] if ev else None, "terminal_error": error}
            self.schemas.validate(record, "AExperimentRecord");experiments.append(record)
            expected = {"dataset_id": self.spec["dataset_id"], "validation_id": self.validation["validation_id"], "metric_id": self.spec["metric"]["metric_id"], "direction": self.spec["metric"]["direction"], "fidelity": "full"}
            if ev and ev["valid"] and not ev["violations"] and row["state"] == "evaluated" and comparable(ev["primary_metric"], expected):
                eligible.append((ev, row["solution_id"]))
            else:
                excluded.append(row["experiment_id"])
        utility = lambda e: e["primary_metric"]["value"] * (1 if e["primary_metric"]["direction"] == "maximize" else -1)
        eligible.sort(key=lambda pair: (-utility(pair[0]), evaluated_order.get(pair[0]["experiment_id"], 2**63)))
        ranking = {"ranking_version": "ranking-1.0", "run_id": self.run_id, "task_spec_id": self.spec["task_spec_id"], "validation_id": self.validation["validation_id"],
                   "protocol_sha256": self.validation["protocol_sha256"], "selection_policy_id": p["selection_policy_id"],
                   "rows": [{"experiment_id": ev["experiment_id"], "solution_id": sid, "evaluation_id": ev["evaluation_id"], "primary_metric": ev["primary_metric"],
                             "rank": rank+1, "eligible": True, "reason_codes": []} for rank, (ev, sid) in enumerate(eligible)],
                   "best_evaluated_experiment_id": eligible[0][0]["experiment_id"] if eligible else None, "published_champion": champion,
                   "excluded_experiment_ids": excluded, "note": "published_output_requires_C18_C21"}
        if ranking["best_evaluated_experiment_id"] != self.store.best():
            raise AgentError("SELECTION_MISMATCH", "Global selection does not match all comparable full evaluations")
        for event in event_records:
            self.schemas.validate(event, "GEvent");self.schemas.validate(event["record"], event["record_schema"])
        ranking_art = self.files.put_json(ranking, "ranking_report")
        events_art = self.files.put_bytes(("\n".join(canonical(e) for e in event_records)+"\n").encode("utf-8"), "graph_events_export", "application/x-ndjson")
        export = {"export_version": "graph-export-1.0", "run_id": self.run_id, "captured_event_seq": cut, "store_revision": selections_revision, "graph_revision": revision,
                  "branches": branches, "heads": heads, "checkpoints": checkpoints, "solutions": solutions, "solution_edges": edges,
                  "branch_seeds": [e["record"] for e in event_records if e["record_schema"] == "GBranchSeed"],
                  "refinement": self.refinement(),
                  "experiment_links": [{"experiment_id": x["experiment_id"], "solution_id": x["solution_id"]} for x in experiments],
                  "ideas": ideas, "ranking_artifact_id": ranking_art["artifact_id"], "artifact_ids": [a["artifact_id"] for a in artifacts]+[ranking_art["artifact_id"], events_art["artifact_id"]],
                  "artifacts": artifacts+[ranking_art, events_art], "experiments": experiments, "evaluations": evaluations,
                  "work_orders": orders, "bindings": bindings, "events_artifact_id": events_art["artifact_id"], "component_trials": trials, "decision_artifact_ids": decisions}
        graph_art = self.files.put_json(export, "graph_export")
        folder = "exports/"+p["scope"]+"-"+str(cut)
        self.artifacts.write_json(folder+"/graph.json", export, "graph_export_projection")
        self.artifacts.write_json(folder+"/ranking.json", ranking, "ranking_projection")
        self.artifacts.write_text(folder+"/graph-events.jsonl", "\n".join(canonical(e) for e in event_records)+"\n", "graph_events_projection", "application/x-ndjson")
        def mutate():
            self.set_metadata_tx("latest_export", {"folder": folder, "graph_artifact_id": graph_art["artifact_id"], "ranking_artifact_id": ranking_art["artifact_id"], "events_artifact_id": events_art["artifact_id"], "captured_event_seq": cut})
            self.event("G11", "export", graph_art["artifact_id"], "ArtifactRef", graph_art)
            return {"graph_artifact": graph_art, "ranking_artifact": ranking_art, "events_artifact": events_art,
                    "best_evaluated_experiment_id": ranking["best_evaluated_experiment_id"], "published_champion_experiment_id": champion["experiment_id"] if champion else None,
                    "captured_event_seq": cut}
        return self.commit("G11", p, key, mutate)
