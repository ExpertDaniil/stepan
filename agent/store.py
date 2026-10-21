import hashlib
import json
from pathlib import Path
import sqlite3
import threading
import uuid
from contextlib import contextmanager

from agent.contracts import canonical
from agent.errors import AgentError


def comparable(a, b):
    return all(a[k] == b[k] for k in ("dataset_id", "validation_id", "metric_id", "direction", "fidelity"))


def is_better(candidate, incumbent):
    if not comparable(candidate, incumbent):
        raise AgentError("INCOMPARABLE_EVALUATIONS", "Metrics come from different protocols")
    return candidate["value"] < incumbent["value"] if candidate["direction"] == "minimize" else candidate["value"] > incumbent["value"]


class Store:
    def __init__(self, path, run_id, config, log):
        self.run_id, self.log = run_id, log
        self.owner_thread = threading.get_ident()
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.graph = None
        self.db.executescript(Path(__file__).with_name("schema.sql").read_text(encoding="utf-8"))
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO runs VALUES(?,?,?)", (run_id, "initializing", canonical(config)))
            self.db.execute("INSERT OR IGNORE INTO selections VALUES(?,?,?)", (run_id, None, 0))

    @contextmanager
    def transaction(self):
        """Nested operations use SAVEPOINT, never commit the caller's transaction."""
        self._writer()
        nested = self.db.in_transaction
        name = "tx_" + uuid.uuid4().hex
        self.db.execute("SAVEPOINT " + name if nested else "BEGIN IMMEDIATE")
        try:
            yield
        except BaseException:
            if nested:
                self.db.execute("ROLLBACK TO " + name)
                self.db.execute("RELEASE " + name)
            else:
                self.db.rollback()
            raise
        else:
            if nested:
                self.db.execute("RELEASE " + name)
            else:
                self.db.commit()

    @staticmethod
    def fingerprint_core(plan, code_hash):
        value = {k: plan[k] for k in ("dataset_id", "task_spec_id", "validation_id", "recipe_id", "model", "feature_steps", "seed", "fidelity")}
        return {**value, "code_sha256": code_hash}

    def _writer(self):
        if threading.get_ident() != self.owner_thread:
            raise AgentError("WRITER_THREAD_VIOLATION", "SQLite mutations belong to the dispatcher thread")

    def set_run_state(self, state):
        self._writer()
        with self.db:
            self.db.execute("UPDATE runs SET state=? WHERE run_id=?", (state, self.run_id))
        self.log.event("run.state_changed", state=state)

    def get(self, ident):
        row = self.db.execute("SELECT * FROM experiments WHERE experiment_id=? AND run_id=?", (ident, self.run_id)).fetchone()
        if not row:
            raise AgentError("EXPERIMENT_NOT_FOUND", ident)
        result = dict(row)
        result["plan"] = json.loads(result["plan"])
        return result

    def reserve(self, plan, code_hash):
        self._writer()
        with self.transaction():
            result = self.reserve_tx(plan, code_hash)
        if result["disposition"] == "created":
            self.log.event("experiment.reserved", experiment_id=result["experiment_id"], fingerprint=result["fingerprint"], researcher_id=plan["researcher_id"])
        return result

    def reserve_tx(self, plan, code_hash):
        self._writer()
        if not self.db.in_transaction:
            raise AgentError("TRANSACTION_REQUIRED", "reserve_tx needs an active transaction")
        identity = self.fingerprint_core(plan, code_hash)
        fingerprint = hashlib.sha256(canonical(identity).encode()).hexdigest()
        old = self.db.execute("SELECT experiment_id,state FROM experiments WHERE run_id=? AND fingerprint=?", (self.run_id, fingerprint)).fetchone()
        if old:
            return {"experiment_id": old["experiment_id"], "fingerprint": fingerprint, "disposition": "duplicate",
                    "state": old["state"], "existing_experiment_id": old["experiment_id"]}
        for parent in plan["parent_ids"]:
            try:
                self.get(parent)
            except AgentError as exc:
                raise AgentError("PARENT_NOT_FOUND", parent) from exc
        ident = "exp-" + uuid.uuid4().hex[:12]
        self.db.execute("INSERT INTO experiments(experiment_id,run_id,fingerprint,plan,state) VALUES(?,?,?,?,?)",
                        (ident, self.run_id, fingerprint, canonical(plan), "queued"))
        for parent in dict.fromkeys(plan["parent_ids"]):
            self.db.execute("INSERT INTO experiment_parents VALUES(?,?)", (ident, parent))
        return {"experiment_id": ident, "fingerprint": fingerprint, "disposition": "created", "state": "queued", "existing_experiment_id": None}

    def start(self, ident, lease):
        self._writer()
        exp = self.get(ident)
        if exp["state"] != "queued":
            raise AgentError("INVALID_STATE_TRANSITION", "Only queued experiments can start")
        attempt = exp["attempt_no"] + 1
        with self.transaction():
            self.db.execute("UPDATE experiments SET state='running',attempt_no=?,lease_id=?,fencing_token=? WHERE experiment_id=?",
                            (attempt, lease["lease_id"], lease["fencing_token"], ident))
        return attempt

    def apply_event(self, event, phase):
        self._writer()
        existing = self.db.execute("SELECT record FROM job_events WHERE event_id=?", (event["event_id"],)).fetchone()
        exp = self.get(event["experiment_id"])
        if existing:
            if json.loads(existing["record"]) != event:
                raise AgentError("IDEMPOTENCY_CONFLICT", "event_id reused with different content")
            disposition = "already_applied"
        else:
            # Refit has a distinct lease; the runner validates its job token.
            if phase == "train" and (event["attempt_no"] != exp["attempt_no"] or event["fencing_token"] != exp["fencing_token"] or event["lease_id"] != exp["lease_id"]):
                raise AgentError("STALE_LEASE", "Late event from an obsolete attempt")
            if phase == "train" and exp["state"] != "running":
                raise AgentError("INVALID_STATE_TRANSITION", "A terminal attempt cannot be overwritten by another event")
            state = "trained" if event["state"] == "succeeded" else event["state"]
            with self.transaction():
                self.db.execute("INSERT INTO job_events VALUES(?,?)", (event["event_id"], canonical(event)))
                if phase == "train":
                    self.db.execute("UPDATE experiments SET state=?,duration=?,error=? WHERE experiment_id=?",
                                    (state, event["duration_seconds"], canonical(event["error"]) if event["error"] else None, event["experiment_id"]))
                    if self.graph:
                        self.graph.capture_event_tx(event)
            disposition = "applied"
            exp = self.get(event["experiment_id"])
        submission_state = ("validating" if event["state"] == "succeeded" else event["state"]) if phase == "refit" else None
        return {"event_id": event["event_id"], "disposition": disposition, "job_state": event["state"],
                "experiment_state": exp["state"], "submission_state": submission_state}

    def invalidate(self, ident, error):
        self._writer()
        with self.transaction():
            self.db.execute("UPDATE experiments SET state='invalid',error=? WHERE experiment_id=?", (str(error), ident))
            if self.graph:
                self.graph.terminal_tx(ident, "invalid", error, "C16")

    def fail_start(self, ident, error):
        self._writer()
        with self.transaction():
            self.db.execute("UPDATE experiments SET state='failed',error=? WHERE experiment_id=?", (str(error), ident))
            if self.graph:
                self.graph.terminal_tx(ident, "failed", error, "C15")

    def save_evaluation(self, record):
        self._writer()
        with self.db:
            self.db.execute("INSERT INTO evaluations VALUES(?,?,?)", (record["evaluation_id"], record["experiment_id"], canonical(record)))

    def evaluation(self, ident):
        row = self.db.execute("SELECT record FROM evaluations WHERE evaluation_id=?", (ident,)).fetchone()
        if not row:
            raise AgentError("EVALUATION_NOT_FOUND", ident)
        return json.loads(row[0])

    def experiment_evaluation(self, ident):
        row = self.db.execute("SELECT record FROM evaluations WHERE experiment_id=? ORDER BY rowid DESC LIMIT 1", (ident,)).fetchone()
        return json.loads(row[0]) if row else None

    def best(self):
        return self.db.execute("SELECT experiment_id FROM selections WHERE run_id=?", (self.run_id,)).fetchone()[0]

    def champion(self):
        row = self.db.execute("SELECT record FROM publications WHERE run_id=?", (self.run_id,)).fetchone()
        return json.loads(row[0]) if row else None

    def commit_evaluation(self, payload):
        self._writer()
        with self.transaction():
            return self.commit_evaluation_tx(payload)

    def commit_evaluation_tx(self, payload):
        self._writer()
        if not self.db.in_transaction:
            raise AgentError("TRANSACTION_REQUIRED", "commit_evaluation_tx needs an active transaction")
        ev = self.evaluation(payload["evaluation_id"])
        exp = self.get(payload["experiment_id"])
        if ev["experiment_id"] != exp["experiment_id"] or exp["state"] not in {"trained", "evaluated"}:
            raise AgentError("INVALID_STATE_TRANSITION", "Evaluation does not match a trained experiment")
        if not ev["valid"] or ev["violations"]:
            raise AgentError("INVALID_EVALUATION", "Only valid trusted evaluations can be committed")
        incumbent = self.best()
        promoted = False
        if ev["primary_metric"]["fidelity"] == "full":
            promoted = not incumbent or is_better(ev["primary_metric"], self.experiment_evaluation(incumbent)["primary_metric"])
        self.db.execute("UPDATE experiments SET state='evaluated' WHERE experiment_id=?", (exp["experiment_id"],))
        if promoted:
            self.db.execute("UPDATE selections SET experiment_id=?,revision=revision+1 WHERE run_id=?", (exp["experiment_id"], self.run_id))
        champion = self.champion()
        revision = self.db.execute("SELECT revision FROM selections WHERE run_id=?", (self.run_id,)).fetchone()[0]
        return {"experiment_state": "evaluated", "best_evaluated_experiment_id": self.best(),
                "published_champion_experiment_id": champion["experiment_id"] if champion else None,
                "should_prepare_submission": promoted, "store_revision": revision}

    def snapshot(self, payload, artifacts):
        rows = self.db.execute("SELECT experiment_id,state,plan,fingerprint FROM experiments WHERE run_id=? ORDER BY rowid DESC LIMIT ?",
                               (self.run_id, payload["limit_recent"])).fetchall()
        summary = []
        best = self.best()
        if best and not any(row["experiment_id"] == best for row in rows):
            rows = list(rows) + [self.db.execute("SELECT experiment_id,state,plan,fingerprint FROM experiments WHERE experiment_id=?", (best,)).fetchone()]
        for row in rows:
            ev = self.experiment_evaluation(row["experiment_id"])
            p = json.loads(row["plan"])
            summary.append({"experiment_id": row["experiment_id"], "state": row["state"], "recipe_id": p["recipe_id"],
                            "hypothesis": p["hypothesis"], "metric": ev["primary_metric"] if ev else None,
                            "hyperparameters": p["model"]["hyperparameters"], "feature_steps": p["feature_steps"],
                            "seed": p["seed"], "is_incumbent": row["experiment_id"] == best})
        art = artifacts.write_json("contexts/snapshot-" + uuid.uuid4().hex[:8] + ".json", summary, "research_summary")
        champion = self.champion()
        revision = self.db.execute("SELECT revision FROM selections WHERE run_id=?", (self.run_id,)).fetchone()[0]
        return {"snapshot_id": art["artifact_id"], "store_revision": revision, "best_evaluated_experiment_id": self.best(),
                "published_champion_experiment_id": champion["experiment_id"] if champion else None,
                "recent_experiment_ids": [r["experiment_id"] for r in rows],
                "reserved_fingerprints": [r[0] for r in self.db.execute("SELECT fingerprint FROM experiments WHERE run_id=? AND state IN ('queued','running')", (self.run_id,))],
                "summary_artifact_id": art["artifact_id"]}

    def prepare_publication(self, record):
        self._writer()
        ident = "pub-" + uuid.uuid4().hex[:12]
        with self.db:
            self.db.execute("INSERT INTO publication_journal VALUES(?,?,?,?)", (ident, self.run_id, "prepared", canonical(record)))
        return ident

    def finish_publication(self, journal_id, champion):
        self._writer()
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO publications VALUES(?,?)", (self.run_id, canonical(champion)))
            self.db.execute("UPDATE publication_journal SET state='committed' WHERE journal_id=?", (journal_id,))

    def pending_publications(self):
        return [(r[0], json.loads(r[1])) for r in self.db.execute("SELECT journal_id,record FROM publication_journal WHERE run_id=? AND state='prepared' ORDER BY rowid", (self.run_id,))]

    def close(self):
        self.db.close()
