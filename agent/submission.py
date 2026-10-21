from pathlib import Path
import hashlib
import json
import os
import shutil
import tempfile
import threading
import uuid

from agent.artifacts import file_sha256
from agent.errors import AgentError
from agent.store import is_better


class Submission:
    def __init__(self, spec, artifacts, store, log):
        self.spec, self.artifacts, self.store, self.log = spec, artifacts, store, log
        self.jobs, self.reports = {}, {}
        self.lock = threading.RLock()

    def change_state(self, ident, state):
        self.jobs[ident]["state"] = state
        self.artifacts.write_json(f"submissions/{ident}/state-{uuid.uuid4().hex[:8]}.json", self.jobs[ident], "submission_state")
        self.log.event("submission.state_changed", submission_job_id=ident, state=state,
                       experiment_id=self.jobs[ident]["experiment_id"])

    def prepare(self, payload):
        if payload["task_spec_id"] != self.spec["task_spec_id"]:
            raise AgentError("TASK_SPEC_CONFLICT", "Submission uses another task")
        experiment = self.store.get(payload["experiment_id"])
        evaluation = self.store.evaluation(payload["evaluation_id"])
        if experiment["state"] != "evaluated" or evaluation["experiment_id"] != experiment["experiment_id"] or evaluation["primary_metric"]["fidelity"] != "full":
            raise AgentError("CANDIDATE_NOT_EVALUATED", "Only a full evaluated candidate can be refit")
        ident = "subjob-" + uuid.uuid4().hex[:12]
        self.jobs[ident] = {**payload, "state": "preparing"}
        self.change_state(ident, "preparing")
        champion = self.store.champion()
        return {"submission_job_id": ident, "state": "preparing",
                "published_champion_experiment_id": champion["experiment_id"] if champion else None,
                "output_path": self.spec["output"]["path"]}

    def validate_output(self, payload):
        import numpy as np
        import pandas as pd
        if payload["task_spec_id"] != self.spec["task_spec_id"] or payload["reference_test_artifact_id"] != self.spec["test_artifact_id"]:
            raise AgentError("TASK_SPEC_CONFLICT", "Output validation uses another task or test file")
        path = self.artifacts.resolve(payload["candidate_artifact_id"])
        test = pd.read_csv(self.artifacts.resolve(self.spec["test_artifact_id"]), dtype=str, keep_default_na=False, encoding="utf-8")
        candidate = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")
        output = self.spec["output"]
        if list(candidate.columns) != output["columns"] or len(candidate) != len(test):
            raise AgentError("INVALID_OUTPUT_SCHEMA", "Output columns or row count mismatch")
        ids = output["id_columns"]
        if ids:
            if candidate[ids].duplicated().any() or candidate[ids].eq("").any().any():
                raise AgentError("INVALID_OUTPUT_VALUES", "Missing or duplicate output identifiers")
            if not candidate[ids].equals(test[ids]):
                raise AgentError("ID_ORDER_MISMATCH", "Output ids must match test order and exact text")
        values = candidate[output["prediction_columns"][0]]
        if self.spec["task_kind"] == "regression":
            try:
                values = pd.to_numeric(values, errors="raise").to_numpy(dtype=float)
            except Exception as exc:
                raise AgentError("INVALID_OUTPUT_VALUES", "Non-numeric regression output") from exc
            if not np.isfinite(values).all():
                raise AgentError("INVALID_OUTPUT_VALUES", "Non-finite output")
        elif values.eq("").any() or not set(values).issubset(self.spec["allowed_labels"]):
            raise AgentError("INVALID_OUTPUT_VALUES", "Unknown or missing class labels")
        ident = "outcheck-" + uuid.uuid4().hex[:12]
        report = {"validation_report_id": ident, "valid": True, "rows": len(candidate), "columns": list(candidate.columns),
                  "id_order_matches": True, "finite_predictions": True, "allowed_labels_only": True,
                  "candidate_sha256": file_sha256(path)}
        self.reports[ident] = {"report": report, "artifact_id": payload["candidate_artifact_id"]}
        self.artifacts.write_json(f"submissions/{ident}.json", report, "output_validation")
        return report

    def publish(self, payload):
        with self.lock:
            report = self.reports.get(payload["validation_report_id"])
            if not report or report["artifact_id"] != payload["candidate_artifact_id"]:
                raise AgentError("INVALID_OUTPUT_SCHEMA", "No trusted validation report for this artifact")
            source = self.artifacts.resolve(payload["candidate_artifact_id"])
            digest = file_sha256(source)
            if digest != payload["candidate_sha256"] or digest != report["report"]["candidate_sha256"]:
                raise AgentError("CANDIDATE_CHANGED", "Candidate changed after validation")
            evaluation = self.store.evaluation(payload["evaluation_id"])
            if evaluation["experiment_id"] != payload["experiment_id"]:
                raise AgentError("CANDIDATE_NOT_EVALUATED", "Candidate/evaluation mismatch")
            current = self.store.champion()
            target = Path(self.spec["output"]["path"])
            if current and current["output_sha256"] == digest and current["experiment_id"] == payload["experiment_id"] and current["evaluation_id"] == payload["evaluation_id"]:
                return {"disposition": "already_published", "champion": current, "output_path": str(target), "publication_event_id": "evt-publication-" + str(current["champion_version"])}
            version = current["champion_version"] if current else 0
            if version != payload["expected_champion_version"]:
                raise AgentError("VERSION_CONFLICT", "Published champion changed; reread and compare", retryable=True)
            if current and not is_better(evaluation["primary_metric"], current["primary_metric"]):
                return {"disposition": "skipped_not_better", "champion": current, "output_path": str(target), "publication_event_id": "evt-publication-" + str(version)}
            champion = {"experiment_id": payload["experiment_id"], "evaluation_id": payload["evaluation_id"],
                        "primary_metric": evaluation["primary_metric"], "submission_artifact_id": payload["candidate_artifact_id"],
                        "output_sha256": digest, "champion_version": version + 1}
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".submission-", suffix=".tmp", delete=False) as handle:
                    temporary = Path(handle.name)
                    with source.open("rb") as incoming:
                        shutil.copyfileobj(incoming, handle)
                    os.chmod(temporary, 0o644)  # Readable by the platform evaluator.
                    handle.flush()
                    os.fsync(handle.fileno())
                if file_sha256(temporary) != digest:
                    raise AgentError("CANDIDATE_CHANGED", "Staged file hash mismatch")
                journal_id = self.store.prepare_publication({"champion": champion, "target_path": str(target), "sha256": digest})
                os.replace(temporary, target)
                self._sync_directory(target.parent)
                self.store.finish_publication(journal_id, champion)
            except AgentError:
                raise
            except Exception as exc:
                raise AgentError("OUTPUT_WRITE_FAILED", str(exc)) from exc
            finally:
                if temporary and temporary.exists():
                    temporary.unlink()
            self.log.event("submission.published", experiment_id=champion["experiment_id"], output_path=str(target),
                           score=champion["primary_metric"]["value"], champion_version=champion["champion_version"])
            return {"disposition": "published", "champion": champion, "output_path": str(target), "publication_event_id": "evt-" + journal_id}

    @staticmethod
    def _sync_directory(path):
        if os.name == "posix":
            handle = os.open(path, os.O_RDONLY)
            try:
                os.fsync(handle)
            finally:
                os.close(handle)

    def recover(self):
        for journal_id, record in self.store.pending_publications():
            target = Path(record["target_path"])
            current = self.store.champion()
            if target.is_file() and file_sha256(target) == record["sha256"]:
                if not current or record["champion"]["champion_version"] > current["champion_version"]:
                    self.store.finish_publication(journal_id, record["champion"])
                    self.log.event("submission.publication_recovered", journal_id=journal_id)

    def require_output(self):
        self.recover()
        champion = self.store.champion()
        target = Path(self.spec["output"]["path"])
        if not champion or not target.is_file() or file_sha256(target) != champion["output_sha256"]:
            raise AgentError("NO_VALID_SUBMISSION", "No verified published output")
        return champion
