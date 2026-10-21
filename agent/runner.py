"""Supervises a whole process group and limits resident memory on Linux."""
from dataclasses import dataclass, field
from pathlib import Path
import json
import os
import signal
import subprocess
import sys
import threading
import time
import uuid

from agent.config import ROOT
from agent.errors import AgentError
from agent.contracts import canonical


def group_processes(pgid):
    if sys.platform.startswith("linux"):
        result = []
        for path in Path("/proc").iterdir():
            if not path.name.isdigit():
                continue
            try:
                stat = (path / "stat").read_text()
                tail = stat[stat.rindex(")") + 2:].split()
                # getpgid is authoritative across PID namespaces / procfs views.
                # Some managed executors expose host IDs inside stat fields.
                if os.getpgid(int(path.name)) == pgid and tail[0] != "Z":
                    result.append(int(path.name))
            except (OSError, ValueError, IndexError):
                continue
        # A restricted procfs may omit our child from directory enumeration.
        if pgid not in result:
            try:
                if os.getpgid(pgid) == pgid:
                    state_path = Path(f"/proc/{pgid}/stat")
                    stat = state_path.read_text() if state_path.is_file() else ""
                    state = stat[stat.rindex(")") + 2:].split()[0] if stat else None
                    if state != "Z":
                        result.append(pgid)
            except (OSError, ValueError, IndexError):
                pass
        return result
    try:
        import psutil
        proc = psutil.Process(pgid)
        return [proc.pid] + [p.pid for p in proc.children(recursive=True) if p.is_running()]
    except ImportError as exc:
        raise AgentError("RESOURCE_MONITOR_UNAVAILABLE", "Use Linux/Docker or install approved psutil") from exc
    except Exception:
        return []


def resident_mb(pgid):
    pids = group_processes(pgid)
    if sys.platform.startswith("linux"):
        total = 0
        for pid in pids:
            try:
                total += int(Path(f"/proc/{pid}/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")
            except (OSError, ValueError, IndexError):
                pass
        return total / (1024 * 1024)
    import psutil
    total = 0
    for pid in pids:
        try:
            total += psutil.Process(pid).memory_info().rss
        except psutil.Error:
            pass
    return total / (1024 * 1024)


@dataclass
class Job:
    job_id: str
    experiment_id: str
    attempt_no: int
    phase: str
    lease: dict
    process: subprocess.Popen
    directory: Path
    started: float
    threads: list = field(default_factory=list)
    peak_memory_mb: float = 0.0
    forced_state: str | None = None
    forced_error: dict | None = None
    reported: bool = False
    last_heartbeat: float = 0.0
    event: dict | None = None
    prediction_path: Path | None = None


class Runner:
    def __init__(self, artifacts, config, budget, log):
        self.artifacts, self.config, self.budget, self.log = artifacts, config, budget, log
        self.jobs = {}

    def _drain(self, stream, path):
        # Redact before bytes reach disk. Retain a bounded log plus marker.
        maximum = 8 * 1024 * 1024
        count = 0
        with path.open("w", encoding="utf-8") as out:
            while True:
                chunk = stream.readline(4096)
                if not chunk:
                    break
                chunk = self.log.redactor.text(chunk)
                if chunk.startswith("AUTOML_WORKER "):
                    try:
                        fields = json.loads(chunk[len("AUTOML_WORKER "):])
                        stage = fields.pop("stage")
                        self.log.event("worker."+stage, job_id=path.parent.name, experiment_id=path.parent.parent.name, **fields)
                    except (ValueError, KeyError, TypeError):
                        pass
                if count < maximum:
                    remaining = maximum - count
                    text = chunk[:remaining]
                    out.write(text)
                    out.flush()
                    count += len(text)
                    if count >= maximum:
                        out.write("\n[OUTPUT LIMIT REACHED; remaining output drained]\n")
            stream.close()

    def _start(self, payload, task_spec, experiment, phase):
        lease = payload["lease"]
        held = self.budget.leases.get(lease["lease_id"])
        if not held or held["lease"]["fencing_token"] != lease["fencing_token"]:
            raise AgentError("STALE_LEASE", "Cannot start without a live lease")
        ident = "job-" + uuid.uuid4().hex[:12]
        directory = self.artifacts.safe_path("experiments/" + experiment["experiment_id"] + "/" + ident)
        directory.mkdir(parents=True, exist_ok=False)
        prediction_path = directory / "validation.csv" if phase == "train" else self.artifacts.safe_path(payload["output_relative_path"])
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        job_spec = {"job_id": ident, "phase": phase, "task_spec": task_spec,
                    "train_path": str(self.artifacts.resolve(task_spec["train_artifact_id"])),
                    "test_path": str(self.artifacts.resolve(task_spec["test_artifact_id"])),
                    "code_path": str(self.artifacts.resolve(experiment["plan"]["code_artifact_id"])),
                    "hyperparameters": experiment["plan"]["model"]["hyperparameters"],
                    "seed": experiment["plan"]["seed"], "cpu_threads": lease["cpu_threads"],
                    "memory_limit_mb": lease["memory_limit_mb"], "timeout_seconds": lease["timeout_seconds"],
                    "prediction_path": str(prediction_path), "model_path": str(directory / "pipeline.joblib"),
                    "result_path": str(directory / "result.json")}
        if phase == "train":
            job_spec.update(train_indices_path=str(self.artifacts.resolve(payload["train_rows_artifact_id"])),
                            valid_indices_path=str(self.artifacts.resolve(payload["valid_rows_artifact_id"])))
        request_path = directory / "job.json"
        request_path.write_text(json.dumps(job_spec, ensure_ascii=False), encoding="utf-8")
        self.artifacts.register(request_path, "worker_job")
        # Explicit allowlist: no OPENAI_*, tokens, proxy credentials or .env.
        env = {k: os.environ[k] for k in ("PATH", "SYSTEMROOT", "WINDIR", "TEMP", "TMP", "TMPDIR", "LD_LIBRARY_PATH") if k in os.environ}
        env.update(PYTHONPATH=str(ROOT), PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8",
                   OMP_NUM_THREADS=str(lease["cpu_threads"]), OPENBLAS_NUM_THREADS=str(lease["cpu_threads"]),
                   MKL_NUM_THREADS=str(lease["cpu_threads"]), NUMEXPR_NUM_THREADS=str(lease["cpu_threads"]),
                   VECLIB_MAXIMUM_THREADS=str(lease["cpu_threads"]))
        kwargs = {"start_new_session": True} if os.name == "posix" else {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
        proc = subprocess.Popen([sys.executable, "-u", "-m", "agent.worker", str(request_path)],
                                cwd=directory, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, encoding="utf-8", errors="replace", **kwargs)
        job = Job(ident, experiment["experiment_id"], payload.get("attempt_no", experiment["attempt_no"]), phase,
                  dict(lease), proc, directory, time.monotonic(), last_heartbeat=time.monotonic())
        job.prediction_path = prediction_path
        self.jobs[ident] = job  # Track before starting drain threads, for cleanup on partial start.
        for name, stream in [("stdout.log", proc.stdout), ("stderr.log", proc.stderr)]:
            thread = threading.Thread(target=self._drain, args=(stream, directory / name), daemon=True)
            thread.start()
            job.threads.append(thread)
        self.log.event("job.started", job_id=ident, experiment_id=job.experiment_id, phase=phase,
                       pid=proc.pid, cpu_threads=lease["cpu_threads"], memory_limit_mb=lease["memory_limit_mb"],
                       recipe_id=experiment["plan"]["recipe_id"], hyperparameters=job_spec["hyperparameters"],
                       timeout_seconds=lease["timeout_seconds"], stderr_log_path=str(directory/"stderr.log"))
        return job

    def start_train(self, payload, spec, experiment):
        if "plan" in payload and canonical(payload["plan"]) != canonical(experiment["plan"]):
            raise AgentError("INVALID_PLAN", "C14 plan differs from the reserved experiment")
        job = self._start(payload, spec, experiment, "train")
        return {"job_id": job.job_id, "state": "accepted", "event_target": "runtime.on_job_event", "heartbeat_seconds": 10}

    def start_refit(self, payload, spec, experiment):
        if payload["code_artifact_id"] != experiment["plan"]["code_artifact_id"] or payload["predict_artifact_id"] != spec["test_artifact_id"]:
            raise AgentError("INVALID_PLAN", "Refit code or prediction data differs from the evaluated experiment")
        job = self._start(payload, spec, experiment, "refit")
        return {"job_id": job.job_id, "state": "accepted", "completion_contract": "C15"}

    def cancel(self, payload):
        job = self.jobs.get(payload["job_id"])
        if not job or job.experiment_id != payload["experiment_id"]:
            raise AgentError("UNKNOWN_JOB", payload["job_id"])
        if job.attempt_no != payload["attempt_no"] or job.lease["fencing_token"] != payload["fencing_token"]:
            raise AgentError("STALE_LEASE", "Cannot cancel another attempt")
        pids = group_processes(job.process.pid)
        if job.process.poll() is None and job.process.pid not in pids:
            pids.append(job.process.pid)  # Never claim a live Popen child is stopped.
        disposition = "already_stopped" if not pids else "stopped"
        if pids:
            if os.name == "posix":
                try:
                    os.killpg(job.process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            else:
                job.process.terminate()
            until = time.monotonic() + min(payload["grace_seconds"], max(0, self.budget.remaining() - 1))
            while group_processes(job.process.pid) and time.monotonic() < until:
                time.sleep(0.05)
            if group_processes(job.process.pid) or job.process.poll() is None:
                if os.name == "posix":
                    try:
                        os.killpg(job.process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                else:
                    subprocess.run(["taskkill", "/PID", str(job.process.pid), "/T", "/F"], capture_output=True, timeout=3)
            try:
                job.process.wait(timeout=3)
            except subprocess.TimeoutExpired as exc:
                raise AgentError("PROCESS_STOP_FAILED", "Worker did not exit") from exc
            until = time.monotonic() + 2
            while group_processes(job.process.pid) and time.monotonic() < until:
                time.sleep(0.05)
            if group_processes(job.process.pid):
                raise AgentError("PROCESS_STOP_FAILED", "Child processes remain alive")
            job.forced_state = payload["terminal_state"]
            job.forced_error = AgentError("JOB_CANCELLED" if job.forced_state == "cancelled" else "JOB_TIMEOUT", payload["reason"]).as_dict()
        terminal = job.forced_state or ("succeeded" if job.process.poll() == 0 else "failed")
        return {"job_id": job.job_id, "disposition": disposition, "terminal_state": terminal,
                "process_tree_stopped": True, "remaining_processes": 0,
                "completion_event_id": "evt-" + job.job_id + "-terminal"}

    def poll(self, job):
        job.peak_memory_mb = max(job.peak_memory_mb, resident_mb(job.process.pid))
        if job.process.poll() is None or group_processes(job.process.pid):
            return None
        if job.event:
            return job.event
        for thread in job.threads:
            thread.join(timeout=2)
        result_file = job.directory / "result.json"
        try:
            result = json.loads(result_file.read_text(encoding="utf-8")) if result_file.is_file() else {}
        except (OSError, json.JSONDecodeError):
            result = {}
        success = job.process.returncode == 0 and result.get("success") is True
        state = job.forced_state or ("succeeded" if success else "timed_out" if result.get("error_type") == "JOB_TIMEOUT" else "failed")
        job.peak_memory_mb = max(job.peak_memory_mb, float(result.get("peak_memory_mb", 0)))
        outputs = []
        output_files = set(job.directory.iterdir())
        if job.prediction_path and job.prediction_path.is_file():
            output_files.add(job.prediction_path)
        for path in sorted(output_files):
            if path.suffix not in {".csv", ".log", ".joblib"}:
                continue
            kind = ("validation_predictions" if job.phase == "train" else "submission") if path == job.prediction_path else "fitted_pipeline" if path.suffix == ".joblib" else "execution_log"
            art = self.artifacts.register(path, kind, "text/csv" if path.suffix == ".csv" else "application/octet-stream")
            outputs.append(art["artifact_id"])
        err = None if state == "succeeded" else job.forced_error or AgentError(result.get("error_type") if result.get("error_type") in {"JOB_TIMEOUT", "JOB_MEMORY_LIMIT"} else "WORKER_FAILED", result.get("error", "Worker exited without a successful result"), details={"exit_code": job.process.returncode}).as_dict()
        event = {"event_id": "evt-" + job.job_id + "-terminal", "job_id": job.job_id,
                 "experiment_id": job.experiment_id, "attempt_no": job.attempt_no,
                 "lease_id": job.lease["lease_id"], "fencing_token": job.lease["fencing_token"], "state": state,
                 "output_artifact_ids": outputs, "duration_seconds": round(time.monotonic() - job.started, 4),
                 "peak_memory_mb": round(job.peak_memory_mb, 2), "error": err}
        job.event = event
        self.log.event("job.finished", level="INFO" if state == "succeeded" else "ERROR", job_id=job.job_id, experiment_id=job.experiment_id,
                       phase=job.phase, state=state, duration_seconds=event["duration_seconds"], peak_memory_mb=event["peak_memory_mb"],
                       exit_code=job.process.returncode, error=err, stderr_log_path=str(job.directory/"stderr.log"))
        return event

    def active(self):
        return [job for job in self.jobs.values() if not job.reported]
