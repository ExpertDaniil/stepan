"""Supervises a whole process group and limits resident memory on Linux."""
from dataclasses import dataclass, field
from pathlib import Path
import json
import hashlib
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
from agent.artifacts import file_sha256


def frozen_execution(artifacts, experiment, task_spec, phase):
    """Reject source/spec/runtime drift before fitting or publishing an evaluated plan."""
    from agent.store import Store
    code_id = experiment['plan']['code_artifact_id']
    actual_code_hash = file_sha256(artifacts.resolve(code_id))
    if actual_code_hash != artifacts.meta(code_id)['sha256']:
        raise AgentError('CANDIDATE_CHANGED', 'Pipeline source bytes changed after registration')
    core = Store.fingerprint_core(experiment['plan'], actual_code_hash)
    if hashlib.sha256(canonical(core).encode()).hexdigest() != experiment['fingerprint']:
        raise AgentError('CANDIDATE_CHANGED', 'Effective pipeline no longer matches the reserved experiment')
    runtime = ['agent/worker.py', 'agent/ml.py', 'agent/preprocessing.py', 'agent/predictions.py',
               'agent/feature_engineering.py', 'agent/feature_schema.py', 'agent/supervised.py', 'agent/data_transport.py', 'agent/fit_diagnostics.py',
               'tools/data.py', 'tools/runtime_guard.py']
    value = {'plan': experiment['plan'], 'task_spec': task_spec, 'code_sha256': actual_code_hash,
             'runtime_sha256': {name: file_sha256(ROOT/name) for name in runtime}}
    protocol_path=artifacts.safe_path('validation/'+experiment['plan']['validation_id']+'.protocol.json')
    value['validation_protocol_sha256']=file_sha256(protocol_path) if protocol_path.is_file() else None
    path = artifacts.safe_path('experiments/'+experiment['experiment_id']+'/execution.json')
    if path.exists():
        if file_sha256(path) != artifacts.meta('execution-'+experiment['experiment_id'])['sha256']:
            raise AgentError('CANDIDATE_CHANGED', 'Frozen execution snapshot bytes changed')
        if json.loads(path.read_text(encoding='utf-8')) != value:
            raise AgentError('CANDIDATE_CHANGED', 'Refit must reproduce the frozen evaluated source/spec/runtime')
    elif phase == 'refit':
        raise AgentError('MISSING_EXECUTION_SNAPSHOT', 'No trusted training execution snapshot for this candidate')
    else:
        artifacts.write_json(path.relative_to(artifacts.root).as_posix(), value, 'execution_snapshot',
                             ident='execution-'+experiment['experiment_id'])
    return value


def _linux_group_members(pgid):
    """Return namespace-local PIDs with their actual procfs paths, excluding zombies."""
    result = []
    status = Path("/proc/self/status").read_text()
    ns_ids = next((line.split()[1:] for line in status.splitlines() if line.startswith("NSpid:")), [str(os.getpid())])
    depth = len(ns_ids) - 1
    namespace = os.readlink("/proc/self/ns/pid") if depth else None
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():
            continue
        try:
            stat = (path / "stat").read_text()
            tail = stat[stat.rindex(")") + 2:].split()
            if tail[0] == "Z":
                continue
            pid = int(path.name)
            if depth:
                # Host-mounted procfs can expose IDs outside our PID namespace.
                # Check namespace identity before mapping; sibling PIDs can collide.
                if os.readlink(path / "ns/pid") != namespace:
                    continue
                status = (path / "status").read_text()
                ids = next(line.split()[1:] for line in status.splitlines() if line.startswith("NSpid:"))
                pid = int(ids[depth])
            if os.getpgid(pid) == pgid:
                result.append((pid, path))
        except (OSError, ValueError, IndexError, StopIteration):
            continue
    # A restricted procfs may omit our directly supervised worker.
    if pgid not in {pid for pid, _ in result}:
        try:
            if os.getpgid(pgid) == pgid:
                path = Path(f"/proc/{pgid}")
                stat = (path / "stat").read_text() if path.is_dir() else ""
                state = stat[stat.rindex(")") + 2:].split()[0] if stat else None
                if state != "Z":
                    result.append((pgid, path))
        except (OSError, ValueError, IndexError):
            pass
    return result


def group_processes(pgid):
    if sys.platform.startswith("linux"):
        return [pid for pid, _ in _linux_group_members(pgid)]
    try:
        import psutil
        proc = psutil.Process(pgid)
        return [proc.pid] + [p.pid for p in proc.children(recursive=True) if p.is_running()]
    except ImportError as exc:
        raise AgentError("RESOURCE_MONITOR_UNAVAILABLE", "Use Linux/Docker or install approved psutil") from exc
    except Exception:
        return []


def resident_mb(pgid):
    if sys.platform.startswith("linux"):
        total = 0
        for _, path in _linux_group_members(pgid):
            try:
                total += int((path / "statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")
            except (OSError, ValueError, IndexError):
                pass
        return total / (1024 * 1024)
    import psutil
    total = 0
    for pid in group_processes(pgid):
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
    diagnostic: bool = False
    lease_owner_id: str | None = None


class Runner:
    def __init__(self, artifacts, config, budget, log):
        self.artifacts, self.config, self.budget, self.log = artifacts, config, budget, log
        self.jobs = {}
        self.model_snapshot_reservations = {}

    def snapshot_allowance(self):
        stored = sum(path.stat().st_size for path in self.artifacts.root.rglob("*.joblib") if path.is_file())
        reserved = sum(self.model_snapshot_reservations.values())
        return min(self.config.model_snapshot_max_bytes, max(0, self.config.model_snapshot_total_bytes-stored-reserved))

    def _drain(self, stream, path):
        # Redact before bytes reach disk. Retain a bounded log plus marker.
        maximum = self.config.worker_log_max_bytes
        marker = b"\n[OUTPUT LIMIT REACHED; remaining output drained]\n"
        count = 0
        truncated = False
        with path.open("wb") as out:
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
                if not truncated:
                    encoded = chunk.encode("utf-8")
                    remaining = maximum-len(marker)-count
                    text = encoded[:remaining].decode("utf-8", errors="ignore").encode("utf-8")
                    out.write(text)
                    out.flush()
                    count += len(text)
                    if len(encoded) > len(text):
                        out.write(marker);out.flush();truncated = True
                        self.log.event("job.log_truncated", job_id=path.parent.name, stream=path.name, max_bytes=maximum)
            stream.close()

    def _start(self, payload, task_spec, experiment, phase):
        lease = payload["lease"]
        held = self.budget.leases.get(lease["lease_id"])
        if not held or held["lease"]["fencing_token"] != lease["fencing_token"]:
            raise AgentError("STALE_LEASE", "Cannot start without a live lease")
        frozen = frozen_execution(self.artifacts, experiment, task_spec, phase)
        ident = "job-" + uuid.uuid4().hex[:12]
        directory = self.artifacts.safe_path("experiments/" + experiment["experiment_id"] + "/" + ident)
        directory.mkdir(parents=True, exist_ok=False)
        prediction_path = directory / "validation.csv" if phase == "train" else self.artifacts.safe_path(payload["output_relative_path"])
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        job_spec = {"job_id": ident, "phase": phase, "task_spec": task_spec,
                    "guard_generated": self.config.research_mode == 'adaptive' and experiment['plan']['model']['family'] == 'custom',
                    "train_path": str(self.artifacts.resolve(task_spec["train_artifact_id"])),
                    "test_path": str(self.artifacts.resolve(task_spec["test_artifact_id"])),
                    "code_path": str(self.artifacts.resolve(experiment["plan"]["code_artifact_id"])),
                    'code_sha256': frozen['code_sha256'],
                    "hyperparameters": experiment["plan"]["model"]["hyperparameters"],
                    "seed": experiment["plan"]["seed"], "cpu_threads": lease["cpu_threads"],
                    "memory_limit_mb": lease["memory_limit_mb"], "timeout_seconds": lease["timeout_seconds"],
                    "prediction_path": str(prediction_path), "model_path": str(directory / "pipeline.joblib"),
                    "model_snapshot_max_bytes": self.snapshot_allowance(),
                    "result_path": str(directory / "result.json")}
        job_spec['preflight'] = phase == 'train' and experiment['plan']['plan_id'].startswith('candidate-')
        protocol_path=self.artifacts.safe_path('validation/'+experiment['plan']['validation_id']+'.protocol.json')
        if protocol_path.is_file():
            proposal=json.loads(protocol_path.read_text(encoding='utf-8'))['proposal']
            job_spec['training_policy']={'strategy':proposal['strategy'],'seed':proposal['seed'],
                'column':proposal['parameters'].get('column')}
        if phase == "train":
            job_spec.update(train_indices_path=str(self.artifacts.resolve(payload["train_rows_artifact_id"])),
                            valid_indices_path=str(self.artifacts.resolve(payload["valid_rows_artifact_id"])))
        if payload.get('diagnostic'):
            job_spec['diagnostic'] = payload['diagnostic']
            job_spec['model_snapshot_max_bytes'] = 0
        from agent.data_transport import transport
        tables = transport(self.artifacts)
        job_spec['input_tables'] = {'train': tables.prepare(task_spec['train_artifact_id'], task_spec)}
        if phase == 'refit':
            job_spec['input_tables']['test'] = tables.prepare(task_spec['test_artifact_id'])
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
        job.diagnostic = bool(payload.get('diagnostic'))
        job.lease_owner_id = payload.get('lease_owner_id')
        self.jobs[ident] = job  # Track before starting drain threads, for cleanup on partial start.
        self.model_snapshot_reservations[ident] = job_spec["model_snapshot_max_bytes"]
        for name, stream in [("stdout.log", proc.stdout), ("stderr.log", proc.stderr)]:
            thread = threading.Thread(target=self._drain, args=(stream, directory / name), daemon=True)
            thread.start()
            job.threads.append(thread)
        self.log.event("job.started", job_id=ident, experiment_id=job.experiment_id, phase=phase,
                       purpose='fit_oof_diagnostics' if job.diagnostic else 'candidate',
                       pid=proc.pid, cpu_threads=lease["cpu_threads"], memory_limit_mb=lease["memory_limit_mb"],
                       recipe_id=experiment["plan"]["recipe_id"], hyperparameters=job_spec["hyperparameters"],
                       timeout_seconds=lease["timeout_seconds"], stderr_log_path=str(directory/"stderr.log"))
        return job

    def start_diagnostics(self, payload, spec, experiment):
        """Trusted fit-only probe; its completion must never go through C15/C16."""
        return self._start(payload, spec, experiment, 'train')

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

    def _stop_process_tree(self, job, grace_seconds):
        """Stop descendants without deciding whether completed computation succeeded."""
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
            until = time.monotonic() + min(grace_seconds, max(0, self.budget.remaining() - 1))
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
        return disposition

    def cancel(self, payload):
        job = self.jobs.get(payload["job_id"])
        if not job or job.experiment_id != payload["experiment_id"]:
            raise AgentError("UNKNOWN_JOB", payload["job_id"])
        if job.attempt_no != payload["attempt_no"] or job.lease["fencing_token"] != payload["fencing_token"]:
            raise AgentError("STALE_LEASE", "Cannot cancel another attempt")
        disposition = self._stop_process_tree(job, payload["grace_seconds"])
        if disposition == "stopped":
            job.forced_state = payload["terminal_state"]
            job.forced_error = AgentError("JOB_CANCELLED" if job.forced_state == "cancelled" else "JOB_TIMEOUT", payload["reason"]).as_dict()
        terminal = job.forced_state or ("succeeded" if job.process.poll() == 0 else "failed")
        return {"job_id": job.job_id, "disposition": disposition, "terminal_state": terminal,
                "process_tree_stopped": True, "remaining_processes": 0,
                "completion_event_id": "evt-" + job.job_id + "-terminal"}

    def poll(self, job):
        job.peak_memory_mb = max(job.peak_memory_mb, resident_mb(job.process.pid))
        if job.process.poll() is None:
            return None
        descendants = group_processes(job.process.pid)
        if descendants:
            # Libraries may leave helper processes after worker exit. Reap those
            # before releasing the lease; preserve exit code + result.json as the
            # authority for success rather than turning housekeeping into cancel.
            self.log.event("job.descendants_cleanup_started", job_id=job.job_id, experiment_id=job.experiment_id,
                           phase=job.phase, exit_code=job.process.returncode, remaining_processes=len(descendants))
            self._stop_process_tree(job, 0.2)
            self.log.event("job.descendants_cleanup_completed", job_id=job.job_id, experiment_id=job.experiment_id,
                           phase=job.phase, exit_code=job.process.returncode, process_tree_stopped=True, remaining_processes=0)
        if job.event:
            return job.event
        for thread in job.threads:
            thread.join(timeout=2)
        result_file = job.directory / "result.json"
        try:
            result = json.loads(result_file.read_text(encoding="utf-8")) if result_file.is_file() else {}
        except (OSError, json.JSONDecodeError):
            result = {}
        (job.directory / "pipeline.joblib.tmp").unlink(missing_ok=True)
        self.model_snapshot_reservations.pop(job.job_id, None)
        if "model_snapshot" in result:
            self.log.event("job.model_snapshot", job_id=job.job_id, experiment_id=job.experiment_id,
                           phase=job.phase, **result["model_snapshot"])
        success = job.process.returncode == 0 and result.get("success") is True
        state = job.forced_state or ("succeeded" if success else "timed_out" if result.get("error_type") == "JOB_TIMEOUT" else "failed")
        job.peak_memory_mb = max(job.peak_memory_mb, float(result.get("peak_memory_mb", 0)))
        outputs = []
        output_files = set(job.directory.iterdir())
        if job.prediction_path and job.prediction_path.is_file():
            output_files.add(job.prediction_path)
        for path in sorted(output_files):
            if path.suffix not in {".csv", ".log", ".joblib"} and path.name != "features.json":
                continue
            kind = ('feature_execution' if path.name == 'features.json' else
                    ('fit_oof_predictions' if job.diagnostic else 'validation_predictions' if job.phase == 'train' else 'submission') if path == job.prediction_path else
                    'fitted_pipeline' if path.suffix == '.joblib' else 'execution_log')
            media = 'application/json' if path.name == 'features.json' else 'text/csv' if path.suffix == '.csv' else 'application/octet-stream'
            art = self.artifacts.register(path, kind, media)
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
