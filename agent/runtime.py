from pathlib import Path
import argparse
import json
import os
import signal
import sys
import time
import uuid

from agent.artifacts import Artifacts
from agent.budget import Budget
from agent.catalog import Catalog
from agent.config import Config, ROOT
from agent.context import demo_metadata, output_schemas
from agent.contracts import ContractBus
from agent.errors import AgentError
from agent.evaluation import Evaluator
from agent.llm.gateway import Gateway
from agent.observability.logging import RunLogger
from agent.orchestrator import Orchestrator
from agent.researcher import Researcher
from agent.runner import Runner
from agent.store import Store
from agent.submission import Submission
from agent.task import resolve_spec
from agent.task_interpreter import TaskInterpreter
from tools.data import profile_data
from tools.files import read_task


def load_env_file(path):
    """Explicit file only. No shell evaluation or variable interpolation."""
    import re
    for no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:]
        key, sep, value = line.partition("=")
        if not sep or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key.strip()):
            raise AgentError("INVALID_CONFIG", f"Invalid .env line {no}")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(key.strip(), value)


def bootstrap(config, started, task_path, run_id, root, log):
    artifacts = Artifacts(root, run_id, log)
    bus = ContractBus(run_id, log)
    budget = Budget(config, started, log)
    store = Store(root / "state.sqlite", run_id, config.public_dict(), log)
    gateway = None
    controller = None
    submission = None
    runner = Runner(artifacts, config, budget, log)
    try:
        bus.call("C00", {"task_path": str(task_path), "work_dir": str(root), "time_budget_seconds": config.time_budget_seconds,
                          "max_workers": config.max_workers},
                 lambda p: {"state": "initializing", "task_path": p["task_path"], "work_dir": p["work_dir"], "time_budget_seconds": p["time_budget_seconds"]}, key=run_id)
        task = bus.call("C01", {"task_path": str(task_path)}, lambda p: read_task(p, artifacts, log))["data"]
        profile = bus.call("C03", {"candidate_paths": task["candidate_paths"], "max_report_bytes": 24000},
                           lambda p: profile_data(p, artifacts))["data"]
        schemas = output_schemas(bus)
        artifacts.write_json("task/agent-output-schemas.json", schemas, "llm_schemas")
        gateway = Gateway(config, budget, artifacts, bus, log, schemas)
        demo = demo_metadata(task["task_text"]) if config.llm_mode == "mock" else None

        interpreter = TaskInterpreter(config, artifacts, bus, gateway, task, profile, demo)
        interpretation = bus.call("C02", {"task_text_artifact_id": task["task_text_artifact_id"],
                                           "candidate_paths": task["candidate_paths"], "tools_catalog_version": "tools-v1"}, interpreter.interpret)
        spec_result = bus.call("C04", {"interpretation_request_id": interpretation["request_id"],
                                        "profile_artifact_id": profile["profile_artifact_id"], "dataset_id": profile["dataset_id"]},
                               lambda p: {"task_spec": resolve_spec(bus.responses[p["interpretation_request_id"]]["data"]["draft"], profile,
                                                                   task["task_root"], artifacts, bus.schemas, task_text=task['task_text']),
                                          "adapter_id": interpretation["data"]["draft"]["task_kind"] + "-v1",
                                          "metric_implementation_id": "sklearn-metrics-v2"})["data"]
        spec = spec_result["task_spec"]
        catalog = Catalog(spec["task_kind"], config)
        bus.call("C06", {"task_spec_id": spec["task_spec_id"], "adapter_id": spec_result["adapter_id"], "cpu_only": True},
                 lambda p: catalog.select(p, artifacts, bus.schemas))
        evaluator = Evaluator(spec, artifacts, store, log)
        proposal = interpreter.choose_validation(spec)
        validation = bus.call("C05", {"task_spec_id": spec["task_spec_id"], "dataset_id": spec["dataset_id"], "proposal": proposal}, evaluator.create_protocol)["data"]
        submission = Submission(spec, artifacts, store, log)
        researcher = Researcher(bus, gateway, artifacts, spec, catalog, config, log,
                                feature_columns=artifacts.read_json(profile['profile_artifact_id'])['train_columns'])
        controller = Orchestrator(config=config, bus=bus, budget=budget, artifacts=artifacts, store=store, runner=runner,
                                  evaluator=evaluator, submission=submission, researcher=researcher, spec=spec,
                                  validation=validation, catalog=catalog, task_text=task["task_text"], profile=artifacts.read_json(profile["profile_artifact_id"]), log=log, demo=demo, task_root=task["task_root"])
        try:
            controller.run()
        except Exception as exc:
            controller.degraded = True
            budget.end_data_research('research_interrupted')
            log.exception("run.research_interrupted", exc)
            if controller.graph:
                controller.graph.close_search("research_interrupted")
            controller.cleanup()
            # A fallback is acceptable only if it was actually published and verified.
            submission.require_output()

        if controller.graph:
            try:
                if not store.db.execute("SELECT 1 FROM g_metadata WHERE run_id=? AND name='latest_export'", (run_id,)).fetchone():
                    controller.graph.invoke("G11", {"task_spec_id": spec["task_spec_id"], "validation_id": validation["validation_id"],
                        "selection_policy_id": "same-protocol-primary-v1", "scope": "final"}, key="fallback-final-export")
            except Exception as exc:
                controller.degraded = True
                log.exception("graph.export_failed", exc)

        def finish(payload):
            if runner.active():
                raise AgentError("ACTIVE_JOBS", "Cannot finish while workers are active")
            champion = submission.require_output()
            state = "degraded" if controller.degraded else "completed"
            store.set_run_state(state)
            summary = write_summary(root, store, budget, config, champion, state)
            art = artifacts.register(summary, "run_summary", "text/markdown")
            return {"state": state, "exit_code": 0, "output_path": spec["output"]["path"],
                    "champion_experiment_id": champion["experiment_id"], "summary_artifact_id": art["artifact_id"]}

        result = bus.call("C23", {"reason": "research_finished", "require_valid_output": True}, finish, key=run_id + "/finish")["data"]
        log.event("run.finished", level="WARNING" if result["state"] == "degraded" else "INFO", state=result["state"],
                  output_path=result["output_path"], run_directory=str(root), token_budget=budget.token_snapshot(),
                  metric_id=submission.require_output()["primary_metric"]["metric_id"],
                  score=submission.require_output()["primary_metric"]["value"])
        return 0
    except BaseException as exc:
        if controller:
            try:
                controller.cleanup()
            except Exception as cleanup_error:
                log.exception("run.cleanup_error", cleanup_error)
        store.set_run_state("failed")
        log.exception("run.failed", exc, code=getattr(exc, "code", "UNHANDLED_ERROR"))
        return 130 if isinstance(exc, KeyboardInterrupt) else 1
    finally:
        if gateway:
            gateway.close()
        store.close()


def write_summary(root, store, budget, config, champion, state):
    rows = store.db.execute("SELECT experiment_id,state,plan,error FROM experiments WHERE run_id=? ORDER BY rowid", (store.run_id,)).fetchall()
    lines = ["# Run report\n", f"State: {state}\n", f"Run: {store.run_id}\n", f"LLM mode: {config.llm_mode}\n",
             f"Elapsed seconds: {time.monotonic() - budget.started:.2f}\n", f"LLM requests: {budget.calls}\n",
             f"Published experiment: {champion['experiment_id']}\n", f"Published metric: {champion['primary_metric']['value']}\n",
             f"Metric: {champion['primary_metric']['metric_id']} ({champion['primary_metric']['direction']})\n",
             f"Research mode: {config.research_mode}\n", f"Token accounting: {json.dumps(budget.token_snapshot())}\n",
             f"Budget profile: {config.profile}; numeric limits: calls={config.llm_max_calls}, output={config.llm_max_total_output_tokens}\n",
             "\n| Experiment | State | Recipe | Metric | Hypothesis |\n|---|---|---|---|---|\n"]
    for row in rows:
        plan = json.loads(row["plan"])
        ev = store.experiment_evaluation(row["experiment_id"])
        score = str(ev["primary_metric"]["value"]) if ev else "—"
        hypothesis = plan["hypothesis"].replace("|", "/").replace("\n", " ")
        lines.append(f"| {row['experiment_id']} | {row['state']} | {plan['recipe_id']} | {score} | {hypothesis} |\n")
    lines.append("\nFull redacted protocol: events.jsonl, messages/, llm/, contexts/, artifacts-index.json.\n")
    confirmation = root/'validation/final-confirmation.json'
    if confirmation.is_file():
        report = json.loads(confirmation.read_text(encoding='utf-8'))
        lines.append('\nFinal comparison: validation/final-confirmation.json; chosen experiment: '+report['chosen_experiment_id']+'.\n')
        lines.append(report['limitations']+'\n')
    if config.graph_enabled:
        row = store.db.execute("SELECT value FROM g_metadata WHERE run_id=? AND name='latest_export'", (store.run_id,)).fetchone()
        branches = store.db.execute("SELECT branch_id,decision,state,revision FROM g_branches WHERE run_id=? ORDER BY rowid", (store.run_id,)).fetchall()
        lines.append("\n## Research graph\n\nEqual peer heads and branch decisions are stored in state.sqlite (g_*).\n")
        if row:
            lines.append("\nGraph/ranking/events export: "+json.loads(row[0])["folder"]+"/\n")
        lines.append("\n| Branch | Decision | State | Revision |\n|---|---|---|---|\n")
        for branch in branches:
            lines.append("| "+" | ".join(map(str, branch))+" |\n")
    path = root / "summary.md"
    path.write_text("".join(lines), encoding="utf-8")
    return path


def main(argv=None, *, started_monotonic=None):
    started = time.monotonic() if started_monotonic is None else started_monotonic
    parser = argparse.ArgumentParser(description="Autonomous ML agent skeleton")
    parser.add_argument("task", nargs="?", default="task.md", type=Path)
    parser.add_argument("--llm-mode", choices=["live", "mock"], default=None)
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument("--env-file", type=Path, help="Explicit .env; existing environment values take precedence")
    parser.add_argument('--profile', choices=['development', 'platform'], default=None)
    args = parser.parse_args(argv)
    try:
        if args.env_file:
            load_env_file(args.env_file)
        config = Config.from_env(llm_mode=args.llm_mode, work_dir=args.work_dir, profile=args.profile)
    except Exception as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    run_id = "run-" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + uuid.uuid4().hex[:8]
    root = config.work_dir / run_id
    root.mkdir(parents=True, exist_ok=False, mode=0o700)
    log = RunLogger(root, run_id, level=config.log_level, log_model_io=config.log_model_io,
                    secrets=[os.environ.get("OPENAI_API_KEY", "")], console_format=config.console_format)
    previous = None
    if threading_is_main():
        previous = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt("SIGTERM")))
    try:
        log.event("run.started", config=config.public_dict(), task_path=str(args.task.resolve()), run_directory=str(root))
        log.event('llm.configuration', profile=config.profile, call_limit=config.llm_max_calls,
                  total_output_limit=config.llm_max_total_output_tokens, reasoning_protocol=config.llm_reasoning_protocol,
                  reasoning_effort=config.llm_reasoning_effort, provider_parameter_application='unverified',
                  request_max_bytes=config.llm_request_max_bytes, refinement_enabled=config.graph_refinement_enabled)
        # Two native threads for profiling/analysis/evaluation, plus at most six
        # leased training threads. The platform cgroup remains the hard limit.
        from threadpoolctl import threadpool_limits
        native_names = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')
        previous_threads = {name: os.environ.get(name) for name in native_names}
        try:
            os.environ.update({name: '2' for name in native_names})
            with threadpool_limits(limits=2):
                return bootstrap(config, started, args.task.resolve(), run_id, root, log)
        finally:
            for name, value in previous_threads.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
    finally:
        if previous is not None:
            signal.signal(signal.SIGTERM, previous)
        log.close()


def threading_is_main():
    import threading
    return threading.current_thread() is threading.main_thread()
