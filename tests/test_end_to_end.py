from contextlib import redirect_stderr
from pathlib import Path
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
import zipfile

from agent.config import ROOT
from agent.contracts import Schemas
from agent.errors import AgentError
from agent.llm.gateway import MockTransport
from agent.orchestrator import Orchestrator
from agent.runtime import main
from scripts.make_demo import make_demo
from scripts.package_solution import build


class EndToEndTests(unittest.TestCase):
    def env(self, experiments="2"):
        return {**os.environ, "LLM_MODE": "mock", "TIME_BUDGET_SECONDS": "90", "MAX_RESEARCH_EXPERIMENTS": experiments,
                "LOG_LEVEL": "ERROR", "MAX_WORKERS": "2", "TRAIN_MEMORY_MB": "10240", "EXPERIMENT_TIMEOUT_SECONDS": "30",
                "RESEARCH_MODE": "catalog", "CATALOG_WARMUP_EXPERIMENTS": "0"}

    def check_run(self, temp, kind, experiments="2", source=ROOT, platform_copy=False):
        import pandas as pd
        task = make_demo(Path(temp) / kind, kind)
        runs = Path(temp) / "runs"
        env = self.env(experiments)
        env.pop("PYTHONPATH", None)
        env.pop("PYTHONHOME", None)
        cwd = temp
        command = [sys.executable, str(source / "solution.py"), str(task)]
        if platform_copy:
            for name in ("solution.py", "requirements.txt"):
                shutil.copy2(source / name, task.parent / name)
            command = [sys.executable, "solution.py", "task.md"]
            cwd = task.parent
        result = subprocess.run(command + ["--llm-mode", "mock", "--work-dir", str(runs)],
                                env=env, cwd=cwd, text=True, capture_output=True, timeout=70)
        self.assertEqual(result.returncode, 0, result.stderr[-6000:])
        output = pd.read_csv(task.parent / "submission.csv", dtype=str, keep_default_na=False)
        test = pd.read_csv(task.parent / "test.csv", dtype=str, keep_default_na=False)
        self.assertEqual(list(output.columns), ["id", "label"])
        self.assertEqual(output["id"].tolist(), test["id"].tolist())
        self.assertEqual(len(output), 90)
        run = next(runs.glob("run-*"))
        events = [json.loads(line) for line in (run / "events.jsonl").read_text().splitlines()]
        self.assertEqual(events[-1]["event"], "run.finished")
        self.assertTrue((run / "summary.md").is_file())
        schemas = Schemas()
        for path in (run / "messages").glob("*.request.json"):
            request = json.loads(path.read_text())
            cid = next(e["contract_id"] for e in events if e.get("request_id") == request["request_id"])
            schemas.validate(request, cid + "_Request")
            response = json.loads(path.with_name(path.name.replace(".request.json", ".response.json")).read_text())
            schemas.validate(response, cid + "_Response")
        return task, run, events

    def test_classification_full_research_loop(self):
        with tempfile.TemporaryDirectory() as temp:
            _, _, events = self.check_run(temp, "classification")
            publications = [e for e in events if e["event"] == "submission.published"]
            self.assertGreaterEqual(len(publications), 2)
            self.assertGreater(publications[-1]["score"], publications[0]["score"])
            workers = {e.get("researcher_id") for e in events if e["event"] == "research.hypothesis_proposed"}
            self.assertEqual(workers, {"researcher-1", "researcher-2"})

    def test_regression_full_research_loop(self):
        with tempfile.TemporaryDirectory() as temp:
            _, _, events = self.check_run(temp, "regression")
            publications = [e for e in events if e["event"] == "submission.published"]
            self.assertGreaterEqual(len(publications), 2)
            self.assertLess(publications[-1]["score"], publications[0]["score"])

    def test_binary_task(self):
        with tempfile.TemporaryDirectory() as temp:
            self.check_run(temp, "binary", "1")

    def test_catalog_warmup_trains_before_llm_research(self):
        with tempfile.TemporaryDirectory() as temp:
            task = make_demo(Path(temp) / "task", "regression")
            settings = {**self.env("6"), "CATALOG_WARMUP_EXPERIMENTS": "4"}
            with patch.dict(os.environ, settings), redirect_stderr(io.StringIO()):
                code = main([str(task), "--llm-mode", "mock", "--work-dir", str(Path(temp)/"runs")])
            self.assertEqual(code, 0)
            events = [json.loads(line) for line in next((Path(temp)/"runs").glob("run-*/events.jsonl")).read_text().splitlines()]
            warmup = next(i for i, e in enumerate(events) if e["event"] == "search.catalog_warmup")
            research = next(i for i, e in enumerate(events) if e["event"] == "research.hypothesis_proposed")
            self.assertLess(warmup, research)
            self.assertTrue(any(e["event"] == "experiment.evaluated" for e in events[warmup:research]))
            self.assertLessEqual(sum(e["event"] == "job.started" and e.get("phase") == "train" for e in events), 7)
            choices = [json.loads(p.read_text())["payload"] for p in next((Path(temp)/"runs").glob("run-*")).glob("messages/*.request.json")]
            self.assertTrue(any(p.get("response_schema_id") == "ResearchChoice-v1" and p["tool_schema_ids"] == [] for p in choices))

    def test_llm_failure_after_baseline_retains_output(self):
        with tempfile.TemporaryDirectory() as temp:
            task = make_demo(Path(temp) / "task")
            original = MockTransport.complete
            def fail_research(transport, request, timeout, context):
                if context["metadata"]["schema_id"] in {"ResearchProposal-v1", "ResearchChoice-v1"}:
                    raise AgentError("LLM_UNAVAILABLE", "simulated API outage")
                return original(transport, request, timeout, context)
            with patch.dict(os.environ, self.env()), patch.object(MockTransport, "complete", fail_research), redirect_stderr(io.StringIO()):
                code = main([str(task), "--llm-mode", "mock", "--work-dir", str(Path(temp) / "runs")])
            self.assertEqual(code, 0)
            self.assertTrue((task.parent / "submission.csv").is_file())
            summary = next((Path(temp) / "runs").glob("run-*/summary.md")).read_text()
            self.assertIn("State: degraded", summary)

    def test_search_deadline_keeps_published_baseline(self):
        with tempfile.TemporaryDirectory() as temp:
            task = make_demo(Path(temp) / "task")
            original = Orchestrator.finalize_best
            def shorten_after_publish(controller):
                result = original(controller)
                if controller.store.champion():
                    controller.budget.deadline = time.monotonic() + 0.3
                return result
            with patch.dict(os.environ, self.env()), patch.object(Orchestrator, "finalize_best", shorten_after_publish), redirect_stderr(io.StringIO()):
                code = main([str(task), "--llm-mode", "mock", "--work-dir", str(Path(temp) / "runs")])
            self.assertEqual(code, 0)
            self.assertTrue((task.parent / "submission.csv").is_file())

    def test_flat_platform_copy_runs_research_without_repository_path(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "solution.zip"
            build(archive)
            unpacked = root / "unpacked"
            with zipfile.ZipFile(archive) as z:
                self.assertEqual(z.namelist(), ["solution.py", "requirements.txt"])
                self.assertEqual(z.read("requirements.txt"), (ROOT / "requirements.organizers.txt").read_bytes())
                z.extractall(unpacked)
            _, _, events = self.check_run(temp, "classification", "2", source=unpacked, platform_copy=True)
            self.assertGreaterEqual(sum(e["event"] == "submission.published" for e in events), 2)

    def test_daily_budget_after_baseline_retains_output(self):
        with tempfile.TemporaryDirectory() as temp:
            task = make_demo(Path(temp) / "task")
            original = MockTransport.complete
            def exhaust_budget(transport, request, timeout, context):
                if context["metadata"]["schema_id"] in {"ResearchProposal-v1", "ResearchChoice-v1"}:
                    exc = RuntimeError("Budget has been exceeded")
                    exc.status_code = 429
                    raise exc
                return original(transport, request, timeout, context)
            with patch.dict(os.environ, self.env()), patch.object(MockTransport, "complete", exhaust_budget), redirect_stderr(io.StringIO()):
                code = main([str(task), "--llm-mode", "mock", "--work-dir", str(Path(temp) / "runs")])
            self.assertEqual(code, 0)
            self.assertTrue((task.parent / "submission.csv").is_file())
            run = next((Path(temp) / "runs").glob("run-*"))
            self.assertIn("State: degraded", (run / "summary.md").read_text())
            self.assertIn("llm.daily_budget_exhausted", (run / "events.jsonl").read_text())
