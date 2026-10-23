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
from agent.budget import Budget
from agent.runtime import main
from scripts.make_demo import make_demo
from scripts.package_solution import build


class EndToEndTests(unittest.TestCase):
    def env(self, experiments="2"):
        return {**os.environ, "LLM_MODE": "mock", "TIME_BUDGET_SECONDS": "60", "MAX_RESEARCH_EXPERIMENTS": experiments,
                "LOG_LEVEL": "ERROR", "MAX_WORKERS": "2", "TRAIN_MEMORY_MB": "10240", "EXPERIMENT_TIMEOUT_SECONDS": "30",
                "RESEARCH_MODE": "catalog", "CATALOG_WARMUP_EXPERIMENTS": "0"}

    def check_run(self, temp, kind, experiments="2", source=ROOT, platform_copy=False, time_budget=None):
        import pandas as pd
        task = make_demo(Path(temp) / kind, kind)
        runs = Path(temp) / "runs"
        env = self.env(experiments)
        if time_budget is not None:
            env['TIME_BUDGET_SECONDS']=str(time_budget)
        env.pop("PYTHONPATH", None)
        env.pop("PYTHONHOME", None)
        cwd = temp
        command = [sys.executable, str(source / "solution.py"), str(task)]
        if platform_copy:
            for path in source.iterdir():
                if path.is_file() and (path.suffix == ".py" or path.name == "requirements.txt"):
                    shutil.copy2(path, task.parent / path.name)
            self.assertFalse((task.parent / "agent").exists())
            command = [sys.executable, "solution.py", "task.md"]
            cwd = task.parent
            env["WORK_DIR"] = str(runs)
        else:
            command += ["--llm-mode", "mock", "--work-dir", str(runs)]
        result = subprocess.run(command,
                                env=env, cwd=cwd, text=True, capture_output=True,
                                timeout=int(env['TIME_BUDGET_SECONDS'])+30)
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
            _, _, events = self.check_run(temp, "classification",time_budget=120)
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
            # Bound the evidence needed by this test, independently of CI speed.
            settings = {**self.env("6"), "CATALOG_WARMUP_EXPERIMENTS": "4", "TIME_BUDGET_SECONDS": "180"}
            original_update=Orchestrator.update_final_reserve
            def finish_after_six_fits(controller):
                original_update(controller)
                if controller.graph.attempted_research_count()>=6:
                    controller.budget.deadline=time.monotonic()+controller.budget.reserve_seconds()+.5
            spawn = subprocess.Popen
            def spawn_with_refit_helper(command, **kwargs):
                if "agent.worker" in command and json.loads(Path(command[-1]).read_text())["phase"] == "refit":
                    wrapper = ("import runpy,subprocess,sys; "
                               "p=subprocess.Popen([sys.executable,'-u','-c',"
                               "'import time; print(\"ready\",flush=True); time.sleep(60)'],stdout=subprocess.PIPE,text=True); "
                               "p.stdout.readline(); sys.argv=['agent.worker',sys.argv[1]]; "
                               "runpy.run_module('agent.worker',run_name='__main__')")
                    command = [command[0], "-u", "-c", wrapper, command[-1]]
                return spawn(command, **kwargs)
            with patch.dict(os.environ, settings), patch.object(Orchestrator,'update_final_reserve',finish_after_six_fits), patch("agent.runner.subprocess.Popen", side_effect=spawn_with_refit_helper), redirect_stderr(io.StringIO()):
                code = main([str(task), "--llm-mode", "mock", "--work-dir", str(Path(temp)/"runs")])
            self.assertEqual(code, 0)
            events = [json.loads(line) for line in next((Path(temp)/"runs").glob("run-*/events.jsonl")).read_text().splitlines()]
            warmup = next(i for i, e in enumerate(events) if e["event"] == "search.catalog_warmup")
            research = next(i for i, e in enumerate(events) if e["event"] == "research.hypothesis_proposed")
            self.assertLess(warmup, research)
            self.assertTrue(any(e["event"] == "experiment.evaluated" for e in events[warmup:research]))
            self.assertGreaterEqual(sum(e["event"] == "job.started" and e.get("phase") == "train" for e in events), 5)
            choices = [json.loads(p.read_text())["payload"] for p in next((Path(temp)/"runs").glob("run-*")).glob("messages/*.request.json")]
            self.assertTrue(any(p.get("response_schema_id") == "ResearchChoice-v1" and p["tool_schema_ids"] == [] for p in choices))
            cleanups = [e for e in events if e["event"] == "job.descendants_cleanup_completed"]
            self.assertTrue(cleanups)
            self.assertTrue(any(e["phase"] == "refit" for e in cleanups))
            self.assertTrue(all(e["process_tree_stopped"] and e["remaining_processes"] == 0 for e in cleanups))
            self.assertTrue(all(e["state"] == "succeeded" for e in events if e["event"] == "job.finished" and e["phase"] == "refit"))
            self.assertGreaterEqual(sum(e["event"] == "submission.published" for e in events), 2)
            self.assertEqual(events[-1]["state"], "completed", json.dumps([e for e in events if e['level'] in {'ERROR','WARNING'}])[-6000:])

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

    def test_budget_clipped_training_timeout_keeps_a_completed_verified_output(self):
        for graph in (False,True):
            with self.subTest(graph=graph),tempfile.TemporaryDirectory() as temp:
                task=make_demo(Path(temp)/'task')
                lease=Budget.lease;can_train=Budget.can_train;override=set();clipped=[]
                spawn=subprocess.Popen;delayed=[]
                def slow_clipped_worker(command, **kwargs):
                    if clipped and not delayed and '-m' in command and 'agent.worker' in command:
                        job = json.loads(Path(command[-1]).read_text())
                        if job['phase'] == 'train':
                            delayed.append(True)
                            # Force the clipped deadline path independently of
                            # which family is selected and how fast this CPU is.
                            wrapper = 'import time,runpy,sys; time.sleep(5); sys.argv=["agent.worker",sys.argv[1]]; runpy.run_module("agent.worker",run_name="__main__")'
                            command = [command[0], '-u', '-c', wrapper, command[-1]]
                    return spawn(command, **kwargs)
                def estimate(budget,plan):
                    if id(budget) in override:
                        return {'allowed':True,'estimated_seconds':.1,'reserve_seconds':budget.reserve_seconds(),'remaining_seconds':budget.remaining(),'family':plan['model']['family']}
                    return can_train(budget,plan)
                def clip(budget,payload):
                    plan=budget.admissions.get(payload['owner_id'])
                    if not clipped and payload['operation']=='acquire' and payload['phase']=='train' and plan and plan['model']['family']!='dummy':
                        clipped.append(True)
                        budget.deadline=time.monotonic()+budget.candidate_reserve(plan)+1.5
                        override.add(id(budget))
                        try:return lease(budget,payload)
                        finally:override.remove(id(budget))
                    return lease(budget,payload)
                settings={**self.env(),'GRAPH_ENABLED':str(int(graph)),'GRAPH_REFINEMENT_ENABLED':'0','TIME_BUDGET_SECONDS':'90'}
                with patch.dict(os.environ,settings),patch.object(Budget,'lease',clip),patch.object(Budget,'can_train',estimate),patch('agent.runner.subprocess.Popen',side_effect=slow_clipped_worker),redirect_stderr(io.StringIO()) as stderr:
                    code=main([str(task),'--work-dir',str(Path(temp)/'runs')])
                self.assertEqual(code,0,stderr.getvalue()[-3000:])
                run=next((Path(temp)/'runs').glob('run-*'))
                events=[json.loads(line) for line in (run/'events.jsonl').read_text().splitlines()]
                self.assertTrue(clipped)
                self.assertTrue(delayed)
                self.assertTrue(any(e['event']=='search.budget_trial_timed_out' for e in events))
                self.assertTrue((task.parent/'submission.csv').is_file())
                self.assertIn('State: completed',(run/'summary.md').read_text())

    def test_flat_platform_copy_runs_research_without_repository_path(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "solution.zip"
            build(archive)
            unpacked = root / "unpacked"
            with zipfile.ZipFile(archive) as z:
                self.assertTrue(all(Path(name).name == name and (name.endswith(".py") or name == "requirements.txt") for name in z.namelist()))
                self.assertIn("agent_graph_controller.py", z.namelist())
                self.assertEqual(z.read("requirements.txt"), (ROOT / "requirements.organizers.txt").read_bytes())
                z.extractall(unpacked)
            _, _, events = self.check_run(temp, "classification", "2", source=unpacked, platform_copy=True)
            self.assertGreaterEqual(sum(e["event"] == "submission.published" for e in events), 2)

    def test_readable_flat_platform_copy_runs_regression(self):
        with tempfile.TemporaryDirectory() as temp:
            archive = Path(temp) / "solution.zip"
            build(archive)
            unpacked = Path(temp) / "unpacked"
            with zipfile.ZipFile(archive) as z:
                z.extractall(unpacked)
            _, _, events = self.check_run(temp, "regression", "2", source=unpacked, platform_copy=True)
            self.assertEqual(events[-1]["state"], "completed")
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

    def test_provider_credit_failure_before_bootstrap_reports_cause_without_fake_output(self):
        with tempfile.TemporaryDirectory() as temp:
            task = make_demo(Path(temp) / "task")
            failure = RuntimeError("OpenrouterException - Insufficient credits")
            failure.status_code = 503
            with patch.dict(os.environ, self.env()), patch.object(MockTransport, "complete", side_effect=failure) as transport, redirect_stderr(io.StringIO()):
                code = main([str(task), "--llm-mode", "mock", "--work-dir", str(Path(temp) / "runs")])
            self.assertEqual(code, 1)
            self.assertEqual(transport.call_count, 1)
            self.assertFalse((task.parent / "submission.csv").exists())
            events = [json.loads(line) for line in next((Path(temp)/"runs").glob("run-*/events.jsonl")).read_text().splitlines()]
            self.assertEqual(events[-1]["event"], "run.failed")
            self.assertEqual(events[-1]["code"], "LLM_PROVIDER_CREDITS_EXHAUSTED")
            self.assertFalse(any(e["event"] in {"llm.daily_budget_exhausted", "job.started"} for e in events))
