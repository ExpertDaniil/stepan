from pathlib import Path
import io
import json
import os
import subprocess
import sys
import time
import unittest
from unittest.mock import patch

from agent.budget import Budget
from agent.orchestrator import Orchestrator
from agent.runner import Job, Runner, group_processes, resident_mb
from agent.catalog import Catalog
from tests.helpers import Harness


@unittest.skipUnless(sys.platform.startswith("linux"), "Process-group tests target the Linux competition container")
class ProcessTests(unittest.TestCase):
    def setUp(self):
        self.h = Harness()
        self.budget = Budget(self.h.config, time.monotonic(), self.h.log)
        self.runner = Runner(self.h.artifacts, self.h.config, self.budget, self.h.log)
        self.processes = []

    def tearDown(self):
        for proc in self.processes:
            if group_processes(proc.pid):
                try:
                    os.killpg(proc.pid, 9)
                except ProcessLookupError:
                    pass
            proc.wait(timeout=3)
            if proc.stdout:
                proc.stdout.close()
        self.h.close()

    def make_job(self, timeout=30, child=False, completed=None, exit_code=0):
        ident = self.h.store.reserve(self.h.plan(model={"family": "linear", "hyperparameters": {"C": len(self.processes) + 1}}), self.h.code["sha256"])["experiment_id"]
        req = {"cpu_threads": 1, "memory_limit_mb": 512, "timeout_seconds": timeout}
        lease = self.budget.lease({"operation": "acquire", "owner_id": ident, "phase": "train", "requested": req, "lease_id": None, "fencing_token": None})["lease"]
        attempt = self.h.store.start(ident, lease)
        job_id = "job-test-" + str(len(self.processes) + 1)
        directory = self.h.run / job_id
        directory.mkdir(exist_ok=True)
        code = "import time; time.sleep(30)"
        if child:
            code = "import subprocess,sys,time; p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); print(p.pid,flush=True); time.sleep(30)"
        if completed is not None:
            result = {"success": completed, "error": "synthetic worker error"}
            write_result = "" if completed == "missing" else f"Path({str(directory / 'result.json')!r}).write_text(json.dumps({result!r})); "
            code = ("import subprocess,sys,json; from pathlib import Path; "
                    "p=subprocess.Popen([sys.executable,'-u','-c',"
                    "'import os,time; print(os.getpid(),flush=True); time.sleep(30)'],stdout=subprocess.PIPE,text=True); "
                    "p.stdout.readline(); print(p.pid,flush=True); "
                    f"{write_result}"
                    f"sys.exit({exit_code})")
        proc = subprocess.Popen([sys.executable, "-u", "-c", code], start_new_session=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        self.processes.append(proc)
        job = Job(job_id, ident, attempt, "train", lease, proc, directory, time.monotonic(), last_heartbeat=time.monotonic())
        self.runner.jobs[job.job_id] = job
        return job

    def controller(self):
        controller = Orchestrator.__new__(Orchestrator)
        for key, value in {"config": self.h.config, "bus": self.h.bus, "budget": self.budget,
                           "runner": self.runner, "store": self.h.store, "log": self.h.log}.items():
            setattr(controller, key, value)
        return controller

    def test_completed_worker_with_live_helper_keeps_success_and_releases_resources(self):
        job = self.make_job(completed=True)
        child_pid = int(job.process.stdout.readline().strip())
        job.process.wait(timeout=3)
        self.assertIn(child_pid, group_processes(job.process.pid))
        self.assertGreater(resident_mb(job.process.pid), 0)
        result = self.controller().wait_jobs([job])[job.job_id]
        self.assertEqual(result["state"], "succeeded")
        self.assertIsNone(result["error"])
        self.assertEqual(self.h.store.get(job.experiment_id)["state"], "trained")
        self.assertFalse(group_processes(job.process.pid))
        self.assertFalse(self.budget.leases)
        events = [json.loads(line) for line in (self.h.run / "events.jsonl").read_text().splitlines()]
        self.assertFalse(any(e.get("contract_id") == "C24" for e in events))

    def test_completed_worker_cleanup_does_not_hide_failure_or_missing_result(self):
        for completed, exit_code in [(False, 1), ("missing", 0), (True, 1)]:
            with self.subTest(completed=completed, exit_code=exit_code):
                job = self.make_job(completed=completed, exit_code=exit_code)
                job.process.stdout.readline()
                job.process.wait(timeout=3)
                result = self.controller().wait_jobs([job])[job.job_id]
                self.assertEqual(result["state"], "failed")
                self.assertEqual(result["error"]["code"], "WORKER_FAILED")
                self.assertFalse(group_processes(job.process.pid))
                self.assertFalse(self.budget.leases)

    def test_cleanup_failure_keeps_lease_and_does_not_report_completion(self):
        from agent.errors import AgentError
        job = self.make_job(completed=True)
        job.process.stdout.readline()
        job.process.wait(timeout=3)
        with patch.object(self.runner, "_stop_process_tree", side_effect=AgentError("PROCESS_STOP_FAILED", "synthetic cleanup failure")):
            with self.assertRaisesRegex(AgentError, "synthetic cleanup failure"):
                self.controller().wait_jobs([job])
        self.assertIn(job.lease["lease_id"], self.budget.leases)
        self.assertFalse(job.reported)
        self.assertIsNone(job.event)

    def test_cancel_stops_parent_and_grandchild_before_release(self):
        job = self.make_job(child=True)
        child_pid = int(job.process.stdout.readline().strip())
        result = self.runner.cancel({"job_id": job.job_id, "experiment_id": job.experiment_id, "attempt_no": job.attempt_no,
                                     "fencing_token": job.lease["fencing_token"], "reason": "unit test", "terminal_state": "cancelled", "grace_seconds": 0})
        self.assertTrue(result["process_tree_stopped"])
        self.assertEqual(group_processes(job.process.pid), [])
        self.assertNotIn(child_pid, group_processes(job.process.pid))

    def test_timeout_routes_through_cancel_event_and_resource_release(self):
        job = self.make_job(timeout=1)
        controller = Orchestrator.__new__(Orchestrator)
        for key, value in {"config": self.h.config, "bus": self.h.bus, "budget": self.budget,
                           "runner": self.runner, "store": self.h.store, "log": self.h.log}.items():
            setattr(controller, key, value)
        result = controller.wait_jobs([job])[job.job_id]
        self.assertEqual(result["state"], "timed_out")
        self.assertEqual(self.h.store.get(job.experiment_id)["state"], "timed_out")
        self.assertFalse(self.budget.leases)
        self.assertFalse(group_processes(job.process.pid))
        contracts = {__import__("json").loads(line).get("contract_id") for line in (self.h.run / "events.jsonl").read_text().splitlines()}
        self.assertTrue({"C24", "C15", "C13"}.issubset(contracts))

    def test_training_environment_does_not_receive_api_credentials(self):
        ident = self.h.store.reserve(self.h.plan(), self.h.code["sha256"])["experiment_id"]
        lease = self.budget.lease({"operation": "acquire", "owner_id": ident, "phase": "train",
                                   "requested": {"cpu_threads": 1, "memory_limit_mb": 512, "timeout_seconds": 5}, "lease_id": None, "fencing_token": None})["lease"]
        attempt = self.h.store.start(ident, lease)
        train = self.h.artifacts.write_json("validation/train.json", [0, 1])
        valid = self.h.artifacts.write_json("validation/valid.json", [2])
        fake = type("FakeProcess", (), {"pid": 999999, "stdout": io.StringIO(""), "stderr": io.StringIO("")})()
        with patch.dict(os.environ, {"OPENAI_API_KEY": "unit-private-secret", "ANOTHER_ACCESS_TOKEN": "also-private"}), \
                patch("agent.runner.subprocess.Popen", return_value=fake) as spawn:
            self.runner.start_train({"lease": lease, "attempt_no": attempt, "train_rows_artifact_id": train["artifact_id"],
                                     "valid_rows_artifact_id": valid["artifact_id"]}, self.h.spec, self.h.store.get(ident))
        env = spawn.call_args.kwargs["env"]
        self.assertNotIn("OPENAI_API_KEY", env)
        self.assertNotIn("ANOTHER_ACCESS_TOKEN", env)
        for job in self.runner.jobs.values():
            for thread in job.threads:
                thread.join(timeout=2)

    def test_worker_has_its_own_memory_watchdog(self):
        jobfile = self.h.run / "guard-job.json"
        result = self.h.run / "guard-result.json"
        jobfile.write_text(json.dumps({"result_path": str(result), "memory_limit_mb": 1, "timeout_seconds": 10}))
        from agent.config import ROOT
        process = subprocess.run([sys.executable, "-m", "agent.worker", str(jobfile)], cwd=ROOT, text=True, capture_output=True, timeout=10)
        self.assertEqual(process.returncode, 137, process.stderr)
        self.assertEqual(json.loads(result.read_text())["error_type"], "JOB_MEMORY_LIMIT")

    def test_failed_spawn_releases_unstarted_lease(self):
        controller = Orchestrator.__new__(Orchestrator)
        train = self.h.artifacts.write_json("validation/train.json", [0, 1])
        valid = self.h.artifacts.write_json("validation/valid.json", [2])
        for key, value in {"config": self.h.config, "bus": self.h.bus, "budget": self.budget,
                           "runner": self.runner, "store": self.h.store, "log": self.h.log,
                           "artifacts": self.h.artifacts, "spec": self.h.spec, "catalog": Catalog(self.h.spec["task_kind"], self.h.config),
                           "validation": {"validation_id": "validation-unit", "train_rows_artifact_id": train["artifact_id"], "valid_rows_artifact_id": valid["artifact_id"]}}.items():
            setattr(controller, key, value)
        from agent.errors import AgentError
        with patch("agent.runner.subprocess.Popen", side_effect=OSError("simulated spawn failure")):
            with self.assertRaises(AgentError):
                controller.run_training([self.h.plan()])
        self.assertFalse(self.budget.leases)
        self.assertEqual(self.h.store.db.execute("SELECT state FROM experiments").fetchone()[0], "failed")
