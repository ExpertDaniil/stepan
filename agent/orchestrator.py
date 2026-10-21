from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import time
import uuid

from agent.catalog import factory_source
from agent.context import ContextBuilder
from agent.errors import AgentError
from tools.code import write_code


class Orchestrator:
    def __init__(self, *, config, bus, budget, artifacts, store, runner, evaluator, submission, researcher,
                 spec, validation, catalog, task_text, profile, log, demo=None):
        self.config, self.bus, self.budget, self.artifacts, self.store = config, bus, budget, artifacts, store
        self.runner, self.evaluator, self.submission, self.researcher = runner, evaluator, submission, researcher
        self.spec, self.validation, self.catalog, self.log = spec, validation, catalog, log
        self.context = ContextBuilder(spec, task_text, profile, catalog, artifacts, log, demo)
        self.researcher.validation_id = validation["validation_id"]
        self.degraded = False

    def call(self, cid, payload, handler, key=None):
        return self.bus.call(cid, payload, handler, key=key)["data"]

    def lease(self, owner, phase, resources):
        return self.call("C13", {"operation": "acquire", "owner_id": owner, "phase": phase, "requested": resources,
                                  "lease_id": None, "fencing_token": None}, self.budget.lease,
                         key="acquire/" + owner + "/" + phase)

    def release(self, job):
        if self.runner.poll(job) is None:
            raise AgentError("PROCESS_STOP_FAILED", "Refusing to release a running process tree")
        self.call("C13", {"operation": "release", "owner_id": job.experiment_id if job.phase == "train" else job.submission_job_id,
                           "phase": job.phase, "requested": self.resources(job.lease["timeout_seconds"]),
                           "lease_id": job.lease["lease_id"], "fencing_token": job.lease["fencing_token"]}, self.budget.lease,
                  key="release/" + job.lease["lease_id"])

    def resources(self, timeout=None):
        return {"cpu_threads": self.config.train_cpu_threads, "memory_limit_mb": self.config.train_memory_mb,
                "timeout_seconds": timeout or self.config.experiment_timeout_seconds}

    def submit_plan(self, payload):
        plan = self.catalog.validate_plan(payload["plan"], self.spec, self.validation["validation_id"], self.artifacts)
        return self.store.reserve(plan, self.artifacts.meta(plan["code_artifact_id"])["sha256"])

    def cancel_job(self, job, reason, state="cancelled"):
        return self.call("C24", {"job_id": job.job_id, "experiment_id": job.experiment_id,
                                  "attempt_no": job.attempt_no, "fencing_token": job.lease["fencing_token"],
                                  "reason": reason, "terminal_state": state, "grace_seconds": 1}, self.runner.cancel,
                         key="cancel/" + job.job_id + "/" + state)

    def _heartbeat_and_limits(self, job):
        now = time.monotonic()
        if job.process.poll() is not None:
            self.cancel_job(job, "Main process exited but child processes remain")
            return
        elapsed = now - job.started
        lease_state = self.budget.leases.get(job.lease["lease_id"])
        if self.budget.remaining() <= 1 or elapsed > job.lease["timeout_seconds"]:
            self.cancel_job(job, "Job or run deadline reached", "timed_out")
            return
        if job.peak_memory_mb > job.lease["memory_limit_mb"]:
            self.cancel_job(job, "Resident memory limit exceeded")
            return
        if lease_state and now > lease_state["heartbeat_deadline"]:
            self.cancel_job(job, "Lease heartbeat expired")
            return
        if now - job.last_heartbeat >= 10:
            self.call("C13", {"operation": "renew", "owner_id": job.experiment_id if job.phase == "train" else job.submission_job_id,
                               "phase": job.phase, "requested": self.resources(job.lease["timeout_seconds"]),
                               "lease_id": job.lease["lease_id"], "fencing_token": job.lease["fencing_token"]}, self.budget.lease)
            job.last_heartbeat = now

    def wait_jobs(self, jobs):
        results = {}
        pending = list(jobs)
        while pending:
            for job in list(pending):
                event = self.runner.poll(job)
                if event is None:
                    self._heartbeat_and_limits(job)
                    event = self.runner.poll(job)
                if event is None:
                    continue
                self.call("C15", {"event": event}, lambda p, phase=job.phase: self.store.apply_event(p["event"], phase), key=event["event_id"])
                self.release(job)
                job.reported = True
                results[job.job_id] = event
                pending.remove(job)
                if job.phase == "train":
                    self.budget.last_train_duration = max(self.budget.last_train_duration, event["duration_seconds"])
            if pending:
                time.sleep(0.1)
        return results

    def run_training(self, plans):
        jobs = []
        for plan in plans:
            record = self.call("C12", {"plan": plan}, self.submit_plan)
            if record["disposition"] == "duplicate":
                self.log.event("experiment.duplicate_skipped", experiment_id=record["experiment_id"])
                continue
            ident = record["experiment_id"]
            lease_reply = self.lease(ident, "train", plan["resources"])
            if not lease_reply["granted"]:
                raise AgentError("RESOURCE_BUSY", "Batch exceeds available execution slots")
            lease = lease_reply["lease"]
            attempt = self.store.start(ident, lease)
            args = {"experiment_id": ident, "attempt_no": attempt, "plan": plan, "lease": lease,
                    "train_rows_artifact_id": self.validation["train_rows_artifact_id"],
                    "valid_rows_artifact_id": self.validation["valid_rows_artifact_id"],
                    "output_kinds": ["validation_predictions", "fitted_pipeline", "execution_log"]}
            try:
                reply = self.call("C14", args, lambda p, eid=ident: self.runner.start_train(p, self.spec, self.store.get(eid)), key=ident + "/" + str(attempt))
            except Exception as exc:
                self.store.fail_start(ident, exc)
                if not any(job.lease["lease_id"] == lease["lease_id"] for job in self.runner.jobs.values()):
                    self.call("C13", {"operation": "release", "owner_id": ident, "phase": "train", "requested": plan["resources"],
                                       "lease_id": lease["lease_id"], "fencing_token": lease["fencing_token"]}, self.budget.lease)
                raise
            jobs.append(self.runner.jobs[reply["job_id"]])
        events = self.wait_jobs(jobs)
        for job in jobs:
            event = events[job.job_id]
            if event["state"] != "succeeded":
                self.degraded = True
                continue
            predictions = next((a for a in event["output_artifact_ids"] if self.artifacts.meta(a)["kind"] == "validation_predictions"), None)
            if not predictions:
                self.store.invalidate(job.experiment_id, "Missing predictions")
                self.degraded = True
                continue
            try:
                result = self.call("C16", {"experiment_id": job.experiment_id, "task_spec_id": self.spec["task_spec_id"],
                                            "dataset_id": self.spec["dataset_id"], "validation_id": self.validation["validation_id"],
                                            "predictions_artifact_id": predictions, "fidelity": "full"}, self.evaluator.evaluate)
                self.call("C17", {"evaluation_id": result["evaluation"]["evaluation_id"], "experiment_id": job.experiment_id,
                                   "selection_policy_id": "same-protocol-primary-v1"}, self.store.commit_evaluation,
                          key=result["evaluation"]["evaluation_id"])
            except AgentError as exc:
                self.store.invalidate(job.experiment_id, str(exc))
                self.degraded = True
                self.log.exception("experiment.rejected", exc, experiment_id=job.experiment_id)
        return len(jobs)

    def baseline_plan(self):
        recipe_id = next(x for x in self.catalog.recipes if x.startswith("dummy-"))
        source = factory_source(recipe_id)
        args = {"researcher_id": "baseline", "filename": "baseline.py", "content": source, "declared_entrypoint": "make_pipeline"}
        code = self.call("C11", args, lambda p: write_code(p, self.artifacts), key="baseline-code")["artifact"]
        return {"plan_id": "plan-baseline", "researcher_id": "baseline", "parent_ids": [],
                "hypothesis": "Create a fast valid fallback before spending the research budget.",
                "expected_effect": "A valid output file and a measured reference score.", "recipe_id": recipe_id,
                "task_spec_id": self.spec["task_spec_id"], "dataset_id": self.spec["dataset_id"],
                "validation_id": self.validation["validation_id"], "seed": self.config.seed, "fidelity": "full",
                "model": {"family": "dummy", "hyperparameters": self.catalog.normalize(recipe_id, {})},
                "feature_steps": [], "code_artifact_id": code["artifact_id"], "resources": self.resources()}

    def finalize_best(self):
        previous = set(self.submission.jobs)
        try:
            return self._finalize_best()
        except Exception:
            for ident in set(self.submission.jobs) - previous:
                if self.submission.jobs[ident]["state"] not in {"failed", "timed_out", "cancelled", "published", "skipped_not_better"}:
                    self.submission.change_state(ident, "failed")
            raise

    def _finalize_best(self):
        ident = self.store.best()
        champion = self.store.champion()
        if not ident or (champion and champion["experiment_id"] == ident):
            return
        evaluation = self.store.experiment_evaluation(ident)
        prepared = self.call("C18", {"experiment_id": ident, "evaluation_id": evaluation["evaluation_id"],
                                     "task_spec_id": self.spec["task_spec_id"], "mode": "refit_full_train",
                                     "expected_champion_version": champion["champion_version"] if champion else 0}, self.submission.prepare,
                             key="prepare/" + ident)
        subjob = prepared["submission_job_id"]
        lease_reply = self.lease(subjob, "refit", self.resources(max(self.config.experiment_timeout_seconds, int(self.budget.last_train_duration * 3 + 10))))
        if not lease_reply["granted"]:
            raise AgentError("RESOURCE_BUSY", "Cannot refit while other jobs hold resources")
        experiment = self.store.get(ident)
        args = {"submission_job_id": subjob, "experiment_id": ident, "task_spec_id": self.spec["task_spec_id"],
                "code_artifact_id": experiment["plan"]["code_artifact_id"], "lease": lease_reply["lease"], "fit_scope": "all_train",
                "predict_artifact_id": self.spec["test_artifact_id"], "output_relative_path": "submissions/" + subjob + "/candidate.csv"}
        result = self.call("C19", args, lambda p: self.runner.start_refit(p, self.spec, experiment), key=subjob)
        job = self.runner.jobs[result["job_id"]]
        job.submission_job_id = subjob
        self.submission.change_state(subjob, "refitting")
        event = self.wait_jobs([job])[job.job_id]
        if event["state"] != "succeeded":
            self.submission.change_state(subjob, event["state"])
            raise AgentError("REFIT_FAILED", "Final training failed; previous published file is retained")
        candidate = next((a for a in event["output_artifact_ids"] if self.artifacts.meta(a)["kind"] == "submission"), None)
        if not candidate:
            raise AgentError("REFIT_FAILED", "Refit did not produce a candidate")
        self.submission.change_state(subjob, "validating")
        report = self.call("C20", {"submission_job_id": subjob, "task_spec_id": self.spec["task_spec_id"],
                                   "candidate_artifact_id": candidate, "reference_test_artifact_id": self.spec["test_artifact_id"]}, self.submission.validate_output)
        current = self.store.champion()
        publication = self.call("C21", {"experiment_id": ident, "evaluation_id": evaluation["evaluation_id"], "candidate_artifact_id": candidate,
                           "validation_report_id": report["validation_report_id"], "candidate_sha256": report["candidate_sha256"],
                           "expected_champion_version": current["champion_version"] if current else 0}, self.submission.publish,
                  key="publish/" + ident + "/" + report["candidate_sha256"])
        self.submission.change_state(subjob, "skipped_not_better" if publication["disposition"] == "skipped_not_better" else "published")

    def control(self, payload):
        self.budget.search_stopped = True
        self.store.set_run_state("finalizing")
        if payload["action"] == "cancel_jobs":
            for ident in payload["cancel_job_ids"]:
                self.cancel_job(self.runner.jobs[ident], payload["reason"])
        return {"run_state": "finalizing", "new_experiments_allowed": False,
                "published_champion_present": self.store.champion() is not None,
                "active_job_ids": [job.job_id for job in self.runner.active()]}

    def run(self):
        self.store.set_run_state("baseline")
        self.run_training([self.baseline_plan()])
        self.finalize_best()
        self.submission.require_output()
        self.store.set_run_state("searching")
        accepted = 0
        for round_no in range(self.config.max_research_experiments + 2):
            if accepted >= self.config.max_research_experiments or not self.budget.snapshot()["search_allowed"]:
                break
            self.context.round_no = round_no
            snapshot = self.call("C07", {"task_spec_id": self.spec["task_spec_id"], "validation_id": self.validation["validation_id"],
                                         "limit_recent": 6, "limit_best": 3}, lambda p: self.store.snapshot(p, self.artifacts))
            requests = []
            for index in range(min(self.config.max_workers, self.config.max_research_experiments - accepted)):
                researcher_id = "researcher-" + str(index + 1)
                assignment = "exploit_incumbent" if index == 0 else "explore_alternative"
                context = self.call("C08", {"task_spec_id": self.spec["task_spec_id"], "snapshot_id": snapshot["snapshot_id"],
                                            "tools_catalog_version": "tools-v1", "researcher_id": researcher_id,
                                            "assignment": assignment, "max_input_tokens": self.config.max_context_tokens}, lambda p: self.context.build(p, snapshot))
                requests.append({"researcher_id": researcher_id, "context_id": context["context_id"], "assignment": assignment,
                                 "base_experiment_ids": [snapshot["best_evaluated_experiment_id"]] if snapshot["best_evaluated_experiment_id"] else [],
                                 "forbidden_fingerprints": snapshot["reserved_fingerprints"], "budget": self.budget.snapshot(), "proposal_limit": 1})
            plans = []
            with ThreadPoolExecutor(max_workers=self.config.max_workers, thread_name_prefix="researcher") as pool:
                futures = [pool.submit(self.call, "C09", request, self.researcher.propose) for request in requests]
                for future in as_completed(futures):
                    try:
                        reply = future.result()
                        if reply["decision"] == "propose":
                            plans.append(reply["plan"])
                    except AgentError as exc:
                        self.degraded = True
                        self.log.exception("research.proposal_failed", exc)
            if not plans:
                break
            try:
                accepted += self.run_training(plans)
                self.finalize_best()
            except AgentError as exc:
                self.degraded = True
                self.log.exception("research.round_failed", exc)
                break
        self.call("C22", {"action": "stop_search", "reason": "search_budget_or_rounds_finished",
                           "wall_seconds_remaining": int(self.budget.remaining()), "cancel_job_ids": []}, self.control)
        self.cleanup()
        return self.submission.require_output()

    def cleanup(self):
        active = self.runner.active()
        for job in active:
            try:
                self.cancel_job(job, "Runtime cleanup")
            except AgentError as exc:
                self.log.exception("job.cleanup_failed", exc, job_id=job.job_id)
                raise
        if active:
            self.wait_jobs(active)
