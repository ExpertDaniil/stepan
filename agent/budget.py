from datetime import datetime, timedelta, timezone
import threading
import time
import uuid
import math

from agent.errors import AgentError
from agent.duration import DurationModel


class Budget:
    def __init__(self, config, started, log):
        self.config, self.started, self.log = config, started, log
        self.deadline = started + config.time_budget_seconds
        self.leases = {}
        self.token = 0
        self.calls = 0
        self.output_tokens_charged = 0
        self.output_tokens_reported = 0
        self.input_tokens_reported = 0
        self.cached_tokens_reported = 0
        self.pending_llm_tokens = {}
        self.last_train_duration = 0.0
        self.duration = DurationModel(config)
        self.final_plan = None
        self.admissions = {}
        self.search_stopped = False
        self.lock = threading.RLock()

    def remaining(self):
        return max(0.0, self.deadline - time.monotonic())

    def reserve_seconds(self):
        with self.lock:
            floor = self.output_reserve()
            plans = ([self.final_plan] if self.final_plan else []) + [item["plan"] for item in self.leases.values() if item.get("plan")]
            return floor + max((self.duration.estimate(plan, "refit") for plan in plans), default=0)

    def output_reserve(self):
        return min(60, max(2, self.config.time_budget_seconds * .03))

    def candidate_reserve(self, plan):
        return max(self.reserve_seconds(), self.output_reserve() + self.duration.estimate(plan, "refit"))

    def can_train(self, plan):
        with self.lock:
            estimate = self.duration.estimate(plan)
            reserve = self.candidate_reserve(plan)
            allowed = (not self.search_stopped and estimate <= self.config.experiment_timeout_seconds
                       and self.remaining() >= estimate + reserve + 1)
            return {"allowed": allowed, "estimated_seconds": estimate, "reserve_seconds": reserve,
                    "remaining_seconds": self.remaining(), "family": plan["model"]["family"]}

    def admit(self, owner, plan):
        with self.lock:
            decision = self.can_train(plan)
            if decision["allowed"]:
                from copy import deepcopy
                self.admissions[owner] = deepcopy(plan)
            return decision

    def set_final_plan(self, plan):
        with self.lock:
            self.final_plan = plan

    def observe(self, plan, phase, event, fit_seconds=None):
        with self.lock:
            self.duration.observe(plan, phase, event["duration_seconds"], fit_seconds=fit_seconds, state=event["state"])

    def research_capacity(self, plans=None):
        """Approximate trials that fit; used to share time between saved branches, not a run cap."""
        with self.lock:
            plans = plans or ([self.final_plan] if self.final_plan else [])
            cost = min((self.duration.estimate(p) for p in plans), default=max(1, self.last_train_duration))
            return max(0, math.floor((self.remaining() - self.reserve_seconds() - 1) / cost))

    def snapshot(self):
        with self.lock:
            cpu = sum(v["lease"]["cpu_threads"] for v in self.leases.values())
            memory = sum(v["lease"]["memory_limit_mb"] for v in self.leases.values())
            return {"wall_seconds_remaining": int(self.remaining()), "finalization_reserve_seconds": int(self.reserve_seconds()),
                    "cpu_threads_available": max(0, 6 - cpu), "memory_mb_available": max(0, 20480 - memory),
                    "llm_calls_remaining": max(0, self.config.llm_max_calls - self.calls),
                    "search_allowed": not self.search_stopped and self.remaining() - self.reserve_seconds() >= 1}

    def llm_call(self):
        with self.lock:
            if self.calls >= self.config.llm_max_calls or self.remaining() < 1:
                raise AgentError("BUDGET_EXHAUSTED", "LLM budget exhausted")
            self.calls += 1
            return self.calls

    def begin_llm_call(self, max_tokens):
        """Reserve per-run token capacity and one API attempt atomically."""
        with self.lock:
            available = self.config.llm_max_total_output_tokens - self.output_tokens_charged - sum(self.pending_llm_tokens.values())
            if max_tokens > available:
                raise AgentError("BUDGET_EXHAUSTED", "Per-run output token budget cannot reserve this request",
                                 details={"requested": max_tokens, "available": max(0, available)})
            call_no = self.llm_call()
            self.pending_llm_tokens[call_no] = max_tokens
            return call_no

    def finish_llm_call(self, call_no, usage=None):
        with self.lock:
            reserved = self.pending_llm_tokens.pop(call_no, None)
            if reserved is None:
                return
            usage = usage or {}
            def count(value):
                return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None
            output = count(usage.get("completion_tokens"))
            self.output_tokens_charged += reserved if output is None else output
            self.output_tokens_reported += output or 0
            self.input_tokens_reported += count(usage.get("prompt_tokens")) or 0
            self.cached_tokens_reported += count((usage.get("prompt_tokens_details") or {}).get("cached_tokens")) or 0

    def token_snapshot(self):
        with self.lock:
            reserved = sum(self.pending_llm_tokens.values())
            return {"output_limit": self.config.llm_max_total_output_tokens, "output_charged": self.output_tokens_charged,
                    "output_reported": self.output_tokens_reported, "output_reserved": reserved,
                    "output_available": max(0, self.config.llm_max_total_output_tokens - self.output_tokens_charged - reserved),
                    "input_reported": self.input_tokens_reported, "cached_input_reported": self.cached_tokens_reported}

    def lease(self, payload):
        with self.lock:
            operation = payload["operation"]
            if operation in {"renew", "release"}:
                item = self.leases.get(payload["lease_id"])
                if item is None and operation == "release":
                    return {"granted": True, "lease": None, "retry_after_seconds": None, "budget": self.snapshot()}
                if not item or item["lease"]["fencing_token"] != payload["fencing_token"] or item["owner_id"] != payload["owner_id"]:
                    raise AgentError("STALE_LEASE", "Invalid lease owner or fencing token")
                if operation == "release":
                    # Caller must have confirmed process_tree_stopped via Runner.
                    del self.leases[payload["lease_id"]]
                    return {"granted": True, "lease": None, "retry_after_seconds": None, "budget": self.snapshot()}
                item["heartbeat_deadline"] = time.monotonic() + 30
                item["lease"]["expires_at"] = (datetime.now(timezone.utc) + timedelta(seconds=30)).isoformat()
                return {"granted": True, "lease": dict(item["lease"]), "retry_after_seconds": None, "budget": self.snapshot()}
            req = payload["requested"]
            snapshot = self.snapshot()
            plan = self.admissions.pop(payload["owner_id"], None)
            reserve = self.candidate_reserve(plan) if plan else self.reserve_seconds()
            phase_remaining = self.remaining() - (self.output_reserve() if payload["phase"] == "refit" else reserve)
            if phase_remaining < 1:
                raise AgentError("BUDGET_EXHAUSTED", "Insufficient time for this phase")
            if plan and (not self.can_train(plan)["allowed"]):
                raise AgentError("BUDGET_EXHAUSTED", "Predicted training and finalization no longer fit")
            if req["cpu_threads"] > 3 or req["memory_limit_mb"] > 10240:
                raise AgentError("INVALID_PLAN", "Per-job resource limit exceeded")
            train_busy = payload["phase"] == "train" and sum(item["phase"] == "train" for item in self.leases.values()) >= getattr(self, "train_worker_limit", self.config.max_workers)
            if train_busy or len(self.leases) >= self.config.max_workers or req["cpu_threads"] > snapshot["cpu_threads_available"] or req["memory_limit_mb"] > snapshot["memory_mb_available"]:
                return {"granted": False, "lease": None, "retry_after_seconds": 1, "budget": snapshot}
            self.token += 1
            lease = {"lease_id": "lease-" + uuid.uuid4().hex[:12], "fencing_token": self.token,
                     "cpu_threads": req["cpu_threads"], "memory_limit_mb": req["memory_limit_mb"],
                     "timeout_seconds": max(1, int(min(req["timeout_seconds"], phase_remaining))),
                     "heartbeat_interval_seconds": 10,
                     "expires_at": (datetime.now(timezone.utc) + timedelta(seconds=30)).isoformat()}
            self.leases[lease["lease_id"]] = {"owner_id": payload["owner_id"], "phase": payload["phase"],
                                             "heartbeat_deadline": time.monotonic() + 30, "lease": lease,
                                             "plan": plan}
            return {"granted": True, "lease": dict(lease), "retry_after_seconds": None, "budget": self.snapshot()}
