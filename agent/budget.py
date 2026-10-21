from datetime import datetime, timedelta, timezone
import threading
import time
import uuid

from agent.errors import AgentError


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
        self.search_stopped = False
        self.lock = threading.RLock()

    def remaining(self):
        return max(0.0, self.deadline - time.monotonic())

    def reserve_seconds(self):
        # Adapt to observed training. Small synthetic demos have smaller reserves.
        floor = min(60, max(2, self.config.time_budget_seconds * 0.05))
        return min(self.config.time_budget_seconds * 0.35, max(floor, self.last_train_duration * 2.5 + 5))

    def snapshot(self):
        with self.lock:
            cpu = sum(v["lease"]["cpu_threads"] for v in self.leases.values())
            memory = sum(v["lease"]["memory_limit_mb"] for v in self.leases.values())
            return {"wall_seconds_remaining": int(self.remaining()), "finalization_reserve_seconds": int(self.reserve_seconds()),
                    "cpu_threads_available": max(0, 6 - cpu), "memory_mb_available": max(0, 20480 - memory),
                    "llm_calls_remaining": max(0, self.config.llm_max_calls - self.calls),
                    "search_allowed": not self.search_stopped and self.remaining() > self.reserve_seconds()}

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
            phase_remaining = self.remaining() if payload["phase"] == "refit" else self.remaining() - self.reserve_seconds()
            if phase_remaining < 1:
                raise AgentError("BUDGET_EXHAUSTED", "Insufficient time for this phase")
            if req["cpu_threads"] > 3 or req["memory_limit_mb"] > 10240:
                raise AgentError("INVALID_PLAN", "Per-job resource limit exceeded")
            if len(self.leases) >= self.config.max_workers or req["cpu_threads"] > snapshot["cpu_threads_available"] or req["memory_limit_mb"] > snapshot["memory_mb_available"]:
                return {"granted": False, "lease": None, "retry_after_seconds": 1, "budget": snapshot}
            self.token += 1
            lease = {"lease_id": "lease-" + uuid.uuid4().hex[:12], "fencing_token": self.token,
                     "cpu_threads": req["cpu_threads"], "memory_limit_mb": req["memory_limit_mb"],
                     "timeout_seconds": max(1, int(min(req["timeout_seconds"], phase_remaining))),
                     "heartbeat_interval_seconds": 10,
                     "expires_at": (datetime.now(timezone.utc) + timedelta(seconds=30)).isoformat()}
            self.leases[lease["lease_id"]] = {"owner_id": payload["owner_id"], "phase": payload["phase"],
                                             "heartbeat_deadline": time.monotonic() + 30, "lease": lease}
            return {"granted": True, "lease": dict(lease), "retry_after_seconds": None, "budget": self.snapshot()}
