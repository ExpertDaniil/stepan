from datetime import datetime, timedelta, timezone
import threading
import time
import uuid
import math
from collections import deque

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
        self.llm_dialogues = {}
        self.last_train_duration = 0.0
        self.duration = DurationModel(config)
        self.final_plan = None
        self.admissions = {}
        self.search_stopped = False
        self.lock = threading.RLock()
        self.phase_call_limit = None
        self.phase_output_limit = None
        self.phase_deadline = None
        self.llm_durations = deque(maxlen=32)
        self.analysis_active = False
        self.analysis_seconds_total = 0.0

    def set_phase_caps(self, *, calls=None, output_tokens=None, deadline=None):
        with self.lock:
            self.phase_call_limit = calls
            self.phase_output_limit = output_tokens
            self.phase_deadline = deadline

    def phase_remaining(self):
        with self.lock:
            return max(0.0, (self.phase_deadline or self.deadline) - time.monotonic())

    def observe_llm(self, seconds):
        with self.lock:
            self.llm_durations.append(seconds)

    def reasoning_estimate(self):
        with self.lock:
            values = sorted(self.llm_durations)
            return (values[math.ceil(.75*len(values))-1]*1.5 if values else
                    min(self.config.llm_operation_timeout_seconds, max(1, self.config.time_budget_seconds*.01)))

    def call_limit(self):
        return min(self.global_call_limit(), self.phase_call_limit if self.phase_call_limit is not None else math.inf)

    def output_limit(self):
        return min(self.global_output_limit(), self.phase_output_limit if self.phase_output_limit is not None else math.inf)

    def global_call_limit(self):
        return self.config.llm_max_calls if self.config.llm_max_calls is not None else math.inf

    def global_output_limit(self):
        return self.config.llm_max_total_output_tokens if self.config.llm_max_total_output_tokens is not None else math.inf

    @staticmethod
    def bounded_count(value):
        return max(0, value) if math.isfinite(value) else None

    def llm_unavailable_reason(self, pending=0):
        with self.lock:
            if self.phase_remaining() <= self.reserve_seconds()+1 or self.remaining() <= self.reserve_seconds()+1:
                return 'time_reserve'
            if self.call_limit()-self.calls-self._dialogue_calls() <= pending:
                return 'llm_call_limit'
            if self.output_limit()-self.output_tokens_charged-sum(self.pending_llm_tokens.values())-self._dialogue_tokens() < (pending+1)*self.config.llm_max_output_tokens:
                return 'llm_output_budget'
            return 'search_closed' if self.search_stopped else None

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
                       and self.remaining() >= estimate + reserve + 1
                       and (self.phase_deadline is None or self.phase_remaining() >= estimate + reserve + 1))
            return {"allowed": allowed, "estimated_seconds": estimate, "reserve_seconds": reserve,
                    "remaining_seconds": self.remaining(), "phase_seconds_remaining": self.phase_remaining(),
                    "family": plan["model"]["family"]}

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
                    "cpu_threads_available": max(0, 6 - cpu - int(self.analysis_active)), "memory_mb_available": max(0, 20480 - memory - 1024*int(self.analysis_active)),
                    "llm_calls_remaining": self.bounded_count(self.call_limit() - self.calls - self._dialogue_calls()),
                    "search_allowed": not self.search_stopped and self.remaining() - self.reserve_seconds() >= 1}

    def _dialogue_calls(self):
        return sum(d['slots'] for d in self.llm_dialogues.values())

    def _dialogue_tokens(self):
        return sum(d['slots']*d['max_tokens'] for d in self.llm_dialogues.values())

    def reserve_dialogue(self, max_tokens, *, repair=True):
        """Protect future final/repair requests; no HTTP attempt is charged yet."""
        with self.lock:
            calls = self.call_limit()-self.calls-self._dialogue_calls()
            tokens = self.output_limit()-self.output_tokens_charged-sum(self.pending_llm_tokens.values())-self._dialogue_tokens()
            slots = min(2 if repair else 1, calls, tokens//max_tokens if math.isfinite(tokens) else math.inf)
            if slots < 1:
                phase = self.call_limit() < self.global_call_limit() or self.output_limit() < self.global_output_limit()
                raise AgentError('SEARCH_PHASE_BUDGET_EXHAUSTED' if phase else 'BUDGET_EXHAUSTED',
                                 'No unreserved capacity for a final research decision')
            ident = 'dialogue-'+uuid.uuid4().hex[:16]
            self.llm_dialogues[ident] = {'slots': slots, 'max_tokens': max_tokens, 'terminal': False}
            return ident, slots

    def release_dialogue(self, ident):
        with self.lock:
            return self.llm_dialogues.pop(ident, None)

    def affordable_dialogue_output(self, ident, requested):
        with self.lock:
            item = self.llm_dialogues[ident]
            free = self.output_limit()-self.output_tokens_charged-sum(self.pending_llm_tokens.values())-self._dialogue_tokens()
            if item['slots'] < 1:
                return item['max_tokens']
            growth = (requested-item['max_tokens'])*item['slots']
            return (requested if growth <= free else
                    item['max_tokens']+max(0, int(free//item['slots'])))

    def prepare_dialogue_turn(self, ident, max_tokens, *, allow_tools):
        """Atomically fund an optional tool request or use the protected final slot."""
        with self.lock:
            item = self.llm_dialogues[ident]
            if item['slots'] < 1:
                raise AgentError('BUDGET_EXHAUSTED', 'The reserved final/repair requests have been consumed')
            free_tokens = self.output_limit()-self.output_tokens_charged-sum(self.pending_llm_tokens.values())-self._dialogue_tokens()
            growth = item['slots']*(max_tokens-item['max_tokens'])
            if growth > free_tokens:
                raise AgentError('BUDGET_EXHAUSTED', 'Cannot enlarge reserved final output without consuming another dialogue reservation')
            item['max_tokens'] = max_tokens
            free_tokens -= growth
            calls = self.call_limit()-self.calls-self._dialogue_calls()
            if allow_tools and not item['terminal'] and calls > 0 and free_tokens >= max_tokens:
                item['slots'] += 1
                return False
            item['terminal'] = True
            return True

    def llm_call(self, dialogue_id=None):
        with self.lock:
            if self.phase_deadline and self.phase_remaining() < 1:
                raise AgentError('SEARCH_PHASE_BUDGET_EXHAUSTED', 'Branch exploration reached its time boundary')
            if self.calls >= self.global_call_limit() or self.remaining() < 1:
                raise AgentError("BUDGET_EXHAUSTED", "LLM budget exhausted")
            own = 1 if dialogue_id is not None and self.llm_dialogues[dialogue_id]['slots'] > 0 else 0
            if self.calls+self._dialogue_calls()-own >= self.call_limit():
                phase = self.call_limit() < self.global_call_limit()
                raise AgentError('SEARCH_PHASE_BUDGET_EXHAUSTED' if phase else 'BUDGET_EXHAUSTED',
                                 'No unreserved API attempts remain; active dialogue and phase reservations are protected')
            self.calls += 1
            if own:
                self.llm_dialogues[dialogue_id]['slots'] -= 1
            return self.calls

    def begin_llm_call(self, max_tokens, dialogue_id=None):
        """Reserve per-run token capacity and one API attempt atomically."""
        with self.lock:
            owner = self.llm_dialogues.get(dialogue_id) if dialogue_id is not None else None
            if dialogue_id is not None and (owner is None or owner['slots'] < 1 or max_tokens > owner['max_tokens']):
                raise AgentError('BUDGET_EXHAUSTED', 'No funded request remains in this dialogue')
            held = self._dialogue_tokens()-(owner['max_tokens'] if owner else 0)
            available = self.global_output_limit() - self.output_tokens_charged - sum(self.pending_llm_tokens.values()) - held
            if max_tokens > available:
                raise AgentError("BUDGET_EXHAUSTED", "Per-run output token budget cannot reserve this request",
                                 details={"requested": max_tokens, "available": max(0, available)})
            phase_available = self.output_limit()-self.output_tokens_charged-sum(self.pending_llm_tokens.values())-held
            if max_tokens > phase_available:
                raise AgentError("SEARCH_PHASE_BUDGET_EXHAUSTED", "Reserved output tokens belong to final refinement",
                                 details={"requested": max_tokens, "available": max(0, phase_available)})
            call_no = self.llm_call(dialogue_id)
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
            inflight, dialogues = sum(self.pending_llm_tokens.values()), self._dialogue_tokens()
            reserved = inflight+dialogues
            return {"output_limit": self.config.llm_max_total_output_tokens, "output_charged": self.output_tokens_charged,
                    "output_reported": self.output_tokens_reported, "output_reserved": reserved,
                    "output_reserved_inflight": inflight, "output_reserved_dialogues": dialogues,
                    "llm_calls_reserved_dialogues": self._dialogue_calls(),
                    "output_available": self.bounded_count(self.output_limit() - self.output_tokens_charged - reserved),
                    "output_available_global": self.bounded_count(self.global_output_limit() - self.output_tokens_charged - reserved),
                    "phase_call_limit": self.phase_call_limit, "phase_output_limit": self.phase_output_limit,
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

    def analysis_slot(self):
        """Serialize 1 CPU / 1 GiB logical analysis admission with train leases.

        Cooperative query limits enforce bounded data and time; this is not a
        new OS-level memory sandbox. The platform's total limit remains binding.
        """
        from contextlib import contextmanager
        @contextmanager
        def slot():
            with self.lock:
                snapshot = self.snapshot()
                if (self.analysis_active or not snapshot['search_allowed']
                        or snapshot['cpu_threads_available'] < 1 or snapshot['memory_mb_available'] < 1024):
                    raise AgentError('ANALYSIS_RESOURCES_BUSY', 'No shared analysis slot; use saved evidence or choose an affordable action', retryable=True)
                self.analysis_active = True
            started = time.monotonic()
            self.log.event('analysis.resource_acquired', cpu_threads=1, memory_mb=1024)
            try:
                yield
            finally:
                elapsed = time.monotonic()-started
                with self.lock:
                    self.analysis_active = False
                    self.analysis_seconds_total += elapsed
                self.log.event('analysis.resource_released', duration_seconds=elapsed,
                               analysis_seconds_total=self.analysis_seconds_total)
        return slot()
