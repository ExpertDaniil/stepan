"""Local calls with the exact C00–C24 envelopes; no HTTP server is needed."""
from copy import deepcopy
import hashlib
import json
import math
import re
import threading
import time
import uuid

from agent.config import ROOT
from agent.errors import AgentError
from agent.observability.logging import CURRENT_CALL


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


class Schemas:
    """Validates the schema vocabulary used by this frozen contract package.

    Fails closed on unsupported validation keywords; uses only stdlib so a demo
    does not need a second dependency installation. Full schemas are also usable
    with the organizers' jsonschema==4.23.0.
    """
    def __init__(self, root=ROOT / "contracts"):
        self.schema = json.loads((root / "contracts.schema.json").read_text(encoding="utf-8"))
        registry = json.loads((root / "contracts.json").read_text(encoding="utf-8"))
        self.operations = {c["id"]: c for c in registry["contracts"]}

    def validate(self, value, name):
        try:
            canonical(value)  # Reject non-JSON objects and NaN even in extensible maps.
            self._check(value, self.schema["$defs"][name], "$")
        except (ValueError, TypeError, KeyError) as exc:
            raise AgentError("CONTRACT_VIOLATION", f"{name}: {exc}") from exc

    def _check(self, v, s, path):
        allowed = {"$ref", "$defs", "$schema", "$id", "title", "description", "type", "enum", "const",
                   "properties", "required", "additionalProperties", "items", "minItems", "maxItems",
                   "minLength", "maxLength", "pattern", "minimum", "maximum", "exclusiveMinimum",
                   "exclusiveMaximum", "oneOf", "anyOf", "default", "examples"}
        if set(s) - allowed:
            raise ValueError(f"unsupported schema keywords: {set(s) - allowed}")
        if "$ref" in s:
            return self._check(v, self.schema["$defs"][s["$ref"].split("/")[-1]], path)
        for key in ("oneOf", "anyOf"):
            if key in s:
                matches = 0
                for candidate in s[key]:
                    try:
                        self._check(v, candidate, path)
                        matches += 1
                    except (ValueError, TypeError, KeyError):
                        pass
                if (key == "oneOf" and matches != 1) or (key == "anyOf" and matches == 0):
                    raise ValueError(f"{path}: {key} matched {matches} schemas")
        if "enum" in s and not any(type(v) is type(e) and v == e for e in s["enum"]):
            raise ValueError(f"{path}: invalid enum {v!r}")
        if "const" in s and (type(v) is not type(s["const"]) or v != s["const"]):
            raise ValueError(f"{path}: const mismatch")
        typ = s.get("type")
        kinds = {"object": isinstance(v, dict), "array": isinstance(v, list), "string": isinstance(v, str),
                 "boolean": isinstance(v, bool), "null": v is None,
                 "integer": isinstance(v, int) and not isinstance(v, bool),
                 "number": isinstance(v, (int, float)) and not isinstance(v, bool)}
        if typ and not kinds[typ]:
            raise ValueError(f"{path}: expected {typ}")
        if isinstance(v, (float, int)) and not isinstance(v, bool) and not math.isfinite(v):
            raise ValueError(f"{path}: non-finite JSON number")
        if typ == "object":
            for k in s.get("required", []):
                if k not in v:
                    raise ValueError(f"{path}: missing {k}")
            for k, x in v.items():
                if k in s.get("properties", {}):
                    self._check(x, s["properties"][k], path + "." + k)
                elif s.get("additionalProperties", True) is False:
                    raise ValueError(f"{path}: unknown field {k}")
                elif isinstance(s.get("additionalProperties"), dict):
                    self._check(x, s["additionalProperties"], path + "." + k)
        if typ == "array":
            if len(v) < s.get("minItems", 0) or len(v) > s.get("maxItems", float("inf")):
                raise ValueError(f"{path}: invalid length")
            for i, x in enumerate(v):
                self._check(x, s.get("items", {}), f"{path}[{i}]")
        if typ == "string":
            if len(v) < s.get("minLength", 0) or len(v) > s.get("maxLength", float("inf")):
                raise ValueError(f"{path}: invalid string length")
            if "pattern" in s and not re.search(s["pattern"], v):
                raise ValueError(f"{path}: invalid string pattern")
        if typ in {"integer", "number"}:
            for key, bad in [("minimum", lambda a, b: a < b), ("maximum", lambda a, b: a > b),
                             ("exclusiveMinimum", lambda a, b: a <= b), ("exclusiveMaximum", lambda a, b: a >= b)]:
                if key in s and bad(v, s[key]):
                    raise ValueError(f"{path}: {key} violation")


class ContractBus:
    def __init__(self, run_id, logger, schemas=None):
        self.run_id, self.log = run_id, logger
        self.schemas = schemas or Schemas()
        self.responses = {}
        self.cache = {}
        self.inflight = {}
        self.lock = threading.RLock()

    def call(self, cid, payload, handler, *, key=None):
        op = self.schemas.operations[cid]
        request_id = "req-" + uuid.uuid4().hex[:16]
        request = {"schema_version": "1.0", "request_id": request_id, "run_id": self.run_id,
                   "idempotency_key": key, "payload": payload}
        started = time.monotonic()
        span_id = "span-" + uuid.uuid4().hex[:12]
        fields = {"contract_id": cid, "request_id": request_id, "trace_id": self.run_id, "span_id": span_id,
                  "parent_span_id": CURRENT_CALL.get().get("span_id"),
                  "sender": op["from"], "receiver": op["to"], "endpoint": op["endpoint"]}
        context_token = CURRENT_CALL.set(fields)
        pending = None
        owns_pending = False
        self.log.event("contract.started", **fields)
        try:
            self.schemas.validate(request, cid + "_Request")
            self.log.snapshot(f"messages/{request_id}.request.json", request)
            digest = hashlib.sha256(canonical(payload).encode()).hexdigest()
            cache_key = (cid, key) if key else None
            with self.lock:
                old = self.cache.get(cache_key) if cache_key else None
                if old and old[0] != digest:
                    raise AgentError("IDEMPOTENCY_CONFLICT", "Idempotency key reused with a different payload")
                if cache_key and not old:
                    pending = self.inflight.get(cache_key)
                    if pending and pending["digest"] != digest:
                        raise AgentError("IDEMPOTENCY_CONFLICT", "Concurrent key reused with a different payload")
                    if pending is None:
                        pending = {"digest": digest, "event": threading.Event(), "failed": False}
                        self.inflight[cache_key] = pending
                        owns_pending = True
            # State-changing handlers have their own transactions / publisher lock.
            # No lock is held during nested calls or slow LLM work.
            if pending and not owns_pending:
                if not pending["event"].wait(op["timeout_seconds"]):
                    raise AgentError("REQUEST_IN_PROGRESS", "Identical operation is still running", retryable=True)
                if pending["failed"]:
                    raise AgentError("UPSTREAM_REQUEST_FAILED", "The identical in-flight request failed")
                with self.lock:
                    old = self.cache[cache_key]
            if old:
                data = deepcopy(old[1])
                self.log.event("contract.replayed", **fields)
            else:
                data = handler(payload)
            response = {"schema_version": "1.0", "request_id": request_id, "run_id": self.run_id,
                        "ok": True, "data": data, "error": None}
            self.schemas.validate(response, cid + "_Response")
            if cache_key:
                with self.lock:
                    self.cache[cache_key] = (digest, deepcopy(data))
        except Exception as exc:
            if owns_pending:
                pending["failed"] = True
            err = exc if isinstance(exc, AgentError) else AgentError("INTERNAL_ERROR", str(exc))
            if err.code == "CONTRACT_VIOLATION":
                self.log.snapshot(f"messages/{request_id}.invalid.json", {"invalid_request_repr": repr(request)[:20000], "error": str(err)})
            response = {"schema_version": "1.0", "request_id": request_id, "run_id": self.run_id,
                        "ok": False, "data": None, "error": err.as_dict()}
            self.schemas.validate(response, cid + "_Response")
            self.log.snapshot(f"messages/{request_id}.response.json", response)
            self.log.event("contract.failed", level="ERROR", **fields, **err.as_dict(),
                           duration_ms=round((time.monotonic() - started) * 1000, 1))
            if err is exc:
                raise
            raise err from exc
        finally:
            if owns_pending:
                with self.lock:
                    self.inflight.pop(cache_key, None)
                    pending["event"].set()
            CURRENT_CALL.reset(context_token)
        self.log.snapshot(f"messages/{request_id}.response.json", response)
        self.log.event("contract.completed", **fields, duration_ms=round((time.monotonic() - started) * 1000, 1))
        with self.lock:
            self.responses[request_id] = response
        return response
