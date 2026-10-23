"""Local calls with the exact C00–C24 envelopes; no HTTP server is needed."""
from copy import deepcopy
from functools import lru_cache
import hashlib
import json
import threading
import time
import uuid

from agent.config import ROOT
from agent.errors import AgentError
from agent.observability.logging import CURRENT_CALL


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


@lru_cache(maxsize=8)
def _validator(serialized):
    """Compile a local contract document once per process; never retrieve URLs."""
    from jsonschema import Draft202012Validator, validators
    from referencing import Registry, Resource
    schema = json.loads(serialized)
    Draft202012Validator.check_schema(schema)
    # Resource quantities and row indices deliberately require JSON integer
    # tokens, preserving the original contract (3.0 is not an integer here).
    checker = Draft202012Validator.TYPE_CHECKER.redefine(
        'integer', lambda _, value: isinstance(value, int) and not isinstance(value, bool))
    validator_type = validators.extend(Draft202012Validator, type_checker=checker)
    registry = Registry().with_resource(schema['$id'], Resource.from_contents(schema))
    return validator_type(schema, registry=registry)


class Schemas:
    """Pinned JSON Schema validation with the project's existing error boundary."""
    def __init__(self, root=ROOT / "contracts"):
        self.schema = json.loads((root / "contracts.schema.json").read_text(encoding="utf-8"))
        registry = json.loads((root / "contracts.json").read_text(encoding="utf-8"))
        self.operations = {c["id"]: c for c in registry["contracts"]}
        self.validator = _validator(canonical(self.schema))

    def validate(self, value, name):
        try:
            self._check(value, self.schema["$defs"][name], "$")
        except (ValueError, TypeError, KeyError) as exc:
            raise AgentError("CONTRACT_VIOLATION", f"{name}: {exc}") from exc

    def _check(self, value, schema, path):
        from jsonschema.exceptions import ValidationError
        from referencing.exceptions import Unresolvable
        canonical(value)  # Finite JSON only, including extensible parameter maps.
        try:
            self.validator.evolve(schema=schema).validate(value)
        except ValidationError as exc:
            location = path + ''.join('['+str(p)+']' if isinstance(p, int) else '.'+p for p in exc.absolute_path)
            detail = ('does not match the permitted schema variants' if exc.validator in {'oneOf', 'anyOf'}
                      else exc.message[:600])
            raise ValueError(location+': '+detail) from exc
        except Unresolvable as exc:
            raise ValueError(path+': schema reference is not in the local registry') from exc


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
