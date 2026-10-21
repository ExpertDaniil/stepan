"""Append-only JSONL, redacted snapshots and bounded child output."""
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import re
import sys
import threading
import traceback
from contextvars import ContextVar

CURRENT_CALL = ContextVar("current_contract_call", default={})


def utcnow():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class Redactor:
    def __init__(self, secrets=()):
        self.secrets = sorted({x for x in secrets if x and len(x) >= 4}, key=len, reverse=True)

    def text(self, value):
        value = str(value)
        for secret in self.secrets:
            value = value.replace(secret, "[REDACTED]")
        return re.sub(r"(?i)(Bearer\s+)[^\s\"']+", r"\1[REDACTED]", value)

    def clean(self, value):
        if isinstance(value, dict):
            return {str(k): "[REDACTED]" if re.search(r"(?i)^(authorization|api_key|openai_api_key|password|access_token)$", str(k))
                    else self.clean(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self.clean(v) for v in value]
        if isinstance(value, str):
            return self.text(value)
        return value


class RunLogger:
    def __init__(self, root, run_id, *, level="INFO", log_model_io=True, secrets=(), console_format="human"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.level = level
        self.log_model_io = log_model_io
        self.console_format = console_format
        self.redactor = Redactor(secrets)
        self.lock = threading.RLock()
        self.handle = (self.root / "events.jsonl").open("a", encoding="utf-8", buffering=1)

    def event(self, event, *, level="INFO", **fields):
        record = self.redactor.clean({"timestamp": utcnow(), "level": level,
                                      "run_id": self.run_id, "event": event, **CURRENT_CALL.get(), **fields})
        line = json.dumps(record, ensure_ascii=False, allow_nan=False, default=str)
        with self.lock:
            self.handle.write(line + "\n")
            self.handle.flush()
            ranks = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40}
            if ranks.get(level, 20) >= ranks.get(self.level, 20):
                if self.console_format == "json":
                    print(line, file=sys.stderr, flush=True)
                else:
                    omitted = {"timestamp", "level", "run_id", "event", "trace_id", "span_id", "parent_span_id", "traceback"}
                    parts = []
                    for key, value in record.items():
                        if key in omitted or value is None:
                            continue
                        value = json.dumps(value, ensure_ascii=False, default=str) if isinstance(value, (dict, list)) else str(value)
                        parts.append(key + "=" + value.replace("\n", "\\n")[:1600])
                    print(f"{record['timestamp']} {level} [{self.run_id}] {event} " + " ".join(parts), file=sys.stderr, flush=True)
                    if record.get("traceback"):
                        print(record["traceback"].rstrip(), file=sys.stderr, flush=True)

    def snapshot(self, relative, value):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        safe = self.redactor.clean(value)
        with self.lock:
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(json.dumps(safe, ensure_ascii=False, indent=2, allow_nan=False, default=str) + "\n", encoding="utf-8")
            os.replace(tmp, path)
        return str(path.relative_to(self.root))

    def exception(self, event, exc, **fields):
        self.event(event, level="ERROR", message=str(exc),
                   traceback="".join(traceback.format_exception(exc)), **fields)

    def close(self):
        with self.lock:
            if not self.handle.closed:
                self.handle.flush()
                self.handle.close()
