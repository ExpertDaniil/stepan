from copy import deepcopy
import json
import os
import re
import threading
import time
import uuid

from agent.errors import AgentError
from tools.code import write_code

DEFAULT_MODEL = "qwen/qwen3.8-27b"  # Permitted by the updated organizers' instructions.


def request_model(mode):
    return os.environ.get("OPENAI_MODEL", "").strip() or (DEFAULT_MODEL if mode == "live" else "mock-only")


def parse_json(text):
    text = (text or "").strip()
    if text.startswith("```"):
        match = re.fullmatch(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        if not match:
            raise ValueError("Malformed JSON code fence")
        text = match[1]
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON key: " + key)
            result[key] = value
        return result
    return json.loads(text, object_pairs_hook=unique_object,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Non-finite JSON: " + value)))


class OpenAITransport:
    def __init__(self):
        missing = [k for k in ("OPENAI_API_KEY", "OPENAI_BASE_URL") if not os.environ.get(k, "").strip()]
        if missing:
            raise AgentError("INVALID_CONFIG", "Missing live API environment variables", details={"missing": missing})
        if not os.environ["OPENAI_BASE_URL"].startswith(("http://", "https://")):
            raise AgentError("INVALID_CONFIG", "OPENAI_BASE_URL must be an HTTP(S) endpoint")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise AgentError("LLM_UNAVAILABLE", "Install requirements.txt to use the provided API") from exc
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url=os.environ["OPENAI_BASE_URL"], max_retries=0)

    def complete(self, request, timeout, context):
        return self.client.chat.completions.create(**request, timeout=timeout).model_dump(exclude_none=True)

    def close(self):
        self.client.close()


class MockTransport:
    """Explicit synthetic-demo adapter. Never selected after a live API failure."""
    def complete(self, request, timeout, context):
        purpose = context["metadata"]["schema_id"]
        demo = context["metadata"].get("demo_spec")
        if not demo:
            raise AgentError("INVALID_CONFIG", "Mock mode requires a task created by scripts/make_demo.py")
        if purpose == "TaskDraft-v1":
            payload = demo["draft"]
        elif purpose == "ValidationProposal-v1":
            payload = demo.get("validation") or {"strategy": "random_holdout" if demo["draft"]["task_kind"] == "regression" else "stratified_holdout",
                                                  "parameters": {"valid_fraction": 0.2}, "seed": 42,
                                                  "reason": "Synthetic demo validation protocol."}
        else:
            from agent.catalog import factory_source
            choices = context["metadata"]["recipe_ids"]
            round_no = context["metadata"]["round_no"]
            researcher = context["metadata"]["researcher_id"]
            preferred = "linear-" if researcher == "researcher-1" else "hist-gradient-"
            recipe = next((x for x in choices if x.startswith(preferred)), choices[0])
            params = {"alpha": 1.0 + round_no} if "regressor" in recipe and recipe.startswith("linear-") else {"C": 1.0 / (round_no + 1)} if recipe.startswith("linear-") else {"max_iter": 40 + 10 * round_no}
            payload = {"recipe_id": recipe, "hyperparameters": params,
                       "hypothesis": "Check a catalogue alternative against the recorded incumbent on the same split.",
                       "expected_effect": "An evidence-based comparison; improvement is not assumed.",
                       "feature_steps": [], "code": factory_source(recipe)}
        return {"id": "mock-" + uuid.uuid4().hex[:12],
                "choices": [{"message": {"role": "assistant", "content": json.dumps(payload, ensure_ascii=False)}, "finish_reason": "stop"}]}

    def close(self):
        pass


class Gateway:
    def __init__(self, config, budget, artifacts, bus, log, output_schemas, transport=None):
        self.config, self.budget, self.artifacts, self.bus, self.log = config, budget, artifacts, bus, log
        self.output_schemas = output_schemas
        self.transport = transport or (MockTransport() if config.llm_mode == "mock" else OpenAITransport())
        self.semaphore = threading.BoundedSemaphore(config.max_workers)
        self.daily_budget_exhausted = threading.Event()

    def check_daily_budget(self):
        if self.daily_budget_exhausted.is_set():
            raise AgentError("LLM_BUDGET_EXHAUSTED", "The shared provider key budget is exhausted; reset is at 03:00 MSK",
                             details={"reset_time_msk": "03:00", "daily_key_limit_rub": 600})

    def complete(self, payload):
        context = self.artifacts.read_json(payload["context_artifact_id"])
        messages = deepcopy(context["messages"])
        expected = self.output_schemas[payload["response_schema_id"]]
        def expanded(value):
            if isinstance(value, dict):
                if "$ref" in value:
                    return expanded(self.bus.schemas.schema["$defs"][value["$ref"].split("/")[-1]])
                return {k: expanded(v) for k, v in value.items()}
            return [expanded(v) for v in value] if isinstance(value, list) else value
        messages.insert(1, {"role": "system", "content": "Final output JSON schema:\n" + json.dumps(expanded(expected), ensure_ascii=False)})
        deadline = min(time.monotonic() + payload["deadline_seconds"], self.budget.deadline - 0.5)
        tools = []
        if "write_code" in payload["tool_schema_ids"]:
            tools.append({"type": "function", "function": {
                "name": "write_code", "description": "Save Python source exporting make_pipeline(task_spec, params).",
                "parameters": self.bus.schemas.schema["$defs"]["C11_Request"]["properties"]["payload"]}})
        usage = {"input_tokens": None, "output_tokens": None}
        calls = []
        last_error = None
        last_art = None
        completion_id = "completion-" + uuid.uuid4().hex[:12]
        for attempt in range(1, payload["attempt_limit"] + 1):
            self.check_daily_budget()
            left = deadline - time.monotonic()
            if left <= 0:
                raise AgentError("LLM_UNAVAILABLE", "LLM operation deadline reached")
            if not self.semaphore.acquire(timeout=left):
                raise AgentError("LLM_UNAVAILABLE", "Timed out waiting for LLM concurrency slot")
            try:
                self.check_daily_budget()
                left = deadline - time.monotonic()
                if left <= 0:
                    raise AgentError("BUDGET_EXHAUSTED", "Deadline reached while waiting for API slot")
                call_no = self.budget.llm_call()
                request = {"model": request_model(self.config.llm_mode), "messages": messages,
                           "max_tokens": payload["max_output_tokens"]}
                if tools:
                    request.update(tools=tools, parallel_tool_calls=False)
                if self.log.log_model_io:
                    self.log.snapshot(f"llm/{call_no:04d}.request.json", request)
                self.log.event("llm.call_started", call_no=call_no, purpose=payload["purpose"],
                               context_artifact_id=payload["context_artifact_id"], mode=self.config.llm_mode,
                               destination=self._destination())
                started = time.monotonic()
                response = self.transport.complete(request, max(0.1, min(left, self.config.llm_timeout_seconds)), context)
                if self.log.log_model_io:
                    self.log.snapshot(f"llm/{call_no:04d}.response.json", response)
                completion_id = response.get("id", completion_id)
                token_data = response.get("usage", {})
                for dst, src in [("input_tokens", "prompt_tokens"), ("output_tokens", "completion_tokens")]:
                    if isinstance(token_data.get(src), int):
                        usage[dst] = (usage[dst] or 0) + token_data[src]
                self.log.event("llm.call_completed", call_no=call_no, duration_ms=round((time.monotonic() - started) * 1000, 1), usage=usage)
            except AgentError:
                raise
            except Exception as exc:
                status = getattr(exc, "status_code", None)
                body = str(exc) + " " + json.dumps(getattr(exc, "body", None), default=str)
                if "budget has been exceeded" in body.casefold():
                    self.daily_budget_exhausted.set()
                    self.log.event("llm.call_failed", level="WARNING", code="LLM_BUDGET_EXHAUSTED",
                                   message="The shared provider daily key budget is exhausted", attempt=attempt,
                                   status_code=status, retryable=False)
                    self.log.event("llm.daily_budget_exhausted", level="WARNING", status_code=status,
                                   reset_time_msk="03:00", daily_key_limit_rub=600)
                    self.check_daily_budget()
                retryable = status == 429 or status is None or (isinstance(status, int) and status >= 500)
                last_error = AgentError("LLM_RATE_LIMIT" if status == 429 else "LLM_UNAVAILABLE", str(exc), retryable=retryable)
                self.log.event("llm.call_failed", level="WARNING", code=last_error.code, message=str(exc),
                               attempt=attempt, status_code=status, retryable=retryable)
                if not retryable or attempt == payload["attempt_limit"]:
                    raise last_error from exc
                delay = min(2 ** (attempt - 1), max(0, deadline - time.monotonic() - 0.1))
                time.sleep(delay)
                continue
            finally:
                self.semaphore.release()
            try:
                message = response["choices"][0]["message"]
                if message.get("tool_calls"):
                    assistant = {k: message[k] for k in ("role", "content", "tool_calls", "reasoning_content") if k in message}
                    messages.append(assistant)
                    for call in message["tool_calls"]:
                        if call["function"]["name"] != "write_code" or not tools:
                            raise AgentError("INVALID_LLM_OUTPUT", "Unexpected model tool call")
                        args = parse_json(call["function"]["arguments"])
                        if args.get("researcher_id") != context["metadata"].get("researcher_id"):
                            raise AgentError("INVALID_LLM_OUTPUT", "Tool call uses another researcher's namespace")
                        result = self.bus.call("C11", args, lambda p: write_code(p, self.artifacts),
                                               key="code/" + self._digest(args))
                        calls.append({"call_id": call["id"], "name": "write_code", "request_id": result["request_id"]})
                        messages.append({"role": "tool", "tool_call_id": call["id"], "content": json.dumps(result, ensure_ascii=False)})
                    continue
                parsed = parse_json(message.get("content"))
                self.bus.schemas._check(parsed, expected, "$")
                last_art = self.artifacts.write_json("llm/parsed-" + uuid.uuid4().hex[:12] + ".json", parsed, "llm_output")
                return {"completion_id": completion_id, "output_artifact_id": last_art["artifact_id"],
                        "parsed_payload": parsed, "tool_calls": calls, "usage": usage, "attempts_used": attempt}
            except AgentError:
                raise
            except (ValueError, KeyError, TypeError, IndexError) as exc:
                last_error = AgentError("INVALID_LLM_OUTPUT", f"Model output did not match {payload['response_schema_id']}: {exc}")
                self.log.event("llm.output_rejected", level="WARNING", message=str(last_error), attempt=attempt)
                messages.append({"role": "user", "content": "Return one valid JSON object matching the requested schema. Error: " + str(exc)[:600]})
        raise last_error or AgentError("INVALID_LLM_OUTPUT", "No final JSON within the bounded tool/repair loop")

    @staticmethod
    def _digest(value):
        import hashlib
        from agent.contracts import canonical
        return hashlib.sha256(canonical(value).encode()).hexdigest()

    def _destination(self):
        from urllib.parse import urlsplit
        if self.config.llm_mode == "mock":
            return "local-mock"
        parsed = urlsplit(os.environ.get("OPENAI_BASE_URL", ""))
        host = parsed.hostname or ""
        if parsed.port:
            host += ":" + str(parsed.port)
        return parsed.scheme + "://" + host + parsed.path

    def close(self):
        self.transport.close()
