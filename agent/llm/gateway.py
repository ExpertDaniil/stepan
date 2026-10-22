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
            if recipe.startswith("linear-"):
                params = {"alpha": 1.0+round_no} if "regressor" in recipe else {"C": 1.0/(round_no+1)}
            elif recipe.startswith("catboost-"):
                params = {"iterations": 100+20*round_no}
            elif recipe.startswith(("random-forest-", "extra-trees-")):
                params = {"n_estimators": 60+10*round_no}
            else:
                params = {"max_iter": 40+10*round_no}
            payload = {"recipe_id": recipe, "hyperparameters": params,
                       "hypothesis": "Check a catalogue alternative against the recorded incumbent on the same split.",
                       "expected_effect": "An evidence-based comparison; improvement is not assumed.",
                       "feature_steps": [], "code": factory_source(recipe)}
            if purpose == "ResearchChoice-v1":
                payload.pop("feature_steps")
                payload.pop("code")
            elif purpose == 'ResearchAction-v1':
                payload.update(action='catalog', code=None, tuning=None, evidence_ids=[])
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
        self.provider_credits_exhausted = threading.Event()
        self.analysis = None

    def check_daily_budget(self):
        if self.daily_budget_exhausted.is_set():
            raise AgentError("LLM_BUDGET_EXHAUSTED", "The shared provider key budget is exhausted; reset is at 03:00 MSK",
                             details={"reset_time_msk": "03:00", "daily_key_limit_rub": 600})

    def check_availability(self):
        if self.provider_credits_exhausted.is_set():
            raise AgentError("LLM_PROVIDER_CREDITS_EXHAUSTED",
                             "The LLM provider reported Insufficient credits. Contact the gateway operator to restore provider access; "
                             "the team's daily balance and reset time are unknown.",
                             details={"scope": "upstream_provider", "team_daily_budget_status": "unknown", "reset_time_msk": None})
        self.check_daily_budget()

    def _progress(self, done, call_no, purpose, started, timeout):
        while not done.wait(self.config.llm_progress_seconds):
            self.log.event("llm.waiting", call_no=call_no, purpose=purpose,
                           elapsed_seconds=round(time.monotonic() - started, 1), request_timeout_seconds=round(timeout, 1),
                           run_seconds_remaining=round(self.budget.remaining(), 1))

    def _request(self, request, context, payload, attempt, deadline):
        from contextvars import copy_context
        request_bytes = len(json.dumps(request, ensure_ascii=False, allow_nan=False).encode("utf-8"))
        byte_limit = self.config.max_context_tokens * 3
        if context["metadata"].get("graph_binding"):
            user = json.loads(context["messages"][1]["content"])
            byte_limit = min(byte_limit, self.config.graph_context_bytes, user["profile_summary"]["context_budget"]["max_api_request_bytes"])
        if request_bytes > byte_limit and context['metadata']['schema_id'] == 'ResearchAction-v1':
            self._compact_research_request(request, byte_limit)
            request_bytes = len(json.dumps(request, ensure_ascii=False, allow_nan=False).encode('utf-8'))
        if request_bytes > byte_limit:
            raise AgentError("CONTEXT_TOO_LARGE", "Complete API request including schema/tools/repair exceeds the byte budget",
                             details={"request_bytes": request_bytes, "limit_bytes": byte_limit})
        self.log.event("llm.request_budget_checked", level="DEBUG", request_bytes=request_bytes, limit_bytes=byte_limit)
        timeout = min(self.config.llm_timeout_seconds, deadline - time.monotonic())
        if timeout < 1:
            raise AgentError("LLM_OPERATION_TIMEOUT", "Insufficient operation time for another API attempt")
        self.check_availability()
        call_no = self.budget.begin_llm_call(request["max_tokens"])
        settled = False
        started = time.monotonic()
        done = threading.Event()
        progress = threading.Thread(target=copy_context().run,
            args=(self._progress, done, call_no, payload["purpose"], started, timeout), daemon=True)
        try:
            if self.log.log_model_io:
                self.log.snapshot(f"llm/{call_no:04d}.request.json", request)
            self.log.event("llm.call_started", call_no=call_no, attempt=attempt, purpose=payload["purpose"],
                           model=request["model"], max_output_tokens=request["max_tokens"],
                           request_timeout_seconds=round(timeout, 1), operation_seconds_remaining=round(deadline-time.monotonic(), 1),
                           context_artifact_id=payload["context_artifact_id"], mode=self.config.llm_mode,
                           destination=self._destination(), token_budget=self.budget.token_snapshot())
            progress.start()
            response = self.transport.complete(request, max(0.1, min(timeout, deadline-time.monotonic())), context)
            if payload['purpose'] == 'research_proposal':
                self.budget.observe_llm(time.monotonic()-started)
            token_data = response.get("usage") or {}
            self.budget.finish_llm_call(call_no, token_data)
            settled = True
            if self.log.log_model_io:
                self.log.snapshot(f"llm/{call_no:04d}.response.json", response)
            choice = (response.get("choices") or [{}])[0]
            message = choice.get("message") or {}
            output = token_data.get("completion_tokens")
            reasoning = (token_data.get("completion_tokens_details") or {}).get("reasoning_tokens")
            self.log.event("llm.call_completed", call_no=call_no, attempt=attempt, purpose=payload["purpose"],
                           duration_ms=round((time.monotonic()-started)*1000, 1), finish_reason=choice.get("finish_reason"),
                           content_chars=len(message.get("content") or ""), tool_calls=len(message.get("tool_calls") or []),
                           input_tokens=token_data.get("prompt_tokens"), output_tokens=output, reasoning_tokens=reasoning,
                           cached_input_tokens=(token_data.get("prompt_tokens_details") or {}).get("cached_tokens"),
                           usage_inconsistent=isinstance(reasoning, int) and isinstance(output, int) and reasoning > output,
                           token_budget=self.budget.token_snapshot())
            return response, call_no
        except AgentError:
            raise
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            body = str(exc) + " " + json.dumps(getattr(exc, "body", None), default=str)
            if "insufficient credits" in body.casefold():
                self.provider_credits_exhausted.set()
                self.log.event("llm.provider_credits_exhausted", level="ERROR", call_no=call_no,
                               status_code=status, reason="insufficient_credits", retryable=False,
                               team_daily_budget_status="unknown", reset_time_msk=None)
                try:
                    self.check_availability()
                except AgentError as error:
                    self.log.event("llm.call_failed", level="ERROR", call_no=call_no, code=error.code,
                                   message=str(error), attempt=attempt, status_code=status, retryable=False)
                    raise error from exc
            if "budget has been exceeded" in body.casefold():
                self.daily_budget_exhausted.set()
                self.log.event("llm.daily_budget_exhausted", level="WARNING", call_no=call_no,
                               status_code=status, reset_time_msk="03:00", daily_key_limit_rub=600)
                self.check_daily_budget()
            retryable = status == 429 or status is None or (isinstance(status, int) and status >= 500)
            error = AgentError("LLM_RATE_LIMIT" if status == 429 else "LLM_UNAVAILABLE", str(exc), retryable=retryable)
            self.log.event("llm.call_failed", level="WARNING", call_no=call_no, code=error.code, message=str(exc),
                           attempt=attempt, status_code=status, retryable=retryable)
            raise error from exc
        finally:
            done.set()
            if progress.ident is not None:
                progress.join(timeout=0.2)
            if not settled:
                # A timed-out request may have been billed without returning usage.
                # Charge its reservation conservatively; never fabricate reported usage.
                self.budget.finish_llm_call(call_no)

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
        reserve = self.budget.reserve_seconds() + 1 if payload["purpose"] == "research_proposal" else .5
        deadline = min(time.monotonic() + payload["deadline_seconds"], self.budget.deadline - reserve)
        phase_limited = False
        if payload['purpose'] == 'research_proposal' and self.budget.phase_deadline:
            # This is a response deadline only. A received proposal can wait in
            # the dispatcher without inheriting the API timeout.
            user = json.loads(context['messages'][1]['content']) if context['metadata'].get('graph_binding') else {}
            base = user.get('base_result')
            recipe = next((r for r in user.get('profile_summary', {}).get('recipes', [])
                           if base and r['id'] == base['recipe_id']), None)
            after_response = self.budget.reserve_seconds()
            if recipe:
                probe = {'recipe_id': recipe['id'], 'model': {'family': recipe['family'], 'hyperparameters': base['hyperparameters']},
                         'resources': {'cpu_threads': self.config.train_cpu_threads}}
                after_response = self.budget.duration.estimate(probe)+self.budget.candidate_reserve(probe)
            phase_deadline = self.budget.phase_deadline-after_response
            phase_limited = phase_deadline < deadline
            deadline = min(deadline, phase_deadline)
        tools = []
        if "write_code" in payload["tool_schema_ids"]:
            tools.append({"type": "function", "function": {
                "name": "write_code", "description": "Save Python source exporting make_pipeline(task_spec, params).",
                "parameters": expanded(self.bus.schemas.schema["$defs"]["C11_Request"]["properties"]["payload"])}})
        if 'analyze_data' in payload['tool_schema_ids']:
            from tools.analysis import ANALYSIS_SCHEMA
            if self.analysis is None:
                raise AgentError('ANALYSIS_UNAVAILABLE', 'No fit-only analysis service is configured')
            tools.append({'type': 'function', 'function': {'name': 'analyze_data',
                'description': 'Read-only fit summary/association/group, unlabelled test comparison, dictionary, or saved report via report_id. No holdout labels. columns contains input features only, never the target: use include_target=true for target summaries. For group, group_by must be one of columns. At most 12 columns; [] selects defaults.',
                'parameters': ANALYSIS_SCHEMA}})
        token_limit = min(payload["max_output_tokens"], self.config.llm_max_output_tokens_limit)
        usage = {"input_tokens": None, "output_tokens": None}
        calls, last_error = [], None
        completion_id = "completion-" + uuid.uuid4().hex[:12]
        for attempt in range(1, payload["attempt_limit"] + 1):
            self.check_availability()
            left = deadline-time.monotonic()
            if left < 1:
                raise AgentError('SEARCH_PHASE_BUDGET_EXHAUSTED' if phase_limited else 'LLM_OPERATION_TIMEOUT',
                                 "LLM operation deadline reached before a new attempt")
            if not self.semaphore.acquire(timeout=left):
                raise AgentError("LLM_OPERATION_TIMEOUT", "Timed out waiting for LLM concurrency slot")
            try:
                request = {"model": request_model(self.config.llm_mode), "messages": deepcopy(messages), "max_tokens": token_limit}
                if tools:
                    request.update(tools=tools, parallel_tool_calls=False)
                response, call_no = self._request(request, context, payload, attempt, deadline)
            except AgentError as exc:
                if phase_limited and time.monotonic() >= deadline:
                    raise AgentError('SEARCH_PHASE_BUDGET_EXHAUSTED', 'Response window ended with time reserved for training and publication') from exc
                if not exc.retryable or attempt == payload["attempt_limit"]:
                    raise
                last_error = exc
                delay = min(2 ** (attempt-1), max(0, deadline-time.monotonic()-1))
                self.log.event("llm.retry_scheduled", level="WARNING", code=exc.code, attempt=attempt, delay_seconds=delay)
                time.sleep(delay)
                continue
            finally:
                self.semaphore.release()
            token_data = response.get("usage") or {}
            completion_id = response.get("id", completion_id)
            for dst, src in (("input_tokens", "prompt_tokens"), ("output_tokens", "completion_tokens")):
                value = token_data.get(src)
                if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                    usage[dst] = (usage[dst] or 0) + value
            choice = (response.get("choices") or [{}])[0]
            finish_reason = choice.get("finish_reason")
            at_cap = isinstance(token_data.get("completion_tokens"), int) and token_data["completion_tokens"] >= token_limit
            try:
                if finish_reason == "length":
                    raise ValueError("Generation reached the output token limit")
                message = choice["message"]
                if message.get("tool_calls"):
                    prepared = []
                    for call in message["tool_calls"]:
                        name = call['function']['name']
                        if name not in payload['tool_schema_ids'] or name not in {'write_code', 'analyze_data'}:
                            raise ValueError("Unexpected model tool call")
                        args = parse_json(call["function"]["arguments"])
                        schema = (self.bus.schemas.schema['$defs']['C11_Request']['properties']['payload'] if name == 'write_code' else ANALYSIS_SCHEMA)
                        self.bus.schemas._check(args, schema, "$.tool")
                        if name == 'write_code' and args.get("researcher_id") != context["metadata"].get("researcher_id"):
                            raise ValueError("Tool call uses another researcher's namespace")
                        prepared.append((call, args))
                    # Provider reasoning is telemetry, not conversation state. Replaying it
                    # both exhausts the byte cap and relies on nonstandard request fields.
                    messages.append({"role": "assistant", "content": message.get("content"),
                                     "tool_calls": message["tool_calls"]})
                    for call, args in prepared:
                        name = call['function']['name']
                        if name == 'write_code':
                            result = self.bus.call("C11", args, lambda p: write_code(p, self.artifacts), key="code/"+self._digest(args))
                            calls.append({"call_id": call["id"], "name": name, "request_id": result["request_id"]})
                        else:
                            try:
                                result = self.analysis.analyze(args)
                            except AgentError as error:
                                result = {'error': error.as_dict()}
                            calls.append({'call_id': call['id'], 'name': name, 'artifact_id': result.get('artifact_id')})
                        messages.append({"role": "tool", "tool_call_id": call["id"], "content": json.dumps(result, ensure_ascii=False)})
                    last_error = AgentError("INVALID_LLM_OUTPUT", "No final JSON after the tool call")
                    continue
                parsed = parse_json(message.get("content"))
                self.bus.schemas._check(parsed, expected, "$")
                art = self.artifacts.write_json("llm/parsed-"+uuid.uuid4().hex[:12]+".json", parsed, "llm_output")
                self.log.event("llm.output_validated", call_no=call_no, schema_id=payload["response_schema_id"],
                               attempts_used=attempt, output_artifact_id=art["artifact_id"])
                return {"completion_id": completion_id, "output_artifact_id": art["artifact_id"],
                        "parsed_payload": parsed, "tool_calls": calls, "usage": usage, "attempts_used": attempt}
            except (ValueError, KeyError, TypeError, IndexError) as exc:
                truncated = finish_reason == "length" or at_cap
                code = "LLM_OUTPUT_TRUNCATED" if truncated else "INVALID_LLM_OUTPUT"
                last_error = AgentError(code, f"Model output did not match {payload['response_schema_id']}: {exc}")
                self.log.event("llm.output_truncated" if truncated else "llm.output_rejected", level="WARNING",
                               call_no=call_no, attempt=attempt, code=code, message=str(last_error), finish_reason=finish_reason,
                               max_output_tokens=token_limit)
                if truncated:
                    new_limit = min(token_limit*2, self.config.llm_max_output_tokens_limit)
                    if new_limit <= token_limit or attempt == payload["attempt_limit"]:
                        raise last_error
                    self.log.event("llm.output_budget_increased", level="WARNING", call_no=call_no,
                                   previous_limit=token_limit, next_limit=new_limit, reason="truncated_generation")
                    token_limit = new_limit
                    messages.append({"role": "user", "content": "Generation was truncated. Return a compact complete final JSON matching the schema; avoid unnecessary code or prose."})
                else:
                    content = (choice.get("message") or {}).get("content")
                    # A malformed full pipeline can be larger than the next request's
                    # entire budget. Its raw response is already logged; retain only a
                    # small invalid reply, with the bounded repair error below.
                    if isinstance(content, str) and content.strip() and len(content.encode('utf-8')) <= 2000:
                        messages.append({"role": "assistant", "content": content})
                    messages.append({"role": "user", "content": "Return one valid JSON object matching the schema. Error: "+str(exc)[:600]})
        raise last_error or AgentError("INVALID_LLM_OUTPUT", "No final JSON within the bounded tool/repair loop")

    @staticmethod
    def _compact_research_request(request, limit):
        """Account for tool results and schemas without deleting the protected core or base."""
        def size():
            return len(json.dumps(request, ensure_ascii=False, allow_nan=False).encode('utf-8'))
        for message in request['messages']:
            if message.get('role') != 'user':
                continue
            try:
                user = json.loads(message['content'])
            except (ValueError, TypeError):
                continue
            if not isinstance(user, dict) or not ('profile_summary' in user or 'profile' in user):
                continue
            profile = user.get('profile_summary', {})
            def refresh():
                message['content'] = json.dumps(user, ensure_ascii=False)
            data = profile.get('data', user.get('profile', {}))
            for field in ['data_dictionary', 'train_dtypes', 'test_dtypes', 'train_missing_fraction', 'test_missing_fraction', 'train_columns', 'test_columns']:
                if size() <= limit:
                    break
                data.pop(field, None);refresh()
            while size() > limit and user.get('facts'):
                user['facts'].pop();refresh()
            memory = profile.get('measured_search', {})
            while size() > limit and len(memory.get('entries', [])) > 4:
                memory['entries'].pop();memory['omitted_count'] = memory.get('omitted_count', 0)+1;refresh()
        # Older reports remain retrievable by artifact_id. Preserve the latest tool
        # result and every assistant/tool-call pair so the API dialogue stays valid.
        tool_messages = [m for m in request['messages'] if m.get('role') == 'tool']
        for message in tool_messages[:-1]:
            if size() <= limit:
                break
            try:
                reply = json.loads(message['content'])
            except (ValueError, TypeError):
                continue
            if not isinstance(reply, dict) or not reply.get('artifact_id') or 'report' not in reply:
                continue
            report = reply['report']
            message['content'] = json.dumps({'artifact_id': reply['artifact_id'], 'report': {
                'scope': report.get('scope'), 'query': report.get('query'), 'details_omitted': True,
                'detail_hint': 'Earlier report compacted; retrieve by kind=report and report_id or query fewer columns.'}},
                ensure_ascii=False)

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
            host += ":"+str(parsed.port)
        return parsed.scheme+"://"+host+parsed.path

    def close(self):
        self.transport.close()
