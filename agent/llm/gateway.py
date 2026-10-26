from copy import deepcopy
import json
import os
import threading
import time
import uuid

from agent.errors import AgentError
from agent.llm.protocol import IncompleteOutput, incomplete_json, parse_final_json, parse_json, prepare_tool_calls
from tools.code import write_code

DEFAULT_MODEL = "qwen/qwen3.8-27b"  # Permitted by the updated organizers' instructions.


class UnavailableTransport:
    def __init__(self, error):
        self.error = error

    def complete(self, request, timeout, context):
        raise self.error

    def close(self):
        pass


def request_model(mode):
    return os.environ.get("OPENAI_MODEL", "").strip() or (DEFAULT_MODEL if mode == "live" else "mock-only")


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
        import httpx
        self.wire_context = threading.local()
        self.log = None
        http_client = httpx.Client(event_hooks={'request': [self._check_wire_request]})
        self.client = OpenAI(http_client=http_client, api_key=os.environ["OPENAI_API_KEY"], base_url=os.environ["OPENAI_BASE_URL"], max_retries=0)

    def _check_wire_request(self, outgoing):
        limit = getattr(self.wire_context, 'limit', None)
        size = len(outgoing.content)
        if limit is not None and size > limit:
            raise AgentError('CONTEXT_TOO_LARGE', 'Serialized HTTP JSON exceeds the effective budget',
                             details={'wire_request_bytes': size, 'limit_bytes': limit})
        if self.log is not None:
            self.log.event('llm.wire_request_budget_checked', wire_request_bytes=size,
                           limit_bytes=limit, tokenizer_estimate=False)

    def complete(self, request, timeout, context):
        self.wire_context.limit = context.get('_request_byte_limit')
        try:
            return self.client.chat.completions.create(**request, timeout=timeout).model_dump(exclude_none=True)
        except Exception as error:
            cause = error
            for _ in range(5):
                if isinstance(cause, AgentError):
                    raise cause
                cause = cause.__cause__
                if cause is None:
                    break
            raise
        finally:
            self.wire_context.limit = None

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
            from agent.semantic_planning import unknown_model
            payload = {**demo['draft'], 'task_model': unknown_model()}
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
            elif recipe.startswith(("strong-", "lightgbm-")):
                params = {"learning_rate": min(0.2, 0.05+0.01*round_no)}
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
                user = json.loads(context['messages'][1]['content'])
                if user.get('profile_summary', user).get('data_research', {}).get('active'):
                    payload['hyperparameters'] = {}
                from agent.feature_search import feature_focus
                column = None
                if feature_focus(user):
                    data = user.get('profile_summary', {}).get('data', user.get('profile', {}))
                    column = next(c for c in data['train_columns'] if c in data['test_columns'] and c != 'id')
                    payload.update(action='features', feature_program=[
                        {'op': 'missing', 'name': '__mock_missing_'+str(round_no), 'columns': [column]}])
                observation = (user.get('profile_summary', user).get('error_research') or {}).get('observation') or {}
                robust = [r for r in observation.get('remedies', []) if r.get('robust') and '#' in r.get('remedy_ref', '')]
                if robust and not recipe.startswith('dummy-'):
                    # Synthetic demo: execute the top tested slice remedy (bound to its measured base).
                    payload.update(action='features', slice_remedies=[robust[0]['remedy_ref']])
                    payload.pop('feature_program', None)
                    column = None
                from agent.scientific_research import synthetic_notes
                payload['research_notes'] = synthetic_notes(payload, user)
                payload['investigation'] = {
                    'question': 'Does this complete synthetic configuration improve the measured base?',
                    'basis_ids': [], 'columns': [column] if column else [],
                    'interpretation': 'exploratory', 'mechanism': 'Synthetic demo reference, not a physical interpretation.',
                    'assumptions': ['Synthetic demo only.'], 'expected_observation': 'A valid independent configuration score.',
                    'alternative_explanation': 'Differences may reflect the joint estimator and representation.',
                    'decision_rule': 'Retain the better valid configuration without asserting a component mechanism.'}
        if context['metadata'].get('pipeline_execution'):
            payload.pop('research_notes', None)
            payload.pop('investigation', None)
        return {"id": "mock-" + uuid.uuid4().hex[:12],
                "choices": [{"message": {"role": "assistant", "content": json.dumps(payload, ensure_ascii=False)}, "finish_reason": "stop"}]}

    def close(self):
        pass


class Gateway:
    def __init__(self, config, budget, artifacts, bus, log, output_schemas, transport=None):
        self.config, self.budget, self.artifacts, self.bus, self.log = config, budget, artifacts, bus, log
        self.output_schemas = output_schemas
        try:
            self.transport = transport or (MockTransport() if config.llm_mode == "mock" else OpenAITransport())
        except AgentError as error:
            # Keep deterministic bootstrapping/training available without API
            # configuration. Calls still fail explicitly; no alternate provider.
            self.transport = UnavailableTransport(error)
        self.semaphore = threading.BoundedSemaphore(config.max_workers)
        self.daily_budget_exhausted = threading.Event()
        self.provider_credits_exhausted = threading.Event()
        self.analysis = None
        self.knowledge = None
        self.feedback = None
        self.cancelled = threading.Event()
        self.availability_lock = threading.RLock()
        self.cooldown_until = 0.0
        self.daily_retry_at = None
        self.daily_probe_active = False
        self.daily_probe_failures = 0
        self.daily_reset_known = False

    def _schedule_daily_probe(self, reset_seconds=None, *, transient=False):
        """Schedule a useful request, without sleeping or assuming provider funds."""
        with self.availability_lock:
            self.daily_probe_failures += 1
            if reset_seconds is not None:
                delay = max(1., reset_seconds)
            else:
                initial, ceiling = (5, 60) if transient else (60, 900)
                delay = min(ceiling, initial*2**min(self.daily_probe_failures-1, 8))
            now = time.monotonic()
            latest = self.budget.deadline-self.budget.reserve_seconds()-1
            self.daily_retry_at = now+delay if now+delay <= latest else None
            self.log.event('llm.budget_probe_scheduled', reset_known=self.daily_reset_known,
                retry_after_seconds=delay if self.daily_retry_at is not None else None,
                reason='temporary_probe_failure' if transient else 'explicit_budget_exhaustion',
                finalization_reserve_preserved=True)

    def check_daily_budget(self):
        if self.daily_budget_exhausted.is_set():
            with self.availability_lock:
                due = (self.daily_retry_at is not None and time.monotonic() >= self.daily_retry_at
                       and not self.daily_probe_active and self.budget.remaining() > self.budget.reserve_seconds()+1)
            if due:return  # One availability probe, not a calendar-based balance reset.
            raise AgentError("LLM_BUDGET_EXHAUSTED", "The provider reported budget exhaustion; resume only after a successful availability probe",
                             details={"reset_known": self.daily_reset_known})

    def check_availability(self):
        if self.provider_credits_exhausted.is_set():
            raise AgentError("LLM_PROVIDER_CREDITS_EXHAUSTED",
                             "The LLM provider reported Insufficient credits. Contact the gateway operator to restore provider access; "
                             "the team's daily balance and reset time are unknown.",
                             details={"scope": "upstream_provider", "team_daily_budget_status": "unknown", "reset_time_msk": None})
        self.check_daily_budget()
        with self.availability_lock:
            left = self.cooldown_until-time.monotonic()
        if left > 0:
            raise AgentError('LLM_COOLDOWN', 'Provider transport is unavailable; retain results and use fallback until retry is allowed',
                             details={'retry_after_seconds': round(left, 1)})

    def _progress(self, done, call_no, purpose, started, timeout):
        while not done.wait(self.config.llm_progress_seconds):
            self.log.event("llm.waiting", call_no=call_no, purpose=purpose,
                           elapsed_seconds=round(time.monotonic() - started, 1), request_timeout_seconds=round(timeout, 1),
                           run_seconds_remaining=round(self.budget.remaining(), 1))

    def _request(self, request, context, payload, attempt, deadline, dialogue_id=None):
        from contextvars import copy_context
        if self.cancelled.is_set():
            raise AgentError('RESEARCH_CANCELLED', 'Research was cancelled before request dispatch')
        delivered = None
        feedback_message = None
        if payload['purpose'] == 'research_proposal' and self.feedback is not None:
            scope = context['metadata'].get('graph_binding', {}).get('branch_id', 'legacy')
            delivered = self.feedback.latest(scope)
            if delivered:
                frozen_scope = context['metadata'].get('detail_scope', {})
                if (frozen_scope and (delivered.get('dataset_id') != frozen_scope.get('dataset_id')
                        or delivered.get('task_spec_id') != frozen_scope.get('task_spec_id')
                        or delivered.get('validation_id') != frozen_scope.get('validation_id'))):
                    raise AgentError('MEMORY_SCOPE_DENIED', 'Feedback identity differs from issued task/validation')
                context['metadata'].setdefault('authorized_feedback_ids', []).append(delivered['artifact_id'])
                from agent.research_feedback import compact_updates, frame_updates
                delivered = frame_updates(compact_updates(delivered), context)
                context['metadata'].setdefault('delivered_experiment_ids', []).extend(
                    r['experiment_id'] for r in [*delivered['results'], *delivered.get('candidate_results', [])])
            if delivered:
                feedback_message = {'role': 'user', 'content': json.dumps(
                    {'research_updates': delivered}, ensure_ascii=False)}
                request['messages'].append(feedback_message)
        if self.config.llm_reasoning_protocol == 'qwen_chat_template':
            # Explicit opt-in only. This is a documented self-hosted Qwen format,
            # not a claim that the organizers' proxy applies these fields.
            request['extra_body'] = {'chat_template_kwargs': {'enable_thinking': True, 'preserve_thinking': True}}
            if self.config.llm_reasoning_effort:
                request['extra_body']['reasoning_effort'] = self.config.llm_reasoning_effort
        self._bound_reasoning_history(request['messages'])
        from agent.context_budget import effective_limit, request_bytes as measure_request
        byte_limit, budget_limits = effective_limit(self.config, context['metadata'], request['max_tokens'])
        request_bytes = measure_request(request)
        # The initial schema/tools/feedback nearly filled the graph limit in the
        # observed classification run. Keep room for answers before requesting them.
        packing_limit = byte_limit-min(16384, byte_limit//6) if request.get('tools') else byte_limit
        if request_bytes > packing_limit and context['metadata']['schema_id'] == 'ResearchAction-v1':
            packed = self._compact_research_request(request, packing_limit,
                archive=(lambda history: self._archive_transcript(history, context))
                        if context['metadata'].get('detail_scope') else None)
            if packed['archived_tool_call_ids']:
                # Keep the bounded transcript on later turns as well.
                context['_packed_messages'] = deepcopy([m for m in request['messages'] if m is not feedback_message])
            request_bytes = measure_request(request)
        if request_bytes > byte_limit:
            raise AgentError('CONTEXT_TOO_LARGE', 'Complete JSON request cannot fit without removing protected contract/base/latest results',
                details={'request_bytes': request_bytes, 'limit_bytes': byte_limit, 'limits': budget_limits,
                         'section_bytes': self._section_bytes(request)})
        context['_request_byte_limit'] = byte_limit
        if isinstance(self.transport, OpenAITransport):
            self.transport.log = self.log
        self.log.event('llm.request_budget_checked', request_bytes=request_bytes, limit_bytes=byte_limit,
            limits=budget_limits, section_bytes=self._section_bytes(request), packing_target_bytes=packing_limit,
            tokenizer_estimate=False, provider_safety_proxy='one_input_byte_per_estimated_token')
        timeout = min(self.config.llm_timeout_seconds, deadline - time.monotonic())
        if timeout < 1:
            raise AgentError("LLM_OPERATION_TIMEOUT", "Insufficient operation time for another API attempt")
        daily_probe = False
        with self.availability_lock:
            self.check_availability()
            if self.daily_budget_exhausted.is_set():
                self.daily_probe_active = daily_probe = True
        try:
            call_no = self.budget.begin_llm_call(request["max_tokens"], dialogue_id)
        except BaseException:
            if daily_probe:
                with self.availability_lock:self.daily_probe_active = False
            raise
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
            if delivered:
                evidence_ids = [delivered['artifact_id']]
                evidence_ids.extend(r['artifact_id'] for r in delivered.get('fit_diagnostics', []))
                for result in [*delivered['results'], *delivered.get('candidate_results', [])]:
                    evidence_ids.extend(result[k] for k in ('experiment_id', 'evaluation_id', 'diagnostics_artifact_id'))
                evidence_ids.extend(s['artifact_id'] for s in delivered['series'])
                evidence_ids.extend(r['artifact_id'] for r in delivered.get('fit_diagnostics', []))
                self.log.event('research.feedback_delivered', call_no=call_no,
                    context_id=payload['context_artifact_id'], branch_id=delivered['branch_id'],
                    revision=delivered['revision'], feedback_artifact_id=delivered['artifact_id'],
                    experiment_ids=[v['experiment_id'] for v in delivered['results']],
                    active_series_ids=[s['series_id'] for s in delivered.get('activity', {}).get('active_searches', [])],
                    series_ids=[s['series_id'] for s in delivered['series']],
                    changed_since_previous_turn=delivered.get('changed_since_previous_turn'),
                    changes=[c['kind'] for c in delivered.get('changes', [])],
                    research_base_experiment_id=(delivered.get('research_frame') or {}).get('research_base_experiment_id'),
                    utility_reference_experiment_id=((delivered.get('research_frame') or {}).get('utility_reference') or {}).get('experiment_id'))
            progress.start()
            response = self.transport.complete(request, max(0.1, min(timeout, deadline-time.monotonic())), context)
            with self.availability_lock:
                budget_recovered = self.daily_budget_exhausted.is_set()
                self.daily_budget_exhausted.clear()
                self.daily_retry_at = None
                self.daily_probe_failures = 0
                self.daily_reset_known = False
            if budget_recovered:
                self.log.event('llm.budget_availability_confirmed', call_no=call_no)
            with self.availability_lock:
                recovered = self.cooldown_until > time.monotonic()
                self.cooldown_until = 0.0
            if recovered:
                self.log.event('llm.transport_recovered', call_no=call_no, reason='another_inflight_request_succeeded')
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
                           reasoning_field=self.config.llm_reasoning_field,
                           reasoning_field_present=isinstance(message.get(self.config.llm_reasoning_field), str),
                           cached_input_tokens=(token_data.get("prompt_tokens_details") or {}).get("cached_tokens"),
                           usage_inconsistent=isinstance(reasoning, int) and isinstance(output, int) and reasoning > output,
                           token_budget=self.budget.token_snapshot())
            if self.cancelled.is_set() or time.monotonic() >= deadline:
                self.log.event('llm.late_result_discarded', call_no=call_no, reason='cancelled' if self.cancelled.is_set() else 'operation_deadline')
                raise AgentError('RESEARCH_CANCELLED' if self.cancelled.is_set() else 'LLM_OPERATION_TIMEOUT',
                                 'Late provider result was charged and archived but cannot become an experiment')
            if delivered:
                context['metadata'].setdefault('delivered_evidence_ids', []).extend(evidence_ids)
                response['_delivered_feedback'] = {'name': 'research_updates', 'call_no': call_no,
                    'artifact_id': delivered['artifact_id'], 'revision': delivered['revision'],
                    'measurement_generation': delivered.get('measurement_generation'),
                    'active_series_ids': [s['series_id'] for s in delivered.get('activity', {}).get('active_searches', [])],
                    'evidence_ids': list(dict.fromkeys(evidence_ids))}
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
                # Only explicit provider metadata supplies the expected reset.
                # Missing/invalid metadata means unknown, never guessed funds.
                from datetime import datetime, timezone
                metadata = getattr(exc, 'body', None) or {}
                if isinstance(metadata, dict):
                    metadata = metadata.get('error', metadata)
                reset = metadata.get('budget_reset_at', metadata.get('reset_at')) if isinstance(metadata, dict) else None
                seconds = None
                try:
                    seconds = max(1., (datetime.fromisoformat(reset.replace('Z','+00:00'))-datetime.now(timezone.utc)).total_seconds())
                except (TypeError, ValueError, AttributeError):pass
                with self.availability_lock:
                    self.daily_budget_exhausted.set()
                    self.daily_reset_known = seconds is not None
                    self._schedule_daily_probe(seconds)
                self.log.event("llm.daily_budget_exhausted", level="WARNING", call_no=call_no,
                               status_code=status, reset_known=self.daily_reset_known)
                self.check_daily_budget()
            retryable = status == 429 or status is None or (isinstance(status, int) and status >= 500)
            error = AgentError("LLM_RATE_LIMIT" if status == 429 else "LLM_UNAVAILABLE", str(exc), retryable=retryable)
            self.log.event("llm.call_failed", level="WARNING", call_no=call_no, code=error.code, message=str(exc),
                           attempt=attempt, status_code=status, retryable=retryable)
            raise error from exc
        finally:
            if daily_probe:
                with self.availability_lock:
                    self.daily_probe_active = False
                    if (self.daily_budget_exhausted.is_set() and self.daily_retry_at is not None
                            and self.daily_retry_at <= time.monotonic()):
                        self._schedule_daily_probe(transient=True)
            done.set()
            if progress.ident is not None:
                progress.join(timeout=0.2)
            if not settled:
                # A timed-out request may have been billed without returning usage.
                # Charge its reservation conservatively; never fabricate reported usage.
                self.budget.finish_llm_call(call_no)

    def _bound_reasoning_history(self, messages):
        field = self.config.llm_reasoning_field
        retained = sum(len(m.get(field, '').encode('utf-8')) for m in messages)
        dropped = 0
        cap = self.config.llm_reasoning_history_bytes if self.config.llm_reasoning_protocol == 'qwen_chat_template' else 0
        for message in messages:
            if retained <= cap:
                break
            value = message.pop(field, '')
            retained -= len(value.encode('utf-8'))
            dropped += len(value.encode('utf-8'))
        if dropped:
            self.log.event('llm.reasoning_history_compacted', retained_bytes=retained, omitted_bytes=dropped,
                           protocol=self.config.llm_reasoning_protocol)

    def complete(self, payload):
        dialogue_started = time.monotonic()
        context = self.artifacts.read_json(payload["context_artifact_id"])
        context['_run_started_monotonic'] = self.budget.started
        messages = deepcopy(context["messages"])
        expected = deepcopy(self.output_schemas[payload["response_schema_id"]])
        executable = bool(context['metadata'].get('pipeline_execution'))
        if executable:
            fields = {'action', 'hypothesis', 'code', 'code_ref', 'recipe_id', 'hyperparameters', 'feature_steps',
                      'feature_program', 'slice_remedies', 'evidence_ids', 'expected_effect', 'next_step', 'tuning'}
            expected['properties'] = {k:v for k,v in expected['properties'].items() if k in fields}
            expected['properties']['action']['enum'] = ['pipeline', 'features', 'catalog', 'research', 'stop']
            expected['required'] = ['action']
            expected['properties']['tuning'] = {'type': 'null',
                'description': 'Background search is controller-owned. Declare SEARCH_SPACE in code; omit tuning.'}
        user = json.loads(context['messages'][1]['content']) if payload['response_schema_id'] == 'ResearchAction-v1' else {}
        if payload['response_schema_id'] == 'ResearchAction-v1':
            from agent.data_research import attach_context
            window = self.budget.data_research_snapshot()
            attach_context(user, window)
            context['messages'][1]['content'] = json.dumps(user, ensure_ascii=False)
            messages = deepcopy(context['messages'])
            if window['active']:
                expected['properties']['action']['enum'] = ['features', 'pipeline', 'research']
                if 'next_step' in expected['properties']:
                    expected['properties']['next_step']['enum'] = ['continue','wait']
        scientific = user.get('profile_summary', user).get('scientific_protocol', {}) if isinstance(user, dict) else {}
        if payload['response_schema_id'] == 'ResearchAction-v1' and not executable and scientific.get('version') == 'scientific-v1':
            expected['required'] = list(dict.fromkeys([*expected['required'], 'research_notes']))
            notes = expected['properties']['research_notes']
            notes['required'] = list(dict.fromkeys([*notes['required'], 'test_plan']))
        if payload['response_schema_id'] == 'TaskDraft-v1':
            for variant in expected['oneOf']:
                if 'target_column' in variant.get('properties', {}):
                    variant['required'] = [*variant['required'], 'task_model']
        elif payload['response_schema_id'] == 'ResearchAction-v1' and context['metadata'].get('semantic_contract'):
            expected['required'] = [*expected['required'], 'investigation']
        def expanded(value):
            if isinstance(value, dict):
                if "$ref" in value:
                    return expanded(self.bus.schemas.schema["$defs"][value["$ref"].split("/")[-1]])
                return {k: expanded(v) for k, v in value.items()}
            return [expanded(v) for v in value] if isinstance(value, list) else value
        messages.insert(1, {"role": "system", "content": "Final output JSON schema:\n" + json.dumps(expanded(expected), ensure_ascii=False)})
        reserve = self.budget.reserve_seconds() + 1 if payload["purpose"] == "research_proposal" else .5
        deadline = min(time.monotonic() + payload["deadline_seconds"], self.budget.deadline - reserve)
        extended = bool(scientific.get('deliberation', {}).get('extended_requested'))
        deep_tokens = min(payload['max_output_tokens']*2, self.config.llm_max_output_tokens_limit)
        available_output = self.budget.token_snapshot()['output_available']
        funded_deep = (payload['response_schema_id'] == 'ResearchAction-v1' and extended
                       and (available_output is None or available_output >= 2*deep_tokens)
                       and deadline-dialogue_started >= self.config.llm_research_dialogue_seconds)
        if payload['response_schema_id'] == 'ResearchAction-v1':
            seconds = max(self.config.llm_research_dialogue_seconds, self.config.research_deep_dialogue_seconds) if funded_deep else self.config.llm_research_dialogue_seconds
            deadline = min(deadline, dialogue_started+seconds)
            self.log.event('llm.deliberation_budget', context_id=payload['context_artifact_id'],
                extended_requested=extended, extended_funded=funded_deep,
                dialogue_seconds=seconds, output_tokens=deep_tokens if funded_deep else payload['max_output_tokens'],
                provider_thinking_application='unverified')
        phase_limited = (payload['purpose'] == 'research_proposal' and deadline >= self.budget.deadline-reserve-.01)
        if payload['purpose'] == 'research_proposal' and self.budget.phase_deadline:
            # This is a response deadline only. A received proposal can wait in
            # the dispatcher without inheriting the API timeout.
            user = json.loads(context['messages'][1]['content']) if context['metadata'].get('graph_binding') else {}
            base = user.get('base_result')
            if base is None and user.get('profile_summary', {}).get('base_result_details'):
                detail_id = context['metadata'].get('detail_artifact_id')
                if detail_id:
                    base = self.artifacts.read_json(detail_id)['sections'].get('base_result')
            recipe = next((r for r in user.get('profile_summary', {}).get('recipes', [])
                           if base and r['id'] == base['recipe_id']), None)
            after_response = self.budget.reserve_seconds()
            if recipe:
                probe = {'recipe_id': recipe['id'], 'model': {'family': recipe['family'], 'hyperparameters': base['hyperparameters']},
                         # Generated pipelines have separate measured histories per source.
                         # Older saved contexts lack the optional identity: use a cold
                         # estimate, never a different pipeline's timing samples.
                         'code_artifact_id': base.get('code_artifact_id') or 'unresolved-source',
                         'resources': {'cpu_threads': self.config.train_cpu_threads}}
                after_response = self.budget.duration.estimate(probe)+self.budget.candidate_reserve(probe)
            phase_deadline = self.budget.phase_deadline-after_response
            phase_limited = phase_limited or phase_deadline < deadline
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
                'description': 'Read-only fit summary/association/group, exact streamed check violation counts, stateless derived expressions/filters, errors for a visible experiment_id, unlabelled test comparison, dictionary or report. Computing queries attach investigation before observations. No holdout labels. columns contains inputs, never targets; include_target=true adds fit-only summaries. check requires checks, group_by=null, include_target=false. errors returns selection-only aggregates. At most 12 columns. limit/offset/limit_bytes paginate schema/report reads; omit them for group/summary queries.',
                'parameters': ANALYSIS_SCHEMA}})
        if 'run_analysis' in payload['tool_schema_ids'] and self.config.python_analysis_enabled:
            from tools.python_analysis import PYTHON_ANALYSIS_SCHEMA
            tools.append({'type': 'function', 'function': {'name': 'run_analysis',
                'description': 'Execute analyze(X_fit,y_fit,X_test,context) Python using approved libraries. Optional experiment_id=best or a visible ID aligns X/y with that version\'s fit-only OOF context["oof"]: predictions, errors, residuals/squared_errors for regression, fold_ids and row_ids. OOF_PENDING means continue other analysis while diagnostics wait. Otherwise a uniform fit sample <=20000 rows. y_fit only with include_target=true; X_test only with compare_test=true. No files/network/processes. Return compact results; tracebacks are retained for repair. No held-out labels.',
                'parameters': PYTHON_ANALYSIS_SCHEMA}})
        if 'read_memory' in payload['tool_schema_ids']:
            from agent.context_details import READ_SCHEMA
            tools.append({'type': 'function', 'function': {'name': 'read_memory',
                'description': 'Read frozen branch/task/validation-scoped details using detail_index.reference_id and section. Follow next_offset in UTF-8 bytes; list sections support section/index. Does not open arbitrary artifacts or files.',
                'parameters': READ_SCHEMA}})
        if 'read_solution' in payload['tool_schema_ids']:
            from agent.context_details import SOLUTION_SCHEMA
            tools.append({'type': 'function', 'function': {'name': 'read_solution',
                'description': 'Read an executable solution by experiment_id, or best for the strongest visible measured solution. '
                    'Default pipeline view returns code, effective parameters, features and metric together. '
                    'Use code/result/features/diagnostics for a smaller exact view. No reference_id or section needed. '
                    'Follow next_read for long outputs. Only this task/branch and delivered results are accessible.',
                'parameters': SOLUTION_SCHEMA}})
        if 'search_knowledge' in payload['tool_schema_ids']:
            from tools.knowledge import KNOWLEDGE_SCHEMA
            if self.knowledge is None:
                raise AgentError('KNOWLEDGE_UNAVAILABLE', 'No offline knowledge library is configured')
            tools.append({'type': 'function', 'function': {'name': 'search_knowledge',
                'description': 'Search offline Kaggle experience and complete project-authored pipeline examples. Returns content, source IDs and current-pipeline cautions, not just links. Use ids to read a known item. External experience is not current-task evidence; cite its IDs only in knowledge_ids. No network.',
                'parameters': KNOWLEDGE_SCHEMA}})
        token_limit = min(deep_tokens if funded_deep else payload["max_output_tokens"], self.config.llm_max_output_tokens_limit)
        if 'edit_solution' in payload['tool_schema_ids']:
            from agent.pipeline_drafts import EDIT_SCHEMA
            tools.append({'type':'function', 'function':{'name':'edit_solution',
                'description':'Save a new source draft with code, or edit a visible experiment_id/best or draft_id using exact old/new replacements. Each old fragment must match once. Returns draft_id; submit action=pipeline, code_ref=draft_id without resending full source. Invalid source is retained for repair.',
                'parameters':EDIT_SCHEMA}})
        dialogue_id, slots = None, 0
        if payload['response_schema_id'] == 'ResearchAction-v1':
            dialogue_id, slots = self.budget.reserve_dialogue(token_limit, repair=payload['attempt_limit'] > 1)
            self.log.event('llm.dialogue_reserved', dialogue_id=dialogue_id, slots=slots,
                           final_request_reserved=True, repair_request_reserved=slots > 1,
                           max_output_tokens=token_limit)
        try:
            answer = self._complete_dialogue(payload, context, messages, expected, tools, deadline, phase_limited,
                                           token_limit, dialogue_id, slots)
            if payload['response_schema_id'] == 'ResearchAction-v1':
                answer['tool_calls'].extend({'name': 'transcript_archive', 'reference_id': aid, 'status': 'archived'}
                    for aid in context.get('_transcript_ids', []))
                checkpoint = self.artifacts.meta(answer['output_artifact_id'])['kind'] == 'controller_research_checkpoint'
                self.log.event('research.dialogue_summary', context_id=payload['context_artifact_id'],
                    duration_seconds=round(time.monotonic()-dialogue_started, 3),
                    tool_results=sum(c.get('name') not in {'research_updates', 'transcript_archive'} for c in answer['tool_calls']),
                    decision_received=not checkpoint, checkpoint_saved=checkpoint,
                    decision_origin='controller' if checkpoint else 'model', tool_seconds_limit=self.config.llm_research_tool_seconds,
                    dialogue_seconds_limit=self.config.llm_research_dialogue_seconds)
            return answer
        except AgentError as error:
            if payload['response_schema_id'] == 'ResearchAction-v1':
                if context['metadata'].get('detail_scope'):
                    self._archive_transcript(messages, context)
                self.log.event('research.dialogue_summary', context_id=payload['context_artifact_id'],
                    duration_seconds=round(time.monotonic()-dialogue_started, 3),
                    decision_received=False, error_code=error.code,
                    tool_seconds_limit=self.config.llm_research_tool_seconds,
                    dialogue_seconds_limit=self.config.llm_research_dialogue_seconds)
            if error.code in {'LLM_UNAVAILABLE', 'LLM_RATE_LIMIT'}:
                until = min(self.budget.deadline, time.monotonic()+60) if error.retryable else self.budget.deadline
                with self.availability_lock:
                    self.cooldown_until = max(self.cooldown_until, until)
                self.log.event('llm.transport_cooldown', reason=error.code, retryable=error.retryable,
                               retry_after_seconds=max(0, until-time.monotonic()))
            if executable and error.code in {'CONTEXT_TOO_LARGE','LLM_OUTPUT_TRUNCATED','INVALID_LLM_OUTPUT',
                    'LLM_OPERATION_TIMEOUT','SEARCH_PHASE_BUDGET_EXHAUSTED'} and not self.cancelled.is_set():
                return self._research_checkpoint(context, messages, error, payload['attempt_limit'])
            raise
        finally:
            if dialogue_id is not None:
                unused = self.budget.release_dialogue(dialogue_id)
                self.log.event('llm.dialogue_released', dialogue_id=dialogue_id, unused_slots=unused['slots'] if unused else 0)

    def _complete_dialogue(self, payload, context, messages, expected, tools, deadline, phase_limited,
                           token_limit, dialogue_id, reserved_slots):
        executable = bool(context['metadata'].get('pipeline_execution'))
        tool_seconds = 0.0
        usage = {"input_tokens": None, "output_tokens": None}
        calls, last_error = [], None
        tool_cache, force_decision = {}, False
        engineer_packet, engineer_used = None, False
        consecutive_errors = 0
        schema_by_name = {tool['function']['name']: tool['function']['parameters'] for tool in tools}
        completion_id = "completion-" + uuid.uuid4().hex[:12]
        # Token/call reservations alone cannot protect wall time. Bound tool
        # requests (including provider latency and concurrency waits) separately.
        now = time.monotonic()
        final_seconds = (min(max(10.0, (deadline-now)*.25, 2*self.budget.reasoning_estimate()), max(0, deadline-now)*.45)
                         if dialogue_id is not None else 0)
        tool_deadline = deadline-final_seconds
        context['_tool_deadline'] = tool_deadline
        if dialogue_id is not None:
            self.log.event('llm.dialogue_time_reserved', dialogue_id=dialogue_id,
                           final_seconds_reserved=round(final_seconds, 3),
                           tool_window_seconds=round(max(0, tool_deadline-now), 3))
        for attempt in range(1, payload["attempt_limit"] + 1):
            active_engineer, engineer_packet = engineer_packet, None
            if self.cancelled.is_set():
                raise AgentError('RESEARCH_CANCELLED', 'Research closed before another dialogue turn')
            self.check_availability()
            left = deadline-time.monotonic()
            if left < 1:
                raise AgentError('SEARCH_PHASE_BUDGET_EXHAUSTED' if phase_limited else 'LLM_OPERATION_TIMEOUT',
                                 "LLM operation deadline reached before a new attempt")
            if dialogue_id is not None and not force_decision and time.monotonic() >= tool_deadline:
                force_decision = True
                self.log.event('llm.tool_phase_closed', context_id=payload['context_artifact_id'],
                    reason='final_time_reserve', attempt=attempt, final_capacity_retained=True)
            waiting = left if force_decision or dialogue_id is None else max(.001, tool_deadline-time.monotonic())
            if not self.semaphore.acquire(timeout=waiting):
                if dialogue_id is not None and time.monotonic() >= tool_deadline and time.monotonic() < deadline:
                    force_decision = True
                    continue
                raise AgentError("LLM_OPERATION_TIMEOUT", "Timed out waiting for LLM concurrency slot")
            terminal = True
            retry_delay = None
            try:
                if tools and dialogue_id is not None and not force_decision and tool_seconds >= self.config.llm_research_tool_seconds:
                    force_decision = True
                    self.log.event('llm.tool_phase_closed', context_id=payload['context_artifact_id'],
                        reason='tool_wall_budget', tool_seconds_limit=self.config.llm_research_tool_seconds,
                        attempt=attempt, final_capacity_retained=True)
                request = {"model": request_model(self.config.llm_mode), "messages": deepcopy(messages), "max_tokens": token_limit}
                terminal = bool(tools and (force_decision or attempt == payload['attempt_limit']))
                if dialogue_id is not None:
                    # Keep the existing attempt cap, reserving its last turns
                    # for final JSON and (when funded) one repair. Time is only
                    # an estimate; server latency can still consume the deadline.
                    allow_tools = bool(reserved_slots > 1 and tools and not force_decision and attempt <= payload['attempt_limit']-reserved_slots
                                       and tool_deadline-time.monotonic() > 1
                                       and deadline-time.monotonic() > self.budget.reasoning_estimate()*reserved_slots)
                    try:
                        terminal = self.budget.prepare_dialogue_turn(dialogue_id, token_limit, allow_tools=allow_tools)
                    except AgentError as capacity_error:
                        self.log.event('llm.dialogue_capacity_exhausted', dialogue_id=dialogue_id,
                                       code=capacity_error.code, message=str(capacity_error), attempt=attempt)
                        if last_error is not None:
                            raise last_error
                        raise
                if terminal:
                    instruction = ('The analysis/tool phase is complete. Return one final JSON object matching the output schema '
                        'using the evidence already available. No tools are available on this final turn.')
                    if context['metadata']['schema_id'] == 'ResearchAction-v1':
                        current_user = json.loads(context['messages'][1]['content'])
                        facts = current_user.get('profile_summary', current_user).get('current_pipeline_facts')
                        if facts:
                            compact_facts = {k: v for k, v in facts.items() if k != 'effective_parameters'}
                            instruction += '\nAuthoritative current-base facts (external experience cannot override them): '+json.dumps(compact_facts, ensure_ascii=False)
                    request['messages'].append({'role': 'user', 'content': instruction})
                    self.log.event('llm.final_decision_requested', attempt=attempt, purpose=payload['purpose'], dialogue_id=dialogue_id)
                elif tools:
                    request.update(tools=tools, parallel_tool_calls=dialogue_id is not None)
                request_deadline = tool_deadline if dialogue_id is not None and not terminal else deadline
                request_context, request_payload = context, payload
                if active_engineer is not None:
                    from agent.llm.repair_engineer import messages as repair_messages
                    # Consume an existing repair attempt/reservation. No recursive
                    # Gateway call, extra tools, mailbox reads or provider slot.
                    request = {"model": request_model(self.config.llm_mode),
                               "messages": repair_messages(active_engineer), "max_tokens": token_limit}
                    request_context = {'metadata': {**deepcopy(context['metadata']), 'schema_id': 'RepairEngineer-v1'}}
                    request_payload = {**payload, 'purpose': 'research_engineering'}
                    request_deadline = min(deadline, time.monotonic()+self.config.llm_repair_engineer_timeout_seconds,
                                           time.monotonic()+max(1., (deadline-time.monotonic())*.35))
                    self.log.event('llm.repair_engineer_started', attempt=attempt, dialogue_id=dialogue_id,
                                   base_experiment_id=active_engineer['base_experiment_id'],
                                   allowed_fields=list(active_engineer['allowed_edits']))
                response, call_no = self._request(request, request_context, request_payload, attempt, request_deadline, dialogue_id)
            except AgentError as exc:
                if (active_engineer is not None and exc.code in {'LLM_UNAVAILABLE', 'LLM_RATE_LIMIT',
                        'LLM_OPERATION_TIMEOUT', 'CONTEXT_TOO_LARGE'} and attempt < payload['attempt_limit']
                        and deadline-time.monotonic() > 1):
                    last_error = exc
                    self.log.event('llm.repair_engineer_fallback', attempt=attempt, code=exc.code,
                                   reason='Engineer unavailable; continue original research repair')
                    continue
                if (dialogue_id is not None and not terminal and time.monotonic() >= tool_deadline
                        and time.monotonic() < deadline and exc.code in {'LLM_OPERATION_TIMEOUT', 'LLM_UNAVAILABLE'}):
                    force_decision = True
                    self.log.event('llm.tool_phase_closed', context_id=payload['context_artifact_id'],
                        reason='final_time_reserve', attempt=attempt, final_capacity_retained=True)
                    continue
                if exc.code == 'CONTEXT_TOO_LARGE' and calls and payload['response_schema_id'] == 'ResearchAction-v1':
                    if not terminal and not force_decision:
                        force_decision = True
                        self.log.event('llm.tool_phase_closed', context_id=payload['context_artifact_id'],
                            reason='context_pressure', attempt=attempt, final_capacity_retained=True)
                        continue
                    # Preserve completed observations and roll over without inventing an
                    # LLM decision, training score or private reasoning transcript.
                    from agent.research_actions import normalize_action
                    evidence = list(dict.fromkeys(c['artifact_id'] for c in calls if c.get('artifact_id') and c.get('name') != 'search_knowledge'))[-12:]
                    question = 'Resume the unresolved research question using saved reports.'
                    for item in reversed(messages):
                        for call in reversed(item.get('tool_calls', [])):
                            try:
                                query = json.loads(call['function']['arguments'])
                                question = query.get('question') or query.get('investigation', {}).get('question') or question
                            except (ValueError, KeyError, TypeError):
                                pass
                            if question != 'Resume the unresolved research question using saved reports.':break
                        if question != 'Resume the unresolved research question using saved reports.':break
                    self._archive_transcript(messages, context)
                    checkpoint = normalize_action({'action': 'research', 'next_step': 'continue',
                        'hypothesis': 'Controller context checkpoint: '+question,
                        'evidence_ids': evidence, 'investigation': {
                            'question': question, 'basis_ids': evidence, 'columns': [],
                            'interpretation': 'exploratory',
                            'mechanism': 'Infrastructure checkpoint only; no scientific conclusion was inferred.',
                            'assumptions': ['Completed reports remain available in scoped memory.'],
                            'expected_observation': 'Read the saved reports in a fresh bounded dialogue.',
                            'alternative_explanation': 'Context pressure does not refute the research hypothesis.',
                            'decision_rule': 'Read transcripts_index and saved reports, repair a failed check if present, '
                                'then decide the next executable experiment for this question.'}})
                    art = self.artifacts.write_json('research/controller-checkpoint-'+uuid.uuid4().hex[:12]+'.json',
                        checkpoint, 'controller_research_checkpoint')
                    self.log.event('research.context_checkpoint', artifact_id=art['artifact_id'],
                                   evidence_ids=evidence, reason=exc.code, decision_origin='controller')
                    return {'completion_id': completion_id, 'output_artifact_id': art['artifact_id'],
                            'parsed_payload': checkpoint, 'tool_calls': calls, 'usage': usage, 'attempts_used': attempt}
                if phase_limited and time.monotonic() >= deadline:
                    raise AgentError('SEARCH_PHASE_BUDGET_EXHAUSTED', 'Response window ended with time reserved for training and publication') from exc
                if not exc.retryable or attempt == payload["attempt_limit"]:
                    raise
                last_error = exc
                consecutive_errors += 1
                remaining = max(0, deadline-time.monotonic())
                retry_reserve = min(max(1, self.budget.reasoning_estimate()), remaining*.5)
                delay = min(2 ** min(consecutive_errors-1, 3), max(0, remaining-retry_reserve-1))
                self.log.event("llm.retry_scheduled", level="WARNING", code=exc.code, attempt=attempt,
                    consecutive_errors=consecutive_errors, delay_seconds=delay)
                retry_delay = delay
            finally:
                self.semaphore.release()
                if '_packed_messages' in context:
                    messages[:] = context.pop('_packed_messages')
            if retry_delay is not None:
                if self.cancelled.wait(retry_delay):
                    raise AgentError('RESEARCH_CANCELLED', 'Research cancelled during provider backoff')
                continue
            consecutive_errors = 0
            token_data = response.get("usage") or {}
            if response.get('_delivered_feedback'):
                calls.append(response.pop('_delivered_feedback'))
            completion_id = response.get("id", completion_id)
            for dst, src in (("input_tokens", "prompt_tokens"), ("output_tokens", "completion_tokens")):
                value = token_data.get(src)
                if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                    usage[dst] = (usage[dst] or 0) + value
            choice = (response.get("choices") or [{}])[0]
            finish_reason = choice.get("finish_reason")
            at_cap = isinstance(token_data.get("completion_tokens"), int) and token_data["completion_tokens"] >= token_limit
            try:
                message = choice["message"]
                if message.get("tool_calls"):
                    if terminal:
                        raise ValueError('The provider returned a tool call although no tools were offered on the final decision turn')
                    prepared = prepare_tool_calls(message['tool_calls'], schema_by_name, self.bus.schemas._check,
                        context['metadata'].get('researcher_id'), at_cap=at_cap or finish_reason == 'length')
                    assistant = {'role': 'assistant', 'content': message.get('content'),
                                 'tool_calls': [item['call'] for item in prepared]}
                    field = self.config.llm_reasoning_field
                    if self.config.llm_reasoning_protocol == 'qwen_chat_template' and isinstance(message.get(field), str):
                        assistant[field] = message[field]
                    messages.append(assistant)
                    self._bound_reasoning_history(messages)
                    tools_started = time.monotonic()
                    new_calls = 0
                    for item in prepared:
                        call, args, key = item['call'], item['args'], item['key']
                        name = call['function']['name']
                        delivered_best = (context.get('_feedback_turn') or {}).get('best', {}).get('experiment_id')
                        if name == 'read_solution' and item['error'] is None and args.get('experiment_id') == 'best' and delivered_best:
                            # "best" names the delivered incumbent: equal reads reuse, a new best reads afresh.
                            from agent.contracts import canonical
                            args = {**args, 'experiment_id': delivered_best}
                            key = (name, canonical(args))
                        if key in tool_cache:
                            previous, original_id = tool_cache[key]
                            result = {'reused': True, 'original_tool_call_id': original_id,
                                      'instruction': 'Reuse the earlier result in this dialogue. Change the query only for an unanswered question.'}
                            for field in ('artifact_id', 'error'):
                                if field in previous:
                                    result[field] = previous[field]
                            if name in {'read_memory', 'read_solution'} and 'text' in previous:
                                result.update({k: deepcopy(v) for k, v in previous.items() if k not in {'reused', 'original_tool_call_id'}})
                            if name in {'analyze_data', 'run_analysis'} and 'report' in previous:
                                # The original reply can have been compacted in the
                                # actual API request; retain the requested report now.
                                result['report'] = deepcopy(previous['report'])
                            messages.append({'role': 'tool', 'tool_call_id': call['id'],
                                             'content': json.dumps(result, ensure_ascii=False)})
                            self.log.event('llm.tool_result_reused', call_no=call_no, tool=name,
                                original_tool_call_id=original_id, operation_key=self._digest(key))
                            continue
                        new_calls += 1
                        if dialogue_id is not None and time.monotonic() >= tool_deadline:
                            force_decision = True
                            result = {'error': {'code': 'TOOL_PHASE_CLOSED', 'message': 'Time is reserved for the final decision; use saved results.'}}
                            trace = {'call_id': call['id'], 'name': name, 'status': 'time_reserve'}
                        elif item['error'] is not None:
                            result = {'error': {'code': 'INVALID_TOOL_ARGUMENTS', 'message': item['error'][:600]},
                                      'instruction': 'Correct the arguments; do not repeat this identical invalid call.'}
                            trace = {'call_id': call['id'], 'name': name, 'status': 'invalid_arguments'}
                        elif name == 'write_code':
                            result = self.bus.call("C11", args, lambda p: write_code(p, self.artifacts), key="code/"+self._digest(args))
                            trace = {"call_id": call["id"], "name": name, "request_id": result["request_id"]}
                        elif name == 'read_solution':
                            from agent.context_details import read_solution
                            try:
                                result = read_solution(args, context, self.artifacts, self.log)
                            except AgentError as error:
                                result = {'error': error.as_dict()}
                            trace = {'call_id': call['id'], 'name': name,
                                     'experiment_id': result.get('experiment_id'), 'evidence_ids': result.get('evidence_ids', [])}
                        elif name == 'edit_solution':
                            from agent.pipeline_drafts import edit_solution
                            context['_elapsed_ms'] = int((time.monotonic()-self.budget.started)*1000)
                            try:result = edit_solution(args, context, self.artifacts, self.log)
                            except AgentError as error:result = {'error':error.as_dict()}
                            trace = {'call_id':call['id'], 'name':name, 'draft_id':result.get('draft_id')}
                        elif name == 'read_memory':
                            from agent.context_details import read_details
                            try:
                                result = read_details(args, context, self.artifacts, self.log)
                                if args['reference_id'] in context['metadata'].get('authorized_feedback_ids', []):
                                    updates = self.artifacts.read_json(args['reference_id'])
                                    measured = {r['experiment_id'] for r in updates.get('results', [])}
                                    context['metadata'].setdefault('delivered_experiment_ids', []).extend(
                                        measured.intersection(result.get('evidence_ids', [])))
                            except AgentError as error:
                                result = {'error': error.as_dict()}
                            trace = {'call_id': call['id'], 'name': name,
                                     'evidence_ids': result.get('evidence_ids', []),
                                     'idea_ids': result.get('idea_ids', [])}
                        elif name == 'run_analysis':
                            from tools.python_analysis import run_analysis
                            try:
                                result = run_analysis(self.analysis, args, context)
                            except AgentError as error:
                                result = {'error': error.as_dict()}
                            trace = {'call_id': call['id'], 'name': name, 'artifact_id': result.get('artifact_id')}
                        elif name == 'analyze_data':
                            try:
                                result = self.analysis.analyze(args, context=context)
                            except AgentError as error:
                                result = {'error': error.as_dict()}
                            trace = {'call_id': call['id'], 'name': name, 'artifact_id': result.get('artifact_id')}
                        else:
                            try:
                                user = json.loads(context['messages'][1]['content'])
                                facts = user.get('profile_summary', user).get('current_pipeline_facts')
                                query = dict(args)
                                if not query.get('ids'):
                                    projection = user.get('profile_summary', user)
                                    decisions = projection.get('research_memory', {}).get('decisions', [])
                                    question = decisions[-1].get('question', '') if decisions else ''
                                    query['query'] = (query.get('query', '')+' '+question)[:1000]
                                result = self.knowledge.search(query, facts)
                            except AgentError as error:
                                result = {'error': error.as_dict()}
                            trace = {'call_id': call['id'], 'name': name, 'artifact_id': result.get('artifact_id'),
                                     'knowledge_ids': [r['id'] for r in result.get('results', [])]}
                        calls.append(trace)
                        tool_error = result.get('error')
                        if isinstance(tool_error, dict):
                            self.log.event('research.tool_failed', level='WARNING', call_no=call_no, tool=name,
                                tool_call_id=call['id'], code=tool_error.get('code'),
                                message=str(tool_error.get('message', ''))[:600],
                                artifact_id=result.get('artifact_id'))
                        if not result.get('error'):
                            delivered_ids = set(result.get('evidence_ids', []))
                            if result.get('artifact_id'):
                                delivered_ids.add(result['artifact_id'])
                            context['metadata'].setdefault('delivered_evidence_ids', []).extend(sorted(delivered_ids))
                        tool_cache[key] = (deepcopy(result), call['id'])
                        if name == 'analyze_data' and result.get('artifact_id') and not result.get('error'):
                            from agent.tool_arguments import report_read
                            report_key = ('analyze_data', json.dumps(report_read(result['artifact_id']),
                                sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False))
                            tool_cache[report_key] = (deepcopy(result), call['id'])
                        messages.append({"role": "tool", "tool_call_id": call["id"], "content": json.dumps(result, ensure_ascii=False)})
                    tool_seconds += time.monotonic()-tools_started
                    self.log.event('llm.tool_batch_normalized', call_no=call_no, raw_calls=len(message['tool_calls']),
                        unique_calls=len(prepared), new_calls=new_calls, reused_calls=len(prepared)-new_calls,
                        invalid_arguments=sum(item['error'] is not None for item in prepared))
                    if not new_calls:
                        force_decision = True
                        self.log.event('llm.tool_no_progress', level='WARNING', call_no=call_no,
                                       reason='All tool requests already have results in this dialogue')
                    last_error = AgentError("INVALID_LLM_OUTPUT", "No final JSON after the tool call")
                    continue
                try:
                    if active_engineer is not None:
                        from agent.llm.repair_engineer import accept
                        parsed = accept(message.get('content'), active_engineer, self.bus.schemas._check)
                        if parsed is None:
                            raise ValueError('Engineer deferred the ambiguous repair to the researcher')
                        recovered = False
                    else:
                        parsed, recovered = parse_final_json(message.get('content'))
                except ValueError as exc:
                    content = message.get('content') or ''
                    # A provider length flag cannot turn duplicate keys, NaN or
                    # a complete semantic error into a request for more tokens.
                    no_final_json = finish_reason == 'length' and isinstance(exc, json.JSONDecodeError) and (
                        not content.strip() or not content.lstrip().startswith(('{', '[', '```')))
                    open_fence = finish_reason == 'length' and content.lstrip().startswith('```') and content.count('```') == 1
                    if no_final_json or open_fence or ((at_cap or finish_reason == 'length') and incomplete_json(content, exc)):
                        raise IncompleteOutput(str(exc)) from exc
                    raise
                if payload['response_schema_id'] == 'ResearchAction-v1':
                    from agent.research_actions import normalize_action
                    if executable:
                        from agent.pipeline_drafts import resolve_candidate
                        try:
                            parsed = resolve_candidate(parsed, context, self.artifacts)
                        except AgentError as error:
                            raise ValueError(error.code+': '+str(error)) from error
                    else:
                        parsed = normalize_action(parsed)
                # Keep the advertised executable form small. Older responses may
                # still include optional explanatory records; accept their known
                # schema without making those records a prerequisite for code.
                accepted_schema = expected
                if executable:
                    accepted_schema = deepcopy(self.output_schemas[payload['response_schema_id']])
                    accepted_schema['properties']['action']['enum'] = expected['properties']['action']['enum']
                self.bus.schemas._check(parsed, accepted_schema, "$")
                if payload['response_schema_id'] == 'ResearchAction-v1':
                    from agent.data_research import require_action
                    try:
                        require_action(parsed, self.budget.data_research_snapshot())
                        validator = getattr(self, 'proposal_validator', None)
                        if validator:
                            validator(parsed, context, {'tool_calls': calls})
                    except AgentError as error:
                        raise ValueError(error.code+': '+str(error)) from error
                if (payload['response_schema_id'] == 'TaskDraft-v1' and 'target_column' in parsed
                        and context['metadata'].get('bootstrap_profile_artifact_id')):
                    # Repair source/column mistakes within the existing bounded
                    # bootstrap dialogue, before they can abort TaskSpec resolution.
                    from agent.semantic_planning import build_task_model
                    source = json.loads(context['messages'][1]['content'])
                    full_profile = self.artifacts.read_json(context['metadata']['bootstrap_profile_artifact_id'])
                    try:
                        build_task_model(parsed['task_model'], source['task_markdown'], full_profile, self.bus.schemas,
                            {'task_spec_id': 'bootstrap-source-check', 'dataset_id': 'bootstrap-source-check',
                             'target_column': parsed['target_column']})
                    except AgentError as error:
                        raise ValueError(error.code+': '+str(error)) from error
                if recovered:
                    self.log.event('llm.json_fence_recovered', call_no=call_no, schema_id=payload['response_schema_id'])
                art = self.artifacts.write_json("llm/parsed-"+uuid.uuid4().hex[:12]+".json", parsed, "llm_output")
                self.log.event("llm.output_validated", call_no=call_no, schema_id=payload["response_schema_id"],
                               attempts_used=attempt, output_artifact_id=art["artifact_id"])
                if active_engineer is not None:
                    self.log.event('llm.repair_engineer_accepted', call_no=call_no, attempt=attempt,
                                   output_artifact_id=art['artifact_id'], intent_preserved=True)
                return {"completion_id": completion_id, "output_artifact_id": art["artifact_id"],
                        "parsed_payload": parsed, "tool_calls": calls, "usage": usage, "attempts_used": attempt}
            except (ValueError, KeyError, TypeError, IndexError) as exc:
                truncated = isinstance(exc, IncompleteOutput)
                code = "LLM_OUTPUT_TRUNCATED" if truncated else "INVALID_LLM_OUTPUT"
                last_error = AgentError(code, f"Model output did not match {payload['response_schema_id']}: {exc}")
                self.log.event("llm.output_truncated" if truncated else "llm.output_rejected", level="WARNING",
                               call_no=call_no, attempt=attempt, code=code, message=str(last_error), finish_reason=finish_reason,
                               max_output_tokens=token_limit)
                if active_engineer is not None:
                    self.log.event('llm.repair_engineer_fallback', call_no=call_no, attempt=attempt,
                                   code=code, reason=str(exc)[:600])
                    messages.append({'role': 'user', 'content':
                        'Interface engineer could not preserve and repair the original proposal. '
                        'Continue from the original rejected action and saved evidence. Error: '+str(exc)[:600]})
                    force_decision = True
                    continue
                if executable and context['metadata'].get('detail_scope'):
                    content = (choice.get('message') or {}).get('content')
                    self._archive_transcript([*messages, {'role':'assistant','content':content}], context)
                    try:
                        raw, _ = parse_final_json(content)
                        if isinstance(raw, dict) and isinstance(raw.get('code'), str) and raw['code']:
                            from agent.pipeline_drafts import edit_solution
                            context['_elapsed_ms'] = int((time.monotonic()-self.budget.started)*1000)
                            draft = edit_solution({'code':raw['code']}, context, self.artifacts, self.log)
                            messages.append({'role':'user','content':'Source retained as draft_id='+draft['draft_id']+
                                '. Use edit_solution to repair fragments; submit code_ref without repeating the full source.'})
                    except (ValueError,AgentError):pass
                if truncated:
                    new_limit = min(token_limit*2, self.config.llm_max_output_tokens_limit)
                    if dialogue_id is not None:
                        new_limit = self.budget.affordable_dialogue_output(dialogue_id, new_limit)
                    if (new_limit <= token_limit and dialogue_id is None) or attempt == payload["attempt_limit"]:
                        raise last_error
                    if new_limit > token_limit:
                        self.log.event("llm.output_budget_increased", level="WARNING", call_no=call_no,
                                       previous_limit=token_limit, next_limit=new_limit, reason="truncated_generation")
                    token_limit = new_limit
                    if dialogue_id is not None:
                        rejected = self.artifacts.write_json('llm/truncated-'+uuid.uuid4().hex[:12]+'.json',
                            {'call_no': call_no, 'content': (choice.get('message') or {}).get('content'), 'error': str(exc)}, 'llm_rejected_output')
                        self.log.event('llm.compact_final_retry', call_no=call_no, dialogue_id=dialogue_id,
                                       rejected_output_artifact_id=rejected['artifact_id'], max_output_tokens=token_limit)
                    messages.append({"role": "user", "content": "Generation was truncated. Return a compact complete final JSON matching the schema; avoid unnecessary code or prose."})
                else:
                    content = (choice.get("message") or {}).get("content")
                    # Adaptive repair needs the actual malformed source/JSON.
                    # Legacy modes keep the previous bounded replay behavior.
                    if dialogue_id is not None:
                        rejected = self.artifacts.write_json('llm/rejected-'+uuid.uuid4().hex[:12]+'.json',
                            {'call_no': call_no, 'content': content, 'error': str(exc)}, 'llm_rejected_output')
                        self.log.event('llm.json_repair_prepared', call_no=call_no, dialogue_id=dialogue_id,
                                       rejected_output_artifact_id=rejected['artifact_id'])
                    if isinstance(content, str) and content.strip() and (dialogue_id is not None or len(content.encode('utf-8')) <= 2000):
                        assistant = {'role': 'assistant', 'content': content}
                        field = self.config.llm_reasoning_field
                        reasoning = (choice.get('message') or {}).get(field)
                        if self.config.llm_reasoning_protocol == 'qwen_chat_template' and isinstance(reasoning, str):
                            assistant[field] = reasoning
                        messages.append(assistant)
                        self._bound_reasoning_history(messages)
                    messages.append({"role": "user", "content": "Return one valid JSON object matching the schema. Error: "+str(exc)[:600]})
                if (not truncated and not engineer_used and executable and dialogue_id is not None
                        and getattr(self.config, 'llm_repair_engineer_enabled', False)
                        and attempt < payload['attempt_limit'] and deadline-time.monotonic() > 2):
                    from agent.llm.repair_engineer import prepare
                    source_user = json.loads(context['messages'][1]['content'])
                    base = source_user.get('base_result')
                    detail = context['metadata'].get('detail_artifact_id')
                    if base is None and detail:
                        base = self.artifacts.read_json(detail).get('sections', {}).get('base_result')
                    engineer_packet = prepare(content, exc, expected, base)
                    if engineer_packet is not None:
                        engineer_used = True
                        record = self.artifacts.write_json('llm/engineer-'+uuid.uuid4().hex[:12]+'.json',
                                                           engineer_packet, 'repair_engineer_packet')
                        self.log.event('llm.repair_engineer_prepared', call_no=call_no,
                                       packet_artifact_id=record['artifact_id'], dialogue_id=dialogue_id)
                if dialogue_id is not None:
                    force_decision = True
        raise last_error or AgentError("INVALID_LLM_OUTPUT", "No final JSON within the bounded tool/repair loop")

    def _archive_transcript(self, messages, context):
        scope = deepcopy(context['metadata'].get('detail_scope', {}))
        value = {k: scope.get(k) for k in ('task_spec_id', 'dataset_id', 'validation_id')}
        value.update(branch_id=scope.get('branch_id') or 'legacy', scope=scope,
            messages=deepcopy(messages), results=[], kind='research_transcript',
            completed_elapsed_ms=int((time.monotonic()-self.budget.started)*1000),
            instruction='Exact observable messages; tool/model statements are not new verified measurements.')
        art = self.artifacts.write_json('research/transcript-'+uuid.uuid4().hex[:16]+'.json',
            self.log.redactor.clean(value), 'research_transcript')
        context['metadata'].setdefault('authorized_feedback_ids', []).append(art['artifact_id'])
        context.setdefault('_transcript_ids', []).append(art['artifact_id'])
        return {'reference_id': art['artifact_id'], 'section': 'transcript_index'}

    def _research_checkpoint(self, context, messages, error, attempts):
        from agent.context_transcript import observations
        from agent.research_actions import normalize_action
        records = observations(messages)
        question = next((r.get('question') for r in reversed(records) if r.get('question')), None)
        if not question:
            user = json.loads(context['messages'][1]['content'])
            question = user.get('profile_summary', user).get('error_research', {}).get('recommendation', {}).get(
                'question', 'Continue the unresolved question from saved reports and drafts.')
        checkpoint = normalize_action({'action': 'research', 'next_step': 'continue',
            'evidence_ids': [],
            'hypothesis': str(question)[:650]+' Continue from the archived results and last rejected proposal; repair it rather than '
                'restarting the investigation. Last failure: '+error.code+': '+str(error)[:300]})
        art = self.artifacts.write_json('research/controller-checkpoint-'+uuid.uuid4().hex[:12]+'.json',
                                       checkpoint, 'controller_research_checkpoint')
        self.log.event('research.context_checkpoint', artifact_id=art['artifact_id'], reason=error.code,
                       decision_origin='controller', question=question)
        return {'completion_id':'checkpoint-'+uuid.uuid4().hex[:12], 'output_artifact_id':art['artifact_id'],
            'parsed_payload':checkpoint, 'tool_calls':[{'name':'transcript_archive','reference_id':aid,'status':'archived'}
                for aid in context.get('_transcript_ids', [])],
            'usage':{'input_tokens':None,'output_tokens':None}, 'attempts_used':attempts}

    def _compact_research_request(self, request, limit, *, archive=None):
        from agent.context_budget import compact_research_request
        return compact_research_request(request, limit, self.log, archive=archive)

    @staticmethod
    def _section_bytes(request):
        from agent.context_budget import request_bytes
        return {'messages': request_bytes(request.get('messages', [])),
                'tools': request_bytes(request.get('tools', [])),
                'other_fields': request_bytes({k: v for k, v in request.items() if k not in {'messages', 'tools'}})}

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
        self.stop_research()
        # Third-party client shutdown is not allowed to hold publication/process
        # exit past the deadline. Research executor threads are daemon-owned.
        closer = threading.Thread(target=self._close_transport, daemon=True, name='llm-close')
        closer.start()
        closer.join(timeout=.2)

    def _close_transport(self):
        try:self.transport.close()
        except Exception:pass

    def stop_research(self):
        self.cancelled.set()
        # A stuck call may already be billable. Settle its maximum reservation
        # conservatively; a late answer is fenced and cannot create a candidate.
        with self.budget.lock:
            for call_no in list(self.budget.pending_llm_tokens):
                self.budget.finish_llm_call(call_no)
            for ident in list(self.budget.llm_dialogues):
                self.budget.release_dialogue(ident)
