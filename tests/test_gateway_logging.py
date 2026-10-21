from dataclasses import replace
from pathlib import Path
import json
import os
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from agent.budget import Budget
from agent.context import output_schemas
from agent.errors import AgentError
from agent.llm.gateway import DEFAULT_MODEL, Gateway, OpenAITransport, parse_json, request_model
from tests.helpers import Harness


class FakeTransport:
    def __init__(self, responses):
        self.responses, self.requests = list(responses), []

    def complete(self, request, timeout, context):
        self.requests.append(request)
        value = self.responses.pop(0)
        if isinstance(value, Exception):
            raise value
        return value

    def close(self):
        pass


class GatewayTests(unittest.TestCase):
    def setUp(self):
        self.h = Harness()

    def tearDown(self):
        self.h.close()

    def context(self):
        return self.h.artifacts.write_json("contexts/unit.json", {"messages": [{"role": "user", "content": "synthetic task"}],
                                                               "metadata": {"researcher_id": "researcher-1", "schema_id": "ResearchProposal-v1"}})["artifact_id"]

    def final_payload(self):
        return {"recipe_id": "linear-classifier-v1", "hyperparameters": {"C": 1.0}, "hypothesis": "Check alternative",
                "expected_effect": "Measure on common split", "feature_steps": [], "code": self.h.code_payload["content"]}

    def args(self):
        return {"purpose": "research_proposal", "context_artifact_id": self.context(), "response_schema_id": "ResearchProposal-v1",
                "tool_schema_ids": ["write_code"], "max_output_tokens": 1024, "deadline_seconds": 5, "attempt_limit": 3}

    def test_native_tool_call_then_structured_plan(self):
        tool = {"id": "call-1", "type": "function", "function": {"name": "write_code", "arguments": json.dumps(self.h.code_payload)}}
        transport = FakeTransport([
            {"id": "first", "choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [tool]}}]},
            {"id": "second", "choices": [{"message": {"role": "assistant", "content": json.dumps(self.final_payload())}}]},
        ])
        budget = Budget(self.h.config, time.monotonic(), self.h.log)
        gateway = Gateway(self.h.config, budget, self.h.artifacts, self.h.bus, self.h.log, output_schemas(self.h.bus), transport)
        result = gateway.complete(self.args())
        self.assertEqual(result["parsed_payload"]["recipe_id"], "linear-classifier-v1")
        self.assertEqual(result["attempts_used"], 2)
        self.assertEqual(result["tool_calls"][0]["name"], "write_code")
        self.assertEqual(transport.requests[1]["messages"][-1]["role"], "tool")
        self.assertTrue(list((self.h.run / "llm").glob("*.request.json")))

    def test_invalid_output_has_bounded_repair(self):
        final = {"choices": [{"message": {"role": "assistant", "content": json.dumps(self.final_payload())}}]}
        transport = FakeTransport([{"choices": [{"message": {"role": "assistant", "content": "not JSON"}}]}, final])
        gateway = Gateway(self.h.config, Budget(self.h.config, time.monotonic(), self.h.log), self.h.artifacts, self.h.bus, self.h.log, output_schemas(self.h.bus), transport)
        self.assertEqual(gateway.complete(self.args())["attempts_used"], 2)

    def test_http_auth_error_is_not_retried(self):
        exc = RuntimeError("unauthorized")
        exc.status_code = 401
        transport = FakeTransport([exc])
        gateway = Gateway(self.h.config, Budget(self.h.config, time.monotonic(), self.h.log), self.h.artifacts, self.h.bus, self.h.log, output_schemas(self.h.bus), transport)
        with self.assertRaises(AgentError):
            gateway.complete(self.args())
        self.assertEqual(len(transport.requests), 1)

    def test_live_mode_requires_organizers_env(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "", "OPENAI_MODEL": "", "OPENAI_BASE_URL": ""}):
            with self.assertRaises(AgentError) as caught:
                OpenAITransport()
        self.assertEqual(caught.exception.code, "INVALID_CONFIG")
        self.assertEqual(caught.exception.details["missing"], ["OPENAI_API_KEY", "OPENAI_BASE_URL"])

    def test_temporary_rate_limit_can_retry(self):
        exc = RuntimeError("temporary rate limit")
        exc.status_code = 429
        transport = FakeTransport([exc, {"choices": [{"message": {"content": json.dumps(self.final_payload())}}]}])
        gateway = Gateway(self.h.config, Budget(self.h.config, time.monotonic(), self.h.log), self.h.artifacts, self.h.bus, self.h.log, output_schemas(self.h.bus), transport)
        with patch("agent.llm.gateway.time.sleep"):
            self.assertEqual(gateway.complete(self.args())["attempts_used"], 2)
        self.assertFalse(gateway.daily_budget_exhausted.is_set())

    def test_model_is_optional_and_platform_override_wins(self):
        with patch.dict(os.environ, {"OPENAI_MODEL": ""}):
            self.assertEqual(request_model("live"), DEFAULT_MODEL)
            self.assertEqual(request_model("mock"), "mock-only")
        with patch.dict(os.environ, {"OPENAI_MODEL": "platform-selected-model"}):
            self.assertEqual(request_model("live"), "platform-selected-model")

    def test_live_transport_needs_no_model_and_uses_only_env_endpoint(self):
        calls = []
        fake = SimpleNamespace(OpenAI=lambda **kwargs: calls.append(kwargs) or SimpleNamespace(close=lambda: None))
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-placeholder", "OPENAI_BASE_URL": "https://provided.invalid/v1", "OPENAI_MODEL": ""}), patch.dict("sys.modules", {"openai": fake}):
            transport = OpenAITransport()
            transport.close()
        self.assertEqual(calls, [{"api_key": "test-placeholder", "base_url": "https://provided.invalid/v1", "max_retries": 0}])

    def test_daily_key_budget_stops_retry_and_followup_calls(self):
        for status, body_in_exception in [(429, False), (403, True)]:
            with self.subTest(status=status):
                exc = RuntimeError("quota error" if body_in_exception else "Budget has been exceeded")
                exc.status_code = status
                if body_in_exception:
                    exc.body = {"error": {"message": "Budget has been exceeded"}}
                transport = FakeTransport([exc])
                gateway = Gateway(self.h.config, Budget(self.h.config, time.monotonic(), self.h.log), self.h.artifacts, self.h.bus, self.h.log, output_schemas(self.h.bus), transport)
                args = self.args()
                for _ in range(2):
                    with self.assertRaises(AgentError) as caught:
                        gateway.complete(args)
                    self.assertEqual(caught.exception.code, "LLM_BUDGET_EXHAUSTED")
                    self.assertFalse(caught.exception.retryable)
                    self.assertEqual(caught.exception.details["reset_time_msk"], "03:00")
                self.assertEqual(len(transport.requests), 1)

    def test_provider_insufficient_credits_stops_retry_without_claiming_team_budget(self):
        for in_body in (False, True):
            with self.subTest(in_body=in_body):
                exc = RuntimeError("upstream failure" if in_body else "OpenrouterException - Insufficient credits")
                exc.status_code = 503
                if in_body:
                    exc.body = {"error": {"message": "OpenrouterException - Insufficient credits"}}
                transport = FakeTransport([exc])
                gateway = Gateway(self.h.config, Budget(self.h.config, time.monotonic(), self.h.log),
                                  self.h.artifacts, self.h.bus, self.h.log, output_schemas(self.h.bus), transport)
                with patch("agent.llm.gateway.time.sleep") as sleep:
                    for _ in range(2):
                        with self.assertRaises(AgentError) as caught:
                            gateway.complete(self.args())
                        self.assertEqual(caught.exception.code, "LLM_PROVIDER_CREDITS_EXHAUSTED")
                        self.assertFalse(caught.exception.retryable)
                        self.assertEqual(caught.exception.details["team_daily_budget_status"], "unknown")
                        self.assertIsNone(caught.exception.details["reset_time_msk"])
                    sleep.assert_not_called()
                self.assertEqual(len(transport.requests), 1)
                self.assertFalse(gateway.daily_budget_exhausted.is_set())

    def test_temporary_503_without_credit_error_can_retry(self):
        exc = RuntimeError("temporary service unavailable")
        exc.status_code = 503
        transport = FakeTransport([exc, {"choices": [{"message": {"content": json.dumps(self.final_payload())}}]}])
        gateway = Gateway(self.h.config, Budget(self.h.config, time.monotonic(), self.h.log),
                          self.h.artifacts, self.h.bus, self.h.log, output_schemas(self.h.bus), transport)
        with patch("agent.llm.gateway.time.sleep"):
            self.assertEqual(gateway.complete(self.args())["attempts_used"], 2)
        self.assertFalse(gateway.provider_credits_exhausted.is_set())

    def test_strict_json_rejects_duplicate_keys_and_nan(self):
        for text in ['{"x":1,"x":2}', '{"x":NaN}', 'prefix {"x":1}']:
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_json(text)

    def test_secrets_redacted_in_events_and_snapshots(self):
        self.h.log.event("unit", message="Bearer abcdefghi unit-private-secret", api_key="different-value")
        self.h.log.snapshot("messages/redaction.json", {"Authorization": "Bearer abcdef", "prompt": "unit-private-secret"})
        for path in [self.h.run / "events.jsonl", self.h.run / "messages/redaction.json"]:
            content = path.read_text(encoding="utf-8")
            self.assertNotIn("unit-private-secret", content)
            self.assertNotIn("abcdef", content)
            self.assertIn("[REDACTED]", content)
