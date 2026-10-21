from dataclasses import replace
from pathlib import Path
import json
import os
import time
import unittest
from unittest.mock import patch

from agent.budget import Budget
from agent.context import output_schemas
from agent.errors import AgentError
from agent.llm.gateway import Gateway, OpenAITransport, parse_json
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
