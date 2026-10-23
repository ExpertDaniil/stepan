"""Regression cases for the observed repeated-call and wrapped-JSON responses."""
import json
import time
import unittest

from agent.budget import Budget
from agent.context import output_schemas
from agent.contracts import Schemas
from agent.llm.gateway import Gateway
from agent.llm.protocol import IncompleteOutput, parse_final_json, parse_json, prepare_tool_calls
from tests import test_gateway_logging as gateway_cases


def tool(args, ident='provider-id'):
    return {'id': ident, 'type': 'function', 'function': {'name': 'write_code', 'arguments': json.dumps(args)}}


class ProtocolRecoveryTests(unittest.TestCase):
    def test_single_fence_with_prose_is_recovered_without_relaxing_json(self):
        self.assertEqual(parse_final_json('A proposed action.\n```json\n{"x": 1}\n```\nDone.'), ({'x': 1}, True))
        self.assertEqual(parse_final_json('{"x":1}'), ({'x': 1}, False))
        for text in ('{"x":1,"x":2}', '{"x":1e10000}', 'Note\n```json\n{"x":NaN}\n```',
                     'Note\n```json\n{"x":1,"x":2}\n```',
                     '```json\n{"x":1}\n```\n```json\n{"x":2}\n```',
                     '{"other":2}\n```json\n{"x":1}\n```',
                     '<tool_call>bad</tool_call>\n```json\n{"x":1}\n```',
                     '{"outer": invalid {"x":1}}'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_final_json(text)
        with self.assertRaises(ValueError):
            parse_json('Explanation\n```json\n{"x":1}\n```')

    def test_repeated_call_ids_and_key_order_do_not_multiply_execution(self):
        schema = {'type': 'object', 'properties': {'x': {'type': 'integer'}, 'y': {'type': 'integer'}},
                  'required': ['x', 'y'], 'additionalProperties': False}
        calls = [tool({'x': 1, 'y': 2}) for _ in range(145)]+[tool({'y': 2, 'x': 1}) for _ in range(145)]
        prepared = prepare_tool_calls(calls, {'write_code': schema}, Schemas()._check, None, at_cap=True)
        self.assertEqual(len(prepared), 1)
        self.assertIsNone(prepared[0]['error'])

    def test_invalid_last_arguments_preserve_valid_siblings(self):
        schema = {'type': 'object', 'properties': {'x': {'type': 'integer'}}, 'required': ['x'], 'additionalProperties': False}
        prepared = prepare_tool_calls([tool({'x': 1}), tool({})], {'write_code': schema}, Schemas()._check, None, at_cap=True)
        self.assertIsNone(prepared[0]['error'])
        self.assertIsNotNone(prepared[1]['error'])
        self.assertEqual(len({item['call']['id'] for item in prepared}), 2)

    def test_only_actual_single_incomplete_arguments_request_more_tokens(self):
        call = tool({})
        call['function']['arguments'] = '{"x":'
        with self.assertRaises(IncompleteOutput):
            prepare_tool_calls([call], {'write_code': {}}, Schemas()._check, None, at_cap=True)
        repeated = prepare_tool_calls([call]*290, {'write_code': {}}, Schemas()._check, None, at_cap=True)
        self.assertIsNotNone(repeated[0]['error'])


class GatewayRecoveryTests(unittest.TestCase):
    setUp = gateway_cases.GatewayTests.setUp
    tearDown = gateway_cases.GatewayTests.tearDown
    context = gateway_cases.GatewayTests.context
    args = gateway_cases.GatewayTests.args
    final_payload = gateway_cases.GatewayTests.final_payload

    def gateway(self, responses):
        transport = gateway_cases.FakeTransport(responses)
        gateway = Gateway(self.h.config, Budget(self.h.config, time.monotonic(), self.h.log),
                          self.h.artifacts, self.h.bus, self.h.log, output_schemas(self.h.bus), transport)
        return gateway, transport

    def final(self, payload=None, finish_reason='stop', tokens=100):
        return {'choices': [{'message': {'content': json.dumps(payload or self.final_payload())},
                             'finish_reason': finish_reason}], 'usage': {'completion_tokens': tokens}}

    def test_290_calls_become_one_history_pair_and_one_execution(self):
        reply = {'choices': [{'message': {'tool_calls': [tool(self.h.code_payload)]*290}, 'finish_reason': 'tool_calls'}],
                 'usage': {'completion_tokens': 1024}}
        gateway, transport = self.gateway([reply, self.final()])
        result = gateway.complete(self.args())
        self.assertEqual(len(result['tool_calls']), 1)
        history = transport.requests[1]['messages']
        assistant = next(m for m in history if m.get('tool_calls'))
        replies = [m for m in history if m['role'] == 'tool']
        self.assertEqual(len(assistant['tool_calls']), 1)
        self.assertEqual(len(replies), 1)
        self.assertEqual(assistant['tool_calls'][0]['id'], replies[0]['tool_call_id'])
        self.assertEqual([r['max_tokens'] for r in transport.requests], [1024, 1024])

    def test_bad_tail_at_cap_is_an_individual_error_not_truncation(self):
        reply = {'choices': [{'message': {'tool_calls': [tool(self.h.code_payload), tool({})]}, 'finish_reason': 'tool_calls'}],
                 'usage': {'completion_tokens': 1024}}
        gateway, transport = self.gateway([reply, self.final()])
        result = gateway.complete(self.args())
        self.assertEqual(len(result['tool_calls']), 2)
        replies = [json.loads(m['content']) for m in transport.requests[1]['messages'] if m['role'] == 'tool']
        self.assertEqual(replies[-1]['error']['code'], 'INVALID_TOOL_ARGUMENTS')
        self.assertEqual([r['max_tokens'] for r in transport.requests], [1024, 1024])

    def test_repeated_error_ends_tool_phase_without_reexecution(self):
        reply = {'choices': [{'message': {'tool_calls': [tool({})]}, 'finish_reason': 'tool_calls'}]}
        gateway, transport = self.gateway([reply, reply, self.final()])
        args = self.args();args['attempt_limit'] = 5
        result = gateway.complete(args)
        self.assertEqual(len(result['tool_calls']), 1)
        self.assertNotIn('tools', transport.requests[2])
        reused = [json.loads(m['content']) for m in transport.requests[2]['messages'] if m['role'] == 'tool'][-1]
        self.assertTrue(reused['reused'])

    def test_complete_schema_error_at_cap_does_not_expand_budget(self):
        bad = self.final_payload();bad.pop('hypothesis')
        gateway, transport = self.gateway([self.final(bad, tokens=1024), self.final()])
        gateway.complete(self.args())
        self.assertEqual([r['max_tokens'] for r in transport.requests], [1024, 1024])

    def test_complete_valid_object_with_length_flag_is_accepted(self):
        gateway, transport = self.gateway([self.final(finish_reason='length', tokens=1024)])
        self.assertEqual(gateway.complete(self.args())['attempts_used'], 1)
        self.assertEqual(len(transport.requests), 1)
