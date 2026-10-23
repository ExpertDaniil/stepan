"""Strict tool arguments and conservative recovery of a single final JSON fence."""
from copy import deepcopy
import json
import re
import uuid

from agent.contracts import canonical
from agent.tool_arguments import normalize_tool_arguments


class IncompleteOutput(ValueError):
    """A syntactically incomplete generation, rather than a semantic/schema error."""


def parse_json(text):
    text = (text or '').strip()
    if text.startswith('```'):
        match = re.fullmatch(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
        if not match:
            raise ValueError('Malformed JSON code fence')
        text = match[1]

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Duplicate JSON key: '+key)
            result[key] = value
        return result

    value = json.loads(text, object_pairs_hook=unique_object,
                       parse_constant=lambda item: (_ for _ in ()).throw(ValueError('Non-finite JSON: '+item)))
    canonical(value)  # Also reject numeric overflow such as 1e10000 -> infinity.
    return value


def parse_final_json(text):
    """Recover only one explicit JSON fence; never scan inside malformed objects."""
    try:
        return parse_json(text), False
    except ValueError as original:
        body = (text or '').strip()
        blocks = list(re.finditer(r'(?m)^[ \t]*```(?:json)?[ \t]*\r?\n([\s\S]*?)^[ \t]*```[ \t]*(?=\r?$)', body))
        if len(blocks) != 1 or body.count('```') != 2:
            raise original
        block = blocks[0]
        outside = body[:block.start()]+body[block.end():]
        if any(c in outside for c in '{}[]') or re.search(r'<\s*/?\s*(?:tool_call|function)\b', outside, re.I):
            raise ValueError('Ambiguous JSON outside the final fence') from original
        return parse_json(block[1]), True


def incomplete_json(text, error):
    if not isinstance(error, json.JSONDecodeError):
        return False
    body = (text or '').strip()
    return not body or error.pos >= len(body) or error.msg.startswith('Unterminated string')


def prepare_tool_calls(raw_calls, schema_by_name, validate, researcher_id, *, at_cap=False):
    """Validate each unique call independently before any execution/history update.

    Canonicalized history has exactly one tool reply for every retained call ID.
    Provider IDs are telemetry; fresh IDs also handle repeated/missing provider IDs.
    An invalid sibling cannot discard other complete, valid calls.
    """
    if not isinstance(raw_calls, list):
        raise ValueError('tool_calls must be an array')
    prepared, seen = [], set()
    for raw in raw_calls:
        if not isinstance(raw, dict) or not isinstance(raw.get('function'), dict):
            raise ValueError('Malformed tool call')
        function = raw['function']
        name, arguments = function.get('name'), function.get('arguments')
        if name not in schema_by_name or not isinstance(arguments, str):
            raise ValueError('Unexpected tool or non-string tool arguments')
        args, error, syntax_error = None, None, None
        try:
            args = normalize_tool_arguments(name, parse_json(arguments))
        except ValueError as exc:
            syntax_error = exc
            error = str(exc)
        key = (name, canonical(args) if syntax_error is None else arguments.strip())
        if key in seen:
            continue
        seen.add(key)
        if syntax_error is None:
            try:
                validate(args, schema_by_name[name], '$.tool')
                if name == 'write_code' and args.get('researcher_id') != researcher_id:
                    raise ValueError("Tool call uses another researcher's namespace")
            except (ValueError, TypeError, KeyError) as exc:
                error = str(exc)
        call = {'id': 'call-'+uuid.uuid4().hex[:16], 'type': 'function',
                'function': {'name': name, 'arguments': key[1]}}
        prepared.append({'call': call, 'args': deepcopy(args), 'key': key,
                         'error': error, 'syntax_error': syntax_error,
                         'provider_call_id': raw.get('id')})
    if len(raw_calls) == len(prepared) == 1 and at_cap:
        item = prepared[0]
        if incomplete_json(item['call']['function']['arguments'], item['syntax_error']):
            raise IncompleteOutput('Tool arguments were cut off at the output limit')
    return prepared
