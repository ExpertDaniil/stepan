"""Bounded observations from completed tool turns; exact transcripts stay scoped."""
from copy import deepcopy
import json


def encoded(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def excerpt(value, maximum):
    text = value if isinstance(value, str) else encoded(value)
    return text.encode('utf-8')[:maximum].decode('utf-8', errors='ignore')


def parse(value):
    try:
        result = json.loads(value)
        return result if isinstance(result, dict) else {}
    except (ValueError, TypeError):
        return {}


def batches(messages):
    """Only complete contiguous batches can be removed, never orphan a reply."""
    result = []
    for start, message in enumerate(messages):
        calls = message.get('tool_calls', []) if message.get('role') == 'assistant' else []
        ids = [c.get('id') for c in calls]
        if not ids or None in ids or len(ids) != len(set(ids)):
            continue
        end = start+1
        while end < len(messages) and messages[end].get('role') == 'tool':
            end += 1
        replies = messages[start+1:end]
        if len(replies) == len(ids) and {r.get('tool_call_id') for r in replies} == set(ids):
            result.append((start, end))
    return result


def observations(messages):
    records = []
    for start, end in batches(messages):
        assistant = messages[start]
        replies = {m['tool_call_id']: parse(m.get('content')) for m in messages[start+1:end]}
        for call in assistant['tool_calls']:
            function = call.get('function', {})
            args, reply = parse(function.get('arguments')), replies[call['id']]
            report = reply.get('report') if isinstance(reply.get('report'), dict) else reply
            record = {'tool': function.get('name'), 'call_id': call['id'], 'preview_only': True}
            question = args.get('question') or (args.get('investigation') or {}).get('question')
            if question:
                record['question'] = excerpt(question, 480)
            if assistant.get('content'):
                record['model_statement_unverified'] = excerpt(assistant['content'], 480)
            query = {k: v for k, v in args.items() if k in {
                'kind', 'columns', 'group_by', 'include_target', 'experiment_id', 'view', 'section', 'ids'}}
            if query:
                record['query_preview'] = excerpt(query, 480)
            for key in ('artifact_id', 'code_artifact_id', 'reference_id', 'section', 'experiment_id', 'next_offset', 'next_read'):
                if key in reply:
                    record[key] = deepcopy(reply[key])
            for key in ('scope', 'validation_id', 'sampled', 'analyzed_rows', 'rows', 'success'):
                if key in report:
                    record[key] = deepcopy(report[key])
            if reply.get('error') or report.get('error'):
                record['error_preview'] = excerpt(reply.get('error') or report['error'], 480)
            else:
                measured = {k: report[k] for k in ('result', 'columns', 'groups', 'associations', 'checks', 'stdout') if k in report}
                if measured:
                    record['observation_preview'] = excerpt(measured, 900)
                elif isinstance(reply.get('text'), str):
                    record['observation_preview'] = excerpt(reply['text'], 900)
            records.append(record)
    return records


def transcript_sections(transcript):
    """Read a result without paging through the original task/schema or Python."""
    index, sections = [], {'updates': transcript}
    messages = transcript.get('messages', [])
    for start, end in batches(messages):
        replies = {m['tool_call_id']: parse(m.get('content')) for m in messages[start+1:end]}
        for call in messages[start]['tool_calls']:
            prefix = 'tool/'+call['id']
            args = parse(call['function'].get('arguments'))
            index.append({'call_id': call['id'], 'tool': call['function']['name'],
                'question': excerpt(args.get('question', ''), 480),
                'result_section': prefix+'/result', 'arguments_section': prefix+'/arguments'})
            sections[prefix+'/result'] = replies[call['id']]
            sections[prefix+'/arguments'] = args or call['function'].get('arguments')
    sections['transcript_index'] = index
    return sections


def visible_transcripts(memory, artifacts, scope):
    """Dispatcher visibility also covers a dialogue that failed before a decision."""
    from agent.errors import AgentError
    decisions = {aid: decision for decision in memory.get('decisions', [])
        for aid in decision.get('transcript_artifact_ids', [])}
    visible = set(memory.get('visible_work_order_ids') or [])
    cutoffs = memory.get('visible_analysis_cutoffs', {})
    with artifacts.lock:
        ids = [aid for aid, entry in artifacts.entries.items() if entry['meta']['kind'] == 'research_transcript']
    for aid in ids:
        transcript = artifacts.read_json(aid)
        source = transcript.get('scope', {})
        decision = decisions.get(aid)
        owner = source.get('work_order_id')
        if not decision and owner not in visible:
            continue
        if (source.get('run_id') != scope['run_id'] or any(source.get(k) != scope.get(k)
                for k in ('task_spec_id', 'dataset_id', 'validation_id'))
                or (decision and owner != decision.get('work_order_id'))):
            if decision:
                raise AgentError('MEMORY_SCOPE_DENIED', 'Transcript differs from its visible decision scope')
            continue
        cutoff = cutoffs.get(owner)
        completed = transcript.get('completed_elapsed_ms')
        if cutoff is not None and (not isinstance(completed, int) or completed > cutoff):
            continue
        records = observations(transcript.get('messages', []))
        question = decision['question'] if decision else (records[-1].get('question') if records else None)
        yield artifacts.meta(aid), (decision or {}).get('decision_artifact_id'), question


def compact_history(request, limit, archive, original_messages, log=None):
    """Archive before removing pairs; preserve the newest batch whenever it fits."""
    def size(value):
        return len(json.dumps(value, ensure_ascii=False, allow_nan=False).encode('utf-8'))
    if size(request) <= limit or archive is None or not batches(request['messages']):
        return []
    messages = request['messages']
    previous = []
    for message in messages:
        value = parse(message.get('content')) if message.get('role') == 'user' else {}
        if 'research_transcript' in value:
            previous.append(message)
    prior_records = [r for m in previous for r in parse(m['content'])['research_transcript'].get('records', [])]
    prior_omitted = sum(parse(m['content'])['research_transcript'].get('omitted_records', 0) for m in previous)
    blocks = batches(messages)
    source_records = observations(original_messages)
    removed_ids = []
    selected = set()
    # Consider the oldest complete turns first. Even a single large parallel
    # batch can be archived; its exact code/results remain readable on demand.
    for start, end in blocks:
        selected.update(range(start, end))
        removed_ids.extend(c['id'] for c in messages[start]['tool_calls'])
        records = prior_records+source_records[:len(removed_ids)]
        body = {'research_transcript': {
            'instruction': 'Tool output excerpts, not new conclusions. Exact calls/results are in the scoped archive. '
                'Reuse completed reports; do not rerun them only because their original tool turn was compacted.',
            'records': records[-8:], 'omitted_records': prior_omitted+max(0, len(records)-8),
            'latest_requested_check': records[-1].get('question') if records else None,
            'next_step': ('Repair the latest failed check from its saved error, then finish the hypothesis.'
                if records and ('error_preview' in records[-1] or records[-1].get('success') is False) else
                'Finish the decision for the latest recorded question using completed reports; submit the next executable experiment.')}}
        summary = {'role': 'user', 'content': encoded(body)}
        survivors = [m for i, m in enumerate(messages) if i not in selected and not any(m is old for old in previous)]
        insert = next((i for i, m in enumerate(survivors) if m.get('role') == 'assistant'), len(survivors))
        # Put the summary before the retained transcript and before a terminal
        # instruction, without moving either system or original task messages.
        insert = min(insert, next((i for i, m in enumerate(survivors)
            if i > 1 and m.get('role') == 'user' and 'Return one final JSON' in (m.get('content') or '')), insert))
        candidate = {**request, 'messages': survivors[:insert]+[summary]+survivors[insert:]}
        # Reserve the actual archive reference and framing before persisting.
        body['research_transcript']['archive'] = {'reference_id': 'art-'+'0'*16, 'section': 'transcript_index'}
        summary['content'] = encoded(body)
        while size(candidate) > limit and len(body['research_transcript']['records']) > 1:
            body['research_transcript']['records'].pop(0)
            body['research_transcript']['omitted_records'] += 1
            summary['content'] = encoded(body)
        if size(candidate) > limit and (start, end) != blocks[-1]:
            continue
        if size(candidate) > limit:
            for record in body['research_transcript']['records']:
                for key in ('observation_preview', 'model_statement_unverified', 'query_preview'):
                    record.pop(key, None)
            summary['content'] = encoded(body)
        if size(candidate) > limit:
            return []  # The protected core itself needs an explicit rejection.
        reference = archive(original_messages)
        body['research_transcript']['archive'] = reference
        summary['content'] = encoded(body)
        if size(candidate) > limit:
            return []
        request['messages'] = candidate['messages']
        if log is not None:
            log.event('llm.tool_history_archived', reference_id=reference['reference_id'],
                removed_tool_call_ids=removed_ids, retained_tool_batches=len(batches(request['messages'])),
                request_bytes=size(request), limit_bytes=limit)
        return removed_ids
    return []
