"""One complete JSON-body budget. Bytes are measured; tokens are not.

Without a proxy tokenizer, one input byte per estimated token is an explicit
conservative safety proxy. Provider tokenization/framing remains authoritative.
"""
import json
from copy import deepcopy


def request_bytes(value):
    return len(json.dumps(value, ensure_ascii=False, allow_nan=False).encode('utf-8'))


def effective_limit(config, metadata, max_output_tokens):
    limits = {'client_bytes': config.llm_request_max_bytes}
    if config.research_mode != 'adaptive':
        limits['legacy_byte_estimate'] = config.max_context_tokens * 3
    if metadata.get('graph_binding'):
        limits['graph_bytes'] = config.graph_context_bytes
        if metadata.get('graph_request_max_bytes') is not None:
            limits['issued_graph_bytes'] = metadata['graph_request_max_bytes']
    limits['provider_safety_byte_proxy'] = max(0, config.llm_context_window_tokens-max_output_tokens-1024)
    return min(limits.values()), limits


def assembly_limit(config, metadata):
    limit, _ = effective_limit(config, metadata, config.llm_max_output_tokens_limit)
    # Semantic commitments extend the output/tool schemas. Reserve up to 32 KiB
    # for their actual JSON framing plus feedback and the bounded tool transcript.
    return max(0, limit-min(32768, limit//3))


def compact_tool_reply(value, maximum):
    """Shorten retrievable details, preserving scope, IDs and whole observations.

    This is a display projection only. Original service artifacts and the tool
    cache remain unchanged. A shortened text page must not claim a complete read.
    """
    result = deepcopy(value)
    if not isinstance(result, dict) or result.get('error') or request_bytes(result) <= maximum:
        return result
    if isinstance(result.get('text'), str) and 'reference_id' in result:
        page = result
    elif isinstance(result.get('report'), dict) and isinstance(result['report'].get('text'), str):
        page = result['report']
    else:
        page = None
    if page is not None:
        original = page['text'].encode('utf-8')
        while request_bytes(result) > maximum and len(page['text'].encode('utf-8')) > 256:
            raw = page['text'].encode('utf-8')
            page['text'] = raw[:max(256, len(raw)//2)].decode('utf-8', errors='ignore')
        if page['text'].encode('utf-8') != original:
            page['next_offset'] = page.get('offset', 0)+len(page['text'].encode('utf-8'))
            page['details_omitted'] = True
            if 'evidence_ids' in result:
                result['evidence_ids'], result['idea_ids'] = [], []
        return result
    report = result.get('report')
    if isinstance(report, dict) and result.get('artifact_id'):
        query = report.get('query', {})
        if isinstance(query, dict) and 'investigation' in query:
            query.pop('investigation')
            report['investigation_in_full_report'] = True
        for field in ('groups', 'columns', 'associations', 'hard_groups', 'checks'):
            records = report.get(field)
            while request_bytes(result) > maximum and isinstance(records, list) and len(records) > 1:
                records.pop()
                report['display_omitted_'+field] = report.get('display_omitted_'+field, 0)+1
        report['detail_hint'] = 'analyze_data kind=report with report_id='+result['artifact_id']+'; use offset/limit_bytes for pages.'
        report['details_omitted'] = True
    if isinstance(result.get('results'), list) and result.get('artifact_id'):
        for record in result['results']:
            # Never turn a complete Python example into invalid partial code.
            for field in ('code_example', 'code', 'text'):
                if field in record:
                    record.pop(field)
                    record.setdefault('display_omitted_fields', []).append(field)
            excerpt = record.get('excerpt')
            if isinstance(excerpt, str) and len(excerpt) > 512:
                record['excerpt'] = excerpt[:512]
                record['next_offset'] = record.get('excerpt_offset', 0)+len(record['excerpt'])
                record['excerpt_truncated'] = True
            record['detail_hint'] = 'search_knowledge ids=['+json.dumps(record['id'])+'] for complete content/code.'
        while request_bytes(result) > maximum and len(result['results']) > 1:
            omitted = result['results'].pop()
            result.setdefault('omitted_result_ids', []).append(omitted['id'])
            result['omitted_results'] = result.get('omitted_results', 0)+1
    return result


def compact_research_request(request, limit, log=None, *, stage="request"):
    """Account for tool results and schemas without deleting the protected core or base."""
    def size():
        return len(json.dumps(request, ensure_ascii=False, allow_nan=False).encode('utf-8'))
    previous_size = size()
    omissions = []
    # Nested JSON is sent as message text. Its whitespace otherwise consumes
    # the same budget again after escaping in the outer request.
    for message in request['messages']:
        if message.get('role') not in {'user', 'tool'}:
            continue
        try:
            value = json.loads(message['content'])
        except (ValueError, TypeError):
            continue
        if isinstance(value, dict):
            message['content'] = json.dumps(value, ensure_ascii=False, separators=(',', ':'))
    for message in request['messages']:
        if message.get('role') != 'user':
            continue
        try:
            user = json.loads(message['content'])
        except (ValueError, TypeError):
            continue
        if not isinstance(user, dict) or not ('profile_summary' in user or 'profile' in user):
            continue
        profile = user.get('profile_summary', user)
        def refresh():
            message['content'] = json.dumps(user, ensure_ascii=False, separators=(',', ':'))
        data = profile.get('data', user.get('profile', {}))
        details = profile.get('detail_index', {})
        # Recent analyses used to copy up to six full 8 KiB queries into every
        # new C08 context. Their investigation prose is already frozen in details.
        if 'analyses' in details.get('sections', {}):
            for index, record in enumerate(profile.get('analysis_history', [])):
                if size() <= limit:
                    break
                query = record.get('query', {})
                if isinstance(query, dict) and 'investigation' in query:
                    query.pop('investigation')
                    record['investigation_details'] = {'reference_id': details['reference_id'],
                                                       'section': 'analyses/'+str(index)}
                    omissions.append('analysis_history_investigations');refresh()
        if 'research_feedback' in details.get('sections', {}):
            for index, record in enumerate(profile.get('research_feedback', [])):
                if size() <= limit:
                    break
                text = record.get('message')
                if isinstance(text, str) and len(text) > 320:
                    record['message'] = text[:320]
                    record['message_is_excerpt'] = True
                    record['detail_hint'] = 'read_memory section=research_feedback/'+str(index)
                    omissions.append('feedback_message_details');refresh()
        if profile.get('research_memory'):
            from agent.research_feedback import compact_memory
            profile['research_memory'] = compact_memory(profile['research_memory'],
                base_experiment_id=user.get('base_experiment_id'), max_bytes=8000)
            omissions.append('bounded_research_memory');refresh()
        if size() > limit and profile.get('scientific_protocol'):
            from agent.scientific_research import compact_protocol
            profile['scientific_protocol'] = compact_protocol(profile['scientific_protocol'])
            omissions.append('scientific_hints_with_scoped_details');refresh()
        # Full definitions/schema stay in the frozen snapshot; the compact
        # column table and input names must remain available to the planner.
        for field in ['data_dictionary', 'train_dtypes', 'test_dtypes', 'train_missing_fraction', 'test_missing_fraction']:
            if size() <= limit:
                break
            if field in data:
                data.pop(field);omissions.append(field);refresh()
        protected = {user.get('base_experiment_id'), *user.get('allowed_donor_experiment_ids', [])}
        protected.update(f['source_id'] for f in user.get('facts', [])[:3])
        while size() > limit and any(f.get('source_id') not in protected for f in user.get('facts', [])):
            pos = next(i for i in range(len(user['facts'])-1, -1, -1) if user['facts'][i].get('source_id') not in protected)
            fact = user['facts'].pop(pos)
            profile.setdefault('compacted_fact_source_ids', []).append(fact['source_id'])
            omissions.append('older_fact_details');refresh()
        memory = profile.get('measured_search', {})
        while size() > limit and len(memory.get('entries', [])) > 4:
            memory['entries'].pop();memory['omitted_count'] = memory.get('omitted_count', 0)+1;refresh()
        recent = user.get('recent_experiments', [])
        while size() > limit and len(recent) > 1:
            index = next((i for i in range(len(recent)-1, -1, -1) if not recent[i].get('is_incumbent')), None)
            if index is None:
                break
            recent.pop(index)
            user['omitted_recent_experiments'] = user.get('omitted_recent_experiments', 0)+1
            omissions.append('older_recent_experiments');refresh()
        archive = profile.get('feature_memory', {})
        while size() > limit and len(archive.get('representations', [])) > 1:
            archive['representations'].pop()
            archive['omitted_representation_count'] = archive.get('omitted_representation_count', 0)+1
            omissions.append('older_feature_program_details');refresh()
        comparisons = profile.get('comparison_evidence', {})
        while size() > limit and len(comparisons.get('comparisons', [])) > 1:
            comparisons['comparisons'].pop()
            comparisons['omitted_comparison_count'] = comparisons.get('omitted_comparison_count', 0)+1
            omissions.append('older_comparison_details');refresh()
        if size() > limit and profile.get('knowledge', {}).get('results'):
            profile['knowledge'] = compact_tool_reply(profile['knowledge'], 1800)
            omissions.append('seed_knowledge_details');refresh()
        if size() > limit and profile.get('detail_index', {}).get('sections', {}).get('recipes'):
            base_recipe = (user.get('base_result') or {}).get('recipe_id')
            for recipe in profile.get('recipes', []):
                if recipe.get('id') != base_recipe:
                    for field in ('hyperparameters', 'parameters', 'parameter_space', 'defaults'):
                        recipe.pop(field, None)
                    recipe['detail_hint'] = 'read_memory section=recipes for the full catalogue.'
            omissions.append('alternative_recipe_parameters');refresh()
        landscape = profile.get('research_landscape', {})
        if size() > limit and len(landscape.get('best_per_family', [])) > 1:
            landscape['omitted_family_details'] = len(landscape['best_per_family'])-1
            landscape['best_per_family'] = landscape['best_per_family'][:1]
            landscape['detail_hint'] = 'read_memory section=research_landscape for all measured families.'
            omissions.append('alternative_family_details');refresh()
        if size() > limit and 'facts' in details.get('sections', {}):
            # A protected fact ID is not a reason to duplicate its entire feature
            # program and parameters alongside the exact base and frozen plan.
            for fact in user.get('facts', []):
                if size() <= limit:
                    break
                try:
                    record = json.loads(fact['summary'])
                except (ValueError, TypeError, KeyError):
                    continue
                if not isinstance(record, dict) or not record.get('experiment_id'):
                    continue
                projection = {k: v for k, v in record.items() if k in {
                    'experiment_id', 'state', 'recipe_id', 'metric', 'seed', 'error', 'is_incumbent'}}
                projection['details_omitted'] = True
                projection['detail_hint'] = 'read_memory section=experiment/'+record['experiment_id']
                summary = json.dumps(projection, ensure_ascii=False, separators=(',', ':'))
                if len(summary.encode('utf-8')) < len(fact['summary'].encode('utf-8')):
                    fact['summary'] = summary
                    omissions.append('duplicate_fact_plan_details');refresh()
        if size() > limit and 'measurements' in details.get('sections', {}):
            for record in memory.get('entries', []):
                if size() <= limit:
                    break
                if request_bytes(record) <= 1200:
                    continue
                for field in ('hyperparameters', 'feature_steps', 'changes_from_base'):
                    record.pop(field, None)
                record['parameter_details_omitted'] = True
                record['detail_hint'] = 'read_memory section=experiment/'+record['experiment_id']
                omissions.append('historical_parameter_details');refresh()
        if size() > limit and profile.get('research_memory'):
            from agent.research_feedback import compact_memory
            profile['research_memory'] = compact_memory(profile['research_memory'],
                base_experiment_id=user.get('base_experiment_id'), max_bytes=4000)
            omissions.append('smaller_research_memory');refresh()
    # Preserve every assistant/tool-call pair. Compact all retrievable result
    # kinds, including the latest answer; keeping only the final report intact
    # could still make the complete request impossible to send.
    for message in request['messages']:
        if size() > limit and message.get('role') == 'assistant' and message.get('tool_calls') and message.get('content'):
            message['content'] = 'Earlier tool request; use its paired results and immutable output artifact.'
            omissions.append('older_assistant_tool_prose')
    tool_messages = [m for m in request['messages'] if m.get('role') == 'tool']
    for message in tool_messages:
        if size() <= limit:
            break
        try:
            reply = json.loads(message['content'])
        except (ValueError, TypeError):
            continue
        if not isinstance(reply, dict):
            continue
        allowance = max(2048, limit-size()+len(message['content'].encode('utf-8'))-256)
        candidate = json.dumps(compact_tool_reply(reply, allowance), ensure_ascii=False, separators=(',', ':'))
        if len(candidate.encode('utf-8')) < len(message['content'].encode('utf-8')):
            message['content'] = candidate
            omissions.append('retrievable_tool_details')
    result = {'previous_bytes': previous_size, 'request_bytes': size(), 'limit_bytes': limit,
              'omitted_fields': omissions, 'retained_tool_call_ids': [
                  m.get('tool_call_id') for m in request['messages'] if m.get('role') == 'tool']}
    if log is not None:
        log.event('context.compacted' if stage == 'assembly' else 'llm.context_compacted', **result)
    return result
