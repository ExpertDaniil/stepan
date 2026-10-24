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
