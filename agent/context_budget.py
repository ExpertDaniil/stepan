"""One complete JSON-body budget. Bytes are measured; tokens are not.

Without a proxy tokenizer, one input byte per estimated token is an explicit
conservative safety proxy. Provider tokenization/framing remains authoritative.
"""
import json


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
