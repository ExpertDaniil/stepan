"""Scoped source drafts. Exact replacements avoid resending large pipelines."""
from copy import deepcopy
import hashlib
import time
import uuid

from agent.errors import AgentError

EDIT_SCHEMA = {'type':'object', 'properties': {
    'experiment_id': {'type':'string', 'minLength':1},
    'draft_id': {'type':'string', 'minLength':1},
    'code': {'type':'string', 'minLength':1},
    'edits': {'type':'array', 'maxItems':16, 'items': {'type':'object', 'properties': {
        'old':{'type':'string','minLength':1}, 'new':{'type':'string'}},
        'required':['old','new'], 'additionalProperties':False}}}, 'additionalProperties':False}


def visible_drafts(context, artifacts):
    scope = context['metadata'].get('detail_scope', {})
    snapshot_id = context['metadata'].get('detail_artifact_id')
    ids = set(context['metadata'].get('authorized_draft_ids', []))
    if snapshot_id:
        snapshot = artifacts.read_json(snapshot_id)
        if snapshot.get('scope') != scope:raise AgentError('MEMORY_SCOPE_DENIED','Draft scope differs from context')
        ids.update(r['draft_id'] for r in snapshot['sections'].get('drafts_index', []))
    return ids


def draft_code(ident, context, artifacts):
    from agent.artifacts import file_sha256
    if ident not in visible_drafts(context, artifacts):
        raise AgentError('MEMORY_SCOPE_DENIED','Draft is outside this issued context; use edit_solution with a visible experiment ID')
    meta = artifacts.meta(ident)
    if meta['kind'] != 'pipeline_draft' or file_sha256(artifacts.resolve(ident)) != meta['sha256']:
        raise AgentError('MEMORY_SOURCE_CHANGED','Draft source changed')
    value = artifacts.read_json(ident)
    frozen = context['metadata']['detail_scope']
    if any(value['scope'].get(k) != frozen.get(k) for k in ('run_id','dataset_id','task_spec_id','validation_id','branch_id')):
        raise AgentError('MEMORY_SCOPE_DENIED','Draft belongs to another task/branch/protocol')
    return value['code']


def edit_solution(query, context, artifacts, log):
    if sum(key in query for key in ('experiment_id','draft_id','code')) != 1:
        raise AgentError('INVALID_SOURCE_EDIT','Choose exactly one source: experiment_id, draft_id, or code for a new draft')
    if 'code' in query:source = query['code']
    elif 'draft_id' in query:source = draft_code(query['draft_id'], context, artifacts)
    else:
        from agent.context_details import read_solution
        request = {'experiment_id':query['experiment_id'], 'view':'code'}
        fragments = []
        while request:
            page = read_solution(request, context, artifacts, log)
            fragments.append(page['text']);request = page['next_read']
        source = ''.join(fragments)
    for edit in query.get('edits', []):
        count = source.count(edit['old'])
        if count != 1:
            raise AgentError('AMBIGUOUS_SOURCE_EDIT',f'Exact old text matched {count} times; include enough surrounding code for one match')
        source = source.replace(edit['old'],edit['new'],1)
    value = {'scope':deepcopy(context['metadata']['detail_scope']), 'code':source,
        'parent_reference':query.get('draft_id',query.get('experiment_id')),
        'completed_elapsed_ms':max(0, int((time.monotonic()-context['_run_started_monotonic'])*1000))
            if '_run_started_monotonic' in context else context.get('_elapsed_ms',0)}
    art = artifacts.write_json('research/draft-'+uuid.uuid4().hex[:16]+'.json',value,'pipeline_draft')
    context['metadata'].setdefault('authorized_draft_ids',[]).append(art['artifact_id'])
    validation = None
    try:
        from tools.code import validate_code
        validate_code(source)
    except (AgentError,ValueError,SyntaxError) as error:validation = str(error)[:600]
    log.event('research.pipeline_draft_saved', draft_id=art['artifact_id'], bytes=len(source.encode()), validation_error=validation)
    return {'draft_id':art['artifact_id'], 'code_sha256':hashlib.sha256(source.encode()).hexdigest(),
        'bytes':len(source.encode()), 'validation_error':validation,
        'instruction':'Edit this draft_id again, or submit action=pipeline with code_ref set to this draft_id. Full code need not be resent.'}


def resolve_candidate(action, context, artifacts):
    from agent.pipeline_candidates import normalize_candidate
    value = deepcopy(action)
    ref = value.pop('code_ref',None)
    if ref is not None:
        if value.get('code') or value.get('action') != 'pipeline':
            raise AgentError('INVALID_SOURCE_EDIT','code_ref is for pipeline actions; omit code when using it')
        value['code'] = draft_code(ref,context,artifacts)
    return normalize_candidate(value)
