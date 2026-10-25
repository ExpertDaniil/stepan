"""Dispatcher-frozen context documents. No SQL/arbitrary artifact reads in tools."""
from copy import deepcopy
import re
import hashlib
import uuid

from agent.contracts import canonical
from agent.errors import AgentError

READ_SCHEMA = {'type': 'object', 'properties': {
    'reference_id': {'type': 'string', 'minLength': 1},
    'section': {'type': 'string', 'minLength': 1, 'maxLength': 80},
    'offset': {'type': 'integer', 'minimum': 0},
    'limit_bytes': {'type': 'integer', 'minimum': 256, 'maximum': 6000}},
    'required': ['reference_id', 'section'], 'additionalProperties': False}

SOLUTION_SCHEMA = {'type': 'object', 'properties': {
    'experiment_id': {'type': 'string', 'minLength': 1},
    'view': {'enum': ['pipeline', 'code', 'result', 'features', 'diagnostics']},
    'offset': {'type': 'integer', 'minimum': 0},
    'limit_bytes': {'type': 'integer', 'minimum': 256, 'maximum': 6000}},
    'required': ['experiment_id'], 'additionalProperties': False}


def solution_record(store, artifacts, experiment_id):
    """Dispatcher-only snapshot; the reader never consults live SQLite."""
    exp = store.get(experiment_id)
    ev = store.experiment_evaluation(experiment_id)
    trusted = ev and ev['valid'] and not ev['violations'] and exp['state'] == 'evaluated'
    docs = {'code': deepcopy(artifacts.meta(exp['plan']['code_artifact_id']))}
    if trusted:
        docs['diagnostics'] = deepcopy(artifacts.meta(ev['diagnostics_artifact_id']))
    return {'experiment_id': experiment_id, 'state': exp['state'], 'plan': deepcopy(exp['plan']),
        'metric': deepcopy(ev['primary_metric']) if trusted else None,
        'execution_error': deepcopy(exp.get('error')), 'documents': docs}


def read_solution(query, context, artifacts, log):
    """One experiment ID, no reference IDs or document-section routing by the LLM."""
    metadata = context['metadata']
    frozen = metadata.get('detail_scope')
    if not frozen:
        raise AgentError('MEMORY_SCOPE_DENIED', 'No scoped solution snapshot is available')
    catalog = {}
    # Newest delivered snapshots override old state; never open a sibling's ID.
    for aid in reversed(metadata.get('authorized_feedback_ids', [])):
        update = artifacts.read_json(aid)
        if update.get('branch_id') != (frozen.get('branch_id') or 'legacy') or any(
                update.get(k) != frozen.get(k) for k in ('dataset_id', 'task_spec_id', 'validation_id')):
            raise AgentError('MEMORY_SCOPE_DENIED', 'Solution feedback outside this task/branch/validation')
        for eid, record in update.get('solution_catalog', {}).items():
            catalog.setdefault(eid, record)
        requested = query['experiment_id']
        if requested in catalog or (requested == 'best' and (update.get('best_solution') or {}).get('experiment_id') in catalog):
            break
    snapshot = artifacts.read_json(metadata['detail_artifact_id'])
    if snapshot['scope'] != frozen or frozen['run_id'] != artifacts.run_id:
        raise AgentError('MEMORY_SCOPE_DENIED', 'Solution snapshot scope mismatch')
    for section, record in snapshot['sections'].items():
        if section.startswith('experiment/'):
            catalog.setdefault(record['experiment_id'], record)
    eid = query['experiment_id']
    if eid == 'best':
        measured = [v for v in catalog.values() if v.get('metric')]
        if not measured:
            raise AgentError('SOLUTION_UNAVAILABLE', 'No measured solution in this scope')
        eid = max(measured, key=lambda v: v['metric']['value']*(1 if v['metric']['direction'] == 'maximize' else -1))['experiment_id']
    if eid not in catalog:
        raise AgentError('MEMORY_SCOPE_DENIED', 'Experiment is not visible in this task/branch snapshot',
                         details={'available_experiment_ids': list(catalog)[:16], 'hint': 'Use experiment_id=best for the scoped incumbent.'})
    record = catalog[eid]
    plan, metric = record['plan'], record.get('metric')
    if any(plan.get(k) != frozen.get(k) for k in ('dataset_id', 'task_spec_id', 'validation_id')):
        raise AgentError('MEMORY_SCOPE_DENIED', 'Solution plan identity differs from the issued task/validation')
    view = query.get('view', 'pipeline')
    if view not in SOLUTION_SCHEMA['properties']['view']['enum']:
        raise AgentError('INVALID_MEMORY_READ', 'Unknown solution view')
    def document(name):
        from agent.artifacts import file_sha256
        meta = record.get('documents', {}).get(name)
        if meta is None:
            # Compatibility with context documents produced before this reader.
            meta = snapshot.get('documents', {}).get(name+'/'+eid)
        if meta is None:
            if name == 'code':
                raise AgentError('SOLUTION_UNAVAILABLE', 'Source code was not retained in this scoped snapshot')
            return None
        path = artifacts.resolve(meta['artifact_id'])
        if file_sha256(path) != meta['sha256']:
            raise AgentError('MEMORY_SOURCE_CHANGED', 'Immutable solution document changed on disk')
        return artifacts.read_json(meta['artifact_id']) if name == 'diagnostics' else path.read_text(encoding='utf-8')
    if view == 'code':
        value = document('code')
    elif view == 'diagnostics':
        value = document('diagnostics')
    elif view == 'features':
        value = {'feature_steps': plan['feature_steps'],
                 'feature_program': plan['model']['hyperparameters'].get('feature_program')}
    elif view == 'result':
        value = {k: record.get(k) for k in ('experiment_id', 'state', 'metric', 'execution_error')}
    else:
        value = {'experiment_id': eid, 'state': record['state'], 'metric': metric,
            'recipe_id': plan['recipe_id'], 'hyperparameters': plan['model']['hyperparameters'],
            'feature_steps': plan['feature_steps'], 'code': document('code'),
            'seed': plan['seed'], 'execution_error': record.get('execution_error')}
    body = (value if view == 'code' and isinstance(value, str) else canonical(value)).encode('utf-8')
    offset, maximum = query.get('offset', 0), query.get('limit_bytes', 6000)
    if (isinstance(offset, bool) or not isinstance(offset, int) or offset < 0 or offset > len(body)
            or (offset < len(body) and body[offset] & 0xC0 == 0x80)
            or isinstance(maximum, bool) or not isinstance(maximum, int) or not 256 <= maximum <= 6000):
        raise AgentError('INVALID_MEMORY_READ', 'Invalid page bounds; follow next_offset')
    fragment = body[offset:offset+maximum].decode('utf-8', errors='ignore')
    end = offset+len(fragment.encode('utf-8'))
    log.event('research.solution_read', experiment_id=eid, view=view, offset_bytes=offset,
              returned_bytes=end-offset, scope=frozen)
    return {'experiment_id': eid, 'view': view, 'reference_id': metadata['detail_artifact_id'],
        'section': 'solution/'+eid+'/'+view, 'scope': frozen, 'offset': offset,
        'next_offset': end if end < len(body) else None, 'total_bytes': len(body), 'text': fragment,
        'evidence_ids': [eid] if offset == 0 and end == len(body) else [], 'idea_ids': [],
        'next_read': ({'experiment_id': eid, 'view': view, 'offset': end, 'limit_bytes': maximum}
                      if end < len(body) else None)}


def scope_for(spec, artifacts):
    return {'run_id': artifacts.run_id, 'task_spec_id': spec.get('task_spec_id'),
            'dataset_id': spec.get('dataset_id'),
            'train_sha256': artifacts.meta(spec['train_artifact_id'])['sha256'],
            'test_sha256': artifacts.meta(spec['test_artifact_id'])['sha256']}


def references(value):
    ids, ideas = set(), set()
    def walk(item):
        if isinstance(item, dict):
            for k, v in item.items():
                if k == 'idea_id' and isinstance(v, str):
                    ideas.add(v)
                if k in {'experiment_id', 'evaluation_id', 'decision_artifact_id', 'artifact_id',
                         'diagnostics_artifact_id', 'record_id', 'model_id'} and isinstance(v, str):
                    ids.add(v)
                if k == 'evidence_ids' and isinstance(v, list):
                    ids.update(v)
                walk(v)
        elif isinstance(item, list):
            for child in item:
                walk(child)
    walk(value)
    return sorted(ids), sorted(ideas)


def read_details(query, context, artifacts, log):
    """Bounded UTF-8 fragments, including for a single oversized record."""
    metadata = context['metadata']
    if set(query)-{'reference_id', 'section', 'offset', 'limit_bytes'}:
        raise AgentError('INVALID_MEMORY_READ', 'Unknown read_memory arguments')
    feedback_read = query['reference_id'] in metadata.get('authorized_feedback_ids', [])
    if query['reference_id'] != metadata.get('detail_artifact_id') and not feedback_read:
        raise AgentError('MEMORY_SCOPE_DENIED', 'reference_id is not authorized by this frozen context')
    snapshot = artifacts.read_json(query['reference_id'])
    if feedback_read:
        frozen = metadata['detail_scope']
        if (snapshot.get('branch_id') != (frozen.get('branch_id') or 'legacy')
                or any(snapshot.get(k) != frozen.get(k) for k in ('dataset_id', 'task_spec_id', 'validation_id'))):
            raise AgentError('MEMORY_SCOPE_DENIED', 'Feedback outside this task/branch/validation')
        if artifacts.meta(query['reference_id'])['kind'] == 'research_transcript':
            from agent.artifacts import file_sha256
            from agent.context_transcript import transcript_sections
            if snapshot.get('scope') != frozen:
                raise AgentError('MEMORY_SCOPE_DENIED', 'Transcript belongs to another issued context')
            if file_sha256(artifacts.resolve(query['reference_id'])) != artifacts.meta(query['reference_id'])['sha256']:
                raise AgentError('MEMORY_SOURCE_CHANGED', 'Immutable transcript changed on disk')
            snapshot = {'scope': frozen, 'sections': transcript_sections(snapshot)}
        else:
            snapshot = {'scope': frozen, 'sections': {'updates': snapshot, 'results': snapshot.get('results', []),
                **{'results/'+str(i): row for i, row in enumerate(snapshot.get('results', []))}}}
    if snapshot['scope'] != metadata.get('detail_scope') or snapshot['scope']['run_id'] != artifacts.run_id:
        raise AgentError('MEMORY_SCOPE_DENIED', 'Task/dataset/validation/context scope mismatch')
    section = query['section']
    documents = snapshot.get('documents', {})
    document_key = section if section in documents else next((key for key, doc in documents.items()
        if doc['format'] == 'transcript' and section.startswith(key+'/')), None)
    if document_key is not None:
        from agent.artifacts import file_sha256
        doc = documents[document_key]
        if file_sha256(artifacts.resolve(doc['artifact_id'])) != doc['sha256']:
            raise AgentError('MEMORY_SOURCE_CHANGED', 'Immutable context source changed on disk')
        if doc['format'] == 'transcript':
            from agent.context_transcript import transcript_sections
            views = transcript_sections(artifacts.read_json(doc['artifact_id']))
            for record in views['transcript_index']:
                for key in ('result_section', 'arguments_section'):
                    record[key] = document_key+'/'+record[key]
            snapshot['sections'][document_key] = views['transcript_index']
            snapshot['sections'].update({document_key+'/'+key: value for key, value in views.items()})
        else:
            snapshot['sections'][section] = (artifacts.read_json(doc['artifact_id']) if doc['format'] == 'json'
                else artifacts.resolve(doc['artifact_id']).read_text(encoding='utf-8'))
    if section == 'detail_index':
        snapshot['sections'][section] = {'reference_id': query['reference_id'],
            'sections': sorted(set(snapshot['sections']) | set(snapshot.get('documents', {}))),
            'solution_tool': 'read_solution(experiment_id="best", view="pipeline")'}
    if section not in snapshot['sections']:
        raise AgentError('INVALID_MEMORY_READ', 'Unknown section for this document', details={
            'reference_id': query['reference_id'], 'available_sections': sorted(k for k in snapshot['sections'] if '/' not in k),
            'context_reference_id': metadata.get('detail_artifact_id'),
            'hint': 'For experiment code/results use read_solution with only experiment_id; for available sections read detail_index.'})
    offset, maximum = query.get('offset', 0), query.get('limit_bytes', 4000)
    if (isinstance(offset, bool) or not isinstance(offset, int) or offset < 0
            or isinstance(maximum, bool) or not isinstance(maximum, int) or not 256 <= maximum <= 6000):
        raise AgentError('INVALID_MEMORY_READ', 'Invalid page bounds')
    value = snapshot['sections'][section]
    body = canonical(value).encode('utf-8')
    if offset > len(body) or (offset < len(body) and body[offset] & 0xC0 == 0x80):
        raise AgentError('INVALID_MEMORY_READ', 'Use next_offset, a UTF-8 byte boundary')
    fragment = body[offset:offset+maximum].decode('utf-8', errors='ignore')
    end = offset+len(fragment.encode('utf-8'))
    # Only a complete record read authorizes citations; fragments authorize none.
    evidence, ideas = references(value) if offset == 0 and end == len(body) else ([], [])
    log.event('research.memory_read', reference_id=query['reference_id'], section=section,
              offset_bytes=offset, returned_bytes=end-offset, scope=snapshot['scope'])
    return {'reference_id': query['reference_id'], 'section': section, 'scope': snapshot['scope'],
            'offset': offset, 'next_offset': end if end < len(body) else None,
            'total_bytes': len(body), 'text': fragment, 'evidence_ids': evidence, 'idea_ids': ideas}


def prepare_details(user, spec, artifacts, log, *, validation_id=None, validation_sha256=None, visibility=None, store=None, experiment_ids=()):
    profile = user.get('profile_summary', user)
    data = profile.get('data', user.get('profile', {}))
    memory = profile.get('research_memory', {})
    task_text = profile.get('task_text', user.get('task', ''))
    from agent.semantic_planning import load_task_model, semantic_view
    model = load_task_model(spec, artifacts)
    if model is not None:
        profile['semantic_task'] = semantic_view(model, memory, user.get('base_result'))
    scope = {**scope_for(spec, artifacts), 'validation_id': validation_id,
             'validation_sha256': validation_sha256,
             'task_text_sha256': hashlib.sha256(task_text.encode('utf-8')).hexdigest(),
             'branch_id': user.get('branch_id'), 'work_order_id': user.get('work_order_id'),
             'visibility': deepcopy(visibility),
             'allowed_donor_experiment_ids': deepcopy(user.get('allowed_donor_experiment_ids', []))}
    from agent.config import ROOT
    sections = {'base_code': user.get('base_code'), 'base_result': deepcopy(user.get('base_result')),
        'full_context': deepcopy(user), 'task_description': task_text, 'tool_guide': (ROOT/'prompts/TOOLS.md').read_text(encoding='utf-8'),
        'decisions': memory.get('decisions', []), 'ideas': memory.get('open_ideas', []),
        'semantics': memory.get('semantic_records', []), 'dictionary': data.get('data_dictionary', []),
        'schema': {k: deepcopy(data.get(k)) for k in ('train_columns', 'test_columns', 'train_dtypes', 'test_dtypes')},
        'facts': user.get('facts', []), 'measurements': profile.get('measured_search', user.get('recent_experiments', [])),
        'knowledge': profile.get('knowledge', {}),
        'representations': profile.get('feature_memory', {}), 'comparisons': profile.get('comparison_evidence', {}),
        'analyses': profile.get('analysis_history', user.get('analysis_history', [])),
        'recipes': deepcopy(profile.get('recipes', [])),
        'research_landscape': deepcopy(profile.get('research_landscape', {})),
        'research_feedback': deepcopy(profile.get('research_feedback', []))}
    if profile.get('scientific_protocol'):
        sections['scientific_protocol'] = deepcopy(profile['scientific_protocol'])
        sections['fit_diagnostics'] = deepcopy(profile['scientific_protocol'].get('fit_diagnostics', []))
        for index, record in enumerate(sections['fit_diagnostics']):
            sections['fit_diagnostics/'+str(index)] = record
    observation = profile.get('error_research', {})
    if observation.get('source_artifact_id'):
        from agent.artifacts import file_sha256
        aid = observation['source_artifact_id']
        if file_sha256(artifacts.resolve(aid)) != artifacts.meta(aid)['sha256']:
            raise AgentError('MEMORY_SOURCE_CHANGED', 'Frozen diagnostic report changed')
        sections['error_observation'] = {'source_experiment_id': observation['source_experiment_id'],
            'source_artifact_id': aid, 'report': artifacts.read_json(aid)}
    if model is not None:
        sections.update(semantic_model=model, quantities=model['quantities'], relations=model['relations'])
    sections['analyses'] = [{**r, 'report': artifacts.read_json(r['artifact_id'])}
                            for r in sections['analyses']]
    analysis_index = []
    for record in sections['analyses']:
        aid, report = record['artifact_id'], record['report']
        query = report.pop('query', {})
        code = query.pop('code', None)
        if code is not None:
            sections['analysis_code/'+aid] = code
        record.get('query', {}).pop('code', None)
        question = query.get('question') or query.get('investigation', {}).get('question', '')
        sections['analysis_result/'+aid] = {'artifact_id': aid, 'question': question,
            'report': deepcopy(report)}
        summary = {'artifact_id': aid, 'question': question[:300],
            'success': report.get('success'), 'result_section': 'analysis_result/'+aid,
            'result_preview': canonical(report.get('result', report.get('error', {})))[:700],
            'preview_only': True}
        if code is not None:
            summary['code_section'] = 'analysis_code/'+aid
        analysis_index.append(summary)
    sections['analyses_index'] = analysis_index
    documents = {}
    transcripts = []
    from agent.context_transcript import visible_transcripts
    for meta, decision_id, question in visible_transcripts(memory, artifacts, scope):
        section = 'transcript/'+meta['artifact_id']
        documents[section] = {**meta, 'format': 'transcript'}
        transcripts.append({'section': section, 'decision_artifact_id': decision_id,
            'question': question, 'preview_only': True})
    sections['transcripts_index'] = transcripts
    sections['drafts_index'] = []
    visible_orders = set(memory.get('visible_work_order_ids', []))
    cutoffs = memory.get('visible_analysis_cutoffs', {})
    with artifacts.lock:
        drafts = [deepcopy(entry['meta']) for entry in artifacts.entries.values() if entry['meta']['kind'] == 'pipeline_draft']
    for meta in drafts:
        value = artifacts.read_json(meta['artifact_id'])
        source = value.get('scope', {})
        owner = source.get('work_order_id')
        if (owner not in visible_orders or any(source.get(k) != scope.get(k) for k in
                ('run_id','task_spec_id','dataset_id','validation_id','branch_id'))
                or value.get('completed_elapsed_ms', 0) > cutoffs.get(owner, float('inf'))):
            continue
        section = 'draft/'+meta['artifact_id']
        documents[section] = {**meta, 'format':'json'}
        sections['drafts_index'].append({'draft_id':meta['artifact_id'], 'section':section,
            'bytes':len(value['code'].encode()), 'instruction':'Use edit_solution(draft_id=...) or pipeline code_ref.'})
    sections['experiments'] = list(dict.fromkeys(experiment_ids))
    if store is not None:
        for eid in sections['experiments']:
            exp = store.get(eid)
            sections['experiment/'+eid] = solution_record(store, artifacts, eid)
            meta = artifacts.meta(exp['plan']['code_artifact_id'])
            documents['code/'+eid] = {'artifact_id': meta['artifact_id'], 'sha256': meta['sha256'], 'format': 'text'}
            ev = store.experiment_evaluation(eid)
            if ev and ev['valid'] and not ev['violations']:
                meta = artifacts.meta(ev['diagnostics_artifact_id'])
                documents['diagnostics/'+eid] = {'artifact_id': meta['artifact_id'], 'sha256': meta['sha256'], 'format': 'json'}
                diagnostic = artifacts.read_json(meta['artifact_id'])
                if diagnostic.get('feature_execution_artifact_id'):
                    meta = artifacts.meta(diagnostic['feature_execution_artifact_id'])
                    documents['feature_execution/'+eid] = {'artifact_id': meta['artifact_id'], 'sha256': meta['sha256'], 'format': 'json'}
            with artifacts.lock:
                execution = artifacts.entries.get('execution-'+eid)
                if execution:
                    meta = execution['meta']
                    documents['execution/'+eid] = {'artifact_id': meta['artifact_id'], 'sha256': meta['sha256'], 'format': 'json'}
    measured = [r for k, r in sections.items() if k.startswith('experiment/') and r.get('metric')]
    if measured:
        best = max(measured, key=lambda r: r['metric']['value']*(1 if r['metric']['direction'] == 'maximize' else -1))
        profile['research_incumbent'] = {'experiment_id': best['experiment_id'], 'metric': best['metric'],
            'recipe_id': best['plan']['recipe_id'], 'read': {'experiment_id': best['experiment_id'], 'view': 'pipeline'},
            'instruction': 'Strongest visible measured solution. The assignment base records graph lineage; improve this solution or a justified alternative.'}
    for name in ('decisions', 'ideas', 'semantics', 'dictionary', 'facts', 'analyses', 'quantities', 'relations', 'research_feedback'):
        for index, item in enumerate(sections.get(name, [])):
            sections[name+'/'+str(index)] = item
    art = artifacts.write_json('research/context-details-'+uuid.uuid4().hex[:16]+'.json',
                               {'scope': scope, 'sections': sections, 'documents': documents}, 'research_context_details')
    profile['source_ids'] = {'task_spec_id': spec.get('task_spec_id'),
        'dictionary_artifact_id': spec.get('dictionary_artifact_id')}
    profile['detail_index'] = {'reference_id': art['artifact_id'], 'tool': 'read_memory',
        'sections': {k: {'bytes': len(canonical(v).encode()), 'records': len(v) if isinstance(v, list) else None}
                     for k, v in sections.items() if '/' not in k},
        'experiment_details': 'Read experiments then experiment/ID, code/ID, diagnostics/ID, feature_execution/ID or execution/ID; only scoped IDs exist.',
        'record_pages': 'List sections support section/index, e.g. decisions/0. offset is UTF-8 bytes; follow next_offset.'}
    text = canonical([user.get('base_result'), memory.get('decisions', [])[-1:]])
    words = set(re.findall(r'[\w]+', text.casefold()))
    dictionary = sections['dictionary']
    ranked = sorted(enumerate(dictionary), key=lambda x: (
        -len(words & set(re.findall(r'[\w]+', canonical(x[1]).casefold()))), x[0]))
    data['relevant_definitions'] = [{'source': 'dictionary', 'source_record': i,
        'verification': 'verbatim_source_excerpt',
        'definition': {k: v[:600] if isinstance(v, str) else v for k, v in row.items()},
        'details_truncated': any(isinstance(v, str) and len(v) > 600 for v in row.values())}
        for i, row in ranked[:6]]
    data['dictionary_index'] = {'total_records': len(dictionary), 'section': 'dictionary',
        'reference_id': art['artifact_id'], 'unknown_units': 'Remain unknown; names/numeric dtype do not establish units.'}
    for field in ('data_dictionary', 'train_dtypes', 'test_dtypes', 'train_missing_fraction', 'test_missing_fraction'):
        data.pop(field, None)
    # Remove only literal Markdown table rows naming input columns. Preserve
    # every narrative constraint and every target row; do not invent a summary.
    from agent.predictions import targets
    inputs = set(data.get('train_columns', []))-set(targets(spec))
    lines, omitted_rows = [], 0
    for line in task_text.splitlines():
        first = line.strip().strip('|').split('|', 1)[0].strip().strip('`')
        if line.strip().startswith('|') and first in inputs:
            omitted_rows += 1
            continue
        lines.append(line)
    if omitted_rows:
        text = '\n'.join(lines)+'\nInput-feature table rows are in read_memory section=task_description and dictionary; relevant definitions appear above.'
        if 'task_text' in profile:
            profile['task_text'] = text
        else:
            user['task'] = text
        data['omitted_task_feature_table_rows'] = omitted_rows
    for result in profile.get('knowledge', {}).get('results', []):
        for key in list(result):
            if key not in {'id', 'kind', 'title', 'tags', 'status', 'excerpt', 'warnings', 'origin'}:
                result.pop(key)
        if isinstance(result.get('excerpt'), str):
            result['excerpt'] = result['excerpt'][:1000]
        result['detail_hint'] = 'Read section=knowledge or search_knowledge by id for complete content/code.'
    for representation in profile.get('feature_memory', {}).get('representations', []):
        representation.pop('parameters', None)
        representation.pop('feature_steps', None)
        specification = representation.pop('execution_specification', None)
        if specification:
            representation['program_id'] = specification['program_id']
        representation['details'] = 'read_memory section=experiment/experiment_id or representations'
    measured = profile.get('measured_search', {})
    if isinstance(measured, dict) and len(measured.get('entries', [])) > 8:
        measured['omitted_count'] = measured.get('omitted_count', 0)+len(measured['entries'])-8
        measured['entries'] = measured['entries'][:8]
    from agent.research_feedback import compact_memory
    profile['research_memory'] = compact_memory(memory, base_experiment_id=user.get('base_experiment_id'))
    log.event('research.memory_selected', reference_id=art['artifact_id'],
        retained_decisions=[d['decision_artifact_id'] for d in profile['research_memory'].get('decisions', [])],
        selection_reason='base_dependency_then_question_overlap_then_recency', scope=scope)
    if model is not None:
        view = profile['semantic_task']
        log.event('research.semantic_context_selected', model_id=model['model_id'],
            quantity_ids=[q['record_id'] for q in view['quantities']], relation_ids=[r['record_id'] for r in view['relations']],
            bytes=len(canonical(view).encode()), selection_reason='previous_question_columns_and_named_program_dependencies', scope=scope)
    return {'detail_artifact_id': art['artifact_id'], 'detail_scope': scope,
            **({'semantic_contract': 'semantic-planning-1'} if model is not None else {})}
