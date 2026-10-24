"""Source-grounded task model and short, executable research commitments.

This checks references and declared consistency, not the truth of LLM prose.
It has no dataset-specific names, model calls, SQL writes or learned state.
"""
from copy import deepcopy
import hashlib
import json

from agent.contracts import canonical
from agent.errors import AgentError


def text_schema(maximum=400):
    return {'type': 'string', 'minLength': 1, 'maxLength': maximum}


SOURCE_SCHEMA = {'type': 'object', 'properties': {
    'source': {'enum': ['task', 'dictionary', 'hypothesis']},
    'record': {'anyOf': [{'type': 'integer', 'minimum': 0}, {'type': 'null'}]},
    'quote': {'type': 'string', 'maxLength': 300}},
    'required': ['source', 'record', 'quote'], 'additionalProperties': False}
QUANTITY_KINDS = ['quantity', 'count', 'ratio', 'rate', 'category', 'ordinal_code',
                  'identifier', 'calendar', 'cyclic', 'unknown']
RELATION_KINDS = ['normalized_by', 'subset_count', 'part_of', 'rate_exposure',
                  'same_window', 'precedes', 'conditional_response', 'proxy', 'other']
QUANTITY_SCHEMA = {'type': 'object', 'properties': {
    'column': text_schema(200), 'meaning': text_schema(), 'kind': {'enum': QUANTITY_KINDS},
    'unit': {'anyOf': [text_schema(80), {'type': 'null'}]},
    'period': {'anyOf': [{'type': 'number', 'exclusiveMinimum': 0}, {'type': 'null'}]},
    'availability': {'enum': ['available', 'unavailable', 'unknown']},
    'source': SOURCE_SCHEMA},
    'required': ['column', 'meaning', 'kind', 'unit', 'period', 'availability', 'source'],
    'additionalProperties': False}
RELATION_SCHEMA = {'type': 'object', 'properties': {
    'kind': {'enum': RELATION_KINDS},
    'columns': {'type': 'array', 'minItems': 2, 'maxItems': 6, 'items': text_schema(200)},
    'statement': text_schema(),
    'assumptions': {'type': 'array', 'maxItems': 4, 'items': text_schema(200)},
    'source': SOURCE_SCHEMA},
    'required': ['kind', 'columns', 'statement', 'assumptions', 'source'], 'additionalProperties': False}
TASK_MODEL_SCHEMA = {'type': 'object', 'properties': {
    'context': {'type': 'object', 'properties': {k: text_schema(400) for k in (
        'entity', 'observation_grain', 'prediction_time', 'target_meaning', 'target_normalization')},
        'required': ['entity', 'observation_grain', 'prediction_time', 'target_meaning', 'target_normalization'],
        'additionalProperties': False},
    'context_sources': {'type': 'array', 'maxItems': 4, 'items': SOURCE_SCHEMA},
    'quantities': {'type': 'array', 'maxItems': 32, 'items': QUANTITY_SCHEMA},
    'relations': {'type': 'array', 'maxItems': 16, 'items': RELATION_SCHEMA},
    'unknowns': {'type': 'array', 'maxItems': 8, 'items': text_schema(240)}},
    'required': ['context', 'context_sources', 'quantities', 'relations', 'unknowns'],
    'additionalProperties': False}
INVESTIGATION_SCHEMA = {'type': 'object', 'properties': {
    'question': text_schema(400),
    'basis_ids': {'type': 'array', 'maxItems': 8, 'items': text_schema(200)},
    'columns': {'type': 'array', 'maxItems': 16, 'items': text_schema(200)},
    'interpretation': {'enum': ['physical', 'mathematical', 'statistical', 'exploratory', 'model_behavior', 'data_quality']},
    'mechanism': text_schema(500),
    'assumptions': {'type': 'array', 'maxItems': 4, 'items': text_schema(200)},
    'expected_observation': text_schema(400),
    'alternative_explanation': text_schema(400),
    'decision_rule': text_schema(500)},
    'required': ['question', 'basis_ids', 'columns', 'interpretation', 'mechanism', 'assumptions',
                 'expected_observation', 'alternative_explanation', 'decision_rule'],
    'additionalProperties': False}


def unknown_model():
    """Legacy/missing metadata is explicit ignorance, never inferred physics."""
    return {'context': {k: 'unknown' for k in TASK_MODEL_SCHEMA['properties']['context']['properties']},
            'context_sources': [], 'quantities': [], 'relations': [],
            'unknowns': ['No source-grounded semantic model was supplied; read task/dictionary before asserting meaning.']}


def build_task_model(raw, task_text, profile, schemas, spec):
    raw = deepcopy(raw if raw is not None else unknown_model())
    schemas.validate(raw, 'TaskModel')
    if len(canonical(raw).encode()) > 32768:
        raise AgentError('INVALID_TASK_MODEL', 'Semantic model exceeds 32 KiB; select relevant quantities/relations')
    names = set(profile['train_columns']) | set(profile['test_columns'])
    dictionary = profile.get('data_dictionary', [])
    def source_status(source):
        kind, index, quote = source['source'], source['record'], source['quote']
        if kind == 'hypothesis':
            if index is not None or quote:
                raise AgentError('INVALID_TASK_MODEL', 'Hypotheses have record=null and quote=""')
            return 'unsupported_hypothesis'
        if not quote.strip():
            raise AgentError('INVALID_TASK_MODEL', 'A source reference needs a verbatim quotation')
        if kind == 'task':
            body = task_text if index is None else ''
        else:
            body = (canonical(dictionary[index]) if index is not None and index < len(dictionary) else '')
            # Also accept exact text inside a CSV cell without JSON escaping.
            if index is not None and index < len(dictionary):
                body += '\n'+'\n'.join(str(v) for v in dictionary[index].values())
        if quote not in body:
            raise AgentError('INVALID_TASK_MODEL', 'Quotation does not occur in the declared task/dictionary source')
        return 'quotation_matched; interpretation_unverified'
    context_status = [source_status(s) for s in raw['context_sources']]
    if not context_status and any(value.casefold() != 'unknown' for value in raw['context'].values()):
        raise AgentError('INVALID_TASK_MODEL', 'Known task context needs source quotations or explicit hypothesis provenance')
    seen = set()
    for quantity in raw['quantities']:
        if quantity['column'] not in names or quantity['column'] in seen:
            raise AgentError('INVALID_TASK_MODEL', 'Quantities must name distinct actual columns')
        seen.add(quantity['column'])
        if quantity['period'] is not None and quantity['kind'] != 'cyclic':
            raise AgentError('INVALID_TASK_MODEL', 'Only a cyclic quantity can declare a period')
        if quantity['kind'] in {'ordinal_code', 'identifier', 'category', 'unknown'} and quantity['unit'] is not None:
            raise AgentError('INVALID_TASK_MODEL', 'Codes/categories/unknown scales cannot declare a physical unit')
        if quantity['source']['source'] == 'hypothesis' and (quantity['unit'] is not None or quantity['period'] is not None):
            raise AgentError('INVALID_TASK_MODEL', 'Unsourced units/periods remain unknown; declare exploratory assumptions in an investigation')
        quantity['source_verification'] = source_status(quantity['source'])
    for relation in raw['relations']:
        if not set(relation['columns']) <= names or len(set(relation['columns'])) != len(relation['columns']):
            raise AgentError('INVALID_TASK_MODEL', 'Relations need distinct actual columns')
        if relation['kind'] in {'normalized_by', 'subset_count', 'precedes'} and len(relation['columns']) != 2:
            raise AgentError('INVALID_TASK_MODEL', 'This relation has exactly two ordered operands')
        relation['source_verification'] = source_status(relation['source'])
    identity = hashlib.sha256(canonical([spec['task_spec_id'], spec['dataset_id'], raw]).encode()).hexdigest()[:24]
    model_id = 'task-model-'+identity
    for collection in ('quantities', 'relations'):
        for item in raw[collection]:
            item['record_id'] = model_id+'-'+hashlib.sha256(canonical(item).encode()).hexdigest()[:16]
    return {**raw, 'model_id': model_id, 'version': 'semantic-planning-1',
            'task_spec_id': spec['task_spec_id'], 'dataset_id': spec['dataset_id'],
            'target_columns': spec.get('target_columns', [spec['target_column']]),
            'context_source_verification': context_status,
            'verification': 'model_interpretation_unverified; source quotations checked separately',
            'predictive_usefulness': 'not_established'}


def load_task_model(spec, artifacts):
    ident = spec.get('semantic_model_artifact_id')
    if not ident:
        return None
    from agent.artifacts import file_sha256
    if file_sha256(artifacts.resolve(ident)) != artifacts.meta(ident)['sha256']:
        raise AgentError('MEMORY_SOURCE_CHANGED', 'Semantic task model changed after bootstrap')
    model = artifacts.read_json(ident)
    if any(model.get(k) != spec.get(k) for k in ('task_spec_id', 'dataset_id')):
        raise AgentError('MEMORY_SCOPE_DENIED', 'Semantic task model belongs to another task/dataset')
    if model['model_id'] != spec.get('semantic_model_id'):
        raise AgentError('MEMORY_SCOPE_DENIED', 'Semantic task model identity mismatch')
    return model


def semantic_view(model, memory, base=None, maximum=10000):
    """Retrieve a small subgraph by actual prior question/column dependencies."""
    if model is None:
        return None
    decisions = memory.get('decisions', [])
    latest = next((r.get('investigation') for r in reversed(decisions) if r.get('investigation')), {})
    focus = set(latest.get('columns', [])) | set(model.get('target_columns', []))
    basis = set(latest.get('basis_ids', []))
    if base:
        for step in base.get('feature_steps', []):
            for op in step.get('program', []):
                focus.update(op.get('columns', []))
    records = [r for r in memory.get('semantic_records', []) if r.get('status') != 'superseded']
    dynamic_relations = [{'record_id': r['record_id'], 'kind': r['content']['properties']['relation_kind'],
        'columns': r['content']['columns'], 'statement': r['content']['statement'],
        'status': r['status'], 'source_verification': r['verification']}
        for r in records if r['content'].get('properties', {}).get('relation_kind')]
    relations = sorted([*dynamic_relations, *model['relations']], key=lambda r: (
        r['record_id'] in basis, len(set(r['columns']) & focus)), reverse=True)[:4]
    focus.update(c for r in relations for c in r['columns'])
    effective_quantities = effective_model(model, records)['quantities']
    quantities = sorted(effective_quantities, key=lambda q: (
        q['column'] in model.get('target_columns', []), q['column'] in focus), reverse=True)[:8]
    recent = []
    for row in decisions[-3:]:
        if row.get('investigation'):
            observations = row.get('observations', [])
            recent.append({'decision_artifact_id': row['decision_artifact_id'],
                'investigation': deepcopy(row['investigation']),
                'observations': deepcopy(observations[-2:]),
                'outcome': ('measured_configuration; mechanism_unresolved'
                    if any(o.get('primary_metric') is not None for o in observations) else
                    'execution_failed; hypothesis_untested'
                    if observations and all(o.get('state') in {'failed', 'invalid', 'timed_out', 'cancelled'} for o in observations)
                    else 'pending_or_unmeasured')})
    view = {'model_id': model['model_id'], 'verification': model['verification'],
        'context': deepcopy(model['context']), 'quantities': deepcopy(quantities),
        'relations': deepcopy(relations), 'unknowns': deepcopy(model['unknowns']),
        'recent_investigations': recent,
        'suggested_checks': relation_checks(relations, model.get('target_columns', [])),
        'omitted_quantities': len(effective_quantities)-len(quantities),
        'omitted_relations': len(model['relations'])+len(dynamic_relations)-len(relations),
        'omitted_unknowns': 0,
        'details': 'read_memory: semantic_model, quantities/index, relations/index. '
            'Model interpretations remain unverified. Missing quantities have unknown meaning/units. '
            'Use scoped semantic_records for later corrections/conflicts; original definitions are immutable.'}
    # Preserve the task definition ahead of historical explanatory prose.
    for field in ('recent_investigations', 'suggested_checks', 'relations', 'quantities', 'unknowns'):
        while len(canonical(view).encode()) > maximum and view[field]:
            view[field].pop()
            if field in {'relations', 'quantities'}:
                view['omitted_'+field] += 1
            elif field == 'unknowns':
                view['omitted_unknowns'] += 1
    if len(canonical(view).encode()) > maximum:
        raise AgentError('CONTEXT_TOO_LARGE', 'Semantic task definition exceeds its protected context allowance')
    return view


def relation_checks(relations, targets=()):
    """Generic logical consequences of declared relations, never automatic fits."""
    checks = []
    for relation in relations:
        if relation.get('status') == 'conflicted' or len(relation['columns']) != 2:
            continue
        left, right = relation['columns']
        if relation['kind'] == 'subset_count':
            check = {'name': 'subset_exceeds_superset', 'violation': {'column': left, 'op': 'gt', 'other_column': right}}
            meaning = 'The proposed subset cannot exceed its superset when observation windows/definitions agree.'
        elif relation['kind'] == 'normalized_by':
            check = {'name': 'zero_normalization_denominator', 'violation': {'column': right, 'op': 'eq', 'value': 0}}
            meaning = 'Inspect denominator validity and target normalization before applying another division.'
        else:
            continue
        inputs = [check['violation']['column']]
        if check['violation'].get('other_column'):
            inputs.append(check['violation']['other_column'])
        if set(inputs) & set(targets):
            continue
        checks.append({'basis_id': relation['record_id'], 'meaning': meaning,
            'query': {'kind': 'check', 'columns': inputs, 'group_by': None,
                      'include_target': False, 'checks': [check]},
            'status': 'advisory consequence of an unverified interpretation; attach investigation before execution'})
    return checks[:2]


def effective_model(model, records):
    """Apply scoped typed declarations; conflicts remove asserted certainty.

    The immutable source model remains in read_memory. Revisions retain their
    own IDs and unverified status; a score never promotes an interpretation.
    """
    if model is None:
        return None
    result = deepcopy(model)
    quantities = {q['column']: q for q in result['quantities']}
    revisions = {}
    for record in records:
        content = record['content']
        props = content.get('properties', {})
        if record.get('status') == 'superseded' or len(content['columns']) != 1:
            continue
        if not any(k in props for k in ('quantity_kind', 'units', 'availability_status', 'period')):
            continue
        name = content['columns'][0]
        quantity = quantities.setdefault(name, {'column': name, 'kind': 'unknown', 'unit': None,
            'period': None, 'availability': 'unknown', 'meaning': content['statement']})
        quantity['record_id'] = record['record_id']
        quantity['source_verification'] = record['verification']
        quantity['revision_status'] = record['status']
        if record['status'] == 'conflicted':
            quantity.update(kind='unknown', unit=None, period=None, availability='unknown')
            revisions.setdefault(name, {})['conflicted'] = True
        else:
            for source, destination in (('quantity_kind', 'kind'), ('units', 'unit'),
                                        ('availability_status', 'availability'), ('period', 'period')):
                if source in props:
                    value = None if source == 'units' and props[source].casefold() == 'unknown' else props[source]
                    values = revisions.setdefault(name, {}).setdefault(destination, [])
                    values.append(value)
                    quantity[destination] = value if all(v == values[0] for v in values) else ('unknown' if destination in {'kind', 'availability'} else None)
    for name, revision in revisions.items():
        if revision.get('conflicted'):
            quantities[name].update(kind='unknown', unit=None, period=None, availability='unknown', revision_status='conflicted')
    result['quantities'] = list(quantities.values())
    return result


def validate_investigation(investigation, context, evidence, columns, schemas, *, program=()):
    metadata = context.get('metadata', {}) if context else {}
    if investigation is None:
        if metadata.get('semantic_contract'):
            raise AgentError('INVALID_INVESTIGATION', 'Supply a short investigation with a discriminating observation and decision_rule')
        return {'verification': 'legacy_no_investigation', 'warnings': []}
    schemas.validate(investigation, 'Investigation')
    for field in ('basis_ids', 'columns'):
        if len(set(investigation[field])) != len(investigation[field]):
            raise AgentError('INVALID_INVESTIGATION', field+' must not repeat IDs/columns')
    user = json.loads(context['messages'][1]['content']) if context else {}
    projection = user.get('profile_summary', user)
    view = projection.get('semantic_task', {})
    delivered = set(evidence) | set(metadata.get('delivered_evidence_ids', []))
    # These fit-only records are already shown in the issued scientific context.
    # Accept the same sources for analysis as Researcher.propose accepts for plans.
    delivered.update(r['artifact_id'] for r in projection.get('scientific_protocol', {}).get('fit_diagnostics', []))
    delivered.update(q['record_id'] for field in ('quantities', 'relations') for q in view.get(field, []))
    delivered.update(q['record_id'] for q in projection.get('research_memory', {}).get('semantic_records', []))
    if view.get('model_id'):
        delivered.add(view['model_id'])
    if not set(investigation['basis_ids']) <= delivered:
        raise AgentError('INVALID_INVESTIGATION', 'Read the source/semantic record before citing its ID')
    if columns is not None and not set(investigation['columns']) <= set(columns):
        raise AgentError('INVALID_INVESTIGATION', 'Investigation references absent columns')
    if investigation['interpretation'] in {'physical', 'mathematical'} and not investigation['basis_ids'] and not investigation['assumptions']:
        raise AgentError('INVALID_INVESTIGATION', 'Declare the source basis or explicit assumptions of a physical/mathematical interpretation')
    generated, dependencies = set(), set()
    for step in program:
        dependencies.update(c for c in [*step['columns'], *([step['group_by']] if step.get('group_by') else [])] if c not in generated)
        generated.add(step['name'])
    if not dependencies <= set(investigation['columns']):
        raise AgentError('INVALID_INVESTIGATION', 'Named feature inputs must be included in investigation.columns')
    return {'verification': 'references_and_dependency_coverage_checked; mechanism_unverified',
            'basis_ids': list(investigation['basis_ids']), 'columns': sorted(dependencies), 'warnings': []}


def check_declared_semantics(program, investigation, model):
    """Reject contradictions to *declared* meanings, never certify physical laws.

    Statistical/exploratory representations may have composite units. Only a
    physical/mathematical claim invokes unit/code arithmetic consistency checks.
    """
    if model is None or investigation is None:
        return []
    quantities = {q['column']: deepcopy(q) for q in model['quantities']}
    strict = investigation['interpretation'] in {'physical', 'mathematical'}
    warnings = []
    for step in program:
        inputs = [quantities.get(c, {'kind': 'unknown', 'unit': None, 'availability': 'unknown'}) for c in step['columns']]
        dependencies = inputs + ([quantities.get(step['group_by'], {'availability': 'unknown'})] if step.get('group_by') else [])
        if any(q['availability'] == 'unavailable' for q in dependencies):
            raise AgentError('INVALID_INVESTIGATION', 'Feature input contradicts declared prediction-time availability; resolve the source first')
        arithmetic = step['op'] in {'sum', 'difference', 'product', 'ratio', 'square', 'log1p', 'signed_log1p'}
        if strict and arithmetic and any(q['kind'] in {'identifier', 'ordinal_code', 'category'} for q in inputs):
            raise AgentError('INVALID_INVESTIGATION', 'Physical/mathematical arithmetic uses a declared code/category/identifier; use a justified mapping or an explicit exploratory interpretation')
        units = [q['unit'] for q in inputs]
        if strict and step['op'] in {'sum', 'difference'} and all(units) and len(set(units)) > 1:
            raise AgentError('INVALID_INVESTIGATION', 'Addition/subtraction has incompatible declared units; make the conversion explicit')
        if strict and step['op'] in {'log1p', 'signed_log1p'} and units[0] not in {None, 'dimensionless'}:
            raise AgentError('INVALID_INVESTIGATION', 'A physical logarithm needs an explicit dimensionless reference scale')
        for column, unit in step.get('input_units', {}).items():
            known = quantities.get(column, {}).get('unit')
            if known is not None and known != unit:
                raise AgentError('INVALID_INVESTIGATION', 'Feature unit contradicts the task model declaration')
            if known is None:
                warnings.append('Unverified unit declaration for '+column)
        if step['op'] in {'cyclic_sin', 'cyclic_cos'}:
            period = inputs[0].get('period')
            if period is not None and period != step['period']:
                raise AgentError('INVALID_INVESTIGATION', 'Cyclic period contradicts the task model declaration')
            if period is None:
                warnings.append('Cyclic period is an assumption for '+step['columns'][0])
        if any(q['kind'] == 'unknown' for q in inputs):
            warnings.append('Unknown input semantics for '+step['name'])
        output_unit = (units[0] if step['op'] in {'sum', 'difference', 'group_mean', 'group_std'} and all(units) and len(set(units)) == 1 else
                       'dimensionless' if step['op'] in {'missing', 'frequency', 'group_rank', 'relative_to_group', 'cyclic_sin', 'cyclic_cos'} else None)
        if all(units) and step['op'] in {'product', 'square', 'ratio'}:
            if step['op'] == 'ratio':
                output_unit = 'dimensionless' if units[0] == units[1] else '('+units[0]+')/('+units[1]+')'
            else:
                factors = units*2 if step['op'] == 'square' else units
                output_unit = '*'.join('('+u+')' for u in sorted(factors) if u != 'dimensionless') or 'dimensionless'
        if 'output_unit' in step:
            if output_unit is not None and step['output_unit'] != output_unit:
                raise AgentError('INVALID_INVESTIGATION', 'Output unit contradicts the operator-derived declaration; omit it or use the explicit symbolic unit')
            if output_unit is None:
                warnings.append('Unverified output unit for '+step['name'])
        quantities[step['name']] = {'kind': 'quantity', 'unit': output_unit, 'availability': 'available'}
    return list(dict.fromkeys(warnings))
