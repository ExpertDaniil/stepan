"""Explicit research decisions, persisted separately from the frozen ExperimentPlan."""
from agent.errors import AgentError
from agent.feature_schema import FEATURE_PROGRAM_SCHEMA
from agent.semantic_planning import INVESTIGATION_SCHEMA

CUSTOM_RECIPE = 'python-pipeline-v1'
IDEA_SCHEMA = {
    'type': 'object', 'properties': {
        'idea_id': {'anyOf': [{'type': 'string', 'minLength': 1}, {'type': 'null'}]},
        'question': {'type': 'string', 'minLength': 1, 'maxLength': 1000},
        'mechanism': {'type': 'string', 'minLength': 1, 'maxLength': 1000},
        'next_test': {'type': 'string', 'minLength': 1, 'maxLength': 1000},
        'why_deferred': {'type': 'string', 'minLength': 1, 'maxLength': 1000},
        'evidence_ids': {'type': 'array', 'items': {'type': 'string', 'minLength': 1}, 'maxItems': 12},
        'knowledge_ids': {'type': 'array', 'items': {'type': 'string', 'minLength': 1}, 'maxItems': 12}},
    'required': ['idea_id', 'question', 'mechanism', 'next_test', 'why_deferred', 'evidence_ids', 'knowledge_ids'],
    'additionalProperties': False}
RESEARCH_NOTES_SCHEMA = {
    'type': 'object', 'properties': {
        'test_plan': {'type': 'object', 'properties': {
            **{key: {'type': 'string', 'minLength': 12, 'maxLength': 1000} for key in
               ('observation', 'prediction', 'falsifier', 'cheap_check', 'counterargument')},
            'evidence_ids': {'type': 'array', 'items': {'type': 'string', 'minLength': 1}, 'maxItems': 12},
            'continuation': {'enum': ['new', 'followup', 'reconsider', 'stop']},
            'switch_reason': {'type': 'string', 'maxLength': 1000},
            'alternatives_unavailable_reason': {'type': 'string', 'maxLength': 1000}},
            'required': ['observation', 'prediction', 'falsifier', 'cheap_check', 'counterargument',
                         'evidence_ids', 'continuation', 'switch_reason', 'alternatives_unavailable_reason'],
            'additionalProperties': False},
        'idea_id': {'anyOf': [{'type': 'string', 'minLength': 1}, {'type': 'null'}]},
        'question': {'type': 'string', 'minLength': 1, 'maxLength': 1000},
        'mechanism': {'type': 'string', 'minLength': 1, 'maxLength': 1000},
        'decision_reason': {'type': 'string', 'minLength': 1, 'maxLength': 2000},
        'previous_result_interpretation': {'type': 'string', 'maxLength': 2000},
        'alternatives': {'type': 'array', 'items': IDEA_SCHEMA, 'maxItems': 8}},
    'required': ['idea_id', 'question', 'mechanism', 'decision_reason', 'previous_result_interpretation', 'alternatives'],
    'additionalProperties': False}
SEMANTIC_UPDATE_SCHEMA = {'type': 'array', 'maxItems': 6, 'items': {
    'type': 'object', 'properties': {
        'topic': {'type': 'string', 'minLength': 1, 'maxLength': 100},
        'kind': {'enum': ['definition', 'observation', 'hypothesis', 'data_error', 'implementation_error', 'execution_error']},
        'statement': {'type': 'string', 'minLength': 1, 'maxLength': 600},
        'columns': {'type': 'array', 'maxItems': 12, 'items': {'type': 'string', 'minLength': 1}},
        'evidence_ids': {'type': 'array', 'maxItems': 8, 'items': {'type': 'string', 'minLength': 1}},
        'supersedes': {'anyOf': [{'type': 'string', 'minLength': 1}, {'type': 'null'}]},
        'properties': {'type': 'object', 'properties': {
            'meaning': {'type': 'string', 'maxLength': 400},
            'units': {'type': 'string', 'maxLength': 100},
            'scale': {'type': 'string', 'maxLength': 100},
            'entity': {'type': 'string', 'maxLength': 100},
            'grain': {'type': 'string', 'maxLength': 100},
            'availability': {'type': 'string', 'maxLength': 200},
            'normalization': {'type': 'string', 'maxLength': 200},
            'relationships': {'type': 'string', 'maxLength': 400},
            'quantity_kind': {'enum': ['quantity', 'count', 'ratio', 'rate', 'category', 'ordinal_code', 'identifier', 'calendar', 'cyclic', 'unknown']},
            'availability_status': {'enum': ['available', 'unavailable', 'unknown']},
            'period': {'type': 'number', 'exclusiveMinimum': 0},
            'relation_kind': {'enum': ['normalized_by', 'subset_count', 'part_of', 'rate_exposure', 'same_window', 'precedes', 'conditional_response', 'proxy', 'other']},
            'unknowns': {'type': 'string', 'maxLength': 400},
            'operation_id': {'type': 'string', 'maxLength': 100}}, 'additionalProperties': False}},
    'required': ['topic', 'kind', 'statement', 'columns', 'evidence_ids'], 'additionalProperties': False}}
DISTRIBUTION_SCHEMA = {'anyOf': [
    {'type': 'object', 'properties': {'type': {'const': 'int'}, 'low': {'type': 'integer'}, 'high': {'type': 'integer'},
        'step': {'type': 'integer', 'minimum': 1}, 'log': {'type': 'boolean'}},
     'required': ['type', 'low', 'high'], 'additionalProperties': False},
    {'type': 'object', 'properties': {'type': {'const': 'float'}, 'low': {'type': 'number'}, 'high': {'type': 'number'},
        'step': {'type': 'number', 'exclusiveMinimum': 0}, 'log': {'type': 'boolean'}},
     'required': ['type', 'low', 'high'], 'additionalProperties': False},
    {'type': 'object', 'properties': {'type': {'const': 'categorical'}, 'choices': {'type': 'array', 'minItems': 1, 'maxItems': 64,
        'items': {'anyOf': [{'type': 'string'}, {'type': 'number'}, {'type': 'boolean'}, {'type': 'null'}]}}},
     'required': ['type', 'choices'], 'additionalProperties': False}]}
TUNING_SCHEMA = {
    'type': 'object', 'properties': {
        'parameters': {'type': 'array', 'items': {'type': 'string', 'minLength': 1}, 'minItems': 1, 'maxItems': 12},
        'search_space': {'type': 'object', 'additionalProperties': DISTRIBUTION_SCHEMA},
        'max_trials': {'type': 'integer', 'minimum': 2, 'maximum': 16},
        'budget_seconds': {'type': 'integer', 'minimum': 1, 'maximum': 3600}},
    'required': ['max_trials', 'budget_seconds'], 'additionalProperties': False}
ACTION_SCHEMA = {
    'type': 'object', 'properties': {
        'action': {'enum': ['research', 'catalog', 'pipeline', 'features', 'tune', 'stop']},
        'feature_program': FEATURE_PROGRAM_SCHEMA,
        'next_step': {'enum': ['continue', 'wait', 'finish']},
        'coordination': {'type': 'object', 'properties': {
            'related_series_ids': {'type': 'array', 'items': {'type': 'string', 'minLength': 1}, 'maxItems': 4},
            'reason': {'type': 'string', 'minLength': 12, 'maxLength': 1000}},
            'required': ['related_series_ids', 'reason'], 'additionalProperties': False},
        'claims': {'type': 'array', 'maxItems': 6, 'items': {
            'type': 'object', 'properties': {
                'scope': {'enum': ['hypothesis', 'configuration', 'component', 'family']},
                'statement': {'type': 'string', 'minLength': 1, 'maxLength': 1000},
                'evidence_ids': {'type': 'array', 'items': {'type': 'string', 'minLength': 1}, 'maxItems': 4},
                'component': {'anyOf': [{'type': 'string', 'minLength': 1}, {'type': 'null'}]}},
            'required': ['scope', 'statement', 'evidence_ids', 'component'], 'additionalProperties': False}},
        'recipe_id': {'type': 'string', 'minLength': 1},
        'hyperparameters': {'type': 'object'},
        'hypothesis': {'type': 'string', 'minLength': 1},
        'expected_effect': {'type': 'string', 'minLength': 1},
        'feature_steps': {'type': 'array', 'items': {'type': 'object'}},
        'code': {'anyOf': [{'type': 'string', 'minLength': 1}, {'type': 'null'}]},
        'tuning': {'anyOf': [TUNING_SCHEMA, {'type': 'null'}]},
        'evidence_ids': {'type': 'array', 'items': {'type': 'string', 'minLength': 1}, 'maxItems': 12},
        'knowledge_ids': {'type': 'array', 'items': {'type': 'string', 'minLength': 1}, 'maxItems': 12},
        'research_notes': RESEARCH_NOTES_SCHEMA, 'semantic_updates': SEMANTIC_UPDATE_SCHEMA,
        'investigation': INVESTIGATION_SCHEMA},
    'required': ['action', 'hypothesis', 'evidence_ids'], 'additionalProperties': False}


def validate_research_notes(notes, memory, evidence, knowledge):
    """Check reference provenance, not the truth of model-authored interpretations."""
    if notes is None:
        return
    if not set(notes.get('test_plan', {}).get('evidence_ids', [])) <= evidence:
        raise AgentError('INVALID_RESEARCH_ACTION', 'Test plan evidence must reference visible completed evidence')
    visible = {idea['idea_id'] for idea in memory.get('open_ideas', [])}
    for idea in [notes, *notes['alternatives']]:
        if idea['idea_id'] is not None and idea['idea_id'] not in visible:
            raise AgentError('INVALID_RESEARCH_ACTION', 'Research notes reference an unseen idea_id')
        if not set(idea.get('evidence_ids', [])) <= evidence:
            raise AgentError('INVALID_RESEARCH_ACTION', 'Alternative evidence_ids must reference visible measured results/analyses')
        if not set(idea.get('knowledge_ids', [])) <= knowledge:
            raise AgentError('INVALID_RESEARCH_ACTION', 'Alternative knowledge_ids must reference content actually read')


def validate_test_plan(action, protocol):
    """New contexts require an explicit, falsifiable record; old artifacts stay readable.

    This validates structure/provenance, never the scientific truth of LLM prose.
    """
    if protocol.get('version') != 'scientific-v1':
        return
    notes = action.get('research_notes')
    if not notes or not notes.get('test_plan'):
        raise AgentError('RESEARCH_DESIGN_REQUIRED', 'Provide research_notes and test_plan: observation, prediction, falsifier, cheap_check, counterargument and continuation. Use actual evidence IDs; no private reasoning is requested.')
    test = notes['test_plan']
    if test['prediction'].casefold().strip() == test['falsifier'].casefold().strip():
        raise AgentError('INVALID_RESEARCH_DESIGN', 'Prediction and falsifier must describe different observable outcomes')
    episode = protocol.get('episode')
    if test['continuation'] == 'followup' and (not episode or notes['idea_id'] != episode['idea_id']):
        raise AgentError('INVALID_RESEARCH_DESIGN', 'A followup must reuse the assigned episode idea_id')
    if test['continuation'] == 'new' and episode and len(test['switch_reason'].strip()) < 12:
        raise AgentError('INVALID_RESEARCH_DESIGN', 'Explain why switching from the existing research question is useful')
    if test['continuation'] == 'reconsider' and len(test['switch_reason'].strip()) < 12:
        raise AgentError('INVALID_RESEARCH_DESIGN', 'Reconsideration needs an explicit cost/new-evidence reason')
    if action['action'] != 'stop' and not notes['alternatives'] and len(test['alternatives_unavailable_reason'].strip()) < 12:
        raise AgentError('INVALID_RESEARCH_DESIGN', 'Compare a concrete alternative explanation, or state why none is useful; do not invent alternatives')


def validate_action(action, catalog):
    kind = action['action']
    if kind in {'catalog', 'tune'} and catalog.recipes.get(action['recipe_id'], {}).get('family') == 'dummy':
        raise AgentError('REFERENCE_ONLY_RECIPE', 'Dummy is the emergency mean/majority reference, not a research/tuning direction. Choose a learned model, features, pipeline or stop.')
    if kind == 'features':
        if (action['code'] is not None or action['tuning'] is not None
                or action['recipe_id'] == CUSTOM_RECIPE or action['recipe_id'].startswith('dummy-')
                or not action.get('feature_program') or action['feature_steps']):
            raise AgentError('INVALID_RESEARCH_ACTION', 'features needs an owned learned recipe, feature_program, code/tuning=null and feature_steps=[]')
    elif 'feature_program' in action:
        raise AgentError('INVALID_RESEARCH_ACTION', 'feature_program belongs only to features actions')
    if kind in {'stop', 'research'}:
        if action['code'] is not None or action['tuning'] is not None:
            raise AgentError('INVALID_RESEARCH_ACTION', 'Stop cannot execute code or start tuning')
        return
    if kind == 'pipeline':
        if action['recipe_id'] != CUSTOM_RECIPE or not action['code'] or action['tuning'] is not None:
            raise AgentError('INVALID_RESEARCH_ACTION', 'A custom pipeline needs explicit source and the custom recipe ID')
    elif kind == 'tune' and action['recipe_id'] == CUSTOM_RECIPE:
        if not action['tuning'] or not action['tuning'].get('search_space'):
            raise AgentError('INVALID_RESEARCH_ACTION', 'Custom tuning needs an explicit typed search_space')
    elif action['code'] is not None or action['recipe_id'] == CUSTOM_RECIPE:
        raise AgentError('INVALID_RESEARCH_ACTION', 'Catalogue/tuning actions execute only the owned constructor')
    if (kind == 'tune') != (action['tuning'] is not None):
        raise AgentError('INVALID_RESEARCH_ACTION', 'Only tune can authorize a local series')
    if kind == 'tune':
        request = action['tuning']
        axes = request.get('parameters', list(request.get('search_space', {})))
        domains = catalog.recipes.get(action['recipe_id'], {}).get('parameters', {})
        if not axes or len(set(axes)) != len(axes) or (action['recipe_id'] != CUSTOM_RECIPE and not set(axes) <= set(domains)):
            raise AgentError('INVALID_RESEARCH_ACTION', 'Tuning axes must be distinct declared parameters; supply search_space')


def normalize_action(action):
    action = dict(action)
    for key, value in {'recipe_id': 'none', 'hyperparameters': {}, 'feature_steps': [],
                       'code': None, 'tuning': None, 'expected_effect': action.get('hypothesis', '')}.items():
        action.setdefault(key, value)
    return action
