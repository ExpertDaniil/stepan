"""Explicit research decisions, persisted separately from the frozen ExperimentPlan."""
from agent.errors import AgentError
from agent.feature_schema import FEATURE_PROGRAM_SCHEMA

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
        'idea_id': {'anyOf': [{'type': 'string', 'minLength': 1}, {'type': 'null'}]},
        'question': {'type': 'string', 'minLength': 1, 'maxLength': 1000},
        'mechanism': {'type': 'string', 'minLength': 1, 'maxLength': 1000},
        'decision_reason': {'type': 'string', 'minLength': 1, 'maxLength': 2000},
        'previous_result_interpretation': {'type': 'string', 'maxLength': 2000},
        'alternatives': {'type': 'array', 'items': IDEA_SCHEMA, 'maxItems': 8}},
    'required': ['idea_id', 'question', 'mechanism', 'decision_reason', 'previous_result_interpretation', 'alternatives'],
    'additionalProperties': False}
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
        'action': {'enum': ['catalog', 'pipeline', 'features', 'tune', 'stop']},
        'feature_program': FEATURE_PROGRAM_SCHEMA,
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
        'research_notes': RESEARCH_NOTES_SCHEMA},
    'required': ['action', 'recipe_id', 'hyperparameters', 'hypothesis', 'expected_effect',
                 'feature_steps', 'code', 'tuning', 'evidence_ids'], 'additionalProperties': False}


def validate_research_notes(notes, memory, evidence, knowledge):
    """Check reference provenance, not the truth of model-authored interpretations."""
    if notes is None:
        return
    visible = {idea['idea_id'] for idea in memory.get('open_ideas', [])}
    for idea in [notes, *notes['alternatives']]:
        if idea['idea_id'] is not None and idea['idea_id'] not in visible:
            raise AgentError('INVALID_RESEARCH_ACTION', 'Research notes reference an unseen idea_id')
        if not set(idea.get('evidence_ids', [])) <= evidence:
            raise AgentError('INVALID_RESEARCH_ACTION', 'Alternative evidence_ids must reference visible measured results/analyses')
        if not set(idea.get('knowledge_ids', [])) <= knowledge:
            raise AgentError('INVALID_RESEARCH_ACTION', 'Alternative knowledge_ids must reference content actually read')


def validate_action(action, catalog):
    kind = action['action']
    if kind == 'features':
        if (action['code'] is not None or action['tuning'] is not None
                or action['recipe_id'] == CUSTOM_RECIPE or action['recipe_id'].startswith('dummy-')
                or not action.get('feature_program') or action['feature_steps']):
            raise AgentError('INVALID_RESEARCH_ACTION', 'features needs an owned learned recipe, feature_program, code/tuning=null and feature_steps=[]')
    elif 'feature_program' in action:
        raise AgentError('INVALID_RESEARCH_ACTION', 'feature_program belongs only to features actions')
    if kind == 'stop':
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
