"""Explicit research decisions, persisted separately from the frozen ExperimentPlan."""
from agent.errors import AgentError

CUSTOM_RECIPE = 'python-pipeline-v1'
TUNING_SCHEMA = {
    'type': 'object', 'properties': {
        'parameters': {'type': 'array', 'items': {'type': 'string', 'minLength': 1}, 'minItems': 1, 'maxItems': 12},
        'max_trials': {'type': 'integer', 'minimum': 2, 'maximum': 16},
        'budget_seconds': {'type': 'integer', 'minimum': 1, 'maximum': 3600}},
    'required': ['parameters', 'max_trials', 'budget_seconds'], 'additionalProperties': False}
ACTION_SCHEMA = {
    'type': 'object', 'properties': {
        'action': {'enum': ['catalog', 'pipeline', 'tune', 'stop']},
        'recipe_id': {'type': 'string', 'minLength': 1},
        'hyperparameters': {'type': 'object'},
        'hypothesis': {'type': 'string', 'minLength': 1},
        'expected_effect': {'type': 'string', 'minLength': 1},
        'feature_steps': {'type': 'array', 'items': {'type': 'object'}},
        'code': {'anyOf': [{'type': 'string', 'minLength': 1}, {'type': 'null'}]},
        'tuning': {'anyOf': [TUNING_SCHEMA, {'type': 'null'}]},
        'evidence_ids': {'type': 'array', 'items': {'type': 'string', 'minLength': 1}, 'maxItems': 12},
        'knowledge_ids': {'type': 'array', 'items': {'type': 'string', 'minLength': 1}, 'maxItems': 12}},
    'required': ['action', 'recipe_id', 'hyperparameters', 'hypothesis', 'expected_effect',
                 'feature_steps', 'code', 'tuning', 'evidence_ids'], 'additionalProperties': False}


def validate_action(action, catalog):
    kind = action['action']
    if kind == 'stop':
        if action['code'] is not None or action['tuning'] is not None:
            raise AgentError('INVALID_RESEARCH_ACTION', 'Stop cannot execute code or start tuning')
        return
    if kind == 'pipeline':
        if action['recipe_id'] != CUSTOM_RECIPE or not action['code'] or action['tuning'] is not None:
            raise AgentError('INVALID_RESEARCH_ACTION', 'A custom pipeline needs explicit source and the custom recipe ID')
    elif action['code'] is not None or action['recipe_id'] == CUSTOM_RECIPE:
        raise AgentError('INVALID_RESEARCH_ACTION', 'Catalogue/tuning actions execute only the owned constructor')
    if (kind == 'tune') != (action['tuning'] is not None):
        raise AgentError('INVALID_RESEARCH_ACTION', 'Only tune can authorize a local series')
    if kind == 'tune':
        domains = catalog.recipes.get(action['recipe_id'], {}).get('parameters', {})
        axes = action['tuning']['parameters']
        if len(set(axes)) != len(axes) or not set(axes) <= set(domains):
            raise AgentError('INVALID_RESEARCH_ACTION', 'Tuning axes must be distinct declared parameters')
