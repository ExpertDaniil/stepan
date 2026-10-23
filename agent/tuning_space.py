"""Validate declarative optimizer domains without importing or executing Optuna."""
from copy import deepcopy
import math

from agent.contracts import canonical
from agent.errors import AgentError


def normalize_space(catalog, plan, request, constraints=None):
    recipe = plan['recipe_id']
    custom = catalog.recipes[recipe]['family'] == 'custom'
    params = catalog.normalize(recipe, plan['model']['hyperparameters'])
    declared = request.get('search_space')
    names = request.get('parameters')
    if names is not None and (len(names) != len(set(names)) or not names):
        raise AgentError('INVALID_TUNING_SPACE', 'parameters must contain distinct names')
    if declared is None:
        if custom or not names:
            raise AgentError('INVALID_TUNING_SPACE', 'Provide a nonempty typed search_space; custom parameters have no inferred domains')
        declared = {}
        for name in names:
            domain = catalog.recipes[recipe]['parameters'].get(name)
            if domain is None:
                raise AgentError('INVALID_TUNING_SPACE', 'Unknown tuning parameter: '+name)
            declared[name] = ({'type': 'categorical', 'choices': deepcopy(domain['enum'])} if 'enum' in domain else
                              {'type': domain['type'], 'low': domain['min'], 'high': domain['max'], 'log': False})
    if not isinstance(declared, dict) or not 1 <= len(declared) <= 12:
        raise AgentError('INVALID_TUNING_SPACE', 'search_space needs 1..12 typed parameters')
    if names is not None and set(names) != set(declared):
        raise AgentError('INVALID_TUNING_SPACE', 'parameters and search_space must name the same axes')
    space = {}
    for name, raw in sorted(declared.items()):
        if not isinstance(name, str) or not name or name.startswith('_') or name not in params:
            raise AgentError('INVALID_TUNING_SPACE', 'Every axis must be a nonreserved parameter in the anchor: '+str(name))
        kind = raw.get('type') if isinstance(raw, dict) else None
        if kind == 'categorical':
            choices = raw.get('choices', [])
            if set(raw) != {'type', 'choices'} or not isinstance(choices, list) or not choices:
                raise AgentError('INVALID_TUNING_SPACE', 'Categorical domain needs choices: '+name)
            for i, value in enumerate(choices):
                if not isinstance(value, (str, bool, int, float, type(None))) or any(value == old for old in choices[:i]):
                    raise AgentError('INVALID_TUNING_SPACE', 'Categorical choices must be distinct finite JSON scalars: '+name)
            values = choices
            domain = {'type': kind, 'choices': deepcopy(choices)}
        elif kind in {'int', 'float'}:
            if set(raw)-{'type', 'low', 'high', 'step', 'log'} or not {'low', 'high'} <= set(raw):
                raise AgentError('INVALID_TUNING_SPACE', 'Numeric domain needs low/high: '+name)
            low, high, step, log = raw['low'], raw['high'], raw.get('step'), raw.get('log', False)
            valid_number = lambda v: isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
            if not valid_number(low) or not valid_number(high) or low > high or not isinstance(log, bool):
                raise AgentError('INVALID_TUNING_SPACE', 'Invalid numeric bounds/log: '+name)
            if step is not None and (not valid_number(step) or step <= 0):
                raise AgentError('INVALID_TUNING_SPACE', 'Step must be positive: '+name)
            if kind == 'int' and (not isinstance(low, int) or not isinstance(high, int) or (step is not None and not isinstance(step, int))):
                raise AgentError('INVALID_TUNING_SPACE', 'Integer domains need integer low/high/step: '+name)
            if log and (low <= 0 or (step is not None and not (kind == 'int' and step == 1))):
                raise AgentError('INVALID_TUNING_SPACE', 'Log domains need positive bounds and no step (int step=1 is allowed): '+name)
            if step is not None:
                count = (high-low)/step
                if not math.isclose(count, round(count), rel_tol=1e-10, abs_tol=1e-10):
                    raise AgentError('INVALID_TUNING_SPACE', 'high must align with low + k*step; bounds are never silently shortened: '+name)
            domain = {'type': kind, 'low': low if kind == 'int' else float(low),
                      'high': high if kind == 'int' else float(high), 'log': log}
            if step is not None:
                domain['step'] = step if kind == 'int' else float(step)
            values = [low, high]
        else:
            raise AgentError('INVALID_TUNING_SPACE', 'Use int, float or categorical domains: '+name)
        try:
            canonical(domain)
        except (ValueError, TypeError) as error:
            raise AgentError('INVALID_TUNING_SPACE', 'Domains must be finite JSON: '+name) from error
        for value in values:
            probe = {**params, name: value}
            normalized = catalog.normalize(recipe, probe)
            if constraints and catalog.constrain_params(recipe, probe, constraints) != normalized:
                raise AgentError('INVALID_TUNING_SPACE', 'Space conflicts with branch constraints: '+name)
        if not custom:
            owned = catalog.recipes[recipe]['parameters'][name]
            if 'enum' in owned and kind != 'categorical':
                raise AgentError('INVALID_TUNING_SPACE', 'Owned enum axes must be categorical: '+name)
            if owned.get('type') == 'int' and kind == 'float':
                raise AgentError('INVALID_TUNING_SPACE', 'Owned integer axes cannot use a float distribution: '+name)
        space[name] = domain
    return space


def contains(domain, value):
    if domain['type'] == 'categorical':
        return any(type(value) is type(v) and value == v for v in domain['choices'])
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return False
    if domain['type'] == 'int' and not isinstance(value, int):
        return False
    if not domain['low'] <= value <= domain['high']:
        return False
    step = domain.get('step')
    return step is None or math.isclose((value-domain['low'])/step, round((value-domain['low'])/step), rel_tol=1e-10, abs_tol=1e-10)
