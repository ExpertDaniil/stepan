"""Feature program syntax shared by controller and worker; no ML imports."""
from copy import deepcopy

OPS = ('missing', 'log1p', 'signed_log1p', 'square', 'sum', 'difference', 'product',
       'ratio', 'frequency', 'cross', 'group_mean', 'group_std', 'group_rank',
       'relative_to_group', 'month', 'dayofweek', 'hour')

GROUP_OPS = ('group_mean', 'group_std', 'group_rank', 'relative_to_group')

FEATURE_PROGRAM_SCHEMA = {
    'type': 'array', 'minItems': 1, 'maxItems': 24,
    'items': {'type': 'object', 'properties': {
        'op': {'enum': list(OPS)}, 'name': {'type': 'string', 'minLength': 1, 'maxLength': 100},
        'columns': {'type': 'array', 'items': {'type': 'string', 'minLength': 1}, 'minItems': 1, 'maxItems': 4},
        'group_by': {'type': 'string', 'minLength': 1}},
        'required': ['op', 'name', 'columns'], 'additionalProperties': False}}

FACTORY_SOURCE = ('from agent.feature_engineering import make_feature_pipeline\n\n'
                  'def make_pipeline(task_spec, params):\n'
                  '    return make_feature_pipeline(task_spec, params)\n')

def validate_program(program, columns=None, excluded=()):
    if not isinstance(program, list) or not 1 <= len(program) <= 24:
        raise ValueError('feature_program must contain 1..24 operations')
    available = set(columns) if columns is not None else None
    generated = set()
    for step in program:
        if not isinstance(step, dict) or set(step)-{'op', 'name', 'columns', 'group_by'}:
            raise ValueError('Unknown feature operation fields')
        op, name, inputs = step.get('op'), step.get('name'), step.get('columns')
        if op not in OPS or not isinstance(name, str) or not 1 <= len(name) <= 100:
            raise ValueError('Invalid feature operator or output name')
        if name in generated or name in excluded or (available is not None and name in available):
            raise ValueError('Feature output overwrites an existing/excluded column: '+name)
        if (not isinstance(inputs, list) or not 1 <= len(inputs) <= 4
                or any(not isinstance(c, str) or not c for c in inputs)):
            raise ValueError('Feature inputs must be 1..4 named columns')
        if op in {'difference', 'ratio'} and len(inputs) != 2:
            raise ValueError(op+' needs exactly two inputs')
        if op not in {'sum', 'product', 'difference', 'ratio', 'cross'} and len(inputs) != 1:
            raise ValueError(op+' needs exactly one input')
        group = step.get('group_by')
        if ((op in GROUP_OPS and not (isinstance(group, str) and group))
                or (op not in GROUP_OPS and 'group_by' in step)):
            raise ValueError('group_by is required only for grouped numeric operations')
        dependencies = inputs + ([group] if group is not None else [])
        if any(c in excluded or (available is not None and c not in available) for c in dependencies):
            raise ValueError('Feature depends on an absent or excluded column')
        generated.add(name)
        if available is not None:
            available.add(name)
    return deepcopy(program)
