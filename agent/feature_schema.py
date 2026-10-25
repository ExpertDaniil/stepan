"""Feature program syntax shared by controller and worker; no ML imports."""
from copy import deepcopy
import math

OPS = ('missing', 'log1p', 'signed_log1p', 'square', 'sum', 'difference', 'product',
       'ratio', 'frequency', 'cross', 'group_mean', 'group_std', 'group_rank',
       'relative_to_group', 'month', 'dayofweek', 'hour', 'cyclic_sin', 'cyclic_cos', 'target_mean', 'target_count')

GROUP_OPS = ('group_mean', 'group_std', 'group_rank', 'relative_to_group')
TARGET_OPS = ('target_mean', 'target_count')

FEATURE_PROGRAM_SCHEMA = {
    'type': 'array', 'minItems': 1, 'maxItems': 24,
    'items': {'type': 'object', 'properties': {
        'op': {'enum': list(OPS)}, 'name': {'type': 'string', 'minLength': 1, 'maxLength': 100},
        'columns': {'type': 'array', 'items': {'type': 'string', 'minLength': 1}, 'minItems': 1, 'maxItems': 4},
        'group_by': {'type': 'string', 'minLength': 1},
        'period': {'type': 'number', 'exclusiveMinimum': 0},
        'origin': {'type': 'number'},
        'alpha': {'type': 'number', 'minimum': 0, 'maximum': 1000000},
        'input_units': {'type': 'object', 'additionalProperties': {'type': 'string', 'minLength': 1, 'maxLength': 100}},
        'output_unit': {'type': 'string', 'minLength': 1, 'maxLength': 100}},
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
        if not isinstance(step, dict) or set(step)-{'op', 'name', 'columns', 'group_by', 'input_units', 'output_unit', 'period', 'origin', 'alpha'}:
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
        if op not in {'sum', 'product', 'difference', 'ratio', 'cross', *TARGET_OPS} and len(inputs) != 1:
            raise ValueError(op+' needs exactly one input')
        alpha=step.get('alpha',10.)
        if op in TARGET_OPS:
            if isinstance(alpha,bool) or not isinstance(alpha,(int,float)) or not math.isfinite(alpha) or not 0<=alpha<=1000000:
                raise ValueError('Target statistics require finite nonnegative smoothing')
        elif 'alpha' in step:
            raise ValueError('alpha belongs only to target statistics')
        if op in {'cyclic_sin', 'cyclic_cos'}:
            period, origin = step.get('period'), step.get('origin', 0)
            if (isinstance(period, bool) or not isinstance(period, (int, float)) or not math.isfinite(period) or period <= 0
                    or isinstance(origin, bool) or not isinstance(origin, (int, float)) or not math.isfinite(origin)):
                raise ValueError('Cyclic operators need a finite positive period and finite origin')
        elif 'period' in step or 'origin' in step:
            raise ValueError('period/origin belong only to cyclic operators')
        group = step.get('group_by')
        if ((op in GROUP_OPS and not (isinstance(group, str) and group))
                or (op not in GROUP_OPS and 'group_by' in step)):
            raise ValueError('group_by is required only for grouped numeric operations')
        dependencies = inputs + ([group] if group is not None else [])
        if any(c in excluded or (available is not None and c not in available) for c in dependencies):
            raise ValueError('Feature depends on an absent or excluded column')
        declared = step.get('input_units', {})
        if (not isinstance(declared, dict) or not set(declared) <= set(dependencies)
                or any(not isinstance(v, str) or not 1 <= len(v) <= 100 for v in declared.values())):
            raise ValueError('input_units must name dependencies with nonempty unit strings')
        if 'output_unit' in step and (not isinstance(step['output_unit'], str) or not 1 <= len(step['output_unit']) <= 100):
            raise ValueError('output_unit must be a nonempty unit string')
        generated.add(name)
        if available is not None:
            available.add(name)
    return deepcopy(program)


PROGRAM_VERSION = 'feature-program-4'


def validate_for_task(program,spec,columns=None):
    excluded=[*spec.get('target_columns',[spec['target_column']]),*spec['excluded_features']]
    result=validate_program(program,columns,excluded)
    if spec['task_kind']!='regression' and any(step['op'] in TARGET_OPS for step in result):
        raise ValueError('Owned target statistics currently support regression only')
    return result


def operation_facts(program):
    """Derived from executable syntax, not a model-authored feature declaration."""
    import hashlib
    import json
    validated = validate_program(program)
    identity = hashlib.sha256(json.dumps([PROGRAM_VERSION, validated], sort_keys=True,
        ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
    return {'schema_version': PROGRAM_VERSION, 'program_id': 'program-'+identity[:24],
        'verification': 'owned_executable_specification', 'predictive_usefulness': 'not_established',
        'operations': [{'order': i, 'operator': step['op'], 'output': step['name'],
            'inputs': list(step['columns']), 'group_by': step.get('group_by'),
            'fit_scope': 'fit_only_target_crossfit' if step['op'] in TARGET_OPS else 'fit_rows_without_target' if step['op'] in (*GROUP_OPS, 'frequency') else 'rowwise',
            'formula': {'missing': 'is_missing(x)', 'log1p': 'log(1+x) for x>=0',
                'signed_log1p': 'sign(x)*log(1+abs(x))', 'square': 'x*x', 'sum': 'sum(inputs)',
                'difference': 'inputs[0]-inputs[1]', 'product': 'product(inputs)',
                'ratio': 'inputs[0]/inputs[1]', 'cross': 'length-prefixed typed category keys',
                'frequency': 'fit_count(key)/fit_rows', 'group_mean': 'fit_group_mean(x)',
                'group_std': 'fit_group_std(x,ddof=1)', 'group_rank': 'fit_group_ECDF(x), right inclusive',
                'relative_to_group': 'x/fit_group_mean(x)', 'month': 'UTC month',
                'dayofweek': 'UTC weekday 0..6', 'hour': 'UTC hour',
                'cyclic_sin': 'sin(2*pi*(x-origin)/period)', 'cyclic_cos': 'cos(2*pi*(x-origin)/period)',
                'target_mean': '(other_fold_target_sum+alpha*other_fold_prior)/(other_fold_count+alpha)',
                'target_count': 'other_fold_count(key); full_fit_count at prediction'}[step['op']],
            **({'alpha': step.get('alpha',10.)} if step['op'] in TARGET_OPS else {}),
            **({'period': step['period'], 'origin': step.get('origin', 0)} if step['op'] in {'cyclic_sin', 'cyclic_cos'} else {}),
            'missing': ('typed missing key participates in counts/cross/group lookup'
                if step['op'] in {'cross', 'frequency', 'group_mean', 'group_std'} else
                'indicator 1 for missing' if step['op'] == 'missing' else 'missing value propagates'),
            'zero_denominator': 'missing', 'unseen_group': 'missing', 'unseen_frequency': 0,
            'units': {'declared_inputs': step.get('input_units', {}), 'declared_output': step.get('output_unit'),
                      'verification': 'declarations_only; not inferred from names/dtypes'},
            'warnings': (['Different declared units in addition/subtraction; check scales or explicitly treat this as exploratory.']
                if step['op'] in {'sum', 'difference'} and len(set(step.get('input_units', {}).values())) > 1 else []),
            'domain': 'log1p requires nonnegative input; invalid/nonfinite numeric outputs become missing'}
            for i, step in enumerate(validated)]}
