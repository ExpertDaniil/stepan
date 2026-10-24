"""AST/schema/resource checks only: no agent imports, tests, fit or API calls.

--sync refreshes declarative JSON schemas from their AST-only definitions.
--output writes the actual static results. This does not execute regressions.
"""
import argparse
import ast
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def expression(node, env):
    """Small declarative evaluator, deliberately unable to import/call runtime code."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return env[node.id]
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        values = [expression(x, env) for x in node.elts]
        return tuple(values) if isinstance(node, ast.Tuple) else set(values) if isinstance(node, ast.Set) else values
    if isinstance(node, ast.Dict):
        result = {}
        for key, value in zip(node.keys, node.values):
            if key is None:
                result.update(expression(value, env))
            else:
                result[expression(key, env)] = expression(value, env)
        return result
    if isinstance(node, ast.Subscript):
        return expression(node.value, env)[expression(node.slice, env)]
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        args = [expression(x, env) for x in node.args]
        if node.func.id == 'list':
            return list(*args)
        if node.func.id == 'text_schema':
            return {'type': 'string', 'minLength': 1, 'maxLength': args[0] if args else 400}
    if isinstance(node, ast.DictComp) and len(node.generators) == 1:
        generator = node.generators[0]
        if isinstance(generator.target, ast.Name) and not generator.ifs and not generator.is_async:
            result = {}
            for value in expression(generator.iter, env):
                local = {**env, generator.target.id: value}
                result[expression(node.key, local)] = expression(node.value, local)
            return result
    raise ValueError('Nondeclarative schema expression: '+ast.dump(node)[:180])


def declarations():
    env = {}
    wanted = {'OPS', 'GROUP_OPS', 'FEATURE_PROGRAM_SCHEMA', 'SOURCE_SCHEMA', 'QUANTITY_KINDS',
        'RELATION_KINDS', 'QUANTITY_SCHEMA', 'RELATION_SCHEMA', 'TASK_MODEL_SCHEMA', 'INVESTIGATION_SCHEMA',
        'CUSTOM_RECIPE', 'IDEA_SCHEMA', 'RESEARCH_NOTES_SCHEMA', 'SEMANTIC_UPDATE_SCHEMA',
        'DISTRIBUTION_SCHEMA', 'TUNING_SCHEMA', 'ACTION_SCHEMA', 'FILTERS_SCHEMA', 'DERIVED_SCHEMA', 'ANALYSIS_SCHEMA'}
    for relative in ('agent/feature_schema.py', 'agent/semantic_planning.py', 'agent/research_actions.py', 'tools/analysis.py'):
        tree = ast.parse((ROOT/relative).read_text())
        for node in tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                name = node.targets[0].id
                if name in wanted:
                    env[name] = expression(node.value, env)
    return env


def synchronized(schema, values):
    result = deepcopy(schema)
    defs = result['$defs']
    defs['TaskModel'] = values['TASK_MODEL_SCHEMA']
    defs['Investigation'] = values['INVESTIGATION_SCHEMA']
    defs['ResearchAction'] = values['ACTION_SCHEMA']
    defs['AnalysisQuery'] = values['ANALYSIS_SCHEMA']
    if 'GResearchContext' in defs:
        defs['GResearchContext']['properties']['metadata']['properties'].update(
            graph_request_max_bytes={'type': 'integer', 'minimum': 1},
            detail_artifact_id={'type': 'string', 'minLength': 1},
            detail_scope={'type': 'object', 'additionalProperties': True},
            semantic_contract={'const': 'semantic-planning-1'})
    defs['TaskSpec']['properties'].update(semantic_model_artifact_id={'type': 'string', 'minLength': 1},
                                         semantic_model_id={'type': 'string', 'minLength': 1})
    draft = defs['C02_Response']['oneOf'][0]['properties']['data']['properties']['draft']
    for variant in draft.get('oneOf', [draft]):
        if 'target_column' in variant.get('properties', {}):
            variant['properties']['task_model'] = {'$ref': '#/$defs/TaskModel'}
    return result


def schema_validator(schema):
    # Extract just the existing JSON-value checker, not the module/import graph.
    tree = ast.parse((ROOT/'agent/contracts.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Schemas')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_check')
    namespace = {'math': math, 're': re}
    exec(compile(ast.Module(body=[method], type_ignores=[]), '<AST-only JSON schema checker>', 'exec'), namespace)
    return type('StaticJSONChecker', (), {'schema': schema, '_check': namespace['_check']})()


def check(sync=False):
    parsed = [path for path in ROOT.rglob('*.py') if '.git' not in path.parts and '__pycache__' not in path.parts]
    for path in parsed:
        ast.parse(path.read_text(encoding='utf-8'), filename=str(path.relative_to(ROOT)))
    values = declarations()
    schemas = []
    for relative in ('contracts/contracts.schema.json', 'docs/architecture/contracts.schema.json'):
        path = ROOT/relative
        original = json.loads(path.read_text())
        expected = synchronized(original, values)
        if sync:
            path.write_text(json.dumps(expected, ensure_ascii=False, indent=2)+'\n')
        elif original != expected:
            raise ValueError('Schema declarations drifted: '+relative)
        schemas.append(expected)
    validator = schema_validator(schemas[0])
    references = []
    def visit(value):
        if isinstance(value, dict):
            if '$ref' in value:
                references.append(value['$ref'])
                if value['$ref'].split('/')[-1] not in schemas[0]['$defs']:
                    raise ValueError('Unresolved schema reference: '+value['$ref'])
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
    visit(schemas[0])
    fixture = json.loads((ROOT/'contracts/examples/semantic-planning.json').read_text())
    for name, value in fixture['valid'].items():
        validator._check(value, schemas[0]['$defs'][name], '$')
    # Validate the frozen C00-C24 examples as JSON values, without handlers.
    examples = 0
    for directory in sorted((ROOT/'contracts/examples').glob('C[0-9][0-9]')):
        for path in sorted(directory.glob('*.json')):
            suffix = 'Request' if 'request' in path.stem else 'Response'
            try:
                validator._check(json.loads(path.read_text()), schemas[0]['$defs'][directory.name+'_'+suffix], '$')
            except ValueError as error:
                raise ValueError(str(path.relative_to(ROOT))+': '+str(error)) from error
            examples += 1
    package = ast.parse((ROOT/'scripts/package_solution.py').read_text())
    resources = next(ast.literal_eval(n.value) for n in package.body if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == 'RESOURCE_FILES' for t in n.targets))
    for relative in resources:
        if not (ROOT/relative).is_file():
            raise ValueError('Missing runtime resource: '+relative)
    runtime = [p for folder in ('agent', 'tools') for p in (ROOT/folder).rglob('*.py') if '__pycache__' not in p.parts]
    names = [p.relative_to(ROOT).as_posix().replace('/__init__.py', '_init.py').replace('/', '_') for p in runtime]
    if len(names) != len(set(names)):
        raise ValueError('Flat runtime filename collision')
    if (ROOT/'requirements.txt').read_bytes() != (ROOT/'requirements.organizers.txt').read_bytes():
        raise ValueError('Official requirements differ')
    test_tree = ast.parse((ROOT/'tests/test_semantic_planning.py').read_text())
    test_methods = sum(isinstance(n, ast.FunctionDef) and n.name.startswith('test_') for n in ast.walk(test_tree))
    json_files = list(ROOT.rglob('*.json'))
    for path in json_files:
        if '.git' not in path.parts:
            json.loads(path.read_text())
    return {'validation': 'static_only; no agent imports, regression execution, fitting or API calls',
        'python_files_parsed': len(parsed), 'runtime_modules': len(runtime), 'runtime_resources': len(resources),
        'schema_declaration_equality': True, 'resolved_schema_references': len(references),
        'new_json_examples_checked': len(fixture['valid']), 'original_contract_examples_checked': examples,
        'flat_module_names_unique': True, 'official_requirements_equal': True,
        'schema_sha256': hashlib.sha256((ROOT/'contracts/contracts.schema.json').read_bytes()).hexdigest(),
        'new_regression_methods_written': test_methods,
        'json_files_parsed': len(json_files),
        'adaptive_prompt_bytes': len((ROOT/'prompts/researcher-adaptive.md').read_bytes()),
        'research_action_schema_bytes': len(json.dumps(values['ACTION_SCHEMA'], ensure_ascii=False).encode()),
        'analysis_tool_schema_bytes': len(json.dumps(values['ANALYSIS_SCHEMA'], ensure_ascii=False).encode()),
        'dynamic_regressions': 'written_not_run', 'agent_runs': 'not_run', 'quality_comparison': 'not_run'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sync', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = check(args.sync)
    if args.output:
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))
