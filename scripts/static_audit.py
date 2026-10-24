"""Read-only source/package audit. Never imports the agent, runs tests or builds a ZIP."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def data_value(node, bindings):
    """Interpret data literals/references only, never execute project code."""
    if isinstance(node, ast.Name):
        return bindings[node.id]
    if isinstance(node, ast.Subscript):
        return data_value(node.value, bindings)[data_value(node.slice, bindings)]
    if isinstance(node, ast.Dict):
        result = {}
        for key, value in zip(node.keys, node.values):
            if key is None:
                result.update(data_value(value, bindings))
            else:
                result[data_value(key, bindings)] = data_value(value, bindings)
        return result
    if isinstance(node, ast.DictComp) and len(node.generators) == 1:
        generator = node.generators[0]
        if not isinstance(generator.target, ast.Name) or generator.ifs or generator.is_async:
            raise ValueError('Unsupported schema comprehension')
        return {data_value(node.key, {**bindings, generator.target.id: value}):
                data_value(node.value, {**bindings, generator.target.id: value})
                for value in data_value(generator.iter, bindings)}
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'text_schema' and len(node.args) <= 1 and not node.keywords:
        return {'type': 'string', 'minLength': 1, 'maxLength': data_value(node.args[0], bindings) if node.args else 400}

    if isinstance(node, (ast.List, ast.Tuple)):
        values = [data_value(v, bindings) for v in node.elts]
        return tuple(values) if isinstance(node, ast.Tuple) else values
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'list' and len(node.args) == 1 and not node.keywords:
        return list(data_value(node.args[0], bindings))
    return ast.literal_eval(node)


def audit(root, official_archive=None):
    root = Path(root)
    files = [root/'solution.py']
    for name in ('agent', 'tools', 'scripts', 'tests'):
        files.extend(sorted((root/name).rglob('*.py')))
    trees = {path.relative_to(root).as_posix(): ast.parse(path.read_bytes(), filename=str(path.relative_to(root))) for path in files}
    for name, tree in trees.items():
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and not node.level and node.module and node.module.split('.')[0] in {'agent', 'tools', 'tests'}:
                module = root/Path(*node.module.split('.'))
                if not module.with_suffix('.py').is_file() and not (module/'__init__.py').is_file():
                    raise ValueError('Missing owned import '+node.module+' in '+name)
    packager = trees['scripts/package_solution.py']
    resources = next(ast.literal_eval(n.value) for n in packager.body if isinstance(n, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == 'RESOURCE_FILES' for t in n.targets))
    for name in resources:
        path = root/name
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError('Missing or unsafe resource: '+name)
        path.read_bytes().decode('utf-8')
    required = {'agent/schema.sql', 'agent/graph/schema.sql', 'contracts/contracts.schema.json',
                'prompts/researcher-adaptive.md', 'recipes/knowledge/cards.json', 'recipes/knowledge/sources.jsonl'}
    if not required <= set(resources):
        raise ValueError('Runtime text resources are not allowlisted')
    modules = [name for name in trees if name.startswith(('agent/', 'tools/'))]
    flat = {name: name.replace('/__init__.py', '_init.py').replace('/', '_') for name in modules}
    if len(set(flat.values())) != len(flat) or any('/' in n or not n.endswith('.py') for n in flat.values()):
        raise ValueError('Flat owned module names collide or violate root-only policy')
    platform = next(n for n in packager.body if isinstance(n, ast.FunctionDef) and n.name == 'platform_files')
    if not any(isinstance(n, ast.Assign) and isinstance(n.value, ast.Name) and n.value.id == 'content' and
               any(isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name) and t.value.id == 'entries' for t in n.targets)
               for n in ast.walk(platform)):
        raise ValueError('Review changed source-byte assignment in the packager')
    requirements = (root/'requirements.txt').read_bytes()
    if requirements != (root/'requirements.organizers.txt').read_bytes():
        raise ValueError('Repository requirements copies differ')
    official_equal = None
    if official_archive:
        with zipfile.ZipFile(official_archive) as archive:
            paths = [n for n in archive.namelist() if n.endswith('/requirements.txt')]
            if len(paths) != 1 or archive.read(paths[0]) != requirements:
                raise ValueError('Requirements differ from the supplied official archive')
        official_equal = True
    schema = json.loads((root/'contracts/contracts.schema.json').read_text())
    allowed = next(ast.literal_eval(n.value) for n in ast.walk(trees['agent/contracts.py'])
                   if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'allowed' for t in n.targets))
    def schema_structure(value):
        if set(value)-allowed:
            raise ValueError('Unsupported local schema keywords: '+repr(set(value)-allowed))
        if '$ref' in value and value['$ref'].split('/')[-1] not in schema['$defs']:
            raise ValueError('Dangling schema reference: '+value['$ref'])
        if 'type' in value and not isinstance(value['type'], str):
            raise ValueError('Local validator requires one type string; use anyOf for nullability')
        for key in ('properties', '$defs'):
            for child in value.get(key, {}).values():
                schema_structure(child)
        for key in ('anyOf', 'oneOf'):
            for child in value.get(key, []):
                schema_structure(child)
        for key in ('items', 'additionalProperties'):
            if isinstance(value.get(key), dict):
                schema_structure(value[key])
    schema_structure(schema)
    declarations = {}
    for module in ('agent/semantic_planning.py', 'agent/feature_schema.py', 'agent/research_actions.py', 'tools/analysis.py', 'agent/context_details.py'):
        for node in trees[module].body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        declarations[target.id] = data_value(node.value, declarations)
    for module, name, definition in [('agent/research_actions.py', 'ACTION_SCHEMA', 'ResearchAction'),
                                      ('tools/knowledge.py', 'KNOWLEDGE_SCHEMA', 'KnowledgeQuery'),
                                      ('tools/analysis.py', 'ANALYSIS_SCHEMA', 'AnalysisQuery'),
                                      ('agent/context_details.py', 'READ_SCHEMA', 'MemoryRead'),
                                      ('agent/context_details.py', 'SOLUTION_SCHEMA', 'SolutionRead')]:
        node = next(n for n in trees[module].body if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == name for t in n.targets))
        if data_value(node.value, declarations) != schema['$defs'][definition]:
            raise ValueError('Runtime/contract schema declarations differ: '+definition)
    frozen = next(n for n in ast.walk(trees['agent/runner.py']) if isinstance(n, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == 'runtime' for t in n.targets))
    if not {'agent/feature_engineering.py', 'agent/feature_schema.py', 'tools/runtime_guard.py'} <= set(ast.literal_eval(frozen.value)):
        raise ValueError('Feature execution dependencies missing from the frozen runtime')
    json_count = 0
    for directory in ('contracts', 'recipes'):
        for path in (root/directory).rglob('*.json'):
            json.loads(path.read_text(encoding='utf-8'));json_count += 1
    knowledge = root/'recipes/knowledge'
    manifest = json.loads((knowledge/'manifest.json').read_text())
    for name, expected in manifest['files'].items():
        if hashlib.sha256((knowledge/name).read_bytes()).hexdigest() != expected:
            raise ValueError('Knowledge manifest mismatch: '+name)
    return {'audit': 'static_source_and_package_inputs', 'python_files_parsed': len(trees),
        'owned_runtime_modules': len(modules), 'allowlisted_text_resources': len(resources),
        'json_files_parsed': json_count, 'schema_vocabulary_and_refs': 'checked_without_importing_project_modules',
        'runtime_contract_schema_declarations': 'byte-independent_data_equality_checked',
        'feature_execution_dependencies': 'included_in_frozen_runtime',
        'official_requirements_byte_equal': official_equal,
        'requirements_sha256': hashlib.sha256(requirements).hexdigest(),
        'knowledge_manifest': 'all_file_hashes_equal',
        'regression_methods_written': sum(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name.startswith('test_')
                                          for name, tree in trees.items() if name.startswith('tests/') for n in ast.walk(tree)),
        'runtime_source_sha256': {name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in sorted(modules)},
        'not_performed': ['project_module_import', 'test_execution', 'training', 'agent_smoke_or_full_run',
                          'live_llm_request', 'dependency_installation', 'zip_build_or_runtime_execution'],
        'limitation': 'Static packaging policy/input checks do not prove generated ZIP bytes or platform execution.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--official-data-archive', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = audit(ROOT, args.official_data_archive)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'runtime_source_sha256'}, ensure_ascii=False, indent=2))
