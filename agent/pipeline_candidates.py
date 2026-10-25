"""Executable versions and controller-owned search spaces; no model-supplied scores."""
import ast
from copy import deepcopy
import hashlib
import os
from pathlib import Path

from agent.contracts import canonical
from agent.errors import AgentError


def enabled(config):
    return (config.research_mode == 'adaptive' and getattr(config, 'graph_enabled', True)
            and getattr(config, 'research_protocol', 'compact') != 'legacy'
            and getattr(config, 'background_tuning_enabled', False))


def declared_settings(source):
    """Read literal declarations without importing or executing generated source."""
    values = {}
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        raise AgentError('INVALID_CODE', str(error)) from error
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in {'DEFAULT_PARAMS', 'SEARCH_SPACE'}:
                try:
                    value = ast.literal_eval(node.value)
                except (ValueError, TypeError) as exc:
                    raise AgentError('INVALID_PIPELINE_SETTINGS', name+' must be a literal dictionary') from exc
                if name in values or not isinstance(value, dict):
                    raise AgentError('INVALID_PIPELINE_SETTINGS', name+' must occur once as a dictionary')
                canonical(value)
                values[name] = value
    return values


def normalize_candidate(action):
    from agent.research_actions import normalize_action
    result = normalize_action({'hypothesis': 'Evaluate the submitted executable pipeline version.', **action})
    result.setdefault('evidence_ids', [])
    if result['action'] == 'pipeline':
        result['recipe_id'] = 'python-pipeline-v1'
        source = result.get('code')
        if source:
            declarations = declared_settings(source)
            result['hyperparameters'] = {**declarations.get('DEFAULT_PARAMS', {}), **result['hyperparameters']}
            # Accept the previous wire format, but never start a researcher-owned
            # tuning series. Only the declared domain moves into the version.
            tuning = result.get('tuning')
            if tuning is not None:
                if not isinstance(tuning, dict):
                    raise AgentError('INVALID_TUNING_SPACE', 'tuning must be an object or null')
                if set(tuning)-{'search_space', 'parameters', 'max_trials', 'budget_seconds'}:
                    raise AgentError('INVALID_TUNING_SPACE', 'Unknown tuning fields; declare typed axes in SEARCH_SPACE')
                space = tuning.get('search_space')
                if space is not None:
                    if not isinstance(space, dict) or not space:
                        raise AgentError('INVALID_TUNING_SPACE', 'Provide a nonempty search_space')
                    if 'SEARCH_SPACE' in declarations and canonical(declarations['SEARCH_SPACE']) != canonical(space):
                        raise AgentError('INVALID_TUNING_SPACE', 'Conflicting SEARCH_SPACE and tuning.search_space; keep one domain')
                    if 'SEARCH_SPACE' not in declarations:
                        result['code'] = source.rstrip()+'\n\nSEARCH_SPACE = '+repr(space)+'\n'
                elif 'SEARCH_SPACE' not in declarations:
                    raise AgentError('INVALID_TUNING_SPACE', 'Custom tuning needs SEARCH_SPACE or tuning.search_space')
                names = tuning.get('parameters')
                if names is not None and set(names) != set(space or declarations.get('SEARCH_SPACE', {})):
                    raise AgentError('INVALID_TUNING_SPACE', 'parameters must match the declared search axes')
                result['tuning'] = None
    return result


def automatic_space(catalog, plan, source):
    """Freeze representation; tune model axes only. Custom source opts in once."""
    params = plan['model']['hyperparameters']
    from agent.feature_schema import FACTORY_SOURCE
    from agent.specialists import FACTORY_SOURCE as SPECIALIST_SOURCE
    wrapped = plan['recipe_id'] == 'python-pipeline-v1' and source.strip() in {FACTORY_SOURCE.strip(),SPECIALIST_SOURCE.strip()}
    if plan['recipe_id'] == 'python-pipeline-v1' and not wrapped:
        space = declared_settings(source).get('SEARCH_SPACE', {})
        if space:
            from agent.tuning_space import normalize_space
            space = normalize_space(catalog, plan, {'search_space': space})
        return {name: (('model', 'hyperparameters', name), domain) for name, domain in space.items()}
    recipe = params['base_recipe_id'] if wrapped else plan['recipe_id']
    fixed = params['model_parameters'] if wrapped else params
    path = ('model', 'hyperparameters', 'model_parameters') if wrapped else ('model', 'hyperparameters')
    excluded = {'numeric_interactions', 'target_transform', 'class_weight', 'strategy'}
    if catalog.config.ensemble_research_enabled:excluded.discard('class_weight')
    result = {}
    for name, domain in catalog.recipes[recipe]['parameters'].items():
        if name in excluded or name not in fixed:
            continue
        if 'enum' in domain:
            dist = {'type': 'categorical', 'choices': deepcopy(domain['enum'])}
        elif domain.get('type') in {'int', 'float'}:
            dist = {'type': domain['type'], 'low': domain['min'], 'high': domain['max'],
                    'log': domain['min'] > 0 and name in {'learning_rate', 'C', 'alpha', 'l2_regularization', 'l2_leaf_reg'}}
        else:
            continue
        result[name] = (path+(name,), dist)
    return result


def version_id(plan, source_hash, space):
    frozen = deepcopy(plan)
    for path, _ in space.values():
        target = frozen
        for name in path[:-1]:
            target = target[name]
        target[path[-1]] = '<optimizer-axis>'
    core = {name: frozen[name] for name in ('task_spec_id', 'dataset_id', 'validation_id', 'recipe_id', 'model', 'feature_steps', 'seed', 'fidelity')}
    return 'pipeline-'+hashlib.sha256(canonical([core, source_hash, space]).encode()).hexdigest()[:20]


def execution_capacity():
    """Leave OS headroom and respect a smaller local cgroup than the platform."""
    cpu = min(7, len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count() or 1)
    memory = 28672
    try:
        physical = os.sysconf('SC_PAGE_SIZE')*os.sysconf('SC_PHYS_PAGES')/1048576
        memory = min(memory, max(256, int(physical*.80)))
    except (ValueError, OSError, AttributeError):
        pass
    try:
        quota, period = Path('/sys/fs/cgroup/cpu.max').read_text().split()
        if quota != 'max':
            cpu = min(cpu, max(1, int(quota)//int(period)))
        limit = Path('/sys/fs/cgroup/memory.max').read_text().strip()
        if limit != 'max':
            memory = min(memory, max(256, int(int(limit)/1048576*.80)))
    except (OSError, ValueError):
        pass
    return max(1, cpu), memory


def execution_ticket(controller, template, parent=None):
    """Controller-only execution order. Never occupies an equal research peer."""
    import uuid
    graph = controller.graph
    work = deepcopy(template['work_order'])
    branch = graph.branch(work['branch_id'], work['branch_revision'])
    if branch['state'] != 'active':
        raise AgentError('INVALID_BRANCH_STATE', 'Pipeline origin branch was closed')
    work.update(work_order_id='background-'+uuid.uuid4().hex[:16], state='background',
                issued_elapsed_ms=graph.elapsed(),
                expires_elapsed_ms=graph.elapsed()+max(1, int(controller.budget.remaining()*1000)))
    if parent:
        work.update(base_experiment_id=parent, base_solution_id=graph.experiment_solution(parent),
                    allowed_donor_experiment_ids=[])
    graph.schemas.validate(work, 'GWorkOrder')
    with controller.store.transaction():
        graph.db.execute('INSERT INTO g_work_orders VALUES(?,?,?,?,?,?,?,?,?)',
            (graph.run_id, work['work_order_id'], work['branch_id'], work['researcher_id'],
             work['branch_revision'], work['base_head_revision'], 'background', work['expires_elapsed_ms'], canonical(work)))
        graph.db.execute('UPDATE g_runs SET graph_revision=graph_revision+1 WHERE run_id=?', (graph.run_id,))
        graph.event('G03', 'work_order', work['work_order_id'], 'GWorkOrder', work)
    return {'work_order': work}
