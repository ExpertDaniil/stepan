"""Mandatory initial data research; scoring and resource limits stay independent."""
from agent.errors import AgentError


INSTRUCTION = (
    'Mandatory initial DATA RESEARCH phase. Investigate the task data, missing values, '
    'distributions, groups, leakage-safe preprocessing and meaningful representations. '
    'Use analyze_data/read_memory and propose features or a pipeline with explicit data/feature steps. '
    'Optuna, tune, model-only catalogue trials and stopping for a sufficient score are unavailable. '
    'For features actions, keep model hyperparameters at recipe defaults; only data-transform switches may change. '
    'Do not disguise hyperparameter changes as data work. Keep a measured reference model; '
    'report negative results honestly and investigate another concrete data hypothesis. '
    'The controller supplies remaining_seconds: wall time, not a trial count or score target.')


def attach_context(user, window):
    profile = user.get('profile_summary', user)
    profile['data_research'] = window
    if window['active']:
        profile['feature_focus'] = True
        user['assignment'] = INSTRUCTION
    elif user.get('assignment') == INSTRUCTION:
        user['assignment'] = 'Choose the next affordable data, pipeline or model-tuning experiment using measured evidence.'


def require_action(action, window):
    if not window['active']:
        return
    if action['action'] in {'features', 'research'}:
        return
    if action['action'] == 'pipeline' and action.get('feature_steps'):
        return
    raise AgentError('DATA_RESEARCH_REQUIRED',
        'The mandatory data research window has '+str(window['remaining_seconds'])+
        ' seconds left. Analyze fit data and propose features or a pipeline with explicit data/feature steps; '
        'model-only trials, Optuna and early stop are unavailable.', details=window)


def require_tuning(budget):
    window = budget.data_research_snapshot()
    if window['active']:
        require_action({'action': 'tune'}, window)


def require_feature_parameters(action, catalog, window, constraints=None):
    """A fixed representation must not disguise a parameter sweep as features."""
    if not window['active'] or action['action'] != 'features':
        return
    constraints = constraints or {'required_capability_ids': [], 'forbidden_capability_ids': []}
    expected = catalog.constrain_params(action['recipe_id'], {}, constraints)
    actual = catalog.constrain_params(action['recipe_id'], action['hyperparameters'], constraints)
    data_switches = {'numeric_interactions', 'target_transform'}
    changed = [name for name in expected if name not in data_switches and actual[name] != expected[name]]
    if changed:
        raise AgentError('DATA_RESEARCH_REQUIRED',
            'Keep model parameters fixed at recipe defaults during data research. '
            'Use hyperparameters={} and change the feature program. Changed model parameters: '+', '.join(changed),
            details=window)
