"""Small controller-authorized feature seed bank, separate from model tuning.

Seeds are hypotheses selected using fit-only schema/statistics, not known task names.
All measurements still pass through normal graph tickets, Runner and Evaluator.
"""
from collections import deque
from copy import deepcopy
import uuid

from agent.contracts import canonical
from agent.feature_schema import FACTORY_SOURCE


def focus_for_round(round_no, peer_index, peer_count):
    # Rotate the assignment instead of adding a permanent feature-only agent.
    return (round_no+peer_index) % max(2, peer_count) == 0


def seed_programs(frame):
    import numpy as np
    import pandas as pd
    prefix = '__feature_seed_'
    while any(str(c).startswith(prefix) for c in frame.columns):
        prefix = '_'+prefix
    numeric = [c for c in frame if pd.api.types.is_numeric_dtype(frame[c])
               and not pd.api.types.is_bool_dtype(frame[c])]
    categorical = [c for c in frame if c not in numeric and 1 < frame[c].nunique() <= 2000]
    categorical.sort(key=lambda c: (frame[c].nunique(), str(c)))
    candidates = []
    missing = [c for c in numeric if frame[c].isna().any()][:24]
    if missing:
        candidates.append([{'op': 'missing', 'name': prefix+'missing_'+str(i), 'columns': [c]}
                           for i, c in enumerate(missing)])
    spread = []
    for c in numeric:
        values = pd.to_numeric(frame[c], errors='coerce').replace([np.inf, -np.inf], np.nan).dropna()
        if len(values) and values.nunique() > 20:
            quantiles = values.abs().quantile([.5, .99])
            ratio = float(quantiles.iloc[1]/max(1e-12, quantiles.iloc[0]))
            if np.isfinite(ratio) and ratio > 3:
                spread.append((ratio, c))
    spread.sort(key=lambda item: (-item[0], str(item[1])))
    if spread:
        candidates.append([{'op': 'signed_log1p', 'name': prefix+'log_'+str(i), 'columns': [c]}
                           for i, (_, c) in enumerate(spread[:3])])
    if categorical:
        candidates.append([{'op': 'frequency', 'name': prefix+'frequency_'+str(i), 'columns': [c]}
                           for i, c in enumerate(categorical[:3])])
    if len(categorical) >= 2:
        candidates.append([
            {'op': 'cross', 'name': prefix+'cross', 'columns': categorical[:2]},
            {'op': 'frequency', 'name': prefix+'cross_frequency', 'columns': [prefix+'cross']}])
    # No arbitrary first-column polynomial expansion. Group context is an explicitly
    # unverified hypothesis; it is not declared relevant by a marginal statistic.
    for group in categorical[:2]:
        if frame[group].nunique() > 100:
            continue
        for _, column in spread[:2]:
            candidates.append([
                {'op': 'relative_to_group', 'name': prefix+'relative', 'columns': [column], 'group_by': group},
                {'op': 'group_rank', 'name': prefix+'rank', 'columns': [column], 'group_by': group}])
    return candidates[:8]


class FeatureSearch:
    def __init__(self, controller):
        self.c = controller
        self.banks = {}
        self.plan_ids = set()
        self.spent_seconds = 0.0

    def next_plan(self, peer, branch, constraints=None):
        c = self.c
        if (not c.config.feature_research_enabled or c.config.research_mode != 'adaptive'
                or not c.budget.snapshot()['search_allowed']
                or self.spent_seconds >= c.config.time_budget_seconds*c.config.feature_seed_budget_fraction
                or (c.graph and c.graph.refinement())):
            return None
        if constraints and not c.catalog.allows_constraints('python-pipeline-v1', constraints):
            return None
        ids = c.graph.visible_history(branch) if c.graph else [r[0] for r in c.store.db.execute(
            'SELECT experiment_id FROM experiments WHERE run_id=?', (c.store.run_id,))]
        eligible = []
        for eid in ids:
            exp = c.store.get(eid)
            plan = exp['plan']
            ev = c.store.experiment_evaluation(eid)
            if (exp['state'] == 'evaluated' and ev and ev['valid'] and not ev['violations']
                    and not plan['recipe_id'].startswith(('dummy-', 'python-'))
                    and c.catalog.allows_constraints(plan['recipe_id'], constraints or {'required_capability_ids': [], 'forbidden_capability_ids': []})):
                eligible.append((ev['primary_metric']['value']*(1 if ev['primary_metric']['direction'] == 'minimize' else -1), plan))
        if not eligible:
            return None
        base = deepcopy(min(eligible, key=lambda item: item[0])[1])
        if branch not in self.banks:
            from agent.predictions import targets
            fit = c.context.analysis.fit
            columns = [name for name in fit if name not in {*targets(c.spec), *c.spec['excluded_features']}]
            self.banks[branch] = deque(seed_programs(fit[columns]))
        while self.banks[branch]:
            program = self.banks[branch].popleft()
            params = c.catalog.constrain_params(base['recipe_id'], base['model']['hyperparameters'], constraints) if constraints else base['model']['hyperparameters']
            compiled = {'base_recipe_id': base['recipe_id'], 'model_parameters': params, 'feature_program': program}
            plan = deepcopy(base)
            plan.update(plan_id='feature-seed-'+uuid.uuid4().hex[:12], researcher_id=peer,
                recipe_id='python-pipeline-v1', model={'family': 'custom', 'hyperparameters': compiled},
                resources=c.resources(),
                hypothesis='Controller-authorized fit-only representation seed: '+canonical(program),
                expected_effect='Measure this complete representation; its benefit and mechanism are unknown.',
                feature_steps=[{'step': 'owned_feature_program', 'program': program, 'fit_scope': 'train_only'}])
            from tools.code import write_code
            args = {'researcher_id': 'feature-seeds', 'filename': 'features.py',
                    'content': FACTORY_SOURCE, 'declared_entrypoint': 'make_pipeline'}
            code = c.call('C11', args, lambda p: write_code(p, c.artifacts))['artifact']
            plan['code_artifact_id'] = code['artifact_id']
            plan['resources']['timeout_seconds'] = min(plan['resources']['timeout_seconds'], c.config.feature_seed_timeout_seconds)
            c.catalog.validate_plan(plan, c.spec, c.validation['validation_id'], c.artifacts)
            if not c.budget.can_train(plan)['allowed']:
                return None
            self.plan_ids.add(plan['plan_id'])
            c.log.event('features.seed_proposed', plan_id=plan['plan_id'], branch_id=branch, researcher_id=peer,
                base_recipe_id=base['recipe_id'], program=program, remaining_seed_programs=len(self.banks[branch]),
                spent_seconds=self.spent_seconds, quality_unverified=True)
            return plan
        return None


def feature_focus(user):
    return bool(user.get('profile_summary', user).get('feature_focus', False))


def validate_focus(action, user):
    if not feature_focus(user) or action['action'] == 'stop':
        return
    if action['action'] == 'features':
        return
    if (action['action'] == 'pipeline' and action['code'] != user.get('base_code')
            and any(s.get('step') not in {'target_transform', 'shifted_log1p', 'numeric_interactions'}
                    for s in action['feature_steps'])):
        return
    from agent.errors import AgentError
    raise AgentError('FEATURE_RESEARCH_REQUIRED', 'This rotating assignment funds a new feature representation. '
        'Return features, a changed custom feature pipeline, or stop with a reason; parameter-only tuning is not admitted.')
