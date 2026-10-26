"""Equal-weight ensemble of measured owned pipelines, executed as one ordinary candidate.

Members are rebuilt from their frozen recipe/parameters with the owned
constructors (catalogue recipes or owned feature-program pipelines), so the
ensemble needs no generated code. Regression and probability tasks average
member outputs; label tasks take a plurality vote, ties going to the best-ranked
member (members are listed best first), so tuned class decisions of each member
are preserved. The controller only proposes an ensemble whose average of the
members' saved validation predictions already beats the best member
(``precheck``); the Evaluator, noise gate and closed confirmation then judge it
like any other candidate.
"""
from copy import deepcopy

import numpy as np
from sklearn.base import BaseEstimator

ENSEMBLE_SOURCE = ('from agent.ensemble import make_ensemble_pipeline\n\n'
                   'def make_pipeline(task_spec, params):\n'
                   '    return make_ensemble_pipeline(task_spec, params)\n')
MAX_MEMBERS = 4


def build_member(task_spec, member, threads, seed):
    params = {**deepcopy(member['hyperparameters']), '_cpu_threads': threads, '_seed': seed}
    if member['kind'] == 'features':
        from agent.feature_engineering import make_feature_pipeline
        return make_feature_pipeline(task_spec, params)
    from agent.ml import build_pipeline
    return build_pipeline(member['recipe_id'], task_spec, params)


def validate_members(members):
    if not isinstance(members, list) or not 2 <= len(members) <= MAX_MEMBERS:
        raise ValueError(f'An ensemble needs 2..{MAX_MEMBERS} members')
    for member in members:
        if (not isinstance(member, dict) or set(member) != {'kind', 'recipe_id', 'hyperparameters', 'experiment_id'}
                or member['kind'] not in {'owned', 'features'} or not isinstance(member['hyperparameters'], dict)):
            raise ValueError('Ensemble members need kind (owned|features), recipe_id, hyperparameters and experiment_id')
        if member['recipe_id'].startswith('dummy-') or member['recipe_id'] == 'python-pipeline-v1' and member['kind'] != 'features':
            raise ValueError('Ensemble members must be learned owned pipelines')
    return deepcopy(members)


class AveragingEnsemble(BaseEstimator):
    def __init__(self, task_spec=None, members=None, cpu_threads=1, seed=42, validation_policy=None):
        self.task_spec, self.members, self.cpu_threads, self.seed = task_spec, members, cpu_threads, seed
        self.validation_policy = validation_policy

    def fit(self, X, y):
        from agent.supervised import bind_validation_policy
        self.members_ = validate_members(self.members)
        self.models_ = []
        for member in self.members_:
            model = build_member(self.task_spec, member, self.cpu_threads, self.seed)
            bind_validation_policy(model, self.validation_policy)
            model.fit(X, y)
            self.models_.append(model)
        if self.task_spec['task_kind'] != 'regression':
            labels = [str(label) for label in self.task_spec.get('allowed_labels') or []]
            seen = {str(c) for model in self.models_ for c in model.classes_}
            self.classes_ = np.asarray([label for label in labels if label in seen])
        return self

    def _aligned(self, model, X):
        raw = np.asarray(model.predict_proba(X), dtype=float)
        result = np.zeros((len(X), len(self.classes_)))
        for index, label in enumerate(model.classes_):
            result[:, int(np.flatnonzero(self.classes_ == str(label))[0])] = raw[:, index]
        return result

    def predict_proba(self, X):
        if self.task_spec['task_kind'] == 'regression':
            raise TypeError('Regression has no class probabilities')
        return np.mean([self._aligned(model, X) for model in self.models_], axis=0)

    def predict(self, X):
        if self.task_spec['task_kind'] == 'regression':
            # The worker wraps multiple targets outside the model, so members are single-target here.
            return np.mean([np.asarray(model.predict(X), dtype=float).reshape(-1) for model in self.models_], axis=0)
        if self.task_spec['metric']['prediction_kind'] == 'probabilities':
            return self.classes_[self.predict_proba(X).argmax(axis=1)]
        return vote([np.asarray(model.predict(X)).reshape(-1).astype(str) for model in self.models_])


def vote(predictions):
    """Plurality per row; ties go to the earliest (best-ranked) member's label."""
    stacked = np.column_stack(predictions)
    result = stacked[:, 0].copy()
    for row in range(len(stacked)):
        values, counts = np.unique(stacked[row], return_counts=True)
        best = counts.max()
        if best > 1:
            winners = set(values[counts == best])
            result[row] = next(label for label in stacked[row] if label in winners)
    return result


def combine(spec, matrices):
    """The ensemble's output from member prediction matrices (validation-column format)."""
    if spec['task_kind'] == 'regression' or spec['metric']['prediction_kind'] == 'probabilities':
        return np.mean([np.asarray(m, dtype=float) for m in matrices], axis=0)
    return vote([np.asarray(m).reshape(-1).astype(str) for m in matrices]).reshape(-1, 1)


def make_ensemble_pipeline(task_spec, params):
    options = dict(params)
    threads = int(options.pop('_cpu_threads', 1))
    seed = int(options.pop('_seed', 42))
    members = options.pop('members')
    if options:
        raise ValueError('Unknown ensemble parameters: '+', '.join(sorted(options)))
    return AveragingEnsemble(task_spec=task_spec, members=validate_members(members), cpu_threads=threads, seed=seed)
