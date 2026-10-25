"""Within-run wall-time estimates, not benchmark promises or external cost data."""
from collections import defaultdict, deque
from copy import deepcopy
import math


class DurationModel:
    @staticmethod
    def sample_key(plan):
        family = plan['model']['family']
        params=plan['model']['hyperparameters']
        if family=='custom' and params.get('base_recipe_id'):
            import hashlib
            from agent.contracts import canonical
            representation=[params['base_recipe_id'],params.get('feature_program'),params.get('representation_source')]
            return 'representation:'+hashlib.sha256(canonical(representation).encode()).hexdigest()
        if family == 'custom':
            import hashlib
            from agent.contracts import canonical
            params = {k:v for k,v in plan['model']['hyperparameters'].items() if not k.startswith('_')}
            return family+':'+plan['code_artifact_id']+':'+hashlib.sha256(canonical(params).encode()).hexdigest()
        return family

    def __init__(self, config):
        self.config = config
        self.rows = {"train": 1, "refit": 1}
        self.features = 1
        self.target_count = 1
        self.test_rows = 1
        self.defaults = {}
        self.samples = defaultdict(lambda: deque(maxlen=32))

    def configure(self, profile, validation_rows, recipes, target_count=1):
        self.rows = {"train": max(1, validation_rows), "refit": max(1, profile["train_rows"])}
        self.features = max(1, len(profile.get("test_columns", [])))
        self.target_count = max(1, target_count)
        self.test_rows = max(1, profile.get('test_rows', 1))
        self.defaults = {key: {name: domain["default"] for name, domain in recipe["parameters"].items()}
                         for key, recipe in recipes.items()}

    def work(self, plan, phase):
        params = plan["model"]["hyperparameters"]
        family = plan["model"]["family"]
        if family=='custom' and params.get('base_recipe_id'):
            return self.work(self.member_plan(plan,params),phase)
        if family=='custom' and params.get('members'):
            return sum(self.work(self.member_plan(plan,m),phase) for m in params['members'])*(params.get('folds',3)+1)
        rows = self.rows[phase]
        features = self.features
        if params.get("numeric_interactions"):
            numeric = min(8, features)
            features += numeric * (numeric + 1) / 2
        if family == "catboost":
            size = params.get("iterations", 800) * 2 ** params.get("depth", 6)
        elif family == "hist_gradient":
            size = params.get("max_iter", 300) * params.get("max_leaf_nodes", 15)
        elif family=='lightgbm':
            size=params.get('n_estimators',300)*params.get('num_leaves',31)
        elif family=='tabular_net':
            size=params.get('epochs',12)*params.get('width',32)**2*(features if params.get('architecture','transformer')=='transformer' else 1)
        elif family in {"random_forest", "extra_trees"}:
            leaf = max(1, params.get("min_samples_leaf", 1))
            depth = min(params.get("max_depth") or rows, max(1, math.log2(max(2, rows / leaf))))
            subset = params.get("max_features", 1.0)
            if subset == "sqrt":
                subset = 1 / math.sqrt(features)
            elif subset == "log2":
                subset = max(1, math.log2(features)) / features
            size = params.get("n_estimators", 100) * depth * (subset if isinstance(subset, (int, float)) else 1)
        elif family == "linear":
            # C affects convergence, but is not linearly proportional to iterations.
            size = 1 + math.log1p(max(0, params.get("C", 0))) / 5
        else:
            size = 1
        threads = max(1, plan.get("resources", {}).get("cpu_threads", 1))
        return max(1e-12, rows * features * size * self.target_count / math.sqrt(threads))

    def observe(self, plan, phase, seconds, *, fit_seconds=None, state="succeeded"):
        if not math.isfinite(seconds) or seconds < 0:
            return
        censored = state == "timed_out"
        if state != "succeeded" and not censored:
            return
        fit = seconds if fit_seconds is None else min(seconds, max(0, fit_seconds))
        self.samples[self.sample_key(plan)].append({
            "work": self.work(plan, phase), "fit": fit, "overhead": max(0, seconds - fit),
            "censored": censored, "phase": phase,
        })

    def estimate(self, plan, phase="train"):
        params=plan['model']['hyperparameters']
        if plan['model']['family']=='custom' and params.get('members'):
            folds=params.get('folds',3)
            members=[{**plan,'model':{'family':'custom','hyperparameters':m}} for m in params['members']]
            return sum(folds*self.estimate(m,'train')+self.estimate(m,phase) for m in members)
        if plan['model']['family']=='custom' and params.get('base_recipe_id') and not self.samples[self.sample_key(plan)]:
            # Do not compare an owned model's work to the empty custom recipe.
            # Reuse the measured native family as a cold representation prior.
            return 1.5*self.estimate(self.member_plan(plan,params),phase)
        family = plan["model"]["family"]
        work = self.work(plan, phase)
        samples = list(self.samples[self.sample_key(plan)])
        complete = [sample for sample in samples if not sample["censored"]]
        if complete:
            rates = sorted(sample["fit"] / sample["work"] for sample in complete)
            rate = rates[min(len(rates) - 1, math.ceil(.75 * len(rates)) - 1)]
            overheads = sorted(s['overhead'] for s in complete)
            overhead = max(overheads[math.ceil(.75 * len(overheads))-1], complete[-1]['overhead'])
            # Prediction and IO live in measured overhead. Extrapolate them too
            # when test is larger than the fit/selection prediction partition.
            prediction_scale = max(1, self.test_rows/max(1, self.rows['refit']-self.rows['train'])) if phase == 'refit' else 1
            growth = max(1, self.rows['refit']/self.rows['train'])**.25 if phase == 'refit' and family == 'custom' else 1
            seconds = (overhead*prediction_scale + work*rate*growth) * 1.35
        else:
            # A bounded initial probe; subsequent estimates use this family's measurements.
            cold = min(self.config.experiment_timeout_seconds, max(2, self.config.time_budget_seconds * .01))
            if family in {"dummy", "linear"}:
                cold = min(cold, max(2, self.rows[phase] / 50000))
            default = deepcopy(plan)
            default["model"]["hyperparameters"] = self.defaults.get(plan["recipe_id"], plan["model"]["hyperparameters"])
            # Imports and process startup remain even for a very small model.
            measured_startup = max((s['overhead'] for s in self.samples['dummy'] if not s['censored']), default=0)
            seconds = max(2, measured_startup*1.35) + cold * self.target_count * max(.15, work / self.work(default, "train"))
        # A timeout is a lower bound on cost, never a fast successful measurement.
        for sample in samples:
            if sample["censored"]:
                seconds = max(seconds, (sample["overhead"] + sample["fit"] * work / sample["work"]) * 1.5)
        return max(1.0, seconds)

    @staticmethod
    def member_plan(parent,member):
        recipe=member['base_recipe_id']
        prefix=recipe.rsplit('-regressor-',1)[0].rsplit('-classifier-',1)[0]
        family={'hist-native':'hist_gradient','hist-gradient':'hist_gradient','extra-trees':'extra_trees',
                'random-forest':'random_forest','tabular-net':'tabular_net'}.get(prefix,prefix)
        return {**parent,'recipe_id':recipe,'model':{'family':family,'hyperparameters':member['model_parameters']}}
