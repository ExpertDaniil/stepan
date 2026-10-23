"""Within-run wall-time estimates, not benchmark promises or external cost data."""
from collections import defaultdict, deque
from copy import deepcopy
import math


class DurationModel:
    @staticmethod
    def sample_key(plan):
        family = plan['model']['family']
        return family+':'+plan['code_artifact_id'] if family == 'custom' else family

    def __init__(self, config):
        self.config = config
        self.rows = {"train": 1, "refit": 1}
        self.features = 1
        self.target_count = 1
        self.defaults = {}
        self.samples = defaultdict(lambda: deque(maxlen=32))

    def configure(self, profile, validation_rows, recipes, target_count=1):
        self.rows = {"train": max(1, validation_rows), "refit": max(1, profile["train_rows"])}
        self.features = max(1, len(profile.get("test_columns", [])))
        self.target_count = max(1, target_count)
        self.defaults = {key: {name: domain["default"] for name, domain in recipe["parameters"].items()}
                         for key, recipe in recipes.items()}

    def work(self, plan, phase):
        params = plan["model"]["hyperparameters"]
        family = plan["model"]["family"]
        rows = self.rows[phase]
        features = self.features
        if params.get("numeric_interactions"):
            numeric = min(8, features)
            features += numeric * (numeric + 1) / 2
        if family == "catboost":
            size = params.get("iterations", 800) * 2 ** params.get("depth", 6)
        elif family == "hist_gradient":
            size = params.get("max_iter", 300) * params.get("max_leaf_nodes", 15)
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
        family = plan["model"]["family"]
        work = self.work(plan, phase)
        samples = list(self.samples[self.sample_key(plan)])
        complete = [sample for sample in samples if not sample["censored"]]
        if complete:
            rates = sorted(sample["fit"] / sample["work"] for sample in complete)
            rate = rates[min(len(rates) - 1, math.ceil(.75 * len(rates)) - 1)]
            overheads = sorted(s['overhead'] for s in complete)
            overhead = max(overheads[math.ceil(.75 * len(overheads))-1], complete[-1]['overhead'])
            seconds = (overhead + work * rate) * 1.35
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
