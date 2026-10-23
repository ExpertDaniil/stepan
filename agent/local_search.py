"""Adaptive Optuna authorizations plus the legacy/emergency catalogue fallback."""
from collections import Counter, deque
import hashlib
import json
import math
import random
import time

from agent.catalog import factory_source
from agent.contracts import canonical
from agent.errors import AgentError


class LocalSearch:
    def __init__(self, controller):
        self.c = controller
        self.series = {}
        self.sequence = Counter()
        self.seen = set()
        self.selected = set()
        self.coverage = Counter()
        self.parameter_turn = Counter()
        self.rowid = 0
        self.authorizations = {}
        self.optuna = None
        if controller.config.research_mode == 'adaptive':
            from agent.optuna_search import OptunaSearch
            self.optuna = OptunaSearch(controller)
            self.authorizations = self.optuna.active

    @staticmethod
    def key(recipe, params):
        return recipe, canonical(params)

    def refresh(self):
        for row in self.c.store.db.execute("SELECT rowid,plan FROM experiments WHERE run_id=? AND rowid>? ORDER BY rowid", (self.c.store.run_id, self.rowid)):
            self.rowid = row["rowid"]
            plan = json.loads(row["plan"])
            recipe = plan["recipe_id"]
            if recipe not in self.c.catalog.recipes or plan["seed"] != self.c.config.seed:
                continue
            actual = self.c.artifacts.meta(plan["code_artifact_id"])["sha256"]
            if actual != hashlib.sha256(factory_source(recipe).encode()).hexdigest():
                continue
            self.seen.add(self.key(recipe, self.c.catalog.normalize(recipe, plan["model"]["hyperparameters"])))
            self.coverage[recipe] += 1

    def repeated(self, plan):
        self.refresh()
        digest = hashlib.sha256(factory_source(plan["recipe_id"]).encode()).hexdigest()
        if self.c.artifacts.meta(plan["code_artifact_id"])["sha256"] != digest:
            return False
        return self.key(plan["recipe_id"], plan["model"]["hyperparameters"]) in self.seen

    def has_series(self, peer, branch):
        if self.optuna is not None:
            return self.optuna.has_series(peer, branch)
        if self.c.config.research_mode == 'adaptive':
            auth = self.authorizations.get((branch, peer))
            if not auth or time.monotonic() >= auth['deadline']:
                self.clear_series(peer, branch)
                return False
        item = self.series.get((branch, peer))
        return bool(item and item[0] > 0 and item[1])

    def clear_series(self, peer, branch):
        if self.optuna is not None:
            self.optuna.stop(peer, branch)
            return
        self.series.pop((branch, peer), None)
        self.authorizations.pop((branch, peer), None)

    def clear_all(self, reason='phase_changed'):
        if self.optuna is not None:
            self.optuna.clear_all(reason)
        self.series.clear()
        self.authorizations.clear()

    def refining(self, branch, base):
        if self.c.config.research_mode == 'adaptive':
            return False  # Empirical feature choices remain revisable in the final phase.
        graph = getattr(self.c, 'graph', None)
        phase = graph.refinement() if graph else None
        return bool(phase and phase['branch_id'] == branch and base and base['model']['family'] != 'dummy')

    def feature_parameters(self):
        graph = getattr(self.c, 'graph', None)
        if graph:
            return {feature['parameter'] for feature in graph.constructor.registry['features'].values()}
        return {'target_transform', 'numeric_interactions', 'class_weight'}

    def variants(self, recipe, params, *, numeric_only=False, offset=0, fine=False, parameter_names=None):
        """Interleave axes; do not spend an entire series on the first parameter."""
        axes = []
        locked = self.feature_parameters() if numeric_only else set()
        domains = list(self.c.catalog.recipes[recipe]['parameters'].items())
        domains = [(name, domain) for name, domain in domains if name not in locked]
        if parameter_names is not None:
            domains = [(name, domain) for name, domain in domains if name in parameter_names]
        if domains:
            offset %= len(domains)
            domains = domains[offset:] + domains[:offset]
        for name, domain in domains:
            current = params[name]
            if "enum" in domain:
                values = domain["enum"]
            else:
                lo, hi = domain["min"], domain["max"]
                values = [current * factor for factor in (.5, .8, 1.25, 2)]
                if current == 0:
                    values += [(hi - lo) * .05]
                if domain["type"] == "int":
                    if fine or current <= 16:
                        values += [current - 1, current + 1]
                    values = [int(round(max(lo, min(hi, v)))) for v in values]
                else:
                    values = [float(f"{max(lo, min(hi, v)):.8g}") for v in values]
            axis = []
            seen = set()
            for value in values:
                if type(value) == type(current) and value == current:
                    continue
                key = canonical(value)
                if key not in seen:
                    axis.append({**params, name: value});seen.add(key)
            axes.append(axis)
        for index in range(max((len(axis) for axis in axes), default=0)):
            for axis in axes:
                if index < len(axis):
                    yield axis[index]

    def start_series(self, peer, branch, plan, *, request=None):
        if self.optuna is not None:
            if request is not None:
                self.optuna.start_series(peer, branch, plan, request)
            return
        adaptive = self.c.config.research_mode == 'adaptive'
        if adaptive and request is None:
            return
        if adaptive:
            if request['budget_seconds'] <= 0:
                return
            self.authorizations[(branch, peer)] = {'deadline': time.monotonic()+request['budget_seconds'],
                'parameters': list(request['parameters']),
                'fixed_parameters': {k: v for k, v in plan['model']['hyperparameters'].items() if k not in request['parameters']}}
        recipe, params = plan["recipe_id"], plan["model"]["hyperparameters"]
        values = deque((recipe, p) for p in self.variants(recipe, params, numeric_only=self.refining(branch, plan),
                       parameter_names=request['parameters'] if adaptive else None))
        self.series[(branch, peer)] = [(request['max_trials'] if adaptive else self.c.config.research_series_size) - 1, values, recipe, params]

    def measured_base(self, peer, branch, recipe, fallback):
        graph = getattr(self.c, 'graph', None)
        if not graph:
            return fallback
        best, chosen = None, fallback
        for eid in graph.visible_history(branch, peer):
            exp = self.c.store.get(eid)
            if exp['state'] != 'evaluated' or exp['plan']['recipe_id'] != recipe:
                continue
            if not graph.head_satisfies({'solution_id': graph.experiment_solution(eid)}, graph.branch(branch)['constraints']):
                continue
            authorization = self.authorizations.get((branch, peer), {})
            if any(exp['plan']['model']['hyperparameters'].get(k) != v
                   for k, v in authorization.get('fixed_parameters', {}).items()):
                continue
            phase = graph.refinement()
            if self.c.config.research_mode != 'adaptive' and phase and phase['branch_id'] == branch and any(
                    exp['plan']['model']['hyperparameters'].get(name) != fallback.get(name)
                    for name in self.feature_parameters()):
                continue
            ev = self.c.store.experiment_evaluation(eid)
            if not ev or not ev['valid'] or ev['violations']:
                continue
            metric = ev['primary_metric']
            value = metric['value'] * (1 if metric['direction'] == 'maximize' else -1)
            if best is None or value > best:
                best, chosen = value, exp['plan']['model']['hyperparameters']
        return chosen

    def _probe(self, recipe, params):
        return {"recipe_id": recipe, "model": {"family": self.c.catalog.recipes[recipe]["family"], "hyperparameters": params},
                "resources": self.c.resources()}

    def next_choice(self, peer, branch, base=None, constraints=None, *, assignment=None):
        if self.optuna is not None and self.optuna.has_series(peer, branch):
            plan = self.optuna.next_plan(peer, branch, constraints)
            return (plan['recipe_id'], plan['model']['hyperparameters']) if plan else None
        self.refresh()
        constraints = constraints or {"required_capability_ids": [], "forbidden_capability_ids": []}
        tried = set()
        final = self.refining(branch, base)
        adaptive = self.c.config.research_mode == 'adaptive'
        authorized = self.has_series(peer, branch) if adaptive else False
        feature_parameters = self.feature_parameters() if final else set()

        def variants(recipe, params):
            return self.variants(recipe, params, numeric_only=final,
                                 offset=self.parameter_turn[(branch, peer, recipe)], fine=final,
                                 parameter_names=self.authorizations[(branch, peer)]['parameters'] if authorized else None)

        def allowed(recipe, params):
            if final and params == base['model']['hyperparameters']:
                return None
            if final and (recipe != base['recipe_id'] or any(params.get(name) != base['model']['hyperparameters'].get(name) for name in feature_parameters)):
                return None
            try:
                params = self.c.catalog.constrain_params(recipe, params, constraints)
            except AgentError:
                return None
            key = self.key(recipe, params)
            if key in self.seen or key in self.selected or key in tried:
                return None
            tried.add(key)
            if not self.c.budget.can_train(self._probe(recipe, params))["allowed"]:
                return None
            self.selected.add(key)
            self.parameter_turn[(branch, peer, recipe)] += 1
            return recipe, params

        item = self.series.get((branch, peer))
        if item and item[0] > 0:
            # Rebase every continuation on this peer's measured family best.
            # A regressed LLM proposal must not anchor three more blind trials.
            recipe = item[2]
            params = self.measured_base(peer, branch, recipe, item[3])
            item[1] = deque((recipe, p) for p in variants(recipe, params))
            while item[1]:
                recipe, params = item[1].popleft()
                choice = allowed(recipe, params)
                if choice:
                    item[0] -= 1
                    return choice
        self.clear_series(peer, branch)
        authorized = False
        if adaptive and self.c.llm_available():
            return None  # Let the planner choose another action; no implicit follow-up.
        recipes = [r for r, value in self.c.catalog.recipes.items()
                   if value["family"] not in {'dummy', 'custom'} and (not final or r == base['recipe_id']) and self.c.catalog.allows_constraints(r, constraints)]
        recipes.sort(key=lambda r: self.coverage[r])
        turn = self.sequence[(branch, peer)]
        self.sequence[(branch, peer)] += 1
        exploit = self.c.config.graph_exploit_steps
        keep_base = final or (assignment != 'explore_alternative' and turn % (exploit+1) < exploit)
        if base and base["recipe_id"] in recipes and keep_base:
            recipes.remove(base["recipe_id"])
            recipes.insert(0, base["recipe_id"])
        elif base and len(recipes) > 1:
            recipes = [recipe for recipe in recipes if recipe != base['recipe_id']]
        for recipe in recipes:
            fallback = base['model']['hyperparameters'] if base and base['recipe_id'] == recipe else {}
            params = self.c.catalog.normalize(recipe, self.measured_base(peer, branch, recipe, fallback))
            for candidate in [params, *variants(recipe, params)]:
                choice = allowed(recipe, candidate)
                if choice:
                    self.start_series(peer, branch, self._probe(*choice))
                    return choice
        rng = random.Random(hashlib.sha256(f"{self.c.config.seed}/{branch}/{peer}/{turn}".encode()).digest())
        for _ in range(128 if recipes else 0):
            recipe = rng.choice(recipes)
            params = self.c.catalog.normalize(recipe, base['model']['hyperparameters'] if final else {})
            domains = self.c.catalog.recipes[recipe]["parameters"]
            names = [name for name in domains if not final or name not in feature_parameters]
            if not names:
                return None
            name = rng.choice(names)
            domain = domains[name]
            if "enum" in domain:
                params[name] = rng.choice(domain["enum"])
            elif final:
                value = params[name]*rng.uniform(.8, 1.25)
                value = max(domain['min'], min(domain['max'], value))
                params[name] = int(round(value)) if domain['type'] == 'int' else float(f'{value:.8g}')
            elif domain["type"] == "int":
                params[name] = rng.randint(domain["min"], domain["max"])
            else:
                lo, hi = domain["min"], domain["max"]
                params[name] = math.exp(rng.uniform(math.log(lo), math.log(hi))) if lo > 0 else rng.uniform(lo, hi)
            choice = allowed(recipe, params)
            if choice:
                self.start_series(peer, branch, self._probe(*choice))
                return choice
        return None
