"""Small related catalogue trials. No additional agent or optimizer dependency."""
from collections import Counter, deque
import hashlib
import json
import math
import random

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
        self.rowid = 0

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
        item = self.series.get((branch, peer))
        return bool(item and item[0] > 0 and item[1])

    def variants(self, recipe, params):
        """One parameter at a time; cheap alternatives precede heavier trials."""
        for name, domain in self.c.catalog.recipes[recipe]["parameters"].items():
            current = params[name]
            if "enum" in domain:
                values = domain["enum"]
            else:
                lo, hi = domain["min"], domain["max"]
                values = [current * factor for factor in (.5, .8, 1.25, 2)]
                if current == 0:
                    values += [(hi - lo) * .05]
                if domain["type"] == "int":
                    values += [current - 1, current + 1]
                    values = [int(round(max(lo, min(hi, v)))) for v in values]
                else:
                    values = [float(f"{max(lo, min(hi, v)):.8g}") for v in values]
            for value in values:
                if type(value) == type(current) and value == current:
                    continue
                yield {**params, name: value}

    def start_series(self, peer, branch, plan):
        recipe, params = plan["recipe_id"], plan["model"]["hyperparameters"]
        values = deque((recipe, p) for p in self.variants(recipe, params))
        self.series[(branch, peer)] = [self.c.config.research_series_size - 1, values]

    def _probe(self, recipe, params):
        return {"recipe_id": recipe, "model": {"family": self.c.catalog.recipes[recipe]["family"], "hyperparameters": params},
                "resources": self.c.resources()}

    def next_choice(self, peer, branch, base=None, constraints=None):
        self.refresh()
        constraints = constraints or {"required_capability_ids": [], "forbidden_capability_ids": []}
        tried = set()

        def allowed(recipe, params):
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
            return recipe, params

        item = self.series.get((branch, peer))
        if item and item[0] > 0:
            while item[1]:
                recipe, params = item[1].popleft()
                choice = allowed(recipe, params)
                if choice:
                    item[0] -= 1
                    return choice
        self.series.pop((branch, peer), None)
        recipes = [r for r, value in self.c.catalog.recipes.items()
                   if value["family"] != "dummy" and self.c.catalog.allows_constraints(r, constraints)]
        recipes.sort(key=lambda r: self.coverage[r])
        turn = self.sequence[(branch, peer)]
        self.sequence[(branch, peer)] += 1
        if base and base["recipe_id"] in recipes and turn % 2 == 0:
            recipes.remove(base["recipe_id"])
            recipes.insert(0, base["recipe_id"])
        for recipe in recipes:
            params = self.c.catalog.normalize(recipe, base["model"]["hyperparameters"] if base and base["recipe_id"] == recipe else {})
            for candidate in [params, *self.variants(recipe, params)]:
                choice = allowed(recipe, candidate)
                if choice:
                    self.start_series(peer, branch, self._probe(*choice))
                    return choice
        rng = random.Random(hashlib.sha256(f"{self.c.config.seed}/{branch}/{peer}/{turn}".encode()).digest())
        for _ in range(128 if recipes else 0):
            recipe = rng.choice(recipes)
            params = self.c.catalog.normalize(recipe, {})
            domains = self.c.catalog.recipes[recipe]["parameters"]
            name = rng.choice(list(domains))
            domain = domains[name]
            if "enum" in domain:
                params[name] = rng.choice(domain["enum"])
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
