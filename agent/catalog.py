from copy import deepcopy
import importlib.util
import json

from agent.config import ROOT
from agent.errors import AgentError


class Catalog:
    def __init__(self, kind, config):
        self.kind, self.config = kind, config
        all_recipes = json.loads((ROOT / "recipes" / "catalog.json").read_text(encoding="utf-8"))
        self.recipes = {r["id"]: r for r in all_recipes if (kind == "regression") == (r["task_kind"] == "regression")
                        and (r.get("requires") is None or importlib.util.find_spec(r["requires"]))}

    def select(self, payload, artifacts, schemas):
        tools = []
        for name, cid, desc, side_effect, timeout in [
            ("read_task", "C01", "Read task Markdown and resolve input files within the task directory.", "none", 10),
            ("profile_data", "C03", "Read technical schema and aggregate missing-value report.", "none", 120),
            ("write_code", "C11", "Save checked Python exporting make_pipeline(task_spec, params).", "artifact_write", 10),
        ]:
            tools.append({"name": name, "description": desc, "input_schema": schemas.schema["$defs"][cid + "_Request"]["properties"]["payload"],
                          "timeout_seconds": timeout, "side_effect": side_effect})
        markdown = "# Tool catalogue v1\n\nEach call has a fresh process only when executed by Runner. Files persist; Python variables do not.\n\n"
        for tool in tools:
            markdown += "## " + tool["name"] + "\n\n" + tool["description"] + "\n\n```json\n" + json.dumps(tool["input_schema"], indent=2) + "\n```\n\n"
        art = artifacts.write_text("task/TOOLS.md", markdown, "tools_catalog", "text/markdown")
        return {"catalog_version": "recipes-v3", "recipe_ids": list(self.recipes), "tools_catalog_version": "tools-v1",
                "tools_markdown_artifact_id": art["artifact_id"], "tools": tools}

    def normalize(self, recipe_id, params):
        if recipe_id not in self.recipes:
            raise AgentError("INVALID_PLAN", f"Unavailable recipe: {recipe_id}")
        recipe = self.recipes[recipe_id]
        if set(params) - set(recipe["parameters"]):
            raise AgentError("INVALID_PLAN", "Unknown hyperparameter", details={"unknown": sorted(set(params) - set(recipe["parameters"]))})
        result = {}
        for name, domain in recipe["parameters"].items():
            value = params.get(name, domain["default"])
            if "enum" in domain and not any((isinstance(value, bool) == isinstance(option, bool)) and value == option for option in domain["enum"]):
                raise AgentError("INVALID_PLAN", f"Invalid {name}")
            if domain.get("type") in {"int", "float"}:
                if isinstance(value, bool) or not isinstance(value, (int, float)) or (domain["type"] == "int" and not isinstance(value, int)):
                    raise AgentError("INVALID_PLAN", f"Invalid numeric type for {name}")
                if not domain["min"] <= value <= domain["max"]:
                    raise AgentError("INVALID_PLAN", f"Out of range: {name}")
            result[name] = value
        return result

    def constrain_params(self, recipe_id, params, constraints):
        from agent.graph.constraints import check
        params = self.normalize(recipe_id, params)
        features = json.loads((ROOT / "recipes/components.json").read_text(encoding="utf-8"))["features"]
        for capability, feature in features.items():
            required = capability in constraints["required_capability_ids"]
            forbidden = capability in constraints["forbidden_capability_ids"]
            if required and feature["parameter"] not in params:
                raise AgentError("UNSUPPORTED_TRANSFER", "Recipe cannot implement " + capability)
            if feature["parameter"] in params and (required or forbidden):
                params[feature["parameter"]] = feature["enabled_value"] if required else feature["disabled_value"]
        params = self.normalize(recipe_id, params)
        caps = ["model."+self.recipes[recipe_id]["family"]] + [cap for cap, feature in features.items() if params.get(feature["parameter"]) == feature["enabled_value"]]
        check({"constraints": constraints, "effective_capability_ids": caps})
        return params

    def allows_constraints(self, recipe_id, constraints):
        try:
            self.constrain_params(recipe_id, {}, constraints)
            return True
        except AgentError:
            return False

    def validate_plan(self, plan, spec, validation_id, artifacts):
        if plan["task_spec_id"] != spec["task_spec_id"] or plan["dataset_id"] != spec["dataset_id"] or plan["validation_id"] != validation_id:
            raise AgentError("STALE_VALIDATION", "Plan references a different task or validation")
        recipe = self.recipes.get(plan["recipe_id"])
        if not recipe or plan["model"]["family"] != recipe["family"]:
            raise AgentError("INVALID_PLAN", "Recipe/model family mismatch")
        plan["model"]["hyperparameters"] = self.normalize(plan["recipe_id"], plan["model"]["hyperparameters"])
        if plan["resources"]["cpu_threads"] > self.config.train_cpu_threads or plan["resources"]["memory_limit_mb"] > self.config.train_memory_mb:
            raise AgentError("INVALID_PLAN", "Resource request exceeds configuration")
        if artifacts.meta(plan["code_artifact_id"])["kind"] != "python_source":
            raise AgentError("INVALID_PLAN", "Expected a Python code artifact")
        return plan

    @staticmethod
    def feature_steps(params):
        result = [{"step": "numeric_interactions", "max_numeric_columns": 8, "fit_scope": "train_only"}] if params.get("numeric_interactions") else []
        if params.get("target_transform") == "shifted_log1p":
            result.append({"step": "target_transform", "name": "shifted_log1p", "fit_scope": "train_only"})
        return result

    def assignment_recipes(self, assignment, recent):
        usable = [key for key, value in self.recipes.items() if value["family"] != "dummy"]
        # Snapshot order is recent-first; the dispatcher supplies the incumbent
        # separately in the assignment context when available.
        focus = next((row.get("recipe_id") for row in recent if row.get("is_incumbent") and row.get("recipe_id") in usable), None)
        focus = focus or next((key for key in usable if key.startswith("catboost-")), usable[0])
        if assignment == "exploit_incumbent":
            return [focus]
        alternatives = [key for key in usable if self.recipes[key]["family"] != self.recipes[focus]["family"]]
        return alternatives or usable

    def warmup_choices(self):
        suffix = "regressor-v1" if self.kind == "regression" else "classifier-v1"
        choices = []
        for family in ("catboost", "hist-gradient", "extra-trees", "linear"):
            key = family + "-" + suffix
            if key not in self.recipes:
                continue
            variants = ["identity", "shifted_log1p"] if self.kind == "regression" and family in {"catboost", "hist-gradient"} else ["identity"]
            for transform in variants:
                params = {"target_transform": transform} if "target_transform" in self.recipes[key]["parameters"] else {}
                choices.append((key, self.normalize(key, params)))
        return choices


def factory_source(recipe_id):
    return 'from agent.ml import build_pipeline\n\ndef make_pipeline(task_spec, params):\n    return build_pipeline(' + repr(recipe_id) + ', task_spec, params)\n'
