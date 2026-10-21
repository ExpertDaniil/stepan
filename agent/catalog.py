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
        return {"catalog_version": "recipes-v1", "recipe_ids": list(self.recipes), "tools_catalog_version": "tools-v1",
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
            if "enum" in domain and value not in domain["enum"]:
                raise AgentError("INVALID_PLAN", f"Invalid {name}")
            if domain.get("type") in {"int", "float"}:
                if isinstance(value, bool) or not isinstance(value, (int, float)) or (domain["type"] == "int" and not isinstance(value, int)):
                    raise AgentError("INVALID_PLAN", f"Invalid numeric type for {name}")
                if not domain["min"] <= value <= domain["max"]:
                    raise AgentError("INVALID_PLAN", f"Out of range: {name}")
            result[name] = value
        return result

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


def factory_source(recipe_id):
    return 'from agent.ml import build_pipeline\n\ndef make_pipeline(task_spec, params):\n    return build_pipeline(' + repr(recipe_id) + ', task_spec, params)\n'
