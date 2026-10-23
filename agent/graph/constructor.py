from copy import deepcopy
import hashlib
import importlib.metadata
import json
import platform
import io
import zipfile

from agent.artifacts import file_sha256
from agent.catalog import factory_source
from agent.config import ROOT
from agent.errors import AgentError
from agent.graph.identity import component, dependency_closure, digest, solution


class Constructor:
    """Only the exact owned factory is catalog_verified; AST is not a semantic proof."""
    def __init__(self, catalog, artifacts, run_id):
        self.catalog, self.artifacts, self.run_id = catalog, artifacts, run_id
        self.registry = json.loads((ROOT / "recipes/components.json").read_text(encoding="utf-8"))
        self.implementation_hash = digest({p: file_sha256(ROOT / p) for p in self.registry["implementation_resources"]})
        self.provenance = None
        self.environment = None
        self.catalog_manifest = None
        self.runtime_archive = None

    def snapshot_runtime(self, requirements_bytes):
        resources = ["agent/schema.sql", "agent/graph/schema.sql", "contracts/contracts.json", "contracts/contracts.schema.json",
                     "recipes/catalog.json", "recipes/components.json", "recipes/graph-policy.json"]
        paths = [ROOT / relative for relative in resources]
        paths += list((ROOT / "prompts").glob("*.md"))
        from tools.knowledge import KNOWLEDGE_RESOURCES
        paths += [ROOT / relative for relative in KNOWLEDGE_RESOURCES if (ROOT / relative).is_file()]
        for directory in ("agent", "tools"):
            paths += [p for p in (ROOT / directory).rglob("*.py") if "__pycache__" not in p.parts]
        files = {}
        for path in paths:
            if path.is_symlink() or not path.resolve().is_relative_to(ROOT):
                raise AgentError("UNSAFE_RUNTIME_RESOURCE", "Owned runtime resource must be a regular in-root file")
            files[path.relative_to(ROOT).as_posix()] = path.read_bytes()
        files["recipes/requirements.lock.txt"] = requirements_bytes
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            for relative, content in sorted(files.items()):
                entry = zipfile.ZipInfo(relative, (2026, 1, 1, 0, 0, 0))
                entry.compress_type = zipfile.ZIP_DEFLATED;entry.external_attr = 0o100644 << 16
                archive.writestr(entry, content, compresslevel=9)
        return self.artifacts.write_bytes("graph/owned-runtime.zip", buffer.getvalue(), "owned_runtime", "application/zip")

    def prepare(self):
        lock = ROOT / "recipes/requirements.lock.txt"
        lock = lock if lock.is_file() else ROOT / "requirements.organizers.txt"
        self.runtime_archive = self.snapshot_runtime(lock.read_bytes())
        requirements = self.artifacts.write_text("graph/requirements.lock.txt", lock.read_text(encoding="utf-8"), "requirements_lock")
        versions = {}
        for package in ("numpy", "pandas", "scipy", "scikit-learn", "catboost", "lightgbm", "xgboost", "optuna", "jsonschema", "joblib"):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                pass
        environment = {"environment_version": "environment-1.0", "python_version": platform.python_version(),
                       "requirements_artifact_id": requirements["artifact_id"], "requirements_sha256": requirements["sha256"],
                       "package_versions": versions, "entrypoint": "solution.py", "container_image_digest": None,
                       "owned_runtime_artifact_id": self.runtime_archive["artifact_id"], "owned_runtime_sha256": self.runtime_archive["sha256"]}
        self.environment = self.artifacts.write_json("graph/environment.json", environment, "environment_manifest")
        provenance = {"provenance_version": "provenance-1.0", "origin": "builtin", "package_name": "owned-catalogue",
                      "package_version": self.registry["catalog_version"], "trained_from_scratch": False,
                      "initial_weights": "none", "training_code_artifact_id": None, "training_dataset_sha256": [],
                      "training_seed": None, "weights_artifact_id": None,
                      "notes": "Unfitted owned factories using approved estimators. No pretrained weights are bundled."}
        self.provenance = self.artifacts.write_json("graph/provenance.json", provenance, "component_provenance")
        components = []
        for recipe_id in self.catalog.recipes:
            components.extend(self.components(recipe_id, self.catalog.normalize(recipe_id, {})))
        by_id = {c["component_id"]: self.external_component(c) for c in components}
        manifest = {"catalog_version": self.registry["catalog_version"], "constructor_version": self.registry["constructor_version"],
                    "recipe_ids": list(self.catalog.recipes), "components": list(by_id.values()),
                    "provenance_artifact_ids": [self.provenance["artifact_id"]]}
        self.catalog_manifest = self.artifacts.write_json("graph/catalog.json", manifest, "component_catalog")
        return self.environment, self.catalog_manifest

    def components(self, recipe_id, params):
        features = self.registry["features"]
        model_params = {k: v for k, v in params.items() if k not in {f["parameter"] for f in features.values()}}
        model = component({"capability_id": "model." + self.catalog.recipes[recipe_id]["family"], "version": "1.0",
                           "implementation_sha256": self.implementation_hash, "parameters": {"recipe_id": recipe_id, **model_params},
                           "dependency_component_ids": [], "origin": "builtin", "provenance_sha256": self.provenance["sha256"]})
        result = [model]
        if self.catalog.recipes[recipe_id]['family'] == 'custom':
            return result  # Never certify generated feature semantics from declared parameter names.
        for capability, feature in sorted(features.items()):
            if params.get(feature["parameter"]) == feature["enabled_value"]:
                result.append(component({"capability_id": capability, "version": "1.0", "implementation_sha256": self.implementation_hash,
                                         "parameters": {"value": feature["enabled_value"]}, "dependency_component_ids": [],
                                         "origin": "builtin", "provenance_sha256": self.provenance["sha256"]}))
        return result

    def external_component(self, core):
        return {k: v for k, v in core.items() if k != "provenance_sha256"} | {"provenance_artifact_id": self.provenance["artifact_id"]}

    def build(self, plan):
        params = self.catalog.normalize(plan["recipe_id"], plan["model"]["hyperparameters"])
        code = self.artifacts.resolve(plan["code_artifact_id"]).read_bytes()
        known = code == factory_source(plan["recipe_id"]).encode("utf-8")
        if known and plan["feature_steps"] != self.catalog.feature_steps(params):
            raise AgentError("MANIFEST_MISMATCH", "feature_steps must describe the actual factory behavior")
        core = {"recipe_id": plan["recipe_id"], "recipe_version": "recipes-v3", "code_sha256": hashlib.sha256(code).hexdigest(),
                "model_family": self.catalog.recipes[plan["recipe_id"]]["family"], "effective_hyperparameters": params,
                "effective_feature_steps": deepcopy(plan["feature_steps"]), "components": self.components(plan["recipe_id"], params),
                "environment_lock_sha256": self.environment["sha256"], "constructor_version": self.registry["constructor_version"],
                "candidate_interface_version": "make_pipeline-2arg-v1"}
        identity = solution({"core": core})
        return {"manifest_version": "solution-1.0", "run_id": self.run_id,
                "solution_id": identity["solution_id"], "solution_hash": identity["solution_hash"], "core": core,
                "component_provenance_artifact_ids": [{"component_id": c["component_id"], "artifact_id": self.provenance["artifact_id"]} for c in core["components"]],
                "verification_level": "catalog_verified" if known else "declared_only"}

    def verify(self, payload):
        plan = payload["plan"]
        actual = self.artifacts.meta(plan["code_artifact_id"])
        if actual != payload["code_artifact"] or actual["run_id"] != self.run_id:
            raise AgentError("MANIFEST_MISMATCH", "Code reference does not match this run")
        if file_sha256(self.artifacts.resolve(actual["artifact_id"])) != actual["sha256"]:
            raise AgentError("ARTIFACT_CORRUPT", "Code changed after it was registered")
        if payload["catalog_manifest_artifact_id"] != self.catalog_manifest["artifact_id"]:
            raise AgentError("MANIFEST_MISMATCH", "The component catalogue is frozen for this run")
        manifest = self.build(plan)
        if manifest != payload["manifest"]:
            raise AgentError("MANIFEST_MISMATCH", "Manifest does not describe the actual normalized executable")
        caps, closure = dependency_closure(manifest["core"]["components"])
        return {"manifest": manifest, "effective_capability_ids": caps, "component_dependency_closure_ids": closure}

    def constrain(self, recipe_id, params, constraints):
        params = self.catalog.normalize(recipe_id, params)
        for capability, feature in self.registry["features"].items():
            required = capability in constraints["required_capability_ids"]
            forbidden = capability in constraints["forbidden_capability_ids"]
            if required and feature["parameter"] not in params:
                raise AgentError("UNSUPPORTED_TRANSFER", "Recipe does not support " + capability)
            if feature["parameter"] in params and (required or forbidden):
                params[feature["parameter"]] = feature["enabled_value"] if required else feature["disabled_value"]
        return self.catalog.normalize(recipe_id, params)

    def patch_params(self, recipe_id, params, patch):
        if recipe_id not in patch["supported_recipe_ids"]:
            raise AgentError("UNSUPPORTED_TRANSFER", "Patch is not registered for this recipe")
        result = self.catalog.normalize(recipe_id, params)
        for operation in patch["operations"]:
            c = operation["component"]
            capability = c["capability_id"] if c else None
            feature = self.registry["features"].get(capability)
            if not feature or operation["operation"] != "add_component":
                raise AgentError("UNSUPPORTED_TRANSFER", "Only verified portable feature additions are supported")
            if feature["parameter"] not in result or c["parameters"] != {"value": feature["enabled_value"]}:
                raise AgentError("UNSUPPORTED_TRANSFER", "Invalid feature patch")
            result[feature["parameter"]] = feature["enabled_value"]
        return self.catalog.normalize(recipe_id, result)
