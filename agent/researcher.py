import hashlib
import uuid

from agent.contracts import canonical
from tools.code import write_code


class Researcher:
    def __init__(self, bus, gateway, artifacts, spec, catalog, config, log):
        self.bus, self.gateway, self.artifacts, self.spec, self.catalog, self.config, self.log = bus, gateway, artifacts, spec, catalog, config, log
        self.validation_id = None

    def propose(self, payload):
        if not payload["budget"]["search_allowed"]:
            return {"proposal_id": "proposal-" + uuid.uuid4().hex[:12], "researcher_id": payload["researcher_id"],
                    "plan": None, "decision": "stop", "stop_reason": "No search time remains"}
        answer = self.bus.call("C10", {"purpose": "research_proposal", "context_artifact_id": payload["context_id"],
                                        "response_schema_id": "ResearchProposal-v1", "tool_schema_ids": ["write_code"],
                                        "max_output_tokens": self.config.llm_max_output_tokens,
                                        "deadline_seconds": self.config.llm_timeout_seconds, "attempt_limit": 3}, self.gateway.complete)["data"]
        proposed = answer["parsed_payload"]
        params = self.catalog.normalize(proposed["recipe_id"], proposed["hyperparameters"])
        args = {"researcher_id": payload["researcher_id"], "filename": "experiment.py", "content": proposed["code"], "declared_entrypoint": "make_pipeline"}
        code = self.bus.call("C11", args, lambda p: write_code(p, self.artifacts),
                             key="code/" + hashlib.sha256(canonical(args).encode()).hexdigest())["data"]["artifact"]
        plan = {"plan_id": "plan-" + uuid.uuid4().hex[:12], "researcher_id": payload["researcher_id"],
                "parent_ids": payload["base_experiment_ids"], "hypothesis": proposed["hypothesis"],
                "expected_effect": proposed["expected_effect"], "recipe_id": proposed["recipe_id"],
                "task_spec_id": self.spec["task_spec_id"], "dataset_id": self.spec["dataset_id"],
                "validation_id": self.validation_id, "seed": self.config.seed, "fidelity": "full",
                "model": {"family": self.catalog.recipes[proposed["recipe_id"]]["family"], "hyperparameters": params},
                "feature_steps": proposed["feature_steps"], "code_artifact_id": code["artifact_id"],
                "resources": {"cpu_threads": self.config.train_cpu_threads, "memory_limit_mb": self.config.train_memory_mb,
                              "timeout_seconds": self.config.experiment_timeout_seconds}}
        self.catalog.validate_plan(plan, self.spec, self.validation_id, self.artifacts)
        self.log.event("research.hypothesis_proposed", researcher_id=payload["researcher_id"],
                       context_id=payload["context_id"], model_output_artifact_id=answer["output_artifact_id"],
                       hypothesis=plan["hypothesis"], expected_effect=plan["expected_effect"], recipe_id=plan["recipe_id"])
        return {"proposal_id": "proposal-" + uuid.uuid4().hex[:12], "researcher_id": payload["researcher_id"],
                "plan": plan, "decision": "propose", "stop_reason": None}
