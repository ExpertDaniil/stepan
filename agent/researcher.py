import hashlib
import json
import uuid

from agent.contracts import canonical
from agent.catalog import factory_source
from agent.errors import AgentError
from tools.code import write_code


class Researcher:
    def __init__(self, bus, gateway, artifacts, spec, catalog, config, log):
        self.bus, self.gateway, self.artifacts, self.spec, self.catalog, self.config, self.log = bus, gateway, artifacts, spec, catalog, config, log
        self.validation_id = None

    def propose(self, payload):
        if not payload["budget"]["search_allowed"]:
            return {"proposal_id": "proposal-" + uuid.uuid4().hex[:12], "researcher_id": payload["researcher_id"],
                    "plan": None, "decision": "stop", "stop_reason": "No search time remains"}
        catalog_mode = self.config.research_mode == "catalog"
        context = self.artifacts.read_json(payload["context_id"])
        answer = self.bus.call("C10", {"purpose": "research_proposal", "context_artifact_id": payload["context_id"],
                                        "response_schema_id": "ResearchChoice-v1" if catalog_mode else "ResearchProposal-v1",
                                        "tool_schema_ids": [] if catalog_mode else ["write_code"],
                                        "max_output_tokens": self.config.llm_max_output_tokens,
                                        "deadline_seconds": self.config.llm_operation_timeout_seconds, "attempt_limit": 4}, self.gateway.complete)["data"]
        proposed = answer["parsed_payload"]
        if catalog_mode and proposed["recipe_id"] not in context["metadata"]["recipe_ids"]:
            raise AgentError("INVALID_PLAN", "Recipe is outside this researcher's assigned catalogue")
        submitted_params = proposed['hyperparameters']
        if catalog_mode and context['metadata'].get('graph_binding'):
            branch_user = json.loads(context['messages'][1]['content'])
            base = branch_user.get('base_result')
            if (base and base['recipe_id'] == proposed['recipe_id']
                    and (branch_user.get('research_phase') == 'refine'
                         or branch_user.get('research_strategy') == 'exploit_incumbent')):
                submitted_params = {**base['hyperparameters'], **submitted_params}
        params = self.catalog.normalize(proposed["recipe_id"], submitted_params)
        constraint_changes = {}
        if context["metadata"].get("graph_binding"):
            branch_user = json.loads(context["messages"][1]["content"])
            effective = self.catalog.constrain_params(proposed["recipe_id"], params, branch_user["restrictions"])
            constraint_changes = {name: {"proposed": value, "effective": effective[name]} for name, value in params.items() if value != effective[name]}
            if constraint_changes:
                self.log.event("research.constraints_applied", researcher_id=payload["researcher_id"], context_id=payload["context_id"],
                               branch_id=branch_user["branch_id"], work_order_id=branch_user["work_order_id"], changes=constraint_changes)
            params = effective
        source = factory_source(proposed["recipe_id"]) if catalog_mode else proposed["code"]
        args = {"researcher_id": payload["researcher_id"], "filename": "experiment.py", "content": source, "declared_entrypoint": "make_pipeline"}
        code = self.bus.call("C11", args, lambda p: write_code(p, self.artifacts),
                             key="code/" + hashlib.sha256(canonical(args).encode()).hexdigest())["data"]["artifact"]
        plan = {"plan_id": "plan-" + uuid.uuid4().hex[:12], "researcher_id": payload["researcher_id"],
                "parent_ids": payload["base_experiment_ids"], "hypothesis": proposed["hypothesis"],
                "expected_effect": proposed["expected_effect"], "recipe_id": proposed["recipe_id"],
                "task_spec_id": self.spec["task_spec_id"], "dataset_id": self.spec["dataset_id"],
                "validation_id": self.validation_id, "seed": self.config.seed, "fidelity": "full",
                "model": {"family": self.catalog.recipes[proposed["recipe_id"]]["family"], "hyperparameters": params},
                "feature_steps": self.catalog.feature_steps(params) if catalog_mode else proposed["feature_steps"], "code_artifact_id": code["artifact_id"],
                "resources": {"cpu_threads": self.config.train_cpu_threads, "memory_limit_mb": self.config.train_memory_mb,
                              "timeout_seconds": self.config.experiment_timeout_seconds}}
        if constraint_changes:
            note = " Effective branch constraints override the proposal: " + canonical(constraint_changes)
            plan["hypothesis"] += note
            plan["expected_effect"] += note
        if context["metadata"].get("graph_binding") and branch_user["restrictions"]["required_capability_ids"]:
            plan["parent_ids"] = list(dict.fromkeys(plan["parent_ids"]+branch_user["allowed_donor_experiment_ids"]))
        self.catalog.validate_plan(plan, self.spec, self.validation_id, self.artifacts)
        self.log.event("research.hypothesis_proposed", researcher_id=payload["researcher_id"],
                       context_id=payload["context_id"], model_output_artifact_id=answer["output_artifact_id"],
                       hypothesis=plan["hypothesis"], expected_effect=plan["expected_effect"], recipe_id=plan["recipe_id"],
                       hyperparameters=params, source_mode=self.config.research_mode, code_artifact_id=code["artifact_id"])
        return {"proposal_id": "proposal-" + uuid.uuid4().hex[:12], "researcher_id": payload["researcher_id"],
                "plan": plan, "decision": "propose", "stop_reason": None}
