import hashlib
import json
import uuid

from agent.contracts import canonical
from agent.catalog import factory_source
from agent.errors import AgentError
from tools.code import write_code
from agent.context import research_schema
from agent.research_actions import validate_action


class Researcher:
    def __init__(self, bus, gateway, artifacts, spec, catalog, config, log):
        self.bus, self.gateway, self.artifacts, self.spec, self.catalog, self.config, self.log = bus, gateway, artifacts, spec, catalog, config, log
        self.validation_id = None

    def propose(self, payload):
        if not payload["budget"]["search_allowed"]:
            return {"proposal_id": "proposal-" + uuid.uuid4().hex[:12], "researcher_id": payload["researcher_id"],
                    "plan": None, "decision": "stop", "stop_reason": "No search time remains"}
        adaptive = self.config.research_mode == 'adaptive'
        catalog_mode = self.config.research_mode == "catalog"
        context = self.artifacts.read_json(payload["context_id"])
        answer = self.bus.call("C10", {"purpose": "research_proposal", "context_artifact_id": payload["context_id"],
                                        "response_schema_id": research_schema(self.config.research_mode),
                                        "tool_schema_ids": (['analyze_data'] + (['search_knowledge'] if getattr(self.gateway, 'knowledge', None) else [])) if adaptive else [] if catalog_mode else ["write_code"],
                                        "max_output_tokens": self.config.llm_max_output_tokens,
                                        "deadline_seconds": self.config.llm_operation_timeout_seconds, "attempt_limit": 8 if adaptive else 4}, self.gateway.complete)["data"]
        proposed = answer["parsed_payload"]
        if adaptive:
            validate_action(proposed, self.catalog)
            user = json.loads(context['messages'][1]['content'])
            evidence = {f['source_id'] for f in user.get('facts', [])}
            for fact in user.get('facts', []):
                evidence.update(fact.get('artifact_ids', []))
            evidence.update(r['experiment_id'] for r in user.get('recent_experiments', []))
            evidence.update(r['experiment_id'] for r in user.get('profile_summary', {}).get('measured_search', {}).get('entries', []))
            evidence.update(r['artifact_id'] for r in user.get('profile_summary', {}).get('analysis_history', []))
            evidence.update(r['artifact_id'] for r in user.get('analysis_history', []))
            evidence.update(r['artifact_id'] for r in user.get('research_feedback', []))
            evidence.update(r['artifact_id'] for r in user.get('profile_summary', {}).get('research_feedback', []))
            evidence.update(call['artifact_id'] for call in answer['tool_calls'] if call.get('artifact_id') and call.get('name') != 'search_knowledge')
            if not set(proposed['evidence_ids']) <= evidence:
                raise AgentError('INVALID_RESEARCH_ACTION', 'Research evidence must reference visible measured results or completed analyses')
            knowledge = user.get('profile_summary', user).get('knowledge', {})
            visible_knowledge = {r['id'] for r in knowledge.get('results', [])}
            visible_knowledge.update(ident for call in answer['tool_calls'] for ident in call.get('knowledge_ids', []))
            if not set(proposed.get('knowledge_ids', [])) <= visible_knowledge:
                raise AgentError('INVALID_RESEARCH_ACTION', 'Unseen knowledge_ids: '+', '.join(sorted(set(proposed.get('knowledge_ids', []))-visible_knowledge))+
                                 '. Read those items using search_knowledge before citing them, or use knowledge_ids=[].')
            if proposed['action'] == 'stop':
                self.log.event('research.stop_requested', researcher_id=payload['researcher_id'], reason=proposed['hypothesis'])
                return {'proposal_id': 'proposal-'+uuid.uuid4().hex[:12], 'researcher_id': payload['researcher_id'],
                        'plan': None, 'decision': 'stop', 'stop_reason': proposed['hypothesis']}
            catalog_mode = proposed['action'] != 'pipeline'
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
        if adaptive and proposed['action'] == 'tune':
            plan['resources']['timeout_seconds'] = min(plan['resources']['timeout_seconds'], proposed['tuning']['budget_seconds'])
        if context["metadata"].get("graph_binding") and branch_user["restrictions"]["required_capability_ids"]:
            plan["parent_ids"] = list(dict.fromkeys(plan["parent_ids"]+branch_user["allowed_donor_experiment_ids"]))
        self.catalog.validate_plan(plan, self.spec, self.validation_id, self.artifacts)
        if adaptive:
            self.artifacts.write_json('research/intent-'+plan['plan_id']+'.json',
                {**{k: proposed[k] for k in ['action', 'tuning', 'evidence_ids', 'hypothesis', 'expected_effect']},
                 'knowledge_ids': proposed.get('knowledge_ids', []),
                 'knowledge_retrieval_artifact_ids': ([knowledge['artifact_id']] if knowledge.get('artifact_id') else [])+
                    [call['artifact_id'] for call in answer['tool_calls'] if call.get('name') == 'search_knowledge' and call.get('artifact_id')]},
                'research_intent', ident='intent-'+plan['plan_id'])
            self.log.event('research.action_proposed', plan_id=plan['plan_id'], researcher_id=payload['researcher_id'],
                           action=proposed['action'], tuning=proposed['tuning'], evidence_ids=proposed['evidence_ids'],
                           knowledge_ids=proposed.get('knowledge_ids', []))
        self.log.event("research.hypothesis_proposed", researcher_id=payload["researcher_id"],
                       context_id=payload["context_id"], model_output_artifact_id=answer["output_artifact_id"],
                       hypothesis=plan["hypothesis"], expected_effect=plan["expected_effect"], recipe_id=plan["recipe_id"],
                       hyperparameters=params, source_mode=self.config.research_mode, code_artifact_id=code["artifact_id"])
        return {"proposal_id": "proposal-" + uuid.uuid4().hex[:12], "researcher_id": payload["researcher_id"],
                "plan": plan, "decision": "propose", "stop_reason": None}
