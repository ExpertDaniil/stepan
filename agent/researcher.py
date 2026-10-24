import hashlib
import json
import time
import uuid

from agent.contracts import canonical
from agent.catalog import factory_source
from agent.errors import AgentError
from tools.code import write_code
from agent.context import research_schema
from agent.research_actions import validate_action, validate_research_notes, validate_test_plan


class Researcher:
    def __init__(self, bus, gateway, artifacts, spec, catalog, config, log, feature_columns=None):
        self.bus, self.gateway, self.artifacts, self.spec, self.catalog, self.config, self.log = bus, gateway, artifacts, spec, catalog, config, log
        self.validation_id = None
        if self.gateway is not None:
            self.gateway.proposal_validator = self.validate_adaptive
        self.feature_columns = list(feature_columns) if feature_columns is not None else None

    def propose(self, payload):
        if not payload["budget"]["search_allowed"]:
            return {"proposal_id": "proposal-" + uuid.uuid4().hex[:12], "researcher_id": payload["researcher_id"],
                    "plan": None, "decision": "stop", "stop_reason": "No search time remains"}
        adaptive = self.config.research_mode == 'adaptive'
        catalog_mode = self.config.research_mode == "catalog"
        context = self.artifacts.read_json(payload["context_id"])
        investigation_check = None
        answer = self.bus.call("C10", {"purpose": "research_proposal", "context_artifact_id": payload["context_id"],
                                        "response_schema_id": research_schema(self.config.research_mode),
                                        "tool_schema_ids": (['analyze_data', 'run_analysis', 'read_memory'] + (['search_knowledge'] if getattr(self.gateway, 'knowledge', None) else [])) if adaptive else [] if catalog_mode else ["write_code"],
                                        "max_output_tokens": self.config.llm_max_output_tokens,
                                        "deadline_seconds": self.config.llm_operation_timeout_seconds,
                                        "attempt_limit": self.config.llm_research_tool_turns + 2 if adaptive else 4}, self.gateway.complete)["data"]
        proposed = answer["parsed_payload"]
        if adaptive:
            checked = self.validate_adaptive(proposed, context, answer)
            user, program = checked['user'], checked['program']
            memory, evidence, knowledge = checked['memory'], checked['evidence'], checked['knowledge']
            protocol, landscape = checked['protocol'], checked['landscape']
            investigation_check = checked['investigation_check']
            if proposed['action'] in {'stop', 'research'}:
                self.save_decision(proposed, None, payload, context, answer, investigation_check=investigation_check)
                self.log.event('research.stop_requested', researcher_id=payload['researcher_id'], reason=proposed['hypothesis'])
                return {'proposal_id': 'proposal-'+uuid.uuid4().hex[:12], 'researcher_id': payload['researcher_id'],
                        'plan': None, 'decision': ('research' if proposed['action'] == 'research' and proposed.get('next_step', 'continue') == 'continue' else 'stop'), 'stop_reason': proposed['hypothesis']}
            catalog_mode = proposed['action'] != 'pipeline' and proposed['recipe_id'] != 'python-pipeline-v1'
        if catalog_mode and proposed["recipe_id"] not in context["metadata"]["recipe_ids"]:
            raise AgentError("INVALID_PLAN", "Recipe is outside this researcher's assigned catalogue")
        submitted_params = proposed['hyperparameters']
        custom_base = None
        if adaptive and proposed['action'] == 'tune' and not catalog_mode and proposed['code'] is None:
            custom_base = user.get('base_result')
            if not custom_base or custom_base['recipe_id'] != 'python-pipeline-v1' or not user.get('base_code'):
                raise AgentError('INVALID_RESEARCH_ACTION', 'To tune custom source, provide code or reuse the current measured custom base_code')
            submitted_params = {**custom_base['hyperparameters'], **submitted_params}
        if catalog_mode and context['metadata'].get('graph_binding'):
            branch_user = user if adaptive else json.loads(context['messages'][1]['content'])
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
        plan_recipe = proposed['recipe_id']
        if (adaptive and proposed.get('investigation', {}).get('interpretation') in {'physical', 'mathematical'}
                and params.get('numeric_interactions') is True):
            raise AgentError('INVALID_INVESTIGATION', 'The effective inherited parameters still enable automatic numeric_interactions; use explicit named operations or an exploratory interpretation')
        feature_manifest = None
        if adaptive and proposed['action'] == 'features':
            from agent.feature_schema import FACTORY_SOURCE, operation_facts
            params = {'base_recipe_id': plan_recipe, 'model_parameters': params, 'feature_program': program}
            plan_recipe = 'python-pipeline-v1'
            source = FACTORY_SOURCE
            feature_manifest = [{'step': 'owned_feature_program', 'program': program, 'fit_scope': 'train_only',
                'execution_specification': operation_facts(program)}]
        else:
            source = factory_source(proposed["recipe_id"]) if catalog_mode else user['base_code'] if custom_base else proposed["code"]
        args = {"researcher_id": payload["researcher_id"], "filename": "experiment.py", "content": source, "declared_entrypoint": "make_pipeline"}
        code = self.bus.call("C11", args, lambda p: write_code(p, self.artifacts),
                             key="code/" + hashlib.sha256(canonical(args).encode()).hexdigest())["data"]["artifact"]
        plan = {"plan_id": "plan-" + uuid.uuid4().hex[:12], "researcher_id": payload["researcher_id"],
                "parent_ids": payload["base_experiment_ids"], "hypothesis": proposed["hypothesis"],
                "expected_effect": proposed["expected_effect"], "recipe_id": plan_recipe,
                "task_spec_id": self.spec["task_spec_id"], "dataset_id": self.spec["dataset_id"],
                "validation_id": self.validation_id, "seed": self.config.seed, "fidelity": "full",
                "model": {"family": self.catalog.recipes[plan_recipe]["family"], "hyperparameters": params},
                "feature_steps": feature_manifest if feature_manifest is not None else self.catalog.feature_steps(params) if catalog_mode else custom_base['feature_steps'] if custom_base else proposed["feature_steps"], "code_artifact_id": code["artifact_id"],
                "resources": {"cpu_threads": self.config.train_cpu_threads, "memory_limit_mb": self.config.train_memory_mb,
                              "timeout_seconds": self.config.experiment_timeout_seconds}}
        if constraint_changes:
            note = " Effective branch constraints override the proposal: " + canonical(constraint_changes)
            plan["hypothesis"] += note
            plan["expected_effect"] += note
        if adaptive and proposed['action'] == 'tune':
            from agent.tuning_space import normalize_space
            normalize_space(self.catalog, plan, proposed['tuning'], user.get('restrictions'))
            plan['resources']['timeout_seconds'] = min(plan['resources']['timeout_seconds'], proposed['tuning']['budget_seconds'])
        if context["metadata"].get("graph_binding") and branch_user["restrictions"]["required_capability_ids"]:
            plan["parent_ids"] = list(dict.fromkeys(plan["parent_ids"]+branch_user["allowed_donor_experiment_ids"]))
        self.catalog.validate_plan(plan, self.spec, self.validation_id, self.artifacts)
        if adaptive:
            self.artifacts.write_json('research/intent-'+plan['plan_id']+'.json',
                {**{k: proposed[k] for k in ['action', 'tuning', 'evidence_ids', 'hypothesis', 'expected_effect']},
                 'knowledge_ids': proposed.get('knowledge_ids', []),
                 'research_notes': proposed.get('research_notes'),
                 'scientific_protocol': protocol,
                 'strongest_measured_reference': landscape.get('strongest_measured_learned', {}).get('experiment_id') if landscape.get('strongest_measured_learned') else None,
                 'investigation': proposed.get('investigation'),
                 'investigation_check': investigation_check,
                 'coordination': proposed.get('coordination', {}),
                 'claims': proposed.get('claims', []), 'visible_evidence_ids': sorted(evidence),
                 'delivered_feedback': [c for c in answer.get('tool_calls', []) if c.get('name') == 'research_updates'],
                 'knowledge_retrieval_artifact_ids': list(dict.fromkeys(
                    ([knowledge['artifact_id']] if knowledge.get('artifact_id') else [])+
                    [call['artifact_id'] for call in answer['tool_calls'] if call.get('name') == 'search_knowledge' and call.get('artifact_id')]))},
                'research_intent', ident='intent-'+plan['plan_id'])
            self.save_decision(proposed, plan, payload, context, answer, investigation_check=investigation_check)
            self.log.event('research.action_proposed', plan_id=plan['plan_id'], researcher_id=payload['researcher_id'],
                           action=proposed['action'], tuning=proposed['tuning'], evidence_ids=proposed['evidence_ids'],
                           knowledge_ids=proposed.get('knowledge_ids', []))
        self.log.event("research.hypothesis_proposed", researcher_id=payload["researcher_id"],
                       context_id=payload["context_id"], model_output_artifact_id=answer["output_artifact_id"],
                       hypothesis=plan["hypothesis"], expected_effect=plan["expected_effect"], recipe_id=plan["recipe_id"],
                       hyperparameters=params, source_mode=self.config.research_mode, code_artifact_id=code["artifact_id"])
        return {"proposal_id": "proposal-" + uuid.uuid4().hex[:12], "researcher_id": payload["researcher_id"],
                "plan": plan, "decision": "propose", "stop_reason": None}

    def save_decision(self, proposed, plan, payload, context, answer, *, investigation_check=None):
        """Immutable artifact only; SQLite/measurement updates stay in the dispatcher."""
        binding = context['metadata'].get('graph_binding', {})
        configuration = None
        if plan is not None:
            from agent.store import Store
            configuration = {'plan_id': plan['plan_id'], 'parent_ids': list(plan['parent_ids']),
                'code_artifact_id': plan['code_artifact_id'],
                'code_sha256': self.artifacts.meta(plan['code_artifact_id'])['sha256'],
                'dataset_id': plan['dataset_id'], 'validation_id': plan['validation_id'],
                'fingerprint_core': Store.fingerprint_core(plan, self.artifacts.meta(plan['code_artifact_id'])['sha256'])}
        record = {key: proposed[key] for key in ('action', 'hypothesis', 'expected_effect', 'evidence_ids')}
        record['evidence_ids'] = list(dict.fromkeys([*record['evidence_ids'],
            *(proposed.get('research_notes') or {}).get('test_plan', {}).get('evidence_ids', [])]))
        record.update(research_notes=proposed.get('research_notes'), knowledge_ids=proposed.get('knowledge_ids', []),
            created_monotonic=time.monotonic(),
            researcher_id=payload['researcher_id'],
            tuning=proposed['tuning'],
            context_id=payload['context_id'],
            work_order_id=binding.get('work_order_id'), branch_id=binding.get('branch_id'),
            output_artifact_id=answer['output_artifact_id'], configuration=configuration)
        record['delivered_feedback'] = [call for call in answer.get('tool_calls', []) if call.get('name') == 'research_updates']
        record['coordination'] = proposed.get('coordination', {})
        with self.artifacts.lock:
            output_kind = self.artifacts.entries.get(answer['output_artifact_id'], {}).get('meta', {}).get('kind')
        record['decision_origin'] = ('controller' if output_kind == 'controller_research_checkpoint'
                                     else 'model' if output_kind == 'llm_output' else 'unknown_legacy')
        record['next_step'] = proposed.get('next_step')
        record['claims'] = proposed.get('claims', [])
        record['model_interpretations_verified'] = False
        record['semantic_updates'] = proposed.get('semantic_updates', [])
        record['investigation'] = proposed.get('investigation')
        record['investigation_check'] = investigation_check
        record['semantic_scope'] = context['metadata'].get('detail_scope')
        if record['semantic_updates']:
            self.log.event('research.semantic_update_saved', context_id=payload['context_id'],
                updates=len(record['semantic_updates']), verification='model_statement_unverified',
                scope=record['semantic_scope'])
        art = self.artifacts.write_json('research/decision-'+uuid.uuid4().hex[:16]+'.json', record, 'research_decision')
        self.log.event('research.decision_saved', artifact_id=art['artifact_id'], researcher_id=payload['researcher_id'],
            action=proposed['action'], notes_supplied=proposed.get('research_notes') is not None,
            test_plan_supplied=bool((proposed.get('research_notes') or {}).get('test_plan')),
            continuation=(proposed.get('research_notes') or {}).get('test_plan', {}).get('continuation'),
            work_order_id=binding.get('work_order_id'))
        if record['investigation']:
            self.log.event('research.investigation_saved', artifact_id=art['artifact_id'],
                question=record['investigation']['question'], basis_ids=record['investigation']['basis_ids'],
                columns=record['investigation']['columns'], interpretation=record['investigation']['interpretation'],
                check=record['investigation_check'], plan_id=plan['plan_id'] if plan else None)

    def validate_adaptive(self, proposed, context, answer):
        program = None
        validate_action(proposed, self.catalog)
        from agent.data_research import require_action, require_feature_parameters
        window = self.gateway.budget.data_research_snapshot()
        require_action(proposed, window)
        user = json.loads(context['messages'][1]['content'])
        detail = context['metadata'].get('detail_artifact_id')
        if detail:
            sections = self.artifacts.read_json(detail)['sections']
            if not user.get('base_code'):
                user['base_code'] = sections.get('base_code')
            if user.get('profile_summary', {}).get('base_result_details'):
                user['base_result'] = sections.get('base_result')
        require_feature_parameters(proposed, self.catalog, window, user.get('restrictions'))
        from agent.feature_search import validate_focus
        validate_focus(proposed, user)
        if proposed['action'] == 'features':
            from agent.feature_schema import validate_program
            from agent.predictions import targets
            profile = user.get('profile_summary', {}).get('data', user.get('profile', {}))
            excluded = [*targets(self.spec), *self.spec['excluded_features']]
            try:
                program = validate_program(proposed['feature_program'],
                    ([c for c in self.feature_columns if c not in excluded] if self.feature_columns is not None
                     else [c for c in profile['train_columns'] if c not in excluded] if 'train_columns' in profile
                     else None), excluded)
            except (ValueError, KeyError) as error:
                raise AgentError('INVALID_FEATURE_PROGRAM', str(error)) from error
        memory = user.get('profile_summary', user).get('research_memory', {})
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
        for delivery in answer['tool_calls']:
            if delivery.get('name') == 'research_updates':
                evidence.update(delivery['evidence_ids'])
        projection = user.get('profile_summary', user)
        protocol = projection.get('scientific_protocol', {})
        for report in protocol.get('fit_diagnostics', []):
            evidence.add(report['artifact_id'])
        episode = protocol.get('episode')
        if episode:
            evidence.update(episode['evidence_ids'])
            evidence.update(episode['measured_experiment_ids'])
            memory = {**memory, 'open_ideas': [*memory.get('open_ideas', []), episode]}
        semantic_task = projection.get('semantic_task', {})
        if semantic_task.get('model_id'):
            evidence.add(semantic_task['model_id'])
        evidence.update(item['record_id'] for field in ('quantities', 'relations') for item in semantic_task.get(field, []))
        evidence.update(item['record_id'] for item in memory.get('semantic_records', []))
        landscape = projection.get('research_landscape', {})
        for result in [*landscape.get('best_per_family', []), *projection.get('feature_memory', {}).get('representations', [])]:
            evidence.add(result['experiment_id'])
        if landscape.get('strongest_measured_learned'):
            evidence.add(landscape['strongest_measured_learned']['experiment_id'])
        for comparison in projection.get('comparison_evidence', {}).get('comparisons', []):
            evidence.update(comparison.get('experiment_ids', []))
        for decision in memory.get('decisions', []):
            evidence.update(decision['evidence_ids'])
            for observation in decision['observations']:
                evidence.add(observation['experiment_id'])
                if observation.get('diagnostics_artifact_id'):
                    evidence.add(observation['diagnostics_artifact_id'])
        for idea in memory.get('open_ideas', []):
            evidence.update(idea['evidence_ids'])
        for delivery in answer['tool_calls']:
            if delivery.get('name') == 'read_memory':
                evidence.update(delivery.get('evidence_ids', []))
                memory.setdefault('open_ideas', []).extend({'idea_id': ident, 'evidence_ids': [], 'knowledge_ids': []} for ident in delivery.get('idea_ids', []))
        evidence.update(i for i in (self.spec.get('dictionary_artifact_id'), self.spec.get('task_spec_id')) if i)
        from agent.research_memory import validate_semantic_updates
        detail_id = context['metadata'].get('detail_artifact_id')
        semantic_records = self.artifacts.read_json(detail_id)['sections']['semantics'] if detail_id else []
        from agent.semantic_planning import validate_investigation, check_declared_semantics, effective_model
        investigation_check = validate_investigation(proposed.get('investigation'), context, evidence,
            self.feature_columns, self.bus.schemas, program=proposed.get('feature_program', []))
        if (proposed.get('investigation', {}).get('interpretation') in {'physical', 'mathematical'}
                and (proposed['hyperparameters'].get('numeric_interactions') is True
                     or 'numeric_interactions' in (proposed.get('tuning') or {}).get('search_space', {})
                     or 'numeric_interactions' in (proposed.get('tuning') or {}).get('parameters', []))):
            raise AgentError('INVALID_INVESTIGATION', 'Automatic numeric_interactions cannot substantiate named physical/mathematical interactions; use an explicit feature_program or an exploratory interpretation')
        frozen_model = self.artifacts.read_json(detail_id)['sections'].get('semantic_model') if detail_id else None
        validate_semantic_updates(proposed.get('semantic_updates', []), memory, evidence,
                                  (self.feature_columns + [step['name'] for step in proposed.get('feature_program', [])]
                                   if self.feature_columns is not None else None),
                                  context['metadata'].get('detail_scope'), semantic_records)
        provisional = list(semantic_records)
        for index, update in enumerate(proposed.get('semantic_updates', [])):
            provisional.append({'record_id': 'provisional-'+str(index), 'content': update,
                                'status': 'active', 'verification': 'model_statement_unverified'})
        superseded = {u.get('supersedes') for u in proposed.get('semantic_updates', []) if u.get('supersedes')}
        provisional = [r for r in provisional if r['record_id'] not in superseded]
        investigation_check['warnings'] = check_declared_semantics(proposed.get('feature_program', []),
            proposed.get('investigation'), effective_model(frozen_model, provisional))
        if not set(proposed['evidence_ids']) <= evidence:
            raise AgentError('INVALID_RESEARCH_ACTION', 'Research evidence must reference visible measured results or completed analyses')
        knowledge = user.get('profile_summary', user).get('knowledge', {})
        visible_knowledge = {r['id'] for r in knowledge.get('results', [])}
        visible_knowledge.update(ident for call in answer['tool_calls'] for ident in call.get('knowledge_ids', []))
        for idea in memory.get('open_ideas', []):
            visible_knowledge.update(idea['knowledge_ids'])
        if not set(proposed.get('knowledge_ids', [])) <= visible_knowledge:
            raise AgentError('INVALID_RESEARCH_ACTION', 'Unseen knowledge_ids: '+', '.join(sorted(set(proposed.get('knowledge_ids', []))-visible_knowledge))+
                             '. Read those items using search_knowledge before citing them, or use knowledge_ids=[].')
        validate_research_notes(proposed.get('research_notes'), memory, evidence, visible_knowledge)
        validate_test_plan(proposed, protocol)
        if proposed['action'] not in {'stop', 'research'}:
            params = proposed['hyperparameters']
            if proposed['action'] == 'tune' and proposed['recipe_id'] == 'python-pipeline-v1' and proposed['code'] is None:
                if not user.get('base_code') or not user.get('base_result') or user['base_result']['recipe_id'] != 'python-pipeline-v1':
                    raise AgentError('INVALID_RESEARCH_ACTION', 'Custom tuning without source needs a measured custom base')
                params = {**user['base_result']['hyperparameters'], **params}
            elif user.get('base_result') and user['base_result']['recipe_id'] == proposed['recipe_id'] and (user.get('research_phase') == 'refine' or user.get('research_strategy') == 'exploit_incumbent'):
                params = {**user['base_result']['hyperparameters'], **params}
            normalized = self.catalog.normalize(proposed['recipe_id'], params)
            if user.get('restrictions'):
                normalized = self.catalog.constrain_params(proposed['recipe_id'], normalized, user['restrictions'])
            if proposed.get('investigation', {}).get('interpretation') in {'physical','mathematical'} and normalized.get('numeric_interactions') is True:
                raise AgentError('INVALID_INVESTIGATION', 'Inherited numeric_interactions has unnamed inputs; disable it or use exploratory interpretation')
            if proposed['action'] == 'tune':
                from agent.tuning_space import normalize_space
                normalize_space(self.catalog, {'recipe_id': proposed['recipe_id'],
                    'model': {'hyperparameters': normalized}}, proposed['tuning'], user.get('restrictions'))
            if proposed['code']:
                from tools.code import validate_code
                validate_code(proposed['code'])
        return dict(user=user, program=program, memory=memory, evidence=evidence, knowledge=knowledge, protocol=protocol, landscape=landscape, investigation_check=investigation_check)
