import json
import re
import uuid

from agent.config import ROOT
from agent.errors import AgentError


def demo_metadata(task_text):
    match = re.search(r"```agent-demo\s*([\s\S]*?)```", task_text)
    return json.loads(match[1]) if match else None


class ContextBuilder:
    def __init__(self, spec, task_text, profile, catalog, artifacts, log, demo=None):
        self.spec, self.task_text, self.profile, self.catalog = spec, task_text, profile, catalog
        self.artifacts, self.log, self.demo = artifacts, log, demo
        self.round_no = 0

    def build(self, payload, snapshot):
        recent = self.artifacts.read_json(snapshot["summary_artifact_id"])
        catalog_mode = self.catalog.config.research_mode == "catalog"
        system = (ROOT / "prompts" / ("researcher-catalog.md" if catalog_mode else "researcher.md")).read_text(encoding="utf-8")
        allowed = list(self.catalog.recipes)
        if catalog_mode:
            allowed = self.catalog.assignment_recipes(payload["assignment"], recent)
        compact_profile = dict(self.profile)
        user = {"task": self.task_text, "task_spec": self.spec, "profile": compact_profile,
                "recipes": [self.catalog.recipes[key] for key in allowed], "recent_experiments": recent,
                "assignment": payload["assignment"], "researcher_id": payload["researcher_id"]}
        messages = [{"role": "system", "content": system}, {"role": "user", "content": json.dumps(user, ensure_ascii=False)}]
        # Conservative byte heuristic, not a claim about the provider tokenizer.
        limit = payload["max_input_tokens"] * 3
        while len(json.dumps(messages, ensure_ascii=False).encode()) > limit and recent:
            recent.pop()
            user["recent_experiments"] = recent
            messages[1]["content"] = json.dumps(user, ensure_ascii=False)
        omitted_profile = []
        for field in ("test_missing_fraction", "test_dtypes", "train_missing_fraction", "train_dtypes", "data_dictionary"):
            if len(json.dumps(messages, ensure_ascii=False).encode()) > limit and field in compact_profile:
                compact_profile.pop(field)
                omitted_profile.append(field)
                messages[1]["content"] = json.dumps(user, ensure_ascii=False)
        if len(json.dumps(messages, ensure_ascii=False).encode()) > limit:
            raise AgentError("CONTEXT_TOO_LARGE", "Task and mandatory context exceed the configured limit")
        metadata = {"schema_id": "ResearchChoice-v1" if catalog_mode else "ResearchProposal-v1", "researcher_id": payload["researcher_id"],
                    "recipe_ids": allowed, "round_no": self.round_no, "demo_spec": self.demo}
        art = self.artifacts.write_json("contexts/" + uuid.uuid4().hex[:12] + ".json",
                                         self.log.redactor.clean({"messages": messages, "metadata": metadata}), "llm_context")
        self.log.event("context.built", context_id=art["artifact_id"], researcher_id=payload["researcher_id"],
                       dropped_profile_fields=omitted_profile, retained_experiments=len(recent), recipe_ids=allowed,
                       estimated_input_tokens=len(json.dumps(messages, ensure_ascii=False).encode()) // 3,
                       context_artifact_path=str(self.artifacts.resolve(art["artifact_id"])))
        return {"context_id": art["artifact_id"], "context_version": self.round_no + 1,
                "messages_artifact_id": art["artifact_id"], "estimated_input_tokens": len(json.dumps(messages, ensure_ascii=False).encode()) // 3,
                "omitted_experiment_count": len(snapshot["recent_experiment_ids"]) - len(recent), "provider_prefix_cache_required": False}


def output_schemas(bus):
    draft = bus.schemas.schema["$defs"]["C02_Response"]["oneOf"][0]["properties"]["data"]["properties"]["draft"]
    validation = bus.schemas.schema["$defs"]["C05_Request"]["properties"]["payload"]["properties"]["proposal"]
    fields = {"recipe_id": {"type": "string", "minLength": 1}, "hyperparameters": {"type": "object"},
              "hypothesis": {"type": "string", "minLength": 1}, "expected_effect": {"type": "string", "minLength": 1},
              "feature_steps": {"type": "array", "items": {"type": "object"}}, "code": {"type": "string", "minLength": 1}}
    choice_fields = {key: value for key, value in fields.items() if key not in {"feature_steps", "code"}}
    return {"TaskDraft-v1": draft, "ValidationProposal-v1": validation,
            "ResearchChoice-v1": {"type": "object", "properties": choice_fields, "required": list(choice_fields), "additionalProperties": False},
            "ResearchProposal-v1": {"type": "object", "properties": fields, "required": list(fields), "additionalProperties": False}}
