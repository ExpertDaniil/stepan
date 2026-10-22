"""Owner B: task semantics and validation hypothesis, through the shared gateway."""
import json
import uuid

from agent.config import ROOT


class TaskInterpreter:
    def __init__(self, config, artifacts, bus, gateway, task, profile, demo=None):
        self.config, self.artifacts, self.bus, self.gateway = config, artifacts, bus, gateway
        self.task, self.profile, self.demo = task, profile, demo

    def ask(self, schema_id, purpose, prompt_file, content):
        messages = [{"role": "system", "content": (ROOT / "prompts" / prompt_file).read_text(encoding="utf-8")},
                    {"role": "user", "content": json.dumps(content, ensure_ascii=False)}]
        context = self.artifacts.write_json("contexts/bootstrap-" + uuid.uuid4().hex[:10] + ".json",
                    self.gateway.log.redactor.clean({"messages": messages, "metadata": {"schema_id": schema_id, "demo_spec": self.demo}}), "llm_context")
        return self.bus.call("C10", {"purpose": purpose, "context_artifact_id": context["artifact_id"],
                                     "response_schema_id": schema_id, "tool_schema_ids": ['inspect_schema'] if getattr(self.gateway, 'tools', None) and self.gateway.tools.inspector is not None else [],
                                     "max_output_tokens": self.config.llm_bootstrap_output_tokens,
                                     "deadline_seconds": self.config.llm_operation_timeout_seconds, "attempt_limit": 3}, self.gateway.complete)["data"]["parsed_payload"]

    def interpret(self, payload):
        draft = self.ask("TaskDraft-v1", "task_interpretation", "task-interpreter.md",
                         {"task_markdown": self.task["task_text"], "candidate_paths": payload["candidate_paths"],
                          "profile": self.profile["summary"], "task_root": self.task["task_root"]})
        return {"draft": draft, "evidence": ["Source task artifact: " + self.task["task_text_artifact_id"]], "unresolved_fields": []}

    def choose_validation(self, spec):
        return self.ask("ValidationProposal-v1", "result_analysis", "validation.md",
                        {"task_markdown": self.task["task_text"], "task_spec": spec, "profile": self.profile["summary"]})
