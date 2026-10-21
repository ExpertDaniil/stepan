from pathlib import Path
import re

from agent.errors import AgentError


def resolve_inputs(task_path):
    task_path = Path(task_path).resolve()
    if not task_path.is_file():
        raise AgentError("TASK_NOT_FOUND", str(task_path))
    root = task_path.parent
    text = task_path.read_text(encoding="utf-8")
    paths = {}
    for role, filename in [("train", "train.csv"), ("test", "test.csv"),
                           ("dictionary", "data_dictionary.csv"), ("sample_submission", "sample_submission.csv")]:
        explicit = []
        for raw in re.findall(r"`([^`\n]+\.csv)`", text):
            candidate = Path(raw)
            if candidate.name != filename:
                continue
            candidate = (root / candidate).resolve()
            if not candidate.is_relative_to(root):
                raise AgentError("PATH_OUTSIDE_TASK", f"Input path escapes task root: {raw}")
            if candidate.is_file():
                explicit.append(candidate)
        matches = sorted(set(explicit)) if explicit else sorted({p.resolve() for p in root.rglob(filename) if p.is_file()})
        if len(matches) > 1:
            raise AgentError("AMBIGUOUS_PATH", f"Multiple candidates for {filename}", details={"candidates": [str(p) for p in matches]})
        if matches:
            if not matches[0].is_relative_to(root):
                raise AgentError("PATH_OUTSIDE_TASK", "Symlink points outside task root")
            paths[role] = str(matches[0])
        elif role in {"train", "test"}:
            raise AgentError("DATA_NOT_FOUND", f"Could not resolve {filename} inside {root}")
    return text, paths


def read_task(payload, artifacts, log):
    task_path = Path(payload["task_path"]).resolve()
    text, paths = resolve_inputs(task_path)
    art = artifacts.write_text("task/task.md", text, "task_description", "text/markdown")
    log.event("task.paths_resolved", candidate_paths=paths)
    return {"task_text_artifact_id": art["artifact_id"], "task_text": text,
            "task_root": str(task_path.parent), "candidate_paths": paths}
