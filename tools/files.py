from pathlib import Path
import os
import re

from agent.errors import AgentError

IGNORED_SEARCH_DIRS = {".git", ".venv", "venv", "__pycache__", "runs", "dist", "build", ".pytest_cache", "node_modules"}


def find_task_files(directory):
    """List task descriptions without reading CSVs or generated run workspaces."""
    directory = Path(directory).resolve()
    result = []
    for current, dirs, files in os.walk(directory, followlinks=False):
        dirs[:] = sorted(name for name in dirs if name not in IGNORED_SEARCH_DIRS and not (Path(current) / name).is_symlink())
        if "task.md" in files:
            path = (Path(current) / "task.md").resolve()
            if path.is_relative_to(directory):
                result.append(path)
    return sorted(result)


def find_named_inputs(root, filename):
    matches = []
    for current, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = sorted(name for name in dirs if name not in IGNORED_SEARCH_DIRS and not (Path(current) / name).is_symlink())
        if filename in files:
            matches.append((Path(current) / filename).resolve())
    return sorted(set(matches))


def choose_input(matches, filename, root, task_path):
    if len(matches) > 1:
        candidates = [str(p.relative_to(root)) if p.is_relative_to(root) else str(p) for p in matches]
        tasks = [str(p.relative_to(root)) for p in find_task_files(root) if p != task_path]
        message = (f"Multiple candidates for {filename}: " + ", ".join(candidates[:8])
                   + ". Run one task directory: python3 solution.py <task-directory>/task.md. "
                     "List task paths with: python3 scripts/list_tasks.py <data-directory>")
        raise AgentError("AMBIGUOUS_PATH", message, details={"candidates": candidates, "task_paths": tasks,
                                                           "task_root": str(root)})
    if matches:
        if not matches[0].is_relative_to(root):
            raise AgentError("PATH_OUTSIDE_TASK", "Symlink points outside task root")
        return str(matches[0])
    return None


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
        if explicit:
            matches = sorted(set(explicit))
        elif role in {"train", "test"}:
            # Adjacent files win over unrelated datasets in descendant directories.
            direct = root / filename
            matches = [direct.resolve()] if direct.is_file() else find_named_inputs(root, filename)
        else:
            # Optional metadata belongs to the selected input directory or task
            # root. Never recursively borrow another dataset's sample/dictionary.
            parents = {Path(paths[key]).parent for key in ("train", "test") if key in paths}
            matches = sorted({(parent / filename).resolve() for parent in parents if (parent / filename).is_file()})
            if not matches and (root / filename).is_file():
                matches = [(root / filename).resolve()]
        selected = choose_input(matches, filename, root, task_path)
        if selected:
            paths[role] = selected
        elif role in {"train", "test"}:
            raise AgentError("DATA_NOT_FOUND", f"Could not resolve {filename} inside {root}. Pass task.md from the dataset directory.")
    return text, paths


def read_task(payload, artifacts, log):
    task_path = Path(payload["task_path"]).resolve()
    text, paths = resolve_inputs(task_path)
    art = artifacts.write_text("task/task.md", text, "task_description", "text/markdown")
    log.event("task.paths_resolved", candidate_paths=paths)
    return {"task_text_artifact_id": art["artifact_id"], "task_text": text,
            "task_root": str(task_path.parent), "candidate_paths": paths}
