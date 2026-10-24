"""Export our readable Python modules to a flat platform ZIP.

Every owned Python module is a separate, byte-identical source file. SQL, JSON
and Markdown are readable string literals in stepan_resources.py. The entry
point copies these files into a temporary package tree; no encoded code or
embedded archive is used.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
MAX_ZIP_BYTES = 100_000_000
RESOURCE_FILES = (
    "agent/schema.sql",
    "agent/graph/schema.sql",
    "contracts/contracts.json",
    "contracts/contracts.schema.json",
    "recipes/catalog.json",
    "recipes/components.json",
    "recipes/graph-policy.json",
    "prompts/task-interpreter.md",
    "prompts/validation.md",
    "prompts/researcher.md",
    "prompts/researcher-catalog.md",
    "prompts/researcher-adaptive.md",
    "prompts/researcher-adaptive-legacy.md",
    "prompts/researcher-pipeline.md",
    "prompts/TOOLS.md",
    "recipes/knowledge/manifest.json",
    "recipes/knowledge/cards.json",
    "recipes/knowledge/competitions.json",
    "recipes/knowledge/sources.jsonl",
    "recipes/knowledge/selection.json",
    "recipes/knowledge/UPSTREAM_LICENSE.txt",
)


def checked_bytes(root, relative):
    path = root / relative
    if any(parent.is_symlink() for parent in (path, *path.parents)):
        raise ValueError("Symlinks are not permitted in a submission: " + relative)
    if not path.is_file() or not path.resolve().is_relative_to(root):
        raise ValueError("Missing or unsafe package input: " + relative)
    return path.read_bytes()


def runtime_files(root):
    files = {name: checked_bytes(root, name) for name in RESOURCE_FILES}
    files["recipes/requirements.lock.txt"] = checked_bytes(root, "requirements.organizers.txt")
    for directory in ("agent", "tools"):
        for path in sorted((root / directory).rglob("*")):
            if path.is_symlink():
                raise ValueError("Symlinks are not permitted in a submission")
            if path.is_file() and "__pycache__" not in path.parts and path.suffix == ".py":
                relative = path.relative_to(root).as_posix()
                files[relative] = checked_bytes(root, relative)
    for required in ("agent/__init__.py", "agent/runtime.py", "agent/worker.py", "tools/__init__.py"):
        if required not in files:
            raise ValueError("Missing runtime module: " + required)
    return files


def write_entry(archive, name, content):
    # Fixed timestamps/permissions make builds byte-for-byte reproducible.
    info = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    archive.writestr(info, content, compresslevel=9)


def flat_module_name(relative):
    return relative.replace("/__init__.py", "_init.py").replace("/", "_")


def make_resources(files):
    lines = ['"""Readable SQL, JSON and prompts. Platform copies only root .py files."""',
             "", "RESOURCE_TEXT = {"]
    for name, content in sorted(files.items()):
        if name.endswith(".py"):
            continue
        lines.append("    " + repr(name) + ": (")
        for line in content.decode("utf-8").splitlines(keepends=True):
            lines.append("        " + repr(line))
        if not content:
            lines.append("        ''")
        lines.append("    ),")
    lines.append("}\n")
    source = "\n".join(lines)
    compile(source, "stepan_resources.py", "exec")
    return source.encode("utf-8")


def make_entrypoint(modules):
    mapping = "\n".join("    " + repr(name) + ": " + repr(relative) + ","
                        for name, relative in sorted(modules.items()))
    source = '''"""Platform entry point: python3 solution.py task.md.

The adjacent agent_*.py and tools_*.py files are our original source modules.
They are copied into package directories under /tmp to preserve normal imports.
"""
import time

STARTED_MONOTONIC = time.monotonic()

from pathlib import Path, PurePosixPath
import shutil
import sys
import tempfile

from stepan_resources import RESOURCE_TEXT

SOURCE_MODULES = {
__MODULES__
}


def runtime_path(root, relative):
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts or "\\\\" in relative:
        raise RuntimeError("Unsafe runtime path")
    destination = root / path
    destination.parent.mkdir(parents=True, exist_ok=True)
    return destination


def run():
    source_root = Path(__file__).resolve().parent
    # The platform permits runtime directories, but copies only root .py files.
    with tempfile.TemporaryDirectory(prefix="automl-runtime-", dir="/tmp") as directory:
        root = Path(directory).resolve()
        for filename, relative in SOURCE_MODULES.items():
            source = source_root / filename
            if source.name != filename or source.is_symlink() or not source.is_file():
                raise RuntimeError("Missing or unsafe source module: " + filename)
            shutil.copyfile(source, runtime_path(root, relative))
        for relative, text in RESOURCE_TEXT.items():
            runtime_path(root, relative).write_text(text, encoding="utf-8")
        sys.path.insert(0, str(root))
        try:
            from agent.runtime import main
            return main(started_monotonic=STARTED_MONOTONIC)
        finally:
            sys.path.remove(str(root))


if __name__ == "__main__":
    raise SystemExit(run())
'''
    source = source.replace("__MODULES__", mapping)
    compile(source, "solution.py", "exec")
    return source.encode("utf-8")


def platform_files(files, requirements):
    modules = {}
    entries = {"requirements.txt": requirements, "stepan_resources.py": make_resources(files)}
    for relative, content in sorted(files.items()):
        if not relative.endswith(".py"):
            continue
        name = flat_module_name(relative)
        if name in entries or name == "solution.py":
            raise ValueError("Flat module filename collision: " + name)
        compile(content, relative, "exec")
        modules[name] = relative
        entries[name] = content
    entries["solution.py"] = make_entrypoint(modules)
    return entries


def build(destination, root=ROOT):
    root = Path(root).resolve()
    destination = Path(destination).resolve()
    requirements = checked_bytes(root, "requirements.txt")
    if requirements != checked_bytes(root, "requirements.organizers.txt"):
        raise ValueError("Platform requirements.txt must match the organizers' list exactly")
    files = runtime_files(root)
    entries = platform_files(files, requirements)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".zip", delete=False) as stream:
            temporary = Path(stream.name)
        with zipfile.ZipFile(temporary, "w") as archive:
            for name in ["solution.py", "requirements.txt", *sorted(set(entries) - {"solution.py", "requirements.txt"})]:
                write_entry(archive, name, entries[name])
        with zipfile.ZipFile(temporary) as archive:
            if (set(archive.namelist()) != set(entries) or len(archive.namelist()) != len(entries)
                    or any(Path(name).name != name or not (name.endswith(".py") or name == "requirements.txt")
                           for name in archive.namelist()) or archive.testzip()):
                raise ValueError("Invalid flat platform ZIP")
        if temporary.stat().st_size > MAX_ZIP_BYTES:
            raise ValueError("Platform ZIP exceeds the 100 MB limit")
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return {"path": str(destination), "files": len(entries), "runtime_files": len(files),
            "bytes": destination.stat().st_size, "sha256": hashlib.sha256(destination.read_bytes()).hexdigest()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "dist/solution.zip")
    args = parser.parse_args()
    print(json.dumps(build(args.output), indent=2))
