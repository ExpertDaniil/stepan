"""Build a flat ZIP containing only solution.py and official requirements.txt.

The modular source tree stays in Git. The generated entry point expands only our
own runtime sources/resources under /tmp. It contains no third-party packages,
datasets, credentials or pre-trained models.
"""
from pathlib import Path
import argparse
import base64
import hashlib
import io
import json
import os
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
MAX_ZIP_BYTES = 100_000_000
RESOURCE_FILES = (
    "agent/schema.sql",
    "contracts/contracts.json",
    "contracts/contracts.schema.json",
    "recipes/catalog.json",
    "prompts/task-interpreter.md",
    "prompts/validation.md",
    "prompts/researcher.md",
    "prompts/TOOLS.md",
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


def make_entrypoint(files):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, content in sorted(files.items()):
            write_entry(archive, name, content)
    bundle = buffer.getvalue()
    encoded = base64.b85encode(bundle).decode("ascii")
    chunks = "\n".join("    " + repr(encoded[i:i + 100]) for i in range(0, len(encoded), 100))
    source = '''"""Generated platform entry point. Edit modular Git sources and rebuild."""
import time

STARTED_MONOTONIC = time.monotonic()

import base64
import hashlib
import io
from pathlib import Path, PurePosixPath
import sys
import tempfile
import zipfile

BUNDLE_SHA256 = __DIGEST__
BUNDLE = (
__CHUNKS__
)


def run():
    payload = base64.b85decode(BUNDLE)
    if hashlib.sha256(payload).hexdigest() != BUNDLE_SHA256:
        raise RuntimeError("Corrupted runtime bundle")
    # Runtime subdirectories in /tmp are explicitly permitted by the platform.
    with tempfile.TemporaryDirectory(prefix="automl-runtime-", dir="/tmp") as directory:
        root = Path(directory).resolve()
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)):
                raise RuntimeError("Duplicate runtime file")
            for info in archive.infolist():
                relative = PurePosixPath(info.filename)
                if (relative.is_absolute() or ".." in relative.parts or "\\\\" in info.filename
                        or info.is_dir() or (info.external_attr >> 16) & 0o170000 == 0o120000):
                    raise RuntimeError("Unsafe runtime file")
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(archive.read(info))
        sys.path.insert(0, str(root))
        try:
            from agent.runtime import main
            return main(started_monotonic=STARTED_MONOTONIC)
        finally:
            sys.path.remove(str(root))


if __name__ == "__main__":
    raise SystemExit(run())
'''
    source = source.replace("__DIGEST__", repr(hashlib.sha256(bundle).hexdigest())).replace("__CHUNKS__", chunks)
    compile(source, "solution.py", "exec")
    return source.encode("utf-8")


def build(destination, root=ROOT):
    root = Path(root).resolve()
    destination = Path(destination).resolve()
    requirements = checked_bytes(root, "requirements.txt")
    if requirements != checked_bytes(root, "requirements.organizers.txt"):
        raise ValueError("Platform requirements.txt must match the organizers' list exactly")
    files = runtime_files(root)
    entrypoint = make_entrypoint(files)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".zip", delete=False) as stream:
            temporary = Path(stream.name)
        with zipfile.ZipFile(temporary, "w") as archive:
            write_entry(archive, "solution.py", entrypoint)
            write_entry(archive, "requirements.txt", requirements)
        with zipfile.ZipFile(temporary) as archive:
            if archive.namelist() != ["solution.py", "requirements.txt"] or archive.testzip():
                raise ValueError("Invalid flat platform ZIP")
        if temporary.stat().st_size > MAX_ZIP_BYTES:
            raise ValueError("Platform ZIP exceeds the 100 MB limit")
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return {"path": str(destination), "files": 2, "runtime_files": len(files),
            "bytes": destination.stat().st_size, "sha256": hashlib.sha256(destination.read_bytes()).hexdigest()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "dist/solution.zip")
    args = parser.parse_args()
    print(json.dumps(build(args.output), indent=2))
