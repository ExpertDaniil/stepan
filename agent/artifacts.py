from pathlib import Path
import hashlib
import json
import os
import threading
import uuid

from agent.errors import AgentError


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Artifacts:
    def __init__(self, root, run_id, log):
        self.root, self.run_id, self.log = Path(root).resolve(), run_id, log
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.entries = {}

    def safe_path(self, relative):
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root) or Path(relative).is_absolute():
            raise AgentError("PATH_OUTSIDE_WORKSPACE", "Artifact path must stay inside this run")
        return path

    def register(self, path, kind, media_type="application/json", *, external=False):
        path = Path(path).resolve()
        if not path.is_file():
            raise AgentError("ARTIFACT_MISSING", str(path))
        if not external and not path.is_relative_to(self.root):
            raise AgentError("PATH_OUTSIDE_WORKSPACE", "Output artifact outside run")
        ident = "art-" + uuid.uuid4().hex[:16]
        relative = str(path.relative_to(self.root)) if not external else "inputs/" + ident + "/" + path.name
        meta = {"artifact_id": ident, "run_id": self.run_id, "kind": kind, "relative_path": relative,
                "sha256": file_sha256(path), "size_bytes": path.stat().st_size, "media_type": media_type}
        with self.lock:
            self.entries[ident] = {"meta": meta, "path": str(path), "external": external}
            tmp = self.root / "artifacts-index.json.tmp"
            tmp.write_text(json.dumps(self.entries, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(tmp, self.root / "artifacts-index.json")
        self.log.event("artifact.registered", artifact_id=ident, kind=kind,
                       relative_path=relative, sha256=meta["sha256"], size_bytes=meta["size_bytes"])
        return meta

    def write_text(self, relative, content, kind="text", media_type="text/plain"):
        path = self.safe_path(relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + "." + uuid.uuid4().hex[:8] + ".tmp")
        tmp.write_text(content, encoding="utf-8")
        os.replace(tmp, path)
        return self.register(path, kind, media_type)

    def write_json(self, relative, value, kind="json"):
        return self.write_text(relative, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                               kind, "application/json")

    def resolve(self, ident):
        with self.lock:
            if ident not in self.entries:
                raise AgentError("ARTIFACT_MISSING", f"Unknown artifact: {ident}")
            return Path(self.entries[ident]["path"])

    def meta(self, ident):
        with self.lock:
            return dict(self.entries[ident]["meta"])

    def read_json(self, ident):
        return json.loads(self.resolve(ident).read_text(encoding="utf-8"))
