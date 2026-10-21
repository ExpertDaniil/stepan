import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import uuid

from agent.artifacts import file_sha256
from agent.contracts import canonical
from agent.errors import AgentError


def sync_directory(path):
    if os.name == "posix":
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


class GraphArtifacts:
    def __init__(self, graph, task_root):
        self.g, self.artifacts = graph, graph.artifacts
        self.task_root = Path(task_root).resolve()

    def prepare_pin(self, ident):
        self.g.store._writer()
        meta = self.artifacts.meta(ident)
        path = self.artifacts.resolve(ident).resolve()
        if meta["run_id"] != self.g.run_id:
            raise AgentError("CROSS_RUN_REFERENCE", ident)
        if path.is_relative_to(self.artifacts.root):
            root_kind, base, external = "run", self.artifacts.root, False
        elif path.is_relative_to(self.task_root) and self.artifacts.entries[ident]["external"]:
            root_kind, base, external = "task", self.task_root, True
        else:
            raise AgentError("PATH_OUTSIDE_WORKSPACE", "Untrusted artifact locator")
        if not path.is_file() or path.stat().st_size != meta["size_bytes"] or file_sha256(path) != meta["sha256"]:
            raise AgentError("ARTIFACT_CORRUPT", ident)
        return meta, {"root_kind": root_kind, "root_relative_path": path.relative_to(base).as_posix(), "external": external}

    def pin_tx(self, prepared):
        meta, loc = prepared
        args = (self.g.run_id, meta["artifact_id"])
        old = self.g.db.execute("SELECT * FROM g_artifacts WHERE run_id=? AND artifact_id=?", args).fetchone()
        if old:
            expected = (meta["kind"], meta["relative_path"], meta["sha256"], meta["size_bytes"], meta["media_type"])
            actual = tuple(old[k] for k in ("kind", "relative_path", "sha256", "size_bytes", "media_type"))
            if expected != actual or old["state"] != "ready":
                raise AgentError("ARTIFACT_CORRUPT", "Pinned artifact identity changed")
            return
        self.g.db.execute("INSERT INTO g_artifacts VALUES(?,?,?,?,?,?,?,?)",
                          (*args, meta["kind"], meta["relative_path"], meta["sha256"], meta["size_bytes"], meta["media_type"], "ready"))
        self.g.db.execute("INSERT INTO g_artifact_locations VALUES(?,?,?,?,?)",
                          (*args, loc["root_kind"], loc["root_relative_path"], int(loc["external"])))

    def pin(self, ident):
        prepared = self.prepare_pin(ident)
        with self.g.store.transaction():
            self.pin_tx(prepared)
        return prepared[0]

    def resolve(self, ident, verify=True):
        row = self.g.db.execute("SELECT a.*,l.root_kind,l.root_relative_path,l.external FROM g_artifacts a JOIN g_artifact_locations l USING(run_id,artifact_id) WHERE a.run_id=? AND a.artifact_id=?",
                                (self.g.run_id, ident)).fetchone()
        if not row or row["state"] != "ready":
            raise AgentError("ARTIFACT_NOT_READY", ident)
        base = self.artifacts.root if row["root_kind"] == "run" else self.task_root
        path = (base / row["root_relative_path"]).resolve()
        if not path.is_relative_to(base) or Path(row["root_relative_path"]).is_absolute():
            raise AgentError("PATH_OUTSIDE_WORKSPACE", ident)
        if not path.is_file() or (verify and (path.stat().st_size != row["size_bytes"] or file_sha256(path) != row["sha256"])):
            raise AgentError("ARTIFACT_CORRUPT", ident)
        meta = {k: row[k] for k in ("artifact_id", "run_id", "kind", "relative_path", "sha256", "size_bytes", "media_type")}
        with self.artifacts.lock:
            self.artifacts.entries[ident] = {"meta": meta, "path": str(path), "external": bool(row["external"])}
        return meta

    def put_json(self, value, kind, *, ident=None):
        return self.put_bytes((canonical(value) + "\n").encode("utf-8"), kind, "application/json", ident=ident)

    def put_bytes(self, data, kind, media_type, *, ident=None):
        self.g.store._writer()
        ident = ident or "art-" + uuid.uuid4().hex
        path = self.artifacts.safe_path("_staging/" + ident + ".blob")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        sync_directory(path.parent)
        payload = {"staging_relative_path": path.relative_to(self.artifacts.root).as_posix(), "kind": kind,
                   "media_type": media_type, "expected_sha256": hashlib.sha256(data).hexdigest(), "expected_size_bytes": len(data)}
        return self.g.invoke("G09", payload, key="blob/"+ident)["artifact"]

    def _validate_body(self, path, kind, media_type):
        name = self.g.kind_schemas.get(kind)
        if name and media_type == "application/json":
            value = json.loads(path.read_text(encoding="utf-8"))
            self.g.schemas.validate(value, name)

    def register_staged(self, payload, key):
        sha, kind = payload["expected_sha256"], payload["kind"]
        path = self.artifacts.safe_path(payload["staging_relative_path"])
        staging_root = self.artifacts.safe_path("_staging")
        if not path.is_relative_to(staging_root):
            raise AgentError("INVALID_STAGING_PATH", "Only _staging/ may be published as graph objects")
        destination = self.artifacts.safe_path("objects/"+sha+"/content")
        ident = path.stem if re.fullmatch(r"art-[0-9a-f]{32}", path.stem) else "art-" + hashlib.sha256((kind+sha).encode()).hexdigest()[:32]
        meta = {"artifact_id": ident, "run_id": self.g.run_id, "kind": kind,
                "relative_path": destination.relative_to(self.artifacts.root).as_posix(), "sha256": sha,
                "size_bytes": payload["expected_size_bytes"], "media_type": payload["media_type"]}
        old = self.g.db.execute("SELECT artifact_id FROM g_artifacts WHERE run_id=? AND kind=? AND sha256=? AND state='ready'",
                                (self.g.run_id, kind, sha)).fetchone()
        if old:
            existing = self.resolve(old[0])
            if existing["size_bytes"] != payload["expected_size_bytes"] or existing["media_type"] != payload["media_type"]:
                raise AgentError("ARTIFACT_CORRUPT", "Inconsistent content metadata")
            result = self.g.commit("G09", payload, key, lambda: {"artifact": existing, "disposition": "existing", "ready": True})
            if path.exists():
                path.unlink()
            return result
        for available in [path] if path.exists() else [destination]:
            if not available.is_file() or available.stat().st_size != meta["size_bytes"] or file_sha256(available) != sha:
                raise AgentError("ARTIFACT_CORRUPT", "Staged bytes do not match the declared hash and size")
            self._validate_body(available, kind, meta["media_type"])
        policy = self.g.policy
        used = self.g.db.execute("SELECT COALESCE(SUM(size_bytes),0) FROM (SELECT sha256,MAX(size_bytes) size_bytes FROM g_artifacts WHERE run_id=? AND state='ready' GROUP BY sha256)", (self.g.run_id,)).fetchone()[0]
        if used + meta["size_bytes"] > policy["retention"]["max_artifact_bytes"] or shutil.disk_usage(self.artifacts.root).free < meta["size_bytes"] + policy["retention"]["min_free_disk_bytes"]:
            raise AgentError("ARTIFACT_BUDGET_EXHAUSTED", "Artifact/disk reserve exhausted; preserve existing pinned results")
        intent_id = "intent-" + hashlib.sha256((canonical(payload)+(key or "")).encode()).hexdigest()[:32]
        intent = {"intent_version": "artifact-intent-1.0", "intent_id": intent_id, "run_id": self.g.run_id,
                  "staging_relative_path": payload["staging_relative_path"], "destination_relative_path": meta["relative_path"],
                  "kind": kind, "media_type": meta["media_type"], "sha256": sha, "size_bytes": meta["size_bytes"], "state": "prepared"}
        with self.g.store.transaction():
            self.g.db.execute("INSERT OR IGNORE INTO g_artifact_intents VALUES(?,?,?,?,?)",
                              (self.g.run_id, intent_id, hashlib.sha256(canonical(payload).encode()).hexdigest(), canonical(intent), "prepared"))
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if destination.stat().st_size != meta["size_bytes"] or file_sha256(destination) != sha:
                raise AgentError("ARTIFACT_CORRUPT", "An immutable object is corrupt; refusing to overwrite it")
        elif path.exists():
            os.replace(path, destination)
            sync_directory(destination.parent)
            sync_directory(destination.parent.parent)
        def finish():
            location = {"root_kind": "run", "root_relative_path": meta["relative_path"], "external": False}
            self.pin_tx((meta, location))
            intent["state"] = "ready"
            self.g.db.execute("UPDATE g_artifact_intents SET state='ready',intent_json=? WHERE run_id=? AND intent_id=?",
                              (canonical(intent), self.g.run_id, intent_id))
            self.g.event("G09", "artifact", ident, "ArtifactRef", meta)
            return {"artifact": meta, "disposition": "created", "ready": True}
        result = self.g.commit("G09", payload, key, finish)
        with self.artifacts.lock:
            self.artifacts.entries[ident] = {"meta": meta, "path": str(destination), "external": False}
        if path.exists():
            path.unlink()
        return result

    def recover_intents(self, repair):
        repaired, errors = [], []
        rows = self.g.db.execute("SELECT * FROM g_artifact_intents WHERE run_id=? AND state NOT IN ('ready','failed')", (self.g.run_id,)).fetchall()
        for row in rows:
            value = json.loads(row["intent_json"])
            if not repair:
                errors.append("Incomplete artifact intent: "+row["intent_id"])
                continue
            try:
                self.register_staged({"staging_relative_path": value["staging_relative_path"], "kind": value["kind"],
                    "media_type": value["media_type"], "expected_sha256": value["sha256"], "expected_size_bytes": value["size_bytes"]}, None)
                with self.g.store.transaction():
                    value["state"] = "ready"
                    self.g.db.execute("UPDATE g_artifact_intents SET state='ready',intent_json=? WHERE run_id=? AND intent_id=?", (canonical(value), self.g.run_id, row["intent_id"]))
                repaired.append(row["intent_id"])
            except (AgentError, OSError, ValueError) as exc:
                errors.append(str(exc))
        return repaired, errors
