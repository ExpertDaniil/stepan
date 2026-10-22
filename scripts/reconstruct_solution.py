"""Materialize one chosen pipeline and a refit job from its pinned owned runtime.

Does not train, load joblib or contact the network. Run the generated command
only in an environment satisfying requirements.lock.txt and package versions.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import sqlite3
import zipfile


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def reconstruct(run_directory, experiment_id, destination, task_root=None):
    root = Path(run_directory).resolve();destination = Path(destination).resolve()
    if destination.exists():
        raise ValueError("Choose a new empty reconstruction directory; existing files are never overwritten")
    db = sqlite3.connect((root / "state.sqlite").as_uri()+"?mode=ro", uri=True);db.row_factory = sqlite3.Row
    try:
        row = db.execute("SELECT e.*,s.manifest_artifact_id FROM experiments e JOIN g_experiment_links l USING(run_id,experiment_id) JOIN g_solutions s USING(run_id,solution_id) WHERE e.experiment_id=?", (experiment_id,)).fetchone()
        if not row:
            raise ValueError("The experiment is absent from this graph")
        run_id = row["run_id"];plan = json.loads(row["plan"])
        def artifact(art_id):
            r = db.execute("SELECT a.*,l.root_kind,l.root_relative_path FROM g_artifacts a JOIN g_artifact_locations l USING(run_id,artifact_id) WHERE a.run_id=? AND a.artifact_id=? AND a.state='ready'", (run_id, art_id)).fetchone()
            if not r or r["root_kind"] != "run":
                raise ValueError("Expected a pinned owned artifact")
            path = (root / r["root_relative_path"]).resolve()
            if not path.is_relative_to(root) or not path.is_file() or path.stat().st_size != r["size_bytes"] or hashlib.sha256(path.read_bytes()).hexdigest() != r["sha256"]:
                raise ValueError("Pinned artifact is missing or corrupt: "+art_id)
            return path
        def metadata(name):
            r = db.execute("SELECT value FROM g_metadata WHERE run_id=? AND name=?", (run_id, name)).fetchone()
            if not r:
                raise ValueError("Required graph metadata is missing: "+name)
            return json.loads(r[0])
        manifest = json.loads(artifact(row["manifest_artifact_id"]).read_text())
        code = artifact(plan["code_artifact_id"]).read_bytes()
        if sha(manifest["core"]) != manifest["solution_hash"] or manifest["solution_id"] != "sol-"+manifest["solution_hash"]:
            raise ValueError("Solution manifest identity mismatch")
        code_hash = hashlib.sha256(code).hexdigest()
        core = {k: plan[k] for k in ("dataset_id", "task_spec_id", "validation_id", "recipe_id", "model", "feature_steps", "seed", "fidelity")}
        core["code_sha256"] = code_hash
        if sha(core) != row["fingerprint"] or code_hash != manifest["core"]["code_sha256"]:
            raise ValueError("Experiment cannot be reconstructed with the pinned code")
        environment = json.loads(artifact(metadata("environment_artifact_id")).read_text())
        runtime = artifact(metadata("runtime_artifact_id"))
        if environment.get("owned_runtime_sha256") != hashlib.sha256(runtime.read_bytes()).hexdigest():
            raise ValueError("Owned runtime does not match the frozen environment")
        contract = json.loads(artifact(metadata("task_contract_artifact_id")).read_text());spec = contract["task_spec"]
        protocol_row = db.execute("SELECT artifact_id FROM g_artifacts WHERE run_id=? AND kind='validation_protocol'", (run_id,)).fetchone()
        protocol = json.loads(artifact(protocol_row[0]).read_text())
        if sha(protocol) != contract["protocol_sha256"]:
            raise ValueError("Frozen validation protocol is corrupt")
        def input_path(art_id):
            r = db.execute("SELECT l.root_relative_path FROM g_artifact_locations l WHERE l.run_id=? AND l.artifact_id=? AND l.root_kind='task'", (run_id, art_id)).fetchone()
            if not r:
                raise ValueError("Input is missing its task-root locator")
            base = Path(task_root or metadata("task_root")).resolve();path = (base/r[0]).resolve()
            if not path.is_relative_to(base):
                raise ValueError("Input locator leaves the supplied task root")
            return str(path)
        train_path,test_path=input_path(spec["train_artifact_id"]),input_path(spec["test_artifact_id"])
        with zipfile.ZipFile(runtime) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)) or sum(i.file_size for i in archive.infolist()) > 100_000_000:
                raise ValueError("Invalid owned runtime archive")
            for info in archive.infolist():
                path = PurePosixPath(info.filename)
                if path.is_absolute() or ".." in path.parts or "\\" in info.filename or info.is_dir() or (info.external_attr >> 16) & 0o170000 == 0o120000 or not info.filename.startswith(("agent/", "tools/", "contracts/", "recipes/", "prompts/")):
                    raise ValueError("Unsafe owned runtime entry")
            destination.mkdir(parents=True, exist_ok=False)
            archive.extractall(destination/"runtime")
        def write(name, value):
            (destination/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
        (destination/"candidate.py").write_bytes(code)
        for name, value in [("plan.json",plan),("manifest.json",manifest),("environment.json",environment),("protocol.json",protocol)]:
            write(name,value)
        write("train_indices.json",protocol["train_indices"]);write("valid_indices.json",protocol["valid_indices"])
        job = {"job_id": "reconstructed-"+experiment_id, "phase": "refit", "task_spec": spec, "train_path": train_path, "test_path": test_path,
               "guard_generated": plan['model']['family'] == 'custom',
               "code_path": str(destination/"candidate.py"), "hyperparameters": plan["model"]["hyperparameters"], "seed": plan["seed"],
               "cpu_threads": plan["resources"]["cpu_threads"], "memory_limit_mb": plan["resources"]["memory_limit_mb"],
               "timeout_seconds": plan["resources"]["timeout_seconds"], "prediction_path": str(destination/"candidate.csv"),
               "model_path": str(destination/"pipeline.joblib"), "result_path": str(destination/"result.json")}
        write("refit-job.json",job)
        inputs = []
        for art_id, path in [(spec["train_artifact_id"], train_path), (spec["test_artifact_id"], test_path)]:
            info = db.execute("SELECT sha256,size_bytes FROM g_artifacts WHERE run_id=? AND artifact_id=?", (run_id, art_id)).fetchone()
            inputs.append({"path": path, "sha256": info[0], "size_bytes": info[1]})
        write("inputs.json", inputs)
        launcher = '''from pathlib import Path
import hashlib
import importlib.metadata
import json
import os
import sys
root = Path(__file__).resolve().parent
env = json.loads((root / "environment.json").read_text())
for name, version in env["package_versions"].items():
    if importlib.metadata.version(name) != version:
        raise RuntimeError("Package version differs from the frozen run: " + name)
for item in json.loads((root / "inputs.json").read_text()):
    path = Path(item["path"])
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    if path.stat().st_size != item["size_bytes"] or value.hexdigest() != item["sha256"]:
        raise RuntimeError("Input differs from the frozen run: " + str(path))
for name in list(os.environ):
    if name.startswith(("OPENAI_", "ANTHROPIC_", "GOOGLE_API_")):
        os.environ.pop(name, None)
sys.path.insert(0, str(root / "runtime"))
sys.argv = ["agent.worker", str(root / "refit-job.json")]
from agent.worker import main
raise SystemExit(main())
'''
        (destination/"run_refit.py").write_text(launcher,encoding="utf-8")
        return {"directory": str(destination), "experiment_id": experiment_id, "solution_id": manifest["solution_id"],
                "run_command": ["python3",str(destination/"run_refit.py")], "executes_training": False,
                "note": "Original input data and approved package versions are required; no packages are installed and no API is called."}
    finally:
        db.close()


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory",type=Path);parser.add_argument("experiment_id")
    parser.add_argument("destination",type=Path);parser.add_argument("--task-root",type=Path)
    args=parser.parse_args()
    print(json.dumps(reconstruct(args.run_directory,args.experiment_id,args.destination,args.task_root),ensure_ascii=False,indent=2))
