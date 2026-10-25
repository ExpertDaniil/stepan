from copy import deepcopy
from pathlib import Path
import json
import tempfile
import uuid

from agent.artifacts import Artifacts
from agent.config import Config, ROOT
from agent.contracts import ContractBus, Schemas
from agent.context import demo_metadata
from agent.observability.logging import RunLogger
from agent.store import Store
from agent.task import resolve_spec
from scripts.make_demo import make_demo
from tools.code import write_code
from tools.data import profile_data


class Harness:
    def __init__(self, kind="classification", with_periods=False, with_duplicates=False):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.task = make_demo(self.root / "task", kind, rows=90, test_rows=12)
        if with_periods:
            import pandas as pd
            for name in ("train.csv", "test.csv"):
                path = self.task.parent / name
                frame = pd.read_csv(path, dtype={"id": str}, keep_default_na=False)
                frame["period"] = [i // 15 for i in range(len(frame))]
                frame.to_csv(path, index=False)
        if with_duplicates:
            import pandas as pd
            path = self.task.parent / "train.csv"
            frame = pd.read_csv(path, dtype={"id": str}, keep_default_na=False)
            features = [c for c in frame.columns if c not in {"id", "label", "target"}]
            # Every feature row appears three times; labels and ids stay as generated.
            frame[features] = frame[features].iloc[[3*(i//3) for i in range(len(frame))]].to_numpy()
            frame.to_csv(path, index=False)
        self.run = self.root / "run"
        self.config = Config(llm_mode="mock", research_mode='catalog', work_dir=self.root, time_budget_seconds=90, log_level="ERROR")
        self.log = RunLogger(self.run, "unit-run", level="ERROR", secrets=["unit-private-secret"])
        self.artifacts = Artifacts(self.run, "unit-run", self.log)
        self.schemas = Schemas()
        self.bus = ContractBus("unit-run", self.log, self.schemas)
        self.store = Store(self.run / "state.sqlite", "unit-run", self.config.public_dict(), self.log)
        self.profile = profile_data({"candidate_paths": {"train": str(self.task.parent / "train.csv"), "test": str(self.task.parent / "test.csv"),
                                                        "sample_submission": str(self.task.parent / "sample_submission.csv")}, "max_report_bytes": 24000}, self.artifacts)
        draft = demo_metadata(self.task.read_text(encoding="utf-8"))["draft"]
        self.spec = resolve_spec(draft, self.profile, self.task.parent, self.artifacts, self.schemas)
        self.code_payload = {"researcher_id": "researcher-1", "filename": "experiment.py", "content": "from agent.ml import build_pipeline\ndef make_pipeline(task_spec, params):\n    return build_pipeline('linear-classifier-v1', task_spec, params)\n", "declared_entrypoint": "make_pipeline"}
        self.code = write_code(self.code_payload, self.artifacts)["artifact"]

    def plan(self, **changes):
        value = deepcopy(json.loads((ROOT / "contracts/examples/C12/request.json").read_text(encoding="utf-8"))["payload"]["plan"])
        value.update(task_spec_id=self.spec["task_spec_id"], dataset_id=self.spec["dataset_id"], validation_id="validation-unit",
                     recipe_id="linear-classifier-v1", code_artifact_id=self.code["artifact_id"], parent_ids=[],
                     model={"family": "linear", "hyperparameters": {"C": 1.0}})
        value.update(changes)
        return value

    def evaluation(self, score):
        plan = self.plan(model={"family": "linear", "hyperparameters": {"C": score + 1}})
        exp = self.store.reserve(plan, self.code["sha256"])["experiment_id"]
        with self.store.db:
            self.store.db.execute("UPDATE experiments SET state='trained' WHERE experiment_id=?", (exp,))
        record = {"evaluation_id": "eval-" + uuid.uuid4().hex[:12], "experiment_id": exp, "valid": True,
                  "primary_metric": {"metric_id": self.spec["metric"]["metric_id"], "direction": self.spec["metric"]["direction"],
                                     "value": score, "validation_id": "validation-unit", "dataset_id": self.spec["dataset_id"], "fidelity": "full"},
                  "diagnostics_artifact_id": self.code["artifact_id"], "violations": []}
        self.store.save_evaluation(record)
        self.store.commit_evaluation({"experiment_id": exp, "evaluation_id": record["evaluation_id"], "selection_policy_id": "unit"})
        return record

    def candidate(self, label="A", reverse=False):
        import pandas as pd
        test = pd.read_csv(self.task.parent / "test.csv", dtype=str, keep_default_na=False)
        frame = test[["id"]].copy()
        if reverse:
            frame = frame.iloc[::-1].reset_index(drop=True)
        frame["label"] = label
        return self.artifacts.write_text("submissions/" + uuid.uuid4().hex[:8] + ".csv", frame.to_csv(index=False), "submission", "text/csv")

    def close(self):
        self.store.close()
        self.log.close()
        self.temp.cleanup()
