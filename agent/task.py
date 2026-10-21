from pathlib import Path
import hashlib

from agent.contracts import canonical
from agent.errors import AgentError
from tools.data import read_table

METRICS = {
    "rmse": ("minimize", "values"), "mae": ("minimize", "values"), "r2": ("maximize", "values"),
    "f1_macro": ("maximize", "labels"), "f1_weighted": ("maximize", "labels"), "accuracy": ("maximize", "labels"),
}


def resolve_spec(draft, profile, task_root, artifacts, schemas):
    import pandas as pd
    import numpy as np
    kind = draft["task_kind"]
    if kind not in {"regression", "binary_classification", "multiclass_classification"}:
        raise AgentError("UNSUPPORTED_TASK_KIND", kind)
    target = draft["target_column"]
    train = read_table(artifacts.resolve(profile["train_artifact_id"]))
    test = read_table(artifacts.resolve(profile["test_artifact_id"]))
    if target not in train or train[target].isna().any():
        raise AgentError("TASK_SPEC_CONFLICT", "Target must exist and have no missing training values")
    metric = dict(draft["metric"])
    if metric["metric_id"] not in METRICS:
        raise AgentError("UNSUPPORTED_METRIC", metric["metric_id"])
    direction, prediction = METRICS[metric["metric_id"]]
    if metric["direction"] != direction or metric["prediction_kind"] != prediction:
        raise AgentError("TASK_SPEC_CONFLICT", "Metric direction or prediction kind is inconsistent")
    if (kind == "regression") != (prediction == "values"):
        raise AgentError("TASK_SPEC_CONFLICT", "Metric is incompatible with task kind")
    if kind == "regression":
        try:
            y = pd.to_numeric(train[target], errors="raise")
        except Exception as exc:
            raise AgentError("TASK_SPEC_CONFLICT", "Regression target is not numeric") from exc
        if not np.isfinite(y.to_numpy(dtype=float)).all():
            raise AgentError("TASK_SPEC_CONFLICT", "Non-finite training target")
        labels = []
    else:
        labels = sorted(train[target].astype(str).unique().tolist())
        if len(labels) < 2 or (kind == "binary_classification" and len(labels) != 2):
            raise AgentError("TASK_SPEC_CONFLICT", "Classification kind does not match observed classes")
        if metric["metric_id"].startswith("f1_"):
            metric["parameters"] = {"labels": labels, "zero_division": 0}
        elif metric["parameters"]:
            raise AgentError("TASK_SPEC_CONFLICT", "This metric implementation takes no parameters")
    if kind == "regression" and metric["parameters"]:
        raise AgentError("TASK_SPEC_CONFLICT", "This regression metric implementation takes no parameters")
    output = dict(draft["output"])
    if len(output["columns"]) != len(set(output["columns"])):
        raise AgentError("TASK_SPEC_CONFLICT", "Output column names must be unique")
    outpath = (Path(task_root) / output["path"]).resolve()
    if not outpath.is_relative_to(Path(task_root).resolve()):
        raise AgentError("TASK_SPEC_CONFLICT", "Output path must stay inside the task directory")
    inputs = [artifacts.resolve(profile[k]) for k in ("train_artifact_id", "test_artifact_id", "dictionary_artifact_id", "sample_submission_artifact_id") if profile[k]]
    if outpath in inputs:
        raise AgentError("TASK_SPEC_CONFLICT", "Output would overwrite an input file")
    output["path"] = str(outpath)
    if len(output["prediction_columns"]) != 1 or output["columns"] != output["id_columns"] + output["prediction_columns"]:
        raise AgentError("UNSUPPORTED_OUTPUT_SCHEMA", "MVP expects id columns followed by one prediction column")
    for col in output["id_columns"]:
        if col not in test:
            raise AgentError("TASK_SPEC_CONFLICT", f"Test does not contain id column: {col}")
    if profile["sample_submission_artifact_id"]:
        sample = pd.read_csv(artifacts.resolve(profile["sample_submission_artifact_id"]), nrows=0)
        if list(sample.columns) != output["columns"]:
            raise AgentError("TASK_SPEC_CONFLICT", "Output columns disagree with sample_submission")
    excluded = list(output["id_columns"])
    features = [c for c in train if c != target and c not in excluded]
    if not features or any(c not in test for c in features):
        raise AgentError("TASK_SPEC_CONFLICT", "Train/test feature columns do not align")
    spec = {"task_spec_id": "task-" + hashlib.sha256(canonical(draft).encode()).hexdigest()[:12],
            "dataset_id": profile["dataset_id"], "task_kind": kind, "target_column": target,
            **{k: profile[k] for k in ("train_artifact_id", "test_artifact_id", "dictionary_artifact_id", "sample_submission_artifact_id")},
            "metric": metric, "allowed_labels": labels, "output": output, "excluded_features": excluded}
    schemas.validate(spec, "TaskSpec")
    artifacts.write_json("task/task-spec.json", spec, "task_spec")
    return spec
