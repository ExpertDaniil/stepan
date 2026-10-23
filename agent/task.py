from pathlib import Path
import hashlib

from agent.contracts import canonical
from agent.errors import AgentError
from tools.data import read_table
from agent.predictions import read_training, targets
from agent.data_transport import transport

METRICS = {
    "rmse": ("minimize", "values"), "mae": ("minimize", "values"), "r2": ("maximize", "values"),
    "f1_macro": ("maximize", "labels"), "f1_weighted": ("maximize", "labels"), "accuracy": ("maximize", "labels"),
    'log_loss': ('minimize', 'probabilities'), 'roc_auc': ('maximize', 'probabilities'),
}


def resolve_spec(draft, profile, task_root, artifacts, schemas, *, task_text=''):
    import pandas as pd
    import numpy as np
    kind = draft["task_kind"]
    if kind not in {"regression", "binary_classification", "multiclass_classification"}:
        raise AgentError("UNSUPPORTED_TASK_KIND", kind+': '+draft.get('reason', 'No declared task adapter'))
    target = draft["target_column"]
    target_columns = targets(draft)
    if not target_columns or len(set(target_columns)) != len(target_columns) or target_columns[0] != target:
        raise AgentError('TASK_SPEC_CONFLICT', 'target_columns must be unique, nonempty and start with target_column')
    if kind != 'regression' and len(target_columns) > 1:
        raise AgentError('UNSUPPORTED_OUTPUT_SCHEMA', 'Multi-target classification requires a separate explicit adapter')
    train = transport(artifacts).read(profile['train_artifact_id'], draft)
    test = transport(artifacts).read(profile['test_artifact_id'])
    if any(c not in train for c in target_columns) or train[target_columns].isna().any().any():
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
            y = train[target_columns].apply(pd.to_numeric, errors="raise")
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
        elif metric['metric_id'] == 'roc_auc':
            positive = metric['parameters'].get('positive_label')
            if len(labels) != 2 or positive not in labels or set(metric['parameters']) != {'positive_label'}:
                raise AgentError('TASK_SPEC_CONFLICT', 'roc_auc adapter requires two classes and an explicit positive_label')
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
    if (len(set(output['id_columns'])) != len(output['id_columns']) or
            len(set(output['prediction_columns'])) != len(output['prediction_columns']) or
            set(output['id_columns']) & set(output['prediction_columns']) or
            len(set(output['id_columns'] + output['prediction_columns'])) != len(output['columns']) or
            set(output['columns']) != set(output['id_columns'] + output['prediction_columns'])):
        raise AgentError('TASK_SPEC_CONFLICT', 'Output must partition columns into distinct identifiers and predictions')
    output_kind = output.setdefault('prediction_kind', prediction)
    if (kind == 'regression') != (output_kind == 'values'):
        raise AgentError('TASK_SPEC_CONFLICT', 'Output prediction kind is incompatible with task kind')
    if output_kind not in {'values', 'labels', 'probabilities'}:
        raise AgentError('UNSUPPORTED_OUTPUT_SCHEMA', 'Unknown prediction kind')
    count = len(output['prediction_columns'])
    if output_kind == 'probabilities':
        if count == len(labels):
            order = output.get('class_labels', output['prediction_columns'])
            if len(order) != len(labels) or len(set(order)) != len(labels) or set(order) != set(labels):
                raise AgentError('TASK_SPEC_CONFLICT', 'class_labels must explicitly map each probability column to its class')
            output['class_labels'] = list(order)
            if output.get('positive_label') is not None:
                raise AgentError('TASK_SPEC_CONFLICT', 'positive_label is only used with one binary probability column')
        elif len(labels) == 2 and count == 1 and output.get('positive_label') in labels:
            if output.get('class_labels'):
                raise AgentError('TASK_SPEC_CONFLICT', 'One probability column uses positive_label, not class_labels')
        else:
            raise AgentError('UNSUPPORTED_OUTPUT_SCHEMA', 'Declare all class probabilities, or one binary column with positive_label')
    elif count != len(target_columns) or output.get('class_labels') or output.get('positive_label') is not None:
        raise AgentError('UNSUPPORTED_OUTPUT_SCHEMA', 'Prediction columns must correspond to target_columns in declared order')
    for col in output["id_columns"]:
        if col in target_columns:
            raise AgentError('TASK_SPEC_CONFLICT', 'A target cannot also be an output identifier')
        if col not in test:
            raise AgentError("TASK_SPEC_CONFLICT", f"Test does not contain id column: {col}")
    if profile["sample_submission_artifact_id"]:
        sample = pd.read_csv(artifacts.resolve(profile["sample_submission_artifact_id"]), nrows=0)
        if list(sample.columns) != output["columns"]:
            raise AgentError("TASK_SPEC_CONFLICT", "Output columns disagree with sample_submission")
    excluded = list(output["id_columns"]) + target_columns[1:]
    features = [c for c in train if c not in target_columns and c not in excluded]
    if not features or any(c not in test for c in features):
        raise AgentError("TASK_SPEC_CONFLICT", "Train/test feature columns do not align")
    spec = {"task_spec_id": "task-" + hashlib.sha256(canonical(draft).encode()).hexdigest()[:12],
            "dataset_id": profile["dataset_id"], "task_kind": kind, "target_column": target,
            'target_columns': target_columns,
            **{k: profile[k] for k in ("train_artifact_id", "test_artifact_id", "dictionary_artifact_id", "sample_submission_artifact_id")},
            "metric": metric, "allowed_labels": labels, "output": output, "excluded_features": excluded}
    from agent.semantic_planning import build_task_model
    semantic_profile = artifacts.read_json(profile['profile_artifact_id']) if profile.get('profile_artifact_id') else profile['summary']
    if task_text or draft.get('task_model') is not None:
        model = build_task_model(draft.get('task_model'), task_text, semantic_profile, schemas, spec)
        semantic_art = artifacts.write_json('task/semantic-model.json', model, 'task_semantic_model')
        spec.update(semantic_model_artifact_id=semantic_art['artifact_id'], semantic_model_id=model['model_id'])
        artifacts.log.event('research.task_model_built', model_id=model['model_id'], artifact_id=semantic_art['artifact_id'],
            quantities=len(model['quantities']), relations=len(model['relations']),
            verification=model['verification'], source='existing_task_interpretation_call')
    schemas.validate(spec, "TaskSpec")
    artifacts.write_json("task/task-spec.json", spec, "task_spec")
    return spec
