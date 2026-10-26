from pathlib import Path
import hashlib

from agent.contracts import canonical
from agent.errors import AgentError
from agent import metrics as metric_registry
from tools.data import read_table
from agent.predictions import read_training, targets
from agent.data_transport import transport


def _default_output_kind(kind, metric, output, labels, profile, artifacts):
    """Submission format when the interpretation omitted it.

    The metric's prediction kind is a scoring input, not the file format: an AUC
    task may still ask for class names. Probabilities are chosen only when the
    columns map one-to-one to classes, a binary column names its positive class,
    or the sample's single column holds probabilities rather than class names;
    such a binary column is then bound to the metric's positive label (in place).
    """
    import pandas as pd
    if kind == 'regression':
        return 'values'
    columns = output.get('prediction_columns') or []
    if metric['prediction_kind'] != 'probabilities':
        return 'labels'
    if len(columns) == len(labels) > 1 or (len(columns) == 1 and output.get('positive_label') in labels):
        return 'probabilities'
    if len(columns) == 1 and profile.get('sample_submission_artifact_id'):
        sample = pd.read_csv(artifacts.resolve(profile['sample_submission_artifact_id']), usecols=columns,
                             dtype=str, keep_default_na=False)[columns[0]]
        numbers = pd.to_numeric(sample, errors='coerce')
        # Class names such as 0/1 make a sample of zeros ambiguous; a probability
        # metric then decides for scores, since ranking metrics on hard labels lose.
        numeric_labels = bool(pd.to_numeric(pd.Series(labels), errors='coerce').notna().all())
        if (len(sample) and numbers.notna().all() and numbers.between(0, 1).all()
                and (numeric_labels or not set(sample) <= set(labels))):
            if len(labels) != 2:
                return 'labels'
            output['positive_label'] = (metric['parameters'].get('positive_label')
                                        or metric_registry.default_positive_label(labels)[0])
            return 'probabilities'
    return 'labels'


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
    # The registry never aborts on an unknown metric name: it resolves an alias or
    # records an explicit proxy in metric.declared (agent/metrics.py).
    metric = metric_registry.resolve(draft["metric"], kind, labels,
                                     truth=train[target_columns] if kind == 'regression' else train[target].astype(str))
    if metric.get('declared'):
        artifacts.log.event('task.metric_resolved', level='WARNING' if metric['declared']['status'] == 'proxy' else 'INFO',
                            metric_id=metric['metric_id'], declared=metric['declared'])
    prediction = metric['prediction_kind']
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
    if 'prediction_kind' not in output:
        output['prediction_kind'] = _default_output_kind(kind, metric, output, labels, profile, artifacts)
    output_kind = output['prediction_kind']
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
