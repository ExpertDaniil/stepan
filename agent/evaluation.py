import hashlib
import math
import uuid

from agent.contracts import canonical
from agent.errors import AgentError
from tools.data import read_table


def error_diagnostics(data, task, fit_indices, valid_indices, y, prediction):
    """Bounded aggregates from the fixed holdout; never expose row-level labels."""
    import numpy as np
    import pandas as pd
    regression = task["task_kind"] == "regression"
    truth = np.asarray(y, dtype=float if regression else str)
    predicted = np.asarray(prediction, dtype=float if regression else str)
    error = np.abs(truth-predicted) if regression else (truth != predicted).astype(float)
    result = {}
    if not regression:
        counts = pd.Series(truth).value_counts()
        # No N-by-N confusion matrix and no unbounded per-class dump.
        labels = counts.iloc[:32].index.tolist()
        classes = []
        for label in labels:
            mask = truth == label
            classes.append({"label": str(label)[:80], "rows": int(mask.sum()), "recall": float(np.mean(predicted[mask] == truth[mask]))})
        result["class_summary"] = {"class_count": len(counts), "worst_classes": sorted(classes, key=lambda c: (c["recall"], -c["rows"]))[:6],
                                   "omitted_classes": max(0, len(counts)-len(labels))}
    groups, used = [], 0
    excluded = set(task["excluded_features"]) | {task["target_column"]}
    for column in data:
        if column in excluded or used >= 4:
            continue
        fit, valid = data[column].iloc[fit_indices], data[column].iloc[valid_indices]
        if pd.api.types.is_numeric_dtype(fit) and not pd.api.types.is_bool_dtype(fit):
            finite = pd.to_numeric(fit, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            if finite.nunique() < 3:
                continue
            cuts = sorted(set(map(float, finite.quantile([1/3, 2/3]).tolist())))
            bucket = np.searchsorted(cuts, pd.to_numeric(valid, errors="coerce").to_numpy(dtype=float), side="right")
            masks = [("bin_"+str(i), bucket == i) for i in range(len(cuts)+1)]
        else:
            values = fit.fillna("__MISSING__").astype(str)
            if not 2 <= values.nunique() <= 20:
                continue
            labels = values.value_counts().index[:4]
            actual = valid.fillna("__MISSING__").astype(str).to_numpy()
            masks = [(str(label)[:80], actual == label) for label in labels]
        used += 1
        for label, mask in masks:
            count = int(mask.sum())
            if count < max(5, int(len(truth)*.02)):
                continue
            record = {"feature": str(column)[:80], "group": label, "rows": count, "mean_error": float(error[mask].mean())}
            if regression:
                record.update(rmse=float(np.sqrt(np.mean((truth[mask]-predicted[mask])**2))), bias=float(np.mean(predicted[mask]-truth[mask])))
            groups.append(record)
    result["hard_groups"] = sorted(groups, key=lambda g: g["mean_error"], reverse=True)[:4]
    return result


class Evaluator:
    def __init__(self, spec, artifacts, store, log):
        self.spec, self.artifacts, self.store, self.log = spec, artifacts, store, log
        self.protocols = {}
        self.baseline_value = None

    def create_protocol(self, payload):
        import numpy as np
        import pandas as pd
        from sklearn.model_selection import train_test_split, GroupShuffleSplit
        frame = read_table(self.artifacts.resolve(self.spec["train_artifact_id"]))
        proposal = payload["proposal"]
        params = proposal["parameters"]
        fraction = params.get("valid_fraction", 0.2)
        if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not 0.05 <= fraction <= 0.5:
            raise AgentError("INVALID_SPLIT", "valid_fraction must be between 0.05 and 0.5")
        indices = np.arange(len(frame))
        strategy = proposal["strategy"]
        y = frame[self.spec["target_column"]]
        try:
            if strategy == "stratified_holdout":
                if self.spec["task_kind"] == "regression":
                    raise AgentError("INVALID_SPLIT", "Stratification requires classification")
                a, b = train_test_split(indices, test_size=fraction, random_state=proposal["seed"], stratify=y.astype(str))
            elif strategy == "random_holdout":
                a, b = train_test_split(indices, test_size=fraction, random_state=proposal["seed"])
            elif strategy == "group_holdout":
                column = params.get("column")
                if column not in frame or column == self.spec["target_column"]:
                    raise AgentError("INVALID_SPLIT", "Invalid group column")
                a, b = next(GroupShuffleSplit(n_splits=1, test_size=fraction, random_state=proposal["seed"]).split(frame, groups=frame[column].fillna("__MISSING__").astype(str)))
            elif strategy == "temporal_holdout":
                column = params.get("column")
                if column not in frame or column == self.spec["target_column"] or frame[column].isna().any():
                    raise AgentError("INVALID_SPLIT", "Temporal column must be present and complete")
                # Whole periods stay together. Never split a timestamp boundary.
                ordered = frame[column]
                if not pd.api.types.is_numeric_dtype(ordered):
                    try:
                        ordered = pd.to_datetime(ordered, format="ISO8601", utc=True, errors="raise")
                    except (ValueError, TypeError) as exc:
                        raise AgentError("INVALID_SPLIT", "Temporal column must be numeric or unambiguous ISO-8601") from exc
                periods = sorted(ordered.unique().tolist())
                if len(periods) < 2:
                    raise AgentError("INVALID_SPLIT", "At least two periods required")
                cutoff = periods[max(1, int(len(periods) * (1 - fraction)))]
                a, b = indices[(ordered < cutoff).to_numpy()], indices[(ordered >= cutoff).to_numpy()]
            else:
                raise AgentError("INVALID_SPLIT", f"Unsupported validation strategy: {strategy}")
        except AgentError:
            raise
        except Exception as exc:
            raise AgentError("INVALID_SPLIT", str(exc)) from exc
        a, b = sorted(map(int, a)), sorted(map(int, b))
        if not a or not b or set(a) & set(b) or len(a) + len(b) != len(frame):
            raise AgentError("INVALID_SPLIT", "Invalid row partition")
        protocol = {"dataset_id": payload["dataset_id"], "metric": self.spec["metric"], "proposal": proposal,
                    "train_indices": a, "valid_indices": b, "evaluator_version": "sklearn-metrics-v1"}
        sha = hashlib.sha256(canonical(protocol).encode()).hexdigest()
        ident = "validation-" + sha[:16]
        train_art = self.artifacts.write_json(f"validation/{ident}.train.json", a, "split_train")
        valid_art = self.artifacts.write_json(f"validation/{ident}.valid.json", b, "split_valid")
        self.artifacts.write_json(f"validation/{ident}.protocol.json", protocol, "validation_protocol")
        self.protocols[ident] = protocol
        self.log.event("validation.created", validation_id=ident, strategy=strategy, parameters=params,
                       metric_id=self.spec["metric"]["metric_id"], direction=self.spec["metric"]["direction"],
                       train_rows=len(a), validation_rows=len(b), protocol_sha256=sha, reason=proposal["reason"])
        return {"validation_id": ident, "train_rows_artifact_id": train_art["artifact_id"],
                "valid_rows_artifact_id": valid_art["artifact_id"], "labels_visibility": "evaluator_only",
                "metric": self.spec["metric"], "protocol_sha256": sha}

    def describe_protocol(self, validation_id):
        """Public research view: rationale and sizes, never validation labels or indices."""
        from copy import deepcopy
        protocol = self.protocols[validation_id]
        return {'validation_id': validation_id,
                'protocol_sha256': hashlib.sha256(canonical(protocol).encode()).hexdigest(),
                'proposal': deepcopy(protocol['proposal']), 'fit_rows': len(protocol['train_indices']),
                'validation_rows': len(protocol['valid_indices']),
                'rule': 'Frozen before research; scores from another split are not comparable. No holdout labels are exposed.'}

    def evaluate(self, payload):
        import numpy as np
        import pandas as pd
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, f1_score, accuracy_score
        if payload["validation_id"] not in self.protocols or payload["dataset_id"] != self.spec["dataset_id"]:
            raise AgentError("STALE_VALIDATION", "Unknown evaluation protocol")
        protocol = self.protocols[payload["validation_id"]]
        path = self.artifacts.resolve(payload["predictions_artifact_id"])
        frame = pd.read_csv(path, dtype={"prediction": str} if self.spec["task_kind"] != "regression" else None, keep_default_na=False)
        if list(frame.columns) != ["__row_index", "prediction"] or frame["__row_index"].tolist() != protocol["valid_indices"]:
            raise AgentError("INVALID_PREDICTIONS", "Validation row order or columns do not match")
        data = read_table(self.artifacts.resolve(self.spec["train_artifact_id"]))
        y = data[self.spec["target_column"]].iloc[protocol["valid_indices"]]
        prediction = frame["prediction"]
        key = self.spec["metric"]["metric_id"]
        if self.spec["task_kind"] == "regression":
            try:
                prediction = pd.to_numeric(prediction, errors="raise").to_numpy(dtype=float)
            except Exception as exc:
                raise AgentError("INVALID_PREDICTIONS", "Non-numeric regression predictions") from exc
            if not np.isfinite(prediction).all():
                raise AgentError("INVALID_PREDICTIONS", "Non-finite predictions")
            value = {"rmse": lambda: math.sqrt(mean_squared_error(y, prediction)),
                     "mae": lambda: mean_absolute_error(y, prediction), "r2": lambda: r2_score(y, prediction)}[key]()
        else:
            prediction = prediction.astype(str)
            if not set(prediction).issubset(self.spec["allowed_labels"]):
                raise AgentError("INVALID_PREDICTIONS", "Unknown predicted label")
            y = y.astype(str)
            value = accuracy_score(y, prediction) if key == "accuracy" else f1_score(y, prediction, average="macro" if key == "f1_macro" else "weighted", labels=self.spec["allowed_labels"], zero_division=0)
        if not math.isfinite(float(value)):
            raise AgentError("METRIC_FAILED", "Metric is not finite")
        experiment = self.store.get(payload["experiment_id"])
        if experiment["plan"]["model"]["family"] == "dummy" and self.baseline_value is None:
            self.baseline_value = float(value)
        direction = self.spec["metric"]["direction"]
        improvement = ((self.baseline_value-float(value)) if direction == "minimize" else (float(value)-self.baseline_value)) if self.baseline_value is not None else None
        diagnostics = {"rows": len(frame), "metric_id": key, "direction": direction, "score": float(value),
                       "baseline_score": self.baseline_value, "improvement_over_baseline": improvement,
                       "improvement_percent": 100*improvement/abs(self.baseline_value) if self.baseline_value and improvement is not None else None}
        if self.spec["task_kind"] == "regression":
            errors = np.abs(np.asarray(y, dtype=float)-prediction)
            diagnostics.update(mae=float(mean_absolute_error(y, prediction)), target_mean=float(np.mean(y)),
                               target_std=float(np.std(y)), absolute_error_p50=float(np.quantile(errors, 0.5)),
                               absolute_error_p90=float(np.quantile(errors, 0.9)))
        diagnostics.update(error_diagnostics(data, self.spec, protocol["train_indices"], protocol["valid_indices"], y, prediction))
        diag = self.artifacts.write_json("diagnostics/" + payload["experiment_id"] + ".json", diagnostics, "diagnostics")
        record = {"evaluation_id": "eval-" + uuid.uuid4().hex[:12], "experiment_id": payload["experiment_id"], "valid": True,
                  "primary_metric": {"metric_id": key, "direction": self.spec["metric"]["direction"], "value": float(value),
                                     "validation_id": payload["validation_id"], "dataset_id": payload["dataset_id"], "fidelity": payload["fidelity"]},
                  "diagnostics_artifact_id": diag["artifact_id"], "violations": []}
        self.store.save_evaluation(record)
        self.log.event("experiment.evaluated", experiment_id=payload["experiment_id"], recipe_id=experiment["plan"]["recipe_id"],
                       hyperparameters=experiment["plan"]["model"]["hyperparameters"], validation_id=payload["validation_id"],
                       diagnostics_artifact_id=diag["artifact_id"], **diagnostics)
        return {"evaluation": record}
