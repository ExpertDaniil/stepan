"""Generate small synthetic tasks only; refuses to overwrite existing inputs."""
from pathlib import Path
import argparse
import csv
import json
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # Also runnable as `python scripts/make_demo.py`.
    sys.path.insert(0, str(ROOT))


def make_demo(directory, kind="classification", rows=360, test_rows=90, metric=None):
    """metric: optional registry id or arbitrary declared name (unknown names exercise proxies)."""
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    if any((directory / name).exists() for name in ("train.csv", "test.csv", "task.md")):
        raise FileExistsError("Demo destination already contains a task; choose an empty directory")
    regression = kind == "regression"
    from agent.metrics import ALIASES, LOG_SQUARED, METRICS, normalize_name
    metric_id = metric or ("rmse" if regression else "f1_macro")
    known = METRICS.get(ALIASES.get(normalize_name(metric_id), ''))
    # Log-scale metrics (RMSLE, MAPE, ...) need a positive, multiplicative target.
    positive = regression and known is not None and known.objective == LOG_SQUARED
    task_kind = "regression" if regression else "binary_classification" if kind == "binary" else "multiclass_classification"
    rng = random.Random(42)
    train, test = [], []
    for i in range(rows + test_rows):
        x, z = rng.uniform(-3, 3), rng.uniform(-2, 2)
        category = rng.choice(["red", "blue", "NA"])
        latent = 2 * x + z + (1 if category == "red" else -1)
        label = (round(10*2.718281828**(0.35*latent+rng.gauss(0, 0.05)), 4) if positive else
                 latent * 3 + rng.gauss(0, 0.3)) if regression else ("yes" if latent > 0 else "no") if kind == "binary" else "A" if latent < -2 else "C" if latent > 2 else "B"
        row = {"id": f"{i:06d}", "x_numeric": x if i % 23 else "", "z_numeric": z,
               "category": category if i % 19 else ""}
        if i < rows:
            train.append({**row, "label": label})
        else:
            test.append(row)
    for name, data, columns in [("train.csv", train, ["id", "x_numeric", "z_numeric", "category", "label"]),
                                ("test.csv", test, ["id", "x_numeric", "z_numeric", "category"]),
                                ("sample_submission.csv", [{"id": r["id"], "label": 0 if regression else "yes" if kind == "binary" else "A"} for r in test], ["id", "label"])]:
        with (directory / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(data)
    draft = {"task_kind": task_kind, "target_column": "label",
             "metric": {"metric_id": metric_id,
                        "direction": known.direction if known else "minimize" if regression else "maximize",
                        "prediction_kind": known.prediction_kind if known else "values" if regression else "labels",
                        "parameters": {}},
             "output": {"path": "submission.csv", "columns": ["id", "label"], "id_columns": ["id"], "prediction_columns": ["label"],
                        "row_order": "test", "encoding": "utf-8", "separator": ",", "write_index": False}}
    description = "# Synthetic integration task\n\nPredict label from train.csv for test.csv.\n"
    description += "Files: `train.csv`, `test.csv`, `sample_submission.csv`.\n"
    description += (f"Metric: {metric_id}.\n" if metric else
                    "Metric: RMSE, lower is better.\n" if regression else "Metric: Macro F1, higher is better.\n")
    description += "Write submission.csv next to this file; columns id,label, test row order, no index.\n"
    description += "\nThe following block is consumed only by the explicit local mock adapter.\n\n```agent-demo\n"
    description += json.dumps({"draft": draft}, ensure_ascii=False, indent=2) + "\n```\n"
    (directory / "task.md").write_text(description, encoding="utf-8")
    return directory / "task.md"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--kind", choices=["regression", "classification", "binary"], default="classification")
    parser.add_argument("--rows", type=int, default=360)
    parser.add_argument("--metric", default=None, help="Registry id or any declared metric name")
    args = parser.parse_args()
    print(make_demo(args.directory, args.kind, args.rows, metric=args.metric))
