import hashlib
import json
from pathlib import Path

from agent.errors import AgentError


def read_table(path, **kwargs):
    import csv
    import pandas as pd
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        header = next(csv.reader(handle), [])
    if not header or len(header) != len(set(header)):
        raise AgentError("INVALID_TABLE", "CSV has an empty or duplicate header")
    # The case states that an empty cell is missing. Literal NA/NULL are values.
    defaults = {"encoding": "utf-8", "keep_default_na": False, "na_values": [""]}
    defaults.update(kwargs)
    return pd.read_csv(path, **defaults)


def profile_data(payload, artifacts):
    import pandas as pd
    registered = {}
    summary = {}
    for role, path in payload["candidate_paths"].items():
        art = artifacts.register(path, "input_" + role, "text/csv", external=True)
        registered[role] = art
        if role in {"train", "test"}:
            try:
                frame = read_table(path)
            except Exception as exc:
                raise AgentError("INVALID_TABLE", f"Cannot read {role}: {exc}") from exc
            if frame.empty or not frame.columns.is_unique:
                raise AgentError("INVALID_TABLE", f"{role} must have rows and unique columns")
            summary[role + "_rows"] = len(frame)
            summary[role + "_columns"] = list(frame.columns)
            summary[role + "_dtypes"] = {c: str(t) for c, t in frame.dtypes.items()}
            summary[role + "_missing_fraction"] = {c: round(float(v), 5) for c, v in frame.isna().mean().items()}
            del frame
    ident = "dataset-" + hashlib.sha256((registered["train"]["sha256"] + registered["test"]["sha256"]).encode()).hexdigest()[:20]
    if "dictionary" in payload["candidate_paths"]:
        dictionary = read_table(payload["candidate_paths"]["dictionary"])
        summary["data_dictionary"] = json.loads(dictionary.to_json(orient="records", force_ascii=False))
    art = artifacts.write_json("task/profile.json", summary, "data_profile")
    # Preserve the complete report as an artifact; bound the LLM-facing response.
    bounded = dict(summary)
    maximum = payload["max_report_bytes"]
    if maximum < 512:
        raise AgentError("INVALID_PLAN", "max_report_bytes must be at least 512")
    if maximum >= 4096:
        from tools.analysis import compact_profile
        bounded['core_profile'] = compact_profile(summary)
    def size():
        return len(json.dumps(bounded, ensure_ascii=False, allow_nan=False).encode("utf-8"))
    if "data_dictionary" in bounded:
        bounded["data_dictionary"] = list(bounded["data_dictionary"])
        while size() > maximum and bounded["data_dictionary"]:
            bounded["data_dictionary"].pop()
        if len(bounded["data_dictionary"]) != len(summary["data_dictionary"]):
            bounded["data_dictionary_truncated"] = True
    for key in ["test_missing_fraction", "train_missing_fraction", "test_dtypes", "train_dtypes", "data_dictionary"]:
        if size() > maximum:
            bounded.pop(key, None)
    for key in ["test_columns", "train_columns"]:
        bounded[key] = list(bounded[key])
        while size() > maximum and len(bounded[key]) > 1:
            bounded[key].pop()
            bounded[key + "_truncated"] = True
    if size() > maximum:
        raise AgentError("CONTEXT_TOO_LARGE", "Technical summary exceeds max_report_bytes")
    result = {"dataset_id": ident, "profile_artifact_id": art["artifact_id"],
              "train_artifact_id": registered["train"]["artifact_id"], "test_artifact_id": registered["test"]["artifact_id"],
              "dictionary_artifact_id": registered.get("dictionary", {}).get("artifact_id"),
                "sample_submission_artifact_id": registered.get("sample_submission", {}).get("artifact_id"), "summary": bounded}
    return result
