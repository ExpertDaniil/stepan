"""Read a consistent exported graph. Never writes SQLite or loads a model."""
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3


def load_graph(directory):
    root = Path(directory).resolve()
    with sqlite3.connect((root / "state.sqlite").as_uri()+"?mode=ro", uri=True) as db:
        row = db.execute("SELECT run_id,value FROM g_metadata WHERE name='latest_export'").fetchone()
        if not row:
            raise ValueError("No committed graph export is available")
        export = json.loads(row[1]);run_id = row[0]
        data = db.execute("SELECT a.sha256,l.root_relative_path FROM g_artifacts a JOIN g_artifact_locations l USING(run_id,artifact_id) WHERE a.run_id=? AND a.artifact_id=? AND a.state='ready' AND l.root_kind='run'", (run_id, export["graph_artifact_id"])).fetchone()
        path = (root / data[1]).resolve()
        if not path.is_relative_to(root) or hashlib.sha256(path.read_bytes()).hexdigest() != data[0]:
            raise ValueError("Graph export is corrupt or outside the run")
        return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    graph = load_graph(args.run_directory)
    if args.json:
        print(json.dumps(graph, ensure_ascii=False, indent=2))
    else:
        print("Run:", graph["run_id"], "event cut:", graph["captured_event_seq"], "revision:", graph["graph_revision"])
        for branch in graph["branches"]:
            print(branch["branch_id"], branch["decision"], branch["state"], "parent="+str(branch["parent_branch_id"]), branch["constraints"])
            for row in graph["heads"]:
                if row["branch_id"] == branch["branch_id"]:
                    h = row["head"];print(" ", h["researcher_id"], "→", h["experiment_id"], "head revision", h["head_revision"])
        print("Solutions:", len(graph["solutions"]), "experiments:", len(graph["experiments"]), "ideas:", len(graph["ideas"]))
