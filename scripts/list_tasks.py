"""Read-only task discovery/preflight. Does not load CSVs or call the model."""
from pathlib import Path
import argparse
import shlex
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.errors import AgentError
from tools.files import find_task_files, resolve_inputs


def main(argv=None):
    parser = argparse.ArgumentParser(description="Find task.md files and check input paths without loading data")
    parser.add_argument("directory", nargs="?", type=Path, default=Path("data_for_participants"))
    args = parser.parse_args(argv)
    if not args.directory.is_dir():
        print(f"Directory not found: {args.directory}", file=sys.stderr)
        return 2
    tasks = find_task_files(args.directory)
    if not tasks:
        print(f"No task.md files found under {args.directory.resolve()}", file=sys.stderr)
        return 1
    valid = 0
    for task in tasks:
        try:
            _, paths = resolve_inputs(task)
        except AgentError as exc:
            print(f"INVALID {task}\n  {exc.code}: {exc}")
            continue
        valid += 1
        print(f"READY {task}")
        for role, path in paths.items():
            print(f"  {role}: {path}")
        print("  " + shlex.join(["python3", "solution.py", str(task)]))
    print(f"Tasks with resolvable input paths: {valid}/{len(tasks)}. No CSVs read and no API calls made.")
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
