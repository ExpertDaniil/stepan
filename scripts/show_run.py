from pathlib import Path
import argparse
import json
import time


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read redacted run history")
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--contract", default=None, help="For example C10")
    parser.add_argument("--experiment", default=None)
    parser.add_argument("--event", default=None)
    parser.add_argument("--follow", action="store_true")
    args = parser.parse_args()
    with (args.run_directory / "events.jsonl").open(encoding="utf-8") as handle:
        while True:
            line = handle.readline()
            if not line:
                if not args.follow:
                    break
                time.sleep(0.2)
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if args.contract and event.get("contract_id") != args.contract:
                continue
            if args.experiment and event.get("experiment_id") != args.experiment:
                continue
            if args.event and args.event not in event.get("event", ""):
                continue
            fields = " ".join(f"{k}={event[k]}" for k in ("contract_id", "experiment_id", "job_id", "score", "message") if k in event)
            print(event["timestamp"], event["level"], event["event"], fields, flush=True)
