"""Platform entry point: python solution.py /path/to/task.md."""
import time

STARTED_MONOTONIC = time.monotonic()  # Before heavy imports; counts preparation too.

if __name__ == "__main__":
    from agent.runtime import main
    raise SystemExit(main(started_monotonic=STARTED_MONOTONIC))
