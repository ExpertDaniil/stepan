"""Provider-neutral CI entry point, using stdlib unittest."""
from pathlib import Path
import compileall
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if __name__ == "__main__":
    if not compileall.compile_dir(ROOT / "agent", quiet=1) or not compileall.compile_dir(ROOT / "tools", quiet=1):
        raise SystemExit(1)
    raise SystemExit(subprocess.call([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"], cwd=ROOT))
