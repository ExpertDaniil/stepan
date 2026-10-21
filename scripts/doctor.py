"""Read-only environment diagnostics; never prints credential values."""
import importlib.metadata
import json
import os
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    pins = {}
    for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        if "==" not in line or line.startswith("#"):
            continue
        name, version = line.split("==")
        try:
            installed = importlib.metadata.version(name.split("[", 1)[0])
        except importlib.metadata.PackageNotFoundError:
            installed = None
        pins[name] = {"expected": version, "installed": installed, "matches": installed == version}
    result = {"python": sys.version, "python_matches": sys.version_info[:2] == (3, 12),
              "platform": platform.system(), "dependencies": pins,
              "live_api_env_present": {k: bool(os.environ.get(k)) for k in ("OPENAI_API_KEY", "OPENAI_MODEL", "OPENAI_BASE_URL")},
              "required_live_api_env": ["OPENAI_API_KEY", "OPENAI_BASE_URL"], "openai_model_optional": True}
    print(json.dumps(result, indent=2))
