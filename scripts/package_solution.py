"""Build a strict allowlisted competition ZIP with solution.py in its root."""
from pathlib import Path
import argparse
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def build(destination, root=ROOT):
    root = Path(root).resolve()
    destination = Path(destination).resolve()
    allowed_files = [root / "solution.py", root / "contracts/contracts.json", root / "contracts/contracts.schema.json"]
    for folder in ("agent", "tools", "recipes", "prompts"):
        for path in (root / folder).rglob("*"):
            if path.is_symlink():
                raise ValueError("Symlinks are not permitted in a submission")
            if path.is_file() and "__pycache__" not in path.parts and path.suffix in {".py", ".sql", ".json", ".md"}:
                allowed_files.append(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(set(allowed_files)):
            if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(root):
                raise ValueError("Missing or unsafe package input: " + str(path))
            archive.write(path, path.relative_to(root).as_posix())
    with zipfile.ZipFile(destination) as archive:
        names = archive.namelist()
        if "solution.py" not in names or len(names) != len(set(names)) or archive.testzip():
            raise ValueError("Invalid platform ZIP")
        if any(name.startswith(("/", "../")) or "/../" in name or ".env" in Path(name).parts for name in names):
            raise ValueError("Unsafe ZIP path")
    return {"path": str(destination), "files": len(names), "sha256": hashlib.sha256(destination.read_bytes()).hexdigest()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "dist/solution.zip")
    args = parser.parse_args()
    print(json.dumps(build(args.output), indent=2))
