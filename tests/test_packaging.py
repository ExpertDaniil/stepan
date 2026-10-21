import ast
import base64
import hashlib
import io
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from agent.config import ROOT
from scripts.package_solution import build


class PackagingTests(unittest.TestCase):
    def test_flat_archive_is_reproducible_and_contains_only_owned_runtime(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first, second = root / "first.zip", root / "second.zip"
            build(first)
            build(second)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            with zipfile.ZipFile(first) as archive:
                self.assertEqual(archive.namelist(), ["solution.py", "requirements.txt"])
                source = archive.read("solution.py").decode("utf-8")
                assignments = {node.targets[0].id: ast.literal_eval(node.value)
                               for node in ast.parse(source).body if isinstance(node, ast.Assign)
                               and isinstance(node.targets[0], ast.Name) and node.targets[0].id in {"BUNDLE", "BUNDLE_SHA256"}}
                payload = base64.b85decode(assignments["BUNDLE"])
                self.assertEqual(hashlib.sha256(payload).hexdigest(), assignments["BUNDLE_SHA256"])
                with zipfile.ZipFile(io.BytesIO(payload)) as runtime:
                    self.assertIn("agent/worker.py", runtime.namelist())
                    self.assertIn("contracts/contracts.schema.json", runtime.namelist())
                    self.assertIn("prompts/researcher.md", runtime.namelist())
                    self.assertTrue(all(name.startswith(("agent/", "tools/", "contracts/", "recipes/", "prompts/")) for name in runtime.namelist()))
                    self.assertFalse(any(name.endswith((".csv", ".whl", ".pkl", ".joblib", ".env")) or "__pycache__" in name for name in runtime.namelist()))

    def test_invalid_build_preserves_previous_archive(self):
        with tempfile.TemporaryDirectory() as temp:
            destination = Path(temp) / "solution.zip"
            destination.write_bytes(b"previous-good-artifact")
            with patch("scripts.package_solution.MAX_ZIP_BYTES", 1), self.assertRaisesRegex(ValueError, "100 MB"):
                build(destination)
            self.assertEqual(destination.read_bytes(), b"previous-good-artifact")
            self.assertEqual(list(Path(temp).iterdir()), [destination])

    def test_changed_official_requirements_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "requirements.txt").write_text("unapproved-package==1.0\n")
            shutil.copy2(ROOT / "requirements.organizers.txt", root / "requirements.organizers.txt")
            with self.assertRaisesRegex(ValueError, "match the organizers"):
                build(root / "solution.zip", root=root)
