import ast
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from agent.config import ROOT
from scripts.package_solution import build, flat_module_name, runtime_files


class PackagingTests(unittest.TestCase):
    def test_flat_archive_is_reproducible_and_contains_only_owned_runtime(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first, second = root / "first.zip", root / "second.zip"
            build(first)
            build(second)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            with zipfile.ZipFile(first) as archive:
                names = archive.namelist()
                self.assertEqual(names[:2], ["solution.py", "requirements.txt"])
                self.assertTrue(all(Path(name).name == name and (name.endswith(".py") or name == "requirements.txt") for name in names))
                for name in ("agent_runtime.py", "agent_orchestrator.py", "agent_graph_controller.py",
                             "agent_graph_service.py", "agent_llm_gateway.py", "agent_worker.py", "tools_code.py"):
                    self.assertIn(name, names)
                source = archive.read("solution.py").decode("utf-8")
                self.assertIn("from agent.runtime import main", source)
                for forbidden in ("BUNDLE", "base64", "b85decode", "zipfile", "exec("):
                    self.assertNotIn(forbidden, source)
                resources_source = archive.read("stepan_resources.py").decode("utf-8")
                resources = next(ast.literal_eval(node.value) for node in ast.parse(resources_source).body
                                 if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                                 and node.targets[0].id == "RESOURCE_TEXT")
                expected = runtime_files(ROOT)
                for relative, content in expected.items():
                    if relative.endswith(".py"):
                        self.assertEqual(archive.read(flat_module_name(relative)), content, relative)
                    else:
                        self.assertEqual(resources[relative].encode("utf-8"), content, relative)
                self.assertEqual(len(names), sum(name.endswith(".py") for name in expected) + 3)
                for name in names:
                    if name.endswith(".py"):
                        compile(archive.read(name), name, "exec")

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
