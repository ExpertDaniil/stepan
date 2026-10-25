import ast
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from agent.config import ROOT
from scripts.package_solution import (PLATFORM_ENV_RESOURCE, build, flat_module_name,
                                     runtime_files, validate_platform_env)


class PackagingTests(unittest.TestCase):
    def test_packaged_entrypoint_applies_public_defaults_without_local_env(self):
        self.run_packaged_config()

    def test_platform_env_and_explicit_file_override_bundled_defaults(self):
        self.run_packaged_config(overrides=True)

    def run_packaged_config(self, overrides=False):
        # Launch the actual flat ZIP in a separate process with no repository on
        # sys.path. Only the expensive ML stage is replaced; CLI, Config and logs run.
        script = '''
import builtins
import json
import os
import runpy
import sys
original_import = builtins.__import__
def bootstrap(config, *args):
    print(json.dumps({"config": config.public_dict(),
                      "platform_credentials_preserved":
                          os.environ.get("OPENAI_API_KEY") == "test-platform-key" and
                          os.environ.get("OPENAI_BASE_URL") == "https://platform.invalid/v1"}))
    return 0
def intercepted_import(name, *args, **kwargs):
    module = original_import(name, *args, **kwargs)
    if name == "agent.runtime":
        module.bootstrap = bootstrap
    return module
builtins.__import__ = intercepted_import
sys.argv = ["solution.py", "task.md", *sys.argv[1:]]
runpy.run_path("solution.py", run_name="__main__")
'''
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive_path = root / "solution.zip"
            build(archive_path)
            with zipfile.ZipFile(archive_path) as archive:
                archive.extractall(root)
            # A task-local .env is not implicitly read or copied from the builder.
            (root / ".env").write_text("LLM_MAX_OUTPUT_TOKENS=128\nOPENAI_API_KEY=wrong\n")
            env = {"PATH": os.environ.get("PATH", os.defpath),
                   "OPENAI_API_KEY": "test-platform-key",
                   "OPENAI_BASE_URL": "https://platform.invalid/v1",
                   "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
            args = []
            if overrides:
                env["LLM_MAX_OUTPUT_TOKENS"] = "4096"
                env["LLM_MAX_TOTAL_OUTPUT_TOKENS"] = "4096"
                (root / "explicit.env").write_text(
                    "LLM_MAX_OUTPUT_TOKENS=2048\nGRAPH_CONTEXT_BYTES=131072\n"
                    "OPENAI_API_KEY=wrong\nOPENAI_BASE_URL=https://wrong.invalid\n")
                args = ["--env-file", "explicit.env", "--llm-mode", "mock"]
            result = subprocess.run([sys.executable, "-c", script, *args], cwd=root, env=env,
                                    capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(result.stdout)
            self.assertTrue(payload["platform_credentials_preserved"])
            config = payload["config"]
            expected = {"profile": "platform", "research_mode": "adaptive",
                        "catalog_warmup_experiments": 4, "research_series_size": 4,
                        "llm_max_calls": None, "llm_max_output_tokens": 32768,
                        "llm_max_output_tokens_limit": 65536, "llm_bootstrap_output_tokens": 8192,
                        "llm_max_total_output_tokens": None, "llm_timeout_seconds": 600,
                        "llm_operation_timeout_seconds": 1200, "llm_progress_seconds": 15,
                        "console_format": "human", "log_level": "INFO", "log_model_io": True,
                        "llm_mode": "live", "graph_context_bytes": 196608,
                        "llm_request_max_bytes": 196608}
            if overrides:
                expected.update(llm_max_output_tokens=4096, llm_max_total_output_tokens=4096,
                                graph_context_bytes=131072, llm_mode="mock")
            self.assertEqual({key: config[key] for key in expected}, expected)
            events = [json.loads(line) for path in (root / "runs").glob("*/events.jsonl")
                      for line in path.read_text().splitlines()]
            self.assertEqual(next(event["config"] for event in events if event["event"] == "run.started"), config)

    def test_build_environment_cannot_leak_into_archive_or_change_settings(self):
        with tempfile.TemporaryDirectory() as temp:
            first, second = Path(temp) / "first.zip", Path(temp) / "second.zip"
            build(first)
            with patch.dict(os.environ, {"OPENAI_API_KEY": "synthetic-build-secret",
                                         "OPENAI_BASE_URL": "https://synthetic-build.invalid/v1",
                                         "LLM_MAX_OUTPUT_TOKENS": "128"}):
                build(second)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            with zipfile.ZipFile(second) as archive:
                self.assertNotIn(b"synthetic-build-secret", b"".join(archive.read(name) for name in archive.namelist()))

    def test_platform_profile_rejects_secret_keys_duplicates_and_invalid_values(self):
        settings = json.loads((ROOT / PLATFORM_ENV_RESOURCE).read_text())
        invalid = [json.dumps({**settings, "OPENAI_API_KEY": "synthetic-secret"}),
                   json.dumps({**settings, "LLM_MAX_CALLS": 0}),
                   json.dumps({**settings, "LOG_LEVEL": "INFO\nBAD=value"}),
                   json.dumps({**settings, "LOG_LEVEL": ""}), "[]",
                   json.dumps(settings)[:-1] + ', "LLM_MAX_CALLS": "32"}']
        for content in invalid:
            with self.subTest(content=content), self.assertRaises(ValueError):
                validate_platform_env(content)
        with patch("scripts.package_solution.validate_platform_env", side_effect=ValueError("invalid profile")):
            with self.assertRaisesRegex(ValueError, "invalid profile"):
                runtime_files(ROOT)

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
