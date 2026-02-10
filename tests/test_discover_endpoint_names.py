"""Tests for endpoint name derivation in discover.py."""

import os
import sys
import tempfile
import unittest
from pathlib import Path

from gen_worker.discover import discover_manifest


def _cleanup_modules(prefix: str) -> None:
    for mod in list(sys.modules.keys()):
        if mod.startswith(prefix):
            del sys.modules[mod]


class TestDiscoverEndpointNames(unittest.TestCase):
    def test_endpoint_name_slugified_from_function_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            original_cwd = os.getcwd()
            original_path = sys.path.copy()

            try:
                os.chdir(root)
                sys.path.insert(0, str(root / "src"))

                (root / "pyproject.toml").write_text(
                    """
[project]
name = "test-project"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = ["gen-worker"]
"""
                )
                (root / "cozy.toml").write_text(
                    """
schema_version = 1
name = "test-project"
main = "endpoint_mod.main"
gen_worker = ">=0"
""".lstrip(),
                    encoding="utf-8",
                )

                src_dir = root / "src" / "endpoint_mod"
                src_dir.mkdir(parents=True)
                (src_dir / "main.py").write_text(
                    """
import msgspec
from gen_worker import ActionContext, worker_function

class Input(msgspec.Struct):
    x: int

class Output(msgspec.Struct):
    y: int

@worker_function()
def medasr_transcribe(ctx: ActionContext, payload: Input) -> Output:
    return Output(y=payload.x)
"""
                )

                manifest = discover_manifest(root)
                funcs = {f["name"]: f for f in manifest["functions"]}
                self.assertEqual(manifest["project_name"], "test-project")
                self.assertEqual(funcs["medasr_transcribe"]["endpoint_name"], "medasr-transcribe")
            finally:
                os.chdir(original_cwd)
                sys.path[:] = original_path
                _cleanup_modules("endpoint_mod")

    def test_endpoint_slug_collision_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            original_cwd = os.getcwd()
            original_path = sys.path.copy()

            try:
                os.chdir(root)
                sys.path.insert(0, str(root / "src"))

                (root / "pyproject.toml").write_text(
                    """
[project]
name = "test-project"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = ["gen-worker"]
"""
                )
                (root / "cozy.toml").write_text(
                    """
schema_version = 1
name = "test-project"
main = "endpoint_mod_collision.main"
gen_worker = ">=0"
""".lstrip(),
                    encoding="utf-8",
                )

                src_dir = root / "src" / "endpoint_mod_collision"
                src_dir.mkdir(parents=True)
                (src_dir / "main.py").write_text(
                    """
import msgspec
from gen_worker import ActionContext, worker_function

class Input(msgspec.Struct):
    x: int

class Output(msgspec.Struct):
    y: int

@worker_function()
def image_worker(ctx: ActionContext, payload: Input) -> Output:
    return Output(y=payload.x)

@worker_function()
def image__worker(ctx: ActionContext, payload: Input) -> Output:
    return Output(y=payload.x)
"""
                )

                with self.assertRaises(ValueError):
                    discover_manifest(root)
            finally:
                os.chdir(original_cwd)
                sys.path[:] = original_path
                _cleanup_modules("endpoint_mod_collision")

    def test_project_name_slugified_from_pyproject(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            original_cwd = os.getcwd()
            original_path = sys.path.copy()

            try:
                os.chdir(root)
                sys.path.insert(0, str(root / "src"))

                (root / "pyproject.toml").write_text(
                    """
[project]
name = "My Cool_Project"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = ["gen-worker"]
"""
                )
                (root / "cozy.toml").write_text(
                    """
schema_version = 1
name = "My Cool_Project"
main = "endpoint_mod_project_name.main"
gen_worker = ">=0"
""".lstrip(),
                    encoding="utf-8",
                )

                src_dir = root / "src" / "endpoint_mod_project_name"
                src_dir.mkdir(parents=True)
                (src_dir / "main.py").write_text(
                    """
import msgspec
from gen_worker import ActionContext, worker_function

class Input(msgspec.Struct):
    x: int

class Output(msgspec.Struct):
    y: int

@worker_function()
def generate(ctx: ActionContext, payload: Input) -> Output:
    return Output(y=payload.x)
"""
                )

                manifest = discover_manifest(root)
                self.assertEqual(manifest["project_name"], "my-cool-project")
            finally:
                os.chdir(original_cwd)
                sys.path[:] = original_path
                _cleanup_modules("endpoint_mod_project_name")


if __name__ == "__main__":
    unittest.main()
