"""Tests for model extraction in discover.py."""

import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from gen_worker.discover import discover_manifest


def _cleanup_modules(prefix: str) -> None:
    """Clean up imported test modules to avoid pollution between tests."""
    for mod in list(sys.modules.keys()):
        if mod.startswith(prefix):
            del sys.modules[mod]


class TestDiscoverModels(unittest.TestCase):
    """Tests for model extraction from function signatures."""

    def test_required_models_extraction(self) -> None:
        """Test that required_models is extracted from injection_json."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            original_cwd = os.getcwd()
            original_path = sys.path.copy()

            try:
                os.chdir(root)
                sys.path.insert(0, str(root / "src"))

                # Create pyproject.toml with models
                pyproject = root / "pyproject.toml"
                pyproject.write_text("""
[project]
name = "test-project"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = ["gen-worker"]

[tool.cozy.functions]
modules = ["funcs_a"]

[tool.cozy.models]
sdxl = "stabilityai/stable-diffusion-xl-base-1.0"
controlnet = "lllyasviel/control_v11p_sd15_canny"
""")

                # Create test module
                src_dir = root / "src" / "funcs_a"
                src_dir.mkdir(parents=True)
                (src_dir / "__init__.py").write_text("""
from typing import Annotated
import msgspec
from gen_worker import ActionContext, worker_function
from gen_worker.injection import ModelRef, ModelRefSource as Src

class Input(msgspec.Struct):
    prompt: str

class Output(msgspec.Struct):
    result: str

class MockPipeline:
    pass

@worker_function()
def generate(
    ctx: ActionContext,
    pipeline: Annotated[MockPipeline, ModelRef(Src.DEPLOYMENT, "sdxl")],
    payload: Input,
) -> Output:
    return Output(result="ok")

@worker_function()
def generate_with_cn(
    ctx: ActionContext,
    pipeline: Annotated[MockPipeline, ModelRef(Src.DEPLOYMENT, "sdxl")],
    cn: Annotated[MockPipeline, ModelRef(Src.DEPLOYMENT, "controlnet")],
    payload: Input,
) -> Output:
    return Output(result="ok")

@worker_function()
def generate_dynamic(
    ctx: ActionContext,
    pipeline: Annotated[MockPipeline, ModelRef(Src.PAYLOAD, "model_key")],
    payload: Input,
) -> Output:
    return Output(result="ok")
""")

                # Run discovery
                manifest = discover_manifest(root)

                # Check functions have required_models
                funcs = {f["name"]: f for f in manifest["functions"]}

                # generate: needs sdxl
                self.assertIn("generate", funcs)
                self.assertEqual(funcs["generate"]["required_models"], ["sdxl"])

                # generate_with_cn: needs sdxl and controlnet
                self.assertIn("generate_with_cn", funcs)
                self.assertEqual(sorted(funcs["generate_with_cn"]["required_models"]), ["controlnet", "sdxl"])

                # generate_dynamic: PAYLOAD source, so no required_models
                self.assertIn("generate_dynamic", funcs)
                self.assertEqual(funcs["generate_dynamic"]["required_models"], [])

                # Top-level models should be present
                self.assertIn("models", manifest)
                self.assertEqual(manifest["models"]["sdxl"], "stabilityai/stable-diffusion-xl-base-1.0")
                self.assertEqual(manifest["models"]["controlnet"], "lllyasviel/control_v11p_sd15_canny")

            finally:
                os.chdir(original_cwd)
                sys.path[:] = original_path
                _cleanup_modules("funcs_a")

    def test_missing_model_key_warning(self) -> None:
        """Test that warning is issued for model keys not in config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            original_cwd = os.getcwd()
            original_path = sys.path.copy()

            try:
                os.chdir(root)
                sys.path.insert(0, str(root / "src"))

                # Create pyproject.toml WITHOUT the required model
                pyproject = root / "pyproject.toml"
                pyproject.write_text("""
[project]
name = "test-project"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = ["gen-worker"]

[tool.cozy.functions]
modules = ["funcs_b"]

[tool.cozy.models]
other = "some/other-model"
""")

                # Create test module that requires sdxl
                src_dir = root / "src" / "funcs_b"
                src_dir.mkdir(parents=True)
                (src_dir / "__init__.py").write_text("""
from typing import Annotated
import msgspec
from gen_worker import ActionContext, worker_function
from gen_worker.injection import ModelRef, ModelRefSource as Src

class Input(msgspec.Struct):
    prompt: str

class Output(msgspec.Struct):
    result: str

class MockPipeline:
    pass

@worker_function()
def generate(
    ctx: ActionContext,
    pipeline: Annotated[MockPipeline, ModelRef(Src.DEPLOYMENT, "sdxl")],
    payload: Input,
) -> Output:
    return Output(result="ok")
""")

                # Run discovery - should print warning but not fail
                stderr = io.StringIO()
                with patch('sys.stderr', stderr):
                    manifest = discover_manifest(root)

                warning = stderr.getvalue()
                self.assertIn("sdxl", warning)
                self.assertIn("not defined", warning.lower())

                # Function should still have required_models
                self.assertEqual(manifest["functions"][0]["required_models"], ["sdxl"])

            finally:
                os.chdir(original_cwd)
                sys.path[:] = original_path
                _cleanup_modules("funcs_b")


class TestManifestModelsField(unittest.TestCase):
    """Tests for top-level models field in manifest."""

    def test_models_from_config(self) -> None:
        """Test that models from [tool.cozy.models] are included."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            original_cwd = os.getcwd()
            original_path = sys.path.copy()

            try:
                os.chdir(root)
                sys.path.insert(0, str(root / "src"))

                pyproject = root / "pyproject.toml"
                pyproject.write_text("""
[project]
name = "test-project"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = ["gen-worker"]

[tool.cozy.functions]
modules = ["funcs_c"]

[tool.cozy.models]
model-a = "org/model-a"
model-b = "org/model-b"
""")

                src_dir = root / "src" / "funcs_c"
                src_dir.mkdir(parents=True)
                (src_dir / "__init__.py").write_text("""
import msgspec
from gen_worker import ActionContext, worker_function

class Input(msgspec.Struct):
    x: int

class Output(msgspec.Struct):
    y: int

@worker_function()
def simple(ctx: ActionContext, payload: Input) -> Output:
    return Output(y=payload.x)
""")

                manifest = discover_manifest(root)

                self.assertIn("models", manifest)
                self.assertEqual(manifest["models"]["model-a"], "org/model-a")
                self.assertEqual(manifest["models"]["model-b"], "org/model-b")

            finally:
                os.chdir(original_cwd)
                sys.path[:] = original_path
                _cleanup_modules("funcs_c")

    def test_no_models_section(self) -> None:
        """Test manifest when no [tool.cozy.models] is defined."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            original_cwd = os.getcwd()
            original_path = sys.path.copy()

            try:
                os.chdir(root)
                sys.path.insert(0, str(root / "src"))

                pyproject = root / "pyproject.toml"
                pyproject.write_text("""
[project]
name = "test-project"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = ["gen-worker"]

[tool.cozy.functions]
modules = ["funcs_d"]
""")

                src_dir = root / "src" / "funcs_d"
                src_dir.mkdir(parents=True)
                (src_dir / "__init__.py").write_text("""
import msgspec
from gen_worker import ActionContext, worker_function

class Input(msgspec.Struct):
    x: int

class Output(msgspec.Struct):
    y: int

@worker_function()
def simple(ctx: ActionContext, payload: Input) -> Output:
    return Output(y=payload.x)
""")

                manifest = discover_manifest(root)

                # No models in config means no models field
                self.assertNotIn("models", manifest)
                # Function should have empty required_models
                self.assertEqual(manifest["functions"][0]["required_models"], [])

            finally:
                os.chdir(original_cwd)
                sys.path[:] = original_path
                _cleanup_modules("funcs_d")


if __name__ == "__main__":
    unittest.main()
