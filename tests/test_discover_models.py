"""Tests for model extraction in discover.py."""

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


class TestDiscoverModels(unittest.TestCase):
    def test_discovery_emits_fixed_and_payload_keyspaces(self) -> None:
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
""".lstrip(),
                    encoding="utf-8",
                )
                (root / "endpoint.toml").write_text(
                    """
schema_version = 1
name = "test-project"
main = "funcs_a"

[models]
sdxl = { ref = "stabilityai/stable-diffusion-xl-base-1.0", dtypes = ["fp16", "bf16"] }

[models.generate_dynamic]
base = { ref = "stabilityai/stable-diffusion-xl-base-1.0", dtypes = ["fp16", "bf16"] }
""".lstrip(),
                    encoding="utf-8",
                )

                src_dir = root / "src" / "funcs_a"
                src_dir.mkdir(parents=True)
                (src_dir / "__init__.py").write_text(
                    """
from typing import Annotated
import msgspec
from gen_worker import RequestContext, worker_function
from gen_worker.injection import ModelRef, ModelRefSource as Src

class Input(msgspec.Struct):
    prompt: str
    model_key: str = "base"

class Output(msgspec.Struct):
    result: str

class MockPipeline:
    pass

@worker_function()
def generate_fixed(
    ctx: RequestContext,
    pipeline: Annotated[MockPipeline, ModelRef(Src.FIXED, "sdxl")],
    payload: Input,
) -> Output:
    return Output(result="ok")

@worker_function()
def generate_dynamic(
    ctx: RequestContext,
    pipeline: Annotated[MockPipeline, ModelRef(Src.PAYLOAD, "model_key")],
    payload: Input,
) -> Output:
    return Output(result="ok")
""".lstrip(),
                    encoding="utf-8",
                )

                manifest = discover_manifest(root)
                funcs_by_python = {f["python_name"]: f for f in manifest["functions"]}

                self.assertEqual(funcs_by_python["generate_fixed"]["required_models"], ["sdxl"])
                self.assertEqual(funcs_by_python["generate_dynamic"]["required_models"], [])
                self.assertEqual(
                    funcs_by_python["generate_dynamic"]["payload_repo_selectors"],
                    [{"field": "model_key", "kind": "short_key"}],
                )

                self.assertEqual(
                    manifest["models"]["sdxl"]["ref"],
                    "stabilityai/stable-diffusion-xl-base-1.0",
                )
                self.assertEqual(
                    manifest["models"]["sdxl"]["dtypes"],
                    ["fp16", "bf16"],
                )
                self.assertEqual(
                    manifest["models_by_function"]["generate-dynamic"]["base"]["ref"],
                    "stabilityai/stable-diffusion-xl-base-1.0",
                )

            finally:
                os.chdir(original_cwd)
                sys.path[:] = original_path
                _cleanup_modules("funcs_a")

    def test_missing_fixed_key_fails_discovery(self) -> None:
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
""".lstrip(),
                    encoding="utf-8",
                )
                (root / "endpoint.toml").write_text(
                    """
schema_version = 1
name = "test-project"
main = "funcs_b"
""".lstrip(),
                    encoding="utf-8",
                )

                src_dir = root / "src" / "funcs_b"
                src_dir.mkdir(parents=True)
                (src_dir / "__init__.py").write_text(
                    """
from typing import Annotated
import msgspec
from gen_worker import RequestContext, worker_function
from gen_worker.injection import ModelRef, ModelRefSource as Src

class Input(msgspec.Struct):
    prompt: str

class Output(msgspec.Struct):
    result: str

class MockPipeline:
    pass

@worker_function()
def generate(
    ctx: RequestContext,
    pipeline: Annotated[MockPipeline, ModelRef(Src.FIXED, "sdxl")],
    payload: Input,
) -> Output:
    return Output(result="ok")
""".lstrip(),
                    encoding="utf-8",
                )

                with self.assertRaises(ValueError) as ctx:
                    discover_manifest(root)
                self.assertIn("missing from endpoint.toml [models]", str(ctx.exception))

            finally:
                os.chdir(original_cwd)
                sys.path[:] = original_path
                _cleanup_modules("funcs_b")

    def test_missing_payload_keyspace_fails_discovery(self) -> None:
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
""".lstrip(),
                    encoding="utf-8",
                )
                (root / "endpoint.toml").write_text(
                    """
schema_version = 1
name = "test-project"
main = "funcs_c"
""".lstrip(),
                    encoding="utf-8",
                )

                src_dir = root / "src" / "funcs_c"
                src_dir.mkdir(parents=True)
                (src_dir / "__init__.py").write_text(
                    """
from typing import Annotated
import msgspec
from gen_worker import RequestContext, worker_function
from gen_worker.injection import ModelRef, ModelRefSource as Src

class Input(msgspec.Struct):
    prompt: str
    model_key: str

class Output(msgspec.Struct):
    result: str

class MockPipeline:
    pass

@worker_function()
def generate(
    ctx: RequestContext,
    pipeline: Annotated[MockPipeline, ModelRef(Src.PAYLOAD, "model_key")],
    payload: Input,
) -> Output:
    return Output(result="ok")
""".lstrip(),
                    encoding="utf-8",
                )

                with self.assertRaises(ValueError) as ctx:
                    discover_manifest(root)
                self.assertIn("missing [models.generate]", str(ctx.exception))

            finally:
                os.chdir(original_cwd)
                sys.path[:] = original_path
                _cleanup_modules("funcs_c")

    def test_batch_dimension_emitted(self) -> None:
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
""".lstrip(),
                    encoding="utf-8",
                )
                (root / "endpoint.toml").write_text(
                    """
schema_version = 1
name = "test-project"
main = "funcs_d"

[functions.caption]
batch_dimension = "items"
""".lstrip(),
                    encoding="utf-8",
                )

                src_dir = root / "src" / "funcs_d"
                src_dir.mkdir(parents=True)
                (src_dir / "__init__.py").write_text(
                    """
import msgspec
from gen_worker import RequestContext, worker_function

class Input(msgspec.Struct):
    items: list[str]

class Output(msgspec.Struct):
    ok: bool

@worker_function()
def caption(ctx: RequestContext, payload: Input) -> Output:
    return Output(ok=True)
""".lstrip(),
                    encoding="utf-8",
                )

                manifest = discover_manifest(root)
                fn = next(f for f in manifest["functions"] if f["name"] == "caption")
                self.assertEqual(fn["batch_dimension"], "items")

            finally:
                os.chdir(original_cwd)
                sys.path[:] = original_path
                _cleanup_modules("funcs_d")

    def test_inline_ref_in_modelref_rejected(self) -> None:
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
""".lstrip(),
                    encoding="utf-8",
                )
                (root / "endpoint.toml").write_text(
                    """
schema_version = 1
name = "test-project"
main = "funcs_e"

[models]
sdxl = "stabilityai/stable-diffusion-xl-base-1.0"
""".lstrip(),
                    encoding="utf-8",
                )

                src_dir = root / "src" / "funcs_e"
                src_dir.mkdir(parents=True)
                (src_dir / "__init__.py").write_text(
                    """
from typing import Annotated
import msgspec
from gen_worker import RequestContext, worker_function
from gen_worker.injection import ModelRef, ModelRefSource as Src

class Input(msgspec.Struct):
    prompt: str

class Output(msgspec.Struct):
    result: str

class MockPipeline:
    pass

@worker_function()
def generate(
    ctx: RequestContext,
    pipeline: Annotated[
        MockPipeline,
        ModelRef(Src.FIXED, "sdxl", ref="stabilityai/stable-diffusion-xl-base-1.0"),
    ],
    payload: Input,
) -> Output:
    return Output(result="ok")
""".lstrip(),
                    encoding="utf-8",
                )

                with self.assertRaises(ValueError) as ctx:
                    discover_manifest(root)
                self.assertIn("inline ref/dtypes", str(ctx.exception))

            finally:
                os.chdir(original_cwd)
                sys.path[:] = original_path
                _cleanup_modules("funcs_e")


if __name__ == "__main__":
    unittest.main()
