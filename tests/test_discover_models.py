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
                (root / "tensorhub.toml").write_text(
                    """
schema_version = 1
name = "test-project"
main = "funcs_a"

[functions.generate_dynamic.models.model_key]
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
from gen_worker import ActionContext, worker_function
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
    ctx: ActionContext,
    pipeline: Annotated[
        MockPipeline,
        ModelRef(
            Src.FIXED,
            "sdxl",
            ref="stabilityai/stable-diffusion-xl-base-1.0",
            dtypes=("fp16", "bf16"),
        ),
    ],
    payload: Input,
) -> Output:
    return Output(result="ok")

@worker_function()
def generate_dynamic(
    ctx: ActionContext,
    pipeline: Annotated[MockPipeline, ModelRef(Src.INPUT_PAYLOAD, "model_key")],
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

                mbf = manifest["models_by_function"]
                self.assertEqual(
                    mbf["generate-fixed"]["fixed"]["sdxl"]["ref"],
                    "stabilityai/stable-diffusion-xl-base-1.0",
                )
                self.assertEqual(
                    mbf["generate-fixed"]["fixed"]["sdxl"]["dtypes"],
                    ["fp16", "bf16"],
                )
                self.assertEqual(
                    mbf["generate-dynamic"]["payload_selectors"]["model_key"]["base"]["ref"],
                    "stabilityai/stable-diffusion-xl-base-1.0",
                )

            finally:
                os.chdir(original_cwd)
                sys.path[:] = original_path
                _cleanup_modules("funcs_a")

    def test_missing_fixed_ref_fails_discovery(self) -> None:
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
                (root / "tensorhub.toml").write_text(
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
    pipeline: Annotated[MockPipeline, ModelRef(Src.FIXED, "sdxl")],
    payload: Input,
) -> Output:
    return Output(result="ok")
""".lstrip(),
                    encoding="utf-8",
                )

                with self.assertRaises(ValueError) as ctx:
                    discover_manifest(root)
                self.assertIn("FIXED model keys", str(ctx.exception))

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
                (root / "tensorhub.toml").write_text(
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
from gen_worker import ActionContext, worker_function
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
    ctx: ActionContext,
    pipeline: Annotated[MockPipeline, ModelRef(Src.INPUT_PAYLOAD, "model_key")],
    payload: Input,
) -> Output:
    return Output(result="ok")
""".lstrip(),
                    encoding="utf-8",
                )

                with self.assertRaises(ValueError) as ctx:
                    discover_manifest(root)
                self.assertIn("[functions.generate.models.model_key]", str(ctx.exception))

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
                (root / "tensorhub.toml").write_text(
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
from gen_worker import ActionContext, worker_function

class Input(msgspec.Struct):
    items: list[str]

class Output(msgspec.Struct):
    ok: bool

@worker_function()
def caption(ctx: ActionContext, payload: Input) -> Output:
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

    def test_fixed_ref_with_scheme_prefix_fails_discovery(self) -> None:
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
                (root / "tensorhub.toml").write_text(
                    """
schema_version = 1
name = "test-project"
main = "funcs_e"
""".lstrip(),
                    encoding="utf-8",
                )

                src_dir = root / "src" / "funcs_e"
                src_dir.mkdir(parents=True)
                (src_dir / "__init__.py").write_text(
                    """
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
    pipeline: Annotated[
        MockPipeline,
        ModelRef(Src.FIXED, "sdxl", ref="hf:stabilityai/stable-diffusion-xl-base-1.0"),
    ],
    payload: Input,
) -> Output:
    return Output(result="ok")
""".lstrip(),
                    encoding="utf-8",
                )

                with self.assertRaises(ValueError) as ctx:
                    discover_manifest(root)
                self.assertIn("must not include a scheme prefix", str(ctx.exception))

            finally:
                os.chdir(original_cwd)
                sys.path[:] = original_path
                _cleanup_modules("funcs_e")


if __name__ == "__main__":
    unittest.main()
