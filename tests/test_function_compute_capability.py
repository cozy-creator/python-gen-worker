import os
import sys
import tempfile
import unittest
from pathlib import Path

from gen_worker.api.decorators import ResourceRequirements
from gen_worker.discovery.discover import discover_manifest


def _cleanup_modules(prefix: str) -> None:
    for mod in list(sys.modules.keys()):
        if mod.startswith(prefix):
            del sys.modules[mod]


class TestFunctionComputeCapability(unittest.TestCase):
    def test_resource_requirements_canonicalizes_compute_capability_min(self) -> None:
        req = ResourceRequirements(compute_capability_min=10)
        self.assertEqual(
            req.to_dict(),
            {"compute_capability": {"min": "10.0"}},
        )

    def test_resource_requirements_rejects_invalid_compute_capability_min(self) -> None:
        with self.assertRaises(ValueError):
            ResourceRequirements(compute_capability_min=0)

    def test_discovery_emits_function_compute_capability_requirement(self) -> None:
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
                (root / "endpoint.toml").write_text(
                    """
schema_version = 1
name = "test-project"
main = "endpoint_mod_cc.main"
""".lstrip(),
                    encoding="utf-8",
                )
                src_dir = root / "src" / "endpoint_mod_cc"
                src_dir.mkdir(parents=True)
                (src_dir / "main.py").write_text(
                    """
import msgspec
from gen_worker import RequestContext, ResourceRequirements, worker_function

class Input(msgspec.Struct):
    x: int

class Output(msgspec.Struct):
    y: int

@worker_function(resources=ResourceRequirements(compute_capability_min=10))
def generate_nvfp4(ctx: RequestContext, payload: Input) -> Output:
    return Output(y=payload.x)
""".lstrip(),
                    encoding="utf-8",
                )

                manifest = discover_manifest(root)
                funcs = manifest.get("functions") or []
                self.assertEqual(len(funcs), 1)
                resources = funcs[0].get("resources") or {}
                self.assertEqual(resources.get("compute_capability"), {"min": "10.0"})
            finally:
                os.chdir(original_cwd)
                sys.path[:] = original_path
                _cleanup_modules("endpoint_mod_cc")


if __name__ == "__main__":
    unittest.main()
