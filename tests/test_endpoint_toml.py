import tempfile
import unittest
from pathlib import Path

from gen_worker.discovery.toml_manifest import (
    constraint_satisfied,
    load_endpoint_toml,
    load_endpoint_toml_with_warnings,
)


class TestEndpointToml(unittest.TestCase):
    def test_schema_version_required(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "endpoint.toml"
            p.write_text("name='x'\nmain='x.main'\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_endpoint_toml(p)

    def test_constraint_satisfied(self) -> None:
        self.assertTrue(constraint_satisfied(">=12.0,<13.0", "12.6"))
        self.assertFalse(constraint_satisfied(">=12.0,<13.0", "13.0"))

    def test_models_top_level_and_function_keyspace_parsed(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "endpoint.toml"
            p.write_text(
                """
schema_version = 1
name = "x"
main = "x.main"

[models]
joycaption = { ref = "fancyfeast/llama-joycaption-beta-one-hf-llava", flavor = "bf16", dtype = "bf16", file_layout = "diffusers" }

[models.generate]
sdxl = { ref = "stabilityai/stable-diffusion-xl-base-1.0#fp16" }
""".lstrip(),
                encoding="utf-8",
            )
            cfg = load_endpoint_toml(p)
            self.assertIn("joycaption", cfg.models)
            self.assertEqual(cfg.models["joycaption"].flavor, "bf16")
            self.assertEqual(cfg.models["joycaption"].dtype, "bf16")
            self.assertEqual(cfg.models["joycaption"].file_layout, "diffusers")
            self.assertIn("generate", cfg.function_models)
            self.assertEqual(
                cfg.function_models["generate"]["sdxl"].ref,
                "stabilityai/stable-diffusion-xl-base-1.0",
            )
            self.assertEqual(cfg.function_models["generate"]["sdxl"].flavor, "fp16")

    def test_legacy_function_models_under_functions_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "endpoint.toml"
            p.write_text(
                """
schema_version = 1
name = "x"
main = "x.main"

[functions.generate.models.model_key]
sdxl = { ref = "stabilityai/stable-diffusion-xl-base-1.0", flavor = "fp16" }
""".lstrip(),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_endpoint_toml(p)

    def test_model_ref_with_scheme_prefix_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "endpoint.toml"
            p.write_text(
                """
schema_version = 1
name = "x"
main = "x.main"

[models]
sdxl = { ref = "cozy:stabilityai/stable-diffusion-xl-base-1.0", flavor = "fp16" }
""".lstrip(),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError) as ctx:
                load_endpoint_toml(p)
            self.assertIn("must not include a scheme prefix", str(ctx.exception))

    def test_legacy_dtypes_field_is_rejected(self) -> None:
        """The legacy `dtypes = [...]` field has been hard-cut."""
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "endpoint.toml"
            p.write_text(
                """
schema_version = 1
name = "x"
main = "x.main"

[models]
m = { ref = "o/r", dtypes = ["bf16", "fp16"] }
""".lstrip(),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError) as ctx:
                load_endpoint_toml(p)
            self.assertIn("'dtypes' field removed", str(ctx.exception))

    def test_invalid_cuda_constraint_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "endpoint.toml"
            p.write_text(
                """
schema_version = 1
name = "x"
main = "x.main"

[host.requirements]
cuda = ">=12.6,<wat"
""".lstrip(),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_endpoint_toml(p)

    def test_compute_capabilities_parsed(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "endpoint.toml"
            p.write_text(
                """
schema_version = 1
name = "x"
main = "x.main"

[host.requirements]
compute_capabilities = ["8.0", "8.x", ">=12.0,<13.0"]
""".lstrip(),
                encoding="utf-8",
            )
            cfg = load_endpoint_toml(p)
            self.assertEqual(cfg.compute_capabilities, ("8.0", "8.x", ">=12.0,<13.0"))

    def test_cuda_compute_capabilities_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "endpoint.toml"
            p.write_text(
                """
schema_version = 1
name = "x"
main = "x.main"

[host.requirements]
cuda_compute_capabilities = ["8.0"]
""".lstrip(),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_endpoint_toml(p)

    def test_function_max_concurrency_in_toml_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "endpoint.toml"
            p.write_text(
                """
schema_version = 1
name = "x"
main = "x.main"

[functions.generate.resources]
max_concurrency = 2
""".lstrip(),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_endpoint_toml(p)

    def test_function_max_inflight_in_toml_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "endpoint.toml"
            p.write_text(
                """
schema_version = 1
name = "x"
main = "x.main"

[functions.generate.resources]
max_inflight_requests = 2
""".lstrip(),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_endpoint_toml(p)

    def test_function_hardware_resources_parsed(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "endpoint.toml"
            p.write_text(
                """
schema_version = 1
name = "x"
main = "x.main"

[functions.cpu_convert.resources]
accelerator = "none"

[functions.gpu_convert.resources]
accelerator = "cuda"
accelerator_preference = "required"
cuda_compute_min = 9
min_vram_gb = 24
vram_multiplier = 1.75
supported_precisions = ["fp8", "nvfp4"]
required_libraries = ["gpu_quant_lib"]
""".lstrip(),
                encoding="utf-8",
            )
            cfg = load_endpoint_toml(p)
            self.assertEqual(cfg.function_resources["cpu-convert"]["accelerator"], "none")
            gpu = cfg.function_resources["gpu-convert"]
            self.assertEqual(gpu["accelerator"], "cuda")
            self.assertEqual(gpu["accelerator_preference"], "required")
            self.assertEqual(gpu["compute_capability_min"], "9.0")
            self.assertEqual(gpu["cuda_compute_min"], "9.0")
            self.assertEqual(gpu["min_vram_gb"], 24.0)
            self.assertEqual(gpu["vram_multiplier"], 1.75)
            self.assertEqual(gpu["supported_precisions"], ["fp8", "nvfp4"])
            self.assertEqual(gpu["required_libraries"], ["gpu_quant_lib"])

    def test_function_batch_dimension_and_endpoint_max_inflight_parsed(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "endpoint.toml"
            p.write_text(
                """
schema_version = 1
name = "x"
main = "x.main"

[functions.caption]
batch_dimension = "items"

[resources]
max_inflight_requests = 2
""".lstrip(),
                encoding="utf-8",
            )
            cfg = load_endpoint_toml(p)
            self.assertEqual(cfg.resources.max_inflight_requests, 2)
            self.assertEqual(cfg.function_batch_dimensions["caption"], "items")

    def test_batch_dimension_path_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "endpoint.toml"
            p.write_text(
                """
schema_version = 1
name = "x"
main = "x.main"

[functions.caption]
batch_dimension_path = "items"
""".lstrip(),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_endpoint_toml(p)

    def test_endpoint_max_inflight_defaults_to_one(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "endpoint.toml"
            p.write_text(
                """
schema_version = 1
name = "x"
main = "x.main"
""".lstrip(),
                encoding="utf-8",
            )
            cfg = load_endpoint_toml(p)
            self.assertEqual(cfg.resources.max_inflight_requests, 1)

class TestEndpointTomlModelSelectors(unittest.TestCase):
    """Hard-cut model selector shape on [models] entries."""

    def _load(self, body: str):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "endpoint.toml"
            p.write_text(body.lstrip(), encoding="utf-8")
            return load_endpoint_toml_with_warnings(p)

    def test_explicit_selector_fields_parse(self) -> None:
        cfg, warnings = self._load(
            """
schema_version = 1
name = "x"
main = "x.main"

[models]
klein = { ref = "paul/klein-4b", flavor = "bf16", dtype = "bf16", file_layout = "diffusers", file_type = "safetensors" }
"""
        )
        spec = cfg.models["klein"]
        self.assertEqual(spec.ref, "paul/klein-4b")
        self.assertEqual(spec.flavor, "bf16")
        self.assertEqual(spec.dtype, "bf16")
        self.assertEqual(spec.file_layout, "diffusers")
        self.assertEqual(spec.file_type, "safetensors")
        self.assertEqual(warnings, [])

    def test_flavor_fallback_list_parses(self) -> None:
        cfg, _ = self._load(
            """
schema_version = 1
name = "x"
main = "x.main"

[models]
klein = { ref = "paul/klein-4b", flavors = ["bf16", "fp8"] }
"""
        )
        spec = cfg.models["klein"]
        self.assertEqual(spec.flavors, ("bf16", "fp8"))

    def test_legacy_dtypes_field_rejected(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            self._load(
                """
schema_version = 1
name = "x"
main = "x.main"

[models]
klein = { ref = "paul/klein-4b", dtypes = ["bf16"] }
"""
        )
        self.assertIn("'dtypes' field removed", str(ctx.exception))

    def test_legacy_attributes_field_rejected(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            self._load(
                """
schema_version = 1
name = "x"
main = "x.main"

[models]
klein = { ref = "paul/klein-4b", attributes = { dtype = "bf16" } }
"""
            )
        self.assertIn("'attributes' field removed", str(ctx.exception))

    def test_bare_string_ref_still_works(self) -> None:
        cfg, warnings = self._load(
            """
schema_version = 1
name = "x"
main = "x.main"

[models]
klein = "paul/klein-4b"
"""
        )
        spec = cfg.models["klein"]
        self.assertEqual(spec.ref, "paul/klein-4b")
        self.assertEqual(spec.flavor, "")
        self.assertEqual(warnings, [])

    def test_ref_flavor_selector_parses(self) -> None:
        cfg, _ = self._load(
            """
schema_version = 1
name = "x"
main = "x.main"

[models]
klein = "paul/klein-4b#bf16"
"""
        )
        spec = cfg.models["klein"]
        self.assertEqual(spec.ref, "paul/klein-4b")
        self.assertEqual(spec.flavor, "bf16")

    def test_selectors_on_function_keyspace(self) -> None:
        cfg, _ = self._load(
            """
schema_version = 1
name = "x"
main = "x.main"

[models.generate]
small = { ref = "paul/model-small", flavor = "bf16" }
large = { ref = "paul/model-large", dtype = "fp16", file_layout = "diffusers" }
"""
        )
        fn = cfg.function_models["generate"]
        self.assertEqual(fn["small"].flavor, "bf16")
        self.assertEqual(fn["large"].dtype, "fp16")
        self.assertEqual(fn["large"].file_layout, "diffusers")


if __name__ == "__main__":
    unittest.main()
