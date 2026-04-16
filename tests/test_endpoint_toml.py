import tempfile
import unittest
from pathlib import Path

from gen_worker.discovery.toml_manifest import constraint_satisfied, load_endpoint_toml


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
joycaption = { ref = "fancyfeast/llama-joycaption-beta-one-hf-llava", dtypes = ["fp16", "bf16"] }

[models.generate]
sdxl = { ref = "stabilityai/stable-diffusion-xl-base-1.0", dtypes = ["fp16"] }
""".lstrip(),
                encoding="utf-8",
            )
            cfg = load_endpoint_toml(p)
            self.assertIn("joycaption", cfg.models)
            self.assertEqual(cfg.models["joycaption"].dtypes, ("fp16", "bf16"))
            self.assertIn("generate", cfg.function_models)
            self.assertEqual(cfg.function_models["generate"]["sdxl"].ref, "stabilityai/stable-diffusion-xl-base-1.0")

    def test_legacy_function_models_under_functions_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "endpoint.toml"
            p.write_text(
                """
schema_version = 1
name = "x"
main = "x.main"

[functions.generate.models.model_key]
sdxl = { ref = "stabilityai/stable-diffusion-xl-base-1.0", dtypes = ["fp16"] }
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
sdxl = { ref = "cozy:stabilityai/stable-diffusion-xl-base-1.0", dtypes = ["fp16"] }
""".lstrip(),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError) as ctx:
                load_endpoint_toml(p)
            self.assertIn("must not include a scheme prefix", str(ctx.exception))

    def test_invalid_dtype_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "endpoint.toml"
            p.write_text(
                """
schema_version = 1
name = "x"
main = "x.main"

[models]
m = { ref = "o/r", dtypes = ["fp16","wat"] }
""".lstrip(),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_endpoint_toml(p)

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
            self.assertEqual(cfg.resources["max_inflight_requests"], 2)
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
            self.assertEqual(cfg.resources["max_inflight_requests"], 1)

    def test_function_gpu_hints_parsed(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "endpoint.toml"
            p.write_text(
                """
schema_version = 1
name = "x"
main = "x.main"

[functions.convert_low_precision.resources]
kind = "conversion"
requires_gpu = true
compute_capability_min = 9.0
min_vram_gb = 24
supported_precisions = ["fp8", "nvfp4"]
supported_conversion_profiles = ["fp8:e4m3", "fp8:mxfp8", "nvfp4"]
""".lstrip(),
                encoding="utf-8",
            )
            cfg = load_endpoint_toml(p)
            hints = cfg.function_resources["convert-low-precision"]
            self.assertEqual(hints["requires_gpu"], True)
            self.assertEqual(hints["compute_capability_min"], "9.0")
            self.assertEqual(hints["min_vram_gb"], 24.0)
            self.assertEqual(hints["supported_precisions"], ["fp8", "nvfp4"])
            self.assertEqual(
                hints["supported_conversion_profiles"],
                ["fp8:e4m3", "fp8:mxfp8", "nvfp4"],
            )

    def test_function_vram_multiplier(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "endpoint.toml"
            p.write_text(
                """
schema_version = 1
name = "x"
main = "x.main"

[functions.convert_quantize_calibrated.resources]
kind = "conversion"
vram_multiplier = 1.5
""".lstrip(),
                encoding="utf-8",
            )
            cfg = load_endpoint_toml(p)
            hints = cfg.function_resources["convert-quantize-calibrated"]
            self.assertEqual(hints["vram_multiplier"], 1.5)

    def test_function_vram_multiplier_must_be_positive(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "endpoint.toml"
            p.write_text(
                """
schema_version = 1
name = "x"
main = "x.main"

[functions.convert_quantize_calibrated.resources]
vram_multiplier = 0
""".lstrip(),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_endpoint_toml(p)

    def test_function_vram_multiplier_must_be_numeric(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "endpoint.toml"
            p.write_text(
                """
schema_version = 1
name = "x"
main = "x.main"

[functions.convert_quantize_calibrated.resources]
vram_multiplier = "high"
""".lstrip(),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_endpoint_toml(p)

    def test_function_requires_gpu_must_be_boolean(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "endpoint.toml"
            p.write_text(
                """
schema_version = 1
name = "x"
main = "x.main"

[functions.convert_low_precision.resources]
requires_gpu = "true"
""".lstrip(),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_endpoint_toml(p)


if __name__ == "__main__":
    unittest.main()
