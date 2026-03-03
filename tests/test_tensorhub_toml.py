import tempfile
import unittest
from pathlib import Path

from gen_worker.tensorhub_toml import constraint_satisfied, load_tensorhub_toml


class TestTensorhubToml(unittest.TestCase):
    def test_schema_version_required(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "tensorhub.toml"
            p.write_text("name='x'\nmain='x.main'\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_tensorhub_toml(p)

    def test_constraint_satisfied(self) -> None:
        self.assertTrue(constraint_satisfied(">=12.0,<13.0", "12.6"))
        self.assertFalse(constraint_satisfied(">=12.0,<13.0", "13.0"))

    def test_top_level_models_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "tensorhub.toml"
            p.write_text(
                """
schema_version = 1
name = "x"
main = "x.main"

[models]
sd15 = "stable-diffusion-v1-5/stable-diffusion-v1-5"
""".lstrip(),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_tensorhub_toml(p)

    def test_function_payload_selector_models_parsed(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "tensorhub.toml"
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
            cfg = load_tensorhub_toml(p)
            self.assertIn("generate", cfg.function_models)
            self.assertIn("model_key", cfg.function_models["generate"])
            self.assertEqual(
                cfg.function_models["generate"]["model_key"]["sdxl"].dtypes,
                ("fp16",),
            )

    def test_model_ref_with_scheme_prefix_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "tensorhub.toml"
            p.write_text(
                """
schema_version = 1
name = "x"
main = "x.main"

[functions.generate.models.model_key]
sdxl = { ref = "cozy:stabilityai/stable-diffusion-xl-base-1.0", dtypes = ["fp16"] }
""".lstrip(),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError) as ctx:
                load_tensorhub_toml(p)
            self.assertIn("must not include a scheme prefix", str(ctx.exception))

    def test_invalid_dtype_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "tensorhub.toml"
            p.write_text(
                """
schema_version = 1
name = "x"
main = "x.main"

[functions.generate.models.model_key]
m = { ref = "o/r", dtypes = ["fp16","wat"] }
""".lstrip(),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_tensorhub_toml(p)

    def test_invalid_cuda_constraint_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "tensorhub.toml"
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
                load_tensorhub_toml(p)

    def test_compute_capabilities_parsed(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "tensorhub.toml"
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
            cfg = load_tensorhub_toml(p)
            self.assertEqual(cfg.compute_capabilities, ("8.0", "8.x", ">=12.0,<13.0"))

    def test_cuda_compute_capabilities_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "tensorhub.toml"
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
                load_tensorhub_toml(p)

    def test_function_max_concurrency_in_toml_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "tensorhub.toml"
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
                load_tensorhub_toml(p)

    def test_function_batch_dimension_and_endpoint_max_inflight_parsed(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "tensorhub.toml"
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
            cfg = load_tensorhub_toml(p)
            self.assertEqual(cfg.resources["max_inflight_requests"], 2)
            self.assertEqual(cfg.function_batch_dimensions["caption"], "items")

    def test_batch_dimension_path_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "tensorhub.toml"
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
                load_tensorhub_toml(p)

    def test_endpoint_max_inflight_defaults_to_one(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "tensorhub.toml"
            p.write_text(
                """
schema_version = 1
name = "x"
main = "x.main"
""".lstrip(),
                encoding="utf-8",
            )
            cfg = load_tensorhub_toml(p)
            self.assertEqual(cfg.resources["max_inflight_requests"], 1)


if __name__ == "__main__":
    unittest.main()
