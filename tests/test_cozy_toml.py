import tempfile
import unittest
from pathlib import Path

from gen_worker.cozy_toml import constraint_satisfied, load_cozy_toml


class TestCozyToml(unittest.TestCase):
    def test_schema_version_required(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "cozy.toml"
            p.write_text("name='x'\nmain='x.main'\ngen_worker='>=0'\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_cozy_toml(p)

    def test_constraint_satisfied(self) -> None:
        self.assertTrue(constraint_satisfied(">=0.2.0,<0.3.0", "0.2.1"))
        self.assertFalse(constraint_satisfied(">=0.2.0,<0.3.0", "0.3.0"))
        self.assertTrue(constraint_satisfied("~=0.2.1", "0.2.9"))
        self.assertFalse(constraint_satisfied("~=0.2.1", "0.3.0"))
        self.assertTrue(constraint_satisfied("0.2.1", "0.2.1"))
        self.assertFalse(constraint_satisfied("0.2.1", "0.2.2"))

    def test_models_string_defaults_dtypes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "cozy.toml"
            p.write_text(
                """
schema_version = 1
name = "x"
main = "x.main"
gen_worker = ">=0"

[models]
sd15 = "hf:stable-diffusion-v1-5/stable-diffusion-v1-5"
""".lstrip(),
                encoding="utf-8",
            )
            cfg = load_cozy_toml(p)
            self.assertEqual(cfg.models["sd15"].ref, "hf:stable-diffusion-v1-5/stable-diffusion-v1-5")
            self.assertEqual(cfg.models["sd15"].dtypes, ("fp16", "bf16"))

    def test_models_table_with_fp8(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "cozy.toml"
            p.write_text(
                """
schema_version = 1
name = "x"
main = "x.main"
gen_worker = ">=0"

[models]
flux = { ref = "hf:black-forest-labs/FLUX.2-klein-4B", dtypes = ["fp8"] }
""".lstrip(),
                encoding="utf-8",
            )
            cfg = load_cozy_toml(p)
            self.assertEqual(cfg.models["flux"].dtypes, ("fp8",))

    def test_invalid_dtype_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "cozy.toml"
            p.write_text(
                """
schema_version = 1
name = "x"
main = "x.main"
gen_worker = ">=0"

[models]
m = { ref = "hf:o/r", dtypes = ["fp16","wat"] }
""".lstrip(),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_cozy_toml(p)

    def test_endpoint_models_override(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "cozy.toml"
            p.write_text(
                """
schema_version = 1
name = "x"
main = "x.main"
gen_worker = ">=0"

[models]
sdxl = "hf:stabilityai/stable-diffusion-xl-base-1.0"

[endpoints.generate.models]
sdxl = { ref = "hf:stabilityai/stable-diffusion-xl-base-1.0", dtypes = ["fp16"] }
""".lstrip(),
                encoding="utf-8",
            )
            cfg = load_cozy_toml(p)
            self.assertIn("generate", cfg.endpoint_models)
            self.assertEqual(cfg.endpoint_models["generate"]["sdxl"].dtypes, ("fp16",))

    def test_invalid_cuda_constraint_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "cozy.toml"
            p.write_text(
                """
schema_version = 1
name = "x"
main = "x.main"
gen_worker = ">=0"

[host.requirements]
cuda = ">=12.6,<wat"
""".lstrip(),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_cozy_toml(p)


if __name__ == "__main__":
    unittest.main()
