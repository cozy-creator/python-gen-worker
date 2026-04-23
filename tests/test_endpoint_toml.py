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
joycaption = { ref = "fancyfeast/llama-joycaption-beta-one-hf-llava", attributes = { dtype = "bf16", file_layout = "diffusers" } }

[models.generate]
sdxl = { ref = "stabilityai/stable-diffusion-xl-base-1.0", attributes = { dtype = "fp16" } }
""".lstrip(),
                encoding="utf-8",
            )
            cfg = load_endpoint_toml(p)
            self.assertIn("joycaption", cfg.models)
            self.assertEqual(
                cfg.models["joycaption"].attributes_as_dict(),
                {"dtype": ["bf16"], "file_layout": ["diffusers"]},
            )
            # DTypes back-filled from attributes["dtype"] during migration.
            self.assertEqual(cfg.models["joycaption"].dtypes, ("bf16",))
            self.assertIn("generate", cfg.function_models)
            self.assertEqual(
                cfg.function_models["generate"]["sdxl"].ref,
                "stabilityai/stable-diffusion-xl-base-1.0",
            )
            self.assertEqual(
                cfg.function_models["generate"]["sdxl"].attributes_as_dict(),
                {"dtype": ["fp16"]},
            )

    def test_legacy_function_models_under_functions_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "endpoint.toml"
            p.write_text(
                """
schema_version = 1
name = "x"
main = "x.main"

[functions.generate.models.model_key]
sdxl = { ref = "stabilityai/stable-diffusion-xl-base-1.0", attributes = { dtype = "fp16" } }
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
sdxl = { ref = "cozy:stabilityai/stable-diffusion-xl-base-1.0", attributes = { dtype = "fp16" } }
""".lstrip(),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError) as ctx:
                load_endpoint_toml(p)
            self.assertIn("must not include a scheme prefix", str(ctx.exception))

    def test_legacy_dtypes_field_is_rejected(self) -> None:
        """The legacy `dtypes = [...]` field has been hard-cut. Publishers must
        migrate to `attributes = { dtype = [...] }`."""
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

class TestEndpointTomlAttributesMigration(unittest.TestCase):
    """Migration tests for tensorhub #229's variant-attribute shape on
    [models] entries. See e2e/agents/progress.json issue #6."""

    def _load(self, body: str):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "endpoint.toml"
            p.write_text(body.lstrip(), encoding="utf-8")
            return load_endpoint_toml_with_warnings(p)

    def test_attributes_shape_parses(self) -> None:
        cfg, warnings = self._load(
            """
schema_version = 1
name = "x"
main = "x.main"

[models]
klein = { ref = "paul/klein-4b", attributes = { dtype = "bf16", file_layout = "diffusers", file_type = "safetensors" } }
"""
        )
        spec = cfg.models["klein"]
        self.assertEqual(
            spec.attributes_as_dict(),
            {"dtype": ["bf16"], "file_layout": ["diffusers"], "file_type": ["safetensors"]},
        )
        # strict_attributes_as_dict unwraps single-valued preferences to str.
        self.assertEqual(
            spec.strict_attributes_as_dict(),
            {"dtype": "bf16", "file_layout": "diffusers", "file_type": "safetensors"},
        )
        # DTypes back-filled from attributes["dtype"].
        self.assertEqual(spec.dtypes, ("bf16",))
        # No deprecation warning for the new shape.
        self.assertTrue(
            all("deprecated" not in w for w in warnings),
            f"unexpected deprecation warning: {warnings!r}",
        )

    def test_attribute_value_preference_list_parses(self) -> None:
        """Canonical multi-preference shape on one axis: prefer bf16, fall
        back to fp16. Other axes stay single-valued."""
        cfg, _ = self._load(
            """
schema_version = 1
name = "x"
main = "x.main"

[models]
klein = { ref = "paul/klein-4b", attributes = { dtype = ["bf16", "fp16"], file_layout = "diffusers" } }
"""
        )
        spec = cfg.models["klein"]
        self.assertEqual(
            spec.attributes_as_dict(),
            {"dtype": ["bf16", "fp16"], "file_layout": ["diffusers"]},
        )
        # strict_attributes_as_dict must raise when any attribute has multiple
        # preferences — resolver callers that want the strict form should
        # handle preference selection themselves.
        with self.assertRaises(ValueError):
            spec.strict_attributes_as_dict()
        # Legacy DTypes back-fill preserves the full preference list.
        self.assertEqual(spec.dtypes, ("bf16", "fp16"))

    def test_multi_axis_preference_lists_rejected(self) -> None:
        """At most one attribute per entry may have multiple preferences;
        two-or-more multi-valued axes require separate keyspace entries."""
        with self.assertRaises(ValueError) as ctx:
            self._load(
                """
schema_version = 1
name = "x"
main = "x.main"

[models]
klein = { ref = "paul/klein-4b", attributes = { dtype = ["bf16", "fp16"], file_layout = ["diffusers", "singlefile"] } }
"""
            )
        self.assertIn("at most one attribute", str(ctx.exception))

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
        # Bare ref → empty attributes (matches any variant).
        self.assertEqual(spec.attributes_as_dict(), {})
        self.assertEqual(spec.dtypes, ())
        self.assertEqual(warnings, [])

    def test_non_string_attribute_value_rejected_accepts_list(self) -> None:
        """int / bool attribute values are rejected; list-of-strings is
        accepted (preference list)."""
        # ints rejected
        with self.assertRaises(ValueError) as ctx:
            self._load(
                """
schema_version = 1
name = "x"
main = "x.main"

[models]
klein = { ref = "paul/klein-4b", attributes = { quant_bits = 4 } }
"""
            )
        self.assertIn("must be a string or list of strings", str(ctx.exception))

    def test_preference_list_with_empty_entry_rejected(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            self._load(
                """
schema_version = 1
name = "x"
main = "x.main"

[models]
klein = { ref = "paul/klein-4b", attributes = { dtype = ["bf16", ""] } }
"""
            )
        self.assertIn("non-empty", str(ctx.exception))

    def test_preference_list_empty_rejected(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            self._load(
                """
schema_version = 1
name = "x"
main = "x.main"

[models]
klein = { ref = "paul/klein-4b", attributes = { dtype = [] } }
"""
            )
        self.assertIn("cannot be empty", str(ctx.exception))

    def test_empty_attribute_value_rejected(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            self._load(
                """
schema_version = 1
name = "x"
main = "x.main"

[models]
klein = { ref = "paul/klein-4b", attributes = { dtype = "" } }
"""
            )
        self.assertIn("non-empty", str(ctx.exception))

    def test_attributes_on_function_keyspace(self) -> None:
        cfg, _ = self._load(
            """
schema_version = 1
name = "x"
main = "x.main"

[models.generate]
small = { ref = "paul/model-small", attributes = { dtype = "bf16" } }
large = { ref = "paul/model-large", attributes = { dtype = "fp16", file_layout = "diffusers" } }
"""
        )
        fn = cfg.function_models["generate"]
        self.assertEqual(fn["small"].attributes_as_dict(), {"dtype": ["bf16"]})
        self.assertEqual(
            fn["large"].attributes_as_dict(),
            {"dtype": ["fp16"], "file_layout": ["diffusers"]},
        )

    def test_unknown_attribute_key_accepted(self) -> None:
        """Forward-compat with future #229 axes — unknown keys pass through."""
        cfg, _ = self._load(
            """
schema_version = 1
name = "x"
main = "x.main"

[models]
klein = { ref = "paul/klein-4b", attributes = { dtype = "bf16", future_axis = "hypothetical" } }
"""
        )
        self.assertEqual(
            cfg.models["klein"].attributes_as_dict()["future_axis"],
            ["hypothetical"],
        )


if __name__ == "__main__":
    unittest.main()
