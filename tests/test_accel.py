"""Tests for :mod:`gen_worker.accel` (#324).

These tests must pass:

* without a GPU,
* without torch installed,
* without any of the optional acceleration deps installed
  (``para_attn``, ``nvidia-modelopt``).

The contract is: the module imports cleanly and every helper either
returns a sensible no-op or raises a typed ImportError pointing at the
canonical install command.
"""

from __future__ import annotations

import logging
import sys
import unittest
from unittest import mock


def _reset_capability_cache() -> None:
    """Clear the gpu_capability() module-level cache between tests."""
    import gen_worker.accel as accel_mod

    accel_mod._CAPABILITY_CACHE = None


class TestModuleImport(unittest.TestCase):
    def test_module_imports_cleanly(self) -> None:
        """Importing the package must not require torch or any optional dep."""
        import gen_worker.accel as accel

        # Public surface present.
        for name in (
            "gpu_capability",
            "compile_diffusion",
            "apply_fbcache",
            "apply_para_attn",
            "apply_nvfp4",
            "GpuCapabilityReport",
            "Arch",
        ):
            self.assertTrue(
                hasattr(accel, name),
                f"gen_worker.accel.{name} missing from public surface",
            )

    def test_exposed_on_top_level_package(self) -> None:
        import gen_worker

        self.assertTrue(hasattr(gen_worker, "accel"))
        self.assertTrue(hasattr(gen_worker.accel, "gpu_capability"))


class TestGpuCapability(unittest.TestCase):
    def setUp(self) -> None:
        _reset_capability_cache()

    def tearDown(self) -> None:
        _reset_capability_cache()

    def test_returns_sensible_report_no_gpu(self) -> None:
        """On a host without CUDA the report has arch='none' and zeroed fields."""
        from gen_worker.accel import GpuCapabilityReport, gpu_capability

        # Force the no-torch path even if torch happens to be installed in
        # the dev venv.
        with mock.patch.dict(sys.modules, {"torch": None}):
            _reset_capability_cache()
            caps = gpu_capability(refresh=True)
        self.assertIsInstance(caps, GpuCapabilityReport)
        self.assertEqual(caps.arch, "none")
        self.assertEqual(caps.compute_capability, "")
        self.assertEqual(caps.vram_gb_total, 0.0)
        self.assertEqual(caps.gpu_count, 0)
        self.assertFalse(caps.has_fp8)
        self.assertFalse(caps.has_nvfp4)

    def test_never_crashes_even_when_probing_blows_up(self) -> None:
        """If torch is broken, the report degrades gracefully — never raises."""
        from gen_worker.accel import gpu_capability

        broken = mock.MagicMock()
        broken.__version__ = "9.9.9"
        broken.cuda.is_available.side_effect = RuntimeError("driver broken")

        with mock.patch.dict(sys.modules, {"torch": broken}):
            _reset_capability_cache()
            caps = gpu_capability(refresh=True)
        # is_available() failing -> arch="none".
        self.assertEqual(caps.arch, "none")
        self.assertEqual(caps.torch_version, "9.9.9")

    def test_classify_arch_known_families(self) -> None:
        """Each known SM family maps to the expected arch label."""
        from gen_worker.accel import _classify_arch

        # Blackwell: SM 10.0 (datacenter) and SM 12.0 (consumer).
        self.assertEqual(_classify_arch(10, 0), "blackwell")
        self.assertEqual(_classify_arch(12, 0), "blackwell")
        # Hopper: SM 9.0.
        self.assertEqual(_classify_arch(9, 0), "hopper")
        # Lovelace: SM 8.9.
        self.assertEqual(_classify_arch(8, 9), "lovelace")
        # Ampere: SM 8.0 / 8.6.
        self.assertEqual(_classify_arch(8, 0), "ampere")
        self.assertEqual(_classify_arch(8, 6), "ampere")
        # Turing: SM 7.5.
        self.assertEqual(_classify_arch(7, 5), "turing")
        # Unknown SM falls through to "unknown".
        self.assertEqual(_classify_arch(6, 0), "unknown")

    def test_synthetic_blackwell_host(self) -> None:
        """Simulate a Blackwell host and verify has_nvfp4 / has_fp8 / arch."""
        from gen_worker.accel import gpu_capability

        torch_stub = mock.MagicMock()
        torch_stub.__version__ = "2.6.0+cu126"
        torch_stub.cuda.is_available.return_value = True
        torch_stub.cuda.device_count.return_value = 1
        torch_stub.cuda.get_device_capability.return_value = (10, 0)
        torch_stub.cuda.get_device_name.return_value = "NVIDIA B200"
        props = mock.MagicMock()
        props.total_memory = 180 * 1024**3
        torch_stub.cuda.get_device_properties.return_value = props

        with mock.patch.dict(sys.modules, {"torch": torch_stub}):
            _reset_capability_cache()
            caps = gpu_capability(refresh=True)

        self.assertEqual(caps.arch, "blackwell")
        self.assertEqual(caps.compute_capability, "10.0")
        self.assertTrue(caps.has_fp8)
        self.assertTrue(caps.has_nvfp4)
        self.assertEqual(caps.device_name, "NVIDIA B200")
        self.assertAlmostEqual(caps.vram_gb_total, 180.0, places=1)
        self.assertEqual(caps.gpu_count, 1)

    def test_synthetic_hopper_host(self) -> None:
        """Hopper has FP8 but not NVFP4."""
        from gen_worker.accel import gpu_capability

        torch_stub = mock.MagicMock()
        torch_stub.__version__ = "2.6.0+cu124"
        torch_stub.cuda.is_available.return_value = True
        torch_stub.cuda.device_count.return_value = 8
        torch_stub.cuda.get_device_capability.return_value = (9, 0)
        torch_stub.cuda.get_device_name.return_value = "NVIDIA H100 80GB HBM3"
        props = mock.MagicMock()
        props.total_memory = 80 * 1024**3
        torch_stub.cuda.get_device_properties.return_value = props

        with mock.patch.dict(sys.modules, {"torch": torch_stub}):
            _reset_capability_cache()
            caps = gpu_capability(refresh=True)

        self.assertEqual(caps.arch, "hopper")
        self.assertTrue(caps.has_fp8)
        self.assertFalse(caps.has_nvfp4)
        self.assertEqual(caps.gpu_count, 8)

    def test_synthetic_ampere_host(self) -> None:
        """Ampere has neither FP8 nor NVFP4."""
        from gen_worker.accel import gpu_capability

        torch_stub = mock.MagicMock()
        torch_stub.__version__ = "2.6.0+cu121"
        torch_stub.cuda.is_available.return_value = True
        torch_stub.cuda.device_count.return_value = 1
        torch_stub.cuda.get_device_capability.return_value = (8, 0)
        torch_stub.cuda.get_device_name.return_value = "NVIDIA A100 80GB PCIe"
        props = mock.MagicMock()
        props.total_memory = 80 * 1024**3
        torch_stub.cuda.get_device_properties.return_value = props

        with mock.patch.dict(sys.modules, {"torch": torch_stub}):
            _reset_capability_cache()
            caps = gpu_capability(refresh=True)

        self.assertEqual(caps.arch, "ampere")
        self.assertFalse(caps.has_fp8)
        self.assertFalse(caps.has_nvfp4)

    def test_result_is_cached(self) -> None:
        """gpu_capability() caches; repeat calls return the same object."""
        from gen_worker.accel import gpu_capability

        _reset_capability_cache()
        a = gpu_capability()
        b = gpu_capability()
        self.assertIs(a, b)

    def test_report_is_frozen(self) -> None:
        """GpuCapabilityReport is a frozen msgspec.Struct."""
        from gen_worker.accel import gpu_capability

        caps = gpu_capability()
        with self.assertRaises((AttributeError, TypeError)):
            caps.arch = "blackwell"  # type: ignore[misc]


class TestCompileDiffusion(unittest.TestCase):
    def test_noop_when_torch_missing(self) -> None:
        """Without torch installed, compile_diffusion returns model unchanged."""
        from gen_worker.accel import compile_diffusion

        sentinel = object()
        with mock.patch.dict(sys.modules, {"torch": None}):
            out = compile_diffusion(sentinel)
        self.assertIs(out, sentinel)

    def test_noop_when_no_cuda(self) -> None:
        """With torch but no CUDA device, compile_diffusion is a no-op."""
        from gen_worker.accel import compile_diffusion

        torch_stub = mock.MagicMock()
        torch_stub.__version__ = "2.6.0+cu126"
        torch_stub.cuda.is_available.return_value = False
        torch_stub.compile.side_effect = AssertionError(
            "torch.compile must not be called without CUDA"
        )

        sentinel = object()
        with mock.patch.dict(sys.modules, {"torch": torch_stub}):
            out = compile_diffusion(sentinel)
        self.assertIs(out, sentinel)

    def test_noop_when_torch_too_old(self) -> None:
        """Torch < 2.5 falls back to passthrough with a stderr warning."""
        from gen_worker.accel import compile_diffusion

        torch_stub = mock.MagicMock()
        torch_stub.__version__ = "2.4.1+cu121"
        torch_stub.cuda.is_available.return_value = True
        torch_stub.compile.side_effect = AssertionError(
            "torch.compile must not be called on torch < 2.5"
        )

        sentinel = object()
        with mock.patch.dict(sys.modules, {"torch": torch_stub}):
            out = compile_diffusion(sentinel)
        self.assertIs(out, sentinel)

    def test_invokes_torch_compile_when_supported(self) -> None:
        """On torch >= 2.5 with CUDA, the helper calls torch.compile with kwargs."""
        from gen_worker.accel import compile_diffusion

        torch_stub = mock.MagicMock()
        torch_stub.__version__ = "2.6.0+cu126"
        torch_stub.cuda.is_available.return_value = True
        torch_stub.compile.return_value = "compiled-model-sentinel"

        with mock.patch.dict(sys.modules, {"torch": torch_stub}):
            out = compile_diffusion(
                "raw-model",
                mode="reduce-overhead",
                fullgraph=False,
                dynamic=False,
            )

        self.assertEqual(out, "compiled-model-sentinel")
        torch_stub.compile.assert_called_once_with(
            "raw-model",
            mode="reduce-overhead",
            fullgraph=False,
            dynamic=False,
        )


class TestApplyFbcache(unittest.TestCase):
    def test_raises_clear_importerror_when_missing(self) -> None:
        """Without para_attn installed, apply_fbcache raises ImportError with a hint."""
        from gen_worker.accel import apply_fbcache

        # Block the import path even if the lib happens to be installed.
        with mock.patch.dict(
            sys.modules,
            {
                "para_attn": None,
                "para_attn.first_block_cache": None,
                "para_attn.first_block_cache.diffusers_adapters": None,
            },
        ):
            with self.assertRaises(ImportError) as ctx:
                apply_fbcache(object())
        self.assertIn("para-attention", str(ctx.exception))

    def test_calls_para_attn_when_available(self) -> None:
        """Happy path: the lazy import finds the adapter and we call it correctly."""
        from gen_worker.accel import apply_fbcache

        recorded: dict[str, object] = {}

        def fake_apply(pipe, *, residual_diff_threshold):  # type: ignore[no-untyped-def]
            recorded["pipe"] = pipe
            recorded["threshold"] = residual_diff_threshold

        adapter_mod = mock.MagicMock()
        adapter_mod.apply_cache_on_pipe = fake_apply

        with mock.patch.dict(
            sys.modules,
            {
                "para_attn": mock.MagicMock(),
                "para_attn.first_block_cache": mock.MagicMock(),
                "para_attn.first_block_cache.diffusers_adapters": adapter_mod,
            },
        ):
            sentinel = object()
            out = apply_fbcache(sentinel, residual_diff_threshold=0.15)

        self.assertIs(out, sentinel)
        self.assertIs(recorded["pipe"], sentinel)
        self.assertEqual(recorded["threshold"], 0.15)


class TestApplyParaAttn(unittest.TestCase):
    def test_raises_importerror_when_missing(self) -> None:
        """Without para_attn installed, apply_para_attn raises ImportError."""
        from gen_worker.accel import apply_para_attn

        with mock.patch.dict(sys.modules, {"para_attn": None}):
            with self.assertRaises(ImportError) as ctx:
                apply_para_attn(object())
        self.assertIn("para-attention", str(ctx.exception))


class TestApplyNvfp4(unittest.TestCase):
    def setUp(self) -> None:
        _reset_capability_cache()

    def tearDown(self) -> None:
        _reset_capability_cache()

    def test_noop_with_warning_on_non_blackwell(self) -> None:
        """When arch != blackwell, apply_nvfp4 logs a warning and returns model."""
        import gen_worker.accel as accel_mod
        from gen_worker.accel import GpuCapabilityReport, apply_nvfp4

        ampere_report = GpuCapabilityReport(
            arch="ampere",
            compute_capability="8.0",
            device_name="NVIDIA A100 80GB PCIe",
            vram_gb_total=80.0,
            gpu_count=1,
            has_fp8=False,
            has_nvfp4=False,
            torch_version="2.6.0+cu126",
        )
        accel_mod._CAPABILITY_CACHE = ampere_report

        sentinel = object()
        with self.assertLogs("gen_worker.accel", level=logging.WARNING) as cm:
            out = apply_nvfp4(sentinel)
        self.assertIs(out, sentinel)
        # The warning surface mentions NVFP4 and arch.
        joined = "\n".join(cm.output)
        self.assertIn("NVFP4", joined)
        self.assertIn("ampere", joined)

    def test_noop_with_warning_on_hopper(self) -> None:
        """Hopper has FP8 but not NVFP4; apply_nvfp4 still no-ops with a warning."""
        import gen_worker.accel as accel_mod
        from gen_worker.accel import GpuCapabilityReport, apply_nvfp4

        accel_mod._CAPABILITY_CACHE = GpuCapabilityReport(
            arch="hopper",
            compute_capability="9.0",
            device_name="NVIDIA H100",
            vram_gb_total=80.0,
            gpu_count=1,
            has_fp8=True,
            has_nvfp4=False,
            torch_version="2.6.0",
        )

        sentinel = object()
        with self.assertLogs("gen_worker.accel", level=logging.WARNING):
            out = apply_nvfp4(sentinel)
        self.assertIs(out, sentinel)

    def test_noop_with_warning_when_no_gpu(self) -> None:
        """No GPU at all -> arch='none' -> no-op + warn. No ImportError."""
        import gen_worker.accel as accel_mod
        from gen_worker.accel import apply_nvfp4

        # Force a "no GPU" report.
        with mock.patch.dict(sys.modules, {"torch": None}):
            accel_mod._CAPABILITY_CACHE = None
            sentinel = object()
            with self.assertLogs("gen_worker.accel", level=logging.WARNING):
                out = apply_nvfp4(sentinel)
        self.assertIs(out, sentinel)

    def test_raises_importerror_on_blackwell_when_modelopt_missing(self) -> None:
        """On a Blackwell host without modelopt, surface a clear ImportError."""
        import gen_worker.accel as accel_mod
        from gen_worker.accel import GpuCapabilityReport, apply_nvfp4

        accel_mod._CAPABILITY_CACHE = GpuCapabilityReport(
            arch="blackwell",
            compute_capability="10.0",
            device_name="NVIDIA B200",
            vram_gb_total=180.0,
            gpu_count=1,
            has_fp8=True,
            has_nvfp4=True,
            torch_version="2.6.0+cu126",
        )

        with mock.patch.dict(
            sys.modules,
            {
                "modelopt": None,
                "modelopt.torch": None,
                "modelopt.torch.quantization": None,
            },
        ):
            with self.assertRaises(ImportError) as ctx:
                apply_nvfp4(object())
        self.assertIn("nvidia-modelopt", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
