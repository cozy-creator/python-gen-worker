"""Smoke tests for #324 acceleration helpers.

These tests must pass WITHOUT GPU and WITHOUT third-party acceleration
deps installed (para-attn, DeepCache, teacache, nvidia-modelopt,
bitsandbytes, xfuser). The contract is: classes/functions import and
construct without side effects; only calling ``.apply()`` (or invoking
the function) probes for the third-party lib and raises a clear error
if it's missing.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest import mock


class TestCacheImports(unittest.TestCase):
    def test_fbcache_construct(self) -> None:
        from gen_worker.cache import FBCache

        fb = FBCache()
        self.assertEqual(fb.threshold, 0.12)
        self.assertFalse(fb.breaks_cross_request_batching)

        fb2 = FBCache(threshold=0.2)
        self.assertEqual(fb2.threshold, 0.2)

    def test_deepcache_construct(self) -> None:
        from gen_worker.cache import DeepCache

        dc = DeepCache()
        self.assertEqual(dc.cache_interval, 3)
        self.assertFalse(dc.breaks_cross_request_batching)

    def test_teacache_construct(self) -> None:
        from gen_worker.cache import TeaCache

        tc = TeaCache()
        self.assertEqual(tc.threshold, 0.6)
        self.assertTrue(tc.breaks_cross_request_batching)

    def test_apply_raises_when_third_party_missing(self) -> None:
        """If para-attn isn't installed, apply() raises CacheUnavailableError."""
        from gen_worker.cache import CacheUnavailableError, FBCache

        fb = FBCache()
        # Block the import path even if the lib happens to be installed
        # on the dev box.
        with mock.patch.dict(sys.modules, {"para_attn": None}):
            with self.assertRaises(CacheUnavailableError):
                fb.apply(object())


class TestCompileImports(unittest.TestCase):
    def test_torch_compile_function_exists(self) -> None:
        from gen_worker.compile_helpers import torch_compile

        self.assertTrue(callable(torch_compile))

    def test_nexfort_compile_raises_when_missing(self) -> None:
        from gen_worker.compile_helpers import (
            CompileUnavailableError,
            nexfort_compile,
        )

        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch not available; nexfort_compile needs it to probe")

        with mock.patch.dict(sys.modules, {"nexfort": None}):
            with self.assertRaises(CompileUnavailableError):
                nexfort_compile(object())

    def test_tensorrt_noop_on_old_hardware(self) -> None:
        """tensorrt() with passthrough fallback returns module unchanged."""
        from gen_worker.compile_helpers import tensorrt

        sentinel = object()
        with mock.patch(
            "gen_worker.compile_helpers._detect_sm_major", return_value=8
        ):
            out = tensorrt(sentinel, fallback="passthrough")
        self.assertIs(out, sentinel)

    def test_compile_exposed_on_top_level_package(self) -> None:
        import gen_worker

        self.assertTrue(hasattr(gen_worker, "compile"))
        self.assertTrue(hasattr(gen_worker.compile, "torch_compile"))
        self.assertTrue(hasattr(gen_worker.compile, "nexfort_compile"))
        self.assertTrue(hasattr(gen_worker.compile, "tensorrt"))


class TestQuantImports(unittest.TestCase):
    def test_quant_module_imports(self) -> None:
        import gen_worker.quant

        self.assertTrue(hasattr(gen_worker.quant, "nvfp4"))
        self.assertTrue(hasattr(gen_worker.quant, "fp8"))
        self.assertTrue(hasattr(gen_worker.quant, "int8"))

    def test_nvfp4_passthrough_on_old_hardware(self) -> None:
        from gen_worker.quant import nvfp4

        sentinel = object()
        with mock.patch(
            "gen_worker.quant._detect_sm_major", return_value=8
        ):
            out = nvfp4(sentinel, fallback="passthrough")
        self.assertIs(out, sentinel)

    def test_fp8_passthrough_on_old_hardware(self) -> None:
        from gen_worker.quant import fp8

        sentinel = object()
        with mock.patch(
            "gen_worker.quant._detect_sm_major", return_value=8
        ):
            out = fp8(sentinel, fallback="passthrough")
        self.assertIs(out, sentinel)

    def test_int8_raises_when_bnb_missing(self) -> None:
        from gen_worker.quant import QuantUnavailableError, int8

        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch not available")

        with mock.patch.dict(sys.modules, {"bitsandbytes": None}):
            with self.assertRaises(QuantUnavailableError):
                int8(object())

    def test_quant_cache_root_uses_env_override(self) -> None:
        from gen_worker.quant import _quant_cache_root

        with mock.patch.dict(os.environ, {"GEN_WORKER_QUANT_CACHE_DIR": "/tmp/_gw_quant_test"}):
            root = _quant_cache_root()
        self.assertEqual(root, Path("/tmp/_gw_quant_test"))

    def test_quant_cache_root_siblings_inductor(self) -> None:
        from gen_worker.quant import _quant_cache_root

        with mock.patch.dict(
            os.environ,
            {"TORCHINDUCTOR_CACHE_DIR": "/tmp/_gw_inductor/cache"},
            clear=False,
        ):
            os.environ.pop("GEN_WORKER_QUANT_CACHE_DIR", None)
            root = _quant_cache_root()
        self.assertEqual(root, Path("/tmp/_gw_inductor/gen_worker_quant_cache"))


class TestParallelism(unittest.TestCase):
    def test_module_imports(self) -> None:
        import gen_worker.parallelism

        self.assertTrue(hasattr(gen_worker.parallelism, "xdit_sequence_parallel"))
        self.assertTrue(hasattr(gen_worker.parallelism, "SequenceParallel"))

    def test_single_gpu_noop(self) -> None:
        from gen_worker.parallelism import xdit_sequence_parallel

        sentinel = object()
        out = xdit_sequence_parallel(sentinel, gpus=1)
        self.assertIs(out, sentinel)

    def test_invalid_gpus_raises(self) -> None:
        from gen_worker.parallelism import xdit_sequence_parallel

        with self.assertRaises(ValueError):
            xdit_sequence_parallel(object(), gpus=0)

    def test_passthrough_when_insufficient_gpus(self) -> None:
        from gen_worker.parallelism import xdit_sequence_parallel

        sentinel = object()
        with mock.patch(
            "gen_worker.parallelism._detect_gpu_count", return_value=1
        ):
            out = xdit_sequence_parallel(sentinel, gpus=4, fallback="passthrough")
        self.assertIs(out, sentinel)

    def test_sequence_parallel_class(self) -> None:
        from gen_worker.parallelism import SequenceParallel

        sp = SequenceParallel(gpus=1)
        sentinel = object()
        self.assertIs(sp.apply(sentinel), sentinel)


if __name__ == "__main__":
    unittest.main()
