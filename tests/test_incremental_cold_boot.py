"""Incremental cold-boot regression tests (gen-worker #21).

Companion to test_incremental_function_readiness.py — verifies the
worker-side half of incremental cold-boot:

  Part 1: orchestrator-supplied order of `supported_repo_refs` is
          preserved through the EndpointConfig handler (no `sorted()`
          re-ordering would defeat per-model demand priority).
  Part 3: after each successful download, the DiffusersModelManager
          fires the worker's eager-load hook. The hook attempts
          `load_model_into_vram` while VRAM has headroom and stops as
          soon as VRAM is full or a load OOMs.
"""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock

import pytest

from gen_worker.pipeline.model_manager import DiffusersModelManager


# --------------------------------------------------------------------------- #
# Part 3: eager-load hook drives load_model_into_vram during prefetch        #
# --------------------------------------------------------------------------- #


def test_prefetch_invokes_eager_load_hook_per_successful_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """For every ref that finishes downloading, the manager fires the
    `_on_model_downloaded_try_eager_load` hook with the (ref, path).
    Returning True keeps the loop trying; returning False stops eager-
    load attempts on subsequent refs but does NOT abort the download
    loop.
    """
    monkeypatch.setattr(
        "gen_worker.models.cache_paths.tensorhub_cas_dir",
        lambda: tmp_path,
    )
    snapshot = tmp_path / "snap"
    snapshot.mkdir()

    download_calls: List[str] = []

    def fake_download(ref: str, dest_dir: str) -> str:
        download_calls.append(ref)
        return str(snapshot)

    eager_calls: List[Tuple[str, Path]] = []

    def eager(ref: str, local_path: Path) -> bool:
        eager_calls.append((ref, local_path))
        # Allow eager-load for the first call, then stop.
        return len(eager_calls) < 2

    manager = DiffusersModelManager.__new__(DiffusersModelManager)
    manager._loader = MagicMock()
    manager._downloader = None
    manager._on_model_downloaded = lambda *_args: None
    manager._on_model_download_failed = None
    manager._on_model_downloaded_try_eager_load = eager

    downloader = MagicMock()
    downloader.download = fake_download

    refs = ["acme/a:latest", "acme/b:latest", "acme/c:latest"]
    asyncio.run(manager.process_supported_models_config(refs, downloader))

    # All three refs still downloaded.
    assert download_calls == refs
    # Eager-load was attempted on the first two; the second call returned
    # False so the third ref doesn't get an eager-load attempt.
    assert [r for r, _ in eager_calls] == ["acme/a:latest", "acme/b:latest"]


def test_prefetch_swallows_eager_load_exception_and_continues_downloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If the eager-load hook raises (e.g. CUDA OOM bubbled up), the
    prefetch loop must:
      - NOT abort: remaining refs still get downloaded.
      - Stop eager-loading subsequent refs (one OOM is enough signal).
    """
    monkeypatch.setattr(
        "gen_worker.models.cache_paths.tensorhub_cas_dir",
        lambda: tmp_path,
    )
    snapshot = tmp_path / "snap"
    snapshot.mkdir()

    download_calls: List[str] = []

    def fake_download(ref: str, dest_dir: str) -> str:
        download_calls.append(ref)
        return str(snapshot)

    eager_calls: List[str] = []

    def eager(ref: str, local_path: Path) -> bool:
        eager_calls.append(ref)
        raise RuntimeError("simulated CUDA OOM")

    manager = DiffusersModelManager.__new__(DiffusersModelManager)
    manager._loader = MagicMock()
    manager._downloader = None
    manager._on_model_downloaded = lambda *_args: None
    manager._on_model_download_failed = None
    manager._on_model_downloaded_try_eager_load = eager

    downloader = MagicMock()
    downloader.download = fake_download

    refs = ["acme/a:latest", "acme/b:latest", "acme/c:latest"]
    asyncio.run(manager.process_supported_models_config(refs, downloader))

    # All three refs downloaded.
    assert download_calls == refs
    # Eager-load was attempted exactly once; the exception turned off
    # further attempts.
    assert eager_calls == ["acme/a:latest"]


def test_prefetch_walks_refs_in_input_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The manager must download refs in the order it received them.
    Part 1 of gen-worker #21 drops the `sorted(supported_set)` upstream
    so this test pins the downstream contract: the manager itself
    must NOT reorder.
    """
    monkeypatch.setattr(
        "gen_worker.models.cache_paths.tensorhub_cas_dir",
        lambda: tmp_path,
    )
    snapshot = tmp_path / "snap"
    snapshot.mkdir()

    download_calls: List[str] = []

    def fake_download(ref: str, dest_dir: str) -> str:
        download_calls.append(ref)
        return str(snapshot)

    manager = DiffusersModelManager.__new__(DiffusersModelManager)
    manager._loader = MagicMock()
    manager._downloader = None
    manager._on_model_downloaded = None
    manager._on_model_download_failed = None
    manager._on_model_downloaded_try_eager_load = None

    downloader = MagicMock()
    downloader.download = fake_download

    # Deliberately non-alphabetical input.
    refs = ["zz/last:latest", "aa/first:latest", "mm/middle:latest"]
    asyncio.run(manager.process_supported_models_config(refs, downloader))

    assert download_calls == refs, (
        f"download order must match input order; got {download_calls}"
    )


# --------------------------------------------------------------------------- #
# Part 3: ModelCache exposes public VRAM accessors                            #
# --------------------------------------------------------------------------- #


def test_model_cache_exposes_vram_budget_accessors() -> None:
    """The eager-load gate reads `cache.vram_free_gb` and
    `cache.max_vram_gb` — both must be exposed as public properties.
    """
    from gen_worker.models.cache import ModelCache

    cache = ModelCache(max_vram_gb=16.0, vram_safety_margin_gb=2.0)
    # Constructor overrides max_vram_gb when explicit; safety margin is
    # only applied during auto-detection.
    assert cache.max_vram_gb == pytest.approx(16.0)
    assert cache.vram_used_gb == pytest.approx(0.0)
    assert cache.vram_free_gb == pytest.approx(16.0)

    cache.mark_loaded_to_vram("acme/a", None, 6.0)
    assert cache.vram_used_gb == pytest.approx(6.0)
    assert cache.vram_free_gb == pytest.approx(10.0)
