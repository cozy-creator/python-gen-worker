"""Issue #20 fix 1: propagate the underlying load exception into
LoadModelResult.error_message.

Today every load failure surfaces to the orchestrator as the opaque
``MMM.load_model_into_vram failed for X`` string, hiding the actual
type/message/traceback. Fix 1 stashes `_last_load_error` +
`_last_load_traceback` on the manager and threads them through worker.py's
LoadModelResult construction.

These tests verify:
  - DiffusersModelManager.load_model_into_vram populates the fields on
    failure and clears them at the top of every attempt.
  - Worker.py's failure-message path reads those fields via getattr() and
    incorporates them into the outbound error_message, while
    third-party managers (no fields) keep the legacy opaque message.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

import pytest

from gen_worker.pipeline.model_manager import DiffusersModelManager


# ---------------------------------------------------------------------------
# Manager-side: error capture on the manager instance.
# ---------------------------------------------------------------------------


def test_load_failure_captures_exception_type_and_message() -> None:
    """A specific exception raised inside _loader.load() should land on
    _last_load_error as 'ExceptionType: message'."""

    mgr = DiffusersModelManager()

    async def _raise(*_a: Any, **_kw: Any) -> Any:
        raise RuntimeError("CUDA out of memory")

    mgr._loader.get = MagicMock(return_value=None)  # type: ignore[method-assign]
    mgr._loader.load = _raise  # type: ignore[assignment]

    ok = asyncio.run(mgr.load_model_into_vram("acme/m"))
    assert ok is False
    assert mgr._last_load_error == "RuntimeError: CUDA out of memory"
    assert mgr._last_load_traceback is not None
    # Standard traceback format includes 'Traceback (most recent call last)'.
    assert "Traceback" in mgr._last_load_traceback
    # And the exception's repr.
    assert "RuntimeError" in mgr._last_load_traceback


def test_load_success_clears_previous_error_state() -> None:
    """A successful load after a failed one must reset the fields so a
    later success isn't reported with stale failure detail."""

    mgr = DiffusersModelManager()

    # First, simulate a failure to populate the fields.
    async def _raise(*_a: Any, **_kw: Any) -> Any:
        raise RuntimeError("first failure")

    mgr._loader.get = MagicMock(return_value=None)  # type: ignore[method-assign]
    mgr._loader.load = _raise  # type: ignore[assignment]
    assert asyncio.run(mgr.load_model_into_vram("m1")) is False
    assert mgr._last_load_error == "RuntimeError: first failure"

    # Then a success: get() returns a truthy pipeline so the load
    # short-circuits.
    mgr._loader.get = MagicMock(return_value=object())  # type: ignore[method-assign]
    assert asyncio.run(mgr.load_model_into_vram("m1")) is True
    assert mgr._last_load_error is None
    assert mgr._last_load_traceback is None


def test_load_model_treats_root_safetensors_repo_as_materialized(tmp_path) -> None:
    """Root-level safetensors HF repos can be component repos, not pipelines.

    They should satisfy LoadModelCommand after download instead of failing the
    generic Diffusers pipeline loader on missing model_index.json.
    """

    mgr = DiffusersModelManager()
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "flux-2-klein-base-4b-fp8.safetensors").write_bytes(b"stub")

    class _Downloader:
        def download(self, model_id: str, cache_dir: str) -> str:
            assert model_id == "black-forest-labs/FLUX.2-klein-base-4b-fp8"
            assert cache_dir
            return str(snapshot)

    mgr._downloader = _Downloader()
    mgr._loader.get = MagicMock(return_value=None)  # type: ignore[method-assign]
    mgr._loader.load = MagicMock()  # type: ignore[assignment]

    model_id = "black-forest-labs/FLUX.2-klein-base-4b-fp8"
    assert asyncio.run(mgr.load_model_into_vram(model_id)) is True
    mgr._loader.load.assert_not_called()
    assert model_id in mgr.get_vram_loaded_models()
    assert mgr._last_load_error is None
    assert mgr._last_load_traceback is None


def test_load_failure_resets_then_repopulates() -> None:
    """Two failures in a row should each see fresh state at the start."""

    mgr = DiffusersModelManager()

    async def _raise_v1(*_a: Any, **_kw: Any) -> Any:
        raise ValueError("v1")

    async def _raise_v2(*_a: Any, **_kw: Any) -> Any:
        raise TypeError("v2")

    mgr._loader.get = MagicMock(return_value=None)  # type: ignore[method-assign]
    mgr._loader.load = _raise_v1  # type: ignore[assignment]
    asyncio.run(mgr.load_model_into_vram("m"))
    assert mgr._last_load_error == "ValueError: v1"

    mgr._loader.load = _raise_v2  # type: ignore[assignment]
    asyncio.run(mgr.load_model_into_vram("m"))
    assert mgr._last_load_error == "TypeError: v2"


# ---------------------------------------------------------------------------
# Worker-side: error_message construction reads manager state.
# ---------------------------------------------------------------------------


def _simulate_worker_error_msg_path(
    model_manager: Any, model_id: str, success: bool
) -> str:
    """Mirror the post-load error_message construction in worker.py around
    the failure branch of `_handle_load_model_cmd`. Kept as a local helper
    so the test doesn't have to spin up a full Worker (the surrounding
    code path requires gRPC stubs we don't care about here)."""

    if success:
        return ""
    last_err = getattr(model_manager, "_last_load_error", None)
    last_tb = getattr(model_manager, "_last_load_traceback", None)
    if last_err:
        parts = [f"MMM.load_model_into_vram failed for '{model_id}': {last_err}"]
        if last_tb:
            tb_trunc = last_tb if len(last_tb) <= 2000 else last_tb[-2000:]
            parts.append(f"traceback:\n{tb_trunc}")
        return "\n".join(parts)
    return f"MMM.load_model_into_vram failed for '{model_id}'."


def test_worker_error_msg_includes_underlying_exception() -> None:
    """When the manager populated _last_load_error, the worker's
    error_message must include the type + message + traceback."""

    mgr = DiffusersModelManager()

    async def _raise(*_a: Any, **_kw: Any) -> Any:
        raise RuntimeError("CUDA out of memory")

    mgr._loader.get = MagicMock(return_value=None)  # type: ignore[method-assign]
    mgr._loader.load = _raise  # type: ignore[assignment]
    asyncio.run(mgr.load_model_into_vram("acme/m"))

    msg = _simulate_worker_error_msg_path(mgr, "acme/m", success=False)
    assert "MMM.load_model_into_vram failed for 'acme/m'" in msg
    assert "RuntimeError: CUDA out of memory" in msg
    assert "traceback:" in msg
    assert "Traceback" in msg


def test_worker_error_msg_falls_back_for_third_party_manager() -> None:
    """A manager that doesn't carry _last_load_* fields must still produce
    a sensible (legacy) error_message."""

    class _ThirdPartyManager:
        pass  # no _last_load_* attrs

    msg = _simulate_worker_error_msg_path(_ThirdPartyManager(), "acme/m", success=False)
    assert msg == "MMM.load_model_into_vram failed for 'acme/m'."


def test_worker_error_msg_truncates_long_traceback() -> None:
    """A pathologically long traceback must be truncated to keep the gRPC
    error_message bounded."""

    class _Mgr:
        _last_load_error = "RuntimeError: x"
        _last_load_traceback = "A" * 5000

    msg = _simulate_worker_error_msg_path(_Mgr(), "acme/m", success=False)
    # 2000 chars of traceback + the framing strings = comfortably under
    # 3000. Without truncation it would carry the full 5000.
    assert len(msg) < 3000
    assert msg.count("A") <= 2000
