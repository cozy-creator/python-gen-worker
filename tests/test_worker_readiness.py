"""Regression tests for worker startup-phase gating (gen-worker 0.7.19, Fix 2).

Pre-fix bug:
- Worker.__init__ spawned `ManifestModelInit` and registered with the
  orchestrator before any model bytes had been pulled. The registration
  message advertised every declared function as ready and an empty
  `loading_functions` list, because the model cache hadn't been
  populated with the "downloading" markers yet.
- `_emit_startup_phase("ready")` then fired from the gRPC connect ACK,
  which made the orchestrator flip `AvailableForRequests=true` and
  dispatch a request to a cold worker.

Fix (Option A, worker-side):
- Pre-mark every required ref as `downloading` in `__init__` so the
  initial registration's `loading_functions` is populated.
- `DiffusersModelManager.process_supported_models_config` actually
  downloads and fires callbacks to update the cache.
- The connect path emits `models_downloading` rather than `ready`; a
  helper emits `ready` only after every required ref is cached to disk.
- Endpoints with no required refs still get `ready` immediately
  (back-compat for marco-polo / idle test workers).
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

from gen_worker.models.cache import ModelCache
from gen_worker.pipeline.model_manager import DiffusersModelManager
from gen_worker.worker import Worker


# --------------------------------------------------------------------------- #
# Worker stub helpers                                                         #
# --------------------------------------------------------------------------- #


class _StubCache:
    """A spy ModelCache substitute that records downloading + cached refs."""

    def __init__(self) -> None:
        self.downloading: set[str] = set()
        self.cached: Dict[str, Path] = {}
        self._lock = threading.Lock()

    def mark_downloading(self, model_id: str, progress: float = 0.0) -> None:  # noqa: ARG002
        with self._lock:
            self.downloading.add(model_id)

    def mark_cached_to_disk(self, model_id: str, disk_path: Path, size_gb: float = 0.0) -> None:  # noqa: ARG002
        with self._lock:
            self.downloading.discard(model_id)
            self.cached[model_id] = disk_path

    def get_disk_models(self) -> List[str]:
        with self._lock:
            return list(self.cached.keys())

    def get_stats(self) -> Any:
        with self._lock:
            ds = list(self.downloading)
        return type("S", (), {"downloading_models": ds})()

    def unload_model(self, model_id: str) -> None:
        with self._lock:
            self.downloading.discard(model_id)
            self.cached.pop(model_id, None)


def _bare_worker_for_readiness(
    *,
    fixed: Optional[Dict[str, str]] = None,
    per_function: Optional[Dict[str, Dict[str, str]]] = None,
    function_names: Optional[List[str]] = None,
) -> Worker:
    """Build a Worker without running __init__. The Fix-2 paths only touch:
    - _model_cache (cache spy)
    - _release_allowed_model_ids (drives the readiness gate)
    - _startup_required_refs_canonical (Fix 2a output)
    - _ready_phase_emitted + _ready_phase_lock (Fix 2c latch)
    - _fixed_model_id_by_key + _payload_model_id_by_key_by_function (drive _loading_function_names)
    - _request_specs / _training_specs / etc (drive _all_declared_function_names)
    - _send_message + _emit_startup_phase capture
    """
    w = Worker.__new__(Worker)
    w._model_cache = _StubCache()
    w._fixed_model_id_by_key = fixed or {}
    w._payload_model_id_by_key_by_function = per_function or {}
    # Drive _all_declared_function_names; the easiest source is _request_specs.
    fn_names = function_names or sorted(
        set((per_function or {}).keys()) | ({"infer"} if fixed else set())
    )
    w._request_specs = {name: object() for name in fn_names}
    w._training_specs = {}
    w._batched_specs = {}
    w._serial_class_specs = {}
    w._disabled_functions_by_name = {}
    w._worker_local_unavailable_functions_by_name = {}

    # Readiness gate state.
    w._ready_phase_emitted = False
    w._ready_phase_lock = threading.Lock()
    w._startup_required_refs_canonical = set()

    # Capture emitted phases + outgoing messages.
    w._emitted_phases = []  # type: ignore[attr-defined]
    w._sent_messages = []

    def _capture_phase(phase: str, *, status: str = "ok", **_: Any) -> None:
        w._emitted_phases.append((phase, status))  # type: ignore[attr-defined]

    w._emit_startup_phase = _capture_phase  # type: ignore[method-assign]
    w._send_message = lambda msg: w._sent_messages.append(msg)
    w._register_worker_calls: List[bool] = []
    w._register_worker = lambda is_heartbeat=False: w._register_worker_calls.append(  # type: ignore[method-assign]
        bool(is_heartbeat)
    )
    w.scheduler_addr = "stub:0"
    return w


# --------------------------------------------------------------------------- #
# Fix 2a: loading_functions is populated at boot                              #
# --------------------------------------------------------------------------- #


def test_pre_marking_populates_loading_functions_for_every_binding() -> None:
    """Simulate the Fix 2a code path: for each ref in _release_allowed_model_ids,
    mark_downloading; then _loading_function_names() must include every
    function whose binding ref is in the downloading set.
    """
    w = _bare_worker_for_readiness(
        per_function={
            "infer": {"unet": "acme/sd-xl:latest"},
            "stage2": {"refiner": "acme/refiner:latest"},
        },
        function_names=["infer", "stage2"],
    )
    # Mirror the Fix 2a body: pre-mark refs as downloading.
    refs = {"acme/sd-xl:latest", "acme/refiner:latest"}
    for r in refs:
        w._model_cache.mark_downloading(r, progress=0.0)
    w._startup_required_refs_canonical = set(refs)

    loading = w._loading_function_names()
    assert loading == ["infer", "stage2"], (
        f"expected both functions in loading_functions, got {loading}"
    )


# --------------------------------------------------------------------------- #
# Fix 2c: startup phase ready only fires after all refs cached                #
# --------------------------------------------------------------------------- #


def test_ready_phase_does_not_fire_before_all_refs_cached() -> None:
    """With required refs declared, calling _emit_ready_if_all_cached
    before any cache writes lands MUST NOT emit `ready`.
    """
    w = _bare_worker_for_readiness(
        per_function={"infer": {"unet": "acme/sd-xl:latest"}},
        function_names=["infer"],
    )
    w._startup_required_refs_canonical = {"acme/sd-xl:latest"}
    w._model_cache.mark_downloading("acme/sd-xl:latest", progress=0.0)

    w._emit_ready_if_all_cached()

    phases = [p for p, _ in w._emitted_phases]
    assert "ready" not in phases, f"ready emitted prematurely; phases={phases}"


def test_ready_phase_fires_after_last_ref_cached() -> None:
    """Once every required ref lands in get_disk_models(),
    _emit_ready_if_all_cached emits the typed `ready` phase exactly once.
    """
    w = _bare_worker_for_readiness(
        per_function={
            "infer": {"unet": "acme/sd-xl:latest"},
            "stage2": {"refiner": "acme/refiner:latest"},
        },
        function_names=["infer", "stage2"],
    )
    w._startup_required_refs_canonical = {"acme/sd-xl:latest", "acme/refiner:latest"}
    for r in w._startup_required_refs_canonical:
        w._model_cache.mark_downloading(r, progress=0.0)

    # First ref lands — not enough.
    w._model_cache.mark_cached_to_disk("acme/sd-xl:latest", Path("/tmp/a"))
    w._emit_ready_if_all_cached()
    assert "ready" not in [p for p, _ in w._emitted_phases]

    # Second ref lands — ready should fire.
    w._model_cache.mark_cached_to_disk("acme/refiner:latest", Path("/tmp/b"))
    w._emit_ready_if_all_cached()
    phases = [p for p, _ in w._emitted_phases]
    assert phases.count("ready") == 1, f"want exactly one `ready`, got {phases}"

    # Idempotent: subsequent calls must not duplicate the emit.
    w._emit_ready_if_all_cached()
    w._emit_ready_if_all_cached()
    phases = [p for p, _ in w._emitted_phases]
    assert phases.count("ready") == 1, f"ready emitted more than once: {phases}"


def test_ready_phase_fires_immediately_when_no_required_refs() -> None:
    """Back-compat: endpoints with an empty required-ref set (marco-polo,
    idle test workers) still get a `ready` emit on first call so the
    orchestrator doesn't park forever.
    """
    w = _bare_worker_for_readiness(function_names=["marco_polo"])
    w._startup_required_refs_canonical = set()

    w._emit_ready_if_all_cached()

    phases = [p for p, _ in w._emitted_phases]
    assert phases == ["ready"], f"want a single 'ready' emit, got {phases}"


# --------------------------------------------------------------------------- #
# Fix 2b: DiffusersModelManager actually downloads + fires callback           #
# --------------------------------------------------------------------------- #


def test_diffusers_model_manager_downloads_and_fires_callback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """process_supported_models_config must:
    - call downloader.download(ref, cache_dir) for every ref
    - fire the _on_model_downloaded callback with (ref, Path)
    """
    import asyncio

    # Point the manager's cache dir into tmp so we don't touch real CAS.
    monkeypatch.setattr(
        "gen_worker.pipeline.model_manager.tensorhub_cas_dir",
        lambda: tmp_path,
        raising=False,
    )
    # The manager imports tensorhub_cas_dir from gen_worker.models.cache_paths
    # via a function-local import; patch there too.
    monkeypatch.setattr(
        "gen_worker.models.cache_paths.tensorhub_cas_dir",
        lambda: tmp_path,
    )

    downloaded_dir = tmp_path / "snapshot"
    downloaded_dir.mkdir()

    download_calls: List[tuple[str, str]] = []

    def fake_download(ref: str, dest_dir: str) -> str:
        download_calls.append((ref, dest_dir))
        return str(downloaded_dir)

    completed: List[tuple[str, Path]] = []

    def on_completed(ref: str, local_path: Path) -> None:
        completed.append((ref, local_path))

    # Skip the diffusers import path inside __init__ by constructing manually.
    manager = DiffusersModelManager.__new__(DiffusersModelManager)
    manager._loader = MagicMock()
    manager._downloader = None
    manager._on_model_downloaded = on_completed
    manager._on_model_download_failed = None

    downloader = MagicMock()
    downloader.download = fake_download

    asyncio.run(manager.process_supported_models_config(["acme/sd-xl:latest"], downloader))

    assert download_calls == [("acme/sd-xl:latest", str(tmp_path))]
    assert completed == [("acme/sd-xl:latest", downloaded_dir)]


def test_diffusers_model_manager_no_op_without_downloader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the worker doesn't pass a downloader, process_supported_models_config
    must not crash — it logs + returns. The startup gate stays at
    `models_downloading` because no completion callback ever fires (which
    is the correct behaviour: the operator needs to fix the config).
    """
    import asyncio

    monkeypatch.setattr(
        "gen_worker.models.cache_paths.tensorhub_cas_dir",
        lambda: tmp_path,
    )

    completed: List[tuple[str, Path]] = []

    def on_completed(ref: str, local_path: Path) -> None:
        completed.append((ref, local_path))

    manager = DiffusersModelManager.__new__(DiffusersModelManager)
    manager._loader = MagicMock()
    manager._downloader = None
    manager._on_model_downloaded = on_completed
    manager._on_model_download_failed = None

    asyncio.run(manager.process_supported_models_config(["acme/sd-xl:latest"], None))

    assert completed == [], "no callback should fire without a downloader"


# --------------------------------------------------------------------------- #
# End-to-end: worker callback + ready gate                                    #
# --------------------------------------------------------------------------- #


def test_manager_completion_callback_updates_cache_and_gates_ready(tmp_path: Path) -> None:
    """The worker's _handle_manager_model_downloaded callback must:
    - mark the ref cached to disk (so it disappears from loading_functions)
    - emit MODEL_AVAILABILITY_DOWNLOAD_COMPLETED + CACHED + READY signals
    - force an is_heartbeat=True registration
    - flip startup_phase=ready when the LAST required ref completes
    """
    w = _bare_worker_for_readiness(
        per_function={"infer": {"unet": "acme/sd-xl:latest"}},
        function_names=["infer"],
    )
    w._startup_required_refs_canonical = {"acme/sd-xl:latest"}
    w._model_cache.mark_downloading("acme/sd-xl:latest", progress=0.0)

    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()

    w._handle_manager_model_downloaded("acme/sd-xl:latest", snapshot)

    # Cache flipped to cached_to_disk.
    assert w._model_cache.get_disk_models() == ["acme/sd-xl:latest"]
    # Force-heartbeat fired with is_heartbeat=True.
    assert any(hb is True for hb in w._register_worker_calls)
    # All three typed model-ready signals emitted.
    kinds = sorted({m.worker_model_ready.kind for m in w._sent_messages if m.HasField("worker_model_ready")})
    assert kinds == [1, 2, 3], f"want kinds [READY, DOWNLOAD_COMPLETED, CACHED] = [1,2,3]; got {kinds}"
    # Ready phase fired.
    phases = [p for p, _ in w._emitted_phases]
    assert "ready" in phases, f"want `ready` in phases, got {phases}"


def test_manager_completion_does_not_fire_ready_until_all_refs_complete(
    tmp_path: Path,
) -> None:
    """With two required refs, _handle_manager_model_downloaded for one
    ref must NOT emit `ready`; only the second completion does.
    """
    w = _bare_worker_for_readiness(
        per_function={
            "infer": {"unet": "acme/sd-xl:latest"},
            "stage2": {"refiner": "acme/refiner:latest"},
        },
        function_names=["infer", "stage2"],
    )
    w._startup_required_refs_canonical = {"acme/sd-xl:latest", "acme/refiner:latest"}
    for r in w._startup_required_refs_canonical:
        w._model_cache.mark_downloading(r, progress=0.0)

    snap_a = tmp_path / "a"
    snap_a.mkdir()
    snap_b = tmp_path / "b"
    snap_b.mkdir()

    w._handle_manager_model_downloaded("acme/sd-xl:latest", snap_a)
    phases = [p for p, _ in w._emitted_phases]
    assert "ready" not in phases, f"ready emitted prematurely: {phases}"

    w._handle_manager_model_downloaded("acme/refiner:latest", snap_b)
    phases = [p for p, _ in w._emitted_phases]
    assert phases.count("ready") == 1, f"want exactly one `ready`, got {phases}"
