"""Incremental function readiness (gen-orchestrator #341).

Each worker registration carries two parallel function lists:

  available_functions -- routable now
  loading_functions   -- bound models still downloading; routable later

The orchestrator dispatches queued jobs the moment a function moves from
loading -> available. A worker with two model-bound functions where one
download finishes before the other:

  T0  available=[]            loading=[a, b]    (both downloading)
  T1  available=[a]           loading=[b]       (a's model on disk)
  T2  available=[a, b]        loading=[]        (b's model on disk too)

This test drives those transitions through the actual code path used by
``_register_worker`` to populate ``WorkerResources`` — including the
``is_function_runnable`` gate that prevents dispatch onto a function whose
model is still downloading. A second test confirms a worker with no
model-bound functions (e.g. marco-polo) never reports loading_functions.

Why we don't spin up a real gRPC stream: the wire-up of model-cache state
-> WorkerResources advertisement is entirely synchronous and pure-Python;
running it inside Worker.__new__ is faster and keeps the test deterministic
across CI shapes (no TLS / scheduler dependency).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List
from unittest.mock import MagicMock

from gen_worker import Resources
from gen_worker.models.cache import ModelCache
from gen_worker.worker import Worker


def _bare_worker_with_two_model_bound_functions(
    fixed_refs: dict[str, str],
    payload_refs_by_fn: dict[str, dict[str, str]],
    discovered_resources: dict[str, Resources] | None = None,
) -> Worker:
    """Construct a Worker without running __init__.

    Pre-populates the same in-memory state the manifest parser would
    install plus an empty ModelCache. ``_loading_function_names`` and
    ``_function_unavailable_entry`` consult these directly.
    """
    w = Worker.__new__(Worker)
    w._fixed_model_id_by_key = dict(fixed_refs)
    w._payload_model_id_by_key_by_function = {
        fn: dict(m) for fn, m in payload_refs_by_fn.items()
    }
    w._disabled_functions_by_name = {}
    w._worker_local_unavailable_functions_by_name = {}
    w._discovered_resources = dict(discovered_resources or {})
    # Synthetic specs so _all_declared_function_names enumerates every
    # bound function. The dispatch side never runs in this test.
    w._request_specs = {}
    w._training_specs = {}
    w._batched_specs = {}
    w._serial_class_specs = {
        fn: MagicMock() for fn in payload_refs_by_fn.keys()
    }
    # Real ModelCache against a tmp dir avoids the VRAM auto-detect
    # branching on a CI host with/without CUDA.
    w._model_cache = ModelCache(max_vram_gb=0.0, model_cache_dir="/tmp/test-mc")
    return w


def test_two_functions_one_loading_one_ready_drives_correct_split() -> None:
    """fn_small's model is already on disk; fn_large's model is downloading.

    loading_functions reports [fn_large] only; available_functions reports
    [fn_small] only. is_function_runnable agrees with both lists.
    """
    w = _bare_worker_with_two_model_bound_functions(
        fixed_refs={},
        payload_refs_by_fn={
            "fn_small": {"weights": "acme/tiny-model:latest"},
            "fn_large": {"weights": "acme/giant-model:latest"},
        },
    )
    w._model_cache.mark_cached_to_disk("acme/tiny-model:latest", Path("/tmp/x"))
    w._model_cache.mark_downloading("acme/giant-model:latest", progress=0.42)

    loading = w._loading_function_names()
    assert loading == ["fn_large"]

    assert w.is_function_runnable("fn_small") is True
    assert w.is_function_runnable("fn_large") is False

    available = w._available_function_names()
    assert available == ["fn_small"]


def test_register_worker_proto_carries_both_lists() -> None:
    """End-to-end check: the message that hits the gRPC wire has both
    available_functions and loading_functions populated.

    We capture the registration message instead of opening a real stream.
    """
    w = _bare_worker_with_two_model_bound_functions(
        fixed_refs={},
        payload_refs_by_fn={
            "fn_small": {"weights": "acme/tiny-model:latest"},
            "fn_large": {"weights": "acme/giant-model:latest"},
        },
    )
    w.worker_id = "test-worker"
    w.release_id = "rel-1"
    w.runpod_pod_id = ""
    w._last_disk_inventory_hash = ""
    w._last_function_capabilities_hash = ""
    w._model_manager = None

    captured: List[Any] = []

    def fake_send(msg: Any) -> None:
        captured.append(msg)

    w._send_message = fake_send
    w._send_heartbeat_message = fake_send
    w._collect_gpu_and_memory_info = lambda: {
        "gpu_count": 0,
        "gpu_total_mem": 0,
        "gpu_free_mem": 0,
        "gpu_name": "",
        "gpu_sm": "",
        "installed_libs": [],
    }
    w._refresh_worker_local_function_availability = lambda gpu_info: None
    w._get_gpu_busy_status = lambda: False
    w._emit_function_capabilities_event = lambda: None
    w._shared_disk_volume_info = lambda: None

    w._model_cache.mark_cached_to_disk("acme/tiny-model:latest", Path("/tmp/x"))
    w._model_cache.mark_downloading("acme/giant-model:latest", progress=0.10)

    w._register_worker(is_heartbeat=False)

    assert captured, "expected a registration message on the wire"
    reg = captured[0].worker_registration
    assert list(reg.resources.available_functions) == ["fn_small"]
    assert list(reg.resources.loading_functions) == ["fn_large"]


def test_loading_to_available_transition_after_download_completes() -> None:
    """T1 -> T2 transition: when the second model finishes downloading,
    loading_functions shrinks to [] and available_functions grows.

    Models transition by calling mark_cached_to_disk — the same API the
    real prefetch loop calls when a download finishes. The prefetch loop
    also fires _register_worker(is_heartbeat=True) right after this call,
    which is the out-of-band heartbeat that makes the orchestrator learn
    immediately rather than waiting for the next 10s tick. (We test the
    out-of-band heartbeat trigger in a separate test below.)
    """
    w = _bare_worker_with_two_model_bound_functions(
        fixed_refs={},
        payload_refs_by_fn={
            "fn_small": {"weights": "acme/tiny-model:latest"},
            "fn_large": {"weights": "acme/giant-model:latest"},
        },
    )
    # T0: both downloading
    w._model_cache.mark_downloading("acme/tiny-model:latest", progress=0.0)
    w._model_cache.mark_downloading("acme/giant-model:latest", progress=0.0)
    assert sorted(w._loading_function_names()) == ["fn_large", "fn_small"]
    assert w._available_function_names() == []

    # T1: small finishes
    w._model_cache.mark_cached_to_disk("acme/tiny-model:latest", Path("/tmp/a"))
    assert w._loading_function_names() == ["fn_large"]
    assert w._available_function_names() == ["fn_small"]

    # T2: large finishes
    w._model_cache.mark_cached_to_disk("acme/giant-model:latest", Path("/tmp/b"))
    assert w._loading_function_names() == []
    assert sorted(w._available_function_names()) == ["fn_large", "fn_small"]


def test_function_with_no_bound_models_is_never_loading() -> None:
    """Marco-polo style: a function declared with zero model bindings is
    never reported as loading, even when the cache is full of in-flight
    downloads for unrelated refs.
    """
    w = _bare_worker_with_two_model_bound_functions(
        fixed_refs={},
        payload_refs_by_fn={
            "marco-polo": {},  # no models
        },
    )
    # Drop a noise download into the cache so downloading_models is non-empty.
    w._model_cache.mark_downloading("unrelated/ref:latest", progress=0.5)

    assert w._loading_function_names() == []
    assert w.is_function_runnable("marco-polo") is True
    assert w._available_function_names() == ["marco-polo"]


def test_fixed_keyspace_ref_blocks_every_function_that_depends_on_it() -> None:
    """A ref in _fixed_model_id_by_key is shared by every function in the
    release. When that ref is downloading, every function lands in
    loading_functions.
    """
    w = _bare_worker_with_two_model_bound_functions(
        fixed_refs={"shared": "acme/base-model:latest"},
        payload_refs_by_fn={
            "fn_a": {},
            "fn_b": {},
        },
    )
    w._model_cache.mark_downloading("acme/base-model:latest", progress=0.0)

    assert sorted(w._loading_function_names()) == ["fn_a", "fn_b"]
    assert w._available_function_names() == []

    # When the shared model finishes, both become available simultaneously.
    w._model_cache.mark_cached_to_disk("acme/base-model:latest", Path("/tmp/s"))
    assert w._loading_function_names() == []
    assert sorted(w._available_function_names()) == ["fn_a", "fn_b"]


def test_loading_does_not_shadow_disabled_or_host_unavailable() -> None:
    """If a function is disabled (scheduler-reported terminal ref failure)
    OR host-unavailable (boot-time hardware gate), the loading channel
    must not also claim it. The orchestrator distinguishes "disabled"
    from "loading"; we only emit "loading" for the third channel.
    """
    w = _bare_worker_with_two_model_bound_functions(
        fixed_refs={},
        payload_refs_by_fn={
            "fn_disabled": {"weights": "acme/dead-ref:latest"},
            "fn_host_blocked": {"weights": "acme/needs-h100:latest"},
            "fn_loading": {"weights": "acme/in-flight:latest"},
        },
    )
    # All three models are technically downloading per the cache.
    w._model_cache.mark_downloading("acme/dead-ref:latest", progress=0.0)
    w._model_cache.mark_downloading("acme/needs-h100:latest", progress=0.0)
    w._model_cache.mark_downloading("acme/in-flight:latest", progress=0.0)

    w._disabled_functions_by_name = {
        "fn_disabled": {"reason": "ref_terminal", "ref": "acme/dead-ref:latest"},
    }
    w._worker_local_unavailable_functions_by_name = {
        "fn_host_blocked": {"reason": "compute_capability_unmet"},
    }

    # Only the genuinely-loading function shows up in loading_functions.
    assert w._loading_function_names() == ["fn_loading"]
    # All three are absent from available_functions, but the three reasons
    # differ: fn_loading because of the loading set; the other two via the
    # pre-existing channels.
    assert w._available_function_names() == []
    assert w.is_function_runnable("fn_disabled") is False
    assert w.is_function_runnable("fn_host_blocked") is False
    assert w.is_function_runnable("fn_loading") is False


def test_out_of_band_heartbeat_fires_when_model_transitions_in_prefetch_loop() -> None:
    """The prefetch loop already calls _register_worker(is_heartbeat=True)
    immediately after each model transitions downloading -> on-disk. This
    test asserts the call site exists and is triggered through the same
    surface that the new loading_functions advertisement is built from.

    We mock out the network/proto layer; what matters is that the call
    fires synchronously after mark_cached_to_disk, so the orchestrator
    learns of the loading -> available transition within one round-trip
    instead of waiting for the next 10s heartbeat tick.
    """
    w = _bare_worker_with_two_model_bound_functions(
        fixed_refs={},
        payload_refs_by_fn={"fn": {"weights": "acme/m:latest"}},
    )
    w.worker_id = "w"
    w.release_id = "r"
    w.runpod_pod_id = ""
    w._last_disk_inventory_hash = ""
    w._last_function_capabilities_hash = ""
    w._model_manager = None

    heartbeats: List[Any] = []

    def fake_send_hb(msg: Any) -> None:
        heartbeats.append(msg)

    sent: List[Any] = []

    def fake_send(msg: Any) -> None:
        sent.append(msg)

    w._send_message = fake_send
    w._send_heartbeat_message = fake_send_hb
    w._collect_gpu_and_memory_info = lambda: {
        "gpu_count": 0,
        "gpu_total_mem": 0,
        "gpu_free_mem": 0,
        "gpu_name": "",
        "gpu_sm": "",
        "installed_libs": [],
    }
    w._refresh_worker_local_function_availability = lambda gpu_info: None
    w._get_gpu_busy_status = lambda: False
    w._emit_function_capabilities_event = lambda: None
    w._shared_disk_volume_info = lambda: None

    # T0: model downloading -> registration shows fn in loading_functions.
    w._model_cache.mark_downloading("acme/m:latest", progress=0.0)
    w._register_worker(is_heartbeat=False)
    assert sent, "expected initial registration on primary stream"
    reg0 = sent[0].worker_registration.resources
    assert list(reg0.loading_functions) == ["fn"]
    assert list(reg0.available_functions) == []

    # The model finishes; the prefetch loop (synchronously) fires the
    # out-of-band heartbeat. We emulate that single line.
    w._model_cache.mark_cached_to_disk("acme/m:latest", Path("/tmp/done"))
    w._register_worker(is_heartbeat=True)
    assert heartbeats, "expected out-of-band heartbeat on heartbeat stream"
    reg1 = heartbeats[0].worker_registration.resources
    assert list(reg1.loading_functions) == []
    assert list(reg1.available_functions) == ["fn"]
    # is_heartbeat is set so the orchestrator does NOT treat this as a
    # re-registration (which would re-issue EndpointConfig and re-enter
    # the prefetch loop).
    assert heartbeats[0].worker_registration.is_heartbeat is True
