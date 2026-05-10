import json
from types import SimpleNamespace

from gen_worker.api.decorators import ResourceRequirements
from gen_worker.worker import Worker


def _worker_shell() -> Worker:
    w = Worker.__new__(Worker)
    w.worker_id = "worker-1"
    w.release_id = "rel-1"
    w.runpod_pod_id = ""
    w.max_concurrency = 0
    w._request_specs = {
        "cpu_function": SimpleNamespace(output_mode="single"),
        "gpu_function": SimpleNamespace(output_mode="single"),
    }
    w._ws_specs = {}
    w._training_specs = {}
    w._function_schemas = {
        "cpu_function": (b"{}", b"{}", None, b"[]"),
        "gpu_function": (b"{}", b"{}", None, b"[]"),
    }
    w._discovered_resources = {
        "cpu_function": ResourceRequirements(accelerator="none"),
        "gpu_function": ResourceRequirements(
            accelerator="cuda",
            cuda_compute_min=9,
            min_vram_gb=16,
            required_libraries=["modelopt"],
        ),
    }
    w._disabled_functions_by_name = {}
    w._worker_local_unavailable_functions_by_name = {}
    w._payload_ref_availability_by_function = {}
    w._last_function_capabilities_hash = ""
    return w


def test_worker_local_availability_disables_cuda_function_on_cpu_host() -> None:
    w = _worker_shell()

    w._refresh_worker_local_function_availability(
        {
            "gpu_count": 0,
            "gpu_total_mem": 0,
            "gpu_sm": "",
            "installed_libs": [],
        }
    )

    assert w.is_function_runnable("cpu_function")
    assert not w.is_function_runnable("gpu_function")
    entry = w._function_unavailable_entry("gpu_function")
    assert entry is not None
    assert entry["reason"] == "cuda_unavailable"
    assert w._available_function_names() == ["cpu_function"]


def test_worker_local_availability_checks_compute_vram_and_optional_libs() -> None:
    w = _worker_shell()

    w._refresh_worker_local_function_availability(
        {
            "gpu_count": 1,
            "gpu_total_mem": 24 * 1024**3,
            "gpu_sm": "90",
            "installed_libs": ["torchao"],
        }
    )

    assert not w.is_function_runnable("gpu_function")
    entry = w._function_unavailable_entry("gpu_function")
    assert entry is not None
    assert entry["reason"] == "missing_optional_library"
    assert entry["axes"]["missing_libraries"] == "modelopt"


def test_registration_advertises_only_host_runnable_functions_and_reports_reasons() -> None:
    w = _worker_shell()
    sent = []

    w._collect_gpu_and_memory_info = lambda: {
        "cpu_cores": 4,
        "memory_bytes": 8 * 1024**3,
        "gpu_count": 0,
        "gpu_total_mem": 0,
        "gpu_used_mem": 0,
        "gpu_free_mem": 0,
        "gpu_name": "",
        "gpu_driver": "",
        "gpu_sm": "",
        "cuda_version": "",
        "torch_version": "",
        "installed_libs": [],
    }
    w._collect_model_inventory = lambda: ([], [], [], False)
    w._send_message = sent.append

    w._register_worker(is_heartbeat=False)

    registration = sent[0].worker_registration
    assert list(registration.resources.available_functions) == ["cpu_function"]
    assert [s.name for s in registration.resources.function_schemas] == ["cpu_function"]

    capability_events = [
        msg.worker_event
        for msg in sent
        if msg.HasField("worker_event") and msg.worker_event.event_type == "worker.function_capabilities"
    ]
    assert len(capability_events) == 1
    payload = json.loads(capability_events[0].payload_json.decode("utf-8"))
    by_name = {fn["function_name"]: fn for fn in payload["functions"]}
    assert by_name["cpu_function"]["available"] is True
    assert by_name["gpu_function"]["available"] is False
    assert by_name["gpu_function"]["unavailable_reason"] == "cuda_unavailable"
