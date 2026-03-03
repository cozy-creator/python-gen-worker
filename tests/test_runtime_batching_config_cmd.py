import json
from types import SimpleNamespace

from gen_worker.decorators import ResourceRequirements
from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.worker import Worker


def test_runtime_batching_config_cmd_applies_and_acks(monkeypatch) -> None:
    w = Worker(user_module_names=[], worker_jwt="dummy-worker-jwt")
    w._task_specs["caption"] = SimpleNamespace(output_mode="incremental")

    sent = []
    monkeypatch.setattr(w, "_send_message", lambda m: sent.append(m))

    cmd = pb.RuntimeBatchingConfigCommand(
        config=pb.RuntimeBatchingConfig(
            function_name="caption",
            batch_size_target=6,
            batch_size_min=2,
            batch_size_max=8,
            prefetch_depth=3,
            max_wait_ms=120,
            version=3,
        )
    )
    w._handle_runtime_batching_config_cmd(cmd)

    cfg = w._runtime_batching_cfg_for_function("caption")
    assert cfg["function_name"] == "caption"
    assert cfg["batch_size_target"] == 6
    assert cfg["batch_size_min"] == 2
    assert cfg["batch_size_max"] == 8
    assert cfg["prefetch_depth"] == 3
    assert cfg["max_wait_ms"] == 120
    assert cfg["version"] == 3

    results = [m.runtime_batching_config_result for m in sent if m.HasField("runtime_batching_config_result")]
    assert len(results) == 1
    assert results[0].function_name == "caption"
    assert results[0].version == 3
    assert results[0].success is True
    assert results[0].error_message == ""


def test_runtime_batching_config_cmd_stale_version_is_ignored(monkeypatch) -> None:
    w = Worker(user_module_names=[], worker_jwt="dummy-worker-jwt")
    w._task_specs["caption"] = SimpleNamespace(output_mode="single")
    w._runtime_batching_config_by_function["caption"] = {
        "function_name": "caption",
        "batch_size_target": 5,
        "batch_size_min": 1,
        "batch_size_max": 6,
        "prefetch_depth": 2,
        "max_wait_ms": 100,
        "version": 4,
    }

    sent = []
    monkeypatch.setattr(w, "_send_message", lambda m: sent.append(m))

    msg = pb.WorkerSchedulerMessage(
        runtime_batching_config_cmd=pb.RuntimeBatchingConfigCommand(
            config=pb.RuntimeBatchingConfig(
                function_name="caption",
                batch_size_target=2,
                batch_size_min=1,
                batch_size_max=2,
                prefetch_depth=1,
                max_wait_ms=50,
                version=3,  # stale
            )
        )
    )
    w._process_message(msg)

    cfg = w._runtime_batching_cfg_for_function("caption")
    assert cfg["version"] == 4
    assert cfg["batch_size_target"] == 5

    results = [m.runtime_batching_config_result for m in sent if m.HasField("runtime_batching_config_result")]
    assert len(results) == 1
    assert results[0].success is True
    assert results[0].version == 3


def test_function_capabilities_event_emits_when_changed(monkeypatch) -> None:
    w = Worker(user_module_names=[], worker_jwt="dummy-worker-jwt")
    w._discovered_resources["caption"] = ResourceRequirements(
        max_concurrency=1,
        batch_size_min=1,
        batch_size_target=4,
        batch_size_max=8,
        prefetch_depth=2,
        max_wait_ms=150,
        memory_hint_mb=12288,
        stage_profile="io_gpu_disaggregated",
        stage_traits=["decode_prefetch", "gpu_decode_overlap"],
    )
    w._task_specs["caption"] = SimpleNamespace(output_mode="incremental")

    sent = []
    monkeypatch.setattr(w, "_send_message", lambda m: sent.append(m))

    w._emit_function_capabilities_event()
    w._emit_function_capabilities_event()  # second send should be deduped

    events = [m.worker_event for m in sent if m.HasField("worker_event")]
    assert len(events) == 1
    assert events[0].event_type == "worker.function_capabilities"
    payload = json.loads(bytes(events[0].payload_json or b"{}").decode("utf-8"))
    fns = list(payload.get("functions") or [])
    assert len(fns) == 1
    assert int(fns[0].get("max_inflight_requests") or 0) == 1
