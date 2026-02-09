from __future__ import annotations

from pathlib import Path

from gen_worker.run_metrics_v1 import RunMetricsV1, best_effort_init_model_metrics


def test_metrics_run_payload_omits_unknown_fields() -> None:
    rm = RunMetricsV1(run_id="r1", function_name="f")
    payload = rm.to_metrics_run_payload()
    assert payload["schema_version"] == 1
    assert payload["function_name"] == "f"
    assert "models" not in payload
    assert "inference_ms" not in payload


def test_canonical_events_include_only_known_values() -> None:
    rm = RunMetricsV1(run_id="r1", function_name="f")
    rm.mark_compute_started()
    rm.fetch_ms = 0
    rm.inference_ms = 123
    rm.mark_compute_completed()
    evs = rm.canonical_events()
    types = [t for (t, _p) in evs]
    assert "metrics.compute.started" in types
    assert "metrics.compute.completed" in types
    assert "metrics.fetch" in types
    assert "metrics.inference" in types
    # gpu_load omitted if unknown
    assert "metrics.gpu_load" not in types


def test_best_effort_init_model_metrics_cache_state() -> None:
    rm = RunMetricsV1(run_id="r1", function_name="f", resolved_cozy_models_by_id={})
    best_effort_init_model_metrics(
        rm,
        ["cozy:org/repo:tag"],
        vram_models=["cozy:org/repo:tag"],
        disk_models=[],
        cache_dir=Path("/tmp/does-not-exist"),
    )
    assert rm.models
    ent = list(rm.models.values())[0]
    assert ent.cache_state == "hot_vram"

