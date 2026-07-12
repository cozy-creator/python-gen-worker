"""diffusers_step_callback (pgw#482): one-line per-step progress for
diffusers pipelines — step/total on the wire, throttled, cancel-aware."""

from __future__ import annotations

import pytest

from gen_worker import CanceledError, RequestContext, diffusers_step_callback


def _ctx(events: list) -> RequestContext:
    return RequestContext(request_id="r1", emitter=events.append)


def _run_steps(cb, total: int) -> None:
    for i in range(total):
        cb(None, i, None, {})


def test_emits_step_total_and_fraction() -> None:
    events: list = []
    cb = diffusers_step_callback(_ctx(events), 4, min_interval_s=0.0)
    _run_steps(cb, 4)
    assert [e["type"] for e in events] == ["request.progress"] * 4
    payloads = [e["payload"] for e in events]
    assert payloads[0] == {"progress": 0.25, "stage": "denoise", "step": 1, "total": 4}
    assert payloads[-1] == {"progress": 1.0, "stage": "denoise", "step": 4, "total": 4}


def test_throttles_but_always_emits_first_and_last() -> None:
    events: list = []
    # Huge interval: only the first and last of 20 steps get through.
    cb = diffusers_step_callback(_ctx(events), 20, min_interval_s=3600.0)
    _run_steps(cb, 20)
    steps = [e["payload"]["step"] for e in events]
    assert steps == [1, 20]


def test_returns_callback_kwargs_unchanged() -> None:
    events: list = []
    cb = diffusers_step_callback(_ctx(events), 2, min_interval_s=0.0)
    kwargs = {"latents": object()}
    assert cb(None, 0, None, kwargs) is kwargs
    assert cb(None, 1, None) == {}  # pipelines that pass no kwargs


def test_custom_stage_and_stage_none() -> None:
    events: list = []
    ctx = _ctx(events)
    diffusers_step_callback(ctx, 1, stage="upscale", min_interval_s=0.0)(None, 0, None, {})
    diffusers_step_callback(ctx, 1, stage=None, min_interval_s=0.0)(None, 0, None, {})
    assert events[0]["payload"]["stage"] == "upscale"
    assert "stage" not in events[1]["payload"]


def test_cancelled_request_aborts_mid_pipeline() -> None:
    events: list = []
    ctx = _ctx(events)
    cb = diffusers_step_callback(ctx, 10, min_interval_s=0.0)
    cb(None, 0, None, {})
    ctx._cancel()
    with pytest.raises(CanceledError):
        cb(None, 1, None, {})
    assert len(events) == 1  # nothing emitted after cancellation


def test_window_maps_progress_into_sub_range_but_step_total_stay_stage_local() -> None:
    events: list = []
    cb = diffusers_step_callback(
        _ctx(events), 4, stage="denoise", min_interval_s=0.0, window=(0.0, 0.6)
    )
    _run_steps(cb, 4)
    payloads = [e["payload"] for e in events]
    assert payloads[0] == {"progress": 0.15, "stage": "denoise", "step": 1, "total": 4}
    assert payloads[-1] == {"progress": 0.6, "stage": "denoise", "step": 4, "total": 4}


def test_two_stage_pipeline_composes_two_windows_without_resetting_bar() -> None:
    events: list = []
    ctx = _ctx(events)
    denoise = diffusers_step_callback(ctx, 2, stage="denoise", min_interval_s=0.0, window=(0.0, 0.6))
    refine = diffusers_step_callback(ctx, 3, stage="refine", min_interval_s=0.0, window=(0.65, 0.9))
    _run_steps(denoise, 2)
    _run_steps(refine, 3)
    payloads = [e["payload"] for e in events]
    # Denoise stage: 0.3, 0.6 (never resets to 0 once refine starts).
    assert payloads[0]["progress"] == pytest.approx(0.3)
    assert payloads[1]["progress"] == pytest.approx(0.6)
    # Refine stage climbs monotonically from where denoise left off.
    assert payloads[2]["progress"] == pytest.approx(0.65 + 0.25 * (1 / 3))
    assert payloads[-1]["progress"] == pytest.approx(0.9)
    assert all(p["progress"] >= payloads[i - 1]["progress"] for i, p in enumerate(payloads) if i)


def test_window_validation_rejects_out_of_range_or_inverted_bounds() -> None:
    ctx = _ctx([])
    for bad_window in [(-0.1, 0.5), (0.5, 1.1), (0.6, 0.4)]:
        with pytest.raises(ValueError):
            diffusers_step_callback(ctx, 4, window=bad_window)  # type: ignore[arg-type]


def test_default_window_is_full_range_backward_compatible() -> None:
    events: list = []
    cb = diffusers_step_callback(_ctx(events), 4, min_interval_s=0.0)
    _run_steps(cb, 4)
    assert [e["payload"]["progress"] for e in events] == [0.25, 0.5, 0.75, 1.0]
