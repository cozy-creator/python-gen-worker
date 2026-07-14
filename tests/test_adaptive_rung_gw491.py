"""gw#491: load-time precision reality must reach ServePlan/FnDegraded.

Three seams under test:
- serve_fit: ``_wanted`` counts a cast directive (storage_dtype), so a
  SUCCESSFUL cast reports wanted=fp8 ran=fp8 instead of masquerading as
  bf16; ``load_rung_engaged`` produces the plan-time emergency shape for
  load-time rung engagement; ``cast_dropped`` reports the actual base
  precision it fell back to.
- loading: an adaptive-fit-rung engagement is stamped on the pipe
  (``_cozy_adaptive_rung``), same pattern as the th#737 cast stamps.
- executor: the stamp is reconciled into serve_plans + the state-delta
  path via the REAL ensure_setup/injection path — a silently nf4-quantized
  pipeline must never report RUN_NATIVE wanted==ran.
"""

from __future__ import annotations

import asyncio
import json
import logging
import struct
from pathlib import Path

import msgspec

import gen_worker.models.loading as loading_mod
from gen_worker.api.binding import Hub
from gen_worker.executor import Executor, ModelStore
from gen_worker.models.serve_fit import (
    RUN_EMERGENCY,
    RUN_FP8_STORAGE,
    ServePlan,
    _wanted,
    cast_dropped,
    load_rung_engaged,
)
from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.registry import EndpointSpec
from gen_worker.api.decorators import Resources


# --------------------------------------------------------------------------
# serve_fit: wanted derivation + plan shapes
# --------------------------------------------------------------------------


def test_wanted_counts_storage_dtype() -> None:
    assert _wanted(Hub("a/b", storage_dtype="fp8")) == "fp8"
    assert _wanted(Hub("a/b", storage_dtype="fp8+te")) == "fp8+te"


def test_wanted_flavor_beats_storage_dtype() -> None:
    assert _wanted(Hub("a/b", flavor="svdq-int4-r128")) == "svdq-int4-r128"
    assert _wanted(Hub("a/b")) == "bf16"


def test_cast_dropped_reports_actual_base_precision() -> None:
    plan = cast_dropped(None, wanted="fp8", detail="no surface", ran="fp16")
    assert plan.degraded
    assert plan.wanted == "fp8" and plan.ran == "fp16"


def test_load_rung_engaged_nf4_is_degraded_emergency() -> None:
    base = ServePlan(serveable=True, run_mode="native", fit="fits",
                     wanted="bf16", ran="bf16")
    plan = load_rung_engaged(base, rung="nf4", detail="tight VRAM")
    assert plan.degraded
    assert plan.run_mode == RUN_EMERGENCY and plan.ran == RUN_EMERGENCY
    assert plan.est_latency_multiplier > 1.0


def test_load_rung_engaged_fp8_is_degraded_fp8_storage() -> None:
    plan = load_rung_engaged(None, rung="fp8", detail="tight VRAM")
    assert plan.degraded
    assert plan.run_mode == RUN_FP8_STORAGE and plan.ran == RUN_FP8_STORAGE


# --------------------------------------------------------------------------
# loading: rung engagement stamped on the pipe
# --------------------------------------------------------------------------


class _Pipe:
    @classmethod
    def from_pretrained(cls, path: str, **kwargs):
        return cls()


def _write_safetensors(path: Path, dtype: str = "BF16", nbytes: int = 1024) -> None:
    header = json.dumps(
        {"w": {"dtype": dtype, "shape": [nbytes], "data_offsets": [0, nbytes]}}
    ).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header)))
        f.write(header)


def _snapshot(tmp_path: Path, components: dict | None = None) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    index: dict = {"_class_name": "Pipe"}
    index.update(components or {"transformer": ["a", "b"]})
    (tmp_path / "model_index.json").write_text(json.dumps(index))
    _write_safetensors(tmp_path / "diffusion_pytorch_model.safetensors")
    return tmp_path


def test_load_stamps_adaptive_rung(tmp_path: Path, monkeypatch) -> None:
    snap = _snapshot(tmp_path / "snapdir")
    monkeypatch.setattr(loading_mod, "_adaptive_fit_rung",
                        lambda *a, **k: ("nf4", ("bnb", {})))
    pipe = loading_mod.load_from_pretrained(_Pipe, snap, dtype="bf16")
    assert getattr(pipe, "_cozy_adaptive_rung", "") == "nf4"


def test_load_no_stamp_when_rung_stays_out(tmp_path: Path, monkeypatch) -> None:
    snap = _snapshot(tmp_path / "snapdir")
    monkeypatch.setattr(loading_mod, "_adaptive_fit_rung",
                        lambda *a, **k: ("", None))
    pipe = loading_mod.load_from_pretrained(_Pipe, snap, dtype="bf16")
    assert getattr(pipe, "_cozy_adaptive_rung", "") == ""


# --------------------------------------------------------------------------
# executor: the stamp reconciles into serve_plans (state-delta path)
# --------------------------------------------------------------------------


class _In(msgspec.Struct):
    x: str = "hi"


class _Out(msgspec.Struct):
    ok: bool = True


class _RungShim(_Pipe):
    def to(self, *a, **k):
        return self


def _executor(spec: EndpointSpec, tmp_path: Path, snap: Path, sent: list) -> Executor:
    async def _send(msg: pb.WorkerMessage) -> None:
        sent.append(msg)

    store = ModelStore(_send, cache_dir=tmp_path, vram_budget_bytes=4 << 30)

    async def _fake_ensure_local(ref, snapshot=None, *, binding=None) -> Path:
        store.residency.track_disk(ref, snap)
        return snap

    store.ensure_local = _fake_ensure_local  # type: ignore[method-assign]
    return Executor([spec], _send, store=store)


def test_executor_records_adaptive_rung(tmp_path, monkeypatch, caplog) -> None:
    monkeypatch.setattr(loading_mod, "_adaptive_fit_rung",
                        lambda *a, **k: ("nf4", ("bnb", {})))

    class Endpoint:
        def setup(self, m: _RungShim) -> None:
            self.m = m

        def run(self, ctx, payload: _In) -> _Out:  # pragma: no cover
            return _Out()

    spec = EndpointSpec(
        name="generate", method=Endpoint.run, kind="inference",
        payload_type=_In, output_mode="single", cls=Endpoint, attr_name="run",
        models={"m": Hub("acme/big-model")},
        resources=Resources(vram_gb=1.0),
    )
    snap = _snapshot(tmp_path / "snapdir")
    sent: list = []

    async def _go() -> None:
        ex = _executor(spec, tmp_path, snap, sent)
        with caplog.at_level(logging.WARNING):
            inst = await ex.ensure_setup(spec)
        assert getattr(inst.m, "_cozy_adaptive_rung", "") == "nf4"
        plan = ex.serve_plans["generate"]
        assert plan.degraded
        assert plan.ran == RUN_EMERGENCY
        assert "LOAD_RUNG_ENGAGED" in caplog.text

    asyncio.run(_go())
