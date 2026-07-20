"""th#567 hot adoption: MODEL_OP_KIND_ADOPT_COMPILE_CACHE re-wraps resident
modules in place — verified seed, one warmup, ADOPTED report; ANY failure
stays eager with a classified adopt_failed:<reason>. Plus th#569 boot-attach:
a compile-cache snapshot on RunJob.snapshots reaches compile_cache.enable."""

from __future__ import annotations

import asyncio
import threading
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import msgspec
import pytest

from gen_worker import Compile, RequestContext, Resources, endpoint
from gen_worker import compile_cache as cc
from gen_worker.models import provision
from gen_worker.api.binding import Hub
from gen_worker.api.binding import wire_ref
from gen_worker.api.errors import RetryableError
from gen_worker.executor import Executor, _Job
from gen_worker.lifecycle import Lifecycle
from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.registry import EndpointSpec, extract_specs

FAMILY = "flux2-klein-4b"
CACHE_REF = f"_system/family-{FAMILY}#inductor-rtx-4090-torch2.9"
MODEL_REF = "acme/klein-finetune:latest"
DIGEST_A = "blake3:" + "a" * 64
DIGEST_B = "blake3:" + "b" * 64
MODEL_DIGEST = "blake3:" + "c" * 64
OP_A = "adopt-operation-a"
OP_B = "adopt-operation-b"


class _In(msgspec.Struct):
    prompt: str = ""


class _Out(msgspec.Struct):
    y: str = ""


class _Denoiser:
    def forward(self, *args, **kwargs):  # pragma: no cover - contract surface
        return None


class _Pipe:
    def __init__(self):
        self.transformer = _Denoiser()


class _LoadablePipe(_Pipe):
    @classmethod
    def from_pretrained(cls, path, **kwargs):  # pragma: no cover - loader is patched
        return cls()


class _AncillaryVae:
    pass


class _ColdEndpoint:
    setups = 0
    warmups = 0
    runs = 0

    def setup(self, pipeline: _LoadablePipe) -> None:
        type(self).setups += 1
        self.pipeline = pipeline

    def warmup(self) -> None:
        type(self).warmups += 1
        _record_fake_warm(self.pipeline)

    def run(self, ctx, payload: _In) -> _Out:
        type(self).runs += 1
        return _Out(y="ok")


class _ColdEndpointB(_ColdEndpoint):
    pass


class _Endpoint:
    warmups = 0

    def setup(self, pipeline: str) -> None:  # pragma: no cover
        pass

    def warmup(self) -> None:
        type(self).warmups += 1
        pipeline = getattr(self, "pipeline", None)
        if pipeline is not None:
            _record_fake_warm(pipeline)

    def run(self, ctx, payload: _In) -> _Out:  # pragma: no cover
        return _Out()


_FAKE_WARM_PROOF = {"hits": 1, "misses": 0}


def _mark_fake_guard(pipeline) -> None:
    setattr(pipeline, cc._MARKER_ATTR, {
        "failure_signal": {
            "callback": None,
            "lock": threading.Lock(),
            "successful_calls": 0,
            "cache_hits": 0,
            "cache_misses": 0,
        },
        "originals": [],
        "regional_mods": [],
    })


def _record_fake_warm(pipeline, *, hits=None, misses=None) -> None:
    marker = getattr(pipeline, cc._MARKER_ATTR, None) or {}
    signal = marker.get("failure_signal")
    if not isinstance(signal, dict):
        return
    lock = signal["lock"]
    with lock:
        activated = signal["cache_hits"] > 0
        signal["successful_calls"] += 1
        signal["cache_hits"] += (
            (0 if activated else _FAKE_WARM_PROOF["hits"])
            if hits is None else hits)
        signal["cache_misses"] += (
            (0 if activated else _FAKE_WARM_PROOF["misses"])
            if misses is None else misses)


def _guarded_apply(pipeline, _cfg, *, cache_ready, guard=True):
    assert cache_ready
    assert guard
    _mark_fake_guard(pipeline)
    return True


def _guarded_enable(pipeline, *_args):
    from gen_worker import fleet_cells

    _mark_fake_guard(pipeline)
    return fleet_cells.ArmOutcome(armed=True)


def _spec(compile_cfg=None) -> EndpointSpec:
    return EndpointSpec(
        name="ep", method=_Endpoint.run, kind="inference",
        payload_type=_In, output_mode="single", cls=_Endpoint,
        attr_name="run", models={"pipeline": Hub("acme/klein-finetune")},
        compile=compile_cfg or Compile(shapes=((768, 768),), family=FAMILY),
    )


def _artifact(
    tmp_path: Path, *, family: str = FAMILY, **meta_overrides,
) -> Path:
    cap = tmp_path / "cap"
    (cap / "inductor" / "g").mkdir(parents=True)
    (cap / "inductor" / "g" / "code.py").write_text("x")
    (cap / "triton").mkdir()
    cfg = Compile(shapes=((768, 768),), family=family)
    signature, weight_contract = cc.execution_contract(_Pipe(), cfg)
    meta = cc.artifact_metadata(
        family=family, shapes=cfg.shapes, targets=cfg.targets,
        graph_signature=signature, weight_contract=weight_contract,
    )
    meta.update(meta_overrides)
    snapdir = tmp_path / "snap"
    snapdir.mkdir(exist_ok=True)
    return cc.pack(cap, snapdir / "inductor-rtx-4090-torch2.9.tar.gz", meta)


def _wire_executor(spec, tmp_path, *, ready=True, resident=True):
    sent: list[pb.WorkerMessage] = []

    async def _send(msg: pb.WorkerMessage) -> None:
        sent.append(msg)

    ex = Executor([spec], _send)
    ex.store._cache_dir = tmp_path / "cas"
    rec = ex._classes[spec.instance_key]
    pipe = _Pipe()
    if ready:
        rec.instance = _Endpoint()
        rec.instance.pipeline = pipe
        rec.ready = True
        model_ref = wire_ref(spec.models["pipeline"])
        rec.held_refs = [model_ref]
        rec.held_snapshot_digests = {model_ref: MODEL_DIGEST}
        rec.held_bindings = [("pipeline", model_ref, MODEL_DIGEST)]
        active = None
        if "#fp8-w8a8" in model_ref:
            setattr(pipe, "_cozy_weight_lane", "w8a8")
            _mark_fake_guard(pipe)
            selection = type("Selection", (), {
                "ref": CACHE_REF + "-w8a8",
                "snapshot_digest": DIGEST_A,
                "path": tmp_path / "already-proven-w8a8.tar.gz",
            })()
            active = {id(pipe): selection}
        ex._install_compile_targets(
            rec,
            spec,
            [pipe],
            active,
            {id(pipe): {spec.name}} if active else None,
        )
    if resident:
        ex.store.residency.track_vram(
            wire_ref(spec.models["pipeline"]), pipe, vram_bytes=1)

    async def _fake_ensure_local(ref, snapshot=None, *, binding=None) -> Path:
        return tmp_path / "snap"

    ex.store.ensure_local = _fake_ensure_local  # type: ignore[method-assign]
    return ex, sent


def _events(sent, state):
    return [m.model_event for m in sent
            if m.WhichOneof("msg") == "model_event" and m.model_event.state == state]


# th#875's hub-side re-arm matcher compares these adopt statuses EXACTLY;
# the worker must keep them bare (no appended detail) on the wire (gw#577).
_TRANSIENT_REASONS = (
    "adopt_failed:model_in_use", "adopt_failed:target_not_ready",
    "adopt_failed:target_replaced", "adopt_failed:download",
)


def _assert_failed(
    sent, error, digest=DIGEST_A, operation_id=OP_A,
    target_incarnation_id=None,
):
    failed = _events(sent, pb.MODEL_STATE_FAILED)
    # gw#577: terminal refusals may carry ": <exact reason>" after the
    # classified code; the th#875 transient vocabulary must stay bare.
    assert failed and (
        failed[-1].error == error
        or failed[-1].error.startswith(error + ":")
    )
    if error in _TRANSIENT_REASONS:
        assert failed[-1].error == error
    assert failed[-1].snapshot_digest == digest
    assert failed[-1].operation_id == operation_id
    if target_incarnation_id is None:
        assert failed[-1].target_incarnation_id
    else:
        assert failed[-1].target_incarnation_id == target_incarnation_id
    return failed[-1]


_TARGET_UNSET = object()


def _target_id(ex) -> str:
    (target,) = ex.compile_targets()
    return target.incarnation_id


def _adopt(
    ex, ref=CACHE_REF, digest=DIGEST_A, operation_id=OP_A,
    target_incarnation_id=_TARGET_UNSET,
):
    if target_incarnation_id is _TARGET_UNSET:
        targets = ex.compile_targets()
        target_incarnation_id = (
            targets[0].incarnation_id if targets else "missing-incarnation")
    asyncio.run(ex.handle_model_op(
        pb.ModelOp(
            op=pb.MODEL_OP_KIND_ADOPT_COMPILE_CACHE,
            ref=ref,
            snapshot=pb.Snapshot(digest=digest),
            operation_id=operation_id,
            target_incarnation_id=target_incarnation_id,
        )
    ))


# ---------------------------------------------------------------------------
# ref helpers
# ---------------------------------------------------------------------------


def test_family_from_ref_and_is_cache_ref():
    assert cc.family_from_ref(CACHE_REF) == FAMILY
    assert cc.family_from_ref(f"_system/family-{FAMILY}:latest@blake3:aa#inductor-x") == FAMILY
    assert cc.family_from_ref("acme/model:latest") == ""
    assert cc.family_from_ref("_system/other-repo#inductor-x") == ""
    assert cc.is_cache_ref(CACHE_REF)
    assert cc.is_cache_ref(CACHE_REF, FAMILY)
    assert not cc.is_cache_ref(CACHE_REF, "sdxl")
    assert not cc.is_cache_ref(f"_system/family-{FAMILY}")  # no inductor flavor
    assert not cc.is_cache_ref("acme/model#inductor-x")


# ---------------------------------------------------------------------------
# adoption success
# ---------------------------------------------------------------------------


def _fake_counters(monkeypatch, *, hits=3, misses=1):
    """Simulate exact-object cache proof inside the endpoint warmup."""
    monkeypatch.setitem(_FAKE_WARM_PROOF, "hits", hits)
    monkeypatch.setitem(_FAKE_WARM_PROOF, "misses", misses)


def test_adopt_success_rewraps_warms_and_reports(tmp_path, monkeypatch):
    _artifact(tmp_path)
    spec = _spec()
    ex, sent = _wire_executor(spec, tmp_path)
    _Endpoint.warmups = 0

    applied: list[tuple] = []

    def _fake_apply(pipeline, cfg, *, cache_ready, guard=True):
        applied.append((pipeline, cfg, cache_ready))
        _mark_fake_guard(pipeline)
        return True

    monkeypatch.setattr(cc, "apply", _fake_apply)
    _fake_counters(monkeypatch, hits=3, misses=1)
    _adopt(ex)

    adopted = _events(sent, pb.MODEL_STATE_ADOPTED)
    assert len(adopted) == 1
    assert adopted[0].ref == CACHE_REF
    assert adopted[0].snapshot_digest == DIGEST_A
    assert adopted[0].operation_id == OP_A
    assert adopted[0].target_incarnation_id == _target_id(ex)
    assert adopted[0].duration_ms >= 0
    assert adopted[0].cache_hits == 3
    assert adopted[0].cache_misses == 1
    assert adopted[0].warmup_s >= 0
    assert not _events(sent, pb.MODEL_STATE_FAILED)
    assert len(applied) == 1 and applied[0][2] is True
    assert isinstance(applied[0][0], _Pipe)
    assert _Endpoint.warmups == 1
    target = ex.compile_targets()[0]
    assert target.active_compile_ref == CACHE_REF
    assert target.active_compile_snapshot_digest == DIGEST_A


def test_adopt_zero_cache_hits_rolls_back_and_fails(tmp_path, monkeypatch):
    """gw#391 honest failure mode: ADOPTED-while-silently-eager is impossible.
    A warmup observing zero fxgraph hits unwraps back to eager and reports
    adopt_failed:cache_miss."""
    _artifact(tmp_path)
    spec = _spec()
    ex, sent = _wire_executor(spec, tmp_path)
    monkeypatch.setattr(cc, "apply", _guarded_apply)
    _fake_counters(monkeypatch, hits=0, misses=2)
    unwrapped: list = []
    monkeypatch.setattr(cc, "unwrap", lambda obj: unwrapped.append(obj))

    _adopt(ex)
    _assert_failed(sent, "adopt_failed:cache_miss")
    assert not _events(sent, pb.MODEL_STATE_ADOPTED)
    assert any(isinstance(o, _Pipe) for o in unwrapped)  # rollback ran
    target = ex.compile_targets()[0]
    assert target.active_compile_ref == ""
    assert target.active_compile_snapshot_digest == ""


def test_endpoint_without_warmup_exposes_no_adopt_target(tmp_path, monkeypatch):
    """An endpoint without a warmup contract advertises no false target."""

    class _NoWarmupEndpoint:
        def setup(self, pipeline: str) -> None:  # pragma: no cover
            pass

        def run(self, ctx, payload: _In) -> _Out:  # pragma: no cover
            return _Out()

    _artifact(tmp_path)
    spec = EndpointSpec(
        name="ep", method=_NoWarmupEndpoint.run, kind="inference",
        payload_type=_In, output_mode="single", cls=_NoWarmupEndpoint,
        attr_name="run", models={"pipeline": Hub("acme/klein-finetune")},
        compile=Compile(shapes=((768, 768),), family=FAMILY),
    )
    ex, sent = _wire_executor(spec, tmp_path)
    rec = ex._classes[spec.instance_key]
    rec.instance = _NoWarmupEndpoint()
    rec.ready = True
    monkeypatch.setattr(
        cc, "apply", lambda *args, **kwargs: pytest.fail("must not adopt"))

    assert ex.compile_targets() == []
    _adopt(ex)
    _assert_failed(sent, "adopt_failed:target_not_ready")
    assert not _events(sent, pb.MODEL_STATE_ADOPTED)


def test_adopt_resident_prep_mode_drift_converges(tmp_path, monkeypatch):
    """gw#588: 'off' and 'vae_only' are both fully-resident preps — hot adopt
    converges the pipeline to the cell's traced mode and adopts instead of
    refusing (ie#501 run 18)."""
    _artifact(tmp_path, low_vram_mode="off")
    spec = _spec()
    ex, sent = _wire_executor(spec, tmp_path)
    obj = ex.store.residency.obj(wire_ref(spec.models["pipeline"]))
    obj._cozy_low_vram_mode = "vae_only"
    monkeypatch.setattr(cc, "apply", _guarded_apply)
    _fake_counters(monkeypatch, hits=2, misses=0)

    _adopt(ex)
    assert len(_events(sent, pb.MODEL_STATE_ADOPTED)) == 1
    assert not _events(sent, pb.MODEL_STATE_FAILED)
    assert obj._cozy_low_vram_mode == "off"


def test_adopt_offload_prep_mode_drift_rejected(tmp_path, monkeypatch):
    """gw#391: an offload prep mode traces genuinely different graphs — a
    pipeline prepped under one can only miss, so adoption still rejects it
    deterministically (key_mismatch) before any wrap or warmup."""
    _artifact(tmp_path, low_vram_mode="off")
    spec = _spec()
    ex, sent = _wire_executor(spec, tmp_path)
    obj = ex.store.residency.obj(wire_ref(spec.models["pipeline"]))
    obj._cozy_low_vram_mode = "model_offload"
    applied: list = []
    monkeypatch.setattr(cc, "apply", lambda *a, **k: applied.append(1) or True)

    _adopt(ex)
    _assert_failed(sent, "adopt_failed:key_mismatch")
    assert not _events(sent, pb.MODEL_STATE_ADOPTED)
    assert not applied  # rejected before the wrap
    assert obj._cozy_low_vram_mode == "model_offload"


def test_adopt_failed_warmup_reports_reason(tmp_path, monkeypatch):
    _artifact(tmp_path)
    spec = _spec()
    ex, sent = _wire_executor(spec, tmp_path)
    monkeypatch.setattr(cc, "apply", _guarded_apply)
    monkeypatch.setattr(_Endpoint, "warmup", lambda self: 1 / 0)

    _adopt(ex)
    _assert_failed(sent, "adopt_failed:warmup")
    assert not _events(sent, pb.MODEL_STATE_ADOPTED)


# ---------------------------------------------------------------------------
# classified failures
# ---------------------------------------------------------------------------


def _case_key_mismatch(tmp_path, monkeypatch):
    _artifact(tmp_path, sku="not-this-gpu")
    ex, sent = _wire_executor(_spec(), tmp_path)
    monkeypatch.setattr(cc, "apply", lambda *a, **k: pytest.fail("must not re-wrap"))
    return ex, sent, {}, None


def _case_model_in_use(tmp_path, monkeypatch):
    _artifact(tmp_path)
    spec = _spec()
    ex, sent = _wire_executor(spec, tmp_path)
    ex.jobs[("r1", 1)] = _Job(request_id="r1", attempt=1, spec=spec)

    calls: list[str] = []

    async def _no_download(ref, snapshot=None, *, binding=None):  # pragma: no cover
        calls.append(ref)
        return tmp_path / "snap"

    ex.store.ensure_local = _no_download  # type: ignore[method-assign]

    def _refused_before_download():
        assert calls == []  # refused before any download

    return ex, sent, {}, _refused_before_download


def _case_family_mismatch(tmp_path, monkeypatch):
    # _adopt's ref names flux2-klein-4b; only sdxl is declared
    spec = _spec(Compile(shapes=((768, 768),), family="sdxl"))
    ex, sent = _wire_executor(spec, tmp_path)
    return ex, sent, {}, None


def _case_target_not_ready(tmp_path, monkeypatch):
    ex, sent = _wire_executor(_spec(), tmp_path, ready=False, resident=False)
    return ex, sent, {}, None


def _case_bad_ref(tmp_path, monkeypatch):
    ex, sent = _wire_executor(_spec(), tmp_path)
    return ex, sent, {"ref": "acme/not-a-cache:latest"}, None


def _case_artifact_missing(tmp_path, monkeypatch):
    (tmp_path / "snap").mkdir()  # empty snapshot dir: no tarball
    ex, sent = _wire_executor(_spec(), tmp_path)
    return ex, sent, {}, None


@pytest.mark.parametrize(
    ("case", "error"),
    [
        (_case_key_mismatch, "adopt_failed:key_mismatch"),
        (_case_model_in_use, "adopt_failed:model_in_use"),
        (_case_family_mismatch, "adopt_failed:target_family_mismatch"),
        (_case_target_not_ready, "adopt_failed:target_not_ready"),
        (_case_bad_ref, "adopt_failed:bad_ref"),
        (_case_artifact_missing, "adopt_failed:artifact_missing"),
    ],
    ids=[
        "key-mismatch-stays-eager", "refuses-while-jobs-in-flight",
        "target-from-another-family", "target-not-ready", "bad-ref",
        "artifact-missing",
    ],
)
def test_adopt_classified_failures(tmp_path, monkeypatch, case, error):
    ex, sent, adopt_kwargs, extra_check = case(tmp_path, monkeypatch)
    _adopt(ex, **adopt_kwargs)
    _assert_failed(sent, error)
    if extra_check is not None:
        extra_check()


@pytest.mark.parametrize(
    ("digest", "operation_id", "error"),
    [
        (None, OP_A, "adopt_failed:missing_snapshot_digest"),
        ("", OP_A, "adopt_failed:missing_snapshot_digest"),
        ("   ", OP_A, "adopt_failed:missing_snapshot_digest"),
        (DIGEST_A, None, "adopt_failed:missing_operation_id"),
        (DIGEST_A, "", "adopt_failed:missing_operation_id"),
        (DIGEST_A, "   ", "adopt_failed:missing_operation_id"),
    ],
)
def test_adopt_missing_identity_fails_before_work_or_resident_inference(
    tmp_path, monkeypatch, digest, operation_id, error,
):
    spec = _spec()
    ex, sent = _wire_executor(spec, tmp_path)
    ex.store._resident_identities[CACHE_REF] = ("resident-digest", 42)

    async def _must_not_adopt(*args, **kwargs):  # pragma: no cover
        pytest.fail("missing digest must fail before adoption work")

    monkeypatch.setattr(ex, "_adopt_compile_cache", _must_not_adopt)
    kwargs = {
        "op": pb.MODEL_OP_KIND_ADOPT_COMPILE_CACHE,
        "ref": CACHE_REF,
    }
    if digest is not None:
        kwargs["snapshot"] = pb.Snapshot(digest=digest)
    if operation_id is not None:
        kwargs["operation_id"] = operation_id
    kwargs["target_incarnation_id"] = _target_id(ex)

    asyncio.run(ex.handle_model_op(pb.ModelOp(**kwargs)))

    failed = _events(sent, pb.MODEL_STATE_FAILED)
    assert len(failed) == 1
    assert failed[0].error == error
    assert failed[0].snapshot_digest == (digest or "")
    assert failed[0].operation_id == (operation_id or "")
    assert failed[0].target_incarnation_id == _target_id(ex)
    assert failed[0].residency_generation == 0
    assert ex.store.resident_identity(CACHE_REF) == ("resident-digest", 42)
    assert not _events(sent, pb.MODEL_STATE_ADOPTED)


def test_adopt_unexpected_failure_echoes_operation_digest(tmp_path, monkeypatch):
    spec = _spec()
    ex, sent = _wire_executor(spec, tmp_path)

    async def _explode(*args, **kwargs):
        raise RuntimeError("unexpected")

    monkeypatch.setattr(ex, "_adopt_compile_cache", _explode)
    _adopt(ex, digest=DIGEST_B)

    failed = _events(sent, pb.MODEL_STATE_FAILED)
    assert len(failed) == 1
    # gw#577: the unexpected-failure event carries the exception text
    assert failed[0].error == "adopt_failed:runtimeerror: unexpected"
    assert failed[0].snapshot_digest == DIGEST_B
    assert failed[0].operation_id == OP_A
    assert failed[0].target_incarnation_id == _target_id(ex)


@pytest.mark.parametrize("digest_b", [DIGEST_A, DIGEST_B])
def test_adopt_serializes_same_ref_ops_across_transport_reconnect(
    tmp_path, monkeypatch, digest_b,
):
    """Old-session A finishes on the new transport before B may mutate."""
    _artifact(tmp_path)
    spec = _spec()
    ex, old_transport = _wire_executor(spec, tmp_path)
    new_transport: list[pb.WorkerMessage] = []
    active_transport = {"sent": old_transport}

    async def _send(msg: pb.WorkerMessage) -> None:
        active_transport["sent"].append(msg)

    ex._send = _send
    monkeypatch.setattr(cc, "apply", _guarded_apply)
    _fake_counters(monkeypatch, hits=1, misses=0)

    async def run():
        a_started = asyncio.Event()
        release_a = asyncio.Event()
        entered = 0

        async def _ensure(ref, snapshot=None, *, binding=None):
            nonlocal entered
            entered += 1
            if entered == 1:
                a_started.set()
                await release_a.wait()
            return tmp_path / "snap"

        ex.store.ensure_local = _ensure  # type: ignore[method-assign]
        op_a = pb.ModelOp(
            op=pb.MODEL_OP_KIND_ADOPT_COMPILE_CACHE,
            ref=CACHE_REF,
            snapshot=pb.Snapshot(digest=DIGEST_A),
            operation_id=OP_A,
            target_incarnation_id=_target_id(ex),
        )
        op_b = pb.ModelOp(
            op=pb.MODEL_OP_KIND_ADOPT_COMPILE_CACHE,
            ref=CACHE_REF,
            snapshot=pb.Snapshot(digest=digest_b),
            operation_id=OP_B,
            target_incarnation_id=_target_id(ex),
        )

        task_a = asyncio.create_task(ex.handle_model_op(op_a))
        await a_started.wait()
        active_transport["sent"] = new_transport
        task_b = asyncio.create_task(ex.handle_model_op(op_b))
        await asyncio.sleep(0)
        assert entered == 1  # B cannot enter download while A owns the lock.
        release_a.set()
        await asyncio.gather(task_a, task_b)
        # Once A proves the target active, an exact replay is acknowledged
        # without work and a different digest requires record reload. Neither
        # may unwrap the active target in place.
        assert entered == 1

    asyncio.run(run())

    assert old_transport == []
    adopted = _events(new_transport, pb.MODEL_STATE_ADOPTED)
    if digest_b == DIGEST_A:
        assert [event.ref for event in adopted] == [CACHE_REF, CACHE_REF]
        assert [event.snapshot_digest for event in adopted] == [DIGEST_A, DIGEST_A]
        assert [event.operation_id for event in adopted] == [OP_A, OP_B]
    else:
        assert [event.ref for event in adopted] == [CACHE_REF]
        failed = _events(new_transport, pb.MODEL_STATE_FAILED)
        assert len(failed) == 1
        assert failed[0].error == "adopt_failed:active_replace_requires_reload"
        assert failed[0].snapshot_digest == DIGEST_B
        assert failed[0].operation_id == OP_B
    assert all(
        event.target_incarnation_id == _target_id(ex)
        for event in adopted + _events(new_transport, pb.MODEL_STATE_FAILED)
    )


def test_ordinary_model_events_never_spoof_an_adoption_operation_id(tmp_path):
    spec = _spec()
    ex, _sent = _wire_executor(spec, tmp_path)
    ordinary = ex.store.model_event(
        MODEL_REF,
        pb.MODEL_STATE_IN_RAM,
        identity=(DIGEST_A, 7),
    )
    assert ordinary.snapshot_digest == DIGEST_A
    assert ordinary.residency_generation == 7
    assert ordinary.operation_id == ""
    assert ordinary.target_incarnation_id == ""


def test_old_worker_semantics_unknown_kind_is_silent(tmp_path):
    """proto3 forward-compat guarantee the hub relies on: an op kind this
    worker doesn't know produces no ModelEvent at all."""
    spec = _spec()
    ex, sent = _wire_executor(spec, tmp_path)
    asyncio.run(ex.handle_model_op(pb.ModelOp(op=99, ref=CACHE_REF)))
    assert not [m for m in sent if m.WhichOneof("msg") == "model_event"]


# ---------------------------------------------------------------------------
# pgw#572 exact compile-target state and execution fencing
# ---------------------------------------------------------------------------


def test_compile_target_state_delta_is_exact_and_ready_only(tmp_path):
    spec = _spec()
    ex, _sent = _wire_executor(spec, tmp_path)
    (target,) = ex.compile_targets()
    rec = ex._classes[spec.instance_key]
    pipe = rec.compile_targets[target.incarnation_id].pipeline

    assert target.family == FAMILY
    assert target.contract_digest == cc.execution_contract_digest(pipe, spec.compile)
    assert list(target.function_names) == ["ep"]
    assert [(b.slot, b.ref, b.snapshot_digest) for b in target.model_bindings] == [
        ("pipeline", wire_ref(spec.models["pipeline"]), MODEL_DIGEST)]
    assert target.active_compile_ref == ""
    assert target.active_compile_snapshot_digest == ""

    lifecycle = Lifecycle(
        SimpleNamespace(worker_jwt="", worker_id="worker"), ex)
    delta = lifecycle._state_delta()
    assert delta.compile_targets == ex.compile_targets()

    rec.ready = False
    assert ex.compile_targets() == []


def test_target_vacate_removes_address_before_replacement(tmp_path):
    spec = _spec()
    ex, _sent = _wire_executor(spec, tmp_path)
    rec = ex._classes[spec.instance_key]
    old_id = _target_id(ex)
    state_snapshots: list[list[str]] = []
    ex._on_state_change = lambda: state_snapshots.append(
        [t.incarnation_id for t in ex.compile_targets()])

    asyncio.run(ex._vacate_record(rec))
    assert ex.compile_targets() == []
    assert state_snapshots and state_snapshots[0] == []

    rec.instance = _Endpoint()
    rec.ready = True
    model_ref = wire_ref(spec.models["pipeline"])
    rec.held_refs = [model_ref]
    rec.held_snapshot_digests = {model_ref: MODEL_DIGEST}
    rec.held_bindings = [("pipeline", model_ref, MODEL_DIGEST)]
    ex._install_compile_targets(rec, spec, [_Pipe()])
    assert _target_id(ex) != old_id


def test_dynamic_sdxl_pick_target_uses_derived_load_time_binding(tmp_path):
    authored = replace(
        _spec(Compile(shapes=((1024, 1024),), family="sdxl")),
        models={"pipeline": Hub("tensorhub/sdxl-default", tag="prod")},
    )
    ex, _sent = _wire_executor(authored, tmp_path, ready=False, resident=False)
    picked_ref = "tensorhub/cyberrealistic-pony:prod"
    picked_digest = "blake3:" + "d" * 64
    derived = replace(
        authored,
        models={"pipeline": Hub("tensorhub/cyberrealistic-pony", tag="prod")},
    )
    rec = ex._class_record(derived)
    rec.instance = _Endpoint()
    rec.ready = True
    rec.held_refs = [picked_ref]
    rec.held_snapshot_digests = {picked_ref: picked_digest}
    rec.held_bindings = [("pipeline", picked_ref, picked_digest)]
    ex._install_compile_targets(rec, derived, [_Pipe()])

    target = next(t for t in ex.compile_targets() if picked_ref in {
        b.ref for b in t.model_bindings})
    assert [(b.slot, b.ref, b.snapshot_digest) for b in target.model_bindings] == [
        ("pipeline", picked_ref, picked_digest)]
    internal = ex._compile_target(target.incarnation_id)
    assert internal is not None and internal[1].spec is derived


def test_shared_instance_target_reports_sorted_function_aliases(tmp_path):
    first = _spec()
    alias = replace(first, name="edit")
    sent = []

    async def _send(msg):
        sent.append(msg)

    ex = Executor([alias, first], _send)
    rec = ex._classes[first.instance_key]
    rec.instance = _Endpoint()
    rec.ready = True
    model_ref = wire_ref(first.models["pipeline"])
    rec.held_refs = [model_ref]
    rec.held_snapshot_digests = {model_ref: MODEL_DIGEST}
    rec.held_bindings = [("pipeline", model_ref, MODEL_DIGEST)]
    ex._install_compile_targets(rec, first, [_Pipe()])
    # A custom object-level warmup cannot prove which sibling handler it
    # covers. Only the setup-initiating handler is honestly addressable.
    assert list(ex.compile_targets()[0].function_names) == ["ep"]


def test_same_family_base_and_lora_targets_remain_distinct(tmp_path):
    class _TurboEndpoint(_Endpoint):
        pass

    base = replace(_spec(), name="base")
    turbo = replace(
        _spec(Compile(
            shapes=((768, 768),), family=FAMILY, lora_bucket=128)),
        name="turbo", cls=_TurboEndpoint, method=_TurboEndpoint.run,
    )

    async def _send(_msg):
        return None

    ex = Executor([base, turbo], _send)
    for spec, lane, digest in (
        (base, "w8a8", MODEL_DIGEST),
        (turbo, "w8a8-lora128", DIGEST_B),
    ):
        rec = ex._classes[spec.instance_key]
        pipe = _Pipe()
        setattr(pipe, "_cozy_weight_lane", lane)
        rec.instance = spec.cls()
        rec.ready = True
        ref = wire_ref(spec.models["pipeline"])
        rec.held_refs = [ref]
        rec.held_snapshot_digests = {ref: digest}
        rec.held_bindings = [("pipeline", ref, digest)]
        selection = type("Selection", (), {
            "ref": CACHE_REF + ("-w8a8-lora128" if spec is turbo else "-w8a8"),
            "snapshot_digest": DIGEST_A,
            "path": tmp_path / ("turbo.tar.gz" if spec is turbo else "base.tar.gz"),
        })()
        _mark_fake_guard(pipe)
        ex._install_compile_targets(
            rec, spec, [pipe], {id(pipe): selection}, {id(pipe): {spec.name}},
        )

    targets = {t.function_names[0]: t for t in ex.compile_targets()}
    assert targets["base"].pipeline_weight_lane == "w8a8"
    assert targets["base"].lora_bucket == 0
    assert targets["turbo"].pipeline_weight_lane == "w8a8-lora128"
    assert targets["turbo"].lora_bucket == 128
    assert targets["base"].incarnation_id != targets["turbo"].incarnation_id


@pytest.mark.parametrize(
    "bindings",
    [
        [("pipeline", MODEL_REF, "")],
        [("pipeline", wire_ref(_spec().models["pipeline"]), MODEL_DIGEST),
         ("pipeline", "acme/other:latest", DIGEST_B)],
    ],
)
def test_malformed_or_duplicate_target_bindings_fail_closed(tmp_path, bindings):
    spec = _spec()
    ex, _sent = _wire_executor(spec, tmp_path, ready=False, resident=False)
    rec = ex._classes[spec.instance_key]
    rec.instance = _Endpoint()
    rec.ready = True
    rec.held_bindings = bindings
    ex._install_compile_targets(rec, spec, [_Pipe()])
    assert ex.compile_targets() == []


def test_unrelated_record_loading_preserves_existing_target(tmp_path):
    class _OtherEndpoint(_Endpoint):
        pass

    first = _spec()
    other = replace(first, name="other", cls=_OtherEndpoint, method=_OtherEndpoint.run)

    async def _send(_msg):
        return None

    ex = Executor([first, other], _send)
    first_rec = ex._classes[first.instance_key]
    first_rec.instance = _Endpoint()
    first_rec.ready = True
    first_rec.held_bindings = [(
        "pipeline", wire_ref(first.models["pipeline"]), MODEL_DIGEST)]
    ex._install_compile_targets(first_rec, first, [_Pipe()])
    first_id = ex.compile_targets()[0].incarnation_id

    other_rec = ex._classes[other.instance_key]
    other_rec.ready = False  # a different record is still loading
    assert [t.incarnation_id for t in ex.compile_targets()] == [first_id]


def test_adopt_rejects_target_replaced_during_download(tmp_path, monkeypatch):
    _artifact(tmp_path)
    spec = _spec()
    ex, sent = _wire_executor(spec, tmp_path)
    target_id = _target_id(ex)

    async def run():
        started = asyncio.Event()
        release = asyncio.Event()

        async def _ensure(ref, snapshot=None, *, binding=None):
            started.set()
            await release.wait()
            return tmp_path / "snap"

        ex.store.ensure_local = _ensure  # type: ignore[method-assign]
        task = asyncio.create_task(ex.handle_model_op(pb.ModelOp(
            op=pb.MODEL_OP_KIND_ADOPT_COMPILE_CACHE,
            ref=CACHE_REF,
            snapshot=pb.Snapshot(digest=DIGEST_A),
            operation_id=OP_A,
            target_incarnation_id=target_id,
        )))
        await started.wait()
        rec = ex._classes[spec.instance_key]
        rec.compile_targets.clear()
        rec.ready = False
        release.set()
        await task

    asyncio.run(run())
    _assert_failed(
        sent, "adopt_failed:target_replaced", target_incarnation_id=target_id)


def test_active_digest_republish_requires_reload_without_unwrap(tmp_path, monkeypatch):
    _artifact(tmp_path)
    spec = _spec()
    ex, sent = _wire_executor(spec, tmp_path)
    monkeypatch.setattr(cc, "apply", _guarded_apply)
    _fake_counters(monkeypatch, hits=2, misses=0)
    _adopt(ex, digest=DIGEST_A)
    target_before = ex.compile_targets()[0]
    assert target_before.active_compile_snapshot_digest == DIGEST_A

    unwraps = []
    monkeypatch.setattr(cc, "unwrap", lambda obj: unwraps.append(obj) or True)
    sent.clear()
    _adopt(ex, digest=DIGEST_B, operation_id=OP_B)
    _assert_failed(
        sent,
        "adopt_failed:active_replace_requires_reload",
        digest=DIGEST_B,
        operation_id=OP_B,
        target_incarnation_id=target_before.incarnation_id,
    )
    target_after = ex.compile_targets()[0]
    assert target_after.active_compile_snapshot_digest == DIGEST_A
    assert unwraps == []


def _cold_spec(binding=None) -> EndpointSpec:
    return EndpointSpec(
        name="cold-generate",
        method=_ColdEndpoint.run,
        kind="inference",
        payload_type=_In,
        output_mode="single",
        cls=_ColdEndpoint,
        attr_name="run",
        models={"pipeline": binding or Hub("acme/klein-finetune")},
        compile=Compile(shapes=((768, 768),), family=FAMILY),
    )


def test_production_setup_stamps_cold_active_identity_after_warmup(
    tmp_path, monkeypatch,
):
    """Real ensure_setup -> fetch -> typed injection -> warmup -> StateDelta."""
    import gen_worker.executor as executor_mod

    artifact = _artifact(tmp_path)
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    spec = _cold_spec()
    model_ref = wire_ref(spec.models["pipeline"])
    sent = []

    async def _send(msg):
        sent.append(msg)

    ex = Executor([spec], _send)
    ex.store._cache_dir = tmp_path / "cas"
    pipe = _LoadablePipe()

    async def _download(ref, **kwargs):
        return artifact.parent if ref == CACHE_REF else model_dir

    monkeypatch.setattr(executor_mod, "ensure_local", _download)
    monkeypatch.setattr(
        provision,
        "load_slot",
        lambda *args, **kwargs: provision.SlotLoad(obj=pipe, is_pipeline=True),
    )
    monkeypatch.setattr(ex, "_enable_compiled", _guarded_enable)
    counter = {"hits": 0}

    def _counters():
        counter["hits"] += 1
        return {"fxgraph_cache_hit": counter["hits"], "fxgraph_cache_miss": 0}

    monkeypatch.setattr(cc, "inductor_counters", _counters)
    _ColdEndpoint.setups = _ColdEndpoint.warmups = 0
    snapshots = {
        model_ref: pb.Snapshot(digest=MODEL_DIGEST),
        CACHE_REF: pb.Snapshot(digest=DIGEST_A),
    }

    instance = asyncio.run(ex.ensure_setup(spec, snapshots))
    assert isinstance(instance, _ColdEndpoint)
    assert _ColdEndpoint.setups == 1 and _ColdEndpoint.warmups == 1
    (target,) = ex.compile_targets()
    assert target.active_compile_ref == CACHE_REF
    assert target.active_compile_snapshot_digest == DIGEST_A
    assert target.model_bindings[0].ref == model_ref
    assert target.model_bindings[0].snapshot_digest == MODEL_DIGEST
    assert target.contract_digest == cc.execution_contract_digest(pipe, spec.compile)


def test_store_served_boot_with_clean_hits_raises_no_compile_alarm(
    tmp_path, monkeypatch,
):
    """gw#587 runtime assertion: a store-served boot (cell delivered, not
    self-minted) that proves clean cache hits must NOT alarm — the whole
    point of a delivered cell is ~0 compile wall time at boot."""
    import gen_worker.executor as executor_mod

    artifact = _artifact(tmp_path)
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    spec = _cold_spec()
    model_ref = wire_ref(spec.models["pipeline"])
    sent = []

    async def _send(msg):
        sent.append(msg)

    ex = Executor([spec], _send)
    ex.store._cache_dir = tmp_path / "cas"
    pipe = _LoadablePipe()

    async def _download(ref, **kwargs):
        return artifact.parent if ref == CACHE_REF else model_dir

    monkeypatch.setattr(executor_mod, "ensure_local", _download)
    monkeypatch.setattr(
        provision,
        "load_slot",
        lambda *args, **kwargs: provision.SlotLoad(obj=pipe, is_pipeline=True),
    )
    monkeypatch.setattr(ex, "_enable_compiled", _guarded_enable)
    counter = {"hits": 0}

    def _counters():
        counter["hits"] += 1
        return {"fxgraph_cache_hit": counter["hits"], "fxgraph_cache_miss": 0}

    monkeypatch.setattr(cc, "inductor_counters", _counters)
    # No real torch.compile runs in this fake-guard rig, so
    # compile_wall_seconds() is naturally ~0 delta across the warmup window —
    # the quiet path is exercised honestly, not forced.
    _ColdEndpoint.setups = _ColdEndpoint.warmups = 0
    snapshots = {
        model_ref: pb.Snapshot(digest=MODEL_DIGEST),
        CACHE_REF: pb.Snapshot(digest=DIGEST_A),
    }

    instance = asyncio.run(ex.ensure_setup(spec, snapshots))
    assert isinstance(instance, _ColdEndpoint)
    adopted = [
        m for m in sent
        if m.HasField("model_event")
        and m.model_event.state == pb.MODEL_STATE_ADOPTED
    ]
    assert adopted == [], "a clean store-served boot must not emit any alarm event"


def test_store_served_boot_with_hidden_compile_fires_alarm(
    tmp_path, monkeypatch, caplog,
):
    """gw#587 runtime assertion, the poisoned/mismatched-cache half: a
    store-served boot proves cache hits (the artifact round-tripped) but the
    process ALSO burns real inductor compile wall time getting there — the
    gw#586 defect class generalized (a cell that claims to serve while the
    boot silently recompiles). Must alarm loudly AND report it hub-side via
    the existing ADOPTED ModelEvent shape."""
    import gen_worker.executor as executor_mod

    artifact = _artifact(tmp_path)
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    spec = _cold_spec()
    model_ref = wire_ref(spec.models["pipeline"])
    sent = []

    async def _send(msg):
        sent.append(msg)

    ex = Executor([spec], _send)
    ex.store._cache_dir = tmp_path / "cas"
    pipe = _LoadablePipe()

    async def _download(ref, **kwargs):
        return artifact.parent if ref == CACHE_REF else model_dir

    monkeypatch.setattr(executor_mod, "ensure_local", _download)
    monkeypatch.setattr(
        provision,
        "load_slot",
        lambda *args, **kwargs: provision.SlotLoad(obj=pipe, is_pipeline=True),
    )
    monkeypatch.setattr(ex, "_enable_compiled", _guarded_enable)
    # before, after — 45s of real inductor compile wall time hidden behind a
    # store-served (delivered, not self-minted) boot.
    wall = iter([0.0, 45.0])
    monkeypatch.setattr(cc, "compile_wall_seconds", lambda: next(wall))
    _ColdEndpoint.setups = _ColdEndpoint.warmups = 0
    snapshots = {
        model_ref: pb.Snapshot(digest=MODEL_DIGEST),
        CACHE_REF: pb.Snapshot(digest=DIGEST_A),
    }

    with caplog.at_level("ERROR", logger="gen_worker.executor"):
        instance = asyncio.run(ex.ensure_setup(spec, snapshots))
    assert isinstance(instance, _ColdEndpoint)
    assert any(
        "STORE_SERVED_BOOT_COMPILED" in r.message for r in caplog.records
    ), "a store-served boot that hid a real compile must alarm loudly"
    adopted = [
        m for m in sent
        if m.HasField("model_event")
        and m.model_event.state == pb.MODEL_STATE_ADOPTED
    ]
    (msg,) = adopted
    assert msg.model_event.ref == CACHE_REF
    assert msg.model_event.snapshot_digest == DIGEST_A
    assert msg.model_event.cache_hits >= 1
    assert msg.model_event.duration_ms == 45000


def test_self_mint_boot_serves_compiled_after_own_warmup_proof(
    tmp_path, monkeypatch, caplog,
):
    """gw#587 serving bootstrap: a mandatory-lane boot with NO delivered cell
    self-mints, runs the SAME warmup proof as a store-served boot (real
    cache-hit accounting on the actual serving graphs), and then ADVERTISES
    its compile target under its own key ref + self-attested digest so the
    hub's self-attested dispatch fence (th#910 PR #488) can dispatch to it.
    The minting boot legitimately burns compile wall time — it must NOT trip
    the STORE_SERVED_BOOT_COMPILED alarm (that line belongs to delivered
    cells only; the store-served side is proven by the sibling tests above)."""
    import gen_worker.executor as executor_mod
    from gen_worker import fleet_cells

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    spec = _cold_spec(Hub("acme/klein-finetune", flavor="fp8-w8a8"))
    model_ref = wire_ref(spec.models["pipeline"])
    mint_key = "ck1-" + "d" * 56
    mint_ref = f"_system/family-{FAMILY}#{mint_key}"
    mint_digest = "blake3:" + "e" * 64
    mint_artifact = tmp_path / "selfmint" / "cell.tar.gz"
    mint_artifact.parent.mkdir()
    mint_artifact.write_bytes(b"cell-bytes")
    sent = []

    async def _send(msg):
        sent.append(msg)

    ex = Executor([spec], _send)
    ex.store._cache_dir = tmp_path / "cas"
    pipe = _LoadablePipe()
    setattr(pipe, "_cozy_weight_lane", "w8a8")

    async def _download(ref, **kwargs):
        return model_dir

    monkeypatch.setattr(executor_mod, "ensure_local", _download)
    monkeypatch.setattr(
        provision,
        "load_slot",
        lambda *args, **kwargs: provision.SlotLoad(obj=pipe, is_pipeline=True),
    )

    def _minting_enable(pipeline, *_args):
        _mark_fake_guard(pipeline)
        return fleet_cells.ArmOutcome(armed=True, self_mint=fleet_cells.SelfMint(
            family=FAMILY, cell_key=mint_key, ref=mint_ref,
            snapshot_digest=mint_digest, artifact=mint_artifact))

    monkeypatch.setattr(ex, "_enable_compiled", _minting_enable)
    # The mint's cold compile happened during setup; simulate real compile
    # wall time visible across the warmup window anyway (regional tails) —
    # a MINTING boot must stay exempt from the store-served alarm.
    wall = iter([0.0, 45.0])
    monkeypatch.setattr(cc, "compile_wall_seconds", lambda: next(wall))
    _ColdEndpoint.setups = _ColdEndpoint.warmups = 0

    with caplog.at_level("ERROR", logger="gen_worker.executor"):
        instance = asyncio.run(ex.ensure_setup(
            spec, {model_ref: pb.Snapshot(digest=MODEL_DIGEST)}))
    assert isinstance(instance, _ColdEndpoint)
    assert _ColdEndpoint.warmups == 1, "the warmup proof must run for a self-mint"
    # Advertised exactly like a delivered cell, under the worker's OWN key.
    (target,) = ex.compile_targets()
    assert target.active_compile_ref == mint_ref
    assert target.active_compile_snapshot_digest == mint_digest
    # No store-served alarm on a minting boot — either loud or on the wire.
    assert not any(
        "STORE_SERVED_BOOT_COMPILED" in r.message for r in caplog.records)
    assert [
        m for m in sent
        if m.HasField("model_event")
        and m.model_event.state == pb.MODEL_STATE_ADOPTED
    ] == []


def test_self_mint_boot_without_warmup_proof_never_reaches_serving(
    tmp_path, monkeypatch,
):
    """Revert-turns-red for the gw#587 serving-bootstrap proof gate: a
    self-minted mandatory-lane cell whose warmup EXERCISES the pipeline but
    proves ZERO cache hits (the mint does not actually serve the serving
    graphs — the gw#586 silent-eager shape) must fail the boot closed
    (CompiledLaneUnavailable), never advertise a target, never serve eager.
    If self-mints are dropped from the warmup proof again (the 0.39.0
    regression this closes), this boot completes and the test goes red."""
    import gen_worker.executor as executor_mod
    from gen_worker import fleet_cells

    model_dir = tmp_path / "model"
    model_dir.mkdir()

    class _NoProofEndpoint(_ColdEndpoint):
        def warmup(self) -> None:
            type(self).warmups += 1
            # Exercised, but every lookup misses: calls>0, hits==0.
            _record_fake_warm(self.pipeline, hits=0, misses=1)

    spec = EndpointSpec(
        name="cold-generate", method=_NoProofEndpoint.run, kind="inference",
        payload_type=_In, output_mode="single", cls=_NoProofEndpoint,
        attr_name="run",
        models={"pipeline": Hub("acme/klein-finetune", flavor="fp8-w8a8")},
        compile=Compile(shapes=((768, 768),), family=FAMILY),
    )
    model_ref = wire_ref(spec.models["pipeline"])
    mint_key = "ck1-" + "f" * 56
    mint_artifact = tmp_path / "selfmint" / "cell.tar.gz"
    mint_artifact.parent.mkdir()
    mint_artifact.write_bytes(b"cell-bytes")

    async def _send(_msg):
        return None

    ex = Executor([spec], _send)
    ex.store._cache_dir = tmp_path / "cas"
    pipe = _LoadablePipe()
    setattr(pipe, "_cozy_weight_lane", "w8a8")

    async def _download(ref, **kwargs):
        return model_dir

    monkeypatch.setattr(executor_mod, "ensure_local", _download)
    monkeypatch.setattr(
        provision,
        "load_slot",
        lambda *args, **kwargs: provision.SlotLoad(obj=pipe, is_pipeline=True),
    )

    def _minting_enable(pipeline, *_args):
        _mark_fake_guard(pipeline)
        return fleet_cells.ArmOutcome(armed=True, self_mint=fleet_cells.SelfMint(
            family=FAMILY, cell_key=mint_key,
            ref=f"_system/family-{FAMILY}#{mint_key}",
            snapshot_digest="blake3:" + "0" * 64, artifact=mint_artifact))

    monkeypatch.setattr(ex, "_enable_compiled", _minting_enable)
    _NoProofEndpoint.setups = _NoProofEndpoint.warmups = _NoProofEndpoint.runs = 0

    with pytest.raises(
        cc.CompiledLaneUnavailableError,
        match="did not serve their own warmup graph",
    ):
        asyncio.run(ex.ensure_setup(
            spec, {model_ref: pb.Snapshot(digest=MODEL_DIGEST)}))
    assert _NoProofEndpoint.warmups == 1, "the proof must have actually run"
    assert ex.compile_targets() == [], "an unproven self-mint must not advertise"
    assert ex.unavailable[spec.name][0] == "compile_cell_failed"


def _pending_mint_rig(tmp_path, monkeypatch, *, pipe, publisher):
    """Wire the REAL fleet_cells miss path (prove-produces-the-mint) into an
    Executor boot: delivered arm refuses (mandatory miss), the cold-capture
    arm opens a real capture dir, and the executor's own warmup proof is the
    only thing that can finalize/publish it."""
    from gen_worker import cell_key as cell_key_mod
    from gen_worker import fleet_cells

    with fleet_cells._PENDING_LOCK:
        fleet_cells._PENDING.clear()

    def _mandatory_miss(*a, **k):
        raise cc.CompiledLaneUnavailableError("no delivered cell")

    monkeypatch.setattr(provision, "enable_compiled", _mandatory_miss)
    monkeypatch.setattr(fleet_cells, "_cuda_ready", lambda: True)
    monkeypatch.setattr(cc, "toolchain_present", lambda: True)

    class _Key:
        digest = "ck1-" + "9" * 56

    monkeypatch.setattr(cell_key_mod, "compute", lambda *a, **k: _Key())
    captured: dict = {}

    def _begin(p, cfg, capture):
        capture.mkdir(parents=True, exist_ok=True)
        captured["dir"] = capture
        _mark_fake_guard(p)

    monkeypatch.setattr(cc, "begin_fleet_mint", _begin)
    return captured, _Key.digest


def test_pending_self_mint_boot_packs_and_publishes_only_the_proven_capture(
    tmp_path, monkeypatch, caplog,
):
    """gw#587 CORRECT FIX direction (a), over the real Executor rig: a
    mandatory-lane miss arms a COLD capture; the endpoint's own warmup —
    the same execution the proof observes — produces the inductor output;
    only after the proof certifies a successful compiled call does the boot
    pack THAT capture, advertise its real digest, and publish those exact
    bytes. Reverting finalize back into the arm path (mint-then-prove)
    turns this red: the publish would happen before the warmup."""
    import gen_worker.executor as executor_mod
    from gen_worker import fleet_cells

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    events: list = []
    published: dict = {}

    class _Pub(fleet_cells.CellPublisher):
        def publish(self, family, artifact, meta):
            events.append("publish")
            published["bytes"] = Path(artifact).read_bytes()
            published["meta"] = dict(meta)
            published["family"] = family
            return "cp-1"

    pub = _Pub(base_url="http://hub", worker_jwt=lambda: "jwt",
               image_digest="sha256:img")
    pipe = _LoadablePipe()
    setattr(pipe, "_cozy_weight_lane", "w8a8")
    captured, mint_key = _pending_mint_rig(
        tmp_path, monkeypatch, pipe=pipe, publisher=pub)

    class _MintingEndpoint(_ColdEndpoint):
        def warmup(self) -> None:
            type(self).warmups += 1
            events.append("warmup")
            # THE real serving execution: cold-compiles the serving graphs
            # into the live capture dir (successful compiled call, all
            # misses — the honest cold-mint signature).
            cap = captured["dir"]
            (cap / "inductor" / "g").mkdir(parents=True, exist_ok=True)
            (cap / "inductor" / "g" / "serving_graph.py").write_text("real")
            (cap / "triton").mkdir(exist_ok=True)
            _record_fake_warm(self.pipeline, hits=0, misses=8)

    spec = EndpointSpec(
        name="cold-generate", method=_MintingEndpoint.run, kind="inference",
        payload_type=_In, output_mode="single", cls=_MintingEndpoint,
        attr_name="run",
        models={"pipeline": Hub("acme/klein-finetune", flavor="fp8-w8a8")},
        compile=Compile(shapes=((768, 768),), family=FAMILY),
    )
    model_ref = wire_ref(spec.models["pipeline"])

    async def _send(_msg):
        return None

    ex = Executor([spec], _send)
    ex.store._cache_dir = tmp_path / "cas"

    async def _download(ref, **kwargs):
        return model_dir

    monkeypatch.setattr(executor_mod, "ensure_local", _download)
    monkeypatch.setattr(
        provision,
        "load_slot",
        lambda *args, **kwargs: provision.SlotLoad(obj=pipe, is_pipeline=True),
    )
    monkeypatch.setattr(
        ex, "_enable_compiled",
        lambda p, cfg, artifact: fleet_cells.enable_compiled(
            p, cfg, ex.store._cache_dir, artifact, publisher=pub))
    _MintingEndpoint.setups = _MintingEndpoint.warmups = 0

    with caplog.at_level("ERROR", logger="gen_worker.executor"):
        instance = asyncio.run(ex.ensure_setup(
            spec, {model_ref: pb.Snapshot(digest=MODEL_DIGEST)}))
    assert isinstance(instance, _MintingEndpoint)
    assert _MintingEndpoint.warmups == 1

    # publish is async; join the publisher thread via the recorded event
    for _ in range(100):
        if "publish" in events:
            break
        import time as _time

        _time.sleep(0.05)
    assert events.index("warmup") < events.index("publish"), (
        "prove-produces-the-mint: publish must come from the proven capture, "
        "never precede the warmup proof")

    (target,) = ex.compile_targets()
    assert target.active_compile_ref == f"_system/family-{FAMILY}#{mint_key}"
    digest = target.active_compile_snapshot_digest
    assert digest.startswith("blake3:")
    # The advertised digest is the digest of exactly the published bytes,
    # and those bytes contain the graphs the WARMUP (not any producer warm
    # loop) compiled.
    from gen_worker.convert.hub import blake3_file

    copy = tmp_path / "published-copy.tar.gz"
    copy.write_bytes(published["bytes"])
    assert digest == "blake3:" + blake3_file(copy)
    import io
    import tarfile

    with tarfile.open(fileobj=io.BytesIO(published["bytes"]), mode="r:*") as tar:
        names = tar.getnames()
    assert any("serving_graph.py" in n for n in names)
    assert published["meta"].get("source_ref") == "self-mint"
    # A minting boot legitimately compiles: no store-served alarm.
    assert not any(
        "STORE_SERVED_BOOT_COMPILED" in r.message for r in caplog.records)


def test_pending_self_mint_unproven_fails_closed_and_never_publishes(
    tmp_path, monkeypatch,
):
    """gw#587 CORRECT FIX direction (b), revert-turns-red: a mandatory-lane
    self-mint capture whose warmup never certifies a successful compiled
    call (the executor ran the warmup, but the compiled targets did not
    serve it — the gw#586 silent-eager shape) must fail the boot closed,
    advertise nothing, publish NOTHING, and abandon the capture. If packing
    or publishing ever moves ahead of the proof again, the publish below
    fires and this test goes red."""
    import gen_worker.executor as executor_mod
    from gen_worker import fleet_cells

    model_dir = tmp_path / "model"
    model_dir.mkdir()

    class _Pub(fleet_cells.CellPublisher):
        def publish(self, family, artifact, meta):
            pytest.fail("an unproven self-mint must NEVER publish")

    pub = _Pub(base_url="http://hub", worker_jwt=lambda: "jwt",
               image_digest="sha256:img")
    pipe = _LoadablePipe()
    setattr(pipe, "_cozy_weight_lane", "w8a8")
    captured, _mint_key = _pending_mint_rig(
        tmp_path, monkeypatch, pipe=pipe, publisher=pub)

    class _UnprovenEndpoint(_ColdEndpoint):
        def warmup(self) -> None:
            type(self).warmups += 1
            # Warmup runs, but no successful compiled call is ever recorded
            # on the pipe (calls=0): the capture certifies nothing.

    spec = EndpointSpec(
        name="cold-generate", method=_UnprovenEndpoint.run, kind="inference",
        payload_type=_In, output_mode="single", cls=_UnprovenEndpoint,
        attr_name="run",
        models={"pipeline": Hub("acme/klein-finetune", flavor="fp8-w8a8")},
        compile=Compile(shapes=((768, 768),), family=FAMILY),
    )
    model_ref = wire_ref(spec.models["pipeline"])

    async def _send(_msg):
        return None

    ex = Executor([spec], _send)
    ex.store._cache_dir = tmp_path / "cas"

    async def _download(ref, **kwargs):
        return model_dir

    monkeypatch.setattr(executor_mod, "ensure_local", _download)
    monkeypatch.setattr(
        provision,
        "load_slot",
        lambda *args, **kwargs: provision.SlotLoad(obj=pipe, is_pipeline=True),
    )
    monkeypatch.setattr(
        ex, "_enable_compiled",
        lambda p, cfg, artifact: fleet_cells.enable_compiled(
            p, cfg, ex.store._cache_dir, artifact, publisher=pub))
    _UnprovenEndpoint.setups = _UnprovenEndpoint.warmups = 0

    with pytest.raises(
        cc.CompiledLaneUnavailableError,
        match="did not serve their own warmup graph",
    ):
        asyncio.run(ex.ensure_setup(
            spec, {model_ref: pb.Snapshot(digest=MODEL_DIGEST)}))
    assert _UnprovenEndpoint.warmups == 1, "the proof must have actually run"
    assert ex.compile_targets() == [], "an unproven self-mint must not advertise"
    assert ex.unavailable[spec.name][0] == "compile_cell_failed"
    # The capture was abandoned, never packed.
    mint_root = captured["dir"].parent
    assert not mint_root.exists(), "an uncertified capture must be discarded"
    with fleet_cells._PENDING_LOCK:
        assert fleet_cells._PENDING == {}


def test_boot_warmup_proves_each_compile_object_independently(
    tmp_path, monkeypatch,
):
    """A hit from pipeline A must never certify its unexecuted sibling B."""
    import gen_worker.executor as executor_mod

    artifact = _artifact(tmp_path)
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    first_ref = "acme/first"
    second_ref = "acme/second"

    class _DualEndpoint:
        def setup(self, first: _LoadablePipe, second: _LoadablePipe) -> None:
            self.first = first
            self.second = second

        def warmup(self) -> None:
            _record_fake_warm(self.first)

        def run(self, ctx, payload: _In) -> _Out:  # pragma: no cover
            return _Out()

    spec = EndpointSpec(
        name="dual", method=_DualEndpoint.run, kind="inference",
        payload_type=_In, output_mode="single", cls=_DualEndpoint,
        attr_name="run",
        models={"first": Hub(first_ref), "second": Hub(second_ref)},
        compile=Compile(shapes=((768, 768),), family=FAMILY),
    )

    async def _send(_msg):
        return None

    ex = Executor([spec], _send)
    ex.store._cache_dir = tmp_path / "cas"
    pipes = {"first": _LoadablePipe(), "second": _LoadablePipe()}

    async def _download(ref, **kwargs):
        return artifact.parent if ref == CACHE_REF else model_dir

    monkeypatch.setattr(executor_mod, "ensure_local", _download)
    monkeypatch.setattr(
        provision,
        "load_slot",
        lambda *args, **kwargs: provision.SlotLoad(
            obj=pipes[kwargs["slot"]], is_pipeline=True),
    )
    monkeypatch.setattr(ex, "_enable_compiled", _guarded_enable)
    instance = asyncio.run(ex.ensure_setup(spec, {
        first_ref: pb.Snapshot(digest=MODEL_DIGEST),
        second_ref: pb.Snapshot(digest=DIGEST_B),
        CACHE_REF: pb.Snapshot(digest=DIGEST_A),
    }))
    assert isinstance(instance, _DualEndpoint)
    targets = {
        target.model_bindings[0].slot: target for target in ex.compile_targets()
    }
    # The custom warmup proves only the first object. The untouched sibling is
    # omitted rather than advertised as an adoptable target it can never prove.
    assert set(targets) == {"first"}
    assert targets["first"].active_compile_ref == CACHE_REF
    assert targets["first"].active_compile_snapshot_digest == DIGEST_A
    assert targets["first"].model_bindings[0].ref == first_ref


def test_sdxl_w8a8_boot_advertises_only_warmup_proven_generate_alias(
    tmp_path, monkeypatch,
):
    """SDXL's legacy Turbo handler rejects W8A8 and is explicitly skipped.

    The ordinary generate warmup still proves the attached cell and reaches
    READY, but its cache hit must not certify the incompatible sibling alias.
    """
    import gen_worker.executor as executor_mod

    family = "sdxl"
    cell_ref = f"_system/family-{family}#inductor-rtx-4090-torch2.9-w8a8"
    artifact = _artifact(tmp_path, family=family)
    model_dir = tmp_path / "sdxl-model"
    model_dir.mkdir()
    calls = {"generate": 0, "generate_turbo": 0}

    @endpoint(
        models={"pipeline": Hub("acme/sdxl", flavor="fp8-w8a8")},
        resources=Resources(vram_gb=24),
        compile=Compile(shapes=((1024, 1024),), family=family),
        warmup={"generate": {"prompt": "warmup"}, "generate_turbo": None},
    )
    class _SdxlEndpoint:
        def setup(self, pipeline: _LoadablePipe) -> None:
            self.pipeline = pipeline

        def generate(self, ctx, payload: _In) -> _Out:
            calls["generate"] += 1
            _record_fake_warm(self.pipeline)
            return _Out(y="ok")

        def generate_turbo(self, ctx, payload: _In) -> _Out:
            calls["generate_turbo"] += 1
            raise AssertionError("legacy Turbo is incompatible with W8A8")

    specs = extract_specs(_SdxlEndpoint)
    generate = next(spec for spec in specs if spec.name == "generate")
    model_ref = wire_ref(generate.models["pipeline"])
    pipe = _LoadablePipe()
    setattr(pipe, "_cozy_weight_lane", "w8a8")

    async def _send(_msg):
        return None

    ex = Executor(specs, _send)
    ex.store._cache_dir = tmp_path / "cas"

    async def _download(ref, **kwargs):
        return artifact.parent if ref == cell_ref else model_dir

    monkeypatch.setattr(executor_mod, "ensure_local", _download)
    monkeypatch.setattr(
        provision,
        "load_slot",
        lambda *args, **kwargs: provision.SlotLoad(obj=pipe, is_pipeline=True),
    )
    monkeypatch.setattr(ex, "_enable_compiled", _guarded_enable)

    asyncio.run(ex.ensure_setup(generate, {
        model_ref: pb.Snapshot(digest=MODEL_DIGEST),
        cell_ref: pb.Snapshot(digest=DIGEST_A),
    }))

    assert calls == {"generate": 1, "generate_turbo": 0}
    (target,) = ex.compile_targets()
    assert list(target.function_names) == ["generate"]
    assert target.active_compile_ref == cell_ref
    assert target.active_compile_snapshot_digest == DIGEST_A


def test_flux_base_w8a8_boot_proves_generate_and_edit_aliases(
    tmp_path, monkeypatch,
):
    """Both aliases recover coherently after one target guard failure."""
    import gen_worker.executor as executor_mod

    cell_ref = CACHE_REF + "-w8a8"
    artifact = _artifact(tmp_path)
    model_dir = tmp_path / "flux-model"
    model_dir.mkdir()
    calls = {"generate": 0, "edit": 0}

    @endpoint(
        models={"pipeline": Hub("acme/flux-base", flavor="fp8-w8a8")},
        resources=Resources(vram_gb=24),
        compile=Compile(shapes=((768, 768),), family=FAMILY),
        warmup={
            "generate": {"prompt": "warmup"},
            "edit": {"prompt": "warmup"},
        },
    )
    class _FluxBaseEndpoint:
        def setup(self, pipeline: _LoadablePipe) -> None:
            self.pipeline = pipeline

        def generate(self, ctx, payload: _In) -> _Out:
            calls["generate"] += 1
            _record_fake_warm(self.pipeline)
            return _Out(y="ok")

        def edit(self, ctx, payload: _In) -> _Out:
            calls["edit"] += 1
            # Reusing an already-loaded graph is a successful wrapper
            # execution, but Inductor records no second lookup/cache hit.
            _record_fake_warm(self.pipeline)
            return _Out(y="ok")

    specs = extract_specs(_FluxBaseEndpoint)
    generate = next(spec for spec in specs if spec.name == "generate")
    model_ref = wire_ref(generate.models["pipeline"])
    pipes = [_LoadablePipe(), _LoadablePipe()]
    for pipe in pipes:
        setattr(pipe, "_cozy_weight_lane", "w8a8")
    remaining_pipes = iter(pipes)
    sent: list[pb.WorkerMessage] = []

    async def _send(msg):
        sent.append(msg)

    ex = Executor(specs, _send)
    ex.store._cache_dir = tmp_path / "cas"

    async def _download(ref, **kwargs):
        return artifact.parent if ref == cell_ref else model_dir

    monkeypatch.setattr(executor_mod, "ensure_local", _download)
    monkeypatch.setattr(
        provision,
        "load_slot",
        lambda *args, **kwargs: provision.SlotLoad(
            obj=next(remaining_pipes), is_pipeline=True),
    )
    monkeypatch.setattr(ex, "_enable_compiled", _guarded_enable)

    snapshots = {
        model_ref: pb.Snapshot(digest=MODEL_DIGEST),
        cell_ref: pb.Snapshot(digest=DIGEST_A),
    }
    asyncio.run(ex.ensure_setup(generate, snapshots))

    assert calls == {"generate": 1, "edit": 1}
    (target,) = ex.compile_targets()
    assert list(target.function_names) == ["edit", "generate"]
    assert target.active_compile_ref == cell_ref

    # Replay binds a real operation identity to the boot-attached active cell,
    # then a runtime guard failure disables every alias on the exact target.
    _adopt(
        ex,
        ref=cell_ref,
        digest=DIGEST_A,
        operation_id=OP_A,
        target_incarnation_id=target.incarnation_id,
    )
    found = ex._compile_target(target.incarnation_id)
    assert found is not None
    rec, internal = found
    ex.unavailable["unrelated-hardware-gate"] = (
        "hardware_unmet", "another record owns this", {"gpu": "too_small"},
    )
    signal = getattr(internal.pipeline, cc._MARKER_ATTR)["failure_signal"]
    callback = signal["callback"]
    assert callable(callback)

    async def _trip() -> None:
        ex._loop = asyncio.get_running_loop()
        await asyncio.to_thread(callback, "compiled graph exploded")
        await asyncio.sleep(0)
        await asyncio.sleep(0)

    asyncio.run(_trip())
    assert rec.stale is True
    assert set(ex.unavailable) >= {"edit", "generate"}
    assert all(
        ex.unavailable[name][0] == "compile_cell_failed"
        for name in ("edit", "generate")
    )
    failed = _events(sent, pb.MODEL_STATE_FAILED)
    assert len(failed) == 1
    assert failed[0].error == "adopt_failed:runtime_guard"
    assert failed[0].operation_id == OP_A

    lifecycle = Lifecycle(
        SimpleNamespace(worker_jwt="", worker_id="worker"), ex)
    failed_delta = lifecycle._state_delta()
    assert "edit" not in failed_delta.available_functions
    assert "generate" not in failed_delta.available_functions

    # Declarative reconciliation replaces the stale incarnation. Only after
    # the new exact active target proves both aliases are its owned marks
    # cleared; unrelated unavailable reasons would remain untouched.
    asyncio.run(ex.ensure_setup(generate, snapshots))
    (recovered,) = ex.compile_targets()
    assert recovered.incarnation_id != target.incarnation_id
    assert recovered.active_compile_ref == cell_ref
    assert list(recovered.function_names) == ["edit", "generate"]
    assert calls == {"generate": 2, "edit": 2}
    assert "edit" not in ex.unavailable
    assert "generate" not in ex.unavailable
    assert ex.unavailable["unrelated-hardware-gate"][0] == "hardware_unmet"
    recovered_delta = lifecycle._state_delta()
    assert {"edit", "generate"}.issubset(recovered_delta.available_functions)


@pytest.mark.parametrize(
    (
        "case", "edit_uses_wrapper", "counter_hits", "expected_names",
        "expected_executions", "expected_hits",
    ),
    (
        ("loaded_graph_reuse", True, (10, 11), ("edit", "generate"), 2, 1),
        ("no_object_hit", True, (10, 10, 10, 10), (), 0, 0),
        ("alias_bypasses_wrapper", False, (10, 11), ("generate",), 1, 1),
    ),
)
def test_flux_real_guard_requires_object_activation_and_each_alias_execution(
    tmp_path,
    monkeypatch,
    case,
    edit_uses_wrapper,
    counter_hits,
    expected_names,
    expected_executions,
    expected_hits,
):
    """One object hit plus one exact wrapper call per alias is causal proof."""
    import gen_worker.executor as executor_mod
    import torch

    artifact = _artifact(tmp_path)
    model_dir = tmp_path / "real-guard-model"
    model_dir.mkdir()
    calls = {"generate": 0, "edit": 0}
    compiled_ready = threading.Event()

    @endpoint(
        models={"pipeline": Hub("acme/flux-base")},
        resources=Resources(vram_gb=24),
        compile=Compile(shapes=((768, 768),), family=FAMILY),
        warmup={
            "generate": {"prompt": "warmup"},
            "edit": {"prompt": "warmup"},
        },
    )
    class _FluxBaseEndpoint:
        def setup(self, pipeline: _LoadablePipe) -> None:
            self.pipeline = pipeline

        def generate(self, ctx, payload: _In) -> _Out:
            calls["generate"] += 1
            self.pipeline.transformer.forward(payload.prompt)
            return _Out(y="ok")

        def edit(self, ctx, payload: _In) -> _Out:
            calls["edit"] += 1
            if edit_uses_wrapper:
                self.pipeline.transformer.forward(payload.prompt)
            return _Out(y="ok")

    specs = extract_specs(_FluxBaseEndpoint)
    generate = next(spec for spec in specs if spec.name == "generate")
    model_ref = wire_ref(generate.models["pipeline"])
    pipe = _Pipe()

    async def _send(_msg):
        return None

    ex = Executor(specs, _send, gpu_slots=2)
    ex.store._cache_dir = tmp_path / "cas"

    async def _download(ref, **kwargs):
        return artifact.parent if ref == CACHE_REF else model_dir

    def _compile(fn, **kwargs):
        compiled_ready.set()
        return fn

    counters = iter(counter_hits)
    counter_reads = []

    def _counters():
        counter_reads.append(case)
        return {
            "fxgraph_cache_hit": next(counters),
            "fxgraph_cache_miss": 1,
        }

    monkeypatch.setattr(executor_mod, "ensure_local", _download)
    monkeypatch.setattr(
        provision,
        "load_slot",
        lambda *args, **kwargs: provision.SlotLoad(obj=pipe, is_pipeline=True),
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch, "compile", _compile)
    monkeypatch.setattr(cc, "inductor_counters", _counters)

    async def scenario() -> None:
        # Hold one of two permits. Setup may stage/arm, but its proof warmup
        # must wait until it can hold the entire worker GPU execution surface.
        await ex._gpu_semaphore.acquire()
        task = asyncio.create_task(ex.ensure_setup(generate, {
            model_ref: pb.Snapshot(digest=MODEL_DIGEST),
            CACHE_REF: pb.Snapshot(digest=DIGEST_A),
        }))
        try:
            assert await asyncio.to_thread(compiled_ready.wait, 10)
            for _ in range(3):
                await asyncio.sleep(0)
            assert calls == {"generate": 0, "edit": 0}
        finally:
            ex._gpu_semaphore.release()
        await task

    asyncio.run(scenario())

    assert calls == {"generate": 1, "edit": 1}
    assert len(counter_reads) == len(counter_hits)
    targets = ex.compile_targets()
    if expected_names:
        (target,) = targets
        assert tuple(target.function_names) == expected_names
        assert target.active_compile_ref == CACHE_REF
        assert cc.execution_count(pipe) == expected_executions
        assert cc.cache_hit_count(pipe) == expected_hits
    else:
        assert targets == []
        assert getattr(pipe, cc._MARKER_ATTR, None) is None


def test_compile_hit_on_other_object_cannot_certify_primary_object(
    tmp_path, monkeypatch,
):
    """Process-wide hit deltas remain owned by the wrapper that observed them."""
    import gen_worker.executor as executor_mod
    import torch

    artifact = _artifact(tmp_path)
    model_dir = tmp_path / "two-object-models"
    model_dir.mkdir()

    @endpoint(
        models={
            "primary": Hub("acme/flux-primary"),
            "other": Hub("acme/flux-other"),
        },
        resources=Resources(vram_gb=24),
        compile=Compile(shapes=((768, 768),), family=FAMILY),
        warmup={"generate": {"prompt": "warmup"}},
    )
    class _TwoObjectEndpoint:
        def setup(
            self, primary: _LoadablePipe, other: _LoadablePipe,
        ) -> None:
            self.primary = primary
            self.other = other

        def generate(self, ctx, payload: _In) -> _Out:
            # The primary wrapper executes but sees no cache activation. The
            # other wrapper then sees the sole process-wide hit.
            self.primary.transformer.forward(payload.prompt)
            self.other.transformer.forward(payload.prompt)
            return _Out(y="ok")

    (spec,) = extract_specs(_TwoObjectEndpoint)
    refs = {slot: wire_ref(binding) for slot, binding in spec.models.items()}
    pipes = {"primary": _Pipe(), "other": _Pipe()}

    async def _send(_msg):
        return None

    ex = Executor([spec], _send)
    ex.store._cache_dir = tmp_path / "cas"

    async def _download(ref, **kwargs):
        return artifact.parent if ref == CACHE_REF else model_dir

    counter_hits = iter((10, 10, 10, 11))
    counter_reads = 0

    def _counters():
        nonlocal counter_reads
        counter_reads += 1
        return {
            "fxgraph_cache_hit": next(counter_hits),
            "fxgraph_cache_miss": 1,
        }

    monkeypatch.setattr(executor_mod, "ensure_local", _download)
    monkeypatch.setattr(
        provision,
        "load_slot",
        lambda *args, **kwargs: provision.SlotLoad(
            obj=pipes[kwargs["slot"]], is_pipeline=True,
        ),
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch, "compile", lambda fn, **kwargs: fn)
    monkeypatch.setattr(cc, "inductor_counters", _counters)

    asyncio.run(ex.ensure_setup(spec, {
        refs["primary"]: pb.Snapshot(digest=MODEL_DIGEST),
        refs["other"]: pb.Snapshot(digest=DIGEST_B),
        CACHE_REF: pb.Snapshot(digest=DIGEST_A),
    }))

    assert counter_reads == 4
    (target,) = ex.compile_targets()
    assert [(binding.slot, binding.ref) for binding in target.model_bindings] == [
        ("other", refs["other"]),
    ]
    assert list(target.function_names) == [spec.name]
    assert target.active_compile_ref == CACHE_REF
    assert getattr(pipes["primary"], cc._MARKER_ATTR, None) is None
    assert cc.execution_count(pipes["other"]) == 1
    assert cc.cache_hit_count(pipes["other"]) == 1


def test_pipeline_target_owns_only_pipeline_not_ancillary_vae(
    tmp_path, monkeypatch,
):
    """Production-shaped SDXL: ancillary bindings cannot certify the graph."""
    import gen_worker.executor as executor_mod

    artifact = _artifact(tmp_path)
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    class _PipelineWithVaeEndpoint:
        def setup(self, pipeline: _LoadablePipe, vae: _AncillaryVae) -> None:
            self.pipeline = pipeline
            self.vae = vae

        def warmup(self) -> None:
            _record_fake_warm(self.pipeline)

        def run(self, ctx, payload: _In) -> _Out:  # pragma: no cover
            return _Out()

    spec = EndpointSpec(
        name="sdxl-with-vae",
        method=_PipelineWithVaeEndpoint.run,
        kind="inference",
        payload_type=_In,
        output_mode="single",
        cls=_PipelineWithVaeEndpoint,
        attr_name="run",
        models={
            "pipeline": Hub("acme/sdxl", flavor="fp8-w8a8"),
            "vae": Hub("acme/sdxl-vae"),
        },
        compile=Compile(shapes=((1024, 1024),), family=FAMILY),
    )
    pipeline_ref = wire_ref(spec.models["pipeline"])
    vae_ref = wire_ref(spec.models["vae"])
    cell_ref = CACHE_REF + "-w8a8"
    pipe = _LoadablePipe()
    setattr(pipe, "_cozy_weight_lane", "w8a8")
    vae = _AncillaryVae()

    async def _send(_msg):
        return None

    ex = Executor([spec], _send)
    ex.store._cache_dir = tmp_path / "cas"

    async def _download(ref, **kwargs):
        return artifact.parent if ref == cell_ref else model_dir

    monkeypatch.setattr(executor_mod, "ensure_local", _download)
    monkeypatch.setattr(
        provision,
        "load_slot",
        lambda *args, **kwargs: provision.SlotLoad(
            obj=pipe if kwargs["slot"] == "pipeline" else vae,
            is_pipeline=kwargs["slot"] == "pipeline",
        ),
    )
    monkeypatch.setattr(ex, "_enable_compiled", _guarded_enable)
    asyncio.run(ex.ensure_setup(spec, {
        pipeline_ref: pb.Snapshot(digest=MODEL_DIGEST),
        vae_ref: pb.Snapshot(digest=DIGEST_B),
        cell_ref: pb.Snapshot(digest=DIGEST_A),
    }))
    (target,) = ex.compile_targets()
    assert [binding.slot for binding in target.model_bindings] == ["pipeline"]

    def run_with_snapshots(snapshots) -> pb.RunJob:
        return pb.RunJob(
            function_name=spec.name,
            models=[
                pb.ModelBinding(slot="pipeline", ref=pipeline_ref),
                pb.ModelBinding(slot="vae", ref=vae_ref),
            ],
            snapshots=snapshots,
            required_compile=pb.RequiredCompileExecution(
                target_incarnation_id=target.incarnation_id,
                cell_ref=target.active_compile_ref,
                cell_snapshot_digest=target.active_compile_snapshot_digest,
                contract_digest=target.contract_digest,
            ),
        )

    # Exact pipeline evidence accepts. VAE identity remains an independent
    # setup/residency concern and cannot replace or broaden the target proof.
    ex._validate_required_compile(spec, run_with_snapshots({
        pipeline_ref: pb.Snapshot(digest=MODEL_DIGEST),
        vae_ref: pb.Snapshot(digest=DIGEST_B),
    }))
    ex._validate_required_compile(spec, run_with_snapshots({
        pipeline_ref: pb.Snapshot(digest=MODEL_DIGEST),
        vae_ref: pb.Snapshot(digest="blake3:" + "e" * 64),
    }))
    with pytest.raises(RetryableError, match="required_compile_binding_missing"):
        ex._validate_required_compile(spec, run_with_snapshots({
            vae_ref: pb.Snapshot(digest=MODEL_DIGEST),
        }))
    with pytest.raises(RetryableError, match="required_compile_binding_mismatch"):
        ex._validate_required_compile(spec, run_with_snapshots({
            pipeline_ref: pb.Snapshot(digest=DIGEST_B),
            vae_ref: pb.Snapshot(digest=MODEL_DIGEST),
        }))


def test_w8a8_without_exact_cell_self_mints_and_fails_typed_without_cuda(
    tmp_path, monkeypatch,
):
    """gw#587: a mandatory-lane miss no longer fail-closes before load — the
    worker proceeds to load and SELF-MINTS its own cell. In a CUDA-less test
    env the mint is impossible, so the quantized lane's typed refusal fires
    from the self-mint exit (never a silent eager serve), and the function
    still lands in the same compile_cell_failed unavailable class."""
    import gen_worker.executor as executor_mod

    spec = _cold_spec(Hub("acme/klein-finetune", flavor="fp8-w8a8"))
    model_ref = wire_ref(spec.models["pipeline"])

    async def _send(_msg):
        return None

    ex = Executor([spec], _send)
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    loads = []
    pipe = _Pipe()
    setattr(pipe, "_cozy_weight_lane", "w8a8")

    async def _download(ref, **kwargs):
        return model_dir

    monkeypatch.setattr(executor_mod, "ensure_local", _download)
    monkeypatch.setattr(
        provision,
        "load_slot",
        lambda *args, **kwargs: loads.append(1)
        or provision.SlotLoad(obj=pipe, is_pipeline=True),
    )
    _ColdEndpoint.setups = _ColdEndpoint.warmups = _ColdEndpoint.runs = 0

    with pytest.raises(cc.CompiledLaneUnavailableError, match="self-mint is unavailable"):
        asyncio.run(ex.ensure_setup(
            spec, {model_ref: pb.Snapshot(digest=MODEL_DIGEST)}))
    # The load is the mint's precondition now (the boot warmup IS the mint).
    assert loads == [1]
    assert _ColdEndpoint.runs == 0
    assert ex.unavailable[spec.name][0] == "compile_cell_failed"


def test_w8a8_partial_handler_proof_fails_loud_without_disabling_skipped_turbo(
    tmp_path, monkeypatch,
):
    """Every required alias must prove the W8A8 cell; explicit skips do not."""
    import gen_worker.executor as executor_mod

    family = "sdxl"
    cell_ref = f"_system/family-{family}#inductor-rtx-4090-torch2.9-w8a8"
    artifact = _artifact(tmp_path, family=family)
    model_dir = tmp_path / "partial-proof-model"
    model_dir.mkdir()

    @endpoint(
        models={"pipeline": Hub("acme/sdxl", flavor="fp8-w8a8")},
        resources=Resources(vram_gb=24),
        compile=Compile(shapes=((1024, 1024),), family=family),
        warmup={
            "generate": {"prompt": "warmup"},
            "edit": {"prompt": "warmup"},
            "generate_turbo": None,
        },
    )
    class _SdxlEndpoint:
        def setup(self, pipeline: _LoadablePipe) -> None:
            self.pipeline = pipeline

        def warmup(self) -> None:
            _record_fake_warm(self.pipeline)

        def generate(self, ctx, payload: _In) -> _Out:
            return _Out(y="ok")

        def edit(self, ctx, payload: _In) -> _Out:
            return _Out(y="eager")

        def generate_turbo(self, ctx, payload: _In) -> _Out:
            raise AssertionError("explicitly skipped Turbo must not run")

    specs = extract_specs(_SdxlEndpoint)
    by_attr = {spec.attr_name: spec for spec in specs}
    generate = by_attr["generate"]
    model_ref = wire_ref(generate.models["pipeline"])
    pipe = _LoadablePipe()
    setattr(pipe, "_cozy_weight_lane", "w8a8")

    async def _send(_msg):
        return None

    ex = Executor(specs, _send)
    ex.store._cache_dir = tmp_path / "cas"

    async def _download(ref, **kwargs):
        return artifact.parent if ref == cell_ref else model_dir

    monkeypatch.setattr(executor_mod, "ensure_local", _download)
    monkeypatch.setattr(
        provision,
        "load_slot",
        lambda *args, **kwargs: provision.SlotLoad(obj=pipe, is_pipeline=True),
    )
    monkeypatch.setattr(ex, "_enable_compiled", _guarded_enable)

    with pytest.raises(
        cc.CompiledLaneUnavailableError,
        match="mandatory quantized-lane function proof incomplete",
    ):
        asyncio.run(ex.ensure_setup(generate, {
            model_ref: pb.Snapshot(digest=MODEL_DIGEST),
            cell_ref: pb.Snapshot(digest=DIGEST_A),
        }))

    required = {by_attr["generate"].name, by_attr["edit"].name}
    assert set(ex.unavailable) == required
    assert all(ex.unavailable[name][0] == "compile_cell_failed" for name in required)
    assert by_attr["generate_turbo"].name not in ex.unavailable
    assert ex.compile_targets() == []


def _merged_lane_endpoint(record_warm):
    """Two w8a8 lane pipes behind ONE handler (the qwen merged shape): the
    declared warmup can only exercise the t2i lane — edit needs an input
    image, so its object has no warmup modality by design (gw#595)."""

    @endpoint(
        models={
            "t2i": Hub("acme/qwen-image", flavor="fp8-w8a8"),
            "edit": Hub("acme/qwen-image-edit", flavor="fp8-w8a8"),
        },
        resources=Resources(vram_gb=48),
        compile=Compile(shapes=((1328, 1328),), family="qwen-image"),
        warmup={"generate": {"prompt": "warmup"}},
    )
    class _MergedEndpoint:
        def setup(self, t2i: _LoadablePipe, edit: _LoadablePipe) -> None:
            self.t2i = t2i
            self.edit = edit

        def generate(self, ctx, payload: _In) -> _Out:
            record_warm(self)
            return _Out(y="ok")

    return _MergedEndpoint


def _wire_merged_lane(ex_cls_specs, tmp_path, monkeypatch):
    import gen_worker.executor as executor_mod

    family = "qwen-image"
    cell_ref = f"_system/family-{family}#inductor-rtx-4090-torch2.9-w8a8"
    artifact = _artifact(tmp_path, family=family)
    model_dir = tmp_path / "merged-lane-model"
    model_dir.mkdir(exist_ok=True)
    pipes = {"t2i": _LoadablePipe(), "edit": _LoadablePipe()}
    for pipe in pipes.values():
        setattr(pipe, "_cozy_weight_lane", "w8a8")

    async def _send(_msg):
        return None

    ex = Executor(ex_cls_specs, _send)
    ex.store._cache_dir = tmp_path / "cas"

    async def _download(ref, **kwargs):
        return artifact.parent if ref == cell_ref else model_dir

    monkeypatch.setattr(executor_mod, "ensure_local", _download)
    monkeypatch.setattr(
        provision,
        "load_slot",
        lambda *args, **kwargs: provision.SlotLoad(
            obj=pipes[kwargs["slot"]], is_pipeline=True),
    )
    monkeypatch.setattr(ex, "_enable_compiled", _guarded_enable)
    return ex, pipes, cell_ref


def test_w8a8_unexercised_sibling_stays_armed_unproven(
    tmp_path, monkeypatch, caplog,
):
    """gw#595(b): an armed MANDATORY-lane object the warmup has no modality
    to exercise must not block adoption by the sibling that proves; it stays
    armed unproven and is logged explicitly."""
    cls = _merged_lane_endpoint(lambda self: _record_fake_warm(self.t2i))
    specs = extract_specs(cls)
    (generate,) = specs
    ex, pipes, cell_ref = _wire_merged_lane(specs, tmp_path, monkeypatch)

    with caplog.at_level("WARNING"):
        asyncio.run(ex.ensure_setup(generate, {
            wire_ref(generate.models["t2i"]): pb.Snapshot(digest=MODEL_DIGEST),
            wire_ref(generate.models["edit"]): pb.Snapshot(digest=DIGEST_B),
            cell_ref: pb.Snapshot(digest=DIGEST_A),
        }))

    targets = {t.model_bindings[0].slot: t for t in ex.compile_targets()}
    assert set(targets) == {"t2i", "edit"}
    assert targets["t2i"].active_compile_ref == cell_ref
    assert targets["t2i"].active_compile_snapshot_digest == DIGEST_A
    # The edit lane is armed (eager is not a w8a8 lane) but unproven.
    assert targets["edit"].active_compile_ref == cell_ref
    assert "armed unproven: no warmup modality" in caplog.text
    assert generate.name not in ex.unavailable


def test_w8a8_exercised_miss_fails_closed_despite_unexercised_sibling(
    tmp_path, monkeypatch,
):
    """gw#595(b) keeps gw#586 shut: an EXERCISED object that misses its own
    warmup graph disproves the cell and fails closed — the unexercised
    sibling exemption never launders a genuine parity defect."""
    cls = _merged_lane_endpoint(
        lambda self: _record_fake_warm(self.t2i, hits=0, misses=2))
    specs = extract_specs(cls)
    (generate,) = specs
    ex, pipes, cell_ref = _wire_merged_lane(specs, tmp_path, monkeypatch)

    with pytest.raises(
        cc.CompiledLaneUnavailableError,
        match="did not serve their own warmup graph",
    ):
        asyncio.run(ex.ensure_setup(generate, {
            wire_ref(generate.models["t2i"]): pb.Snapshot(digest=MODEL_DIGEST),
            wire_ref(generate.models["edit"]): pb.Snapshot(digest=DIGEST_B),
            cell_ref: pb.Snapshot(digest=DIGEST_A),
        }))
    assert ex.compile_targets() == []


def test_w8a8_all_objects_unexercised_fails_closed(tmp_path, monkeypatch):
    """gw#595(b): with ZERO proven objects the cell is entirely unverified —
    a warmup that exercises nothing cannot arm anything."""
    cls = _merged_lane_endpoint(lambda self: None)
    specs = extract_specs(cls)
    (generate,) = specs
    ex, pipes, cell_ref = _wire_merged_lane(specs, tmp_path, monkeypatch)

    with pytest.raises(
        cc.CompiledLaneUnavailableError,
        match="did not serve their own warmup graph",
    ):
        asyncio.run(ex.ensure_setup(generate, {
            wire_ref(generate.models["t2i"]): pb.Snapshot(digest=MODEL_DIGEST),
            wire_ref(generate.models["edit"]): pb.Snapshot(digest=DIGEST_B),
            cell_ref: pb.Snapshot(digest=DIGEST_A),
        }))
    assert ex.compile_targets() == []


def test_production_w8a8_ignores_legacy_compile_environment_fallbacks(
    tmp_path, monkeypatch,
):
    """DesiredInstance and RunJob require Tensorhub-attached exact evidence."""
    import gen_worker.executor as executor_mod

    artifact = _artifact(tmp_path)
    monkeypatch.setenv("GEN_WORKER_COMPILE_CACHE", str(artifact))
    monkeypatch.setenv("GEN_WORKER_COMPILE_CACHE_URL", "https://ignored/cell")
    monkeypatch.setenv("GEN_WORKER_COMPILE_ALLOW_COLD", "1")
    spec = _cold_spec(Hub("acme/klein-finetune", flavor="fp8-w8a8"))
    model_ref = wire_ref(spec.models["pipeline"])

    async def _send(_msg):
        return None

    ex = Executor([spec], _send)
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    async def _download(ref, **kwargs):
        return model_dir

    monkeypatch.setattr(executor_mod, "ensure_local", _download)
    pipe = _Pipe()
    setattr(pipe, "_cozy_weight_lane", "w8a8")
    monkeypatch.setattr(
        provision,
        "load_slot",
        lambda *args, **kwargs: provision.SlotLoad(obj=pipe, is_pipeline=True),
    )
    snapshot = pb.Snapshot(digest=MODEL_DIGEST)
    desired = pb.DesiredInstance(
        function_name=spec.name,
        models=[pb.ModelBinding(slot="pipeline", ref=model_ref)],
    )
    # gw#587: the miss proceeds to the self-mint, IGNORING the inherited
    # local/producer env cells (if the env were honored this would arm and
    # succeed); in a CUDA-less env the typed quantized refusal fires from
    # the self-mint exit.
    with pytest.raises(cc.CompiledLaneUnavailableError, match="self-mint is unavailable"):
        asyncio.run(ex.ensure_desired_instance(
            desired, {model_ref: snapshot},
        ))

    # This is the first production RunJob fence, before setup/mutation. An
    # inherited local/producer env cannot substitute for scheduler-selected
    # RequiredCompileExecution evidence.
    with pytest.raises(RetryableError, match="required_compile_missing"):
        ex._validate_required_compile(
            spec,
            pb.RunJob(
                function_name=spec.name,
                models=[pb.ModelBinding(slot="pipeline", ref=model_ref)],
                snapshots={model_ref: snapshot},
            ),
        )


def test_w8a8_binding_cannot_advertise_plain_materialized_pipeline(tmp_path):
    spec = replace(
        _spec(), models={"pipeline": Hub(
            "acme/klein-finetune", flavor="fp8-w8a8")})

    async def _send(_msg):
        return None

    ex = Executor([spec], _send)
    rec = ex._classes[spec.instance_key]
    rec.ready = True
    rec.instance = _Endpoint()
    ref = wire_ref(spec.models["pipeline"])
    rec.held_bindings = [("pipeline", ref, MODEL_DIGEST)]
    pipe = _Pipe()  # loader silently lost the W8A8 lane
    selection = type("Selection", (), {
        "ref": CACHE_REF + "-w8a8",
        "snapshot_digest": DIGEST_A,
        "path": tmp_path / "cell.tar.gz",
    })()
    with pytest.raises(cc.CompiledLaneUnavailableError, match="materialized pipeline lane"):
        ex._install_compile_targets(
            rec, spec, [pipe], {id(pipe): selection}, {id(pipe): {spec.name}},
        )


def test_w8a8_setup_with_no_addressable_compile_object_fails(tmp_path, monkeypatch):
    import gen_worker.executor as executor_mod

    artifact = _artifact(tmp_path)
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    spec = _cold_spec(Hub("acme/klein-finetune", flavor="fp8-w8a8"))
    model_ref = wire_ref(spec.models["pipeline"])
    cell_ref = CACHE_REF + "-w8a8"

    async def _send(_msg):
        return None

    ex = Executor([spec], _send)
    ex.store._cache_dir = tmp_path / "cas"

    async def _download(ref, **kwargs):
        return artifact.parent if ref == cell_ref else model_dir

    class _SupportObject:
        pass  # no transformer/vae target despite typed setup annotation

    monkeypatch.setattr(executor_mod, "ensure_local", _download)
    monkeypatch.setattr(
        provision,
        "load_slot",
        lambda *args, **kwargs: provision.SlotLoad(
            obj=_SupportObject(), is_pipeline=True),
    )
    monkeypatch.setattr(ex, "_enable_compiled", _guarded_enable)

    with pytest.raises(cc.CompiledLaneUnavailableError, match="no addressable"):
        asyncio.run(ex.ensure_setup(spec, {
            model_ref: pb.Snapshot(digest=MODEL_DIGEST),
            cell_ref: pb.Snapshot(digest=DIGEST_A),
        }))
    assert ex.compile_targets() == []


def test_desired_w8a8_cell_digest_and_ref_changes_vacate_then_rebuild(
    tmp_path, monkeypatch,
):
    import gen_worker.executor as executor_mod

    artifact = _artifact(tmp_path)
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    spec = _cold_spec(Hub("acme/klein-finetune", flavor="fp8-w8a8"))
    model_ref = wire_ref(spec.models["pipeline"])
    cell_a = CACHE_REF + "-w8a8"
    cell_b = f"_system/family-{FAMILY}#inductor-rtx-5090-torch2.9-w8a8"

    async def _send(_msg):
        return None

    ex = Executor([spec], _send)
    ex.store._cache_dir = tmp_path / "cas"

    async def _download(ref, **kwargs):
        return artifact.parent if ref.startswith("_system/") else model_dir

    monkeypatch.setattr(executor_mod, "ensure_local", _download)

    loaded = []

    def _load(*args, **kwargs):
        pipe = _LoadablePipe()
        setattr(pipe, "_cozy_weight_lane", "w8a8")
        loaded.append(pipe)
        return provision.SlotLoad(obj=pipe, is_pipeline=True)

    monkeypatch.setattr(provision, "load_slot", _load)
    monkeypatch.setattr(ex, "_enable_compiled", _guarded_enable)
    counter = {"hits": 0}

    def _counters():
        counter["hits"] += 1
        return {"fxgraph_cache_hit": counter["hits"], "fxgraph_cache_miss": 0}

    monkeypatch.setattr(cc, "inductor_counters", _counters)
    _ColdEndpoint.setups = _ColdEndpoint.warmups = 0

    def snapshots(cell_ref, cell_digest):
        return {
            model_ref: pb.Snapshot(digest=MODEL_DIGEST),
            cell_ref: pb.Snapshot(digest=cell_digest),
        }

    desired = pb.DesiredInstance(
        function_name=spec.name,
        models=[pb.ModelBinding(slot="pipeline", ref=model_ref)],
    )

    def reconcile(cell_ref, cell_digest):
        asyncio.run(ex.ensure_desired_instance(
            desired, snapshots(cell_ref, cell_digest)))

    reconcile(cell_a, DIGEST_A)
    first = ex.compile_targets()[0]
    assert (first.active_compile_ref,
            first.active_compile_snapshot_digest) == (cell_a, DIGEST_A)
    vacated_targets = []
    vacate = ex._vacate_record

    async def _observed_vacate(rec):
        released = await vacate(rec)
        vacated_targets.append(ex.compile_targets())
        return released

    monkeypatch.setattr(ex, "_vacate_record", _observed_vacate)

    # Same mutable cell ref, new immutable bytes: no hot unwrap; declarative
    # reconcile vacates and creates a fresh incarnation against the new digest.
    reconcile(cell_a, DIGEST_B)
    second = ex.compile_targets()[0]
    assert second.incarnation_id != first.incarnation_id
    assert (second.active_compile_ref,
            second.active_compile_snapshot_digest) == (cell_a, DIGEST_B)
    assert vacated_targets[-1] == []

    # Chosen compatible ref/SKU changes: the same state-driven reload path.
    reconcile(cell_b, DIGEST_A)
    third = ex.compile_targets()[0]
    assert third.incarnation_id not in {
        first.incarnation_id, second.incarnation_id}
    assert (third.active_compile_ref,
            third.active_compile_snapshot_digest) == (cell_b, DIGEST_A)
    assert _ColdEndpoint.setups == 3


def test_desired_plain_cell_change_vacates_then_rebuilds(
    tmp_path, monkeypatch,
):
    """Optional acceleration moves only from explicit desired cell evidence."""
    import gen_worker.executor as executor_mod

    artifact = _artifact(tmp_path)
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    spec = _cold_spec()
    model_ref = wire_ref(spec.models["pipeline"])
    cell_a = CACHE_REF
    cell_b = f"_system/family-{FAMILY}#inductor-rtx-5090-torch2.9"

    async def _send(_msg):
        return None

    ex = Executor([spec], _send)
    ex.store._cache_dir = tmp_path / "cas"

    async def _download(ref, **kwargs):
        return artifact.parent if ref.startswith("_system/") else model_dir

    monkeypatch.setattr(executor_mod, "ensure_local", _download)
    monkeypatch.setattr(
        provision, "load_slot",
        lambda *args, **kwargs: provision.SlotLoad(
            obj=_LoadablePipe(), is_pipeline=True),
    )
    monkeypatch.setattr(ex, "_enable_compiled", _guarded_enable)
    counter = {"hits": 0}

    def _counters():
        counter["hits"] += 1
        return {"fxgraph_cache_hit": counter["hits"], "fxgraph_cache_miss": 0}

    monkeypatch.setattr(cc, "inductor_counters", _counters)
    desired = pb.DesiredInstance(
        function_name=spec.name,
        models=[pb.ModelBinding(slot="pipeline", ref=model_ref)],
    )

    def reconcile(cell_ref, cell_digest):
        asyncio.run(ex.ensure_desired_instance(desired, {
            model_ref: pb.Snapshot(digest=MODEL_DIGEST),
            cell_ref: pb.Snapshot(digest=cell_digest),
        }))

    reconcile(cell_a, DIGEST_A)
    first = ex.compile_targets()[0]
    assert (first.active_compile_ref,
            first.active_compile_snapshot_digest) == (cell_a, DIGEST_A)
    vacated_targets = []
    vacate = ex._vacate_record

    async def _observed_vacate(rec):
        released = await vacate(rec)
        vacated_targets.append(ex.compile_targets())
        return released

    monkeypatch.setattr(ex, "_vacate_record", _observed_vacate)
    reconcile(cell_b, DIGEST_B)
    second = ex.compile_targets()[0]
    assert vacated_targets == [[]]
    assert second.incarnation_id != first.incarnation_id
    assert (second.active_compile_ref,
            second.active_compile_snapshot_digest) == (cell_b, DIGEST_B)


def test_missing_desired_w8a8_cell_keeps_workers_own_armed_target(tmp_path, monkeypatch):
    """gw#587 flips this outcome BY DESIGN: cells are worker-owned (th#883
    pull-by-key + self-mint), so a hub delivery that does NOT attach the
    cell is no longer authority to tear down a worker's own armed, proven
    target — the worker minted (or can re-mint) that cell itself.
    Invalidation still flows through the real channels (adoption ops,
    artifact_drift, cell_selection_bug), never through non-delivery.
    Pre-gw#587 this asserted the fail-closed teardown."""
    spec = replace(
        _spec(), models={"pipeline": Hub(
            "acme/klein-finetune", flavor="fp8-w8a8")})
    ex, _sent = _wire_executor(spec, tmp_path)
    _active_w8a8_target(ex)
    model_ref = wire_ref(spec.models["pipeline"])

    asyncio.run(ex.ensure_setup(
        spec, {model_ref: pb.Snapshot(digest=MODEL_DIGEST)}))
    assert len(ex.compile_targets()) == 1
    assert ex._classes[spec.instance_key].ready


def test_concurrent_same_ref_setups_keep_each_loaded_snapshot_identity(
    tmp_path, monkeypatch,
):
    """A loads digest A, B advances ref-global disk state to B before A's
    load lock; A's record/target must still say A, never the current B."""
    import gen_worker.executor as executor_mod

    first = _cold_spec()
    second = replace(
        first,
        name="cold-generate-b",
        cls=_ColdEndpointB,
        method=_ColdEndpointB.run,
    )

    async def _send(_msg):
        return None

    ex = Executor([first, second], _send)
    ex.store._cache_dir = tmp_path / "cas"
    model_ref = wire_ref(first.models["pipeline"])
    digest_a = "blake3:" + "1" * 64
    digest_b = "blake3:" + "2" * 64
    paths = {
        digest_a: tmp_path / ("1" * 64),
        digest_b: tmp_path / ("2" * 64),
    }
    for path in paths.values():
        path.mkdir()
    downloaded = {digest_a: asyncio.Event(), digest_b: asyncio.Event()}

    async def _download(ref, *, snapshot=None, **kwargs):
        digest = snapshot.snapshot_digest
        downloaded[digest].set()
        return paths[digest]

    monkeypatch.setattr(executor_mod, "ensure_local", _download)
    monkeypatch.setattr(
        provision,
        "load_slot",
        lambda *args, **kwargs: provision.SlotLoad(
            obj=_LoadablePipe(), is_pipeline=True),
    )
    from gen_worker import fleet_cells

    monkeypatch.setattr(
        ex, "_enable_compiled",
        lambda *args: fleet_cells.ArmOutcome(armed=False))

    async def scenario():
        await ex._load_lock.acquire()
        task_a = asyncio.create_task(ex.ensure_setup(
            first, {model_ref: pb.Snapshot(digest=digest_a)}))
        await downloaded[digest_a].wait()
        await asyncio.sleep(0)  # A queues on the held global load lock.

        task_b = asyncio.create_task(ex.ensure_setup(
            second, {model_ref: pb.Snapshot(digest=digest_b)}))
        await downloaded[digest_b].wait()
        await asyncio.sleep(0)  # B has advanced the ref-global disk identity.
        assert ex.store._disk_identities[model_ref][0] == digest_b

        ex._load_lock.release()
        await asyncio.gather(task_a, task_b)

    asyncio.run(scenario())

    targets = {target.function_names[0]: target for target in ex.compile_targets()}
    assert targets[first.name].model_bindings[0].snapshot_digest == digest_a
    assert targets[second.name].model_bindings[0].snapshot_digest == digest_b
    assert ex._classes[first.instance_key].held_snapshot_digests[model_ref] == digest_a
    assert ex._classes[second.instance_key].held_snapshot_digests[model_ref] == digest_b
    assert ex.store.resident_identity(model_ref)[0] == digest_b


def _active_w8a8_target(ex: Executor, *, digest=DIGEST_A):
    target_id = _target_id(ex)
    found = ex._compile_target(target_id)
    assert found is not None
    _rec, target = found
    setattr(target.pipeline, "_cozy_weight_lane", "w8a8")
    ex._refresh_compile_target(target)
    target.active_compile_ref = CACHE_REF + "-w8a8"
    target.active_compile_snapshot_digest = digest
    return target


def _required_run(spec: EndpointSpec, target, **overrides) -> pb.RunJob:
    model_ref = wire_ref(spec.models["pipeline"])
    required = dict(
        target_incarnation_id=target.incarnation_id,
        cell_ref=target.active_compile_ref,
        cell_snapshot_digest=target.active_compile_snapshot_digest,
        contract_digest=target.contract_digest,
    )
    required.update(overrides)
    return pb.RunJob(
        request_id="required-compile",
        attempt=1,
        function_name=spec.name,
        input_payload=msgspec.msgpack.encode(_In(prompt="cat")),
        models=[pb.ModelBinding(slot="pipeline", ref=model_ref)],
        snapshots={model_ref: pb.Snapshot(digest=MODEL_DIGEST)},
        required_compile=pb.RequiredCompileExecution(**required),
    )


@pytest.mark.parametrize(
    ("override", "reason"),
    [
        ({"target_incarnation_id": "gone"}, "required_compile_replaced"),
        ({"cell_ref": CACHE_REF + "-other"}, "required_compile_identity_mismatch"),
        ({"cell_snapshot_digest": DIGEST_B}, "required_compile_identity_mismatch"),
        ({"contract_digest": "bad-contract"}, "required_compile_identity_mismatch"),
        ({"cell_ref": ""}, "required_compile_invalid"),
    ],
)
def test_required_compile_rejects_wrong_target_cell_digest_or_contract(
    tmp_path, override, reason,
):
    spec = replace(
        _spec(), models={"pipeline": Hub(
            "acme/klein-finetune", flavor="fp8-w8a8")})
    ex, _sent = _wire_executor(spec, tmp_path)
    target = _active_w8a8_target(ex)
    run = _required_run(spec, target, **override)
    with pytest.raises(RetryableError, match=reason):
        ex._validate_required_compile(spec, run)


def test_required_compile_rejects_missing_fence_and_binding_digest_drift(tmp_path):
    spec = replace(
        _spec(), models={"pipeline": Hub(
            "acme/klein-finetune", flavor="fp8-w8a8")})
    ex, _sent = _wire_executor(spec, tmp_path)
    target = _active_w8a8_target(ex)
    model_ref = wire_ref(spec.models["pipeline"])

    with pytest.raises(RetryableError, match="required_compile_missing"):
        ex._validate_required_compile(spec, pb.RunJob(function_name=spec.name))

    run = _required_run(spec, target)
    run.snapshots[model_ref].digest = DIGEST_B
    with pytest.raises(RetryableError, match="required_compile_binding_mismatch"):
        ex._validate_required_compile(spec, run)

    other = replace(
        spec, models={"pipeline": Hub("acme/other-klein", flavor="fp8-w8a8")})
    other_ref = wire_ref(other.models["pipeline"])
    run = _required_run(spec, target)
    del run.snapshots[model_ref]
    run.snapshots[other_ref].CopyFrom(pb.Snapshot(digest=MODEL_DIGEST))
    with pytest.raises(RetryableError, match="required_compile_binding_mismatch"):
        ex._validate_required_compile(other, run)


def test_runtime_guard_revokes_state_emits_one_causal_failure_and_quarantines(
    tmp_path,
):
    spec = replace(
        _spec(), models={"pipeline": Hub(
            "acme/klein-finetune", flavor="fp8-w8a8")})
    ex, sent = _wire_executor(spec, tmp_path)
    active = _active_w8a8_target(ex)
    active_ref = active.active_compile_ref
    active_digest = active.active_compile_snapshot_digest
    active_id = active.incarnation_id
    required_run = _required_run(spec, active)
    _adopt(
        ex,
        ref=active_ref,
        digest=active_digest,
        operation_id=OP_A,
        target_incarnation_id=active_id,
    )
    sent.clear()
    found = ex._compile_target(active_id)
    assert found is not None
    rec, internal = found
    signal = getattr(internal.pipeline, cc._MARKER_ATTR)["failure_signal"]
    callback = signal["callback"]
    assert callable(callback)

    async def _trip() -> None:
        ex._loop = asyncio.get_running_loop()
        await asyncio.to_thread(callback, "compiled graph exploded")
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        # The wrapper calls the callback only once, but duplicate delivery is
        # harmless and cannot fabricate a second causal terminal event.
        await asyncio.to_thread(callback, "duplicate")
        await asyncio.sleep(0)

    asyncio.run(_trip())
    (revoked,) = ex.compile_targets()
    assert revoked.incarnation_id == active_id
    assert revoked.active_compile_ref == ""
    assert revoked.active_compile_snapshot_digest == ""
    assert rec.stale is True
    failed = _events(sent, pb.MODEL_STATE_FAILED)
    assert len(failed) == 1
    assert failed[0].error == "adopt_failed:runtime_guard"
    assert failed[0].ref == active_ref
    assert failed[0].snapshot_digest == active_digest
    assert failed[0].operation_id == OP_A
    assert failed[0].target_incarnation_id == active_id

    with pytest.raises(RetryableError, match="required_compile_identity_mismatch"):
        ex._validate_required_compile(spec, required_run)

    sent.clear()
    # A controller-only desired/order rewrite can mint a new operation ID but
    # does not change executable identity. The same target + immutable cell
    # remains quarantined and cannot repeat wrap/warmup.
    _adopt(
        ex,
        ref=active_ref,
        digest=active_digest,
        operation_id=OP_B,
        target_incarnation_id=active_id,
    )
    _assert_failed(
        sent,
        "adopt_failed:cell_quarantined",
        digest=active_digest,
        operation_id=OP_B,
        target_incarnation_id=active_id,
    )


def test_cell_quarantine_survives_successful_different_cell_adoption(
    tmp_path, monkeypatch,
):
    """A failure stays sticky on its incarnation after B becomes active."""
    _artifact(tmp_path)
    spec = _spec()
    ex, sent = _wire_executor(spec, tmp_path)
    monkeypatch.setattr(cc, "apply", _guarded_apply)
    _fake_counters(monkeypatch, hits=1, misses=0)
    target_id = _target_id(ex)

    _adopt(
        ex,
        digest=DIGEST_A,
        operation_id=OP_A,
        target_incarnation_id=target_id,
    )
    found = ex._compile_target(target_id)
    assert found is not None
    _rec, target = found
    callback = getattr(target.pipeline, cc._MARKER_ATTR)[
        "failure_signal"
    ]["callback"]
    assert callable(callback)

    async def _trip_a() -> None:
        ex._loop = asyncio.get_running_loop()
        await asyncio.to_thread(callback, "cell A failed")
        await asyncio.sleep(0)
        await asyncio.sleep(0)

    asyncio.run(_trip_a())
    assert (CACHE_REF, DIGEST_A) in target.failed_compile_identities

    sent.clear()
    _adopt(
        ex,
        digest=DIGEST_B,
        operation_id=OP_B,
        target_incarnation_id=target_id,
    )
    adopted = _events(sent, pb.MODEL_STATE_ADOPTED)
    assert len(adopted) == 1
    assert adopted[0].snapshot_digest == DIGEST_B
    assert (CACHE_REF, DIGEST_A) in target.failed_compile_identities

    sent.clear()
    _adopt(
        ex,
        digest=DIGEST_A,
        operation_id="adopt-operation-c",
        target_incarnation_id=target_id,
    )
    _assert_failed(
        sent,
        "adopt_failed:cell_quarantined",
        digest=DIGEST_A,
        operation_id="adopt-operation-c",
        target_incarnation_id=target_id,
    )
    (still_b,) = ex.compile_targets()
    assert still_b.active_compile_snapshot_digest == DIGEST_B


def test_multifunction_adoption_keeps_target_identity_through_guard_failure(
    tmp_path, monkeypatch,
):
    """ADOPTED -> StateDelta -> causal guard failure keeps one target key.

    Tensorhub correlates the pending operation to the pre-adoption target
    identity, including its function aliases. Adoption therefore proves the
    entire advertised set and never narrows it in place before a later guard
    event can quarantine the exact operation.
    """
    import gen_worker.executor as executor_mod

    artifact = _artifact(tmp_path)
    model_dir = tmp_path / "multifunction-model"
    model_dir.mkdir()

    @endpoint(
        models={"pipeline": Hub("acme/flux-base")},
        resources=Resources(vram_gb=24),
        compile=Compile(shapes=((768, 768),), family=FAMILY),
        warmup={
            "generate": {"prompt": "warmup"},
            "edit": {"prompt": "warmup"},
        },
    )
    class _FluxBaseEndpoint:
        def setup(self, pipeline: _LoadablePipe) -> None:
            self.pipeline = pipeline

        def generate(self, ctx, payload: _In) -> _Out:
            _record_fake_warm(self.pipeline)
            return _Out(y="ok")

        def edit(self, ctx, payload: _In) -> _Out:
            _record_fake_warm(self.pipeline)
            return _Out(y="ok")

    specs = extract_specs(_FluxBaseEndpoint)
    generate = next(spec for spec in specs if spec.name == "generate")
    model_ref = wire_ref(generate.models["pipeline"])
    pipe = _Pipe()
    sent: list[pb.WorkerMessage] = []

    async def _send(msg):
        sent.append(msg)

    ex = Executor(specs, _send)
    ex.store._cache_dir = tmp_path / "cas"

    async def _download(ref, **kwargs):
        return model_dir

    monkeypatch.setattr(executor_mod, "ensure_local", _download)
    monkeypatch.setattr(
        provision,
        "load_slot",
        lambda *args, **kwargs: provision.SlotLoad(obj=pipe, is_pipeline=True),
    )
    asyncio.run(ex.ensure_setup(generate, {
        model_ref: pb.Snapshot(digest=MODEL_DIGEST),
    }))
    (before,) = ex.compile_targets()
    assert list(before.function_names) == ["edit", "generate"]
    assert before.active_compile_ref == ""

    async def _local_cell(ref, snapshot=None, *, binding=None):
        return artifact.parent

    ex.store.ensure_local = _local_cell  # type: ignore[method-assign]
    monkeypatch.setattr(cc, "apply", _guarded_apply)
    _fake_counters(monkeypatch, hits=1, misses=0)
    _adopt(
        ex,
        operation_id=OP_A,
        target_incarnation_id=before.incarnation_id,
    )
    adopted = _events(sent, pb.MODEL_STATE_ADOPTED)
    assert len(adopted) == 1
    assert adopted[0].target_incarnation_id == before.incarnation_id

    lifecycle = Lifecycle(
        SimpleNamespace(worker_jwt="", worker_id="worker"), ex)
    (after_adopt,) = lifecycle._state_delta().compile_targets
    assert after_adopt.incarnation_id == before.incarnation_id
    assert list(after_adopt.function_names) == list(before.function_names)
    assert after_adopt.active_compile_ref == CACHE_REF

    found = ex._compile_target(before.incarnation_id)
    assert found is not None
    _rec, internal = found
    signal = getattr(internal.pipeline, cc._MARKER_ATTR)["failure_signal"]
    callback = signal["callback"]
    assert callable(callback)

    async def _trip() -> None:
        ex._loop = asyncio.get_running_loop()
        await asyncio.to_thread(callback, "compiled graph exploded")
        await asyncio.sleep(0)
        await asyncio.sleep(0)

    asyncio.run(_trip())
    failed = _events(sent, pb.MODEL_STATE_FAILED)
    assert len(failed) == 1
    assert failed[0].error == "adopt_failed:runtime_guard"
    assert failed[0].operation_id == OP_A
    assert failed[0].target_incarnation_id == before.incarnation_id

    (after_guard,) = lifecycle._state_delta().compile_targets
    assert after_guard.incarnation_id == before.incarnation_id
    assert list(after_guard.function_names) == list(before.function_names)
    assert after_guard.active_compile_ref == ""
    assert after_guard.active_compile_snapshot_digest == ""


def test_hot_adoption_rejects_an_unproven_advertised_function_alias(
    tmp_path, monkeypatch,
):
    """One function's cache hit cannot certify its advertised sibling."""
    import gen_worker.executor as executor_mod

    artifact = _artifact(tmp_path)
    model_dir = tmp_path / "partial-function-proof-model"
    model_dir.mkdir()

    @endpoint(
        models={"pipeline": Hub("acme/flux-base")},
        resources=Resources(vram_gb=24),
        compile=Compile(shapes=((768, 768),), family=FAMILY),
        warmup={
            "generate": {"prompt": "warmup"},
            "edit": {"prompt": "warmup"},
        },
    )
    class _PartiallyCoveredEndpoint:
        def setup(self, pipeline: _LoadablePipe) -> None:
            self.pipeline = pipeline

        def generate(self, ctx, payload: _In) -> _Out:
            _record_fake_warm(self.pipeline)
            return _Out(y="ok")

        def edit(self, ctx, payload: _In) -> _Out:
            return _Out(y="eager-only")

    specs = extract_specs(_PartiallyCoveredEndpoint)
    generate = next(spec for spec in specs if spec.name == "generate")
    model_ref = wire_ref(generate.models["pipeline"])
    pipe = _Pipe()
    sent: list[pb.WorkerMessage] = []

    async def _send(msg):
        sent.append(msg)

    ex = Executor(specs, _send)
    ex.store._cache_dir = tmp_path / "cas"

    async def _download(ref, **kwargs):
        return model_dir

    monkeypatch.setattr(executor_mod, "ensure_local", _download)
    monkeypatch.setattr(
        provision,
        "load_slot",
        lambda *args, **kwargs: provision.SlotLoad(obj=pipe, is_pipeline=True),
    )
    asyncio.run(ex.ensure_setup(generate, {
        model_ref: pb.Snapshot(digest=MODEL_DIGEST),
    }))
    (before,) = ex.compile_targets()
    assert list(before.function_names) == ["edit", "generate"]

    async def _local_cell(ref, snapshot=None, *, binding=None):
        return artifact.parent

    ex.store.ensure_local = _local_cell  # type: ignore[method-assign]
    monkeypatch.setattr(cc, "apply", _guarded_apply)
    _fake_counters(monkeypatch, hits=1, misses=0)
    _adopt(ex, target_incarnation_id=before.incarnation_id)

    _assert_failed(
        sent,
        "adopt_failed:function_alias_unproven",
        target_incarnation_id=before.incarnation_id,
    )
    assert not _events(sent, pb.MODEL_STATE_ADOPTED)
    (after,) = ex.compile_targets()
    assert after.incarnation_id == before.incarnation_id
    assert list(after.function_names) == list(before.function_names)
    assert after.active_compile_ref == ""


def test_guard_revocation_between_intake_and_gpu_turn_fails_final_fence(
    tmp_path,
):
    spec = replace(
        _spec(), models={"pipeline": Hub(
            "acme/klein-finetune", flavor="fp8-w8a8")})
    ex, sent = _wire_executor(spec, tmp_path)
    active = _active_w8a8_target(ex)
    _adopt(
        ex,
        ref=active.active_compile_ref,
        digest=active.active_compile_snapshot_digest,
        operation_id=OP_A,
        target_incarnation_id=active.incarnation_id,
    )
    sent.clear()
    found = ex._compile_target(active.incarnation_id)
    assert found is not None
    _rec, internal = found
    callback = getattr(
        internal.pipeline, cc._MARKER_ATTR)["failure_signal"]["callback"]
    run = _required_run(spec, active)

    async def scenario() -> None:
        ex._loop = asyncio.get_running_loop()
        first_has_gpu = asyncio.Event()
        sibling_validated = asyncio.Event()

        async def first_request() -> None:
            async with ex._gpu_semaphore:
                first_has_gpu.set()
                await sibling_validated.wait()
                # The active request trips its guard before releasing the GPU
                # turn to a sibling that validated during queueing.
                await asyncio.to_thread(callback, "first request guard failed")

        async def sibling_request() -> None:
            await first_has_gpu.wait()
            ex._validate_required_compile(spec, run)  # intake fence passes
            sibling_validated.set()
            async with ex._gpu_semaphore:
                # This is the production final fence immediately after GPU
                # acquisition. Revocation must already be visible here.
                with pytest.raises(
                    RetryableError, match="required_compile_identity_mismatch",
                ):
                    ex._validate_required_compile(spec, run)

        await asyncio.gather(first_request(), sibling_request())
        await asyncio.sleep(0)

    asyncio.run(scenario())
    assert len(_events(sent, pb.MODEL_STATE_FAILED)) == 1


def test_target_replacement_between_assignment_and_gpu_never_runs_handler(tmp_path):
    spec = replace(
        _spec(), models={"pipeline": Hub(
            "acme/klein-finetune", flavor="fp8-w8a8")})
    ex, sent = _wire_executor(spec, tmp_path)
    old = _active_w8a8_target(ex)
    run = _required_run(spec, old)
    run.compute.CopyFrom(pb.ResolvedCompute(accelerator="cuda", gpu_index=0))
    _Endpoint.runs = 0
    original_run = _Endpoint.run

    def _counted_run(self, ctx, payload):
        _Endpoint.runs += 1
        return original_run(self, ctx, payload)

    async def scenario():
        await ex._gpu_semaphore.acquire()
        initial_validated = asyncio.Event()
        calls = 0
        validate = ex._validate_required_compile

        def _observed_validate(effective, incoming):
            nonlocal calls
            validate(effective, incoming)
            calls += 1
            if calls == 1:
                initial_validated.set()

        ex._validate_required_compile = _observed_validate  # type: ignore[method-assign]
        setattr(_Endpoint, "run", _counted_run)
        try:
            await ex.handle_run_job(run)
            job = ex.jobs[(run.request_id, run.attempt)]
            assert job.task is not None
            await initial_validated.wait()
            await asyncio.sleep(0)

            rec = ex._classes[spec.instance_key]
            rec.compile_targets.clear()
            new_pipe = _Pipe()
            setattr(new_pipe, "_cozy_weight_lane", "w8a8")
            selection = type("Selection", (), {
                "ref": old.active_compile_ref,
                "snapshot_digest": old.active_compile_snapshot_digest,
                "path": tmp_path / "replacement.tar.gz",
            })()
            _mark_fake_guard(new_pipe)
            ex._install_compile_targets(
                rec,
                spec,
                [new_pipe],
                {id(new_pipe): selection},
                {id(new_pipe): {spec.name}},
            )
            assert _target_id(ex) != old.incarnation_id
            ex._gpu_semaphore.release()
            await job.task
            assert calls == 1  # second validation raises before increment
        finally:
            setattr(_Endpoint, "run", original_run)
            if ex._gpu_semaphore.locked():
                ex._gpu_semaphore.release()

    asyncio.run(scenario())
    results = [m.job_result for m in sent if m.WhichOneof("msg") == "job_result"]
    assert results and results[-1].status == pb.JOB_STATUS_RETRYABLE
    assert _Endpoint.runs == 0


# ---------------------------------------------------------------------------
# th#569 boot-attach: cache snapshot on RunJob.snapshots
# ---------------------------------------------------------------------------


def test_fetch_compile_snapshot_finds_family_cache_and_ignores_others(tmp_path):
    artifact = _artifact(tmp_path)
    spec = _spec()
    ex, _sent = _wire_executor(spec, tmp_path)
    snapshots = {
        MODEL_REF: pb.Snapshot(digest="blake3:aa"),
        CACHE_REF: pb.Snapshot(digest="blake3:bb"),
    }
    got = asyncio.run(ex._fetch_compile_snapshot(spec, snapshots))
    assert got is not None
    assert got.path == artifact
    assert got.ref == CACHE_REF
    assert got.snapshot_digest == "blake3:bb"
    # Other families' cells and an absent snapshot map both resolve to None.
    other = {"_system/family-sdxl#inductor-rtx-4090-torch2.9": pb.Snapshot()}
    assert asyncio.run(ex._fetch_compile_snapshot(spec, other)) is None
    assert asyncio.run(ex._fetch_compile_snapshot(spec, None)) is None


@pytest.mark.parametrize("lane", ["w8a8", "plain"])
def test_fetch_compile_snapshot_selects_exact_lane(tmp_path, lane):
    """The spec's weight lane picks exactly its own cell — w8a8 specs take
    the -w8a8 cell, plain specs ignore it — and only that cell is fetched."""
    if lane == "w8a8":
        spec = replace(
            _spec(), models={"pipeline": Hub(
                "acme/klein-finetune", flavor="fp8-w8a8")},
        )
        extra_ref = "acme/klein-finetune#fp8-w8a8"
    else:
        spec = _spec()
        extra_ref = "acme/unselected#fp8-w8a8"
    ex, _sent = _wire_executor(spec, tmp_path)
    plain = tmp_path / "plain"
    w8a8 = tmp_path / "w8a8"
    plain.mkdir()
    w8a8.mkdir()
    (plain / "plain.tar.gz").write_bytes(b"plain")
    (w8a8 / "w8a8.tar.gz").write_bytes(b"w8a8")
    seen: list[str] = []

    async def _ensure(ref, snapshot=None, *, binding=None):
        seen.append(ref)
        return w8a8 if ref.endswith("-w8a8") else plain

    ex.store.ensure_local = _ensure  # type: ignore[method-assign]
    plain_ref = f"_system/family-{FAMILY}#inductor-rtx-4090-torch2.9"
    w8a8_ref = plain_ref + "-w8a8"
    snapshots = {
        plain_ref: pb.Snapshot(digest=DIGEST_A),
        w8a8_ref: pb.Snapshot(digest=DIGEST_B),
        extra_ref: pb.Snapshot(),
    }
    got = asyncio.run(ex._fetch_compile_snapshot(spec, snapshots))
    assert got is not None
    if lane == "w8a8":
        assert got.path == w8a8 / "w8a8.tar.gz"
        assert got.ref == w8a8_ref and got.snapshot_digest == DIGEST_B
        assert seen == [w8a8_ref]
    else:
        assert got.path == plain / "plain.tar.gz"
        assert got.ref == plain_ref and got.snapshot_digest == DIGEST_A
        assert seen == [plain_ref]


def test_prepare_with_explicit_artifact_seeds(tmp_path):
    artifact = _artifact(tmp_path)
    meta = cc.prepare(FAMILY, cache_dir=tmp_path / "cache", artifact=artifact)
    assert meta is not None and meta["family"] == FAMILY
    assert (tmp_path / "cache" / "compile-cache" / "inductor" / "g" / "code.py").exists()


def test_manifest_carries_compile_block():
    from gen_worker.discovery.discover import _extract_entries

    @endpoint(compile=Compile(shapes=((768, 768), (1024, 1024)), family=FAMILY))
    class Ep:
        def gen(self, ctx: RequestContext, data: _In) -> _Out:
            return _Out()

    (entry,) = _extract_entries(Ep, "testmod")
    assert entry["compile"] == {
        "family": FAMILY,
        "shapes": [[768, 768], [1024, 1024]],
        "targets": ["transformer", "vae.decode"],
    }


# ---------------------------------------------------------------------------
# ensure_local digest guard (e2e#117 live find #7): a cached materialization
# of the same ref must NOT short-circuit when the snapshot digest changed
# (flavor re-published — digest-change re-adoption fetched the stale bytes).


def test_ensure_local_redownloads_on_digest_change(tmp_path, monkeypatch):
    import gen_worker.executor as executor_mod

    async def _noop_send(msg):
        return None

    async def run():
        store = executor_mod.ModelStore(_noop_send, cache_dir=tmp_path)
        old_dir = tmp_path / "snapshots" / "aa11"
        old_dir.mkdir(parents=True)
        ref = "_system/family-fam#inductor-rtx-4090-torch2.9"
        store.residency.track_disk(ref, old_dir)

        new_dir = tmp_path / "snapshots" / "bb22"
        new_dir.mkdir(parents=True)
        calls = []

        async def fake_download(r, **kw):
            calls.append(r)
            return new_dir

        monkeypatch.setattr(executor_mod, "ensure_local", fake_download)
        # same digest -> cache hit, no download
        got = await store.ensure_local(ref, pb.Snapshot(digest="blake3:aa11"))
        assert got == old_dir and calls == []
        # digest change -> stale cache bypassed, downloader invoked
        got = await store.ensure_local(ref, pb.Snapshot(digest="blake3:bb22"))
        assert got == new_dir and calls == [ref]

    asyncio.run(run())
