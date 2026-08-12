"""th#567 hot adoption: MODEL_OP_KIND_ADOPT_COMPILE_CACHE re-wraps resident
modules in place — verified seed, one warmup, ADOPTED report; ANY failure
stays eager with a classified adopt_failed:<reason>. Plus th#569 boot-attach:
a compile-cache snapshot on RunJob.snapshots reaches compile_cache.enable."""

from __future__ import annotations

import asyncio
import logging
import threading
from dataclasses import replace
from pathlib import Path

import msgspec
import pytest

from typing import Annotated

from gen_worker import (
    AxisClass,
    Compile,
    CompileAxis,
    RequestContext,
    Resources,
    endpoint,
)
from gen_worker import compile_cache as cc
from gen_worker.models import provision
from gen_worker.api.binding import Hub
from gen_worker.api.binding import wire_ref
from gen_worker.api.errors import RetryableError
from gen_worker.config import Settings
from gen_worker.executor import Executor
from gen_worker.lifecycle import Lifecycle
from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.registry import EndpointSpec, extract_specs

FAMILY = "flux2-klein-4b"
CACHE_REF = f"root/family-{FAMILY}#inductor-rtx-4090-torch2.9"
MODEL_REF = "acme/klein-finetune:latest"
DIGEST_A = "blake3:" + "a" * 64
DIGEST_B = "blake3:" + "b" * 64
MODEL_DIGEST = "blake3:" + "c" * 64
OP_A = "adopt-operation-a"
OP_B = "adopt-operation-b"


class _In(msgspec.Struct):
    prompt: str = ""


class _AxisIn(msgspec.Struct):
    """Payload with a CompileAxis-partitioned guidance field (SDK v2)."""

    prompt: str = ""
    guidance_scale: Annotated[float, CompileAxis(classes=(
        AxisClass("cfg_off", match=lambda v: v == 0, warm=0.0),
        AxisClass("cfg_on", match=lambda v: v != 0, warm=5.0),
    ))] = 5.0


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


def _cell_arm(artifact, ref=None, digest=None):
    """pgw#904: a delivered cell is an exact ORDER (`Arm.artifact` ->
    `_ArmOrder`), never a snapshot entry the worker scans for. These rigs
    fake the arm itself, so no expected identity or publisher rides it."""
    from gen_worker import executor as executor_mod

    return executor_mod._ArmOrder(
        backend="aot_cell",
        selection=executor_mod._CompileArtifactSelection(
            path=Path(artifact), ref=ref or CACHE_REF,
            snapshot_digest=digest or DIGEST_A))


def _seeded_enable(ex, artifact, ref=None, digest=None):
    """The REAL hub-less arming policy, handed its artifact at the test seam
    (pgw#904 deleted the connected fetch): the seeded dynamo arm and the
    warmup proof run unchanged."""
    from gen_worker import fleet_cells

    def _enable(pipe, cfg, art, delivered=None, arm=None, boot_local_key=""):
        return fleet_cells.enable_compiled(
            pipe, cfg, ex.store._cache_dir, Path(artifact),
            delivered_ref=ref or CACHE_REF,
            delivered_digest=digest or DIGEST_A)

    return _enable


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


#: pgw#1148: the w8a8 lane no longer arrives as a `#fp8-w8a8` ref TOKEN
#: (§1.32(d) deleted that address). It arrives as the pipeline's own
#: `_cozy_weight_lane` stamp — what the weights ARE — so these fixtures say
#: so explicitly instead of sniffing a ref.
#: pgw#1148: the mandate itself now comes from the HUB-RESOLVED lane — the
#: only channel that carries evidence rather than an assertion in a ref.
W8A8_LANE = "fp8-w8a8-dynamic+compiled"


def _declare_w8a8_lane(ex, ref: str) -> None:
    """State the w8a8 lane the way production does since th#1803/pgw#1148:
    a hub RESOLUTION naming the execution lane, not a `#fp8-w8a8` ref."""
    ex._model_resolutions = {ref: (ref, "", W8A8_LANE)}


def _wire_executor(spec, tmp_path, *, ready=True, resident=True, w8a8=False):
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
        if w8a8:
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


def _target_id(ex) -> str:
    (target,) = ex.compile_targets()
    return target.incarnation_id


# ---------------------------------------------------------------------------
# pgw#1032 deleted the whole HOT-ADOPTION suite that lived here (the `_adopt`
# harness, `_assert_failed`, the success/classified-failure sections, the
# target-replaced and republish-reload cases, and the per-target quarantine
# tests). They drove `Executor.handle_model_op`, which answered the hub's
# `ModelOp{ADOPT_COMPILE_CACHE}` push — a push keyed off the COMPUTED cell key,
# a space with no producer since pgw#1010, so no stack has ever dispatched one.
# The behaviour they guarded is gone with the handler; nothing was weakened to
# keep a test green. What remains below is the LIVE path — the worker arming a
# cell it ACQUIRED itself (`aot_cells` fetch-and-filter; th#1702 deletes the
# hub's snapshot attach too, so nothing is pushed to a pod any more) — and the
# runtime-guard revocation the dispatch fence rides.
# ---------------------------------------------------------------------------


def _fake_counters(monkeypatch, *, hits=3, misses=1):
    """Simulate exact-object cache proof inside the endpoint warmup."""
    monkeypatch.setitem(_FAKE_WARM_PROOF, "hits", hits)
    monkeypatch.setitem(_FAKE_WARM_PROOF, "misses", misses)


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
        Settings(bootstrap_worker_jwt="", worker_id="worker"), ex)
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
    picked_ref = "tensorhub/cyberrealistic-pony"
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
    # SDK v2: lora_bucket is a decorator kwarg -> EndpointSpec.lora_bucket,
    # no longer a Compile field.
    turbo = replace(
        _spec(), name="turbo", cls=_TurboEndpoint, method=_TurboEndpoint.run,
        lora_bucket=128,
    )

    async def _send(_msg):
        return None

    ex = Executor([base, turbo], _send)
    for spec, execution_lane, digest in (
        (base, "w8a8", MODEL_DIGEST),
        (turbo, "w8a8-lora128", DIGEST_B),
    ):
        rec = ex._classes[spec.instance_key]
        pipe = _Pipe()
        setattr(pipe, "_cozy_weight_lane", execution_lane)
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
        compile=Compile(shapes=((768, 768),), family=FAMILY, text_len=0),
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
    snapshots = {model_ref: pb.Snapshot(digest=MODEL_DIGEST)}

    instance = asyncio.run(
        ex.ensure_setup(spec, snapshots, arm=_cell_arm(artifact)))
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
    snapshots = {model_ref: pb.Snapshot(digest=MODEL_DIGEST)}

    instance = asyncio.run(
        ex.ensure_setup(spec, snapshots, arm=_cell_arm(artifact)))
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
    snapshots = {model_ref: pb.Snapshot(digest=MODEL_DIGEST)}

    with caplog.at_level("ERROR", logger="gen_worker.executor"):
        instance = asyncio.run(
            ex.ensure_setup(spec, snapshots, arm=_cell_arm(artifact)))
    assert isinstance(instance, _ColdEndpoint)
    assert any(
        "STORE_SERVED_BOOT_COMPILED" in r.message for r in caplog.records
    ), "a store-served boot that hid a real compile must alarm loudly"
    # pgw#923: the alarm has its OWN typed event. It used to ride
    # `ModelEvent{ADOPTED}` with `duration_ms` redefined to mean "inductor
    # compile wall" — a second meaning for the one field the adoption
    # measurement lane (`compile_cache_adopt`) percentiles over, on the only
    # other boot-path sender of that message. Two meanings for one field is
    # how the lane could never be read.
    alarms = [
        m for m in sent
        if m.HasField("activity_update")
        and m.activity_update.kind == "store_served_boot_compiled"
    ]
    (alarm,) = alarms
    assert CACHE_REF in alarm.activity_update.detail
    assert DIGEST_A in alarm.activity_update.detail
    assert alarm.activity_update.duration_ms == 45000
    assert not [
        m for m in sent
        if m.HasField("model_event")
        and m.model_event.state == pb.MODEL_STATE_ADOPTED
        and m.model_event.duration_ms == 45000
    ], "the alarm still hijacks the adoption measurement's duration field"


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
    spec = _cold_spec(Hub("acme/klein-finetune"))
    model_ref = wire_ref(spec.models["pipeline"])
    mint_key = "ek1-" + "d" * 56
    mint_ref = f"root/family-{FAMILY}#{mint_key}"
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
        models={"pipeline": Hub("acme/klein-finetune")},
        compile=Compile(shapes=((768, 768),), family=FAMILY, text_len=0),
    )
    model_ref = wire_ref(spec.models["pipeline"])
    mint_key = "ek1-" + "f" * 56
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
            ref=f"root/family-{FAMILY}#{mint_key}",
            snapshot_digest="sha256:" + "0" * 64, artifact=mint_artifact))

    monkeypatch.setattr(ex, "_enable_compiled", _minting_enable)
    _NoProofEndpoint.setups = _NoProofEndpoint.warmups = _NoProofEndpoint.runs = 0

    # pgw#672: the disproven mandatory-lane mint DEGRADES to explicit eager
    # instead of failing the boot closed — the function stays dispatchable,
    # nothing is advertised, and the identity is quarantined in-process.
    asyncio.run(ex.ensure_setup(
        spec, {model_ref: pb.Snapshot(digest=MODEL_DIGEST)}))
    assert _NoProofEndpoint.warmups == 1, "the proof must have actually run"
    assert ex.compile_targets() == [], "an unproven self-mint must not advertise"
    assert spec.name not in ex.unavailable
    assert ex.serving_tiers() == {spec.name: "eager"}
    assert cc.cell_quarantined_in_process(f"root/family-{FAMILY}#{mint_key}")


def _sim_guard_closure(pipe, cfg, label=""):
    """A closed pgw#681 manifest for rigs whose compiles never touch dynamo."""
    return {"v": 1, "graphs": [{"target": "transformer", "code": "sim",
                                "entry": 0, "guards": []}],
            "verdicts": {}, "leaks": []}


# pgw#1010: `_pending_mint_rig` and its two boot tests (pack-and-publish-only-
# the-proven-capture, unproven-fails-closed-and-never-publishes) covered the
# IN-PROCESS capture — the executor arming a live pipe cold, packing the dir
# its own warmup filled, and publishing those bytes. That whole route built a
# DYNAMO cell, which `aot_cells` rejects by name, so it is deleted rather than
# re-pointed. Its surviving claims live on the delegated route:
# `test_mint_wiring_pgw784.py` (adopt-then-publish, sibling coverage) and
# `test_fleet_cells.py` (the publish/withhold gate).


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
        compile=Compile(shapes=((768, 768),), family=FAMILY, text_len=0),
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
    }, arm=_cell_arm(artifact)))
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


def test_sdxl_w8a8_boot_proves_both_aliases_through_their_own_runs(
    tmp_path, monkeypatch,
):
    """pgw#654: the derived plan runs one warm forward PER ALIAS (causal
    per-alias proof — a sibling's run never certifies an unexercised code
    path), and the class-union contract keeps both aliases on ONE cell, so
    turbo serves compiled on w8a8 instead of failing closed (gap #1)."""
    import gen_worker.executor as executor_mod

    family = "sdxl"
    cell_ref = f"root/family-{family}#inductor-rtx-4090-torch2.9-w8a8"
    artifact = _artifact(tmp_path, family=family)
    model_dir = tmp_path / "sdxl-model"
    model_dir.mkdir()
    calls = {"generate": 0, "generate_turbo": 0}

    @endpoint(
        models={"pipeline": Hub("acme/sdxl")},
        resources=Resources(gpu=True),
        compile=Compile(shapes=((1024, 1024),), family=family, text_len=0),
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
            _record_fake_warm(self.pipeline)
            return _Out(y="turbo")

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
    }, arm=_cell_arm(artifact, ref=cell_ref)))

    assert calls == {"generate": 1, "generate_turbo": 1}
    (target,) = ex.compile_targets()
    assert list(target.function_names) == ["generate", "generate-turbo"]
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
        models={"pipeline": Hub("acme/flux-base")},
        resources=Resources(gpu=True),
        compile=Compile(shapes=((768, 768),), family=FAMILY, text_len=0),
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

    snapshots = {model_ref: pb.Snapshot(digest=MODEL_DIGEST)}
    asyncio.run(ex.ensure_setup(
        generate, snapshots, arm=_cell_arm(artifact, ref=cell_ref)))

    assert calls == {"generate": 1, "edit": 1}
    (target,) = ex.compile_targets()
    assert list(target.function_names) == ["edit", "generate"]
    assert target.active_compile_ref == cell_ref

    # A runtime guard failure revokes the compiled identity — pgw#672: the
    # aliases STAY dispatchable at explicit eager tier (a broken optimization
    # never kills a serving worker); the identity is quarantined in-process so
    # it is never re-armed this boot.
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
    # pgw#672: no reload churn, no function disable — serving continues.
    assert rec.stale is False
    assert "edit" not in ex.unavailable
    assert "generate" not in ex.unavailable
    (tripped,) = ex.compile_targets()
    assert tripped.active_compile_ref == ""
    assert ex.serving_tiers() == {"edit": "eager", "generate": "eager"}
    assert cc.cell_quarantined_in_process(cell_ref)
    # pgw#1032: revocation is state-only. The `adopt_failed:runtime_guard`
    # ModelEvent terminated a hub-commanded adoption operation, and there is no
    # operation to terminate — the tier flip above is the wire-visible signal.
    assert _events(sent, pb.MODEL_STATE_FAILED) == []

    lifecycle = Lifecycle(
        Settings(bootstrap_worker_jwt="", worker_id="worker"), ex)
    failed_delta = lifecycle._state_delta()
    assert "edit" in failed_delta.available_functions
    assert "generate" in failed_delta.available_functions
    assert ex.unavailable["unrelated-hardware-gate"][0] == "hardware_unmet"


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
        resources=Resources(gpu=True),
        compile=Compile(shapes=((768, 768),), family=FAMILY, text_len=0),
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
    monkeypatch.setattr(ex, "_enable_compiled", _seeded_enable(ex, artifact))

    async def scenario() -> None:
        # Hold one of two permits. Setup may stage/arm, but its proof warmup
        # must wait until it can hold the entire worker GPU execution surface.
        await ex._gpu_semaphore.acquire()
        task = asyncio.create_task(ex.ensure_setup(generate, {
            model_ref: pb.Snapshot(digest=MODEL_DIGEST),
        }, arm=_cell_arm(artifact)))
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
        resources=Resources(gpu=True),
        compile=Compile(shapes=((768, 768),), family=FAMILY, text_len=0),
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
    monkeypatch.setattr(ex, "_enable_compiled", _seeded_enable(ex, artifact))

    asyncio.run(ex.ensure_setup(spec, {
        refs["primary"]: pb.Snapshot(digest=MODEL_DIGEST),
        refs["other"]: pb.Snapshot(digest=DIGEST_B),
    }, arm=_cell_arm(artifact)))

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


def test_second_checkpoint_served_from_dynamo_inmemory_cache_proves(
    tmp_path, monkeypatch,
):
    """pgw#637: cell keys are checkpoint-free, so the 2nd checkpoint of an
    already-proven family serves its warmup off dynamo's in-memory compiled
    code — calls>0 with ZERO FX/AOT counter movement. That signature used to
    disprove the cell (`CompiledLaneUnavailableError` -> FnUnavailable
    `compile_cell_failed`) on every multi-checkpoint session. It now proves,
    but ONLY with dynamo confirming live compiled code for this object's
    targets: the sibling-hit guard above must keep failing closed."""
    import gen_worker.executor as executor_mod
    import torch
    from torch._dynamo import eval_frame

    from gen_worker import settings_authority

    # Production boot order (pgw#719): the canonical config is imposed
    # BEFORE any mint/artifact exists, so the cell's recorded seal and the
    # post-bootstrap arm state agree (the executor's pgw#654 TF32 bootstrap
    # is a no-op re-assertion of the same canonical values).
    settings_authority.impose_torch()
    artifact = _artifact(tmp_path)
    model_dir = tmp_path / "family-checkpoints"
    model_dir.mkdir()
    with cc._PROVEN_CELLS_LOCK:
        cc._PROVEN_CELLS.clear()

    def _make_spec(model: str):
        @endpoint(
            models={"pipeline": Hub(model)},
            resources=Resources(gpu=True),
            compile=Compile(shapes=((768, 768),), family=FAMILY, text_len=0),
            )
        class _CheckpointEndpoint:
            def setup(self, pipeline: _LoadablePipe) -> None:
                self.pipeline = pipeline

            def generate(self, ctx, payload: _In) -> _Out:
                self.pipeline.transformer.forward(payload.prompt)
                return _Out(y="ok")

        (spec,) = extract_specs(_CheckpointEndpoint)
        return spec

    async def _send(_msg):
        return None

    async def _download(ref, **kwargs):
        return artifact.parent if ref == CACHE_REF else model_dir

    monkeypatch.setattr(executor_mod, "ensure_local", _download)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch, "compile", lambda fn, **kwargs: fn)
    # Torch boundary only: dynamo genuinely holds an entry for the class
    # `__code__` both checkpoints share (the pgw#637 mechanism). torch.compile
    # is faked here, so the real cache stays empty — say so explicitly.
    monkeypatch.setattr(
        eval_frame, "_debug_get_cache_entry_list", lambda _code: [object()])

    def _run(model: str, hits, pipe):
        spec = _make_spec(model)
        counters = iter(hits)
        monkeypatch.setattr(cc, "inductor_counters", lambda: {
            "fxgraph_cache_hit": next(counters), "fxgraph_cache_miss": 1,
        })
        monkeypatch.setattr(
            provision, "load_slot",
            lambda *a, **kw: provision.SlotLoad(obj=pipe, is_pipeline=True))
        ex = Executor([spec], _send)
        ex.store._cache_dir = tmp_path / "cas"
        monkeypatch.setattr(ex, "_enable_compiled", _seeded_enable(ex, artifact))
        asyncio.run(ex.ensure_setup(spec, {
            wire_ref(spec.models["pipeline"]): pb.Snapshot(digest=MODEL_DIGEST),
        }, arm=_cell_arm(artifact)))
        return ex

    # Checkpoint 1 mints/hits normally and registers the cell as proven here.
    first_pipe = _Pipe()
    first = _run("acme/checkpoint-one", (10, 11), first_pipe)
    (first_target,) = first.compile_targets()
    assert first_target.active_compile_ref == CACHE_REF
    assert cc.cell_proven_in_process(CACHE_REF) is True

    # Checkpoint 2: same cell, new pipeline object, ZERO counter movement.
    second_pipe = _Pipe()
    second = _run("acme/checkpoint-two", (10, 10), second_pipe)
    (second_target,) = second.compile_targets()
    assert second_target.active_compile_ref == CACHE_REF
    assert cc.execution_count(second_pipe) == 1
    assert cc.cache_hit_count(second_pipe) == 0
    assert cc.cache_miss_count(second_pipe) == 0
    # Neither lane was unwrapped: checkpoint 1 still serves compiled.
    assert getattr(second_pipe, cc._MARKER_ATTR, None) is not None
    assert getattr(first_pipe, cc._MARKER_ATTR, None) is not None

    # Same signature WITHOUT live dynamo code is still a disproof.
    monkeypatch.setattr(
        eval_frame, "_debug_get_cache_entry_list", lambda _code: [])
    third_pipe = _Pipe()
    third = _run("acme/checkpoint-three", (10, 10), third_pipe)
    assert third.compile_targets() == []
    assert getattr(third_pipe, cc._MARKER_ATTR, None) is None


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
            "pipeline": Hub("acme/sdxl"),
            "vae": Hub("acme/sdxl-vae"),
        },
        compile=Compile(shapes=((1024, 1024),), family=FAMILY, text_len=0),
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
    }, arm=_cell_arm(artifact, ref=cell_ref)))
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

    spec = _cold_spec(Hub("acme/klein-finetune"))
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

    with pytest.raises(cc.CompiledExecutionLaneUnavailableError, match="self-mint is unavailable"):
        asyncio.run(ex.ensure_setup(
            spec, {model_ref: pb.Snapshot(digest=MODEL_DIGEST)}))
    # The load is the mint's precondition now (the boot warmup IS the mint).
    assert loads == [1]
    assert _ColdEndpoint.runs == 0
    assert ex.unavailable[spec.name][0] == "compile_cell_failed"


def test_w8a8_custom_warmup_proof_attributes_to_all_compatible_siblings(
    tmp_path, monkeypatch,
):
    """gw#603 ruling, amended by pgw#654: proof is a property of the WARMED
    OBJECT and the graph set actually exercised, not of the initiating
    handler's name. A custom object-level warmup's proof attributes to
    EVERY contract-compatible sibling alias of the exact proven object.
    The old ``warmup={...: None}`` per-alias opt-out died with the declared
    warmup surface itself (the warm plan is derived), so there is no
    author skip left to honor — an incompatible sibling is expressed as a
    separate @endpoint class (its own contract), never a skip row.
    Live motivation: LTX serves generate+edit(+extend) from ONE class with
    ONE custom warmup that warms every declared graph — under the ac0bab9
    single-name attribution no >=0.38.8 worker could EVER boot it compiled,
    delivered cells included."""
    import gen_worker.executor as executor_mod

    family = "sdxl"
    cell_ref = f"root/family-{family}#inductor-rtx-4090-torch2.9-w8a8"
    artifact = _artifact(tmp_path, family=family)
    model_dir = tmp_path / "partial-proof-model"
    model_dir.mkdir()

    @endpoint(
        models={"pipeline": Hub("acme/sdxl")},
        resources=Resources(gpu=True),
        compile=Compile(shapes=((1024, 1024),), family=family, text_len=0),
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
            return _Out(y="turbo")

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

    instance = asyncio.run(ex.ensure_setup(generate, {
        model_ref: pb.Snapshot(digest=MODEL_DIGEST),
    }, arm=_cell_arm(artifact, ref=cell_ref)))
    assert isinstance(instance, _SdxlEndpoint)

    # The object proof covers EVERY contract-compatible sibling (pgw#654:
    # the declared-skip carve-out died with the declared warmup surface).
    (target,) = ex.compile_targets()
    proven = set(target.function_names)
    assert by_attr["generate"].name in proven
    assert by_attr["edit"].name in proven
    assert by_attr["generate_turbo"].name in proven
    assert not ex.unavailable


def test_w8a8_custom_warmup_multi_alias_boot_serves_all_siblings(
    tmp_path, monkeypatch,
):
    """The exact live gw#603 shape (LTX): one class, generate+edit aliases,
    ONE custom warmup() covering every declared graph, NO decorator warmup
    rows. Under single-name attribution this boot failed closed forever
    ("expected=['edit','generate'] proven=['edit']") on delivered AND
    self-mint cells alike; under the gw#603 ruling the proven object
    certifies both siblings and the boot serves."""
    import gen_worker.executor as executor_mod

    family = "ltx-shaped"
    cell_ref = f"root/family-{family}#inductor-rtx-4090-torch2.9-w8a8"
    artifact = _artifact(tmp_path, family=family)
    model_dir = tmp_path / "ltx-shaped-model"
    model_dir.mkdir()

    @endpoint(
        models={"pipeline": Hub("acme/ltx-shaped")},
        resources=Resources(gpu=True),
        compile=Compile(shapes=((1024, 1024),), family=family, text_len=0),
    )
    class _LtxShapedEndpoint:
        def setup(self, pipeline: _LoadablePipe) -> None:
            self.pipeline = pipeline

        def warmup(self) -> None:
            # The instance-level synthetic warms EVERY declared graph.
            _record_fake_warm(self.pipeline)

        def generate(self, ctx, payload: _In) -> _Out:
            return _Out(y="ok")

        def edit(self, ctx, payload: _In) -> _Out:
            return _Out(y="ok")

    specs = extract_specs(_LtxShapedEndpoint)
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

    instance = asyncio.run(ex.ensure_setup(generate, {
        model_ref: pb.Snapshot(digest=MODEL_DIGEST),
    }, arm=_cell_arm(artifact, ref=cell_ref)))
    assert isinstance(instance, _LtxShapedEndpoint)
    (target,) = ex.compile_targets()
    assert set(target.function_names) == {
        by_attr["generate"].name, by_attr["edit"].name}
    assert not ex.unavailable


def _merged_execution_lane_endpoint(record_warm):
    """Two w8a8 lane pipes behind ONE handler (the qwen merged shape): the
    declared warmup can only exercise the t2i lane — edit needs an input
    image, so its object has no warmup modality by design (gw#595)."""

    @endpoint(
        models={
            "t2i": Hub("acme/qwen-image"),
            "edit": Hub("acme/qwen-image-edit"),
        },
        resources=Resources(gpu=True),
        compile=Compile(shapes=((1328, 1328),), family="qwen-image", text_len=0),
    )
    class _MergedEndpoint:
        def setup(self, t2i: _LoadablePipe, edit: _LoadablePipe) -> None:
            self.t2i = t2i
            self.edit = edit

        def generate(self, ctx, payload: _In) -> _Out:
            record_warm(self)
            return _Out(y="ok")

    return _MergedEndpoint


def _wire_merged_execution_lane(ex_cls_specs, tmp_path, monkeypatch):
    import gen_worker.executor as executor_mod

    family = "qwen-image"
    cell_ref = f"root/family-{family}#inductor-rtx-4090-torch2.9-w8a8"
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
    return ex, pipes, cell_ref, artifact


def test_w8a8_unexercised_sibling_stays_armed_unproven(
    tmp_path, monkeypatch, caplog,
):
    """gw#595(b): an armed MANDATORY-lane object the warmup has no modality
    to exercise must not block adoption by the sibling that proves; it stays
    armed unproven and is logged explicitly."""
    cls = _merged_execution_lane_endpoint(lambda self: _record_fake_warm(self.t2i))
    specs = extract_specs(cls)
    (generate,) = specs
    ex, pipes, cell_ref, artifact = _wire_merged_execution_lane(
        specs, tmp_path, monkeypatch)

    with caplog.at_level("WARNING"):
        asyncio.run(ex.ensure_setup(generate, {
            wire_ref(generate.models["t2i"]): pb.Snapshot(digest=MODEL_DIGEST),
            wire_ref(generate.models["edit"]): pb.Snapshot(digest=DIGEST_B),
        }, arm=_cell_arm(artifact, ref=cell_ref)))

    targets = {t.model_bindings[0].slot: t for t in ex.compile_targets()}
    assert set(targets) == {"t2i", "edit"}
    assert targets["t2i"].active_compile_ref == cell_ref
    assert targets["t2i"].active_compile_snapshot_digest == DIGEST_A
    # The edit lane is armed (eager is not a w8a8 lane) but unproven.
    assert targets["edit"].active_compile_ref == cell_ref
    assert "armed unproven: no warmup modality" in caplog.text
    assert generate.name not in ex.unavailable


def test_w8a8_exercised_miss_degrades_despite_unexercised_sibling(
    tmp_path, monkeypatch, caplog,
):
    """gw#595(b) keeps gw#586 shut: an EXERCISED object that misses its own
    warmup graph disproves the cell — the unexercised sibling exemption
    never launders a genuine parity defect. pgw#672: the disproof now
    degrades to explicit eager instead of killing the boot."""
    cls = _merged_execution_lane_endpoint(
        lambda self: _record_fake_warm(self.t2i, hits=0, misses=2))
    specs = extract_specs(cls)
    (generate,) = specs
    ex, pipes, cell_ref, artifact = _wire_merged_execution_lane(
        specs, tmp_path, monkeypatch)

    # pgw#672: the disproven proof DEGRADES to explicit eager — setup
    # completes, nothing is advertised, and the gw#608 self-discriminating
    # counts (compile_seconds ~0 = crediting bug vs minutes = recompile)
    # land in the loud degrade record instead of a fatal raise.
    with caplog.at_level(logging.ERROR, logger="gen_worker.executor"):
        asyncio.run(ex.ensure_setup(generate, {
            wire_ref(generate.models["t2i"]): pb.Snapshot(digest=MODEL_DIGEST),
            wire_ref(generate.models["edit"]): pb.Snapshot(digest=DIGEST_B),
        }, arm=_cell_arm(artifact, ref=cell_ref)))
    assert ex.compile_targets() == []
    assert generate.name not in ex.unavailable
    degrade = [
        r for r in caplog.records
        if "did not serve their own warmup graph" in r.getMessage()
    ]
    assert degrade and "compile_seconds=" in degrade[0].getMessage()
    assert "DEGRADED" in degrade[0].getMessage()


def test_store_served_failure_names_diverging_fx_key_component(
    tmp_path, monkeypatch, caplog,
):
    """gw#608 forensics wiring: a store-served warmup-proof failure diffs the
    boot's freshly saved FX entries against the seeded cell's and puts the
    diverging FxGraphHashDetails component in the CompiledLaneUnavailable
    detail. Revert the executor wiring and this goes red."""
    cls = _merged_execution_lane_endpoint(
        lambda self: _record_fake_warm(self.t2i, hits=0, misses=2))
    specs = extract_specs(cls)
    (generate,) = specs
    ex, pipes, cell_ref, artifact = _wire_merged_execution_lane(
        specs, tmp_path, monkeypatch)

    monkeypatch.setattr(
        cc, "fx_cache_failure_report",
        lambda path: ("cell_keys=1; live_keys=2; fresh_keys=1; divergence: "
                      "inductor_config[foo]: cell=cell-value != "
                      "boot=boot-value"))

    # pgw#672: the disproof degrades; the forensics ride the degrade record.
    with caplog.at_level(logging.ERROR, logger="gen_worker.executor"):
        asyncio.run(ex.ensure_setup(generate, {
            wire_ref(generate.models["t2i"]): pb.Snapshot(digest=MODEL_DIGEST),
            wire_ref(generate.models["edit"]): pb.Snapshot(digest=DIGEST_B),
        }, arm=_cell_arm(artifact, ref=cell_ref)))
    assert generate.name not in ex.unavailable
    (detail,) = [
        r.getMessage() for r in caplog.records
        if "did not serve their own warmup graph" in r.getMessage()
    ]
    assert "fx forensics" in detail
    assert "inductor_config[foo]" in detail
    assert "cell=cell-value" in detail and "boot=boot-value" in detail


def test_w8a8_all_objects_unexercised_degrades_to_eager(tmp_path, monkeypatch):
    """gw#595(b): with ZERO proven objects the cell is entirely unverified —
    a warmup that exercises nothing cannot arm anything. pgw#672: the boot
    completes at explicit eager instead of failing closed."""
    cls = _merged_execution_lane_endpoint(lambda self: None)
    specs = extract_specs(cls)
    (generate,) = specs
    ex, pipes, cell_ref, artifact = _wire_merged_execution_lane(
        specs, tmp_path, monkeypatch)

    asyncio.run(ex.ensure_setup(generate, {
        wire_ref(generate.models["t2i"]): pb.Snapshot(digest=MODEL_DIGEST),
        wire_ref(generate.models["edit"]): pb.Snapshot(digest=DIGEST_B),
    }, arm=_cell_arm(artifact, ref=cell_ref)))
    assert ex.compile_targets() == []
    assert generate.name not in ex.unavailable
    assert ex.serving_tiers()[generate.name] == "eager"


def test_production_w8a8_ignores_legacy_compile_environment_fallbacks(
    tmp_path, monkeypatch,
):
    """DesiredInstance and RunJob require Tensorhub-attached exact evidence."""
    import gen_worker.executor as executor_mod

    artifact = _artifact(tmp_path)
    monkeypatch.setenv("GEN_WORKER_COMPILE_CACHE", str(artifact))
    monkeypatch.setenv("GEN_WORKER_COMPILE_CACHE_URL", "https://ignored/cell")
    monkeypatch.setenv("GEN_WORKER_COMPILE_ALLOW_COLD", "1")
    spec = _cold_spec(Hub("acme/klein-finetune"))
    model_ref = wire_ref(spec.models["pipeline"])

    async def _send(_msg):
        return None

    ex = Executor([spec], _send)
    _declare_w8a8_lane(ex, model_ref)
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
    with pytest.raises(cc.CompiledExecutionLaneUnavailableError, match="self-mint is unavailable"):
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
            "acme/klein-finetune")})

    async def _send(_msg):
        return None

    ex = Executor([spec], _send)
    rec = ex._classes[spec.instance_key]
    rec.ready = True
    rec.instance = _Endpoint()
    ref = wire_ref(spec.models["pipeline"])
    _declare_w8a8_lane(ex, ref)
    rec.held_bindings = [("pipeline", ref, MODEL_DIGEST)]
    pipe = _Pipe()  # loader silently lost the W8A8 lane
    selection = type("Selection", (), {
        "ref": CACHE_REF + "-w8a8",
        "snapshot_digest": DIGEST_A,
        "path": tmp_path / "cell.tar.gz",
    })()
    with pytest.raises(cc.CompiledExecutionLaneUnavailableError, match="materialized pipeline lane"):
        ex._install_compile_targets(
            rec, spec, [pipe], {id(pipe): selection}, {id(pipe): {spec.name}},
        )


def test_w8a8_setup_with_no_addressable_compile_object_serves_eager(tmp_path, monkeypatch):
    import gen_worker.executor as executor_mod

    artifact = _artifact(tmp_path)
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    spec = _cold_spec(Hub("acme/klein-finetune"))
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

    # pgw#672: no addressable compile target => the functions serve
    # explicit eager; the boot never dies for a missing optimization.
    asyncio.run(ex.ensure_setup(spec, {
        model_ref: pb.Snapshot(digest=MODEL_DIGEST),
        cell_ref: pb.Snapshot(digest=DIGEST_A),
    }))
    assert ex.compile_targets() == []
    assert spec.name not in ex.unavailable
    assert ex.serving_tiers()[spec.name] == "eager"


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
            "acme/klein-finetune")})
    ex, _sent = _wire_executor(spec, tmp_path, w8a8=True)
    _declare_w8a8_lane(ex, wire_ref(spec.models["pipeline"]))
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
            "acme/klein-finetune")})
    ex, _sent = _wire_executor(spec, tmp_path, w8a8=True)
    _declare_w8a8_lane(ex, wire_ref(spec.models["pipeline"]))
    target = _active_w8a8_target(ex)
    run = _required_run(spec, target, **override)
    with pytest.raises(RetryableError, match=reason):
        ex._validate_required_compile(spec, run)


def test_required_compile_rejects_missing_fence_and_binding_digest_drift(tmp_path):
    spec = replace(
        _spec(), models={"pipeline": Hub(
            "acme/klein-finetune")})
    ex, _sent = _wire_executor(spec, tmp_path, w8a8=True)
    _declare_w8a8_lane(ex, wire_ref(spec.models["pipeline"]))
    target = _active_w8a8_target(ex)
    model_ref = wire_ref(spec.models["pipeline"])

    with pytest.raises(RetryableError, match="required_compile_missing"):
        ex._validate_required_compile(spec, pb.RunJob(function_name=spec.name))

    run = _required_run(spec, target)
    run.snapshots[model_ref].digest = DIGEST_B
    with pytest.raises(RetryableError, match="required_compile_binding_mismatch"):
        ex._validate_required_compile(spec, run)

    other = replace(
        spec, models={"pipeline": Hub("acme/other-klein")})
    other_ref = wire_ref(other.models["pipeline"])
    run = _required_run(spec, target)
    del run.snapshots[model_ref]
    run.snapshots[other_ref].CopyFrom(pb.Snapshot(digest=MODEL_DIGEST))
    with pytest.raises(RetryableError, match="required_compile_binding_mismatch"):
        ex._validate_required_compile(other, run)


def test_runtime_guard_revokes_state_and_quarantines_the_cell(tmp_path):
    """The LIVE revocation path, on a cell this worker armed at boot.

    pgw#1032 removed two things this test used to also assert: the `_adopt`
    call that established the active identity (the hub-commanded adoption is
    gone — a cell is ACQUIRED by the worker's own `aot_cells` discovery and
    armed at boot, which is what `_active_w8a8_target` stands in for), and the
    causal `adopt_failed:runtime_guard` ModelEvent,
    which existed only to terminate an adoption operation nothing issues. What
    revocation MEANS is unchanged and still asserted: the target drops its
    active identity, the record stays serving at explicit eager, the cell is
    quarantined process-wide, and the dispatch fence refuses the pinned run.
    """
    spec = replace(
        _spec(), models={"pipeline": Hub(
            "acme/klein-finetune")})
    ex, sent = _wire_executor(spec, tmp_path, w8a8=True)
    _declare_w8a8_lane(ex, wire_ref(spec.models["pipeline"]))
    active = _active_w8a8_target(ex)
    active_ref = active.active_compile_ref
    active_id = active.incarnation_id
    required_run = _required_run(spec, active)
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
    # pgw#672: the record is NOT marked stale and the aliases stay
    # dispatchable — the object serves explicit eager; the identity is
    # quarantined process-wide, which is the half `fleet_cells` reads on the
    # arm path so this boot never re-arms the cell that just exploded.
    assert rec.stale is False
    assert spec.name not in ex.unavailable
    assert cc.cell_quarantined_in_process(active_ref)
    # pgw#1032: no ModelEvent. The causal terminal belonged to a hub-commanded
    # adoption operation, and there is no operation to terminate.
    assert _events(sent, pb.MODEL_STATE_FAILED) == []

    with pytest.raises(RetryableError, match="required_compile_identity_mismatch"):
        ex._validate_required_compile(spec, required_run)


def test_guard_revocation_between_intake_and_gpu_turn_fails_final_fence(
    tmp_path,
):
    spec = replace(
        _spec(), models={"pipeline": Hub(
            "acme/klein-finetune")})
    ex, sent = _wire_executor(spec, tmp_path, w8a8=True)
    _declare_w8a8_lane(ex, wire_ref(spec.models["pipeline"]))
    active = _active_w8a8_target(ex)
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
    # pgw#1032: revocation is state-only now; the causal ModelEvent went with
    # the adoption operation it used to terminate.
    assert _events(sent, pb.MODEL_STATE_FAILED) == []


def test_target_replacement_between_assignment_and_gpu_never_runs_handler(tmp_path):
    spec = replace(
        _spec(), models={"pipeline": Hub(
            "acme/klein-finetune")})
    ex, sent = _wire_executor(spec, tmp_path, w8a8=True)
    _declare_w8a8_lane(ex, wire_ref(spec.models["pipeline"]))
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


def test_seeding_an_explicit_artifact_writes_the_live_cache(tmp_path):
    """pgw#1035: `cc.prepare` — a None-returning wrapper with no production
    caller — is gone. `seed_artifact` is what the adopt lane calls, and it is
    the same transaction: stage, verify in isolation, activate under the lock."""
    artifact = _artifact(tmp_path)
    meta = cc.seed_artifact(artifact, FAMILY, cache_dir=tmp_path / "cache")
    assert meta is not None and meta["family"] == FAMILY
    assert (tmp_path / "cache" / "compile-cache" / "inductor" / "g" / "code.py").exists()


def test_manifest_carries_compile_block():
    """SDK v2 manifest compile block: text_len/dynamic/shape_contract_digest
    ride along; guidance_scales derive from the payload's CompileAxis
    classes (Compile(guidance_scales=...) is deleted); lora_bucket comes
    from the decorator kwarg — both fn-level and inside the compile block."""
    from gen_worker.discovery.discover import _extract_entries

    @endpoint(
        lora_bucket=64,
        compile=Compile(
            shapes=((768, 768), (1024, 1024)), family=FAMILY, text_len=0,
        ),
    )
    class Ep:
        def gen(self, ctx: RequestContext, data: _AxisIn) -> _Out:
            return _Out()

    (spec,) = extract_specs(Ep)
    assert spec.lora_bucket == 64
    cell = spec.compile_cell()
    assert cell is not None
    (entry,) = _extract_entries(Ep, "testmod")
    assert entry["lora_bucket"] == 64
    assert entry["compile_axes"] == [{
        "field": "guidance_scale",
        "classes": ["cfg_off", "cfg_on"],
        "warm": [0.0, 5.0],
    }]
    assert entry["compile"] == {
        "family": FAMILY,
        "shapes": [[768, 768], [1024, 1024]],
        "targets": ["transformer", "vae.decode"],
        "text_len": 0,
        # pgw#654 gap #6: the class's per-lane pin union rides too.
        "text_lens": [0],
        "guidance_scales": [0.0, 5.0],
        "shape_contract_digest": cell.contract_digest(),
        "lora_bucket": 64,
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
        ref = "root/family-fam#inductor-rtx-4090-torch2.9"
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


# pgw#1032: `test_fresh_boot_advertises_candidate_cell_lookups` is deleted with
# `Executor.cell_lookups`. gw#605 wrote it to keep a fresh boot advertising
# pre-load CANDIDATE keys so the hub could attach a stored cell before setup —
# but those candidates are COMPUTED (kind="inductor") keys, and since pgw#1010
# no mint publishes into that space, so the attach it protected could never
# have resolved. Cold-boot adoption is `aot_cells` fetch-and-filter now
# (pgw#904 owns its replacement), which does not go through the hub's key
# lookup at all.


# ---------------------------------------------------------------------------
# gw#612: multi-lane self-mint — sibling handoff must complete, publish gated
# on full capture coverage (ie#501 run 26 / gw#611 qwen variant)
# ---------------------------------------------------------------------------


# pgw#1010: the two-lane IN-PROCESS mint boots that stood here
# (`_dual_mint_boot` / `_routed_mint_boot` + three publish/withhold tests) are
# deleted with the route they drove. Each armed live pipes cold, packed one
# shared inductor capture and published the union as a family cell — a DYNAMO
# artifact with no consumer. gw#612's sibling-coverage rule, which is the claim
# they carried, is asserted on the surviving delegated route in
# `test_mint_wiring_pgw784.py::test_shared_sharers_mint_one_cell_between_them`
# and in `test_fleet_cells.py`'s withhold/publish gate.
