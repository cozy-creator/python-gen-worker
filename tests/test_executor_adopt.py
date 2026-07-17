"""th#567 hot adoption: MODEL_OP_KIND_ADOPT_COMPILE_CACHE re-wraps resident
modules in place — verified seed, one warmup, ADOPTED report; ANY failure
stays eager with a classified adopt_failed:<reason>. Plus th#569 boot-attach:
a compile-cache snapshot on RunJob.snapshots reaches compile_cache.enable."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path

import msgspec
import pytest

from gen_worker import Compile, RequestContext, endpoint
from gen_worker import compile_cache as cc
from gen_worker.api.binding import Hub
from gen_worker.api.binding import wire_ref
from gen_worker.executor import Executor, _Job
from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.registry import EndpointSpec

FAMILY = "flux2-klein-4b"
CACHE_REF = f"_system/family-{FAMILY}#inductor-rtx-4090-torch2.9"
MODEL_REF = "acme/klein-finetune:latest"
DIGEST_A = "blake3:" + "a" * 64
DIGEST_B = "blake3:" + "b" * 64


class _In(msgspec.Struct):
    prompt: str = ""


class _Out(msgspec.Struct):
    y: str = ""


class _Pipe:
    pass


class _Endpoint:
    warmups = 0

    def setup(self, pipeline: str) -> None:  # pragma: no cover
        pass

    def warmup(self) -> None:
        type(self).warmups += 1

    def run(self, ctx, payload: _In) -> _Out:  # pragma: no cover
        return _Out()


def _spec(compile_cfg=None) -> EndpointSpec:
    return EndpointSpec(
        name="ep", method=_Endpoint.run, kind="inference",
        payload_type=_In, output_mode="single", cls=_Endpoint,
        attr_name="run", models={"pipeline": Hub("acme/klein-finetune")},
        compile=compile_cfg or Compile(shapes=((768, 768),), family=FAMILY),
    )


def _artifact(tmp_path: Path, **meta_overrides) -> Path:
    cap = tmp_path / "cap"
    (cap / "inductor" / "g").mkdir(parents=True)
    (cap / "inductor" / "g" / "code.py").write_text("x")
    (cap / "triton").mkdir()
    cfg = Compile(shapes=((768, 768),), family=FAMILY)
    signature, weight_contract = cc.execution_contract(_Pipe(), cfg)
    meta = cc.artifact_metadata(
        family=FAMILY, shapes=cfg.shapes, targets=cfg.targets,
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
    if ready:
        rec.instance = _Endpoint()
        rec.ready = True
    if resident:
        ex.store.residency.track_vram(wire_ref(spec.models["pipeline"]), _Pipe(), vram_bytes=1)

    async def _fake_ensure_local(ref, snapshot=None, *, binding=None) -> Path:
        return tmp_path / "snap"

    ex.store.ensure_local = _fake_ensure_local  # type: ignore[method-assign]
    return ex, sent


def _events(sent, state):
    return [m.model_event for m in sent
            if m.WhichOneof("msg") == "model_event" and m.model_event.state == state]


def _adopt(ex, ref=CACHE_REF, digest=DIGEST_A):
    asyncio.run(ex.handle_model_op(
        pb.ModelOp(
            op=pb.MODEL_OP_KIND_ADOPT_COMPILE_CACHE,
            ref=ref,
            snapshot=pb.Snapshot(digest=digest),
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
    """Simulate inductor counters advancing across the adoption warmup."""
    state = {"n": 0}

    def counters():
        state["n"] += 1
        if state["n"] == 1:  # before-warmup snapshot
            return {"fxgraph_cache_hit": 10, "fxgraph_cache_miss": 5}
        return {"fxgraph_cache_hit": 10 + hits, "fxgraph_cache_miss": 5 + misses}

    monkeypatch.setattr(cc, "inductor_counters", counters)


def test_adopt_success_rewraps_warms_and_reports(tmp_path, monkeypatch):
    _artifact(tmp_path)
    spec = _spec()
    ex, sent = _wire_executor(spec, tmp_path)
    _Endpoint.warmups = 0

    applied: list[tuple] = []

    def _fake_apply(pipeline, cfg, *, cache_ready, guard=True):
        applied.append((pipeline, cfg, cache_ready))
        return True

    monkeypatch.setattr(cc, "apply", _fake_apply)
    _fake_counters(monkeypatch, hits=3, misses=1)
    _adopt(ex)

    adopted = _events(sent, pb.MODEL_STATE_ADOPTED)
    assert len(adopted) == 1
    assert adopted[0].ref == CACHE_REF
    assert adopted[0].snapshot_digest == DIGEST_A
    assert adopted[0].duration_ms >= 0
    # cache_hits/cache_misses/warmup_s are computed internally (gating the
    # cache_miss rollback below) but intentionally not sent on the wire —
    # the orchestrator has no reader for them (pgw#514/P3).
    assert adopted[0].cache_hits == 0
    assert adopted[0].cache_misses == 0
    assert adopted[0].warmup_s == 0
    assert not _events(sent, pb.MODEL_STATE_FAILED)
    assert len(applied) == 1 and applied[0][2] is True
    assert isinstance(applied[0][0], _Pipe)
    assert _Endpoint.warmups == 1


def test_adopt_zero_cache_hits_rolls_back_and_fails(tmp_path, monkeypatch):
    """gw#391 honest failure mode: ADOPTED-while-silently-eager is impossible.
    A warmup observing zero fxgraph hits unwraps back to eager and reports
    adopt_failed:cache_miss."""
    _artifact(tmp_path)
    spec = _spec()
    ex, sent = _wire_executor(spec, tmp_path)
    monkeypatch.setattr(cc, "apply", lambda *a, **k: True)
    _fake_counters(monkeypatch, hits=0, misses=2)
    unwrapped: list = []
    monkeypatch.setattr(cc, "unwrap", lambda obj: unwrapped.append(obj))

    _adopt(ex)
    failed = _events(sent, pb.MODEL_STATE_FAILED)
    assert failed and failed[-1].error == "adopt_failed:cache_miss"
    assert not _events(sent, pb.MODEL_STATE_ADOPTED)
    assert any(isinstance(o, _Pipe) for o in unwrapped)  # rollback ran


def test_adopt_prep_mode_drift_is_key_mismatch(tmp_path, monkeypatch):
    """A cell traced under a different low-VRAM prep mode can only miss —
    rejected deterministically before any warmup (gw#391 parity)."""
    from gen_worker.models.memory import _COZY_MODE_ATTR

    _artifact(tmp_path, low_vram_mode="off")
    spec = _spec()
    ex, sent = _wire_executor(spec, tmp_path)
    obj = ex.store.residency.obj(wire_ref(spec.models["pipeline"]))
    setattr(obj, _COZY_MODE_ATTR, "vae_only")
    monkeypatch.setattr(cc, "apply", lambda *a, **k: pytest.fail("must not re-wrap"))

    _adopt(ex)
    failed = _events(sent, pb.MODEL_STATE_FAILED)
    assert failed and failed[-1].error == "adopt_failed:key_mismatch"
    assert failed[-1].snapshot_digest == DIGEST_A
    assert not _events(sent, pb.MODEL_STATE_ADOPTED)


def test_adopt_without_warmup_is_unprovable(tmp_path, monkeypatch):
    """An endpoint without warmup() cannot prove the cell hits — the honest
    answer is adopt_failed:no_warmup, never a blind ADOPTED."""

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
    monkeypatch.setattr(cc, "apply", lambda *a, **k: True)
    _fake_counters(monkeypatch, hits=5, misses=0)  # counters moved elsewhere: still unprovable

    _adopt(ex)
    failed = _events(sent, pb.MODEL_STATE_FAILED)
    assert failed and failed[-1].error == "adopt_failed:no_warmup"
    assert not _events(sent, pb.MODEL_STATE_ADOPTED)


def test_adopt_prep_mode_drift_rejected(tmp_path, monkeypatch):
    """gw#391: the producer's low-VRAM prep mode is part of the graph key — a
    pipeline prepped under a different mode can only miss, so adoption rejects
    it deterministically (key_mismatch) before any wrap or warmup."""
    _artifact(tmp_path, low_vram_mode="off")
    spec = _spec()
    ex, sent = _wire_executor(spec, tmp_path)
    obj = ex.store.residency.obj(wire_ref(spec.models["pipeline"]))
    obj._cozy_low_vram_mode = "vae_only"
    applied: list = []
    monkeypatch.setattr(cc, "apply", lambda *a, **k: applied.append(1) or True)

    _adopt(ex)
    failed = _events(sent, pb.MODEL_STATE_FAILED)
    assert failed and failed[-1].error == "adopt_failed:key_mismatch"
    assert not _events(sent, pb.MODEL_STATE_ADOPTED)
    assert not applied  # rejected before the wrap


def test_adopt_failed_warmup_reports_reason(tmp_path, monkeypatch):
    _artifact(tmp_path)
    spec = _spec()
    ex, sent = _wire_executor(spec, tmp_path)
    monkeypatch.setattr(cc, "apply", lambda *a, **k: True)
    monkeypatch.setattr(_Endpoint, "warmup", lambda self: 1 / 0)

    _adopt(ex)
    failed = _events(sent, pb.MODEL_STATE_FAILED)
    assert failed and failed[-1].error == "adopt_failed:warmup"
    assert not _events(sent, pb.MODEL_STATE_ADOPTED)


# ---------------------------------------------------------------------------
# classified failures
# ---------------------------------------------------------------------------


def test_adopt_key_mismatch_stays_eager(tmp_path, monkeypatch):
    _artifact(tmp_path, sku="not-this-gpu")
    spec = _spec()
    ex, sent = _wire_executor(spec, tmp_path)
    monkeypatch.setattr(cc, "apply", lambda *a, **k: pytest.fail("must not re-wrap"))

    _adopt(ex)
    failed = _events(sent, pb.MODEL_STATE_FAILED)
    assert failed and failed[-1].error == "adopt_failed:key_mismatch"


def test_adopt_refuses_while_jobs_in_flight(tmp_path):
    _artifact(tmp_path)
    spec = _spec()
    ex, sent = _wire_executor(spec, tmp_path)
    ex.jobs[("r1", 1)] = _Job(request_id="r1", attempt=1, spec=spec)

    calls: list[str] = []

    async def _no_download(ref, snapshot=None, *, binding=None):  # pragma: no cover
        calls.append(ref)
        return tmp_path / "snap"

    ex.store.ensure_local = _no_download  # type: ignore[method-assign]
    _adopt(ex)
    failed = _events(sent, pb.MODEL_STATE_FAILED)
    assert failed and failed[-1].error == "adopt_failed:model_in_use"
    assert calls == []  # refused before any download


def test_adopt_no_endpoint_for_family(tmp_path):
    spec = _spec(Compile(shapes=((768, 768),), family="sdxl"))
    ex, sent = _wire_executor(spec, tmp_path)
    _adopt(ex)  # ref names flux2-klein-4b; only sdxl is declared
    failed = _events(sent, pb.MODEL_STATE_FAILED)
    assert failed and failed[-1].error == "adopt_failed:no_endpoint"


def test_adopt_not_resident(tmp_path):
    spec = _spec()
    ex, sent = _wire_executor(spec, tmp_path, ready=False, resident=False)
    _adopt(ex)
    failed = _events(sent, pb.MODEL_STATE_FAILED)
    assert failed and failed[-1].error == "adopt_failed:not_resident"


def test_adopt_bad_ref(tmp_path):
    spec = _spec()
    ex, sent = _wire_executor(spec, tmp_path)
    _adopt(ex, ref="acme/not-a-cache:latest")
    failed = _events(sent, pb.MODEL_STATE_FAILED)
    assert failed and failed[-1].error == "adopt_failed:bad_ref"


def test_adopt_artifact_missing(tmp_path):
    (tmp_path / "snap").mkdir()  # empty snapshot dir: no tarball
    spec = _spec()
    ex, sent = _wire_executor(spec, tmp_path)
    _adopt(ex)
    failed = _events(sent, pb.MODEL_STATE_FAILED)
    assert failed and failed[-1].error == "adopt_failed:artifact_missing"


@pytest.mark.parametrize("digest", [None, "", "   "])
def test_adopt_missing_digest_fails_before_work_or_identity_inference(
    tmp_path, monkeypatch, digest,
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

    asyncio.run(ex.handle_model_op(pb.ModelOp(**kwargs)))

    failed = _events(sent, pb.MODEL_STATE_FAILED)
    assert len(failed) == 1
    assert failed[0].error == "adopt_failed:missing_snapshot_digest"
    assert failed[0].snapshot_digest == ""
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
    assert failed[0].error == "adopt_failed:runtimeerror"
    assert failed[0].snapshot_digest == DIGEST_B


def test_adopt_same_ref_out_of_order_keeps_each_operation_digest(tmp_path, monkeypatch):
    """A late result for digest A must remain distinguishable from newer B."""
    _artifact(tmp_path)
    spec = _spec()
    ex, sent = _wire_executor(spec, tmp_path)
    monkeypatch.setattr(cc, "apply", lambda *args, **kwargs: True)

    counter = {"hits": 0}

    def _counters():
        counter["hits"] += 1
        return {"fxgraph_cache_hit": counter["hits"], "fxgraph_cache_miss": 0}

    monkeypatch.setattr(cc, "inductor_counters", _counters)

    async def run():
        a_started = asyncio.Event()
        release_a = asyncio.Event()

        async def _ensure(ref, snapshot=None, *, binding=None):
            if snapshot is not None and snapshot.digest == DIGEST_A:
                a_started.set()
                await release_a.wait()
            return tmp_path / "snap"

        ex.store.ensure_local = _ensure  # type: ignore[method-assign]
        op_a = pb.ModelOp(
            op=pb.MODEL_OP_KIND_ADOPT_COMPILE_CACHE,
            ref=CACHE_REF,
            snapshot=pb.Snapshot(digest=DIGEST_A),
        )
        op_b = pb.ModelOp(
            op=pb.MODEL_OP_KIND_ADOPT_COMPILE_CACHE,
            ref=CACHE_REF,
            snapshot=pb.Snapshot(digest=DIGEST_B),
        )

        task_a = asyncio.create_task(ex.handle_model_op(op_a))
        await a_started.wait()
        await ex.handle_model_op(op_b)
        release_a.set()
        await task_a

    asyncio.run(run())

    adopted = _events(sent, pb.MODEL_STATE_ADOPTED)
    assert [event.ref for event in adopted] == [CACHE_REF, CACHE_REF]
    assert [event.snapshot_digest for event in adopted] == [DIGEST_B, DIGEST_A]


def test_old_worker_semantics_unknown_kind_is_silent(tmp_path):
    """proto3 forward-compat guarantee the hub relies on: an op kind this
    worker doesn't know produces no ModelEvent at all."""
    spec = _spec()
    ex, sent = _wire_executor(spec, tmp_path)
    asyncio.run(ex.handle_model_op(pb.ModelOp(op=99, ref=CACHE_REF)))
    assert not [m for m in sent if m.WhichOneof("msg") == "model_event"]


# ---------------------------------------------------------------------------
# th#569 boot-attach: cache snapshot on RunJob.snapshots
# ---------------------------------------------------------------------------


def test_fetch_compile_snapshot_finds_family_cache(tmp_path):
    artifact = _artifact(tmp_path)
    spec = _spec()
    ex, _sent = _wire_executor(spec, tmp_path)
    snapshots = {
        MODEL_REF: pb.Snapshot(digest="blake3:aa"),
        CACHE_REF: pb.Snapshot(digest="blake3:bb"),
    }
    got = asyncio.run(ex._fetch_compile_snapshot(spec, snapshots))
    assert got == artifact


def test_fetch_compile_snapshot_ignores_other_families(tmp_path):
    _artifact(tmp_path)
    spec = _spec()
    ex, _sent = _wire_executor(spec, tmp_path)
    snapshots = {"_system/family-sdxl#inductor-rtx-4090-torch2.9": pb.Snapshot()}
    assert asyncio.run(ex._fetch_compile_snapshot(spec, snapshots)) is None
    assert asyncio.run(ex._fetch_compile_snapshot(spec, None)) is None


def test_fetch_compile_snapshot_selects_exact_w8a8_lane(tmp_path):
    spec = replace(
        _spec(), models={"pipeline": Hub(
            "acme/klein-finetune", flavor="fp8-w8a8")},
    )
    ex, _sent = _wire_executor(spec, tmp_path)
    plain = tmp_path / "plain"
    w8a8 = tmp_path / "w8a8"
    plain.mkdir()
    w8a8.mkdir()
    (plain / "plain.tar.gz").write_bytes(b"plain")
    wanted = w8a8 / "w8a8.tar.gz"
    wanted.write_bytes(b"w8a8")
    seen: list[str] = []

    async def _ensure(ref, snapshot=None, *, binding=None):
        seen.append(ref)
        return w8a8 if ref.endswith("-w8a8") else plain

    ex.store.ensure_local = _ensure  # type: ignore[method-assign]
    plain_ref = f"_system/family-{FAMILY}#inductor-rtx-4090-torch2.9"
    w8a8_ref = plain_ref + "-w8a8"
    snapshots = {
        plain_ref: pb.Snapshot(),
        w8a8_ref: pb.Snapshot(),
        "acme/klein-finetune#fp8-w8a8": pb.Snapshot(),
    }
    got = asyncio.run(ex._fetch_compile_snapshot(spec, snapshots))
    assert got == wanted
    assert seen == [w8a8_ref]


def test_fetch_compile_snapshot_plain_lane_ignores_w8a8_cell(tmp_path):
    spec = _spec()
    ex, _sent = _wire_executor(spec, tmp_path)
    plain = tmp_path / "plain"
    w8a8 = tmp_path / "w8a8"
    plain.mkdir()
    w8a8.mkdir()
    wanted = plain / "plain.tar.gz"
    wanted.write_bytes(b"plain")
    (w8a8 / "w8a8.tar.gz").write_bytes(b"w8a8")
    seen: list[str] = []

    async def _ensure(ref, snapshot=None, *, binding=None):
        seen.append(ref)
        return w8a8 if ref.endswith("-w8a8") else plain

    ex.store.ensure_local = _ensure  # type: ignore[method-assign]
    plain_ref = f"_system/family-{FAMILY}#inductor-rtx-4090-torch2.9"
    w8a8_ref = plain_ref + "-w8a8"
    snapshots = {
        w8a8_ref: pb.Snapshot(),
        plain_ref: pb.Snapshot(),
        "acme/unselected#fp8-w8a8": pb.Snapshot(),
    }
    got = asyncio.run(ex._fetch_compile_snapshot(spec, snapshots))
    assert got == wanted
    assert seen == [plain_ref]


def test_prepare_with_explicit_artifact_seeds(tmp_path, monkeypatch):
    monkeypatch.delenv(cc.ENV_CACHE_PATH, raising=False)
    monkeypatch.delenv(cc.ENV_CACHE_URL, raising=False)
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
