"""pgw#532: worker-side dynamic slot materialization (the last th#767 piece).

The fc157 live failure: a hub-connected worker materialized a declared
``Slot``'s ``default_checkpoint`` from its RAW upstream (Civitai 827184) at
setup -> ``civitai_not_found`` -> setup failed -> every function unavailable,
cascading ``load_failed`` onto the healthy hub-binding refs. Independently,
``resolved_models[slot]`` (the hub-resolved pick the scheduler routed and
pre-drove) was never injected — a pick would silently have run the default's
weights.

Covered here:
  1. boot: a hub-connected worker NEVER touches a Slot's raw upstream
     default (no Civitai/HF fetch, no eager setup); the function still
     advertises available (dispatch materializes the hub-resolved ref).
  2. dispatch pick != default: the executor loads the PICKED checkpoint and
     ``ctx.slots[name].ref`` reflects the pick, not the code default.
  3. instance-per-pick residency: two picks = two resident instances (one
     ``setup()`` per (class, resolved pick)); UNLOAD evicts one whole
     instance while the other stays warm; a later LOAD for the evicted pick
     matches its derived record and re-sets it up.
  4. default-only path with a CAS default_checkpoint: unchanged (one
     instance, base record, ctx.slots ref = declared default).
  5. an unusable (non-CAS) resolved_models stamp with a raw upstream
     default fails the job RETRYABLE — never an upstream self-fetch.
  6. hub-less (`cozy run`) path: the raw default_checkpoint still resolves
     through its upstream provider, unchanged.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import msgspec

from gen_worker.api.binding import HF, Civitai, Hub, ModelRef
from gen_worker.api.slot import Slot
from gen_worker.config.settings import Settings
from gen_worker.executor import Executor
from gen_worker.families.base import FamilyDefaults, family
from gen_worker.lifecycle import Lifecycle
from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.registry import EndpointSpec


@family("pgw532-testfam")
class _Fam(FamilyDefaults):
    steps: int = 7


class _StubPipeline:
    """Slot compat class only — setup() slots are str-annotated so the
    executor injects local PATHS (no torch / placement machinery)."""


class _In(msgspec.Struct):
    prompt: str = ""
    model: str = ""


class _Out(msgspec.Struct):
    slot_ref: str
    pipeline_path: str


PIPE_DEFAULT_RAW = Civitai("827184", version="2883731")   # WAI-Illustrious (raw upstream)
VAE_DEFAULT_RAW = HF("madebyollin/sdxl-vae-fp16-fix")     # raw upstream
PIPE_DEFAULT_CAS = Hub("acme/default-model", tag="prod")  # CAS-declared default

PICK_A = "tensorhub/cyberrealistic-pony:prod"
PICK_B = "tensorhub/wai-illustrious:prod"
VAE_MIRROR = "tensorhub/sdxl-vae-fp16-fix:prod"


def _slot_spec(
    name: str,
    setup_calls: List[Tuple[str, str]],
    *,
    pipe_default: ModelRef = PIPE_DEFAULT_RAW,
    vae_default: ModelRef = VAE_DEFAULT_RAW,
) -> EndpointSpec:
    class Endpoint:
        def setup(self, pipeline: str, vae: str) -> None:
            # setup-held state, exactly the sdxl template's shape: the
            # instance binds whatever THIS setup received, forever.
            self.pipeline_path = pipeline
            setup_calls.append((name, pipeline))

        def generate(self, ctx: Any, payload: _In) -> _Out:
            resolved = ctx.slots["pipeline"]
            ref = resolved.ref
            return _Out(
                slot_ref=f"{ref.source}:{ref.path}:{ref.tag}",
                pipeline_path=self.pipeline_path,
            )

    slots = {
        "pipeline": Slot(
            _StubPipeline, selected_by="model",
            default_checkpoint=pipe_default, default_config=_Fam(),
        ),
        "vae": Slot(_StubPipeline, default_checkpoint=vae_default,
                    default_config=_Fam()),
    }
    return EndpointSpec(
        name=name, method=Endpoint.generate, kind="inference",
        payload_type=_In, output_mode="single", cls=Endpoint,
        attr_name="generate",
        models={"pipeline": pipe_default, "vae": vae_default},
        slots=slots,
        slot_family={"pipeline": "pgw532-testfam", "vae": "pgw532-testfam"},
    )


def _snapshot(digest: str = "ab" * 32) -> pb.Snapshot:
    return pb.Snapshot(digest=digest, files=[pb.SnapshotFile(
        path="model.safetensors", size_bytes=5, blake3="cd" * 32,
        url="http://r2.invalid/presigned")])


def _harness(tmp_path: Path, monkeypatch, setup_calls: List[Tuple[str, str]]):
    """Executor over the REAL ModelStore/dispatch orchestration; only the
    network download primitive is faked, and it REFUSES raw-upstream refs —
    the gw#465 invariant under test."""
    sent: List[pb.WorkerMessage] = []
    downloads: List[Dict[str, Any]] = []

    async def _send(msg: pb.WorkerMessage) -> None:
        sent.append(msg)

    ex = Executor([_slot_spec("generate", setup_calls)], _send)

    async def _fake_download(ref: str, **kwargs: Any) -> Path:
        provider = kwargs.get("provider")
        assert provider in (None, "tensorhub"), (
            f"hub-connected worker attempted an UPSTREAM fetch: "
            f"ref={ref!r} provider={provider!r}")
        assert "827184" not in ref and "madebyollin" not in ref, (
            f"raw upstream default leaked into the hub-connected path: {ref!r}")
        downloads.append({"ref": ref, **kwargs})
        p = tmp_path / ref.replace("/", "_").replace(":", "_")
        p.mkdir(parents=True, exist_ok=True)
        return p

    import gen_worker.executor as ex_mod
    monkeypatch.setattr(ex_mod, "ensure_local", _fake_download)
    monkeypatch.setattr(ex_mod, "_MISSING_SNAPSHOT_WAIT_S", 0.2)
    return ex, sent, downloads


def _run_job(rid: str, *, model: str = "", models: List[pb.ModelBinding],
             snapshots: Dict[str, pb.Snapshot] = {}) -> pb.RunJob:
    return pb.RunJob(
        request_id=rid, attempt=1, function_name="generate",
        input_payload=msgspec.msgpack.encode(_In(prompt="a cat", model=model)),
        models=models, snapshots=snapshots,
    )


async def _dispatch(ex: Executor, sent: List[pb.WorkerMessage], run: pb.RunJob) -> pb.JobResult:
    await ex.handle_run_job(run)
    job = ex.jobs[(run.request_id, run.attempt)]
    assert job.task is not None
    await job.task
    results = [m.job_result for m in sent if m.WhichOneof("msg") == "job_result"
               and m.job_result.request_id == run.request_id]
    assert results, f"no job_result for {run.request_id}"
    return results[-1]


def _ready_records(ex: Executor) -> List[Any]:
    seen: set = set()
    out = []
    for rec in ex._classes.values():
        if id(rec) in seen:
            continue
        seen.add(id(rec))
        if rec.ready:
            out.append(rec)
    return out


# ---------------------------------------------------------------------------
# 1. boot: never touch the raw upstream default; function still advertises
# ---------------------------------------------------------------------------


def test_boot_never_fetches_raw_default_and_advertises(tmp_path, monkeypatch, caplog) -> None:
    setup_calls: List[Tuple[str, str]] = []
    ex, sent, downloads = _harness(tmp_path, monkeypatch, setup_calls)

    ensured: List[str] = []
    orig = ex.store.ensure_local

    async def _spy(ref: str, *a: Any, **kw: Any) -> Path:
        ensured.append(ref)
        return await orig(ref, *a, **kw)

    monkeypatch.setattr(ex.store, "ensure_local", _spy)

    lc = Lifecycle(Settings(orchestrator_public_addr="localhost:1"), ex)
    lc.hardware = {"gpu_count": 1, "gpu_total_mem": 32 * 1024**3,
                   "gpu_free_mem": 30 * 1024**3, "gpu_sm": "90", "installed_libs": []}
    with caplog.at_level(logging.INFO, logger="gen_worker.lifecycle"):
        asyncio.run(lc.startup())

    assert ensured == [], f"boot materialized Slot seeds: {ensured}"
    assert setup_calls == [], f"boot eagerly set up a dynamic-slot spec: {setup_calls}"
    assert ex.available_functions() == ["generate"]
    assert ex.loading_functions() == []
    assert "generate" not in ex.unavailable
    failed = [m.model_event for m in sent if m.WhichOneof("msg") == "model_event"
              and m.model_event.state == pb.MODEL_STATE_FAILED]
    assert not failed, f"boot emitted failures: {failed}"


# ---------------------------------------------------------------------------
# 2. dispatch pick != default loads the PICKED weights; ctx.slots shows it
# ---------------------------------------------------------------------------


def test_dispatch_pick_loads_picked_checkpoint(tmp_path, monkeypatch) -> None:
    setup_calls: List[Tuple[str, str]] = []
    ex, sent, downloads = _harness(tmp_path, monkeypatch, setup_calls)

    async def _run() -> pb.JobResult:
        return await _dispatch(ex, sent, _run_job(
            "r1", model="cyberrealistic-pony",
            models=[pb.ModelBinding(slot="pipeline", ref=PICK_A),
                    pb.ModelBinding(slot="vae", ref=VAE_MIRROR)],
            snapshots={PICK_A: _snapshot("aa" * 32), VAE_MIRROR: _snapshot("bb" * 32)},
        ))

    res = asyncio.run(_run())
    assert res.status == pb.JOB_STATUS_OK, res.safe_message
    out = msgspec.msgpack.decode(res.inline, type=_Out)
    # ctx.slots reflects the RESOLVED pick, not the code default.
    assert out.slot_ref == "tensorhub:tensorhub/cyberrealistic-pony:prod"
    # setup() received the PICK's materialized path.
    assert "cyberrealistic-pony" in out.pipeline_path
    assert setup_calls == [("generate", out.pipeline_path)]
    assert {d["ref"] for d in downloads} == {PICK_A, VAE_MIRROR}


# ---------------------------------------------------------------------------
# 3. instance-per-pick: two picks = two instances; UNLOAD evicts one whole
#    instance; LOAD re-materializes the evicted pick's derived record
# ---------------------------------------------------------------------------


def test_two_picks_two_instances_then_lru_evicts_whole_instance(tmp_path, monkeypatch) -> None:
    setup_calls: List[Tuple[str, str]] = []
    ex, sent, downloads = _harness(tmp_path, monkeypatch, setup_calls)

    async def _run() -> None:
        r1 = await _dispatch(ex, sent, _run_job(
            "r1", model="a",
            models=[pb.ModelBinding(slot="pipeline", ref=PICK_A),
                    pb.ModelBinding(slot="vae", ref=VAE_MIRROR)],
            snapshots={PICK_A: _snapshot("aa" * 32), VAE_MIRROR: _snapshot("bb" * 32)}))
        assert r1.status == pb.JOB_STATUS_OK, r1.safe_message
        r2 = await _dispatch(ex, sent, _run_job(
            "r2", model="b",
            models=[pb.ModelBinding(slot="pipeline", ref=PICK_B),
                    pb.ModelBinding(slot="vae", ref=VAE_MIRROR)],
            snapshots={PICK_B: _snapshot("cc" * 32), VAE_MIRROR: _snapshot("bb" * 32)}))
        assert r2.status == pb.JOB_STATUS_OK, r2.safe_message

        # setup() ran once per (class, resolved pick) — setup-held state
        # (self.pipeline_path) is coherent per instance.
        assert len(setup_calls) == 2
        picked_paths = [p for _, p in setup_calls]
        assert any("cyberrealistic-pony" in p for p in picked_paths)
        assert any("wai-illustrious" in p for p in picked_paths)
        assert len(_ready_records(ex)) == 2, "two picks must be two resident instances"

        # Same pick again: the resident instance serves — NO third setup.
        r3 = await _dispatch(ex, sent, _run_job(
            "r3", model="a",
            models=[pb.ModelBinding(slot="pipeline", ref=PICK_A),
                    pb.ModelBinding(slot="vae", ref=VAE_MIRROR)]))
        assert r3.status == pb.JOB_STATUS_OK, r3.safe_message
        assert len(setup_calls) == 2, "resident pick must not re-run setup()"

        # Hub-directed UNLOAD of pick A vacates THAT instance only.
        await ex.handle_model_op(pb.ModelOp(op=pb.MODEL_OP_KIND_UNLOAD, ref=PICK_A))
        ready = _ready_records(ex)
        assert len(ready) == 1, "UNLOAD must evict exactly one pick's instance"
        held = {r for rec in ready for r in rec.held_refs}
        assert PICK_B in held and PICK_A not in held

        # LOAD for the evicted pick matches its derived record and re-sets up.
        await ex.handle_model_op(pb.ModelOp(
            op=pb.MODEL_OP_KIND_LOAD, ref=PICK_A, snapshot=_snapshot("aa" * 32)))
        assert len(setup_calls) == 3
        assert len(_ready_records(ex)) == 2

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# 4. default-only path with a CAS default_checkpoint: unchanged
# ---------------------------------------------------------------------------


def test_cas_default_dispatch_uses_declared_binding(tmp_path, monkeypatch) -> None:
    setup_calls: List[Tuple[str, str]] = []
    sent: List[pb.WorkerMessage] = []
    downloads: List[Dict[str, Any]] = []

    async def _send(msg: pb.WorkerMessage) -> None:
        sent.append(msg)

    spec = _slot_spec("generate", setup_calls,
                      pipe_default=PIPE_DEFAULT_CAS,
                      vae_default=Hub("acme/vae", tag="prod"))
    ex = Executor([spec], _send)

    async def _fake_download(ref: str, **kwargs: Any) -> Path:
        downloads.append({"ref": ref})
        p = tmp_path / ref.replace("/", "_").replace(":", "_")
        p.mkdir(parents=True, exist_ok=True)
        return p

    import gen_worker.executor as ex_mod
    monkeypatch.setattr(ex_mod, "ensure_local", _fake_download)

    async def _run() -> pb.JobResult:
        # The hub stamps the declared default for a no-pick request.
        return await _dispatch(ex, sent, _run_job(
            "r1",
            models=[pb.ModelBinding(slot="pipeline", ref="acme/default-model:prod"),
                    pb.ModelBinding(slot="vae", ref="acme/vae:prod")],
            snapshots={"acme/default-model:prod": _snapshot("aa" * 32),
                       "acme/vae:prod": _snapshot("bb" * 32)}))

    res = asyncio.run(_run())
    assert res.status == pb.JOB_STATUS_OK, res.safe_message
    out = msgspec.msgpack.decode(res.inline, type=_Out)
    assert out.slot_ref == "tensorhub:acme/default-model:prod"
    assert len(setup_calls) == 1
    # The declared binding was reused: the base record, no derived sibling.
    assert _ready_records(ex)[0] is ex._classes[spec.instance_key]


# ---------------------------------------------------------------------------
# 5. unusable (non-CAS) stamp over a raw default: RETRYABLE, no upstream fetch
# ---------------------------------------------------------------------------


def test_raw_stamp_over_raw_default_fails_retryable_without_fetch(tmp_path, monkeypatch) -> None:
    setup_calls: List[Tuple[str, str]] = []
    ex, sent, downloads = _harness(tmp_path, monkeypatch, setup_calls)

    async def _run() -> pb.JobResult:
        # The hub stamped the UNMIRRORED raw default (a bare civitai id —
        # not CAS grammar). The worker must refuse, retryably.
        return await _dispatch(ex, sent, _run_job(
            "r1",
            models=[pb.ModelBinding(slot="pipeline", ref="827184"),
                    pb.ModelBinding(slot="vae", ref=VAE_MIRROR)],
            snapshots={VAE_MIRROR: _snapshot("bb" * 32)}))

    res = asyncio.run(_run())
    assert res.status == pb.JOB_STATUS_RETRYABLE, res.safe_message
    assert "pipeline" in res.safe_message
    assert setup_calls == []
    assert downloads == [], f"a refused dispatch must not download: {downloads}"
    assert "generate" not in ex.unavailable, "a per-request refusal must not disable the function"


# ---------------------------------------------------------------------------
# LOAD for a never-dispatched pick: bytes banked, typed failure, no setup
# ---------------------------------------------------------------------------


def test_load_of_unknown_pick_banks_bytes_and_fails_typed(tmp_path, monkeypatch) -> None:
    setup_calls: List[Tuple[str, str]] = []
    ex, sent, downloads = _harness(tmp_path, monkeypatch, setup_calls)

    asyncio.run(ex.handle_model_op(pb.ModelOp(
        op=pb.MODEL_OP_KIND_LOAD, ref=PICK_A, snapshot=_snapshot("aa" * 32))))

    # Pre-warm degraded to a download: bytes + snapshot banked for the next
    # dispatch, no instance guessed into existence, typed failure reported.
    assert {d["ref"] for d in downloads} == {PICK_A}
    assert ex.store.has_snapshot(PICK_A)
    assert setup_calls == []
    failed = [m.model_event for m in sent if m.WhichOneof("msg") == "model_event"
              and m.model_event.state == pb.MODEL_STATE_FAILED]
    assert [(e.ref, e.error) for e in failed] == [(PICK_A, "load_failed")]


# ---------------------------------------------------------------------------
# 6. hub-less path: default_checkpoint raw source resolves upstream, unchanged
# ---------------------------------------------------------------------------


def test_hubless_default_path_unchanged(monkeypatch, tmp_path) -> None:
    from gen_worker.models import provision

    resolved: List[Tuple[str, str]] = []

    def _fake_resolve(*, ref: str, provider: str, **kw: Any) -> str:
        resolved.append((ref, provider))
        return str(tmp_path)

    monkeypatch.setattr(provision, "resolve_local_path", _fake_resolve)
    paths = provision.resolve_bindings(
        {"pipeline": PIPE_DEFAULT_RAW, "vae": VAE_DEFAULT_RAW},
        offline=False, emit=lambda e: None,
        slots={"pipeline": Slot(_StubPipeline, selected_by="model",
                                default_checkpoint=PIPE_DEFAULT_RAW)},
        payload=_In(prompt="x"),
    )
    assert paths == {"pipeline": str(tmp_path), "vae": str(tmp_path)}
    assert ("827184", "civitai") in resolved
    assert ("madebyollin/sdxl-vae-fp16-fix", "huggingface") in resolved
