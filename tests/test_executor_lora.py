"""gw#393 + gw#399 per-request LoRA overlays with adapter residency.

RunJob ModelBinding.loras reaches the executor; adapter snapshots ride the
normal ensure_local path; state dicts hit the digest-keyed RAM LRU; adapters
stay ATTACHED to the resident pipeline while requests toggle the ACTIVE set
(explicit activation — nothing active unless the request named it, disabled
on EVERY exit path). Attachments are LRU-capped and dropped on demotion.
Validation mirrors the hub gates worker-side.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional

import msgspec
import pytest

from gen_worker.api.binding import Hub, wire_ref
from gen_worker.api.errors import RefCompatibilitySurprise, ValidationError
from gen_worker.executor import Executor
from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.registry import EndpointSpec
from gen_worker.utils import lora as lora_util

MODEL_REF = "acme/sdxl-base"
LORA_A = "user1/lora-anime"
LORA_B = "user1/lora-sketch"


class _FakeTensor:
    def __init__(self, nbytes: int = 64) -> None:
        self.nbytes = nbytes
        self.shape = (4, 4)


def _fake_state_dict(nbytes: int = 64) -> Dict[str, Any]:
    return {"unet.mid_block.attn.to_q.lora_down.weight": _FakeTensor(nbytes)}


class _FakeLoraPipe:
    """Residency-capable fake: tracks attachments, the active set, and the
    peft-level enabled flag like a diffusers pipeline."""

    def __init__(self, fail_load: bool = False) -> None:
        self.calls: List[Any] = []
        self.attached: List[str] = []
        self.active: List[str] = []
        self.weights: List[float] = []
        self.enabled = True
        self.fail_load = fail_load
        self.fail_disable = False

    def load_lora_weights(self, state_dict, adapter_name: str = "") -> None:
        if self.fail_load:
            raise RuntimeError("size mismatch for lora_up.weight")
        self.calls.append(("load", adapter_name))
        self.attached.append(adapter_name)

    def set_adapters(self, names, adapter_weights=None) -> None:
        self.calls.append(("set", tuple(names), tuple(adapter_weights)))
        self.active = list(names)
        self.weights = list(adapter_weights)

    def enable_lora(self) -> None:
        self.calls.append(("enable",))
        self.enabled = True

    def disable_lora(self) -> None:
        if self.fail_disable:
            raise RuntimeError("disable failed")
        self.calls.append(("disable",))
        self.enabled = False

    def delete_adapters(self, name: str) -> None:
        self.calls.append(("delete", name))
        self.attached.remove(name)

    def unload_lora_weights(self) -> None:
        self.calls.append(("unload",))
        self.attached = []
        self.active = []
        self.weights = []

    def to(self, device: str) -> "_FakeLoraPipe":
        self.calls.append(("to", device))
        return self

    @property
    def live(self) -> List[str]:
        """Adapters actually affecting compute right now."""
        return self.active if self.enabled else []


class _In(msgspec.Struct):
    prompt: str = ""
    fail: bool = False


class _Out(msgspec.Struct):
    active_adapters: int = 0
    weights: List[float] = []
    lora_meta: List[List[Any]] = []
    adapter_ref_pinned: bool = False


class _Endpoint:
    pipe: Optional[_FakeLoraPipe] = None
    store: Any = None

    def setup(self, pipeline: str) -> None:  # pragma: no cover
        pass

    def run(self, ctx, payload: _In) -> _Out:
        if payload.fail:
            raise RuntimeError("boom")
        pipe = type(self).pipe
        meta = [
            [slot, ov["ref"], ov["weight"]]
            for slot, loras in sorted(ctx.loras.items()) for ov in loras
        ]
        pinned = bool(type(self).store and type(self).store.residency.in_use(LORA_A))
        return _Out(
            active_adapters=len(pipe.live), weights=list(pipe.weights),
            lora_meta=meta, adapter_ref_pinned=pinned,
        )


def _spec() -> EndpointSpec:
    return EndpointSpec(
        name="txt2img", method=_Endpoint.run, kind="inference",
        payload_type=_In, output_mode="single", cls=_Endpoint,
        attr_name="run", models={"pipeline": Hub(MODEL_REF)},
    )


class _Harness:
    def __init__(self, tmp_path: Path, monkeypatch, *, pipe: Optional[_FakeLoraPipe] = None) -> None:
        self.sent: List[pb.WorkerMessage] = []
        self.ensured: List[str] = []
        self.parsed: List[str] = []
        self.tmp_path = tmp_path

        async def _send(msg: pb.WorkerMessage) -> None:
            self.sent.append(msg)

        spec = _spec()
        self.spec = spec
        self.executor = Executor([spec], _send)
        self.pipe = pipe or _FakeLoraPipe()
        _Endpoint.pipe = self.pipe
        _Endpoint.store = self.executor.store
        rec = self.executor._classes[spec.instance_key]
        rec.instance = _Endpoint()
        rec.ready = True
        self.executor.store.residency.track_vram(
            wire_ref(spec.models["pipeline"]), self.pipe, vram_bytes=1)

        async def _fake_ensure_local(ref, snapshot=None, *, binding=None) -> Path:
            self.ensured.append(ref)
            d = tmp_path / ref.replace("/", "--")
            d.mkdir(parents=True, exist_ok=True)
            (d / "adapter.safetensors").write_bytes(b"x" * 8)
            self.executor.store.residency.track_disk(ref, d)
            return d

        self.executor.store.ensure_local = _fake_ensure_local  # type: ignore[method-assign]

        def _fake_parse(path: Path, *, ref: str = "") -> Dict[str, Any]:
            self.parsed.append(ref)
            return _fake_state_dict()

        monkeypatch.setattr(lora_util, "load_adapter_state_dict", _fake_parse)

    async def run(self, *, loras=(), payload: Optional[_In] = None,
                  request_id: str = "r1", snapshots=None) -> pb.JobResult:
        run = pb.RunJob(
            request_id=request_id, attempt=1, function_name="txt2img",
            input_payload=msgspec.msgpack.encode(payload or _In()),
            models=[pb.ModelBinding(slot="pipeline", ref=MODEL_REF, loras=list(loras))],
        )
        for ref, digest in (snapshots or {}).items():
            run.snapshots[ref].digest = digest
        await self.executor.handle_run_job(run)
        job = self.executor.jobs[(request_id, 1)]
        assert job.task is not None
        await job.task
        results = [m.job_result for m in self.sent if m.WhichOneof("msg") == "job_result"]
        assert results, f"no job_result; sent={self.sent}"
        return results[-1]


def _ov(ref: str, weight: float = 1.0) -> pb.LoraOverlay:
    return pb.LoraOverlay(ref=ref, weight=weight)


# ---------------------------------------------------------------------------
# Wire -> apply -> unload
# ---------------------------------------------------------------------------


def test_adapters_active_during_handler_and_disabled_after(tmp_path, monkeypatch) -> None:
    async def _run() -> None:
        h = _Harness(tmp_path, monkeypatch)
        res = await h.run(loras=[_ov(LORA_A, 0.8), _ov(LORA_B, -0.5)])
        assert res.status == pb.JOB_STATUS_OK
        out = msgspec.msgpack.decode(res.inline, type=_Out)
        assert out.active_adapters == 2
        assert out.weights == [0.8, -0.5]
        assert out.adapter_ref_pinned  # executing() covers adapter refs
        assert h.ensured == [LORA_A, LORA_B]
        kinds = [c[0] for c in h.pipe.calls]
        assert kinds == ["load", "load", "set", "enable", "disable"]
        assert h.pipe.live == []          # nothing active after the request
        assert len(h.pipe.attached) == 2  # attachments stay resident

    asyncio.run(_run())


def test_ctx_loras_surfaces_refs_and_weights(tmp_path, monkeypatch) -> None:
    async def _run() -> None:
        h = _Harness(tmp_path, monkeypatch)
        res = await h.run(loras=[_ov(LORA_A, 0.8)])
        out = msgspec.msgpack.decode(res.inline, type=_Out)
        assert out.lora_meta == [["pipeline", LORA_A, 0.8]]

    asyncio.run(_run())


def test_no_loras_is_a_no_op(tmp_path, monkeypatch) -> None:
    async def _run() -> None:
        h = _Harness(tmp_path, monkeypatch)
        res = await h.run()
        assert res.status == pb.JOB_STATUS_OK
        assert h.pipe.calls == []
        assert h.ensured == []

    asyncio.run(_run())


def test_deactivate_runs_when_handler_raises(tmp_path, monkeypatch) -> None:
    async def _run() -> None:
        h = _Harness(tmp_path, monkeypatch)
        res = await h.run(loras=[_ov(LORA_A)], payload=_In(fail=True))
        assert res.status == pb.JOB_STATUS_FATAL
        assert h.pipe.calls[-1] == ("disable",)
        assert h.pipe.live == []

    asyncio.run(_run())


def test_failed_adapter_load_rolls_back_and_is_invalid(tmp_path, monkeypatch) -> None:
    async def _run() -> None:
        h = _Harness(tmp_path, monkeypatch, pipe=_FakeLoraPipe(fail_load=True))
        # Second adapter fails to load -> the first is rolled back; the job
        # fails INVALID (ref_compatibility_surprise), never reaches the handler.
        h.pipe.fail_load = False
        real_load = h.pipe.load_lora_weights

        def _load_second_fails(state_dict, adapter_name=""):
            if len([c for c in h.pipe.calls if c[0] == "load"]) >= 1:
                raise RuntimeError("size mismatch for lora_up.weight")
            real_load(state_dict, adapter_name=adapter_name)

        h.pipe.load_lora_weights = _load_second_fails
        res = await h.run(loras=[_ov(LORA_A), _ov(LORA_B)])
        assert res.status == pb.JOB_STATUS_INVALID
        assert "failed to load onto base pipeline" in res.safe_message
        assert h.pipe.calls[-1] == ("disable",)  # rollback deactivates
        assert h.pipe.live == []

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# Validation (worker-side mirrors of the hub gates)
# ---------------------------------------------------------------------------


def test_weight_out_of_bounds_is_invalid(tmp_path, monkeypatch) -> None:
    async def _run() -> None:
        h = _Harness(tmp_path, monkeypatch)
        res = await h.run(loras=[_ov(LORA_A, 4.5)])
        assert res.status == pb.JOB_STATUS_INVALID
        assert "out of bounds" in res.safe_message
        assert h.pipe.calls == []
        assert h.ensured == []  # rejected before any download

    asyncio.run(_run())


def test_too_many_adapters_is_invalid(tmp_path, monkeypatch) -> None:
    async def _run() -> None:
        h = _Harness(tmp_path, monkeypatch)
        overlays = [_ov(f"u/l{i}") for i in range(lora_util.MAX_LORAS_PER_REQUEST + 1)]
        res = await h.run(loras=overlays)
        assert res.status == pb.JOB_STATUS_INVALID
        assert "too many lora adapters" in res.safe_message
        assert h.ensured == []

    asyncio.run(_run())


def test_unknown_slot_is_invalid(tmp_path, monkeypatch) -> None:
    async def _run() -> None:
        h = _Harness(tmp_path, monkeypatch)
        run = pb.RunJob(
            request_id="r1", attempt=1, function_name="txt2img",
            input_payload=msgspec.msgpack.encode(_In()),
            models=[
                pb.ModelBinding(slot="pipeline", ref=MODEL_REF),
                pb.ModelBinding(slot="ghost", ref="x/y", loras=[_ov(LORA_A)]),
            ],
        )
        await h.executor.handle_run_job(run)
        await h.executor.jobs[("r1", 1)].task
        results = [m.job_result for m in h.sent if m.WhichOneof("msg") == "job_result"]
        assert results[-1].status == pb.JOB_STATUS_INVALID
        assert "unknown model slot" in results[-1].safe_message

    asyncio.run(_run())


def test_slot_without_worker_managed_pipeline_is_invalid(tmp_path, monkeypatch) -> None:
    async def _run() -> None:
        h = _Harness(tmp_path, monkeypatch)
        # Simulate a tenant-loaded slot: residency has no object for the ref.
        h.executor.store.residency.evict(wire_ref(h.spec.models["pipeline"]), force=True)
        res = await h.run(loras=[_ov(LORA_A)])
        assert res.status == pb.JOB_STATUS_INVALID
        assert "no worker-managed pipeline" in res.safe_message

    asyncio.run(_run())


def test_weight_zero_means_default_one(tmp_path, monkeypatch) -> None:
    async def _run() -> None:
        h = _Harness(tmp_path, monkeypatch)
        res = await h.run(loras=[_ov(LORA_A, 0.0)])  # proto3 unset
        out = msgspec.msgpack.decode(res.inline, type=_Out)
        assert out.weights == [1.0]

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# Adapter RAM cache
# ---------------------------------------------------------------------------


def test_repeat_request_hits_adapter_cache(tmp_path, monkeypatch) -> None:
    async def _run() -> None:
        h = _Harness(tmp_path, monkeypatch)
        snaps = {LORA_A: "sha256:abc123"}
        await h.run(loras=[_ov(LORA_A)], request_id="r1", snapshots=snaps)
        await h.run(loras=[_ov(LORA_A)], request_id="r2", snapshots=snaps)
        assert h.parsed == [LORA_A]  # parsed once, served from RAM after
        assert h.executor._adapter_cache.hits == 1

    asyncio.run(_run())


def test_digest_change_invalidates_cache_key(tmp_path, monkeypatch) -> None:
    async def _run() -> None:
        h = _Harness(tmp_path, monkeypatch)
        await h.run(loras=[_ov(LORA_A)], request_id="r1",
                    snapshots={LORA_A: "sha256:aaa"})
        await h.run(loras=[_ov(LORA_A)], request_id="r2",
                    snapshots={LORA_A: "sha256:bbb"})
        assert h.parsed == [LORA_A, LORA_A]  # re-parsed under the new digest

    asyncio.run(_run())


def test_adapter_cache_lru_byte_cap() -> None:
    cache = lora_util.AdapterCache(max_bytes=100)
    cache.put("a", _fake_state_dict(60))
    cache.put("b", _fake_state_dict(60))  # over cap -> LRU 'a' evicted
    assert cache.get("a") is None
    assert cache.get("b") is not None
    cache.put("huge", _fake_state_dict(1000))  # larger than cap -> not cached
    assert cache.get("huge") is None
    assert len(cache) == 1


# ---------------------------------------------------------------------------
# Adapter residency (gw#399): attach once, toggle the active set
# ---------------------------------------------------------------------------


def test_repeat_request_reuses_attachment(tmp_path, monkeypatch) -> None:
    async def _run() -> None:
        h = _Harness(tmp_path, monkeypatch)
        snaps = {LORA_A: "sha256:abc"}
        await h.run(loras=[_ov(LORA_A, 0.9)], request_id="r1", snapshots=snaps)
        await h.run(loras=[_ov(LORA_A, 0.9)], request_id="r2", snapshots=snaps)
        loads = [c for c in h.pipe.calls if c[0] == "load"]
        sets = [c for c in h.pipe.calls if c[0] == "set"]
        assert len(loads) == 1  # attached once, reused on repeat
        assert len(sets) == 2   # every request toggles the active set
        assert len(h.pipe.attached) == 1

    asyncio.run(_run())


def test_weight_change_on_resident_adapter_needs_no_reload(tmp_path, monkeypatch) -> None:
    async def _run() -> None:
        h = _Harness(tmp_path, monkeypatch)
        snaps = {LORA_A: "sha256:abc"}
        await h.run(loras=[_ov(LORA_A, 0.9)], request_id="r1", snapshots=snaps)
        res = await h.run(loras=[_ov(LORA_A, 0.4)], request_id="r2", snapshots=snaps)
        out = msgspec.msgpack.decode(res.inline, type=_Out)
        assert out.weights == [0.4]
        assert len([c for c in h.pipe.calls if c[0] == "load"]) == 1

    asyncio.run(_run())


def test_interleaved_lora_bare_lora_never_bleeds(tmp_path, monkeypatch) -> None:
    """The zero-leakage invariant: bare requests between LoRA requests run
    with NOTHING live, even though attachments stay resident."""
    async def _run() -> None:
        h = _Harness(tmp_path, monkeypatch)
        snaps = {LORA_A: "sha256:abc"}

        res = await h.run(loras=[_ov(LORA_A, 0.9)], request_id="r1", snapshots=snaps)
        assert msgspec.msgpack.decode(res.inline, type=_Out).active_adapters == 1

        res = await h.run(request_id="r2")  # bare
        assert msgspec.msgpack.decode(res.inline, type=_Out).active_adapters == 0
        assert len(h.pipe.attached) == 1  # still attached, just not live

        res = await h.run(loras=[_ov(LORA_A, 0.9)], request_id="r3", snapshots=snaps)
        assert msgspec.msgpack.decode(res.inline, type=_Out).active_adapters == 1

        res = await h.run(request_id="r4")  # bare again
        assert msgspec.msgpack.decode(res.inline, type=_Out).active_adapters == 0
        assert h.pipe.live == []

    asyncio.run(_run())


def test_bare_request_recovers_from_failed_deactivation(tmp_path, monkeypatch) -> None:
    """Crash-leak guard: if a LoRA request's teardown fails, the next bare
    request on the same pipeline explicitly deactivates before running."""
    async def _run() -> None:
        h = _Harness(tmp_path, monkeypatch)
        snaps = {LORA_A: "sha256:abc"}
        h.pipe.fail_disable = True
        await h.run(loras=[_ov(LORA_A, 0.9)], request_id="r1", snapshots=snaps)
        assert h.pipe.enabled  # teardown failed; adapters still live
        h.pipe.fail_disable = False
        res = await h.run(request_id="r2")  # bare request must self-protect
        assert msgspec.msgpack.decode(res.inline, type=_Out).active_adapters == 0
        assert not h.pipe.enabled

    asyncio.run(_run())


def test_lru_eviction_over_attachment_caps(tmp_path, monkeypatch) -> None:
    async def _run() -> None:
        h = _Harness(tmp_path, monkeypatch)
        h.executor._adapters = lora_util.AdapterResidency(max_attached=2)
        for i, ref in enumerate([LORA_A, LORA_B, "user1/lora-third"]):
            await h.run(loras=[_ov(ref)], request_id=f"r{i}",
                        snapshots={ref: f"sha256:{i}"})
        assert len(h.pipe.attached) == 2
        deletes = [c for c in h.pipe.calls if c[0] == "delete"]
        assert len(deletes) == 1
        # LRU victim was the first-attached adapter. Cache keys carry the
        # bare-hex digest spelling (gw#491: algo prefix stripped so one
        # adapter never mints two identities).
        assert deletes[0][1] == lora_util.adapter_name(f"{LORA_A}@0")

    asyncio.run(_run())


def test_demotion_drops_attachments_and_reattaches_lazily(tmp_path, monkeypatch) -> None:
    async def _run() -> None:
        h = _Harness(tmp_path, monkeypatch)
        monkeypatch.setattr(
            "gen_worker.models.residency.get_available_ram_gb", lambda: 999.0)
        snaps = {LORA_A: "sha256:abc"}
        ref = wire_ref(h.spec.models["pipeline"])
        await h.run(loras=[_ov(LORA_A)], request_id="r1", snapshots=snaps)
        assert len(h.pipe.attached) == 1

        assert h.executor.store.residency.demote(ref)  # VRAM -> RAM
        assert h.pipe.attached == []  # pre_demote hook dropped attachments
        assert ("unload",) in h.pipe.calls

        h.executor.store.residency.promote(ref, device="cpu")
        await h.run(loras=[_ov(LORA_A)], request_id="r2", snapshots=snaps)
        assert len(h.pipe.attached) == 1  # lazily re-attached
        assert len([c for c in h.pipe.calls if c[0] == "load"]) == 2
        # state dict itself still came from the RAM cache (no re-parse)
        assert h.parsed == [LORA_A]

    asyncio.run(_run())


def test_replaced_pipeline_object_resets_attachment_state(tmp_path, monkeypatch) -> None:
    async def _run() -> None:
        h = _Harness(tmp_path, monkeypatch)
        snaps = {LORA_A: "sha256:abc"}
        ref = wire_ref(h.spec.models["pipeline"])
        await h.run(loras=[_ov(LORA_A)], request_id="r1", snapshots=snaps)

        new_pipe = _FakeLoraPipe()  # cold reload produced a fresh object
        _Endpoint.pipe = new_pipe
        h.executor.store.residency.track_vram(ref, new_pipe, vram_bytes=1)
        await h.run(loras=[_ov(LORA_A)], request_id="r2", snapshots=snaps)
        assert len([c for c in new_pipe.calls if c[0] == "load"]) == 1
        assert len(new_pipe.attached) == 1

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# State-dict validation (key stuffing, weights, files)
# ---------------------------------------------------------------------------


def test_lora_key_patterns() -> None:
    good = [
        "lora_unet_down_blocks_0_attn.lora_down.weight",
        "lora_unet_down_blocks_0_attn.lora_up.weight",
        "lora_unet_down_blocks_0_attn.alpha",
        "unet.mid.attn.to_q.lora_A.weight",
        "unet.mid.attn.to_q.lora_B.weight",
        "text_encoder.layers.0.q.lora_A.default.weight",
        "unet.up.attn.processor.to_v_lora.down.weight",
        "unet.up.attn.processor.to_v_lora.up.weight",
        "unet.mid.attn.to_q.lora_magnitude_vector",  # DoRA
        "unet.mid.attn.to_q.dora_scale",
    ]
    lora_util.validate_lora_keys(good)

    for bad in (
        "unet.mid_block.attn.to_q.weight",       # full-weight replacement
        "state_dict_pickle_payload",             # junk
        "unet.conv_in.bias",                     # bias stuffing
        "text_model.embeddings.token_embedding.weight",  # embedding stuffing
    ):
        with pytest.raises(RefCompatibilitySurprise):
            lora_util.validate_lora_keys(good + [bad])


def test_weight_bound_helper() -> None:
    assert lora_util.validate_overlay_weight(0.0) == 1.0
    assert lora_util.validate_overlay_weight(-4.0) == -4.0
    with pytest.raises(ValidationError):
        lora_util.validate_overlay_weight(4.01)
    with pytest.raises(ValidationError):
        lora_util.validate_overlay_weight(float("nan"))


def test_find_adapter_file_prefers_safetensors(tmp_path) -> None:
    d = tmp_path / "snap"
    d.mkdir()
    (d / "small.safetensors").write_bytes(b"x" * 4)
    (d / "big.safetensors").write_bytes(b"x" * 400)
    (d / "README.md").write_text("readme")
    assert lora_util.find_adapter_file(d).name == "big.safetensors"

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(RefCompatibilitySurprise):
        lora_util.find_adapter_file(empty)


def test_state_dict_parse_validates_and_injects_alpha(tmp_path) -> None:
    st = pytest.importorskip("safetensors.torch")
    torch = pytest.importorskip("torch")

    f = tmp_path / "ok.safetensors"
    st.save_file({
        "lora_unet_mid_attn.lora_down.weight": torch.zeros(4, 8),
        "lora_unet_mid_attn.lora_up.weight": torch.zeros(8, 4),
    }, str(f))
    sd = lora_util.load_adapter_state_dict(f)
    assert float(sd["lora_unet_mid_attn.alpha"]) == 4.0  # alpha = rank

    stuffed = tmp_path / "stuffed.safetensors"
    st.save_file({
        "lora_unet_mid_attn.lora_down.weight": torch.zeros(4, 8),
        "unet.conv_in.weight": torch.zeros(3, 3),  # full-weight stuffing
    }, str(stuffed))
    with pytest.raises(RefCompatibilitySurprise):
        lora_util.load_adapter_state_dict(stuffed)


def test_oversized_adapter_rejected(tmp_path, monkeypatch) -> None:
    pytest.importorskip("safetensors.torch")
    monkeypatch.setattr(lora_util, "MAX_LORA_FILE_BYTES", 4)
    f = tmp_path / "big.safetensors"
    f.write_bytes(b"x" * 64)
    with pytest.raises(ValidationError):
        lora_util.load_adapter_state_dict(f)
