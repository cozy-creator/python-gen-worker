"""pgw#654 warm-tax fix: warm RUNS are contract-keyed, never instance-keyed.

The ie#546 sdxl canary measured the 0.61.0 derived warm plan re-running per
CHECKPOINT INSTANCE: ~30-min first boots (full class x bucket cross-product
of real eager denoises) and a ~9-min warm tax on every juggle swap whose
genuine cost (download + VRAM load) was ~74s. Pinned here, through the REAL
executor setup path (fakes only at the download boundary):

  1. boot on an eager lane runs ONE shape representative per (function,
     guidance class) — not the cross-product — with step fields clamped to
     their declared floor;
  2. a SECOND checkpoint instance of the same family/contract runs exactly
     ONE verification job (swap cost = transfer + load, never a warm-plan
     re-run), and a third does the same;
  3. the warm contract key unifies across checkpoint refs but splits on
     lane facts (flavor/storage_dtype) and component overrides;
  4. select_runs keeps the full plan while tracing, keeps the max-area
     bucket for numeric shape axes, and honors msgspec Meta floors in the
     step clamp;
  5. pgw#647 gap #2: a component override inherits the base COMPOSITION's
     compute dtype (fp8-stored base => bf16 compute), never the override's
     on-disk dtype;
  6. ctx.adjustments is the public, immutable read side of the ledger.
"""

from __future__ import annotations

import asyncio
import json
import struct
from pathlib import Path
from typing import Annotated, Any, List, Optional, Tuple

import msgspec
import pytest

from gen_worker import (
    AxisClass,
    CompileAxis,
    RequestContext,
    Resources,
    Slot,
    endpoint,
    worker_function,
)
from gen_worker import warmup as warmup_mod
from gen_worker.executor import Executor
from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.registry import extract_specs

_GUIDANCE_AXIS = CompileAxis(classes=(
    AxisClass("cfg_off", match=lambda v: v == 0, warm=0.0),
    AxisClass("cfg_on", match=lambda v: v != 0, warm=7.5),
))
_ASPECT_AXIS = CompileAxis(classes=(
    AxisClass("sq", match=lambda v: v == "1:1", warm="1:1"),
    AxisClass("wide", match=lambda v: v == "16:9", warm="16:9"),
    AxisClass("tall", match=lambda v: v == "9:16", warm="9:16"),
))


class GenIn(msgspec.Struct):
    prompt: str
    aspect_ratio: Annotated[str, _ASPECT_AXIS] = "1:1"
    guidance_scale: Annotated[Optional[float], _GUIDANCE_AXIS] = None
    num_inference_steps: int = 8


class TurboIn(msgspec.Struct):
    # Distilled contract: no guidance field (classifies as the 0-class).
    prompt: str
    aspect_ratio: Annotated[str, _ASPECT_AXIS] = "1:1"
    num_inference_steps: int = 4


class Out(msgspec.Struct):
    y: str = "ok"


# (function, pipeline path, guidance, steps, aspect) per warm invocation.
CALLS: List[Tuple[str, str, Optional[float], int, str]] = []


@endpoint(
    models={"pipeline": Slot(str)},
    resources=Resources(gpu=True),
)
class Family:
    def setup(self, pipeline: str) -> None:
        self.pipeline = pipeline

    @worker_function()
    def generate(self, ctx: RequestContext, p: GenIn) -> Out:
        CALLS.append((
            "generate", self.pipeline, p.guidance_scale,
            p.num_inference_steps, p.aspect_ratio,
        ))
        return Out()

    @worker_function()
    def generate_turbo(self, ctx: RequestContext, p: TurboIn) -> Out:
        CALLS.append((
            "generate_turbo", self.pipeline, None,
            p.num_inference_steps, p.aspect_ratio,
        ))
        return Out()


def _executor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Executor:
    sent: List[pb.WorkerMessage] = []

    async def _send(msg: pb.WorkerMessage) -> None:
        sent.append(msg)

    specs = extract_specs(Family)
    ex = Executor(specs, _send)
    ex.store._cache_dir = tmp_path / "cas"

    async def _fake_download(ref: str, **kwargs: Any) -> Path:
        p = tmp_path / ref.replace("/", "_").replace(":", "_").replace("#", "_")
        p.mkdir(parents=True, exist_ok=True)
        return p

    import gen_worker.executor as ex_mod

    monkeypatch.setattr(ex_mod, "ensure_local", _fake_download)
    return ex


def _pick(ex: Executor, name: str, ref: str) -> Any:
    spec = ex.specs[name]
    run = pb.RunJob(
        function_name=name,
        models=[pb.ModelBinding(slot="pipeline", ref=ref)],
    )
    return ex._effective_spec(spec, run)


def _snapshots(ref: str, digest: str) -> dict:
    return {ref: pb.Snapshot(digest=digest, files=[pb.SnapshotFile(
        path="model.safetensors", size_bytes=5, blake3="cd" * 32,
        url="http://r2.invalid/presigned")])}


def test_eager_boot_reduces_and_instance_swaps_verify_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The regression test Paul asked for: N instances of one family — the
    warm plan runs once; each additional instance costs ONE verification
    run, never a re-run of the plan."""
    CALLS.clear()
    ex = _executor(tmp_path, monkeypatch)

    async def _run() -> None:
        # --- boot: first checkpoint instance -------------------------------
        eff1 = _pick(ex, "generate", "acme/sdxl-base")
        await ex.ensure_setup(eff1, _snapshots("acme/sdxl-base", "d1" * 16))
        boot_calls = list(CALLS)
        # generate's full eager cross-product would be 6 runs (2 cfg x 3
        # aspects). The eager reduction runs one shape representative per
        # guidance class: cfg_off + cfg_on.
        assert len(boot_calls) == 2, boot_calls
        assert {(fn, g) for fn, _p, g, _s, _a in boot_calls} == {
            ("generate", 0.0), ("generate", 7.5),
        }
        # One shape representative only (first declared class for a
        # non-numeric axis), steps clamped to the declared floor.
        assert all(a == "1:1" for *_x, a in boot_calls)
        assert all(s == 1 for _f, _p, _g, s, _a in boot_calls)

        # A sibling-alias dispatch on the SAME pick joins the ready record
        # without any further warm work.
        CALLS.clear()
        turbo1 = _pick(ex, "generate-turbo", "acme/sdxl-base")
        assert turbo1.instance_key == eff1.instance_key
        await ex.ensure_setup(
            turbo1, _snapshots("acme/sdxl-base", "d1" * 16))
        assert CALLS == []

        # --- juggle swap: second checkpoint, same family/contract ----------
        CALLS.clear()
        eff2 = _pick(ex, "generate", "acme/cyberrealistic-xl")
        assert eff2.instance_key != eff1.instance_key
        await ex.ensure_setup(eff2, _snapshots("acme/cyberrealistic-xl", "d2" * 16))
        assert len(CALLS) == 1, (
            "an instance swap must be a warm-contract cache hit "
            f"(one verification run), got {CALLS}"
        )

        # --- third instance: same again ------------------------------------
        CALLS.clear()
        eff3 = _pick(ex, "generate", "acme/nova-anime-xl")
        await ex.ensure_setup(eff3, _snapshots("acme/nova-anime-xl", "d3" * 16))
        assert len(CALLS) == 1, CALLS

        # All three instances are READY and share ONE warm contract.
        ready = [r for r in ex._classes.values() if r.ready]
        assert len(ready) == 3
        assert len(ex._warm_contract_runs) == 1
        (memory,) = ex._warm_contract_runs.values()
        assert len(memory) == 2  # cfg_off + cfg_on; verify re-runs a member

    asyncio.run(_run())


def test_warm_contract_key_splits_on_execution_lane_and_overrides_not_ref(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    ex = _executor(tmp_path, monkeypatch)
    base = _pick(ex, "generate", "acme/sdxl-base")
    fine_tune = _pick(ex, "generate", "acme/cyberrealistic-xl")
    assert ex._warm_contract_key(base) == ex._warm_contract_key(fine_tune)

    quant = _pick(ex, "generate", "acme/sdxl-base#fp8-w8a8")
    assert ex._warm_contract_key(quant) != ex._warm_contract_key(base)

    run = pb.RunJob(function_name="generate", models=[pb.ModelBinding(
        slot="pipeline", ref="acme/sdxl-base",
        components={"vae": "tensorhub/sdxl-vae-fp16-fix"},
    )])
    overridden = ex._effective_spec(ex.specs["generate"], run)
    assert ex._warm_contract_key(overridden) != ex._warm_contract_key(base)


# ---------------------------------------------------------------------------
# select_runs unit rows
# ---------------------------------------------------------------------------


def _family_jobs() -> List[warmup_mod.WarmupJob]:
    jobs, skips = warmup_mod.plan(extract_specs(Family), decl_warmup=None)
    assert not skips
    return jobs


def test_select_runs_full_while_tracing() -> None:
    jobs = _family_jobs()
    runs, mode = warmup_mod.select_runs(jobs, tracing=True)
    assert mode == "full" and runs == jobs


def test_select_runs_verify_only_when_contract_covered() -> None:
    jobs = _family_jobs()
    executed = {wj.graph_key for wj in jobs}
    runs, mode = warmup_mod.select_runs(jobs, tracing=True, executed=executed)
    assert mode == "verify" and len(runs) == 1
    # Partial coverage never collapses to verify.
    partial = set(list(executed)[:-1])
    _runs, mode = warmup_mod.select_runs(jobs, tracing=True, executed=partial)
    assert mode == "full"


class MpIn(msgspec.Struct):
    prompt: str
    megapixels: Annotated[float, CompileAxis(classes=(
        AxisClass("mp1", match=lambda v: v <= 1.5, warm=1.0),
        AxisClass("mp2", match=lambda v: v > 1.5, warm=2.0),
    ))] = 1.0


class FloorIn(msgspec.Struct):
    prompt: str
    num_inference_steps: Annotated[int, msgspec.Meta(ge=4)] = 20


def test_eager_reduction_keeps_max_numeric_bucket() -> None:
    @endpoint(models={"pipeline": Slot(str)}, resources=Resources(gpu=True))
    class Z:
        def setup(self, pipeline: str) -> None:
            self.pipeline = pipeline

        @worker_function()
        def generate(self, ctx: RequestContext, p: MpIn) -> Out:
            return Out()

    jobs, _ = warmup_mod.plan(extract_specs(Z), decl_warmup=None)
    assert len(jobs) == 2
    runs, mode = warmup_mod.select_runs(jobs, tracing=False)
    assert mode == "eager" and len(runs) == 1
    # The allocator-peak (max-area) bucket is the one kept.
    payload = runs[0].build(".")
    assert payload.megapixels == 2.0


def test_step_clamp_honors_meta_floor() -> None:
    @endpoint(models={"pipeline": Slot(str)}, resources=Resources(gpu=True))
    class F:
        def setup(self, pipeline: str) -> None:
            self.pipeline = pipeline

        @worker_function()
        def generate(self, ctx: RequestContext, p: FloorIn) -> Out:
            return Out()

    jobs, _ = warmup_mod.plan(extract_specs(F), decl_warmup=None)
    (job,) = jobs
    assert job.build(".").num_inference_steps == 4


# ---------------------------------------------------------------------------
# pgw#647 gap #2: component-override dtype inherits the composition's
# ---------------------------------------------------------------------------


def _write_safetensors(dir_path: Path, dtype: str) -> None:
    dir_path.mkdir(parents=True, exist_ok=True)
    header = {"weight": {
        "dtype": dtype, "shape": [1], "data_offsets": [0, 4],
    }}
    blob = json.dumps(header).encode()
    (dir_path / "model.safetensors").write_bytes(
        struct.pack("<Q", len(blob)) + blob + b"\x00\x00\x00\x00")


def test_composition_compute_dtype_inherits_base_not_override(
    tmp_path: Path,
) -> None:
    from gen_worker.models.loading import composition_compute_dtype

    bf16_base = tmp_path / "base-bf16"
    _write_safetensors(bf16_base, "BF16")
    fp8_base = tmp_path / "base-fp8"
    _write_safetensors(fp8_base, "F8_E4M3")
    fp32_base = tmp_path / "base-fp32"
    _write_safetensors(fp32_base, "F32")

    # Declared binding dtype wins outright.
    assert composition_compute_dtype(bf16_base, "fp16") == "fp16"
    # No declared dtype: the base tree's compute dtype, with fp8 storage
    # mapping to its bf16 compute default (the ie#546 canary case — the
    # fp32-stored fp16-fix VAE must land bf16 in an fp8-w8a16 composition).
    assert composition_compute_dtype(bf16_base) == "bf16"
    assert composition_compute_dtype(fp8_base) == "bf16"
    # An fp32/undetectable base stays "" (caller falls back).
    assert composition_compute_dtype(fp32_base) == ""


def test_load_component_override_prefers_composition_dtype(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gen_worker.models import loading as loading_mod

    base = tmp_path / "base"
    base.mkdir()
    (base / "model_index.json").write_text(json.dumps({
        "_class_name": "FakePipe", "vae": ["fake_lib", "FakeVae"],
    }))
    _write_safetensors(base / "vae", "F8_E4M3")  # composition computes bf16
    override = tmp_path / "override"
    _write_safetensors(override, "F32")  # fp32-stored fp16-fix shape

    seen: dict = {}

    class FakeVae:
        @classmethod
        def from_pretrained(cls, path: str, **kwargs: Any) -> "FakeVae":
            seen.update(kwargs)
            return cls()

    import types

    fake_lib = types.ModuleType("fake_lib")
    fake_lib.FakeVae = FakeVae  # type: ignore[attr-defined]
    monkeypatch.setitem(__import__("sys").modules, "fake_lib", fake_lib)

    captured: dict = {}

    def _get_torch_dtype(name: str) -> str:
        captured["wanted"] = name
        raise ImportError("torch-less test rig")

    monkeypatch.setattr(loading_mod, "get_torch_dtype", _get_torch_dtype)
    loading_mod.load_component_override(base, "vae", override)
    assert captured["wanted"] == "bf16", (
        "override must inherit the base composition's compute dtype, "
        "never its own on-disk fp32"
    )


# ---------------------------------------------------------------------------
# ctx.adjustments: the public read side
# ---------------------------------------------------------------------------


def test_ctx_adjustments_is_public_and_immutable() -> None:
    ctx: RequestContext[Any] = RequestContext(
        request_id="t", local_output_dir=".", models={})
    assert len(ctx.adjustments) == 0
    ctx.clamp("guidance_scale", 15.0, hi=10.0, reason="model maximum")
    rows = ctx.adjustments
    assert len(rows) == 1
    row = rows[0]
    assert row["field"] == "guidance_scale"
    assert row["requested"] == "15.0" and row["applied"] == "10.0"
    with pytest.raises(AttributeError):
        rows.append({})  # noqa: B038 — proves immutability


# ---------------------------------------------------------------------------
# benchmark harness: content-address plan (CPU-safe half)
# ---------------------------------------------------------------------------


def test_benchmark_component_digest_discriminates(tmp_path: Path) -> None:
    """The swap benchmark's component plan: identical component subtrees
    share a content address (no bytes move on swap); differing ones split."""
    sys_path = __import__("sys").path
    root = str(Path(__file__).resolve().parents[1])
    if root not in sys_path:
        sys_path.insert(0, root)
    from benchmarks.swap_latency import component_digest

    a = tmp_path / "a" / "vae"
    b = tmp_path / "b" / "vae"
    a.mkdir(parents=True)
    b.mkdir(parents=True)
    (a / "w.safetensors").write_bytes(b"same-bytes")
    (b / "w.safetensors").write_bytes(b"same-bytes")
    assert component_digest(tmp_path / "a", "vae") == component_digest(
        tmp_path / "b", "vae")
    (b / "w.safetensors").write_bytes(b"other-bytes")
    assert component_digest(tmp_path / "a", "vae") != component_digest(
        tmp_path / "b", "vae")
    assert component_digest(tmp_path / "a", "absent") == ""
