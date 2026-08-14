"""#384: per-SKU torch.compile cache artifacts — key, packaging, safety policy."""

from __future__ import annotations

import logging
import os
import threading

import msgspec
import pytest

from typing import Annotated

from gen_worker import AxisClass, Compile, CompileAxis, DynamicDim, Resources, endpoint
from gen_worker import compile_cache as cc
from gen_worker.registry import CompileCell, collect_from_namespace


# ---------------------------------------------------------------------------
# key
# ---------------------------------------------------------------------------


def test_sku_slug():
    assert cc.sku_slug("NVIDIA GeForce RTX 4090") == "rtx-4090"
    assert cc.sku_slug("NVIDIA H100 80GB HBM3") == "h100-80gb-hbm3"
    assert cc.sku_slug("NVIDIA RTX 5090") == "rtx-5090"
    assert cc.sku_slug("") == ""


def test_flavor_label():
    assert cc.flavor_label("rtx-4090", "2.9.1+cu128") == "inductor-rtx-4090-torch2.9"
    assert cc.flavor_label("h100-80gb-hbm3", "2.11.0") == "inductor-h100-80gb-hbm3-torch2.11"


@pytest.mark.parametrize(
    ("execution_lane", "bucket"),
    [
        ("", 0),
        ("fp8-hooks", 0),
        ("w8a16", 0),
        ("w8a8", 0),
        ("lora16", 16),
        ("fp8-hooks-lora32", 32),
        ("w8a16-lora64", 64),
        ("w8a8-lora128", 128),
    ],
)
def test_compile_target_execution_lane_vocabulary_matches_tensorhub(execution_lane, bucket):
    assert cc.compile_target_execution_lane_error(execution_lane, bucket) == ""


@pytest.mark.parametrize(
    ("execution_lane", "bucket"),
    [
        ("w8a8-row", 0),
        ("fp8", 0),
        ("w8a8-lora8", 8),
        ("w8a8-lora32-sparse", 32),
        ("w8a8-lora32", 16),
        ("w8a8", 32),
    ],
)
def test_compile_target_execution_lane_vocabulary_rejects_impossible_states(execution_lane, bucket):
    assert cc.compile_target_execution_lane_error(execution_lane, bucket)


# pgw#1181 REMOVED 20 rows whose subject is the `torch-inductor-cache` format
# itself: `verify` (2), `mode_drift` (2), `artifact_metadata` (2), `pack` /
# `unpack` (3), the stage/seed/merge transaction (5:
# `test_failed_seed_never_mutates_live_cache`,
# `test_pipeline_mismatch_never_activates_staged_cache`,
# `test_cache_collision_and_merge_failure_leave_live_tree_unchanged`,
# `test_seed_activation_blocks_concurrent_cold_arming`,
# `test_seeding_rejects_a_key_mismatch`), `capture_env` (1), `contract_drift`
# (1), `_reconcile_resident_mode` (2: the two `enable_reconciles_*` rows) and
# the w8a8 identity gate over `verify`'s axis set (1).
#
# The format's last writer died with `mint_artifact` in pgw#1178 and the format
# itself is deleted here, so every one of these builds a cell no pod can
# produce and drives a transaction no pod can enter (§4.34: they die with
# their subject, never ported). What each fenced survives on the exported
# lane by CONSTRUCTION rather than by comparison — sm, the declared contract,
# the env seal, the lane and the graph are all axes of `ck1`, so a cell that
# disagrees has a different key and never resolves; see
# `tests/test_cell_key_pgw1059.py`.


def test_execution_contract_uses_structure_not_checkpoint_values():
    torch = pytest.importorskip("torch")

    class _Pipe:
        def __init__(self, hidden: int, fill: float) -> None:
            self.transformer = torch.nn.Sequential(
                torch.nn.Linear(hidden, hidden), torch.nn.SiLU(),
            )
            with torch.no_grad():
                self.transformer[0].weight.fill_(fill)

    cfg = Compile(shapes=((1024, 1024),), targets=("transformer",), family="sdxl")
    a_sig, a_weights = cc.execution_contract(_Pipe(16, 1.0), cfg)
    b_sig, b_weights = cc.execution_contract(_Pipe(16, 9.0), cfg)
    c_sig, _ = cc.execution_contract(_Pipe(32, 1.0), cfg)
    assert a_sig == b_sig
    assert a_weights == b_weights == {"lane": ""}
    assert c_sig != a_sig


def test_execution_contract_records_dynamic_w8a8_exclusions():
    torch = pytest.importorskip("torch")

    class _Scaled(torch.nn.Module):
        _cozy_w8a8_linear = True

        def __init__(self) -> None:
            super().__init__()
            self.in_features = self.out_features = 16
            self.register_buffer("weight", torch.empty(
                16, 16, dtype=getattr(torch, "float8_e4m3fn")))
            self.register_buffer("weight_scale", torch.ones(16, 1))
            self.input_scale = None

        def forward(self, x):
            return x

    class _Target(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fast = _Scaled()
            self.sensitive = torch.nn.Linear(16, 16)

    class _Pipe:
        def __init__(self) -> None:
            self.transformer = _Target()
            self._cozy_weight_lane = "w8a8"

    cfg = Compile(shapes=((1024, 1024),), targets=("transformer",), family="sdxl")
    _signature, contract = cc.execution_contract(_Pipe(), cfg)
    assert contract["operator"] == "torch._scaled_mm"
    assert contract["activation_scaling"] == ["dynamic-per-row"]
    assert [r["path"] for r in contract["quantized"]] == ["transformer:fast"]
    assert [r["path"] for r in contract["excluded"]] == ["transformer:sensitive"]


def test_execution_contract_digest_covers_every_runtime_graph_axis():
    torch = pytest.importorskip("torch")

    class _Scaled(torch.nn.Module):
        _cozy_w8a8_linear = True

        def __init__(self, *, per_tensor: bool, fill: float) -> None:
            super().__init__()
            self.in_features = self.out_features = 8
            self.register_buffer("weight", torch.full(
                (8, 8), fill, dtype=getattr(torch, "float8_e4m3fn")))
            self.register_buffer("weight_scale", torch.ones(8, 1))
            self.input_scale = None
            self.gemm_mode = "pertensor" if per_tensor else "rowwise"

        def forward(self, value):
            return value

    class _Target(torch.nn.Module):
        def __init__(self, *, per_tensor: bool, fill: float) -> None:
            super().__init__()
            self.proj = _Scaled(per_tensor=per_tensor, fill=fill)

    class _Pipe:
        def __init__(
            self, *, per_tensor: bool = False, fill: float = 1.0,
            low_vram: str = "",
        ) -> None:
            self.transformer = _Target(per_tensor=per_tensor, fill=fill)
            self._cozy_weight_lane = "w8a8"
            self._cozy_low_vram_mode = low_vram

    base_cfg = Compile(
        shapes=((768, 768),), targets=("transformer",), family="flux2-klein-4b")
    base = cc.execution_contract_digest(_Pipe(fill=1.0), base_cfg)

    # Checkpoint values are deliberately excluded: compatible fine-tunes
    # share a family cell.
    assert cc.execution_contract_digest(_Pipe(fill=9.0), base_cfg) == base
    # Every consumer compatibility axis changes the fence identity.
    assert cc.execution_contract_digest(_Pipe(per_tensor=True), base_cfg) != base
    assert cc.execution_contract_digest(
        _Pipe(), Compile(
            shapes=((1024, 1024),), targets=("transformer",),
            family="flux2-klein-4b")) != base
    assert cc.execution_contract_digest(
        _Pipe(), Compile(
            shapes=((768, 768),), targets=("transformer",),
            family="flux2-klein-4b", regional=True)) != base
    assert cc.execution_contract_digest(_Pipe(low_vram="model_offload"), base_cfg) != base


def test_w8a8_guard_degrades_to_eager_and_revokes(caplog):
    """pgw#672/pgw#673: a failing mandatory-lane compiled call DEGRADES to
    explicit eager (revocation flips the wire tier) instead of raising —
    a broken optimization must never kill a serving worker. The old
    behavior (raise, function disabled, pod retired) produced the sm120
    CantSplit $0.25-for-nothing pods and the L4 finalize churn loop."""
    calls = {"eager": 0, "revoked": []}

    def eager(value):
        calls["eager"] += 1
        return value

    def broken(_value):
        raise RuntimeError("graph miss")

    signal = {"callback": calls["revoked"].append}
    guarded = cc._guarded(
        eager, broken, "transformer", fail_closed=True, failure_signal=signal)
    with caplog.at_level(logging.ERROR, logger="gen_worker.compile_cache"):
        assert guarded(1) == 1
    assert guarded(2) == 2
    assert calls["eager"] == 2
    # Revoked exactly once, loudly, before the first eager fallback.
    assert len(calls["revoked"]) == 1
    assert "graph miss" in calls["revoked"][0]
    assert any("DEGRADED" in r.message for r in caplog.records)


def test_guard_revocation_failure_latches_fail_closed_for_optional_execution_lane():
    calls = {"compiled": 0, "eager": 0, "callback": 0}

    def eager(value):
        calls["eager"] += 1
        return value

    def broken(_value):
        calls["compiled"] += 1
        raise RuntimeError("graph failed")

    def revoke(_detail):
        calls["callback"] += 1
        raise RuntimeError("state path unavailable")

    signal = {"callback": revoke}
    guarded = cc._guarded(eager, broken, "transformer", failure_signal=signal)
    with pytest.raises(
        cc.CompiledExecutionLaneUnavailableError, match="revocation failed",
    ):
        guarded(1)
    with pytest.raises(
        cc.CompiledExecutionLaneUnavailableError, match="revocation failed",
    ):
        guarded(2)
    assert calls == {"compiled": 1, "eager": 0, "callback": 1}


def test_guard_records_cache_proof_on_the_exact_wrapped_object(monkeypatch):
    counters = iter((
        {"fxgraph_cache_hit": 10, "fxgraph_cache_miss": 2},
        {"fxgraph_cache_hit": 13, "fxgraph_cache_miss": 3},
    ))
    monkeypatch.setattr(cc, "inductor_counters", lambda: next(counters))
    signal = {
        "callback": None,
        "lock": threading.Lock(),
        "successful_calls": 0,
        "cache_hits": 0,
        "cache_misses": 0,
    }

    guarded = cc._guarded(
        lambda value: value,
        lambda value: value + 1,
        "transformer",
        failure_signal=signal,
    )
    assert guarded(4) == 5
    assert guarded(5) == 6
    assert signal["successful_calls"] == 2
    assert signal["cache_hits"] == 3
    assert signal["cache_misses"] == 1


def test_unwrap_restores_eager():
    class _Mod:
        def forward(self, x):  # pragma: no cover
            return x

    class _Pipe2:
        def __init__(self):
            self.transformer = _Mod()

    pipe = _Pipe2()
    original = pipe.transformer.forward
    # simulate an armed pipeline the way apply() records it
    pipe._cozy_compile = {
        "targets": ["transformer"],
        "shapes": [(768, 768)],
        "cache": True,
        "originals": [(pipe.transformer, "forward", original)],
    }
    pipe.transformer.forward = lambda x: x  # the "compiled" wrap
    assert cc.unwrap(pipe) is True
    assert pipe.transformer.forward == original
    assert getattr(pipe, "_cozy_compile", None) is None
    assert cc.unwrap(pipe) is False  # idempotent


def test_regional_clear_and_guard():
    """ie#381 regional mode: blocks are compiled in place (nn.Module.compile
    sets _compiled_call_impl); rollback must CLEAR them, and the guard's
    first failure does so before retrying eager."""

    class _Block:
        _compiled_call_impl = None

    class _Mod:
        def __init__(self):
            self.b1, self.b2 = _Block(), _Block()

        def modules(self):
            return [self, self.b1, self.b2]

        def forward(self, x):
            if getattr(self.b1, "_compiled_call_impl", None) is not None:
                raise RuntimeError("compiled block exploded")
            return x + 1

    mod = _Mod()
    mod.b1._compiled_call_impl = object()  # "compiled"
    mod.b2._compiled_call_impl = object()
    guarded = cc._guarded_regional(mod, mod.forward, "transformer")
    assert guarded(1) == 2  # failure -> cleared -> eager retry succeeds
    assert mod.b1._compiled_call_impl is None
    assert mod.b2._compiled_call_impl is None
    assert guarded(2) == 3  # stays eager


def test_unwrap_clears_regional_mods():
    class _Block:
        _compiled_call_impl = object()

    class _Mod:
        def __init__(self):
            self.block = _Block()

        def modules(self):
            return [self, self.block]

    class _Pipe3:
        pass

    pipe = _Pipe3()
    mod = _Mod()
    pipe._cozy_compile = {
        "targets": ["transformer"], "shapes": [(960, 544, 241)],
        "cache": True, "originals": [], "regional_mods": [mod],
    }
    assert cc.unwrap(pipe) is True
    assert mod.block._compiled_call_impl is None


def test_system_repo():
    assert cc.system_repo("sd15") == "root/family-sd15"
    with pytest.raises(ValueError):
        cc.system_repo("")


# ---------------------------------------------------------------------------
# pack / unpack
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# fleet delivery axes + named refusal reasons
# ---------------------------------------------------------------------------


def _module_tree_pipe(wrapper_name: str):
    torch = pytest.importorskip("torch")

    class _Tree:
        def __init__(self):
            self.transformer = torch.nn.Sequential(
                torch.nn.Linear(16, 16), torch.nn.SiLU(),
            )

    return type(wrapper_name, (_Tree,), {})()


def test_execution_contract_ignores_pipeline_wrapper_class():
    """gw#577 axis (c): conversion traces via generic DiffusionPipeline
    (-> LTX2Pipeline) while serving wraps the SAME module tree in
    LTX2ConditionPipeline. torch.compile wraps target callables — the
    pipeline class never enters a traced graph — so the signature must key
    on the traced module structure only (dual-load probe: identical trees,
    sig 2625baca vs e0f356f5 under the old class-name hash)."""
    cfg = Compile(shapes=((1024, 1024),), targets=("transformer",), family="fam")
    conv_sig, conv_wc = cc.execution_contract(
        _module_tree_pipe("LTX2Pipeline"), cfg)
    serve_sig, serve_wc = cc.execution_contract(
        _module_tree_pipe("LTX2ConditionPipeline"), cfg)
    assert conv_sig == serve_sig
    assert conv_wc == serve_wc
    # genuine structural drift still produces a different signature
    torch = pytest.importorskip("torch")
    other = _module_tree_pipe("LTX2Pipeline")
    other.transformer = torch.nn.Sequential(torch.nn.Linear(32, 32))
    assert cc.execution_contract(other, cfg)[0] != conv_sig


def test_w8a8_enable_refusal_carries_exact_reason(tmp_path, monkeypatch):
    """gw#577 axis (a): the raised CompiledExecutionLaneUnavailableError is the
    ONLY wire-visible diagnostic on a serve pod, so it must name the cause.

    pgw#1181 leaves it ONE cause. The key-mismatch and drift halves refused a
    delivered `torch-inductor-cache` cell, and that format has no writer and
    is deleted; a w8a8 pipeline that arms nothing now has exactly one reason —
    no cell — and `enable` takes no artifact with which to have another."""
    pytest.importorskip("torch")
    pipe = _module_tree_pipe("Serving")
    pipe._cozy_weight_lane = "w8a8"
    cfg = Compile(shapes=((768, 768),), targets=("transformer",), family="fam")

    with pytest.raises(cc.CompiledExecutionLaneUnavailableError) as exc:
        cc.enable(pipe, cfg)
    message = str(exc.value)
    assert "no cell artifact delivered" in message
    assert "W8A8" in message
    assert "not a W8A8 production lane" in message


class _FakePipe:
    pass


def test_apply_stays_eager_without_cache():
    """No verified artifact and no explicit cold opt-in => untouched pipeline."""
    pipe = _FakePipe()
    cfg = Compile(shapes=((768, 768),))
    assert cc.apply(pipe, cfg, cache_ready=False) is False
    assert getattr(pipe, "_cozy_compile", None) is None


# ---------------------------------------------------------------------------
# declaration plumbing
# ---------------------------------------------------------------------------


def test_compile_struct_validation():
    c = Compile(shapes=[[768, 768], (1024, 1024)], family=" sd15 ")
    assert c.shapes == ((768, 768), (1024, 1024))
    assert c.targets == ("transformer", "vae.decode")
    assert c.family == "sd15"
    assert c.text_len is None       # None = undeclared (walk-time lint)
    assert c.dynamic == ()
    assert c.regional is False
    assert Compile(shapes=((960, 544, 241),), targets=("transformer",),
                   family="ltx-2.3", regional=True).regional is True
    # SDK v2 text axis: 0 = explicitly unconditioned; >0 = pinned length.
    assert Compile(shapes=((1024, 1024),), text_len=0).text_len == 0
    assert Compile(shapes=((1024, 1024),), text_len=512).text_len == 512
    seq = DynamicDim(dim="sequence", min=2, max=512)
    c2 = Compile(shapes=((1024, 1024),), dynamic=(seq,))
    assert c2.sequence_dynamic == seq and c2.batch_dynamic is None
    with pytest.raises(ValueError):
        Compile(shapes=())
    with pytest.raises(ValueError):
        Compile(shapes=((0, 768),))
    with pytest.raises(ValueError):
        Compile(shapes=((768, 768),), targets=())
    with pytest.raises(ValueError):
        Compile(shapes=((768, 768),), text_len=-1)
    with pytest.raises(TypeError):
        Compile(shapes=((768, 768),), dynamic=("sequence",))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="repeats"):
        Compile(shapes=((768, 768),), dynamic=(
            DynamicDim(dim="batch", min=2, max=4),
            DynamicDim(dim="batch", min=2, max=8),
        ))


def test_dynamic_dim_validation():
    d = DynamicDim(dim=" Batch ", min=2, max=8)
    assert (d.dim, d.min, d.max) == ("batch", 2, 8)
    # The two-literal "batch"|"sequence" wall is REPLACED — wan's
    # latent-spatial axis is the red case (ie#550/ie#566 hit it from both
    # sides). A named axis constructs; validation moved to Compile, which
    # cross-references the name against declared Compile.dims.
    lat = DynamicDim(dim="latent_h", min=90, max=160)
    assert lat.dim == "latent_h"
    with pytest.raises(ValueError, match="latent_h"):
        Compile(shapes=((768, 768),), dynamic=(lat,))  # no Dim declares it
    with pytest.raises(ValueError):
        DynamicDim(dim="not an identifier", min=2, max=8)
    with pytest.raises(ValueError, match=">= 2"):
        DynamicDim(dim="batch", min=1, max=8)  # torch 0/1 specialization
    with pytest.raises(ValueError, match="exceed"):
        DynamicDim(dim="sequence", min=4, max=4)


def test_compile_struct_video_shapes():
    """(w, h, frames) rows — video graphs key on the frame axis."""
    c = Compile(
        family="ltx-2.3",
        shapes=[[1280, 704, 121], (960, 544, 241), (1920, 1088, 241)],
        targets=("transformer",),
    )
    assert c.shapes == ((1280, 704, 121), (960, 544, 241), (1920, 1088, 241))
    # mixed image + video rows are fine (one endpoint, two modality functions)
    m = Compile(shapes=((768, 768), (1280, 704, 121)))
    assert m.shapes == ((768, 768), (1280, 704, 121))
    with pytest.raises(ValueError):
        Compile(shapes=((1280, 704, 0),))
    with pytest.raises(ValueError):
        Compile(shapes=((1280,),))
    with pytest.raises(ValueError):
        Compile(shapes=((1280, 704, 121, 24),))


def test_guidance_regimes_are_an_envelope_axis():
    """SDK v2: warm guidance regimes live on the enriched CompileCell (from
    payload CompileAxis classes), and remain part of the declared envelope.

    pgw#1181 asserts that on the DECLARATION rather than on a
    `torch-inductor-cache` cell's metadata: `declared_compile_facts` is the
    canonical form of the declared envelope, it is what the arm identity and
    the cell key are both built from, and it outlived the format."""
    cfg = CompileCell(
        shapes=((1024, 1024),), targets=("transformer",), family="sdxl",
        regional=False, text_len=0, dynamic=(), lora_bucket=0,
        guidance_scales=(5.0, 0.0),
    )
    assert cfg.guidance_scales == (5.0, 0.0)
    assert cc.declared_compile_facts(cfg)["guidance"] == [0.0, 5.0]


class In(msgspec.Struct):
    prompt: str = ""


class Out(msgspec.Struct):
    ok: bool = True


class GuidedIn(msgspec.Struct):
    """SDK v2: CFG regimes are payload-field equivalence classes."""

    prompt: str = ""
    guidance_scale: Annotated[float, CompileAxis(classes=(
        AxisClass("cfg_on", match=lambda v: v != 0, warm=5.0),
        AxisClass("cfg_off", match=lambda v: v == 0, warm=0.0),
    ))] = 5.0


def test_endpoint_compile_reaches_spec():
    import types

    @endpoint(
        resources=Resources(vram_gb_hint=4),
        compile=Compile(shapes=((768, 768),), text_len=0),
    )
    class Ep:
        def setup(self) -> None:
            pass

        def gen(self, ctx, p: GuidedIn) -> Out:
            return Out()

    mod = types.SimpleNamespace(Ep=Ep)
    specs = collect_from_namespace(mod)
    assert len(specs) == 1
    assert specs[0].compile is not None
    assert specs[0].compile.shapes == ((768, 768),)
    cell = specs[0].compile_cell()
    assert cell is not None
    assert cell.shapes == ((768, 768),)
    assert cell.text_len == 0
    # Warm guidance derives from the payload CompileAxis classes, in class
    # declaration order (Compile(guidance_scales=...) is deleted in v2).
    assert cell.guidance_scales == (0.0, 5.0)  # class union, sorted

    with pytest.raises(TypeError, match="compile="):
        @endpoint(compile="yes")  # type: ignore[arg-type]
        def bad(ctx, p: In) -> Out:
            return Out()

    # SDK v2 lint: an inference compile= endpoint must declare its
    # text-sequence axis — text_len or a dynamic "sequence" dim.
    with pytest.raises(ValueError, match="text-sequence"):
        @endpoint(compile=Compile(shapes=((768, 768),)))
        def unlinted(ctx, p: In) -> Out:
            return Out()

    @endpoint(compile=Compile(
        shapes=((768, 768),),
        dynamic=(DynamicDim(dim="sequence", min=2, max=512),),
    ))
    def dyn_ok(ctx, p: In) -> Out:
        return Out()

    # Constructing a bare Compile outside @endpoint never lints.
    assert Compile(shapes=((768, 768),)).text_len is None


def test_flavor_label_carries_weight_lane_gw534() -> None:
    from gen_worker.compile_cache import flavor_label, execution_lane_token

    assert flavor_label("rtx-4090", "2.9.1+cu128") == "inductor-rtx-4090-torch2.9"
    assert flavor_label("h100-80gb-hbm3", "2.13.0+cu130", "w8a8") == (
        "inductor-h100-80gb-hbm3-torch2.13-w8a8")
    assert flavor_label("rtx-4090", "2.9.1", "fp8-hooks") == (
        "inductor-rtx-4090-torch2.9-w8a16")
    assert execution_lane_token("") == "" and execution_lane_token("w8a8") == "w8a8"


def test_resolve_pipeline_class_gw586() -> None:
    """gw#586 call-path parity: a mint may name the SERVING pipeline class;
    unknown names refuse loudly — a silent generic fallback would trace the
    wrong call path and publish a cell no serving lookup can hit."""
    from gen_worker.compile_cache import resolve_pipeline_class

    cls = resolve_pipeline_class("DiffusionPipeline")
    assert callable(getattr(cls, "from_pretrained", None))

    with pytest.raises(RuntimeError, match="wrong call path"):
        resolve_pipeline_class("NoSuchPipelineClass")
    with pytest.raises(RuntimeError, match="non-empty"):
        resolve_pipeline_class("   ")
    # A diffusers attribute that is not a loadable pipeline class refuses too.
    with pytest.raises(RuntimeError, match="wrong call path"):
        resolve_pipeline_class("__version__")


# ---------------------------------------------------------------------------
# resident prep-mode drift (off <-> vae_only) converges to the cell
# ---------------------------------------------------------------------------


# `test_arm_staged_artifact_reconciles_resident_drift` is deleted with
# `arm_staged_artifact` itself — the STRICT arm entry point existed only for
# hub-commanded hot adoption, which nothing has ever dispatched. The resident
# convergence it checked is the same one `enable()` performs, and
# `test_enable_reconciles_*` above/below cover it on the live path.


# pgw#1181 REMOVED `test_reconcile_resident_mode_unit` with its subject.
# `models.memory.reconcile_resident_mode` adjusted a live pipeline's offload
# mode to match the `low_vram_mode` a DELIVERED `torch-inductor-cache` cell had
# recorded, and `compile_cache._reconcile_resident_mode` was its only caller.
# The mode is still READ as a fact — `execution_contract_digest` folds it into
# the contract — but nothing reconciles a pipeline to a cell's recorded mode,
# because no cell records one.


def test_aot_autograd_cache_disabled_for_portability(monkeypatch, tmp_path):
    """gw#608 revert-turns-red: the AOTAutogradCache key embeds the decomp
    table function's process memory address (ASLR), so its entries can never
    hit across pods and an AOT miss skips the portable on-disk FX entries
    (live: 8/8 misses on byte-identical FX keys, two hosts). Both the
    capture/seed env contract and apply() must pin the AOT layer OFF so the
    FX cache is the lookup surface, symmetrically for producer and consumer."""
    import torch._functorch.config as fconf

    from gen_worker import settings_authority as sa

    monkeypatch.delenv("TORCHINDUCTOR_AUTOGRAD_CACHE", raising=False)
    monkeypatch.setattr(fconf, "enable_autograd_cache", True)
    # The pin is `settings_authority.disable_autograd_cache`, the ONE
    # writer of torch settings. `capture_env` used to call it on the
    # way to writing a `torch-inductor-cache` capture and is deleted with that
    # format; the invariant is unchanged and is asserted at its owner.
    sa.disable_autograd_cache()
    assert os.environ["TORCHINDUCTOR_AUTOGRAD_CACHE"] == "0"
    assert fconf.enable_autograd_cache is False

    monkeypatch.setattr(fconf, "enable_autograd_cache", True)

    class _P:
        pass

    class _Cfg:
        shapes = ((64, 64),)
        targets = ("transformer",)
        regional = False

    # CPU box: apply() exits eager after the CUDA check — the AOT disable
    # must already have happened (it precedes every compile decision).
    cc.apply(_P(), _Cfg(), cache_ready=False)
    assert fconf.enable_autograd_cache is False


def test_aot_autograd_cache_disabled_across_threads(monkeypatch, tmp_path):
    """gw#608 residual (live-disproven 0.40.5, B200, 2026-07-21): torch>=2.13
    config user overrides are ContextVars — THREAD-LOCAL. The arming thread's
    ``enable_autograd_cache = False`` was invisible to the warmup thread that
    actually compiled, so the mint still packed ASLR-keyed AOT entries and the
    store-served sibling still missed 8/8. The disable must bind EVERY thread
    (entry-level env force), not just the caller's."""
    import threading

    import torch._functorch.config as fconf

    entry = fconf._config["enable_autograd_cache"]
    had_force = "env_value_force" in entry.__dict__
    old_force = entry.__dict__.get("env_value_force")
    monkeypatch.delenv("TORCHINDUCTOR_AUTOGRAD_CACHE", raising=False)
    monkeypatch.setattr(fconf, "enable_autograd_cache", True)
    try:
        entry.__dict__.pop("env_value_force", None)  # simulate no pre-import env
        from gen_worker import settings_authority as sa

        sa.disable_autograd_cache()  # the pin's owner, see above

        seen: dict = {}

        def probe() -> None:
            seen["value"] = fconf.enable_autograd_cache

        t = threading.Thread(target=probe)
        t.start()
        t.join()
        assert seen["value"] is False, (
            "AOT autograd cache still enabled on a sibling thread — the "
            "warmup/compile thread would repack ASLR-keyed AOT entries and "
            "consumers would miss 8/8 (gw#608)"
        )
    finally:
        if had_force:
            entry.env_value_force = old_force
        else:
            entry.__dict__.pop("env_value_force", None)


# ---------------------------------------------------------------------------
# FX-key forensics
# ---------------------------------------------------------------------------


# pgw#1200 REMOVED the three `fx_key_forensics` rows and the two
# `fx_cache_failure_report` rows that survived pgw#1181.
#
# `fx_key_forensics` diffed the CELL's recorded FxGraphHashDetails lines
# against the boot's; with the `torch-inductor-cache` format deleted there is
# no cell side to diff, and `fx_cache_failure_report` was its only production
# caller — pgw#1181 kept all three alive by anchoring them here, which is the
# reachability trap this rewrite exists to close. The helpers go with them.
#
# The report itself is NOT deleted: the dynamo lane still asks for its live
# FX-cache state on a failed warmup proof. What it can honestly say — the live
# key census, the missing-directory case and the extern-libs key — is driven
# in `tests/test_fx_report_live_only_pgw1200.py`, against the real function.


# pgw#1181 REMOVED `test_fx_cache_failure_report_names_b2_samekey_resave`.
# `samekey_resaves` counts entry files that a boot re-saved under a key the
# CELL had already seeded, so it is undefined without a cell — and the cell
# half of `fx_cache_failure_report` read FX entries out of a
# `torch-inductor-cache` tarball, a format with no writer, deleted here. The
# live-directory census the sibling row now fences is the half a pod can still
# reach: what the executor hands this function is an exported cell, which
# carries no `inductor/` tree at all.

