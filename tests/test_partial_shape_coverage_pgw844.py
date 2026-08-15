"""One undispatchable aspect bucket must not cost the pod its compiled lane.

The shape of the hazard: a transformer block sees ``(B, H_lat*W_lat, C)`` — the
token PRODUCT — while entries are keyed on the latent H and W separately, so
sdxl's nine aspect buckets collapse to four distinct token counts and eight of
the nine admit more than one entry, refusing ``entry_ambiguous``. Under the
all-or-nothing rule that cost even 1024x1024, whose 128x128 latent is unique and
whose entry was armed and correct.

Two independent defects produced that, and both are asserted here on the real
paths (:class:`aot_serve.EntryDispatch` for dispatch, the executor's own
derived warm plan and ``ensure_setup`` for the boot):

1. **the all-or-nothing coverage rule** — an alias attributed to an object
   only when EVERY declared graph class proved there, so one unservable
   bucket withheld the whole target (``target_applicability_incomplete``) and
   the pod advertised eager for every shape;
2. **the exported lane was never asked for its revocation signal** —
   ``_bind_compile_guard`` probed the TRT lane (since deleted) and dynamo
   only, and
   ``provision.enable_compiled`` returns as soon as ``arm_aot`` succeeds, so
   an AOT-armed pipeline carries no ``compile_cache`` ``failure_signal``
   marker at all.  Every AOT arm therefore answered "no runtime guard
   revocation signal" and had its ``active_compile_ref`` cleared — a compiled
   AOT serve was structurally unreachable on the boot path even with a cell
   that dispatches perfectly.

``boot_ended_uncompiled`` now means "nothing is dispatchable", never
"something wasn't", and every shape that falls back is named at ingress and
charged ``fallback_reason=ingress_refused`` on its own request row.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated, Any, List, Tuple, cast

import msgspec
import pytest

torch = pytest.importorskip("torch")

from gen_worker import (  # noqa: E402
    AxisClass,
    Compile,
    CompileAxis,
    Resources,
    endpoint,
)
from gen_worker import aot_serve  # noqa: E402
from torch_compiled_graphs import (  # noqa: E402
    CallIngress,
    CallInput,
    CompiledGraphRunner,
)
from gen_worker import compile_cache as cc  # noqa: E402
from gen_worker import serving_mode  # noqa: E402
import gen_worker.executor as executor_mod  # noqa: E402
from gen_worker.api.binding import Hub, wire_ref  # noqa: E402
from gen_worker.executor import Executor, _Job  # noqa: E402
from gen_worker.models import provision  # noqa: E402
from gen_worker.pb import worker_scheduler_pb2 as pb  # noqa: E402
from gen_worker.registry import extract_specs  # noqa: E402
from gen_worker.models import store as store_mod

FAMILY = "sdxl"
COMPILED_GRAPH_KEY = "cg-key-v1-" + "8" * 56
CELL_REF = f"root/family-{FAMILY}#{COMPILED_GRAPH_KEY}"
CELL_DIGEST = "blake3:" + "a" * 64
MODEL_DIGEST = "blake3:" + "c" * 64
CHANNELS = 320
TEXT_LEN = 77

#: sdxl's nine declared aspect buckets, as LATENT extents (pixels // 8) —
#: exactly the rows attempt eleven's cell was minted over.
SDXL_BUCKETS: Tuple[Tuple[int, int], ...] = (
    (128, 128),
    (152, 104), (104, 152),
    (168, 96), (96, 168), (144, 112), (112, 144),
    (192, 80), (80, 192),
)


def _entry_name(h: int, w: int, *, cfg: bool = False) -> str:
    return (f"unet/block=BasicTransformerBlock#0,cfg={str(cfg).lower()}"
            f"/B=1,H_lat={h},T_txt={TEXT_LEN},W_lat={w}")


def _entry_contract(h: int, w: int) -> CallIngress:
    """One entry block exactly as the regional mint packs it: keyed on H_lat
    and W_lat, but the BLOCK's own input carries only their product."""
    return CallIngress(
        parameters=("hidden_states", "encoder_hidden_states"),
        flat_arity=2,
        inputs=(
            CallInput(
                "hidden_states", 0, "hidden_states", 0, (), "hidden_states",
                "bfloat16", (1, h * w, CHANNELS),
            ),
            CallInput(
                "encoder_hidden_states", 1, "encoder_hidden_states", 1, (),
                "encoder_hidden_states", "bfloat16", (1, TEXT_LEN, 2048),
            ),
        ),
    )


class _FakeTCGRunner:
    bound = True
    declared_fqns: tuple[str, ...] = ()

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, *feeds: Any) -> Any:
        self.calls += 1
        return feeds[0]


def _entry_runner(contract: CallIngress, entry: str) -> aot_serve.TCGEntryRunner:
    """The live worker policy around the minimal TCG-owned runner surface."""
    return aot_serve.TCGEntryRunner(
        runner=cast(CompiledGraphRunner, _FakeTCGRunner()),
        contract=contract,
        module_name="unet",
        entry=entry,
        family=FAMILY,
    )


def _runner(h: int, w: int) -> aot_serve.TCGEntryRunner:
    return _entry_runner(_entry_contract(h, w), _entry_name(h, w))


def _dispatch(buckets: Tuple[Tuple[int, int], ...]) -> aot_serve.EntryDispatch:
    return aot_serve.EntryDispatch(tuple(
        (_entry_name(h, w), _runner(h, w)) for h, w in buckets))


def _call(h: int, w: int) -> Tuple[Any, ...]:
    return (
        torch.zeros(1, h * w, CHANNELS, dtype=torch.bfloat16),
        torch.zeros(1, TEXT_LEN, 2048, dtype=torch.bfloat16),
    )


# ---------------------------------------------------------------------------
# 1. The ambiguity itself, at the real dispatch boundary
# ---------------------------------------------------------------------------


def test_sdxl_nine_buckets_collapse_to_four_token_counts_at_real_dispatch():
    """The pod's own table, reproduced through ``EntryDispatch.select``.

    This is the reproduction, not an illustration: the entries are parsed by
    TCG's closed :class:`CallIngress` and admitted by the worker's real
    ingress path. One bucket dispatches; eight are ``entry_ambiguous``.
    """
    dispatch = _dispatch(SDXL_BUCKETS)

    name, _runner_ = dispatch.select(_call(128, 128), {})
    assert name == _entry_name(128, 128)

    groups = {15360: 2, 15808: 2, 16128: 4}
    for h, w in SDXL_BUCKETS:
        if (h, w) == (128, 128):
            continue
        with pytest.raises(aot_serve.IngressContractError) as err:
            dispatch.select(_call(h, w), {})
        assert err.value.reason == "entry_ambiguous"
        assert f"{groups[h * w]} entries admit this call" in str(err.value)


def test_the_collapsed_declaration_dispatches_every_bucket_uniquely():
    """pgw#829's remedy, verified from the SERVING side: one entry over the
    token hull admits every declared bucket and is unique by construction."""
    hull = min(h * w for h, w in SDXL_BUCKETS), max(
        h * w for h, w in SDXL_BUCKETS)
    collapsed = aot_serve.EntryDispatch(((
        "unet/block=BasicTransformerBlock#0,cfg=false",
        _entry_runner(
            CallIngress(
                parameters=("hidden_states", "encoder_hidden_states"),
                flat_arity=2,
                inputs=(
                    CallInput(
                        "hidden_states", 0, "hidden_states", 0, (),
                        "hidden_states", "bfloat16", (1, "s_tok", CHANNELS),
                    ),
                    CallInput(
                        "encoder_hidden_states", 1, "encoder_hidden_states", 1,
                        (), "encoder_hidden_states", "bfloat16",
                        (1, TEXT_LEN, 2048),
                    ),
                ),
                symbols=(("s_tok", hull),),
            ),
            "unet/block=BasicTransformerBlock#0,cfg=false",
        ),
    )))

    for h, w in SDXL_BUCKETS:
        name, _r = collapsed.select(_call(h, w), {})
        assert name == "unet/block=BasicTransformerBlock#0,cfg=false"


# ---------------------------------------------------------------------------
# 2. The boot: a partially dispatchable cell keeps the compiled lane
# ---------------------------------------------------------------------------

_LATENT = {"1:1": (128, 128), "16:9": (192, 80), "9:16": (80, 192)}
_ASPECT_AXIS = CompileAxis(classes=(
    AxisClass("sq", match=lambda v: v == "1:1", warm="1:1"),
    AxisClass("wide", match=lambda v: v == "16:9", warm="16:9"),
    AxisClass("tall", match=lambda v: v == "9:16", warm="9:16"),
))


class _In(msgspec.Struct):
    prompt: str = ""
    aspect_ratio: Annotated[str, _ASPECT_AXIS] = "1:1"


class _Out(msgspec.Struct):
    y: str = ""


class _Unet:
    def __init__(self) -> None:
        self.eager_calls = 0

    def forward(self, hidden: Any, encoder: Any) -> Any:
        self.eager_calls += 1
        return hidden


class _Pipe:
    def __init__(self) -> None:
        self.unet = _Unet()

    @classmethod
    def from_pretrained(cls, path, **kwargs):  # pragma: no cover - patched
        return cls()


@endpoint(
    models={"pipeline": Hub("acme/sdxl")},
    resources=Resources(gpu=True),
    compile=Compile(
        shapes=((1024, 1024),), family=FAMILY, text_len=TEXT_LEN,
        targets=("unet",), regional=True),
)
class _SdxlRegional:
    def setup(self, pipeline: _Pipe) -> None:
        self.pipeline = pipeline

    def generate(self, ctx, payload: _In) -> _Out:
        h, w = _LATENT[payload.aspect_ratio]
        self.pipeline.unet.forward(*_call(h, w))
        return _Out(y="ok")


def _arm(pipe: _Pipe, *_args: Any) -> Any:
    """The compile-arm LEAF, faked exactly as far as the ``.pt2`` load: the
    dispatch, the wrap, the marker and the refusal path are real."""
    from gen_worker import fleet_cells

    meta = {"family": FAMILY, "sku": "l4", "torch": str(torch.__version__),
            "precision": "w8a8"}
    aot_serve.wrap_module(
        module=pipe.unet,
        runner=_dispatch(tuple(_LATENT.values())),
        meta=meta,
        target="unet",
    )
    # The pipeline-level format-3 marker `arm_entry` publishes — the shape
    # `is_armed` / `execution_count` / `proven_since` / the guard + refusal
    # callbacks all read.
    module_marker = getattr(pipe.unet, aot_serve._MARKER_ATTR, {})
    setattr(pipe, aot_serve._MARKER_ATTR, {
        "meta": meta,
        "targets": {"unet": {
            "module": pipe.unet, "attr": "forward",
            "state": module_marker.get("state", {})}},
    })
    return fleet_cells.ArmOutcome(armed=True)


def _cell_snapshot(tmp_path: Path) -> Path:
    """Opaque transport bytes carried through the executor's exact order.

    This suite stubs the compile-arm leaf below, so no Engine imports these
    bytes. The proof here is snapshot/selection plumbing plus live dispatch,
    not TCG artifact admission (which has its own tests)."""
    artifact = tmp_path / "compiled-graph-transfer.tar.gz"
    artifact.write_bytes(b"\x00tcg-transfer-placeholder")
    return artifact


def _arm_dynamo(pipe: _Pipe, *_args: Any) -> Any:
    """The DYNAMO control arm: same partial coverage, other backend.

    The compiled callable serves the unique token count and falls back for the
    two that collide — the same serving shape, but scored by FX cache hits
    instead of artifact invocations.
    """
    import threading

    from gen_worker import fleet_cells

    signal = {
        "callback": None, "lock": threading.Lock(),
        "successful_calls": 0, "cache_hits": 0, "cache_misses": 0,
    }
    setattr(pipe, cc._MARKER_ATTR, {
        "failure_signal": signal, "originals": [], "regional_mods": []})
    unet = pipe.unet

    def wrapped(hidden: Any, encoder: Any) -> Any:
        if int(hidden.shape[1]) == 128 * 128:
            with signal["lock"]:
                signal["successful_calls"] += 1
                signal["cache_hits"] += 1
            return hidden
        unet.eager_calls += 1
        return hidden

    unet.forward = wrapped
    return fleet_cells.ArmOutcome(armed=True)


def _boot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, exported: bool = True,
) -> Tuple[Executor, Any, _Pipe, List[Tuple[str, str, str]]]:
    events: List[Tuple[str, str, str]] = []
    monkeypatch.setattr(
        executor_mod.activity_mod, "emit_event",
        lambda kind, detail, phase="", duration_ms=0, **_kw: events.append(
            (kind, phase, detail)))
    # An `aot_serve.note_aot_key(COMPILED_GRAPH_KEY)` stood here, with a comment
    # arguing this process is "TOLD the flavor is an AOT cell exactly as a
    # Plan's `Arm.artifact` tells a pod". Production is told no such thing — it
    # LEARNS at the wrap (`arm_entry`, pgw#1141b), and the route that
    # believed the convention instead is what cost four pods. The line is
    # deleted rather than moved: `_arm` below publishes the pipeline-level
    # marker `arm_entry` publishes, so `holds_exported_cell` answers the
    # lane question off the OBJECT and the registry is not consulted at all.
    artifact = _cell_snapshot(tmp_path)
    model_dir = tmp_path / "sdxl-model"
    model_dir.mkdir()
    specs = extract_specs(_SdxlRegional)
    generate = next(s for s in specs if s.name == "generate")
    model_ref = wire_ref(generate.models["pipeline"])
    pipe = _Pipe()
    setattr(pipe, "_cozy_weight_lane", "w8a8")

    async def _send(_msg: Any) -> None:
        return None

    async def _download(ref, **kwargs):
        return artifact.parent if ref == CELL_REF else model_dir

    ex = Executor(specs, _send)
    ex.store._cache_dir = tmp_path / "cas"
    monkeypatch.setattr(store_mod, "ensure_local", _download)
    monkeypatch.setattr(
        provision, "load_slot",
        lambda *args, **kwargs: provision.SlotLoad(obj=pipe, is_pipeline=True))
    monkeypatch.setattr(
        ex, "_enable_compiled", _arm if exported else _arm_dynamo)

    # The cell is an exact ORDER (Arm.artifact), never a snapshot
    # entry the worker scans for.
    arm_order = executor_mod._ArmOrder(
        backend="aot_cell",
        selection=executor_mod._CompileArtifactSelection(
            path=artifact, ref=CELL_REF, snapshot_digest=CELL_DIGEST))
    asyncio.run(ex.ensure_setup(generate, {
        model_ref: pb.Snapshot(digest=MODEL_DIGEST),
    }, arm=arm_order))
    return ex, generate, pipe, events


def test_one_undispatchable_bucket_does_not_cost_the_boot_its_compiled_execution_lane(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """THE pgw#844 assertion, on the real derived warm plan.

    Three declared aspect classes; two of them (192x80 / 80x192) share a token
    count and are ``entry_ambiguous``, one (128x128) is unique.  The armed cell
    must survive: the target keeps its active identity, the boot does not end
    uncompiled, and the two eager classes are NAMED rather than inferred from
    a healthy-looking silence.
    """
    ex, generate, pipe, events = _boot(tmp_path, monkeypatch)

    (target,) = ex.compile_targets()
    assert target.active_compile_ref == CELL_REF, (
        "the armed, correct, unambiguous 1:1 entry must serve compiled even "
        "though two sibling buckets are undispatchable")
    assert target.active_compile_snapshot_digest == CELL_DIGEST
    assert list(target.function_names) == ["generate"]
    assert aot_serve.is_armed(pipe)

    kinds = [(kind, phase) for kind, phase, _d in events]
    assert ("self_mint_skipped", "boot_ended_uncompiled") not in kinds, (
        "`boot_ended_uncompiled` must mean nothing is dispatchable, not that "
        "something wasn't")
    assert ("serve_eager_posture", "target_applicability_incomplete") \
        not in kinds

    # The exported lane's OWN revocation signal is what
    # keeps this target advertisable — nothing installed a dynamo marker.
    assert not hasattr(pipe, cc._MARKER_ATTR)


def test_the_exported_lane_boots_on_the_eager_warm_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """pgw#1184 — an ARMED `.pt2` costs ONE warm generate, not one per class.

    The fixture's three aspect classes share one eager group (one function,
    one guidance class, one text pin), so the eager plan is one run and the
    class x bucket cross-product is three. sdxl's real declaration is 18
    classes in 2 eager groups: the same ratio, and the 18-where-2-would-do the
    tree's own comment quoted while leaving it in place.

    Nothing is being traced. A `.pt2` is ahead-of-time machine code that
    performs no FX lookup, there is no per-class cache-hit ledger to move, and
    §4.31 deleted the warm plan as a prerequisite to arming — so every run
    past the first bought nothing but wall clock.

    RED on unmodified master: `the boot ran 3 full warm generates against an
    ARMED exported cell (1 dispatched, 2 refused at ingress), want 1`.
    """
    _ex, _generate, pipe, _events = _boot(tmp_path, monkeypatch)

    dispatched = aot_serve.execution_count(pipe)
    refused = aot_serve.ingress_refusals(pipe)
    assert dispatched + refused == 1, (
        f"the boot ran {dispatched + refused} full warm generates against an "
        f"ARMED exported cell ({dispatched} dispatched, {refused} refused at "
        f"ingress), want 1")


def test_an_ambiguous_request_is_charged_ingress_refused_not_counted_compiled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """A compiled lane that fell back for ONE request must not contaminate the
    compiled measurement with an eager sample (pgw#764's whole premise).

    Only reachable once the lane stays armed through a partial plan, which is
    why it lands with pgw#844 rather than before it.
    """
    from gen_worker import postmortem

    ex, generate, pipe, _events = _boot(tmp_path, monkeypatch)
    rec = ex._classes[generate.instance_key]
    instance = rec.instance

    def _served(aspect: str, request_id: str) -> Any:
        job = _Job(request_id=request_id, attempt=1, spec=generate)
        ex.jobs[request_id] = job
        token = postmortem.note_inflight(
            "request", generate.name, request_id=request_id)
        try:
            instance.generate(None, _In(aspect_ratio=aspect))
        finally:
            postmortem.clear_inflight(token)
        return ex._served_identity(generate, job)

    compiled = _served("1:1", "req-compiled")
    assert compiled.serving_mode == serving_mode.MODE_AOT_CELL
    assert compiled.served_cell_ref == CELL_REF
    assert compiled.served_eager_fallback is False
    assert compiled.fallback_reason == ""

    refused = _served("16:9", "req-ambiguous")
    assert refused.serving_mode == serving_mode.MODE_AOT_CELL
    assert refused.served_eager_fallback is True
    assert refused.fallback_reason == serving_mode.FALLBACK_INGRESS_REFUSED


def test_the_partial_coverage_claim_is_scoped_to_the_exported_execution_lane(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """The relaxation is a statement about the EXPORTED lane only.

    An exported artifact refuses a shape it cannot serve BY NAME, counts it,
    emits ``aot_ingress_refused`` and stays armed — the degradation is per
    shape and fully visible, which is what makes "compiled for these, eager
    for those" an honest thing to advertise.  An unproven dynamo graph class
    is an unannounced recompile at serve time, so the same warm plan on the
    dynamo lane must NOT produce a partial-coverage claim.
    """
    _ex, _generate, pipe, _events = _boot(tmp_path, monkeypatch, exported=False)

    # The FULL plan, all three classes: one served by the compiled callable,
    # two fell through to eager. On this lane an unproven class is an
    # unannounced recompile, so the runs are the only detector there is —
    # which is exactly why pgw#1184 cut the exported lane out of the plan and
    # left this one alone.
    assert cc.execution_count(pipe) == 1
    assert pipe.unet.eager_calls == 2
