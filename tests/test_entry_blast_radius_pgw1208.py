"""pgw#1208: one class that cannot export must not cost the other thirty-five.

WHAT HAPPENED. The first sdxl full-circle leg reached the compile phase and
died there, deterministically, on pod ``fry7suf4xgycie`` (RTX A4000, 0.113.2):

    entry 'unet/adapter=true,cfg=true/B=2,H_lat=80,T_txt=77,W_lat=192':
    torch.export(strict=True) failed for UNet2DConditionModel:
    Unsupported: Skip inlining `torch.compiler.disable()`d function
      Explanation: … <function ModuleGroup.onload_> … wrapped with
      `torch.compiler.disable`

``ModuleGroup.onload_`` is **diffusers group offloading**. The mint child ran
the worker's SERVING placement ladder, the ladder engaged group offload (a
serving parent was already resident on that card), and the offload hooks put a
``@torch.compiler.disable``d function in the traced path — which
``torch.export(strict=True)`` refuses, fatally.

THE CONTROLLED COMPARISON, which is what makes the diagnosis rather than a
guess: the SAME entry name, on the SAME wheel (0.113.2), exported in **91.07 s**
on an A4500 (`ft5vr2zg86jwmi`) and was `Unsupported` on the A4000. Six
adapter-bearing entries exported clean there. The adapter axis is exonerated —
rows are ordered adapter-bearing first (`aot_mint._rows_source`), so the
adapter entry was simply the first one tried.

TWO FIXES, AND THE SECOND IS THE MORE IMPORTANT
-----------------------------------------------
1. The mint child no longer PLACES on the weight-free path (``place=False``) —
   the pgw#1124 seam, applied one door over. The ladder is for a pipeline that
   will run a forward, and this one only ever exports. The hooks are never
   installed rather than stripped afterwards, which is why this is small.
2. **A deterministic per-entry export failure no longer kills the mint.** That
   pod threw away nothing (it died on its first entry), but the A4500 shows the
   shape that matters: 6 classes exported clean, and under the old code a
   refusal at row 7 would have discarded all six. The entry is already the unit
   of identity and of publish (pgw#718); it is now the unit of FAILURE too.

Fail-closed stays fail-closed AT THE ENTRY: a skipped class is never packed and
never published, and serving covers it eager by the same mechanism that covers
any shape outside a cell's declared envelope (pgw#844). Only the blast radius
changed.
"""

from __future__ import annotations

import types
from pathlib import Path
from typing import Any, Dict, List

import pytest

torch = pytest.importorskip("torch")

import torch.nn as nn  # noqa: E402

from torch._dynamo.exc import Unsupported  # noqa: E402

from gen_worker import aot_mint, aot_serve, compile_cache  # noqa: E402
from gen_worker.api.decorators import Compile  # noqa: E402
from gen_worker.api.export_contract import (  # noqa: E402
    Dim,
    GraphClass,
    Input,
    register_export_declaration,
    reset_export_declarations,
)

pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")

FAMILY = "tiny1208"


class TinyUNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, sample: Any) -> Any:
        return torch.tanh(self.lin(sample)) + 1.0


def _declare() -> Any:
    """TWO declared graph classes, so 'one failed' and 'the rest survived' are
    distinguishable outcomes."""
    return register_export_declaration(Compile(
        family=FAMILY,
        targets=("unet",),
        dims=(Dim("B", carried_by=(("sample", 0),)),),
        classes=(GraphClass(dims={"B": 2}), GraphClass(dims={"B": 1})),
        inputs=(Input("sample", shape=("B", 4), dtype="model"),),
        shape_strategy="static-rows",
        warm_changes_key=False,
    ))


@pytest.fixture(autouse=True)
def _fresh_registry() -> Any:
    reset_export_declarations()
    yield
    reset_export_declarations()


@pytest.fixture
def fake_sm(monkeypatch: pytest.MonkeyPatch) -> Dict[str, str]:
    full = {"sku": "", "sm": "sm_89", "torch": str(torch.__version__), "cuda": ""}
    monkeypatch.setattr(compile_cache, "runtime_key", lambda: dict(full))
    monkeypatch.setattr(aot_serve, "runtime_key", lambda: dict(full))
    return full


def _mint(tmp: Path) -> aot_mint.MintResult:
    pipe = types.SimpleNamespace(unet=TinyUNet())
    spec = aot_mint.ExportSpec(family=FAMILY, target="")
    # K=1: this file is about the export loop, and the serial path exercises
    # the same `_rows_source` the overlapped one does (one body, by design).
    return aot_mint.mint(pipe, spec, tmp, entry_workers=1)


def _fail_one(monkeypatch: pytest.MonkeyPatch, *, on: str, exc: BaseException) -> List[str]:
    """Make ONE declared class refuse at export. Returns the names attempted."""
    real = aot_mint._export_entry
    seen: List[str] = []

    def _patched(pipeline: Any, spec: Any, plan: Any, decl: Any, **kw: Any) -> Any:
        from gen_worker import aot_declaration as _decl

        name = _decl.plan_entry_name(plan)
        seen.append(name)
        if on in name:
            raise exc
        return real(pipeline, spec, plan, decl, **kw)

    monkeypatch.setattr(aot_mint, "_export_entry", _patched)
    return seen


# ---------------------------------------------------------------------------
# The blast radius
# ---------------------------------------------------------------------------


def test_a_deterministic_refusal_skips_ONE_class_and_mints_the_rest(
    tmp_path: Path, fake_sm: Dict[str, str], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The whole issue. Before this, the mint died here and the class that HAD
    exported was discarded with it."""
    _declare()
    seen = _fail_one(
        monkeypatch, on="B=1",
        exc=Unsupported("Skip inlining `torch.compiler.disable()`d function\n"
                        "  Explanation: <function ModuleGroup.onload_>"))

    result = _mint(tmp_path)

    assert len(seen) == 2, "both classes must be ATTEMPTED — one refusing must not stop the loop"
    assert result.timings.get("skipped_entries") == 1.0
    entries = result.metadata.get("entries") or {}
    assert len(entries) == 1, "the surviving class must still be packed"
    assert not any("B=1" in name for name in entries), (
        "the class that refused must NOT be in the cell — fail-closed is per "
        "ENTRY, not abandoned")


def test_the_skipped_class_is_NAMED_with_the_construct_that_refused(
    tmp_path: Path, fake_sm: Dict[str, str], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A skipped class is only actionable if the reason names the thing someone
    has to change."""
    _declare()
    _fail_one(monkeypatch, on="B=1", exc=Unsupported(
        "Skip inlining `torch.compiler.disable()`d function\n"
        "  Explanation: Skip inlining function <function ModuleGroup.onload_> "
        "since it was wrapped with `torch.compiler.disable`\n"
        "  Developer debug context: <function ModuleGroup.onload_>"))

    events: List[Any] = []
    monkeypatch.setattr(
        aot_mint.activity_mod, "emit_event",
        lambda kind, detail, **kw: events.append((kind, detail, kw)))

    _mint(tmp_path)

    skips = [e for e in events if e[0] == aot_mint.KIND_ENTRY_EXPORT_UNSUPPORTED]
    assert len(skips) == 1, "the hub must be told, once, which class was skipped"
    detail = skips[0][1]
    assert "ModuleGroup.onload_" in detail, (
        "the event must name the CONSTRUCT that refused, not just the entry")
    assert "B=1" in detail


def test_a_RESOURCE_shortfall_still_aborts_the_WHOLE_mint(
    tmp_path: Path, fake_sm: Dict[str, str], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The distinction the whole change rests on.

    An OOM says nothing about the graph class — it says the pod is out of room,
    and the mint must abort so the parent can retry narrower. Skipping here
    would publish a cell whose missing classes are an artifact of memory
    pressure and would have exported fine on the retry.
    """
    _declare()
    boom = aot_mint.MintResourceExhausted("host memory exhausted mid-export")
    _fail_one(monkeypatch, on="B=1", exc=boom)

    with pytest.raises(Exception) as caught:
        _mint(tmp_path)
    assert caught.value is boom or "exhaust" in str(caught.value).lower()


def test_every_class_skipped_still_REFUSES(
    tmp_path: Path, fake_sm: Dict[str, str], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cell with no entries is not a partial cell, it is not a cell. It fails
    closed on the path it already failed closed on (`_pack`), rather than
    through a second check."""
    _declare()
    _fail_one(monkeypatch, on="B=", exc=Unsupported("nope"))

    with pytest.raises(aot_mint.MintRefused, match="no entries"):
        _mint(tmp_path)


# ---------------------------------------------------------------------------
# The classifier, directly
# ---------------------------------------------------------------------------


def test_only_deterministic_local_failures_are_skippable() -> None:
    assert aot_mint._export_skippable(Unsupported("whatever")), (
        "torch's own export refusal is the skippable case")

    # ...and ONLY torch's own. `_export_entry` also warms, gates and compiles;
    # skipping any of those would be this issue's own defect in a new hat.
    assert not aot_mint._export_skippable(RuntimeError("boom in forward")), (
        "a broken forward is not an unsupported construct")
    assert not aot_mint._export_skippable(
        aot_mint.MintRefused("mint-warm: the declared warm forward failed")), (
        "pgw#758 made a warm failure a NAMED REFUSAL — a cell whose classes "
        "were never warm-proven must not publish")
    assert not aot_mint._export_skippable(
        aot_mint.MintRefused("folding fence: lifted weight absent")), (
        "a correctness gate that can be skipped is not a gate")

    assert not aot_mint._export_skippable(KeyboardInterrupt()), (
        "a shutdown is not a property of the graph")
    assert not aot_mint._export_skippable(SystemExit(1))
    assert not aot_mint._export_skippable(
        aot_mint.MintResourceExhausted("no room")), (
        "a resource shortfall must abort so the parent can retry narrower")

    marked = RuntimeError("host memory")
    setattr(marked, "mint_resource_shortfall", True)
    assert not aot_mint._export_skippable(marked), (
        "the duck-typed shortfall marker must be honoured")


def test_a_cuda_oom_is_never_skippable() -> None:
    oom = torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 50.00 MiB")
    assert not aot_mint._export_skippable(oom)


def test_the_construct_namer_lifts_dynamos_own_explanation() -> None:
    exc = Unsupported(
        "Skip inlining `torch.compiler.disable()`d function\n"
        "  Explanation: Skip inlining function <function ModuleGroup.onload_> "
        "since it was wrapped with `torch.compiler.disable` (reason: None)\n"
        "  Hint: Remove the `torch.compiler.disable` call\n"
        "  Developer debug context: <function ModuleGroup.onload_>\n"
        "  " + "traceback noise " * 200)

    named = aot_mint._unsupported_construct(exc)

    assert "ModuleGroup.onload_" in named
    assert "Unsupported" in named
    assert len(named) <= 600, "this rides an event; it may not carry a traceback"
    assert "traceback noise traceback noise" not in named


# ---------------------------------------------------------------------------
# The MECHANISM, reproduced locally: why the A4000 could not export at all
# ---------------------------------------------------------------------------


def test_a_compiler_disabled_call_in_the_traced_path_kills_strict_export() -> None:
    """The pod's failure, in five lines, on CPU.

    This is what diffusers' group-offload hook does to a traced module — its
    ``ModuleGroup.onload_`` carries exactly this decorator. Reproduced so the
    fix's premise is a measured property of torch rather than a reading of a
    stack trace.
    """
    @torch.compiler.disable
    def _onload(x: Any) -> Any:
        return x + 1

    class Offloaded(nn.Module):
        def forward(self, x: Any) -> Any:
            return _onload(x) * 2

    class Plain(nn.Module):
        def forward(self, x: Any) -> Any:
            return (x + 1) * 2

    args = (torch.randn(2, 4),)

    torch.export.export(Plain().eval(), args, strict=True)  # the control

    with pytest.raises(Exception) as caught:
        torch.export.export(Offloaded().eval(), args, strict=True)

    # Asserted by TYPE, because the wording moves: torch 2.13 says "Skip
    # CALLING" for a free function and "Skip INLINING" for a bound method —
    # the pod's UNet hit the second (`ModuleGroup.onload_` is a method) and
    # this reproduction hits the first. Same exception, same cause, and a
    # message-matching test would have gone red on that difference alone
    # while the mechanism was identical.
    assert type(caught.value).__name__ == "Unsupported", (
        f"expected dynamo's Unsupported, got {type(caught.value).__name__}")
    said = str(caught.value).lower()
    assert "torch.compiler.disable" in said and "skip" in said, (
        f"expected a skip-because-disabled refusal, got: {str(caught.value)[:200]}")


# ---------------------------------------------------------------------------
# (b) The mint child never runs the serving placement ladder on the
#     weight-free path — the pgw#1124 seam, one door over
# ---------------------------------------------------------------------------


class _Sentinel(Exception):
    """Stops the mint at the load, so the assertion is about the CALL."""


def _drive_to_load(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, structure_ok: bool,
) -> List[Dict[str, Any]]:
    """Run the REAL `mint_child.mint()` as far as its first load and record
    every `run_setup` call it makes."""
    from gen_worker import mint_child
    from gen_worker import mint_process as mp
    from gen_worker.api.binding import ModelRef
    from gen_worker.cli import run as cli_run
    from gen_worker.models.structure_only import StructureOnlyUnsupported

    calls: List[Dict[str, Any]] = []

    def _spy(instance: Any, resolved: Any, **kwargs: Any) -> Any:
        calls.append(dict(kwargs))
        if kwargs.get("structure_only") and not structure_ok:
            # The family is stranded — mint_child falls back to real weights,
            # which is the second call this test wants to see.
            raise StructureOnlyUnsupported(
                component="unet", cls_name="X", lacks="no config surface")
        raise _Sentinel()

    monkeypatch.setattr(cli_run, "run_setup", _spy)

    tree = tmp_path / "tree"
    tree.mkdir()
    request = mp.MintRequest(
        function="composed-echo", modules=("harness.toy_endpoints",),
        family="sdxl", arm_token="arm1-abc",
        target=str(tmp_path / "cell.tar.gz"),
        work_root=str(tmp_path / "capture"),
        report=str(tmp_path / mp.REPORT_NAME),
        cfg=mp.CompileCellSpec(family="sdxl", shapes=((1024, 1024),),
                               targets=("unet",)),
        handler_proof="test: the parent proved it",
        slots={"pipeline": mp.MintSlot(
            ref=ModelRef(source="tensorhub", path="harness/composed",
                         tag="prod"),
            path=str(tree))},
    )
    with pytest.raises(BaseException):
        mint_child.mint(request)
    return calls


def test_the_weight_free_load_asks_for_NO_serving_placement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """THE sdxl A4000 fix.

    The ladder engaged diffusers group offloading on a card that already held
    a serving parent, and offload hooks put a `@torch.compiler.disable`d
    function in the traced path. This child exports and never runs a forward,
    so it has no business on the ladder at all — pgw#1124 made exactly this
    argument for the boot-trace child.
    """
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parent))
    calls = _drive_to_load(monkeypatch, tmp_path, structure_ok=True)

    assert calls, "the child never reached its load"
    weight_free = [c for c in calls if c.get("structure_only")]
    assert weight_free, "no structure-only load was attempted"
    for call in weight_free:
        assert call.get("place") is False, (
            "the weight-free mint asked for serving placement — that is the "
            "ladder that engaged group offload and made the graph "
            "unexportable")


def test_the_REAL_WEIGHT_fallback_keeps_its_placement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The scope of the fix, fenced. A stranded family loads its checkpoint
    into THIS process and runs the pgw#984 warm proof — a real forward, which
    needs a placed pipeline. Only the path that never executes may skip the
    ladder, and widening it would break the fallback silently."""
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parent))
    calls = _drive_to_load(monkeypatch, tmp_path, structure_ok=False)

    fallback = [c for c in calls if not c.get("structure_only")]
    assert fallback, "the fallback load never happened"
    for call in fallback:
        assert call.get("place", True) is not False, (
            "the real-weight fallback lost its placement — it runs a forward")


# ---------------------------------------------------------------------------
# (a) The pipeline that cannot be traced AS LOADED refuses ONCE, before export
# ---------------------------------------------------------------------------


def _offloaded() -> Any:
    """A pipeline carrying REAL diffusers group offloading. No mock: the whole
    point is that the marker is torch's and the hooks are diffusers'."""
    from diffusers.hooks.group_offloading import apply_group_offloading

    class Blocks(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.ModuleList([nn.Linear(8, 8) for _ in range(4)])

        def forward(self, x: Any) -> Any:
            for layer in self.layers:
                x = layer(x)
            return x

    unet = Blocks()
    apply_group_offloading(
        unet, onload_device=torch.device("cpu"),
        offload_device=torch.device("cpu"),
        offload_type="block_level", num_blocks_per_group=1)
    return types.SimpleNamespace(unet=unet)


def test_a_clean_pipeline_is_traceable() -> None:
    from gen_worker.models import traceability

    assert traceability.untraceable_hooks(
        types.SimpleNamespace(unet=TinyUNet())) == ()
    assert traceability.untraceable_reason(
        types.SimpleNamespace(unet=TinyUNet())) == ""


def test_real_group_offloading_is_detected_and_the_function_is_NAMED() -> None:
    """The pod's exact construct, found through the real diffusers path.

    `ModuleGroup.onload_` is `@torch.compiler.disable`d, and the hook that
    reaches it is registered in diffusers' own `_diffusers_hook` registry — not
    in torch's `_forward_pre_hooks` — so a walk that only knew torch's dicts
    would have seen nothing at all on the one case this exists for.
    """
    pytest.importorskip("diffusers")
    from gen_worker.models import traceability

    hits = traceability.untraceable_hooks(_offloaded())

    assert hits, "real group offloading was not detected"
    assert all(fn == "ModuleGroup.onload_" for _p, _w, fn in hits), (
        f"the disabled callable must be NAMED, got {sorted({f for _p,_w,f in hits})}")
    assert all("group_offloading" in where for _p, where, _f in hits)


def test_the_mint_refuses_ONCE_before_export_naming_the_construct() -> None:
    """(a). Without it, pgw#1208's per-entry skip would dutifully skip all 36
    classes and publish nothing — thirty-six typed refusals and an hour of wall
    clock to say once what was knowable before the first export began."""
    pytest.importorskip("diffusers")
    from gen_worker import mint_child
    from gen_worker import mint_process as mp

    request = mp.MintRequest(
        function="f", modules=(), family="sdxl", arm_token="a",
        target="t", work_root="w", report="r",
        cfg=mp.CompileCellSpec(family="sdxl", targets=("unet",)))

    # A traceable pipeline passes silently.
    mint_child.assert_traceable_as_loaded(
        types.SimpleNamespace(unet=TinyUNet()), request)

    with pytest.raises(mint_child.MintChildRefused) as caught:
        mint_child.assert_traceable_as_loaded(_offloaded(), request)

    said = str(caught.value)
    assert "mint_requires_resident_parent" in said, "the refusal must be TYPED"
    assert "ModuleGroup.onload_" in said, "and must name the construct"
    assert "serve this family eager" in said, (
        "and must say what this pod CAN still do — a refusal that does not is "
        "read as the pod being broken")


def test_the_check_is_on_the_REAL_WEIGHT_path_only() -> None:
    """Scope, fenced. The weight-free child never places (pgw#1208 fix 1), so
    it never acquires these hooks and must not pay a walk over every module of
    every component on the path that is already correct."""
    import inspect

    from gen_worker import mint_child

    source = inspect.getsource(mint_child)
    # The call sits after `pick_compile_target`, which both paths reach — but
    # the weight-free load cannot produce hooks, so this is a no-op there by
    # construction rather than by a second branch. What must NOT happen is the
    # check being skipped on the fallback.
    assert "assert_traceable_as_loaded(loaded_pipe, request)" in source
    assert source.index("assert_traceable_as_loaded(loaded_pipe") > \
        source.index("place=False"), (
        "the traceability check must run after the load, not before it")
