"""An untraceable loaded pipeline must refuse before TCG compilation.

The failure shape: the mint child ran the worker's SERVING placement ladder, the
ladder engaged diffusers group offloading, and the offload hooks put a
``@torch.compiler.disable``d function (``ModuleGroup.onload_``) in the traced
path, which ``torch.export(strict=True)`` refuses fatally.

Two surviving properties are guarded:

1. The mint child does not PLACE on the weight-free path (``place=False``). The
   ladder is for a pipeline that will run a forward; this one only ever exports,
   so the hooks are never installed rather than stripped afterwards.
2. A real-weight fallback carrying untraceable hooks refuses once, before any
   graph class is exported, and names the disabled construct.

TCG owns graph-class declaration, compilation, and refusal. This file retains
only the worker placement and preflight boundary that prevents the known
diffusers hook failure from reaching TCG at all.
"""

from __future__ import annotations

import types
from pathlib import Path
from typing import Any, Dict, List

import pytest

torch = pytest.importorskip("torch")

import torch.nn as nn  # noqa: E402

from gen_worker import child_preflight
from gen_worker import child_contract

pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")


class TinyUNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, sample: Any) -> Any:
        return torch.tanh(self.lin(sample)) + 1.0


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
        cfg=child_contract.CompileSpec(family="sdxl", shapes=((1024, 1024),),
                               targets=("unet",)),
        handler_proof="test: the parent proved it",
        slots={"pipeline": child_contract.MintSlot(
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
        cfg=child_contract.CompileSpec(family="sdxl", targets=("unet",)))

    # A traceable pipeline passes silently.
    mint_child.assert_traceable_as_loaded(
        types.SimpleNamespace(unet=TinyUNet()), request)

    with pytest.raises(child_preflight.PreflightRefused) as caught:
        mint_child.assert_traceable_as_loaded(_offloaded(), request)

    said = str(caught.value)
    assert "mint_pipeline_not_traceable" in said, "the refusal must be TYPED"
    assert "ModuleGroup.onload_" in said, "and must name the construct"
    assert "serves the family eager" in said, (
        "and must say what this pod CAN still do — a refusal that does not is "
        "read as the pod being broken")
    assert "resident" not in said.lower() and "fits" not in said.lower(), (
        "the refusal must NOT read as a capacity verdict: z-image refused on a "
        "48 GiB A40 holding a 19 GiB model, and §1.35 rules the card-filter "
        "concept out of existence — feasibility is never asked")


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
