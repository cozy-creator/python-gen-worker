"""pgw#1215 — a family with NO structure-only path still mints, K-wide.

The keystone moved the composition into the compile child, and a composition
that only ever asks for a weight-free target answers a question the old one
never asked: *may this family mint at all?* It may. ``mint_child._load`` has
preferred structure-only and fallen back to real weights since pgw#1123 —
a stranded family (a quantized artifact lane, a class the tree's
``model_index.json`` does not name) is a correct, more expensive mint, not a
refusal — and moving the trace one process down must not quietly turn that
into a capability regression on every such family.

**NO CAPABILITY REGRESSION is the invariant here**: a family that minted on
master mints on the K-wide path. It is worth a file because the regression is
invisible from the parent — the child refuses typed, terminally, and with a
perfectly good sentence, and the mint just never produces an artifact for a
family that used to have one.

The real-composition row deliberately stops at the composed target: the
compile itself is a POD leg (pgw#1215 step 3, and mints are remote-only), and
what is under test is which pipeline the child hands the trace, not what
inductor makes of it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

torch = pytest.importorskip("torch")

from gen_worker import aot_compile_child as child  # noqa: E402
from gen_worker import aot_compile_pool as pool  # noqa: E402
from gen_worker import aot_mint  # noqa: E402
from gen_worker.api.binding import ModelRef  # noqa: E402
from gen_worker.child_contract import CompileSpec, MintSlot  # noqa: E402
from gen_worker.child_preflight import PreflightRefused  # noqa: E402
from gen_worker.models import structure_only  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")

ENDPOINT_MODULE = "harness.tiny_diffusion_endpoint"
FUNCTION = "rig-generate"


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    from harness.tiny_diffusion import build_checkpoint

    return build_checkpoint(tmp_path_factory.mktemp("stranded") / "checkpoint")


def _job(checkpoint: Path, tmp_path: Path) -> pool.EntryJob:
    from harness import tiny_diffusion_endpoint as ep

    return pool.EntryJob(
        function=FUNCTION,
        modules=(ENDPOINT_MODULE,),
        cfg=CompileSpec(
            family=ep.FAMILY, targets=("unet",), shapes=(ep.PIXEL_SHAPE,)),
        slots={"pipeline": MintSlot(
            ref=ModelRef(source="tensorhub", path="rig/tiny-diffusion",
                         tag="prod"),
            path=str(checkpoint))},
        share="share-000", share_index=0, share_count=1,
        out_dir=str(tmp_path / "artifacts"), work=str(tmp_path / "work"),
        report=str(tmp_path / "report.json"))


# ---------------------------------------------------------------------------
# 1. The real composition, on the family that has no weight-free path
# ---------------------------------------------------------------------------


def test_a_family_with_no_structure_only_path_still_has_a_target_to_mint(
    checkpoint: Path, tmp_path: Path,
) -> None:
    """RED if the compile child REQUIRES structure-only.

    ``TinyUNet`` has no ``load_config``/``from_config`` surface and the tree's
    ``model_index.json`` does not name it, so the weight-free forge refuses it
    by name — the same refusal ``mint_child`` has fallen back from since
    pgw#1123. Sever the fallback and this reds with
    ``StructureOnlyUnsupported: … stranded on the real-weight mint``, which is
    exactly what the fleet would see: a family that minted last week and mints
    nothing this week.
    """
    pipeline, spec, decl = child.build_pipeline(_job(checkpoint, tmp_path))

    # The fallback really is the path that ran: a weight-free composition
    # stamps its facts on the pipeline, and this one carries none.
    assert not structure_only.facts_of(pipeline), (
        "this family has no structure-only path, so a composition that "
        "reports structure-only facts is not the one under test")
    params = list(pipeline.unet.parameters())
    assert params and not any(p.is_meta for p in params), (
        "the fallback composes REAL weights — a meta parameter here would "
        "export weight-scale fakes while the child reports weightless "
        "(ie#638)")

    # ...and there is something to compile: the share's own class enumeration
    # is what the trace loop iterates, so a non-empty row set is the whole
    # difference between "mints" and "refuses".
    rows = aot_mint.declared_class_rows(pipeline, spec, decl)
    assert rows, (
        f"{spec.family!r} composed but declares no graph class to this "
        f"child, so the share would pack nothing")


# ---------------------------------------------------------------------------
# 2. The shape of the fallback, and the one refusal that stays fatal
# ---------------------------------------------------------------------------


class _Recorder:
    """Stands in for ``cli.run.run_setup`` and records how it was asked."""

    def __init__(self, *, raises: List[Any]) -> None:
        self.calls: List[Dict[str, Any]] = []
        self._raises = list(raises)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append(dict(kwargs))
        exc = self._raises.pop(0) if self._raises else None
        if exc is not None:
            raise exc
        return {"pipeline": _Loaded()}


class _Unet:
    def forward(self, sample: Any) -> Any:  # pragma: no cover - never traced
        return sample


class _Loaded:
    def __init__(self) -> None:
        self.unet = _Unet()


def _unsupported() -> structure_only.StructureOnlyUnsupported:
    return structure_only.StructureOnlyUnsupported(
        component="unet", cls_name="TinyUNet",
        lacks="the tree does not name it")


def _not_honored() -> structure_only.StructureNotHonored:
    return structure_only.StructureNotHonored(
        component="unet", cls_name="TinyUNet",
        lacks="the pipeline rebuilt it from the checkpoint")


def _patch_run_setup(
    monkeypatch: pytest.MonkeyPatch, recorder: _Recorder,
) -> None:
    from gen_worker.cli import run as cli_run

    monkeypatch.setattr(cli_run, "run_setup", recorder)


def _build(job: pool.EntryJob) -> Tuple[Any, Any, Any]:
    return child.build_pipeline(job)


def test_the_fallback_asks_for_real_weights_and_still_never_places_them(
    checkpoint: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The second ask is the SAME load without the weight-free request.

    ``place=False`` survives the fallback, and that is not a copy-paste
    detail: this process never runs a forward, so the serving placement
    ladder would only move real non-target components onto the card the
    serving parent is resident on (pgw#1124) and install the offload hooks
    that make the target untraceable.
    """
    recorder = _Recorder(raises=[_unsupported(), None])
    _patch_run_setup(monkeypatch, recorder)
    monkeypatch.setattr(
        aot_mint, "export_declaration", lambda family: object())
    monkeypatch.setattr(
        "gen_worker.fleet_cells.aot_export_spec",
        lambda pipe, cfg: type("S", (), {"family": cfg.family})())

    _build(_job(checkpoint, tmp_path))

    assert len(recorder.calls) == 2, recorder.calls
    assert recorder.calls[0]["structure_only"] == ("unet",)
    assert "structure_only" not in recorder.calls[1], (
        "the fallback must ask for the REAL load, not the same weight-free "
        "one a second time")
    assert recorder.calls[1]["place"] is False


def test_a_weight_free_target_the_pipeline_discarded_still_fails_closed(
    checkpoint: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``StructureNotHonored`` is a SUBCLASS, so the order of the two handlers
    is the whole behaviour.

    It means the target WAS built weight-free and the composed pipeline threw
    it away — falling back there exports ~weight-scale REAL tensors while the
    child reports weightless, which is ie#638's silent 40 GiB OOM. One load,
    then a typed refusal.
    """
    recorder = _Recorder(raises=[_not_honored()])
    _patch_run_setup(monkeypatch, recorder)

    with pytest.raises(PreflightRefused) as raised:
        _build(_job(checkpoint, tmp_path))

    assert "did not carry it" in str(raised.value)
    assert len(recorder.calls) == 1, (
        "a pipeline that discarded the weight-free target must not be "
        "re-loaded with real weights")
