"""pgw#756: the guard-closure classifier lost its veto.

Paul's ruling. The classifier protects against WASTED REUSE, not
incorrectness — dynamo re-evaluates the same guards at the consumer, where
pgw#680 fail-on-recompile turns a real leak into an eager serve plus a named
``guard_miss``. A classifier BUG, by contrast, refuses every mint on every
family, which happened twice fleet-wide (pgw#691, pgw#733) while the gate
caught zero real leaks. So classification is now advisory.

Real-dynamo tapes (CPU, ``backend="eager"``): the guard machinery is
production's; only inductor codegen is skipped.

RED-VERIFY, both directions:
  * a mint that USED to refuse on a classification leak now completes, packs,
    and carries the leak in its manifest, with a typed ``guard_leak`` event;
  * a genuinely broken mint (nothing compiled) and a real ingress boundary
    violation still fail exactly as before.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Tuple

import pytest

torch = pytest.importorskip("torch")

from gen_worker import activity as activity_mod
from gen_worker import compile_cache as cc
from gen_worker import guard_closure as gc
from gen_worker.registry import CompileCell


@pytest.fixture(autouse=True)
def _fresh_dynamo() -> Iterator[None]:
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


@pytest.fixture
def events(monkeypatch: pytest.MonkeyPatch) -> List[Any]:
    """Capture real ``ActivityUpdate`` protos off the live emission path."""
    captured: List[Any] = []
    monkeypatch.setattr(activity_mod, "_sink", captured.append)
    return captured


def _cfg(**overrides: Any) -> CompileCell:
    base: Dict[str, Any] = dict(
        shapes=((64, 64),), targets=("transformer",), family="toyfam",
        regional=False, text_len=None, dynamic=(), lora_bucket=0,
        guidance_scales=(), text_lens=(),
    )
    base.update(overrides)
    return CompileCell(**base)


def _armed_pipe(forward: Callable[..., Any]) -> Tuple[Any, Callable[..., Any]]:
    class _Mod(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = torch.nn.Linear(8, 8)

    _Mod.forward = forward  # type: ignore[method-assign]

    class _Pipe:
        pass

    pipe = _Pipe()
    mod = _Mod()
    pipe.transformer = mod  # type: ignore[attr-defined]
    setattr(pipe, cc._MARKER_ATTR, {
        "targets": ["transformer"], "shapes": [(64, 64)], "cache": True,
        "originals": [(mod, "forward", mod.forward)], "regional_mods": [],
        "failure_signal": {
            "callback": None, "lock": threading.Lock(),
            "successful_calls": 0, "cache_hits": 0, "cache_misses": 0,
            "router": None,
        },
    })
    return pipe, mod.forward


def _leaking_pipe() -> Any:
    """An armed, really-compiled pipe whose guards pin an UNDECLARED scalar
    (3.25 is in no contract axis) — the exact shape that refused the mint
    before pgw#756."""

    def forward(self: Any, x: Any, scale: float) -> Any:
        return self.lin(x) * scale

    pipe, fn = _armed_pipe(forward)
    torch.compile(fn, backend="eager", dynamic=None)(torch.randn(2, 4, 8), 3.25)
    return pipe


def _guard_leak_events(events: List[Any]) -> List[Any]:
    return [e for e in events if e.kind == activity_mod.KIND_GUARD_LEAK]


# ---------------------------------------------------------------------------
# GREEN: a classification leak no longer refuses
# ---------------------------------------------------------------------------


def test_classification_leak_completes_the_mint_naming_the_variable(
    events: List[Any],
) -> None:
    manifest = gc.closure_manifest(_leaking_pipe(), _cfg(), label="toyfam")

    leaks = "\n".join(manifest["leaks"])
    assert "L['scale']" in leaks and "3.25" in leaks, leaks
    assert manifest[gc.GATE_KEY] == gc.GATE_ADVISORY
    assert manifest["graphs"], "the manifest still carries the full dump"

    emitted = _guard_leak_events(events)
    assert len(emitted) == 1, [e.kind for e in events]
    assert emitted[0].phase == "leak"
    assert "L['scale']" in emitted[0].detail
    assert "toyfam" in emitted[0].detail


def test_leaking_mint_packs_and_publishes_the_leak(
    tmp_path: Path, events: List[Any],
) -> None:
    """End to end through the real finalize: the cell EXISTS, and the leak
    rides it into CAS. Before pgw#756 this raised and packed nothing."""
    pipe = _leaking_pipe()
    capture = tmp_path / "capture"
    fx_entry = capture / "inductor" / "fxgraph" / "aa" / "bb"
    fx_entry.mkdir(parents=True)
    (fx_entry / "entry").write_bytes(b"fx")
    target = tmp_path / "cell.tar.gz"

    meta = cc.finish_fleet_mint(pipe, _cfg(), "toyfam", target, capture)

    assert target.exists(), "an undeclared scalar must not refuse the mint"
    packed = gc.load_manifest(target)
    assert packed == meta[gc.MANIFEST_KEY]
    assert any("L['scale']" in row for row in packed["leaks"])
    assert _guard_leak_events(events), "the leak must be countable hub-side"


def test_unreadable_guard_entry_is_unproven_not_fatal(
    monkeypatch: pytest.MonkeyPatch, events: List[Any],
) -> None:
    """A torch build whose guard debug surface we cannot walk is a fact
    about OUR diagnostics, not about the mint: recorded, emitted, survived."""
    pipe = _leaking_pipe()

    def _boom(entry: Any) -> List[Tuple[str, str, str]]:
        raise gc.GuardClosureError("guard_manager surface changed")

    monkeypatch.setattr(gc, "_entry_guard_rows", _boom)
    manifest = gc.closure_manifest(pipe, _cfg(), label="toyfam")

    assert manifest["unproven"], manifest
    assert "guard_manager surface changed" in manifest["unproven"][0]
    assert manifest["leaks"] == [], "an unreadable entry is not a leak claim"
    emitted = _guard_leak_events(events)
    assert emitted and emitted[0].phase == "unproven"


def test_clean_mint_stays_silent(events: List[Any]) -> None:
    def forward(self: Any, x: Any) -> Any:
        return self.lin(x) + 1

    pipe, fn = _armed_pipe(forward)
    torch.compile(fn, backend="eager", dynamic=None)(torch.randn(2, 4, 8))

    manifest = gc.closure_manifest(pipe, _cfg(), label="toyfam")
    assert manifest["leaks"] == [] and manifest["unproven"] == []
    assert not _guard_leak_events(events), "no finding, no event"


# ---------------------------------------------------------------------------
# RED: refusals that PROVE a defect survive the demotion
# ---------------------------------------------------------------------------


def test_nothing_compiled_still_refuses() -> None:
    """Not a classification verdict: the mint compiled nothing, so the cell
    would be empty. Still red."""

    class _Mod:
        def forward(self, x: Any) -> Any:  # pragma: no cover - never called
            return x

    class _Pipe:
        pass

    pipe = _Pipe()
    mod = _Mod()
    pipe.transformer = mod  # type: ignore[attr-defined]
    setattr(pipe, cc._MARKER_ATTR, {
        "targets": ["transformer"], "shapes": [(64, 64)], "cache": True,
        "originals": [(mod, "forward", mod.forward)], "regional_mods": [],
        "failure_signal": {
            "callback": None, "lock": threading.Lock(),
            "successful_calls": 0, "cache_hits": 0, "cache_misses": 0,
            "router": None,
        },
    })
    with pytest.raises(gc.GuardClosureError, match="nothing was compiled"):
        gc.closure_manifest(pipe, _cfg(), label="toyfam")


def test_ingress_dtype_drift_still_raises() -> None:
    """ie#544's boundary is a REAL assertion on REAL values — a different
    mechanism from classification, untouched by the demotion."""

    def f(x: Any) -> Any:
        return x * 2

    wrapped = gc.canonical_ingress(
        torch.compile(f, backend="eager", dynamic=None), "transformer")
    wrapped(torch.randn(4, 8))
    with pytest.raises(gc.GuardBoundaryError, match="dtype drift"):
        wrapped(torch.randn(4, 8).to(torch.float64))


def test_non_canonical_posture_still_refuses() -> None:
    """pgw#695 measures live torch state against exact canonical values —
    a measurement, not an inference. Still red."""

    def forward(self: Any, x: Any) -> Any:
        return self.lin(x) + 1

    pipe, fn = _armed_pipe(forward)
    torch.compile(fn, backend="eager", dynamic=None)(torch.randn(2, 4, 8))

    with torch.no_grad():
        with pytest.raises(gc.PostureError, match="grad_enabled"):
            gc.closure_manifest(pipe, _cfg(), label="toyfam")
