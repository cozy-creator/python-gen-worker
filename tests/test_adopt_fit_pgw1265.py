"""The adopt asks whether a SERVE still fits, and can give the arm back.

pgw#1265. Live four times on 2026-08-14 as
``ComputeProcessDied: cause=signal:SIGSEGV function=generate`` on z-image:
resident 32.01 GiB of 44.42 GiB (three functions, two full pipelines), the
adopt loads a runner and runs the §4.32 gate's two forwards beside it, and the
process dies with 17.9 MiB free. Under ``expandable_segments:True`` that
exhaustion is a native mapping abort no ``except`` sees, so
``aot_serve._bind_headroom``'s attempt-and-catch — the honest gate for a
CATCHABLE bind OOM — cannot be the whole answer.

Three invariants, one per section:

1. every device-touching span of the adopt is preceded by a verdict taken from
   live free memory against a peak MEASURED in this process;
2. a negative verdict ABANDONS — the entry de-arms, its runner is released, and
   the worker keeps serving eager;
3. an abandoned adopt cannot brick the worker: nothing is advertised, the flip
   never happens, and the class cannot be re-armed in this process.

These drive the REAL ``provision.arm_aot``, the REAL
``Executor._advertise_compiled_graphs``, the REAL ``adopt_fit`` arithmetic and
the REAL ``aot_serve.EntryDispatch``. The doubles are the CUDA device, the load
and the §4.32 gate — the GPU work this box may not do. Nothing is compiled or
minted here.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional

import sys

import pytest

from gen_worker import activity as activity_mod
from gen_worker import adopt_fit, mint_workers
from gen_worker import compile_cache
from gen_worker.cell_adopt import AdoptOutcome
from gen_worker.models import provision

_GIB = 1 << 30

_META: Dict[str, Any] = {
    "family": "z-image",
    "weight_lane": "lora128",
    "compiled_graph_key": "cg-key-v1-" + "0" * 56,
    "graph_class": {"name": "transformer/e0", "target": "transformer"},
}


# ---------------------------------------------------------------------------
# The card, and only the card, is a double.
# ---------------------------------------------------------------------------


class FakeCard:
    """A CUDA device that answers the three questions `adopt_fit` asks it.

    Everything above it — the free/reclaimable construction, the watermark
    subtraction, the verdict — is the production code under test.
    """

    def __init__(self, *, free: int, reserved: int = 0, allocated: int = 0):
        self.free = int(free)
        self.reserved = int(reserved)
        self.allocated = int(allocated)
        self.peak = int(allocated)
        self.is_available = lambda: True  # noqa: E731 — attribute, not method

    # -- the torch surface `adopt_fit` uses -------------------------------
    @property
    def cuda(self) -> "FakeCard":
        return self

    def current_device(self) -> int:
        return 0

    def mem_get_info(self, _index: int) -> tuple:
        return (self.free, 48 * _GIB)

    def memory_reserved(self, _index: int) -> int:
        return self.reserved

    def memory_allocated(self, _index: int) -> int:
        return self.allocated

    def max_memory_allocated(self, _index: int) -> int:
        return self.peak

    # -- what a forward or a load does to it ------------------------------
    def run_forward(self, transient: int) -> None:
        """One forward: allocates `transient`, frees it, moves the peak."""
        self.peak = max(self.peak, self.allocated + int(transient))

    def take(self, held: int) -> None:
        """Somebody (a load, a sibling instance) takes `held` bytes for good."""
        self.allocated += int(held)
        self.reserved += int(held)
        self.free -= int(held)
        self.peak = max(self.peak, self.allocated)


@pytest.fixture(autouse=True)
def _fresh_measurements() -> Iterator[None]:
    adopt_fit.reset()
    yield
    adopt_fit.reset()


@pytest.fixture
def card(monkeypatch: pytest.MonkeyPatch) -> FakeCard:
    """A card with 40 GiB free that has served one 12 GiB forward."""
    dev = FakeCard(free=40 * _GIB, reserved=2 * _GIB, allocated=1 * _GIB)
    monkeypatch.setattr(adopt_fit, "_torch", lambda: dev)
    # pgw#896: the free/reclaimable construction itself now lives in the ONE
    # home (`hostfacts.headroom_bytes`), which imports torch on its own — so
    # the fake card has to BE torch for the production formula to be the one
    # under test.
    monkeypatch.setitem(sys.modules, "torch", dev)
    with adopt_fit.forward_watermark(0):
        dev.run_forward(12 * _GIB)
    assert adopt_fit.forward_peak(0) == 12 * _GIB
    return dev


class _Trace:
    def __init__(self) -> None:
        self.load = 0
        self.gate = 0
        self.disarmed: List[tuple] = []
        self.flushed = 0
        self.events: List[tuple] = []


def _install(
    monkeypatch: pytest.MonkeyPatch,
    *,
    on_load: Optional[Callable[[], None]] = None,
    on_gate: Optional[Callable[[], None]] = None,
    armed: bool = True,
    gate_passes: bool = True,
) -> _Trace:
    """Real arm_aot; the load and the §4.32 gate are doubled."""
    from gen_worker import aot_serve

    trace = _Trace()
    monkeypatch.setattr(mint_workers, "adopt_watermark", lambda _d=None: (0, 0))
    monkeypatch.setattr(mint_workers, "device_of", lambda _p: 0)

    def _enable(*_a: Any, **_k: Any) -> AdoptOutcome:
        trace.load += 1
        if on_load is not None:
            on_load()
        return (AdoptOutcome.hit("armed") if armed
                else AdoptOutcome.miss("load_failed", "no"))

    def _gate(*_a: Any, **_k: Any) -> bool:
        trace.gate += 1
        if on_gate is not None:
            on_gate()
        return gate_passes

    monkeypatch.setattr(aot_serve, "enable", _enable)
    monkeypatch.setattr(aot_serve, "armed_metadata", lambda _p: dict(_META))
    def _disarm(_p: Any, name: str, reason: str) -> bool:
        trace.disarmed.append((name, reason))
        return True

    monkeypatch.setattr(aot_serve, "disarm_entry", _disarm)
    monkeypatch.setattr(aot_serve, "entry_states", lambda _p: {})
    monkeypatch.setattr(provision, "gate_cell_numerics", _gate)
    monkeypatch.setattr(
        provision, "flush_memory",
        lambda: trace.__setattr__("flushed", trace.flushed + 1))
    monkeypatch.setattr(
        activity_mod, "emit_event",
        lambda kind, detail, **kw: trace.events.append((kind, kw.get("phase", ""))))
    return trace


def _arm(tmp_path: Path, **kw: Any) -> AdoptOutcome:
    return provision.arm_aot(
        object(), type("Cfg", (), {"family": "z-image"})(), None,
        tmp_path / "cell.tar.gz", 0, dict(_META), **kw)


# ===========================================================================
# INVARIANT 1 — the verdict is taken BEFORE the span that would spend the card.
# ===========================================================================


def test_the_LOAD_is_refused_when_a_measured_forward_no_longer_fits(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, card: FakeCard,
) -> None:
    """A sibling instance took the card down to 1 GiB of usable memory. The
    adopt must not even start: `aot_serve.enable` is never called."""
    card.take(39 * _GIB)
    trace = _install(monkeypatch)

    outcome = _arm(tmp_path, verify_numerics=True)

    assert trace.load == 0, (
        "the adopt loaded a runner onto a card that cannot fit a forward — "
        "this is the allocation that raised the residency floor and killed "
        "the process")
    assert not outcome.armed
    assert outcome.reason == adopt_fit.REASON
    # The refusal states BOTH measured terms: 1 GiB free + 1 GiB reclaimable
    # against the 12 GiB forward this process actually ran.
    assert "2.000 GiB of usable" in outcome.detail, outcome.detail
    assert "12.000 GiB transient" in outcome.detail, outcome.detail


def test_the_VERIFY_FORWARDS_are_refused_when_the_LOAD_ate_the_headroom(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, card: FakeCard,
) -> None:
    """The exact production shape: the card had room for the load, and the
    §4.32 gate's two forwards — `probe_cell` -> `gate_cell_numerics`, the frame
    the z-image pod actually died in — no longer fit beside the runner."""
    trace = _install(monkeypatch, on_load=lambda: card.take(35 * _GIB))

    outcome = _arm(tmp_path, verify_numerics=True)

    assert trace.load == 1, "the load itself was affordable and must have run"
    assert trace.gate == 0, (
        "the §4.32 gate ran two forwards on a card with no room for one — the "
        "adopt's peak device moment, unchecked")
    assert outcome.reason == adopt_fit.REASON


def test_an_ADOPTER_that_runs_no_gate_is_still_checked_before_it_serves(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, card: FakeCard,
) -> None:
    """`verify_numerics=False` is every adopting pod. It runs no gate, so the
    verdict that protects it is the one about the state the arm LEAVES."""
    trace = _install(monkeypatch, on_load=lambda: card.take(35 * _GIB))

    outcome = _arm(tmp_path, verify_numerics=False)

    assert trace.gate == 0
    assert outcome.reason == adopt_fit.REASON
    assert trace.disarmed, "the loaded runner was left resident"


def test_the_state_the_ARM_LEAVES_is_checked_after_the_gate_has_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, card: FakeCard,
) -> None:
    """The card had room for the load AND for the §4.32 gate, and by the time
    the gate finished it no longer had room to SERVE — a sibling instance
    rotated, or the gate's own forwards left the allocator holding blocks this
    process cannot hand back. The caller is about to advertise this arm and
    serve through it, so the last verdict is about the state the arm actually
    leaves behind."""
    trace = _install(monkeypatch, on_gate=lambda: card.take(38 * _GIB))

    outcome = _arm(tmp_path, verify_numerics=True)

    assert (trace.load, trace.gate) == (1, 1), (
        "both spans were affordable when they were asked about")
    assert outcome.reason == adopt_fit.REASON, (
        "an arm was returned ARMED on a card that can no longer fit a forward "
        "— the caller advertises it and the next request is the one that dies")
    assert trace.disarmed == [("transformer/e0", adopt_fit.REASON)]


def test_a_card_that_fits_adopts_exactly_as_before(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, card: FakeCard,
) -> None:
    """The gate is not allowed to cost a pod that has room."""
    trace = _install(monkeypatch, on_load=lambda: card.take(2 * _GIB))

    outcome = _arm(tmp_path, verify_numerics=True)

    assert outcome.armed
    assert (trace.load, trace.gate) == (1, 1)
    assert trace.disarmed == []


# --- the two terms of the verdict, severed one at a time -------------------


def test_NOTHING_MEASURED_fits_by_construction(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """No forward has run in this process, so there is no evidence to refuse
    on. An honest under-refusal beats a number invented to look like a bound
    (pgw#1175's lesson, not re-learned)."""
    dev = FakeCard(free=1 << 20, reserved=0, allocated=44 * _GIB)
    monkeypatch.setattr(adopt_fit, "_torch", lambda: dev)
    trace = _install(monkeypatch)

    assert adopt_fit.forward_peak(0) == 0
    assert _arm(tmp_path, verify_numerics=True).armed
    assert trace.load == 1


def test_an_UNPROBEABLE_device_fits_by_construction(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.setattr(adopt_fit, "_torch", lambda: None)
    trace = _install(monkeypatch)

    assert adopt_fit.refusal("x", 0) == ""
    assert _arm(tmp_path, verify_numerics=True).armed
    assert trace.load == 1


def test_RECLAIMABLE_CACHE_counts_as_headroom(card: FakeCard) -> None:
    """Driver-free alone would refuse a card holding a large idle allocator
    cache, which is memory this process can hand back on demand."""
    card.free = 4 * _GIB
    card.reserved = card.allocated + 9 * _GIB

    assert adopt_fit.headroom(0) == 13 * _GIB
    assert adopt_fit.refusal("x", 0) == "", "13 GiB usable, 12 GiB needed"


def test_there_is_no_MARGIN_no_FACTOR_and_no_FLOOR(card: FakeCard) -> None:
    """Paul's standing rule. The verdict is `have >= need` on two measured
    quantities: exactly enough is enough, and one byte short refuses."""
    card.free = 12 * _GIB
    card.reserved = card.allocated
    assert adopt_fit.headroom(0) == 12 * _GIB
    assert adopt_fit.refusal("x", 0) == ""

    card.free -= 1
    assert adopt_fit.refusal("x", 0) != ""


def test_the_verdict_reads_no_BANK() -> None:
    """pgw#1205's device-peak bank stays measurement-only, and pgw#1164's
    deleted estimate is not rebuilt: this module's whole input surface is the
    live device plus this process's own forwards."""
    source = Path(adopt_fit.__file__).read_text()
    for forbidden in (
        "device_peak", "mint_workers", "fleet_cells", "compile_cache",
        "vram_gb", "resources_vram",
    ):
        assert forbidden not in source.split('"""')[2], (
            f"adopt_fit reads {forbidden!r} — the verdict must come from "
            f"measurement at the decision point, never a bank or a declaration")


def test_the_watermark_never_resets_the_shared_peak_counter(
    card: FakeCard,
) -> None:
    """`max_memory_allocated` is process-monotone and other readers share it
    (pgw#652's activation learning). A watermark that reset it would zero a
    measurement the admission ladder is built on."""
    assert not hasattr(card, "reset_peak_memory_stats_called")
    assert "reset_peak_memory_stats" not in Path(adopt_fit.__file__).read_text()


# ===========================================================================
# INVARIANT 2 — a negative verdict ABANDONS: de-arm, release, keep serving.
# ===========================================================================


def test_a_refused_adopt_DE_ARMS_the_entry_it_loaded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, card: FakeCard,
) -> None:
    trace = _install(monkeypatch, on_load=lambda: card.take(35 * _GIB))

    _arm(tmp_path, verify_numerics=True)

    assert trace.disarmed == [("transformer/e0", adopt_fit.REASON)], (
        "the runner stayed armed after the adopt was refused — the residency "
        "floor it raised is permanent and every later request pays it")


def test_a_refused_adopt_RELEASES_the_bytes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, card: FakeCard,
) -> None:
    """De-arming drops the reference; `flush_memory` is what turns a dropped
    reference into free bytes. Both terms are load-bearing."""
    trace = _install(monkeypatch, on_load=lambda: card.take(35 * _GIB))

    _arm(tmp_path, verify_numerics=True)

    assert trace.flushed == 1, (
        "the dropped runner was never collected, so the floor did not come "
        "back down")


def test_a_refused_adopt_RETURNS_and_the_worker_keeps_serving(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, card: FakeCard,
) -> None:
    """The refusal is a returned verdict, never a raise: `adopt_delegated_mint`
    is documented to return None when nothing adopted, and an escaping
    exception there destroys the typed refusal and the quarantine."""
    _install(monkeypatch, on_load=lambda: card.take(35 * _GIB))

    outcome = _arm(tmp_path, verify_numerics=True)

    assert outcome.armed is False
    assert outcome.reason == adopt_fit.REASON


def test_the_refusal_is_a_CAPACITY_verdict_not_a_contract_one(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, card: FakeCard,
) -> None:
    """A contract verdict quarantines a key (th#1819). A full card says
    nothing about the artifact — another pod adopts it fine."""
    trace = _install(monkeypatch, on_load=lambda: card.take(35 * _GIB))

    outcome = _arm(tmp_path, verify_numerics=True)

    assert ("adopt_headroom_refused", adopt_fit.REASON) in trace.events
    assert "another pod" in outcome.detail
    for reason in ("numerics_refused", "artifact_failed", "contract_invalid"):
        assert reason not in outcome.reason


# ===========================================================================
# INVARIANT 3 — an abandoned adopt cannot brick the worker.
# ===========================================================================


def test_a_refused_class_cannot_be_RE_ARMED_in_this_process() -> None:
    """§4.31's de-arm is sticky, which is what stops a refused adopt becoming
    a retry loop that pays the load again on a card that is already full.
    This drives the REAL dispatch registry."""
    from gen_worker import aot_serve

    dispatch = aot_serve.EntryDispatch()
    runner = type("Runner", (), {"calls": 0})()
    dispatch.add("transformer/e0", runner)  # type: ignore[arg-type]
    assert dispatch.remove("transformer/e0", adopt_fit.REASON)

    with pytest.raises(compile_cache.AdoptError) as caught:
        dispatch.add("transformer/e0", runner)  # type: ignore[arg-type]
    assert caught.value.reason == "entry_de_armed"


# -- the executor's advertisement -------------------------------------------


class _FakeTarget:
    def __init__(self, pipeline: Any) -> None:
        self.pipeline = pipeline
        self.state_lock = threading.Lock()
        self.active_compile_ref = ""
        self.active_compile_snapshot_digest = ""
        self.active_self_mint = False
        self.incarnation_id = "inc-1"


def _advertise(
    monkeypatch: pytest.MonkeyPatch, *, guard_binds: bool = True,
    probe: Optional[Callable[[], None]] = None,
) -> tuple:
    """Drive the REAL `Executor._advertise_compiled_graphs`."""
    from gen_worker import aot_serve, executor as executor_mod
    from gen_worker import hot_swap

    pipe = object()
    target = _FakeTarget(pipe)
    rec = type("Rec", (), {
        "compile_targets": {"t": target}, "eager_posture": "declined"})()
    bg = type("Bg", (), {
        "pipes": {1: pipe}, "spec": type("Spec", (), {"name": "generate"})()})()
    act = type("Act", (), {"phase": staticmethod(lambda _p: None)})()
    outcome = type("Outcome", (), {"ref": "cg-ref", "snapshot_digest": "d"})()

    enabled: List[Any] = []
    disarmed: List[tuple] = []
    events: List[tuple] = []
    monkeypatch.setattr(mint_workers, "device_of", lambda _p: 0)
    def _enable(p: Any, *_a: Any, **_k: Any) -> bool:
        enabled.append(p)
        return True

    def _disarm(_p: Any, name: str, reason: str) -> bool:
        disarmed.append((name, reason))
        return True

    monkeypatch.setattr(hot_swap, "enable", _enable)
    monkeypatch.setattr(aot_serve, "entry_states", lambda _p: {"e0": {}})
    monkeypatch.setattr(aot_serve, "disarm_entry", _disarm)
    monkeypatch.setattr(executor_mod, "flush_memory", lambda: None)
    monkeypatch.setattr(
        activity_mod, "emit_event",
        lambda kind, detail, **kw: events.append((kind, kw.get("phase", ""))))
    monkeypatch.setattr(
        executor_mod.Executor, "_refresh_compile_target",
        lambda self, t: probe() if probe is not None else None)
    monkeypatch.setattr(
        executor_mod.Executor, "_bind_compile_guard",
        lambda self, r, t: guard_binds)

    ex = object.__new__(executor_mod.Executor)
    executor_mod.Executor._advertise_compiled_graphs(
        ex, rec, bg, act, {1: outcome})
    return target, enabled, disarmed, events


def test_the_ADVERTISEMENT_is_refused_before_active_compile_ref_flips(
    monkeypatch: pytest.MonkeyPatch, card: FakeCard,
) -> None:
    """The named site. Between the arm and here sit the publish and whatever a
    sibling did to the card, so the flip takes the verdict again — and a claim
    that this pipe SERVES compiled must not be made on a card that cannot fit
    a forward."""
    card.take(39 * _GIB)

    target, enabled, disarmed, events = _advertise(monkeypatch)

    assert target.active_compile_ref == "", (
        "the target advertises a compiled ref on a card with no room to serve "
        "one")
    assert target.active_self_mint is False
    assert enabled == [], "eager-while-compiling was turned on anyway"
    assert disarmed == [("e0", adopt_fit.REASON)]
    assert ("adopt_headroom_refused", adopt_fit.REASON) in events


def test_a_fitting_card_advertises_and_enables_exactly_as_before(
    monkeypatch: pytest.MonkeyPatch, card: FakeCard,
) -> None:
    target, enabled, disarmed, _events = _advertise(monkeypatch)

    assert target.active_compile_ref == "cg-ref"
    assert target.active_self_mint is True
    assert len(enabled) == 1
    assert disarmed == []


def test_hot_swap_is_NOT_enabled_when_no_target_kept_its_ref(
    monkeypatch: pytest.MonkeyPatch, card: FakeCard,
) -> None:
    """Every target rolled back for want of a guard revocation signal, and the
    pipe was still switched to concurrent routing. Unconditional was wrong."""
    target, enabled, _disarmed, _events = _advertise(
        monkeypatch, guard_binds=False)

    assert target.active_compile_ref == ""
    assert enabled == [], (
        "concurrent routing was enabled on a pipe advertising nothing")


def test_the_advertisement_span_is_named_as_a_COMPILE(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, card: FakeCard,
) -> None:
    """pgw#1262's marker, one question over: a signal death at the flip must be
    charged to the compile, so pgw#714's eager-only reboot fires instead of the
    pod re-minting the same key and crash-looping while the hub reads the
    crashes as demand (th#1959)."""
    from gen_worker import postmortem

    monkeypatch.setattr(postmortem, "INFLIGHT_PATH", tmp_path / "inflight.json")
    postmortem.clear_inflight()
    seen: List[List[Dict[str, Any]]] = []

    def _look() -> None:
        seen.append([
            row for row in postmortem.take_inflight(postmortem.INFLIGHT_PATH)
            if row.get("kind") == postmortem.COMPILE_KIND])

    _advertise(monkeypatch, probe=_look)

    assert seen and seen[0], (
        "no compile marker was in flight while the advertisement touched the "
        "live compiled objects")
    assert seen[0][0]["function"] == "compile:advertise:generate"
