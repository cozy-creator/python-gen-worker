"""pgw#1052 — export and compile were sequential by CODE, not by necessity.

Attempt 26/30 measured the two phases at 65-68 min (export, one core, serial
in the mint parent) and 92-97 min (compile, K=2 children) with ZERO overlap:
the pool was constructed only after the last row exported. Producer ~113 s/row
against a pool consuming ~127 s/row means the phases almost perfectly shadow
each other, so handing each row to the pool AS IT EXPORTS collapses the mint
wall toward max(export, compile).

What must hold, and is asserted here for real (real ``torch.export``, real
``aot_compile`` children, CPU):

* entries REACH the pool while the export source is still open — proven
  structurally (the producer refuses to yield its last row until the pool has
  completed an earlier one), never by wall clock;
* pgw#917's merge/refuse decision moves to ARRIVAL: a duplicate class row is
  aliased before any compile is spent on it, and a same-ingress
  different-identity collision refuses at row N (bounded waste, stated);
* the overlapped mint and the serial mint produce the SAME cell key — the
  overlap is a PROCESS change under pgw#846's rule, byte-invisible in the
  artifact;
* the phase books stay honest: ``export_all_s`` still exists, and the pool
  ledger charges producer time to its own named bucket (``idle_source_s``).
"""

from __future__ import annotations

import types
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

import pytest

torch = pytest.importorskip("torch")

import torch.nn as nn  # noqa: E402

from gen_worker import aot_compile_pool as pool_mod  # noqa: E402
from gen_worker import aot_flatten, aot_mint, aot_serve, compile_cache  # noqa: E402
from gen_worker.api.decorators import Compile  # noqa: E402
from gen_worker.api.export_contract import (  # noqa: E402
    Dim,
    GraphClass,
    Input,
    register_export_declaration,
    reset_export_declarations,
)
from harness.progress_wait import Cadence, await_progress  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")

_GIB = 1024 ** 3
_HIDDEN = 64


# ---------------------------------------------------------------------------
# The pool consumes while the producer is still producing
# ---------------------------------------------------------------------------


def _program(seed: int) -> Any:
    class Tiny(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.a = nn.Linear(_HIDDEN, _HIDDEN)

        def forward(self, x: Any) -> Any:
            return torch.tanh(self.a(x)) * (1.0 + seed)

    return torch.export.export(Tiny(), (torch.randn(4, _HIDDEN),))


def test_entries_reach_the_pool_as_they_export(tmp_path: Path) -> None:
    """The producer WAITS for the pool to finish an earlier entry before it
    yields its last one. If the pool only started consuming after the source
    was exhausted — the pre-pgw#1052 shape — this test would never complete;
    the progress predicate is the pool's own completion record."""
    width = pool_mod.entry_workers(
        3, limit=2, vcpus=16, available_bytes=64 * _GIB,
        device_lock=True)
    assert width.workers == 2
    box = pool_mod.EntryCompilePool(
        tmp_path / "pool", width=width,
        inductor_configs={"compile_threads": 2},
        cache_dir=str(tmp_path / "cache"))

    overlap_seen: List[int] = []

    def _reports_on_disk() -> int:
        # The parent is single-threaded and is INSIDE this producer while it
        # waits, so it cannot have collected anything yet — the honest
        # overlap evidence is the CHILD's own report file, written by a
        # process the parent is not driving.
        return len(list((tmp_path / "pool").glob("entry-*/report.json")))

    def _source() -> Iterator[Tuple[str, Any]]:
        yield "unet/dim=0", _program(0)
        yield "unet/dim=1", _program(1)
        # The whole point: an earlier entry COMPLETES (its child writes its
        # report) while this source is still open.
        await_progress(
            _reports_on_disk,
            lambda n: n >= 1,
            what="an entry child finishes while the source is open",
            cadence=Cadence(floor_s=300.0))
        overlap_seen.append(_reports_on_disk())
        yield "unet/dim=2", _program(2)

    out = box.compile(_source(), expected_total=3)

    assert set(out) == {"unet/dim=0", "unet/dim=1", "unet/dim=2"}
    assert overlap_seen and overlap_seen[0] >= 1, (
        "the pool never completed an entry while the export source was open "
        "— the phases are still sequential")
    assert box.ledger.entries == 3
    facts = box.ledger.facts()
    assert "idle_source_s" in facts, (
        "producer time must be charged to its own named bucket — an idle "
        "split that hides the source starves pgw#1000's fused-child case of "
        "its number")


def test_a_sequence_still_compiles_exactly_as_before(tmp_path: Path) -> None:
    """The pre-existing list-shaped callers (and the pgw#848 resume path) are
    untouched: a fully-exported list goes through the same loop."""
    width = pool_mod.entry_workers(
        2, limit=2, vcpus=16, available_bytes=64 * _GIB,
        device_lock=True)
    box = pool_mod.EntryCompilePool(
        tmp_path / "pool", width=width,
        inductor_configs={"compile_threads": 2},
        cache_dir=str(tmp_path / "cache"))
    out = box.compile([("unet/dim=1", _program(1)), ("unet/dim=0", _program(0))])
    assert list(out) == sorted(out)
    assert set(out) == {"unet/dim=0", "unet/dim=1"}


# ---------------------------------------------------------------------------
# alias before a compile is spent; refuse at row N
# ---------------------------------------------------------------------------

XATTN = 128
TEXT_LEN = 7
AREA_PRESERVING = ((4, 6), (6, 4), (8, 3))   # all products 24


class _Block(nn.Module):
    def forward(self, hidden_states: Any, encoder_hidden_states: Any) -> Any:
        return hidden_states.mean() + encoder_hidden_states.mean()


class _OtherBlock(nn.Module):
    def forward(self, hidden_states: Any, encoder_hidden_states: Any) -> Any:
        return hidden_states.sum() - encoder_hidden_states.sum()


def _entry(name: str, h: int, w: int, module: nn.Module | None = None) -> Any:
    args = (torch.zeros(2, h * w, _HIDDEN), torch.zeros(2, TEXT_LEN, XATTN))
    program = torch.export.export((module or _Block()).eval(), args, strict=True)
    return aot_mint._MintedEntry(
        name=name,
        spec=aot_mint.ExportSpec(
            family="sdxl", target="unet",
            fork=((aot_mint.ADAPTER_FORK, False),),
            class_dims=(("B", 2), ("H_lat", h), ("W_lat", w)),
        ),
        module=None, owner=None, program=program,
        input_names=("hidden_states", "encoder_hidden_states"),
        flat_leaves=tuple(
            aot_flatten.Leaf(param=n, param_position=i, path=())
            for i, n in enumerate(
                ("hidden_states", "encoder_hidden_states"))),
        files=[], timings={})


def test_arrival_alias_merges_without_a_compile() -> None:
    canon = aot_mint._ArrivalCanon()
    first = _entry("unet/r0", *AREA_PRESERVING[0])
    assert canon.admit(first) is None
    for i, (h, w) in enumerate(AREA_PRESERVING[1:], start=1):
        keeper = canon.admit(_entry(f"unet/r{i}", h, w))
        assert keeper is first, (
            "an area-preserving sibling must alias onto the FIRST arrival — "
            "compiling it buys nothing and makes the cell undispatchable")


def test_arrival_collision_refuses_naming_the_axis() -> None:
    canon = aot_mint._ArrivalCanon()
    assert canon.admit(_entry("unet/r0", 4, 6)) is None
    with pytest.raises(aot_mint.MintRefused) as err:
        canon.admit(_entry("unet/r1", 6, 4, module=_OtherBlock()))
    assert "graph" in str(err.value), (
        "the refusal must NAME the differing identity axis (pgw#917)")
    assert "arrival" in str(err.value).lower()


def test_arrival_and_late_alias_maps_merge_and_rehome() -> None:
    a, b, c = (_entry(f"unet/r{i}", 4, 6) for i in range(3))
    merged = aot_mint._merge_alias_maps(
        {"unet/r1": [c]},                 # arrival: c aliased onto keeper r1
        {"unet/r0": (b,)},                # drain: r1's OBJECT b merged onto r0
    )
    _ = a
    # r1 was itself merged away at drain, so its arrival alias re-homes.
    assert set(merged) == {"unet/r0", "unet/r1"} or set(merged) == {"unet/r0"}
    # the strict requirement: no declared row falls out of the audit trail
    named = {row.name for rows in merged.values() for row in rows}
    assert {"unet/r1", "unet/r2"} <= named | set(merged)


# ---------------------------------------------------------------------------
# The overlapped mint IS the serial mint, artifact-wise (pgw#846's gate)
# ---------------------------------------------------------------------------

FAMILY = "tiny1052"


class TinyUNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, sample: Any) -> Any:
        return torch.tanh(self.lin(sample)) + 1.0


def _declare() -> Any:
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
def _fresh_registry():
    reset_export_declarations()
    yield
    reset_export_declarations()


@pytest.fixture
def fake_sm(monkeypatch):
    full = {"sku": "", "sm": "sm_89", "torch": str(torch.__version__),
            "cuda": ""}
    monkeypatch.setattr(compile_cache, "runtime_key", lambda: dict(full))
    monkeypatch.setattr(aot_serve, "runtime_key", lambda: {
        "sku": full["sku"], "sm": full["sm"], "torch": full["torch"],
        "cuda": full["cuda"]})
    return full


@pytest.fixture
def wide_pool(monkeypatch):
    """The width is STATED, not derived (the pgw#809 discipline): a 4-vCPU CI
    runner honestly derives K=1, and every pool-path assertion below would
    then pass while exercising no pool at all. The REAL policy still runs —
    only its resource inputs are pinned to a box wide enough to pool."""
    real = pool_mod.entry_workers

    def _wide(entries: int, **kw: Any) -> Any:
        kw.update(vcpus=16, available_bytes=64 * _GIB, device_lock=True)
        return real(entries, **kw)

    monkeypatch.setattr(pool_mod, "entry_workers", _wide)


def _mint(tmp: Path, *, entry_workers: int = 0) -> aot_mint.MintResult:
    pipe = types.SimpleNamespace(unet=TinyUNet())
    spec = aot_mint.ExportSpec(family=FAMILY, target="")
    return aot_mint.mint(pipe, spec, tmp, entry_workers=entry_workers)


def test_overlapped_and_serial_mints_share_one_cell_key(
    tmp_path: Path, fake_sm: Dict[str, str], wide_pool: None,
) -> None:
    """pgw#846: the overlap is a process change. The serial path (a forced
    K=1) and the overlapped pool path must stamp the SAME cell key — and the
    overlapped result must carry the overlap's own books."""
    _declare()
    serial = _mint(tmp_path / "serial", entry_workers=1)
    overlapped = _mint(tmp_path / "overlapped")

    # a mint produces a KEY SET, so "same cell key" becomes "the
    # same classes, keyed identically" — which is the stronger claim the row
    # always meant. An overlapped mint that agreed on a combined digest while
    # one class differed would have passed the old assertion.
    assert serial.keys == overlapped.keys, (
        "the overlapped mint re-keyed the cell — pgw#1052 must be "
        "byte-invisible in the artifact")
    assert overlapped.timings.get("entry_workers", 0) > 1, (
        "the overlapped mint never took the pool path on this box; the "
        "comparison proved nothing")
    assert "export_all_s" in overlapped.timings, (
        "the export phase's own wall must survive the overlap (the pgw#1052 "
        "acceptance names it)")
    # The phase table is a property of the MINT RUN, carried on
    # every entry's metadata. Read it off one entry rather than off a result
    # that no longer has a single metadata.
    pool_block = ((overlapped.entries[0].metadata.get("mint_phases") or {})
                  .get("pool") or {})
    assert "idle_source_s" in pool_block, (
        "the pool ledger must charge producer time to its named bucket")


def test_beats_interleave_export_and_pool(
    tmp_path: Path, fake_sm, wide_pool: None,
) -> None:
    """The obligation beat stays honest while two phases run concurrently:
    trace_graph and inductor_compile positions INTERLEAVE on the wire rather
    than forming two disjoint blocks."""
    _declare()
    beats: List[Tuple[str, int, int]] = []
    pipe = types.SimpleNamespace(unet=TinyUNet())
    spec = aot_mint.ExportSpec(family=FAMILY, target="")
    aot_mint.mint(
        pipe, spec, tmp_path / "out",
        on_progress=lambda phase, step, total, note: beats.append(
            (phase, step, total)))
    phases = [p for p, _s, _t in beats]
    assert "trace_graph" in phases and "inductor_compile" in phases
    first_pool = phases.index("inductor_compile")
    last_trace = len(phases) - 1 - phases[::-1].index("trace_graph")
    assert first_pool < last_trace, (
        "every inductor_compile beat came after the last trace_graph beat — "
        "the phases did not overlap on the wire")
