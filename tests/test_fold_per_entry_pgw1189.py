"""pgw#1189 — an ABANDONED mint's per-compiled graph partition was dropped, so the one
question the compile loop kept re-opening was structurally unanswerable.

pgw#830 measures the child's whole wall (`child_seal_s`, `child_program_load_s`,
`child_interp_s`, `child_boot_s`, `reap_lag_s`, `parent_stage_s`…) and pgw#832's
seal split rides the overlays. Both reach the phase table through
``_fold_pool_results`` — which runs ONLY after ``_drive_pool`` RETURNS. The
``finally`` above it refreshes only ``progress.pool_ledger``.

So a mint killed mid-pool kept its LEDGER and lost its per-compiled graph SPANS. Verified
against the standing hub, 2026-08-12: all three recorded gen-worker 0.112.0
mints carry the six inductor leaf timers and `overlays={}` — not one child span —
and the two sdxl mints were abandoned at 1/36 and 16/36. **The 16/36 one is the
run th#1834's P0-E read its figures from**, which is why "what is the ~39 s
residual" was answered by inference: the split was never on the wire.

pgw#848 had already written the rule this violates, for the ledger, one line
away: *"A mint killed at compiled graph 30 of 36 then leaves 30 compiled graphs' worth of
measurement on disk instead of one bare 'no compiled graph produced' row."* The fold is
now per compiled graph, at the same `_tick` that banks the ledger, so an compiled graph's numbers
are durable the moment the compiled graph finishes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pytest

from gen_worker import aot_mint


class _StubPool:
    """The pool's read surface, with a child's REAL span shape on each compiled graph.

    Not a mock of the thing under test: `_drive_pool`/`_fold_pool_results` are
    the production functions here, and this stands in only for the subprocess
    fleet, which cannot run on a box with no torch and no GPU.
    """

    def __init__(self, names: Sequence[str], *, fail_at: Optional[int] = None):
        self._names = list(names)
        self._fail_at = fail_at
        self.compiled_graph_seconds: Dict[str, float] = {}
        self.compiled_graph_phases: Dict[str, Dict[str, float]] = {}
        self.compiled_graph_overlays: Dict[str, Dict[str, float]] = {}
        self.compiled_graph_metrics_raw: Dict[str, Dict[str, float]] = {}
        self.peak_rss_bytes = 0
        self.width = type("W", (), {"workers": 3, "facts": lambda self: {
            "compiled_graph_workers": 3, "binding": "cores"}})()
        self.completed: List[str] = []

    def facts(self) -> Dict[str, Any]:
        return {"compiled_graph_workers": 3}

    def _finish(self, name: str) -> None:
        self.compiled_graph_seconds[name] = 127.8
        # The shape `_close_compiled_graph_partition` really produces: the inductor
        # leaves PLUS the child partition. It is the second half that has
        # never reached a reader.
        self.compiled_graph_phases[name] = {
            "codegen_s": 40.2, "host_compile_s": 19.7, "lowering_s": 15.6,
            "graph_passes_s": 11.6, "autotune_s": 1.8, "triton_s": 1.0,
            "compile_wall_s": 91.9, "child_program_load_s": 36.0,
            "child_seal_s": 12.4, "child_interp_s": 2.1,
            "child_boot_s": 2.4, "reap_lag_s": 0.3, "parent_stage_s": 1.1,
        }
        self.compiled_graph_overlays[name] = {
            "seal_libhash_s": 0.07, "seal_config_s": 11.2,
            "seal_scrub_s": 0.4, "seal_effective_s": 0.5,
        }
        self.completed.append(name)

    def compile(self, compiled_graphs: Any, *, on_compiled_graph: Any = None,
                expected_total: int = 0) -> Dict[str, List[str]]:
        out: Dict[str, List[str]] = {}
        for i, (name, _program) in enumerate(compiled_graphs):
            if self._fail_at is not None and i == self._fail_at:
                # A drain / co-tenancy kill mid-pool: the compiled graphs BEFORE it
                # finished and their seconds are real.
                raise aot_mint.aot_compile_pool.CompiledGraphCompileFailed(
                    name, f"compiled_graph {name!r}: the pod drained mid-compile")
            self._finish(name)
            out[name] = [f"/tmp/{name}.so"]
            if on_compiled_graph is not None:
                on_compiled_graph(name, len(out), expected_total or len(self._names))
        return out


def _compiled_graph(name: str) -> Any:
    return aot_mint._MintedCompiledGraph(
        name=name, spec=None, module=None, owner=None, program=None,
        input_names=(), flat_leaves=(), files=[], timings={"export_s": 86.7})


CHILD_SPANS = ("child_program_load_s", "child_seal_s", "child_interp_s",
               "compile_wall_s")


def _drive(names: Sequence[str], *, fail_at: Optional[int] = None,
           progress: Any = None):
    pool = _StubPool(names, fail_at=fail_at)
    rows = [_compiled_graph(n) for n in names]
    fold = aot_mint._compiled_graph_timing_folder(rows, pool)
    source = ((r.name, None) for r in rows)
    try:
        by_compiled_graph = aot_mint._drive_pool(
            pool, source, expected_total=len(rows), progress=progress,
            on_compiled_graph_complete=fold)
    except aot_mint.MintRefused:
        by_compiled_graph = None
    return pool, rows, by_compiled_graph


# ---------------------------------------------------------------------------
# 1. THE DEFECT: an abandoned mint kept its ledger and lost its compiled graphs' spans
# ---------------------------------------------------------------------------


def test_an_abandoned_mint_keeps_the_partition_of_every_FINISHED_compiled_graph() -> None:
    """RED on master: `_fold_pool_results` never runs when `_drive_pool`
    raises, so compiled graphs 1-2 — which finished, and whose seconds were really
    spent — carry `export_s` and nothing else.

    This is the row a reader needs MOST (pgw#848's own words about the ledger),
    and it is the row every sdxl mint on record produced."""
    pool, rows, by_compiled_graph = _drive(["a", "b", "c"], fail_at=2)
    assert by_compiled_graph is None, "the drain must still abandon the mint"
    assert pool.completed == ["a", "b"], "two compiled_graphs really finished"

    for row in rows[:2]:
        have = set(row.timings.get("phases") or {})
        assert have.issuperset(CHILD_SPANS), (
            f"compiled_graph {row.name!r} finished and its child partition was "
            f"DISCARDED — the phase table carries {sorted(have)}; the "
            "abandoned mint is exactly the one whose numbers a reader needs")
        assert row.timings.get("overlays"), (
            f"compiled_graph {row.name!r} lost pgw#832's seal split")
        assert row.timings["compile_s"] == 127.8


def test_EVERY_production_drive_pool_call_folds_per_compiled_graph() -> None:
    """THE FENCE. The per-compiled graph fold is opt-in on `_drive_pool`, so the defect
    returns the moment a call site forgets it — and the symptom is invisible
    (a phase table that looks complete until the mint is abandoned).

    RED on master, structurally: neither call site passed a folder, because
    the parameter did not exist. Checked on the AST rather than by running a
    mint, so it covers the path this box cannot execute.
    """
    import ast

    src = Path(aot_mint.__file__).read_text()
    calls = [
        node for node in ast.walk(ast.parse(src))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_drive_pool"
    ]
    assert len(calls) == 2, (
        f"expected the overlapped and pre-exported drivers, found {len(calls)}")
    for call in calls:
        kwargs = {kw.arg for kw in call.keywords}
        assert "on_compiled_graph_complete" in kwargs, (
            f"a _drive_pool call site at line {call.lineno} does not fold per "
            "compiled_graph — an abandoned mint there loses every finished compiled_graph's "
            "measurement, which is pgw#1189 reintroduced")


def test_the_compiled_graph_that_never_finished_claims_nothing() -> None:
    """The other half: a fold that ran early would invent numbers for an compiled graph
    that never compiled. Absence here must stay absence."""
    _pool, rows, _ = _drive(["a", "b", "c"], fail_at=2)
    assert "phases" not in rows[2].timings
    assert "compile_s" not in rows[2].timings


def test_a_completed_mint_carries_every_compiledgraphs_partition() -> None:
    """The A4500 anomaly this issue would not close without: a mint that
    COMPLETED (status=minted, K=3) also reached the hub with leaves only, and
    abandonment cannot explain that one."""
    _pool, rows, by_compiled_graph = _drive(["a", "b", "c"])
    assert by_compiled_graph is not None and len(by_compiled_graph) == 3
    for row in rows:
        have = set(row.timings.get("phases") or {})
        assert have.issuperset(CHILD_SPANS), (
            f"a COMPLETED mint dropped {row.name!r}'s child partition")
        assert row.timings.get("overlays")


def test_the_fold_is_idempotent_across_the_per_compiled_graph_and_final_passes() -> None:
    """`_fold_pool_results` still runs at the end (it assembles `files` and
    raises the short-compiled graph refusal). Folding twice must not double anything —
    the timings are assignments, never accumulations."""
    pool, rows, by_compiled_graph = _drive(["a", "b"])
    assert by_compiled_graph is not None
    before = {r.name: dict(r.timings["phases"]) for r in rows}
    aot_mint._fold_pool_results(rows, pool, by_compiled_graph)
    for row in rows:
        assert row.timings["phases"] == before[row.name]
        assert row.timings["compile_s"] == 127.8
        assert len(row.files) == 1, "files folded twice"


# ---------------------------------------------------------------------------
# 2. The reading trap this lane measured, pinned where the next reader hits it
# ---------------------------------------------------------------------------


def test_the_leaf_counters_are_not_a_partition_and_say_so() -> None:
    """MEASURED on the hub, 2026-08-12: over 20 recorded sdxl compiled graphs the
    residual `compile_s - sum(leaves)` has median 37.7 s and **minimum −2.9 s**.
    A negative residual is proof the six leaves OVERLAP; anyone summing them as
    a breakdown of compile time is wrong, and P0-E's "unattributed ~39 s" was
    read that way.

    `compile_wall_s` is the honest denominator, and `PARTITION_KEYS` is the set
    that really partitions it."""
    from gen_worker import aot_compile_spans

    assert "compile_wall_s" not in aot_compile_spans.PARTITION_KEYS
    # The warning has to live where the NAME lives, not in a tracker: the
    # reader who is about to sum these is looking at this constant.
    src = Path(aot_compile_spans.__file__).read_text()
    head = src.split("PARTITION_KEYS")[0]
    assert "OVERLAP" in head.upper() and "compile_wall_s" in head, (
        "the leaf counters overlap and nothing says so where a reader looks")
