"""pgw#1356 — a SUCCESSFUL mint's roll-up announced itself as still running.

MEASURED on a rented A40 (pod ``zco8e1bx0t1jgk``, 2026-08-17, $0.54). Two
compile children, 36 of 36 graph classes, ``status=compiled exit=0``, pool
efficiency 99.89 % — and the one roll-up the hub sees said::

    aot_mint_phases  in_flight  status=in_flight  n_entries=0  3608490ms

Sixty minutes of measured work reported as *"still in flight, nothing
produced"*. An operator reading that row cannot tell a finished mint from a
mint the pod killed, which is the exact distinction pgw#848 built the on-disk
snapshot to preserve.

TWO INDEPENDENT DEFECTS, both of which have to be fixed for the row to be
readable, and neither of which is the ``key_axis_divergence`` refusal the same
pod also hit (that one is pgw#1340, fixed and unreleased):

1. **The reader and the writer disagreed about an address.**
   ``mint_graph_classes`` stamps the phase table onto ``MintedArtifact
   .mint_phases``; ``mint_supervisor._mint_phase_table_of`` read
   ``entry.metadata["mint_phases"]``. ``metadata`` is TCG's CLOSED artifact
   vocabulary — ``test_tcg_mint_parent_pgw1270`` asserts ``"mint_phases" not
   in survivor.metadata`` BY NAME — so the reader could never find it, and
   ``phase_table`` fell through to the snapshot on every successful mint. The
   snapshot's terminus is ``in_flight``, because a snapshot is a beat.

2. **The table's entry rows had no writer.** ``_mint_phase_table`` folded
   ``MintProgress.minted``, a list the SERIAL driver appended to as it
   exported in-process. pgw#1215 deleted that driver (``aot_mint.mint`` does
   not exist) and nothing has written the field since, so ``n_entries`` was
   ``0`` and ``entries``/``phases`` were empty for a mint of any size. The
   K-wide pool holds the real thing per GRAPH CLASS in
   ``EntryCompilePool.class_spans``.

Driven over the REAL relay — ``mint_graph_classes`` -> ``_mint_phase_table_of``
-> ``phase_table`` -> ``_emit_aot_phases`` — because the defect was a seam
between those four and any one of them tested alone reads green.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from gen_worker import aot_compile_pool, aot_mint, mint_supervisor

_CLASSES = ("unet/a", "unet/b", "unet/c")


def _metadata(name: str, key: str) -> Dict[str, Any]:
    """TCG's closed envelope — the only shape a packed class can carry."""
    return {
        "compiled_graph_format": 1,
        "kind": "aot-inductor",
        "compiled_graph_key": key,
        "graph_class": {"name": name, "class_hash": key[-16:], "graph": {}},
        "sm": "cpu-test",
        "toolchain": {"torch": "test"},
    }


def _spans(export_s: float, compile_s: float) -> Dict[str, float]:
    """One class's spans, in the FLAT shape ``aot_compile_child`` writes."""
    return {
        "export_s": export_s, "compile_s": compile_s, "reuse_s": 0.0,
        "nodes": 128.0, "lowering_s": 0.5, "codegen_s": 0.25,
        "host_compile_s": 1.0, "graph_passes_s": 0.125,
    }


class _Pool:
    """The K-wide pool, stubbed at the ONE boundary a laptop cannot cross.

    Everything the defect lives in — the fold, the stamp, the parent-side
    relay — is the real code. What is replaced is the spawn of compile
    children, which is a real AOTI compile and therefore a pod leg.
    """

    peak_rss_bytes = 0

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        self.entry_seconds: Dict[str, float] = {
            "share-000": 4.0, "share-001": 4.0}
        self.class_spans: Dict[str, Dict[str, float]] = {}
        self.entry_overlays: Dict[str, Dict[str, float]] = {}
        self.landed: List[str] = []

    def compile(self, *_args: Any, **kwargs: Any) -> Any:
        """Land the classes one at a time, beating between each.

        The pool populates ``class_spans`` as reports arrive, so a mint killed
        part-way has to leave the landed rows behind it — pinned by
        :func:`test_a_killed_mint_leaves_the_classes_that_landed_on_disk`.
        """
        on_share = kwargs.get("on_class")  # pgw#1371: per-class beats
        packed: Dict[str, aot_compile_pool.PackedGraphClass] = {}
        for index, name in enumerate(_CLASSES):
            key = "cg-key-v1-" + f"{index}".rjust(56, "0")
            self.class_spans[name] = _spans(1.0, 2.0)
            self.entry_overlays[f"share-{index:03d}"] = {"autotune_s": 0.5}
            self.landed.append(name)
            packed[name] = aot_compile_pool.PackedGraphClass(
                name=name, key=key, artifact=f"/tmp/{key}.tar.gz",
                metadata=json.dumps(_metadata(name, key)))
            if on_share is not None:
                on_share(name, index + 1, len(_CLASSES))
        return packed


@pytest.fixture(autouse=True)
def _pool(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(aot_compile_pool, "EntryCompilePool", _Pool)
    monkeypatch.setattr(aot_mint, "_pool_facts", lambda _pool: {})
    monkeypatch.setattr(
        aot_mint, "canonicalize_packed_classes", lambda _blocks, _metas: {})


def _mint(
    tmp_path: Path, snapshot: Path | None = None,
) -> aot_mint.MintResult:
    width = aot_compile_pool.entry_workers(
        len(_CLASSES), limit=2, vcpus=16,
        available_bytes=64 * 1024 ** 3, device_lock=True)
    return aot_mint.mint_graph_classes(
        aot_compile_pool.EntryJob(
            function="generate", modules=("m",), out_dir=str(tmp_path)),
        workdir=tmp_path / "pool", width=width,
        spec=aot_mint.ExportSpec(family="sdxl", target="unet"),
        phase_snapshot=snapshot)


def _wire(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> List[Dict[str, Any]]:
    """The ROWS THE HUB SEES, through the parent-side relay verbatim.

    The whole defect was a seam between the fold, the stamp, the supervisor's
    two-source rule and the emitter, so the assertion has to be made on the
    far side of all four — on the same ``aot_mint_phases`` row the A40 pod
    put on the wire. The snapshot file is REAL and is read the way
    :func:`mint_supervisor.SupervisedMint` reads it, because the fallback to
    it is the defect's second half: without it the roll-up reads ``aborted``
    instead of the production ``in_flight``, which is a different wrong answer
    to the same question.
    """
    snapshot = tmp_path / "phases.json"
    result = _mint(tmp_path, snapshot=snapshot)
    rows: List[Dict[str, Any]] = []

    def _emit(kind: str, detail: str, **kw: Any) -> None:
        rows.append({"kind": kind, "detail": detail, **kw})

    monkeypatch.setattr(aot_mint.activity_mod, "emit_event", _emit)
    mint_supervisor._emit_aot_phases(
        mint_supervisor.phase_table(
            mint_supervisor._mint_phase_table_of(result),
            mint_supervisor._read_snapshot(snapshot)),
        family="sdxl", execution_lane="bf16-w16a16",
        terminus="", elapsed_s=3608.49)
    assert rows, "a mint that produced N classes emitted no roll-up at all"
    return rows


def _rollup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> Dict[str, Any]:
    return _wire(tmp_path, monkeypatch)[0]


def test_a_finished_mint_does_not_announce_itself_as_in_flight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """THE PRODUCTION RED, on the exact wire row the A40 pod emitted.

    Before pgw#1356 the supervisor could not read the table the mint wrote, so
    ``phase_table`` fell through to the on-disk snapshot and the roll-up
    carried the snapshot's ``in_flight`` terminus — on a mint that reached its
    own terminus with every declared class packed.
    """
    row = _rollup(tmp_path, monkeypatch)

    assert row["kind"] == aot_mint.MINT_PHASES_KIND
    assert row["phase"] == aot_mint.activity_mod.PHASE_MINTED, (
        "status=in_flight on a mint that finished — the operator cannot tell "
        "it from a mint the pod killed")
    assert "status=minted" in row["detail"]


def test_the_rollup_counts_the_graph_classes_the_mint_produced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``n_entries=0`` on a 36-class mint, measured on the A40.

    The count is the survivor set, so it is the number of artifacts that can
    be published — not the number of shares, and not zero.
    """
    rows = _wire(tmp_path, monkeypatch)

    assert f"n_entries={len(_CLASSES)}" in rows[0]["detail"]
    assert {row["phase"] for row in rows if row["phase"].startswith("entry:")} \
        == {f"entry:{name}" for name in _CLASSES}, (
        "the per-class rows are how a slow mint says WHICH class it was slow "
        "in; an empty entries block emits none of them")


def test_the_rollup_carries_the_per_class_seconds_the_children_measured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Paul's question — *"why is AOT mint so much slower than JIT?"* — is
    answered by WHICH class the minutes are in. An empty ``entries`` block and
    empty ``phases`` totals answer nothing, and that is what every mint since
    pgw#1215 has reported.

    ``phases`` folds the flat ``torchcg.spans`` partition labels the child
    actually writes, rather than a nested block no child has ever written.
    """
    table = mint_supervisor._mint_phase_table_of(_mint(tmp_path))

    assert table["totals"]["export_s"] == pytest.approx(1.0 * len(_CLASSES))
    assert table["totals"]["compile_s"] == pytest.approx(2.0 * len(_CLASSES))
    assert table["phases"]["host_compile_s"] == pytest.approx(
        1.0 * len(_CLASSES))
    assert table["phases"]["lowering_s"] == pytest.approx(0.5 * len(_CLASSES))
    assert table["entries"]["unet/a"]["nodes"] == 128.0
    assert set(table["entries"]) == set(_CLASSES)


def test_overlays_are_reported_at_the_granularity_they_are_measured_at(
    tmp_path: Path,
) -> None:
    """pgw#830: overlays nest INSIDE partition members, so they ride beside
    ``phases`` and are never summed into it. The child discards its per-class
    overlay split, so the honest source is the pool's per-SHARE ledger — an
    invented per-class attribution would be worse than none.
    """
    table = mint_supervisor._mint_phase_table_of(_mint(tmp_path))

    assert table["overlays"]["autotune_s"] == pytest.approx(
        0.5 * len(_CLASSES))
    assert "autotune_s" not in table["phases"]


def test_a_killed_mint_leaves_the_classes_that_landed_on_disk(
    tmp_path: Path,
) -> None:
    """pgw#848's promise, unmet since pgw#1215 deleted the field it read.

    ``MintProgress`` binds the pool's LIVE ``class_spans`` rather than copying
    a snapshot of it, so the table written on every beat names what has landed
    so far. Attempt sixteen was killed at class 30 of 36 and reported zero
    rows; this is the fence against that returning.
    """
    snapshot = tmp_path / "phases.json"
    _mint(tmp_path, snapshot=snapshot)
    table = json.loads(snapshot.read_text())

    assert table["terminus"] == "in_flight", (
        "a beat is a beat — only the mint's own terminus may claim otherwise")
    assert table["n_entries"] > 0, (
        "a killed mint that had already packed classes must not report zero")
    assert set(table["entries"]) <= set(_CLASSES)


def test_the_supervisor_reads_the_table_off_the_typed_field(
    tmp_path: Path,
) -> None:
    """Stated so the address cannot drift back.

    TCG's artifact vocabulary is CLOSED and refuses metadata carrying an extra
    field, so ``metadata["mint_phases"]`` is not a place the writer is allowed
    to put it — the previous reader was asking for something that could not
    legally exist.
    """
    result = _mint(tmp_path)
    survivor = result.entries[0]

    assert survivor.mint_phases["n_entries"] == len(_CLASSES)
    assert "mint_phases" not in survivor.metadata
    assert mint_supervisor._mint_phase_table_of(result) == survivor.mint_phases
