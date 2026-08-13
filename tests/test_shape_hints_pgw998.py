"""pgw#998 — the export handoff carries the ShapeEnv's symbol values.

THE DEFECT, measured by the micro-mint rig's first grid-shaped cycle (CPU,
torch 2.13.0+cu130, $0). ``aot_compile_pool`` saves each ``ExportedProgram``
and ``aot_compile_child`` loads it in another interpreter. The round trip
rebuilds ``ShapeEnv.var_to_val`` keyed by the size EXPRESSIONS instead of by
the free symbols::

    parent   {s11: 32, s37: 32, s18: 16, s57: 16}  replacements {s11: 2*s18, …}
    child    {2*s18: 32, 2*s57: 32, 4*s18*s57: 1024}   replacements {}

Inductor resolves an extent by substituting ``backed_var_to_val`` into it, so
an extent that IS one of those keys still resolves and every other one cannot::

    LoweringException: RuntimeError: ('unexpected None!', 512*s18*s57)
      target: aten.addmm.default

THE TRIGGER IS A DERIVED SYMBOL, NOT NONLINEARITY — and that correction is
this file's first test. A dim declared with ``multiple_of`` exports as
``2*s18``; a matmul M multiplying two such axes is ``512*s18*s57``, which is
no key. The SAME graph with the SAME product extent and no ``multiple_of``
survives the round trip, because there its keys are the bare symbols.

WHY IT MATTERS BEYOND THE RIG. z-image declares ``H_lat``/``W_lat`` with
``multiple_of=2`` on a 4-D latent under ``dynamic-collapse``. Any patch-embed
or attention reshape that folds the spatial extents into one matmul M is this
shape exactly, which is why pgw#998 is a prerequisite for that family's next
mint rather than a rig curiosity.

No GPU and no compile: every row here is an export plus a save/load, which is
the whole of the contract under test. The compiled proof is the rig
(`task rig:micro`), which mints the grid-shaped declaration end to end.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict, Tuple

import pytest

from gen_worker import aot_shape_hints
from gen_worker.aot_compile_pool import CompiledGraphJob

torch = pytest.importorskip("torch")


class _Grid(torch.nn.Module):
    """A matmul whose M extent is the PRODUCT of two dynamic axes."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = torch.nn.Linear(4, 128)

    def forward(self, x: Any) -> Any:
        c, h, w = x.shape
        return self.lin(x.permute(1, 2, 0).reshape(h * w, c)).sum()


def _export(multiple_of: int) -> Any:
    height = torch.export.Dim("H_lat_u", min=8, max=32)
    width = torch.export.Dim("W_lat_u", min=8, max=32)
    if multiple_of > 1:
        height, width = multiple_of * height, multiple_of * width
    return torch.export.export(
        _Grid().eval(), (torch.randn(4, 32, 32),), {},
        dynamic_shapes={"x": {1: height, 2: width}}, strict=True)


def _round_trip(program: Any) -> Any:
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "program.pt2")
        torch.export.save(program, path)
        return torch.export.load(path)


def _free_symbol_values(program: Any) -> Dict[str, Any]:
    env = aot_shape_hints.shape_env(program)
    return {
        str(key.name): value
        for key, value in (getattr(env, "var_to_val", {}) or {}).items()
        if getattr(key, "name", None)
    }


def _extent_symbols(program: Any) -> Tuple[str, ...]:
    env = aot_shape_hints.shape_env(program)
    assert env is not None
    out = set()
    for node in program.graph_module.graph.nodes:
        if node.op != "placeholder":
            continue
        for dim in getattr(node.meta.get("val"), "shape", ()) or ():
            expr = getattr(getattr(dim, "node", None), "expr", None)
            for symbol in getattr(expr, "free_symbols", ()) or ():
                out.add(str(symbol.name))
    return tuple(sorted(out))


# ---------------------------------------------------------------------------
# 1. THE DEFECT, AND WHAT ACTUALLY TRIGGERS IT
# ---------------------------------------------------------------------------


def test_the_round_trip_loses_the_values_of_derived_symbols() -> None:
    """RED: the child's map is keyed by expressions, so the free symbols the
    extents are written in have no value at all."""
    program = _export(multiple_of=2)
    parent = _free_symbol_values(program)
    assert set(_extent_symbols(program)) <= set(parent), parent

    loaded = _round_trip(program)

    orphaned = set(_extent_symbols(loaded)) - set(_free_symbol_values(loaded))
    assert orphaned, "the round trip is supposed to have dropped these"
    assert aot_shape_hints.unhinted_extents(loaded), \
        "an extent nothing can evaluate must be visible as a gap"


def test_without_a_multiple_of_the_same_product_survives() -> None:
    """The correction to the filing: nonlinearity is the victim, the DERIVED
    symbol is the cause. Same graph, same H*W extent, no coefficient."""
    loaded = _round_trip(_export(multiple_of=1))

    assert set(_extent_symbols(loaded)) <= set(_free_symbol_values(loaded))
    assert aot_shape_hints.unhinted_extents(loaded) == []


# ---------------------------------------------------------------------------
# 2. THE FIX — the parent's values, restored, from ONE authority
# ---------------------------------------------------------------------------


def test_restoring_the_parents_values_closes_every_gap() -> None:
    program = _export(multiple_of=2)
    values = aot_shape_hints.symbol_values(program)
    loaded = _round_trip(program)

    restored = aot_shape_hints.restore_symbol_values(loaded, values)

    assert restored, "nothing was restored, so nothing was carried"
    assert aot_shape_hints.unhinted_extents(loaded) == []
    assert set(_extent_symbols(loaded)) <= set(_free_symbol_values(loaded))


def test_the_shipped_values_are_symbols_not_expressions() -> None:
    """What crosses the wire is the parent's OWN map, free symbols only — the
    expression keys are the disease, not a fact worth shipping."""
    values = aot_shape_hints.symbol_values(_export(multiple_of=2))

    assert values, values
    assert all(name.isidentifier() for name in values), values
    assert all(isinstance(v, int) for v in values.values()), values


def test_the_job_carries_them_across_the_process_boundary() -> None:
    """The wire itself: a compile job that does not ship the values is a
    child that cannot lower, so the field is pinned here rather than left to
    be noticed on a pod."""
    import msgspec

    program = _export(multiple_of=2)
    job = CompiledGraphJob(
        compiled_graph="denoiser/cfg=false", program="/tmp/p.pt2", report="/tmp/r.json",
        symbol_values=aot_shape_hints.symbol_values(program))

    decoded = msgspec.json.decode(msgspec.json.encode(job), type=CompiledGraphJob)

    assert decoded.symbol_values == job.symbol_values
    assert decoded.symbol_values, "the job shipped an empty map"


def test_a_program_with_no_symbols_is_a_no_op() -> None:
    """A fully static compiled graph — sdxl's shape — must not acquire machinery."""
    program = torch.export.export(
        _Grid().eval(), (torch.randn(4, 32, 32),), {}, strict=True)

    assert aot_shape_hints.symbol_values(program) == {}
    assert aot_shape_hints.unhinted_extents(_round_trip(program)) == []
    assert aot_shape_hints.restore_symbol_values(_round_trip(program), {}) == 0


# ---------------------------------------------------------------------------
# 3. THE REFUSAL NAMES SOMETHING AN AUTHOR WROTE
# ---------------------------------------------------------------------------


def test_the_gap_is_reported_with_the_input_the_axis_and_the_dim() -> None:
    """`512*s18*s57` cost an hour of bisection because it names nothing. The
    refusal must name the input, the axis, and the DECLARED dim."""
    program = _export(multiple_of=2)
    labels = aot_shape_hints.symbol_labels(program)
    loaded = _round_trip(program)

    gaps = aot_shape_hints.unhinted_extents(loaded, labels)

    assert gaps, "the round trip left an extent nothing can evaluate"
    assert all(g.startswith("x[") for g in gaps), gaps
    assert all("carry no value" in g for g in gaps), gaps
    joined = " ".join(gaps)
    assert "H_lat_u" in joined and "W_lat_u" in joined, gaps
