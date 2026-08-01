"""pgw#825 — a regional cell's branch pair is BINDABLE, and the gate says so
before the compile is paid for.

Measured on a real L4 (gen-worker 0.84.0, release ``c47531a8577477048739cf1b``,
sdxl 0.2.104, lane ``w8a8-lora64``, recipe ``aot-regional``, pod
``0oqmb9dgs9lttz``): the mint compiled, reached per-entry publication, and was
refused at **351.73 s** —

    entry 'unet/adapter=true,block=BasicTransformerBlock#0,cfg=true/...':
    bindability gate: 20 declared state_dict constant(s) are absent from the
    resident module's state_dict, e.g. ['attn1.to_q.lora_a', ...]

The mechanism, proven here with a real ``torch.export`` + AOTI pack:

* ``w8a8_lora.alloc_branch_buffers`` registers ``lora_a``/``lora_b`` on the
  w8a8 lane's scaled linears with **persistent=False**, so a checkpoint never
  carries a zeroed adapter;
* ``module.state_dict()`` omits non-persistent buffers — but ``torch.export``
  still lifts them as BUFFER inputs and AOTInductor still declares them
  ``ConstantType::Buffer`` under their real FQN, i.e. ``source=state_dict``;
* so a bind table built from ``state_dict()`` declares constants no lookup
  could resolve. The gate was right; the TEMPLATE was wrong, on the mint side
  and at both arm sites.

The lift (pgw#725) is not the answer here and is not the defect: it wraps the
DENOISER's forward, and a regional entry is exported one block deep from the
block's own signature. A regional block's branch pair stays module-resident and
binds per instance BY REFERENCE (``user_managed=True``), which buys regional
the same property lifting buys the family graph — an adapter swap is an
in-place buffer write, never a rebind and never a recompile.

Three things are asserted, because attempt eight paid for all three:

1. the bindability template covers non-persistent buffers, on BOTH sides;
2. the mismatch is refused BEFORE the entry's compile, not after it;
3. an ABORTED mint still reports its per-phase table.
"""

from __future__ import annotations

import types
from typing import Any, Dict, List

import pytest

torch = pytest.importorskip("torch")

import torch.nn as nn  # noqa: E402

from gen_worker import (  # noqa: E402
    aot_mint,
    aot_package,
    aot_regional,
    aot_serve,
    compile_cache,
)
from gen_worker.api.decorators import Compile  # noqa: E402
from gen_worker.api.export_contract import (  # noqa: E402
    Dim,
    GraphClass,
    Input,
    register_export_declaration,
    reset_export_declarations,
)

FAMILY = "tiny825"
WIDTH = 8
BUCKET = 4


class BranchLinear(nn.Module):
    """One branch-capable leaf in the shape the w8a8 lane produces.

    ``alloc_branch_buffers``' scaled-linear arm verbatim: the pair is a
    REGISTERED buffer (it must move with the module and be read natively by
    the forward) and NON-PERSISTENT (a checkpoint must not carry a zeroed
    adapter). That combination is the whole defect — see
    ``test_the_production_allocator_really_produces_this_shape``.
    """

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(WIDTH, WIDTH, bias=False)
        self.register_buffer(
            "lora_a", torch.zeros(BUCKET, WIDTH), persistent=False)
        self.register_buffer(
            "lora_b", torch.zeros(WIDTH, BUCKET), persistent=False)

    def forward(self, x: Any) -> Any:
        return self.lin(x) + (x @ self.lora_a.t()) @ self.lora_b.t()


class TinyBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn1 = BranchLinear()
        self.ff = BranchLinear()

    def forward(self, hidden: Any) -> Any:
        return self.ff(torch.tanh(self.attn1(hidden)))


class TinyUNet(nn.Module):
    """A denoiser that declares its own repeated block, as diffusers does."""

    _repeated_blocks = ("TinyBlock",)

    def __init__(self, depth: int = 3) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([TinyBlock() for _ in range(depth)])

    def forward(self, sample: Any) -> Any:
        for block in self.blocks:
            sample = block(sample)
        return sample


def _declare() -> Any:
    reset_export_declarations()
    return register_export_declaration(Compile(
        family=FAMILY,
        targets=("unet",),
        regional=True,
        dims=(Dim("B", carried_by=(("sample", 0),)),),
        classes=(GraphClass(dims={"B": 2}),),
        inputs=(Input("sample", shape=("B", WIDTH)),),
        shape_strategy="static-rows",
        warm_changes_key=False,
    ))


def _pipe() -> Any:
    return types.SimpleNamespace(unet=TinyUNet().eval())


def _spec() -> aot_mint.ExportSpec:
    return aot_mint.ExportSpec(family=FAMILY, target="")


def _fake_sm(mp) -> None:
    full = {"sku": "", "sm": "sm_89", "torch": str(torch.__version__), "cuda": ""}
    mp.setattr(compile_cache, "runtime_key", lambda: dict(full))
    mp.setattr(aot_serve, "runtime_key", lambda: dict(full))


@pytest.fixture(autouse=True)
def _fresh_registry():
    reset_export_declarations()
    yield
    reset_export_declarations()


# ---------------------------------------------------------------------------
# The shape of the defect — proven against the PRODUCTION allocator
# ---------------------------------------------------------------------------


def test_the_production_allocator_really_produces_this_shape() -> None:
    """``alloc_branch_buffers`` on the w8a8 lane's own scaled linear: a
    registered, NON-PERSISTENT pair that ``state_dict()`` does not report."""
    from gen_worker.models import w8a8, w8a8_lora

    mod = w8a8.fp8_scaled_linear_class()(
        WIDTH, WIDTH, bias=False, compute_dtype=torch.float32,
        static_input_scale=False)
    w8a8_lora.alloc_branch_buffers(mod, BUCKET)

    assert "lora_a" in dict(mod.named_buffers())
    assert "lora_b" in dict(mod.named_buffers())
    # The defect's precondition, in one line.
    assert "lora_a" not in mod.state_dict()
    assert "lora_b" not in mod.state_dict()
    # And the fixed template DOES see them.
    resident = aot_serve.resident_constants(mod)
    assert {"lora_a", "lora_b"} <= set(resident)


def test_resident_constants_is_a_superset_of_state_dict() -> None:
    block = TinyBlock().eval()
    resident = aot_serve.resident_constants(block)
    assert set(block.state_dict()) < set(resident)
    assert {"attn1.lora_a", "attn1.lora_b", "ff.lora_a", "ff.lora_b"} <= set(
        resident)


# ---------------------------------------------------------------------------
# The real regional mint: RED at base, both fork classes here after the fix
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cell(tmp_path_factory, request) -> Dict[str, Any]:
    """ONE real regional mint (torch.export + AOTI, CPU, no harness stand-in).

    At base this raises ``MintRefused: ... bindability gate: 4 declared
    state_dict constant(s) are absent from the resident module's state_dict``
    — the pod's sentence, at 1/5th the entry count and none of the dollars —
    and every assertion below is unreachable.
    """
    from _pytest.monkeypatch import MonkeyPatch

    mp = MonkeyPatch()
    request.addfinalizer(mp.undo)
    _fake_sm(mp)
    _declare()
    tmp = tmp_path_factory.mktemp("cell825")
    pipe = _pipe()
    result = aot_mint.mint(
        pipe, _spec(), tmp / "out", allow_regressed_lanes=True)
    reset_export_declarations()
    return {"pipe": pipe, "result": result}


def test_a_regional_mint_packs_its_block_entry(cell) -> None:
    entries = cell["result"].metadata["entries"]
    assert entries, "the regional mint produced no entry"
    assert all(aot_regional.BLOCK_FORK in dict(
        [tuple(row) for row in block["fork"]]) for block in entries.values())


def test_the_branch_pair_is_DECLARED_and_BINDABLE(cell) -> None:
    """The gate's subject, from the packed artifact's own constant table: the
    pair is declared ``source=state_dict`` — and the block the serve arm binds
    from carries it."""
    entries = cell["result"].metadata["entries"]
    block_meta = next(iter(entries.values()))
    declared = {row["fqn"]: row["source"] for row in block_meta["constants"]}
    for name in ("attn1.lora_a", "attn1.lora_b", "ff.lora_a", "ff.lora_b"):
        assert declared.get(name) == aot_serve.SOURCE_STATE_DICT, declared
    resident = aot_serve.resident_constants(cell["pipe"].unet.blocks[0])
    unresolved = [fqn for fqn, source in declared.items()
                  if source == aot_serve.SOURCE_STATE_DICT
                  and fqn not in resident]
    assert unresolved == []


def test_a_block_entry_declares_NO_lifted_inputs(cell) -> None:
    """A block never carries the lifted signature — the lift wraps the
    denoiser. Inheriting the family's ``lifted_inputs`` here would record a
    contract the entry's program does not have."""
    for block_meta in cell["result"].metadata["entries"].values():
        assert block_meta["graph"]["lifted_inputs"] == []


# ---------------------------------------------------------------------------
# The pre-pay fix — the refusal must precede the compile
# ---------------------------------------------------------------------------


def test_the_program_side_gate_names_the_same_constants() -> None:
    block = TinyBlock().eval()
    program = torch.export.export(
        block, (torch.randn(2, WIDTH),), strict=False)

    # The old template: state_dict() alone.
    stale = aot_package.unbindable_program_constants(
        program, tuple(block.state_dict()))
    assert stale and "lora_a" in stale[0]

    # The fixed template.
    assert aot_package.unbindable_program_constants(
        program, tuple(aot_serve.resident_constants(block))) == []


def test_a_mismatch_is_REFUSED_BEFORE_a_kernel_is_built(
    tmp_path, monkeypatch,
) -> None:
    """pgw#825's second ask. Attempt eight paid ~4-6 minutes of L4 per entry
    and THEN learned the entry was unpublishable. Compiling here is a test
    failure, not a slow test."""
    _fake_sm(monkeypatch)
    _declare()

    def _never(*_a: Any, **_k: Any) -> Any:
        raise AssertionError("a kernel was compiled for an unbindable entry")

    monkeypatch.setattr(aot_mint, "compile_entry_files", _never)
    # Put the template back the way it was: the arm cannot see the branch pair.
    monkeypatch.setattr(
        aot_serve, "resident_constants", lambda m: dict(m.state_dict()))

    with pytest.raises(aot_mint.MintRefused) as excinfo:
        aot_mint.mint(
            _pipe(), _spec(), tmp_path / "out", allow_regressed_lanes=True)
    assert "pre-compile bindability gate" in str(excinfo.value)
    assert "lora_a" in str(excinfo.value)


# ---------------------------------------------------------------------------
# pgw#725 G3, in the shape a regional entry can be asked it
# ---------------------------------------------------------------------------


def test_a_registered_branch_pair_SURVIVES_export_as_a_bindable_buffer() -> None:
    block = TinyBlock().eval()
    program = torch.export.export(
        block, (torch.randn(2, WIDTH),), strict=False)
    assert aot_mint._regional_branch_gaps(block, program, BUCKET) == []


def test_a_dict_home_branch_is_REFUSED_as_a_baked_adapter() -> None:
    """The cast-hook/plain-Linear lanes keep the pair in the module
    ``__dict__``, where export lifts it as a tensor CONSTANT whose bytes then
    ship in the literal payload — a permanently zeroed adapter that serves the
    base model silently for every attach. Measured here, not argued: the pair
    lands in ``ep.constants`` and never in the program's buffer set."""
    from gen_worker.models import w8a8_lora

    class PlainBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attn1 = nn.Linear(WIDTH, WIDTH, bias=False)

        def forward(self, x: Any) -> Any:
            return self.attn1(x)

    block = PlainBlock().eval()
    w8a8_lora.alloc_branch_buffers(block.attn1, BUCKET)
    assert "lora_a" in block.attn1.__dict__          # the __dict__ home
    program = torch.export.export(
        block, (torch.randn(2, WIDTH),), strict=False)
    assert "attn1.lora_a" in (program.constants or {})
    assert "attn1.lora_a" not in aot_package.program_state_dict_fqns(program)

    gaps = aot_mint._regional_branch_gaps(block, program, BUCKET)
    assert gaps and "attn1.lora_a" in gaps[0]
    # A bucket-0 (branchless) class is not asked the question.
    assert aot_mint._regional_branch_gaps(block, program, 0) == []


# ---------------------------------------------------------------------------
# The instrumentation gap — an ABORTED mint still reports where it spent
# ---------------------------------------------------------------------------


def test_an_aborted_mint_carries_its_partial_phase_table(
    tmp_path, monkeypatch,
) -> None:
    """`aot_mint_phases` emitted only ``total_s`` on an abort, so attempt eight
    could not produce a single on-pod compile number despite paying for
    compiles. Every terminus reports."""
    _fake_sm(monkeypatch)
    _declare()

    real_gate = aot_mint._gate_and_declare_entry

    def _refuse_at_publication(row: Any, package: Any) -> Any:
        raise aot_mint.MintRefused(f"entry {row.name!r}: bindability gate: nope")

    monkeypatch.setattr(
        aot_mint, "_gate_and_declare_entry", _refuse_at_publication)
    assert real_gate is not None

    with pytest.raises(aot_mint.MintRefused) as excinfo:
        aot_mint.mint(
            _pipe(), _spec(), tmp_path / "out", allow_regressed_lanes=True)

    table = getattr(excinfo.value, "mint_phases", {})
    assert table, "an aborted mint reported no phase table"
    assert table["terminus"] == "aborted"
    assert table["n_entries"] >= 1
    assert table["totals"]["export_s"] > 0
    assert table["totals"]["compile_s"] > 0
    assert table["totals"]["total_s"] > 0
    # Per ENTRY, which is the number the splitter needs.
    assert all("compile_s" in t for t in table["entries"].values())


def test_the_aborted_roll_up_is_NOT_reported_as_minted(monkeypatch) -> None:
    from gen_worker import activity as activity_mod

    seen: List[Dict[str, Any]] = []
    monkeypatch.setattr(
        activity_mod, "emit_event",
        lambda kind, detail, **kw: seen.append(
            {"kind": kind, "detail": detail, **kw}))
    aot_mint.emit_phase_events(
        family=FAMILY, lane="w8a8-lora64", terminus="aborted",
        table={
            "n_entries": 2, "totals": {"total_s": 351.73},
            "phases": {}, "autotune": {},
            "entries": {"unet/block=A#0/B=2": {"export_s": 3.0, "compile_s": 200.0}},
        })
    roll_up = [row for row in seen if "entry=" not in row["detail"]]
    assert roll_up and roll_up[0]["phase"] == "aborted"
    assert roll_up[0]["duration_ms"] == 351730
    per_entry = [row for row in seen if "entry=" in row["detail"]]
    assert per_entry and per_entry[0]["duration_ms"] == 203000
