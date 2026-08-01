"""pgw#829 — a conv-free block class collapses its SHAPE axis to one entry.

Attempt nine's sdxl regional mint published 72 entries: 9 aspect rows x 2 CFG
arms x 2 adapter arms x 2 block classes. pgw#830 then measured what those 72
actually cost — **a per-ENTRY fixed cost** (interpreter boot, the env seal,
the torch import, the staged-program load, the reap lag), paid 72 times, on a
pool that was already >= 95.5 % efficient. There were no stragglers to
recover; there was a constant being multiplied.

pgw#812 measured the lever: dynamic inner dims are **0.0 %** at serve on a
conv-FREE region, because ``decide_layout_opt`` bails only on conv + free
symbol and a conv-free region has no channels-last layout opt to lose. #730's
``static-rows`` verdict (+7.2 %) was always a statement about CONVS, and
sdxl's convs are in the SHELL, which regional leaves eager.

So this file proves, on real ``torch.export`` + real AOTInductor, CPU, $0:

1. a conv-free block class serves every declared class row from ONE entry —
   8 static entries become 2 (the fork axis is structural and stays);
2. every shape that used to have its own entry dispatches onto the collapsed
   one and is **no further from eager than the 8-entry cell it replaces**,
   per SHAPE and per block INSTANCE. Stated that way on purpose: a dynamic
   kernel is different compiled code from a static one, so demanding
   identical bytes would over-specify. What may not happen is drift, and
   #730's standing rule makes any of it disqualifying;
3. a conv-BEARING block class does NOT collapse — the premise is checked per
   class off the live module, never assumed fleet-wide;
4. pgw#831's folded-constant refusal still fires under the collapse (dynamic
   dims do not change what inductor folds, and a regional cell that bakes the
   prototype's weights must stay unpublishable);
5. the per-entry fixed cost pgw#830 named really does multiply with the entry
   count on a real pool — which is the whole arithmetic behind 72 -> 8.

Nothing here is mocked on the mint or the serve path: both cells are minted,
adopted through ``models.provision.arm_aot``, and CALLED.
"""

from __future__ import annotations

import copy
import types
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

torch = pytest.importorskip("torch")

import torch.nn as nn  # noqa: E402

from gen_worker import (  # noqa: E402
    aot_compile_pool,
    aot_declaration,
    aot_mint,
    aot_regional,
    aot_serve,
    compile_cache,
)
from gen_worker.api.decorators import Compile  # noqa: E402
from gen_worker.api.export_contract import (  # noqa: E402
    DYNAMIC_COLLAPSE,
    Dim,
    Fork,
    GraphClass,
    Input,
    register_export_declaration,
    reset_export_declarations,
)
from gen_worker.models import provision  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")

WIDTH = 8
DEPTH = 3
#: Four (H, W) latent rows -> block token counts 64, 128, 256, 384, all
#: distinct. Deliberately NOT symmetric: sdxl's aspect table is
#: (1536x640, 640x1536, ...), and a symmetric pair hands the block the
#: IDENTICAL shape, which is its own defect —
#: :func:`test_two_class_rows_that_reduce_to_ONE_block_shape_are_REFUSED`
#: owns that case, and this row set keeps the static A/B baseline
#: dispatchable so the collapse can be measured against something that works.
ROWS: Tuple[Tuple[int, int], ...] = ((8, 8), (8, 16), (16, 16), (16, 24))
#: The block token hull those rows produce.
HULL = (64, 384)
#: A symmetric pair — 8*16 == 16*8 — which is exactly sdxl's shape.
SYMMETRIC_ROWS: Tuple[Tuple[int, int], ...] = ((8, 16), (16, 8))


class AttnBlock(nn.Module):
    """The conv-free repeated block — sdxl's ``BasicTransformerBlock`` shape.

    Bias-FREE deliberately: the compiler folds a ``Linear`` bias into the
    epilogue and pgw#831's gate refuses a regional mint that does so. The
    bias case has its own test below, where it is the SUBJECT.
    """

    def __init__(self) -> None:
        super().__init__()
        self.attn1 = nn.Linear(WIDTH, WIDTH, bias=False)
        self.ff = nn.Linear(WIDTH, WIDTH, bias=False)

    def forward(self, hidden_states: Any) -> Any:
        return self.ff(torch.tanh(self.attn1(hidden_states)))


class BiasedBlock(AttnBlock):
    """Same block, with the bias inductor folds — pgw#831's subject."""

    def __init__(self) -> None:
        super().__init__()
        self.ff = nn.Linear(WIDTH, WIDTH, bias=True)


class ConvBlock(AttnBlock):
    """A repeated block that carries a CONVOLUTION.

    This is the class pgw#829 must NOT collapse: a free symbol on a
    conv-bearing graph turns off inductor's channels-last layout opt, which
    is the +7.2 % #730 measured and the reason sdxl declares static rows for
    its whole forward in the first place.
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv1d(WIDTH, WIDTH, 3, padding=1, bias=False)

    def forward(self, hidden_states: Any) -> Any:
        out = super().forward(hidden_states)
        return self.conv(out.transpose(1, 2)).transpose(1, 2)


def _shell(block_cls: type) -> type:
    """A CONV-BEARING shell over ``block_cls`` — sdxl's actual shape.

    The convs live here and stay eager; only the repeated blocks are
    compiled. The shell also flattens (H, W) into a single token axis, which
    is the reason this issue needs a derivation at all: the declaration binds
    ``H``/``W`` to ``sample``, and the block is handed a ``(B, H*W, C)``
    hidden state carrying neither declared name.
    """

    class ToyUNet(nn.Module):
        _repeated_blocks = (block_cls.__name__,)

        def __init__(self, depth: int = DEPTH) -> None:
            super().__init__()
            self.conv_in = nn.Conv2d(WIDTH, WIDTH, 3, padding=1, bias=False)
            self.blocks = nn.ModuleList([block_cls() for _ in range(depth)])
            self.config = {"width": WIDTH, "depth": depth}

        def forward(self, sample: Any) -> Any:
            x = self.conv_in(sample)
            b, c, h, w = x.shape
            tokens = x.permute(0, 2, 3, 1).reshape(b, h * w, c)
            for block in self.blocks:
                tokens = block(tokens)
            return tokens.reshape(b, h, w, c).permute(0, 3, 1, 2)

    return ToyUNet


def _declare(
    family: str, *, regional_strategy: str = "",
    rows: Tuple[Tuple[int, int], ...] = ROWS,
) -> Any:
    """sdxl's declaration in miniature: conv shell, CFG fork, aspect rows."""
    return register_export_declaration(Compile(
        family=family,
        targets=("unet",),
        regional=True,
        dims=(Dim("B", carried_by=(("sample", 0),)),
              Dim("H", carried_by=(("sample", 2),)),
              Dim("W", carried_by=(("sample", 3),))),
        forks=(Fork("cfg", served=(True, False)),),
        classes=tuple(
            GraphClass(dims={"B": b, "H": h, "W": w}, fork={"cfg": c})
            for h, w in rows for c, b in ((True, 2), (False, 1))),
        inputs=(Input("sample", shape=("B", WIDTH, "H", "W")),),
        # The SHELL keeps its convs and therefore keeps static rows. Only the
        # block population collapses — that split is the declaration change.
        shape_strategy="static-rows",
        regional_shape_strategy=regional_strategy,
        warm_changes_key=False,
    ))


def _cfg(family: str) -> Any:
    return types.SimpleNamespace(family=family, lora_bucket=0)


def _fake_sm(mp: Any) -> None:
    full = {"sku": "", "sm": "sm_89", "torch": str(torch.__version__),
            "cuda": ""}
    mp.setattr(compile_cache, "runtime_key", lambda: dict(full))
    mp.setattr(aot_serve, "runtime_key", lambda: dict(full))


@pytest.fixture(autouse=True)
def _fresh_registry():
    reset_export_declarations()
    yield
    reset_export_declarations()


def _mint(
    family: str, out: Path, *, block_cls: type = AttnBlock,
    model: Any = None,
) -> Dict[str, Any]:
    """One real regional mint. ``model`` lets two cells be minted from the
    SAME weights, which the A/B needs: two independently constructed toys
    would disagree numerically for reasons that have nothing to do with the
    collapse."""
    unet = copy.deepcopy(model) if model is not None \
        else _shell(block_cls)().eval()
    pipe = types.SimpleNamespace(unet=unet)
    result = aot_mint.mint(
        pipe, aot_mint.ExportSpec(family=family, target=""), out,
        allow_regressed_lanes=True, entry_workers=1)
    return {"result": result, "pipe": pipe}


# ---------------------------------------------------------------------------
# The A/B: one model, one toolchain, the SHAPE axis collapsed
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cells(tmp_path_factory, request) -> Dict[str, Any]:
    """Two REAL regional mints of one model — per-shape and collapsed."""
    from _pytest.monkeypatch import MonkeyPatch

    from gen_worker import host_isa

    try:
        host_isa.impose()
    except Exception:  # pragma: no cover — torchless / non-x86 runner
        pass
    mp = MonkeyPatch()
    request.addfinalizer(mp.undo)
    _fake_sm(mp)
    tmp = tmp_path_factory.mktemp("cells829")

    # ONE model, minted twice: the A/B is about the entry axis, and two
    # independently initialised toys could not be compared bit for bit.
    model = _shell(AttnBlock)().eval()
    out: Dict[str, Any] = {}
    for tag, strategy in (("static", ""), ("collapsed", DYNAMIC_COLLAPSE)):
        reset_export_declarations()
        family = f"tiny829{tag}"
        _declare(family, regional_strategy=strategy)
        out[tag] = _mint(family, tmp / tag, model=model)
        out[tag]["family"] = family
    reset_export_declarations()
    return out


def test_the_conv_free_block_collapses_EIGHT_entries_into_TWO(cells) -> None:
    """The headline count, derived rather than asserted from a target.

    8 = 4 declared aspect rows x 2 CFG arms, one block class. The CFG fork is
    STRUCTURAL — two batch sizes, two graph classes — so it survives; the
    SHAPE axis is what collapses. On the real sdxl declaration the same
    arithmetic reads 9 rows x 2 CFG x 2 adapter arms x 2 block classes = 72
    -> 2 x 2 x 2 = 8.
    """
    static = dict(cells["static"]["result"].metadata["entries"])
    collapsed = dict(cells["collapsed"]["result"].metadata["entries"])

    assert len(static) == len(ROWS) * 2 == 8
    assert sorted(collapsed) == [
        "unet/block=AttnBlock#0,cfg=false",
        "unet/block=AttnBlock#0,cfg=true",
    ], (
        f"the collapsed cell must carry ONE entry per structural fork "
        f"coordinate, not per shape row: {sorted(collapsed)}")

    # Named by the fork alone: the admitted range is the entry's recorded
    # CONTRACT, never its name (aot_declaration.entry_coordinates).
    assert all("H=" not in name for name in collapsed)
    # …and the per-shape names really did carry the dims, so this is a
    # collapse and not a renaming.
    assert all("H=" in name for name in static)


def test_the_collapsed_entry_declares_the_HULL_of_every_row(cells) -> None:
    """The derived contract, read off the artifact.

    The declaration binds ``H``/``W`` to ``sample``; the block never sees
    ``sample``. The token axis is therefore derived from what the shell
    actually fed the block on each declared row — 64, 128, 128, 256 — and the
    entry admits [64, 256].
    """
    entries = dict(cells["collapsed"]["result"].metadata["entries"])
    for name, block in entries.items():
        contract = aot_serve.contract_from_meta(block)
        (spec,) = [i for i in contract.inputs if i.name == "hidden_states"]
        # dim 0 is the CFG batch and stays STATIC — it is what discriminates
        # the two entries at dispatch.
        assert spec.shape[0] == (2 if "cfg=true" in name else 1)
        symbol = spec.shape[1]
        assert isinstance(symbol, str), (
            f"{name}: the block's token axis exported STATIC ({symbol}) — "
            f"there is nothing to collapse onto")
        assert contract.symbols[symbol] == HULL, contract.symbols
        # The feature axis never varied and must not have become symbolic.
        assert spec.shape[2] == WIDTH


def test_the_declaration_splits_the_two_POPULATIONS(cells) -> None:
    """``shape_strategy`` still governs the whole-graph route; only
    ``regional_shape_strategy`` governs the blocks. A family that flipped the
    first to serve the second would silently re-price the conv-bearing shell
    it also owns."""
    _declare("tiny829split", regional_strategy=DYNAMIC_COLLAPSE)
    from gen_worker.api.export_contract import export_declaration

    decl = export_declaration("tiny829split")
    assert decl is not None
    assert aot_declaration.effective_shape_strategy(decl) == "static-rows"
    assert aot_declaration.effective_shape_strategy(
        decl, regional=True) == DYNAMIC_COLLAPSE
    # Unset inherits — which is exactly the pre-pgw#829 behaviour.
    _declare("tiny829inherit")
    plain = export_declaration("tiny829inherit")
    assert plain is not None
    assert aot_declaration.effective_shape_strategy(
        plain, regional=True) == "static-rows"


def test_a_regional_strategy_without_regional_is_REFUSED() -> None:
    """A declaration nothing reads is how a fact quietly stops being true."""
    from gen_worker.api.export_contract import DeclarationError

    with pytest.raises(DeclarationError, match="regional=True"):
        register_export_declaration(Compile(
            family="tiny829bad", targets=("unet",), regional=False,
            dims=(Dim("B", carried_by=(("sample", 0),)),),
            classes=(GraphClass(dims={"B": 2}),),
            inputs=(Input("sample", shape=("B", WIDTH)),),
            shape_strategy="static-rows",
            regional_shape_strategy=DYNAMIC_COLLAPSE,
            warm_changes_key=False))


def test_the_ADAPTER_fork_survives_the_collapse(cells) -> None:
    """The doubling pgw#827 depends on is STRUCTURAL, not shape-driven.

    ``BlockDispatch`` routes by adapter arm first and only then by ingress,
    because a regional entry is exported from the block's own signature and
    both arms therefore declare the SAME contract (pgw#825/#827). So the arm
    axis can never be collapsed by a shape decision — and the collapse must
    not disturb it. Asserted at the plan level, where the interaction lives:
    the fork carries the arm, and ``with_adapter_arm`` preserves the row set
    the collapse is derived FROM.
    """
    from gen_worker.api.export_contract import export_declaration

    _declare("tiny829arm", regional_strategy=DYNAMIC_COLLAPSE)
    decl = export_declaration("tiny829arm")
    assert decl is not None
    plans = aot_declaration.cell_plans(decl, regional=True)
    assert len(plans) == 2, [p.fork for p in plans]
    assert all(len(p.rows) == len(ROWS) for p in plans)

    armed = [aot_mint.with_adapter_arm(p, arm)
             for p in plans for arm in (True, False)]
    assert len(armed) == 4
    # The doubling is real, the collapse is preserved (each armed plan still
    # spans every class row), and every coordinate still names a distinct
    # entry.
    assert all(len(p.rows) == len(ROWS) for p in armed)
    names = {aot_declaration.plan_entry_name(p) for p in armed}
    assert len(names) == 4, names
    assert all("H=" not in n for n in names), names
    # 4 plans x 1 conv-free block class = 4 entries here; on sdxl the same
    # arithmetic is 2 CFG x 2 arms x 2 block classes = 8, from 72.
    assert {aot_mint.adapter_arm(p.fork) for p in armed} == {True, False}


# ---------------------------------------------------------------------------
# PARITY — per shape AND per instance, against the cell it replaces
# ---------------------------------------------------------------------------


def _armed(cell: Dict[str, Any], tmp_path: Path) -> Any:
    pipe = copy.deepcopy(cell["pipe"])
    assert provision.arm_aot(
        pipe, _cfg(cell["family"]), tmp_path, cell["result"].artifact, 0,
        cell["result"].metadata), "the cell did not adopt"
    return pipe


def _sample(batch: int, h: int, w: int) -> Any:
    torch.manual_seed(0x829)
    return torch.randn(batch, WIDTH, h, w)


def test_EVERY_shape_dispatches_onto_the_collapsed_entry_and_matches(
    cells, tmp_path, monkeypatch,
) -> None:
    """The acceptance bar, made per-coordinate.

    For all 8 declared coordinates: the collapsed cell serves every block
    INSTANCE through ONE artifact, and its output is identical to the
    per-shape cell's and to eager's. A regional cell that got instance
    binding wrong (pgw#827) or constant folding wrong (pgw#831) fails here on
    instances 1..N-1 while looking perfect on instance 0, which is why the
    comparison is per instance and not on the prototype.
    """
    _fake_sm(monkeypatch)
    eager = copy.deepcopy(cells["static"]["pipe"])
    per_shape = _armed(cells["static"], tmp_path / "static")
    collapsed = _armed(cells["collapsed"], tmp_path / "collapsed")

    served_by: Dict[str, int] = {}
    for h, w in ROWS:
        for batch in (1, 2):
            sample = _sample(batch, h, w)
            before = aot_serve.execution_count(collapsed)
            with torch.no_grad():
                want = eager.unet(sample).clone()
                old = per_shape.unet(sample).clone()
                new = collapsed.unet(sample).clone()

            # Every one of the DEPTH instances went through the artifact —
            # not the prototype alone, and no silent eager fallback.
            assert aot_serve.execution_count(collapsed) == before + DEPTH, (
                f"B={batch} {h}x{w}: only "
                f"{aot_serve.execution_count(collapsed) - before} of {DEPTH} "
                f"block instances served through the collapsed entry; the "
                f"rest fell back to eager, which is a dispatch failure "
                f"wearing a correct answer")
            # The acceptance bar, stated as "no degradation against the
            # cell it replaces" rather than as bit-equality: a dynamic
            # kernel is different compiled code from a static one, so
            # demanding identical bytes would be over-specifying. What may
            # NOT happen is the collapsed cell drifting further from eager
            # than the 8-entry cell does.
            drift_old = (old - want).abs().max().item()
            drift_new = (new - want).abs().max().item()
            assert torch.allclose(new, old, atol=1e-6, rtol=1e-6), (
                f"B={batch} {h}x{w}: the collapsed cell disagrees with the "
                f"per-shape cell by {(new - old).abs().max().item():.3e} — "
                f"the whole premise is that one dynamic entry computes what "
                f"{len(ROWS)} static ones did")
            assert drift_new <= max(drift_old, 1e-6), (
                f"B={batch} {h}x{w}: the collapsed cell is FURTHER from "
                f"eager ({drift_new:.3e}) than the per-shape cell it "
                f"replaces ({drift_old:.3e}) — that is a quality "
                f"regression, and pgw#730's standing rule makes it "
                f"disqualifying")
            assert drift_new <= 1e-6, (
                f"B={batch} {h}x{w}: vs eager {drift_new:.3e}")

    # And it really was ONE entry per fork doing all of it.
    for name, count in _entry_calls(collapsed).items():
        served_by[name] = count
    assert len(served_by) == 2, served_by
    assert all(count == len(ROWS) * DEPTH for count in served_by.values()), (
        f"each collapsed entry must have served every declared row: "
        f"{served_by}")


def _entry_calls(pipeline: Any) -> Dict[str, int]:
    """Per-entry served-call counts, summed over every armed block instance."""
    out: Dict[str, int] = {}
    for state in aot_serve._marker_states(pipeline):
        runner = state.get("runner")
        if runner is None:
            continue
        for name, count in runner.entry_calls().items():
            out[name] = out.get(name, 0) + count
    return out


def test_a_shape_outside_the_derived_hull_is_a_NAMED_refusal(
    cells, tmp_path, monkeypatch,
) -> None:
    """The collapse widens what one entry admits; it must not widen what it
    SILENTLY admits. A token count above the derived hull is an ingress
    refusal served eager — the pgw#704 B2 property, on a derived range."""
    _fake_sm(monkeypatch)
    eager = copy.deepcopy(cells["collapsed"]["pipe"])
    pipe = _armed(cells["collapsed"], tmp_path / "oob")
    sample = _sample(2, 32, 32)  # 1024 tokens, hull is [64, 384]

    with torch.no_grad():
        want = eager.unet(sample)
        got = pipe.unet(sample)
    assert torch.allclose(got, want, atol=1e-6)
    assert aot_serve.ingress_refusals(pipe) == DEPTH, (
        "an out-of-hull call must be refused BY NAME on every instance and "
        "served eager, not admitted")
    reasons = _last_refusals(pipe)
    # `EntryDispatch.select` reports the aggregate `no_entry_admits` and
    # carries each entry's own reason inside it, so the DERIVED range is what
    # rejected the call — not a rank or dtype accident.
    assert all("no_entry_admits" in r and "range_violation" in r
               for r in reasons), reasons


def _last_refusals(pipeline: Any) -> List[str]:
    return [str(state.get("last_refusal") or "")
            for state in aot_serve._marker_states(pipeline)]


# ---------------------------------------------------------------------------
# The premise guard, and pgw#831's gate
# ---------------------------------------------------------------------------


def test_a_CONV_BEARING_block_class_does_NOT_collapse(tmp_path) -> None:
    """Per block class, off the live module — never a fleet-wide assumption.

    pgw#812's 0.0 % is measured on a CONV-FREE region. A block class that
    carries a conv keeps one static entry per declared row, because a free
    symbol would cost it the channels-last layout opt (#730's +7.2 %).
    """
    from _pytest.monkeypatch import MonkeyPatch

    from gen_worker import host_isa

    try:
        host_isa.impose()
    except Exception:  # pragma: no cover
        pass
    mp = MonkeyPatch()
    try:
        _fake_sm(mp)
        assert aot_regional.block_has_conv(ConvBlock())
        assert not aot_regional.block_has_conv(AttnBlock())

        _declare("tiny829conv", regional_strategy=DYNAMIC_COLLAPSE)
        cell = _mint("tiny829conv", tmp_path / "conv",
                     block_cls=ConvBlock)
        entries = sorted(cell["result"].metadata["entries"])
    finally:
        mp.undo()

    assert len(entries) == len(ROWS) * 2, (
        f"a conv-bearing block class must keep one STATIC entry per class "
        f"row even under dynamic-collapse: {entries}")
    assert all("H=" in name for name in entries), entries
    # Every entry is fully specialized — no symbol survived onto a conv graph.
    for block in cell["result"].metadata["entries"].values():
        assert not aot_serve.contract_from_meta(block).symbols


def test_two_class_rows_that_reduce_to_ONE_block_shape_are_REFUSED(
    tmp_path,
) -> None:
    """The defect pgw#829's own A/B walked into, and the gate that closes it.

    A regional entry is exported one block deep, and the shell has already
    flattened the latent extents into a token count — so two DECLARED rows
    that are different coordinates upstream can hand the block the identical
    shape. ``EntryDispatch`` then admits both and refuses ``entry_ambiguous``
    on every call: the cell arms, reports armed, and serves those coordinates
    100 % EAGER while looking healthy.

    This is not hypothetical, and it is worse than transposition. What
    collides is the token PRODUCT: sdxl's NINE aspect rows carry only FOUR
    distinct token counts — 15360 (1536x640, 640x1536), 15808 (1216x832,
    832x1216), 16128 (1344x768, 1152x896, 896x1152, 768x1344 — a quadruple
    whose members are not each other's transpose, since 96*168 == 112*144),
    and 16384 (1024x1024, the only unique row). Attempt nine's 72-entry cell
    could therefore have served exactly ONE of its nine aspect ratios
    compiled; the other eight were `entry_ambiguous` -> eager, per (CFG arm x
    adapter arm x block class). The mint now refuses it by name, and the
    collapse is the remedy the refusal names.
    """
    from _pytest.monkeypatch import MonkeyPatch

    from gen_worker import host_isa

    try:
        host_isa.impose()
    except Exception:  # pragma: no cover
        pass
    mp = MonkeyPatch()
    try:
        _fake_sm(mp)
        _declare("tiny829sym", rows=SYMMETRIC_ROWS)
        with pytest.raises(aot_mint.MintRefused) as err:
            _mint("tiny829sym", tmp_path / "sym")
        assert "dispatch-ambiguity" in str(err.value), str(err.value)
        assert "entry_ambiguous" in str(err.value)

        # …and the SAME declaration collapsed is admitted, because one entry
        # over the hull is unique by construction.
        reset_export_declarations()
        _declare("tiny829symc", regional_strategy=DYNAMIC_COLLAPSE,
                 rows=SYMMETRIC_ROWS)
        cell = _mint("tiny829symc", tmp_path / "symc")
    finally:
        mp.undo()
    assert sorted(cell["result"].metadata["entries"]) == [
        "unet/block=AttnBlock#0,cfg=false",
        "unet/block=AttnBlock#0,cfg=true",
    ]


def test_pgw831s_FOLDED_CONSTANT_gate_still_refuses_under_the_collapse(
    tmp_path,
) -> None:
    """A dynamic dim must not become a way around the pgw#827 refusal.

    pgw#831: inductor folds a ``Linear`` bias into the epilogue, and a
    REGIONAL artifact serving N instances then bakes the PROTOTYPE's folded
    bytes into all of them (measured: instance 0 exact, instance 1 wrong by
    0.53). The mint refuses that by name. Whether the token axis is symbolic
    changes nothing about it, and this test is what says so — if constant
    folding ever stops firing under a dynamic dim, this goes GREEN and the
    refusal has silently become unreachable for collapsed entries.
    """
    from _pytest.monkeypatch import MonkeyPatch

    from gen_worker import host_isa

    try:
        host_isa.impose()
    except Exception:  # pragma: no cover
        pass
    mp = MonkeyPatch()
    try:
        _fake_sm(mp)
        _declare("tiny829bias", regional_strategy=DYNAMIC_COLLAPSE)
        with pytest.raises(aot_mint.MintRefused) as err:
            _mint("tiny829bias", tmp_path / "bias",
                  block_cls=BiasedBlock)
    finally:
        mp.undo()
    assert "ff.bias" in str(err.value), str(err.value)


def test_a_NON_CONTIGUOUS_captured_feed_is_traced_CONTIGUOUS(cells) -> None:
    """The third defect this lane's A/B found, and its regression test.

    The toy shell does diffusers' own
    ``permute(0, 2, 3, 1).reshape(b, h*w, c)``, which hands the block a
    NON-contiguous view. A block feed is CAPTURED from a live forward
    (pgw#812 S5) rather than constructed, so whatever layout the shell
    happens to produce is what gets traced — and AOTInductor generates
    against that layout. Measured off-pod on this exact toy:

        traced non-contiguous, served with the pgw#791 realign   max|d| 0.1645
        traced non-contiguous, served with the realign DISABLED  max|d| 0.1690
        traced CONTIGUOUS,     served with the realign           max|d| 1.5e-08

    Nothing refused any of it: the ingress contract records shapes and
    dtypes, never strides, so a 16 % wrong answer was admitted silently.
    RED at base — before the mint made the traced feed contiguous, the parity
    matrix above fails on its first coordinate.

    Asserted through the capture itself, not through a private flag: the
    shell really does produce a non-contiguous feed (or this test proves
    nothing), and the mint really does export from a contiguous one.
    """
    unet = copy.deepcopy(cells["collapsed"]["pipe"]).unet
    groups = aot_regional.repeated_block_groups(unet)
    assert groups

    seen: Dict[str, Any] = {}
    handle = groups[0].prototype.register_forward_pre_hook(
        lambda _m, args: seen.setdefault("feed", args[0]))
    try:
        with torch.no_grad():
            unet(_sample(2, 8, 8))
    finally:
        handle.remove()
    assert not seen["feed"].is_contiguous(), (
        "the toy no longer reproduces the non-contiguous capture, so the "
        "regression it guards is untested — restore the permute/reshape")

    # And the mint's own feed transform makes it contiguous.
    assert aot_mint._contiguous_feed(seen["feed"]).is_contiguous()
    assert torch.equal(aot_mint._contiguous_feed(seen["feed"]), seen["feed"])


def test_an_axis_that_reaches_ONE_must_fork_not_collapse() -> None:
    """torch's 0/1 specialization is not overridable (ie#543): a derived
    symbol that could take 1 would be silently specialized and the artifact
    would serve one shape while advertising a range."""
    with pytest.raises(aot_mint.MintRefused, match="0/1 specialization"):
        aot_mint._regional_derived_dynamic(
            "unet/block=X#0", ("hidden_states",),
            [((1, 8),), ((4, 8),)])


def test_lockstep_carriers_share_ONE_symbol() -> None:
    """pgw#812 D1, one level down: two block arguments that move together
    across the declared rows are one logical axis, and exporting them as
    independent symbols is what strict export refuses outright. The block's
    arguments are unnamed, so lockstep over the observed rows is the only
    evidence there is."""
    rows = [((2, 64), (2, 64, 4)), ((2, 256), (2, 256, 4))]
    dims = aot_mint._regional_derived_dynamic(
        "unet/block=X#0", ("hidden_states", "ids"), rows)
    assert {(d.input_name, d.axis) for d in dims} == {
        ("hidden_states", 1), ("ids", 1)}
    assert len({d.dim for d in dims}) == 1, [d.as_row() for d in dims]
    assert {(d.min, d.max) for d in dims} == {(64, 256)}


def test_a_rank_change_across_rows_is_a_FORK_not_a_dim() -> None:
    with pytest.raises(aot_mint.MintRefused, match="changes RANK"):
        aot_mint._regional_derived_dynamic(
            "unet/block=X#0", ("hidden_states",), [((2, 64),), ((2, 64, 4),)])


def test_a_DECLARED_dim_may_not_split_a_derived_LOCKSTEP_group() -> None:
    """The merge's one refusal. Symbol identity is how
    :func:`dynamic_shapes_spec` shares carriers, so renaming one member of an
    observed lockstep group and not the others discards the observation — and
    strict export then refuses the declaration with a `Constraints violated`
    on a symbol nobody wrote (pgw#812 D1). Refused by name at the merge, where
    both facts are in hand."""
    derived = aot_mint._regional_derived_dynamic(
        "unet/block=X#0", ("hidden_states", "ids"),
        [((2, 64), (2, 64, 4)), ((2, 256), (2, 256, 4))])
    declared = (aot_mint.DynamicDim(
        input_name="hidden_states", axis=1, min=32, max=512,
        multiple_of=1, dim="T_img"),)

    with pytest.raises(aot_mint.MintRefused, match="LOCKSTEP"):
        aot_mint._merge_dynamic("unet/block=X#0", derived, declared)

    # A declared row that agrees with the derived SYMBOL still merges, and it
    # widens to the union — the case the docstring calls "wider on purpose".
    same = (aot_mint.DynamicDim(
        input_name="hidden_states", axis=1, min=32, max=512,
        multiple_of=1, dim=derived[0].dim),)
    merged = {(d.input_name, d.axis): d
              for d in aot_mint._merge_dynamic("unet/block=X#0", derived, same)}
    assert (merged[("hidden_states", 1)].min,
            merged[("hidden_states", 1)].max) == (32, 512)
    # …and a carrier that is NOT in a lockstep group is renameable, because
    # there is no group to split.
    solo = aot_mint._regional_derived_dynamic(
        "unet/block=X#0", ("hidden_states",), [((2, 64),), ((2, 256),)])
    renamed = aot_mint._merge_dynamic("unet/block=X#0", solo, declared)
    assert [d.dim for d in renamed] == ["T_img"]


# ---------------------------------------------------------------------------
# The arithmetic behind 72 -> 8: the per-entry fixed cost MULTIPLIES
# ---------------------------------------------------------------------------


def _tiny_program(seed: int) -> Any:
    class Tiny(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.a = torch.nn.Linear(96, 96)

        def forward(self, x: Any) -> Any:
            return torch.tanh(self.a(x)) * (1.0 + seed)

    return torch.export.export(Tiny(), (torch.randn(4, 96),))


_FIXED = ("child_boot_s", "child_seal_s", "child_program_load_s",
          "child_devlock_s", "reap_lag_s")


def _pool_fixed_phases(tmp_path: Path, n: int) -> List[Dict[str, float]]:
    """One REAL pooled compile of ``n`` entries; each entry's fixed-phase
    ledger, straight off pgw#830's."""
    entries = [(f"unet/dim={i}", _tiny_program(i)) for i in range(n)]
    width = aot_compile_pool.entry_workers(
        len(entries), limit=2, vcpus=16, available_bytes=64 * 1024**3,
        free_vram_bytes=0, device_lock=True)
    box = aot_compile_pool.EntryCompilePool(
        tmp_path / f"pool{n}", width=width,
        inductor_configs={"compile_threads": 2},
        cache_dir=str(tmp_path / f"cache{n}"))
    box.compile(entries)
    return [
        {k: float(box.entry_phases[name].get(k, 0.0)) for k in _FIXED}
        for name, _ in entries]


@pytest.mark.integration
def test_the_per_entry_fixed_cost_MULTIPLIES_with_the_entry_count(
    tmp_path,
) -> None:
    """pgw#830's finding, turned into the lever this issue pulls.

    The pool is already >= 95.5 % efficient, so there is no scheduling loss
    to recover. What 72 entries buy is 72 interpreter boots, 72 env seals, 72
    staged-program loads and 72 reaps — a CONSTANT, multiplied. This test
    asserts that EVERY entry pays EVERY fixed phase — which is what makes the
    cost multiply, and is what stops being true if it ever amortizes.

    If it ever stops scaling — a persistent pool worker, a forked child —
    this test is where that shows up, and pgw#829's arithmetic would need
    re-deriving rather than re-quoting.

    pgw#845: it used to prove this with a stopwatch — a `rel=1.5` band around a
    6-sample mean, and `1.4 <= total_4 / total_2 <= 2.8` across two real pooled
    compiles. On a shared runner under `-n 4` those seconds are the box's, not
    the pool's: CI measured 31.26s vs 23.05s, ratio **1.36**, and reported it as
    "the per-entry fixed cost does not exist". The LEDGER carries the same claim
    exactly and deterministically — a persistent worker or a forked child would
    make some entry skip its boot/seal/load/reap, and that is a missing phase,
    not a smaller number. Asserting it that way also halves what this test costs
    the box, because one entry count is now enough.
    """
    ledger = _pool_fixed_phases(tmp_path, 4)
    assert len(ledger) == 4

    for index, phases in enumerate(ledger):
        amortized = [k for k in _FIXED if phases.get(k, 0.0) <= 0.0]
        assert not amortized, (
            f"entry {index} paid nothing for {amortized} — the per-entry fixed "
            f"cost is being amortized across entries, and pgw#829's 72 -> 8 "
            f"arithmetic assumes it is not (full ledger: {ledger})")


@pytest.mark.integration
def test_the_COLLAPSE_removes_the_multiplied_fixed_cost(tmp_path) -> None:
    """The A/B's own entry counts, priced on a real pool ledger.

    The test above proves the fixed cost is a per-ENTRY constant. This one
    spends it at the two counts this issue actually produces —
    ``test_the_conv_free_block_collapses_EIGHT_entries_into_TWO`` mints 8
    per-shape entries and 2 collapsed ones — and asserts the multiplication
    is GONE, not merely smaller: the six entries that stopped existing took
    six whole copies of the constant with them.

    That is the entire claim, and it is why this is a ledger test rather
    than a stopwatch: the collapse changes no kernel and saves no compile
    work per entry. It removes ENTRIES, and pgw#830 measured that an entry
    costs a fixed 8.18 s on the pod before inductor does anything. sdxl
    reads the same way one axis up (72 -> 8), so 64 copies stop being paid.

    pgw#845: the docstring already said "ledger, not stopwatch" while the body
    asserted `2.8 <= total_8/total_2 <= 5.2` and a `rel=0.6` band on a
    subtraction of seconds. Seconds on a shared runner are the box's. The
    claim is a COUNT — one copy of the constant per entry, so removing six
    entries removes six copies — and a count is exact.
    """
    per_shape = _pool_fixed_phases(tmp_path, 8)
    collapsed = _pool_fixed_phases(tmp_path, 2)

    # Denominator first: the loops below say nothing about an empty ledger.
    assert (len(per_shape), len(collapsed)) == (8, 2)

    # One copy of the constant per entry, on both sides: an entry that skipped
    # a phase would mean the cost amortizes and pgw#829 buys nothing.
    for label, ledger in (("per-shape", per_shape), ("collapsed", collapsed)):
        for index, phases in enumerate(ledger):
            amortized = [k for k in _FIXED if phases.get(k, 0.0) <= 0.0]
            assert not amortized, (
                f"{label} entry {index} paid nothing for {amortized} — the "
                f"fixed cost is amortized across entries, so the collapse "
                f"removes nothing and pgw#829's arithmetic needs re-deriving "
                f"rather than re-quoting (full ledger: {ledger})")

    # ...therefore the collapse removes exactly the six copies it removes
    # entries. This is the whole payoff, and it is arithmetic, not a stopwatch.
    assert len(per_shape) - len(collapsed) == 6, (
        f"{len(per_shape)} per-shape entries collapsed to {len(collapsed)}: "
        f"the A/B this issue produces is 8 -> 2, and the saving IS the six "
        f"entries that stopped existing")
