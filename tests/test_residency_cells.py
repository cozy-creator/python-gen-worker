"""Coarse residency cells and the hold-while-hot schedule.

# pgw#1515: the packing-vs-scheduling split, phase 3's residency half.

Two claims are under test, and both are ARITHMETIC the plan publishes rather
than assertions about a past run:

1. **Packing eliminates the per-leaf granularity tax.** pgw#1507 measured it on
   the card: 1.830 GiB of sd1.5 weights held in 2.072 GiB of span across 239
   per-leaf regions — 248 MiB, **13.2 %**, forced by 2 MiB VMM granularity plus
   independent per-leaf back/unback. Every test below computes BOTH prices from
   the same leaf census, so the elimination is a subtraction and the baseline
   can be checked against the measured number.
2. **The schedule is decided in the plan.** A budget that admits holding the
   whole active component for a call gets ``CALL_BOUNDARY`` — the per-CALL
   amortization ``model_offload`` gets, on this rung's mechanism — and one that
   does not gets ``PER_STEP``. Deterministic from budget + cell layout, never a
   runtime reaction to pressure.

Real ``nn.Module`` trees throughout for anything touching the mechanism; the
planner tests feed it :class:`LeafCost`, which is its real input type, not a
mock of one. CPU only: the cell layer is arithmetic plus the same host/device
mover the CPU arm already exercises.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

# A real static import, not `nn = torch.nn` — the rebinding form leaves mypy no
# base type for `class X(nn.Module)`. Same idiom as `test_stream_residency.py`.
import torch.nn as nn  # noqa: E402

from gen_worker.models.stream_residency import (  # noqa: E402
    LEAF_CELLS,
    VMM_GRANULARITY_BYTES,
    CellPolicy,
    LeafCost,
    ResidencySchedule,
    StreamedResidency,
    module_roots,
    pack_cells,
    plan_residency,
)

MIB = 1 << 20


# ---------------------------------------------------------------------------
# Fixtures: a real two-component tree, and a leaf census shaped like the one
# pgw#1507 measured.
# ---------------------------------------------------------------------------


class Block(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(width, width * 2)
        self.fc2 = nn.Linear(width * 2, width)
        self.norm = nn.LayerNorm(width)

    def forward(self, x):  # type: ignore[no-untyped-def]
        return self.norm(x + self.fc2(torch.relu(self.fc1(x))))


class Tower(nn.Module):
    def __init__(self, width: int, depth: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([Block(width) for _ in range(depth)])

    def forward(self, x):  # type: ignore[no-untyped-def]
        for block in self.blocks:
            x = block(x)
        return x


class TwoComponent:
    """A pipeline-shaped holder: two independent module trees under one object.

    ``module_roots`` reads exactly this shape off the serve path (an author
    model object whose attributes are modules), so the components the schedule
    reasons about are the ones the real walk produces.
    """

    def __init__(self, width: int = 192, depth: int = 4) -> None:
        torch.manual_seed(1515)
        self.unet = Tower(width, depth).eval()
        self.text_encoder = Tower(width // 2, max(1, depth // 2)).eval()


def _census(n: int = 239, total_bytes: int = int(1.830 * (1 << 30))) -> list[LeafCost]:
    """A leaf census with pgw#1507's shape: 239 leaves, 1.830 GiB, remainders
    spread across the 2 MiB granularity so the per-leaf tax lands in the
    measured band. Deterministic — a fixed LCG, no RNG dependency."""
    sizes: list[int] = []
    state = 1515
    for _ in range(n):
        state = (state * 1103515245 + 12345) % (1 << 31)
        # fp16 tensors: even byte counts, sizes spread over two orders of
        # magnitude the way an attention tower's leaves are.
        sizes.append(2 * (1 + state % (4 * MIB)))
    scale = total_bytes / sum(sizes)
    return [
        LeafCost(f"unet.leaf{i:03d}", max(2, 2 * int(size * scale / 2)))
        for i, size in enumerate(sizes)
    ]


def _vmm(cell_bytes: int) -> CellPolicy:
    return CellPolicy.vmm(cell_bytes)


# ---------------------------------------------------------------------------
# 1. Packing
# ---------------------------------------------------------------------------


def test_packing_is_deterministic_under_input_order() -> None:
    """The cells must be a function of the SET of leaves, never the sequence
    they arrived in — a mint's graph specializations and a residency
    reservation both key on the answer."""
    costs = _census(64)
    first = pack_cells(costs, policy=_vmm(64 * MIB), min_stream_bytes=1)
    shuffled = pack_cells(
        list(reversed(costs)), policy=_vmm(64 * MIB), min_stream_bytes=1
    )
    assert first == shuffled
    assert len(first) < len(costs), "64 MiB cells must actually pack this census"


def test_packing_eliminates_the_per_leaf_granularity_tax() -> None:
    """THE claim. Both prices come out of the same census, so the baseline is
    checkable against pgw#1507's measured 13.2 % and the residual is a number,
    not a promise.

    The residual is NOT zero and is not claimed to be: a cell still pays one
    page remainder, so the tax falls as ``cells x granularity / 2`` — 244 MiB
    over 239 leaf regions becomes 32 MiB over 33 cells at 64 MiB and 6 MiB over
    8 cells at 256 MiB. What is eliminated is the PER-LEAF term."""
    costs = _census()
    weights = sum(c.resident_bytes for c in costs)

    leafwise = pack_cells(costs, policy=_vmm(0), min_stream_bytes=1)
    leaf_span = sum(c.span_bytes for c in leafwise)
    leaf_tax = (leaf_span - weights) / weights
    assert len(leafwise) == len(costs), "leaf-granular means one region per leaf"
    # The measured baseline this packing exists to beat: 13.2 % on 239 regions.
    assert 0.10 <= leaf_tax <= 0.16, f"census is off the measured shape: {leaf_tax:.3%}"

    for target, ceiling in ((64 * MIB, 0.02), (256 * MIB, 0.005)):
        packed = pack_cells(costs, policy=_vmm(target), min_stream_bytes=1)
        packed_span = sum(c.span_bytes for c in packed)
        packed_tax = (packed_span - weights) / weights
        assert packed_tax < ceiling, f"{target // MIB} MiB cells: {packed_tax:.3%}"
        assert packed_span < leaf_span
        # BOUNDED, not merely small: at most one page remainder per cell.
        assert packed_span - weights <= len(packed) * VMM_GRANULARITY_BYTES


def test_the_residual_tax_is_bounded_by_the_geometry_not_by_the_leaves() -> None:
    """The exact shape of the elimination, and the reason it is a DESIGN and not
    a heuristic: the leaf-granular tax is unbounded as leaves shrink (a 64 KiB
    leaf wastes 97 % of its 2 MiB page, and sd1.5's census is full of them),
    while the packed residual can never exceed ``granularity / cell_bytes`` —
    3.1 % at 64 MiB cells, 0.8 % at 256 MiB — whatever the leaves look like."""
    costs = [LeafCost(f"unet.s{i:04d}", 64 * 1024 + 2 * i) for i in range(4096)]
    weights = sum(c.resident_bytes for c in costs)

    leafwise = sum(
        c.span_bytes for c in pack_cells(costs, policy=_vmm(0), min_stream_bytes=1)
    )
    leaf_tax = (leafwise - weights) / weights
    assert leaf_tax > 20, "a 64 KiB leaf wastes a whole 2 MiB page"

    for target in (64 * MIB, 256 * MIB):
        packed = pack_cells(costs, policy=_vmm(target), min_stream_bytes=1)
        span = sum(c.span_bytes for c in packed)
        residual = (span - weights) / weights
        assert residual <= VMM_GRANULARITY_BYTES / target
        assert residual < leaf_tax / 500


def test_bigger_cells_monotonically_shrink_the_residual_tax() -> None:
    """The sizing knob has to be a real dial, not a switch — tomorrow's pricing
    sweep is {leaf, 64 MiB, 256 MiB, component} and a non-monotone tax would
    make that table unreadable."""
    costs = _census()
    weights = sum(c.resident_bytes for c in costs)
    taxes = [
        sum(c.span_bytes for c in pack_cells(costs, policy=_vmm(size), min_stream_bytes=1))
        - weights
        for size in (0, 16 * MIB, 64 * MIB, 256 * MIB, 1 << 30)
    ]
    assert taxes == sorted(taxes, reverse=True)
    assert taxes[0] > 100 * MIB and taxes[-1] < 4 * MIB


def test_small_leaves_are_packed_with_each_other_never_welded_to_a_big_one() -> None:
    """Small-FIRST is the mechanism, not a tie-break, and this is the outcome it
    buys that a descending fill does not.

    Filling ascending, a leaf at or above the cell target always lands alone:
    the batch before it flushes, and the next leaf is at least as large so it
    overflows immediately. Filling DESCENDING, the big leaf goes down first with
    room left over and the small leaves get appended to it — which welds a 4 KiB
    norm's residency to a 90 MiB block, so the norm can no longer be moved
    without moving 90 MiB. On this fixture that is the difference between the
    twenty small leaves being one 20 MiB unit and ten of them being hostages."""
    small = [LeafCost(f"unet.s{i:02d}", MIB) for i in range(20)]
    large = [LeafCost("unet.big", 90 * MIB)]
    cells = pack_cells(small + large, policy=_vmm(100 * MIB), min_stream_bytes=1)

    big_cells = [c for c in cells if "unet.big" in c.members]
    assert [c.members for c in big_cells] == [("unet.big",)], (
        "a leaf near the cell target must land alone, not collect small leaves"
    )
    packed = [c for c in cells if "unet.big" not in c.members]
    assert sum(len(c.members) for c in packed) == 20
    assert max(len(c.members) for c in packed) == 20, "and they pack together"


def test_a_cell_never_mixes_forced_and_streamable_leaves() -> None:
    """A cell has ONE fate. A forced member would pin the whole cell onto the
    card, which is how a packing layer silently un-does a budget."""
    costs = [LeafCost(f"unet.l{i}", (i + 1) * MIB) for i in range(8)]
    cells = pack_cells(
        costs, policy=_vmm(64 * MIB), min_stream_bytes=1, exclude=("unet.l7",)
    )
    for cell in cells:
        assert cell.forced == ("unet.l7" in cell.members) or "unet.l7" not in cell.members
    forced_cells = [c for c in cells if c.forced]
    assert [c.members for c in forced_cells] == [("unet.l7",)]


def test_a_cell_never_straddles_two_components() -> None:
    """The call-boundary schedule swaps a COMPONENT; a cell spanning two of
    them could not be moved at a call boundary at all."""
    costs = [LeafCost(f"unet.l{i}", MIB) for i in range(4)] + [
        LeafCost(f"vae.l{i}", MIB) for i in range(4)
    ]
    cells = pack_cells(costs, policy=_vmm(64 * MIB), min_stream_bytes=1)
    for cell in cells:
        assert len({m.split(".", 1)[0] for m in cell.members}) == 1
        assert all(m.startswith(cell.component + ".") for m in cell.members)


# ---------------------------------------------------------------------------
# 2. The plan reports the tax, and the fill admits whole cells
# ---------------------------------------------------------------------------


def test_the_plan_reports_the_tax_it_did_not_pay() -> None:
    costs = _census()
    budget = int(1.2 * (1 << 30))
    leafwise = plan_residency(
        costs, budget_bytes=budget, min_stream_bytes=1, cells=_vmm(0)
    )
    packed = plan_residency(
        costs, budget_bytes=budget, min_stream_bytes=1, cells=_vmm(64 * MIB)
    )
    assert leafwise.leaf_granular_tax_ratio == pytest.approx(
        leafwise.granularity_tax_ratio
    ), "leaf-granular IS the baseline, so it can eliminate nothing"
    assert leafwise.tax_eliminated_bytes == 0

    # Measured by this plan, on this census: 8.0 % leaf-granular -> 1.6 % packed.
    assert leafwise.granularity_tax_ratio > 0.05
    assert packed.granularity_tax_ratio < 0.02
    assert packed.leaf_granular_tax_ratio > 0.05
    assert packed.granularity_tax_ratio < packed.leaf_granular_tax_ratio / 5
    assert packed.tax_eliminated_bytes > 0


def test_coarser_cells_buy_tax_and_pay_window_and_the_plan_prices_both() -> None:
    """The counter-force, and it is NOT small — banked here because the GPU
    sweep has to price it and a planner that only reported the win would hide
    the reason a bigger cell can be worse.

    A cell is the stream unit, so under PER_STEP the in-flight reservation is
    ``streams x the largest streamed CELL``: at a 1.2 GiB budget on this census
    the window goes 20 MiB (leaf) -> 115 MiB (64 MiB cells) -> 508 MiB (256 MiB
    cells), and it comes straight out of the resident set. So the tax falls
    monotonically while the WEIGHT HELD peaks at a moderate cell size — 1118 /
    1112 / 1086 / 630 MiB across those four geometries. Under CALL_BOUNDARY
    nothing streams mid-forward and the elimination is pure gain; under PER_STEP
    it is a trade, and tomorrow's {leaf, 64 MiB, 256 MiB, component} x budget
    table is the measurement of exactly this curve."""
    costs = _census()
    budget = int(1.2 * (1 << 30))
    plans = [
        plan_residency(costs, budget_bytes=budget, min_stream_bytes=1, cells=_vmm(size))
        for size in (0, 16 * MIB, 64 * MIB, 256 * MIB)
    ]
    windows = [p.window_bytes for p in plans]
    taxes = [p.granularity_tax_ratio for p in plans]
    assert windows == sorted(windows), "a coarser cell reserves a bigger window"
    assert taxes[-1] < taxes[0] / 10, "and buys the tax back"
    assert plans[-1].resident_bytes < plans[0].resident_bytes, (
        "at 256 MiB the window costs more than the tax it saves"
    )


def test_the_fill_charges_a_cell_its_mapped_span_not_its_weight() -> None:
    """``fits`` has to stay exact under a mechanism that pages — pgw#1507's
    numbers matched the driver's mapped bytes precisely because regions were
    priced at their aligned span."""
    costs = [LeafCost(f"unet.l{i}", 3 * MIB + 1024) for i in range(8)]
    plan = plan_residency(
        costs, budget_bytes=20 * MIB, min_stream_bytes=1, cells=_vmm(0)
    )
    held = [c for c in plan.cells if c.index in plan.resident_cells]
    assert plan.resident_span_bytes == sum(c.span_bytes for c in held)
    assert plan.resident_span_bytes > plan.resident_bytes, "3 MiB + 1 KiB maps 4 MiB"
    assert plan.device_bytes == plan.resident_span_bytes + plan.window_bytes
    assert plan.fits and plan.device_bytes <= plan.budget_bytes


def test_a_cells_members_share_one_fate() -> None:
    costs = _census(80)
    plan = plan_residency(
        costs,
        budget_bytes=int(0.4 * (1 << 30)),
        min_stream_bytes=1,
        cells=_vmm(64 * MIB),
    )
    assert plan.resident and plan.streamed, "this budget must genuinely split"
    resident, streamed = set(plan.all_resident), set(plan.streamed)
    for cell in plan.cells:
        members = set(cell.members)
        assert members <= resident or members <= streamed


def test_the_window_reserves_the_largest_streamed_CELL() -> None:
    """A cell is what moves, so the in-flight reservation grows with the cell
    size. Coarser cells are not free and the plan says so."""
    costs = [LeafCost(f"unet.l{i}", 4 * MIB) for i in range(16)]
    plan = plan_residency(
        costs, budget_bytes=40 * MIB, streams=2, min_stream_bytes=1, cells=_vmm(16 * MIB)
    )
    assert plan.streamed_cells
    largest = max(plan.cells[i].cast_bytes for i in plan.streamed_cells)
    assert plan.window_bytes == 2 * largest
    assert plan.device_bytes <= plan.budget_bytes


def test_the_default_policy_is_the_pre_cell_planner_exactly() -> None:
    """The reduction that makes this a safe layer to add: state no geometry and
    every number is the one pgw#1497 shipped."""
    costs = [LeafCost(f"unet.l{i}", (i + 1) * 1_000_000) for i in range(6)]
    plan = plan_residency(costs, budget_bytes=14_000_000, min_stream_bytes=1)
    assert plan.policy == LEAF_CELLS
    assert len(plan.cells) == len(costs)
    assert all(len(c.members) == 1 for c in plan.cells)
    assert plan.resident_span_bytes == plan.resident_bytes
    assert plan.granularity_tax_bytes == 0 and plan.leaf_granular_tax_bytes == 0
    assert plan.device_bytes == plan.resident_bytes + plan.window_bytes


def test_the_vram_ram_pair_shape_survives_the_cell_layer() -> None:
    """pgw#1497's pair is the plan's shape and cells do not narrow it."""
    from gen_worker.models.stream_residency import MemoryBudget

    costs = _census(64)
    tail = 200 * MIB
    plan = plan_residency(
        costs,
        budget_bytes=MemoryBudget(vram_bytes=300 * MIB, ram_bytes=tail),
        min_stream_bytes=1,
        cells=_vmm(64 * MIB),
    )
    assert plan.ram_budget_bytes == tail
    assert plan.host_bytes == plan.streamed_bytes
    assert plan.host_fits is (plan.host_bytes <= tail)
    unstated = plan_residency(
        costs, budget_bytes=300 * MIB, min_stream_bytes=1, cells=_vmm(64 * MIB)
    )
    assert unstated.ram_budget_bytes == 0 and unstated.host_fits


# ---------------------------------------------------------------------------
# 3. The schedule
# ---------------------------------------------------------------------------


def _two_components(unet_mb: int = 400, text_mb: int = 100) -> list[LeafCost]:
    return [LeafCost(f"unet.l{i}", 4 * MIB) for i in range(unet_mb // 4)] + [
        LeafCost(f"text_encoder.l{i}", 4 * MIB) for i in range(text_mb // 4)
    ]


def test_a_budget_that_holds_the_hot_component_schedules_at_call_boundaries() -> None:
    costs = _two_components()
    plan = plan_residency(
        costs, budget_bytes=600 * MIB, streams=2, min_stream_bytes=1, cells=_vmm(64 * MIB)
    )
    assert plan.schedule is ResidencySchedule.CALL_BOUNDARY
    assert plan.hold_while_hot == ("unet", "text_encoder")
    assert plan.hot_component_bytes + plan.streams * max(
        c.cast_bytes for c in plan.cells if not c.forced
    ) <= plan.budget_bytes


def test_a_budget_below_the_hot_component_falls_to_per_step() -> None:
    costs = _two_components()
    plan = plan_residency(
        costs, budget_bytes=200 * MIB, streams=2, min_stream_bytes=1, cells=_vmm(64 * MIB)
    )
    assert plan.schedule is ResidencySchedule.PER_STEP
    assert plan.hold_while_hot == ()
    assert plan.hot_component_bytes > plan.budget_bytes


def test_the_schedule_flips_exactly_where_the_arithmetic_says() -> None:
    """Deterministic from budget + layout: one byte on either side of the
    stated inequality, and nothing else moves."""
    costs = _two_components()
    probe = plan_residency(
        costs, budget_bytes=1 << 40, streams=2, min_stream_bytes=1, cells=_vmm(64 * MIB)
    )
    threshold = probe.hot_component_bytes + probe.streams * max(
        c.cast_bytes for c in probe.cells if not c.forced
    )
    at = plan_residency(
        costs, budget_bytes=threshold, streams=2, min_stream_bytes=1, cells=_vmm(64 * MIB)
    )
    below = plan_residency(
        costs, budget_bytes=threshold - 1, streams=2, min_stream_bytes=1,
        cells=_vmm(64 * MIB),
    )
    assert at.schedule is ResidencySchedule.CALL_BOUNDARY
    assert below.schedule is ResidencySchedule.PER_STEP


def test_a_stated_ram_half_can_force_per_step_on_a_card_that_would_admit_it() -> None:
    """The pair is one decision, not two. VRAM admitting the hot component is
    worthless if the RAM half cannot hold what parks while it is hot."""
    from gen_worker.models.stream_residency import MemoryBudget

    costs = _two_components()
    roomy = plan_residency(
        costs,
        budget_bytes=MemoryBudget(600 * MIB, 1 << 30),
        streams=2,
        min_stream_bytes=1,
        cells=_vmm(64 * MIB),
    )
    assert roomy.schedule is ResidencySchedule.CALL_BOUNDARY

    starved = plan_residency(
        costs,
        budget_bytes=MemoryBudget(600 * MIB, 8 * MIB),
        streams=2,
        min_stream_bytes=1,
        cells=_vmm(64 * MIB),
    )
    assert starved.cold_component_bytes > 8 * MIB
    assert starved.schedule is ResidencySchedule.PER_STEP


def test_the_schedule_is_deterministic_and_component_spans_are_published() -> None:
    costs = _two_components()
    first = plan_residency(
        costs, budget_bytes=600 * MIB, min_stream_bytes=1, cells=_vmm(64 * MIB)
    )
    again = plan_residency(
        list(reversed(costs)), budget_bytes=600 * MIB, min_stream_bytes=1,
        cells=_vmm(64 * MIB),
    )
    assert first == again
    assert [name for name, _ in first.component_spans] == ["unet", "text_encoder"]
    assert first.hot_component_bytes == first.component_spans[0][1]
    assert first.cold_component_bytes == sum(v for _, v in first.component_spans[1:])


# ---------------------------------------------------------------------------
# 4. Holding a component hot, over a real tree
# ---------------------------------------------------------------------------


def test_holding_a_component_hot_parks_every_other_component(
) -> None:
    model = TwoComponent()
    roots = module_roots(model)
    assert {name for name, _ in roots} == {"unet", "text_encoder"}
    residency = StreamedResidency(
        roots, device="cpu", budget_bytes=1 << 40, min_stream_bytes=1,
        cells=CellPolicy.vmm(4 * MIB),
    )
    plan = residency.engage()
    assert plan.schedule is ResidencySchedule.CALL_BOUNDARY

    hot = residency.hold_component("unet")
    assert hot.hot_component == "unet"
    assert all(n.startswith("unet.") for n in hot.resident)
    assert all(n.startswith("text_encoder.") for n in hot.streamed)
    assert hot.streamed, "the cold component must actually park"

    cold = residency.hold_component("text_encoder")
    assert all(n.startswith("text_encoder.") for n in cold.resident)
    assert all(n.startswith("unet.") for n in cold.streamed)


def test_a_held_component_computes_the_same_answer_as_a_resident_one() -> None:
    """The swap is a placement, never a numeric change — the property the whole
    rung rests on, re-checked across a call-boundary rotation."""
    model = TwoComponent()
    x = torch.randn(2, 192)
    y = torch.randn(2, 96)
    with torch.no_grad():
        want_unet = model.unet(x).clone()
        want_text = model.text_encoder(y).clone()

    residency = StreamedResidency(
        module_roots(model), device="cpu", budget_bytes=1 << 40, min_stream_bytes=1,
        cells=CellPolicy.vmm(4 * MIB),
    )
    residency.engage()
    for _ in range(2):
        residency.hold_component("unet")
        with torch.no_grad():
            assert torch.equal(model.unet(x), want_unet)
            assert torch.equal(model.text_encoder(y), want_text)
        residency.hold_component("text_encoder")
        with torch.no_grad():
            assert torch.equal(model.unet(x), want_unet)
            assert torch.equal(model.text_encoder(y), want_text)
    residency.release()
    with torch.no_grad():
        assert torch.equal(model.unet(x), want_unet)


def test_hold_component_refuses_when_the_budget_cannot_admit_it() -> None:
    """A refusal, never a silent overshoot: the plan already knows the answer
    before a byte moves, which is the whole reason the schedule is planned."""
    model = TwoComponent()
    residency = StreamedResidency(
        module_roots(model), device="cpu", budget_bytes=64 * 1024, min_stream_bytes=1,
        cells=CellPolicy.vmm(4 * MIB),
    )
    plan = residency.engage()
    assert plan.schedule is ResidencySchedule.PER_STEP
    with pytest.raises(ValueError, match="admits no whole component"):
        residency.hold_component("unet")


def test_hold_component_refuses_a_component_this_tree_does_not_have() -> None:
    model = TwoComponent()
    residency = StreamedResidency(
        module_roots(model), device="cpu", budget_bytes=1 << 40, min_stream_bytes=1,
        cells=CellPolicy.vmm(4 * MIB),
    )
    residency.engage()
    with pytest.raises(ValueError, match="no such component"):
        residency.hold_component("vae")


def test_the_forced_core_stays_resident_through_a_call_boundary_swap() -> None:
    """An exclusion is a statement about RESIDENCY (pgw#1497 defect 2), and a
    call boundary does not repeal it — LoRA adapters live in this set."""
    model = TwoComponent()
    roots = module_roots(model)
    residency = StreamedResidency(
        roots,
        device="cpu",
        budget_bytes=1 << 40,
        min_stream_bytes=1,
        exclude=("text_encoder.blocks.0.fc1",),
        cells=CellPolicy.vmm(4 * MIB),
    )
    residency.engage()
    hot = residency.hold_component("unet")
    assert "text_encoder.blocks.0.fc1" in hot.forced
    assert "text_encoder.blocks.0.fc1" not in hot.streamed
    # And it is really still where it was: a parked leaf would be sitting in
    # pinned host memory instead.
    params = dict(model.text_encoder.named_parameters())
    assert not params["blocks.0.fc1.weight"].is_pinned()


def test_a_real_tree_reports_its_own_tax_elimination() -> None:
    """End to end on a real ``nn.Module`` census: the same tree, two geometries,
    and the plan's own numbers show what the packing bought."""
    # 512-wide, 8 deep: ~33 MB of real parameters, enough that a 2 MiB page is
    # a rounding detail rather than the whole model.
    model = TwoComponent(width=512, depth=8)
    roots = module_roots(model)
    leafwise = StreamedResidency(
        roots, device="cpu", budget_bytes=1 << 40, min_stream_bytes=1,
        cells=CellPolicy.vmm(0),
    )
    packed = StreamedResidency(
        roots, device="cpu", budget_bytes=1 << 40, min_stream_bytes=1,
        cells=CellPolicy.vmm(64 * MIB),
    )
    a = leafwise.engage()
    b = packed.engage()
    assert a.resident_bytes == b.resident_bytes, "same weights, different geometry"
    assert a.granularity_tax_ratio > 0.5, "small leaves in 2 MiB regions are brutal"
    # The claim is the SHAPE: a per-CELL remainder where there was a per-LEAF
    # one. On a 33 MB tree two cells' remainders are still ~10 % of it, which is
    # the geometry being honest about a model smaller than a few pages.
    held = [c for c in b.cells if c.index in b.resident_cells]
    assert b.granularity_tax_bytes <= len(held) * VMM_GRANULARITY_BYTES
    assert b.granularity_tax_ratio < a.granularity_tax_ratio / 5
    assert b.tax_eliminated_bytes > 0
    packed.release()
    leafwise.release()


def test_the_forced_core_does_not_inflate_the_call_boundary_reservation() -> None:
    """Packing makes this matter. A tree with hundreds of sub-floor leaves has
    ONE large forced cell, and that cell is resident in every arrangement — it
    never travels. Reserving the swap ring for it would deny the call-boundary
    schedule to budgets that plainly admit it."""
    costs = [LeafCost(f"unet.tiny{i:03d}", 128 * 1024) for i in range(512)] + [
        LeafCost(f"unet.l{i}", 2 * MIB) for i in range(3)
    ]
    # The floor forces every 128 KiB leaf resident; they pack into ONE 64 MiB
    # cell, eight times larger than the only cell that can actually move.
    plan = plan_residency(
        costs,
        budget_bytes=100 * MIB,
        streams=2,
        min_stream_bytes=1 * MIB,
        cells=_vmm(64 * MIB),
    )
    forced_cells = [c for c in plan.cells if c.forced]
    movable = [c for c in plan.cells if not c.forced]
    assert max(c.cast_bytes for c in forced_cells) > 8 * max(
        c.cast_bytes for c in movable
    ), "the packed forced core must dwarf every cell that can move"
    # RED ARM: reserving the ring for the forced core would put this over budget.
    assert plan.hot_component_bytes + plan.streams * max(
        c.cast_bytes for c in plan.cells
    ) > plan.budget_bytes
    assert plan.schedule is ResidencySchedule.CALL_BOUNDARY
