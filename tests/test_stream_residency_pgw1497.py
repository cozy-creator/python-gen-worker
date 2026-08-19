"""pgw#1497 — per-module budgeted partial residency + streamed weight cast.

Every test here runs the REAL mechanism over a REAL ``nn.Module`` tree: the
planner's arithmetic, the forward hooks, the cast ring and the partial
unload/load transitions. Nothing is mocked, because the defects this rung can
have (a weight left bound to a reused cast buffer, a plan that fits at rest
and overshoots in flight, an unload that promotes) are all defects of the
interaction, not of a unit.

The CPU arm is not a simulation of the CUDA one: it is the same driver loop
with a null stream and synchronous copies, so every ordering decision is
exercised here and only the OVERLAP is left for the card to prove.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from gen_worker.models.stream_residency import (  # noqa: E402
    DEFAULT_STREAMS,
    ENGAGED_ATTR,
    LeafCost,
    ResidencyPlan,
    StreamedResidency,
    module_roots,
    plan_residency,
    plan_transition,
    stream_residency_active,
)


# ---------------------------------------------------------------------------
# Trees
# ---------------------------------------------------------------------------


class Block(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(width, width * 2)
        self.fc2 = nn.Linear(width * 2, width)
        self.norm = nn.LayerNorm(width)

    def forward(self, x):  # type: ignore[no-untyped-def]
        return self.norm(x + self.fc2(torch.relu(self.fc1(x))))


class Stack(nn.Module):
    """Eight blocks of decreasing width — a tree where a budget genuinely
    splits, instead of one that is all-or-nothing."""

    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([Block(256 - 16 * i) for i in range(8)])
        self.proj = nn.Linear(256, 256)

    def forward(self, x):  # type: ignore[no-untyped-def]
        out = self.proj(x)
        for block in self.blocks:
            width = block.norm.normalized_shape[0]
            out = torch.cat([block(out[..., :width]), out[..., width:]], dim=-1)
        return out


@pytest.fixture()
def tree() -> nn.Module:
    torch.manual_seed(1497)
    model = Stack().eval()
    return model


def _pyramid(n: int = 6, unit: int = 1_000_000) -> list[LeafCost]:
    return [LeafCost(f"m{i}", (i + 1) * unit) for i in range(n)]


# ---------------------------------------------------------------------------
# 1. The budget split
# ---------------------------------------------------------------------------


def test_budget_split_is_deterministic_and_largest_first() -> None:
    costs = _pyramid()
    first = plan_residency(costs, budget_bytes=14_000_000, min_stream_bytes=1)
    # Same inputs in a different order must give the same answer: a mint's
    # traced graph class and a residency reservation both depend on it.
    shuffled = plan_residency(
        list(reversed(costs)), budget_bytes=14_000_000, min_stream_bytes=1
    )
    assert first == shuffled
    assert first.resident, "a 14 MB budget on a 21 MB model must hold something"
    resident_sizes = [int(n[1:]) for n in first.resident]
    assert resident_sizes == sorted(resident_sizes, reverse=True)


def test_the_in_flight_window_is_reserved_out_of_the_budget() -> None:
    """The subtle half of the port. A plan that ignores the window fits at
    rest and overshoots the instant two casts are in flight."""
    costs = _pyramid()
    plan = plan_residency(
        costs, budget_bytes=14_000_000, streams=2, min_stream_bytes=1
    )
    assert plan.streamed, "this budget must leave a tail"
    largest_streamed = max(int(n[1:]) + 1 for n in plan.streamed) * 1_000_000
    assert plan.window_bytes == 2 * largest_streamed
    assert plan.device_bytes <= plan.budget_bytes
    assert plan.fits

    # RED ARM: had the window not been reserved, the same fill would have put
    # this much on the card and the tail's casts would have had nowhere to go.
    unreserved = sum(
        c.resident_bytes
        for c in sorted(costs, key=lambda c: -c.resident_bytes)
        if _fits_ignoring_window(c, costs, 14_000_000)
    )
    assert unreserved + plan.window_bytes > plan.budget_bytes


def _fits_ignoring_window(cost: LeafCost, costs: list[LeafCost], budget: int) -> bool:
    mem = 0
    for c in sorted(costs, key=lambda c: -c.resident_bytes):
        if mem + c.resident_bytes <= budget:
            mem += c.resident_bytes
            if c is cost:
                return True
    return False


def test_a_budget_at_the_model_size_means_full_residency() -> None:
    """The window must terminate at zero when nothing streams. ComfyUI's
    lookahead formula reserves for a tail that isn't there and reports an
    empty resident set on a budget that fits the whole model."""
    costs = _pyramid()
    total = sum(c.resident_bytes for c in costs)
    plan = plan_residency(costs, budget_bytes=total, min_stream_bytes=1)
    assert not plan.streamed
    assert plan.window_bytes == 0
    assert plan.resident_bytes == total


def test_a_budget_below_one_window_reports_that_it_does_not_fit() -> None:
    plan = plan_residency(_pyramid(), budget_bytes=1_000_000, min_stream_bytes=1)
    assert not plan.resident
    assert not plan.fits, "an unservable budget must confess, not clamp"


def test_leaves_below_the_floor_are_forced_resident() -> None:
    costs = [LeafCost("big", 8_000_000), LeafCost("tiny", 4_096)]
    plan = plan_residency(costs, budget_bytes=0, min_stream_bytes=1 << 20)
    assert plan.forced == ("tiny",)
    assert plan.streamed == ("big",)


def test_excluded_leaves_are_never_streamed() -> None:
    costs = _pyramid()
    plan = plan_residency(
        costs, budget_bytes=0, min_stream_bytes=1, exclude=("m5", "m0")
    )
    assert set(plan.forced) == {"m5", "m0"}
    assert "m5" not in plan.streamed and "m0" not in plan.streamed


def test_an_unload_never_promotes() -> None:
    """The greedy fill is NOT monotone in the budget — dropping a large leaf
    frees room a smaller one can take — so a re-plan at a lower budget can
    genuinely want to move bytes ONTO the card. A call asked to free bytes
    must never answer by claiming some.

    These three sizes are a measured instance of that: at a 9 MB budget the
    5 MB leaf is resident, and at 8 MB it is the 3 MB leaf instead."""
    mib = 1_000_000
    costs = [LeafCost("a", 5 * mib), LeafCost("b", 4 * mib), LeafCost("c", 3 * mib)]
    wide = plan_residency(costs, budget_bytes=9 * mib, streams=1, min_stream_bytes=1)
    narrow = plan_residency(costs, budget_bytes=8 * mib, streams=1, min_stream_bytes=1)
    assert wide.resident == ("a",) and narrow.resident == ("c",)

    # The instrument's own red arm: with promotion allowed this transition
    # DOES promote, so a test that could not see it would be measuring nothing.
    assert plan_transition(wide, narrow, costs, allow_promote=True).promote == ("c",)

    move = plan_transition(wide, narrow, costs, allow_promote=False)
    assert move.promote == ()
    assert move.demote == ("a",)
    assert move.freed_bytes == 5 * mib


# ---------------------------------------------------------------------------
# 2. The streaming tail over a real tree
# ---------------------------------------------------------------------------


def test_every_residency_state_computes_the_same_answer(tree: nn.Module) -> None:
    """Resident, fully streamed, and partially streamed must be numerically
    indistinguishable. This is the test the whole rung rests on."""
    x = torch.randn(2, 256)
    with torch.no_grad():
        want = tree(x).clone()

    residency = StreamedResidency(
        module_roots(tree), device="cpu", budget_bytes=0, min_stream_bytes=1
    )
    plan = residency.engage()
    assert plan.streamed and not plan.resident
    with torch.no_grad():
        assert torch.equal(tree(x), want), "fully streamed diverged"

    residency.rebudget(residency.total_bytes // 2)
    assert residency.plan is not None
    assert residency.plan.resident and residency.plan.streamed, "wanted a split"
    with torch.no_grad():
        assert torch.equal(tree(x), want), "partially streamed diverged"

    residency.promote_to_device()
    assert residency.plan is not None and not residency.plan.streamed
    with torch.no_grad():
        assert torch.equal(tree(x), want), "promoted diverged"

    residency.release()
    with torch.no_grad():
        assert torch.equal(tree(x), want), "released diverged"


def test_a_streamed_weight_is_unbound_from_the_cast_buffer_after_its_forward(
    tree: nn.Module,
) -> None:
    """The single most dangerous defect this rung can have: a weight left
    pointing into a buffer the ring is about to refill for another leaf. The
    bytes would be silently wrong, load-order dependent, and only sometimes."""
    residency = StreamedResidency(
        module_roots(tree), device="cpu", budget_bytes=0, min_stream_bytes=1
    )
    residency.engage()
    with torch.no_grad():
        tree(torch.randn(2, 256))
    for name, leaf in residency._streamed.items():
        for slot in leaf.slots:
            live = getattr(leaf.module, slot.attr)
            assert live.data_ptr() == slot.host.data_ptr(), (
                f"{name}.{slot.attr} is still bound to the cast buffer after "
                f"its forward returned"
            )


def test_the_cast_buffer_costs_the_window_not_the_tail(tree: nn.Module) -> None:
    """The memory argument, measured: the ring holds ``streams`` buffers sized
    to the largest streamed leaf, never the tail's own size — and the planner's
    reservation is exactly that number, not an estimate of it."""
    residency = StreamedResidency(
        module_roots(tree),
        device="cpu",
        budget_bytes=0,
        streams=DEFAULT_STREAMS,
        min_stream_bytes=1,
    )
    plan = residency.engage()
    with torch.no_grad():
        tree(torch.randn(2, 256))
    largest = max(c.cast_bytes for c in residency.costs)
    assert residency.buffer_peak_bytes == largest
    assert plan.window_bytes == DEFAULT_STREAMS * largest
    assert plan.window_bytes < residency.total_bytes


def test_partial_unload_trims_the_cold_tail_instead_of_dropping_the_model(
    tree: nn.Module,
) -> None:
    """The eviction primitive: freeing N bytes leaves the model SERVING."""
    x = torch.randn(2, 256)
    with torch.no_grad():
        want = tree(x).clone()
    residency = StreamedResidency(
        module_roots(tree),
        device="cpu",
        budget_bytes=1 << 30,
        min_stream_bytes=1,
    )
    plan = residency.engage()
    assert not plan.streamed, "a huge budget must start fully resident"

    freed = residency.partial_unload(residency.total_bytes // 3)
    assert freed >= residency.total_bytes // 3
    assert residency.plan is not None
    assert residency.plan.resident, "an unload must not drop the whole model"
    assert residency.plan.streamed
    with torch.no_grad():
        assert torch.equal(tree(x), want)

    # Smallest-first is the point: the tail that leaves is the CHEAP end, so
    # the expensive leaves keep their residency.
    by_name = {c.name: c for c in residency.costs}
    smallest_resident = min(
        by_name[n].resident_bytes for n in residency.plan.all_resident
    )
    largest_streamed = max(by_name[n].resident_bytes for n in residency.plan.streamed)
    assert largest_streamed >= smallest_resident


def test_partial_unload_and_partial_load_round_trip(tree: nn.Module) -> None:
    residency = StreamedResidency(
        module_roots(tree), device="cpu", budget_bytes=1 << 30, min_stream_bytes=1
    )
    residency.engage()
    before = residency.plan.resident_bytes if residency.plan else 0
    freed = residency.partial_unload(residency.total_bytes // 2)
    claimed = residency.partial_load(freed)
    assert residency.plan is not None
    assert residency.plan.resident_bytes == before
    assert claimed == freed


def test_the_host_tier_is_this_rung_at_its_end_stops(tree: nn.Module) -> None:
    x = torch.randn(2, 256)
    with torch.no_grad():
        want = tree(x).clone()
    residency = StreamedResidency(
        module_roots(tree), device="cpu", budget_bytes=1 << 30, min_stream_bytes=1
    )
    residency.engage()
    residency.demote_to_host()
    assert residency.plan is not None and not residency.plan.resident
    assert stream_residency_active(tree)
    residency.promote_to_device()
    assert residency.plan is not None and not residency.plan.streamed
    assert not getattr(tree, ENGAGED_ATTR, False)
    with torch.no_grad():
        assert torch.equal(tree(x), want)


def test_release_removes_every_hook(tree: nn.Module) -> None:
    residency = StreamedResidency(
        module_roots(tree), device="cpu", budget_bytes=0, min_stream_bytes=1
    )
    residency.engage()
    assert any(m._forward_pre_hooks for m in tree.modules())
    residency.release()
    for module in tree.modules():
        assert not module._forward_pre_hooks
        assert not module._forward_hooks


def test_a_module_owning_both_children_and_weights_is_never_hooked() -> None:
    """Its post-hook would fire AFTER its children's forwards, holding a
    cast-buffer view alive while the ring hands the same buffer to a child."""

    class Mixed(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.scale = nn.Parameter(torch.ones(1024, 256))
            self.child = nn.Linear(256, 256)

        def forward(self, x):  # type: ignore[no-untyped-def]
            return self.child(x) * self.scale.mean()

    model = Mixed().eval()
    residency = StreamedResidency(
        module_roots(model), device="cpu", budget_bytes=0, min_stream_bytes=1
    )
    names = {c.name for c in residency.costs}
    assert any(n.endswith("child") for n in names)
    assert not any(n == "Mixed" for n in names), (
        "a module with children AND its own weights was offered to the ring"
    )


def test_attached_lora_leaves_are_forced_resident() -> None:
    """Our LoRA is attach-based, so an adapter is a pair of tiny leaves next
    to the base layer, never a patch fused into the base weight. They stay
    resident and the base layer streams unchanged — which is why no
    LowVramPatch equivalent is needed (see the module docstring)."""

    class Attached(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.base_layer = nn.Linear(512, 512)
            self.lora_A = nn.Linear(512, 8, bias=False)
            self.lora_B = nn.Linear(8, 512, bias=False)

        def forward(self, x):  # type: ignore[no-untyped-def]
            return self.base_layer(x) + self.lora_B(self.lora_A(x))

    model = Attached().eval()
    x = torch.randn(2, 512)
    with torch.no_grad():
        want = model(x).clone()
    residency = StreamedResidency(
        module_roots(model), device="cpu", budget_bytes=0, min_stream_bytes=1
    )
    plan = residency.engage()
    assert any(n.endswith("lora_A") for n in plan.forced)
    assert any(n.endswith("lora_B") for n in plan.forced)
    assert any(n.endswith("base_layer") for n in plan.streamed)
    with torch.no_grad():
        assert torch.equal(model(x), want)


# ---------------------------------------------------------------------------
# 3. The serve-loop host tier
# ---------------------------------------------------------------------------


def test_the_serve_loop_backend_tiers_a_real_author_model() -> None:
    """``demote_to_host``/``promote_to_device`` raised NotImplementedError
    until this issue; the manager was run with a zero host budget so eviction
    was always a drop. Drive the REAL backend arms over a real tree."""
    from gen_worker.serving.serve_loop import _InstanceBackend

    class AuthorModel:
        def __init__(self) -> None:
            torch.manual_seed(7)
            self.pipe = Stack().eval()

    backend = _InstanceBackend.__new__(_InstanceBackend)
    backend.model_cls = AuthorModel
    backend.model = AuthorModel()
    backend._stream_residency = None

    x = torch.randn(2, 256)
    with torch.no_grad():
        want = backend.model.pipe(x).clone()

    backend.demote_to_host()
    assert backend._stream_residency is not None
    assert not backend._stream_residency.plan.resident
    with torch.no_grad():
        assert torch.equal(backend.model.pipe(x), want), (
            "a host-tier instance must still compute the same answer"
        )

    backend.promote_to_device()
    assert not backend._stream_residency.plan.streamed
    with torch.no_grad():
        assert torch.equal(backend.model.pipe(x), want)


def test_a_tier_move_on_a_weightless_instance_refuses_loudly() -> None:
    """Silence is the one answer a placement report must never give: an
    instance with no module tree cannot be tiered, and must say so rather than
    report a successful demote that moved nothing."""
    from gen_worker.serving.residency import ResidencyError
    from gen_worker.serving.serve_loop import _InstanceBackend

    class Weightless:
        pass

    backend = _InstanceBackend.__new__(_InstanceBackend)
    backend.model_cls = Weightless
    backend.model = Weightless()
    backend._stream_residency = None
    with pytest.raises(ResidencyError, match="no nn.Module tree"):
        backend.demote_to_host()


def test_module_roots_finds_the_three_shapes_the_serve_path_holds() -> None:
    bare = nn.Linear(4, 4)
    assert module_roots(bare) == [("Linear", bare)]

    class Pipeline:
        def __init__(self) -> None:
            self.components = {"unet": nn.Linear(4, 4), "cfg": "not a module"}

    pipeline = Pipeline()
    assert [n for n, _ in module_roots(pipeline)] == ["unet"]

    class Author:
        def __init__(self) -> None:
            self.pipe = pipeline
            self.extra = nn.Linear(4, 4)
            self._private = nn.Linear(4, 4)

    names = [n for n, _ in module_roots(Author())]
    assert "pipe.unet" in names and "extra" in names
    assert not any(n.startswith("_") for n in names)


def test_plans_are_frozen_records() -> None:
    plan = plan_residency(_pyramid(), budget_bytes=0, min_stream_bytes=1)
    assert isinstance(plan, ResidencyPlan)
    with pytest.raises(Exception):
        plan.resident = ()  # type: ignore[misc]
