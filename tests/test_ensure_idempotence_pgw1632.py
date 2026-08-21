"""pgw#1632(a): the `ensure_*` idempotence harness — the fence pgw#1596 needed.

A fill on the weight path is supposed to be a pure function of (manifest,
store): objects the CAS already holds are skipped, the rest are fetched, and
progress IS bytes-present. Three consequences follow and NONE of them had an
instrument before this file:

1. **Called twice** — the second call moves zero network bytes and hands back
   the same path.
2. **Killed at k%, then re-entered** — the second pass fetches exactly the
   bytes the first did not bank. Not "approximately": every object either
   landed or did not, so the arithmetic is exact and any restart-from-zero is
   a numeric failure, not a judgement call.
3. **Disk high-water ≤ 1x tree + the fan-out's in-flight files** — a resumable
   fill never needs a second copy of the tree, which is the property pgw#1596
   violated in the GATE (it demanded 2x) and th#2246 violated in the WRITER
   (it wrote 2x).

Plus the cross-layer property that was the actual defect: **the headroom gate
and the fill loop must agree about what is missing.** Proven here by running
both and comparing numbers — the gate's refusal quotes `required_bytes`, the
origin counts what the fill then really fetched, and pgw#1596 is the case where
those two disagreed by 86 GB.

Everything runs the production code path against a real HTTP origin and a real
`LocalCAS`. The only injected pieces are the origin's byte fuse (that is what
"the fetcher raised" MEANS) and `ModelStore`'s own `disk_free_bytes_fn`
constructor parameter, which expresses "this volume holds one copy of the tree"
as arithmetic over the real files the fill really wrote.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from pathlib import Path
from typing import Any, Iterator

import pytest

from gen_worker import activity
from gen_worker.capability import InsufficientDiskError
from gen_worker.models import store as store_mod
from gen_worker.models.refs import WireRef
from gen_worker.models.store import _DISK_GC_MARGIN_BYTES, ModelStore
from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.transfer.grants import DEFAULT_PARALLEL

from fill_fixture import (
    FILL_OPS,
    FillContext,
    HighWater,
    Op,
    Origin,
    Tree,
    build_tree,
    disk_used,
    resident_bytes,
    run_fill,
)

#: 32 objects resolves every k the design asks for (3.1% granularity) and keeps
#: the whole 4-op x 4-k matrix inside a CI budget. The RATIOS that decide
#: pass/fail are size independent — this is the pgw#1596 test's own scaling
#: trick, and the incident it descends from was 105 GB.
OBJECTS = 32
OBJECT_BYTES = 256 * 1024

#: What a fill legitimately holds ABOVE the tree: the fan-out's in-flight
#: temporaries, one per worker. Bounded by the transfer width, never by the
#: tree size — which is exactly why a 2x-tree requirement is a bug and not a
#: bigger constant.
INFLIGHT_SLACK = DEFAULT_PARALLEL * OBJECT_BYTES

K_PERCENTS = (10, 50, 80, 99)


@pytest.fixture(autouse=True)
def _clean_activity() -> Iterator[None]:
    activity.reset_for_tests()
    yield
    activity.reset_for_tests()


@pytest.fixture(scope="module")
def origin() -> Iterator[Origin]:
    served = Origin()
    try:
        yield served
    finally:
        served.close()


@pytest.fixture(scope="module")
def tree(origin: Origin) -> Tree:
    return build_tree(origin, objects=OBJECTS, object_bytes=OBJECT_BYTES)


@pytest.fixture
def one_tree_budget(tree: Tree) -> int:
    """A volume that holds exactly ONE copy of the tree, plus production margin.

    `_DISK_GC_MARGIN_BYTES` is the gate's own reserve, so a disk sized without
    it would refuse every fill for reasons that have nothing to do with
    re-entry. What this budget deliberately does NOT have room for is a SECOND
    copy of the tree.
    """

    return tree.total_bytes + _DISK_GC_MARGIN_BYTES + INFLIGHT_SLACK


def _ctx(tmp_path: Path, tree: Tree, origin: Origin, budget: int | None = None) -> FillContext:
    ctx = FillContext(cache_dir=tmp_path / "cas", tree=tree, origin=origin)
    if budget is not None:
        ctx.disk_free_bytes_fn = ctx.budget_fn(budget)
    return ctx


def _no_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    """A kill is one interruption, not a retry storm.

    `ModelStore` retries a failed download three times with backoff, and that
    inner retry is itself a re-entry — which is real behaviour but a different
    test, and paying its 5 s of `asyncio.sleep` in all sixteen cells buys
    nothing this file is measuring. The property under test is what happens
    when the operation is ENTERED AGAIN, so the first entry is made to stop.
    """

    monkeypatch.setattr(store_mod, "_DOWNLOAD_RETRIES", 1)


# ---------------------------------------------------------------------------
# Property 1 — called twice
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("op", FILL_OPS, ids=lambda o: o.name)
def test_called_twice_moves_zero_network_bytes(
    tmp_path: Path, tree: Tree, origin: Origin, op: Op
) -> None:
    """The second call is free, and it is the SAME answer.

    An `ensure_*` that re-fetches what it already holds is not an ensure at
    all; it is a download with an optimistic name. The origin's own counter is
    the check, because it is the only number no layer under test computed.
    """

    ctx = _ctx(tmp_path, tree, origin)
    origin.reset()

    first_path = run_fill(ctx, op)
    first_wire = origin.wire_bytes
    assert first_wire == tree.total_bytes, "the cold pass fetches the whole tree"

    origin.reset()
    second_path = run_fill(ctx, op)

    assert origin.wire_bytes == 0, (
        f"{op.name} re-fetched {origin.wire_bytes} bytes it already held"
    )
    assert second_path == first_path
    assert disk_used(ctx.cache_dir) <= tree.total_bytes + INFLIGHT_SLACK, (
        "a second call must not leave a second copy behind"
    )


def test_a_second_call_leaves_no_open_download_record(
    tmp_path: Path, tree: Tree, origin: Origin
) -> None:
    """th#2204/th#2205's LIABILITY, as a property of the fill.

    A hub-side `model_download` row is opened by `DOWNLOADING` and only
    `ON_DISK`/`FAILED`/`EVICTED` closes it; while it is open the hub vetoes
    idle retirement and parks placement (th#2205: 179 minutes at $1.59/hr).
    So the fence on a re-entry is not "say nothing" — it is "leave nothing
    open", on a call that moved no bytes at all.

    MEASURED HERE, and deliberately recorded rather than tightened: a FRESH
    PROCESS over a warm CAS does still emit one `DOWNLOADING`, because the
    identity bank that suppresses it (`_disk_identities`) is in-memory and a
    restart has none. It closes within the same materialization, so it is a
    brief honest "verifying" row and not the immortal one th#2204 cost a rental
    to — but a change that makes it stop closing fails right here.
    """

    ctx = _ctx(tmp_path, tree, origin)
    origin.reset()
    run_fill(ctx, FILL_OPS[0])

    origin.reset()
    ctx.wire = type(ctx.wire)()
    run_fill(ctx, FILL_OPS[0])

    assert origin.wire_bytes == 0
    events = ctx.wire.model_events
    opened = sum(1 for e in events if e.state == pb.MODEL_STATE_DOWNLOADING)
    closed = sum(
        1 for e in events
        if e.state in (
            pb.MODEL_STATE_ON_DISK, pb.MODEL_STATE_FAILED, pb.MODEL_STATE_EVICTED,
        )
    )
    assert closed >= opened, (
        f"{opened} download record(s) opened, {closed} closed: "
        f"{[e.state for e in events]}"
    )
    moved = [e.bytes_done for e in events if e.state == pb.MODEL_STATE_DOWNLOADING]
    assert all(int(b) == 0 for b in moved), (
        f"a fully resident re-entry reported bytes moving: {moved}"
    )


# ---------------------------------------------------------------------------
# Property 2 — killed at k%, re-entered
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("op", FILL_OPS, ids=lambda o: o.name)
@pytest.mark.parametrize("k", K_PERCENTS)
def test_killed_at_k_percent_resumes_without_refetching(
    tmp_path: Path,
    tree: Tree,
    origin: Origin,
    monkeypatch: pytest.MonkeyPatch,
    op: Op,
    k: int,
) -> None:
    """THE REGRESSION, generalized. Interrupt the fill, re-enter, and the second
    pass must fetch exactly the remainder — never the tree.

    The pre-`d2e31047` gate fails this at every k (proven by
    `test_the_harness_is_red_against_the_pre_pgw1596_gate` below); a fill that
    restarted from zero would fail it by fetching `total` again.
    """

    _no_retry(monkeypatch)
    ctx = _ctx(tmp_path, tree, origin)
    total = tree.total_bytes

    origin.reset()
    origin.arm(total * k // 100)
    with HighWater(ctx.cache_dir) as killed_pass:
        with pytest.raises(BaseException):
            run_fill(ctx, op)

    banked = resident_bytes(ctx.cache_dir, tree)
    assert 0 < banked < total, (
        f"k={k} must interrupt the fill mid-flight; banked {banked} of {total}"
    )
    assert origin.wire_bytes == banked, (
        "every byte the origin served must be a byte the CAS banked — anything "
        "else is a partial object the resume will have to pay for again"
    )

    origin.reset()
    origin.arm(None)
    with HighWater(ctx.cache_dir) as resume_pass:
        path = run_fill(ctx, op)

    assert path.exists()
    assert origin.wire_bytes == total - banked, (
        f"{op.name} re-entered at {100 * banked // total}% and fetched "
        f"{origin.wire_bytes} bytes; exactly {total - banked} were missing"
    )
    peak = max(killed_pass.peak, resume_pass.peak)
    assert peak <= total + INFLIGHT_SLACK, (
        f"disk high-water {peak} exceeded one tree ({total}) plus the "
        f"fan-out's in-flight files ({INFLIGHT_SLACK}) — a resumable fill "
        f"never needs a second copy"
    )


def test_the_harness_is_red_against_the_pre_pgw1596_gate(
    tmp_path: Path,
    tree: Tree,
    origin: Origin,
    one_tree_budget: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RED ARM. Restore the pre-`d2e31047` precondition and the harness fails.

    Before pgw#1596 the gate compared free space against the ENTIRE manifest
    with nothing subtracting the resident part. On a disk that fits exactly one
    copy of the tree that is unsatisfiable the moment anything has landed —
    which is what killed a 105 GB pull 157 MB from the end on a real H200.

    A harness that cannot go red proves nothing, so the pre-fix behaviour is
    reconstructed exactly — a plan whose every object reads as MISSING, which
    is what a gate priced off the request rather than the delta computes — and
    the resume is asserted to REFUSE.
    """

    _no_retry(monkeypatch)
    ctx = _ctx(tmp_path, tree, origin, budget=one_tree_budget)
    total = tree.total_bytes

    origin.reset()
    origin.arm(total * 80 // 100)
    with pytest.raises(BaseException):
        run_fill(ctx, FILL_OPS[0])
    banked = resident_bytes(ctx.cache_dir, tree)
    assert banked > INFLIGHT_SLACK

    # The post-fix gate lets the resume through on the same disk.
    origin.reset()
    origin.arm(None)
    run_fill(ctx, FILL_OPS[0])
    assert origin.wire_bytes == total - banked

    # Now the pre-fix gate, on the same state: it demands the whole tree free.
    from gen_worker.models import fill_plan as fill_plan_mod

    real_plan = fill_plan_mod.plan_for_snapshot

    def _price_the_request(cache_dir: object, files: list[object]) -> object:
        """The pre-pgw#1596 shape: every object of the manifest is 'missing'."""
        whole = real_plan(cache_dir, files)
        return fill_plan_mod.FillPlan(missing=whole.present + whole.missing)

    monkeypatch.setattr(fill_plan_mod, "plan_for_snapshot", _price_the_request)
    fresh = _ctx(tmp_path, tree, origin, budget=one_tree_budget)
    with pytest.raises(InsufficientDiskError) as caught:
        run_fill(fresh, FILL_OPS[0])
    assert caught.value.required_bytes == total, (
        "the reconstructed pre-fix gate must charge for the whole manifest — "
        "if it does not, this arm is not proving the harness can go red"
    )


# ---------------------------------------------------------------------------
# Property 3 — precondition consistency
# ---------------------------------------------------------------------------


def test_the_gate_and_the_fill_agree_about_what_is_missing(
    tmp_path: Path,
    tree: Tree,
    origin: Origin,
    one_tree_budget: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """THE NAMED ANTI-PATTERN: *a precondition computed from a stale view of
    state the operation itself changes.*

    Both layers are run against one partially-filled store and their numbers
    are compared: what the gate says is still needed, and what the fill then
    actually pulls off the wire. pgw#1596 is precisely the case where those
    disagreed — the gate said 105 GB, the fill needed 157 MB.

    This is the structural form of the check. Its cheap twin (the gate may not
    re-derive a size at all) is `test_the_gate_derives_its_cost_from_the_one_predicate`.
    """

    _no_retry(monkeypatch)
    ctx = _ctx(tmp_path, tree, origin, budget=one_tree_budget)
    total = tree.total_bytes

    origin.reset()
    origin.arm(total * 50 // 100)
    with pytest.raises(BaseException):
        run_fill(ctx, FILL_OPS[0])
    banked = resident_bytes(ctx.cache_dir, tree)
    assert 0 < banked < total

    # What the GATE says is still required, taken from its own refusal on a
    # disk with nothing free.
    starved = FillContext(cache_dir=ctx.cache_dir, tree=tree, origin=origin)
    starved.disk_free_bytes_fn = lambda: 0
    snapshot = tree.snapshot()
    import asyncio

    from gen_worker.models.fill_plan import plan_for_snapshot

    plan = plan_for_snapshot(ctx.cache_dir, list(snapshot.files))
    with pytest.raises(InsufficientDiskError) as caught:
        asyncio.run(
            starved.store()._ensure_disk_headroom(
                WireRef("acme/harness-model"), plan,
            )
        )
    gate_says = caught.value.required_bytes

    # What the FILL actually moves, from the origin's own counter.
    origin.reset()
    origin.arm(None)
    run_fill(ctx, FILL_OPS[0])
    fill_moved = origin.wire_bytes

    assert gate_says == fill_moved == total - banked, (
        f"the gate required {gate_says} bytes and the fill moved {fill_moved}; "
        "a gate that does not share the fill's skip predicate is the pgw#1596 "
        "defect regardless of which direction it is wrong in"
    )


def _gate_source() -> ast.FunctionDef:
    source = textwrap.dedent(inspect.getsource(ModelStore._ensure_disk_headroom))
    tree_ = ast.parse(source)
    node = tree_.body[0]
    assert isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    return node  # type: ignore[return-value]


def test_the_gate_derives_its_cost_from_the_one_predicate() -> None:
    """THE LINT. The gate may not compute a size; it may only be handed one.

    Every instance of this bug class looks the same in source: a precondition
    that re-derives its cost from the REQUEST (walk the manifest, sum the
    sizes) instead of from the DELTA (ask the store what it already holds). So
    the rule is structural and cheap — inside `_ensure_disk_headroom` there is
    no size arithmetic over files, and the resident figure comes from the same
    content-addressed predicate the fill skips on.

    A rewrite that feeds the gate the fill's PLAN satisfies this by
    construction: a plan is a delta, so there is nothing left to re-derive.
    """

    node = _gate_source()
    names = {
        n.attr for n in ast.walk(node) if isinstance(n, ast.Attribute)
    } | {
        n.id for n in ast.walk(node) if isinstance(n, ast.Name)
    }

    assert "size_bytes" not in names, (
        "`_ensure_disk_headroom` reads a file size directly. That is the "
        "pgw#1596 shape: the gate is re-deriving cost from the request instead "
        "of from what the store is missing."
    )
    assert not any(
        isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "sum"
        for n in ast.walk(node)
    ), "the gate must be HANDED its cost, never sum one"
    assert "plan" in names, (
        "the gate must consume the fill's own plan; nothing else can be proven "
        "to agree with what the fill will skip"
    )
    # pgw#1631: and the plan is the ONLY size input. A manifest or a total in
    # the signature is a thing the gate could price differently.
    params = [a.arg for a in node.args.args + node.args.kwonlyargs]
    assert "files" not in params and "needed_bytes" not in params, (
        f"the gate's signature still admits a manifest to re-price: {params}"
    )


# ---------------------------------------------------------------------------
# The coverage fence
# ---------------------------------------------------------------------------

#: `ensure_*` functions in `models/` that move no bytes, with the reason. The
#: point of naming them is that a NEW `ensure_*` belongs in one list or the
#: other, and cannot quietly belong to neither.
NON_FILL_ENSURES = {
    "_ensure_objects": "the fill's inner loop; driven through every op above",
    "ensure_snapshot": "the method behind `ensure_snapshot_async`, covered by it",
    "_ensure_disk_headroom": "a gate — it moves no bytes, and has its own tests",
    "ensure_pinned": "writes one manifest pin, not tree bytes (pgw#1526)",
    "ensure_resident": "a VRAM lease scope; no disk fill",
    "ensure_warm": "checkpoint juggle over already-materialized trees",
}

#: `ensure_*` functions the harness drives, by the name in `FILL_OPS`.
COVERED_ENSURES = {"ensure_local", "_materialize_local", "ensure_snapshot_async"}


def test_every_fill_in_models_is_covered_or_classified() -> None:
    """A new `ensure_*` cannot be added without answering "is it idempotent?".

    The design's whole point is that this property belongs to the CLASS, not to
    whichever function last had an incident. So the fence enumerates the class.
    """

    root = Path(store_mod.__file__).parent
    found: dict[str, str] = {}
    for path in sorted(root.glob("*.py")):
        module = ast.parse(path.read_text())
        for node in ast.walk(module):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if node.name.lstrip("_").startswith("ensure_") or node.name == "_materialize_local":
                found[node.name] = path.name

    unclassified = {
        name: where
        for name, where in found.items()
        if name not in COVERED_ENSURES and name not in NON_FILL_ENSURES
    }
    assert not unclassified, (
        f"unclassified fill/ensure functions: {unclassified}. Add each to "
        f"FILL_OPS (and prove it idempotent) or to NON_FILL_ENSURES with the "
        f"reason it moves no bytes."
    )
    assert COVERED_ENSURES <= set(found), (
        f"the harness claims to cover {COVERED_ENSURES - set(found)}, which no "
        f"longer exists in models/"
    )
    assert {op.name for op in FILL_OPS} and len(FILL_OPS) >= 4
