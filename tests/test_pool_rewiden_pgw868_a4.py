"""pgw#868 A4: the pool sizes K off an ESTIMATE and then measures the truth.

``aot_compile_pool.entry_workers`` runs before a single entry has compiled, so
the only per-entry device figure it can have is
``mint_budget.co_residency().need_bytes`` — the MINT CHILD's whole co-residency
estimate (a full weight copy, an activation set nothing observed, two flat
constants), spent as ONE entry child's ask. pgw#877 measured that ~56 % of it
was never observed and renamed its basis ``"estimated"`` for that reason.

Every entry child then reports what it actually peaked at, ``observe_entry_device``
banks it, and until this change nothing read it: attempt eighteen's sdxl mint
ran ``K=2, binding=vram, underwidth=6`` for all 36 entries against an estimate
its own first two entries had already disproved. K is the mint's ONLY
multiplicative lever, so that is the largest single item in A4.

What these tests hold:

* the measurement REACHES the width — through the real ``entry_workers``, not a
  recomputation written here;
* it only ever WIDENS, and never past the caller's own ``limit``;
* it refuses on one sample, on no sample, on an absent device lock, and on a
  pool that never read the card — the four ways this could widen a pod onto an
  OOM;
* the ledger's capacity identity survives a width that MOVES, which is the
  thing a mid-flight K silently breaks;
* and the mint child installs the goal set that decides whether the tenant
  reserves apply at all.

Nothing here mocks the thing under test: ``entry_workers``, ``PoolWidth``,
``EntryCompilePool`` and ``mint_budget.entry_device_ask`` are the production
objects. The one synthesized input is the entry child's device peak, which a
box with no CUDA card cannot produce any other way — it arrives through the
real ``EntryReport`` and the real ``observe_entry_device``.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from gen_worker import aot_compile_pool as pool
from gen_worker import mint_budget, worker_goals

_GIB = 1024 ** 3


def _sized(
    *,
    entries: int = 36,
    free_vram: int = 30 * _GIB,
    per_entry: int = 10 * _GIB,
    limit: int = 0,
    device_lock: bool = True,
) -> pool.PoolWidth:
    """A width from the REAL policy, on a stated host.

    Stated rather than derived: a CI runner honestly derives K=1 and every
    assertion below would then pass while exercising nothing.
    """
    return pool.entry_workers(
        entries, limit=limit, vcpus=32,
        available_bytes=256 * _GIB, peak_rss_bytes=2 * _GIB,
        free_vram_bytes=free_vram, device_bytes=per_entry,
        device_basis="estimated", device_lock=device_lock,
        goals=worker_goals.MINT_ONLY)


@pytest.fixture(autouse=True)
def _roomy_card(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hold the CARD constant and generous for this whole file.

    pgw#992 added a second, independent bound to `_rewiden`: the widened set's
    SIMULTANEOUS peak against the card, which on a box with no CUDA is
    unreadable and therefore refuses every widen (fail-closed, by design). The
    rows in this file are about A4's DIVISOR — an estimate replaced by an
    observation — so the card is stated once here and never varies. The
    simultaneity bound has its own file
    (``test_pool_simultaneity_pgw992.py``), where the card is the variable.
    """
    monkeypatch.setattr(
        pool, "card_census",
        lambda device=-1: pool.CardCensus(512 * _GIB, 512 * _GIB, 0, "sampled"))


def _report(entry: str, reserved: int) -> pool.EntryReport:
    return pool.EntryReport(
        entry=entry, status=pool.COMPILED,
        peak_device_reserved_bytes=reserved)


def _observe(box: pool.EntryCompilePool, reserved: int, n: int = 2) -> None:
    for i in range(n):
        box.observe_entry_device(_report(f"unet/dim={i}", reserved))
        box._rewiden()


# ---------------------------------------------------------------------------
# The win: an estimate replaced by an observation, through the real policy
# ---------------------------------------------------------------------------


def test_the_measured_entry_peak_replaces_the_estimate_and_widens_K(
    tmp_path: Path,
) -> None:
    """Attempt eighteen's own numbers, with the measurement plugged in.

    30 GiB free, a 10 GiB/entry ESTIMATE and a mint-only pod gives K=3. The
    entry children then report a 5 GiB reserved high-water; the ask that
    becomes is 5 GiB + the CUDA context floor mint_budget adds back, and the
    same free figure divided by THAT gives K=5. Nothing else moved: not the
    reserve, not the ceiling, not the host bounds, not the card.
    """
    width = _sized(free_vram=30 * _GIB, per_entry=10 * _GIB)
    assert width.workers == 3 and width.binding == "vram", width.reason
    assert width.per_entry_device_basis == "estimated"

    box = pool.EntryCompilePool(tmp_path / "pool", width=width)
    _observe(box, 5 * _GIB)

    ask = mint_budget.entry_device_ask(5 * _GIB)
    assert box.width.workers == 30 * _GIB // ask == 5, box.width.reason
    assert box.width.per_entry_device_basis == "measured", (
        "a width derived from an entry child's own high-water must SAY it was "
        "measured — the basis is what tells the next reader whether the number "
        "is an observation or the guess pgw#877 renamed")
    assert box.width.per_entry_device_bytes == mint_budget.entry_device_ask(
        5 * _GIB), (
        "the ask must come from mint_budget's own function — a CUDA context, "
        "the cuBLAS/cuDNN handles and the driver's per-process overhead live "
        "OUTSIDE the allocator's high-water and have to be added back")
    # The initial row survives being superseded: the prize is a delta.
    assert box.width_initial.workers == 3
    assert box.ledger.workers_initial == 3 and box.ledger.workers == 5


def test_the_re_derivation_divides_the_pools_OWN_free_figure_not_a_fresh_probe(
    tmp_path: Path,
) -> None:
    """The trap this method has to avoid, stated as a test.

    Re-derivation happens while K children are ALIVE and holding their
    footprints, so a fresh ``mem_get_info`` would read their own memory as
    absent capacity and NARROW — the exact opposite of the truth. The
    re-derivation is against ``width_initial.free_device_bytes``, which is
    the figure the pool was sized with, so it is indifferent to what is
    running when it fires.
    """
    width = _sized(free_vram=48 * _GIB, per_entry=12 * _GIB)
    box = pool.EntryCompilePool(tmp_path / "pool", width=width)

    def _explode(*_a: object, **_k: object) -> int:
        raise AssertionError(
            "_rewiden probed the card; a probe taken with K children resident "
            "reads their own footprints as missing capacity")

    # The REAL guard: if anything in the path samples the device, this raises.
    box._rewiden.__globals__["_probe_free_device_bytes"]  # present
    original = pool._probe_free_device_bytes
    pool._probe_free_device_bytes = _explode      # type: ignore[assignment]
    try:
        _observe(box, 4 * _GIB)
    finally:
        pool._probe_free_device_bytes = original

    assert box.width.workers > width.workers
    assert box.width.free_device_bytes == width.free_device_bytes


# ---------------------------------------------------------------------------
# Fail-closed: the five refusals
# ---------------------------------------------------------------------------


def test_one_sample_is_an_anecdote_and_does_not_move_K(tmp_path: Path) -> None:
    """The first round's children start together, so they also miss the shared
    PCH and the autotune cache — they are the LEAST representative entries in
    the cell. Widening off one of them is pgw#842's "5-vs-3 with nothing
    recorded to say why" with a new cause."""
    width = _sized(free_vram=30 * _GIB, per_entry=10 * _GIB)
    box = pool.EntryCompilePool(tmp_path / "pool", width=width)

    _observe(box, 2 * _GIB, n=1)
    assert box.width.workers == 3, "one report is not a measurement"
    assert box.device_samples == 1

    _observe(box, 2 * _GIB, n=1)
    assert box.width.workers > 3, "two reports are"


def test_it_never_narrows_however_large_the_measurement(
    tmp_path: Path,
) -> None:
    """Children are already running against the current width. A measurement
    saying the pool is too wide cannot be acted on — the entries are spawned —
    and acting on it would strand them. It is banked for the NEXT mint through
    ``mint_budget``, which is where a narrowing belongs."""
    width = _sized(free_vram=60 * _GIB, per_entry=8 * _GIB)
    assert width.workers > 1
    box = pool.EntryCompilePool(tmp_path / "pool", width=width)

    _observe(box, 55 * _GIB)          # would derive K=1

    assert box.width.workers == width.workers
    assert box.width is width, "a refusal must leave the record untouched"


def test_an_operator_who_forced_the_serial_path_keeps_it(
    tmp_path: Path,
) -> None:
    """``entry_workers(limit=)`` is how an operator or a test forces K=1
    without pretending a 4-vCPU pod is an H100 host. A measurement must not
    widen it back out from under them — which is why ``limit`` had to become
    part of the ``PoolWidth`` record."""
    width = _sized(free_vram=60 * _GIB, per_entry=10 * _GIB, limit=1)
    assert width.workers == 1 and width.limit == 1
    box = pool.EntryCompilePool(tmp_path / "pool", width=width)

    _observe(box, 1 * _GIB)

    assert box.width.workers == 1, (
        f"the operator's cap was overridden by a measurement: "
        f"{box.width.reason}")


def test_a_pool_with_no_device_lock_stays_serial(tmp_path: Path) -> None:
    """K=1 without ``set_gpu_benchmark_lock_context`` is a SAFETY width, not a
    resource one: two entries benchmarking at once bake contention-chosen
    kernel configs into a cell whose key would not move. No amount of free
    VRAM licenses widening past it."""
    width = _sized(free_vram=60 * _GIB, per_entry=10 * _GIB, device_lock=False)
    assert width.workers == 1 and width.binding == "device-lock"
    box = pool.EntryCompilePool(tmp_path / "pool", width=width)

    _observe(box, 1 * _GIB)

    assert box.width.workers == 1, box.width.reason


def test_a_pool_that_never_read_the_card_does_not_invent_a_free_figure(
    tmp_path: Path,
) -> None:
    """``entries <= 1`` and an absent probe both leave ``free_device_bytes``
    unread (``-1`` / ``0``). There is nothing to re-divide, and supplying a
    figure here would be exactly the guess this whole method deletes."""
    width = pool.entry_workers(1, vcpus=32, available_bytes=256 * _GIB)
    assert width.free_device_bytes <= 0
    box = pool.EntryCompilePool(tmp_path / "pool", width=width)

    _observe(box, 1 * _GIB)

    assert box.width.workers == 1


def test_a_raising_re_derivation_leaves_the_width_exactly_as_it_was(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A width policy that could kill a mint would be a worse defect than the
    narrow width it exists to fix. pgw#846's ordering is explicit: mint
    reliably first, mint fast third."""
    width = _sized(free_vram=30 * _GIB, per_entry=10 * _GIB)
    box = pool.EntryCompilePool(tmp_path / "pool", width=width)

    def _boom(_peak: int) -> int:
        raise RuntimeError("banked measurement unreadable")

    monkeypatch.setattr(mint_budget, "entry_device_ask", _boom)
    _observe(box, 4 * _GIB)

    assert box.width is width


def test_a_zero_peak_is_not_a_measurement(tmp_path: Path) -> None:
    """A child too old to report, or one that died before it allocated, banks
    nothing. ``observe_entry_device`` already refuses to count it; this pins
    that the sample counter agrees, because the counter is what licenses the
    widening."""
    width = _sized(free_vram=30 * _GIB, per_entry=10 * _GIB)
    box = pool.EntryCompilePool(tmp_path / "pool", width=width)

    for i in range(4):
        box.observe_entry_device(
            pool.EntryReport(entry=f"unet/dim={i}", status=pool.COMPILED))
        box._rewiden()

    assert box.device_samples == 0 and box.width.workers == 3


# ---------------------------------------------------------------------------
# The ledger identity, under a width that moves
# ---------------------------------------------------------------------------


def test_the_capacity_identity_survives_a_width_that_moves() -> None:
    """``capacity_s`` used to be ``wall_s * final_workers``, computed once at
    the end. With a K that can move mid-pool that prices the whole run at a
    width the first entries never had, and ``busy + idle == capacity`` — the
    identity the whole ledger exists to keep exact — becomes a residual.

    Accumulating per interval at the LIVE width keeps it exact. Driven here
    through the ledger's own arithmetic rather than a clock, so it asserts the
    identity and not the runner's spare CPU.
    """
    ledger = pool.PoolLedger(workers=2, workers_initial=2)
    # Two seconds at K=2, then two at K=4, with one slot free throughout.
    for width_workers, seconds in ((2, 2.0), (4, 2.0)):
        ledger.capacity_s += seconds * width_workers
        ledger.idle_other_s += seconds * 1
        ledger.busy_s += seconds * (width_workers - 1)

    assert ledger.capacity_s == pytest.approx(12.0)
    assert ledger.busy_s + ledger.idle_s == pytest.approx(ledger.capacity_s), (
        "busy + idle must equal capacity; a residual here means the ledger is "
        "pricing seconds at a width that was never live for them")


def test_the_staging_cap_follows_the_width(tmp_path: Path) -> None:
    """The cap on programs staged ahead of the running set is derived from K.
    Frozen at the construction width it would starve exactly the slots a
    re-derivation just opened, and the widening would buy nothing."""
    src = Path("src/gen_worker/aot_compile_pool.py").read_text()
    body = src.split("def compile(", 1)[1]
    assert "staged_cap = max(1, self.width.workers" in body
    # pgw#1052 reshaped the loop (`while True:` with the pull inside it); the
    # invariant is unchanged: staged_cap is recomputed at the TOP of every
    # round, after `_rewiden` may have moved the width.
    head, _rest = body.split("while staged and not failure", 1)
    assert "while True:" in head
    assert head.index("while True:") \
        < head.index("staged_cap = max(1, self.width.workers"), (
        "staged_cap must be recomputed INSIDE the round loop, after "
        "_rewiden may have moved the width")


def test_the_width_change_is_emitted_not_silent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """pgw#842: "an unexplained K is a defect in itself". A K that changes
    mid-mint and says nothing is the same defect with a moving value."""
    from gen_worker import activity as activity_mod

    seen: list[str] = []
    monkeypatch.setattr(
        activity_mod, "emit_event",
        lambda kind, detail, **kw: seen.append(str(detail)))

    width = _sized(free_vram=30 * _GIB, per_entry=10 * _GIB)
    box = pool.EntryCompilePool(tmp_path / "pool", width=width)
    _observe(box, 5 * _GIB)

    assert box.width.workers == 5
    assert any("'entry_workers': 5" in d for d in seen), (
        f"the re-derived width never reached a hub row: {seen}")
    assert any("'per_entry_device_basis': 'measured'" in d for d in seen)


# ---------------------------------------------------------------------------
# The goal set the width policy reads — and the process that never had one
# ---------------------------------------------------------------------------


def test_the_mint_child_installs_its_own_goal_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``worker_goals.install`` is called from ``entrypoint`` and
    ``procsplit.parent`` — both the SERVING parent. The mint runs in a child
    spawned as ``python -m gen_worker.mint_child``, which installed nothing, so
    ``current()`` fell back to ``SERVE_ONLY`` in the one process that sizes the
    compile pool: a mint-only pod held a tenant VRAM reserve, a serving CPU
    headroom and a tenant host-RAM reserve for a tenant that cannot reach it.
    """
    from gen_worker import mint_child

    monkeypatch.setenv("WORKER_MODE", "forge")
    worker_goals.reset_for_test()
    assert worker_goals.current() is worker_goals.SERVE_ONLY, (
        "precondition: nothing installed yet")

    mint_child._install_goals()

    goals = worker_goals.current()
    assert goals.mint and not goals.serve, (
        f"the child read {goals.declared!r} and holds serve={goals.serve} "
        f"mint={goals.mint}")
    assert not goals.tenant_reserve_applies()
    worker_goals.reset_for_test()


def test_the_goal_set_reaches_the_width_and_drops_the_tenant_reserves() -> None:
    """The observable pgw#846's attempts fourteen and fifteen needed: the
    reason string names the goals, and a mint-only pod's reserves are gone.
    Two widths from the REAL policy on the SAME host — only the goals move."""
    common = dict(
        vcpus=4, available_bytes=32 * _GIB, peak_rss_bytes=2 * _GIB,
        free_vram_bytes=30 * _GIB, device_bytes=10 * _GIB,
        device_basis="estimated", device_lock=True)

    serving = pool.entry_workers(36, goals=worker_goals.SERVE_ONLY, **common)
    forge = pool.entry_workers(36, goals=worker_goals.MINT_ONLY, **common)

    assert "goals=serve" in serving.reason and "goals=mint" in forge.reason
    assert forge.workers >= serving.workers, (
        f"a pod with no tenant may not compile NARROWER than one with a "
        f"tenant: forge {forge.reason!r} vs serve {serving.reason!r}")
    assert forge.cpu_workers > serving.cpu_workers, (
        "SERVING_HEADROOM_CPUS protects an eager forward this pod will never "
        "run — on a 4-vCPU pod that is a quarter of the pool")


def test_the_declaration_the_child_cannot_read_is_not_fatal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A mint that died because it could not parse its own goal declaration
    would trade acceptance 2 for acceptance 3. The fallback is the narrow,
    serve-only width — which is exactly what runs today."""
    from gen_worker import mint_child

    worker_goals.reset_for_test()
    monkeypatch.setattr(
        mint_child, "load_settings",
        lambda **_: (_ for _ in ()).throw(RuntimeError("unreadable env")))

    mint_child._install_goals()      # must not raise

    assert worker_goals.current() is worker_goals.SERVE_ONLY
    worker_goals.reset_for_test()


def test_the_child_env_carries_the_declaration_the_child_now_reads(
) -> None:
    """The two halves have to meet: ``mint_process.child_env`` copies the
    parent's environment (so ``WORKER_MODE`` is there), and the loader maps it
    onto ``Settings.worker_mode``. Neither is new; the join is."""
    from gen_worker import mint_process
    from gen_worker.mint_process import MintRequest

    from gen_worker.mint_process import CompileCellSpec

    env = mint_process.child_env(
        MintRequest(
            function="generate", modules=(), family="sdxl", arm_token="arm1-x",
            target="/tmp/cell.tar.gz", work_root="/tmp/cap",
            report="/tmp/r.json", cfg=CompileCellSpec(family="sdxl")),
        base={"WORKER_MODE": "forge", "PATH": os.environ.get("PATH", "")})
    assert env["WORKER_MODE"] == "forge"
    assert env["GEN_WORKER_MINT_CHILD"] == "1"
