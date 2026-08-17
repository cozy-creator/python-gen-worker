"""pgw#1137 / DESIGN-RULINGS §4.30 — a mint on the USER'S OWN MACHINE is polite.

pgw#1127 §7 item 6 filed this and left it unowned: *"posture-aware K, niced
children, never saturate the user's only machine… before any cozy-local
release."* Its own §6 states the evidence — **zero ``os.nice`` calls anywhere in
``src/``**, and every clamp in ``aot_compile_pool`` written for a serving pod
with a co-resident TENANT rather than for a desktop with a co-resident HUMAN.

Since pgw#1127 S1 (`b29a5e95`) ``cozy serve`` drives the ordinary
``mint_supervisor`` -> ``aot_compile_pool`` -> ``aot_compile_child`` chain, so that saturation
is now a shipping product behaviour and not a hypothetical.

WHAT IS PROVEN HERE, and what each section would have looked like before

1. **The posture is DECLARED.** ``local_serve`` — the one entry that knows —
   states it; it is not sniffed off ``trust_class()`` or off ``publisher is
   None`` (both are facts about the SINK, and a rented community-cloud pod
   matches them while having nobody sitting at it), and it is not an env var
   (§1.17: an env may carry a VALUE, not a DECISION). RED: ``MintTask`` and
   ``MintRequest`` had no posture field and ``compile_posture`` did not exist.
2. **CPU.** The pool halves its core budget and the mint child drops its own
   scheduling priority — by the CHILD, on itself, never through a
   ``preexec_fn``. RED: ``entry_workers`` had no posture parameter and nothing
   in ``src/`` called ``os.nice``.
3. **Memory.** More host RAM is left alone, because ``MemAvailable`` counts the
   user's page cache as free and a mint that OOMs a desktop is worse than a
   slow one. RED: one reserve constant, sized for a pod.
4. **Pods do not regress.** The default posture is ``FLEET`` and a pod's width
   is arithmetically identical to what it was — proven against an independent
   re-implementation of the pre-issue formula, not against the code under test.
5. **Honesty.** The user can see the compile and what it is costing them. RED:
   the only progress signal on this path was an activity addressed to a hub
   cozy-local does not have, plus ``logger.info`` inside a subprocess.
6. **Interruptibility.** The mint child dies with its parent, and what it
   finished survives into the next run. RED: the child holds its own session
   (``start_new_session=True``), so a closed terminal left a full-speed compile
   tree running with nobody to reap it.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import msgspec
import pytest

from gen_worker import (
    aot_compile_child, aot_compile_pool, compile_cache, compile_posture,
    local_serve, mint_child, mint_process, mint_supervisor)
from gen_worker.compile_posture import FLEET, USER_MACHINE, CompilePosture
from gen_worker.child_contract import (
    CompileSpec, MintFrame, MintSlot)
from gen_worker.mint_process import MintRequest

SRC = Path(aot_compile_pool.__file__).parent


@pytest.fixture(autouse=True)
def _no_installed_posture() -> Iterator[None]:
    """The posture is a per-process publication; a test installs a value and
    every other test must not inherit it."""
    compile_posture.install(FLEET)
    yield
    compile_posture.install(FLEET)


def _width(
    entries: int, *, vcpus: int, avail_gib: float, posture: CompilePosture,
    free_vram_gib: float = 0.0, device_gib: float = 0.0, limit: int = 0,
) -> aot_compile_pool.PoolWidth:
    """``entry_workers`` driven off SUPPLIED facts only.

    Every reading the function would otherwise probe (cores, cgroup, card) is
    passed in, so these cases describe a laptop and a workstation from a
    32-core CI-less box without simulating anything the policy decides.
    """
    return aot_compile_pool.entry_workers(
        entries, vcpus=vcpus,
        available_bytes=int(avail_gib * 1024**3),
        device_lock=True, posture=posture, limit=limit)


# ---------------------------------------------------------------------------
# 1. The posture is DECLARED — not sniffed off the sink, not an env var
# ---------------------------------------------------------------------------


class _Task:
    """Captures the ``MintTask`` ``local_serve`` builds, without minting."""

    def __init__(self) -> None:
        self.task: Any = None
        self.watch: Any = None

    def supervise(self, task: Any, **kw: Any) -> Any:
        self.task = task
        self.watch = kw.get("watch")

        class _Coro:
            def close(self) -> None:
                pass

            def __await__(self) -> Any:  # pragma: no cover - never awaited here
                yield
                return None

        return _Coro()


class _Cfg:
    shapes: Tuple[Tuple[int, int], ...] = ((64, 64),)
    targets: Tuple[str, ...] = ("transformer",)
    family = "micro-diffusion"
    lora_bucket = 0
    guidance_scales: Tuple[float, ...] = ()
    text_lens: Tuple[int, ...] = ()


class _Pending:
    family = "micro-diffusion"
    arm_token = "arm2-" + "1" * 64
    mint_root = "/tmp/pgw1137-does-not-exist"
    cfg = _Cfg()


def test_the_local_serve_entry_DECLARES_the_user_machine_posture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """§4.30's input, stated at the one site that knows.

    RED before this issue: ``MintTask`` had no ``posture`` field at all, so the
    mint child sized its pool from a fleet policy on a desktop.
    """
    cap = _Task()
    monkeypatch.setattr(mint_supervisor, "supervise", cap.supervise)
    monkeypatch.setattr(local_serve, "_drive", lambda coro: None)
    monkeypatch.setattr(
        local_serve.cc, "cell_base_execution_lane", lambda pipe: "bf16")
    monkeypatch.setattr(local_serve, "_say", lambda line: None)
    monkeypatch.setattr(
        local_serve.fleet_cells, "terminus_of", lambda p: "already-ended")

    ctx = local_serve.mint_context(
        function="generate", module="micro_diffusion.endpoint",
        slots={"pipeline": MintSlot(ref="cozy/micro#1", path="/tmp/micro")})
    local_serve._mint_here(object(), _Pending(), ctx)  # type: ignore[arg-type]

    assert cap.task is not None, "the local path must build a MintTask"
    assert cap.task.posture == USER_MACHINE, (
        "cozy-local runs on the machine the user is sitting at; the posture "
        "that sizes and nices the mint has to be DECLARED here, because "
        "nothing downstream can measure whether a person is present")


def test_the_posture_is_not_derived_from_the_sink_or_from_the_trust_class(
) -> None:
    """The three near-miss proxies, fenced.

    pgw#1127 §2's own warning is *"two derivations of one fact"*. Trust
    (``local_cell_store.trust_class()``) answers whether the HUB will accept a
    cell from this hardware; ``publisher is None`` answers whether a sink was
    wired. A community-cloud pod satisfies both and is rented by the second
    with nobody on it — being polite there would slow work we are paying for.
    """
    tree = ast.parse((SRC / "compile_posture.py").read_text())
    # Identifiers the CODE touches. Read off the AST and not off the text, so
    # the module may (and does) explain in prose why each proxy was rejected
    # without the fence mistaking the explanation for a dependency.
    used = {
        node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
    } | {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    } | set(_module_names(SRC / "compile_posture.py"))
    for proxy in ("trust_class", "local_cell_store", "CellPublisher",
                  "worker_goals", "publisher", "fleet_cells"):
        assert proxy not in used, (
            f"compile_posture must not derive the posture from {proxy!r} — "
            f"it is a different question with a different authority")


def test_no_environment_variable_can_switch_politeness() -> None:
    """§1.17, verbatim: *"Envs are for secrets and configuration, not for logic
    gates."* Politeness changes the nice level of a whole process tree and
    halves K, so it is a decision and it travels as a typed value.
    """
    tree = ast.parse((SRC / "compile_posture.py").read_text())
    reads = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and node.attr in ("environ", "getenv", "environb")
    ]
    assert not reads, (
        "the posture must not be readable from the environment: an ambient "
        "toggle can be flipped by a stray shell export and cannot be reasoned "
        "about from the request that produced a cell")
    assert "os" not in {
        alias.name
        for node in ast.walk(tree) if isinstance(node, ast.Import)
        for alias in node.names
    }, "compile_posture holds policy, not process manipulation"


def test_the_posture_survives_the_parent_to_child_WIRE() -> None:
    """The child sizes the pool, so the declaration has to reach it.

    It rides ``MintRequest`` — the same JSON file that already carries
    ``device`` — so the round trip through msgspec is
    the property, not the in-process object identity.
    """
    task = mint_process.MintTask(
        pending=_Pending(), pipe=None, function="generate",
        modules=("micro_diffusion.endpoint",), posture=USER_MACHINE)
    request = mint_process.build_request(
        task, workdir=Path("/tmp/pgw1137"))
    assert request.posture == USER_MACHINE

    decoded = msgspec.json.decode(
        msgspec.json.encode(request), type=MintRequest)
    assert decoded.posture.user_machine is True, (
        "the width is computed INSIDE the mint child, whose only input is "
        "this file — a posture that does not survive the encode is a posture "
        "the pool never sees")


def test_a_fleet_mint_declares_nothing_and_gets_the_fleet_posture() -> None:
    """The non-regression that makes the whole design safe: FLEET is the
    struct default, so every caller that has not heard of this issue — the
    executor, every scheduled mint — is unchanged by construction."""
    task = mint_process.MintTask(
        pending=_Pending(), pipe=None, function="generate", modules=("m",))
    assert task.posture == FLEET
    request = mint_process.build_request(
        task, workdir=Path("/tmp/pgw1137"))
    assert request.posture == FLEET
    assert MintRequest(
        function="f", modules=(), family="x", arm_token="", target="",
        work_root="", report="",
        cfg=CompileSpec()).posture == FLEET


# ---------------------------------------------------------------------------
# 2. CPU — half the cores, and the lowest priority the scheduler offers
# ---------------------------------------------------------------------------


def test_a_user_machine_pool_takes_at_most_HALF_the_cores() -> None:
    """``CPUS_PER_ENTRY_WORKER`` is an AVERAGE (one core for ~71 % of an entry,
    up to ``compile_threads`` for the rest), so a pod deliberately overcommits:
    at the burst K*compile_threads asks for ~2x the box, which is right when
    the box is ours and idle. Halving the budget makes the burst ask for the
    machine instead of double it.

    RED: ``entry_workers`` had no posture, so a 32-core desktop sized K off
    (32-2)//2 = 15 exactly as a 32-core pod did.
    """
    pod = _width(36, vcpus=32, avail_gib=256, posture=FLEET)
    desk = _width(36, vcpus=32, avail_gib=256, posture=USER_MACHINE)
    assert pod.cpu_workers == 15
    assert desk.cpu_workers == 8, (
        "half of 32 cores, over 2 cores per entry worker — a quarter of the "
        "machine's cores as a steady ask, one whole machine at the burst")


@pytest.mark.parametrize(
    "vcpus,expected", [(2, 1), (4, 1), (8, 2), (16, 4), (32, 8)])
def test_the_core_budget_degrades_sanely_from_a_laptop_to_a_workstation(
    vcpus: int, expected: int,
) -> None:
    """Both bounds compose at the ends: on a 4-core laptop the serving headroom
    binds ((4-2)//2 = 1, i.e. the serial in-process path, which is the honest
    answer); on a 32-core workstation the half binds."""
    assert _width(
        36, vcpus=vcpus, avail_gib=256,
        posture=USER_MACHINE).cpu_workers == expected


def test_a_four_core_laptop_compiles_SERIALLY() -> None:
    """The product statement of the row above. K=1 is the pre-pgw#809 serial
    path and it is what a machine with four cores and a person on it should
    do."""
    assert _width(
        36, vcpus=4, avail_gib=16, posture=USER_MACHINE).workers == 1


def test_the_entry_CEILING_halves_on_a_user_machine() -> None:
    """K children mean K concurrent ``cc1plus``, K inductor caches being
    written and K working sets in the page cache — and here that disk and that
    page cache are the user's own. RED: one ceiling, 8, for every machine."""
    fat = _width(36, vcpus=128, avail_gib=512, posture=USER_MACHINE)
    assert fat.ceiling == compile_posture.USER_MACHINE_MAX_ENTRY_WORKERS == 4
    assert fat.workers == 4, (
        "a workstation big enough to ignore every other bound still stops at "
        "the posture ceiling")
    assert _width(
        36, vcpus=128, avail_gib=512,
        posture=FLEET).ceiling == aot_compile_pool.MAX_ENTRY_WORKERS


def test_a_caller_cap_below_the_posture_ceiling_still_wins() -> None:
    """Both narrow; neither widens. An operator forcing K=2 must not be
    widened to 4 by the posture, and the posture must not be widened to 8 by
    an operator asking for it."""
    assert _width(
        36, vcpus=128, avail_gib=512, posture=USER_MACHINE,
        limit=2).ceiling == 2
    assert _width(
        36, vcpus=128, avail_gib=512, posture=USER_MACHINE,
        limit=8).ceiling == 4


class _Nice:
    def __init__(self) -> None:
        self.levels: List[int] = []

    def __call__(self, inc: int) -> int:
        self.levels.append(int(inc))
        return int(inc)


def _drive_child_posture(
    monkeypatch: pytest.MonkeyPatch, posture: CompilePosture,
) -> Tuple[_Nice, List[bool]]:
    nice = _Nice()
    armed: List[bool] = []
    monkeypatch.setattr(os, "nice", nice)
    monkeypatch.setattr(
        aot_compile_pool, "arm_parent_death_signal",
        lambda: armed.append(True) or True)
    request = MintRequest(
        function="generate", modules=("m",), family="micro-diffusion",
        arm_token="arm2-x", target="/tmp/c.tar.gz", work_root="/tmp",
        report="/tmp/r.json", cfg=CompileSpec(),
        posture=posture)
    mint_child._install_posture(request)
    return nice, armed


def test_a_user_machine_mint_child_NICES_ITSELF(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The CPU half of politeness, and the only lever that actually preserves
    interactivity — a core reservation does not stop the scheduler putting a
    compile on the core the compositor wants; priority does.

    RED before this issue: pgw#1127 §6 recorded **zero** ``os.nice`` calls
    anywhere in ``src/``.
    """
    nice, _ = _drive_child_posture(monkeypatch, USER_MACHINE)
    assert nice.levels == [compile_posture.USER_MACHINE_NICE] == [19]
    assert compile_posture.current() == USER_MACHINE


def test_a_FLEET_mint_child_never_nices_itself(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The serving pod compiles exactly as fast as it did. A pod's mint already
    competes with a tenant whose reserves are held back explicitly; de-
    prioritising it on top of that would slow paid work for no gain."""
    nice, armed = _drive_child_posture(monkeypatch, FLEET)
    assert nice.levels == []
    assert armed == []
    assert compile_posture.current() == FLEET


def test_a_kernel_that_refuses_the_nice_leaves_a_RUDE_mint_not_a_DEAD_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Politeness is not correctness. A container without ``CAP_SYS_NICE``
    must still mint."""
    def _refuse(inc: int) -> int:
        raise OSError("not permitted")

    monkeypatch.setattr(os, "nice", _refuse)
    monkeypatch.setattr(
        aot_compile_pool, "arm_parent_death_signal", lambda: False)
    request = MintRequest(
        function="generate", modules=("m",), family="f", arm_token="",
        target="", work_root="", report="",
        cfg=CompileSpec(), posture=USER_MACHINE)
    mint_child._install_posture(request)   # must not raise
    assert compile_posture.current() == USER_MACHINE


def test_the_nice_is_applied_by_the_CHILD_and_never_through_a_preexec_fn(
) -> None:
    """A guard, and a deliberate coordination point with pgw#1111.

    ``aot_compile_pool.arm_parent_death_signal``'s own docstring already writes
    down why: a ``preexec_fn`` forces ``fork()`` instead of ``posix_spawn()``
    for a process that has live gRPC threads with ``pthread_atfork`` handlers,
    and only async-signal-safe work is legal in the forked child.

    Nicing in the mint child also covers strictly MORE. Since pgw#1080 every
    production mint is weight-free and ``aot_mint`` forces ``parallel=False``
    for those — so the entry pool spawns no children at all today and the
    compile runs in the mint child itself. A ``preexec_fn`` on the pool's spawn
    would nice a path that does not currently run; nice is inherited across
    ``fork``/``exec``, so one call in the child covers the serial compile, the
    entry children when parallelism returns, inductor's compile workers and
    every ``cc1plus`` under them.
    """
    for name in ("aot_compile_pool.py", "mint_process.py", "mint_child.py"):
        tree = ast.parse((SRC / name).read_text())
        passed = {
            kw.arg for node in ast.walk(tree) if isinstance(node, ast.Call)
            for kw in node.keywords
        }
        assert "preexec_fn" not in passed, (
            f"{name} must not spawn through preexec_fn — see "
            f"arm_parent_death_signal's docstring for the fork/gRPC hazard")


# ---------------------------------------------------------------------------
# 3. Memory — a mint that OOMs a desktop is worse than a slow one
# ---------------------------------------------------------------------------


def test_a_user_machine_leaves_more_HOST_RAM_alone() -> None:
    """``MemAvailable`` counts reclaimable page cache as free, so a pool sized
    against it evicts the working set of every application the user has open
    before it meets any limit — and on a desktop the OOM killer's most
    attractive target is the browser.

    RED: one reserve constant, 4 GiB, sized for a pod where the only thing
    being protected is a serving process whose RSS is already excluded.
    """
    pod = _width(36, vcpus=64, avail_gib=28, posture=FLEET)
    desk = _width(36, vcpus=64, avail_gib=28, posture=USER_MACHINE)
    # (28 - 4) / 3 = 8   vs   (28 - 8) / 3 = 6
    assert pod.mem_workers == 8
    assert desk.mem_workers == 6
    assert compile_posture.USER_MACHINE_RSS_RESERVE_BYTES == 8 * 1024**3


def test_a_sixteen_gig_desktop_under_load_mints_serially() -> None:
    """The case this bound exists for: a laptop with a browser open. 10 GiB
    available minus the reserve leaves nothing for a second entry."""
    assert _width(
        36, vcpus=16, avail_gib=10, posture=USER_MACHINE).workers == 1


# ---------------------------------------------------------------------------
# 4. The serving pod does not regress — proven against the OLD formula
# ---------------------------------------------------------------------------


def _fleet_width(
    entries: int, *, vcpus: int, avail: int, limit: int = 0,
) -> Dict[str, int]:
    """``entry_workers``' arithmetic, restated independently.

    Deliberately a re-implementation and not a call into the code under test:
    a non-regression test that asks the new code whether it changed can only
    ever answer no.

    pgw#1175 removed the device term from BOTH sides. The rest of pgw#1137's
    claim — that the FLEET posture is arithmetically the plain policy — is
    what these rows are for, and it is unaffected.
    """
    cpu_workers = max(
        1, (vcpus - aot_compile_pool.SERVING_HEADROOM_CPUS)
        // aot_compile_pool.CPUS_PER_ENTRY_WORKER)
    per_entry = aot_compile_pool.DEFAULT_ENTRY_PEAK_RSS_BYTES
    mem_workers = max(
        1, max(0, avail - aot_compile_pool.ENTRY_RSS_RESERVE_BYTES)
        // per_entry) if avail > 0 else 1
    ceiling = (
        min(aot_compile_pool.MAX_ENTRY_WORKERS, limit) if limit > 0
        else aot_compile_pool.MAX_ENTRY_WORKERS)
    return {
        "cpu_workers": cpu_workers, "mem_workers": mem_workers,
        "ceiling": ceiling,
        "workers": max(1, min(cpu_workers, mem_workers, ceiling, entries)),
    }


@pytest.mark.parametrize(
    "entries,vcpus,avail_gib,limit", [
        (18, 8, 32, 0),      # a modest serving pod
        (36, 64, 200, 0),    # a fat H100 pod
        (36, 128, 500, 0),   # the widest real pod
        (18, 16, 64, 0),     # a CPU-only cell (no card at all)
        (36, 64, 200, 3),    # an operator-forced narrow pool
        (4, 4, 8, 0),        # a small pod where RAM binds
    ])
def test_a_POD_width_is_arithmetically_the_plain_policy(
    entries: int, vcpus: int, avail_gib: float, limit: int,
) -> None:
    """§4.30's constraint the other way round: *"aggressive on a dedicated
    serving pod"*. Every bound, over a matrix spanning the real fleet."""
    got = _width(
        entries, vcpus=vcpus, avail_gib=avail_gib, posture=FLEET, limit=limit)
    want = _fleet_width(
        entries, vcpus=vcpus, avail=int(avail_gib * 1024**3), limit=limit)
    assert {
        "cpu_workers": got.cpu_workers, "mem_workers": got.mem_workers,
        "ceiling": got.ceiling, "workers": got.workers,
    } == want


def test_the_DEFAULT_posture_is_the_fleet_one_when_nothing_installed() -> None:
    """Nothing on the fleet path installs a posture, so the fallback IS the
    fleet policy — the same shape ``worker_goals.current()`` uses, and the
    reason no serving code path had to change."""
    assert compile_posture.current() == FLEET
    assert _width(
        36, vcpus=32, avail_gib=256, posture=FLEET).cpu_workers \
        == aot_compile_pool.entry_workers(
            36, vcpus=32, available_bytes=256 * 1024**3, device_lock=True).cpu_workers


def test_the_width_row_SAYS_which_posture_chose_K() -> None:
    """pgw#842's rule applied to the new bound: an unexplained K is a defect in
    itself, and a K held down for a human at a keyboard is a different fact
    from a K held down for a tenant."""
    desk = _width(36, vcpus=32, avail_gib=256, posture=USER_MACHINE)
    assert desk.facts()["posture"] == "user-machine"
    assert desk.facts()["nice"] == 19
    assert "§4.30 user-machine" in desk.reason
    assert _width(
        36, vcpus=32, avail_gib=256, posture=FLEET).facts()["posture"] == "fleet"


# ---------------------------------------------------------------------------
# 5. Honesty — the user can see the compile and what it costs
# ---------------------------------------------------------------------------


def test_the_user_is_told_what_the_compile_COSTS_before_it_starts() -> None:
    """*"A silent 20-minute CPU hog is a support ticket."* Four questions the
    notice has to answer: what is happening, will it happen again, what is it
    costing me, and may I stop it.

    RED: no user-facing surface existed on this path at all — the only signals
    were an activity addressed to a hub cozy-local does not have and a
    ``logger.info`` inside a subprocess.
    """
    facts = aot_compile_pool.CpuFacts(
        vcpus=32, basis="caller", os_cpu_count=32, affinity_cpus=32,
        quota_cores=-1.0)
    said = local_serve.compile_notice(
        "micro-diffusion", USER_MACHINE, cpu=facts,
        store_root=Path("/home/u/.cache/cozy/compile-cells"))
    assert "micro-diffusion" in said
    assert "ONCE" in said, "the user must know this is not every run"
    assert "/home/u/.cache/cozy/compile-cells" in said, (
        "where the result goes is what makes the promise checkable")
    assert "nice 19" in said and "32 cores" in said, (
        "what it is taking, in the units of the machine it is taking it from")
    assert "Ctrl-C is safe" in said


def test_progress_reaches_the_user_WHILE_the_compile_runs() -> None:
    """A cadence, not a burst: the frames arrive at wildly different rates (one
    per export, one per compiled entry, then silence through a single long link
    step) and a user watching a machine they can no longer type on needs a
    steady "still working" line."""
    clock = [0.0]
    lines: List[str] = []
    watch = local_serve._Progress(
        "micro-diffusion", say=lines.append, interval_s=10.0,
        clock=lambda: clock[0])

    watch(MintFrame(phase="trace_graph", note="exporting"))
    assert len(lines) == 1 and "micro-diffusion" in lines[0]

    clock[0] = 3.0
    watch(MintFrame(phase="compile_entries", step=1, total=18))
    assert len(lines) == 1, "throttled: three seconds is not a new line"

    clock[0] = 45.0
    watch(MintFrame(phase="compile_entries", step=7, total=18))
    assert len(lines) == 2
    assert "7/18" in lines[1] and "0m45s" in lines[1], (
        "how far along, and how long it has been — the two numbers a user "
        "deciding whether to wait actually needs")


def test_the_terminal_watcher_never_displaces_the_HUB_activity() -> None:
    """The fleet's own reporting is unchanged: the watcher is an ADDITIONAL
    sink, and a fleet mint passes none."""
    phases: List[Tuple[str, int, int]] = []
    notes: List[str] = []

    class _Act:
        def phase(self, p: str, step: int = 0, total: int = 0) -> None:
            phases.append((p, step, total))

        def note(self, n: str) -> None:
            notes.append(n)

    seen: List[MintFrame] = []
    frame = MintFrame(phase="compile_entries", step=2, total=18, note="hi")

    mint_supervisor._on_frame(_Act())(frame)          # the fleet shape
    assert phases == [("compile_entries", 2, 18)] and notes == ["hi"]

    mint_supervisor._on_frame(_Act(), seen.append)(frame)
    assert phases == [("compile_entries", 2, 18)] * 2
    assert seen == [frame]


# ---------------------------------------------------------------------------
# 6. Interruptibility — a stopped mint costs nothing and leaves nothing running
# ---------------------------------------------------------------------------


def test_a_user_machine_mint_child_DIES_WITH_ITS_PARENT(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``mint_process`` spawns the child with ``start_new_session=True``, so it
    holds its OWN session and a terminal's Ctrl-C or SIGHUP never reaches it.

    On a pod that is correct — the parent reaps the group deliberately when it
    abandons a mint. On a desktop the parent is a CLI a human closes, and
    without ``PR_SET_PDEATHSIG`` a closed terminal leaves a full-speed compile
    tree running with nobody left to reap it. That is precisely the "my machine
    is at a crawl and I don't know why" ticket this issue exists to prevent.

    RED: ``arm_parent_death_signal`` had exactly one caller, the entry-pool
    child, and the mint child armed nothing.
    """
    _, armed = _drive_child_posture(monkeypatch, USER_MACHINE)
    assert armed == [True]


def test_a_stopped_local_mint_reuses_only_the_canonical_tcg_store(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ctrl-C may cost the in-flight class, never a second resume substrate.

    Every completed class is already in TCG's canonical tensorfs CAS. The
    worker request therefore carries no private bank path, and the compile
    child opens the same production engine factory as import and serving.
    """
    from gen_worker.models import cache_paths

    workdir = Path("/tmp/pgw1137/child-1")
    request = mint_process.build_request(
        mint_process.MintTask(
            pending=_Pending(), pipe=None, function="generate",
            modules=("m",), posture=USER_MACHINE),
        workdir=workdir)
    assert "resume" not in MintRequest.__struct_fields__
    assert not hasattr(request, "resume")

    opened: List[Optional[Path]] = []
    engine = object()
    monkeypatch.setattr(
        cache_paths,
        "open_worker_engine",
        lambda root=None: (opened.append(root), engine)[1],
    )
    monkeypatch.setattr(
        compile_cache,
        "runtime_key",
        lambda: {"sm": "cpu"},
    )
    monkeypatch.setattr(
        compile_cache, "toolchain_digest", lambda: (("torch", "test"),),
    )

    actual, _runtime = aot_compile_child._tcg_runtime()
    assert actual is engine
    assert opened == [None], "production compile invented a private CAS root"


def test_the_local_notice_and_the_local_posture_cannot_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One posture object feeds both the policy and the sentence describing it,
    so the numbers a user is shown are the numbers the pool used."""
    facts = aot_compile_pool.CpuFacts(
        vcpus=8, basis="caller", os_cpu_count=8, affinity_cpus=8,
        quota_cores=-1.0)
    said = local_serve.compile_notice("f", USER_MACHINE, cpu=facts,
                                      store_root=Path("/r"))
    width = _width(36, vcpus=8, avail_gib=256, posture=USER_MACHINE)
    assert f"{width.ceiling} parallel worker(s)" in said
    assert f"nice {width.posture.nice_level()}" in said


def _module_names(path: Path) -> List[str]:
    tree = ast.parse(path.read_text())
    out: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            out.append(node.module or "")
            out += [a.name for a in node.names]
        elif isinstance(node, ast.Import):
            out += [a.name for a in node.names]
    return out


def test_the_politeness_wiring_adds_no_transport_to_the_local_serve_entry(
) -> None:
    """pgw#1127 §4's fence, re-asserted against THIS issue's imports: the local
    serve entry gained ``aot_compile_pool``, ``compile_posture`` and
    ``local_cell_store``, and none of them may drag a publisher in."""
    names = _module_names(SRC / "local_serve.py")
    assert "compile_posture" in names and "aot_compile_pool" in names
    for banned in ("cell_publish", "cas_client", "httpx", "requests",
                   "aiohttp"):
        assert banned not in names


def test_the_posture_module_holds_POLICY_and_nothing_else() -> None:
    """Every politeness term lives in one place, so a reader can price the
    whole trade without walking four modules — and so the notice, the pool and
    the child cannot each grow their own idea of what polite means."""
    posture = CompilePosture(user_machine=True)
    assert posture.nice_level() == 19
    assert posture.entry_ceiling(8) == 4
    assert posture.entry_ceiling(2) == 2, "never widens"
    assert posture.rss_reserve_bytes(4 * 1024**3) == 8 * 1024**3
    assert posture.rss_reserve_bytes(16 * 1024**3) == 16 * 1024**3, \
        "never shrinks"
    assert posture.cpu_budget_cores(32, headroom=2) == 16
    assert posture.cpu_budget_cores(4, headroom=2) == 2

    fleet = CompilePosture()
    assert fleet.nice_level() == 0
    assert fleet.entry_ceiling(8) == 8
    assert fleet.rss_reserve_bytes(4 * 1024**3) == 4 * 1024**3
    assert fleet.cpu_budget_cores(32, headroom=2) == 30


_ = Optional, Iterator, Any
