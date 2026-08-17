"""The cozy-local serve entry: the fleet arming brain, no sink.

Untrusted hardware (community cloud, cozy-local) mints for ITSELF: local cell,
local repo-CAS, reused across its own boots — never uploaded, never requested.

This is deliberately a MODULE rather than a line in ``cli/run.py``, because
the never-publish property has to be STRUCTURAL: ``publisher=None`` at one
call site is a convention, whereas this module — the ONLY arming entry the
local CLI has — contains no ``CellPublisher`` construction, no publish call
and no transport import at all, which a test reads out of the source tree. The
obligation's terminus is ``fleet_cells.keep_self_mint_local``, which takes no
publisher and therefore cannot grow into one.

It adds no mint code (``mint_supervisor`` is used unchanged — the same
supervisor the fleet's serving parent drives), no second key scheme, no coordinator,
no mint-request in any address form, no trust self-declaration and no env
behaviour switch. A cozy-local box learns it keeps its own cells from
``no_publish_sink_reason``'s ``no_publish_sink`` — a fact about the SINK,
never a claim about itself.
"""

from __future__ import annotations

import asyncio
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from . import activity as activity_mod
from . import aot_compile_pool
from . import compile_cache as cc
from . import compile_posture, fleet_cells, handler_proof
from . import local_cell_store, mint_supervisor
from .child_contract import MintFrame, MintSlot
from .serving_facts import FactsUnavailable

logger = logging.getLogger(__name__)

#: Seconds between progress lines while a compile runs. A mint is minutes-to-
#: an-hour of work, so a line a second would be noise and a line a minute
#: would read as a hang. Ten is the same cadence the fleet's own beat uses.
NOTICE_INTERVAL_S = 10.0


@dataclass(frozen=True)
class LocalMintContext:
    """Everything a mint child needs that the local serve process knows.

    The same three facts the executor hands ``mint_supervisor.MintTask``: WHICH
    routable function, WHICH modules to rediscover it in, and the parent's own
    resolution of every setup slot (identity + bytes + composition, as one
    value — see ``child_contract.MintSlot``). A context with no function or
    no modules cannot be minted from and is declared ``incomplete``: the child
    re-runs discovery, so a module list it cannot import is a silent
    "compiles nothing, forever".
    """

    function: str = ""
    modules: Tuple[str, ...] = ()
    slots: Mapping[str, MintSlot] = field(default_factory=dict)
    device: Optional[int] = None

    @property
    def incomplete(self) -> str:
        if not self.function:
            return "no routable function name"
        if not self.modules:
            return "no declaring module for the child to rediscover"
        return ""


@dataclass
class DeferredMint:
    """A mint this machine owes, held until the endpoint's handler has RUN.

    The mint child does not prove the handler itself: on a weight-free mint
    that would mean materialising a whole checkpoint in a process that holds
    none. The proof belongs to the resident parent — but on cozy-local the
    parent reaches ``enable_compiled`` from inside the SLOT LOAD, before
    ``setup()`` has bound the pipeline to the endpoint instance,
    so there is no handler to run yet. So the mint waits: arm from the local
    store now (that must stay in the load, or the endpoint's ``setup()`` sees
    an unarmed pipeline), and mint after ``setup()`` and ``warmup()`` have
    returned, when one real forward can be run on the resident weights.

    This is the same ORDER the fleet already boots in — eager-first, setup
    complete, handler warmed, then the background mint.
    """

    pipe: Any
    pending: Any
    mint: "LocalMintContext"


def enable_compiled(
    pipe: Any,
    cfg: Any,
    cache_dir: Optional[Path] = None,
    *,
    mint: Optional[LocalMintContext] = None,
    defer: Optional[List["DeferredMint"]] = None,
) -> bool:
    """Arm ``pipe`` from this machine's own cells, minting one if it has none.

    The ordering is ``fleet_cells``' and is not restated here: delivered
    artifact -> in-process ledgers -> **this machine's local store** ->
    delegated mint. On a local-store hit no network is touched and no mint is
    opened, which is the whole of compile-once-run-forever.

    Returns True when this pipeline is now serving compiled. False is never an
    error: a machine with no CUDA, a family with no export declaration, or a
    mint that could not fit on the card all serve eager, exactly as they did
    before — except on a mandatory (w8a8/w4a4) lane, whose typed refusal
    ``fleet_cells`` raises and this function deliberately does not catch.
    """
    outcome = fleet_cells.enable_compiled(
        pipe, cfg, cache_dir,
        # §4.28, and the reason this module exists: user hardware is untrusted
        # tier by definition, so there is no sink to pass and never will be.
        publisher=None,
    )
    if outcome.armed:
        return True
    pending = outcome.self_mint
    if not isinstance(pending, fleet_cells.PendingSelfMint):
        # An eager exit with a classified reason (`no_toolchain`, a declined
        # family, a quarantined identity). `fleet_cells` already named it.
        return False
    if mint is None or mint.incomplete:
        # A pending nobody can drive is an obligation with no terminus. End it
        # here rather than leaving the capture dir and the ledger entry behind
        # for a run that will never come.
        fleet_cells.abandon_self_mint(pending)
        logger.warning(
            "local-serve: %s opened a mint this process cannot drive (%s); "
            "serving eager", pending.family,
            mint.incomplete if mint is not None else "no mint context")
        return False
    if defer is not None:
        # The handler cannot run yet — see `DeferredMint`.
        defer.append(DeferredMint(pipe=pipe, pending=pending, mint=mint))
        return False
    return _mint_here(pipe, pending, mint)


def run_deferred(
    deferred: List[DeferredMint], *, instance: Any, specs: Any,
    execution_lane: str = "", config: Optional[Dict[str, Any]] = None,
) -> None:
    """Prove the handler on the resident pipeline, then mint what is owed.

    ``setup()`` and ``warmup()`` have returned, so the endpoint instance is
    bound to the pipeline this machine will serve from, and ONE warm forward
    through its own handler proves it — on real checkpoint values, in the
    process that already holds them, for the price of one forward instead of a
    second copy of the model.

    A handler that does not run mints NOTHING and says so: a cell must not
    seal for a handler that cannot serve, and that is reached before a compile
    is paid for rather than after.
    """
    if not deferred:
        return
    for owed in deferred:
        try:
            handler_proof.prove(
                instance, specs, owed.mint.function,
                execution_lane=execution_lane, config=config,
                origin=f"cozy-local:{owed.pending.family}")
        except handler_proof.HandlerProofFailed as exc:
            _say(
                f"{owed.pending.family} is not compiled on this machine: the "
                f"endpoint's own handler does not run ({exc}). Serving eager.")
            fleet_cells.abandon_self_mint(owed.pending)
            continue
        _mint_here(owed.pipe, owed.pending, owed.mint)


def _say(line: str) -> None:
    """One line to the person at the keyboard.

    ``print`` to stderr and not ``logger``: the CLI configures logging for the
    endpoint's own output, a mint runs before any of that matters, and every
    existing progress signal on this path already goes somewhere a user does
    not look (an activity addressed to a hub that cozy-local does not have, or
    a ``logger.info`` inside a subprocess).
    """
    print(f"cozy: {line}", file=sys.stderr, flush=True)


def compile_notice(
    family: str, posture: compile_posture.CompilePosture,
    *, cpu: Optional[aot_compile_pool.CpuFacts] = None,
    store_root: Optional[Path] = None,
) -> str:
    """What a user is told BEFORE a compile starts — §4.30's honesty half.

    It has to answer four questions a support ticket would otherwise ask:
    *what is happening*, *will it happen again*, *what is it costing me right
    now*, and *may I stop it*. The cost figures are this machine's own
    (``cpu_facts`` reads the same cgroup/affinity/host triple the pool does),
    and they are stated as a CEILING because the mint child narrows further on
    memory and VRAM — overstating what we took would be the same defect as
    saying nothing.
    """
    facts = cpu if cpu is not None else aot_compile_pool.cpu_facts()
    vcpus = max(1, int(facts.vcpus))
    cores = max(
        1, posture.cpu_budget_cores(
            vcpus, headroom=aot_compile_pool.SERVING_HEADROOM_CPUS))
    workers = posture.entry_ceiling(aot_compile_pool.MAX_ENTRY_WORKERS)
    root = store_root if store_root is not None else local_cell_store.store_root()
    reserve = posture.rss_reserve_bytes(
        aot_compile_pool.ENTRY_RSS_RESERVE_BYTES) // 1024**3
    return (
        f"compiling {family} for this machine — this happens ONCE; every "
        f"later run of this endpoint arms from {root} with no compile and no "
        f"network. It takes a while.\n"
        f"      This is your machine, so the compile is polite: lowest CPU "
        f"priority (nice {posture.nice_level()}), at most {workers} parallel "
        f"worker(s) sized against {cores} of your {vcpus} cores, and "
        f"{reserve} GiB of RAM left alone.\n"
        f"      Ctrl-C is safe — finished work is kept and the next run "
        f"picks up where this one stopped."
    )


class _Progress:
    """Renders the child's frames onto the user's terminal, throttled.

    Deliberately a throttle on TIME and not on frame count: the frames arrive
    at wildly different rates (one per export, one per compiled entry, then
    silence through a single long link step), and a user watching a machine
    they can no longer type on needs a steady "still working" cadence, not a
    burst followed by nothing.
    """

    def __init__(
        self, family: str, *, say: Any = _say,
        interval_s: float = NOTICE_INTERVAL_S, clock: Any = time.monotonic,
    ) -> None:
        self.family = family
        self.say = say
        self.interval_s = float(interval_s)
        self.clock = clock
        self.started = clock()
        self._last: Optional[float] = None
        self.lines = 0

    def elapsed(self) -> str:
        secs = int(max(0.0, self.clock() - self.started))
        return f"{secs // 60}m{secs % 60:02d}s"

    def __call__(self, frame: MintFrame) -> None:
        now = self.clock()
        if self._last is not None and now - self._last < self.interval_s:
            return
        self._last = now
        step = (
            f" {frame.step}/{frame.total}"
            if frame.total > 0 else "")
        self.say(
            f"[{self.family}] {frame.phase or 'compiling'}{step} — "
            f"{self.elapsed()} elapsed")
        self.lines += 1


def _mint_here(
    pipe: Any, pending: "fleet_cells.PendingSelfMint", mint: LocalMintContext,
) -> bool:
    """Supervise this machine's own compile children and keep what they pack.

    Identical to the executor's ``_supervise_mint`` in everything that decides
    correctness — same ``mint_supervisor.supervise``, same compile children,
    same ``adopt_delegated_mint`` gate — and different in the three things a
    desktop does not have: there is no serving loop to keep beating (this call
    is the boot, and it blocks), there is nowhere to publish, so the obligation
    ends at ``keep_self_mint_local`` instead of at the publish gate, and there
    is a PERSON on the machine — so §4.30's posture is declared here, and what
    the compile is doing to their computer is said out loud.
    """
    result: Optional[mint_supervisor.SupervisedResult] = None
    family = str(pending.family)
    proof = handler_proof.provenance(mint.function)
    # §4.30: the ONE site in the tree that declares a user-machine posture. It
    # is stated here, never derived from `publisher is None` or from
    # `local_cell_store.trust_class()` — both are facts about the SINK, and a
    # community-cloud pod matches them while having no human on it.
    posture = compile_posture.USER_MACHINE
    _say(compile_notice(family, posture))
    watch = _Progress(family)
    with activity_mod.running(activity_mod.KIND_SELF_MINT_COMPILE) as act:
        try:
            result = _drive(mint_supervisor.supervise(
                mint_supervisor.MintTask(
                    pending=pending,
                    pipe=pipe,
                    function=mint.function,
                    modules=mint.modules,
                    slots=dict(mint.slots),
                    # Cell IDENTITY's lane, the same probe the mint's own
                    # `stamp_lane` memoizes — so what this machine looks up on
                    # its next boot is what this mint stamps.
                    weight_lane=cc.cell_base_execution_lane(pipe),
                    device=mint.device,
                    posture=posture,
                    handler_proof=proof,
                ),
                act=act, watch=watch))
        except Exception as exc:  # noqa: BLE001 — a mint must never kill a serve
            logger.warning(
                "local-serve: the mint for %s failed (%s: %s); serving eager "
                "and keeping nothing", pending.family, type(exc).__name__, exc)
    if result is not None and result.ok:
        _say(
            f"compiled {family} in {watch.elapsed()} — kept at "
            f"{local_cell_store.store_root()}; later runs arm from it.")
    else:
        _say(
            f"{family} is not compiled on this machine ({watch.elapsed()} "
            f"spent); serving eager. Finished compile work, if any, is kept "
            f"for the next run.")
    if result is not None and result.ok:
        # The cell is ALREADY in this machine's store — `adopt_delegated_mint`
        # put it there before any publish could be attempted, which is what
        # makes a process killed here cost nothing.
        fleet_cells.keep_self_mint_local(pending)
        return True
    if not fleet_cells.terminus_of(pending):
        # Every obligation ends somewhere nameable. `supervise` abandons its
        # own failures; an ABANDONED one it deliberately leaves open for a
        # caller that might drive it again.
        fleet_cells.abandon_self_mint(pending)
    logger.info(
        "local-serve: %s has no cell on this machine yet (%s); serving eager",
        pending.family,
        (result.detail or result.status) if result is not None else "mint failed")
    return False


def _drive(coro: Any) -> Any:
    """Run one coroutine to completion from this synchronous serve path.

    ``cozy run`` and ``cozy serve`` both call ``run_setup`` off any event loop
    — the CLI's own async dispatch runs inside ``asyncio.run`` per request,
    never around setup — so this is the ordinary case. It is asserted rather
    than assumed: a loop already running here would mean the mint is being
    driven from inside a request, which is the one place a 40-minute compile
    must not be.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    coro.close()
    raise RuntimeError(
        "local-serve: a delegated mint was driven from inside a running event "
        "loop — the local serve path arms at setup, never on the request path")


def mint_context(
    *, function: str, module: str, slots: Mapping[str, MintSlot],
    device: Optional[int] = None,
) -> LocalMintContext:
    """Build a :class:`LocalMintContext` from what the local CLI holds.

    ``module`` is the endpoint's own declaring module, which is exactly what
    ``executor._mint_modules`` hands a fleet mint: ``registry`` walks it and
    its submodules, so the child rediscovers this class AND its class-scoped
    siblings without the parent serializing anything live.
    """
    named = str(module or "").strip()
    return LocalMintContext(
        function=str(function or ""),
        modules=(named,) if named else (),
        slots=dict(slots),
        device=device,
    )


def slot_map(
    paths: Mapping[str, str],
    bindings: Mapping[str, Any],
) -> Dict[str, MintSlot]:
    """The parent's resolution of every setup slot, as the child reads it.

    A slot the local run did not resolve is ABSENT, never a present entry with
    a hole in it — ``MintSlot`` refuses to be constructed without both halves,
    and ``child_preflight.assert_slots_resolvable`` refuses a declared,
    non-optional slot that never arrived.

    pgw#1333: a hub-less local serve consults no catalog, so it has no serving
    facts to forward and says exactly that. A local run of a function that
    DECLARES ``objectives=`` therefore refuses by name — which is honest: the
    declaration is checked against evidence, and there is none here.
    """
    out: Dict[str, MintSlot] = {}
    for slot, path in paths.items():
        binding = bindings.get(slot)
        if binding is None or not str(path or ""):
            continue
        out[slot] = MintSlot(
            ref=binding, path=str(path),
            facts=FactsUnavailable(
                owed_by="a hub-less local serve (no catalog resolves this "
                        "slot's objective/distilled facts)"))
    return out


__all__ = [
    "LocalMintContext", "compile_notice", "enable_compiled", "mint_context",
    "slot_map",
]
