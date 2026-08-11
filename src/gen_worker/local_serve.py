"""pgw#1127 / §4.28 — the cozy-local serve entry: the fleet arming brain, no sink.

DESIGN-RULINGS §4.28 (Paul, 2026-08-10): *"Untrusted hardware (community
cloud, cozy-local) mints for ITSELF: local cell, local repo-CAS, reused
across its own boots — never uploaded, never requested."*

pgw#1096 built the store that clause needs (``local_cell_store``) and wired it
into ``fleet_cells._arming_policy``, local-first, before any mint is opened.
It left ONE thing owed, and the omission made the whole build unreachable from
the machine it was written for: ``cli/run.py`` still armed through
``local_cells`` — the JIT path — so ``_arming_policy`` was never entered from
``cozy serve`` and cozy-local got compile-once-run-forever on JIT only, on the
recipe pgw#1086 wave 1 deletes.

This module is that re-point, and it is deliberately a MODULE rather than a
line in ``cli/run.py``: the never-publish property has to be structural.

WHAT IS STRUCTURAL HERE, and what merely happens to be true
-----------------------------------------------------------
``publisher=None`` at one call site is a convention. What pins the property is
that this module — the ONLY arming entry the local CLI has — contains no
``CellPublisher`` construction, no publish call and no transport import at all,
and ``tests/test_local_serve_no_publisher_pgw1127.py`` reads that out of the
source tree. The obligation's terminus is ``fleet_cells.keep_self_mint_local``,
which takes no publisher and therefore cannot grow into one.

WHAT THIS DOES NOT ADD
----------------------
No mint code (the delegated ``mint_delegate`` -> ``mint_process`` ->
``mint_child`` chain is used unchanged, weight-free since pgw#1080), no second
key scheme, no coordinator, no mint-request in any address form, no trust
self-declaration, and no env behaviour switch. A cozy-local box learns it keeps
its own cells from ``local_keep_reason``'s ``no_publish_sink`` — a fact about
the SINK, never a claim about itself.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

from . import activity as activity_mod
from . import compile_cache as cc
from . import fleet_cells, mint_delegate
from .mint_process import MintSlot

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LocalMintContext:
    """Everything a mint child needs that the local serve process knows.

    The same three facts the executor hands ``mint_delegate.MintTask``: WHICH
    routable function, WHICH modules to rediscover it in, and the parent's own
    resolution of every setup slot (identity + bytes + pgw#617 composition, as
    one value — see ``mint_process.MintSlot``). A context with no function or
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


def enable_compiled(
    pipe: Any,
    cfg: Any,
    cache_dir: Optional[Path] = None,
    *,
    mint: Optional[LocalMintContext] = None,
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
        # A pending nobody can drive is the pgw#815 shape: an obligation with
        # no terminus. End it here rather than leaving the capture dir and the
        # ledger entry behind for a run that will never come.
        fleet_cells.abandon_self_mint(pending)
        logger.warning(
            "local-serve: %s opened a mint this process cannot drive (%s); "
            "serving eager", pending.family,
            mint.incomplete if mint is not None else "no mint context")
        return False
    return _mint_here(pipe, pending, mint)


def _mint_here(
    pipe: Any, pending: "fleet_cells.PendingSelfMint", mint: LocalMintContext,
) -> bool:
    """Run the delegated child mint on THIS machine and keep what it produces.

    Identical to the executor's ``_delegated_mint_run`` in everything that
    decides correctness — same ``build_cell``, same child, same
    ``adopt_delegated_mint`` gate — and different in the two things a desktop
    does not have: there is no serving loop to keep beating (this call is the
    boot, and it blocks), and there is nowhere to publish, so the obligation
    ends at ``keep_self_mint_local`` instead of at the publish gate.
    """
    result: Optional[mint_delegate.DelegatedResult] = None
    with activity_mod.running(activity_mod.KIND_SELF_MINT_COMPILE) as act:
        try:
            result = _drive(mint_delegate.build_cell(
                mint_delegate.MintTask(
                    pending=pending,
                    pipe=pipe,
                    function=mint.function,
                    modules=mint.modules,
                    slots=dict(mint.slots),
                    # Cell IDENTITY's lane, the same probe the mint's own
                    # `stamp_lane` memoizes — so what this machine looks up on
                    # its next boot is what this mint stamps (pgw#686).
                    weight_lane=cc.cell_base_execution_lane(pipe),
                    device=mint.device,
                ),
                act=act))
        except Exception as exc:  # noqa: BLE001 — a mint must never kill a serve
            logger.warning(
                "local-serve: the mint for %s failed (%s: %s); serving eager "
                "and keeping nothing", pending.family, type(exc).__name__, exc)
    if result is not None and result.ok:
        # The cell is ALREADY in this machine's store — `adopt_delegated_mint`
        # put it there before any publish could be attempted, which is what
        # makes a process killed here cost nothing (th#1643's SUNK case).
        fleet_cells.keep_self_mint_local(pending)
        return True
    if not fleet_cells.terminus_of(pending):
        # pgw#815: every obligation ends somewhere nameable. `build_cell`
        # abandons its own failures; a DECLINED one (no room on the card) it
        # deliberately leaves open for a caller that might have another.
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
    component_paths: Optional[Mapping[str, Mapping[str, str]]] = None,
) -> Dict[str, MintSlot]:
    """The parent's resolution of every setup slot, as the child reads it.

    A slot the local run did not resolve is ABSENT, never a present entry with
    a hole in it — ``MintSlot`` refuses to be constructed without both halves,
    and ``mint_child.assert_slots_resolvable`` refuses a declared,
    non-optional slot that never arrived.
    """
    out: Dict[str, MintSlot] = {}
    for slot, path in paths.items():
        binding = bindings.get(slot)
        if binding is None or not str(path or ""):
            continue
        out[slot] = MintSlot(
            ref=binding, path=str(path),
            component_paths=dict((component_paths or {}).get(slot) or {}),
        )
    return out


__all__ = [
    "LocalMintContext", "enable_compiled", "mint_context", "slot_map",
]
