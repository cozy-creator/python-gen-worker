"""pgw#1371: the CALLER. WHEN a serving worker fills its own holes.

:mod:`gen_worker.serving.mint` is the mint — fetch, compile, publish, arm,
and the progress-gated guard over all of it. It landed with **no production
caller**: every reference to ``mint_holes`` at ``origin/master`` was a
definition or a re-export, so no pod could mint anything and the typed
``self_mint_wedged`` / ``self_mint_arm_missed`` events pgw#985 added were
unreachable in production. This module is that caller.

**THE TRIGGER POLICY, stated so it is not re-derived.** The mint starts the
moment the adopt session has REGISTERED ITS HOLES — boot for the
:class:`~gen_worker.serving.host.EndpointHost` shape (``setup()`` loads
eagerly, so the holes exist when it returns), first model load for the
:class:`~gen_worker.serving.serve_loop.ServeLoop` shape (residency is lazy,
so the holes exist when the first request's load runs ``ctx.compile``). One
policy, both shapes, and it is the earliest instant at which the work-list is
KNOWN.

Three alternatives were considered and rejected, each for a stated reason:

* **After the first request.** A pre-warm pod is an ordinary serving pod that
  nobody sends traffic to (§4.28's restored shape: an operator warms a
  (lane x sm) by booting one there). A trigger that needs a request never
  fires on exactly the pod whose whole job is to mint.
* **After a boot-idle delay.** A delay is a clock, and a clock against work is
  the anti-pattern this tier already paid for twice. The contention mechanism
  is the one mint.py implements and measured: ``entry_workers <= vcpus -
  SERVING_RESERVE_CPUS`` plus a niced mint tree. A sleep would add a magic
  number on top of a mechanism that does not need one.
* **On the request path.** Never. The mint compiles in child processes off a
  daemon thread; the serving loop is not involved at any point.

**AND IT REPORTS, COUNTED.** The mint opens a RUNNING activity of kind
``self_mint_compile`` — the vocabulary the hub already consumes
(``SelfMintActivityRunning``, the stall monitor, the autoscale cold-window) —
carrying a ``compile:self_mint_graphs`` counter that advances once per landed
graph, beaten on this module's own cadence from its own thread. The activity
is constructed rather than :func:`~gen_worker.activity.begin`-ed on purpose:
``begin`` swaps the module-global ``_current`` and would strand or steal a
concurrently open request activity on the same process.

:meth:`SelfMint.status` never has a silent state. ``not_armed`` (nothing has
asked for a mint) and ``nothing_to_mint`` (a mint was asked for and the adopt
session had zero holes) are DIFFERENT answers, and neither renders as a zero
count: an instrument that cannot tell "not running" from "nothing to do" is
the C11 defect this tier has been bitten by, so both are named states here.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

from .. import activity as activity_mod
from .. import compile_posture
from .. import progress as progress_mod
from .mint import (
    DEFAULT_SILENCE_WINDOW_S,
    SERVING_RESERVE_CPUS,
    Compiler,
    MintOutcome,
    MintedHole,
    hole_work_list,
    mint_holes,
)

logger = logging.getLogger(__name__)


def _artifacts_root() -> Path:
    """The box artifacts cache — pgw#1526's ONE address.

    Imported inside the function, not at module scope: this module is inside
    the adopt-only serve closure (pgw#1328) and must not acquire the CLI
    package merely to learn a path. The answer still comes from
    `cli/workspace.py` and is not restated here — a second spelling of a store
    address is how a build and a lookup end up at different directories.
    """
    from ..cli.workspace import artifacts_root

    return artifacts_root()


#: The counted observable. ``compile:`` prefixes into
#: :data:`gen_worker.progress.STALL_WINDOW_S`'s 600s window, which is the
#: right order of magnitude for one inductor compile of one graph specialization.
COUNTER = "compile:self_mint_graphs"

#: The activity kind. Deliberately the one the hub ALREADY consumes rather
#: than a new spelling: `self_mint_compile` is in the hub's
#: `selfMintActivityKinds`, so a serving-side mint is visible to the stall
#: monitor, the cap-turnover test and the autoscale cold-window with no
#: hub-side change at all.
KIND = activity_mod.KIND_SELF_MINT_COMPILE

#: The event a worker emits when it was ASKED to mint and had nothing to do.
#: A hub-known event kind, and the reason `nothing_to_mint` is never silent.
KIND_SKIPPED = "self_mint_skipped"

#: Beat cadence for the counter-carrying RUNNING update. Not a budget and not
#: a deadline — the hub's stall clock runs on counter NON-ADVANCEMENT, so this
#: only decides how often the truth is re-stated.
DEFAULT_BEAT_S = 10.0

#: Every state :meth:`SelfMint.status` can report. No others exist.
NOT_ARMED = "not_armed"
NOTHING_TO_MINT = "nothing_to_mint"
UNAVAILABLE = "unavailable"
MINTING = "minting"
COMPLETE = "complete"
PARTIAL = "partial"
CONDEMNED = "condemned"


@dataclass(frozen=True)
class SelfMintStatus:
    """What this worker's own mint is doing, or why it is not doing it.

    ``state`` is the whole answer and is never inferred from the counts: a
    reader must be able to tell a worker that was never asked to mint from one
    that was asked and found every graph already adopted, and both from one
    that failed to start. All three would otherwise render ``0 / 0``.
    """

    state: str = NOT_ARMED
    holes: int = 0
    landed: int = 0
    failed: int = 0
    reason: str = ""
    elapsed_s: float = 0.0

    @property
    def remaining(self) -> int:
        """Holes neither landed nor failed. Meaningless unless ``holes``."""
        return max(0, self.holes - self.landed - self.failed)

    @property
    def running(self) -> bool:
        return self.state == MINTING

    def facts(self) -> Dict[str, Any]:
        return {
            "state": self.state,
            "holes": self.holes,
            "landed": self.landed,
            "failed": self.failed,
            "remaining": self.remaining,
            "reason": self.reason,
            "elapsed_s": round(self.elapsed_s, 3),
        }


@dataclass
class SelfMint:
    """One serving worker's background mint, and the policy that starts it.

    Every input is a seam the wiring states: where artifacts land, the CAS the
    compile child admits into, and — for the local no-GPU proof — an explicit
    ``compiler``. Nothing here reads a global, and nothing here decides
    whether the pod HAS holes; :meth:`arm` is handed a booted host and reads
    its work-list.
    """

    store: Any = None
    #: pgw#1526: the box cache, from `cli/workspace.py`. This is the
    #: SERVE-coordinated half of the same work ledger `gen-worker compile`
    #: drives, so the two must resolve the same address — a background
    #: mint writing where the explicit verb does not look is a hole that
    #: refills forever and reads as a cache miss every boot.
    artifacts_dir: Path = field(default_factory=_artifacts_root)
    cas_dir: Optional[Path] = None
    target_arch: str = ""
    toolchain: Mapping[str, str] = field(default_factory=dict)
    #: The local seam (no GPU, no inductor). ``None`` runs the production
    #: one-child-process-per-graph compile.
    compiler: Optional[Compiler] = None
    program_source: Optional[Callable[[str, Path], Path]] = None
    posture: Optional[compile_posture.CompilePosture] = None
    reserve: int = SERVING_RESERVE_CPUS
    window_s: float = DEFAULT_SILENCE_WINDOW_S
    beat_s: float = DEFAULT_BEAT_S
    vcpus: Optional[int] = None

    def __post_init__(self) -> None:
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._done = threading.Event()
        self._status = SelfMintStatus()
        self._started_at = 0.0
        #: The armed work-list, read ONCE in :meth:`arm` (pgw#1564). Declared,
        #: not a getattr hedge — pgw#1534 is what a hedge costs.
        self._work_list: Tuple[Any, ...] = ()

    # -- the trigger --------------------------------------------------------

    def arm(self, host: Any) -> SelfMintStatus:
        """Start this worker's mint, ONCE, off the serving path.

        Called the instant the adopt session has registered its holes. Never
        raises: a worker must never die of its own background mint, so a
        refusal to start is a named ``unavailable`` status and an eager pod.
        Idempotent — a second call returns the live status rather than opening
        a second mint against the same work-list.
        """
        with self._lock:
            if self._thread is not None:
                return self._status
            try:
                holes = hole_work_list(host)
            except Exception as exc:  # noqa: BLE001 — a boot never dies here
                self._status = SelfMintStatus(
                    state=UNAVAILABLE, reason=f"{type(exc).__name__}: {exc}")
                logger.warning("self-mint: could not read the work-list: %s", exc)
                return self._status
            if not holes:
                # NOT a zero-count mint: the adopt session answered every
                # graph, so there is nothing to compile. Said on the wire,
                # because "the pod never minted" and "the pod had nothing to
                # mint" are the same silence otherwise.
                self._status = SelfMintStatus(state=NOTHING_TO_MINT)
                self._done.set()
                activity_mod.emit_event(
                    KIND_SKIPPED,
                    "adopt-first boot left no holes for this (lane x sm): "
                    "every claimed graph is already armed from the store",
                    phase="no_holes",
                )
                return self._status
            if self.compiler is None and (
                self.cas_dir is None or not self.target_arch
            ):
                self._status = SelfMintStatus(
                    state=UNAVAILABLE, holes=len(holes),
                    reason="no compiler: this worker states neither a "
                           "`cas_dir` + `target_arch` for the production "
                           "compile child nor an explicit compiler seam")
                self._done.set()
                logger.warning("self-mint: %s", self._status.reason)
                return self._status
            self._status = SelfMintStatus(state=MINTING, holes=len(holes))
            self._started_at = time.monotonic()
            # READ ONCE (pgw#1564): this tuple — the one whose length the
            # status just declared — IS the mint's work-list. `run` used to
            # re-read the live `host.holes` property, and a second read that
            # answered empty settled as `completed 0/N` in milliseconds.
            self._work_list = holes
            self._thread = threading.Thread(
                target=self._run, args=(host, len(holes)),
                name="self-mint", daemon=True)
            self._thread.start()
            logger.info(
                "self-mint: armed on %d hole(s) — background, off the "
                "request path", len(holes))
            return self._status

    # -- observation --------------------------------------------------------

    def status(self) -> SelfMintStatus:
        """This mint's state, counted. Safe from any thread, always typed."""
        with self._lock:
            status = self._status
            if status.state != MINTING:
                return status
            return SelfMintStatus(
                state=MINTING, holes=status.holes, landed=status.landed,
                failed=status.failed,
                elapsed_s=time.monotonic() - self._started_at,
            )

    def join(self, timeout: Optional[float] = None) -> SelfMintStatus:
        """Wait for the mint to finish. For shutdown and for tests — the
        serving path never calls this."""
        self._done.wait(timeout)
        return self.status()

    # -- the run ------------------------------------------------------------

    def _run(self, host: Any, holes: int) -> None:
        act = activity_mod.Activity(KIND)
        act.phase("self_mint_holes", 0, holes)
        counter = act.counter(COUNTER, "graphs", total=float(holes))
        beat = threading.Event()
        beater = threading.Thread(
            target=self._beat, args=(act, beat), name="self-mint-beat",
            daemon=True)
        beater.start()

        def landed(entry: MintedHole) -> None:
            with self._lock:
                self._status = SelfMintStatus(
                    state=MINTING, holes=holes,
                    landed=self._status.landed + 1,
                    failed=self._status.failed,
                )
                done = self._status.landed
            counter.set_done(float(done))
            act.phase("self_mint_holes", done, holes)

        outcome: Optional[MintOutcome] = None
        error = ""
        try:
            outcome = mint_holes(
                host,
                store=self.store,
                compiler=self.compiler,
                cas_dir=self.cas_dir,
                target_arch=self.target_arch,
                toolchain=dict(self.toolchain),
                artifacts_dir=Path(self.artifacts_dir),
                program_source=self.program_source,
                posture=self.posture,
                reserve=self.reserve,
                window_s=self.window_s,
                vcpus=self.vcpus,
                on_landed=landed,
                work_list=self._work_list or None,
            )
        except Exception as exc:  # noqa: BLE001 — the worker outlives its mint
            error = f"{type(exc).__name__}: {exc}"
            logger.exception("self-mint: the mint raised; the pod serves eager")
        finally:
            beat.set()
            self._settle(outcome, error, holes)
            if outcome is not None and not outcome.condemned and not error:
                act.completed()
            else:
                act.failed(RuntimeError(
                    (outcome.condemned if outcome is not None else "")
                    or error or "the mint ended without an outcome"))
            self._done.set()

    def _settle(
        self, outcome: Optional[MintOutcome], error: str, holes: int,
    ) -> None:
        if outcome is None:
            status = SelfMintStatus(
                state=UNAVAILABLE, holes=holes, reason=error,
                elapsed_s=time.monotonic() - self._started_at)
        elif outcome.condemned:
            status = SelfMintStatus(
                state=CONDEMNED, holes=outcome.holes, landed=outcome.landed,
                failed=len(outcome.failed), reason=outcome.condemned,
                elapsed_s=outcome.elapsed_s)
        elif outcome.complete:
            status = SelfMintStatus(
                state=COMPLETE, holes=outcome.holes, landed=outcome.landed,
                elapsed_s=outcome.elapsed_s)
        else:
            status = SelfMintStatus(
                state=PARTIAL, holes=outcome.holes, landed=outcome.landed,
                failed=len(outcome.failed),
                reason="; ".join(
                    f"{f.graph}: {f.reason}" for f in outcome.failed)[:2000],
                elapsed_s=outcome.elapsed_s)
        with self._lock:
            self._status = status
        logger.info("self-mint: %s", status.facts())
        self._say_verdict(status, outcome, holes)

    def _say_verdict(
        self, status: SelfMintStatus, outcome: Optional[MintOutcome], armed_holes: int,
    ) -> None:
        """The mint's TERMINAL verdict as a DURABLE activity row (pgw#1564).

        Twice in one day a pod's mint declared itself done in milliseconds and
        the one line naming WHY lived in a log no SSH could reach (the
        pgw#1541/#1542 lesson, relearned). So the verdict rides the wire:

        * the COUNTS, armed-at-arm vs processed-at-run — a divergence between
          them IS the 13/23 ms no-op class, named here instead of being
          inferred across two other rows that disagree;
        * the REASON (failures / condemnation / error), deduped upstream;
        * the EXECUTED CODE IDENTITY — the falsifying rental could only say
          "pin inferred from the build chain"; a verdict that states its own
          `gen_worker_version` + parent contract digest ends that inference.

        Never raises: a verdict emitter that can kill the worker it reports on
        is worse than the silence it replaces.
        """
        try:
            from ..toolchain import gen_worker_version
            from .mint_child import contract_digest

            run_holes = outcome.holes if outcome is not None else 0
            divergence = ""
            if run_holes != armed_holes:
                divergence = (
                    f" DIVERGENT WORK-LIST: armed {armed_holes} hole(s), the "
                    f"run processed {run_holes} — the pgw#1564 no-op class; "
                    f"a second live read (or older bytes) emptied the list."
                )
            try:
                identity = (
                    f"gen-worker {gen_worker_version()}, "
                    f"contract {contract_digest() or 'no-source'}"
                )
            except Exception:  # noqa: BLE001 — identity must not cost the row
                identity = "identity unreadable"
            activity_mod.emit_event(
                KIND,
                f"self-mint TERMINAL: {status.state} — landed "
                f"{status.landed}, failed {status.failed}, armed "
                f"{armed_holes} hole(s), {status.elapsed_s:.3f}s."
                f"{divergence} reason: {status.reason or 'none'}; {identity}",
                phase=f"terminal_{status.state}",
                step=status.landed,
                total_steps=armed_holes,
            )
        except Exception:  # noqa: BLE001
            logger.debug("self-mint: verdict row failed to emit", exc_info=True)

    def _beat(self, act: activity_mod.Activity, stop: threading.Event) -> None:
        """Carry the counter to the hub on this mint's OWN thread.

        Not :func:`gen_worker.activity.on_beat`: that reads the module-global
        current activity, and a background mint is deliberately not it. The
        beat therefore says what THIS mint has landed, and a serving request's
        counter can never refresh this mint's stall clock (pgw#894's defect,
        arrived at from the other side).
        """
        while not stop.wait(self.beat_s):
            try:
                snap = progress_mod.freshest(act.id)
                if snap is None:
                    continue
                act.progress_beat(
                    snap,
                    self_stalled=progress_mod.self_diagnosis(act.id) is not None,
                )
            except Exception:  # noqa: BLE001 — reporting never breaks the work
                logger.debug("self-mint: beat dropped", exc_info=True)


def production_mint(
    *,
    store: Any,
    artifacts_dir: Path,
    cas_dir: Path,
    sm: str,
    posture: Optional[compile_posture.CompilePosture] = None,
) -> SelfMint:
    """The mint a real serving pod runs: real child compiles, this host's sm.

    The toolchain block is the worker's own recorded one — the same digests
    the artifact's key axis is built from — so a mint can never record a
    toolchain it did not compile with.
    """
    from ..toolchain import toolchain_digest

    return SelfMint(
        store=store,
        artifacts_dir=Path(artifacts_dir),
        cas_dir=Path(cas_dir),
        target_arch=str(sm),
        toolchain=dict(toolchain_digest()),
        posture=posture,
    )


__all__ = [
    "COMPLETE",
    "COUNTER",
    "CONDEMNED",
    "DEFAULT_BEAT_S",
    "KIND",
    "KIND_SKIPPED",
    "MINTING",
    "NOTHING_TO_MINT",
    "NOT_ARMED",
    "PARTIAL",
    "SelfMint",
    "SelfMintStatus",
    "UNAVAILABLE",
    "production_mint",
]
