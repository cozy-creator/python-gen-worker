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
    from ..cli.workspace import artifacts_root

    return artifacts_root()


COUNTER = "compile:self_mint_graphs"

KIND = activity_mod.KIND_SELF_MINT_COMPILE

KIND_SKIPPED = "self_mint_skipped"

DEFAULT_BEAT_S = 10.0

NOT_ARMED = "not_armed"
NOTHING_TO_MINT = "nothing_to_mint"
UNAVAILABLE = "unavailable"
MINTING = "minting"
COMPLETE = "complete"
PARTIAL = "partial"
CONDEMNED = "condemned"


@dataclass(frozen=True)
class SelfMintStatus:
    """What this worker's own mint is doing, or why it is not doing it."""

    state: str = NOT_ARMED
    holes: int = 0
    landed: int = 0
    failed: int = 0
    reason: str = ""
    elapsed_s: float = 0.0

    @property
    def remaining(self) -> int:
        """Holes neither landed nor failed."""
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
    """One serving worker's background mint, and the policy that starts it."""

    store: Any = None
    artifacts_dir: Path = field(default_factory=_artifacts_root)
    cas_dir: Optional[Path] = None
    target_arch: str = ""
    toolchain: Mapping[str, str] = field(default_factory=dict)
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
        self._work_list: Tuple[Any, ...] = ()

    def arm(self, host: Any) -> SelfMintStatus:
        """Start this worker's mint, ONCE, off the serving path."""
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
            self._work_list = holes
            self._thread = threading.Thread(
                target=self._run, args=(host, len(holes)),
                name="self-mint", daemon=True)
            self._thread.start()
            logger.info(
                "self-mint: armed on %d hole(s) — background, off the "
                "request path", len(holes))
            return self._status

    def status(self) -> SelfMintStatus:
        """This mint's state, counted."""
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
        """Wait for the mint to finish."""
        self._done.wait(timeout)
        return self.status()

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
    """The mint a real serving pod runs: real child compiles, this host's sm."""
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
