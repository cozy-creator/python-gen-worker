from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

from . import layout_rung
from .host import CompileStack

from .. import activity as activity_mod

logger = logging.getLogger(__name__)

KIND_ADOPT_REFUSED = "adopt_refused"

PERMANENT_REFUSALS = frozenset({
    "eager_permanent",
    "environment_mismatch",
    "no_document",
    "EnvironmentMismatch",
    "no_local_cas",
})

EAGER_BY_DESIGN_REFUSALS = frozenset({
    "eager_permanent",
    "release_not_stamped",
})

class ServeAdoption:
    """The release's compiled graphs, adopted into a live ``ServeLoop``."""

    def __init__(
        self,
        release_id: str,
        *,
        sm: str,
        artifacts_dir: Path,
        cas_dir: Optional[Path] = None,
        transport: Any = None,
        stack: Optional[CompileStack] = None,
        loader: Optional[Callable[[Path, Any, Any], Any]] = None,
        on_adopted: Optional[Callable[[Any, Any], Any]] = None,
    ) -> None:
        self.release_id = str(release_id)
        self.sm = str(sm)
        self.artifacts_dir = Path(artifacts_dir)
        self.cas_dir = Path(cas_dir) if cas_dir is not None else None
        self._transport = transport
        self._stack = dict(stack) if stack is not None else None
        self._loader = loader
        self._on_adopted = on_adopted
        self._lock = threading.Lock()
        self._sessions: Dict[Tuple[type, str, str], Any] = {}
        self._stores: Dict[Tuple[str, str], Any] = {}
        self._class_keys: Dict[type, Tuple[type, str, str]] = {}
        self._triggered: set[Tuple[type, str, str]] = set()
        self.adoption: Any = None
        self.store: Any = None
        self.refusal: str = ""
        self.refusal_phase: str = ""
        self.refusal_permanent: bool = False
        self.contract: str = ""

    def sink_for(
        self, model_cls: type, lane: Any, binding: Any,
    ) -> Optional[Callable[..., Any]]:
        """``ctx.compile``'s sink for one model class — the adopt arm.

        Sessions are keyed by (class, Bind Contract, lane), while classes that
        share a bind reuse its graph store. A class bound to a second config
        tree gets that Bind Contract's graphs and can never inherit the
        primary's.
        """
        from .model import lane_handle

        contract = lane_handle(lane) if lane is not None else ""
        bind = getattr(binding, "bind_contract", None)
        bind_digest = str(getattr(bind, "digest", "") or "").strip()
        key = (model_cls, bind_digest, contract)
        with self._lock:
            self._class_keys[model_cls] = key
        if lane is None:
            self._refuse("eager_permanent", "this model class declares no lane")
            return None
        if not bind_digest:
            self._refuse(
                "bind_contract_missing",
                f"{model_cls.__name__} carries no Bind Contract",
            )
            return None
        bind_key = (bind_digest, contract)
        with self._lock:
            session = self._sessions.get(key)
            if session is None:
                try:
                    built = self._build(lane, bind_digest, bind_key)
                except Exception as exc:  # noqa: BLE001 — adoption never kills a boot
                    self._refuse(type(exc).__name__, str(exc))
                    return None
                if built is None:
                    return None
                store, session = built
                self._stores[bind_key] = store
                self._sessions[key] = session
                if self.adoption is None:
                    self.adoption, self.store, self.contract = session, store, contract
        return self._sink(session)

    def _sink(self, session: Any = None) -> Optional[Callable[..., Any]]:
        from .adapter_guard import sink as guarded_sink

        selected = self.adoption if session is None else session
        if selected is None:
            return None
        return guarded_sink(selected.adopt)

    def loaded(self, model_cls: type = type(None), lane: Any = None) -> None:
        """THE MINT'S TRIGGER: the author's ``load(ctx)`` has returned."""
        with self._lock:
            key = self._class_keys.get(model_cls)
            if key is None or key in self._triggered:
                return
            self._triggered.add(key)
            session = self._sessions.get(key)
            store = self._stores.get((key[1], key[2]))
            hook = self._on_adopted
            contract = key[2]
        if session is None:
            self._say_boot_end(session, contract, armed=0, claimed=0, mint_running=False)
            return
        self._say_outcome(session, contract)
        armed, holes = len(session.adopted), len(session.holes)
        mint_running = False
        if hook is not None:
            try:
                started = hook(store, session)
                running = getattr(started, "running", None)
                mint_running = bool(holes) if running is None else bool(running)
            except Exception:  # noqa: BLE001 — the pod serves either way
                logger.exception("adopt: the post-load mint trigger raised")
        self._say_boot_end(session, contract, armed=armed, claimed=armed + holes,
                           mint_running=mint_running)

    def _say_boot_end(
        self, session: Any, contract: str, *, armed: int, claimed: int,
        mint_running: bool,
    ) -> None:
        from ..compiled_graph_adopt import EagerPhase
        from .self_mint import KIND_SKIPPED

        if armed or mint_running:
            return
        declared = session is not None or (
            bool(self.refusal_phase)
            and self.refusal_phase not in EAGER_BY_DESIGN_REFUSALS
        )
        if not declared:
            return
        why = (
            f"the release declared {claimed} graph specialization(s) for this "
            f"(lane x sm) and none armed"
            if session is not None
            else f"the adopt refused ({self.refusal})"
        )
        activity_mod.emit_event(
            KIND_SKIPPED,
            f"release={self.release_id} lane={contract or '(unresolved)'} "
            f"sm={self.sm}: boot ended with ZERO armed graph specializations "
            f"and no mint in flight — {why}. This pod serves EAGER for the "
            f"rest of its life against a release that declared compile.",
            phase=EagerPhase.BOOT_ENDED_UNCOMPILED.value,
            step=armed,
            total_steps=claimed,
        )

    def _build(
        self,
        lane: Any,
        bind_contract_digest: str,
        bind_key: Tuple[str, str],
    ) -> Optional[Tuple[Any, Any]]:
        from .._vendor.torchcg.adopt import AdoptSession
        from ..env_identity import installed_stack_drift
        from .mint_store import graph_store
        from .hub_store import (
            BrokerReleaseGraphTransport,
            HubGraphStore,
            ReleaseNotStamped,
        )
        from .model import lane_handle

        contract = lane_handle(lane)
        transport = (
            self._transport if self._transport is not None
            else BrokerReleaseGraphTransport()
        )
        store = self._stores.get(bind_key)
        upstream: Any = (
            getattr(store, "upstream", None)
            if store is not None
            else HubGraphStore(
                transport, self.release_id, bind_contract_digest, contract, self.sm
            )
        )
        try:
            document = upstream.get_graphs(self.release_id)
        except ReleaseNotStamped as exc:
            self._refuse("release_not_stamped", str(exc))
            return None
        if document is None:
            self._refuse(
                "no_document",
                "the adopt route answered, and the answer rebuilt to no lane "
                "document for this (release x lane x sm)")
            return None
        if getattr(document, "eager_permanent", False):
            self._refuse("eager_permanent", "the release document is eager-permanent")
            return None
        stack = self._stack if self._stack is not None else dict(document.stack)
        for row in installed_stack_drift(dict(document.stack)):
            logger.warning("adopt: compile-stack drift vs this venv: %s", row)
        if self.cas_dir is None:
            self._refuse(
                "no_local_cas",
                "this worker states no local CAS, so a fetched artifact could "
                "not be banked and a minted one could not be published — the "
                "adopt would be read-only against the fleet pool forever")
            return None
        if store is None:
            store = graph_store(self.cas_dir, upstream)
        session = AdoptSession(
            store, document, contract, self.sm,
            loader=self._loader,
            artifacts_dir=self.artifacts_dir,
            stack=stack,
        )
        logger.info(
            "adopt: release=%s lane=%s sm=%s — %d graph(s) claimed by the "
            "document; what this boot arms is decided by the author's "
            "ctx.compile calls and reported after load",
            self.release_id, contract, self.sm,
            len(session.lane.graphs),
        )
        return store, session

    def _say_outcome(self, session: Any, contract: str) -> None:
        if session is None:
            return
        adopted, holes = len(session.adopted), len(session.holes)
        marks = tuple(session.unclaimed_marks)
        silent = bool(session.silently_eager())
        ambiguous = len(session.ambiguous)
        activity_mod.emit_event(
            activity_mod.KIND_BOOT_ADOPT_SUMMARY,
            f"release={self.release_id} lane={contract} sm={self.sm}: "
            f"{adopted} graph(s) adopted from the store, {holes} hole(s) for "
            f"the background mint, {ambiguous} disarmed as ambiguous, "
            f"{len(session.unclaimed)} unclaimed, "
            f"{len(marks)} marked module(s) that matched nothing",
            phase=(
                "marks_unmatched" if silent else
                "reused" if holes == 0 and adopted else
                "minting" if holes else
                "all_ambiguous" if ambiguous and not adopted else "empty_lane"
            ),
            step=adopted,
            total_steps=adopted + holes,
        )
        if marks:
            logger.warning(
                "adopt: %d module(s) marked with ctx.compile matched NO graph "
                "in lane %s. They serve EAGER, permanently, and no mint will "
                "change that: the graphs exist, the MATCH does not.",
                len(marks), contract,
            )
            for mark in marks:
                logger.warning("adopt:   %s", mark.describe())

    def _refuse(self, phase: str, detail: str) -> None:
        self.refusal = f"{phase}: {detail}"
        self.refusal_phase = phase
        self.refusal_permanent = phase in PERMANENT_REFUSALS
        logger.warning(
            "adopt: serving eager (%s) — %s",
            "PERMANENT" if self.refusal_permanent else "retryable",
            self.refusal,
        )
        activity_mod.emit_event(
            KIND_ADOPT_REFUSED,
            f"release {self.release_id} sm={self.sm} "
            f"({'PERMANENT' if self.refusal_permanent else 'retryable'}): "
            f"{detail}"[:2000],
            phase=phase,
        )

    @property
    def holes(self) -> Tuple[Any, ...]:
        """The ordered mint work-list, in canonical document order."""
        return tuple(hole for session in self._sessions.values() for hole in session.holes)

    @property
    def layout_rungs(self) -> Tuple[Any, ...]:
        """Each armed graph's layout position, read off the artifact serving it."""
        return tuple(
            rung for session in self._sessions.values()
            for rung in layout_rung.rungs_of(session)
        )

    def facts(self) -> Dict[str, Any]:
        """Counted, and never silent: `adopted`/`holes` are absent (not zero) when no session was ever built, and `refusal` says why."""
        if not self._sessions:
            return {"adopting": False, "refusal": self.refusal or "not_attempted"}
        sessions = tuple(self._sessions.values())
        marks = tuple(mark for session in sessions for mark in session.unclaimed_marks)
        rungs = self.layout_rungs
        return {
            "adopting": True,
            "bind_sessions": len(sessions),
            "adopted": sum(len(session.adopted) for session in sessions),
            "holes": sum(len(session.holes) for session in sessions),
            "unclaimed": sum(len(session.unclaimed) for session in sessions),
            "unmatched_marks": len(marks),
            "unmatched": [mark.describe() for mark in marks],
            "ambiguous": sum(len(session.ambiguous) for session in sessions),
            # pgw#1645. Counted the same way everything else here is, and the
            # rows carry BOTH layouts: a pod on a lower rung must read as
            # EARNING rather than as a slow pod nobody can explain.
            "layout_earning": sum(1 for rung in rungs if rung.earning),
            "layout_rungs": [rung.facts() for rung in rungs],
        }


__all__ = [
    "EAGER_BY_DESIGN_REFUSALS",
    "KIND_ADOPT_REFUSED",
    "PERMANENT_REFUSALS",
    "ServeAdoption",
]
