"""pgw#1372's PRODUCTION half: adopt-first boot for the :class:`ServeLoop`.

The adopt machinery all landed — ``HubGraphStore`` over th#2133's route,
``BrokerReleaseGraphTransport`` over the split's action broker, torchcg's
``AdoptSession`` behind ``ctx.compile`` — and the only thing that ever built
it was :mod:`gen_worker.serving.__main__`, the cozy-local / CI runner. The
process a real pod runs (:class:`gen_worker.worker.Worker` -> ``ServeLoop``)
passed ``compile_sink_for=None``, so **every ``ctx.compile`` on every real
serving pod was a transparent pass-through**: no artifact was ever adopted,
no hole was ever registered, and pgw#1371's mint had nothing to consume even
once it acquired a caller. This module is that missing construction.

**LAZY, BY THE SHAPE OF THE LOOP.** ``EndpointHost.setup`` loads every model
at boot, so its session can be built in one place before the first load. The
``ServeLoop`` makes instances under residency leases — the first load happens
inside the first request — so the session is built at the first
``sink_for`` ask and cached. That is not a deferral of the adopt: it is the
first moment a model is loaded at all, and adoption is meaningless earlier.

**IT NEVER FAILS A BOOT.** An unstamped release, a hub that will not answer,
a pod with no visible GPU: each is an eager pod and a stated reason, never a
dead worker. The one exception is torchcg's own environment audit, which
refuses BEFORE author code runs and is a build-system defect worth surfacing
— it is caught here too, because a pod that cannot adopt can still serve.

The object doubles as the mint's HOST: :attr:`holes` and :attr:`adoption` are
exactly what :func:`gen_worker.serving.mint.hole_work_list` and
``BackgroundMint._mint_one`` read, so the same work-list the boot registered
is the one the mint fills — one path, not two.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

from .. import activity as activity_mod

logger = logging.getLogger(__name__)

#: The typed wire fact for "this pod will serve eager for its whole life, and
#: here is why". A serve pod's stdout goes nowhere (pgw#760), so a refusal
#: that reaches only the logger is a pod that silently never adopts — the
#: exact shape that made this gap invisible for as long as it existed.
KIND_ADOPT_REFUSED = "adopt_refused"


def aoti_loader(path: Path, record: Any) -> Any:
    """Load one packaged compiled graph as its graph class's forward."""
    import torch._inductor

    return torch._inductor.aoti_load_package(str(path))


class ServeAdoption:
    """The release's compiled graphs, adopted into a live ``ServeLoop``.

    Constructed at worker boot with the pod's own identity; it asks the hub
    once, on the first model load, and thereafter serves the session it built.
    """

    def __init__(
        self,
        release_id: str,
        *,
        sm: str,
        artifacts_dir: Path,
        cas_dir: Optional[Path] = None,
        transport: Any = None,
        installed: Optional[Mapping[str, str]] = None,
        loader: Optional[Callable[[Path, Any], Any]] = None,
        on_adopted: Optional[Callable[["ServeAdoption"], None]] = None,
    ) -> None:
        self.release_id = str(release_id)
        self.sm = str(sm)
        self.artifacts_dir = Path(artifacts_dir)
        #: The pod's own tensorfs CAS — the local tier of the store below.
        #: ``None`` keeps the hub as the only tier, which is read-only, so a
        #: mint on that shape banks nothing and says so.
        self.cas_dir = Path(cas_dir) if cas_dir is not None else None
        self._transport = transport
        self._installed = dict(installed) if installed is not None else None
        self._loader = loader if loader is not None else aoti_loader
        self._on_adopted = on_adopted
        self._lock = threading.Lock()
        self._settled = False
        self._triggered = False
        #: The torchcg session, once one exists. ``None`` = serving eager.
        self.adoption: Any = None
        #: The store the session read, which is also the mint's publish sink.
        self.store: Any = None
        #: Why this pod serves eager, when it does. Empty while adopting.
        self.refusal: str = ""
        #: The lane the session was built against, for the outcome event.
        self.contract: str = ""

    # -- the ServeLoop seam -------------------------------------------------

    def sink_for(self, model_cls: type, lane: Any) -> Optional[Callable[..., Any]]:
        """``ctx.compile``'s sink for one model class — the adopt arm.

        ``None`` means the eager bridge, which is what every refusal below
        resolves to. Called under the residency load gate, so the one-time
        construction is already serialized; the lock is here because two
        DIFFERENT residency keys can load concurrently.
        """
        with self._lock:
            if self._settled:
                return self.adoption.adopt if self.adoption is not None else None
            self._settled = True
            try:
                self._build(lane)
            except Exception as exc:  # noqa: BLE001 — adoption never kills a boot
                self._refuse(type(exc).__name__, str(exc))
                return None
        return self.adoption.adopt if self.adoption is not None else None

    def loaded(self, model_cls: type = type(None), lane: Any = None) -> None:
        """THE MINT'S TRIGGER: the author's ``load(ctx)`` has returned.

        Not :meth:`sink_for`, which is a full model load too early — the sink
        is handed over BEFORE ``load(ctx)`` runs, so the holes it would read
        are an empty list and the mint would find nothing to do. Holes are
        registered by the author's own ``ctx.compile`` calls INSIDE the load,
        so the first instant the work-list is complete is the instant the load
        returns, and that is here. (Measured: triggering at ``sink_for``
        reported ``nothing_to_mint`` on a pod with two real holes.)

        Fires once. Never raises — a mint that will not start is an eager pod.
        """
        with self._lock:
            if self._triggered or self.adoption is None:
                return
            self._triggered = True
            hook = self._on_adopted
            contract = self.contract
        self._say_outcome(contract)
        if hook is None:
            return
        try:
            hook(self)
        except Exception:  # noqa: BLE001 — the pod serves either way
            logger.exception("adopt: the post-load mint trigger raised")

    def _build(self, lane: Any) -> None:
        from .._vendor.torchcg.adopt import AdoptSession
        from .._vendor.torchcg.graph_identity import installed_closure
        from .mint_store import worker_store
        from .hub_store import (
            BrokerReleaseGraphTransport,
            HubGraphStore,
            ReleaseNotStamped,
        )
        from .model import lane_handle

        if lane is None:
            self._refuse("eager_permanent", "this model class declares no lane")
            return
        contract = lane_handle(lane)
        transport = (
            self._transport if self._transport is not None
            else BrokerReleaseGraphTransport()
        )
        store = HubGraphStore(transport, self.release_id, contract, self.sm)
        try:
            document = store.get_graphs(self.release_id)
        except ReleaseNotStamped as exc:
            # Not a failure: an unstamped release HAS no adopt story, and the
            # worker's answer is its eager path (th#2134 gates the stamp).
            self._refuse("release_not_stamped", str(exc))
            return
        if document is None:
            self._refuse(
                "no_document",
                "the adopt route answered, and the answer rebuilt to no lane "
                "document for this (release x lane x sm)")
            return
        if getattr(document, "eager_permanent", False):
            self._refuse("eager_permanent", "the release document is eager-permanent")
            return
        installed = (
            self._installed if self._installed is not None else installed_closure()
        )
        # ONE store for both directions: the adopt reads through it (local CAS
        # before the hub, so a restarted pod adopts what it already minted) and
        # the mint publishes through it. Two objects here would mean two
        # answers to "do I have this graph".
        self.store = (
            worker_store(self.cas_dir, store) if self.cas_dir is not None else store
        )
        self.contract = contract
        self.adoption = AdoptSession(
            self.store, document, contract, self.sm,
            loader=self._loader,
            artifacts_dir=self.artifacts_dir,
            installed=installed,
        )
        logger.info(
            "adopt: release=%s lane=%s sm=%s — %d adopted, %d hole(s)",
            self.release_id, contract, self.sm,
            len(self.adoption.adopted), len(self.adoption.holes),
        )

    def _say_outcome(self, contract: str) -> None:
        """THE REUSE FACT, on the wire, counted — emitted after the load.

        A serve pod's stdout goes nowhere (pgw#760), so "this pod adopted N
        graphs and compiled none" cannot be a log line: it is the single
        observation the whole mint-and-reuse program is judged on, and it has
        to be readable off-pod. The two boots of a reuse proof read straight
        off it:

            first  boot -> phase=minting  step=0 total_steps=N
            second boot -> phase=reused   step=N total_steps=N

        **Its own kind, `boot_adopt_summary` (pgw#1441).** `boot_adopt` is a
        PER-KEY event whose `phase` is `hit`/`miss`/`no_export_declaration` —
        one row per graph. This is a PER-BOOT verdict over all of them, and
        putting both under one kind gives that kind two `phase` vocabularies:
        `count(*) where kind='boot_adopt' and phase='reused'` then reads 0 on
        every pod that predates this code, which is indistinguishable from
        "nothing reused". `warmup`/`warmup_summary` is the same split, made
        for the same reason, one incident earlier (pgw#1067).

        **The counts are NUMERIC.** `step` = graphs adopted, `total_steps` =
        graphs claimed, so the reuse ratio is a query instead of a regex over
        `detail`. Prose in `detail` alone is how a reader ends up building a
        metric on whatever nearby column looks numeric.

        Emitted from :meth:`loaded`, not from :meth:`_build`, because the
        counts are only final once the author's ``ctx.compile`` calls have
        run — at build time every record is still unclaimed.
        """
        session = self.adoption
        if session is None:
            return
        adopted, holes = len(session.adopted), len(session.holes)
        activity_mod.emit_event(
            activity_mod.KIND_BOOT_ADOPT_SUMMARY,
            f"release={self.release_id} lane={contract} sm={self.sm}: "
            f"{adopted} graph(s) adopted from the store, {holes} hole(s) for "
            f"the background mint, {len(session.unclaimed)} unclaimed",
            phase="reused" if holes == 0 and adopted else (
                "minting" if holes else "empty_lane"),
            step=adopted,
            total_steps=adopted + holes,
        )

    def _refuse(self, phase: str, detail: str) -> None:
        self.refusal = f"{phase}: {detail}"
        logger.warning("adopt: serving eager — %s", self.refusal)
        activity_mod.emit_event(
            KIND_ADOPT_REFUSED,
            f"release {self.release_id} sm={self.sm}: {detail}"[:2000],
            phase=phase,
        )

    # -- the mint's host protocol -------------------------------------------

    @property
    def holes(self) -> Tuple[Any, ...]:
        """The ordered mint work-list, in canonical document order."""
        return tuple(self.adoption.holes) if self.adoption is not None else ()

    def facts(self) -> Dict[str, Any]:
        """Counted, and never silent: `adopted`/`holes` are absent (not zero)
        when no session was ever built, and `refusal` says why."""
        if self.adoption is None:
            return {"adopting": False, "refusal": self.refusal or "not_attempted"}
        return {
            "adopting": True,
            "adopted": len(self.adoption.adopted),
            "holes": len(self.adoption.holes),
            "unclaimed": len(self.adoption.unclaimed),
        }


__all__ = ["KIND_ADOPT_REFUSED", "ServeAdoption", "aoti_loader"]
