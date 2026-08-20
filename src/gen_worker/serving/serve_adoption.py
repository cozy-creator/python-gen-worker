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
from typing import Any, Callable, Dict, Optional, Tuple

from .host import CompileStack

from .. import activity as activity_mod

logger = logging.getLogger(__name__)

#: The typed wire fact for "this pod will serve eager for its whole life, and
#: here is why". A serve pod's stdout goes nowhere (pgw#760), so a refusal
#: that reaches only the logger is a pod that silently never adopts — the
#: exact shape that made this gap invisible for as long as it existed.
KIND_ADOPT_REFUSED = "adopt_refused"

#: The refusal phases that will NEVER fix themselves on this pod (pgw#1472).
#: A refusal that is unconditional is indistinguishable from one that is
#: correct — both read as "serving eager" — and that is how an env-identity
#: mismatch could have made every pod eager forever while looking healthy. So
#: the distinction is CARRIED, not inferred later from a log: `permanent`
#: means "a successor with the same inputs refuses identically; stop waiting".
PERMANENT_REFUSALS = frozenset({
    "eager_permanent",       # the release document says so
    "environment_mismatch",  # the pod is not the release's env; a retry cannot help
    "no_document",           # the release is stamped with no lane for this (lane x sm)
    "EnvironmentMismatch",   # the same, arriving as an exception type name
    "no_local_cas",          # pgw#1573: no local tier — nothing can be banked or minted
})

#: The refusal phases that mean the RELEASE NEVER DECLARED compiled serving
#: (pgw#1480). Orthogonal to :data:`PERMANENT_REFUSALS`, which answers "will a
#: retry help" — this one answers "was anything promised".
#:
#: The boot-end verdict below stays SILENT on these, and that silence is the
#: whole reason the verdict is trustworthy: `boot_ended_uncompiled` must mean
#: **DECLARED and didn't**, never "this pod runs eager". An eager-by-design
#: release is the contract working, and an instrument that fires on it is an
#: alarm operators learn to ignore.
#:
#: `no_document` is deliberately NOT here. The release IS stamped — it declared
#: compiled serving and simply has no lane document for THIS (lane x sm) — so a
#: pod serving eager under it is exactly the "declared compile, served eager"
#: shape se#780's row 6 exists to catch.
EAGER_BY_DESIGN_REFUSALS = frozenset({
    "eager_permanent",      # the release document (or the model class) says eager
    "release_not_stamped",  # nothing was ever declared for this release at all
})

# pgw#1460: THERE IS NO ARTIFACT LOADER IN THIS MODULE ANY MORE, and its
# absence is the fix. What lived here was
# `torch._inductor.aoti_load_package(str(path))` — the `GraphRecord` accepted
# and discarded — so every adopted graph ran with an EMPTY constant buffer
# (torchcg mints WEIGHTLESS artifacts by sealed policy; constants ship beside
# them and bind by reference) and with the author's nested kwargs where the
# package wanted flat positional feeds. Wrong numerics or a hard AOTI error,
# on the exact path the adopt-and-reuse program is judged on.
#
# The bytes->callable path is torchcg's own (`torchcg.serve.aoti_loader`,
# tcg#58) and it is the DEFAULT `AdoptSession` loader, so stating nothing is
# both correct and the only reasonable thing to write. `loader=` below survives
# as a TEST seam and nothing else; a raw `aoti_load_package` must not grow
# back here or anywhere else in `src/`.


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
        stack: Optional[CompileStack] = None,
        loader: Optional[Callable[[Path, Any, Any], Any]] = None,
        #: The mint trigger. May ANSWER with its mint's status (anything
        #: carrying `.running`); `None` back means "unknown" (pgw#1480).
        on_adopted: Optional[Callable[["ServeAdoption"], Any]] = None,
    ) -> None:
        self.release_id = str(release_id)
        self.sm = str(sm)
        self.artifacts_dir = Path(artifacts_dir)
        #: The pod's own tensorfs CAS — the local tier of the store below, and
        #: REQUIRED since pgw#1573: it is where a fleet hit is cached and where
        #: a mint publishes and arms from. ``None`` is a typed refusal at
        #: build time (`no_local_cas`), not a read-only degraded mode nobody
        #: could tell from a working one.
        self.cas_dir = Path(cas_dir) if cas_dir is not None else None
        self._transport = transport
        self._stack = dict(stack) if stack is not None else None
        #: `None` = torchcg's own production loader (tcg#58). A value here
        #: is a TEST DOUBLE; production never states one.
        self._loader = loader
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
        #: The refusal's PHASE alone (pgw#1480). `refusal` is a joined
        #: sentence, and classifying a refusal by substring-matching a sentence
        #: is how a vocabulary becomes folklore — the boot-end verdict needs the
        #: token, so the token is carried.
        self.refusal_phase: str = ""
        #: Whether that refusal is PERMANENT for this pod (pgw#1472). Read off
        #: the object rather than scraped: `sink_for` swallows every failure
        #: into eager-forever by design, so "refused once, will refuse always"
        #: has to be a state something can COUNT.
        self.refusal_permanent: bool = False
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
                return self._sink()
            self._settled = True
            try:
                self._build(lane)
            except Exception as exc:  # noqa: BLE001 — adoption never kills a boot
                self._refuse(type(exc).__name__, str(exc))
                return None
        return self._sink()

    def _sink(self) -> Optional[Callable[..., Any]]:
        """``adopt``, GUARDED (pgw#1573/pgw#1571).

        A peft adapter attached after arming does not execute inside a compiled
        graph — the parent's forward is the traced computation and never enters
        the wrapped submodules — so the base model is served bit-identically
        with no refusal and no log. The guard rides every module this sink
        arms; `adapter_guard` owns the argument.
        """
        from .adapter_guard import sink as guarded_sink

        if self.adoption is None:
            return None
        return guarded_sink(self.adoption.adopt)

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

        **It is also BOOT-END (pgw#1480).** The ``ServeLoop`` makes instances
        lazily under residency leases, so there is no single boot-end moment in
        v2 — but there IS a first moment at which this pod's compiled-serving
        story is finished and a request is about to be served against it, and
        this is that moment: the sink was handed over, the author's
        ``ctx.compile`` calls have run, the adopt session is settled, and the
        mint (if any) has been given its chance. Everything after this is
        serving. So the terminal verdict is taken here, on FACTS, never on a
        timer.
        """
        with self._lock:
            if self._triggered:
                return
            self._triggered = True
            session = self.adoption
            hook = self._on_adopted
            contract = self.contract
        if session is None:
            # A refused adopt: no session, no counts, and `_refuse` already
            # said why on the wire. The only thing left to decide is whether
            # the release had DECLARED anything, which the verdict owns.
            self._say_boot_end(armed=0, claimed=0, mint_running=False)
            return
        self._say_outcome(contract)
        armed, holes = len(session.adopted), len(session.holes)
        mint_running = False
        if hook is not None:
            try:
                # The hook may answer with its mint's status (anything with a
                # `.running`). An unanswering hook is READ AS A RUNNING MINT
                # when there are holes: a false alarm on a pod that is about to
                # fix itself is worse than a missed one, because the operator
                # stops reading the instrument (tracker: "a wrongly placed
                # emitter is WORSE than a dead one").
                started = hook(self)
                running = getattr(started, "running", None)
                mint_running = bool(holes) if running is None else bool(running)
            except Exception:  # noqa: BLE001 — the pod serves either way
                logger.exception("adopt: the post-load mint trigger raised")
        self._say_boot_end(armed=armed, claimed=armed + holes,
                           mint_running=mint_running)

    def _say_boot_end(
        self, *, armed: int, claimed: int, mint_running: bool
    ) -> None:
        """`boot_ended_uncompiled` — THE EMIT THIS PHASE NEVER HAD (pgw#1480).

        `EagerPhase.BOOT_ENDED_UNCOMPILED` was defined and emitted by nothing,
        so ``self_mint_skipped/boot_ended_uncompiled`` read ABSENT on every pod
        that has ever run — **including every pod it exists to condemn** — and
        se#780 was about to judge an 80 GB rental on that absence being its
        PASS. A proof condition written against an instrument that cannot fire
        is not a weak proof; it is not a proof.

        The predicate is three FACTS, all of them owned by this object at this
        instant, and not one of them a clock:

        1. **DECLARED** — the release promised compiled serving. Either a lane
           document was resolved for this (release x lane x sm) (a session
           exists), or the adopt refused for a reason that is not
           :data:`EAGER_BY_DESIGN_REFUSALS`. A pod that was never promised
           anything is silent: eager is its contract.
        2. **ZERO ARMED** — no graph specialization adopted. Not "few", not
           "some holes": pgw#844's rule is that this token means *nothing is
           dispatchable*, never "partial", so one armed specialization silences
           it.
        3. **TERMINAL** — no mint is running that could end the eager window.
           A boot whose mint is in flight is `mint_in_progress`, a different
           and non-terminal state; `boot_adopt_summary phase='minting' step=0`
           already carries it and is the row se#780 should read for boot 1.

        Deliberately on the kind the enum's own wire contract names —
        ``self_mint_skipped``, which tensorhub already stores and already knows
        (`ActivityKindSelfMintSkipped`) — so the fleet query is live with NO
        hub-side change. `step`/`total_steps` carry armed/claimed as NUMBERS,
        because "0 of 6 armed" as prose in `detail` is a metric nobody can
        group by.
        """
        from ..compiled_graph_adopt import EagerPhase
        from .self_mint import KIND_SKIPPED

        if armed or mint_running:
            return
        declared = self.adoption is not None or (
            bool(self.refusal_phase)
            and self.refusal_phase not in EAGER_BY_DESIGN_REFUSALS
        )
        if not declared:
            return
        why = (
            f"the release declared {claimed} graph specialization(s) for this "
            f"(lane x sm) and none armed"
            if self.adoption is not None
            else f"the adopt refused ({self.refusal})"
        )
        activity_mod.emit_event(
            KIND_SKIPPED,
            f"release={self.release_id} lane={self.contract or '(unresolved)'} "
            f"sm={self.sm}: boot ended with ZERO armed graph specializations "
            f"and no mint in flight — {why}. This pod serves EAGER for the "
            f"rest of its life against a release that declared compile.",
            phase=EagerPhase.BOOT_ENDED_UNCOMPILED.value,
            step=armed,
            total_steps=claimed,
        )

    def _build(self, lane: Any) -> None:
        from .._vendor.torchcg.adopt import AdoptSession
        from ..env_identity import installed_stack_drift
        from .mint_store import graph_store
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
        # pgw#1489: the env half of the key is the COMPILE STACK the
        # document stamped off the endpoint's uv.lock. The pod's own venv is
        # compared to it diagnostically — a mismatch there costs a failed
        # artifact load and a re-mint, never a wrong answer, so it warns.
        stack = self._stack if self._stack is not None else dict(document.stack)
        for row in installed_stack_drift(dict(document.stack)):
            logger.warning("adopt: compile-stack drift vs this venv: %s", row)
        # ONE store for both directions and for every entry point (pgw#1573):
        # the adopt reads through it (local CAS before the hub, so a restarted
        # pod adopts what it already minted, and a fleet hit is BANKED locally
        # on the way through), and the mint publishes into it and arms out of
        # it. Two objects here would mean two answers to "do I have this
        # graph"; a hub tier with no local one would mean a pod that
        # re-downloads its own artifacts on every boot.
        if self.cas_dir is None:
            self._refuse(
                "no_local_cas",
                "this worker states no local CAS, so a fetched artifact could "
                "not be banked and a minted one could not be published — the "
                "adopt would be read-only against the fleet pool forever")
            return
        self.store = graph_store(self.cas_dir, store)
        self.contract = contract
        self.adoption = AdoptSession(
            self.store, document, contract, self.sm,
            loader=self._loader,
            artifacts_dir=self.artifacts_dir,
            stack=stack,
        )
        # WHAT IS KNOWN AT CONSTRUCTION, and only that (pgw#1534). This line
        # used to report "%d adopted, %d hole(s)" from here — and it ran BEFORE
        # any `ctx.compile` had, so both counts were structurally always 0. A
        # log line that cannot say anything else is not a reading; the real
        # verdict is `_say_outcome`, emitted from `loaded` for exactly this
        # reason, and its own docstring already said so.
        logger.info(
            "adopt: release=%s lane=%s sm=%s — %d graph(s) claimed by the "
            "document; what this boot arms is decided by the author's "
            "ctx.compile calls and reported after load",
            self.release_id, contract, self.sm,
            len(self.adoption.lane.graphs),
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
        marks = tuple(session.unclaimed_marks)
        # `adopted=0, holes=0` HAD TWO CAUSES AND ONE PHASE (pgw#1534).
        #
        # An endpoint that marks nothing is an eager endpoint and `empty_lane`
        # is the truth about it. An endpoint whose `ctx.compile`-ed module fits
        # NO record in the lane is a compiled path silently switched off — over
        # a store that may hold every artifact the document names. Both landed
        # on `empty_lane` with the same two zeros, so no query could separate
        # "nothing to do" from "the compiled path is off and nobody noticed".
        # Measured exactly that way on this box: 14 graphs in the document, 14
        # artifacts present, `adopted: 0, holes: 0, armed: 0`.
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
            # LOUD, and at WARNING: this is the author's own mark producing
            # nothing, which is never what the author meant. Eager still serves
            # — correctness is untouched — so it is stated, not raised.
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
        # The COUNT is by `phase`, which is already a typed wire field: the
        # permanence question is answered by `phase IN PERMANENT_REFUSALS`.
        # No new wire field is invented here — `emit_event`'s keyword set is
        # the proto's, and widening it is a coordinated proto/wire change, not
        # something a serving module does on its own. What this fix adds is
        # that the SET is named in code instead of being folklore, and that
        # the answer is readable off the live object.
        activity_mod.emit_event(
            KIND_ADOPT_REFUSED,
            f"release {self.release_id} sm={self.sm} "
            f"({'PERMANENT' if self.refusal_permanent else 'retryable'}): "
            f"{detail}"[:2000],
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
        marks = tuple(self.adoption.unclaimed_marks)
        return {
            "adopting": True,
            "adopted": len(self.adoption.adopted),
            "holes": len(self.adoption.holes),
            "unclaimed": len(self.adoption.unclaimed),
            # pgw#1534: the MODULE side. Without it `adopted: 0, holes: 0` is
            # the same reading for "this endpoint marks nothing" and "this
            # endpoint's marked module fits no graph in its own lane", and a
            # caller deciding whether the compiled path is live cannot tell
            # them apart. `unmatched_marks` is the count; `unmatched` is the
            # per-module reason, which is what makes it actionable rather than
            # merely non-zero.
            "unmatched_marks": len(marks),
            "unmatched": [mark.describe() for mark in marks],
            # pgw#1534: the THIRD way to reach `adopted: 0, holes: 0` — every
            # record disarmed as a literal-twin. The artifacts are present and
            # valid; the dispatcher refuses to guess between two graphs whose
            # tensor structures are identical, so it disarms BOTH. Counted here
            # because a caller asking "is the compiled path live" gets three
            # different answers behind one pair of zeros.
            "ambiguous": len(self.adoption.ambiguous),
        }


__all__ = [
    "EAGER_BY_DESIGN_REFUSALS",
    "KIND_ADOPT_REFUSED",
    "PERMANENT_REFUSALS",
    "ServeAdoption",
]
