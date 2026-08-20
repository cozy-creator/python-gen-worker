"""THE compiled-graph store (pgw#1371, made canonical by pgw#1573).

:func:`graph_store` is the ONE constructor. Paul's ruling, 2026-08-20:
*"check local and remote hub … it's literally that simple. You likely created
a bunch of unnecessary extra code paths."* Before pgw#1573 six call sites built
a store six ways — three of them a bare ``LocalGraphStore`` with no hub tier at
all, one of them (`gen-worker compile`) documenting a fetch-first hub consult it
did not have — so "do I have this graph" had three different answers depending
on which entry point asked. There is one now, and every reader and every
publisher goes through it.

One ``GraphStore``, three tiers, and the reason they are one object is that a
worker must not have two answers to "do I have this graph".

* **LOCAL** — a ``LocalGraphStore`` over the pod's own tensorfs CAS. Every
  minted artifact lands here FIRST, as it lands, so the durability contract
  in :mod:`gen_worker.serving.mint` ("on disk before published, published
  before armed") is satisfied by something that cannot refuse. A worker
  restarted on a pod that already minted adopts its own artifacts back
  instead of paying for them twice.
* **UPSTREAM** — the boot-side :class:`~gen_worker.serving.hub_store.
  HubGraphStore`. It owns the release DOCUMENT, the graph BLOBS and the
  fleet's already-minted artifacts, and it is READ-ONLY by construction:
  ``publish_artifact`` raises, because the fleet publish is the
  intent/complete control-plane leg, not the adopt route.
* **BAKED** — the IMAGE's read-only exported-program CAS
  (``/app/.tensorhub/derive-cas``), consulted for ``fetch_program`` ONLY.
  pgw#1462 part 2. th#2162 argued no hub route was needed for graph blobs
  because the miner *"already falls through to its local CAS by content
  address"* — it does, but into ``<TENSORHUB_CACHE_DIR>/cas``, which is NOT
  where the builder bakes them. Every baked blob has been unreachable since
  the first image carried one, silently, because a cache miss reports nothing.

**THE UPSTREAM PUBLISH IS A STATED HOLE, NOT A SWALLOWED ERROR.** The fleet
leg (``compiled_graphs.publish_intent`` / ``publish_complete``) is allowlisted
in :mod:`gen_worker.procsplit.actions` and has no worker-side caller at HEAD —
pgw#1373 deleted it with ``executor.py`` and pgw#1368/th#2132 own restoring
it. Until it exists, a serving worker's mint is durable ON ITS POD and the
fleet does not get the bytes. That is a materially different world from "the
mint works", so it is emitted as a typed event ONCE per mint rather than
logged per graph or, worse, raised — a refusal that killed each hole would
turn a working local mint into a wholly failed one.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Dict, Optional

from .. import activity as activity_mod

logger = logging.getLogger(__name__)


def _store_faults() -> tuple[type[BaseException], ...]:
    """The store's own "these bytes are bad" vocabulary, imported not guessed."""
    from .._vendor.tensorfs.local import DigestMismatch
    from .._vendor.torchcg.store import StoreError

    return (StoreError, DigestMismatch, FileNotFoundError, ValueError)


_STORE_FAULTS = _store_faults()

#: The typed fact: this pod minted a graph and the FLEET does not have it.
#: A mint that is only pod-local is not the mint this program promises, so it
#: says so on the wire rather than reading as a clean publish.
KIND_PUBLISH_LOCAL_ONLY = "self_mint_publish_local_only"


class ProgramBlobUnreachable(RuntimeError):
    """The serialized graph this hole must compile cannot be fetched.

    Costs its own graph and names its owner. NEVER a re-trace: running author
    code at mint time is exactly what the blob-in design removes.
    """


class TieredGraphStore:
    """Local CAS first, hub second — for reads AND for the mint's publishes.

    Implements torchcg's ``GraphStore`` protocol. Reads prefer the local tier
    because bytes this pod already holds need no network and cannot be
    revoked mid-boot; writes go to the local tier unconditionally and to the
    upstream tier when it can take them.
    """

    def __init__(self, local: Any, upstream: Any = None, baked: Any = None) -> None:
        self.local = local
        self.upstream = upstream
        #: The IMAGE's read-only exported-program CAS (pgw#1462 part 2). A
        #: THIRD tier and deliberately not a second `local`: nothing is ever
        #: written here, it is consulted for `fetch_program` ONLY, and it holds
        #: serialized graphs rather than compiled artifacts. Folding it into
        #: the local tier would make a read-only directory look writable to
        #: every publish path.
        self.baked = baked
        self._lock = threading.Lock()
        #: Graphs this pod minted that the fleet did not receive. Counted, and
        #: named — an empty tuple after a mint that landed graphs is the proof
        #: the fleet leg worked, which no boolean could give.
        self.local_only: tuple[str, ...] = ()
        self._said = False

    # -- reads --------------------------------------------------------------

    def get_graphs(self, name: str) -> Any:
        """The release document. Upstream owns it: the hub stores the derive's
        rows and the local CAS never holds a lane the hub did not stamp."""
        if self.upstream is None:
            return self.local.get_graphs(name)
        return self.upstream.get_graphs(name)

    def put_graphs(self, name: str, document: Any) -> None:
        self.local.put_graphs(name, document)

    def has_artifact(self, graph: str, env: Any) -> bool:
        if self.local.has_artifact(graph, env):
            return True
        return self.upstream is not None and self.upstream.has_artifact(graph, env)

    def artifact_skew(self, graph: str, env: Any) -> Optional[str]:
        """The LOCAL position's shape verdict, passed through (pgw#1573).

        pgw#1561 left this as its one non-blocking observation: the skew gate
        lived only in the compile CLI's own bare ``LocalGraphStore``, so a
        serving pod holding a skewed local position still answered
        ``has_artifact`` TRUE and repaired itself the expensive way — an
        adoption hole, then a full re-compile. With the gate on the store every
        reader of the store asks the same question.

        Only the local tier: the upstream answer is verified by digest on the
        way in and has no position to be skewed at.
        """
        probe = getattr(self.local, "artifact_skew", None)
        return None if probe is None else probe(graph, env)

    def fetch_artifact(self, graph: str, env: Any, destination: Any) -> Any:
        """Branches ① and ② of the canonical flow, and nothing else.

        ① LOCAL HIT — bytes this pod already holds. No network, unrevocable
        mid-boot, and the common case on every boot after the first.

        ② REMOTE HIT — the fleet pool has it and this box does not. The bytes
        are fetched, digest-verified by the upstream tier, and **BANKED INTO
        THE LOCAL CAS on the way through**, so the next boot takes branch ①.
        Without that bank a pod re-downloaded every artifact on every restart
        and a hub outage turned a warm box cold — "check local then remote" is
        only half a cache if the remote answer is never kept.

        Banking is BEST-EFFORT and never costs the arm: the bytes are already
        in hand and already verified, so a full disk or a raced sibling publish
        means one more download next boot, not a hole.
        """
        if self.local.has_artifact(graph, env):
            return self.local.fetch_artifact(graph, env, destination)
        if self.upstream is None:
            return None
        fetched = self.upstream.fetch_artifact(graph, env, destination)
        if fetched is not None:
            self._bank(graph, env, Path(fetched))
        return fetched

    def _bank(self, graph: str, env: Any, artifact: Path) -> None:
        """Keep a verified upstream artifact in this box's own CAS."""
        manifest = None
        try:
            manifest = self.upstream.get_manifest(graph, env)
        except Exception:  # noqa: BLE001 — a manifest-less hit is still a hit
            logger.debug("adopt: upstream manifest unreadable for %s", graph,
                         exc_info=True)
        if manifest is None:
            # NOT banked without one. `LocalGraphStore.publish_artifact` stores
            # the manifest beside the bytes and `EndpointHost.setup` reads it
            # back to check this host satisfies the artifact's floors; banking
            # bytes with no manifest would make a cached artifact adoptable on
            # a machine the fleet answer would have refused.
            logger.info(
                "adopt: %s fetched from the fleet pool and NOT cached locally "
                "— the answer carried no requirements manifest, and an "
                "artifact with no stated floors must not become a local hit",
                graph,
            )
            return
        try:
            self.local.publish_artifact(graph, env, artifact, manifest)
        except Exception as exc:  # noqa: BLE001 — the arm has its bytes already
            logger.warning(
                "adopt: %s fetched from the fleet pool but not cached locally "
                "(%s: %s); the next boot re-downloads it",
                graph, type(exc).__name__, exc,
            )

    def get_manifest(self, graph: str, env: Any) -> Any:
        found = self.local.get_manifest(graph, env)
        if found is not None or self.upstream is None:
            return found
        return self.upstream.get_manifest(graph, env)

    def fetch_program(self, graph: str, destination: Path) -> Path:
        """This box's serialized ``ExportedProgram`` for one GRAPH IDENTITY.

        KEYED BY IDENTITY, NOT BY ADDRESS (Paul, 2026-08-19, address-free). The
        document states which graph a hole is; it states nothing about where
        bytes live, because a serialized program's digest is machine-scoped —
        pgw#1462p2 measured 14/14 graph identities reproducing across boxes and
        0/14 blob digests. A box can only use bytes it made, and the identity is
        the one key every box agrees on, so it is the key here.

        Two tiers, cheapest first: this pod's own store, then the IMAGE's baked
        one. Both answer the same question under the same key, so the order is a
        cost decision and never a correctness one.

        **THE BAKED TIER (pgw#1462 part 2) STAYS, with its scope narrowed.** The
        pod's store is ``<TENSORHUB_CACHE_DIR>/cas`` and the builder bakes to
        ``/app/.tensorhub/derive-cas`` — two directories, which is why baked
        programs were unreachable for as long as they existed. It is still
        exactly right for the REGENERATE arm, where the build container derived
        the document and the bytes together so its programs answer for the
        identities the document names. It cannot serve a release stamped from a
        committed lock — those bytes were never on this machine — and that case
        is rescued by the identity instead, which lets a LOCAL derive supply
        them.

        **THE BYTES ARE VERIFIED BEFORE THEY ARE TRUSTED**, by the store's own
        scrub: ``LocalGraphStore.fetch_program`` verifies the object behind the
        graph ref and raises rather than hand back bytes that no longer hash to
        what was banked. These go straight to a compiler.

        **A CORRUPT TIER IS A MISS, NOT A DEATH** — the next tier may hold a
        good copy. The corruption is REMEMBERED: if nothing serves the program,
        the refusal names it, because "no tier had it" and "a tier had it and it
        was rotten" have very different remedies.

        A program no tier holds stays a TYPED PER-GRAPH refusal — it costs its
        own graph and never the boot.
        """
        # A MALFORMED KEY IS NOT CORRUPT BYTES. The store validates the graph
        # spelling and raises `StoreError` for a bad one — indistinguishable,
        # inside the loop below, from "these bytes are rotten", which would
        # send the reader to scrub a disk over a wrong argument. Checked once,
        # here, against torchcg's own predicate.
        from .._vendor.torchcg import is_graph_hash

        if not is_graph_hash(graph):
            raise ProgramBlobUnreachable(
                f"{graph!r} is not a cg-graph-v1 identity, so no store can be "
                f"asked for a program under it. A mint hole is keyed by the "
                f"graph the document names — not by a blob digest, which since "
                f"the address-free ruling is not in the document at all."
            )
        fetch = getattr(self.upstream, "fetch_program", None)
        if fetch is not None:
            return Path(fetch(graph, destination))
        rotten: list[str] = []
        for tier, store in (("this pod's own", self.local),
                            ("the image's baked", self.baked)):
            if store is None:
                continue
            try:
                found = store.fetch_program(graph, destination)
            except _STORE_FAULTS as exc:
                # NARROW ON PURPOSE. A blanket `except Exception` here reported
                # an `AttributeError` — a wiring mistake — as "corrupted at
                # rest", which sends the reader to scrub a disk over a typo.
                # Only the store's OWN failure vocabulary means "these bytes
                # are bad"; anything else is this code being wrong and must
                # surface as itself.
                rotten.append(f"{tier} store ({type(exc).__name__}: {exc})")
                continue
            if found is not None:
                return Path(found)
        if rotten:
            raise ProgramBlobUnreachable(
                f"the serialized program for {graph} is present but does not "
                f"survive its own integrity scrub in " + "; ".join(rotten) +
                " — truncated or corrupted at rest, and no other tier holds a "
                "good copy."
            )
        raise ProgramBlobUnreachable(
            f"this box holds no serialized program for graph {graph}. That is "
            f"ORDINARY on a pod that has not derived this endpoint yet, and the "
            f"remedy is LOCAL: derive here (`gen-worker lock`, or `compile`) so "
            f"the programs land in this machine's own store under the identity "
            f"the release stamped. Nothing is fetched from anywhere — a "
            f"serialized program's bytes are machine-scoped (pgw#1462p2), so "
            f"the only ones this box can compile are the ones it made."
        )

    # -- the mint's publish -------------------------------------------------

    def publish_artifact(
        self, graph: str, env: Any, artifact: Any, manifest: Any,
    ) -> Any:
        """Bank the artifact locally, then offer it to the fleet.

        Local first and unconditionally: that write is what makes a pod killed
        mid-mint strictly better off, and it must not be contingent on a
        network. The upstream offer is best-effort and its absence is a
        counted fact, never an exception that would cost the graph.
        """
        outcome = self.local.publish_artifact(graph, env, artifact, manifest)
        if self.upstream is None:
            return outcome
        try:
            return self.upstream.publish_artifact(graph, env, artifact, manifest)
        except Exception as exc:  # noqa: BLE001 — a local mint is still a mint
            with self._lock:
                self.local_only = self.local_only + (graph,)
                first = not self._said
                self._said = True
            if first:
                activity_mod.emit_event(
                    KIND_PUBLISH_LOCAL_ONLY,
                    f"minted artifacts are durable on THIS POD and the fleet "
                    f"is not receiving them: {type(exc).__name__}: {exc}. The "
                    f"fleet publish leg (compiled_graphs.publish_intent / "
                    f"publish_complete) has no worker-side caller at HEAD — "
                    f"pgw#1368/th#2132 own restoring it.",
                    phase=type(exc).__name__,
                    graph_specialization=str(graph)[:300],
                )
            logger.warning(
                "self-mint: %s published locally only: %s", graph, exc)
            return outcome

    def facts(self) -> Dict[str, Any]:
        return {
            "tier": "local+hub" if self.upstream is not None else "local",
            "local_only": len(self.local_only),
        }


def graph_store(
    cas_dir: Path,
    upstream: Optional[Any] = None,
    baked_root: Optional[Path] = None,
) -> TieredGraphStore:
    """THE compiled-graph store — every entry point builds it here (pgw#1573).

    A pod, ``gen-worker up``, ``gen-worker compile`` and the CI runner all take
    this constructor, so a graph is present-or-absent for all of them at once.
    ``upstream`` is the fleet pool (:class:`~gen_worker.serving.hub_store.
    HubGraphStore`) when this process has a release to adopt for and nothing
    when it does not — the ONE thing that differs between a pod and a box, and
    it is a stated argument rather than a different class.

    ``baked_root`` is the IMAGE's read-only exported-program CAS; omitted, it
    is resolved from settings (`baked_program_cas_dir`), which is what every
    production call site wants. Passing it explicitly is for tests and for a
    cozy-local run whose blobs are somewhere else.
    """
    from .._vendor.tensorfs import LocalCAS
    from ..models.cache_paths import baked_program_cas_dir

    from .._vendor.torchcg.store import LocalGraphStore

    root = baked_program_cas_dir() if baked_root is None else Path(baked_root)
    # A read-only tier, and a GraphStore rather than a bare CAS because the
    # programs are addressed by graph identity now. Opened on an EXISTING
    # directory only: `LocalCAS.__init__` mkdirs, so absence is decided before
    # it is constructed, not by it.
    baked = (
        LocalGraphStore(LocalCAS(root))
        if root is not None and Path(root).is_dir()
        else None
    )
    return TieredGraphStore(LocalGraphStore(LocalCAS(Path(cas_dir))), upstream, baked)


__all__ = [
    "KIND_PUBLISH_LOCAL_ONLY",
    "ProgramBlobUnreachable",
    "TieredGraphStore",
    "graph_store",
]
