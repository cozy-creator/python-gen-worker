"""The store a serving worker adopts from and mints into (pgw#1371).

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

    def fetch_artifact(self, graph: str, env: Any, destination: Any) -> Any:
        if self.local.has_artifact(graph, env):
            return self.local.fetch_artifact(graph, env, destination)
        if self.upstream is None:
            return self.local.fetch_artifact(graph, env, destination)  # its refusal
        return self.upstream.fetch_artifact(graph, env, destination)

    def get_manifest(self, graph: str, env: Any) -> Any:
        found = self.local.get_manifest(graph, env)
        if found is not None or self.upstream is None:
            return found
        return self.upstream.get_manifest(graph, env)

    def fetch_program(self, digest: str, destination: Path) -> Path:
        """One graph's serialized ``ExportedProgram`` — the mint's INPUT.

        THREE tiers, cheapest-and-most-trusted first: an upstream that can
        serve one, then the pod's own CAS, then the IMAGE's baked CAS. All
        three are keyed by CONTENT ADDRESS, so the order is a cost decision
        and never a correctness one — a digest is a digest.

        **THE BAKED TIER IS pgw#1462 PART 2, and it closes a gap that was
        assumed closed.** th#2162 argued no hub route was needed because
        *"pgw's miner already falls through to its local CAS by content
        address"*. The fallthrough is real; the directory was wrong. The pod's
        CAS is ``<TENSORHUB_CACHE_DIR>/cas`` (``/tmp/tensorhub-cache/cas``) and
        the builder bakes to ``/app/.tensorhub/derive-cas``, so every baked blob
        has been unreachable since the first image carried one — a permanent
        silent miss, which is why nothing ever reported it.

        **THE BYTES ARE VERIFIED BEFORE THEY ARE TRUSTED, BY THE STORE'S OWN
        SCRUB.** These bytes are fed straight to a compiler, so each tier is
        probed with ``verify_object`` — presence AND integrity in one call —
        rather than with ``contains`` plus a hash written here. Two reasons it
        is that call and not the obvious one: a second implementation of
        integrity checking is a second thing that can disagree, and the two
        tensorfs snapshots in this tree DISAGREE about ``contains`` — the
        vendored one rehashes and RAISES ``DigestMismatch``, tensorfs master's
        is "presence, not integrity" and answers True for bytes corrupted in
        place. ``verify_object`` means the same thing in both, so this code
        does not silently change behaviour when the vendor is re-synced.

        **A CORRUPT TIER IS A MISS, NOT A DEATH** — the next tier may hold a
        good copy, and preferring it is strictly better than refusing. But the
        corruption is REMEMBERED: if nothing serves the blob, the refusal names
        it, because "no tier had it" and "a tier had it and it was rotten" have
        very different remedies.

        A blob no tier holds stays a TYPED PER-GRAPH refusal — it costs its own
        graph and never the boot.
        """
        fetch = getattr(self.upstream, "fetch_program", None)
        if fetch is not None:
            return Path(fetch(digest, destination))
        rotten: list[str] = []
        for tier, cas in (("this pod's own", getattr(self.local, "cas", None)),
                          ("the image's baked", self.baked)):
            if cas is None:
                continue
            try:
                source = Path(cas.verify_object(digest))
            except FileNotFoundError:
                continue  # an ordinary miss: this tier does not hold it
            except Exception as exc:  # noqa: BLE001 - any scrub failure is an answer
                rotten.append(f"{tier} CAS ({type(exc).__name__}: {exc})")
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(source.read_bytes())
            return destination
        if rotten:
            raise ProgramBlobUnreachable(
                f"graph blob {digest} is present but does not survive its own "
                f"integrity scrub in " + "; ".join(rotten) + " — the object was "
                f"truncated or corrupted at rest, and no other tier holds a good "
                f"copy. Refusing to compile bytes that are not the graph the "
                f"release stamped."
            )
        raise ProgramBlobUnreachable(
            f"graph blob {digest} is in neither this pod's CAS nor the image's "
            f"baked program CAS, and the adopt answer offers no transport for "
            f"it: th#2133's per-graph rows carry the `program` DIGEST but "
            f"presign only the compiled artifact. An image whose build used a "
            f"COMMITTED endpoint.lock bakes no programs (th#2162 renders no "
            f"derive step for it), which is the ordinary way to reach this, and "
            f"the blobs are NOT regenerable here: a re-trace reproduces the "
            f"graph identity but not the serialized bytes, so it would address "
            f"different digests than the release stamped. pgw#1370 owns "
            f"emitting these blobs where the pod can reach them. "
            f"This mint will NEVER re-trace to work around it (author code "
            f"does not run at mint time)."
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


def worker_store(
    cas_dir: Path,
    upstream: Optional[Any] = None,
    baked_root: Optional[Path] = None,
) -> TieredGraphStore:
    """The store a real serving pod uses, over its own tensorfs CAS.

    ``baked_root`` is the IMAGE's read-only exported-program CAS; omitted, it
    is resolved from settings (`baked_program_cas_dir`), which is what both
    production call sites want. Passing it explicitly is for tests and for a
    cozy-local run whose blobs are somewhere else.
    """
    from .._vendor.tensorfs import LocalCAS
    from ..models.cache_paths import baked_program_cas_dir

    from .._vendor.torchcg.store import LocalGraphStore

    root = baked_program_cas_dir() if baked_root is None else Path(baked_root)
    # A read-only tier: LocalCAS is opened on an EXISTING directory only, so a
    # missing bake never creates one. `LocalCAS.__init__` mkdirs, so absence is
    # decided before it is constructed, not by it.
    baked = LocalCAS(root) if root is not None and Path(root).is_dir() else None
    return TieredGraphStore(LocalGraphStore(LocalCAS(Path(cas_dir))), upstream, baked)


__all__ = [
    "KIND_PUBLISH_LOCAL_ONLY",
    "ProgramBlobUnreachable",
    "TieredGraphStore",
    "worker_store",
]
