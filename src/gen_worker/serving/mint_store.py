"""The store a serving worker adopts from and mints into (pgw#1371).

One ``GraphStore``, two tiers, and the reason they are one object is that a
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

    def __init__(self, local: Any, upstream: Any = None) -> None:
        self.local = local
        self.upstream = upstream
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

        Upstream first when it can, then the pod's own CAS by content address:
        a digest is a digest, so a blob this pod already holds needs no hop.

        **A THIRD UNWIRED LEG, named rather than crashed.** th#2133's adopt
        answer carries each graph's ``program`` digest and NO transport for
        it — its ``transport.files`` presign the compiled ARTIFACT, not the
        serialized graph — so a pod whose CAS has never seen the blob has no
        route to it. pgw#1370 owns publishing the blob and the hub owns
        answering with a way to fetch it. Until then this raises a typed
        refusal per graph, which the mint records as a ``MintFailure`` naming
        the owner instead of dying on an ``AttributeError``.
        """
        fetch = getattr(self.upstream, "fetch_program", None)
        if fetch is not None:
            return Path(fetch(digest, destination))
        cas = getattr(self.local, "cas", None)
        if cas is not None and cas.contains(digest):
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(Path(cas.object_path(digest)).read_bytes())
            return destination
        raise ProgramBlobUnreachable(
            f"graph blob {digest} is not in this pod's CAS and the adopt "
            f"answer offers no transport for it: th#2133's per-graph rows "
            f"carry the `program` DIGEST but presign only the compiled "
            f"artifact. pgw#1370 owns publishing the serialized "
            f"ExportedProgram and the hub owns answering with a fetch route. "
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


def worker_store(cas_dir: Path, upstream: Optional[Any] = None) -> TieredGraphStore:
    """The store a real serving pod uses, over its own tensorfs CAS."""
    from .._vendor.tensorfs import LocalCAS
    from .._vendor.torchcg.store import LocalGraphStore

    return TieredGraphStore(LocalGraphStore(LocalCAS(Path(cas_dir))), upstream)


__all__ = [
    "KIND_PUBLISH_LOCAL_ONLY",
    "ProgramBlobUnreachable",
    "TieredGraphStore",
    "worker_store",
]
