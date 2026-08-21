from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Dict, Optional

from .. import activity as activity_mod

logger = logging.getLogger(__name__)


def _store_faults() -> tuple[type[BaseException], ...]:
    from .._vendor.tensorfs.local import DigestMismatch
    from .._vendor.torchcg.store import StoreError

    return (StoreError, DigestMismatch, FileNotFoundError, ValueError)


_STORE_FAULTS = _store_faults()

KIND_PUBLISH_LOCAL_ONLY = "self_mint_publish_local_only"


class ProgramBlobUnreachable(RuntimeError):
    """The serialized graph this hole must compile cannot be fetched."""


class TieredGraphStore:
    """Local CAS first, hub second — for reads AND for the mint's publishes."""

    def __init__(self, local: Any, upstream: Any = None, baked: Any = None) -> None:
        self.local = local
        self.upstream = upstream
        self.baked = baked
        self._lock = threading.Lock()
        self.local_only: tuple[str, ...] = ()
        self._said = False

    def get_graphs(self, name: str) -> Any:
        """The release document."""
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
        probe = getattr(self.local, "artifact_skew", None)
        return None if probe is None else probe(graph, env)

    def fetch_artifact(self, graph: str, env: Any, destination: Any) -> Any:
        """Branches ① and ② of the canonical flow, and nothing else."""
        if self.local.has_artifact(graph, env):
            return self.local.fetch_artifact(graph, env, destination)
        if self.upstream is None:
            return None
        fetched = self.upstream.fetch_artifact(graph, env, destination)
        if fetched is not None:
            self._bank(graph, env, Path(fetched))
        return fetched

    def _bank(self, graph: str, env: Any, artifact: Path) -> None:
        manifest = None
        try:
            manifest = self.upstream.get_manifest(graph, env)
        except Exception:  # noqa: BLE001 — a manifest-less hit is still a hit
            logger.debug("adopt: upstream manifest unreadable for %s", graph,
                         exc_info=True)
        if manifest is None:
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
        """This box's serialized ``ExportedProgram`` for one GRAPH IDENTITY."""
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

    def publish_artifact(
        self, graph: str, env: Any, artifact: Any, manifest: Any,
    ) -> Any:
        """Bank the artifact locally, then offer it to the fleet."""
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
    from .._vendor.tensorfs import LocalCAS
    from ..models.cache_paths import baked_program_cas_dir

    from .._vendor.torchcg.store import LocalGraphStore

    root = baked_program_cas_dir() if baked_root is None else Path(baked_root)
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
