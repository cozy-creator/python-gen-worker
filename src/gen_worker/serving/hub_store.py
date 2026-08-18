"""The hub-backed GraphStore (pgw#1372): th#2133's adopt route, thin.

One ask per boot — ``GET /v1/worker/releases/<release>/compiled-graphs
?lane=<lane>&sm=<sm>`` — answers, for every graph in the release's lane, the
artifact for THIS release's exact env + this sm (content digest, presigned
transport URL, the mint's requirements manifest) or a per-graph MISS.
PARTIAL-HIT is the wire contract; exact-env is the ruling (no compat
ranking on this route).

The TRANSPORT is a seam (:class:`ReleaseGraphTransport`): the route lands in
th#2133 and the worker's HTTP/procsplit wiring follows it — this store only
states the answer shape and the verification. Publishing never happens here:
the boot-side store is read-only, and mint publishes ride pgw#1371's
publisher.

Answer shape (built to th#2133's spec; the route is the authority once
merged)::

    {
      "document":  {...GraphSetDocument...},
      "artifacts": {"cg-graph-v1-...": {"digest": "<sha256 hex>",
                                        "url": "<presigned>",
                                        "manifest": {...RequirementsManifest...}}},
      "misses":    ["cg-graph-v1-..."]
    }
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Optional, Protocol

from .._vendor.torchcg.document import DocumentError, GraphSetDocument
from .._vendor.torchcg.graph_identity import EnvIdentity
from .._vendor.torchcg.requirements import RequirementsError, RequirementsManifest
from .._vendor.torchcg.store import PublishOutcome, StoreError


class ReleaseGraphTransport(Protocol):
    """What the hub wiring must provide — and all it must provide."""

    def release_compiled_graphs(
        self, release_id: str, lane: str, sm: str
    ) -> Mapping[str, Any]:
        """The one boot ask: the release's graph document + per-graph answer."""
        ...

    def fetch_blob(self, url: str) -> bytes:
        """Follow one presigned artifact URL."""
        ...


class HubGraphStore:
    """torchcg's ``GraphStore``, read side, over the th#2133 answer."""

    def __init__(
        self, transport: ReleaseGraphTransport, release_id: str, lane: str, sm: str
    ) -> None:
        self._transport = transport
        self._release_id = str(release_id)
        self._lane = str(lane)
        self._sm = str(sm)
        self._answer: Optional[Mapping[str, Any]] = None
        self._document: Optional[GraphSetDocument] = None

    # -- the one ask --------------------------------------------------------

    def _resolve(self) -> Mapping[str, Any]:
        if self._answer is None:
            answer = self._transport.release_compiled_graphs(
                self._release_id, self._lane, self._sm
            )
            if not isinstance(answer, Mapping):
                raise StoreError(
                    f"release {self._release_id}: adopt route answered "
                    f"{type(answer).__name__}, not an object"
                )
            self._answer = answer
        return self._answer

    def _env(self) -> EnvIdentity:
        document = self.get_graphs(self._release_id)
        if document is None:
            raise StoreError(f"release {self._release_id} stamped no graph document")
        return EnvIdentity(closure=document.closure, sm=self._sm)

    def _entry(self, graph: str, env: EnvIdentity) -> Optional[Mapping[str, Any]]:
        if env != self._env():
            # This store holds exactly one env — the release's own. Any other
            # ask is a clean miss, never a compat answer (exact-env ruling).
            return None
        artifacts = self._resolve().get("artifacts")
        entry = artifacts.get(graph) if isinstance(artifacts, Mapping) else None
        return entry if isinstance(entry, Mapping) else None

    # -- GraphStore ---------------------------------------------------------

    def get_graphs(self, name: str) -> Optional[GraphSetDocument]:
        if name != self._release_id:
            return None
        if self._document is None:
            raw = self._resolve().get("document")
            if raw is None:
                return None
            try:
                self._document = GraphSetDocument.decode(raw)
            except DocumentError as exc:
                raise StoreError(
                    f"release {self._release_id} graph document is unreadable: {exc}"
                ) from exc
        return self._document

    def put_graphs(self, name: str, document: GraphSetDocument) -> None:
        raise StoreError(
            "the release graph document is stamped by the publish pipeline "
            "(th#2134); a worker never writes it"
        )

    def has_artifact(self, graph: str, env: EnvIdentity) -> bool:
        return self._entry(graph, env) is not None

    def fetch_artifact(
        self, graph: str, env: EnvIdentity, destination: str | Path
    ) -> Optional[Path]:
        entry = self._entry(graph, env)
        if entry is None:
            return None
        url = str(entry.get("url") or "")
        digest = str(entry.get("digest") or "")
        if not url or not digest:
            raise StoreError(
                f"artifact ({graph}, {env.value}): the answer row is missing "
                f"its url or digest"
            )
        payload = self._transport.fetch_blob(url)
        observed = hashlib.sha256(payload).hexdigest()
        if observed != digest:
            raise StoreError(
                f"artifact ({graph}, {env.value}) failed digest verification: "
                f"answer said {digest[:16]}..., bytes hash {observed[:16]}..."
            )
        target = Path(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{target.name}.", dir=target.parent
        )
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
            os.replace(temporary_name, target)
        finally:
            Path(temporary_name).unlink(missing_ok=True)
        return target

    def publish_artifact(
        self,
        graph: str,
        env: EnvIdentity,
        artifact: str | Path,
        manifest: RequirementsManifest,
    ) -> PublishOutcome:
        raise StoreError(
            "the boot-side hub store is read-only; a minted artifact publishes "
            "through the pgw#1371 mint publisher, never the adopt route"
        )

    def get_manifest(
        self, graph: str, env: EnvIdentity
    ) -> Optional[RequirementsManifest]:
        entry = self._entry(graph, env)
        if entry is None:
            return None
        raw = entry.get("manifest")
        if raw is None:
            return None
        try:
            return RequirementsManifest.decode(raw)
        except RequirementsError as exc:
            raise StoreError(
                f"manifest ({graph}, {env.value}) is unreadable: {exc}"
            ) from exc


__all__ = ["HubGraphStore", "ReleaseGraphTransport"]
