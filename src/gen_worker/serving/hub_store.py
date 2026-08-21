"""The hub-backed GraphStore (pgw#1372): th#2133's adopt route, thin.

One ask per boot — ``GET /v1/worker/releases/<release>/compiled-graphs
?lane=<lane>&sm=<sm>`` — answers, for every graph in the release's lane, the
artifact for THIS release's exact env + this sm (content digest, presigned
transport manifest, the mint's requirements manifest) or a per-graph MISS.
PARTIAL-HIT is the wire contract; exact-env is the ruling (no compat
ranking on this route).

The TRANSPORT is a seam (:class:`ReleaseGraphTransport`) with two
implementations that live here because both are two-liners over machinery
that already exists: :class:`BrokerReleaseGraphTransport` (production — the
call is parent-mediated, since the compute child holds no credential) and
``__main__``'s plain-HTTP one (cozy-local / CI). Publishing never happens
here: the boot-side store is read-only, and mint publishes ride pgw#1371's
publisher.

THE ANSWER SHAPE IS th#2133's, verbatim — this module was written against a
SPECULATED ``{document, artifacts, misses}`` shape before the route landed,
and the route answers something else::

    {"object": "release_compiled_graphs", "release_id": "...",
     "binding_generation": 0, "env_compile_stack": [["torch", "2.13.0"], ...],
     "lane": "sdxl.diffusers@1+plain.bf16@1", "lane_stamped": true,
     "lane_contract": {"stamp": ..., "contract_digest": ..., "document": {...},
                       "requires": ...},
     "sm": "sm_89", "empty": false, "hits": 30, "misses": 6,
     "graphs": [{"graph_hash": "cg-graph-v1:...", "graph_specialization": "...",
                 "status": "hit"|"miss"|"transport_unavailable",
                 "found": true, "module_path": "...", "ingress": {...},
                 "content_digest": "...", "artifact_path": "...",
                 "requirements_manifest": {...},
                 "transport": {"snapshot_digest": ..., "files": [...]}}, ...],
     "targets": [...], "unobserved_targets": [...], "passes": [...]}

The lane's graph document is REBUILT from that answer rather than shipped
whole: the hub stores the derive's rows, not the derive's bytes, and the
rebuild is the one place that knows it.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Protocol

from .._vendor.torchcg.document import (
    DOCUMENT_FORMAT,
    DocumentError,
    GraphSetDocument,
)
from .._vendor.torchcg.graph_identity import EnvIdentity
from .._vendor.torchcg.requirements import RequirementsError, RequirementsManifest
from .._vendor.torchcg.store import PublishOutcome, StoreError

HTTP_TIMEOUT_S = 60.0

ADOPT_PATH = "/v1/worker/releases/{release_id}/compiled-graphs"

HIT = "hit"


class ReleaseNotStamped(Exception):
    """The release carries no stamped compiled-graph document."""


class ReleaseGraphTransport(Protocol):
    """What the hub wiring must provide — and all it must provide."""

    def release_compiled_graphs(
        self, release_id: str, lane: str, sm: str
    ) -> Mapping[str, Any]:
        """The one boot ask: the release's per-graph adopt answer."""
        ...

    def fetch_blob(self, url: str) -> bytes:
        """Follow one presigned artifact URL."""
        ...


class BrokerReleaseGraphTransport:
    """The PRODUCTION transport: the adopt ask, parent-mediated."""

    def __init__(
        self,
        *,
        base_url: str = "",
        bearer: str = "",
        timeout_s: float = HTTP_TIMEOUT_S,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.bearer = bearer
        self.timeout_s = timeout_s

    def release_compiled_graphs(
        self, release_id: str, lane: str, sm: str
    ) -> Mapping[str, Any]:
        from ..procsplit import broker

        response = broker.request(
            "GET",
            ADOPT_PATH.format(release_id=str(release_id)),
            base_url=self.base_url,
            bearer=self.bearer,
            params={"lane": str(lane), "sm": str(sm)},
            timeout=self.timeout_s,
        )
        if response.status_code == 404:
            raise ReleaseNotStamped(
                f"release {release_id} carries no stamped compiled-graph "
                f"document; serving eager"
            )
        if response.status_code != 200:
            raise StoreError(
                f"adopt route answered {response.status_code} for release "
                f"{release_id} lane={lane} sm={sm}: {response.text[:400]}"
            )
        payload = response.json()
        if not isinstance(payload, Mapping):
            raise StoreError("adopt route answered non-object JSON")
        return payload

    def fetch_blob(self, url: str) -> bytes:
        import requests

        got = requests.get(url, timeout=self.timeout_s)
        got.raise_for_status()
        return bytes(got.content)


def _graph_rows(answer: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    rows = answer.get("graphs")
    if not isinstance(rows, list):
        return ()
    return tuple(row for row in rows if isinstance(row, Mapping))


class HubGraphStore:

    def __init__(
        self, transport: ReleaseGraphTransport, release_id: str, lane: str, sm: str
    ) -> None:
        self._transport = transport
        self._release_id = str(release_id)
        self._lane = str(lane)
        self._sm = str(sm)
        self._answer: Optional[Mapping[str, Any]] = None
        self._document: Optional[GraphSetDocument] = None
        self._rows: Optional[dict[str, Mapping[str, Any]]] = None

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

    def _row(self, graph: str) -> Optional[Mapping[str, Any]]:
        if self._rows is None:
            self._rows = {
                str(row.get("graph_hash") or ""): row
                for row in _graph_rows(self._resolve())
            }
        return self._rows.get(str(graph))

    @property
    def misses(self) -> tuple[str, ...]:
        """The graph hashes this env still needs, in answer order."""

        return tuple(
            str(row.get("graph_hash") or "")
            for row in _graph_rows(self._resolve())
            if str(row.get("status") or "") == "miss"
        )

    def _env(self) -> EnvIdentity:
        document = self.get_graphs(self._release_id)
        if document is None:
            raise StoreError(f"release {self._release_id} stamped no graph document")
        return document.env(self._sm)

    def _entry(self, graph: str, env: EnvIdentity) -> Optional[Mapping[str, Any]]:
        if env != self._env():
            return None
        row = self._row(graph)
        if row is None or str(row.get("status") or "") != HIT:
            return None
        return row

    def get_graphs(self, name: str) -> Optional[GraphSetDocument]:
        """Rebuild THIS lane's graph document out of the adopt answer."""

        if name != self._release_id:
            return None
        if self._document is not None:
            return self._document
        answer = self._resolve()
        raw_stack = answer.get("env_compile_stack")
        if not isinstance(raw_stack, (list, tuple)) or not raw_stack:
            raise StoreError(
                f"release {self._release_id} answers no `env_compile_stack`; "
                f"the artifact key is (trace digest, sm, compile stack) since "
                f"pgw#1489 and this hub still states only a closure hash"
            )
        stack = [
            [str(row[0]), str(row[1])]
            for row in raw_stack
            if isinstance(row, (list, tuple)) and len(row) == 2
        ]
        lanes: list[dict[str, Any]] = []
        if not answer.get("empty") and answer.get("lane_stamped"):
            records: list[dict[str, Any]] = []
            for row in _graph_rows(answer):
                ingress = row.get("ingress")
                if not isinstance(ingress, Mapping):
                    raise StoreError(
                        f"release {self._release_id} graph "
                        f"{row.get('graph_hash')!r} carries no ingress contract; "
                        f"the adopt answer cannot be dispatched against"
                    )
                records.append({
                    "graph": str(row.get("graph_hash") or ""),
                    "target": str(row.get("module_path") or ""),
                    "ingress": ingress,
                })
            observed = {record["target"] for record in records}
            declared = [
                str(target) for target in (answer.get("targets") or [])
                if isinstance(target, str)
            ]
            unobserved = [
                str(target) for target in (answer.get("unobserved_targets") or [])
                if isinstance(target, str)
            ]
            targets = sorted(observed | set(declared) | set(unobserved))
            lanes.append({
                "contract": str(answer.get("lane") or self._lane),
                "targets": targets,
                "graphs": records,
                "unobserved_targets": sorted(set(unobserved) - observed),
                "passes": [
                    str(name) for name in (answer.get("passes") or [])
                    if isinstance(name, str)
                ],
            })
        try:
            self._document = GraphSetDocument.decode(
                {"v": DOCUMENT_FORMAT, "stack": stack, "lanes": lanes}
            )
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
        digest = str(entry.get("content_digest") or "")
        artifact_path = str(entry.get("artifact_path") or "")
        if not digest or not artifact_path:
            raise StoreError(
                f"artifact ({graph}, {env.value}): the answer row is missing "
                f"its content digest or artifact path"
            )
        payload = self._fetch_from_transport(graph, env, entry, artifact_path)
        observed = hashlib.sha256(payload).hexdigest()
        if observed != digest.removeprefix("sha256:"):
            raise StoreError(
                f"artifact ({graph}, {env.value}) failed digest verification: "
                f"answer said {digest[:23]}..., bytes hash {observed[:16]}..."
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

    def _fetch_from_transport(
        self,
        graph: str,
        env: EnvIdentity,
        entry: Mapping[str, Any],
        artifact_path: str,
    ) -> bytes:

        transport = entry.get("transport")
        files = transport.get("files") if isinstance(transport, Mapping) else None
        if not isinstance(files, list):
            raise StoreError(
                f"artifact ({graph}, {env.value}) answered a hit with no "
                f"transport manifest"
            )
        for row in files:
            if not isinstance(row, Mapping) or str(row.get("path") or "") != artifact_path:
                continue
            chunks = row.get("chunks")
            if isinstance(chunks, list) and chunks:
                blob = bytearray()
                for chunk in chunks:
                    if not isinstance(chunk, Mapping):
                        continue
                    blob += self._transport.fetch_blob(str(chunk.get("url") or ""))
                return bytes(blob)
            url = str(row.get("url") or "")
            if not url:
                raise StoreError(
                    f"artifact ({graph}, {env.value}): transport row "
                    f"{artifact_path!r} carries neither url nor chunks"
                )
            return self._transport.fetch_blob(url)
        raise StoreError(
            f"artifact ({graph}, {env.value}): the transport manifest names no "
            f"file {artifact_path!r}"
        )

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
        raw = entry.get("requirements_manifest")
        if raw is None:
            return None
        try:
            return RequirementsManifest.decode(raw)
        except RequirementsError as exc:
            raise StoreError(
                f"manifest ({graph}, {env.value}) is unreadable: {exc}"
            ) from exc


__all__ = [
    "ADOPT_PATH",
    "BrokerReleaseGraphTransport",
    "HubGraphStore",
    "ReleaseGraphTransport",
    "ReleaseNotStamped",
]
