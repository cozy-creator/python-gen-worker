from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

from ..config import get_settings
from .refs import TensorhubRef
import requests


# In-memory shape of a resolved cozy: ref. PRODUCTION workers never resolve
# these themselves — the orchestrator pre-resolves every cozy ref a job needs
# and ships the manifest + presigned URLs via
# JobExecutionRequest.resolved_repos_by_id over gRPC. STANDALONE clients
# (gen-worker run/serve/prefetch under cozy, #379) resolve the same shape over
# HTTP via ``resolve_repo`` against tensorhub's public resolve route (th#560).


@dataclass(frozen=True)
class WorkerResolvedChunk:
    """One CAS object of a chunked file (manifest v2, th#1303).

    ``sha256`` is bare lowercase hex; ``length`` comes from the manifest so a
    chunk's cumulative offset is arithmetic, not an assumption about the
    chunking policy.
    """

    sha256: str
    url: str
    length: int


@dataclass(frozen=True)
class WorkerResolvedRepoFile:
    path: str
    size_bytes: int
    #: LEGACY MIRROR (manifest v1): bare blake3 hex. EMPTY on every v2 entry.
    #: Never read this to decide integrity — read ``cas_ref()``, which carries
    #: the algorithm. Guarding on this field's truthiness is exactly how a v2
    #: snapshot passes a check that never ran (th#1303).
    blake3: str
    url: Optional[str]
    transfer_grant: Optional[dict[str, Any]] = None
    #: Algorithm-tagged whole-file digest ("sha256:<hex>"). v2 populates this;
    #: v1 leaves it empty and the blake3 mirror carries the ref.
    digest: str = ""
    #: Present only when the file is stored as chunks (size > chunk_size).
    chunks: tuple["WorkerResolvedChunk", ...] = ()
    chunk_size_bytes: int = 0

    def cas_ref(self) -> str:
        """The algorithm-tagged digest, whichever manifest version produced it.

        THE one place the v1/v2 dual-read is expressed for a resolved entry.
        Raises rather than returning a bare or empty string, because every
        caller of this is an integrity check and an unreadable digest must
        fail closed, never degrade into "no check".
        """
        d = (self.digest or "").strip().lower()
        if d:
            if ":" not in d:
                raise ValueError(
                    f"resolved file {self.path!r}: digest {d[:16]}… is untagged; "
                    "v2 entries must carry an algorithm prefix"
                )
            return d
        b3 = (self.blake3 or "").strip().lower()
        if b3:
            return b3 if ":" in b3 else f"blake3:{b3}"
        raise ValueError(f"resolved file {self.path!r} carries no digest")


@dataclass(frozen=True)
class WorkerResolvedFlavor:
    """One (tag, flavor) selector row of the resolved tag (th#697)."""

    flavor: str
    size_bytes: int
    placement: Optional[dict[str, Any]] = None


@dataclass(frozen=True)
class WorkerResolvedRepo:
    snapshot_digest: str
    files: List[WorkerResolvedRepoFile]
    # th#697: total checkpoint size + the tag's sibling flavor rows — the
    # local precision ladder's candidate set. Empty on older hubs.
    size_bytes: int = 0
    sibling_flavors: List[WorkerResolvedFlavor] = field(default_factory=list)
    # th#964: the resolved checkpoint's architecture family ("sdxl-pony",
    # ...) — drives the local family lane policy. "" on hubs not sending it.
    model_family: str = ""
    # pgw#654: the resolved checkpoint's hub-stamped training objective
    # ("epsilon" | "v_prediction" | "flow"; "" = unstamped) and distillation
    # fact.
    objective: str = ""
    distilled: bool = False


class HubResolveError(RuntimeError):
    """Base for standalone tensorhub resolve failures."""


class HubRepoNotFoundError(HubResolveError):
    """404: unknown repo/tag/flavor OR a private repo the caller may not see
    (the route deliberately never distinguishes these)."""


class HubAuthError(HubResolveError):
    """401/403: the supplied TENSORHUB_TOKEN was rejected."""


def hub_base_url(base_url: Optional[str] = None) -> str:
    return (base_url or get_settings().tensorhub_url).strip().rstrip("/")


def _parse_chunks(
    ref: TensorhubRef, path: str, raw: Any
) -> tuple[WorkerResolvedChunk, ...]:
    """Parse a v2 entry's ordered chunk list. Order is the file's byte order.

    A malformed chunk list is a HARD failure, never a silent empty list: an
    empty list is indistinguishable from "stored whole", and reading a chunked
    file as whole is how a 40 GiB shard becomes a 0-byte one.
    """
    if not raw:
        return ()
    if not isinstance(raw, list):
        raise HubResolveError(
            f"tensorhub resolve for {ref.canonical()}: {path!r} chunks is not a list"
        )
    out: list[WorkerResolvedChunk] = []
    for i, c in enumerate(raw):
        if not isinstance(c, dict):
            raise HubResolveError(
                f"tensorhub resolve for {ref.canonical()}: {path!r} chunk[{i}] is not an object"
            )
        digest = str(c.get("digest") or c.get("sha256") or "").strip().lower()
        digest = digest.removeprefix("sha256:")
        url = str(c.get("url") or "").strip()
        length = int(c.get("len") or c.get("length") or 0)
        if len(digest) != 64 or not url or length <= 0:
            raise HubResolveError(
                f"tensorhub resolve for {ref.canonical()}: {path!r} chunk[{i}] "
                f"missing digest/url/len"
            )
        out.append(WorkerResolvedChunk(sha256=digest, url=url, length=length))
    return tuple(out)


def resolve_repo(
    ref: TensorhubRef,
    *,
    base_url: Optional[str] = None,
    token: Optional[str] = None,
    timeout: float = 60.0,
) -> WorkerResolvedRepo:
    """Resolve a Hub ref against ``GET /api/v1/repos/:tenant/:name/resolve``.

    Anonymous for public repos; bearer ``token`` (default ``TENSORHUB_TOKEN``)
    for private. Returns the manifest + presigned GET URLs ready for
    ``cozy_snapshot.ensure_snapshot_async``.
    """

    base = hub_base_url(base_url)
    if not base:
        raise HubResolveError(
            "no tensorhub base URL: set TENSORHUB_URL (e.g. https://tensorhub.com)"
        )
    tok = (token or get_settings().tensorhub_token).strip()
    headers = {"Authorization": f"Bearer {tok}"} if tok else {}
    params: dict[str, str] = {}
    if ref.digest:
        params["digest"] = ref.digest
    elif ref.tag:
        params["tag"] = ref.tag
    if ref.flavor:
        params["flavor"] = ref.flavor

    url = f"{base}/api/v1/repos/{ref.owner}/{ref.repo}/resolve"
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=timeout)
    except requests.RequestException as e:
        raise HubResolveError(f"tensorhub resolve failed for {ref.canonical()}: {e}") from e

    if resp.status_code == 404:
        # te#125/th#1238: only the HUB's own 404 means "no such repo". A proxy
        # answering 404 means the hub is unreachable (restarting), and calling
        # that "repo not found" sends the operator hunting a catalog problem
        # that does not exist.
        from ..http_origin import is_proxy_outage

        if is_proxy_outage(resp):
            raise HubResolveError(
                f"tensorhub unreachable resolving {ref.canonical()} — a proxy "
                "answered 404 (backend offline, e.g. hub restarting); retry shortly"
            )
        raise HubRepoNotFoundError(
            f"tensorhub repo {ref.canonical()} not found (unknown repo/tag/"
            "flavor, or a private repo — set TENSORHUB_TOKEN for private pulls)"
        )
    if resp.status_code in (401, 403):
        raise HubAuthError(
            f"tensorhub rejected the token for {ref.canonical()} (HTTP {resp.status_code})"
        )
    if resp.status_code == 429:
        raise HubResolveError(
            f"tensorhub rate-limited the resolve for {ref.canonical()}; retry later"
        )
    if resp.status_code != 200:
        raise HubResolveError(
            f"tensorhub resolve for {ref.canonical()} returned HTTP {resp.status_code}"
        )

    try:
        body = resp.json()
    except ValueError as e:
        raise HubResolveError(f"tensorhub resolve returned invalid JSON: {e}") from e

    digest = str(body.get("snapshot_digest") or "").strip()
    if digest.lower().startswith("blake3:"):
        digest = digest[7:]
    files: List[WorkerResolvedRepoFile] = []
    for ent in body.get("files") or []:
        if not isinstance(ent, dict):
            continue
        path = str(ent.get("path") or "").strip()
        b3 = str(ent.get("blake3") or "").strip().lower()
        if b3.startswith("blake3:"):
            b3 = b3[7:]
        tagged = str(ent.get("digest") or "").strip().lower()
        u = str(ent.get("url") or "").strip() or None
        chunks = _parse_chunks(ref, path, ent.get("chunks"))
        # A chunked entry has NO whole-file url — its bytes only exist as
        # chunks. Requiring `url` here is what would classify every v2
        # snapshot as unresolvable.
        if not path or (not tagged and not b3) or (not u and not chunks):
            raise HubResolveError(
                f"tensorhub resolve for {ref.canonical()}: manifest entry "
                f"missing path/digest/url ({ent.get('path')!r})"
            )
        files.append(WorkerResolvedRepoFile(
            path=path, size_bytes=int(ent.get("size_bytes") or 0), blake3=b3, url=u,
            digest=tagged, chunks=chunks,
            chunk_size_bytes=int(ent.get("chunk_size_bytes") or 0),
        ))
    if not digest or not files:
        raise HubResolveError(
            f"tensorhub resolve for {ref.canonical()}: empty snapshot manifest"
        )
    siblings: List[WorkerResolvedFlavor] = []
    for row in body.get("sibling_flavors") or []:
        if not isinstance(row, dict):
            continue
        placement = row.get("placement")
        siblings.append(WorkerResolvedFlavor(
            flavor=str(row.get("flavor") or "").strip(),
            size_bytes=int(row.get("size_bytes") or 0),
            placement=placement if isinstance(placement, dict) else None,
        ))
    return WorkerResolvedRepo(
        snapshot_digest=digest, files=files,
        size_bytes=int(body.get("size_bytes") or 0),
        sibling_flavors=siblings,
        model_family=str(body.get("model_family") or "").strip(),
        objective=str(body.get("objective") or "").strip(),
        distilled=bool(body.get("distilled") or False),
    )
