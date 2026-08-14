from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Mapping, Optional

from ..config import Settings, current_or

_STANDALONE = Settings()
from .refs import TensorhubRef
import requests


# In-memory shape of a resolved cozy: ref. PRODUCTION workers never resolve
# these themselves — the orchestrator pre-resolves every cozy ref a job needs
# and ships the manifest + presigned URLs via
# JobExecutionRequest.resolved_repos_by_id over gRPC. STANDALONE clients
# (gen-worker run/serve/prefetch under cozy, #379) resolve the same shape over
# HTTP via ``resolve_repo`` against tensorhub's public resolve route.


@dataclass(frozen=True)
class WorkerResolvedChunk:
    """One CAS object of a chunked file (manifest v2).

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
    url: Optional[str]
    #: Algorithm-tagged whole-file digest ("sha256:<hex>"). Every resolved
    #: entry carries one; an entry that does not is refused, not skipped.
    digest: str = ""
    #: Present only when the file is stored as chunks (size > chunk_size).
    chunks: tuple["WorkerResolvedChunk", ...] = ()
    chunk_size_bytes: int = 0

    def cas_ref(self) -> str:
        """The algorithm-tagged digest of this entry.

        Raises rather than returning a bare or empty string, because every
        caller of this is an integrity check and an unreadable digest must
        fail closed, never degrade into "no check".
        """
        return resolved_entry_digest(
            {"digest": self.digest},
            what=f"resolved file {self.path!r}",
        )


def resolved_entry_digest(
    ent: Mapping[str, Any], *, what: str = "manifest entry"
) -> str:
    """The algorithm-tagged digest of one RAW resolve-manifest entry.

    Callers that never build a ``WorkerResolvedRepoFile`` — anything reading
    ``/resolve`` JSON directly, e.g. AOT cell discovery — must go through here
    rather than reaching for a field name. Raises on an absent or untagged
    digest: an integrity check with no digest is a REFUSAL, never a skip.

    the legacy ``blake3`` mirror is no longer read. Before the
    repoint an entry could carry bare blake3 hex and nothing else; after it,
    an entry with no ``digest`` is a stale pointer, and the only safe answer
    is to refuse it. Reinstating the mirror arm here re-opens the vacuous
    guard this whole program exists to close.
    """
    d = str(ent.get("digest") or "").strip().lower()
    if not d:
        raise ValueError(f"{what} carries no digest")
    if ":" not in d:
        raise ValueError(
            f"{what}: digest {d[:16]}… is untagged; "
            "every entry must carry an algorithm prefix"
        )
    return d


@dataclass(frozen=True)
class WorkerResolvedRepo:
    snapshot_digest: str
    files: List[WorkerResolvedRepoFile]
    # Total checkpoint size. There is deliberately no `sibling_flavors`: the
    # hub emits `tag_members` (checkpoint rows) and no flavor row exists to
    # parse. Selection within a tag group is contract compatibility (§1.33).
    size_bytes: int = 0
    # The resolved checkpoint's architecture family ("sdxl-pony",
    # ...) — drives the local family lane policy. "" on hubs not sending it.
    # resolved objective and distillation value plus the
    # evidence status that distinguishes explicit false from unknown. Empty
    # status is the backward-compatible shape from an older hub.
    objective: str = ""
    distilled: bool = False
    distilled_status: str = ""


class HubResolveError(RuntimeError):
    """Base for standalone tensorhub resolve failures."""


class HubRepoNotFoundError(HubResolveError):
    """404: unknown repo/tag OR a private repo the caller may not see
    (the route deliberately never distinguishes these)."""


class HubAuthError(HubResolveError):
    """401/403: the supplied TENSORHUB_TOKEN was rejected."""


def hub_base_url(base_url: Optional[str] = None) -> str:
    return (base_url or current_or(_STANDALONE).tensorhub_url).strip().rstrip("/")


def parse_chunk_list(
    what: str, path: str, raw: Any, urls: Any = None
) -> tuple[WorkerResolvedChunk, ...]:
    """Parse a v2 entry's ordered chunk list. Order is the file's byte order.

    THE WIRE SHAPE, read off the hub's serializer rather than assumed:
    `catalog.SnapshotManifestFile` carries `chunks: [{digest, len}]` — the
    canonical manifest bytes, which cannot contain URLs because they are part of
    the content-addressed identity — plus a SEPARATE, INDEX-ALIGNED
    `chunk_urls: [...]` that exists only at resolve time. The URL is therefore
    NOT nested inside each chunk on the REST path. (The gRPC path differs: its
    `ChunkRef` carries `sha256`/`url`/`len` together, so a nested `url` is
    accepted too.)

    Getting this wrong is not cosmetic: requiring a nested `url` made every real
    chunked checkpoint unparseable, which is how a v2 publish looks fine on the
    hub and is unreachable to every standalone client.

    A malformed chunk list is a HARD failure, never a silent empty list: an
    empty list is indistinguishable from "stored whole", and reading a chunked
    file as whole is how a 40 GiB shard becomes a 0-byte one. A URL list of the
    WRONG LENGTH is equally fatal — index alignment is the only thing tying a
    URL to its digest, so a short list must never mean "fetch fewer chunks".
    """
    if not raw:
        return ()
    if not isinstance(raw, list):
        raise HubResolveError(
            f"{what}: {path!r} chunks is not a list"
        )
    url_list: list[str] = []
    if urls is not None:
        if not isinstance(urls, list):
            raise HubResolveError(
                f"{what}: {path!r} chunk_urls is not a list"
            )
        url_list = [str(u or "").strip() for u in urls]
        if url_list and len(url_list) != len(raw):
            raise HubResolveError(
                f"{what}: {path!r} has {len(raw)} "
                f"chunks but {len(url_list)} chunk_urls — index alignment is the "
                "only thing binding a URL to its digest"
            )
    out: list[WorkerResolvedChunk] = []
    for i, c in enumerate(raw):
        if not isinstance(c, dict):
            raise HubResolveError(
                f"{what}: {path!r} chunk[{i}] is not an object"
            )
        digest = str(c.get("digest") or c.get("sha256") or "").strip().lower()
        digest = digest.removeprefix("sha256:")
        url = str(c.get("url") or "").strip() or (url_list[i] if i < len(url_list) else "")
        length = int(c.get("len") or c.get("length") or 0)
        if len(digest) != 64 or not url or length <= 0:
            raise HubResolveError(
                f"{what}: {path!r} chunk[{i}] "
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
    tok = (token or current_or(_STANDALONE).tensorhub_token).strip()
    headers = {"Authorization": f"Bearer {tok}"} if tok else {}
    params: dict[str, str] = {}
    if ref.digest:
        params["digest"] = ref.digest
    elif ref.tag:
        params["tag"] = ref.tag

    url = f"{base}/api/v1/repos/{ref.owner}/{ref.repo}/resolve"
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=timeout)
    except requests.RequestException as e:
        raise HubResolveError(f"tensorhub resolve failed for {ref.canonical()}: {e}") from e

    if resp.status_code == 404:
        # Only the HUB's own 404 means "no such repo". A proxy
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
            f"tensorhub repo {ref.canonical()} not found (unknown repo or "
            "tag, or a private repo — set TENSORHUB_TOKEN for private pulls)"
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
    files: List[WorkerResolvedRepoFile] = []
    for ent in body.get("files") or []:
        if not isinstance(ent, dict):
            continue
        path = str(ent.get("path") or "").strip()
        tagged = str(ent.get("digest") or "").strip().lower()
        u = str(ent.get("url") or "").strip() or None
        chunks = parse_chunk_list(
            f"tensorhub resolve for {ref.canonical()}", path,
            ent.get("chunks"), ent.get("chunk_urls"))
        # A chunked entry has NO whole-file url — its bytes only exist as
        # chunks. Requiring `url` here is what would classify every v2
        # snapshot as unresolvable.
        if not path or not tagged or (not u and not chunks):
            raise HubResolveError(
                f"tensorhub resolve for {ref.canonical()}: manifest entry "
                f"missing path/digest/url ({ent.get('path')!r})"
            )
        files.append(WorkerResolvedRepoFile(
            path=path, size_bytes=int(ent.get("size_bytes") or 0), url=u,
            digest=tagged, chunks=chunks,
            chunk_size_bytes=int(ent.get("chunk_size_bytes") or 0),
        ))
    if not digest or not files:
        raise HubResolveError(
            f"tensorhub resolve for {ref.canonical()}: empty snapshot manifest"
        )
    distilled_status = str(body.get("distilled_status") or "").strip()
    if distilled_status not in ("", "classified", "unclassified", "inconclusive"):
        raise HubResolveError(
            f"tensorhub resolve for {ref.canonical()}: unknown "
            f"distilled_status {distilled_status!r}"
        )
    return WorkerResolvedRepo(
        snapshot_digest=digest, files=files,
        size_bytes=int(body.get("size_bytes") or 0),
        objective=str(body.get("objective") or "").strip(),
        distilled=bool(body.get("distilled") or False),
        distilled_status=distilled_status,
    )
