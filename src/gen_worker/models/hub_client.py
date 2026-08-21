from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Mapping, Optional

from ..config import Settings, current_or

_STANDALONE = Settings()
from .refs import TensorhubRef
import requests


@dataclass(frozen=True)
class WorkerResolvedChunk:
    """One CAS object of a chunked file (manifest v2)."""

    sha256: str
    url: str
    length: int


@dataclass(frozen=True)
class WorkerResolvedRepoFile:
    path: str
    size_bytes: int
    url: Optional[str]
    digest: str = ""
    chunks: tuple["WorkerResolvedChunk", ...] = ()

    def cas_ref(self) -> str:
        """The algorithm-tagged digest of this entry."""
        return resolved_entry_digest(
            {"digest": self.digest},
            what=f"resolved file {self.path!r}",
        )


def resolved_entry_digest(
    ent: Mapping[str, Any], *, what: str = "manifest entry"
) -> str:
    """The algorithm-tagged digest of one RAW resolve-manifest entry."""
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
    size_bytes: int = 0
    objective: str = ""
    distilled: bool = False
    distilled_status: str = ""


class HubResolveError(RuntimeError):
    """Base for standalone tensorhub resolve failures."""


class HubRepoNotFoundError(HubResolveError):
    """404: unknown repo/release OR a private repo the caller may not see (the route deliberately never distinguishes these)."""


class HubAuthError(HubResolveError):
    """401/403: the supplied TENSORHUB_TOKEN was rejected."""


def hub_base_url(base_url: Optional[str] = None) -> str:
    return (base_url or current_or(_STANDALONE).tensorhub_url).strip().rstrip("/")


def parse_chunk_list(
    what: str, path: str, raw: Any, urls: Any = None
) -> tuple[WorkerResolvedChunk, ...]:
    """Parse a v2 entry's ordered chunk list; order is the file's byte order. The wire shape (read off the hub's serializer): chunks: [{digest, len}] — canonical manifest bytes, which cannot carry URLs — plus a SEPARATE, INDEX-ALIGNED chunk_urls list that exists only at resolve time (a nested url is accepted when present). A malformed chunk list is a HARD failure, never a silent empty list: empty is indistinguishable from "stored whole", and a URL list of the WRONG LENGTH is equally fatal — index alignment is the only thing tying a URL to its digest. REST-only; the gRPC path reads typed ChunkRef and must not be mirrored here."""
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
        digest = str(c.get("digest") or "").strip().lower()
        digest = digest.removeprefix("sha256:")
        url = str(c.get("url") or "").strip() or (url_list[i] if i < len(url_list) else "")
        length = int(c.get("len") or 0)
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
    """Resolve a Hub ref against ``GET /api/v1/repos/:tenant/:name/resolve``."""

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
    elif ref.release:
        params["release"] = ref.release

    url = f"{base}/api/v1/repos/{ref.owner}/{ref.repo}/resolve"
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=timeout)
    except requests.RequestException as e:
        raise HubResolveError(f"tensorhub resolve failed for {ref.canonical()}: {e}") from e

    if resp.status_code == 404:
        from ..http_origin import is_proxy_outage

        if is_proxy_outage(resp):
            raise HubResolveError(
                f"tensorhub unreachable resolving {ref.canonical()} — a proxy "
                "answered 404 (backend offline, e.g. hub restarting); retry shortly"
            )
        raise HubRepoNotFoundError(
            f"tensorhub repo {ref.canonical()} not found (unknown repo or "
            "release, or a private repo — set TENSORHUB_TOKEN for private pulls)"
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
        if not path or not tagged or (not u and not chunks):
            raise HubResolveError(
                f"tensorhub resolve for {ref.canonical()}: manifest entry "
                f"missing path/digest/url ({ent.get('path')!r})"
            )
        files.append(WorkerResolvedRepoFile(
            path=path, size_bytes=int(ent.get("size_bytes") or 0), url=u,
            digest=tagged, chunks=chunks,
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
