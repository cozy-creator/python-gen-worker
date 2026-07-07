from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, List, Optional

from .refs import TensorhubRef


# In-memory shape of a resolved cozy: ref. PRODUCTION workers never resolve
# these themselves — the orchestrator pre-resolves every cozy ref a job needs
# and ships the manifest + presigned URLs via
# JobExecutionRequest.resolved_repos_by_id over gRPC. STANDALONE clients
# (gen-worker run/serve/prefetch under cozy, #379) resolve the same shape over
# HTTP via ``resolve_repo`` against tensorhub's public resolve route (th#560).


@dataclass(frozen=True)
class WorkerResolvedRepoFile:
    path: str
    size_bytes: int
    blake3: str
    url: Optional[str]
    transfer_grant: Optional[dict[str, Any]] = None


@dataclass(frozen=True)
class WorkerResolvedRepo:
    snapshot_digest: str
    files: List[WorkerResolvedRepoFile]


class HubResolveError(RuntimeError):
    """Base for standalone tensorhub resolve failures."""


class HubRepoNotFoundError(HubResolveError):
    """404: unknown repo/tag/flavor OR a private repo the caller may not see
    (the route deliberately never distinguishes these)."""


class HubAuthError(HubResolveError):
    """401/403: the supplied TENSORHUB_TOKEN was rejected."""


def hub_base_url(base_url: Optional[str] = None) -> str:
    return (base_url or os.getenv("TENSORHUB_URL") or "").strip().rstrip("/")


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
    import requests

    base = hub_base_url(base_url)
    if not base:
        raise HubResolveError(
            "no tensorhub base URL: set TENSORHUB_URL (e.g. https://tensorhub.com)"
        )
    tok = (token or os.getenv("TENSORHUB_TOKEN") or "").strip()
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
        u = str(ent.get("url") or "").strip() or None
        if not path or not b3 or not u:
            raise HubResolveError(
                f"tensorhub resolve for {ref.canonical()}: manifest entry "
                f"missing path/blake3/url ({ent.get('path')!r})"
            )
        files.append(WorkerResolvedRepoFile(
            path=path, size_bytes=int(ent.get("size_bytes") or 0), blake3=b3, url=u,
        ))
    if not digest or not files:
        raise HubResolveError(
            f"tensorhub resolve for {ref.canonical()}: empty snapshot manifest"
        )
    return WorkerResolvedRepo(snapshot_digest=digest, files=files)
