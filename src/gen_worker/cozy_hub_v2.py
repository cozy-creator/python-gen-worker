from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional

import aiohttp


class CozyHubError(RuntimeError):
    pass


class CozyHubNoCompatibleArtifactError(CozyHubError):
    def __init__(self, message: str, *, debug: Optional[object] = None) -> None:
        super().__init__(message)
        self.debug = debug


@dataclass(frozen=True)
class CozyHubArtifact:
    label: str
    file_layout: str
    file_type: str
    quantization: str


@dataclass(frozen=True)
class CozyHubSnapshotFile:
    path: str
    size_bytes: int
    blake3: str
    url: Optional[str]


@dataclass(frozen=True)
class CozyHubResolveArtifactResult:
    repo_revision_seq: int
    snapshot_digest: str
    artifact: Optional[CozyHubArtifact]
    files: List[CozyHubSnapshotFile]


class CozyHubV2Client:
    """
    Cozy Hub v2 model APIs (resolve_artifact).

    Endpoint:
      - POST /api/v1/repos/<owner>/<repo>/resolve_artifact

    Response (v1):
      - repo_revision_seq: number
      - snapshot_digest: hex
      - artifact: {label, file_layout, file_type, quantization}
      - snapshot_manifest: {version, files:[{path,size_bytes,blake3,url?}]}
    """

    def __init__(self, base_url: str, token: Optional[str] = None, timeout_s: int = 30) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = (token or "").strip() or None
        self.timeout_s = timeout_s

    def _headers(self) -> Dict[str, str]:
        h: Dict[str, str] = {"Content-Type": "application/json"}
        if self.token:
            h["Authorization"] = f"Bearer {self.token}"
        return h

    async def resolve_artifact(
        self,
        *,
        owner: str,
        repo: str,
        tag: str,
        include_urls: bool,
        preferences: Mapping[str, Any],
        capabilities: Mapping[str, Any],
    ) -> CozyHubResolveArtifactResult:
        if not owner or not repo:
            raise ValueError("owner/repo required")
        tag = (tag or "").strip() or "latest"

        url = f"{self.base_url}/api/v1/repos/{owner}/{repo}/resolve_artifact"
        payload = {
            "tag": tag,
            "include_urls": bool(include_urls),
            "preferences": dict(preferences),
            "capabilities": dict(capabilities),
        }

        timeout = aiohttp.ClientTimeout(total=self.timeout_s)
        async with aiohttp.ClientSession(timeout=timeout, headers=self._headers()) as session:
            async with session.post(url, json=payload) as resp:
                if resp.status == 409:
                    try:
                        data = await resp.json()
                    except Exception:
                        data = {}
                    raise CozyHubNoCompatibleArtifactError(
                        "no compatible artifact for worker",
                        debug=data.get("debug") if isinstance(data, dict) else None,
                    )
                resp.raise_for_status()
                data = await resp.json()
                if not isinstance(data, dict):
                    raise ValueError("unexpected response shape")

        return _parse_resolve_artifact_response(data, include_urls=include_urls)

    async def get_snapshot_manifest(self, *, owner: str, repo: str, digest: str) -> List[CozyHubSnapshotFile]:
        """
        Fetch a snapshot manifest by digest (already pinned).

        Endpoint:
          - GET /api/v1/repos/<owner>/<repo>/snapshots/<digest>/manifest
        """
        if not owner or not repo or not digest:
            raise ValueError("owner/repo/digest required")
        url = f"{self.base_url}/api/v1/repos/{owner}/{repo}/snapshots/{digest}/manifest"

        timeout = aiohttp.ClientTimeout(total=self.timeout_s)
        async with aiohttp.ClientSession(timeout=timeout, headers=self._headers()) as session:
            async with session.get(url) as resp:
                resp.raise_for_status()
                data = await resp.json()
                if not isinstance(data, dict):
                    raise ValueError("unexpected response shape")

        manifest = data.get("files")
        if not isinstance(manifest, list):
            manifest = data.get("root_files")
        if not isinstance(manifest, list):
            raise ValueError("missing files list")
        out: List[CozyHubSnapshotFile] = []
        for ent in manifest:
            if not isinstance(ent, dict):
                continue
            path = str(ent.get("path") or "").strip()
            if not path:
                continue
            out.append(
                CozyHubSnapshotFile(
                    path=path,
                    size_bytes=int(ent.get("size_bytes") or 0),
                    blake3=str(ent.get("blake3") or "").strip().lower(),
                    url=str(ent.get("url") or "").strip() or None,
                )
            )
        if not out:
            raise ValueError("empty files list")
        return out


def _parse_resolve_artifact_response(data: Mapping[str, Any], *, include_urls: bool) -> CozyHubResolveArtifactResult:
    repo_revision_seq = int(data.get("repo_revision_seq") or 0)
    snapshot_digest = str(data.get("snapshot_digest") or "").strip()
    art = data.get("artifact")
    if not isinstance(art, dict):
        raise ValueError("missing artifact")
    artifact = CozyHubArtifact(
        label=str(art.get("label") or "").strip(),
        file_layout=str(art.get("file_layout") or "").strip(),
        file_type=str(art.get("file_type") or "").strip(),
        quantization=str(art.get("quantization") or "").strip(),
    )
    if repo_revision_seq <= 0 or not snapshot_digest:
        raise ValueError("missing snapshot_digest/repo_revision_seq")
    if not artifact.label:
        raise ValueError("missing artifact.label")

    manifest = data.get("snapshot_manifest")
    if not isinstance(manifest, dict):
        raise ValueError("missing snapshot_manifest")
    files_raw = manifest.get("files")
    if not isinstance(files_raw, list):
        raise ValueError("missing snapshot_manifest.files")

    files: List[CozyHubSnapshotFile] = []
    for ent in files_raw:
        if not isinstance(ent, dict):
            continue
        path = str(ent.get("path") or "").strip()
        if not path:
            continue
        size_bytes = int(ent.get("size_bytes") or 0)
        blake3_hex = str(ent.get("blake3") or "").strip().lower()
        url = str(ent.get("url") or "").strip() if include_urls else ""
        files.append(
            CozyHubSnapshotFile(
                path=path,
                size_bytes=size_bytes,
                blake3=blake3_hex,
                url=url or None,
            )
        )
    if not files:
        raise ValueError("empty snapshot file list")

    return CozyHubResolveArtifactResult(
        repo_revision_seq=repo_revision_seq,
        snapshot_digest=snapshot_digest,
        artifact=artifact,
        files=files,
    )
