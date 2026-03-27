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


class CozyHubPublicModelPendingError(CozyHubError):
    def __init__(self, ingest_job_id: str) -> None:
        ingest_job_id = (ingest_job_id or "").strip() or "unknown"
        super().__init__(f"public model ingest pending (ingest_job_id={ingest_job_id})")
        self.ingest_job_id = ingest_job_id


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
    Cozy Hub v2 model APIs (resolve).

    Endpoint:
      - POST /api/v1/repos/<owner>/<repo>/resolve

    Response compatibility:
      - Current shape: {version_id, repo_revision_seq, variant, snapshot_manifest:{entries:[...]}}
      - Legacy shape:  {snapshot_digest, repo_revision_seq, artifact, snapshot_manifest:{files:[...]}}
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
        digest: Optional[str] = None,
        include_urls: bool,
        preferences: Mapping[str, Any],
        capabilities: Mapping[str, Any],
    ) -> CozyHubResolveArtifactResult:
        if not owner or not repo:
            raise ValueError("owner/repo required")
        tag = (tag or "").strip() or "latest"
        digest = (digest or "").strip() or None

        url = f"{self.base_url}/api/v1/repos/{owner}/{repo}/resolve"
        payload = {
            "tag": tag,
            "include_urls": bool(include_urls),
            "preferences": dict(preferences),
            "capabilities": dict(capabilities),
        }
        if digest:
            payload["digest"] = digest

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
        Fetch a snapshot manifest by digest (already pinned) via resolve.
        """
        if not owner or not repo or not digest:
            raise ValueError("owner/repo/digest required")
        res = await self.resolve_artifact(
            owner=owner,
            repo=repo,
            tag="latest",
            digest=digest,
            include_urls=True,
            preferences={},
            capabilities={},
        )
        return res.files

    async def request_public_model(
        self,
        *,
        model_ref: str,
        dtypes: Optional[List[str]] = None,
        file_types: Optional[List[str]] = None,
        file_layouts: Optional[List[str]] = None,
        include_urls: bool = True,
    ) -> CozyHubResolveArtifactResult:
        """
        Request a (public) model via Cozy Hub.

        Endpoint:
          - POST /api/v1/public/models/request

        Responses:
          - 200: {owner,repo,tag,variant_label,snapshot_digest,snapshot_manifest:{files:[...]}}
          - 202: {ingest_job_id,...} (pending)
          - 403: forbidden (model not mirrored and invoker not authenticated)
          - 409: no compatible variant
        """
        model_ref = (model_ref or "").strip()
        if not model_ref:
            raise ValueError("model_ref required")

        url = f"{self.base_url}/api/v1/public/models/request"
        payload = {
            "model_ref": model_ref,
            "constraints": {
                "dtypes": [str(x) for x in (dtypes or []) if str(x).strip()],
                "file_types": [str(x) for x in (file_types or []) if str(x).strip()],
                "file_layouts": [str(x) for x in (file_layouts or []) if str(x).strip()],
            },
            "include_urls": bool(include_urls),
        }

        timeout = aiohttp.ClientTimeout(total=self.timeout_s)
        async with aiohttp.ClientSession(timeout=timeout, headers=self._headers()) as session:
            async with session.post(url, json=payload) as resp:
                if resp.status == 202:
                    try:
                        data = await resp.json()
                    except Exception:
                        data = {}
                    ingest_job_id = str(data.get("ingest_job_id") or "").strip() if isinstance(data, dict) else ""
                    raise CozyHubPublicModelPendingError(ingest_job_id or "")
                if resp.status == 409:
                    try:
                        data = await resp.json()
                    except Exception:
                        data = {}
                    raise CozyHubNoCompatibleArtifactError(
                        "no compatible artifact for worker",
                        debug=data if isinstance(data, dict) else None,
                    )
                if resp.status == 403:
                    try:
                        data = await resp.json()
                    except Exception:
                        data = {}
                    msg = "forbidden"
                    if isinstance(data, dict):
                        msg = str(data.get("message") or data.get("error") or msg)
                    raise CozyHubError(msg)
                resp.raise_for_status()
                data = await resp.json()
                if not isinstance(data, dict):
                    raise ValueError("unexpected response shape")

        return _parse_request_public_model_response(data, include_urls=include_urls)


def _parse_resolve_artifact_response(data: Mapping[str, Any], *, include_urls: bool) -> CozyHubResolveArtifactResult:
    repo_revision_seq = _coerce_int(data.get("repo_revision_seq")) or _coerce_int(data.get("revision")) or _coerce_int(data.get("version")) or 0
    snapshot_digest = str(data.get("snapshot_digest") or data.get("version_id") or "").strip()
    if not snapshot_digest:
        raise ValueError("missing snapshot_digest/version_id")

    artifact = _parse_artifact_meta(data)
    files = _parse_snapshot_files(data, include_urls=include_urls)
    if not files:
        raise ValueError("empty snapshot file list")

    return CozyHubResolveArtifactResult(repo_revision_seq=repo_revision_seq, snapshot_digest=snapshot_digest, artifact=artifact, files=files)


def _parse_request_public_model_response(data: Mapping[str, Any], *, include_urls: bool) -> CozyHubResolveArtifactResult:
    repo_revision_seq = _coerce_int(data.get("repo_revision_seq")) or _coerce_int(data.get("revision")) or _coerce_int(data.get("version")) or 0
    snapshot_digest = str(data.get("snapshot_digest") or data.get("version_id") or "").strip()
    if not snapshot_digest:
        raise ValueError("missing snapshot_digest/version_id")

    variant_label = str(data.get("variant_label") or "").strip()
    artifact = None
    if variant_label:
        artifact = CozyHubArtifact(label=variant_label, file_layout="", file_type="", quantization="")

    files = _parse_snapshot_files(data, include_urls=include_urls)
    if not files:
        raise ValueError("empty snapshot file list")

    return CozyHubResolveArtifactResult(repo_revision_seq=repo_revision_seq, snapshot_digest=snapshot_digest, artifact=artifact, files=files)


def _coerce_int(v: Any) -> Optional[int]:
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v
    s = str(v or "").strip()
    if s == "":
        return None
    try:
        return int(s)
    except Exception:
        return None


def _extract_blake3_hex(ent: Mapping[str, Any]) -> str:
    b3 = str(ent.get("blake3") or "").strip().lower()
    if b3:
        return b3
    dig = str(ent.get("digest") or "").strip().lower()
    if dig.startswith("blake3:"):
        return dig.split(":", 1)[1].strip().lower()
    if len(dig) == 64 and all(ch in "0123456789abcdef" for ch in dig):
        return dig
    return ""


def _parse_snapshot_files(data: Mapping[str, Any], *, include_urls: bool) -> List[CozyHubSnapshotFile]:
    manifest = data.get("snapshot_manifest")
    if not isinstance(manifest, dict):
        raise ValueError("missing snapshot_manifest")
    files_raw = manifest.get("files")
    if not isinstance(files_raw, list):
        files_raw = manifest.get("entries")
    if not isinstance(files_raw, list):
        raise ValueError("missing snapshot_manifest.files|entries")

    files: List[CozyHubSnapshotFile] = []
    for ent in files_raw:
        if not isinstance(ent, dict):
            continue
        path = str(ent.get("path") or "").strip()
        if not path:
            continue
        url = str(ent.get("url") or "").strip() if include_urls else ""
        files.append(
            CozyHubSnapshotFile(
                path=path,
                size_bytes=int(ent.get("size_bytes") or 0),
                blake3=_extract_blake3_hex(ent),
                url=url or None,
            )
        )
    return files


def _parse_artifact_meta(data: Mapping[str, Any]) -> Optional[CozyHubArtifact]:
    art = data.get("artifact")
    if isinstance(art, Mapping):
        label = str(art.get("label") or "").strip()
        file_layout = str(art.get("file_layout") or "").strip()
        file_type = str(art.get("file_type") or "").strip()
        quantization = str(art.get("quantization") or "").strip()
        if label or file_layout or file_type or quantization:
            return CozyHubArtifact(label=label, file_layout=file_layout, file_type=file_type, quantization=quantization)

    variant = data.get("variant")
    if isinstance(variant, Mapping):
        label = str(variant.get("label") or "").strip()
        file_layout = str(variant.get("file_layout") or "").strip()
        file_type = str(variant.get("file_type") or "").strip()
        quantization = str(variant.get("quantization") or variant.get("dtype") or "").strip()
        if label or file_layout or file_type or quantization:
            return CozyHubArtifact(label=label, file_layout=file_layout, file_type=file_type, quantization=quantization)
    return None
