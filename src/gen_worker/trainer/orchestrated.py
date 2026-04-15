from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping
from urllib import request
from urllib.parse import quote, urlparse
from blake3 import blake3

from .uploader import ArtifactUploadError, ArtifactUploader
from gen_worker.models.ref_downloader import ModelRefDownloader


def is_truthy(value: object) -> bool:
    s = str(value or "").strip().lower()
    return s in {"1", "true", "t", "yes", "y"}


class StartupContractError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


class InputMaterializationError(RuntimeError):
    pass


@dataclass
class RuntimeCancelPolicy:
    cancel_file_path: str | None = None
    max_runtime_seconds: int = 0
    started_monotonic_s: float = 0.0
    _reason: str = ""

    def start(self) -> None:
        self.started_monotonic_s = time.monotonic()

    @property
    def reason(self) -> str:
        return self._reason or "canceled"

    def is_canceled(self) -> bool:
        if self.max_runtime_seconds > 0 and self.started_monotonic_s > 0:
            elapsed = time.monotonic() - self.started_monotonic_s
            if elapsed >= float(self.max_runtime_seconds):
                self._reason = "timeout"
                return True

        if self.cancel_file_path:
            p = Path(self.cancel_file_path)
            if p.exists():
                try:
                    raw = p.read_text(encoding="utf-8").strip().lower()
                except Exception:
                    raw = "1"
                if raw in {"", "1", "true", "t", "yes", "y", "cancel", "canceled"}:
                    self._reason = "canceled"
                    return True
        return False


@dataclass(frozen=True)
class UploadEndpoints:
    metrics_url: str = ""
    checkpoint_url: str = ""
    sample_url: str = ""
    terminal_url: str = ""

    def enabled(self) -> bool:
        return any([self.metrics_url, self.checkpoint_url, self.sample_url, self.terminal_url])


class JsonHttpArtifactUploader(ArtifactUploader):
    def __init__(
        self,
        *,
        request_id: str,
        token: str | None,
        endpoints: UploadEndpoints,
        tensorhub_url: str | None = None,
        owner: str = "",
        destination_repo: str = "",
        job_id: str = "",
        execution_kind: str = "training",
    ) -> None:
        self._request_id = request_id
        self._token = (token or "").strip() or None
        self._endpoints = endpoints
        self._tensorhub_url = (tensorhub_url or "").strip().rstrip("/")
        self._owner = str(owner or "").strip()
        self._destination_repo = str(destination_repo or "").strip().strip("/")
        self._job_id = str(job_id or "").strip()
        self._execution_kind = str(execution_kind or "").strip().lower() or "training"
        self._final_uploaded_ref = ""
        self._final_uploaded_sha256 = ""

    def _post_json(self, url: str, payload: Mapping[str, object]) -> dict[str, object]:
        url = (url or "").strip()
        if not url:
            return {"skipped": True}
        body = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        req = request.Request(url, data=body, headers=headers, method="POST")
        try:
            with request.urlopen(req, timeout=30) as resp:  # noqa: S310
                raw = resp.read().decode("utf-8") if resp.length is None or resp.length > 0 else "{}"
                if resp.status < 200 or resp.status >= 300:
                    raise ArtifactUploadError(f"upload endpoint rejected request: status={resp.status}")
                try:
                    data = json.loads(raw) if raw else {}
                except Exception:
                    data = {}
                if isinstance(data, dict):
                    return {str(k): v for (k, v) in data.items()}
                return {"ok": True}
        except ArtifactUploadError:
            raise
        except Exception as exc:
            raise ArtifactUploadError(f"failed to upload artifact payload to {url}") from exc

    @staticmethod
    def _file_size(path: str) -> int:
        try:
            return int(Path(path).stat().st_size)
        except Exception:
            return 0

    def upload_checkpoint(self, *, local_path: str, step: int) -> Mapping[str, Any]:
        uploaded = self._upload_artifact_file(
            local_path=local_path,
            category="checkpoints",
            step=step,
            final=Path(local_path).name == "final.json",
        )
        if uploaded and bool(uploaded.get("final", False)):
            self._final_uploaded_ref = str(uploaded.get("ref") or "").strip()
            self._final_uploaded_sha256 = str(uploaded.get("sha256") or "").strip()
        return self._post_json(
            self._endpoints.checkpoint_url,
            {
                "request_id": self._request_id,
                "kind": "checkpoint",
                "step": int(step),
                "local_path": str(local_path),
                "size_bytes": self._file_size(local_path),
                "final": Path(local_path).name == "final.json",
                "file_ref": str(uploaded.get("ref") or "") if uploaded else "",
                "sha256": str(uploaded.get("sha256") or "") if uploaded else "",
                "blob_digest": str(uploaded.get("blob_digest") or "") if uploaded else "",
                "snapshot_digest": str(uploaded.get("snapshot_digest") or "") if uploaded else "",
                "uploaded_size_bytes": int(uploaded.get("size_bytes") or 0) if uploaded else 0,
            },
        )

    def upload_sample(self, *, local_path: str, step: int) -> Mapping[str, Any]:
        uploaded = self._upload_artifact_file(
            local_path=local_path,
            category="samples",
            step=step,
            final=False,
        )
        return self._post_json(
            self._endpoints.sample_url,
            {
                "request_id": self._request_id,
                "kind": "sample",
                "step": int(step),
                "local_path": str(local_path),
                "size_bytes": self._file_size(local_path),
                "file_ref": str(uploaded.get("ref") or "") if uploaded else "",
                "sha256": str(uploaded.get("sha256") or "") if uploaded else "",
                "uploaded_size_bytes": int(uploaded.get("size_bytes") or 0) if uploaded else 0,
            },
        )

    def upload_metrics(self, *, metrics: Mapping[str, float], step: int) -> Mapping[str, Any]:
        return self._post_json(
            self._endpoints.metrics_url,
            {
                "request_id": self._request_id,
                "kind": "metrics",
                "step": int(step),
                "metrics": {str(k): float(v) for (k, v) in metrics.items()},
            },
        )

    def upload_terminal(self, *, status: str, step: int, final_checkpoint: str | None, error: str = "") -> Mapping[str, Any]:
        final_checkpoint_ref = self._final_uploaded_ref
        final_checkpoint_sha256 = self._final_uploaded_sha256
        if not final_checkpoint_ref and final_checkpoint:
            uploaded = self._upload_artifact_file(
                local_path=str(final_checkpoint),
                category="checkpoints",
                step=int(step),
                final=True,
            )
            if uploaded:
                final_checkpoint_ref = str(uploaded.get("ref") or "").strip()
                final_checkpoint_sha256 = str(uploaded.get("sha256") or "").strip()
                self._final_uploaded_ref = final_checkpoint_ref
                self._final_uploaded_sha256 = final_checkpoint_sha256

        return self._post_json(
            self._endpoints.terminal_url,
            {
                "request_id": self._request_id,
                "kind": "terminal",
                "status": str(status),
                "step": int(step),
                "final_checkpoint": str(final_checkpoint_ref or final_checkpoint or ""),
                "final_checkpoint_ref": str(final_checkpoint_ref),
                "final_checkpoint_sha256": str(final_checkpoint_sha256),
                "error": str(error or ""),
            },
        )

    def _upload_artifact_file(self, *, local_path: str, category: str, step: int, final: bool) -> dict[str, object] | None:
        from gen_worker.presigned_upload import blake3_hash_file, presigned_upload_file

        if not self._tensorhub_url or not self._token or not self._owner:
            return None
        p = Path(local_path)
        if not p.exists() or not p.is_file():
            return None

        safe_name = p.name.replace("/", "_")
        slot = "final" if final else f"step-{int(step):08d}"
        ref = f"v1/{self._owner}/runs/{self._request_id}/{category}/{slot}-{safe_name}"
        repo_job_scope = None
        if category == "checkpoints" and self._execution_kind in {"training", "conversion"} and self._destination_repo and self._job_id:
            if "/" in self._destination_repo:
                repo_owner, repo_name = self._destination_repo.split("/", 1)
                repo_owner = repo_owner.strip()
                repo_name = repo_name.strip()
                if repo_owner and repo_name:
                    repo_job_scope = (repo_owner, repo_name, self._job_id)
        if category == "checkpoints" and self._execution_kind in {"training", "conversion"} and repo_job_scope is None:
            raise ArtifactUploadError(
                "checkpoint upload requires repo-cas job scope (destination_repo and job_id)"
            )

        # Hash the file before uploading.
        blake3_hex = blake3_hash_file(p)
        size_bytes = int(p.stat().st_size)

        headers = {
            "Authorization": f"Bearer {self._token}",
            "X-Cozy-Owner": self._owner,
        }

        if repo_job_scope is None:
            create_payload: dict[str, object] = {
                "ref": ref,
                "request_id": str(self._request_id or ""),
            }
            endpoint_path = "/api/v1/media/uploads"
        else:
            repo_owner, repo_name, job_id = repo_job_scope
            artifact_path = f"{category}/{slot}-{safe_name}"
            create_payload = {
                "path": artifact_path,
                "request_id": str(self._request_id or ""),
            }
            endpoint_path = (
                f"/api/v1/repos/{quote(repo_owner, safe='')}/{quote(repo_name, safe='')}/"
                f"jobs/{quote(job_id, safe='')}/uploads"
            )

        try:
            result = presigned_upload_file(
                file_path=str(p),
                base_url=self._tensorhub_url,
                endpoint_path=endpoint_path,
                headers=headers,
                create_payload=create_payload,
                blake3_hex=blake3_hex,
                size_bytes=size_bytes,
            )
            meta = result.meta
            return {
                "ref": str(meta.get("ref") or meta.get("filename") or ref),
                "sha256": str(meta.get("sha256") or ""),
                "blake3": str(meta.get("blake3") or blake3_hex),
                "size_bytes": int(meta.get("size_bytes") or size_bytes),
                "blob_digest": str(meta.get("blob_digest") or ""),
                "snapshot_digest": str(meta.get("snapshot_digest") or ""),
                "final": bool(final),
            }
        except Exception as exc:
            raise ArtifactUploadError(f"failed to upload artifact file to tensorhub: {p}") from exc


class RuntimeInputDownloader:
    def __init__(self, *, root_dir: str, capability_token: str | None = None) -> None:
        self._root = Path(root_dir)
        self._root.mkdir(parents=True, exist_ok=True)
        self._token = (capability_token or "").strip() or None
        self._model_downloader = ModelRefDownloader(
            cozy_base_url=os.getenv("TENSORHUB_URL"),
            cozy_token=os.getenv("TENSORHUB_TOKEN"),
            hf_home=os.getenv("HF_HOME"),
            hf_token=os.getenv("HF_TOKEN"),
        )

    def _hash_name(self, ref: str) -> str:
        return hashlib.sha256(ref.encode("utf-8")).hexdigest()

    def _is_url(self, ref: str) -> bool:
        scheme = urlparse(ref).scheme.lower()
        return scheme in {"http", "https", "file"}

    def _download_url_to_file(self, ref: str, dst: Path) -> str:
        dst.parent.mkdir(parents=True, exist_ok=True)
        headers: dict[str, str] = {}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        req = request.Request(ref, headers=headers, method="GET")
        try:
            with request.urlopen(req, timeout=60) as resp:  # noqa: S310
                data = resp.read()
                if resp.status < 200 or resp.status >= 300:
                    raise InputMaterializationError(f"download failed: status={resp.status}")
        except InputMaterializationError:
            raise
        except Exception as exc:
            raise InputMaterializationError(f"failed to download input ref: {ref}") from exc
        dst.write_bytes(data)
        return str(dst)

    def _resolve_existing_path(self, ref: str) -> str | None:
        p = Path(ref)
        if p.exists():
            return str(p.resolve())
        return None

    def download_weights(self, ref: str) -> str:
        ref = str(ref or "").strip()
        if not ref:
            raise InputMaterializationError("empty model ref")

        existing = self._resolve_existing_path(ref)
        if existing:
            p = Path(existing)
            return str(p if p.is_dir() else p.parent)

        if self._is_url(ref):
            out = self._root / "models" / self._hash_name(ref) / "weights.bin"
            self._download_url_to_file(ref, out)
            return str(out.parent)

        try:
            return str(Path(self._model_downloader.download(ref, str(self._root / "models"))).resolve())
        except Exception as exc:
            raise InputMaterializationError(f"failed to materialize model ref: {ref}") from exc

    def download_dataset_parquet(self, dataset_ref: str) -> str:
        ref = str(dataset_ref or "").strip()
        if not ref:
            raise InputMaterializationError("empty dataset parquet ref")

        existing = self._resolve_existing_path(ref)
        if existing:
            return existing

        if self._is_url(ref):
            ext = Path(urlparse(ref).path).suffix or ".parquet"
            out = self._root / "datasets" / f"{self._hash_name(ref)}{ext}"
            return self._download_url_to_file(ref, out)

        raise InputMaterializationError(f"unsupported dataset parquet ref: {ref}")

    def download_resume_checkpoint(self, checkpoint_ref: str) -> str:
        ref = str(checkpoint_ref or "").strip()
        if not ref:
            raise InputMaterializationError("empty resume checkpoint ref")

        existing = self._resolve_existing_path(ref)
        if existing:
            return existing

        if self._is_url(ref):
            ext = Path(urlparse(ref).path).suffix or ".json"
            out = self._root / "resume" / f"{self._hash_name(ref)}{ext}"
            return self._download_url_to_file(ref, out)

        raise InputMaterializationError(f"unsupported resume checkpoint ref: {ref}")


__all__ = [
    "ArtifactUploadError",
    "InputMaterializationError",
    "JsonHttpArtifactUploader",
    "RuntimeCancelPolicy",
    "RuntimeInputDownloader",
    "StartupContractError",
    "UploadEndpoints",
    "is_truthy",
]
