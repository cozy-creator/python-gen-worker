from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import mimetypes
import os
from pathlib import Path
import time
from typing import Any, Mapping
from urllib import request
from urllib.parse import quote, urlparse

from .uploader import ArtifactUploadError, ArtifactUploader
from ..model_ref_downloader import ModelRefDownloader


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
        run_id: str,
        token: str | None,
        endpoints: UploadEndpoints,
        tensorhub_url: str | None = None,
        owner: str = "",
    ) -> None:
        self._run_id = run_id
        self._token = (token or "").strip() or None
        self._endpoints = endpoints
        self._tensorhub_url = (tensorhub_url or "").strip().rstrip("/")
        self._owner = str(owner or "").strip()
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
                "run_id": self._run_id,
                "kind": "checkpoint",
                "step": int(step),
                "local_path": str(local_path),
                "size_bytes": self._file_size(local_path),
                "final": Path(local_path).name == "final.json",
                "file_ref": str(uploaded.get("ref") or "") if uploaded else "",
                "sha256": str(uploaded.get("sha256") or "") if uploaded else "",
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
                "run_id": self._run_id,
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
                "run_id": self._run_id,
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
                "run_id": self._run_id,
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
        if not self._tensorhub_url or not self._token or not self._owner:
            return None
        p = Path(local_path)
        if not p.exists() or not p.is_file():
            return None

        safe_name = p.name.replace("/", "_")
        slot = "final" if final else f"step-{int(step):08d}"
        ref = f"v1/{self._owner}/runs/{self._run_id}/{category}/{slot}-{safe_name}"
        url = f"{self._tensorhub_url}/api/v1/file/{quote(ref, safe='/')}"
        mime_type = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
        body = p.read_bytes()
        headers = {
            "Content-Type": mime_type,
            "Authorization": f"Bearer {self._token}",
        }
        req = request.Request(url, data=body, headers=headers, method="PUT")
        try:
            with request.urlopen(req, timeout=120) as resp:  # noqa: S310
                raw = resp.read().decode("utf-8") if resp.length is None or resp.length > 0 else "{}"
                if resp.status < 200 or resp.status >= 300:
                    raise ArtifactUploadError(f"tensorhub file upload rejected status={resp.status}")
                parsed: dict[str, object] = {}
                if raw:
                    try:
                        obj = json.loads(raw)
                        if isinstance(obj, dict):
                            parsed = {str(k): v for (k, v) in obj.items()}
                    except Exception:
                        parsed = {}
                return {
                    "ref": str(parsed.get("ref") or ref),
                    "sha256": str(parsed.get("sha256") or ""),
                    "size_bytes": int(parsed.get("size_bytes") or p.stat().st_size),
                    "final": bool(final),
                }
        except ArtifactUploadError:
            raise
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
            allow_tensorhub_api_resolve=is_truthy(os.getenv("WORKER_ALLOW_TENSORHUB_API_RESOLVE")),
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
