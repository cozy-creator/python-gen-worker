from __future__ import annotations

import asyncio
import contextvars
from pathlib import Path
from typing import Any, Mapping, Optional

import random
import time

from .cozy_cas import CozyHubClient, CozySnapshotDownloader
from .cozy_snapshot_v2_downloader import ensure_snapshot_sync
from .downloader import ModelDownloader
from .cozy_hub_v2 import (
    CozyHubError,
    CozyHubPublicModelPendingError,
    CozyHubV2Client,
)
from .hf_downloader import HuggingFaceHubDownloader
from .model_refs import CozyRef, ParsedModelRef, parse_model_ref

# Per-task resolved manifests provided by gen-orchestrator (issue #92).
# Shape: {canonical_model_id: ResolvedCozyModel-like object}
_resolved_cozy_models_by_id: contextvars.ContextVar[Optional[Mapping[str, Any]]] = contextvars.ContextVar(
    "resolved_cozy_models_by_id", default=None
)

# Per-task best-effort model download preferences.
#
# Shape:
#   {canonical_model_ref: {"dtypes": ["bf16","fp16"], "file_type": "safetensors"}}
_cozy_model_download_prefs_by_ref: contextvars.ContextVar[Optional[Mapping[str, Any]]] = contextvars.ContextVar(
    "cozy_model_download_prefs_by_ref", default=None
)


def set_resolved_cozy_models_by_id(mapping: Optional[Mapping[str, Any]]) -> contextvars.Token:
    return _resolved_cozy_models_by_id.set(mapping)


def reset_resolved_cozy_models_by_id(token: contextvars.Token) -> None:
    _resolved_cozy_models_by_id.reset(token)


def set_cozy_model_download_prefs_by_ref(mapping: Optional[Mapping[str, Any]]) -> contextvars.Token:
    return _cozy_model_download_prefs_by_ref.set(mapping)


def reset_cozy_model_download_prefs_by_ref(token: contextvars.Token) -> None:
    _cozy_model_download_prefs_by_ref.reset(token)


def _get_prefs_for_ref(canonical_ref: str) -> Mapping[str, Any]:
    m = _cozy_model_download_prefs_by_ref.get()
    if not isinstance(m, Mapping):
        return {}
    v = m.get(canonical_ref)
    return v if isinstance(v, Mapping) else {}


class ModelRefDownloader(ModelDownloader):
    """Composite downloader for phase-1 model refs.

    Supported schemes:
      - Cozy Hub snapshots (default): owner/repo[:tag] or owner/repo@sha256:<digest>
        optionally prefixed with "cozy:".
      - Hugging Face: hf:owner/repo[@revision]

    Security posture (issue #92):
      - By default, cozy: downloads MUST use orchestrator-provided resolved manifests
        (presigned URLs). The worker should not call Cozy Hub APIs.
      - If WORKER_ALLOW_COZY_HUB_API_RESOLVE=1, the worker may call Cozy Hub APIs as
        a dev-only fallback.

    Returns a local directory path for both schemes.
    """

    DEFAULT_PUBLIC_MODEL_REQUEST_WAIT_TIMEOUT_S = 15 * 60

    def __init__(
        self,
        cozy_base_url: Optional[str] = None,
        cozy_token: Optional[str] = None,
        hf_home: Optional[str] = None,
        hf_token: Optional[str] = None,
        *,
        allow_cozy_hub_api_resolve: bool = False,
    ) -> None:
        self._cozy_base_url = (cozy_base_url or "").strip() or None
        self._cozy_token = (cozy_token or "").strip() or None
        self._allow_cozy_hub_api_resolve = bool(allow_cozy_hub_api_resolve)

        self._hf = HuggingFaceHubDownloader(hf_home=hf_home, hf_token=hf_token)

        self._cozy_v2: Optional[CozyHubV2Client] = None
        if self._cozy_base_url:
            # Token is optional. If set, it enables ingest-if-missing for public HF models.
            self._cozy_v2 = CozyHubV2Client(base_url=self._cozy_base_url, token=self._cozy_token)

        # Legacy snapshot/object downloader kept for backward compatibility with older Cozy Hub
        # endpoints; only enabled when API resolve is explicitly allowed.
        self._cozy_legacy: Optional[CozySnapshotDownloader] = None
        if self._allow_cozy_hub_api_resolve and self._cozy_base_url:
            client = CozyHubClient(self._cozy_base_url, token=self._cozy_token)
            self._cozy_legacy = CozySnapshotDownloader(client)

    async def _download_async(self, parsed: ParsedModelRef, dest_dir: Path) -> Path:
        if parsed.scheme == "hf" and parsed.hf is not None:
            # Prefer Cozy Hub public model request when Cozy Hub is configured.
            # If Cozy Hub cannot serve it (e.g. not mirrored and no auth), fall back to HF directly.
            if self._cozy_v2 is not None:
                try:
                    canonical = parsed.hf.canonical()
                    prefs = _get_prefs_for_ref(canonical)
                    resolved = await self._request_public_model_with_wait(canonical, prefs=prefs)
                    return ensure_snapshot_sync(
                        base_dir=dest_dir,
                        ref=CozyRef(owner="public", repo="public", tag="latest"),
                        base_url=self._cozy_base_url or "",
                        token=None,
                        resolved=resolved,
                    )
                except CozyHubError:
                    pass
            return self._hf.download(parsed.hf).local_dir

        if parsed.scheme == "cozy" and parsed.cozy is not None:
            canonical = parsed.cozy.canonical()
            resolved = _resolved_cozy_models_by_id.get()
            resolved_entry = resolved.get(canonical) if resolved is not None else None

            if resolved_entry is not None:
                return ensure_snapshot_sync(
                    base_dir=dest_dir,
                    ref=parsed.cozy,
                    base_url=self._cozy_base_url or "",
                    token=None,
                    resolved=resolved_entry,
                )

            # Public model request path.
            # This is allowed even when api-resolve is disabled, but requires COZY_HUB_URL.
            if self._cozy_v2 is not None and parsed.cozy.digest is None:
                prefs = _get_prefs_for_ref(canonical)
                resolved = await self._request_public_model_with_wait(canonical, prefs=prefs)
                return ensure_snapshot_sync(
                    base_dir=dest_dir,
                    ref=parsed.cozy,
                    base_url=self._cozy_base_url or "",
                    token=None,
                    resolved=resolved,
                )

            if not self._allow_cozy_hub_api_resolve:
                raise RuntimeError(
                    "cozy model download requires orchestrator-resolved URLs (missing resolved_cozy_models_by_id entry)"
                )
            if not self._cozy_base_url:
                raise RuntimeError("cozy downloads require COZY_HUB_URL")

            # Prefer Cozy Hub v2 resolve_artifact flow.
            try:
                return ensure_snapshot_sync(
                    base_dir=dest_dir,
                    ref=parsed.cozy,
                    base_url=self._cozy_base_url,
                    token=self._cozy_token,
                    resolved=None,
                )
            except Exception:
                # Fall back to legacy object-based downloader if the hub is old.
                if self._cozy_legacy is None:
                    raise
                return await self._cozy_legacy.ensure_snapshot(dest_dir, parsed.cozy)

        raise ValueError("invalid parsed model ref")

    def _dtype_candidates(self, prefs: Mapping[str, Any]) -> list[str]:
        raw = prefs.get("dtypes")
        if isinstance(raw, str):
            dtypes = [raw]
        elif isinstance(raw, list):
            dtypes = [str(x) for x in raw if str(x).strip()]
        else:
            dtypes = []
        dtypes = [d.strip().lower() for d in dtypes if d.strip()]

        # Default preference: bf16, then fp16.
        if not dtypes:
            return ["bf16", "fp16"]

        # If fp16 is acceptable, bf16 is also acceptable (and preferred).
        if "fp16" in dtypes and "bf16" not in dtypes:
            dtypes = ["bf16"] + dtypes

        # Prefer bf16 first when present.
        if "bf16" in dtypes:
            dtypes = ["bf16"] + [d for d in dtypes if d != "bf16"]

        # Dedupe preserving order.
        out: list[str] = []
        for d in dtypes:
            if d not in out:
                out.append(d)
        return out

    def _file_type_candidates(self, prefs: Mapping[str, Any]) -> list[str]:
        ft = str(prefs.get("file_type") or "").strip().lower()
        if ft in ("flashpack", "safetensors"):
            return [ft]
        # Default to safetensors for broad compatibility.
        return ["safetensors"]

    def _file_layout_candidates(self, prefs: Mapping[str, Any]) -> list[str]:
        lo = str(prefs.get("file_layout") or "").strip().lower()
        if lo in ("diffusers",):
            return [lo]
        return ["diffusers"]

    async def _request_public_model_with_wait(self, model_ref: str, *, prefs: Mapping[str, Any]):
        if self._cozy_v2 is None:
            raise RuntimeError("cozy downloads require COZY_HUB_URL")

        deadline = time.monotonic() + self.DEFAULT_PUBLIC_MODEL_REQUEST_WAIT_TIMEOUT_S
        delay = 0.5
        dtypes = self._dtype_candidates(prefs)
        file_types = self._file_type_candidates(prefs)
        file_layouts = self._file_layout_candidates(prefs)
        while True:
            try:
                return await self._cozy_v2.request_public_model(
                    model_ref=model_ref,
                    dtypes=dtypes,
                    file_types=file_types,
                    file_layouts=file_layouts,
                    include_urls=True,
                )
            except CozyHubPublicModelPendingError as e:
                now = time.monotonic()
                if now >= deadline:
                    raise RuntimeError(
                        f"timed out waiting for public model ingest (ingest_job_id={e.ingest_job_id})"
                    ) from e
                # Exponential backoff with a little jitter.
                sleep_s = min(delay, max(0.1, deadline - now))
                sleep_s = max(0.1, sleep_s + random.random() * 0.25)
                await asyncio.sleep(sleep_s)
                delay = min(10.0, delay * 1.5)

    def download(self, model_ref: str, dest_dir: str, filename: Optional[str] = None) -> str:
        parsed = parse_model_ref(model_ref)
        base = Path(dest_dir)
        # We ignore filename; snapshots/refs already define structure.
        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
                # Nested loop scenario; run in a new loop in a thread.
                return _run_in_thread(self._download_async(parsed, base))
        except RuntimeError:
            pass
        return asyncio.run(self._download_async(parsed, base)).as_posix()


def _run_in_thread(coro) -> str:
    out: dict[str, str] = {}
    err: dict[str, BaseException] = {}

    def runner() -> None:
        try:
            out["v"] = asyncio.run(coro).as_posix()
        except BaseException as e:
            err["e"] = e

    import threading

    t = threading.Thread(target=runner, daemon=True)
    t.start()
    t.join()
    if "e" in err:
        raise err["e"]
    return out["v"]
