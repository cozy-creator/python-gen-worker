from __future__ import annotations

import asyncio
import contextvars
from pathlib import Path
from typing import Any, Mapping, Optional

from .cozy_cas import CozyHubClient, CozySnapshotDownloader
from .cozy_snapshot_v2_downloader import ensure_snapshot_sync
from .downloader import ModelDownloader
from .hf_downloader import HuggingFaceHubDownloader
from .model_refs import ParsedModelRef, parse_model_ref

# Per-task resolved manifests provided by gen-orchestrator (issue #92).
# Shape: {canonical_model_id: ResolvedCozyModel-like object}
_resolved_cozy_models_by_id: contextvars.ContextVar[Optional[Mapping[str, Any]]] = contextvars.ContextVar(
    "resolved_cozy_models_by_id", default=None
)


def set_resolved_cozy_models_by_id(mapping: Optional[Mapping[str, Any]]) -> contextvars.Token:
    return _resolved_cozy_models_by_id.set(mapping)


def reset_resolved_cozy_models_by_id(token: contextvars.Token) -> None:
    _resolved_cozy_models_by_id.reset(token)


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

        # Legacy snapshot/object downloader kept for backward compatibility with older Cozy Hub
        # endpoints; only enabled when API resolve is explicitly allowed.
        self._cozy_legacy: Optional[CozySnapshotDownloader] = None
        if self._allow_cozy_hub_api_resolve and self._cozy_base_url:
            client = CozyHubClient(self._cozy_base_url, token=self._cozy_token)
            self._cozy_legacy = CozySnapshotDownloader(client)

    async def _download_async(self, parsed: ParsedModelRef, dest_dir: Path) -> Path:
        if parsed.scheme == "hf" and parsed.hf is not None:
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
