from __future__ import annotations

import asyncio
import contextvars
from pathlib import Path
from typing import Any, Coroutine, Mapping, Optional

from .cozy_snapshot_v2 import ensure_snapshot_async
from .downloader import ModelDownloader
from .hf_downloader import HuggingFaceHubDownloader
from .refs import ParsedModelRef, parse_model_ref

# Per-request resolved manifests provided by gen-orchestrator (issue #92).
# Shape: {canonical_model_id: ResolvedRepo-like object}
_resolved_repos_by_id: contextvars.ContextVar[Optional[Mapping[str, Any]]] = contextvars.ContextVar(
    "resolved_repos_by_id", default=None
)


def set_resolved_repos_by_id(mapping: Optional[Mapping[str, Any]]) -> contextvars.Token:
    return _resolved_repos_by_id.set(mapping)


def reset_resolved_repos_by_id(token: contextvars.Token) -> None:
    _resolved_repos_by_id.reset(token)


class ModelRefDownloader(ModelDownloader):
    """Composite downloader for phase-1 model refs.

    Supported schemes:
      - Cozy Hub snapshots: owner/repo[:tag] or owner/repo@sha256:<digest>,
        optionally prefixed with "cozy:". Requires orchestrator-provided presigned URLs.
      - Hugging Face: hf:owner/repo[@revision]

    Returns a local directory path for both schemes.
    """

    def __init__(
        self,
        hf_home: Optional[str] = None,
        hf_token: Optional[str] = None,
    ) -> None:
        self._hf = HuggingFaceHubDownloader(hf_home=hf_home, hf_token=hf_token)

    async def _download_async(self, parsed: ParsedModelRef, dest_dir: Path) -> Path:
        if parsed.scheme == "hf" and parsed.hf is not None:
            # Workers download HuggingFace refs directly from HF. Any
            # Cozy-Hub mirroring of HF repos is orchestrator-side
            # (pre-resolved into resolved_repos_by_id as a cozy: ref).
            return self._hf.download(parsed.hf).local_dir

        if parsed.scheme == "cozy" and parsed.cozy is not None:
            canonical = parsed.cozy.canonical()
            resolved_mapping = _resolved_repos_by_id.get()
            resolved_entry = resolved_mapping.get(canonical) if resolved_mapping is not None else None

            if resolved_entry is None:
                # Workers never resolve directly against tensorhub. The
                # orchestrator pre-resolves every cozy: ref a job needs and
                # ships the manifest + presigned URLs via
                # JobExecutionRequest.resolved_repos_by_id. Missing entry
                # here means the orchestrator didn't pre-resolve this ref
                # — that's an orchestrator-side bug, not a worker fallback.
                raise RuntimeError(
                    f"cozy ref {canonical!r} not in resolved_repos_by_id "
                    "— orchestrator must pre-resolve before dispatching the job"
                )

            return await ensure_snapshot_async(
                base_dir=dest_dir,
                ref=parsed.cozy,
                resolved=resolved_entry,
            )

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


def _run_in_thread(coro: Coroutine[Any, Any, Path]) -> str:
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
