from __future__ import annotations

import asyncio
import contextvars
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any, Callable, Dict, Optional, Tuple,
)


from ..measured_posture import normalize_backend
from . import attention_modes
from .cache_paths import tensorhub_cas_dir
from .errors import UrlExpiredError, non_cas_refusal
from .loading import (
    model_index_components,
)
from .refs import parse_model_ref

__all__ = ["model_index_components"]

logger = logging.getLogger(__name__)

EmitFn = Callable[[Dict[str, Any]], None]


class ModelResolutionError(Exception):
    """A model binding cannot be resolved locally (CLI exit 3)."""


@dataclass(frozen=True)
class _AppliedLaneContext:
    applied: list[Any]


_APPLIED_LANE_CTX: "contextvars.ContextVar[Optional[_AppliedLaneContext]]" = (
    contextvars.ContextVar("gen_worker_applied_lane_ctx", default=None)
)


class AppliedLaneScope:
    """Context manager the executor/CLI holds open around one ``setup()`` call so ``report_applied_lane()`` lands on that instance."""

    def __init__(self) -> None:
        self._applied: list[Any] = []
        self._value = _AppliedLaneContext(applied=self._applied)
        self._token: Optional["contextvars.Token[Optional[_AppliedLaneContext]]"] = None

    def __enter__(self) -> "AppliedLaneScope":
        self._token = _APPLIED_LANE_CTX.set(self._value)
        return self

    def __exit__(self, *_exc_info: Any) -> None:
        if self._token is not None:
            _APPLIED_LANE_CTX.reset(self._token)
            self._token = None

    @property
    def applied(self) -> tuple[Any, ...]:
        """Every ``AppliedLane`` reported inside this setup scope, in order."""
        return tuple(self._applied)


def report_applied_lane(
    component: str,
    lane_body: str,
    *,
    modules: int = 0,
    kept_bf16: int = 0,
) -> bool:
    """Report the lane a serve-time recipe just APPLIED to ``component``'s weights."""
    from . import execution_lanes

    body = str(lane_body or "").strip().lower()
    if not execution_lanes.valid_execution_lane_body(body):
        raise ValueError(
            f"report_applied_lane({component!r}, {lane_body!r}): not a known "
            "lane body (known: "
            f"{', '.join(execution_lanes.known_execution_lane_bodies())})")
    ctx = _APPLIED_LANE_CTX.get()
    if ctx is None:
        logger.info(
            "gen_worker.report_applied_lane(): no active setup scope; "
            "%s applied %s is not attributed to an instance", component, body)
        return False
    ctx.applied.append(execution_lanes.AppliedLane(
        component=str(component or "").strip() or "instance",
        body=body,
        modules=max(0, int(modules)),
        kept_bf16=max(0, int(kept_bf16)),
    ))
    return True


@dataclass(frozen=True)
class _AppliedAttentionContext:
    applied: list[Any]


_APPLIED_ATTENTION_CTX: (
    "contextvars.ContextVar[Optional[_AppliedAttentionContext]]"
) = contextvars.ContextVar("gen_worker_applied_attention_ctx", default=None)


class AppliedAttentionScope:
    """Held open by the executor around one ``setup()`` so a report lands on that instance and cannot be forged from a handler or a background thread."""

    def __init__(self) -> None:
        self._applied: list[Any] = []
        self._value = _AppliedAttentionContext(applied=self._applied)
        self._token: Optional[
            "contextvars.Token[Optional[_AppliedAttentionContext]]"] = None

    def __enter__(self) -> "AppliedAttentionScope":
        self._token = _APPLIED_ATTENTION_CTX.set(self._value)
        return self

    def __exit__(self, *_exc_info: Any) -> None:
        if self._token is not None:
            _APPLIED_ATTENTION_CTX.reset(self._token)
            self._token = None

    @property
    def applied(self) -> tuple[Any, ...]:
        return tuple(self._applied)


def report_applied_attention(
    component: str,
    mode: str,
    *,
    k_blocks: int = 0,
    block_size: int = 0,
    density: float = 0.0,
    selector: str = "",
    index_ref: str = "",
) -> bool:
    """Report the attention path that was actually INSTALLED on ``component``."""
    tok = str(mode or "").strip().lower()
    if not attention_modes.valid_attention_mode(tok):
        raise ValueError(
            f"report_applied_attention({component!r}, {mode!r}): not a valid "
            "attention mode (expected 'dense' or 'sparse-k<N>')")
    k = attention_modes.sparse_k_of(tok)
    if k is not None and k_blocks and int(k_blocks) != k:
        raise ValueError(
            f"report_applied_attention({component!r}, {mode!r}): k_blocks="
            f"{k_blocks} contradicts the mode token")
    ctx = _APPLIED_ATTENTION_CTX.get()
    if ctx is None:
        logger.info(
            "gen_worker.report_applied_attention(): no active setup scope; "
            "%s applied %s is not attributed to an instance", component, tok)
        return False
    ctx.applied.append(attention_modes.AppliedAttention(
        component=str(component or "").strip() or "instance",
        mode=tok,
        k_blocks=int(k_blocks or k or 0),
        block_size=max(0, int(block_size)),
        density=max(0.0, float(density)),
        selector=str(selector or "").strip(),
        index_ref=str(index_ref or "").strip(),
    ))
    return True


def report_attention_backend(
    component: str,
    backend: str,
    *,
    wanted: str = "",
) -> bool:
    engaged = normalize_backend(backend)
    asked = normalize_backend(wanted)
    if not engaged:
        raise ValueError(
            f"report_attention_backend({component!r}, {backend!r}): the "
            "engaged backend is required (report what RAN; `wanted` is the "
            "optional half)")
    ctx = _APPLIED_ATTENTION_CTX.get()
    if ctx is None:
        logger.info(
            "gen_worker.report_attention_backend(): no active setup scope; "
            "%s backend %s is not attributed to an instance", component, engaged)
        return False
    ctx.applied.append(attention_modes.AppliedAttention(
        component=str(component or "").strip() or "instance",
        backend=engaged,
        backend_wanted=asked,
    ))
    return True


def _hub_ref_map_path(cache_dir: Path, thref: Any) -> Path:
    name = str(thref.release or "_bare")
    safe = "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in name)
    return cache_dir / "refs" / str(thref.owner) / str(thref.repo) / safe


def _remember_hub_ref(cache_dir: Path, thref: Any, digest: str) -> None:
    try:
        p = _hub_ref_map_path(cache_dir, thref)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(digest)
    except OSError:
        pass


def _fetch_tensorhub_snapshot(
    thref: Any, *, cache_dir: Path, emit: EmitFn, components: Tuple[str, ...] = (),
) -> str:
    from .cozy_snapshot import ensure_snapshot_async, snapshot_dir_key
    from .hub_client import HubResolveError, resolve_repo

    canonical = thref.canonical()

    def _resolve() -> Any:
        try:
            return resolve_repo(thref)
        except HubResolveError as e:
            raise ModelResolutionError(str(e)) from e

    emit({"kind": "model_fetch.started", "ref": canonical, "provider": "tensorhub"})
    resolved = _resolve()

    key = snapshot_dir_key(resolved.snapshot_digest, components)
    snap_dir = cache_dir / "snapshots" / key
    if snap_dir.exists():
        if not components:
            _remember_hub_ref(cache_dir, thref, resolved.snapshot_digest)
        emit({"kind": "model_fetch.completed", "ref": canonical,
              "provider": "tensorhub", "local_dir": str(snap_dir)})
        return str(snap_dir)

    last_at = [0.0]

    def _progress(done: int, total: Optional[int]) -> None:
        now = time.monotonic()
        if now - last_at[0] < 1.0 and (total is None or done < total):
            return
        last_at[0] = now
        emit({"kind": "model_fetch.progress", "ref": canonical,
              "provider": "tensorhub", "done_bytes": int(done),
              "total_bytes": int(total) if total else None})

    async def _download(res: Any) -> Path:
        return await ensure_snapshot_async(
            base_dir=cache_dir, ref=thref, resolved=res, progress=_progress,
            components=components,
        )

    try:
        try:
            snap = asyncio.run(_download(resolved))
        except UrlExpiredError:
            emit({"kind": "model_fetch.reresolve", "ref": canonical,
                  "provider": "tensorhub", "reason": "url_expired"})
            snap = asyncio.run(_download(_resolve()))
    except ModelResolutionError:
        raise
    except Exception as e:
        raise ModelResolutionError(
            f"failed to download tensorhub snapshot for {canonical}: {e}"
        ) from e
    if not components:
        _remember_hub_ref(cache_dir, thref, resolved.snapshot_digest)
    emit({"kind": "model_fetch.completed", "ref": canonical,
          "provider": "tensorhub", "local_dir": str(snap)})
    return str(snap)


def resolve_local_path(
    *, ref: str, provider: str, offline: bool, emit: EmitFn,
    components: Tuple[str, ...] = (),
) -> str:
    """Resolve one model ref to a local tensorfs CAS snapshot dir."""

    cache_dir = Path(tensorhub_cas_dir())

    try:
        parsed = parse_model_ref(ref, provider=provider)
    except Exception as e:
        raise ModelResolutionError(
            f"failed to parse model ref {ref!r} (provider={provider!r}): {e}"
        ) from e

    if parsed.provider == "tensorhub" and parsed.tensorhub and parsed.tensorhub.digest:
        digest = parsed.tensorhub.digest.split(":", 1)[-1]
        snap_dir = cache_dir / "snapshots" / digest
        if snap_dir.exists():
            return str(snap_dir)

    if parsed.provider == "tensorhub" and parsed.tensorhub is not None:
        if offline:
            ref_map = _hub_ref_map_path(cache_dir, parsed.tensorhub)
            if ref_map.exists():
                snap = cache_dir / "snapshots" / ref_map.read_text().strip()
                if snap.exists():
                    return str(snap)
            raise ModelResolutionError(
                f"--offline: tensorhub ref {parsed.tensorhub.canonical()} not in local "
                f"CAS ({cache_dir}); warm the cache by running without "
                "--offline once (or set TENSORHUB_CACHE_DIR to a cache root "
                "with the snapshot pre-seeded)."
            )
        return _fetch_tensorhub_snapshot(
            parsed.tensorhub, cache_dir=cache_dir, emit=emit, components=components,
        )

    if parsed.provider in ("hf", "civitai", "modelscope"):
        raise ModelResolutionError(str(non_cas_refusal(
            ref=str(ref), provider=parsed.provider)))

    raise ModelResolutionError(
        f"unsupported model ref: {ref!r} (provider={provider!r})"
    )
