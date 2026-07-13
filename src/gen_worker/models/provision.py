"""The ONE model load+place core, plus the CLI's hub-less resolve (pgw#515).

Production (the executor's setup injection) and the local CLI
(``gen-worker run`` / ``serve``) drive the SAME code for turning a resolved
snapshot into a ready slot value: annotation-typed injection, binding
dtype / storage-dtype honoring, the pre-load cast gate (th#737), the
adaptive fit ladder outcome stamps (gw#491), worker-owned placement, and
compiled-artifact arming. Structural reporting (ServePlan / FnDegraded)
stays with the executor — :class:`SlotLoad` carries the outcomes so the
caller reports them however it reports.

Resolution differs by necessity: the executor's bytes come from
orchestrator-resolved snapshots (``ModelStore.ensure_local``); the CLI has
no orchestrator, so :func:`resolve_local_path` resolves standalone — local
CAS, tensorhub's public resolve route (th#560), direct HF / Civitai /
ModelScope downloads — through the same download layer.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

from ..config import get_settings
from .loading import model_index_components

__all__ = ["model_index_components"]  # re-export: single source in loading.py (gw#521)

logger = logging.getLogger(__name__)

EmitFn = Callable[[Dict[str, Any]], None]


class ModelResolutionError(Exception):
    """A model binding cannot be resolved locally (CLI exit 3)."""


# ---------------------------------------------------------------------------
# Shared load + place + compile (executor and CLI)
# ---------------------------------------------------------------------------


@dataclass
class SlotLoad:
    """Outcome of loading one setup slot.

    ``obj`` is what the slot receives: the local path (``str``/``Path``
    annotations and unknown annotations), or a constructed + placed pipeline
    for a class annotation exposing ``from_pretrained``. The remaining fields
    are non-default only on the pipeline lane; detail fields are non-empty
    exactly when the caller should report that degradation (the decision
    logic lives here, once)."""

    obj: Any
    is_pipeline: bool = False
    ran: str = ""                # compute precision label ("bf16" default)
    # th#737 pre-load gate: the cast directive had no cast surface and was
    # dropped before the load.
    pre_drop_wanted: str = ""
    pre_drop_detail: str = ""
    # gw#491: the loader engaged an emergency fit rung ("fp8" / "nf4").
    rung: str = ""
    rung_detail: str = ""
    # th#737 backstop: the resolved cast was attempted at load and failed on
    # every component.
    cast_fail_wanted: str = ""
    cast_fail_detail: str = ""
    # place_pipeline outcome ({} when placement was skipped).
    placed: Dict[str, Any] = field(default_factory=dict)


def load_slot(
    annotation: Any,
    path: str,
    *,
    binding: Any = None,
    slot: str = "",
    ref: str = "",
    mode: str = "auto",
    components: Optional[Dict[str, Any]] = None,
    device: str = "",
) -> SlotLoad:
    """Typed slot injection: the slot receives exactly what its ``setup``
    annotation says — a ``str``/``Path`` local path, or a constructed
    pipeline for a class annotation exposing ``from_pretrained`` (the
    binding's dtype/storage_dtype honored, the worker's placement/offload
    policy applied). Blocking; callers on an event loop run it via
    ``asyncio.to_thread``.

    ``mode`` is the placement mode (plan-time offload verdicts / learned
    degraded floors — the executor's knowledge; the CLI passes ``auto``).
    ``device="cpu"`` (CLI ``--device cpu``) skips placement entirely.
    ``components`` are preloaded shared modules (gw#479) forwarded to
    ``from_pretrained``.
    """
    if annotation is None or annotation is str:
        return SlotLoad(obj=path)
    if annotation is Path:
        return SlotLoad(obj=Path(path))
    if not (isinstance(annotation, type)
            and callable(getattr(annotation, "from_pretrained", None))):
        return SlotLoad(obj=path)

    from .loading import load_from_pretrained
    from .memory import place_pipeline

    dtype = str(getattr(binding, "dtype", "") or "")
    storage_dtype = str(getattr(binding, "storage_dtype", "") or "")
    out = SlotLoad(obj=None, is_pipeline=True, ran=(dtype or "bf16"))

    # th#737: a cast directive on a denoiser-less diffusers tree is a
    # load-time no-op that would silently serve bf16. Gate it up front when
    # the snapshot's model_index proves there is no cast surface (unknown
    # layouts pass through; the post-load outcome check below is the
    # backstop).
    if storage_dtype in ("fp8", "fp8+te"):
        comps = model_index_components(path)
        if comps and not ({"transformer", "unet"} & comps):
            out.pre_drop_wanted = storage_dtype
            out.pre_drop_detail = (
                f"cast {storage_dtype!r} dropped for slot {slot!r}: pipeline "
                f"has no denoiser/cast surface (components: {sorted(comps)}); "
                "serving at base precision")
            storage_dtype = ""

    pipe = load_from_pretrained(
        annotation, path, dtype=dtype, storage_dtype=storage_dtype,
        components=components or None,
    )
    out.obj = pipe

    rung = str(getattr(pipe, "_cozy_adaptive_rung", "") or "")
    cast_failed = getattr(
        pipe, "_cozy_fp8_storage_requested", False
    ) and not getattr(pipe, "_cozy_fp8_storage_ok", True)
    if rung == "nf4" or (rung == "fp8" and not cast_failed):
        # gw#491: the loader engaged an emergency rung because free VRAM at
        # load was tighter than planning assumed.
        out.rung = rung
        out.rung_detail = (
            f"adaptive fit rung {rung!r} engaged at load for slot {slot!r} "
            f"({type(pipe).__name__}); free VRAM below the stored-precision "
            "footprint")
    elif cast_failed and not rung:
        # th#737 backstop: the RESOLUTION cast was attempted at load and
        # failed on every target — structural report, not a silent bf16
        # fallback. (A failed adaptive fp8 is not a plan deviation: the plan
        # was base precision.)
        out.cast_fail_wanted = storage_dtype or "fp8"
        out.cast_fail_detail = (
            f"fp8 storage failed on every component of slot {slot!r} "
            f"({type(pipe).__name__}); serving at base precision")

    # Worker-owned placement/offload policy: one decider for the whole
    # worker; endpoints never write device/offload code. A CUDA OOM inside
    # is a ladder transition, not a failure.
    if device.strip().lower() != "cpu":
        out.placed = place_pipeline(pipe, mode=mode, ref=ref)
    return out


def enable_compiled(
    pipe: Any, cfg: Any, cache_dir: Optional[Path] = None,
    artifact: Optional[Path] = None,
) -> bool:
    """Arm the best available compiled path for a freshly loaded pipeline:
    a TRT engine artifact swaps the module (fail-soft), anything else goes
    through the torch.compile cache policy (which also covers the no-
    artifact and ALLOW_COLD lanes)."""
    from .. import compile_cache, trt_engine

    if artifact is not None:
        try:
            meta = trt_engine.unpack_metadata(Path(artifact))
        except Exception:
            meta = None
        if meta is not None and str(meta.get("kind") or "") == "trt-engine":
            if trt_engine.enable(pipe, cfg, cache_dir, artifact):
                return True
            artifact = None  # unusable engine: fall through to eager policy
    return compile_cache.enable(pipe, cfg, cache_dir, artifact)


# ---------------------------------------------------------------------------
# Standalone (hub-less) resolution — the CLI's half. The executor's bytes
# come from orchestrator-resolved snapshots via ModelStore.ensure_local.
# ---------------------------------------------------------------------------


def resolve_bindings(
    bindings: Mapping[str, Any],
    *,
    offline: bool,
    emit: EmitFn,
    slots: Optional[Mapping[str, Any]] = None,
    payload: Any = None,
) -> Dict[str, str]:
    """Resolve every binding to a local path / loader-ready string.

    ``slots``/``payload`` (pgw#520): when a binding's slot is Slot-declared
    with a ``selected_by`` field, and this hub-less run has no hub to
    resolve a curated/BYOM pick against, a payload that actually NAMES a
    model (a non-empty ``selected_by`` field value) is a clear usage error
    instead of silently running the slot's default — ``cozy run`` only ever
    runs a Slot's ``default_checkpoint`` ref locally.
    """
    from ..api.binding import ModelRef, wire_ref

    out: Dict[str, str] = {}
    for param_name, binding in bindings.items():
        slot = (slots or {}).get(param_name)
        selected_by = str(getattr(slot, "selected_by", "") or "") if slot is not None else ""
        if selected_by and payload is not None:
            picked = str(getattr(payload, selected_by, "") or "").strip()
            if picked:
                raise ModelResolutionError(
                    f"slot {param_name!r}: payload names model {picked!r} via "
                    f"{selected_by!r}, but no hub is configured — "
                    "hub-less mode (`cozy run` / `gen-worker run`) only runs "
                    "a Slot's default_checkpoint= ref; configure HUB= (or "
                    f"drop the {selected_by!r} field) to run against a hub."
                )
        if not isinstance(binding, ModelRef):
            raise ModelResolutionError(
                f"unknown binding type for param {param_name!r}: "
                f"{type(binding).__name__}"
            )
        out[param_name] = resolve_local_path(
            ref=wire_ref(binding), provider=binding.provider,
            offline=offline, emit=emit,
            allow_patterns=tuple(getattr(binding, "files", ()) or ()),
            civitai_version_id=str(getattr(binding, "version", "") or ""),
        )
    return out


def _hub_ref_map_path(cache_dir: Path, thref: Any) -> Path:
    """CAS-local memory of tag->snapshot resolutions, so a previously-fetched
    tag ref keeps working offline: cas/refs/<owner>/<repo>/<tag>[#flavor]."""
    name = str(thref.tag or "latest")
    if thref.flavor:
        name += "#" + str(thref.flavor)
    safe = "".join(ch if (ch.isalnum() or ch in "._#-") else "_" for ch in name)
    return cache_dir / "refs" / str(thref.owner) / str(thref.repo) / safe


def _remember_hub_ref(cache_dir: Path, thref: Any, digest: str) -> None:
    try:
        p = _hub_ref_map_path(cache_dir, thref)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(digest)
    except OSError:
        pass


def _fetch_tensorhub_snapshot(
    thref: Any, *, cache_dir: Path, emit: EmitFn,
) -> str:
    """Resolve a Hub ref via th#560 and download its snapshot into the CAS.

    One re-resolve retry on a presigned-URL expiry mid-download (the same
    contract the orchestrator honors on ``url_expired``).
    """
    import asyncio

    from .cozy_snapshot import ensure_snapshot_async
    from .errors import UrlExpiredError
    from .hub_client import HubResolveError, resolve_repo

    canonical = thref.canonical()

    def _resolve() -> Any:
        try:
            return resolve_repo(thref)
        except HubResolveError as e:
            raise ModelResolutionError(str(e)) from e

    emit({"kind": "model_fetch.started", "ref": canonical, "provider": "tensorhub"})
    resolved = _resolve()

    # Already materialized under the resolved digest? No download.
    snap_dir = cache_dir / "snapshots" / resolved.snapshot_digest
    if snap_dir.exists():
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
    _remember_hub_ref(cache_dir, thref, resolved.snapshot_digest)
    emit({"kind": "model_fetch.completed", "ref": canonical,
          "provider": "tensorhub", "local_dir": str(snap)})
    return str(snap)


def resolve_local_path(
    *, ref: str, provider: str, offline: bool, emit: EmitFn,
    allow_patterns: Tuple[str, ...] = (),
    civitai_version_id: str = "",
) -> str:
    """Resolve one model ref to a local snapshot dir / loader-ready string.

    Order matches the live worker:
      1. local CAS lookup (digest-pinned snapshot dirs).
      2. HF refs → ``download_hf`` (auto-fetches from HF).
      3. ModelScope refs → ``modelscope.snapshot_download``.
      4. Cozy refs missing from CAS: standalone resolve against tensorhub's
         public resolve route (th#560); ``--offline`` stays CAS-only (exit 3).
      5. Civitai refs → model → latest-version lookup (or the pinned
         version), then ``download_civitai``.
    """
    from .cache_paths import tensorhub_cas_dir
    from .refs import parse_model_ref

    env_cas = get_settings().tensorhub_cas_dir.strip()
    cache_dir = Path(env_cas) if env_cas else Path(tensorhub_cas_dir())

    # Decode the bare ref into typed parts using the explicit provider.
    # No string-prefix sniffing — provider is the source of truth.
    try:
        parsed = parse_model_ref(ref, provider=provider)
    except Exception as e:
        raise ModelResolutionError(
            f"failed to parse model ref {ref!r} (provider={provider!r}): {e}"
        ) from e

    if parsed.provider == "tensorhub" and parsed.tensorhub and parsed.tensorhub.digest:
        # Snapshot dirs are keyed by the bare hex digest (no algo prefix).
        digest = parsed.tensorhub.digest.split(":", 1)[-1]
        snap_dir = cache_dir / "snapshots" / digest
        if snap_dir.exists():
            return str(snap_dir)

    # HF refs: fall through to the shared HF downloader.
    if parsed.provider == "hf" and parsed.hf is not None:
        if offline:
            # Best-effort: check the HF cache (huggingface_hub manages this
            # itself; a cache hit returns a path, miss raises).
            try:
                from ..net import hf
                p = hf().snapshot_download(
                    repo_id=parsed.hf.repo_id,
                    revision=parsed.hf.revision,
                    local_files_only=True,
                    cache_dir=get_settings().hf_home or None,
                    token=get_settings().hf_token or None,
                    allow_patterns=list(allow_patterns) or None,
                )
                return str(p)
            except Exception as e:
                raise ModelResolutionError(
                    f"--offline: huggingface ref {parsed.hf.canonical()} not "
                    f"in local cache ({e}); warm the cache by running without "
                    "--offline first."
                ) from e

        emit({"kind": "model_fetch.started", "ref": parsed.hf.canonical()})
        try:
            from .download import download_hf

            local_dir = download_hf(
                parsed.hf,
                hf_home=get_settings().hf_home or None,
                hf_token=get_settings().hf_token or None,
                allow_patterns=tuple(allow_patterns),
            )
        except Exception as e:
            raise ModelResolutionError(
                f"failed to fetch huggingface ref {parsed.hf.canonical()}: {e}"
            ) from e
        emit({
            "kind": "model_fetch.completed",
            "ref": parsed.hf.canonical(),
            "local_dir": str(local_dir),
        })
        return str(local_dir)

    # ModelScope refs: fetch directly via modelscope.snapshot_download. This is
    # file-oriented (allow_patterns) and has NO diffusers-layout requirement, so
    # it handles ComfyUI/DiffSynth split checkpoints the HF resolver rejects.
    if parsed.provider == "modelscope" and parsed.modelscope is not None:
        try:
            from modelscope import snapshot_download as _ms_snap
        except Exception as e:
            raise ModelResolutionError(
                f"modelscope is required for modelscope refs ({parsed.modelscope.canonical()}): {e}"
            ) from e
        kwargs: Dict[str, Any] = {}
        if parsed.modelscope.revision:
            kwargs["revision"] = parsed.modelscope.revision
        if allow_patterns:
            kwargs["allow_patterns"] = list(allow_patterns)
        if offline:
            kwargs["local_files_only"] = True
        emit({"kind": "model_fetch.started", "ref": parsed.modelscope.canonical(), "provider": "modelscope"})
        try:
            local = _ms_snap(model_id=parsed.modelscope.repo_id, **kwargs)
        except Exception as e:
            raise ModelResolutionError(
                f"failed to fetch modelscope ref {parsed.modelscope.canonical()}: {e}"
            ) from e
        emit({"kind": "model_fetch.completed", "ref": parsed.modelscope.canonical(), "local_dir": str(local)})
        return str(local)

    # Cozy refs that miss the CAS (#379): resolve standalone against
    # tensorhub's public resolve route (th#560) and feed the shared
    # cozy_snapshot downloader. TENSORHUB_URL selects the hub; TENSORHUB_TOKEN
    # (optional) unlocks private repos. Offline stays CAS-only.
    if parsed.provider == "tensorhub" and parsed.tensorhub is not None:
        if offline:
            # Tag refs: a previous online resolve remembered tag->digest.
            ref_map = _hub_ref_map_path(cache_dir, parsed.tensorhub)
            if ref_map.exists():
                snap = cache_dir / "snapshots" / ref_map.read_text().strip()
                if snap.exists():
                    return str(snap)
            raise ModelResolutionError(
                f"--offline: tensorhub ref {parsed.tensorhub.canonical()} not in local "
                f"CAS ({cache_dir}); warm the cache by running without "
                "--offline once (or set TENSORHUB_CAS_DIR to a path with the "
                "snapshot pre-seeded)."
            )
        return _fetch_tensorhub_snapshot(
            parsed.tensorhub, cache_dir=cache_dir, emit=emit,
        )

    # Civitai refs: download the model-version files directly. Auth (for gated
    # creators) comes from CIVITAI_API_KEY; public models need none.
    if parsed.provider == "civitai" and parsed.civitai is not None:
        if offline:
            raise ModelResolutionError(
                f"--offline: civitai ref {ref!r} not available offline (no local "
                "civitai cache); run once online to fetch it."
            )
        from .download import (
            download_civitai,
            fetch_civitai_model,
            parse_civitai_version_id,
        )
        api_key = get_settings().civitai_api_key

        if civitai_version_id:
            # Explicit version pin via Civitai(version="<id>"). The pinned id
            # IS a model-VERSION id, so use it directly — no model lookup.
            try:
                version_id = parse_civitai_version_id(civitai_version_id)
            except Exception as e:
                raise ModelResolutionError(
                    f"bad civitai version pin {civitai_version_id!r} on ref {ref!r}: {e}"
                ) from e
        else:
            # Civitai's ref is a MODEL id by convention; map it to its latest
            # version id. No silent fallback: if the lookup fails or the model
            # has no versions, the ref is wrong (e.g. a bare version id was
            # passed where a model id was expected) — surface it rather than
            # guessing and downloading an unrelated model.
            try:
                model_id = parse_civitai_version_id(parsed.civitai.model_id)
            except Exception as e:
                raise ModelResolutionError(f"bad civitai ref {ref!r}: {e}") from e
            try:
                model = fetch_civitai_model(model_id, api_key=api_key)
            except Exception as e:
                raise ModelResolutionError(
                    f"failed to resolve civitai model {model_id} for ref {ref!r}: {e}; "
                    "Civitai's ref must be a MODEL id (pin a specific version "
                    'with .version("<version_id>")).'
                ) from e
            versions = model.get("modelVersions") or []
            version_id = int(versions[0].get("id") or 0) if versions else 0
            if version_id <= 0:
                raise ModelResolutionError(
                    f"civitai model {model_id} (ref {ref!r}) has no published "
                    'version to download (pin one with .version("<version_id>")).'
                )
        out_dir = cache_dir / "civitai" / str(version_id)
        emit({"kind": "model_fetch.started", "ref": ref, "provider": "civitai"})
        try:
            local = download_civitai(version_id, out_dir, api_key=api_key)
        except Exception as e:
            raise ModelResolutionError(
                f"failed to fetch civitai ref {ref!r} (resolved version {version_id}): {e}"
            ) from e
        emit({"kind": "model_fetch.completed", "ref": ref, "local_dir": str(local)})
        return str(local)

    raise ModelResolutionError(
        f"unsupported model ref: {ref!r} (provider={provider!r})"
    )
