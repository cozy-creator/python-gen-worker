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

__all__ = ["model_index_components"]  # re-export: single source in loading.py (gw#521)

logger = logging.getLogger(__name__)

EmitFn = Callable[[Dict[str, Any]], None]


class ModelResolutionError(Exception):
    """A model binding cannot be resolved locally (CLI exit 3)."""


# ---------------------------------------------------------------------------
# pgw#1104: the APPLIED-LANE report. `metrics.lane` used to be a pure function
# of the binding, so a recipe that quantized in setup() served fp8 under a
# bf16 label — and the lane id is a KEY (th#935 verdicts, compiled graphs,
# pricing, the executed-lane proof). A static `handles=`-style declaration
# cannot fix it: the recipe is runtime-gated (sm89 for w8a8, the compile
# preflight), so a declaration would over-claim on the card that skips it.
# Only the code that converted the weights can report provably, so it does —
# through the same contextvar scope `arm_compile` uses, so the report is
# attributed to exactly the setup() that made it and cannot be forged later.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _AppliedLaneContext:
    applied: list[Any]  # list[execution_lanes.AppliedLane]; owned by the scope


_APPLIED_LANE_CTX: "contextvars.ContextVar[Optional[_AppliedLaneContext]]" = (
    contextvars.ContextVar("gen_worker_applied_lane_ctx", default=None)
)


class AppliedLaneScope:
    """Context manager the executor/CLI holds open around one ``setup()`` call
    so ``report_applied_lane()`` lands on that instance. Re-entrant-safe."""

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
    """Report the lane a serve-time recipe just APPLIED to ``component``'s
    weights. Call it from ``setup()`` immediately after the conversion
    returns — the way ``arm_compile()`` is called after placement.

    ``lane_body`` is one of ``known_execution_lane_bodies()`` (the th#1050
    vocabulary, e.g. ``"fp8-w8a8-dynamic"``); an unknown token raises
    ``ValueError`` — the lane vocabulary is shared with the hub and is never
    extended from an endpoint. The execution axis is NOT the author's: the
    worker composes ``+compiled``/``+eager`` from live compile state.

    Returns whether the report was recorded. Outside a setup scope (hub-less
    ``cozy run``, a unit rig) it logs once and returns False — never raises,
    so every endpoint can call it unconditionally."""
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


# ---------------------------------------------------------------------------
# The attention axis (pgw#1043 §PRODUCTIZATION) — same shape as the lane report
# above, deliberately: only the code that INSTALLED the attention path can prove
# what it installed, and a static declaration would over-claim on a card whose
# kernel gate refused (the exact reason pgw#1104 rejected position 2).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _AppliedAttentionContext:
    applied: list[Any]  # list[attention_modes.AppliedAttention]


_APPLIED_ATTENTION_CTX: (
    "contextvars.ContextVar[Optional[_AppliedAttentionContext]]"
) = contextvars.ContextVar("gen_worker_applied_attention_ctx", default=None)


class AppliedAttentionScope:
    """Held open by the executor around one ``setup()`` so a report lands on
    that instance and cannot be forged from a handler or a background thread."""

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
    """Report the attention path that was actually INSTALLED on ``component``.

    Call it from ``setup()`` right after the processor/dispatch is patched —
    the way ``report_applied_lane()`` is called after ``quantize_()`` returns.
    ``mode`` is ``"dense"`` or ``"sparse-k<N>"``; an ungrammatical token raises
    ``ValueError``. Reporting nothing means dense, so no endpoint is obliged to
    call this.

    ``density`` is the MEASURED kept fraction, not the budget: ``k`` is what was
    asked for and the density is what the geometry produced, and the wall is a
    function of the second. Returns whether the report was recorded; outside a
    setup scope it logs once and returns False rather than raising."""
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
    """Report the attention KERNEL that was actually engaged (th#1871 P1).

    Call it from ``setup()`` wherever the backend is chosen — right after the
    attention processor is installed, or right after the ``try: import
    flash_attn`` that decided it. ``backend`` is one of ``fa3``, ``fa2``,
    ``sdpa``, ``xformers``, ``eager`` (the ecosystem's own spellings —
    ``flash_attention_2``, ``torch_sdpa``, … — are accepted and normalized).
    ``wanted`` is what the code ASKED FOR, and passing it is what makes a
    fallback visible: ``wanted="fa2", backend="sdpa"`` is ie#707 exactly.

    WHY THIS IS A SECOND FUNCTION AND NOT A LOOSER GRAMMAR ON THE FIRST.
    ``report_applied_attention`` reports SPARSITY (``dense`` / ``sparse-k<N>``)
    and correctly refuses ``"sdpa"`` — the kernel is not a sparsity budget.
    Those are two independent axes: the same ``sparse-k8`` costs roughly twice
    as much on ``sdpa`` as on ``fa3``. Widening one token to cover both would
    make every measurement of either uninterpretable, which is the vocabulary
    collapse th#1871 §1.3 measured one layer up.

    Reporting nothing is honest and stays the default: an unreported backend is
    UNKNOWN to the hub, never "fine". Returns whether the report was recorded;
    outside a setup scope it logs once and returns False rather than raising.
    An ungrammatical backend token raises ``ValueError`` — the reporter is the
    last place a fourth vocabulary can be stopped."""
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


# ---------------------------------------------------------------------------
# Standalone (hub-less) resolution — the CLI's half. The executor's bytes
# come from orchestrator-resolved snapshots via ModelStore.ensure_local.
# ---------------------------------------------------------------------------


def _hub_ref_map_path(cache_dir: Path, thref: Any) -> Path:
    """CAS-local memory of release->snapshot resolutions, so a
    previously-fetched release ref keeps working offline:
    cas/refs/<owner>/<repo>/<release>. A ref naming no release memoizes under
    `_bare`, which is a repo-identity slot and resolves nothing on its own."""
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
    """Resolve a Hub ref via th#560 and download its snapshot into the CAS.

    One re-resolve retry on a presigned-URL expiry mid-download (the same
    contract the orchestrator honors on ``url_expired``).

    ``components`` (pgw#505): th#560's resolve route always returns the
    FULL repo manifest today (selective CAS resolve is the hub-side
    desired-snapshot scoping — a separate, not-yet-built platform change).
    Until then this narrows client-side: the worker fully owns this
    resolve+download+materialize loop (unlike the production executor path,
    which digest-verifies against an orchestrator-issued file list), so it
    can safely fetch only the declared components — ``ensure_snapshot_async``
    keys the materialized directory by ``(digest, components)`` so a partial
    fetch never collides with a full one of the same ref. NOTE: offline
    reuse (``--offline`` / the ``_hub_ref_map_path`` tag memory below) only
    covers the FULL-repo case — a components=-scoped ref must be fetched
    online at least once per component set.
    """
    # Deferred: cozy_snapshot pulls +305 modules onto the `import gen_worker`
    # path — the single largest boot-cost import in the SDK.
    from .cozy_snapshot import ensure_snapshot_async, snapshot_dir_key
    # Deferred: hub_client pulls +129 modules onto the `import gen_worker` path.
    from .hub_client import HubResolveError, resolve_repo

    canonical = thref.canonical()

    def _resolve() -> Any:
        try:
            return resolve_repo(thref)
        except HubResolveError as e:
            raise ModelResolutionError(str(e)) from e

    emit({"kind": "model_fetch.started", "ref": canonical, "provider": "tensorhub"})
    resolved = _resolve()

    # Already materialized under the resolved (digest, components) key? No download.
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
    """Resolve one model ref to a local tensorfs CAS snapshot dir.

    Order matches the live worker, and since pgw#1524 both have exactly one
    weight source:
      1. local CAS lookup (digest-pinned snapshot dirs).
      2. Cozy refs missing from CAS: standalone resolve against tensorhub's
         public resolve route (th#560), then the shared ``cozy_snapshot``
         downloader; ``--offline`` stays CAS-only (exit 3).
      3. anything else — hf, civitai, modelscope — is an INGEST source and is
         REFUSED by name (``NonCasWeightSourceRefused``). The direct-download
         rungs are DELETED: a hub-less CLI that quietly served un-normalized
         upstream bytes would be a second answer to "what is a servable model",
         and the whole point of the hardcut is that there is only one.

    ``components`` (pgw#505) narrows the tensorhub fetch to the named pipeline
    component subfolders (+ root config files) — see
    ``download.select_component_paths`` / ``cozy_snapshot.snapshot_dir_key``.
    """

    cache_dir = Path(tensorhub_cas_dir())

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

    # Cozy refs that miss the CAS (#379): resolve standalone against
    # tensorhub's public resolve route (th#560) and feed the shared
    # cozy_snapshot downloader. TENSORHUB_URL selects the hub; TENSORHUB_TOKEN
    # (optional) unlocks private repos. Offline stays CAS-only.
    if parsed.provider == "tensorhub" and parsed.tensorhub is not None:
        if offline:
            # Release refs: a previous online resolve remembered release->digest.
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

    # THE HARDCUT (pgw#1524). hf / civitai / modelscope are INGEST sources;
    # this resolver serves, so it refuses them by name and points at the route.
    if parsed.provider in ("hf", "civitai", "modelscope"):
        raise ModelResolutionError(str(non_cas_refusal(
            ref=str(ref), provider=parsed.provider)))

    raise ModelResolutionError(
        f"unsupported model ref: {ref!r} (provider={provider!r})"
    )
