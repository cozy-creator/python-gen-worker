"""Self-minted local compile cells for the cozy-local runtime (gw#555).

Production compile cells are minted by the platform's first-party compile
job and DELIVERED to workers (#384/#569); a production worker that has no
cell stays eager — it never compiles and never publishes. cozy-local users
run on GPUs the platform may never have compiled for (an RTX 5080, say), so
the LOCAL runtime is allowed to mint its own cell: when a compile-enabled
endpoint loads and no stored cell matches this runtime, compile once (the
endpoint's own ``Compile`` recipe — same shapes/targets/regional mode the
production producer uses), pack the result with the production artifact
format, and save it in the user's cozy directory. Subsequent boots adopt it
through the exact delivered-cell code path.

TRUST BOUNDARY — local cells are never uploaded or shared. A compile cell
is user-generated EXECUTABLE code (compiled kernels + generated C++/Triton
sources); accepting one from a user machine into shared storage would let
any user ship arbitrary code into other people's GPU workers. Enforcement
is structural, not a flag:

* this module has no publish path — it is not a CAS client, imports no
  upload/transport machinery, and writes only under the local store root;
* only the local CLI serve path (``cli/run.py``) calls into it — the
  production executor arms compile exclusively through hub-DELIVERED
  artifacts (``executor._enable_compiled``) and never imports this module.

Key discipline is unchanged from production: adoption reuses
``compile_cache.verify()`` EXACT semantics verbatim (artifact format, SKU,
torch, triton, gen-worker version, lib versions, family) plus the same
``mode_drift``/``lane_drift``/``compile_mode`` parity checks. Any mismatch
re-mints and atomically replaces the stored cell — stale kernels are never
served.

Store layout (one cell per (family, SKU/torch/lane) key)::

    <store>/<family>/<flavor_label>.tar.gz     # production artifact format
    <store>/.mint/*                            # in-progress capture dirs

The store root is ``$GEN_WORKER_LOCAL_CELLS_DIR`` (exported by cozy-local's
``workerEnv``, cozy-local #47), defaulting to ``~/.cache/cozy/compile-cells``
— a path relocation like ``TENSORHUB_CACHE_DIR``, not a behavior knob.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import tarfile
import time
from pathlib import Path
from typing import Any, Optional

from . import compile_cache as cc

logger = logging.getLogger(__name__)

ENV_STORE_DIR = "GEN_WORKER_LOCAL_CELLS_DIR"
_MINT_DIR = ".mint"
_STALE_MINT_S = 24 * 3600


def store_root() -> Path:
    env = os.environ.get(ENV_STORE_DIR, "").strip()
    if env:
        return Path(env).expanduser()
    return Path.home() / ".cache" / "cozy" / "compile-cells"


def cell_path(family: str, weight_lane: str = "", root: Optional[Path] = None) -> Path:
    """Where this runtime's cell for ``family`` lives in the local store."""
    key = cc.runtime_key()
    label = cc.flavor_label(key["sku"], key["torch"], weight_lane)
    return (root or store_root()) / family / f"{label}.tar.gz"


def store_verdict(artifact: Path, family: str, pipe: Any, cfg: Any) -> str:
    """'' when the stored cell is adoptable by this runtime + pipeline, else
    the mismatch reason (production verify() + drift parity, verbatim)."""
    try:
        with tarfile.open(artifact, mode="r:*") as tar:  # metadata only
            member = tar.extractfile(cc.METADATA_NAME)
            if member is None:
                return "no metadata.json"
            meta = json.loads(member.read().decode())
    except Exception as exc:  # noqa: BLE001 — any unreadable cell re-mints
        return f"unreadable artifact ({exc})"
    # th#883/gw#581: ONE compatibility brain — when both sides can state a
    # cell key (local mints stamp one via artifact_metadata), the verdict is
    # the exact key comparison fleet workers use; pre-key cells fall back to
    # the legacy axis-by-axis verify.
    reason = ""
    try:
        from . import cell_key
        from .models.loading import pipeline_weight_lane

        want = cell_key.compute(
            family, pipeline_weight_lane(pipe),
            int(getattr(cfg, "lora_bucket", 0) or 0),
            regional=bool(getattr(cfg, "regional", False)),
        )
        reason = cell_key.mismatch(meta, want)
        if reason and "records no computable key" in reason:
            reason = cc.verify(meta, family=family)
    except Exception:
        reason = cc.verify(meta, family=family)
    if reason:
        return reason
    reason = cc.mode_drift(meta, pipe) or cc.lane_drift(meta, pipe)
    if reason:
        return reason
    want_mode = "regional" if getattr(cfg, "regional", False) else "whole"
    have = str(meta.get("compile_mode") or "whole")
    if have != want_mode:
        return f"compile_mode {have!r} != declared {want_mode!r}"
    return ""


def _estimate_minutes(cfg: Any) -> int:
    """Rough one-time compile estimate: ~45s per image shape, ~3min per
    video (w, h, frames) shape — order-of-magnitude UX, not a promise."""
    mins = 0.0
    for shape in getattr(cfg, "shapes", ()) or ():
        mins += 3.0 if len(shape) >= 3 else 0.75
    return max(1, round(mins))


def _cuda_ready() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _say(msg: str) -> None:
    logger.info("%s", msg)
    print(msg, file=sys.stderr)


def _sweep_stale_mints(root: Path) -> None:
    base = root / _MINT_DIR
    if not base.is_dir():
        return
    now = time.time()
    for d in base.iterdir():
        try:
            if now - d.stat().st_mtime > _STALE_MINT_S:
                shutil.rmtree(d, ignore_errors=True)
        except OSError:
            continue


def _compile_and_warm(pipe: Any, cfg: Any, *, steps: int = 2) -> None:
    """Cold-compile ``pipe`` over the declared shape table (the only part of
    a mint that needs CUDA + a toolchain). ``guard=False``: a failing warm
    call must fail the mint — a silently-eager capture must never be saved."""
    if not cc.apply(pipe, cfg, cache_ready=False, guard=False, allow_cold=True):
        raise RuntimeError(f"no compile targets resolved on {type(pipe).__name__}")
    import torch

    decode = any(t.startswith("vae") for t in cfg.targets)
    for shape in cfg.shapes:
        torch.cuda.synchronize()
        t0 = time.monotonic()
        cc._warm_call(
            pipe, shape, steps=steps,
            prompt="cache warm-up: a lighthouse on a cliff at dawn, detailed",
            decode=decode,
            guidance_scales=getattr(cfg, "guidance_scales", ()),
        )
        torch.cuda.synchronize()
        key = "x".join(str(v) for v in shape)
        _say(f"  compiled {key} in {time.monotonic() - t0:.0f}s")


def _mint(pipe: Any, cfg: Any, target: Path, family: str) -> Path:
    """Compile this pipeline once and save the cell atomically at ``target``.

    The capture uses the production artifact recipe end to end
    (``capture_env`` -> warm the shape table -> ``artifact_metadata`` ->
    deterministic ``pack``), so the saved cell is byte-compatible with a
    delivered one and adopts through the identical code path.
    """
    root = store_root()
    _sweep_stale_mints(root)
    capture = root / _MINT_DIR / f"{target.name[: -len('.tar.gz')]}-{os.getpid()}"
    cc.capture_env(capture)
    started = time.monotonic()

    _compile_and_warm(pipe, cfg)

    captured = [p for p in (capture / "inductor").rglob("*") if p.is_file()]
    if not captured:
        raise RuntimeError(
            "compile warm-up captured nothing under TORCHINDUCTOR_CACHE_DIR"
        )
    from .models.loading import pipeline_weight_lane
    from .models.memory import low_vram_mode

    # gw#564: record the execution contract exactly like the production
    # build — w8a8 cells are contract_drift-gated on the graph signature and
    # weight-lane manifest, so a mint without them can never re-adopt.
    graph_signature, weight_contract = cc.execution_contract(pipe, cfg)
    meta = cc.artifact_metadata(
        family=family,
        source_ref="local-mint",
        shapes=cfg.shapes,
        targets=cfg.targets,
        guidance_scales=getattr(cfg, "guidance_scales", ()),
        low_vram_mode=low_vram_mode(pipe),
        compile_mode="regional" if getattr(cfg, "regional", False) else "whole",
        weight_lane=pipeline_weight_lane(pipe),
        lora_bucket=int(getattr(cfg, "lora_bucket", 0) or 0),
        graph_signature=graph_signature,
        weight_contract=weight_contract,
    )
    tmp = target.with_suffix(".part")
    target.parent.mkdir(parents=True, exist_ok=True)
    cc.pack(capture, tmp, meta)
    os.replace(tmp, target)
    _say(
        f"compile cell saved: {target} "
        f"({target.stat().st_size / 1e6:.1f} MB, {time.monotonic() - started:.0f}s total); "
        "future loads reuse it with no compile"
    )
    return target


def _fail_closed(pipe: Any, reason: str) -> bool:
    """The production quantized-lane policy re-asserted at the local exits:
    a w8a8/w4a4 pipeline never silently serves the slow eager GEMM path —
    when the local mint cannot produce a cell either, the refusal stays
    TYPED (the same CompiledLaneUnavailableError the executor maps). Plain
    lanes keep the gw#555 never-raise miss policy (eager)."""
    from .models.loading import pipeline_weight_lane

    lane = pipeline_weight_lane(pipe)
    if lane.startswith(("w8a8", "w4a4")):
        raise cc.CompiledLaneUnavailableError(
            f"{lane[:4].upper()} requires a compile cell and the local mint "
            f"is unavailable ({reason})")
    return False


def enable_compiled(
    pipe: Any, cfg: Any, cache_dir: Optional[Path] = None,
    artifact: Optional[Path] = None,
) -> bool:
    """Local-CLI compile arming: an explicitly delivered artifact first,
    then the local cell store — adopt a matching stored cell, or mint one.

    Store/mint failures log and serve eager (the production miss policy) —
    EXCEPT on w8a8 lanes, whose fail-closed refusal stays typed
    (:func:`_fail_closed`): production raises CompiledLaneUnavailableError
    when no cell can arm, and the local runtime only downgrades that to a
    mint attempt, never to silent slow eager (gw#564 found the raise
    aborting the mint path live on a 4090).
    """
    from .models import provision

    family = str(getattr(cfg, "family", "") or "")
    try:
        if provision.enable_compiled(pipe, cfg, cache_dir, artifact):
            return True
    except cc.CompiledLaneUnavailableError:
        # No delivered w8a8 cell => production would refuse here. The LOCAL
        # runtime's whole purpose is minting that cell (gw#555): fall
        # through to the store/mint path — the typed refusal re-asserts at
        # every exit that cannot produce one.
        logger.info(
            "local-cells: no delivered w8a8 cell; trying the local store/mint")
    if not family:
        logger.info("local-cells: Compile decl has no family; staying eager")
        return _fail_closed(pipe, "Compile decl has no family")
    if not _cuda_ready():
        # apply() already logged; nothing to mint for
        return _fail_closed(pipe, "CUDA unavailable")

    from .models.loading import pipeline_weight_lane

    # gw#561: the eager-miss rollback in provision.enable_compiled dropped
    # the branch lane; local store/mint must key + trace the DECLARED graph
    # family, so re-apply it for the rest of this arming attempt.
    bucket = int(getattr(cfg, "lora_bucket", 0) or 0)
    if bucket:
        cc.apply_lora_lane(pipe, bucket)

    target = cell_path(family, pipeline_weight_lane(pipe))
    if target.exists():
        reason = store_verdict(target, family, pipe, cfg)
        if not reason:
            # the exact delivered-cell consumer path (verify + drift + arm)
            try:
                if cc.enable(pipe, cfg, cache_dir, artifact=target):
                    _say(f"local-cells: adopted stored cell {target.name}")
                    return True
                reason = "seed/arm failed"
            except cc.CellSelectionBugError as exc:
                # th#883 invariant, local edition: a stored cell whose axes
                # describe exactly this runtime refused to arm — a bug in
                # the one selection brain. Loud, then re-mint (the local
                # store can always replace its cell).
                _say(f"local-cells: cell_selection_bug: {exc}")
                reason = f"cell_selection_bug: {exc}"
            except cc.CompiledLaneUnavailableError:
                reason = "seed/arm failed (quantized-lane fail-closed)"
        _say(f"local-cells: stored cell no longer matches ({reason}); re-minting")

    if not cc.toolchain_present():
        _say(
            "local-cells: this endpoint supports torch.compile but no C "
            "compiler is installed (need cc/gcc/clang); serving eager. "
            "Install one to let cozy compile once and cache the result."
        )
        if bucket:
            cc.drop_lora_lane(pipe)
        return _fail_closed(pipe, "no C compiler for the one-time local mint")
    _say(
        f"local-cells: no compile cell for this GPU/torch yet — compiling "
        f"{len(tuple(cfg.shapes))} graph shape(s) once "
        f"(~{_estimate_minutes(cfg)} min). The cell will be saved to "
        f"{target} and reused forever."
    )
    try:
        _mint(pipe, cfg, target, family)
    except Exception as exc:  # noqa: BLE001 — mint failure => eager, never fatal
        logger.warning("local-cells: mint failed (%s); serving eager", exc)
        cc.unwrap(pipe)
        if bucket:
            cc.drop_lora_lane(pipe)
        return _fail_closed(pipe, f"local mint failed: {exc}")
    # Adopt the just-saved cell through the delivered-cell path (drops the
    # unguarded mint wrappers; re-traces hit the captured FX cache). This
    # also proves the saved artifact round-trips before we rely on it.
    cc.unwrap(pipe)
    try:
        if cc.enable(pipe, cfg, cache_dir, artifact=target):
            return True
    except cc.CellSelectionBugError as exc:
        _say(f"local-cells: cell_selection_bug on freshly minted cell: {exc}")
        if bucket:
            cc.drop_lora_lane(pipe)
        return _fail_closed(pipe, f"cell_selection_bug: {exc}")
    except cc.CompiledLaneUnavailableError as exc:
        logger.warning("local-cells: minted cell failed re-adoption (%s)", exc)
        if bucket:
            cc.drop_lora_lane(pipe)
        raise
    logger.warning("local-cells: minted cell failed re-adoption; serving eager")
    if bucket:
        cc.drop_lora_lane(pipe)
    return _fail_closed(pipe, "minted cell failed re-adoption")


__all__ = [
    "ENV_STORE_DIR",
    "cell_path",
    "enable_compiled",
    "store_root",
    "store_verdict",
]
