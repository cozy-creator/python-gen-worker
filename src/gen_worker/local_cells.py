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
    reason = cc.verify(meta, family=family)
    if reason:
        return reason
    reason = cc.mode_drift(meta, pipe) or cc.lane_drift(meta, pipe)
    if reason:
        return reason
    want = "regional" if getattr(cfg, "regional", False) else "whole"
    have = str(meta.get("compile_mode") or "whole")
    if have != want:
        return f"compile_mode {have!r} != declared {want!r}"
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

    meta = cc.artifact_metadata(
        family=family,
        source_ref="local-mint",
        shapes=cfg.shapes,
        targets=cfg.targets,
        low_vram_mode=low_vram_mode(pipe),
        compile_mode="regional" if getattr(cfg, "regional", False) else "whole",
        weight_lane=pipeline_weight_lane(pipe),
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


def enable_compiled(
    pipe: Any, cfg: Any, cache_dir: Optional[Path] = None,
    artifact: Optional[Path] = None,
) -> bool:
    """Local-CLI compile arming: delivered/env artifacts first (production
    policy, incl. TRT and the manual GEN_WORKER_COMPILE_CACHE overrides),
    then the local cell store — adopt a matching stored cell, or mint one.

    Never raises: any store or mint failure logs and serves eager, exactly
    like the production miss policy.
    """
    from .models import provision

    family = str(getattr(cfg, "family", "") or "")
    if provision.enable_compiled(pipe, cfg, cache_dir, artifact):
        return True
    if not family:
        logger.info("local-cells: Compile decl has no family; staying eager")
        return False
    if not _cuda_ready():
        return False  # apply() already logged; nothing to mint for

    from .models.loading import pipeline_weight_lane

    target = cell_path(family, pipeline_weight_lane(pipe))
    if target.exists():
        reason = store_verdict(target, family, pipe, cfg)
        if not reason:
            # the exact delivered-cell consumer path (verify + drift + arm)
            if cc.enable(pipe, cfg, cache_dir, artifact=target):
                _say(f"local-cells: adopted stored cell {target.name}")
                return True
            reason = "seed/arm failed"
        _say(f"local-cells: stored cell no longer matches ({reason}); re-minting")

    if not cc.toolchain_present():
        _say(
            "local-cells: this endpoint supports torch.compile but no C "
            "compiler is installed (need cc/gcc/clang); serving eager. "
            "Install one to let cozy compile once and cache the result."
        )
        return False
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
        return False
    # Adopt the just-saved cell through the delivered-cell path (drops the
    # unguarded mint wrappers; re-traces hit the captured FX cache). This
    # also proves the saved artifact round-trips before we rely on it.
    cc.unwrap(pipe)
    if cc.enable(pipe, cfg, cache_dir, artifact=target):
        return True
    logger.warning("local-cells: minted cell failed re-adoption; serving eager")
    return False


__all__ = [
    "ENV_STORE_DIR",
    "cell_path",
    "enable_compiled",
    "store_root",
    "store_verdict",
]
