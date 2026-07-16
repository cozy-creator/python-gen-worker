"""Per-SKU torch.compile cache artifacts (#384).

torch.compile wins 15-34% warm latency on flux-class models but costs 20-46s
of compile per (model, resolution) and needs a C toolchain the prod worker
images don't ship. The split: a platform compile job (training-endpoints
``produce-inductor-cache``) compiles once per GPU SKU and publishes the
inductor+triton cache dirs as a repo flavor; workers that opt in via
``@endpoint(compile=Compile(...))`` seed those dirs before load and hit the
cache with no compiler and no stall.

Policy: cache miss / key mismatch / no artifact => eager, never a boot stall
or a runtime compile attempt in prod. The compile job itself opts into cold
compilation with ``GEN_WORKER_COMPILE_ALLOW_COLD=1`` (requires a toolchain).

Artifacts are FAMILY-keyed (settled 2026-07-06): torch.compile caches key on
the traced graph + shapes, not the weights, so one artifact serves every
fine-tune of a model family. They live in a system-owned repo per family
(``_system/family-<family>``), one flavor per (SKU, torch) cell — and they
are CODE: only the platform's first-party compile job publishes shared ones.

Artifact = deterministic ``.tar.gz``::

    metadata.json      key: family, sku/torch/triton/cuda, shapes, targets,
                       diffusers/transformers versions (+ source_ref info)
    inductor/**        TORCHINDUCTOR_CACHE_DIR contents
    triton/**          TRITON_CACHE_DIR contents

Key sensitivity (all exact-match): family (graph identity), GPU SKU
(autotune choices + cubin arch), torch (fx-graph cache key), triton
(cubin/launcher cache key), diffusers (the traced graph is its code), and
gen-worker itself plus the producer's low-VRAM prep mode (gw#391: the
worker's load/wrap/placement code shapes the traced graph — a cell produced
by a different gen-worker, or traced under different low-VRAM flags, can
pass every other key yet miss inductor's FX-graph cache at trace time,
serving eager while reporting adopted). ``source_ref`` records which family member the producer
compiled from — informational, never part of the match.
"""

from __future__ import annotations

import functools
import gzip
import hashlib
import io
import json
import logging
import os
import shutil
import tarfile
import time
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

logger = logging.getLogger(__name__)

ENV_CACHE_PATH = "GEN_WORKER_COMPILE_CACHE"       # local artifact (tar) or seeded dir
ENV_CACHE_URL = "GEN_WORKER_COMPILE_CACHE_URL"    # http(s) URL to the artifact
ENV_ALLOW_COLD = "GEN_WORKER_COMPILE_ALLOW_COLD"  # compile without an artifact (needs cc)

METADATA_NAME = "metadata.json"
# 2 (gw#391): key gained the producer gen-worker version; format-1 cells
# (gw#384 era) demonstrably miss the FX-graph cache on current code.
ARTIFACT_FORMAT = 2
_MARKER_ATTR = "_cozy_compile"
_JUNK_SUFFIXES = (".lock", ".tmp", ".pid")


# ---------------------------------------------------------------------------
# Key
# ---------------------------------------------------------------------------


def sku_slug(gpu_name: str) -> str:
    """Deterministic SKU slug: ``NVIDIA GeForce RTX 4090`` -> ``rtx-4090``,
    ``NVIDIA H100 80GB HBM3`` -> ``h100-80gb-hbm3``."""
    s = str(gpu_name or "").lower()
    for noise in ("nvidia", "geforce"):
        s = s.replace(noise, " ")
    out = "".join(c if c.isalnum() else "-" for c in s).strip("-")
    while "--" in out:
        out = out.replace("--", "-")
    return out


def runtime_key() -> Dict[str, str]:
    """The consumer-side half of the cache key, probed from this process."""
    key = {"sku": "", "torch": "", "triton": "", "cuda": ""}
    try:
        import torch

        key["torch"] = str(torch.__version__)
        key["cuda"] = str(torch.version.cuda or "")
        if torch.cuda.is_available():
            key["sku"] = sku_slug(torch.cuda.get_device_name(0))
    except Exception:
        pass
    try:
        import triton

        key["triton"] = str(triton.__version__)
    except Exception:
        pass
    return key


def _lib_versions() -> Dict[str, str]:
    out: Dict[str, str] = {}
    for lib in ("diffusers", "transformers"):
        try:
            out[lib] = str(__import__(lib).__version__)
        except Exception:
            pass
    return out


def gen_worker_version() -> str:
    try:
        from importlib.metadata import version

        return str(version("gen-worker"))
    except Exception:
        return ""


def lane_token(weight_lane: str) -> str:
    """Label token for a traced weight lane (gw#534): cells of different
    lanes are DIFFERENT graphs and must not collide on one flavor label.
    "" (plain resident, incl. bf16-resident) stays unsuffixed. LoRA-branch
    lanes (``w8a8-lora<bucket>``, gw#547) pass through — one graph family
    per rank bucket."""
    return {"": "", "fp8-hooks": "w8a16", "w8a8": "w8a8"}.get(
        str(weight_lane or ""), str(weight_lane))


def flavor_label(sku: str, torch_version: str, weight_lane: str = "") -> str:
    """Repo-flavor label for an artifact: ``inductor-rtx-4090-torch2.9`` (+
    ``-w8a8``/``-w8a16`` for non-plain weight lanes, gw#534). The full
    versions live in metadata; the label is for humans + selection. MUST stay
    byte-compatible with tensorhub's compilecache.FlavorLabel."""
    short = ".".join(str(torch_version).split("+")[0].split(".")[:2])
    label = f"inductor-{sku}-torch{short}"
    tok = lane_token(weight_lane)
    return f"{label}-{tok}" if tok else label


def system_repo(family: str) -> str:
    """The system-owned repo holding one family's compiled-artifact cells."""
    fam = str(family or "").strip()
    if not fam:
        raise ValueError("compile-cache family must be non-empty")
    return f"_system/family-{fam}"


def parse_cell_ref(ref: str) -> Tuple[str, str]:
    """(family, flavor) from a system cell ref
    (``_system/family-<f>[:tag][@digest][#<flavor>]``) via the ONE ref
    grammar (gw#492); ('', '') when the ref is not a system-family ref."""
    from .models.refs import parse_model_ref

    try:
        parsed = parse_model_ref(str(ref or ""))
    except ValueError:
        return "", ""
    th = parsed.tensorhub
    if th is None or th.owner != "_system" or not th.repo.startswith("family-"):
        return "", ""
    return th.repo[len("family-"):], th.flavor or ""


def family_from_ref(ref: str) -> str:
    """Family encoded in a compile-cache ref; '' when the ref is not a
    system-family cell ref."""
    return parse_cell_ref(ref)[0]


def is_cache_ref(ref: str, family: str = "") -> bool:
    """True when ``ref`` names an inductor compile-cache cell (optionally of
    one specific family)."""
    fam, flavor = parse_cell_ref(ref)
    if not fam or (family and fam != family):
        return False
    return flavor.startswith("inductor-")


def artifact_metadata(
    *,
    family: str,
    source_ref: str = "",
    source_digest: str = "",
    shapes: Iterable[Tuple[int, ...]] = (),
    targets: Iterable[str] = (),
    low_vram_mode: str = "",
    storage_dtype: str = "",
    compile_mode: str = "whole",
    weight_lane: str = "",
) -> Dict[str, Any]:
    """Producer-side metadata for :func:`pack` (no timestamps: artifacts of
    identical content must be byte-identical). ``source_ref``/``source_digest``
    record the family member compiled from — informational only.
    ``low_vram_mode`` is the prep mode the producer pipeline was traced under
    (gw#391): its flags are traced into the graphs, so a consumer prepped in a
    different mode must reject the cell. ``storage_dtype`` records the weight
    storage the binding REQUESTED — informational only. ``weight_lane`` is the
    lane the built pipeline ACTUALLY traced under (gw#534:
    ``loading.pipeline_weight_lane`` — "" plain-resident, "fp8-hooks"
    layerwise-cast; the hooks are traced INTO the graphs, ie#381) and is
    parity-checked at :func:`enable` like ``low_vram_mode``. Shape rows are
    (w, h) or (w, h, frames) — see ``Compile``."""
    return {
        "format": ARTIFACT_FORMAT,
        "kind": "torch-inductor-cache",
        **runtime_key(),
        "gen_worker": gen_worker_version(),
        "family": str(family or ""),
        "source_ref": str(source_ref or ""),
        "source_digest": str(source_digest or ""),
        "shapes": [[int(v) for v in s] for s in shapes],
        "targets": list(targets),
        "low_vram_mode": str(low_vram_mode or ""),
        "storage_dtype": str(storage_dtype or ""),
        "compile_mode": str(compile_mode or "whole"),
        "weight_lane": str(weight_lane or ""),
        "libs": _lib_versions(),
    }


def verify(meta: Dict[str, Any], *, family: str = "") -> str:
    """'' when the artifact matches this runtime, else the mismatch reason.

    Family is the graph-identity half of the key: checked when both sides
    declare one (fine-tunes of the same family share caches by design)."""
    if int(meta.get("format") or 0) != ARTIFACT_FORMAT:
        return f"format {meta.get('format')!r} != {ARTIFACT_FORMAT}"
    here = runtime_key()
    for field in ("sku", "torch", "triton"):
        want, have = str(meta.get(field) or ""), here[field]
        if want != have:
            return f"{field} {want!r} != runtime {have!r}"
    want_gw, have_gw = str(meta.get("gen_worker") or ""), gen_worker_version()
    if want_gw != have_gw:
        # gw#391: the producer's gen-worker shapes the traced graph; a version
        # drift means the FX-graph cache keys may no longer match.
        return f"gen_worker {want_gw!r} != runtime {have_gw!r}"
    for lib, have in _lib_versions().items():
        want = str((meta.get("libs") or {}).get(lib) or "")
        if want and want != have:
            return f"{lib} {want!r} != runtime {have!r}"
    want_fam = str(meta.get("family") or "")
    if family and want_fam and want_fam != family:
        return f"family {want_fam!r} != {family!r}"
    return ""


# ---------------------------------------------------------------------------
# Pack / unpack
# ---------------------------------------------------------------------------


def _clean_tarinfo(ti: tarfile.TarInfo, executable: bool = False) -> tarfile.TarInfo:
    ti.uid = ti.gid = 0
    ti.uname = ti.gname = ""
    ti.mtime = 0
    ti.mode = 0o755 if executable else 0o644
    return ti


def pack(cache_root: Path, out_path: Path, metadata: Dict[str, Any]) -> Path:
    """Deterministic artifact from a capture root holding ``inductor/`` and
    ``triton/``: sorted entries, zeroed times/owners, gzip mtime 0 — identical
    content always packs to identical bytes."""
    cache_root = Path(cache_root)
    out_path = Path(out_path)
    files: list[Path] = []
    for sub in ("inductor", "triton"):
        base = cache_root / sub
        if base.is_dir():
            files.extend(
                p for p in base.rglob("*")
                if p.is_file() and not p.name.endswith(_JUNK_SUFFIXES)
            )
    files.sort(key=lambda p: str(p.relative_to(cache_root)))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as raw:
        with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as gz:
            with tarfile.open(fileobj=gz, mode="w") as tar:
                meta_bytes = json.dumps(metadata, sort_keys=True, indent=1).encode()
                ti = _clean_tarinfo(tarfile.TarInfo(METADATA_NAME))
                ti.size = len(meta_bytes)
                tar.addfile(ti, io.BytesIO(meta_bytes))
                for p in files:
                    rel = str(p.relative_to(cache_root))
                    ti = _clean_tarinfo(
                        tarfile.TarInfo(rel), executable=os.access(p, os.X_OK)
                    )
                    ti.size = p.stat().st_size
                    with open(p, "rb") as f:
                        tar.addfile(ti, f)
    return out_path


def unpack(artifact: Path, dest_root: Path) -> Dict[str, Any]:
    """Extract an artifact's ``inductor/``+``triton/`` trees into ``dest_root``
    (merging with whatever is already seeded) and return its metadata."""
    dest_root = Path(dest_root)
    dest_root.mkdir(parents=True, exist_ok=True)
    meta: Dict[str, Any] = {}
    with tarfile.open(artifact, mode="r:*") as tar:
        for member in tar:
            name = member.name.lstrip("./")
            if name == METADATA_NAME:
                f = tar.extractfile(member)
                meta = json.loads(f.read().decode()) if f else {}
                continue
            if not member.isfile():
                continue
            parts = Path(name).parts
            if (
                not parts
                or parts[0] not in ("inductor", "triton")
                or ".." in parts
                or Path(name).is_absolute()
            ):
                raise ValueError(f"unsafe or unknown member in compile-cache artifact: {member.name!r}")
            target = dest_root.joinpath(*parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            src = tar.extractfile(member)
            assert src is not None
            with open(target, "wb") as out:
                shutil.copyfileobj(src, out)
            if member.mode & 0o100:
                target.chmod(0o755)
    if not meta:
        raise ValueError(f"compile-cache artifact {artifact} has no {METADATA_NAME}")
    return meta


# ---------------------------------------------------------------------------
# Capture (producer) / seed (consumer)
# ---------------------------------------------------------------------------


def capture_env(root: Path) -> Path:
    """Point inductor+triton at the dirs under ``root`` (producer capture and
    consumer seeding share this contract). Safe mid-process: latched inductor
    path caches are cleared so a hot adoption's re-seed actually takes effect
    (gw#391 — the worker has been serving eager long before seeding)."""
    root = Path(root)
    for sub, env in (("inductor", "TORCHINDUCTOR_CACHE_DIR"), ("triton", "TRITON_CACHE_DIR")):
        d = root / sub
        d.mkdir(parents=True, exist_ok=True)
        os.environ[env] = str(d)
    _reset_inductor_latch()
    return root


def _reset_inductor_latch() -> None:
    """Clear inductor's in-memory caches that may have latched the previous
    cache-dir paths (torch's own ``temporary_cache_dir`` does the same)."""
    import sys

    if "torch" not in sys.modules:
        return
    try:
        from torch._inductor.utils import clear_caches

        clear_caches()
    except Exception:
        logger.debug("compile-cache: inductor latch reset unavailable", exc_info=True)


# seeding reuses the same env contract
seed_env = capture_env


def inductor_counters() -> Dict[str, int]:
    """This process's inductor FX-graph cache counters (monotonic). The delta
    across a warmup is the honest adopted-vs-silently-eager signal (gw#391):
    zero hits means the seeded cell never served the trace."""
    try:
        from torch._dynamo.utils import counters

        c = counters["inductor"]
        return {
            k: int(c.get(k, 0))
            for k in ("fxgraph_cache_hit", "fxgraph_cache_miss", "fxgraph_cache_bypass")
        }
    except Exception:
        return {}


def counters_delta(before: Dict[str, int], after: Dict[str, int]) -> Dict[str, int]:
    return {k: int(after.get(k, 0)) - int(before.get(k, 0)) for k in after}


def toolchain_present() -> bool:
    return any(shutil.which(c) for c in ("cc", "gcc", "g++", "clang"))


class AdoptError(RuntimeError):
    """Classified adoption failure (ModelEvent ``adopt_failed:<reason>``)."""

    def __init__(self, reason: str, detail: str = "") -> None:
        self.reason = reason
        super().__init__(detail or reason)


def find_artifact(root: Path) -> Optional[Path]:
    """The compile-cache tarball inside a downloaded snapshot dir (or the
    file itself)."""
    root = Path(root)
    if root.is_file():
        return root
    return next(iter(sorted(root.rglob("*.tar.gz"))), None)


def seed_artifact(
    artifact: Path, family: str, cache_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """Unpack + verify + seed one artifact for this runtime. Returns its
    metadata; raises :class:`AdoptError` (never seeds) on any mismatch."""
    root = (Path(cache_dir) if cache_dir else Path.home() / ".cache" / "gen-worker")
    root = root / "compile-cache"
    try:
        meta = unpack(Path(artifact), root)
    except Exception as exc:
        raise AdoptError("artifact_invalid", str(exc)) from exc
    reason = verify(meta, family=family)
    if reason:
        raise AdoptError("key_mismatch", reason)
    seed_env(root)
    return meta


def mode_drift(meta: Dict[str, Any], pipeline: Any) -> str:
    """'' when the producer's low-VRAM prep mode matches this pipeline's, else
    the mismatch (gw#391). The prep flags (VAE tiling/slicing, attention
    slicing, offload hooks) are traced into the FX graphs, so a mode drift is
    a guaranteed cache miss. Enforced only when the producer recorded one —
    the check is per-pipeline, so it lives outside :func:`verify`."""
    want = str(meta.get("low_vram_mode") or "")
    if not want:
        return ""
    from .models.memory import low_vram_mode

    have = low_vram_mode(pipeline)
    if want != have:
        return f"low_vram_mode {want!r} != pipeline {have!r}"
    return ""


def lane_drift(meta: Dict[str, Any], pipeline: Any) -> str:
    """'' when the cell's traced weight lane matches this pipeline's, else the
    mismatch (gw#534). Enforced SYMMETRICALLY (unlike ``mode_drift``): a
    bf16-resident pipeline must never adopt hook-cast-traced graphs and vice
    versa — both directions are guaranteed FX-graph misses that would serve
    eager while reporting adopted (the gw#391 bug class)."""
    want = str(meta.get("weight_lane") or "")
    from .models.loading import pipeline_weight_lane

    have = pipeline_weight_lane(pipeline)
    if want != have:
        return f"weight_lane {want!r} != pipeline {have!r}"
    return ""


def _fetch_url(url: str, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    name = hashlib.sha256(url.encode()).hexdigest()[:16] + ".tar.gz"
    target = dest_dir / name
    if target.exists():
        return target
    t = time.monotonic()
    tmp = target.with_suffix(".part")
    with urllib.request.urlopen(url, timeout=120) as resp, open(tmp, "wb") as out:
        shutil.copyfileobj(resp, out)
    tmp.rename(target)
    logger.info(
        "compile-cache: downloaded artifact (%d bytes in %.1fs)",
        target.stat().st_size, time.monotonic() - t,
    )
    return target


def prepare(
    family: str,
    cache_dir: Optional[Path] = None,
    artifact: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
    """Locate, verify, and seed a compile-cache artifact for this runtime.

    Sources, in order: explicit ``artifact`` (a hub-attached snapshot, #569),
    then ``GEN_WORKER_COMPILE_CACHE`` (local tar), then
    ``GEN_WORKER_COMPILE_CACHE_URL``. Returns the artifact metadata on a
    verified hit (cache dirs seeded), else None with the reason logged.

    ``local``/``url`` are raw env reads, not Settings fields — no production
    launcher has ever set them (pgw#514 dead-config sweep), but they're a
    real, tested manual-override path (see test_compile_cache.py) for local
    dev / the compile-cell producer job, kept as library-standalone knobs.
    """
    local = os.environ.get(ENV_CACHE_PATH, "").strip()
    url = os.environ.get(ENV_CACHE_URL, "").strip()
    root = Path(cache_dir) if cache_dir else Path.home() / ".cache" / "gen-worker"
    root = root / "compile-cache"
    try:
        if artifact is not None:
            artifact = Path(artifact)
            if not artifact.exists():
                logger.warning("compile-cache: attached artifact %s does not exist", artifact)
                return None
        elif local:
            artifact = Path(local)
            if not artifact.exists():
                logger.warning("compile-cache: %s=%s does not exist", ENV_CACHE_PATH, local)
                return None
        elif url:
            artifact = _fetch_url(url, root / "artifacts")
        else:
            logger.info("compile-cache: no artifact configured; staying eager")
            return None
        meta = seed_artifact(artifact, family, cache_dir=cache_dir)
        logger.info(
            "compile-cache: seeded verified artifact (sku=%s torch=%s shapes=%s)",
            meta.get("sku"), meta.get("torch"), meta.get("shapes"),
        )
        return meta
    except Exception as exc:
        logger.warning("compile-cache: artifact unusable (%s); staying eager", exc)
        return None


# ---------------------------------------------------------------------------
# Apply
# ---------------------------------------------------------------------------


def _resolve_target(pipeline: Any, target: str) -> Optional[Tuple[Any, str, Callable[..., Any]]]:
    """``"transformer"`` -> (module, 'forward', fn); ``"vae.decode"`` ->
    (vae, 'decode', fn). None when the pipeline has no such attribute."""
    obj = pipeline
    parts = target.split(".")
    for part in parts[:-1]:
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    leaf = getattr(obj, parts[-1], None)
    if leaf is None:
        return None
    if callable(getattr(leaf, "forward", None)) and parts[-1] != "forward":
        # a Module: compile its bound forward
        return leaf, "forward", leaf.forward
    if callable(leaf):
        return obj, parts[-1], leaf
    return None


def _guarded(original: Callable[..., Any], compiled: Callable[..., Any], label: str) -> Callable[..., Any]:
    """Never fail a request on compile problems: first error permanently
    unwraps to eager (prod images can't compile uncached shapes)."""
    state = {"failed": False}

    @functools.wraps(original)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if state["failed"]:
            return original(*args, **kwargs)
        try:
            return compiled(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 — any compile failure => eager
            state["failed"] = True
            logger.warning(
                "compile-cache: compiled %s failed (%s: %s); eager for the rest "
                "of this process", label, type(exc).__name__, exc,
            )
            return original(*args, **kwargs)

    return wrapper


def _clear_regional(mod: Any) -> None:
    """Undo nn.Module.compile() on every submodule (regional rollback)."""
    for m in mod.modules():
        if getattr(m, "_compiled_call_impl", None) is not None:
            m._compiled_call_impl = None


def _guarded_regional(mod: Any, original: Callable[..., Any], label: str) -> Callable[..., Any]:
    """Regional analogue of :func:`_guarded`: blocks are compiled in place,
    so eager fallback must first CLEAR the block compilations, then retry."""
    state = {"failed": False}

    @functools.wraps(original)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if not state["failed"]:
            try:
                return original(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001 — any compile failure => eager
                state["failed"] = True
                logger.warning(
                    "compile-cache: regional-compiled %s failed (%s: %s); eager "
                    "for the rest of this process", label, type(exc).__name__, exc,
                )
                _clear_regional(mod)
        return original(*args, **kwargs)

    return wrapper


def apply(
    pipeline: Any,
    cfg: Any,
    *,
    cache_ready: bool,
    guard: bool = True,
    allow_cold: Optional[bool] = None,
) -> bool:
    """Wrap ``cfg.targets`` on ``pipeline`` with compiled callables.

    Only compiles when a verified cache artifact was seeded (``cache_ready``)
    or the process explicitly opted into cold compilation AND has a C
    toolchain (``allow_cold``; defaults to the ``GEN_WORKER_COMPILE_ALLOW_COLD``
    env var, read raw — not a Settings field, see ``prepare()``). Anything
    else is a logged no-op — eager, never a stall.

    ``guard=True`` (consumer): a failing compiled call permanently unwraps to
    eager. ``guard=False`` (compile job): failures must raise, a silently
    eager warm-up would publish an empty artifact as success.
    """
    if getattr(pipeline, _MARKER_ATTR, None) is not None:
        return True
    try:
        import torch
    except Exception:
        return False
    if not torch.cuda.is_available():
        logger.info("compile-cache: no CUDA; staying eager")
        return False
    if not cache_ready:
        if allow_cold is None:
            allow_cold = os.environ.get(ENV_ALLOW_COLD, "").strip().lower() in ("1", "true", "yes")
        if not allow_cold:
            logger.info("compile-cache: no verified cache artifact; staying eager")
            return False
        if not toolchain_present():
            logger.warning(
                "compile-cache: %s set but no C compiler on PATH; staying eager",
                ENV_ALLOW_COLD,
            )
            return False

    # Dynamo's per-code-object recompile limit defaults to 8; a preset table
    # bigger than that (LTX: 12 video graphs, ie#381) would silently fall
    # back to eager for every shape past the limit. Size it to the declared
    # shape set — never lower an operator-raised value.
    try:
        import torch._dynamo

        want = len(tuple(cfg.shapes)) + 8
        torch._dynamo.config.cache_size_limit = max(
            int(torch._dynamo.config.cache_size_limit), want)
        if hasattr(torch._dynamo.config, "recompile_limit"):
            torch._dynamo.config.recompile_limit = max(
                int(torch._dynamo.config.recompile_limit), want)
    except Exception:
        logger.debug("compile-cache: could not raise recompile limit", exc_info=True)

    regional = bool(getattr(cfg, "regional", False))
    applied: list[str] = []
    originals: list[Tuple[Any, str, Callable[..., Any]]] = []
    regional_mods: list[Any] = []
    for target in cfg.targets:
        resolved = _resolve_target(pipeline, target)
        if resolved is None:
            logger.debug("compile-cache: pipeline has no target %r; skipping", target)
            continue
        owner, attr, fn = resolved
        if (
            regional
            and attr == "forward"
            and callable(getattr(owner, "compile_repeated_blocks", None))
        ):
            # Per-block graphs (ie#381): bounded memory under fp8 layerwise
            # casting + much cheaper cold compile. Blocks are compiled in
            # place; the guard wrapper clears them on the first failure.
            owner.compile_repeated_blocks(dynamic=False)
            if guard:
                setattr(owner, attr, _guarded_regional(owner, fn, target))
                originals.append((owner, attr, fn))
            regional_mods.append(owner)
            applied.append(target)
            continue
        if regional:
            logger.info(
                "compile-cache: %r has no compile_repeated_blocks; "
                "whole-forward compile for it", target)
        if target.startswith("vae"):
            # channels_last + compiled decode is the measured win combo (#382);
            # memory format changes strides, so it is part of the cache key —
            # producer and consumer both come through here.
            vae = getattr(pipeline, "vae", None)
            if vae is not None:
                vae.to(memory_format=torch.channels_last)
        compiled = torch.compile(fn, dynamic=False)
        setattr(owner, attr, _guarded(fn, compiled, target) if guard else compiled)
        applied.append(target)
        originals.append((owner, attr, fn))
    if not applied:
        return False
    setattr(pipeline, _MARKER_ATTR, {
        "targets": applied,
        "shapes": [tuple(s) for s in cfg.shapes],
        "cache": bool(cache_ready),
        "originals": originals,
        "regional_mods": regional_mods,
    })
    logger.info(
        "compile-cache: torch.compile armed for %s (cache=%s regional=%s)",
        applied, cache_ready, regional)
    return True


def unwrap(pipeline: Any) -> bool:
    """Restore the eager callables :func:`apply` wrapped and drop dynamo's
    in-memory compiled code so a later :func:`apply` re-traces against the
    then-seeded caches. Used on adoption rollback (zero cache hits => back to
    true eager, gw#391) and before re-adoption of a re-published cell."""
    marker = getattr(pipeline, _MARKER_ATTR, None)
    if marker is None:
        return False
    for owner, attr, fn in marker.get("originals") or ():
        try:
            setattr(owner, attr, fn)
        except Exception:
            logger.warning("compile-cache: could not restore eager %s.%s", type(owner).__name__, attr)
    for mod in marker.get("regional_mods") or ():
        try:
            _clear_regional(mod)
        except Exception:
            logger.warning("compile-cache: could not clear regional compile on %s", type(mod).__name__)
    try:
        delattr(pipeline, _MARKER_ATTR)
    except AttributeError:
        setattr(pipeline, _MARKER_ATTR, None)
    try:
        import torch._dynamo

        torch._dynamo.reset()
    except Exception:
        pass
    return True


def enable(
    pipeline: Any,
    cfg: Any,
    cache_dir: Optional[Path] = None,
    artifact: Optional[Path] = None,
) -> bool:
    """The one consumer entry point (executor + local CLI): seed a verified
    artifact if one is configured or attached, then arm compile under the
    safety policy."""
    meta = prepare(
        getattr(cfg, "family", "") or "", cache_dir=cache_dir, artifact=artifact
    )
    if meta is not None:
        drift = mode_drift(meta, pipeline) or lane_drift(meta, pipeline)
        if drift:
            logger.warning("compile-cache: %s; staying eager", drift)
            meta = None
    if meta is not None:
        want = "regional" if getattr(cfg, "regional", False) else "whole"
        have = str(meta.get("compile_mode") or "whole")
        if have != want:
            logger.warning(
                "compile-cache: cell compile_mode %r != declared %r; staying "
                "eager (graphs would miss)", have, want)
            meta = None
    return apply(pipeline, cfg, cache_ready=meta is not None)


# ---------------------------------------------------------------------------
# Build (the compile job / conversion producer)
# ---------------------------------------------------------------------------


def _warm_call(
    pipe: Any,
    shape: Tuple[int, ...],
    *,
    steps: int,
    prompt: str,
    decode: bool,
) -> None:
    """One warm-up call for ``shape``. (w, h) is the classic image call;
    (w, h, frames) is a video call (ie#381): the DiT graph keys on the token
    count only, so a plain single-pipeline call traces the same graph the
    serving path (including a two-stage refine, whose latents arrive from an
    upsampler of identical shape) will look up. Video calls force the
    batch-1 no-CFG serving regime (CFG is a graph shape — ``Compile``) and
    skip decode unless a vae target is declared."""
    import inspect

    import torch

    kwargs: Dict[str, Any] = dict(
        prompt=prompt,
        num_inference_steps=int(steps),
        width=int(shape[0]),
        height=int(shape[1]),
        generator=torch.Generator(device="cuda").manual_seed(0),
    )
    if len(shape) == 3:
        params = inspect.signature(type(pipe).__call__).parameters
        kwargs["num_frames"] = int(shape[2])
        kwargs["output_type"] = "np" if decode else "latent"
        if "frame_rate" in params:
            kwargs["frame_rate"] = 24.0
        if "guidance_scale" in params:
            kwargs["guidance_scale"] = 1.0
        if "audio_guidance_scale" in params:
            kwargs["audio_guidance_scale"] = 1.0
    pipe(**kwargs)


def build(
    model_path: str | Path,
    out_dir: str | Path,
    *,
    shapes: Iterable[Tuple[int, ...]],
    targets: Iterable[str] = ("transformer", "vae.decode"),
    family: str = "",
    source_ref: str = "",
    source_digest: str = "",
    dtype: str = "bf16",
    storage_dtype: str = "",
    regional: bool = False,
    steps: int = 2,
    prompt: str = "cache warm-up: a lighthouse on a cliff at dawn, detailed",
) -> Tuple[Path, Dict[str, Any], Dict[str, float]]:
    """Compile a diffusers pipeline over ``shapes`` and package the resulting
    inductor+triton caches as a per-SKU artifact.

    ``storage_dtype`` mirrors the serving binding's weight-storage lane
    (gw#389 fp8 layerwise casting): the cast hooks are traced INTO the FX
    graphs, so a cell for an fp8-served model must be built from an
    fp8-loaded pipeline or every request misses the cache (ie#381).

    Runs on the TARGET GPU SKU with a C toolchain present (cold compile).
    Returns ``(artifact_path, metadata, per-shape warm-up seconds)`` — the
    first call per shape is the compile cost. Raises on any compile failure
    or an empty capture; a silently-eager build must never publish.
    """
    from .api.decorators import Compile as CompileCfg
    from .models.loading import load_from_pretrained
    from .models.memory import place_pipeline

    if not toolchain_present():
        raise RuntimeError(
            "compile-cache build needs a C toolchain (cc/gcc); run in the "
            "compile-job image, not a prod worker image"
        )
    out_dir = Path(out_dir)
    capture_root = out_dir / "capture"
    capture_env(capture_root)

    import torch
    from diffusers import DiffusionPipeline

    if not torch.cuda.is_available():
        raise RuntimeError("compile-cache build requires CUDA")

    cfg = CompileCfg(shapes=tuple(shapes), targets=tuple(targets), regional=bool(regional))
    pipe = load_from_pretrained(
        DiffusionPipeline, str(model_path), dtype=dtype,
        storage_dtype=storage_dtype)
    # Producer/consumer graph parity (gw#391): the worker prepares pipelines
    # with place_pipeline (placement + vae/attention low-VRAM flags), and
    # those flags are traced INTO the graphs — the FX-graph cache key. A cell
    # built from a differently-prepared pipeline misses at request time, so
    # the producer must come through the exact same prep, and the mode it
    # traced under travels in the metadata for adopt-time parity checks. Run
    # on a pod with the same free-VRAM class as the target workers.
    placed = place_pipeline(pipe)
    if callable(getattr(pipe, "set_progress_bar_config", None)):
        pipe.set_progress_bar_config(disable=True)
    # Child processes (inductor compile workers) inherit the opt-in; this
    # process opts in explicitly via the `allow_cold` param.
    os.environ[ENV_ALLOW_COLD] = "1"
    if not apply(pipe, cfg, cache_ready=False, guard=False, allow_cold=True):
        raise RuntimeError(f"no compile targets resolved on {type(pipe).__name__}")

    timings: Dict[str, float] = {}
    decode = any(t.startswith("vae") for t in cfg.targets)
    for shape in cfg.shapes:
        torch.cuda.synchronize()
        t = time.monotonic()
        _warm_call(pipe, shape, steps=int(steps), prompt=prompt, decode=decode)
        torch.cuda.synchronize()
        key = "x".join(str(v) for v in shape)
        timings[key] = round(time.monotonic() - t, 2)
        logger.info("compile-cache build: warmed %s in %.1fs", key, timings[key])

    captured = [p for p in (capture_root / "inductor").rglob("*") if p.is_file()]
    if not captured:
        raise RuntimeError(
            "compile warm-up captured nothing under TORCHINDUCTOR_CACHE_DIR — "
            "was inductor already latched to another dir in this process?"
        )

    from .models.loading import pipeline_weight_lane

    meta = artifact_metadata(
        family=family, source_ref=source_ref, source_digest=source_digest,
        shapes=cfg.shapes, targets=cfg.targets,
        low_vram_mode=str(placed.get("mode") or ""),
        storage_dtype=storage_dtype,
        compile_mode="regional" if regional else "whole",
        # gw#534: the lane the pipeline ACTUALLY traced under — the loader may
        # have upgraded a requested fp8 cast to bf16-resident on this pod.
        weight_lane=pipeline_weight_lane(pipe),
    )
    label = flavor_label(meta["sku"], meta["torch"], meta.get("weight_lane", ""))
    artifact = pack(capture_root, out_dir / f"{label}.tar.gz", meta)
    return artifact, meta, timings


__all__ = [
    "ARTIFACT_FORMAT",
    "AdoptError",
    "build",
    "ENV_ALLOW_COLD",
    "ENV_CACHE_PATH",
    "ENV_CACHE_URL",
    "apply",
    "artifact_metadata",
    "capture_env",
    "counters_delta",
    "enable",
    "family_from_ref",
    "parse_cell_ref",
    "find_artifact",
    "flavor_label",
    "gen_worker_version",
    "inductor_counters",
    "is_cache_ref",
    "lane_token",
    "lane_drift",
    "mode_drift",
    "pack",
    "prepare",
    "runtime_key",
    "seed_artifact",
    "seed_env",
    "sku_slug",
    "system_repo",
    "toolchain_present",
    "unpack",
    "unwrap",
    "verify",
]
