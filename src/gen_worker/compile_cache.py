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
(cubin/launcher cache key), diffusers (the traced graph is its code).
``source_ref`` records which family member the producer compiled from —
informational, never part of the match.
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
ARTIFACT_FORMAT = 1
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


def flavor_label(sku: str, torch_version: str) -> str:
    """Repo-flavor label for an artifact: ``inductor-rtx-4090-torch2.9``.
    The full versions live in metadata; the label is for humans + selection."""
    short = ".".join(str(torch_version).split("+")[0].split(".")[:2])
    return f"inductor-{sku}-torch{short}"


def system_repo(family: str) -> str:
    """The system-owned repo holding one family's compiled-artifact cells."""
    fam = str(family or "").strip()
    if not fam:
        raise ValueError("compile-cache family must be non-empty")
    return f"_system/family-{fam}"


def family_from_ref(ref: str) -> str:
    """Family encoded in a compile-cache ref
    (``_system/family-<f>[:tag][@digest][#inductor-...]``); '' when the ref
    is not a compile-cache ref."""
    repo = str(ref or "").split("#", 1)[0].split("@", 1)[0].split(":", 1)[0]
    owner, _, name = repo.partition("/")
    if owner == "_system" and name.startswith("family-"):
        return name[len("family-"):]
    return ""


def is_cache_ref(ref: str, family: str = "") -> bool:
    """True when ``ref`` names an inductor compile-cache cell (optionally of
    one specific family)."""
    fam = family_from_ref(ref)
    if not fam or (family and fam != family):
        return False
    flavor = ref.split("#", 1)[1] if "#" in ref else ""
    return flavor.startswith("inductor-")


def artifact_metadata(
    *,
    family: str,
    source_ref: str = "",
    source_digest: str = "",
    shapes: Iterable[Tuple[int, int]] = (),
    targets: Iterable[str] = (),
) -> Dict[str, Any]:
    """Producer-side metadata for :func:`pack` (no timestamps: artifacts of
    identical content must be byte-identical). ``source_ref``/``source_digest``
    record the family member compiled from — informational only."""
    return {
        "format": ARTIFACT_FORMAT,
        "kind": "torch-inductor-cache",
        **runtime_key(),
        "family": str(family or ""),
        "source_ref": str(source_ref or ""),
        "source_digest": str(source_digest or ""),
        "shapes": [[int(w), int(h)] for w, h in shapes],
        "targets": list(targets),
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
    """Point inductor+triton at fresh dirs under ``root`` for a compile job.
    Must run before this process first touches torch.compile / triton."""
    root = Path(root)
    for sub, env in (("inductor", "TORCHINDUCTOR_CACHE_DIR"), ("triton", "TRITON_CACHE_DIR")):
        d = root / sub
        d.mkdir(parents=True, exist_ok=True)
        os.environ[env] = str(d)
    return root


# seeding reuses the same env contract
seed_env = capture_env


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
    """
    local = (os.getenv(ENV_CACHE_PATH) or "").strip()
    url = (os.getenv(ENV_CACHE_URL) or "").strip()
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


def apply(pipeline: Any, cfg: Any, *, cache_ready: bool, guard: bool = True) -> bool:
    """Wrap ``cfg.targets`` on ``pipeline`` with compiled callables.

    Only compiles when a verified cache artifact was seeded (``cache_ready``)
    or the process explicitly opted into cold compilation AND has a C
    toolchain. Anything else is a logged no-op — eager, never a stall.

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
        allow_cold = (os.getenv(ENV_ALLOW_COLD) or "").strip().lower() in ("1", "true", "yes")
        if not allow_cold:
            logger.info("compile-cache: no verified cache artifact; staying eager")
            return False
        if not toolchain_present():
            logger.warning(
                "compile-cache: %s set but no C compiler on PATH; staying eager",
                ENV_ALLOW_COLD,
            )
            return False

    applied: list[str] = []
    for target in cfg.targets:
        resolved = _resolve_target(pipeline, target)
        if resolved is None:
            logger.debug("compile-cache: pipeline has no target %r; skipping", target)
            continue
        owner, attr, fn = resolved
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
    if not applied:
        return False
    setattr(pipeline, _MARKER_ATTR, {
        "targets": applied,
        "shapes": [tuple(s) for s in cfg.shapes],
        "cache": bool(cache_ready),
    })
    logger.info("compile-cache: torch.compile armed for %s (cache=%s)", applied, cache_ready)
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
    return apply(pipeline, cfg, cache_ready=meta is not None)


# ---------------------------------------------------------------------------
# Build (the compile job / conversion producer)
# ---------------------------------------------------------------------------


def build(
    model_path: str | Path,
    out_dir: str | Path,
    *,
    shapes: Iterable[Tuple[int, int]],
    targets: Iterable[str] = ("transformer", "vae.decode"),
    family: str = "",
    source_ref: str = "",
    source_digest: str = "",
    dtype: str = "bf16",
    steps: int = 2,
    prompt: str = "cache warm-up: a lighthouse on a cliff at dawn, detailed",
) -> Tuple[Path, Dict[str, Any], Dict[str, float]]:
    """Compile a diffusers pipeline over ``shapes`` and package the resulting
    inductor+triton caches as a per-SKU artifact.

    Runs on the TARGET GPU SKU with a C toolchain present (cold compile).
    Returns ``(artifact_path, metadata, per-shape warm-up seconds)`` — the
    first call per shape is the compile cost. Raises on any compile failure
    or an empty capture; a silently-eager build must never publish.
    """
    from .api.decorators import Compile as CompileCfg
    from .models.loading import load_from_pretrained

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

    cfg = CompileCfg(shapes=tuple(shapes), targets=tuple(targets))
    pipe = load_from_pretrained(DiffusionPipeline, str(model_path), dtype=dtype)
    pipe.to("cuda")
    if callable(getattr(pipe, "set_progress_bar_config", None)):
        pipe.set_progress_bar_config(disable=True)
    os.environ[ENV_ALLOW_COLD] = "1"
    if not apply(pipe, cfg, cache_ready=False, guard=False):
        raise RuntimeError(f"no compile targets resolved on {type(pipe).__name__}")

    timings: Dict[str, float] = {}
    for w, h in cfg.shapes:
        torch.cuda.synchronize()
        t = time.monotonic()
        pipe(
            prompt=prompt,
            num_inference_steps=int(steps),
            width=int(w),
            height=int(h),
            generator=torch.Generator(device="cuda").manual_seed(0),
        )
        torch.cuda.synchronize()
        timings[f"{w}x{h}"] = round(time.monotonic() - t, 2)
        logger.info("compile-cache build: warmed %dx%d in %.1fs", w, h, timings[f"{w}x{h}"])

    captured = [p for p in (capture_root / "inductor").rglob("*") if p.is_file()]
    if not captured:
        raise RuntimeError(
            "compile warm-up captured nothing under TORCHINDUCTOR_CACHE_DIR — "
            "was inductor already latched to another dir in this process?"
        )

    meta = artifact_metadata(
        family=family, source_ref=source_ref, source_digest=source_digest,
        shapes=cfg.shapes, targets=cfg.targets,
    )
    label = flavor_label(meta["sku"], meta["torch"])
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
    "enable",
    "family_from_ref",
    "find_artifact",
    "flavor_label",
    "is_cache_ref",
    "pack",
    "prepare",
    "runtime_key",
    "seed_artifact",
    "seed_env",
    "sku_slug",
    "system_repo",
    "toolchain_present",
    "unpack",
    "verify",
]
