"""Per-SKU torch.compile cache artifacts (#384).

torch.compile wins 15-34% warm latency on flux-class models but costs 20-46s
of compile per (model, resolution) and needs a C toolchain the prod worker
images don't ship. The split: a platform compile job (training-endpoints
``produce-inductor-cache``) compiles once per GPU SKU and publishes the
inductor+triton cache dirs as a repo flavor; workers that opt in via
``@endpoint(compile=Compile(...))`` seed those dirs before load and hit the
cache with no compiler and no stall.

Policy: cache miss / key mismatch / no artifact leaves ordinary lanes eager,
never causing a boot stall or a runtime compile attempt in prod. A declared
W8A8 lane instead fails retryably: eager/dequantized execution cannot claim
W8A8. The compile job itself opts into cold compilation through the explicit
``allow_cold=True`` library argument (requires a toolchain).

Artifacts are FAMILY-keyed (settled 2026-07-06): torch.compile caches key on
the traced graph + shapes, not the weights, so one artifact serves every
fine-tune of a model family. They live in a system-owned repo per family
(``_system/family-<family>``), one flavor per (SKU, torch) cell — and they
are CODE: only the platform's first-party compile job publishes shared ones.

Artifact = deterministic ``.tar.gz``::

    metadata.json      key: family, sku/torch/triton/cuda, shapes, targets,
                       image guidance regimes, diffusers/transformers
                       versions (+ source_ref info)
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

import ctypes
import filecmp
import functools
import gzip
import hashlib
import io
import json
import logging
import os
import re
import shutil
import tarfile
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

from .api.errors import RetryableError

logger = logging.getLogger(__name__)

METADATA_NAME = "metadata.json"
# 2 (gw#391): key gained the producer gen-worker version. ie#496 extends its
# metadata with the canonical module graph, shape/target table and weight-lane
# schema without gratuitously invalidating proven non-W8A8 cells. New W8A8
# consumers require those fields; checkpoint bytes remain deliberately absent.
ARTIFACT_FORMAT = 2
_MARKER_ATTR = "_cozy_compile"
_JUNK_SUFFIXES = (".lock", ".tmp", ".pid")
# Cache directories and torch's in-process cache latches are process-global.
# Serialize the complete seed+arm transaction so another setup can never arm
# against a half-merged artifact. RLock keeps prepare -> seed_artifact and
# seed_artifact -> capture_env composable without another configuration layer.
_SEED_ARM_LOCK = threading.RLock()
_LOCK_TYPE = type(threading.Lock())


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


def _cuda_driver_version() -> str:
    """CUDA driver API version without shelling out to provider tooling."""
    try:
        lib = ctypes.CDLL("libcuda.so.1")
        value = ctypes.c_int()
        if lib.cuInit(0) != 0 or lib.cuDriverGetVersion(ctypes.byref(value)) != 0:
            return ""
        return str(int(value.value))
    except Exception:
        return ""


def runtime_key() -> Dict[str, str]:
    """The consumer-side half of the cache key, probed from this process."""
    key = {
        "sku": "", "sm": "", "torch": "", "triton": "", "cuda": "",
        "cuda_driver": "", "image_digest": os.environ.get(
            "WORKER_IMAGE_DIGEST", "").strip(),
    }
    try:
        import torch

        key["torch"] = str(torch.__version__)
        key["cuda"] = str(torch.version.cuda or "")
        if torch.cuda.is_available():
            key["sku"] = sku_slug(torch.cuda.get_device_name(0))
            major, minor = torch.cuda.get_device_capability(0)
            key["sm"] = f"sm_{major}{minor}"
            # CUDA's integer encoding (e.g. 13000), obtained from libcuda
            # rather than provider-specific nvidia-smi output.
            key["cuda_driver"] = _cuda_driver_version()
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


def lane_bucket(lane: str) -> Tuple[str, int]:
    """(base lane, rank bucket) for a weight lane in stamp OR label-token
    form: ``"w8a8-lora128"`` -> ``("w8a8", 128)``, ``"lora32"`` -> ``("", 32)``,
    ``"w8a8"`` -> ``("w8a8", 0)``. Sparse stamps (eager-only, never cells)
    do not parse as bucketed — they pass through as their whole string."""
    m = re.search(r"(?:^|-)lora(\d+)$", str(lane or ""))
    if m is None:
        return str(lane or ""), 0
    base = lane[: m.start()]
    return base, int(m.group(1))


def lane_token(weight_lane: str) -> str:
    """Label token for a traced weight lane (gw#534): cells of different
    lanes are DIFFERENT graphs and must not collide on one flavor label.
    "" (plain resident, incl. bf16-resident) stays unsuffixed. LoRA-branch
    lanes (gw#547/gw#561) keep their base lane's token + the bucket suffix:
    ``w8a8-lora128`` -> ``w8a8-lora128``, ``fp8-hooks-lora32`` ->
    ``w8a16-lora32``, ``lora32`` -> ``lora32`` — one graph family per
    (base lane, rank bucket)."""
    base, bucket = lane_bucket(str(weight_lane or ""))
    tok = {"": "", "fp8-hooks": "w8a16", "w8a8": "w8a8",
           "w4a4": "w4a4"}.get(base, base)
    if bucket:
        return f"{tok}-lora{bucket}" if tok else f"lora{bucket}"
    return tok


def compile_target_lane_error(weight_lane: str, lora_bucket: int) -> str:
    """Return why a worker compile-target lane is not wire-canonical.

    This is the Python half of Tensorhub's compile-target descriptor contract:
    the worker reports the *raw pipeline lane* (Tensorhub maps ``fp8-hooks`` to
    the ``w8a16`` cell token), with an optional exact canonical LoRA suffix.
    Keeping this vocabulary explicit prevents a test or future loader from
    advertising a target the scheduler must reject.
    """
    lane = str(weight_lane or "")
    declared = int(lora_bucket or 0)
    base, observed = lane_bucket(lane)
    if base not in ("", "fp8-hooks", "w8a16", "w8a8", "w4a4"):
        return f"unsupported pipeline_weight_lane {lane!r}"
    from .models.w8a8_lora import RANK_BUCKETS

    if observed not in (0, *RANK_BUCKETS):
        return f"unsupported LoRA bucket {observed} in lane {lane!r}"
    canonical = f"{base}-lora{observed}" if base and observed else (
        f"lora{observed}" if observed else base
    )
    if lane != canonical:
        return f"non-canonical pipeline_weight_lane {lane!r}; expected {canonical!r}"
    if observed != declared:
        return (
            f"pipeline lane LoRA bucket {observed} != declared "
            f"Compile.lora_bucket {declared}"
        )
    return ""


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


def cell_lane(ref: str) -> str:
    """The compiled weight-lane token encoded in a system-cell ref.

    The flavor is human/routing metadata; artifact metadata remains the
    authority. This narrow parser exists so a worker presented several cells
    for one family tries the exact lane instead of whichever mapping entry
    happened to arrive first (ie#496).
    """
    _family, flavor = parse_cell_ref(ref)
    _prefix, sep, suffix = flavor.partition("-torch")
    if not sep:
        return ""
    _version, sep, lane = suffix.partition("-")
    return lane if sep else ""


def family_from_ref(ref: str) -> str:
    """Family encoded in a compile-cache ref; '' when the ref is not a
    system-family cell ref."""
    return parse_cell_ref(ref)[0]


def is_cache_ref(ref: str, family: str = "") -> bool:
    """True when ``ref`` names an inductor compile-cache cell (optionally of
    one specific family). Cells are flavored either with the legacy human
    label (``inductor-<sku>-torch<mm>[-lane]``) or, post-th#883, with the
    worker-computed cell key itself (``ck1-<sha256>`` — pull-by-key)."""
    from . import cell_key

    fam, flavor = parse_cell_ref(ref)
    if not fam or (family and fam != family):
        return False
    return flavor.startswith("inductor-") or cell_key.is_key(flavor)


def artifact_metadata(
    *,
    family: str,
    source_ref: str = "",
    source_digest: str = "",
    shapes: Iterable[Tuple[int, ...]] = (),
    targets: Iterable[str] = (),
    guidance_scales: Iterable[float] = (),
    low_vram_mode: str = "",
    storage_dtype: str = "",
    compile_mode: str = "whole",
    weight_lane: str = "",
    lora_bucket: int = 0,
    graph_signature: str = "",
    weight_contract: Optional[Dict[str, Any]] = None,
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
    (w, h) or (w, h, frames); ``guidance_scales`` records the image CFG /
    no-CFG graph regimes captured for every 2-D row — see ``Compile``."""
    meta: Dict[str, Any] = {
        "format": ARTIFACT_FORMAT,
        "kind": "torch-inductor-cache",
        **runtime_key(),
        "gen_worker": gen_worker_version(),
        "family": str(family or ""),
        "source_ref": str(source_ref or ""),
        "source_digest": str(source_digest or ""),
        "shapes": [[int(v) for v in s] for s in shapes],
        "targets": list(targets),
        "guidance_scales": [float(v) for v in guidance_scales],
        "low_vram_mode": str(low_vram_mode or ""),
        "storage_dtype": str(storage_dtype or ""),
        "compile_mode": str(compile_mode or "whole"),
        "weight_lane": str(weight_lane or ""),
        "lora_bucket": int(lora_bucket or 0),
        "graph_signature": str(graph_signature or ""),
        "weight_contract": dict(weight_contract or {}),
        "libs": _lib_versions(),
    }
    # gw#581/th#883: stamp the worker-owned cell key the recorded axes
    # describe. Derived FROM the metadata (never probed separately), so the
    # stamp can never disagree with the axes it summarizes. Callers that
    # later override a key axis (build()'s serving image digest) re-stamp.
    from . import cell_key

    return cell_key.stamp(meta)


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
    # Extended runtime axes are exact when a cell records them. Old proven
    # non-W8A8 format-2 cells remain usable; W8A8 requires every field in
    # contract_drift below and therefore never gets this compatibility path.
    # cuda_driver is deliberately NOT here (gw#577): inductor/triton artifacts
    # are keyed by torch/triton/cuda-runtime/SM-arch — triton's disk cache key
    # is (source, ptxas_version-from-the-wheel, sm arch, options); the host
    # libcuda build never enters any compiled-artifact key. Pinning it made
    # cell delivery a host lottery across same-image same-SKU pods. The driver
    # stays recorded in metadata for observability only.
    for field in ("sm", "cuda", "image_digest"):
        want, have = str(meta.get(field) or ""), here[field]
        if want and want != have:
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
            name = member.name
            if name == METADATA_NAME:
                if not member.isfile():
                    raise ValueError(
                        f"unsafe {METADATA_NAME} member in compile-cache artifact"
                    )
                f = tar.extractfile(member)
                meta = json.loads(f.read().decode()) if f else {}
                continue
            posix = PurePosixPath(name)
            parts = posix.parts
            if (
                not member.isfile()
                or not parts
                or parts[0] not in ("inductor", "triton")
                or any(part in ("", ".", "..") for part in parts)
                or posix.is_absolute()
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


class CellSelectionBugError(RuntimeError):
    """A SELF-REQUESTED, identity-verified cell failed to arm (th#883).

    Under worker-owned selection the worker never refuses a cell it asked
    for: the artifact's axes describe exactly the key this runtime computed
    for itself, so any arm failure is by construction a bug in the one
    shared selection/parity brain — never a compatibility outcome. Callers
    must surface it as the ``cell_selection_bug`` event class (loud, wire-
    visible), never as a silent eager fallback."""

    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


class CompiledLaneUnavailableError(RetryableError):
    """A precision lane whose production contract requires a cell is unsafe."""


def find_artifact(root: Path) -> Optional[Path]:
    """The compile-cache tarball inside a downloaded snapshot dir (or the
    file itself)."""
    root = Path(root)
    if root.is_file():
        return root
    return next(iter(sorted(root.rglob("*.tar.gz"))), None)


def _merge_staged_cache(staged: Path, live: Path) -> None:
    """Safely add one already-verified staging tree to ``live``.

    Inductor/Triton paths are content-addressed: an existing path must be
    byte-identical, never overwritten. New files become visible one at a time
    via ``os.replace`` (there is no portable whole-directory union swap), but
    the process lock prevents normal arming consumers from observing that
    interval. An in-process failure removes every newly added file. A process
    crash can leave only complete, verified new files; replay treats those as
    identical and finishes the additive merge.
    """
    files = sorted(
        path
        for sub in ("inductor", "triton")
        for path in (staged / sub).rglob("*")
        if path.is_file()
    )
    additions: list[tuple[Path, Path]] = []
    for source in files:
        target = live / source.relative_to(staged)
        if target.exists():
            if not target.is_file() or not filecmp.cmp(source, target, shallow=False):
                raise AdoptError(
                    "cache_collision",
                    f"verified cache path {source.relative_to(staged)!s} "
                    "already exists with different bytes",
                )
            continue
        additions.append((source, target))

    live.mkdir(parents=True, exist_ok=True)
    added: list[Path] = []
    try:
        for source, target in additions:
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(source, target)
            added.append(target)
    except BaseException:
        for target in reversed(added):
            target.unlink(missing_ok=True)
        raise


@dataclass
class _StagedArtifact:
    metadata: Dict[str, Any]
    staged_root: Path
    live_root: Path
    temporary: tempfile.TemporaryDirectory[str]
    activated: bool = False

    def close(self) -> None:
        self.temporary.cleanup()


def stage_artifact(
    artifact: Path, family: str, cache_dir: Optional[Path] = None,
) -> _StagedArtifact:
    """Extract and validate an artifact without touching process-global state."""
    root = (Path(cache_dir) if cache_dir else Path.home() / ".cache" / "gen-worker")
    root = root / "compile-cache"
    root.parent.mkdir(parents=True, exist_ok=True)
    temporary = tempfile.TemporaryDirectory(
        prefix="compile-cache-stage-", dir=root.parent,
    )
    staged = Path(temporary.name) / "cache"
    try:
        meta = unpack(Path(artifact), staged)
        reason = verify(meta, family=family)
        if reason:
            raise AdoptError("key_mismatch", reason)
        return _StagedArtifact(meta, staged, root, temporary)
    except AdoptError:
        temporary.cleanup()
        raise
    except Exception as exc:
        temporary.cleanup()
        raise AdoptError("artifact_invalid", str(exc)) from exc


def _activate_staged(staged: _StagedArtifact) -> Dict[str, Any]:
    """Publish a verified staging tree while holding ``_SEED_ARM_LOCK``."""
    if not staged.activated:
        _merge_staged_cache(staged.staged_root, staged.live_root)
        staged.activated = True
    seed_env(staged.live_root)
    return staged.metadata


def seed_artifact(
    artifact: Path, family: str, cache_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """Verify in isolation, then seed one artifact under the process lock.

    A malformed, unsafe, corrupt, or runtime-mismatched tar never writes into
    the live Inductor/Triton cache. Returns metadata or raises
    :class:`AdoptError` without changing the live tree.
    """
    staged = stage_artifact(artifact, family, cache_dir=cache_dir)
    try:
        try:
            with _SEED_ARM_LOCK:
                return _activate_staged(staged)
        except AdoptError:
            raise
        except Exception as exc:
            raise AdoptError("activation_failed", str(exc)) from exc
    finally:
        staged.close()


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


def apply_lora_lane(pipeline: Any, bucket: int) -> bool:
    """Put the pipeline on the branch-bearing graph family for ``bucket``
    (gw#561): canonical zeroed rank-``bucket`` branches on every
    branch-capable denoiser Linear (the gw#547 compiled-lane contract) + the
    ``<base>-lora<bucket>`` lane stamp, so :func:`lane_drift` admits exactly
    the matching lora cells. Raises when the pipeline has no branch-capable
    denoiser — a declared bucket that cannot trace must fail loud, not
    publish/adopt the wrong graph."""
    if not bucket:
        return False
    from .models import w8a8_lora

    denoiser = w8a8_lora.branch_target(pipeline)
    if denoiser is None:
        raise RuntimeError(
            "Compile.lora_bucket declared but the pipeline has no "
            "branch-capable denoiser (transformer/unet)"
        )
    w8a8_lora.enable_lora_branches(denoiser, int(bucket))
    w8a8_lora.stamp_lane(pipeline, denoiser)
    return True


def drop_lora_lane(pipeline: Any) -> None:
    """Undo :func:`apply_lora_lane`: drop the branch buffers and restore the
    branchless lane stamp (the eager rollback — canonical zeroed branches
    cost +21-32% eager, gw#547)."""
    from .models import w8a8_lora

    denoiser = w8a8_lora.branch_target(pipeline)
    if denoiser is None:
        return
    w8a8_lora.disable_lora_branches(denoiser)
    w8a8_lora.stamp_lane(pipeline, denoiser)


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


def prepare(
    family: str,
    cache_dir: Optional[Path] = None,
    artifact: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
    """Verify and seed one explicitly delivered artifact for this runtime.

    Production obtains ``artifact`` from Tensorhub's immutable RunJob/
    DesiredInstance snapshot attachment. Local tooling passes an explicit path
    or uses the explicit local-cell store; environment fallbacks deliberately
    do not participate in serving placement.
    """
    try:
        if artifact is None:
            logger.info("compile-cache: no delivered artifact; staying eager")
            return None
        artifact = Path(artifact)
        if not artifact.exists():
            logger.warning("compile-cache: attached artifact %s does not exist", artifact)
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


def has_compile_target(pipeline: Any, cfg: Any) -> bool:
    """Whether ``pipeline`` owns at least one callable declared by ``cfg``.

    A setup may inject support objects (for example SDXL's standalone VAE)
    alongside the actual pipeline. Only the object whose graph targets resolve
    is a compile-adoption target; family-wide scans must not try to wrap every
    resident model object.
    """
    return any(
        _resolve_target(pipeline, str(target)) is not None
        for target in tuple(getattr(cfg, "targets", ()) or ())
    )


def _type_name(value: Any) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _direct_tensor_schema(module: Any) -> list[list[Any]]:
    """Names/shapes/dtypes only; tensor values and checkpoint IDs stay out."""
    rows: list[list[Any]] = []
    for kind, method in (
        ("parameter", getattr(module, "named_parameters", None)),
        ("buffer", getattr(module, "named_buffers", None)),
    ):
        if not callable(method):
            continue
        try:
            tensors = method(recurse=False)
        except TypeError:
            tensors = method()
        for name, tensor in tensors:
            rows.append([
                kind, str(name), [int(v) for v in getattr(tensor, "shape", ())],
                str(getattr(tensor, "dtype", "")),
            ])
    return sorted(rows)


def execution_contract(pipeline: Any, cfg: Any) -> Tuple[str, Dict[str, Any]]:
    """Canonical family-graph and weight-lane contract for one loaded model.

    Fine-tunes with the same module graph produce the same result: no ref,
    tag, source/checkpoint digest or tensor value is read. A structural
    SDXL/Pony/Illustrious incompatibility (different module class/shape or
    different scaled-mm exclusion surface) produces a different signature
    and is rejected before adoption.

    The signature hashes ONLY the traced module structure — the resolved
    targets' module types, paths and tensor schemas — never the wrapping
    pipeline class (gw#577): torch.compile wraps target callables such as
    ``transformer.forward``; the pipeline class never enters any traced
    graph. Conversion producers load via generic ``DiffusionPipeline``
    (model_index -> e.g. LTX2Pipeline) while serving loads the endpoint's
    declared wrapper (e.g. LTX2ConditionPipeline) over a byte-identical
    module tree — proven identical-graph, and must share one cell.
    """
    graph_targets: list[Dict[str, Any]] = []
    quantized: list[Dict[str, Any]] = []
    excluded: list[Dict[str, Any]] = []
    seen_modules: set[int] = set()

    for target in tuple(getattr(cfg, "targets", ()) or ()):
        resolved = _resolve_target(pipeline, str(target))
        if resolved is None:
            graph_targets.append({"target": str(target), "missing": True})
            continue
        owner, attr, _fn = resolved
        modules: list[Dict[str, Any]] = []
        named = getattr(owner, "named_modules", None)
        module_rows = list(named()) if callable(named) else [("", owner)]
        for name, module in module_rows:
            path = f"{target}:{name}" if name else str(target)
            modules.append({
                "path": path,
                "type": _type_name(module),
                "tensors": _direct_tensor_schema(module),
            })
            # A target such as vae.decode can overlap another declaration;
            # record each module once in the W8A8 manifest.
            if id(module) in seen_modules:
                continue
            seen_modules.add(id(module))
            in_features = getattr(module, "in_features", None)
            out_features = getattr(module, "out_features", None)
            if not isinstance(in_features, int) or not isinstance(out_features, int):
                continue
            row = {
                "path": path,
                "in_features": int(in_features),
                "out_features": int(out_features),
            }
            if bool(getattr(module, "_cozy_w8a8_linear", False)):
                # gw#564: the activation-scale granularity is a graph property
                # (per-row rowwise sm_90+, per-tensor epilogue sm_89).
                if getattr(module, "input_scale", None) is not None:
                    row["activation"] = "static"
                elif getattr(module, "gemm_mode", "") == "pertensor":
                    row["activation"] = "dynamic-per-tensor"
                else:
                    row["activation"] = "dynamic-per-row"
                quantized.append(row)
            elif bool(getattr(module, "_cozy_w4a4_linear", False)):
                # gw#540: block scales are always dynamic per-16-block; the
                # graph property is the second-level activation scale mode.
                row["activation"] = (
                    "static" if getattr(module, "input_scale", None)
                    is not None else "dynamic-per-tensor")
                if getattr(module, "pre_quant_scale", None) is not None:
                    row["pre_quant_scale"] = True
                quantized.append(row)
            else:
                row["type"] = _type_name(module)
                excluded.append(row)
        graph_targets.append({
            "target": str(target), "attr": str(attr), "modules": modules,
        })

    graph = {
        "targets": graph_targets,
    }
    encoded = json.dumps(graph, sort_keys=True, separators=(",", ":")).encode()
    from .models.loading import pipeline_weight_lane

    lane = pipeline_weight_lane(pipeline)
    weight_contract: Dict[str, Any] = {"lane": lane}
    if lane.startswith(("w8a8", "w4a4")):
        activations = sorted({str(r["activation"]) for r in quantized})
        weight_contract.update({
            "artifact_schema": (
                "nvfp4-w4a4-v1" if lane.startswith("w4a4") else "fp8-w8a8-v1"),
            "operator": "torch._scaled_mm",
            "weight_scaling": (
                "per-16-block+per-tensor" if lane.startswith("w4a4")
                else "per-output-channel"),
            "activation_scaling": activations,
            "quantized": sorted(quantized, key=lambda r: str(r["path"])),
            "excluded": sorted(excluded, key=lambda r: str(r["path"])),
        })
    return hashlib.sha256(encoded).hexdigest(), weight_contract


def execution_contract_digest(pipeline: Any, cfg: Any) -> str:
    """Digest every graph-compatibility axis enforced by the consumer.

    ``execution_contract()[0]`` is intentionally only the module-graph
    signature. Scheduler fencing needs the complete contract: declared graph
    shapes/targets/CFG regimes, whole-vs-regional mode, actual weight lane and
    activation-scaling schema, LoRA bucket, and observed low-VRAM preparation.
    Tensor values and checkpoint identities remain excluded so compatible
    fine-tunes share one family cell.
    """
    graph_signature, weight_contract = execution_contract(pipeline, cfg)
    from .models.memory import low_vram_mode

    payload = {
        "version": 1,
        "family": str(getattr(cfg, "family", "") or ""),
        "shapes": sorted(
            [int(v) for v in row] for row in getattr(cfg, "shapes", ())
        ),
        "targets": [str(v) for v in getattr(cfg, "targets", ())],
        "guidance_scales": [
            float(v) for v in getattr(cfg, "guidance_scales", ())
        ],
        "compile_mode": (
            "regional" if bool(getattr(cfg, "regional", False)) else "whole"
        ),
        "lora_bucket": int(getattr(cfg, "lora_bucket", 0) or 0),
        "low_vram_mode": low_vram_mode(pipeline),
        "graph_signature": graph_signature,
        "weight_contract": weight_contract,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _first_contract_difference(cell: Dict[str, Any], here: Dict[str, Any]) -> str:
    """Name the first differing weight-contract key with compact values, so a
    refusal is diagnosable from the reason alone (gw#577)."""
    for key in sorted(set(cell) | set(here)):
        want, have = cell.get(key), here.get(key)
        if want == have:
            continue
        if isinstance(want, list) and isinstance(have, list):
            return (
                f"{key}: cell has {len(want)} row(s), consumer {len(have)}; "
                f"first cell-only rows {[r for r in want if r not in have][:2]!r} "
                f"vs consumer-only {[r for r in have if r not in want][:2]!r}"
            )
        return f"{key}: cell {want!r} != consumer {have!r}"
    return "identical keys, differing encoding"


def contract_drift(meta: Dict[str, Any], pipeline: Any, cfg: Any) -> str:
    """Mismatch between the cell's declared graph and the loaded consumer."""
    shapes = sorted(
        [int(v) for v in row] for row in getattr(cfg, "shapes", ()))
    cell_shapes = sorted(
        [int(v) for v in row] for row in (meta.get("shapes") or ()))
    if cell_shapes != shapes:
        return f"shapes {cell_shapes!r} != declared {shapes!r}"
    targets = [str(v) for v in getattr(cfg, "targets", ())]
    if meta.get("targets") != targets:
        return f"targets {meta.get('targets')!r} != declared {targets!r}"
    guidance_scales = [float(v) for v in getattr(cfg, "guidance_scales", ())]
    cell_guidance_scales = [float(v) for v in (meta.get("guidance_scales") or ())]
    if cell_guidance_scales != guidance_scales:
        return (
            f"guidance_scales {cell_guidance_scales!r} != declared "
            f"{guidance_scales!r}"
        )
    signature, weight_contract = execution_contract(pipeline, cfg)
    meta_signature = str(meta.get("graph_signature") or "")
    meta_weights = meta.get("weight_contract") or {}
    if not meta_signature and not meta_weights and not str(
        weight_contract.get("lane") or ""
    ).startswith(("w8a8", "w4a4")):
        return ""  # legacy format-2 non-quantized-lane cell
    if meta_signature != signature:
        return (
            f"module graph signature: cell {meta_signature[:12]!r} != "
            f"consumer {signature[:12]!r}"
        )
    if meta_weights != weight_contract:
        return (
            "weight-lane artifact schema/exclusion manifest mismatch: "
            + _first_contract_difference(meta_weights, weight_contract)
        )
    cell_lane_base = str(weight_contract.get("lane") or "")
    if cell_lane_base.startswith(("w8a8", "w4a4")):
        activations = weight_contract.get("activation_scaling") or []
        if cell_lane_base.startswith("w8a8"):
            # DYNAMIC only, one homogeneous granularity per graph (gw#564:
            # per-row = rowwise sm_90+, per-tensor = the sm_89 epilogue lane).
            if activations not in (["dynamic-per-row"], ["dynamic-per-tensor"]):
                return (f"W8A8 activation scaling must be dynamic "
                        f"(per-row or per-tensor), got {activations!r}")
        else:
            # gw#540: one homogeneous second-level activation scale mode per
            # graph (static = calibrated input_scale, the production mode).
            if activations not in (["static"], ["dynamic-per-tensor"]):
                return (f"W4A4 activation scaling must be homogeneous "
                        f"static or dynamic-per-tensor, got {activations!r}")
        if not weight_contract.get("quantized"):
            return (f"{cell_lane_base[:4].upper()} graph contains no "
                    "torch._scaled_mm modules")
        here_digest = runtime_key()["image_digest"]
        # cuda_driver excluded (gw#577): host-lottery axis, see verify().
        for field in ("sm", "cuda", "image_digest"):
            if not str(meta.get(field) or ""):
                if field == "image_digest" and not here_digest:
                    # Bare-metal local runtime (gw#555 self-mint): no image
                    # identity axis exists on either side. Production images
                    # always carry WORKER_IMAGE_DIGEST, so fleet cells stay
                    # fully pinned.
                    continue
                return f"quantized-lane cell missing {field} identity"
    return ""


def _guarded(
    original: Callable[..., Any], compiled: Callable[..., Any], label: str,
    *, fail_closed: bool = False,
    failure_signal: Optional[Dict[str, Any]] = None,
) -> Callable[..., Any]:
    """Guard one exact compiled callable and record its own warm-call proof.

    The process-wide Dynamo counters are sampled *inside this wrapper* around
    this object's compiled call.  Executor adoption therefore cannot use a
    cache hit produced by a different resident pipeline as proof for this one.
    """
    state: Dict[str, Any] = {
        "failed": False,
        "detail": "",
        "revocation_error": "",
    }

    def revoke(detail: str) -> None:
        callback = (failure_signal or {}).get("callback")
        if callable(callback):
            try:
                callback(detail)
            except Exception as exc:
                state["revocation_error"] = (
                    "compiled-state revocation failed: "
                    f"{type(exc).__name__}: {exc}"
                )
                logger.exception("compile-cache: %s", state["revocation_error"])
                raise CompiledLaneUnavailableError(
                    state["revocation_error"]
                ) from exc

    def proof_before() -> Optional[Dict[str, int]]:
        signal = failure_signal or {}
        lock = signal.get("lock")
        if not isinstance(lock, _LOCK_TYPE):
            return None
        # Inductor counts graph lookup/compile hits, not every execution of an
        # already-loaded graph. Capture activation once for this exact wrapper;
        # successful_calls below remains a per-invocation alias proof. Executor
        # proof warmups exclude concurrent GPU work, so the process-wide delta
        # cannot come from another resident object.
        with lock:
            if int(signal.get("cache_hits", 0)) > 0:
                return None
        return inductor_counters()

    def record_success(before: Optional[Dict[str, int]]) -> None:
        signal = failure_signal or {}
        lock = signal.get("lock")
        if not isinstance(lock, _LOCK_TYPE):
            return
        stats = counters_delta(before, inductor_counters()) if before is not None else {}
        with lock:
            signal["successful_calls"] = int(signal.get("successful_calls", 0)) + 1
            signal["cache_hits"] = int(signal.get("cache_hits", 0)) + max(
                0, int(stats.get("fxgraph_cache_hit", 0)))
            signal["cache_misses"] = int(signal.get("cache_misses", 0)) + max(
                0, int(stats.get("fxgraph_cache_miss", 0)))

    @functools.wraps(original)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if state["revocation_error"]:
            raise CompiledLaneUnavailableError(state["revocation_error"])
        if state["failed"]:
            if fail_closed:
                raise CompiledLaneUnavailableError(state["detail"])
            return original(*args, **kwargs)
        before = proof_before()
        try:
            result = compiled(*args, **kwargs)
            record_success(before)
            return result
        except Exception as exc:  # noqa: BLE001 — optional lanes may use eager
            state["failed"] = True
            state["detail"] = (
                f"compiled {'W8A8 ' if fail_closed else ''}target {label} failed: "
                f"{type(exc).__name__}: {exc}"
            )
            # Revoke scheduler-visible compiled proof synchronously before
            # either the optional eager fallback or mandatory W8A8 error.
            revoke(state["detail"])
            if fail_closed:
                logger.error("compile-cache: %s", state["detail"])
                raise CompiledLaneUnavailableError(state["detail"]) from exc
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


def _guarded_regional(
    mod: Any,
    original: Callable[..., Any],
    label: str,
    *,
    fail_closed: bool = False,
    failure_signal: Optional[Dict[str, Any]] = None,
) -> Callable[..., Any]:
    """Regional analogue of :func:`_guarded`: blocks are compiled in place,
    so eager fallback must first CLEAR the block compilations, then retry."""
    state: Dict[str, Any] = {
        "failed": False,
        "detail": "",
        "revocation_error": "",
    }

    def proof_before() -> Optional[Dict[str, int]]:
        signal = failure_signal or {}
        lock = signal.get("lock")
        if not isinstance(lock, _LOCK_TYPE):
            return None
        # See _guarded: one exact-object activation is sufficient; subsequent
        # aliases still need their own successful wrapper invocation.
        with lock:
            if int(signal.get("cache_hits", 0)) > 0:
                return None
        return inductor_counters()

    def record_success(before: Optional[Dict[str, int]]) -> None:
        signal = failure_signal or {}
        lock = signal.get("lock")
        if not isinstance(lock, _LOCK_TYPE):
            return
        stats = counters_delta(before, inductor_counters()) if before is not None else {}
        with lock:
            signal["successful_calls"] = int(signal.get("successful_calls", 0)) + 1
            signal["cache_hits"] = int(signal.get("cache_hits", 0)) + max(
                0, int(stats.get("fxgraph_cache_hit", 0)))
            signal["cache_misses"] = int(signal.get("cache_misses", 0)) + max(
                0, int(stats.get("fxgraph_cache_miss", 0)))

    @functools.wraps(original)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if state["revocation_error"]:
            raise CompiledLaneUnavailableError(state["revocation_error"])
        if not state["failed"]:
            before = proof_before()
            try:
                result = original(*args, **kwargs)
                record_success(before)
                return result
            except Exception as exc:  # noqa: BLE001 — optional lanes may use eager
                state["failed"] = True
                state["detail"] = (
                    f"regional compiled {'W8A8 ' if fail_closed else ''}"
                    f"target {label} failed: "
                    f"{type(exc).__name__}: {exc}"
                )
                # Regional eager state is real only after the in-place block
                # compilations are gone. Revoke proof after that mutation and
                # before a state delta can be scheduled.
                _clear_regional(mod)
                callback = (failure_signal or {}).get("callback")
                if callable(callback):
                    try:
                        callback(state["detail"])
                    except Exception as callback_exc:
                        state["revocation_error"] = (
                            "compiled-state revocation failed: "
                            f"{type(callback_exc).__name__}: {callback_exc}"
                        )
                        logger.exception(
                            "compile-cache: %s", state["revocation_error"])
                        raise CompiledLaneUnavailableError(
                            state["revocation_error"]
                        ) from callback_exc
                if fail_closed:
                    logger.error("compile-cache: %s", state["detail"])
                    raise CompiledLaneUnavailableError(state["detail"]) from exc
                logger.warning(
                    "compile-cache: regional-compiled %s failed (%s: %s); eager "
                    "for the rest of this process", label, type(exc).__name__, exc,
                )
        if fail_closed:
            raise CompiledLaneUnavailableError(state["detail"])
        return original(*args, **kwargs)

    return wrapper


def _vae_supports_channels_last(vae: Any) -> bool:
    """True only when every VAE weight is rank<=4 (2D convs). channels_last
    is a rank-4 memory format; rank-5 Conv3d weights (causal/video VAEs)
    raise on it (gw#574)."""
    try:
        return all(p.dim() <= 4 for p in vae.parameters())
    except Exception:
        return False


def apply(
    pipeline: Any,
    cfg: Any,
    *,
    cache_ready: bool,
    guard: bool = True,
    allow_cold: bool = False,
) -> bool:
    """Wrap ``cfg.targets`` on ``pipeline`` with compiled callables.

    Only compiles when a verified cache artifact was seeded (``cache_ready``)
    or explicit producer/local tooling passes ``allow_cold=True`` and has a C
    toolchain. Production serving never consults an environment fallback.
    Anything else is a logged no-op — eager, never a stall.

    ``guard=True`` (consumer): a failing ordinary compiled call permanently
    unwraps to eager; W8A8 fails closed. ``guard=False`` (compile job): all
    failures raise, because a silently eager warm-up would publish an empty
    artifact as success.
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
        if not allow_cold:
            logger.info("compile-cache: no verified cache artifact; staying eager")
            return False
        if not toolchain_present():
            logger.warning(
                "compile-cache: cold compile explicitly requested but no C "
                "compiler is on PATH; staying eager",
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
    from .models.loading import pipeline_weight_lane

    fail_closed = pipeline_weight_lane(pipeline).startswith(("w8a8", "w4a4"))
    failure_signal: Dict[str, Any] = {
        "callback": None,
        "lock": threading.Lock(),
        "successful_calls": 0,
        "cache_hits": 0,
        "cache_misses": 0,
    }
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
                setattr(owner, attr, _guarded_regional(
                    owner,
                    fn,
                    target,
                    fail_closed=fail_closed,
                    failure_signal=failure_signal,
                ))
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
            # producer and consumer both come through here. channels_last is a
            # RANK-4 format: causal/video VAEs (Conv3d, rank-5 weights — qwen,
            # LTX) crash on it (gw#574), so gate on the actual weight ranks.
            # The gate is deterministic per model class, so producer and
            # consumer always agree on the resulting strides.
            vae = getattr(pipeline, "vae", None)
            if vae is not None and _vae_supports_channels_last(vae):
                vae.to(memory_format=torch.channels_last)
        compiled = torch.compile(fn, dynamic=False)
        setattr(owner, attr, _guarded(
            fn,
            compiled,
            target,
            fail_closed=fail_closed,
            failure_signal=failure_signal,
        ) if guard else compiled)
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
        "failure_signal": failure_signal,
    })
    logger.info(
        "compile-cache: torch.compile armed for %s (cache=%s regional=%s)",
        applied, cache_ready, regional)
    return True


def set_guard_failure_callback(
    pipeline: Any, callback: Callable[[str], None],
) -> bool:
    """Bind scheduler-state revocation to an armed consumer guard."""
    marker = getattr(pipeline, _MARKER_ATTR, None) or {}
    signal = marker.get("failure_signal")
    if not isinstance(signal, dict):
        return False
    signal["callback"] = callback
    return True


def _proof_count(pipeline: Any, key: str) -> int:
    marker = getattr(pipeline, _MARKER_ATTR, None) or {}
    signal = marker.get("failure_signal")
    if not isinstance(signal, dict):
        return 0
    lock = signal.get("lock")
    if isinstance(lock, _LOCK_TYPE):
        with lock:
            return int(signal.get(key, 0))
    return int(signal.get(key, 0))


def execution_count(pipeline: Any) -> int:
    """Successful compiled calls observed on this exact pipeline object."""
    return _proof_count(pipeline, "successful_calls")


def cache_hit_count(pipeline: Any) -> int:
    """FX-graph cache hits observed inside this exact pipeline's guard."""
    return _proof_count(pipeline, "cache_hits")


def cache_miss_count(pipeline: Any) -> int:
    """FX-graph cache misses observed inside this exact pipeline's guard."""
    return _proof_count(pipeline, "cache_misses")


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


def _reconcile_resident_mode(meta: Optional[Dict[str, Any]], pipeline: Any) -> None:
    """gw#588: 'off' and 'vae_only' are both fully-resident preps differing
    only in flag groups — converge the pipeline to the cell's traced mode so
    an honest :func:`mode_drift` passes. Offload drift keeps refusing."""
    if not meta:
        return
    want = str(meta.get("low_vram_mode") or "")
    from .models.memory import low_vram_mode, reconcile_resident_mode

    resident = ("off", "vae_only")
    have = low_vram_mode(pipeline)
    if want != have and want in resident and have in resident:
        reconcile_resident_mode(pipeline, want)


def artifact_drift(meta: Dict[str, Any], pipeline: Any, cfg: Any) -> str:
    """The complete pipeline/config compatibility verdict for one cell."""
    drift = (
        mode_drift(meta, pipeline)
        or lane_drift(meta, pipeline)
        or contract_drift(meta, pipeline, cfg)
    )
    if drift:
        return drift
    want = "regional" if getattr(cfg, "regional", False) else "whole"
    have = str(meta.get("compile_mode") or "whole")
    if have != want:
        return f"cell compile_mode {have!r} != declared {want!r}"
    return ""


def arm_staged_artifact(
    pipeline: Any,
    cfg: Any,
    staged: _StagedArtifact,
) -> Dict[str, Any]:
    """Activate and arm an already-verified artifact under the process lock.

    This strict entry point is used by hot adoption: unlike :func:`enable`, a
    mismatch is returned as a classified :class:`AdoptError` instead of an
    eager fallback. Expensive tar extraction happened before the executor's
    model/GPU locks; the process lock covers only atomic cache activation and
    wrapper installation.
    """
    try:
        with _SEED_ARM_LOCK:
            meta = staged.metadata
            _reconcile_resident_mode(meta, pipeline)
            drift = artifact_drift(meta, pipeline, cfg)
            if drift:
                raise AdoptError("key_mismatch", drift)
            _activate_staged(staged)
            unwrap(pipeline)
            try:
                if not apply(pipeline, cfg, cache_ready=True):
                    raise AdoptError("no_target")
            except Exception:
                unwrap(pipeline)
                raise
            return meta
    finally:
        staged.close()


def enable(
    pipeline: Any,
    cfg: Any,
    cache_dir: Optional[Path] = None,
    artifact: Optional[Path] = None,
) -> bool:
    """The one consumer entry point (executor + local CLI): seed an explicitly
    attached verified artifact, then arm compile under the safety policy.

    A W8A8 refusal names its exact cause — the mismatched key axis with the
    cell-vs-runtime values, the drift verdict, or the missing delivery
    (gw#577): the raise IS the wire-visible job error, and serve pods expose
    no logs, so a generic message makes a refused cell undiagnosable."""
    staged: Optional[_StagedArtifact] = None
    refusal = "no cell artifact delivered"
    if artifact is not None:
        try:
            staged = stage_artifact(
                Path(artifact), getattr(cfg, "family", "") or "",
                cache_dir=cache_dir,
            )
        except Exception as exc:
            refusal = f"cell rejected: {exc}"
            logger.warning("compile-cache: artifact unusable (%s); staying eager", exc)
    try:
        with _SEED_ARM_LOCK:
            meta: Optional[Dict[str, Any]] = None
            self_key = ""
            if staged is not None:
                meta = staged.metadata
                # th#883/gw#581: is this MY cell — the artifact whose axes
                # describe exactly the key this runtime computes for itself
                # with the one shared brain? If so, a refusal below is by
                # construction a selection/parity bug, never compatibility.
                try:
                    from . import cell_key
                    from .models.loading import (
                        pipeline_weight_lane as _pwl,
                    )

                    want = cell_key.compute(
                        str(getattr(cfg, "family", "") or ""),
                        _pwl(pipeline),
                        int(getattr(cfg, "lora_bucket", 0) or 0),
                        regional=bool(getattr(cfg, "regional", False)),
                    )
                    if not cell_key.mismatch(meta, want):
                        self_key = want.digest
                except Exception:
                    self_key = ""
                _reconcile_resident_mode(meta, pipeline)
                drift = artifact_drift(meta, pipeline, cfg)
                if drift:
                    # low_vram prep mode is DYNAMIC (free-VRAM placement at
                    # load) and outside the key: its drift is a legitimate
                    # miss even on a self-requested cell, never the bug class.
                    if self_key and not drift.startswith("low_vram_mode"):
                        raise CellSelectionBugError(
                            f"self-requested cell {self_key} refused to "
                            f"arm: {drift}"
                        )
                    refusal = f"cell rejected: {drift}"
                    logger.warning("compile-cache: %s; staying eager", drift)
                    meta = None
                else:
                    try:
                        _activate_staged(staged)
                    except Exception as exc:
                        refusal = f"cell activation failed: {exc}"
                        logger.warning(
                            "compile-cache: cache activation failed (%s); "
                            "staying eager", exc)
                        meta = None
            armed = apply(pipeline, cfg, cache_ready=meta is not None)
            if meta is not None and not armed and self_key:
                raise CellSelectionBugError(
                    f"self-requested cell {self_key} activated but armed "
                    "no compile target"
                )
            from .models.loading import pipeline_weight_lane

            quant_lane = pipeline_weight_lane(pipeline)
            if quant_lane.startswith(("w8a8", "w4a4")) and not armed:
                if meta is not None:
                    refusal = "verified cell armed no compile target"
                lane_name = quant_lane[:4].upper()
                raise CompiledLaneUnavailableError(
                    f"{lane_name} requires an exact compatible Forge cell "
                    f"({refusal}); eager/dequantized execution is not a "
                    f"{lane_name} production lane"
                )
            return armed
    finally:
        if staged is not None:
            staged.close()


# ---------------------------------------------------------------------------
# Build (the compile job / conversion producer)
# ---------------------------------------------------------------------------


def resolve_pipeline_class(name: str) -> Any:
    """Resolve a serving pipeline class name for a mint (gw#586).

    The traced FX graphs depend on the pipeline's CALL path, not just the
    module tree — an unknown name must refuse loudly, because a silent
    generic-load fallback would trace the wrong call and publish a cell no
    serving lookup can ever hit.
    """
    import diffusers

    cleaned = str(name or "").strip()
    if not cleaned:
        raise RuntimeError("pipeline_class must be a non-empty class name")
    cls = getattr(diffusers, cleaned, None)
    if cls is None or not callable(getattr(cls, "from_pretrained", None)):
        raise RuntimeError(
            f"pipeline_class {cleaned!r} is not a loadable diffusers "
            "pipeline class in this producer image; a generic-load fallback "
            "would trace the wrong call path (gw#586), so the mint refuses"
        )
    return cls


def _warm_call(
    pipe: Any,
    shape: Tuple[int, ...],
    *,
    steps: int,
    prompt: str,
    decode: bool,
    guidance_scales: Iterable[float] = (),
) -> None:
    """One warm-up call for ``shape``. (w, h) is the classic image call;
    (w, h, frames) is a video call (ie#381): the DiT graph keys on the token
    count only, so a plain single-pipeline call traces the same graph the
    serving path (including a two-stage refine, whose latents arrive from an
    upsampler of identical shape) will look up. Video calls force the
    batch-1 no-CFG serving regime (CFG is a graph shape — ``Compile``) and
    skip decode unless a vae target is declared. Image calls run once per
    explicitly declared guidance scale, capturing CFG batch-2 and no-CFG
    batch-1 graphs in one family cell."""
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
        return

    scales = tuple(float(v) for v in guidance_scales)
    if not scales:
        pipe(**kwargs)
        return
    params = inspect.signature(type(pipe).__call__).parameters
    accepts_guidance = "guidance_scale" in params or any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
    )
    if not accepts_guidance:
        raise RuntimeError(
            f"{type(pipe).__name__} cannot warm declared guidance_scales; "
            "its call signature has no guidance_scale"
        )
    for scale in scales:
        pipe(**kwargs, guidance_scale=scale)


def build(
    model_path: str | Path,
    out_dir: str | Path,
    *,
    shapes: Iterable[Tuple[int, ...]],
    targets: Iterable[str] = ("transformer", "vae.decode"),
    guidance_scales: Iterable[float] = (),
    family: str = "",
    source_ref: str = "",
    source_digest: str = "",
    dtype: str = "bf16",
    storage_dtype: str = "",
    regional: bool = False,
    steps: int = 2,
    prompt: str = "cache warm-up: a lighthouse on a cliff at dawn, detailed",
    declared_vram_gb: float = 0.0,
    serving_image_digest: str = "",
    lora_bucket: int = 0,
    requested_cell_key: str = "",
    pipeline_class: str = "",
) -> Tuple[Path, Dict[str, Any], Dict[str, float]]:
    """Compile a diffusers pipeline over ``shapes`` and package the resulting
    inductor+triton caches as a per-SKU artifact.

    ``storage_dtype`` mirrors the serving binding's weight-storage lane
    (gw#389 fp8 layerwise casting): the cast hooks are traced INTO the FX
    graphs, so a cell for an fp8-served model must be built from an
    fp8-loaded pipeline or every request misses the cache (ie#381).

    ``pipeline_class`` (gw#586) names the diffusers pipeline class the
    SERVING endpoint declares (e.g. ``"LTX2ConditionPipeline"``). The traced
    FX graphs depend on the pipeline's CALL path, not just the module tree:
    LTX2ConditionPipeline drives the DiT with PER-TOKEN timestep/modulation
    tensors even for a plain unconditioned call, while the generic
    model_index class broadcasts them — structurally different graphs, so a
    cell minted through the generic load can never serve the serving path's
    lookups (found live: warmups=1, cache_hits=0). The gw#577
    ``graph_signature`` remains class-agnostic (same module tree) — this is
    call-path parity, not module identity. Unknown class names refuse
    loudly: a silent generic fallback would re-open the exact parity gap.

    Runs on the TARGET GPU SKU with a C toolchain present (cold compile).
    Returns ``(artifact_path, metadata, per-shape warm-up seconds)`` — the
    first call per shape is the compile cost. Raises on any compile failure
    or an empty capture; a silently-eager build must never publish.
    """
    from .api.decorators import Compile as CompileCfg
    from .models.loading import load_from_pretrained
    from .models.memory import place_pipeline

    _W8A8_MINT_NEEDS_DIGEST = (
        "W8A8 cell mint requires serving_image_digest (the endpoint "
        "serving image's immutable OCI digest); a cell stamped with the "
        "producer image identity can never be adopted by the fleet"
    )
    if (("w8a8" in str(storage_dtype) or "w4a4" in str(storage_dtype))
            and not str(serving_image_digest).strip()):
        # gw#577 finding (b): contract_drift requires image_digest identity on
        # W8A8 cells and verify() pins it exactly. Without the SERVING digest
        # the artifact stamps the PRODUCER pod's WORKER_IMAGE_DIGEST — every
        # serving worker then rejects it and W8A8 serves NOTHING
        # (fail-closed). Refuse loudly before any load or compile.
        raise RuntimeError(_W8A8_MINT_NEEDS_DIGEST)
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

    cfg = CompileCfg(
        shapes=tuple(shapes), targets=tuple(targets),
        guidance_scales=tuple(guidance_scales), regional=bool(regional),
    )
    load_cls: Any = DiffusionPipeline
    if str(pipeline_class or "").strip():
        # gw#586 call-path parity: trace through the SERVING pipeline class.
        load_cls = resolve_pipeline_class(str(pipeline_class))
    pipe = load_from_pretrained(
        load_cls, str(model_path), dtype=dtype,
        storage_dtype=storage_dtype,
        # Producer/consumer LANE parity (ie#381): the serving worker decides
        # the bf16-resident upgrade against its function's declared envelope;
        # the producer must decide with the same input or it traces the
        # other lane and the cell never adopts.
        declared_vram_gb=declared_vram_gb)
    # Producer/consumer graph parity (gw#391): the worker prepares pipelines
    # with place_pipeline (placement + vae/attention low-VRAM flags), and
    # those flags are traced INTO the graphs — the FX-graph cache key. A cell
    # built from a differently-prepared pipeline misses at request time, so
    # the producer must come through the exact same prep, and the mode it
    # traced under travels in the metadata for adopt-time parity checks. Run
    # on a pod with the same free-VRAM class as the target workers.
    placed = place_pipeline(pipe)
    from .models.loading import pipeline_weight_lane as _traced_lane

    if _traced_lane(pipe).startswith(("w8a8", "w4a4")) and not str(
        serving_image_digest
    ).strip():
        # The lane can materialize as w8a8/w4a4 from the SOURCE flavor alone
        # (e.g. storage_dtype="fp8+te" over an fp8-w8a8 checkpoint), so the
        # authoritative check is on the traced lane, before the compile.
        raise RuntimeError(_W8A8_MINT_NEEDS_DIGEST)
    # gw#561: branch-bearing cells trace WITH canonical zeroed rank-bucket
    # branches installed — zeroed slots are bit-exact with branchless output
    # (gw#547), so the warm calls render normally while the traced graphs
    # carry the branch GEMMs.
    apply_lora_lane(pipe, int(lora_bucket))
    if callable(getattr(pipe, "set_progress_bar_config", None)):
        pipe.set_progress_bar_config(disable=True)
    # Cold compilation is an explicit producer-library operation; serving
    # workers have no environment switch that can enter this path.
    if not apply(pipe, cfg, cache_ready=False, guard=False, allow_cold=True):
        raise RuntimeError(f"no compile targets resolved on {type(pipe).__name__}")

    timings: Dict[str, float] = {}
    decode = any(t.startswith("vae") for t in cfg.targets)
    for shape in cfg.shapes:
        torch.cuda.synchronize()
        t = time.monotonic()
        _warm_call(
            pipe, shape, steps=int(steps), prompt=prompt, decode=decode,
            guidance_scales=cfg.guidance_scales,
        )
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

    graph_signature, weight_contract = execution_contract(pipe, cfg)
    meta = artifact_metadata(
        family=family, source_ref=source_ref, source_digest=source_digest,
        shapes=cfg.shapes, targets=cfg.targets,
        guidance_scales=cfg.guidance_scales,
        low_vram_mode=str(placed.get("mode") or ""),
        storage_dtype=storage_dtype,
        compile_mode="regional" if regional else "whole",
        # gw#534: the lane the pipeline ACTUALLY traced under — the loader may
        # have upgraded a requested fp8 cast to bf16-resident on this pod.
        weight_lane=pipeline_weight_lane(pipe),
        lora_bucket=int(lora_bucket or 0),
        graph_signature=graph_signature,
        weight_contract=weight_contract,
    )
    if serving_image_digest:
        # The producer image contains a compiler; the graph is consumed by
        # the endpoint's serving image. Tensorhub supplies that immutable OCI
        # digest from the release, so it—not the producer container—is the
        # identity the worker must match.
        meta["image_digest"] = str(serving_image_digest).strip()
    if str(pipeline_class or "").strip():
        # gw#586 observability only — NOT a key axis (graph_signature and the
        # ck1 key stay class-agnostic; the class shapes the traced CALL, and
        # a wrong class shows up as serving cache misses, which the warmup
        # proof refuses loudly).
        meta["pipeline_class"] = str(pipeline_class).strip()
    # gw#581/th#883: re-stamp the key over the final axes, then honor the
    # forge's echo — a demand-driven mint names the exact worker-computed
    # key it must satisfy, and publishing a cell under a key its own axes
    # do not describe would be a permanently un-armable store entry.
    from . import cell_key

    cell_key.stamp(meta)
    if str(requested_cell_key or "").strip():
        reason = cell_key.mismatch(meta, str(requested_cell_key).strip())
        if reason:
            try:
                axes = cell_key.from_artifact_metadata(meta).canonical()
            except cell_key.CellKeyError:
                axes = "<not key-complete>"
            raise RuntimeError(
                "cell mint does not satisfy the requested cell key "
                f"({reason}); producer axes: {axes}"
            )
    label = flavor_label(meta["sku"], meta["torch"], meta.get("weight_lane", ""))
    artifact = pack(capture_root, out_dir / f"{label}.tar.gz", meta)
    return artifact, meta, timings


def _compile_and_warm(pipe: Any, cfg: Any, *, steps: int = 2, say: Any = None) -> None:
    """Cold-compile ``pipe`` over the declared shape table (the only part of
    a mint that needs CUDA + a toolchain). ``guard=False``: a failing warm
    call must fail the mint — a silently-eager capture must never be saved."""
    _say = say if callable(say) else (lambda msg: logger.info("%s", msg))
    if not apply(pipe, cfg, cache_ready=False, guard=False, allow_cold=True):
        raise RuntimeError(f"no compile targets resolved on {type(pipe).__name__}")
    import torch

    decode = any(t.startswith("vae") for t in cfg.targets)
    for shape in cfg.shapes:
        torch.cuda.synchronize()
        t0 = time.monotonic()
        _warm_call(
            pipe, shape, steps=steps,
            prompt="cache warm-up: a lighthouse on a cliff at dawn, detailed",
            decode=decode,
            guidance_scales=getattr(cfg, "guidance_scales", ()),
        )
        torch.cuda.synchronize()
        shape_key = "x".join(str(v) for v in shape)
        _say(f"  compiled {shape_key} in {time.monotonic() - t0:.0f}s")


def mint_artifact(
    pipe: Any,
    cfg: Any,
    family: str,
    target: Path,
    capture: Path,
    *,
    steps: int = 2,
    say: Any = None,
) -> Dict[str, Any]:
    """Self-mint (gw#555/gw#587): compile THIS pipeline over its declared
    shape table, capture the inductor/triton output, and pack the production
    artifact atomically at ``target``. Returns the stamped metadata (incl.
    the cell key its axes describe).

    The capture uses the production artifact recipe end to end
    (``capture_env`` -> warm the shape table -> ``artifact_metadata`` ->
    deterministic ``pack``), so the saved cell is byte-compatible with a
    delivered one and adopts through the identical code path. Shared by the
    cozy-local store mint and the fleet self-mint
    (fleet_cells) — ONE mint brain, different publish sinks.

    ``guard=False`` on the warm calls: a failing warm call must fail the
    mint — a silently-eager capture must never be saved.
    """
    _say = say if callable(say) else (lambda msg: logger.info("%s", msg))
    capture_env(capture)
    _compile_and_warm(pipe, cfg, steps=steps, say=_say)

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
    graph_signature, weight_contract = execution_contract(pipe, cfg)
    meta = artifact_metadata(
        family=family,
        source_ref="self-mint",
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
    pack(capture, tmp, meta)
    os.replace(tmp, target)
    return meta


__all__ = [
    "ARTIFACT_FORMAT",
    "AdoptError",
    "CellSelectionBugError",
    "CompiledLaneUnavailableError",
    "build",
    "apply",
    "apply_lora_lane",
    "artifact_metadata",
    "capture_env",
    "drop_lora_lane",
    "cell_lane",
    "contract_drift",
    "counters_delta",
    "cache_hit_count",
    "cache_miss_count",
    "enable",
    "execution_count",
    "execution_contract",
    "execution_contract_digest",
    "family_from_ref",
    "parse_cell_ref",
    "find_artifact",
    "flavor_label",
    "gen_worker_version",
    "has_compile_target",
    "inductor_counters",
    "is_cache_ref",
    "lane_bucket",
    "lane_token",
    "lane_drift",
    "mint_artifact",
    "mode_drift",
    "pack",
    "prepare",
    "resolve_pipeline_class",
    "runtime_key",
    "seed_artifact",
    "seed_env",
    "set_guard_failure_callback",
    "sku_slug",
    "system_repo",
    "toolchain_present",
    "unpack",
    "unwrap",
    "verify",
]
