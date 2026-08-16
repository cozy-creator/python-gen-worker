"""SVDQuant 4-bit loader mode.

A ``#svdq-fp4-*`` / ``#svdq-int4-*`` flavor is a normal diffusers tree whose
denoiser directory holds ONE nunchaku-FORMAT single-file checkpoint instead of
plain safetensors shards. The file self-describes in its safetensors
``__metadata__``: ``model_class`` names the transformer class and
``quantization_config`` carries ``{"method": "svdquant", "weight": {"dtype":
"fp4_e2m1_all" | "int4"}, "rank": N}``. Loading swaps the decoded denoiser
into the standard pipeline — same pipeline class, same handler, no new
endpoint.

"nunchaku" here is a FORMAT LINEAGE, not a runtime. pgw#1298 deleted the
nunchaku engine: it was slower than our own decoder (843 vs 785 ms/step on a
5090, same seeds, equal peak VRAM), covered strictly less silicon (sm_120/121
against native's sm_100/103/120/121), and was UNREACHABLE — the only caller
(``models/loading.py``) never passed an engine pin, and for fp4 the ladder put
native first on every card nunchaku could serve. Every svdq load now runs
through :mod:`gen_worker.models.svdq_native`, which reads the same bytes
bit-exactly.

ONE fleet image still installs the nunchaku WHEEL — ``inference-endpoints/
qwen-image-svdq-bench`` (a benchmark fixture pending teardown). That is
deliberate and must stay until th#2055: the hub admits an svdq-fp4 row only
when the worker reports ``nunchaku`` in ``installed_libs``
(``models/hub_policy.py`` probes it; tensorhub ``precision/ladder.go``'s
``admitted()`` requires it), so removing the wheel — or dropping the probe —
would stop that endpoint BINDING a flavor it can serve perfectly well on
native. The wheel is now purely an admission token.

fp4 serves; int4 is a typed refusal (:class:`SvdqInt4Unsupported`) — the
native module is nvfp4 block-scaled ``_scaled_mm`` and int4 svdq is a
different (single-level, group-64) scale path. Nothing produces int4
(``convert/svdq_produce.py`` pins ``nvfp4``), so this is the honest spelling
of a refusal that already happened."""

from __future__ import annotations

import json
import logging
import struct
from dataclasses import dataclass
from pathlib import Path
from ..component_vocab import denoiser_components
from .safetensors_header import header_len_ok
from typing import Any, Optional

logger = logging.getLogger(__name__)

SVDQ_METHOD = "svdquant"
# The SM windows of nunchaku's kernels. NOT a serving constraint any more —
# nothing in this module reads them, because the native engine's own window
# (`svdq_native.SVDQ_NATIVE_FP4_SMS`, a strict superset for fp4) is what
# decides whether a load can run. They survive for exactly two consumers, both
# of which die together:
#   - `models/ladder.py:99-101`, the published placement stamp
#   - tensorhub's `internal/wirecontract/WORKER_PARSED_CONSTANTS`, which pins
#     these two symbol names to this file by hand
# Both go with th#2055 + pgw#1300 (the placement-reader/producer cut). Deleting
# them here would break that stamp and falsify the hub's peer pin, so they stay
# until that coordinated cut, and not one line longer.
SVDQ_FP4_SMS = (120, 121)
SVDQ_INT4_SMS = (75, 80, 86, 89)


class SvdqError(RuntimeError):
    """Base class for typed svdq loader-mode failures."""


class SvdqHardwareError(SvdqError):
    """The artifact's precision has no kernels on this GPU."""


class SvdqInt4Unsupported(SvdqError):
    """svdq-int4: no native engine, and no other engine is installed."""


SVDQ_INT4_REFUSAL = (
    "{path} is svdq-int4. The native engine implements the nvfp4 "
    "block-scaled path only; int4 svdq is a different (single-level, "
    "group-64) scale path with no native implementation, so loading it would "
    "require the nunchaku runtime this worker deliberately does not use "
    "(pgw#1298 deleted that engine). Serve the fp4 artifact of this family, "
    "or file int4 native support as a pgw# follow-up."
)


# ---------------------------------------------------------------------------
# Artifact detection — safetensors __metadata__ sniff
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SvdqArtifact:
    component: str      # denoiser dir name inside the tree ("transformer"/"unet")
    file: Path          # the nunchaku single-file checkpoint
    model_class: str    # e.g. "NunchakuZImageTransformer2DModel"
    precision: str      # "fp4" | "int4"
    rank: int


def _read_safetensors_metadata(path: Path) -> dict:
    try:
        with open(path, "rb") as f:
            raw = f.read(8)
            if len(raw) < 8:
                return {}
            (n,) = struct.unpack("<Q", raw)
            if not header_len_ok(n):
                return {}
            header = json.loads(f.read(n))
    except (OSError, ValueError):
        return {}
    meta = header.get("__metadata__") if isinstance(header, dict) else None
    return meta if isinstance(meta, dict) else {}


def _svdq_from_file(component: str, path: Path) -> Optional[SvdqArtifact]:
    meta = _read_safetensors_metadata(path)
    model_class = str(meta.get("model_class") or "")
    qc_raw = meta.get("quantization_config")
    try:
        qc = json.loads(qc_raw) if isinstance(qc_raw, str) else (qc_raw or {})
    except ValueError:
        qc = {}
    if not isinstance(qc, dict) or str(qc.get("method") or "") != SVDQ_METHOD:
        return None
    if not model_class.startswith("Nunchaku"):
        return None
    weight_dtype = str((qc.get("weight") or {}).get("dtype") or "").lower()
    if "fp4" in weight_dtype:
        precision = "fp4"
    elif "int4" in weight_dtype:
        precision = "int4"
    else:
        logger.warning("svdq artifact %s has unknown weight dtype %r", path, weight_dtype)
        return None
    return SvdqArtifact(
        component=component,
        file=path,
        model_class=model_class,
        precision=precision,
        rank=int(qc.get("rank") or 0),
    )


def detect_svdq_artifact(model_path: Path) -> Optional[SvdqArtifact]:
    """Find the nunchaku single-file checkpoint inside a snapshot: the
    denoiser dir's (or, for a bare artifact, the root's) sole svdq-tagged
    safetensors. Cheap — header reads only."""
    root = Path(model_path)
    if root.is_file():
        return _svdq_from_file("", root) if root.suffix == ".safetensors" else None
    if not root.is_dir():
        return None
    for comp in denoiser_components():
        comp_dir = root / comp
        if not comp_dir.is_dir():
            continue
        for f in sorted(comp_dir.glob("*.safetensors")):
            art = _svdq_from_file(comp, f)
            if art is not None:
                return art
    for f in sorted(root.glob("*.safetensors")):
        art = _svdq_from_file("", f)
        if art is not None:
            return art
    return None


# ---------------------------------------------------------------------------
# Loading — the native decoder wired into the standard pipeline
# ---------------------------------------------------------------------------

def check_svdq_servable(art: SvdqArtifact, path: Any = "") -> None:
    """Every typed refusal for ``art``, raised on the DETECTED artifact.

    Called before a single weight byte is read, so an unservable flavor is
    refused by name rather than dying inside a load. int4 is refused on the
    artifact's own precision — no probing, no environment, no engine ladder,
    because there is nothing to probe: the refusal is a property of the
    checkpoint."""
    if str(art.precision) == "int4":
        raise SvdqInt4Unsupported(
            SVDQ_INT4_REFUSAL.format(path=path or art.file))
    # Deferred: svdq_native adds +2 modules to the `import gen_worker` path.
    from .svdq_native import svdq_native_reason

    reason = svdq_native_reason()
    if reason is not None:
        raise SvdqHardwareError(
            f"cannot serve svdq-{art.precision} here — {reason}")


def load_svdq_pipeline(cls: Any, path: Path, art: SvdqArtifact) -> Any:
    """Serve an svdq artifact on the native engine.

    The single entry point the loading layer uses. There is one engine, so
    there is nothing to select: the artifact is either servable here or it is
    a typed refusal."""
    check_svdq_servable(art, path)
    from .svdq_native import load_svdq_native_pipeline

    logger.info("svdq: native engine (%s %s r%d, file %s)",
                art.precision, art.component, art.rank, art.file.name)
    return load_svdq_native_pipeline(cls, path, art)


__all__ = [
    "SVDQ_FP4_SMS",
    "SVDQ_INT4_REFUSAL",
    "SVDQ_INT4_SMS",
    "SvdqArtifact",
    "SvdqError",
    "SvdqHardwareError",
    "SvdqInt4Unsupported",
    "check_svdq_servable",
    "detect_svdq_artifact",
    "load_svdq_pipeline",
]
