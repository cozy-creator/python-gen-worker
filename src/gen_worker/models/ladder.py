"""Precision-ladder spec (th#697) — flavor precision classes + placement requirements.

A flavor's *precision class* names its quantization lane (``fp8``,
``svdq-int4``, ...). A :class:`Placement` states which silicon can run it:
a discrete SM allow-list (fail-closed — kernel wheels are per-arch), an
open-ended SM floor, and the importable engine libraries the lane needs.

Produced flavors carry their placement in ``checkpoints.metadata["placement"]``
(stamped at publish by :func:`gen_worker.convert.publish.publish_flavors`).
Unstamped/mirrored rows fall back to the token-derived defaults here — the
same defaults the stamping writes, so both paths agree. The ladder walk
itself (rung ordering per arch class) lands with the shared Go+Py vector
spec; this module is the classification + placement half both sides share.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional

from .svdq import SVDQ_FP4_SMS, SVDQ_INT4_SMS

logger = logging.getLogger(__name__)

CLASS_BASE = "base"  # bare bf16/fp16/fp32 row — runs anywhere a card fits it
CLASS_FP8 = "fp8"  # fp8-E4M3 storage; universal (bf16-upcast path needs no fp8 silicon)
CLASS_SVDQ_FP4 = "svdq-fp4"  # nunchaku SVDQuant fp4 — consumer Blackwell only
CLASS_SVDQ_INT4 = "svdq-int4"  # nunchaku SVDQuant int4 — sm_75-89
CLASS_NVFP4 = "nvfp4"  # plain nvfp4 artifact — Blackwell datacenter, TRT lane (not a diffusers rung)

_BASE_TOKENS = ("", "bf16", "fp16", "fp32")


@dataclass(frozen=True)
class Placement:
    """Arch requirements for one flavor. Empty fields = unconstrained."""

    precision_class: str
    sm_allowed: tuple[int, ...] = ()  # discrete allow-list (gpu_sm as int, e.g. 89, 120)
    sm_min: int = 0  # open-ended floor; 0 = none
    engines: tuple[str, ...] = ()  # importable libraries required to load

    def admits_sm(self, gpu_sm: int) -> bool:
        if self.sm_allowed and gpu_sm not in self.sm_allowed:
            return False
        if self.sm_min and gpu_sm < self.sm_min:
            return False
        return True


def classify_flavor_token(flavor: str) -> str:
    """Flavor token -> precision class; "" when unrecognized (gguf/trt/etc.
    stay opaque — never ladder rungs)."""
    token = str(flavor or "").strip().lower()
    if token in _BASE_TOKENS:
        return CLASS_BASE
    if token.startswith("svdq-fp4"):
        return CLASS_SVDQ_FP4
    if token.startswith("svdq-int4"):
        return CLASS_SVDQ_INT4
    if token == "fp8" or token.startswith("fp8-"):
        return CLASS_FP8
    if token == "nvfp4" or token.startswith("nvfp4-"):
        return CLASS_NVFP4
    return ""


def default_placement(precision_class: str) -> Optional[Placement]:
    """Token-derived placement defaults — the fallback for unstamped rows
    and the source the publish-time stamp writes."""
    if precision_class == CLASS_BASE:
        return Placement(CLASS_BASE)
    if precision_class == CLASS_FP8:
        return Placement(CLASS_FP8)  # fp8-storage serves on any silicon
    if precision_class == CLASS_SVDQ_FP4:
        return Placement(CLASS_SVDQ_FP4, sm_allowed=tuple(SVDQ_FP4_SMS), engines=("nunchaku",))
    if precision_class == CLASS_SVDQ_INT4:
        return Placement(CLASS_SVDQ_INT4, sm_allowed=tuple(SVDQ_INT4_SMS), engines=("nunchaku",))
    if precision_class == CLASS_NVFP4:
        return Placement(CLASS_NVFP4, sm_min=100)
    return None


def placement_for_flavor(flavor: str) -> Optional[Placement]:
    return default_placement(classify_flavor_token(flavor))


def placement_to_metadata(p: Placement) -> dict[str, Any]:
    """The ``checkpoints.metadata["placement"]`` wire/storage shape."""
    out: dict[str, Any] = {"precision_class": p.precision_class}
    if p.sm_allowed:
        out["sm_allowed"] = list(p.sm_allowed)
    if p.sm_min:
        out["sm_min"] = p.sm_min
    if p.engines:
        out["engines"] = list(p.engines)
    return out


def placement_from_metadata(meta: Mapping[str, Any] | None) -> Optional[Placement]:
    """Parse a checkpoint metadata mapping (the whole bag or the placement
    block itself). Unknown keys ignored; malformed values fail soft (None)."""
    if not isinstance(meta, Mapping):
        return None
    block = meta.get("placement", meta)
    if not isinstance(block, Mapping):
        return None
    cls = str(block.get("precision_class", "") or "").strip().lower()
    if not cls:
        return None
    try:
        sm_allowed = tuple(int(v) for v in (block.get("sm_allowed") or ()))
        sm_min = int(block.get("sm_min") or 0)
    except (TypeError, ValueError):
        return None
    engines = tuple(
        s for s in (str(e).strip() for e in (block.get("engines") or ())) if s
    )
    return Placement(cls, sm_allowed=sm_allowed, sm_min=sm_min, engines=engines)


# ---------------------------------------------------------------------------
# The ladder walk (th#697 P2) — shared spec with tensorhub's
# internal/orchestrator/precision package. Both implementations must pass the
# byte-identical tests/testdata/precision_ladder_vectors.json. Any semantic
# change edits the vector file in BOTH repos.
#
# Preference per arch class (Paul's fp8 ruling): best admitted 4-bit svdq
# flavor (fp4 > int4, rank desc) first, then SM-aware fp8/bf16 ordering:
#   SM >= 89 (Ada/Hopper/Blackwell, fp8 tensor cores): fp8 (stored artifact >
#     cast-at-load) > bf16 base — fp8 is faster AND smaller there.
#   SM < 89 (no fp8 compute; storage upcasts to bf16 at the same speed):
#     bf16 base > fp8 — fp8 stays a FIT FALLBACK when bf16 won't fit
#     (compact storage still halves resident weights on any card).
# fp8 is NEVER a refusal: storage is universal (loading.apply_fp8_storage —
# fp8 bytes resident, per-layer bf16 upcast, no fp8 silicon required).
# Rung gates: quality floor, placement SM admission, engines within installed
# libs, est VRAM <= free. local=True appends emergency-nf4 then CPU-offload
# (the hub never schedules those).
# ---------------------------------------------------------------------------

# Native fp8 tensor-core compute exists on SM >= 89 (sm_89 Ada, sm_90 Hopper,
# sm_100+/120 Blackwell). Below that, fp8 storage still SERVES (bf16-upcast
# path) — this floor gates only the fp8-over-bf16 PREFERENCE, never admission.
FP8_COMPUTE_MIN_SM = 89

CAST_FP8_VRAM_FACTOR = 0.75  # pipeline-level est, gw#389 measured 62-76%
EMERGENCY_NF4_VRAM_FACTOR = 0.45  # nf4 denoiser, encoders/VAE at compute dtype

MODE_NATIVE = "native"
MODE_EMERGENCY = "emergency"
MODE_OFFLOAD = "offload"

REFUSE_NO_GPU = "no_cuda_gpu"
REFUSE_NO_RUNG = "no_runnable_precision"

# quality rank: higher = closer to baseline quality
_QUALITY = {CLASS_BASE: 3, CLASS_FP8: 2, CLASS_SVDQ_FP4: 1, CLASS_SVDQ_INT4: 1}
_FLOOR_RANK = {"": 0, "4bit": 1, "fp8": 2, "bf16": 3}

_RANK_RE = re.compile(r"-r(\d+)$")


@dataclass(frozen=True)
class FlavorRow:
    """One catalog flavor row for the model being resolved."""

    token: str
    size_gb: float = 0.0
    placement: Optional[Placement] = None  # stamped metadata; None -> token defaults


@dataclass(frozen=True)
class LadderModel:
    """The catalog view of one bound model."""

    base_size_gb: float = 0.0  # 0 = no bare/base row
    fp8_cast_vram_gb: float = 0.0  # 0 = use CAST_FP8_VRAM_FACTOR * base
    flavors: tuple[FlavorRow, ...] = ()


@dataclass(frozen=True)
class Resolution:
    flavor: str = ""  # stored-flavor token to download ("" = bare/base ref)
    cast: str = ""  # load-time transform: "" | "fp8" | "nf4"
    mode: str = MODE_NATIVE
    refusal: str = ""  # non-empty = no runnable rung (flavor/cast/mode unset)


def _placement_of(row: FlavorRow) -> Optional[Placement]:
    return row.placement or placement_for_flavor(row.token)


def _admitted(row: FlavorRow, gpu_sm: int, libs: frozenset[str]) -> bool:
    p = _placement_of(row)
    if p is None:
        return False
    return p.admits_sm(gpu_sm) and all(e in libs for e in p.engines)


def _rank_of(token: str) -> int:
    m = _RANK_RE.search(token)
    return int(m.group(1)) if m else 0


def _quality_ok(precision_class: str, quality_floor: str) -> bool:
    return _QUALITY.get(precision_class, 0) >= _FLOOR_RANK.get(quality_floor, 0)


def _four_bit_candidates(
    model: LadderModel, gpu_sm: int, libs: frozenset[str]
) -> list[FlavorRow]:
    rows = [
        r
        for r in model.flavors
        if classify_flavor_token(r.token) in (CLASS_SVDQ_FP4, CLASS_SVDQ_INT4)
        and _admitted(r, gpu_sm, libs)
    ]
    rows.sort(
        key=lambda r: (
            classify_flavor_token(r.token) != CLASS_SVDQ_FP4,  # fp4 first
            -_rank_of(r.token),
            r.token,
        )
    )
    return rows


def _stored_fp8(model: LadderModel, gpu_sm: int, libs: frozenset[str]) -> Optional[FlavorRow]:
    rows = [
        r
        for r in model.flavors
        if classify_flavor_token(r.token) == CLASS_FP8 and _admitted(r, gpu_sm, libs)
    ]
    rows.sort(key=lambda r: (len(r.token), r.token))
    return rows[0] if rows else None


def _rungs(
    model: LadderModel, gpu_sm: int, libs: frozenset[str], quality_floor: str
) -> list[tuple[str, str, float]]:
    """Eligible native rungs in preference order (ignoring VRAM fit):
    (flavor, cast, est_vram_gb). fp8 vs bf16 order is SM-aware: fp8 first on
    fp8-compute silicon (SM>=89, faster AND smaller), bf16 first below it
    (fp8 upcasts there — same speed, so fp8 is only the fit fallback)."""
    out: list[tuple[str, str, float]] = []
    if _quality_ok(CLASS_SVDQ_FP4, quality_floor):
        for row in _four_bit_candidates(model, gpu_sm, libs):
            out.append((row.token, "", row.size_gb))
    fp8_rung: Optional[tuple[str, str, float]] = None
    if _quality_ok(CLASS_FP8, quality_floor):
        stored = _stored_fp8(model, gpu_sm, libs)
        if stored is not None:
            fp8_rung = (stored.token, "", stored.size_gb)
        elif model.base_size_gb > 0:
            est = model.fp8_cast_vram_gb or CAST_FP8_VRAM_FACTOR * model.base_size_gb
            fp8_rung = ("", "fp8", est)
    base_rung: Optional[tuple[str, str, float]] = None
    if model.base_size_gb > 0 and _quality_ok(CLASS_BASE, quality_floor):
        base_rung = ("", "", model.base_size_gb)
    pair = (fp8_rung, base_rung) if gpu_sm >= FP8_COMPUTE_MIN_SM else (base_rung, fp8_rung)
    out.extend(r for r in pair if r is not None)
    return out


def resolve(
    model: LadderModel,
    *,
    gpu_sm: int,
    free_vram_gb: float,
    libs: Iterable[str] = (),
    quality_floor: str = "",
    local: bool = False,
    allow_offload: bool = True,
) -> Resolution:
    """Walk the precision ladder for ONE model on ONE card. The shared-spec
    entry point — tensorhub's Go resolver implements the identical walk."""
    if gpu_sm <= 0:
        return Resolution(refusal=REFUSE_NO_GPU)
    lib_set = frozenset(str(lib).strip() for lib in libs if str(lib).strip())
    rungs = _rungs(model, gpu_sm, lib_set, quality_floor)
    for flavor, cast, est in rungs:
        if est <= free_vram_gb:
            return Resolution(flavor=flavor, cast=cast, mode=MODE_NATIVE)
    if local:
        if quality_floor == "" and model.base_size_gb > 0:
            est = EMERGENCY_NF4_VRAM_FACTOR * model.base_size_gb
            if est <= free_vram_gb:
                stored = _stored_fp8(model, gpu_sm, lib_set)
                return Resolution(
                    flavor=stored.token if stored else "",
                    cast="nf4",
                    mode=MODE_EMERGENCY,
                )
        if allow_offload and rungs:
            flavor, cast, _ = rungs[0]
            return Resolution(flavor=flavor, cast=cast, mode=MODE_OFFLOAD)
    return Resolution(refusal=REFUSE_NO_RUNG)


# ---------------------------------------------------------------------------
# Local (no-orchestrator) resolution — cozy-local's half of th#697. The hub
# path delivers picks via HelloAck; this walks the SAME ladder against the
# hub catalog's sibling-flavor rows and rebinds bare tensorhub bindings to
# the best NATIVE rung. Emergency-nf4 / offload stay the loading layer's
# fit machinery (gw#389) — a refusal here just keeps the declared binding.
# ---------------------------------------------------------------------------


def ladder_model_from_resolved(rr: Any) -> LadderModel:
    """Build a LadderModel from a hub_client.WorkerResolvedRepo."""
    rows = []
    for sib in getattr(rr, "sibling_flavors", None) or ():
        token = str(getattr(sib, "flavor", "") or "").strip()
        if not token:
            continue
        rows.append(FlavorRow(
            token=token,
            size_gb=float(getattr(sib, "size_bytes", 0) or 0) / 1e9,
            placement=placement_from_metadata(getattr(sib, "placement", None)),
        ))
    return LadderModel(
        base_size_gb=float(getattr(rr, "size_bytes", 0) or 0) / 1e9,
        flavors=tuple(rows),
    )


def resolve_local_bindings(
    bindings: Mapping[str, Any],
    *,
    caps: Any,
    free_vram_gb: float,
    resolver: Any,
    quality_floor: str = "",
) -> dict[str, Any]:
    """Rebind bare tensorhub bindings to this card's best native rung.

    ``resolver(thref) -> WorkerResolvedRepo`` (hub_client.resolve_repo shape).
    Only bindings with no author-pinned flavor/storage_dtype/digest are
    laddered; any failure keeps the declared binding (fail-open — the
    loading layer's fit ladder still applies at load time).
    """
    import dataclasses

    from .refs import parse_model_ref

    out: dict[str, Any] = {}
    for name, binding in bindings.items():
        out[name] = binding
        if getattr(binding, "provider", "") != "tensorhub":
            continue
        if getattr(binding, "flavor", "") or getattr(binding, "storage_dtype", ""):
            continue  # author override — never laddered
        try:
            from ..api.binding import wire_ref

            base_ref = wire_ref(binding)
            thref = parse_model_ref(base_ref).tensorhub
            if thref is None or thref.digest:
                continue
            model = ladder_model_from_resolved(resolver(thref))
        except Exception as exc:  # resolver/network failures fail open
            logger.debug("local ladder: resolve of %s failed (%s); keeping declared", name, exc)
            continue
        res = resolve(
            model,
            gpu_sm=int(getattr(caps, "gpu_sm", 0) or 0),
            free_vram_gb=free_vram_gb,
            libs=tuple(getattr(caps, "installed_libs", ()) or ()),
            quality_floor=quality_floor,
        )
        if res.refusal or (not res.flavor and not res.cast):
            continue
        rebound = binding
        if res.flavor:
            rebound = dataclasses.replace(rebound, flavor=res.flavor)
        if res.cast:
            rebound = dataclasses.replace(rebound, storage_dtype=res.cast)
        logger.info(
            "local precision ladder: %s %s -> flavor=%s cast=%s (sm%d, %.1f GB free)",
            name, base_ref, res.flavor or "-", res.cast or "-",
            int(getattr(caps, "gpu_sm", 0) or 0), free_vram_gb,
        )
        out[name] = rebound
    return out


__all__ = [
    "CAST_FP8_VRAM_FACTOR",
    "FP8_COMPUTE_MIN_SM",
    "CLASS_BASE",
    "CLASS_FP8",
    "CLASS_NVFP4",
    "CLASS_SVDQ_FP4",
    "CLASS_SVDQ_INT4",
    "EMERGENCY_NF4_VRAM_FACTOR",
    "FlavorRow",
    "LadderModel",
    "MODE_EMERGENCY",
    "MODE_NATIVE",
    "MODE_OFFLOAD",
    "Placement",
    "REFUSE_NO_GPU",
    "REFUSE_NO_RUNG",
    "Resolution",
    "classify_flavor_token",
    "ladder_model_from_resolved",
    "resolve_local_bindings",
    "default_placement",
    "placement_for_flavor",
    "placement_from_metadata",
    "placement_to_metadata",
    "resolve",
]
