"""The POSTURE a request was served under, typed; vocabulary wire-shared with tensorhub's measurement relation. This module deliberately does NOT digest: posture identity is the hub's measurement.Posture.Digest() alone — a worker-side digest would be a second canonicalization free to disagree, and the byte-identical vector corpus (tests/testdata/posture_wire_vectors.json) keeps the two sides honest. Rules: `applied` is ORDERED and never collapsed; `wanted` and `applied` stay two fields on every axis; magnitudes are data, never identity."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .pb import worker_scheduler_pb2 as pb

logger = logging.getLogger(__name__)

BACKEND_FA3 = "fa3"
BACKEND_FA2 = "fa2"
BACKEND_SDPA = "sdpa"
BACKEND_XFORMERS = "xformers"
BACKEND_EAGER = "eager"

ATTENTION_BACKENDS: frozenset[str] = frozenset({
    BACKEND_FA3, BACKEND_FA2, BACKEND_SDPA, BACKEND_XFORMERS, BACKEND_EAGER,
})

_BACKEND_ALIASES: Dict[str, str] = {
    "flash": BACKEND_FA2,
    "flash_attn": BACKEND_FA2,
    "flash_attention": BACKEND_FA2,
    "flash_attn_2": BACKEND_FA2,
    "flash_attention_2": BACKEND_FA2,
    "flash_attn_3": BACKEND_FA3,
    "flash_attention_3": BACKEND_FA3,
    "flashattention": BACKEND_FA2,
    "flashattention2": BACKEND_FA2,
    "flashattention3": BACKEND_FA3,
    "torch_sdpa": BACKEND_SDPA,
    "scaled_dot_product_attention": BACKEND_SDPA,
    "math": BACKEND_EAGER,
    "vanilla": BACKEND_EAGER,
}

BACKEND_PACKAGE: Dict[str, str] = {
    BACKEND_FA3: "flash_attn",
    BACKEND_FA2: "flash_attn",
    BACKEND_XFORMERS: "xformers",
}

COMPILE_COMPILED = "compiled"
COMPILE_EAGER = "eager"

RESIDENCY_ALL_RESIDENT = "all_resident"

PLACEMENT_RESIDENT = "resident"
PLACEMENT_OFFLOADED = "offloaded"
PLACEMENT_CPU = "cpu"
PLACEMENT_DISK = "disk"

TECHNIQUE_FP8_STORAGE = "fp8_storage"
TECHNIQUE_PARTIAL_RESIDENT = "partial_resident"
TECHNIQUE_MODEL_OFFLOAD = "model_offload"
TECHNIQUE_GROUP_OFFLOAD = "group_offload"
TECHNIQUE_SEQUENTIAL = "sequential"
TECHNIQUE_CPU = "cpu"
TECHNIQUE_DISK_OFFLOAD = "disk_offload"
TECHNIQUE_VAE_TILING = "vae_tiling"
TECHNIQUE_VAE_SLICING = "vae_slicing"
TECHNIQUE_ATTENTION_SLICING = "attention_slicing"
TECHNIQUE_ATTENTION_FALLBACK = "attention_fallback"

REASON_VRAM_SHORTFALL = "vram_shortfall"
REASON_CUDA_OOM = "cuda_oom"
REASON_KERNEL_UNAVAILABLE = "kernel_unavailable"
REASON_LANE_CAST_DROPPED = "lane_cast_dropped"
REASON_NO_CUDA = "no_cuda"
REASON_BELOW_DECLARED_MINIMUM = "below_declared_minimum"
REASON_SERVING_FACTS_UNEVIDENCED = "serving_facts_unevidenced"

RESOURCE_VRAM = "vram"
RESOURCE_HOST_RAM = "host_ram"
RESOURCE_DISK = "disk"

_GIB = 1 << 30


def normalize_backend(raw: str) -> str:
    """Canonical attention-backend token, or ``""`` when nothing was stated."""
    tok = str(raw or "").strip().lower().replace("-", "_")
    if not tok:
        return ""
    tok = _BACKEND_ALIASES.get(
        tok, _BACKEND_ALIASES.get(tok.replace("_", ""), tok))
    if tok not in ATTENTION_BACKENDS:
        raise ValueError(
            f"attention backend {raw!r} is not one of "
            f"{sorted(ATTENTION_BACKENDS)} (aliases: "
            f"{sorted(_BACKEND_ALIASES)})")
    return tok


@dataclass(frozen=True)
class AppliedTechnique:
    """One lever the worker reached for, with the reason it reached."""

    name: str
    component: str = ""
    wanted: str = ""
    reason: str = ""
    est_slowdown: float = 0.0

    def to_proto(self) -> pb.AppliedTechnique:
        return pb.AppliedTechnique(
            name=self.name, component=self.component, wanted=self.wanted,
            reason=self.reason, est_slowdown=float(self.est_slowdown))


@dataclass(frozen=True)
class ComponentPosture:
    """What one component ended up as."""

    component: str
    applied_quant: str = ""
    bound_quant: str = ""
    placement: str = ""
    size_bytes: int = 0

    def to_proto(self) -> pb.ComponentPosture:
        return pb.ComponentPosture(
            component=self.component, applied_quant=self.applied_quant,
            bound_quant=self.bound_quant, placement=self.placement,
            bytes=int(self.size_bytes))


@dataclass(frozen=True)
class ResourceShortfall:
    """The quantified WHY (§1.36 amendment part 2)."""

    resource: str
    component: str = ""
    needed_bytes: int = 0
    available_bytes: int = 0

    @property
    def short_by_bytes(self) -> int:
        """The deficit."""
        return max(0, self.needed_bytes - self.available_bytes)

    @classmethod
    def from_gb(
        cls, resource: str, needed_gb: float, available_gb: float, *,
        component: str = "",
    ) -> "ResourceShortfall":
        """Bytes are the unit every consumer compares in; GB is what the placement path happens to think in."""
        return cls(
            resource=resource, component=component,
            needed_bytes=int(max(0.0, needed_gb) * _GIB),
            available_bytes=int(max(0.0, available_gb) * _GIB))

    def to_proto(self) -> pb.ResourceShortfall:
        return pb.ResourceShortfall(
            resource=self.resource, component=self.component,
            needed_bytes=int(self.needed_bytes),
            available_bytes=int(self.available_bytes))


@dataclass(frozen=True)
class MeasuredPosture:
    """The full set of conditions one measurement was taken under."""

    execution_lane: str = ""
    attention_backend: str = ""
    attention_backend_wanted: str = ""
    compile_state: str = ""
    compile_state_wanted: str = ""
    residency_mode: str = ""
    applied: Tuple[AppliedTechnique, ...] = ()
    components: Tuple[ComponentPosture, ...] = ()
    shortfall: Optional[ResourceShortfall] = None

    @property
    def observed(self) -> bool:
        """True when the worker actually OBSERVED this instance's posture."""
        return bool(
            self.attention_backend or self.attention_backend_wanted
            or self.residency_mode or self.applied or self.components
            or self.shortfall)

    @property
    def degraded(self) -> bool:
        """Any lever applied, any stated axis unmet, or a residency rung that had to move something off the card."""
        if self.applied:
            return True
        if self.residency_mode and self.residency_mode != RESIDENCY_ALL_RESIDENT:
            return True
        for wanted, applied in (
            (self.attention_backend_wanted, self.attention_backend),
            (self.compile_state_wanted, self.compile_state),
        ):
            if wanted and wanted != applied:
                return True
        return False

    def to_proto(self) -> pb.MeasuredPosture:
        msg = pb.MeasuredPosture(
            execution_lane=self.execution_lane,
            attention_backend=self.attention_backend,
            attention_backend_wanted=self.attention_backend_wanted,
            compile_state=self.compile_state,
            compile_state_wanted=self.compile_state_wanted,
            residency_mode=self.residency_mode,
        )
        for technique in self.applied:
            msg.applied.append(technique.to_proto())
        for component in self.components:
            msg.components.append(component.to_proto())
        if self.shortfall is not None:
            msg.shortfall.CopyFrom(self.shortfall.to_proto())
        return msg


_PLACEMENT_RESIDENCY: Dict[str, str] = {
    "": "",
    "off": RESIDENCY_ALL_RESIDENT,
    "vae_only": RESIDENCY_ALL_RESIDENT,
    "partial_resident": TECHNIQUE_PARTIAL_RESIDENT,
    "model_offload": TECHNIQUE_MODEL_OFFLOAD,
    "group_offload": TECHNIQUE_GROUP_OFFLOAD,
    "sequential": TECHNIQUE_SEQUENTIAL,
    "cpu": TECHNIQUE_CPU,
}

_TECHNIQUE_SLOWDOWN: Dict[str, float] = {
    TECHNIQUE_PARTIAL_RESIDENT: 1.3,
    TECHNIQUE_MODEL_OFFLOAD: 2.5,
    TECHNIQUE_GROUP_OFFLOAD: 3.0,
    TECHNIQUE_SEQUENTIAL: 4.0,
    TECHNIQUE_CPU: 20.0,
}


def residency_for_placement(mode: str) -> str:
    """The residency rung a placement mode means, ``""`` when unprepped."""
    return _PLACEMENT_RESIDENCY.get(str(mode or "").strip().lower(), "")


def placement_for_residency(residency: str) -> str:
    """Where a component's weights live, given the rung the pipeline ran on."""
    rung = str(residency or "").strip().lower()
    if not rung:
        return ""
    if rung == RESIDENCY_ALL_RESIDENT:
        return PLACEMENT_RESIDENT
    if rung == TECHNIQUE_CPU:
        return PLACEMENT_CPU
    if rung == TECHNIQUE_DISK_OFFLOAD:
        return PLACEMENT_DISK
    return PLACEMENT_OFFLOADED


@dataclass
class PostureLedger:
    """The per-instance accumulator, written at the ONE choke point every degradation already passes through."""

    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _applied: List[AppliedTechnique] = field(default_factory=list)
    _components: Dict[str, ComponentPosture] = field(default_factory=dict)
    _shortfall: Optional[ResourceShortfall] = None
    _residency: str = ""
    _attention: str = ""
    _attention_wanted: str = ""

    def technique(
        self, name: str, *, component: str = "", wanted: str = "",
        reason: str = "", est_slowdown: float = 0.0,
    ) -> None:
        if not name:
            return
        slowdown = est_slowdown or _TECHNIQUE_SLOWDOWN.get(name, 0.0)
        entry = AppliedTechnique(
            name=name, component=component, wanted=wanted, reason=reason,
            est_slowdown=slowdown)
        with self._lock:
            for existing in self._applied:
                if existing.name == name and existing.component == component:
                    return
            self._applied.append(entry)

    def shortfall(self, value: ResourceShortfall) -> None:
        """Keep the DEEPEST shortfall seen."""
        with self._lock:
            if (self._shortfall is None
                    or value.short_by_bytes > self._shortfall.short_by_bytes):
                self._shortfall = value

    def residency(self, mode: str) -> None:
        rung = residency_for_placement(mode)
        if rung:
            with self._lock:
                self._residency = rung

    def component(
        self, name: str, *, applied_quant: str = "", bound_quant: str = "",
        size_bytes: int = 0,
    ) -> None:
        if not name:
            return
        with self._lock:
            prior = self._components.get(name)
            self._components[name] = ComponentPosture(
                component=name,
                applied_quant=applied_quant or (
                    prior.applied_quant if prior else ""),
                bound_quant=bound_quant or (prior.bound_quant if prior else ""),
                size_bytes=size_bytes or (prior.size_bytes if prior else 0),
            )

    def attention(self, backend: str, *, wanted: str = "") -> None:
        """Record the engaged kernel, and the fallback if one happened."""
        engaged = normalize_backend(backend)
        asked = normalize_backend(wanted)
        with self._lock:
            self._attention = engaged or self._attention
            self._attention_wanted = asked or self._attention_wanted
        if asked and engaged and asked != engaged:
            self.technique(
                TECHNIQUE_ATTENTION_FALLBACK,
                reason=(REASON_KERNEL_UNAVAILABLE
                        if _package_missing(asked) else ""))

    def snapshot(
        self, *, execution_lane: str = "", compile_state: str = "",
        compile_state_wanted: str = "",
    ) -> MeasuredPosture:
        """The record, as of now."""
        with self._lock:
            residency = self._residency
            applied = tuple(self._applied)
            components = tuple(
                ComponentPosture(
                    component=c.component, applied_quant=c.applied_quant,
                    bound_quant=c.bound_quant,
                    placement=placement_for_residency(residency),
                    size_bytes=c.size_bytes)
                for c in self._components.values())
            shortfall = self._shortfall
            attention = self._attention
            attention_wanted = self._attention_wanted
        return MeasuredPosture(
            execution_lane=execution_lane,
            attention_backend=attention,
            attention_backend_wanted=attention_wanted,
            compile_state=compile_state,
            compile_state_wanted=compile_state_wanted,
            residency_mode=residency,
            applied=applied,
            components=components,
            shortfall=shortfall,
        )

    def clear(self) -> None:
        with self._lock:
            self._applied.clear()
            self._components.clear()
            self._shortfall = None
            self._residency = ""
            self._attention = ""
            self._attention_wanted = ""


def _package_missing(backend: str) -> bool:
    package = BACKEND_PACKAGE.get(backend, "")
    if not package:
        return False
    try:
        import importlib.util

        return importlib.util.find_spec(package) is None
    except (ImportError, ValueError):
        return True


def compile_axis(serving_mode: str) -> str:
    """``compiled`` | ``eager`` from ``ServedIdentity.serving_mode``."""
    return COMPILE_EAGER if str(serving_mode or "") == "eager" else COMPILE_COMPILED


def compile_axis_of_lane(lane: str) -> str:
    """What a lane descriptor DECLARES on the compile axis, ``""`` when it declares nothing."""
    text = str(lane or "").strip().lower()
    if "+" not in text:
        return ""
    suffix = text.rsplit("+", 1)[1]
    if suffix in (COMPILE_COMPILED, COMPILE_EAGER):
        return suffix
    return ""


def technique_for_run_mode(run_mode: str, to_rung: str) -> str:
    """The technique name a ladder transition means."""
    rung = str(to_rung or "").strip().lower()
    named = _PLACEMENT_RESIDENCY.get(rung, "")
    if named and named != RESIDENCY_ALL_RESIDENT:
        return named
    mode = str(run_mode or "").strip().lower()
    if mode == "fp8_storage":
        return TECHNIQUE_FP8_STORAGE
    if mode == "cpu":
        return TECHNIQUE_CPU
    if mode == "offload":
        return TECHNIQUE_MODEL_OFFLOAD
    return ""


def summarize(posture: MeasuredPosture) -> str:
    """One line for the human channel."""
    parts: List[str] = []
    if posture.execution_lane:
        parts.append(f"lane={posture.execution_lane}")
    if posture.residency_mode:
        parts.append(f"residency={posture.residency_mode}")
    if posture.attention_backend or posture.attention_backend_wanted:
        engaged = posture.attention_backend or "?"
        asked = posture.attention_backend_wanted
        parts.append(
            f"attention={engaged}" if not asked or asked == engaged
            else f"attention={asked}>{engaged}")
    if posture.compile_state_wanted and (
            posture.compile_state_wanted != posture.compile_state):
        parts.append(
            f"compile={posture.compile_state_wanted}>{posture.compile_state}")
    elif posture.compile_state:
        parts.append(f"compile={posture.compile_state}")
    if posture.applied:
        parts.append("applied=" + ">".join(
            t.name if not t.component else f"{t.name}:{t.component}"
            for t in posture.applied))
    short = posture.shortfall
    if short is not None and short.short_by_bytes > 0:
        parts.append(
            f"short={short.resource}"
            f"{'/' + short.component if short.component else ''} "
            f"needed={short.needed_bytes} available={short.available_bytes} "
            f"by={short.short_by_bytes}")
    return " ".join(parts)


__all__ = [
    "ATTENTION_BACKENDS",
    "AppliedTechnique",
    "BACKEND_EAGER",
    "BACKEND_FA2",
    "BACKEND_FA3",
    "BACKEND_PACKAGE",
    "BACKEND_SDPA",
    "BACKEND_XFORMERS",
    "COMPILE_COMPILED",
    "COMPILE_EAGER",
    "ComponentPosture",
    "MeasuredPosture",
    "PLACEMENT_CPU",
    "PLACEMENT_DISK",
    "PLACEMENT_OFFLOADED",
    "PLACEMENT_RESIDENT",
    "PostureLedger",
    "REASON_CUDA_OOM",
    "REASON_KERNEL_UNAVAILABLE",
    "REASON_LANE_CAST_DROPPED",
    "REASON_NO_CUDA",
    "REASON_VRAM_SHORTFALL",
    "RESIDENCY_ALL_RESIDENT",
    "RESOURCE_DISK",
    "RESOURCE_HOST_RAM",
    "RESOURCE_VRAM",
    "ResourceShortfall",
    "TECHNIQUE_ATTENTION_FALLBACK",
    "TECHNIQUE_ATTENTION_SLICING",
    "TECHNIQUE_CPU",
    "TECHNIQUE_DISK_OFFLOAD",
    "TECHNIQUE_FP8_STORAGE",
    "TECHNIQUE_GROUP_OFFLOAD",
    "TECHNIQUE_MODEL_OFFLOAD",
    "TECHNIQUE_SEQUENTIAL",
    "TECHNIQUE_VAE_SLICING",
    "TECHNIQUE_VAE_TILING",
    "compile_axis",
    "compile_axis_of_lane",
    "normalize_backend",
    "placement_for_residency",
    "residency_for_placement",
    "summarize",
    "technique_for_run_mode",
]
