"""The POSTURE a request was served under, typed (th#1871 P1, pgw#1225).

DESIGN-RULINGS §1.36 stage 2: *"a benchmark measured through a silent
degradation is not a benchmark"*. The deciding specimen is ie#707 — a family
served on ``sdpa`` because flash-attn was absent from its image while its lane
declared flash, and every number taken from those runs was filed as a
measurement OF the flash lane. Nothing downstream could tell.

The hub now keys its measurement relation on the posture
(``endpoint_measurements``, tensorhub ``4f57e6e6``: ``(release, function,
gpu_sku_id, execution_lane, posture_digest, source)``). Until this module
existed, every ``source='serve'`` row got the UNREPORTED posture — the hub's
reducer deliberately refuses to parse the prose we already emit into
``worker_activity_events.detail``, because parsing prose to recover structure
is the defect the relation exists to end.

So this is a change of REPRESENTATION, not of instrumentation. Every field here
is something the worker already knows and already says, in words, on a channel
no decision reads.

WHAT THIS MODULE DELIBERATELY DOES NOT DO
-----------------------------------------
**It does not digest.** The posture's identity is
``measurement.Posture.Digest()`` on the hub and there is exactly one
implementation of it. A worker-side digest would be a second one — two
canonicalizations of the same object, each green against its own tests, free to
disagree exactly like the two topology decoders did (pgw#1188). The worker
reports the structure; the hub decides what it is identical to. What keeps the
two sides honest is the byte-identical vector fence
(``tests/testdata/posture_wire_vectors.json`` + ``scripts/posture-vector-drift.sh``),
not a duplicated hash.

**It does not decide anything.** No refusal, no gate, no placement input.
§1.35's second amendment stands: loudness is diagnostics.

**It does not recommend a card.** The worker knows what it was short by; only
the hub knows the catalog and the prices (§1.36 amendment, clause 3). ``recommended_vram_gb`` was the author's own declared hint handed back, and
th#1867 deleted it from the wire outright rather than leaving it to be read as
a measurement.

THE THREE RULES THAT ARE THE HUB'S, RESTATED SO THE PRODUCER OBEYS THEM
----------------------------------------------------------------------
1. **``applied`` is ORDERED and never collapsed.** The order the worker reached
   for levers is a fact about the run — ``group_offload`` AFTER ``model_offload``
   failed is not ``group_offload`` instead of it. The retired ``FnDegraded.ran``
   projected ``model_offload``/``group_offload``/``sequential`` onto the single
   token ``offload``, so a 2.5x, a 3.0x and a 4.0x degradation were
   indistinguishable to the hub, and tiling, slicing and attention fallback had
   no representation at all.
2. **``wanted`` and applied are two fields on every axis, never one reconciled
   value.** "eager was chosen" and "compiled was asked for and eager happened"
   are different facts with different owners. A schema holding only the resolved
   value makes the second UNREPRESENTABLE at the moment it matters most.
3. **Magnitudes are data, not identity.** Two runs short by 6.1 and 6.3 GiB that
   reached for the same levers in the same order are one cell with two samples.
   Report the bytes; the hub excludes them from the key.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .pb import worker_scheduler_pb2 as pb

logger = logging.getLogger(__name__)

# --- vocabularies -----------------------------------------------------------
# Wire-shared with tensorhub (internal/orchestrator/measurement/posture.go).
# A value not in these sets is not "unknown" to the hub, it is a THIRD
# vocabulary — which is the defect th#1871 §1.3 measured: three writers keying
# one relation three ways, so no two sources could ever collide or join.

#: What actually ran the attention. `eager` here is the math fallback, NOT the
#: compile axis — the two words collide across axes and must not be conflated.
BACKEND_FA3 = "fa3"
BACKEND_FA2 = "fa2"
BACKEND_SDPA = "sdpa"
BACKEND_XFORMERS = "xformers"
BACKEND_EAGER = "eager"

ATTENTION_BACKENDS: frozenset[str] = frozenset({
    BACKEND_FA3, BACKEND_FA2, BACKEND_SDPA, BACKEND_XFORMERS, BACKEND_EAGER,
})

#: Spellings the ecosystem uses for the same kernel. Normalized rather than
#: rejected: an endpoint that reports `flash_attention_2` is being honest and
#: precise, and refusing it is how `report_applied_attention` ended up with 23
#: of 29 families reporting nothing at all.
_BACKEND_ALIASES: Dict[str, str] = {
    "flash": BACKEND_FA2,
    "flash_attn": BACKEND_FA2,
    "flash_attention": BACKEND_FA2,
    "flash_attn_2": BACKEND_FA2,
    "flash_attention_2": BACKEND_FA2,
    "flash_attn_3": BACKEND_FA3,
    "flash_attention_3": BACKEND_FA3,
    # The concatenated spellings the upstream projects actually print
    # (`FlashAttention2`, `FlashAttention-3`). Separators are normalized to `_`
    # before this lookup, so only the run-together forms need naming.
    "flashattention": BACKEND_FA2,
    "flashattention2": BACKEND_FA2,
    "flashattention3": BACKEND_FA3,
    "torch_sdpa": BACKEND_SDPA,
    "scaled_dot_product_attention": BACKEND_SDPA,
    "math": BACKEND_EAGER,
    "vanilla": BACKEND_EAGER,
}

#: The import a backend needs to exist at all. An absent package is the ie#707
#: cause, and it is a fact the worker can state without being told.
BACKEND_PACKAGE: Dict[str, str] = {
    BACKEND_FA3: "flash_attn",
    BACKEND_FA2: "flash_attn",
    BACKEND_XFORMERS: "xformers",
}

#: The compile axis. `eager` collides spelling-wise with BACKEND_EAGER and is a
#: different axis; both are the hub's vocabulary and neither may be renamed here.
COMPILE_COMPILED = "compiled"
COMPILE_EAGER = "eager"

#: Residency rungs. ALL_RESIDENT is the only undegraded one; an UNSTATED mode is
#: unknown, and unknown must never render as fine.
RESIDENCY_ALL_RESIDENT = "all_resident"

#: Where a component's weights actually live.
PLACEMENT_RESIDENT = "resident"
PLACEMENT_OFFLOADED = "offloaded"
PLACEMENT_CPU = "cpu"
PLACEMENT_DISK = "disk"

#: Technique names — the SDK's OWN rungs (`models/rung.py`) plus the levers that
#: had no wire representation at all.
TECHNIQUE_FP8_STORAGE = "fp8_storage"
TECHNIQUE_MODEL_OFFLOAD = "model_offload"
TECHNIQUE_GROUP_OFFLOAD = "group_offload"
TECHNIQUE_SEQUENTIAL = "sequential"
TECHNIQUE_CPU = "cpu"
TECHNIQUE_DISK_OFFLOAD = "disk_offload"
TECHNIQUE_VAE_TILING = "vae_tiling"
TECHNIQUE_VAE_SLICING = "vae_slicing"
TECHNIQUE_ATTENTION_SLICING = "attention_slicing"
TECHNIQUE_ATTENTION_FALLBACK = "attention_fallback"

#: Machine-readable causes. Prose belongs in the activity event, never here.
REASON_VRAM_SHORTFALL = "vram_shortfall"
REASON_CUDA_OOM = "cuda_oom"
REASON_KERNEL_UNAVAILABLE = "kernel_unavailable"
REASON_LANE_CAST_DROPPED = "lane_cast_dropped"
REASON_NO_CUDA = "no_cuda"

#: Shortfall resources.
RESOURCE_VRAM = "vram"
RESOURCE_HOST_RAM = "host_ram"
RESOURCE_DISK = "disk"

_GIB = 1 << 30


def normalize_backend(raw: str) -> str:
    """Canonical attention-backend token, or ``""`` when nothing was stated.

    Raises ``ValueError`` on a token that is neither canonical nor a known
    alias: a value the hub does not share is not a measurement, it is a fourth
    vocabulary, and the reporter is the last place it can be caught.
    """
    tok = str(raw or "").strip().lower().replace("-", "_")
    if not tok:
        return ""
    # Separators are noise on this axis: `FlashAttention-3`, `flash_attn_3` and
    # `flashattention3` are one kernel written three ways, and an author who
    # reports the one their own library prints is being precise, not sloppy.
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
    """One lever the worker reached for, with the reason it reached.

    ``wanted`` empty means the lever was FORCED. A forced lever is a
    degradation but NOT a mismatch — a mismatch means a *stated* expectation was
    violated, and counting forced levers twice inflates the report whose whole
    job is surfacing SILENT substitutions.
    """

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
    """What one component ended up as.

    "The transformer is offloaded" is actionable; "the pipeline is degraded" is
    not. ``applied_quant`` is what the weights were actually LOADED as, which is
    not always what the lane BOUND — the live fleet reports
    ``applied=fp8-w8a8-dynamic ... bound=bf16-w16a16`` today, in prose.
    """

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
    """The quantified WHY (§1.36 amendment part 2).

    "Insufficient VRAM" is the old vocabulary; ``needed N, had M, short by N-M
    for component X`` is the diagnostic. These numbers already exist as locals
    inside the worker's own placement decision; this type is where they land
    instead of being discarded.
    """

    resource: str
    component: str = ""
    needed_bytes: int = 0
    available_bytes: int = 0

    @property
    def short_by_bytes(self) -> int:
        """The deficit. A shortfall that is not short is not a shortfall."""
        return max(0, self.needed_bytes - self.available_bytes)

    @classmethod
    def from_gb(
        cls, resource: str, needed_gb: float, available_gb: float, *,
        component: str = "",
    ) -> "ResourceShortfall":
        """Bytes are the unit every consumer compares in; GB is what the
        placement path happens to think in. Convert once, here — a GB/bytes
        mixup is a silent factor of 2^30."""
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
    """The full set of conditions one measurement was taken under.

    Two measurements may be compared if and only if their postures are equal.
    The hub decides that (``Posture.Digest``); this type only has to be complete
    and honest.
    """

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
        """True when the worker actually OBSERVED this instance's posture.

        Deliberately ignores lane and compile state: those two are known for
        every request from the served identity alone, so including them would
        make every posture "observed" and the flag would answer nothing.

        This is the guard on the emit path, and the asymmetry it protects is the
        point. An all-empty posture is NOT a clean posture — it is the absence
        of a report, and the hub keys the two differently on purpose (the
        unreported posture has its own digest, so it can never be mistaken for a
        cell measured on a known-clean one). Sending one would claim "measured,
        nothing applied" on behalf of a worker that never looked: ie#707 with
        the polarity flipped.
        """
        return bool(
            self.attention_backend or self.attention_backend_wanted
            or self.residency_mode or self.applied or self.components
            or self.shortfall)

    @property
    def degraded(self) -> bool:
        """Any lever applied, any stated axis unmet, or a residency rung that
        had to move something off the card. Mirrors the hub's ``Degraded()`` so
        the worker's own log line cannot disagree with the hub's reading."""
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
        # Ordered on the wire because the order is the fact (rule 1). Components
        # are NOT sorted here: the hub sorts before digesting, and a producer
        # that pre-sorts hides a producer that does not.
        for technique in self.applied:
            msg.applied.append(technique.to_proto())
        for component in self.components:
            msg.components.append(component.to_proto())
        if self.shortfall is not None:
            msg.shortfall.CopyFrom(self.shortfall.to_proto())
        return msg


# ---------------------------------------------------------------------------
# The ledger
# ---------------------------------------------------------------------------

#: `memory.low_vram_mode` placement modes -> (residency rung, technique).
#: `off` and `vae_only` are BOTH fully resident (pgw#750: the vae_only
#: refinement only toggles VAE slicing, it moves no weights), so they are the
#: undegraded rung — and `vae_only` still contributes its slicing technique,
#: because a sliced decode is a different traced graph and a different number.
_PLACEMENT_RESIDENCY: Dict[str, str] = {
    "": "",
    "off": RESIDENCY_ALL_RESIDENT,
    "vae_only": RESIDENCY_ALL_RESIDENT,
    "model_offload": TECHNIQUE_MODEL_OFFLOAD,
    "group_offload": TECHNIQUE_GROUP_OFFLOAD,
    "sequential": TECHNIQUE_SEQUENTIAL,
    "cpu": TECHNIQUE_CPU,
}

#: The honest price of each rung vs a resident run. Reported, never keyed: it is
#: the worker's estimate OF the lever, not a property of the run, and two SDK
#: versions estimating it differently must not fork one posture's identity.
_TECHNIQUE_SLOWDOWN: Dict[str, float] = {
    TECHNIQUE_MODEL_OFFLOAD: 2.5,
    TECHNIQUE_GROUP_OFFLOAD: 3.0,
    TECHNIQUE_SEQUENTIAL: 4.0,
    TECHNIQUE_CPU: 20.0,
}


def residency_for_placement(mode: str) -> str:
    """The residency rung a placement mode means, ``""`` when unprepped.

    Unprepped stays EMPTY rather than defaulting to all-resident: a pipeline
    nobody placed is unknown, and "unknown" rendering as "fine" is the exact
    zero-vs-null defect §1.36's measurement columns are nullable to avoid.
    """
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
    """The per-instance accumulator, written at the ONE choke point every
    degradation already passes through.

    It is a ledger and not a set of flags for the reason ``applied`` is ordered:
    the sequence of levers is the finding. Appends are idempotent per
    (name, component) — the placement path can be re-entered on a retry, and a
    lever recorded twice would read as a deeper descent than actually happened.
    """

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
        """Keep the DEEPEST shortfall seen. A descent that was short by 2 GiB
        and then, one rung down, by 9 GiB is described by the 9."""
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
        """Record the engaged kernel, and the fallback if one happened.

        This is ie#707's whole shape: the fallback was SILENT, so the numbers
        were filed under the lane that was declared rather than the one that
        ran. When the wanted backend needs a package this image does not have,
        the cause is stated rather than left to a reader's inference.
        """
        engaged = normalize_backend(backend)
        asked = normalize_backend(wanted)
        with self._lock:
            self._attention = engaged or self._attention
            self._attention_wanted = asked or self._attention_wanted
        if asked and engaged and asked != engaged:
            # The lever is recorded WITHOUT a `wanted`, deliberately. The
            # attention axis already carries wanted-vs-applied as first-class
            # fields, and the hub derives a mismatch finding per technique by
            # comparing `wanted` against the technique's own NAME — so a
            # backend token there would both double-count the finding and render
            # as `attention_fallback:fa2>attention_fallback`, which is not a
            # sentence. One axis, one reading (ie#655's rule).
            self.technique(
                TECHNIQUE_ATTENTION_FALLBACK,
                reason=(REASON_KERNEL_UNAVAILABLE
                        if _package_missing(asked) else ""))

    def snapshot(
        self, *, execution_lane: str = "", compile_state: str = "",
        compile_state_wanted: str = "",
    ) -> MeasuredPosture:
        """The record, as of now. The three per-REQUEST axes are arguments
        rather than ledger state: they are decided at the terminal, from the
        same ``ServedIdentity`` the rest of ``JobMetrics`` is stamped from, so
        the posture cannot disagree with ``metrics.serving_mode`` about eager
        (ie#655's rule, one axis one reading)."""
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
        """Forget everything — the instance was torn down and reloaded, so its
        levers are no longer facts about what is serving."""
        with self._lock:
            self._applied.clear()
            self._components.clear()
            self._shortfall = None
            self._residency = ""
            self._attention = ""
            self._attention_wanted = ""


def _package_missing(backend: str) -> bool:
    """Whether the backend's kernel package is absent from THIS image.

    Probed with ``find_spec`` and never by import: importing a kernel package to
    ask whether it exists is how a probe becomes a side effect.
    """
    package = BACKEND_PACKAGE.get(backend, "")
    if not package:
        return False
    try:
        import importlib.util

        return importlib.util.find_spec(package) is None
    except (ImportError, ValueError):
        return True


def compile_axis(serving_mode: str) -> str:
    """``compiled`` | ``eager`` from ``ServedIdentity.serving_mode``.

    ``jit_cell`` and ``aot_cell`` are BOTH compiled — the artifact kind is a
    different axis (``metrics.serving_mode`` carries it), and folding it in here
    would split every cell in two on a fact §1.30 ruled is a cache question.
    """
    return COMPILE_EAGER if str(serving_mode or "") == "eager" else COMPILE_COMPILED


def compile_axis_of_lane(lane: str) -> str:
    """What a lane descriptor DECLARES on the compile axis, ``""`` when it
    declares nothing. ``fp8-w8a8-dynamic+compiled`` -> ``compiled``.

    Never split a compiled_graph KEY this way — this parses a LANE, whose
    ``+`` separator is the platform's execution axis (§1.30).
    """
    text = str(lane or "").strip().lower()
    if "+" not in text:
        return ""
    suffix = text.rsplit("+", 1)[1]
    if suffix in (COMPILE_COMPILED, COMPILE_EAGER):
        return suffix
    return ""


def technique_for_run_mode(run_mode: str, to_rung: str) -> str:
    """The technique name a ladder transition means.

    ``run_mode`` is the hub-wire projection (``serve_fit.RUN_*``) and is
    deliberately COARSE — ``offload`` covers three rungs whose prices differ by
    60%. So the named rung wins whenever there is one, which is the whole point
    of item 5 of th#1871 §6.6: those three stop sharing one token.
    """
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
        # An offload transition that named no rung: the token is all we have,
        # and guessing which of the three it was would be a fabricated fact.
        return TECHNIQUE_MODEL_OFFLOAD
    return ""


def summarize(posture: MeasuredPosture) -> str:
    """One line for the human channel. The typed record is the contract; this
    is for the operator reading pod logs, and it is derived FROM the record so
    the two cannot say different things."""
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
