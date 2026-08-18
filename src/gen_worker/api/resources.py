"""``Resources`` — the machine envelope ONE declaration states.

Extracted from the deleted ``api/decorators.py`` by the pgw#1373 hardcut: the
struct is v2 surface (``@entrypoint(resources=Resources(...))``, pgw#1396) and
survives the ``@endpoint`` deletion that used to host it. Nothing else in that
module did.

The other half of this contract is the hub's ``extractStaffingEnvelope`` /
``function_requirements.go`` ingest — every refusal below mirrors one there,
so a declaration that would be refused after an image bake is refused here,
at the author's own line.
"""

from __future__ import annotations

from typing import Any, Dict

import msgspec

from ..models.tensor_layout_contract import (
    LayoutRequirements, parse_layout_requirements,
)


def force(obj: Any, field: str, value: Any) -> None:
    """Assign through a frozen msgspec.Struct's normalization."""
    object.__setattr__(obj, field, value)


# hub's builder ingest (internal/builder/staffing_envelope.go
# parallelMechanisms) and orchestrator/topology's Parallel* constants — an
# unknown token refuses here, at declaration, before a build is spent.
_PARALLEL_MECHANISMS = frozenset({"sequence", "cfg"})


class Resources(msgspec.Struct, frozen=True, omit_defaults=True):
    """Hardware envelope for one function: ONLY what the endpoint
    CANNOT RUN WITHOUT — ``Resources(gpu, gpu_count, libraries, vcpus)``.

    "What it needs to run WELL" belongs to the fit ladder / residency
    planner / economics gate — all of which have measurements the endpoint
    author does not.

    th#1867 (DESIGN-RULINGS §1.35): this struct carries NO VRAM marker. Paul's
    ruling in one line: *"We should be able to run any model on any GPU. The
    challenge is not IF it can run — it's: is it an EFFICIENT choice?"* A card
    that looks too small is
    a card whose best (GPU, lane) pair sits further down the operator's ladder,
    and the RUNTIME picks that rung from what it MEASURES (``models/memory.py``
    ``select_auto_mode``), never from what the author guessed. §1.2 measured the
    guess against live serve profiles and it was wrong in BOTH directions —
    anima declared 8 GB against a 10.6 GiB peak, sdxl declared 20 against a
    proven 9.3 GiB run on a 16 GB A4000.

    ``requires`` is the FUNCTION scope of the one requirement vocabulary
    (pgw#1313) — the same grammar and the same terms as
    ``Slot(layout_requirements={handle: ...})``, for code with no contract to
    hang a requirement on: trainers, converters, encoders. It exists because
    training-endpoints has ZERO ``Slot(...)`` model slots (te#209), so its
    endpoints cannot express a floor any other way::

        Resources(requires="sm89+, vram80g")
        Resources(requires=LayoutRequirements(
            minimum="sm80+, vram48g", recommended="sm90+, vram80g, ram64g"))

    The compact form is the MINIMUM. A minimum gates ADMISSION — a
    config-write check on a pick a human is making — and NEVER execution; a
    ``recommended`` gates nothing at all, ever (th#1867/th#1720: the hub
    learned a monotone buy floor from the last recommendation that travelled).
    Declaring ``min_sm`` or ``min_vram_gb`` implies ``gpu=True``;
    ``min_host_ram_gb`` does not, and is declarable at ``recommended`` only.

    It SUBSUMES the two axes deleted with it. ``min_sm`` is the hard
    GPU-architecture floor (pgw#660) — a producer whose kernel is
    ``torch._scaled_mm`` cannot run below sm_89 at any precision, on any rung,
    ever, and with no carrier the scheduler placed the fp8 producer on sm_80
    A100s. It is spelled BARE (89, 100), tensorhub's own spelling and the one
    wire spelling on both sides. ``min_host_ram_gb`` (pgw#670) is declarable at
    the RECOMMENDED level only: Paul 2026-07-11 ruled that RunPod GPU pods
    cannot select or guarantee host RAM, so a host-RAM MINIMUM is unenforceable
    theater. Unmet, it is a degrade warning, like the offload rungs.

    There is no disk axis: th#1233 sizes a pod's container disk from the bytes
    the job will materialize, which covers every live case.

    ``vcpus`` declares the host-side vCPU ask (CPU-heavy encode);
    it does not imply ``gpu=True``. ``gpu_count`` declares how many devices
    one instance needs; endpoint code NEVER picks devices.

    ``max_gpu_count`` + ``max_gpus_per_execution_group`` + ``parallel``
    are the author ENVELOPE — the SDK half of the
    hub builder's ``extractStaffingEnvelope`` contract. ``gpu_count`` is the
    floor (devices ONE materialization requires); ``max_gpu_count`` is how
    many GPUs the POD may hold; ``max_gpus_per_execution_group`` is how many
    of them ONE REQUEST may be spread across; ``parallel`` names the platform
    mechanisms the function survives at degree > 1 (``"sequence"`` is the
    only one with a worker runtime). The author never writes a degree, a
    tier, a packing or a device id — those are the hub's decisions.

    The last two are INDEPENDENT axes::

        max_gpu_count=4                                  -> 1x4: one request across 4 cards
        max_gpu_count=4, max_gpus_per_execution_group=2  -> 2x2: two slots, each request across 2

    Omitting ``max_gpus_per_execution_group`` means "no opinion on that axis"
    and lets the hub use the whole ceiling as one group. It is NOT defaulted
    to a legal value: the declared domain is 2 or more, and 0/1 are refused
    here and at ingest, so "declared nothing" can never be confused with a
    declaration. Both are elided from the manifest when they carry nothing
    (``omit_defaults``). Validation mirrors the builder's ingest refusals so a
    contradiction costs a declaration-time ValueError instead of a build.
    """

    gpu: bool = False
    gpu_count: int = 1
    libraries: tuple[str, ...] = ()
    vcpus: int | None = None
    requires: Any = None
    max_gpu_count: int | None = None
    max_gpus_per_execution_group: int | None = None
    parallel: tuple[str, ...] = ()

    def manifest_dict(self) -> Dict[str, Any]:
        """The manifest ``resources{}`` projection.

        ``requires`` travels as the requirement ROW — declared terms only, per
        level (``LayoutRequirements.manifest_row``) — under its own key, the
        same shape the slot scope already emits, and the hub reads it
        (th#2072).

        NO compatibility projection is made. The ``compute_capability``
        back-projection this method carried was unreachable the moment th#2072
        landed: the hub takes ``requires`` wherever it is present and only
        falls back to ``compute_capability`` when the vector left the axis
        undeclared, and this method emitted the projection ONLY when
        ``min_sm`` was declared — i.e. only when ``requires`` already answered.
        The hub's remaining arm exists for PUBLISHED wheels (0.119.0 and
        older) that emit no ``requires`` at all; th#2074 retires it. No wheel
        built from this source is one of those.

        Two projections are deliberately NOT made, because each would resurrect
        a floor a ruling removed. ``min_host_ram_gb`` does not become the
        builder's ``ram_gb`` (a recommendation must never become an
        allocation minimum — that is th#1720 exactly), and ``min_vram_gb``
        does not become the builder's ``min_vram_gb`` (th#1867 deleted every
        VRAM marker on this struct; arming the lane VRAM floor is th#2073's,
        with the buy-side fail-open closed in the same change).
        """
        raw = msgspec.to_builtins(self)
        out: Dict[str, Any] = dict(raw) if isinstance(raw, dict) else {}
        out.pop("requires", None)
        requirement = self.requirement()
        if requirement is not None:
            out["requires"] = requirement.manifest_row()
        return out

    def requirement(self) -> LayoutRequirements | None:
        """The parsed function-scope requirement, or None if undeclared."""
        return self.requires if isinstance(
            self.requires, LayoutRequirements) else None

    def __post_init__(self) -> None:
        force = msgspec.structs.force_setattr
        if self.libraries:
            force(self, "libraries", tuple(
                str(x).strip() for x in self.libraries if str(x).strip()
            ))
        n_gpu = int(self.gpu_count)
        if n_gpu <= 0:
            raise ValueError(f"gpu_count must be positive, got {self.gpu_count}")
        force(self, "gpu_count", n_gpu)
        # The FUNCTION scope of the one requirement vocabulary. Normalized
        # here, at the declaration site, so the traceback names the
        # `Resources(...)` the author wrote rather than a manifest key.
        if self.requires is not None:
            force(self, "requires", parse_layout_requirements(
                self.requires, where="Resources(requires=)"))
        requirement = self.requirement()
        implies_gpu = requirement is not None and bool(
            requirement.min_terms().min_sm
            or requirement.min_terms().min_vram_gb
            or requirement.recommended_terms().min_sm
            or requirement.recommended_terms().min_vram_gb)
        if n_gpu > 1 or implies_gpu:
            force(self, "gpu", True)
        if self.vcpus is not None:
            c = int(self.vcpus)
            if c <= 0:
                raise ValueError(f"vcpus must be positive, got {c}")
            force(self, "vcpus", c)
        # Mirror the builder's extractStaffingEnvelope refusals at declaration
        # time.
        if self.max_gpu_count is not None:
            m = int(self.max_gpu_count)
            if m != self.max_gpu_count or m <= 0:
                raise ValueError(
                    f"max_gpu_count must be a positive whole number of "
                    f"devices, got {self.max_gpu_count}")
            if m < n_gpu:
                raise ValueError(
                    f"max_gpu_count {m} is below gpu_count {n_gpu}")
            force(self, "max_gpu_count", m)
            force(self, "gpu", True)
        # The degree axis: legal domain is [2, max_gpu_count]; absence means
        # "no opinion", never a defaulted legal value.
        if self.max_gpus_per_execution_group is not None:
            d = int(self.max_gpus_per_execution_group)
            if d != self.max_gpus_per_execution_group:
                raise ValueError(
                    f"max_gpus_per_execution_group must be a whole number of "
                    f"devices, got {self.max_gpus_per_execution_group}")
            if d < 2:
                raise ValueError(
                    f"max_gpus_per_execution_group {d} declares a sharding "
                    f"width that does not shard; omit the field to let the hub "
                    f"use the whole max_gpu_count as one group")
            if self.max_gpu_count is None or d > self.max_gpu_count:
                raise ValueError(
                    f"max_gpus_per_execution_group {d} exceeds max_gpu_count "
                    f"{self.max_gpu_count} — one request cannot span more GPUs "
                    f"than the pod may hold")
            force(self, "max_gpus_per_execution_group", d)
            force(self, "gpu", True)
        if self.parallel:
            mechanisms = tuple(
                str(x).strip().lower() for x in self.parallel if str(x).strip()
            )
            unknown = [m for m in mechanisms if m not in _PARALLEL_MECHANISMS]
            if unknown:
                raise ValueError(
                    f"parallel mechanism(s) {unknown} not implemented by the "
                    f"platform; known: {sorted(_PARALLEL_MECHANISMS)}")
            if len(set(mechanisms)) != len(mechanisms):
                raise ValueError(f"parallel repeats a mechanism: {mechanisms}")
            if self.max_gpu_count is None or self.max_gpu_count <= n_gpu:
                raise ValueError(
                    f"parallel {list(mechanisms)} declared without a "
                    f"max_gpu_count above gpu_count {n_gpu} — a mechanism "
                    f"is only reachable with device headroom")
            force(self, "parallel", mechanisms)
            force(self, "gpu", True)
        elif self.max_gpus_per_execution_group is not None:
            # A group width only bounds a MECHANISM; with none declared the
            # degree is always 1 and the field is inert. Mirrors the hub's own
            # ingest refusal.
            raise ValueError(
                f"max_gpus_per_execution_group "
                f"{self.max_gpus_per_execution_group} declared without a "
                f"parallel mechanism — nothing shards a group, so the "
                f"declaration is inert")




__all__ = ["Resources"]
