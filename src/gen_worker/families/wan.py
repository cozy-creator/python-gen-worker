"""Wan family vocabulary (pgw#520 / th#767) — Wan 2.2 A14B MoE (dual-expert
high-noise/low-noise transformer) inference defaults, the animegen-lane
onboarding of a second family alongside SDXL.

Registered under tensorhub's CANONICAL architecture root ``"wan22"``
(``internal/modelfamily/modelfamily.go`` canonicalFamilies / th#586) — this
is deliberately NOT the same string as gen-worker's ``Compile(family=)``
values used by the wan-2.2 endpoint (``"wan-2.2-t2v-a14b"`` /
``"-i2v-a14b"`` / ``"-ti2v-5b"``). Those are narrower, per-function
compile-cell keys (shape-grid/cell selection); repo-metadata validation on
the tensorhub side keys on the REPO's own classified ``model_family``
column (``internal/api/repo_inference_defaults.go``'s
``repoForInferenceDefaults``), which is the shared architecture root every
wan22-envelope checkpoint carries regardless of which endpoint function
serves it. One ``WanDefaults`` vocabulary covers base Wan2.2-T2V/I2V-A14B,
TI2V-5B, and fine-tunes sharing the envelope (AnimeGen-T2V, ...).
"""

from __future__ import annotations

from typing import Optional

from .base import FamilyDefaults, family


@family("wan22")
class WanDefaults(FamilyDefaults, frozen=True):
    """Wan 2.2 MoE inference defaults + constraints.

    ``guidance`` is the high-noise expert's CFG scale; ``guidance_2`` is the
    low-noise expert's. Diffusers' ``WanPipeline.__call__`` defaults
    ``guidance_scale_2`` to ``guidance_scale`` when the caller leaves it
    unset and the pipeline config carries a ``boundary_ratio`` (the MoE
    dual-expert split) — ``max_guidance`` clamps BOTH fields explicitly at
    the endpoint's resolution site rather than relying on that internal
    defaulting, so a distilled lineage's CFG ceiling holds regardless of
    which field a caller overrides.

    ``shift`` (ie#522 live finding #3): a step-distilled lineage (e.g. a
    lightx2v/Wan2.2-Lightning fuse) is picky about the flow-matching
    scheduler's ``shift`` — the AnimeGen card's own recipe pins
    ``FlowMatchEulerDiscreteScheduler(shift=3.0)`` alongside its 8-step/
    guidance=1.0 settings. ``None`` (the default, and every non-distilled
    repo's metadata) leaves the pipeline's own scheduler config untouched;
    only a repo that explicitly publishes a ``shift`` value gets one
    applied at the endpoint's resolution site.
    """

    steps: int = 40
    guidance: float = 4.0
    guidance_2: float = 3.0
    max_guidance: Optional[float] = None
    shift: Optional[float] = None


__all__ = ["WanDefaults"]
