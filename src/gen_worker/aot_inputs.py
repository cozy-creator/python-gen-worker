"""Generic export-input machinery for the AOT mint (#723, #739).

``torch.export`` needs example inputs with the exact structure the target's
forward takes. That structure is FAMILY knowledge — and per Paul's SDK-generic
rule (pgw#739) it is a DECLARATION in the endpoint spec, never worker code:
the endpoint declares ``Compile(dims=..., forks=..., classes=..., inputs=...)``
and :func:`gen_worker.aot_declaration.declared_inputs` derives the example
inputs generically. This module carried the sdxl contract as per-family SDK
code during bootstrap; that family knowledge is deleted (it lives in the sdxl
endpoint's declaration now) and what remains is generic: composition, the
lifted-LoRA kwargs, dtype/device introspection, and a registration hook for
callers that need a hand-written builder while a family's declaration is
still being written.

Composition itself is NOT family-specific and mirrors the SERVING path's own
composition: ``load_from_pretrained`` -> ``place_pipeline``. (It used to say
"mirrors ``compile_cache.build``"; that whole-pipeline dynamo mint had no caller
and pgw#1035 deleted it — the requirement it named is unchanged.) That is a
parity requirement, not tidiness — the placement/low-VRAM flags are traced
INTO the graph, so a cell composed differently from the serving path is a cell
the serving path can never use (gw#391, ie#381 are both that bug).

The ONE deliberate divergence is the LoRA lane: the dynamo mint stops at
``compile_cache.apply_lora_lane``, which allocates zeroed branch BUFFERS —
module state that export would bake. The exported lane goes one step further
through ``lora_lifted.arm_lifted_lora_lanes``, which wraps those same
containers in the lifted forward so the adapter arrives as call arguments
(#725 option 2). Same bucket, one extra step, because the two lanes disagree
about what "dynamic" means — and skipping that step is pgw#822.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, Dict, Mapping, Tuple

from .compile_cache import resolve_pipeline_class
from .models import lora_lifted
from .models.loading import load_from_pretrained
from .models.memory import place_pipeline
from .api.export_contract import export_declaration
from .aot_contract import DynamicDim

if TYPE_CHECKING:  # pragma: no cover — typing only, no runtime import cycle
    from .aot_contract import DynamicDim, ExportSpec

logger = logging.getLogger(__name__)

InputBuilder = Callable[[Any, "ExportSpec"], Tuple[Tuple[Any, ...], Dict[str, Any]]]

#: Keyed by ``(family, target)`` — NOT by family (ie#566 G1). A family's compile
#: targets are unrelated modules with unrelated call contracts: wan's span the
#: denoiser AND the VAE, and a family-only key made ``vae.decode`` unmintable for
#: EVERY family, not just wan. ``target=""`` registers a family-wide fallback.
#: This registry is the ESCAPE HATCH while a family's declaration is being
#: written; a registered export DECLARATION (pgw#739) always wins over it.
_BUILDERS: Dict[Tuple[str, str], InputBuilder] = {}


class InputContractError(RuntimeError):
    """No export-input contract is declared or registered for a family, or it
    cannot be satisfied from the composed pipeline."""


def inputs_for(family: str, target: str = "") -> Callable[[InputBuilder], InputBuilder]:
    """Register a hand-written example-input builder for one ``(family,
    target)``. ``target=""`` registers a family-wide fallback.

    Prefer the declaration (``Compile(inputs=...)``): a registered export
    declaration supersedes anything registered here.
    """

    def register(fn: InputBuilder) -> InputBuilder:
        _BUILDERS[(str(family), str(target))] = fn
        return fn

    return register


def builder_for(family: str, target: str = "") -> InputBuilder:
    """The builder for ``(family, target)``: the family's registered export
    DECLARATION when it carries ``inputs=`` rows, else the hand-registered
    builder, else a named refusal.

    Exact ``(family, target)`` wins over a family-wide registration. A family
    with a declared denoiser but no VAE contract must FAIL for the VAE rather
    than silently feed it denoiser inputs.
    """

    fam, tgt = str(family), str(target)
    decl = export_declaration(fam)
    if decl is not None and decl.inputs:
        from . import aot_declaration

        def declared(module: Any, spec: "ExportSpec") -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
            return aot_declaration.declared_inputs(module, spec, decl)

        return declared
    for key in ((fam, tgt), (fam, "")):
        fn = _BUILDERS.get(key)
        if fn is not None:
            return fn
    known = sorted(f"{f}/{t or '*'}" for f, t in _BUILDERS)
    raise InputContractError(
        f"no export-input contract for {fam!r} target {tgt!r}: no registered "
        f"export declaration (Compile(inputs=...), pgw#739) and no "
        f"hand-registered builder (have: {known!r}) — a target's call "
        f"contract is family knowledge and must be DECLARED by the endpoint "
        f"before it can be exported")


def lifted_lora_values(module: Any, spec: "ExportSpec") -> Dict[str, Any]:
    """The mandatory ``lora_a``/``lora_b`` call values for a LoRA-bucket mint,
    keyed by parameter NAME — the builder binds them to their POSITIONAL
    slots (all-positional example feeds are a mint obligation: pod 9's
    kwarg-traced package armed and silently revoked to eager on first call,
    pgw#723 residuals).

    #725 option 2: the adapter rides as a flat 1-D pair with static per-layer
    offsets, passed as CALL ARGUMENTS. The pair must be present at trace time —
    tracing without it traces the branchless graph and bakes the absence, which
    is the "missing FQN is the same bug in a different hat" case — and it must be
    NON-ZERO, because a zeroed B lets constant folding erase the branch and
    leaves the gate passing on nothing.

    Empty dict for a bucket-0 mint: the branchless pipeline is its own graph
    class and carries no adapter arguments at all.
    """
    if not int(spec.lora_bucket or 0):
        return {}
    import torch

    plan = lora_lifted.build_plan(module, int(spec.lora_bucket))
    lora_a, lora_b = plan.alloc(module_dtype_device(module)[1])
    return {
        "lora_a": torch.randn_like(lora_a),
        "lora_b": torch.randn_like(lora_b),
    }


def module_dtype_device(module: Any) -> Tuple[Any, Any]:
    """The resident dtype/device of a module, read off its own tensors."""
    import torch

    for source in (module.parameters, module.buffers):
        try:
            first = next(iter(source()))
        except (StopIteration, AttributeError, TypeError):
            continue
        return first.dtype, first.device
    return torch.bfloat16, torch.device("meta")


def latent_dims(
    spec: "ExportSpec",
    *,
    input_name: str = "sample",
    vae_scale: int = 8,
    downsamples: int = 8,
) -> Tuple["DynamicDim", ...]:
    """Declared symbolic latent H/W dims spanning ``spec.shapes``.

    Generic derivation for REQUEST-level dynamic rows (a family without a
    class declaration): bounds derive from the declared aspect set rather
    than hand-written numbers, so "these are the aspect ratios we serve" is
    stated once and the admissible range cannot drift from it.
    ``downsamples`` is the divisibility the network itself requires; a range
    not expressible as that multiple is refused at
    ``aot_mint.dynamic_shapes_spec`` rather than silently 0/1-specialized.

    Bounds are ROUNDED OUT to the multiple, never in: rounding in would
    exclude a declared aspect ratio from the artifact's own contract, and B2
    measured that nothing at runtime would tell us.

    Families with a class declaration never call this — their bounds derive
    from the class rows (``aot_declaration.derived_dynamic``).
    """

    if not spec.shapes:
        raise InputContractError(
            "cannot derive latent dims without declared shape rows")
    widths = [int(row[0]) // vae_scale for row in spec.shapes]
    heights = [int(row[1]) // vae_scale for row in spec.shapes]
    both = widths + heights
    low = (min(both) // downsamples) * downsamples
    high = -(-max(both) // downsamples) * downsamples
    if low < downsamples:
        low = downsamples
    return (
        DynamicDim(input_name, 2, low, high, multiple_of=downsamples),
        DynamicDim(input_name, 3, low, high, multiple_of=downsamples),
    )


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------


def compose(
    model: str, spec: "ExportSpec", request: Mapping[str, Any],
) -> Tuple[Any, Callable[[Any], Tuple[Tuple[Any, ...], Dict[str, Any]]]]:
    """Compose the pipeline for a mint; return it with its input builder.

    The sequence mirrors the serving path's composition exactly (see the
    module docstring for why that is a requirement rather than a convention). The
    returned builder takes the RESOLVED target module — not the pipeline — so
    widths and dtypes come off the thing actually being exported.
    """
    from diffusers import DiffusionPipeline

    builder = builder_for(spec.family, spec.target)
    load_cls: Any = DiffusionPipeline
    pipeline_class = str(request.get("pipeline_class") or "").strip()
    if pipeline_class:
        # gw#586 call-path parity: trace through the SERVING pipeline class.
        load_cls = resolve_pipeline_class(pipeline_class)
    pipe = load_from_pretrained(
        load_cls, str(model),
        dtype=str(request.get("dtype") or "bf16"),
        storage_dtype=str(request.get("storage_dtype") or ""),
        declared_vram_gb=float(request.get("declared_vram_gb") or 0.0),
    )
    place_pipeline(pipe)
    if int(spec.lora_bucket or 0):
        # #725 option 2 through pgw#822's ONE arm: containers then lifted
        # signature. The dynamo lane stops after the containers (module state
        # export would bake); the exported lane lifts the adapter to call
        # arguments ON TOP of them — `install_lifted_lora_execution_lanes` alone has no
        # container to lay its flat pair out over.
        lora_lifted.arm_lifted_lora_execution_lanes(pipe, int(spec.lora_bucket))
    if callable(getattr(pipe, "set_progress_bar_config", None)):
        pipe.set_progress_bar_config(disable=True)
    return pipe, lambda module: builder(module, spec)


__all__ = [
    "InputContractError",
    "builder_for",
    "compose",
    "inputs_for",
    "latent_dims",
    "lifted_lora_values",
    "module_dtype_device",
]
