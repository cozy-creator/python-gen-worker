"""Generic export-input machinery for the AOT mint.

``torch.export`` needs example inputs with the exact structure the target's
forward takes. That structure is FAMILY knowledge — and per Paul's SDK-generic
rule it is a DECLARATION in the endpoint spec, never worker code: the endpoint
declares ``Compile(dims=..., forks=..., classes=..., inputs=...)`` and
:func:`gen_worker.aot_declaration.declared_inputs` derives the example inputs
generically. What lives here is the minimal worker-owned input vocabulary,
lifted-LoRA values, dtype/device introspection, and the temporary registration
hook for families whose declaration is still being written. Pipeline
composition belongs to the compile child and serving loader, not this contract.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

from .compile_cache import execution_lane_label as _execution_lane_label
from .models import lora_lifted
from .api.export_contract import export_declaration

logger = logging.getLogger(__name__)


class MintRefused(RuntimeError):
    """A named, terminal refusal to produce or publish a compiled graph."""

    def __init__(
        self,
        *args: Any,
        mint_phases: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__(*args)
        self.mint_phases: Dict[str, Any] = dict(mint_phases or {})


# The SDK-owned fork coordinate of a LoRA-bucket family: one class with the
# lifted adapter inputs and one without. This is worker declaration policy,
# not compiled-graph identity or artifact metadata.
ADAPTER_FORK = "adapter"


@dataclass(frozen=True)
class DynamicDim:
    """One declared symbolic dimension of one export input."""

    input_name: str
    axis: int
    min: int
    max: int
    multiple_of: int = 1
    dim: str = ""

    def as_row(self) -> Dict[str, Any]:
        return {
            "input": self.input_name,
            "axis": self.axis,
            "min": self.min,
            "max": self.max,
            "multiple_of": self.multiple_of,
        }


@dataclass
class ExportSpec:
    """Worker-only tracing inputs for one declared compiled graph class."""

    family: str
    target: str
    weight_lane: str = ""
    precision: str = ""
    lora_bucket: int = 0
    shapes: Tuple[Tuple[int, ...], ...] = ()
    text_lens: Tuple[int, ...] = ()
    guidance_scales: Tuple[float, ...] = ()
    dynamic: Tuple[DynamicDim, ...] = ()
    fork: Tuple[Tuple[str, Any], ...] = ()
    class_dims: Tuple[Tuple[str, int], ...] = ()
    specialization: Dict[str, Any] = field(default_factory=dict)
    lora_fqns: Tuple[str, ...] = ()
    lifted_inputs: Tuple[str, ...] = ()
    strict: bool = True
    source_ref: str = ""
    source_digest: str = ""

    def execution_lane_label(self) -> str:
        return _execution_lane_label(self.weight_lane, self.lora_bucket)

InputBuilder = Callable[[Any, ExportSpec], Tuple[Tuple[Any, ...], Dict[str, Any]]]

#: Keyed by ``(family, target)`` — NOT by family. A family's compile
#: targets are unrelated modules with unrelated call contracts: wan's span the
#: denoiser AND the VAE, and a family-only key made ``vae.decode`` unmintable for
#: EVERY family, not just wan. ``target=""`` registers a family-wide fallback.
#: This registry is the ESCAPE HATCH while a family's declaration is being
#: written; a registered export DECLARATION always wins over it.
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

        def declared(module: Any, spec: ExportSpec) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
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


def lifted_lora_values(module: Any, spec: ExportSpec) -> Dict[str, Any]:
    """The mandatory ``lora_a``/``lora_b`` call values for a LoRA-bucket mint,
    keyed by parameter NAME — the builder binds them to their POSITIONAL slots
    (all-positional example feeds are a mint obligation: a kwarg-traced package
    arms and then silently revokes to eager on first call).

    The adapter rides as a flat 1-D pair with static per-layer offsets, passed
    as CALL ARGUMENTS. The pair must be present at trace time —
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
    spec: ExportSpec,
    *,
    input_name: str = "sample",
    vae_scale: int = 8,
    downsamples: int = 8,
) -> Tuple[DynamicDim, ...]:
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


__all__ = [
    "ADAPTER_FORK",
    "DynamicDim",
    "ExportSpec",
    "InputContractError",
    "MintRefused",
    "builder_for",
    "inputs_for",
    "latent_dims",
    "lifted_lora_values",
    "module_dtype_device",
]
