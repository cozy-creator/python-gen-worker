"""Generic export-input machinery for the AOT mint.

``torch.export`` needs example inputs with the exact structure the target's
forward takes. That structure is FAMILY knowledge — and per Paul's SDK-generic
rule it is a DECLARATION in the endpoint spec, never worker code: the endpoint
declares ``Compile(dims=..., forks=..., classes=..., inputs=...)`` and
:func:`gen_worker.aot_declaration.declared_inputs` derives the example inputs
generically. What lives here is the minimal worker-owned input vocabulary,
lifted-LoRA values, and dtype/device introspection. Pipeline composition belongs
to the compile child and serving loader, not this contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

from .compile_cache import execution_lane_label as _execution_lane_label
from .models import lora_lifted
from .api.export_contract import export_declaration

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
    # Measurement/reporting only; TCG graph metadata and identity do not carry
    # a second worker precision stamp.
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

class InputContractError(RuntimeError):
    """No declared export-input contract exists for a family."""


def builder_for(family: str, target: str = "") -> InputBuilder:
    """Build inputs from the family's authored export declaration."""

    fam, tgt = str(family), str(target)
    decl = export_declaration(fam)
    if decl is None or not decl.inputs:
        raise InputContractError(
            f"no export-input contract for {fam!r} target {tgt!r}: the "
            "endpoint must DECLARE Compile(inputs=...) before export"
        )
    from . import aot_declaration

    def declared(
        module: Any, spec: ExportSpec,
    ) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        return aot_declaration.declared_inputs(module, spec, decl)

    return declared


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


__all__ = [
    "ADAPTER_FORK",
    "DynamicDim",
    "ExportSpec",
    "InputContractError",
    "MintRefused",
    "builder_for",
    "lifted_lora_values",
    "module_dtype_device",
]
