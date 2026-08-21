"""Static re-specialization of a symbolic parent program (tcg#88, pgw#1603).

The derive exports ONE symbolic parent per structural variant and stamps
static buckets by binding concrete shapes to it — and the program STORE banks
that parent under every bucket's graph identity (the CAS dedups the bytes to
one blob, which is what collapses 28 near-identical serialized programs into
a handful). So a compile position asking for a STATIC record can be handed a
SYMBOLIC program: this module re-specializes it at the compile seam, from the
record's own ingress, and refuses to proceed unless the re-derived program
restates the EXACT graph identity it was asked for. The bind is not a guess:
pgw#1603's spike proved it byte-identical to a direct static export (hash and
ingress, real and fake mode, torch 2.13).
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any

from .graph_identity import graph_hash

if TYPE_CHECKING:
    from .declaration import GraphSpecialization


class BindError(RuntimeError):
    """A symbolic parent cannot restate the static record it was banked for."""


def _fake_mode_of(module: Any) -> Any:
    from torch._subclasses.fake_tensor import FakeTensor

    modes = {
        parameter.fake_mode
        for parameter in module.parameters()
        if isinstance(parameter, FakeTensor)
    }
    if len(modes) > 1:
        raise BindError(
            "the parent program holds fake parameters from two fake modes; "
            "it cannot be re-exported under one"
        )
    return modes.pop() if modes else None


def _placeholder_values(program: Any) -> dict[str, Any]:
    return {
        node.name: (node.meta or {}).get("val")
        for node in program.graph_module.graph.nodes
        if node.op == "placeholder"
    }


def respecialize(program: Any, ingress: Any, *, strict: bool = False) -> Any:
    """Bind ``ingress``'s concrete shapes to a symbolic parent program.

    The call is rebuilt from the parent's own signature: non-tensor leaves
    come back as the ``ConstantArgument`` values the trace recorded, tensor
    leaves as zeros of the INGRESS's concrete shape in the parent's dtype and
    device — exactly the synthesis discovery derives from, so the bound
    program is the one a direct static export would have produced.
    """

    import contextlib

    import torch
    from torch.export.graph_signature import ConstantArgument, InputKind
    from torch.utils import _pytree as pytree

    rows = {row.exported_name: row for row in ingress.inputs}
    values = _placeholder_values(program)
    leaves: list[Any] = []
    fake_mode = _fake_mode_of(program.graph_module)
    synthesis = fake_mode if fake_mode is not None else contextlib.nullcontext()
    with synthesis:
        for spec in program.graph_signature.input_specs:
            if spec.kind != InputKind.USER_INPUT:
                continue
            if isinstance(spec.arg, ConstantArgument):
                leaves.append(spec.arg.value)
                continue
            name = str(getattr(spec.arg, "name", "") or "")
            row = rows.get(name)
            template = values.get(name)
            if row is None or template is None:
                raise BindError(
                    f"the record's ingress names no input for the parent's "
                    f"placeholder {name!r}; the parent cannot restate this "
                    f"record"
                )
            if any(not isinstance(d, int) for d in row.shape):
                raise BindError(
                    f"ingress input {row.name!r} still carries a symbol in "
                    f"{row.shape!r}; a static bind needs concrete shapes"
                )
            leaves.append(
                torch.zeros(
                    tuple(int(d) for d in row.shape),
                    dtype=template.dtype,
                    device=template.device,
                )
            )
        args, kwargs = pytree.tree_unflatten(leaves, program.call_spec.in_spec)
        return torch.export.export(
            program.module(), args, kwargs, strict=strict
        )


def bind_static_spec(spec: GraphSpecialization) -> GraphSpecialization:
    """The compile-seam gate: a symbolic program at a static position binds.

    No-op when the program is already static, or when the record itself is
    dynamic (its ingress states symbols — the range program IS the compile
    input there). Otherwise the parent is re-specialized and the result must
    hash to EXACTLY the graph identity this position was asked to mint;
    anything else is refused, never compiled under the wrong name.
    """

    if not getattr(spec.program, "range_constraints", None):
        return spec
    if spec.ingress.symbols:
        return spec
    bound = respecialize(spec.program, spec.ingress)
    derived = graph_hash(bound, spec.ingress)
    if derived != spec.name:
        raise BindError(
            f"binding the symbolic parent to this record's shapes derives "
            f"{derived}, not the requested graph {spec.name!r}. Either the "
            f"banked parent does not cover this record, or the record was "
            f"hashed with transform passes this seam cannot restate — "
            f"refusing rather than minting under the wrong identity."
        )
    return replace(spec, program=bound)


_DIAGNOSTIC_META = (
    "stack_trace",
    "nn_module_stack",
    "source_fn_stack",
    "torch_fn",
)


def strip_diagnostics(program: Any) -> Any:
    """Drop per-node diagnostic strings before a program is serialized.

    Measured (pgw#1603, sd15): per-node ``stack_trace``/``nn_module_stack``/
    ``torch_fn`` strings are ~60% of every serialized program — 1.8 MB of a
    3.1 MB graph JSON — and nothing on the mint path reads them: the graph
    hash renders ops, args and values, never ``node.meta``, and AOTI compiles
    without them. They are debug provenance for a human reading ONE program,
    paid on every blob forever.
    """

    for node in program.graph_module.graph.nodes:
        for key in _DIAGNOSTIC_META:
            node.meta.pop(key, None)
    return program


__all__ = ["BindError", "bind_static_spec", "respecialize", "strip_diagnostics"]
