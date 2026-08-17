"""The worker relies on TCG's public declaration for literal identity."""

from __future__ import annotations

from typing import Any

import pytest
from gen_worker._vendor.torchcg import (
    DeclarationError,
    GraphClassSpec,
    build_call_ingress,
)

torch: Any = pytest.importorskip("torch")


class _LiteralModule(torch.nn.Module):  # type: ignore[misc]
    def __init__(self, literal: Any) -> None:
        super().__init__()
        self.literal = literal

    def forward(self, value: Any) -> Any:
        return value + self.literal


def _exported(literal: Any, value: Any) -> tuple[Any, tuple[Any, ...]]:
    args = (value,)
    return torch.export.export(_LiteralModule(literal), args), args


def _spec(program: Any, args: tuple[Any, ...]) -> GraphClassSpec:
    ingress = build_call_ingress(program, ("value",), args, {})
    return GraphClassSpec(
        graph_class="model/literal",
        target="denoiser",
        program=program,
        graph={
            "v": 3,
            "lifted_inputs": [],
            "pytree": {"in": "", "out": "", "ingress": ingress.as_dict()},
            "specialization": {},
        },
    )


def test_public_tcg_declaration_refuses_unreadable_literals() -> None:
    program, args = _exported(torch.ones(4), torch.ones(4))
    (name,) = program.graph_signature.lifted_tensor_constants
    original = program.constants.pop(name)
    with pytest.raises(DeclarationError, match=f"{name!r} carries no value"):
        _spec(program, args).declare()

    class Hostile:
        dtype = "weird"
        shape = (1,)

        def detach(self) -> Any:
            raise RuntimeError("no bytes")

    program.constants[name] = Hostile()
    with pytest.raises(DeclarationError, match=f"{name!r} could not be digested"):
        _spec(program, args).declare()
    program.constants[name] = original


def test_public_tcg_declaration_digests_complex64_literals() -> None:
    angles = torch.arange(4, dtype=torch.float32)
    literal = torch.polar(torch.ones_like(angles), angles).to(torch.complex64)
    program, args = _exported(literal, torch.ones(4, dtype=torch.complex64))

    declaration = _spec(program, args).declare()

    assert len(declaration.literal_values) == 32
    assert declaration.graph["literal_values"] == declaration.literal_values
    assert declaration.graph["constant_fqns"] == ["literal"]
