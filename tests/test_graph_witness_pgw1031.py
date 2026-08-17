"""TCG alone derives graph witnesses and graph-class identity."""

from __future__ import annotations

import inspect
from typing import Any

import pytest
from gen_worker._vendor.torchcg import CallIngress, CallInput, GraphClassSpec

from gen_worker import aot_mint, boot_key, boot_trace_child
from gen_worker.aot_inputs import ExportSpec

torch: Any = pytest.importorskip("torch")


class Operation(torch.nn.Module):  # type: ignore[misc]
    def __init__(self, *, add: bool = False, weight: float = 2.0) -> None:
        super().__init__()
        self.add = add
        self.weight = torch.nn.Parameter(torch.tensor(weight))

    def forward(self, value: Any) -> Any:
        return value + self.weight if self.add else value * self.weight


def _program(*, add: bool = False, weight: float = 2.0) -> Any:
    return torch.export.export(
        Operation(add=add, weight=weight),
        (torch.ones(2),),
    )


def _graph() -> dict[str, Any]:
    ingress = CallIngress(
        parameters=("value",),
        flat_arity=1,
        inputs=(
            CallInput(
                "value", 0, "value", 0, (), "value", "float32", (2,),
            ),
        ),
    )
    return {
        "v": 3,
        "lifted_inputs": [],
        "pytree": {
            "in": "leaf",
            "out": "leaf",
            "ingress": ingress.as_dict(),
        },
        "specialization": {},
    }


def _spec(program: Any) -> GraphClassSpec:
    return GraphClassSpec("model", "denoiser", program, _graph())


def test_tcg_witness_ignores_weight_values_and_separates_graph_bodies() -> None:
    first = _spec(_program(weight=2.0)).declare()
    fine_tune = _spec(_program(weight=9.0)).declare()
    changed_body = _spec(_program(add=True)).declare()

    assert first == fine_tune
    assert first.graph_witness == fine_tune.graph_witness
    assert first.class_hash == fine_tune.class_hash
    assert first.graph_witness != changed_body.graph_witness
    assert first.class_hash != changed_body.class_hash


def test_boot_and_compile_share_one_worker_to_tcg_translation() -> None:
    program = _program()
    traced = aot_mint.TracedClass(
        name="model",
        block={
            "target": "denoiser",
            "fork": [],
            "class_dims": [],
            "graph": _graph(),
        },
        nodes=len(program.graph_module.graph.nodes),
        program=program,
        declared=1,
    )
    export_spec = ExportSpec(family="tiny", target="denoiser")

    shared = aot_mint.tcg_graph_class_spec(traced, export_spec).declare()
    direct = _spec(program).declare()
    assert shared == direct

    wire = boot_key.serialize_declaration(shared)
    assert boot_key.declaration_hashes({"model": wire}) == {
        "model": shared.class_hash,
    }


def test_boot_wire_reconstructs_every_public_tcg_fact() -> None:
    declaration = GraphClassSpec(
        "nested/model",
        "denoiser",
        _program(),
        _graph(),
        fork=(("adapter", True),),
        class_dims=(("height", 64),),
        strict=True,
        lora_bucket=128,
    ).declare()
    wire = boot_key.serialize_declaration(declaration)

    assert boot_key.declaration_hashes({"nested/model": wire}) == {
        "nested/model": declaration.class_hash,
    }


def test_boot_child_has_no_worker_graph_identity_implementation() -> None:
    source = inspect.getsource(boot_trace_child.run)
    assert "tcg_graph_class_spec" in source
    assert ".declare()" in source
    assert "serialize_declaration" in source
    assert "graph_hash" not in source
    assert "stamp_entry" not in source
