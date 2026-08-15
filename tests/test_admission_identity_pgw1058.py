"""Declaration-side dtype identity remains explicit after TCG extraction."""

from __future__ import annotations

import msgspec
import pytest

from gen_worker.api.export_contract import (
    INPUT_DTYPES,
    MODEL_DTYPE,
    DeclarationError,
    Input,
)


def test_an_input_row_without_a_dtype_is_refused_at_declaration_time() -> None:
    with pytest.raises(DeclarationError) as excinfo:
        Input("timestep", shape=(), value=1.0)
    message = str(excinfo.value)
    assert "timestep" in message
    assert MODEL_DTYPE in message
    assert "pgw#1058" in message


def test_an_unknown_dtype_word_is_refused_by_name() -> None:
    with pytest.raises(DeclarationError, match="unknown dtype 'bfloat17'"):
        Input("timestep", shape=(), dtype="bfloat17")


def test_the_explicit_inheritance_word_constructs() -> None:
    row = Input("sample", shape=("B", 4), dtype=MODEL_DTYPE)
    assert row.dtype == MODEL_DTYPE
    assert MODEL_DTYPE in INPUT_DTYPES


def test_a_row_that_dodged_validation_is_refused_at_the_feed_builder() -> None:
    torch = pytest.importorskip("torch")
    from gen_worker.aot_declaration import declared_inputs
    from gen_worker.aot_inputs import ExportSpec, MintRefused
    from gen_worker.api.decorators import Compile
    from gen_worker.api.export_contract import Dim, GraphClass

    row = Input("timestep", shape=(), dtype=MODEL_DTYPE, value=1.0)
    msgspec.structs.force_setattr(row, "dtype", "")
    decl = Compile(
        family="pgw1058",
        targets=("unet",),
        dims=(Dim("B", carried_by=(("sample", 0),)),),
        classes=(GraphClass(dims={"B": 1}),),
        inputs=(Input("sample", shape=("B", 4), dtype=MODEL_DTYPE), row),
        shape_strategy="static-rows",
    )
    module = torch.nn.Linear(4, 4)
    spec = ExportSpec(family="pgw1058", target="unet")
    with pytest.raises(MintRefused, match="declares no dtype"):
        declared_inputs(module, spec, decl)
