"""Worker-owned input planning and export guards retained across the TCG cut."""

from __future__ import annotations

from typing import Any

import pytest

from gen_worker import aot_declaration, aot_inputs, aot_mint
from gen_worker.api.decorators import Compile
from gen_worker.api.export_contract import Dim, Fork, GraphClass

torch: Any = pytest.importorskip("torch")


def test_dotted_targets_preserve_each_bound_signature() -> None:
    class Vae(torch.nn.Module):  # type: ignore[misc]
        def forward(self, value: Any) -> Any:
            return value

        def decode(self, latent: Any, scale: float = 1.0) -> Any:
            return latent * scale

        def encode(self, pixels: Any) -> Any:
            return pixels

    owner = Vae()
    decode = aot_mint._CallableTarget(owner, "decode")
    encode = aot_mint._CallableTarget(owner, "encode")

    decode_names = aot_mint._input_names(decode, (torch.zeros(1),), {})
    assert decode_names == ("latent",)
    assert 0 in aot_mint.dynamic_shapes_spec(
        (aot_inputs.DynamicDim("latent", 0, 2, 8),), decode_names,
    )["latent"]
    assert aot_mint._input_names(encode, (torch.zeros(1),), {}) == ("pixels",)


def test_batch_is_a_declared_class_coordinate_not_a_guidance_guess() -> None:
    declaration = Compile(
        family="pgw1270-g2",
        targets=("unet",),
        text_len=77,
        dims=(Dim("B", carried_by=(("x", 0),)),),
        forks=(Fork("cfg", served=(True, False)),),
        classes=(
            GraphClass(dims={"B": 2}, fork={"cfg": True}),
            GraphClass(dims={"B": 1}, fork={"cfg": False}),
        ),
        shape_strategy="static-rows",
        warm_changes_key=False,
    )

    batched = aot_declaration.select_plan(
        declaration, "unet", fork={"cfg": True},
    )
    unbatched = aot_declaration.select_plan(
        declaration, "unet", fork={"cfg": False},
    )
    assert batched.seed.dim_map["B"] == 2
    assert unbatched.seed.dim_map["B"] == 1


class _GenuineDependence(torch.nn.Module):  # type: ignore[misc]
    def forward(self, x: Any, tokens: Any) -> Any:
        batch, channels, height, width = x.shape
        return x.reshape(batch, channels, height * width) + tokens.reshape(1, 1, -1)


class _NumericCoincidence(torch.nn.Module):  # type: ignore[misc]
    def forward(self, x: Any, tokens: Any) -> Any:
        return x.flatten(2).sum(-1) + tokens.sum()


def _spatial_export(module: Any) -> Any:
    height, width = 16, 24
    return torch.export.export(
        module.eval(),
        (torch.randn(2, 4, height, width), torch.randn(height * width)),
        {},
        dynamic_shapes={
            "x": {
                2: torch.export.Dim("h", min=8, max=32),
                3: torch.export.Dim("w", min=8, max=32),
            },
            "tokens": None,
        },
        strict=True,
    )


_SPATIAL_DIMS = (
    aot_inputs.DynamicDim("x", 2, 8, 32),
    aot_inputs.DynamicDim("x", 3, 8, 32),
)


def test_range_gate_separates_real_dependence_from_numeric_coincidence() -> None:
    gaps = aot_mint.declared_range_gaps(
        _spatial_export(_GenuineDependence()), _SPATIAL_DIMS,
    )
    assert gaps
    assert "PINS" in " ".join(gaps)
    assert aot_mint.declared_range_gaps(
        _spatial_export(_NumericCoincidence()), _SPATIAL_DIMS,
    ) == []


def test_range_gate_does_not_treat_model_widths_as_spatial_dependence() -> None:
    class Sdxlish(torch.nn.Module):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(2048, 4)

        def forward(self, sample: Any, context: Any, pooled: Any) -> Any:
            return sample.flatten(2).sum(-1) + self.linear(context).sum() + pooled.sum()

    program = torch.export.export(
        Sdxlish().eval(),
        (
            torch.randn(2, 4, 16, 16),
            torch.randn(2, 77, 2048),
            torch.randn(2, 1280),
        ),
        {},
        dynamic_shapes={
            "sample": {
                2: torch.export.Dim("h", min=8, max=32),
                3: torch.export.Dim("w", min=8, max=32),
            },
            "context": None,
            "pooled": None,
        },
        strict=True,
    )
    dims = (
        aot_inputs.DynamicDim("sample", 2, 8, 32),
        aot_inputs.DynamicDim("sample", 3, 8, 32),
    )
    assert aot_mint.declared_range_gaps(program, dims) == []


def test_range_gate_names_a_declared_input_the_program_does_not_take() -> None:
    program = torch.export.export(
        _NumericCoincidence().eval(),
        (torch.randn(2, 4, 16, 24), torch.randn(384)),
        strict=True,
    )
    gaps = aot_mint.declared_range_gaps(
        program, (aot_inputs.DynamicDim("missing", 0, 2, 8),),
    )
    assert gaps
    assert "not a user input" in " ".join(gaps)
