"""Dispatch decoration-time validation.

The `@inference_function(models={...})` decorator validates Dispatch bindings
against the payload struct's static type:

- `dispatch.field` must name a real payload field.
- That field must be `Literal[...]`-typed (Optional[Literal[...]] OK).
- Every key in `dispatch.table` must be a member of that Literal.
"""

from typing import Literal

import msgspec
import pytest

from gen_worker import Repo, RequestContext, Resources, dispatch, inference_function


class _Pipe:
    pass


class _LiteralPayload(msgspec.Struct):
    variant: Literal["a", "b"]
    prompt: str


class _PlainStrPayload(msgspec.Struct):
    variant: str
    prompt: str


class _OptionalLiteralPayload(msgspec.Struct):
    variant: "Literal['a', 'b'] | None" = None
    prompt: str = ""


class _DispatchPayload(msgspec.Struct):
    variant: Literal["nf4", "int8"]
    prompt: str


class _Out(msgspec.Struct):
    ok: bool


def test_dispatch_validates_field_exists_on_payload() -> None:
    with pytest.raises(ValueError, match="no such field"):

        @inference_function(
            resources=Resources(requires_gpu=True, min_vram_gb=4.0),
            models={
                "pipeline": dispatch(
                    field="not_a_field",
                    table={"a": Repo("x/y")},
                ),
            },
        )
        def fn(ctx: RequestContext, pipeline: _Pipe, payload: _LiteralPayload) -> _Out:
            return _Out(ok=True)


def test_dispatch_field_must_be_literal() -> None:
    with pytest.raises(ValueError, match="Literal"):

        @inference_function(
            resources=Resources(requires_gpu=True, min_vram_gb=4.0),
            models={
                "pipeline": dispatch(
                    field="variant",
                    table={"a": Repo("x/y")},
                ),
            },
        )
        def fn(ctx: RequestContext, pipeline: _Pipe, payload: _PlainStrPayload) -> _Out:
            return _Out(ok=True)


def test_dispatch_table_keys_must_be_literal_members() -> None:
    with pytest.raises(ValueError, match="not a member"):

        @inference_function(
            resources=Resources(requires_gpu=True, min_vram_gb=4.0),
            models={
                "pipeline": dispatch(
                    field="variant",
                    table={"a": Repo("x/y"), "c": Repo("x/y")},  # "c" not in Literal
                ),
            },
        )
        def fn(ctx: RequestContext, pipeline: _Pipe, payload: _LiteralPayload) -> _Out:
            return _Out(ok=True)


def test_dispatch_optional_literal_accepted() -> None:
    """`Literal[...] | None` is acceptable — discriminator can be optional."""

    @inference_function(
        resources=Resources(requires_gpu=True, min_vram_gb=4.0),
        models={
            "pipeline": dispatch(
                field="variant",
                table={"a": Repo("x/y"), "b": Repo("x/y").flavor("nf4")},
            ),
        },
    )
    def fn(ctx: RequestContext, pipeline: _Pipe, payload: _OptionalLiteralPayload) -> _Out:
        return _Out(ok=True)

    bindings = getattr(fn, "__gen_worker_bindings__")
    assert "pipeline" in bindings


def test_dispatch_happy_path() -> None:
    @inference_function(
        resources=Resources(requires_gpu=True, min_vram_gb=14.0),
        models={
            "pipeline": dispatch(
                field="variant",
                table={
                    "nf4": Repo("acme/r").flavor("nf4"),
                    "int8": Repo("acme/r").flavor("int8"),
                },
            ).allow_override(_Pipe),
        },
    )
    def fn(ctx: RequestContext, pipeline: _Pipe, payload: _DispatchPayload) -> _Out:
        return _Out(ok=True)

    bindings = getattr(fn, "__gen_worker_bindings__")
    binding = bindings["pipeline"]
    assert binding._allow_override is True
    assert binding._pipeline_classes == ("test_dispatch_validation._Pipe",)
    assert binding.field == "variant"
