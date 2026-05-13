"""@inference_function(models=...) decorator-time validation.

Beyond Dispatch-specific checks (covered in test_dispatch_validation.py),
the decorator runs three other validation passes:

1. Rejects payload structs that use the reserved `_models` field name.
2. Validates `resources.vram_scales_with` + `resources.runtime_scales_with`
   reference real payload fields.
3. Rejects `models[param_name]` for params that don't exist on the function.
"""

import msgspec
import pytest

from gen_worker import Repo, RequestContext, Resources, inference_function


class _Pipe:
    pass


class _ReservedPayload(msgspec.Struct):
    prompt: str
    _models: dict = {}


class _BasicPayload(msgspec.Struct):
    prompt: str


class _WithWidthPayload(msgspec.Struct):
    prompt: str
    width: int = 1024


class _WithSpecsPayload(msgspec.Struct):
    prompt: str
    specs: list = []


class _Out(msgspec.Struct):
    ok: bool


def test_payload_with_reserved_models_field_is_rejected() -> None:
    with pytest.raises(ValueError, match="_models"):

        @inference_function(resources=Resources(requires_gpu=True, min_vram_gb=4.0))
        def fn(ctx: RequestContext, payload: _ReservedPayload) -> _Out:
            return _Out(ok=True)


def test_vram_scales_with_unknown_field_rejected() -> None:
    with pytest.raises(ValueError, match="unknown_payload_field"):

        @inference_function(
            resources=Resources(
                requires_gpu=True,
                min_vram_gb=4.0,
                vram_scales_with=("not_a_field",),
            ),
        )
        def fn(ctx: RequestContext, payload: _WithWidthPayload) -> _Out:
            return _Out(ok=True)


def test_runtime_scales_with_unknown_field_rejected() -> None:
    with pytest.raises(ValueError, match="unknown_payload_field"):

        @inference_function(
            resources=Resources(runtime_scales_with=("nope",)),
        )
        def fn(ctx: RequestContext, payload: _BasicPayload) -> _Out:
            return _Out(ok=True)


def test_scales_with_dotted_path_first_segment_validated() -> None:
    """Dotted paths like 'specs[0].scheme' validate only the head ('specs')."""

    @inference_function(
        resources=Resources(vram_scales_with=("specs[0].scheme",)),
    )
    def fn(ctx: RequestContext, payload: _WithSpecsPayload) -> _Out:
        return _Out(ok=True)


def test_models_param_must_exist_on_signature() -> None:
    with pytest.raises(ValueError, match="parameter that does not exist"):

        @inference_function(
            resources=Resources(requires_gpu=True, min_vram_gb=4.0),
            models={"not_a_param": Repo("x/y")},
        )
        def fn(ctx: RequestContext, payload: _BasicPayload) -> _Out:
            return _Out(ok=True)


def test_models_value_must_be_binding_instance() -> None:
    with pytest.raises(TypeError, match="Repo or Dispatch"):

        @inference_function(
            resources=Resources(requires_gpu=True, min_vram_gb=4.0),
            models={"pipeline": "not-a-binding"},  # type: ignore[dict-item]
        )
        def fn(ctx: RequestContext, pipeline: _Pipe, payload: _BasicPayload) -> _Out:
            return _Out(ok=True)


def test_bare_inference_function_works() -> None:
    """A bare @inference_function with no models / no resources still works."""

    @inference_function
    def fn(ctx: RequestContext, payload: _BasicPayload) -> _Out:
        return _Out(ok=True)

    assert getattr(fn, "_is_inference_function") is True
    assert getattr(fn, "__gen_worker_bindings__") == {}
