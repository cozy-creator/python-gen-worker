"""pgw#1370: the lanes-only @endpoint surface (code + lanes, nothing else)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from gen_worker import Endpoint
from gen_worker.api.decorators import ATTR, endpoint


class _In:
    pass


def test_lanes_class_records_the_declaration() -> None:
    @endpoint(lanes=("sdxl.diffusers-bf16@1",))
    class Ep(Endpoint):
        def setup(self, ctx) -> None: ...
        def generate(self, ctx, payload: _In) -> _In: ...

    decl = getattr(Ep, ATTR)
    assert decl.lanes == ("sdxl.diffusers-bf16@1",)
    assert [attr for attr, _ in Ep.__gen_worker_handlers__] == ["generate"]


def test_a_contract_object_is_a_valid_lane() -> None:
    contract = SimpleNamespace(name="cozy.sdxl-fp8-rowwise", version=1)

    @endpoint(lanes=(contract,))
    class Ep(Endpoint):
        def generate(self, ctx, payload: _In) -> _In: ...

    assert getattr(Ep, ATTR).lanes == (contract,)


def test_lanes_do_not_compose_with_the_catalog_surface() -> None:
    from gen_worker import HF

    with pytest.raises(ValueError, match="does not compose"):
        @endpoint(lanes=("sdxl.diffusers-bf16@1",), model=HF("org/repo"))
        class Ep(Endpoint):
            def setup(self, ctx) -> None: ...
            def generate(self, ctx, payload: _In) -> _In: ...


def test_lanes_require_a_class() -> None:
    with pytest.raises(TypeError, match="class"):
        @endpoint(lanes=("sdxl.diffusers-bf16@1",))
        def handler(ctx, payload: _In) -> _In: ...


def test_lanes_require_the_endpoint_base() -> None:
    with pytest.raises(TypeError, match="gen_worker.Endpoint"):
        @endpoint(lanes=("sdxl.diffusers-bf16@1",))
        class Ep:
            def generate(self, ctx, payload: _In) -> _In: ...


def test_a_misshapen_setup_hook_refuses() -> None:
    with pytest.raises(TypeError, match="setup"):
        @endpoint(lanes=("sdxl.diffusers-bf16@1",))
        class Ep(Endpoint):
            def setup(self, ctx, extra) -> None: ...
            def generate(self, ctx, payload: _In) -> _In: ...


def test_duplicate_lane_contracts_refuse() -> None:
    with pytest.raises(ValueError, match="unique"):
        @endpoint(lanes=("sdxl.diffusers-bf16@1", "sdxl.diffusers-bf16@1"))
        class Ep(Endpoint):
            def generate(self, ctx, payload: _In) -> _In: ...


def test_a_lane_that_is_no_contract_reference_refuses() -> None:
    with pytest.raises(ValueError, match="contract"):
        @endpoint(lanes=(object(),))
        class Ep(Endpoint):
            def generate(self, ctx, payload: _In) -> _In: ...
