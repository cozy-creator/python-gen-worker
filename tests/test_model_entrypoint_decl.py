"""pgw#1370: the Model + @entrypoint author surface (code + lanes)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from gen_worker import Model
from gen_worker.api.entrypoint import entrypoint
from gen_worker.api.model_base import model_lanes, model_model_type


class _In:
    pass


class _MT:
    class Defaults:
        pass


def test_lanes_class_kwargs_record_the_declaration() -> None:
    class M(Model[_MT], lanes=("sdxl.diffusers-bf16@1",)):
        pass

    assert model_lanes(M) == ("sdxl.diffusers-bf16@1",)
    assert model_model_type(M) is _MT


def test_a_contract_object_is_a_valid_lane() -> None:
    contract = SimpleNamespace(name="cozy.sdxl-fp8-rowwise", version=1)

    class M(Model[_MT], lanes=(contract,)):
        pass

    assert model_lanes(M) == (contract,)


def test_no_lanes_means_eager_and_states_nothing() -> None:
    class M(Model[_MT]):
        pass

    assert model_lanes(M) == ()


def test_duplicate_lane_contracts_refuse() -> None:
    with pytest.raises(ValueError, match="unique"):
        class M(Model[_MT], lanes=("sdxl.diffusers-bf16@1", "sdxl.diffusers-bf16@1")):
            pass


def test_a_lane_that_is_no_contract_reference_refuses() -> None:
    with pytest.raises(ValueError, match="contract"):
        class M(Model[_MT], lanes=(object(),)):
            pass


def test_empty_lanes_tuple_refuses() -> None:
    with pytest.raises(ValueError, match="at least one"):
        class M(Model[_MT], lanes=()):
            pass


def test_entrypoint_marks_and_shape_checks() -> None:
    @entrypoint
    def run(payload: _In, model: Any, ctx: Any) -> _In:
        return payload

    assert getattr(run, "__gen_worker_entrypoint__") is True

    with pytest.raises(TypeError, match="payload, model, ctx"):
        @entrypoint
        def wrong(payload: _In, ctx: Any) -> _In:
            return payload
