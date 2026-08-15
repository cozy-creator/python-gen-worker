"""Adapter fan-out remains worker planning; TCG receives the resulting classes."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest

from gen_worker import aot_declaration, aot_inputs, aot_mint
from gen_worker.api.decorators import Compile
from gen_worker.api.export_contract import Dim, GraphClass, Input

torch: Any = pytest.importorskip("torch")

_BUCKET = 16


class _TinyDenoiser(torch.nn.Module):  # type: ignore[misc]
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(_BUCKET, _BUCKET)

    def forward(self, sample: Any) -> Any:
        return self.linear(sample)


def _declaration() -> Compile:
    return Compile(
        family="pgw1270-adapter-plans",
        targets=("unet",),
        dims=(Dim("B", carried_by=(("sample", 0),)),),
        classes=(GraphClass(dims={"B": 2}),),
        inputs=(Input("sample", shape=("B", _BUCKET), dtype="model"),),
        shape_strategy="static-rows",
        warm_changes_key=False,
    )


def _spec(bucket: int) -> aot_inputs.ExportSpec:
    return aot_inputs.ExportSpec(
        family="pgw1270-adapter-plans",
        target="",
        lora_bucket=bucket,
    )


def test_bucket_family_forks_branch_capable_target_adapter_first() -> None:
    plans = aot_declaration.cell_plans(_declaration())
    rows = aot_mint.adapter_arm_plans(
        plans, SimpleNamespace(unet=_TinyDenoiser().eval()), _spec(_BUCKET),
    )

    assert [arm for _plan, arm in rows] == [True, False]
    assert [aot_declaration.plan_entry_name(plan) for plan, _arm in rows] == [
        "unet/adapter=true/B=2",
        "unet/adapter=false/B=2",
    ]


def test_bucket_zero_family_does_not_fork() -> None:
    rows = aot_mint.adapter_arm_plans(
        aot_declaration.cell_plans(_declaration()),
        SimpleNamespace(unet=_TinyDenoiser().eval()),
        _spec(0),
    )

    assert [arm for _plan, arm in rows] == [None]
    assert aot_declaration.plan_entry_name(rows[0][0]) == "unet/B=2"


def test_non_branch_target_does_not_fork_in_a_bucket_family() -> None:
    plans = tuple(
        replace(plan, target="vae")
        for plan in aot_declaration.cell_plans(_declaration())
    )
    rows = aot_mint.adapter_arm_plans(
        plans,
        SimpleNamespace(
            unet=_TinyDenoiser().eval(),
            vae=_TinyDenoiser().eval(),
        ),
        _spec(_BUCKET),
    )

    assert [arm for _plan, arm in rows] == [None]
