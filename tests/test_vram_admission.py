"""VRAM admission for cold dynamic model picks.

Live SDXL loaded WAI, Cyber and Nova before a never-seen Realism pick. The
fixed VAE had a small prior hint, so the old sum-only estimate admitted the
unknown checkpoint as though only that VAE were incoming. Admission must use
the endpoint's full declared requirement whenever any setup ref is unknown.
"""

from __future__ import annotations

import asyncio

import msgspec
import pytest

torch = pytest.importorskip("torch")

from gen_worker.api.binding import HF
from gen_worker.api.decorators import Resources
from gen_worker.executor import Executor, ModelStore
from gen_worker.models import residency as residency_mod
from gen_worker.models.residency import Tier
from gen_worker.registry import EndpointSpec

_GiB = 1024 ** 3


class _In(msgspec.Struct):
    prompt: str = "test"


class _Pipe:
    def to(self, device: str) -> "_Pipe":
        return self


class _Endpoint:
    def setup(self, pipeline: _Pipe, vae: _Pipe) -> None:  # pragma: no cover
        self.pipeline = pipeline
        self.vae = vae

    def run(self, ctx, payload: _In):  # pragma: no cover
        return payload


def _spec() -> EndpointSpec:
    return EndpointSpec(
        name="generate", method=_Endpoint.run, kind="inference",
        payload_type=_In, output_mode="single", cls=_Endpoint,
        attr_name="run",
        models={"pipeline": HF("tensorhub/realism"),
                "vae": HF("madebyollin/sdxl-vae-fp16-fix")},
        resources=Resources(vram_gb=12),
    )


def _executor(spec: EndpointSpec, tmp_path) -> Executor:
    async def _send(message) -> None:
        return None

    store = ModelStore(
        _send, cache_dir=tmp_path, vram_budget_bytes=24 * _GiB,
    )
    return Executor([spec], _send, store=store)


def test_partial_hint_uses_full_declared_requirement(
        tmp_path, monkeypatch) -> None:
    spec = _spec()
    ex = _executor(spec, tmp_path)
    res = ex.store.residency
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(residency_mod, "get_available_ram_gb", lambda: 128.0)

    # Three warm picks plus the known companion VAE leave only 5GiB free.
    # A 1GiB-only ask would fit; the honest 12GiB + margin ask must evict.
    for ref in ("tensorhub/wai", "tensorhub/cyber", "tensorhub/nova"):
        res.track_vram(ref, _Pipe(), vram_bytes=6 * _GiB)
    res.track_vram(
        "madebyollin/sdxl-vae-fp16-fix", _Pipe(), vram_bytes=1 * _GiB,
    )

    asyncio.run(ex._make_room_for(spec, ["pipeline", "vae"]))

    assert res.tier("tensorhub/wai") is Tier.RAM
    assert res.tier("tensorhub/cyber") is Tier.RAM
    assert res.tier("tensorhub/nova") is Tier.VRAM
    assert res.tier("madebyollin/sdxl-vae-fp16-fix") is Tier.VRAM
