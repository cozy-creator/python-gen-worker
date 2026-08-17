"""Toy family declarations for the pgw#1332 suite.

Two of them, because the two halves of the vocabulary fail differently:
:data:`TOY_DIFFUSION` is a staged composition (a counted denoiser then a
decoder) and :data:`TOY_AR` is a host-owned loop (prefill then decode, session
state on the host). Both are tiny enough that a real fake-tensor export runs in
well under a second, so the suite exercises the ACTUAL export path rather than
a hand-written snapshot — which is the only way the test can catch a change in
what ``torch.export`` records.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from gen_worker.model import (
    Bucket,
    CallExample,
    GraphModelSpec,
    Loop,
    LoopKind,
    Parameter,
    Runner,
    Scheduler,
    SessionState,
    Stage,
    TunedValues,
)

WIDTH = 8


class ToyTuned(TunedValues, frozen=True):
    """The toy family's tuned schema — shape here, values from the catalog."""

    steps: int = 4
    guidance: float = 3.5


def _torch() -> Any:
    import torch

    return torch


def _denoiser(layout: str) -> Any:
    import torch
    from torch import nn

    class _Denoiser(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Linear(WIDTH, WIDTH)

        def forward(self, hidden_states: Any, timestep: Any) -> Any:
            return self.proj(hidden_states) * (1.0 + timestep.to(hidden_states.dtype))

    del torch, layout
    return _Denoiser().eval()


def _denoiser_example(bucket: Mapping[str, int], layout: str) -> CallExample:
    torch = _torch()
    del layout
    tokens = int(bucket["resolution"]) // 64
    return CallExample(
        params=("hidden_states", "timestep"),
        kwargs={
            "hidden_states": torch.zeros(1, tokens, WIDTH, dtype=torch.float32),
            "timestep": torch.zeros((), dtype=torch.float32),
        },
    )


def _decoder(layout: str) -> Any:
    from torch import nn

    class _Decoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.out = nn.Linear(WIDTH, 3)

        def forward(self, latents: Any) -> Any:
            return self.out(latents)

    del layout
    return _Decoder().eval()


def _decoder_example(bucket: Mapping[str, int], layout: str) -> CallExample:
    torch = _torch()
    del layout
    tokens = int(bucket["resolution"]) // 64
    return CallExample(
        params=("latents",),
        kwargs={"latents": torch.zeros(1, tokens, WIDTH, dtype=torch.float32)},
    )


TOY_DIFFUSION = GraphModelSpec(
    name="toy_diffusion",
    tuned=ToyTuned,
    buckets=(Bucket("resolution", (64, 128)),),
    runners=(
        Runner("decoder", build=_decoder, example=_decoder_example, axes=("resolution",),
               component="vae.decoder"),
        Runner("denoiser", build=_denoiser, example=_denoiser_example, axes=("resolution",),
               component="transformer"),
    ),
    loop=Loop(stages=(Stage("denoiser", repeat="steps"), Stage("decoder"))),
    parameters=(Parameter("steps", minimum=1, maximum=100),),
    scheduler=Scheduler("euler_discrete", {"shift": 3.0}),
)


class ToyArTuned(TunedValues, frozen=True):
    """An autoregressive family's tuned schema."""

    temperature: float = 0.7


def _step(layout: str) -> Any:
    from torch import nn

    class _Step(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.head = nn.Linear(WIDTH, WIDTH)

        def forward(self, tokens: Any) -> Any:
            return self.head(tokens)

    del layout
    return _Step().eval()


def _step_example(bucket: Mapping[str, int], layout: str) -> CallExample:
    torch = _torch()
    del layout
    return CallExample(
        params=("tokens",),
        kwargs={"tokens": torch.zeros(1, int(bucket["context"]), WIDTH, dtype=torch.float32)},
    )


TOY_AR = GraphModelSpec(
    name="toy_ar",
    tuned=ToyArTuned,
    buckets=(Bucket("context", (16, 32)),),
    runners=(
        Runner("decode", build=_step, example=_step_example, axes=("context",)),
        Runner("prefill", build=_step, example=_step_example, axes=("context",)),
    ),
    # No repeat count anywhere: the recipe vocabulary refuses one under a host
    # loop, because a fabricated bound reads to a second implementation as real.
    loop=Loop(
        stages=(Stage("prefill"), Stage("decode")),
        kind=LoopKind.HOST,
        session_state=SessionState.HOST,
    ),
)

__all__ = ["TOY_AR", "TOY_DIFFUSION", "WIDTH", "ToyArTuned", "ToyTuned"]


def toy_loaded_tree() -> Any:
    """A stand-in for what the LOADER produces on a real pod.

    Not a mock of the SDK: these are the declaration's own modules, built by the
    declaration's own `build` callables, arranged the way a diffusers-style
    pipeline arranges them — a `.transformer` and a `.vae.decoder`. That shape
    is the whole point, because `Runner.component` is what maps a runner onto
    it, and a test that hand-built the map would not be testing the map.
    """

    from types import SimpleNamespace

    return SimpleNamespace(
        transformer=_denoiser("bf16"),
        vae=SimpleNamespace(decoder=_decoder("bf16")),
    )
