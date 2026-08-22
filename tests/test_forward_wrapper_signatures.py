"""A wrapper installed onto a module's ``forward`` must present the signature it wrapped — because that signature is what decides whether the module is COMPILED."""

from __future__ import annotations

import inspect
from typing import Any, Dict, Tuple

import torch

from gen_worker._vendor.torchcg import CallIngress, CallInput
from gen_worker.graphs.adopt import AdoptSession, _forward_parameters
from gen_worker.graphs.document import (
    GraphRecord,
    GraphSetDocument,
    LaneGraphs,
)
from gen_worker.models import oom_ladder

SM = "sm_89"
STACK: Tuple[Tuple[str, str], ...] = (("torch", "2.13.0"),)
#: The v2 stamp PAIR (pgw#1621). `tiny.plain-bf16@1` was ONE handle and
#: `LaneGraphs` refuses one now. The pair is a REAL corpus lane rather than a
#: fixture invention, and it is sd1.5's on purpose: `PARAMETERS` below is the
#: real sd1.5 unet's ingress, so the lane and the signature under test name
#: the same denoiser.
LANE = "sd15.diffusers@1+plain.bf16@1"
TARGET = "unet"
GRAPH = "cg-graph-v1-" + "c" * 56

PARAMETERS = ("sample", "timestep", "encoder_hidden_states", "return_dict")


class Unet(torch.nn.Module):
    """A denoiser-shaped module: NAMED forward parameters, as every real one has."""

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Any = None,
        encoder_hidden_states: Any = None,
        return_dict: bool = True,
    ) -> torch.Tensor:
        return sample


class Pipe:
    """The pipeline surface ``oom_ladder`` needs, and nothing else."""

    def __init__(self) -> None:
        self.unet = Unet()
        self.vae = None

    def enable_attention_slicing(self, *args: Any, **kwargs: Any) -> None:
        return None


def document() -> GraphSetDocument:
    ingress = CallIngress(
        parameters=PARAMETERS,
        flat_arity=1,
        inputs=(CallInput("sample", 0, "sample", 0, (), "sample", "float32", (1, 4)),),
    )
    return GraphSetDocument(
        stack=STACK,
        lanes=(
            LaneGraphs(
                contract=LANE,
                targets=(TARGET,),
                graphs=(GraphRecord(graph=GRAPH, target=TARGET, ingress=ingress),),
            ),
        ),
    )


def session(tmp_path: Any) -> AdoptSession:
    return AdoptSession(
        None, document(), LANE, SM,
        artifacts_dir=tmp_path / "adopted", stack=STACK,
    )


def test_the_oom_ladder_leaves_the_denoiser_claimable(tmp_path: Any) -> None:
    """The real `install`, on the real seam."""
    pipe = Pipe()
    before = _forward_parameters(pipe.unet)
    assert set(PARAMETERS) <= before

    armed = oom_ladder.install(pipe)
    assert armed.get("attention_sliced_retry") is True, "the ladder must still arm"

    after = _forward_parameters(pipe.unet)
    assert after == before, "the wrapper must present the signature it wrapped"

    live = session(tmp_path)
    live.adopt(pipe.unet)
    assert [hole.record.graph for hole in live.holes] == [GRAPH]


def test_the_ladder_is_still_installed_and_still_wraps(tmp_path: Any) -> None:
    """Carrying the signature must not have quietly turned the wrapper into a no-op — a fence that passes by removing the feature is worse than the bug."""
    pipe = Pipe()
    original = pipe.unet.forward
    oom_ladder.install(pipe)
    assert pipe.unet.forward is not original
    assert pipe.unet.forward.__module__ == oom_ladder.__name__
    out = pipe.unet(torch.zeros(1, 4))
    assert out.shape == torch.Size((1, 4))


def test_a_signature_erasing_wrapper_is_what_switches_adoption_off(
    tmp_path: Any,
) -> None:
    pipe = Pipe()
    inner = pipe.unet.forward

    def erased(*args: Any, **kwargs: Any) -> Any:
        return inner(*args, **kwargs)

    pipe.unet.forward = erased  # type: ignore[method-assign]
    assert _forward_parameters(pipe.unet) == frozenset()

    live = session(tmp_path)
    live.adopt(pipe.unet)
    assert live.adopted == ()
    assert live.holes == (), "not even mint work — the record was never claimed"
    assert len(live.unclaimed) == 1


def test_carrying_a_signature_survives_a_forward_that_has_none(tmp_path: Any) -> None:
    """A wrapped object with no introspectable signature must not fail a load."""
    marker: Dict[str, Any] = {}

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return None

    oom_ladder._carry_signature(wrapper, len)
    marker["ok"] = True
    assert marker["ok"]

    class Exploding:
        @property
        def __signature__(self) -> Any:
            raise ZeroDivisionError("a descriptor can raise anything")

        def __call__(self) -> None:
            return None

    oom_ladder._carry_signature(wrapper, Exploding())
    assert callable(wrapper)


def test_the_predicate_this_fence_uses_is_torchcgs_own() -> None:
    """The rule is IMPORTED, never restated here — so it moves with its owner.

    tcg#90 moved that owner: the claiming half of adoption is pgw's now
    (`gen_worker.graphs.adopt`), while torchcg keeps `program -> keyed artifact`.
    What the fence protects is unchanged — that this predicate is not a second
    copy of the parameter-kind rule.
    """
    module = inspect.getmodule(_forward_parameters)
    assert module is not None
    assert module.__name__ == "gen_worker.graphs.adopt", module.__name__


def test_the_erased_signature_reaches_a_READER_and_names_the_remedy(
    tmp_path: Any,
) -> None:
    """The half that makes the red arm above actionable instead of merely red."""
    pipe = Pipe()
    inner = pipe.unet.forward

    def erased(*args: Any, **kwargs: Any) -> Any:
        return inner(*args, **kwargs)

    pipe.unet.forward = erased  # type: ignore[method-assign]
    live = session(tmp_path)
    live.adopt(pipe.unet)

    assert live.silently_eager() is True, "two zeros, and NOT nothing-to-do"
    (mark,) = live.unclaimed_marks
    assert mark.module == "Unet"
    assert mark.parameters == ()
    described = mark.describe()
    assert "accepts NO named parameters" in described
    assert "__signature__" in described, "the remedy is in the message"


def test_the_real_ladder_leaves_no_unclaimed_mark_behind(tmp_path: Any) -> None:
    """And with the fix in place the instrument stays quiet on the real path."""
    pipe = Pipe()
    oom_ladder.install(pipe)
    live = session(tmp_path)
    live.adopt(pipe.unet)

    assert live.unclaimed_marks == ()
    assert live.silently_eager() is False
    assert [hole.record.graph for hole in live.holes] == [GRAPH]
