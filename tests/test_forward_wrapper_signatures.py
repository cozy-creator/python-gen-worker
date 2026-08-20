"""A wrapper installed onto a module's ``forward`` must present the signature
it wrapped — because that signature is what decides whether the module is
COMPILED.

``nn.Module.__call__`` reads ``self.forward`` as an instance attribute, so
installing a wrapper there is a legitimate and widely used seam in this repo.
But it is not only where a compiled graph is armed; it is where adoption
decides whether to arm one. ``torchcg`` claims a graph record for a
``ctx.compile``-ed module when the module's forward accepts every parameter the
record's ingress names, and it reads those names off
``inspect.signature(module.forward)``.

A bare ``*args, **kwargs`` wrapper has no named parameters at all. So it does
not degrade adoption — it switches it OFF for that module, permanently and
everywhere, and no mint can turn it back on, because nothing is missing. That
is what happened (pgw#1534): the OOM-retry ladder — armed on every rung by
pgw#1499, correctly — presented a 13-parameter unet forward as an empty set, and
a real boot with all 14 artifacts sitting in its store reported
``adopted: 0, holes: 0, armed: 0``.

The predicate here is IMPORTED from torchcg, never restated: a local
reimplementation of "which parameters count" would pass while the real claim
still failed.
"""

from __future__ import annotations

import inspect
from typing import Any, Dict, Tuple

import torch

from gen_worker._vendor.torchcg import CallIngress, CallInput
from gen_worker._vendor.torchcg.adopt import AdoptSession, _forward_parameters
from gen_worker._vendor.torchcg.document import (
    GraphRecord,
    GraphSetDocument,
    LaneGraphs,
)
from gen_worker.models import oom_ladder

SM = "sm_89"
STACK: Tuple[Tuple[str, str], ...] = (("torch", "2.13.0"),)
LANE = "tiny.plain-bf16@1"
TARGET = "unet"
GRAPH = "cg-graph-v1-" + "c" * 56

#: The real sd1.5 unet's ingress, as the derive recorded it.
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
    # No store: every claimed record becomes a HOLE, which is exactly the
    # signal wanted here — "was this record claimed at all" without needing a
    # real artifact to arm.
    return AdoptSession(
        None, document(), LANE, SM,
        artifacts_dir=tmp_path / "adopted", stack=STACK,
    )


def test_the_oom_ladder_leaves_the_denoiser_claimable(tmp_path: Any) -> None:
    """The real `install`, on the real seam. This is the regression."""
    pipe = Pipe()
    before = _forward_parameters(pipe.unet)
    assert set(PARAMETERS) <= before

    armed = oom_ladder.install(pipe)
    assert armed.get("attention_sliced_retry") is True, "the ladder must still arm"

    after = _forward_parameters(pipe.unet)
    assert after == before, "the wrapper must present the signature it wrapped"

    # And the claim itself, through torchcg's own session rather than through a
    # restatement of its rule.
    live = session(tmp_path)
    live.adopt(pipe.unet)
    assert [hole.record.graph for hole in live.holes] == [GRAPH]


def test_the_ladder_is_still_installed_and_still_wraps(tmp_path: Any) -> None:
    """Carrying the signature must not have quietly turned the wrapper into a
    no-op — a fence that passes by removing the feature is worse than the bug."""
    pipe = Pipe()
    original = pipe.unet.forward
    oom_ladder.install(pipe)
    assert pipe.unet.forward is not original
    assert pipe.unet.forward.__module__ == oom_ladder.__name__
    # It still forwards.
    out = pipe.unet(torch.zeros(1, 4))
    assert out.shape == torch.Size((1, 4))


def test_a_signature_erasing_wrapper_is_what_switches_adoption_off(
    tmp_path: Any,
) -> None:
    """THE RED ARM, built by hand: the shape the ladder had before pgw#1534.

    Nothing is missing, nothing raises, the module works — and the record is
    never claimed, so it is not even a hole. `adopted=0, holes=0`.
    """
    pipe = Pipe()
    inner = pipe.unet.forward

    def erased(*args: Any, **kwargs: Any) -> Any:  # no named parameters
        return inner(*args, **kwargs)

    # The exact assignment `oom_ladder` makes, and mypy is right that it is a
    # method assignment — that IS the seam under test.
    pipe.unet.forward = erased  # type: ignore[method-assign]
    assert _forward_parameters(pipe.unet) == frozenset()

    live = session(tmp_path)
    live.adopt(pipe.unet)
    assert live.adopted == ()
    assert live.holes == (), "not even mint work — the record was never claimed"
    assert len(live.unclaimed) == 1


def test_carrying_a_signature_survives_a_forward_that_has_none(tmp_path: Any) -> None:
    """A wrapped object with no introspectable signature must not fail a load.

    An unarmed ladder never fails a load, and neither may an unsigned one.
    """
    marker: Dict[str, Any] = {}

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return None

    # `len` is a builtin: `inspect.signature` may or may not describe it, and
    # either way this must return quietly rather than raise.
    oom_ladder._carry_signature(wrapper, len)
    marker["ok"] = True
    assert marker["ok"]

    # And a callable whose `__signature__` RAISES takes the same path.
    # `inspect.signature` runs arbitrary code through that descriptor, so the
    # exception set is not enumerable and the catch is deliberately broad.
    class Exploding:
        @property
        def __signature__(self) -> Any:
            raise ZeroDivisionError("a descriptor can raise anything")

        def __call__(self) -> None:
            return None

    oom_ladder._carry_signature(wrapper, Exploding())
    assert callable(wrapper)


def test_the_predicate_this_fence_uses_is_torchcgs_own() -> None:
    """If torchcg changes which parameter kinds count, this fence must move
    with it — so the rule is imported, never restated here."""
    module = inspect.getmodule(_forward_parameters)
    assert module is not None
    assert module.__name__.endswith("torchcg.adopt")


def test_the_erased_signature_reaches_a_READER_and_names_the_remedy(
    tmp_path: Any,
) -> None:
    """The half that makes the red arm above actionable instead of merely red.

    Before tcg#70 the state above was recorded nowhere: `_adopt_module` ended
    in a bare `return`, `adopted` and `holes` both read 0, and an operator with
    a full store had nothing to go on. The mark now carries the module, its
    (empty) signature and the remedy, so the diagnosis is one line of the boot
    log rather than an afternoon.
    """
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
