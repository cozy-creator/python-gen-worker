"""The rank command wire: what rank 0 may say to a follower, and in what bytes.

**Why this module exists.** The command channel's writer is rank 0 — the
compute process that imports TENANT endpoint code and marshals tenant-supplied
model-call arguments into every RUN. Its readers are the D−1 followers.
Unpickling that queue makes every follower a deserialization gadget driven by
tenant-reachable input, so the wire is msgspec msgpack and the
decoder never names a class the payload chose: commands decode as a CLOSED
tagged union, and a RUN's leaves decode as plain msgpack data (scalars, lists,
maps, bytes). There is no constructor reachable from a payload.

**Tensors ride this queue, so msgspec alone does not close it.** The model call
is the SPMD unit and its arguments are latents, embeddings and masks. Tensors
therefore travel as **safetensors** bytes — the org's mandated tensor container,
already a hard dependency, and not a second serialization format: msgspec owns
the envelope, safetensors owns the tensor.

**The value vocabulary is CLOSED.** msgpack scalars, list, tuple, str-keyed
dict, `torch.Tensor` and `torch.Generator` — that is all. Anything else is a
typed refusal raised on rank 0 at send time, with the group still coherent and
the followers still parked at ``queue.get``. Under pickle the vocabulary was
open by construction (a PIL image, a numpy array or an endpoint's own class
crossed silently), and "it pickled" is not the same question as "every rank can
rebuild it identically".

A `torch.Generator` is not crossable at all and is carried as its seed: the
seed is the only part that has to agree, and every rank rebuilds an identical
generator on its own device.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple, Union, cast

import msgspec

from .plan import BootPlan, GroupPlan

# The one key that distinguishes an envelope from a plain value. A model-call
# argument that happens to use it as a dict key is refused rather than
# escaped — an escape scheme is a second grammar to get wrong.
_TAG = "__sp__"
_VAL = "v"


class UncrossableArgument(TypeError):
    """A model-call argument outside the wire's value vocabulary."""


class Arm(msgspec.Struct, tag="arm", tag_field="op"):
    """Build the SAME pipeline, then obey THIS plan. Sent once per group."""

    boot: BootPlan
    plan: GroupPlan


class Run(msgspec.Struct, tag="run", tag_field="op"):
    """One model call, marshalled. Leaves are wire values, never objects."""

    args: Tuple[Any, ...] = ()
    kwargs: Dict[str, Any] = {}


class Close(msgspec.Struct, tag="close", tag_field="op"):
    """Teardown. The only way a follower is supposed to exit."""


Command = Union[Arm, Run, Close]

_ENCODER = msgspec.msgpack.Encoder()
_DECODER = msgspec.msgpack.Decoder(Command)


def encode(command: Command) -> bytes:
    return _ENCODER.encode(command)


def decode(raw: bytes) -> Command:
    """Bytes -> one of exactly three commands. Never anything else."""
    return cast(Command, _DECODER.decode(raw))


# ---------------------------------------------------------------------------
# values
# ---------------------------------------------------------------------------


def marshal(value: Any, *, path: str = "") -> Any:
    """One call argument -> msgpack-native tagged data, or refuse by name."""
    import torch

    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return value
    if isinstance(value, torch.Tensor):
        from safetensors.torch import save

        return {_TAG: "tensor",
                _VAL: save({"t": value.detach().to("cpu").contiguous()})}
    if isinstance(value, torch.Generator):
        return {_TAG: "generator", _VAL: int(value.initial_seed())}
    if isinstance(value, tuple):
        return {_TAG: "tuple",
                _VAL: [marshal(v, path=f"{path}[{i}]")
                       for i, v in enumerate(value)]}
    if isinstance(value, list):
        return [marshal(v, path=f"{path}[{i}]") for i, v in enumerate(value)]
    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise UncrossableArgument(
                    f"{path or 'argument'}: dict key {key!r} is "
                    f"{type(key).__name__}, and only str keys cross the rank "
                    "boundary")
            if key == _TAG:
                raise UncrossableArgument(
                    f"{path or 'argument'}: {_TAG!r} is the wire's own tag key "
                    "and cannot be a model-call dict key")
            out[key] = marshal(item, path=f"{path}.{key}" if path else key)
        return out
    raise UncrossableArgument(
        f"{path or 'argument'} is a {type(value).__name__}, which cannot cross "
        "the rank boundary. Rank commands carry msgpack scalars, lists, "
        "tuples, str-keyed dicts, torch tensors (as safetensors) and "
        "torch.Generator (as its seed) — a closure, a callback or a live "
        "handle cannot be broadcast to follower ranks")


def unmarshal(value: Any, *, device: str) -> Any:
    """Wire data -> the argument, on THIS rank's device."""
    import torch

    if isinstance(value, dict):
        tag = value.get(_TAG)
        if tag is None:
            return {k: unmarshal(v, device=device) for k, v in value.items()}
        payload = value.get(_VAL)
        if tag == "tensor" and isinstance(payload, bytes):
            from safetensors.torch import load

            return load(payload)["t"].to(device)
        if tag == "generator" and isinstance(payload, int):
            return torch.Generator(device=device).manual_seed(payload)
        if tag == "tuple" and isinstance(payload, list):
            return tuple(unmarshal(v, device=device) for v in payload)
        raise UncrossableArgument(
            f"malformed wire envelope: tag={tag!r} payload="
            f"{type(payload).__name__}")
    if isinstance(value, list):
        return [unmarshal(v, device=device) for v in value]
    return value


def run_command(args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Run:
    """Marshal a model call. Raises `UncrossableArgument` naming the argument
    — on rank 0, before anything is queued."""
    return Run(
        args=tuple(marshal(a, path=f"args[{i}]") for i, a in enumerate(args)),
        kwargs={k: marshal(v, path=k) for k, v in kwargs.items()},
    )


def run_call(command: Run, *, device: str) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
    """A follower's side of `run_command`."""
    args: List[Any] = [unmarshal(a, device=device) for a in command.args]
    kwargs = {k: unmarshal(v, device=device) for k, v in command.kwargs.items()}
    return tuple(args), kwargs
