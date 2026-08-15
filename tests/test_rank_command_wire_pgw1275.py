"""The rank command channel is not a deserialization gadget (pgw#1275).

The channel's writer is rank 0 — the compute process that imports TENANT
endpoint code and marshals tenant-supplied model-call arguments — and its
readers are the spawned rank siblings. While ``FollowerChannel.next_command``
unpickled, every follower executed whatever the bytes on that queue described.

Every row here drives the REAL channel: a real ``spawn`` process reading a real
``mp.Queue`` through the production ``FollowerChannel``. The full formed group
(TCPStore rendezvous, gloo process group, arm/run/close over the same wire) is
exercised by ``test_sp_group_isolation_pgw773_774.py``.

RED before the fix: row 1's marker file appears, because the poisoned payload's
``__reduce__`` runs inside the follower.
"""

from __future__ import annotations

import multiprocessing as mp
import pickle
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from gen_worker.parallel import wire  # noqa: E402
from gen_worker.parallel.group import FollowerChannel  # noqa: E402


def _touch(path: str) -> str:
    """The payload's effect. It proves EXECUTION and does nothing harmful."""
    Path(path).write_text("executed", encoding="utf-8")
    return path


class _Poisoned:
    """A tenant-shaped object whose mere DESERIALIZATION runs code."""

    def __init__(self, marker: str) -> None:
        self.marker = marker

    def __reduce__(self) -> Any:
        return (_touch, (self.marker,))


def _read_one(channel: FollowerChannel, out: Any) -> None:  # pragma: no cover - spawned
    try:
        out.put(("decoded", type(channel.next_command(timeout=60)).__name__))
    except BaseException as exc:  # noqa: BLE001 — the refusal is the result
        out.put(("refused", type(exc).__name__))


def _echo_one(channel: FollowerChannel, out: Any) -> None:  # pragma: no cover - spawned
    """A follower's real decode, reported back as plain describable facts."""
    try:
        cmd = channel.next_command(timeout=60)
        assert isinstance(cmd, wire.Run), cmd
        args, kwargs = wire.run_call(cmd, device="cpu")
        out.put(("ok", _describe(args), _describe(kwargs)))
    except BaseException as exc:  # noqa: BLE001
        out.put(("raised", f"{type(exc).__name__}: {exc}", None))


def _describe(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return ("tensor", str(value.dtype), tuple(value.shape),
                value.flatten().tolist())
    if isinstance(value, torch.Generator):
        return ("generator", int(value.initial_seed()))
    if isinstance(value, tuple):
        return ("tuple", [_describe(v) for v in value])
    if isinstance(value, list):
        return ("list", [_describe(v) for v in value])
    if isinstance(value, dict):
        return ("dict", {k: _describe(v) for k, v in value.items()})
    return value


def _drive(target: Any, raw: bytes) -> Any:
    """Put RAW bytes on a real command queue and let a real follower read."""
    ctx = mp.get_context("spawn")
    channel = FollowerChannel(commands=ctx.Queue(), ready=ctx.Queue())
    out = ctx.Queue()
    proc = ctx.Process(target=target, args=(channel, out), daemon=True)
    proc.start()
    try:
        channel.commands.put(raw)
        result = out.get(timeout=120)
    finally:
        proc.join(timeout=30)
        if proc.is_alive():  # pragma: no cover - the follower must exit
            proc.kill()
            proc.join(timeout=10)
    return result


def test_a_pickle_on_the_command_queue_executes_nothing(tmp_path: Path) -> None:
    """RED before pgw#1275: `next_command` did `pickle.loads(raw)`, so this
    marker was written by the follower. The payload is exactly what rank 0
    would have produced from a hostile model-call argument."""
    marker = tmp_path / "executed-in-the-follower"
    outcome, detail = _drive(_read_one, pickle.dumps(_Poisoned(str(marker))))

    assert not marker.exists(), (
        "a pickle payload on the rank command queue EXECUTED inside the "
        "follower — the channel is a deserialization gadget again"
    )
    assert outcome == "refused", (outcome, detail)


def test_the_wire_carries_tensors_generators_and_nesting() -> None:
    """What actually flows: model-call arguments, tensors included. The
    ledger's "msgspec closes it for free" is half the story — a tensor is not
    msgpack-native, so it travels as safetensors bytes and must arrive
    bit-identical, with tuple/list nesting preserved (a pipeline that gets a
    list where rank 0 had a tuple is a silent divergence)."""
    latents = torch.arange(8, dtype=torch.float32).reshape(2, 4) / 3.0
    mask = torch.tensor([[True, False]], dtype=torch.bool)
    half = torch.ones(3, dtype=torch.bfloat16)
    gen = torch.Generator().manual_seed(1234)

    command = wire.run_command(
        (latents, "a prompt", 30),
        {
            "generator": gen,
            "sizes": (512, 512),
            "steps": [1, 2, 3],
            "extra": {"mask": mask, "guidance": 3.5, "half": half},
            "flags": (True, None, b"raw"),
        },
    )
    outcome, described_args, described_kwargs = _drive(
        _echo_one, wire.encode(command))
    assert outcome == "ok", described_args

    assert described_args == ("tuple", [
        ("tensor", "torch.float32", (2, 4), latents.flatten().tolist()),
        "a prompt",
        30,
    ])
    kind, kwargs = described_kwargs
    assert kind == "dict"
    assert kwargs["generator"] == ("generator", 1234)
    # tuple stays a tuple, list stays a list — msgpack has one array type, so
    # this is a property of the wire, not of msgpack.
    assert kwargs["sizes"] == ("tuple", [512, 512])
    assert kwargs["steps"] == ("list", [1, 2, 3])
    assert kwargs["flags"] == ("tuple", [True, None, b"raw"])
    assert kwargs["extra"] == ("dict", {
        "mask": ("tensor", "torch.bool", (1, 2), [True, False]),
        "guidance": 3.5,
        "half": ("tensor", "torch.bfloat16", (3,), [1.0, 1.0, 1.0]),
    })


class _TenantObject:
    """Picklable, and therefore silently crossable before pgw#1275."""

    def __init__(self) -> None:
        self.value = 1


@pytest.mark.parametrize(
    "argument",
    [
        pytest.param(_TenantObject(), id="an-endpoint-class"),
        pytest.param(lambda *a: None, id="a-callback-closure"),
        pytest.param({1: "int-key"}, id="a-non-str-dict-key"),
        pytest.param({"__sp__": "shadowing-the-tag"}, id="the-wire-tag-key"),
    ],
)
def test_an_argument_outside_the_vocabulary_is_refused_by_name(
    argument: Any,
) -> None:
    """The vocabulary is CLOSED, and the refusal names the argument. Only the
    closure was refused before — a plain picklable object crossed silently,
    which is the open-by-construction half of the defect."""
    with pytest.raises(wire.UncrossableArgument) as excinfo:
        wire.run_command((), {"callback": argument})
    assert "callback" in str(excinfo.value)


def test_every_command_decodes_to_its_own_closed_type() -> None:
    """A payload never names a class, so it can never reach a constructor:
    the decoder admits exactly three shapes and nothing else."""
    from gen_worker.parallel.plan import BootPlan, GroupPlan

    arm = wire.Arm(boot=BootPlan(function_name="f", degree=2),
                   plan=GroupPlan(sp_degree=2, loras=(("l", 0.5),)))
    assert wire.decode(wire.encode(arm)) == arm
    assert isinstance(wire.decode(wire.encode(wire.Close())), wire.Close)
    assert isinstance(wire.decode(wire.encode(wire.Run())), wire.Run)

    import msgspec

    with pytest.raises(msgspec.ValidationError):
        wire.decode(msgspec.msgpack.encode({"op": "exec", "code": "boom"}))
