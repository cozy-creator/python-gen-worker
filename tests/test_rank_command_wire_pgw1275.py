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
    Path(path).write_text("executed", encoding="utf-8")
    return path


class _Poisoned:

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
    marker = tmp_path / "executed-in-the-follower"
    outcome, detail = _drive(_read_one, pickle.dumps(_Poisoned(str(marker))))

    assert not marker.exists(), (
        "a pickle payload on the rank command queue EXECUTED inside the "
        "follower — the channel is a deserialization gadget again"
    )
    assert outcome == "refused", (outcome, detail)


def test_the_wire_carries_tensors_generators_and_nesting() -> None:
    """What actually flows: model-call arguments, tensors included."""
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
    assert kwargs["sizes"] == ("tuple", [512, 512])
    assert kwargs["steps"] == ("list", [1, 2, 3])
    assert kwargs["flags"] == ("tuple", [True, None, b"raw"])
    assert kwargs["extra"] == ("dict", {
        "mask": ("tensor", "torch.bool", (1, 2), [True, False]),
        "guidance": 3.5,
        "half": ("tensor", "torch.bfloat16", (3,), [1.0, 1.0, 1.0]),
    })


class _TenantObject:

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
    """The vocabulary is CLOSED, and the refusal names the argument."""
    with pytest.raises(wire.UncrossableArgument) as excinfo:
        wire.run_command((), {"callback": argument})
    assert "callback" in str(excinfo.value)


def test_every_command_decodes_to_its_own_closed_type() -> None:
    """A payload never names a class, so it can never reach a constructor: the decoder admits exactly three shapes and nothing else."""
    from gen_worker.parallel.plan import BootPlan, GroupPlan

    arm = wire.Arm(boot=BootPlan(function_name="f", degree=2),
                   plan=GroupPlan(sp_degree=2, loras=(("l", 0.5),)))
    assert wire.decode(wire.encode(arm)) == arm
    assert isinstance(wire.decode(wire.encode(wire.Close())), wire.Close)
    assert isinstance(wire.decode(wire.encode(wire.Run())), wire.Run)

    import msgspec

    with pytest.raises(msgspec.ValidationError):
        wire.decode(msgspec.msgpack.encode({"op": "exec", "code": "boom"}))
