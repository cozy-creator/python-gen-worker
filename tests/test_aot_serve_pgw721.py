"""Worker policy around TCG-owned compiled graph runners."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

from gen_worker import aot_serve as aot
from torch_compiled_graphs import CallIngress, CallInput, CompiledGraphRunner

FAMILY = "sdxl-base"
KEY = "cg-key-v1-" + "7" * 56
ENTRY = "unet/h=128,w=128"


class FakeTensor:
    def __init__(self, shape: tuple[int, ...], dtype: str = "torch.bfloat16") -> None:
        self.shape = shape
        self.dtype = dtype


class FakeTCGRunner:
    def __init__(self, *, raises: str = "") -> None:
        self.bound = False
        self.calls = 0
        self.declared_fqns = ("weight",)
        self.bound_fqns: tuple[str, ...] = ()
        self.raises = raises
        self.binds: list[tuple[dict[str, Any], str]] = []

    def bind(self, state: dict[str, Any], *, device: str) -> None:
        self.binds.append((dict(state), device))
        self.bound_fqns = tuple(sorted(state))
        self.bound = True

    def __call__(self, *feeds: object) -> str:
        if self.raises:
            raise RuntimeError(self.raises)
        self.calls += 1
        return "COMPILED"


class Module:
    device = "cpu"

    def __init__(self) -> None:
        self.weight = FakeTensor((1,))
        self.eager_calls = 0

    def state_dict(self) -> dict[str, Any]:
        return {"weight": self.weight}

    def named_buffers(self) -> tuple[()]:
        return ()

    def forward(self, *_args: object, **_kwargs: object) -> str:
        self.eager_calls += 1
        return "EAGER"


class Pipeline:
    def __init__(self) -> None:
        self.unet = Module()


class Cfg:
    family = FAMILY
    lora_bucket = 0


def contract() -> CallIngress:
    return CallIngress(
        parameters=("sample", "timestep"),
        flat_arity=2,
        inputs=(
            CallInput(
                "sample", 0, "sample", 0, (), "sample", "bfloat16",
                (2, 4, "h", "w"),
            ),
            CallInput(
                "timestep", 1, "timestep", 1, (), "timestep", "int64", (),
            ),
        ),
        symbols=(("h", (64, 160)), ("w", (64, 160))),
    )


def graph_metadata() -> dict[str, object]:
    return {
        "compiled_graph_format": 1,
        "kind": "aot-inductor",
        "compiled_graph_key": KEY,
        "sm": "cpu-x86_64-v1",
        "toolchain": {"torch": "test"},
        "host_isa": {"machine": "x86_64", "march": "", "simdlen": "0", "level": ""},
        "graph_class": {
            "name": ENTRY,
            "target": "unet",
            "class_hash": "1" * 32,
            "graph": {
                "pytree": {"ingress": contract().as_dict()},
                "constant_fqns": ["weight"],
            },
        },
    }


def entry_runner(raw: FakeTCGRunner | None = None) -> tuple[aot.TCGEntryRunner, FakeTCGRunner]:
    runner = raw or FakeTCGRunner()
    wrapped = aot.TCGEntryRunner(
        cast(CompiledGraphRunner, runner), contract(), "unet", ENTRY, FAMILY
    )
    return wrapped, runner


def in_range(h: int = 128, w: int = 128) -> tuple[FakeTensor, FakeTensor]:
    return FakeTensor((2, 4, h, w)), FakeTensor((), "torch.int64")


def install(pipe: Pipeline, wrapped: aot.TCGEntryRunner) -> None:
    dispatch = aot.EntryDispatch(declared=(ENTRY,))
    dispatch.add(ENTRY, wrapped)
    aot.wrap_module(pipe.unet, dispatch, {"family": FAMILY, "compiled_graph_key": KEY})
    setattr(pipe, aot._MARKER_ATTR, {
        "meta": {"family": FAMILY, "compiled_graph_key": KEY},
        "targets": {"unet": {
            "module": pipe.unet,
            "attr": "forward",
            "state": getattr(pipe.unet, aot._MARKER_ATTR)["state"],
        }},
        "entries": {ENTRY: {"compiled_graph_key": KEY, "target": "unet"}},
        "bound_constants": {"pools": {}, "literals": {}},
    })


def test_unbound_tcg_runner_never_crosses_the_aoti_boundary() -> None:
    wrapped, raw = entry_runner()
    with pytest.raises(aot.ConstantsUnboundError) as excinfo:
        wrapped(*in_range())
    assert excinfo.value.reason == "constants_unbound"
    assert raw.calls == 0


def test_out_of_contract_call_falls_back_without_disarming() -> None:
    pipe = Pipeline()
    wrapped, raw = entry_runner()
    raw.bind(pipe.unet.state_dict(), device="cpu")
    install(pipe, wrapped)

    assert pipe.unet.forward(*in_range()) == "COMPILED"
    assert pipe.unet.forward(*in_range(256, 256)) == "EAGER"
    assert aot.entry_states(pipe)[ENTRY] == {
        "state": "armed", "target": "unet", "calls": 1,
    }
    assert raw.calls == 1
    assert pipe.unet.eager_calls == 1


def test_runner_failure_dearms_only_its_entry_and_serves_eager() -> None:
    pipe = Pipeline()
    wrapped, raw = entry_runner(FakeTCGRunner(raises="kernel failed"))
    raw.bind(pipe.unet.state_dict(), device="cpu")
    install(pipe, wrapped)

    assert pipe.unet.forward(*in_range()) == "EAGER"
    assert pipe.unet.forward(*in_range()) == "EAGER"
    assert aot.entry_states(pipe)[ENTRY]["state"] == "de_armed"
    assert pipe.unet.eager_calls == 2


def test_exact_tcg_key_is_the_aot_ref_identity() -> None:
    ref = f"root/family-{FAMILY}#{KEY}"
    assert aot.is_aot_ref(ref, family=FAMILY)
    assert not aot.is_aot_ref(ref, family="other")
    assert not aot.is_aot_ref(f"root/family-{FAMILY}#aot-l4")


def test_arm_resolves_and_loads_the_same_destination_then_binds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = FakeTCGRunner()
    calls: list[tuple[str, str, Path]] = []

    class Stored:
        metadata = graph_metadata()

    class Engine:
        def resolve(self, key: str, destination: Path) -> Stored:
            calls.append(("resolve", key, Path(destination)))
            return Stored()

        def runner(self, key: str, destination: Path) -> CompiledGraphRunner:
            calls.append(("runner", key, Path(destination)))
            return cast(CompiledGraphRunner, raw)

    monkeypatch.setattr(aot, "open_worker_engine", lambda _root=None: Engine())
    pipe = Pipeline()
    original = pipe.unet.forward

    meta = aot.arm_compiled_graph(pipe, Cfg(), KEY, tmp_path)

    assert calls[0][0] == "resolve" and calls[1][0] == "runner"
    assert calls[0][1:] == calls[1][1:]
    assert raw.binds == [({"weight": pipe.unet.weight}, "cpu")]
    assert pipe.unet.forward is not original
    assert meta["compiled_graph_key"] == KEY
    assert aot.armed_entries(pipe) == {ENTRY: KEY}
    assert aot.entry_states(pipe)[ENTRY]["state"] == "armed"


def test_failed_bind_does_not_mutate_the_live_module(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Refusing(FakeTCGRunner):
        def bind(self, state: dict[str, Any], *, device: str) -> None:
            from torch_compiled_graphs import ConstantBindingError

            raise ConstantBindingError("constant_unresolved", "weight absent")

    raw = Refusing()

    class Stored:
        metadata = graph_metadata()

    class Engine:
        def resolve(self, _key: str, _destination: Path) -> Stored:
            return Stored()

        def runner(self, _key: str, _destination: Path) -> CompiledGraphRunner:
            return cast(CompiledGraphRunner, raw)

    monkeypatch.setattr(aot, "open_worker_engine", lambda _root=None: Engine())
    pipe = Pipeline()
    original = pipe.unet.forward

    with pytest.raises(Exception, match="weight absent"):
        aot.arm_compiled_graph(pipe, Cfg(), KEY, tmp_path)

    assert pipe.unet.forward == original
    assert not hasattr(pipe, aot._MARKER_ATTR)
