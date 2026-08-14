from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, cast

import pytest

from gen_worker import aot_serve, compiled_graph_store, fleet_cells


KEY = "cg-key-v1-" + "1" * 56


class Tensor:
    shape = (2,)
    dtype = "torch.float32"


class Module:
    device = "cpu"

    def __init__(self) -> None:
        self.eager_calls = 0

    def forward(self, _x: Tensor) -> str:
        self.eager_calls += 1
        return "eager"

    def state_dict(self) -> dict[str, object]:
        return {}

    def named_buffers(self) -> tuple[()]:
        return ()


class Runner:
    key = KEY
    graph_class = "denoiser-b2"
    declared_fqns: tuple[str, ...] = ()

    def __init__(self, *, bind_error: Exception | None = None) -> None:
        self.bound = False
        self.calls = 0
        self.bind_error = bind_error
        self.bound_state: Mapping[str, Any] | None = None

    def bind(self, state: Mapping[str, Any], *, device: str) -> None:
        assert device == "cpu"
        if self.bind_error is not None:
            raise self.bind_error
        self.bound_state = state
        self.bound = True

    def __call__(self, *_feeds: object) -> str:
        assert self.bound
        self.calls += 1
        return "compiled"


def _loaded(runner: Runner) -> compiled_graph_store.LoadedCompiledGraph:
    metadata = {
        "compiled_graph_format": 1,
        "compiled_graph_key": KEY,
        "sm": "cpu-x86_64-v1",
        "toolchain": {"torch": "test"},
        "graph_class": {
            "name": "denoiser-b2",
            "target": "denoiser",
            "class_hash": "1" * 16,
            "graph": {
                "inputs": [{
                    "name": "x",
                    "position": 0,
                    "dtype": "float32",
                    "shape": [2],
                }],
                "symbols": {},
                "excluded_inputs": [],
            },
        },
    }
    graph = SimpleNamespace(key=KEY, metadata=metadata)
    return compiled_graph_store.LoadedCompiledGraph(graph, runner)  # type: ignore[arg-type]


def test_tcg_runner_binds_before_the_first_live_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = SimpleNamespace(denoiser=Module())
    runner = Runner()
    monkeypatch.setattr(
        compiled_graph_store,
        "load_runner",
        lambda key, root=None: _loaded(runner),
    )

    outcome = aot_serve.enable_compiled_graph(
        pipeline,
        SimpleNamespace(family="micro", lora_bucket=0),
        KEY,
        tmp_path,
    )

    assert outcome.armed is True
    assert runner.bound is True
    assert pipeline.denoiser.forward(Tensor()) == "compiled"
    assert runner.calls == 1
    assert pipeline.denoiser.eager_calls == 0


def test_failed_tcg_bind_leaves_the_pipeline_eager(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from torch_compiled_graphs import ConstantBindingError

    module = Module()
    original = module.forward
    pipeline = SimpleNamespace(denoiser=module)
    runner = Runner(
        bind_error=ConstantBindingError("constant_unresolved", "missing weight")
    )
    monkeypatch.setattr(
        compiled_graph_store,
        "load_runner",
        lambda key, root=None: _loaded(runner),
    )

    outcome = aot_serve.enable_compiled_graph(
        pipeline,
        SimpleNamespace(family="micro", lora_bucket=0),
        KEY,
        tmp_path,
    )

    assert outcome.armed is False
    assert outcome.reason == "constant_unresolved"
    assert module.forward == original
    assert module.forward(Tensor()) == "eager"


def test_self_mint_stage_uses_the_childs_exact_tcg_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = tmp_path / f"{KEY}.tar.gz"
    artifact.write_bytes(b"owned by TCG")
    calls: list[tuple[Path, str]] = []

    def store(path: Path, *, key: str, **_kwargs: object) -> object:
        calls.append((path, key))
        return SimpleNamespace(compiled_graph_key=key)

    monkeypatch.setattr(fleet_cells, "no_publish_sink_reason", lambda _sink: "")
    monkeypatch.setattr(compiled_graph_store, "store", store)

    pending = SimpleNamespace(
        family="micro",
        arm_token="arm-v1-test",
        publisher=SimpleNamespace(),
    )
    assert fleet_cells._stage_durable(cast(Any, pending), artifact) == KEY
    assert calls == [(artifact, KEY)]

    wrong_name = tmp_path / "compiled-graph.tar.gz"
    wrong_name.write_bytes(b"not addressed")
    assert fleet_cells._stage_durable(cast(Any, pending), wrong_name) == ""
    assert calls == [(artifact, KEY)]
