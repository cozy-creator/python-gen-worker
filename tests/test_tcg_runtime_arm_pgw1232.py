from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, cast

import pytest
from torch_compiled_graphs import CallIngress, CallInput

from gen_worker import aot_serve, compiled_graph_store, fleet_cells, shape_growth
from gen_worker.models import provision


KEY = "cg-key-v1-" + "1" * 56


class Tensor:
    dtype = "torch.float32"

    def __init__(self, shape: tuple[int, ...] = (2,)) -> None:
        self.shape = shape


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
    ingress = CallIngress(
        parameters=("x",),
        flat_arity=1,
        inputs=(CallInput(
            name="x",
            position=0,
            param="x",
            param_position=0,
            path=(),
            exported_name="x",
            dtype="float32",
            shape=(2,),
        ),),
    )
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
                "v": 3,
                "lifted_inputs": [],
                "pytree": {
                    "user_inputs": ["x"],
                    "in_spec": "leaf",
                    "out_spec": "leaf",
                    "ingress": ingress.as_dict(),
                },
                "specialization": {},
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
    assert aot_serve.is_armed(pipeline) is True
    assert aot_serve.armed_entries(pipeline) == {"denoiser-b2": KEY}
    assert pipeline.denoiser.forward(Tensor()) == "compiled"
    assert runner.calls == 1
    assert pipeline.denoiser.eager_calls == 0


def test_provisioning_arms_only_by_exact_compiled_graph_key(
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

    outcome = provision.enable_compiled(
        pipeline,
        SimpleNamespace(family="micro", lora_bucket=0),
        tmp_path,
        KEY,
    )

    assert outcome.armed is True
    assert pipeline.denoiser.forward(Tensor()) == "compiled"


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


def test_unavailable_or_quarantined_exact_key_never_reaches_a_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = Module()
    original = module.forward
    pipeline = SimpleNamespace(denoiser=module)
    monkeypatch.setattr(
        compiled_graph_store,
        "load_runner",
        lambda key, root=None: None,
    )

    outcome = aot_serve.enable_compiled_graph(
        pipeline,
        SimpleNamespace(family="micro", lora_bucket=0),
        KEY,
        tmp_path,
    )

    assert outcome.armed is False
    assert outcome.reason == "compiled_graph_unavailable"
    assert module.forward == original
    assert not hasattr(pipeline, aot_serve._MARKER_ATTR)


def test_out_of_envelope_call_falls_back_without_disarming_tcg_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = SimpleNamespace(denoiser=Module())
    runner = Runner()
    gaps: list[shape_growth.ShapeGap] = []
    monkeypatch.setattr(
        compiled_graph_store,
        "load_runner",
        lambda key, root=None: _loaded(runner),
    )
    monkeypatch.setattr(shape_growth, "report_and_submit", gaps.append)

    outcome = aot_serve.enable_compiled_graph(
        pipeline,
        SimpleNamespace(family="micro", lora_bucket=0),
        KEY,
        tmp_path,
    )

    assert outcome.armed is True
    assert pipeline.denoiser.forward(Tensor((3,))) == "eager"
    assert pipeline.denoiser.eager_calls == 1
    assert runner.calls == 0
    assert aot_serve.is_armed(pipeline) is True
    assert aot_serve.ingress_refusals(pipeline) == 1
    assert len(gaps) == 1

    assert pipeline.denoiser.forward(Tensor()) == "compiled"
    assert runner.calls == 1


def test_non_key_is_refused_before_store_lookup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        compiled_graph_store,
        "load_runner",
        lambda *_args, **_kwargs: pytest.fail("invalid key reached store lookup"),
    )

    outcome = aot_serve.enable_compiled_graph(
        SimpleNamespace(denoiser=Module()),
        SimpleNamespace(family="micro", lora_bucket=0),
        str(tmp_path / "legacy-cell.tar.gz"),
        tmp_path,
    )

    assert outcome.armed is False
    assert outcome.reason == "compiled_graph_key_invalid"


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
