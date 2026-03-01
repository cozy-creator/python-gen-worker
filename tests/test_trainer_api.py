from __future__ import annotations

import sys
import types

import pytest

from gen_worker.trainer import StepContext, StepResult, TrainingJobSpec, load_trainer_plugin


def _ctx() -> StepContext:
    return StepContext(job=TrainingJobSpec(run_id="r", max_steps=1))


def test_load_trainer_plugin_from_class(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = types.ModuleType("tmp_trainer_class_mod")

    class MyTrainer:
        def setup(self, ctx: StepContext) -> None:
            _ = ctx

        def configure(self, ctx: StepContext) -> dict[str, object]:
            _ = ctx
            return {"step": 0}

        def prepare_batch(self, raw_batch: object, state: dict[str, object], ctx: StepContext) -> object:
            _ = state
            _ = ctx
            return raw_batch

        def train_step(self, batch: object, state: dict[str, object], ctx: StepContext) -> StepResult:
            _ = state
            _ = ctx
            return StepResult(metrics={"ok": float(batch)})

        def state_dict(self, state: dict[str, object]) -> dict[str, object]:
            return dict(state)

        def load_state_dict(self, state: dict[str, object], payload: dict[str, object], ctx: StepContext) -> None:
            _ = ctx
            state.update(payload)

    mod.MyTrainer = MyTrainer
    monkeypatch.setitem(sys.modules, "tmp_trainer_class_mod", mod)

    trainer = load_trainer_plugin("tmp_trainer_class_mod:MyTrainer")
    ctx = _ctx()
    trainer.setup(ctx)
    state = trainer.configure(ctx)
    prepared = trainer.prepare_batch(3, state, ctx)
    out = trainer.train_step(prepared, state, ctx)
    assert out.metrics["ok"] == 3.0


def test_load_trainer_plugin_from_instance(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = types.ModuleType("tmp_trainer_instance_mod")

    class MyTrainer:
        def setup(self, ctx: StepContext) -> None:
            _ = ctx

        def configure(self, ctx: StepContext) -> dict[str, object]:
            _ = ctx
            return {}

        def prepare_batch(self, raw_batch: object, state: dict[str, object], ctx: StepContext) -> object:
            _ = state
            _ = ctx
            return raw_batch

        def train_step(self, batch: object, state: dict[str, object], ctx: StepContext) -> StepResult:
            _ = state
            _ = ctx
            return StepResult(metrics={"loss": float(batch)})

        def state_dict(self, state: dict[str, object]) -> dict[str, object]:
            return state

        def load_state_dict(self, state: dict[str, object], payload: dict[str, object], ctx: StepContext) -> None:
            _ = ctx
            state.update(payload)

    mod.trainer = MyTrainer()
    monkeypatch.setitem(sys.modules, "tmp_trainer_instance_mod", mod)

    trainer = load_trainer_plugin("tmp_trainer_instance_mod:trainer")
    out = trainer.train_step(2, {}, _ctx())
    assert out.metrics["loss"] == 2.0


def test_load_trainer_plugin_rejects_plain_callable(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = types.ModuleType("tmp_trainer_callable_mod")

    def step_only(ctx: StepContext, batch: object) -> StepResult:
        _ = ctx
        return StepResult(metrics={"x": float(batch)})

    mod.step_only = step_only
    monkeypatch.setitem(sys.modules, "tmp_trainer_callable_mod", mod)

    with pytest.raises(TypeError, match="plain callables are unsupported"):
        load_trainer_plugin("tmp_trainer_callable_mod:step_only")


def test_load_trainer_plugin_rejects_missing_hooks(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = types.ModuleType("tmp_trainer_bad_class_mod")

    class BadTrainer:
        def setup(self, ctx: StepContext) -> None:
            _ = ctx

    mod.BadTrainer = BadTrainer
    monkeypatch.setitem(sys.modules, "tmp_trainer_bad_class_mod", mod)

    with pytest.raises(TypeError, match="missing configure"):
        load_trainer_plugin("tmp_trainer_bad_class_mod:BadTrainer")


def test_load_trainer_plugin_from_module_only_single_candidate(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = types.ModuleType("tmp_trainer_module_only_mod")

    class ModuleTrainer:
        def setup(self, ctx: StepContext) -> None:
            _ = ctx

        def configure(self, ctx: StepContext) -> dict[str, object]:
            _ = ctx
            return {}

        def prepare_batch(self, raw_batch: object, state: dict[str, object], ctx: StepContext) -> object:
            _ = state
            _ = ctx
            return raw_batch

        def train_step(self, batch: object, state: dict[str, object], ctx: StepContext) -> StepResult:
            _ = state
            _ = ctx
            return StepResult(metrics={"loss": float(batch)})

        def state_dict(self, state: dict[str, object]) -> dict[str, object]:
            return state

        def load_state_dict(self, state: dict[str, object], payload: dict[str, object], ctx: StepContext) -> None:
            _ = ctx
            state.update(payload)

    mod.ModuleTrainer = ModuleTrainer
    monkeypatch.setitem(sys.modules, "tmp_trainer_module_only_mod", mod)

    trainer = load_trainer_plugin("tmp_trainer_module_only_mod")
    out = trainer.train_step(5, {}, _ctx())
    assert out.metrics["loss"] == 5.0


def test_load_trainer_plugin_from_module_only_prefers_trainer_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = types.ModuleType("tmp_trainer_module_alias_mod")

    class AliasTrainer:
        def setup(self, ctx: StepContext) -> None:
            _ = ctx

        def configure(self, ctx: StepContext) -> dict[str, object]:
            _ = ctx
            return {}

        def prepare_batch(self, raw_batch: object, state: dict[str, object], ctx: StepContext) -> object:
            _ = state
            _ = ctx
            return raw_batch

        def train_step(self, batch: object, state: dict[str, object], ctx: StepContext) -> StepResult:
            _ = state
            _ = ctx
            return StepResult(metrics={"loss": float(batch)})

        def state_dict(self, state: dict[str, object]) -> dict[str, object]:
            return state

        def load_state_dict(self, state: dict[str, object], payload: dict[str, object], ctx: StepContext) -> None:
            _ = ctx
            state.update(payload)

    mod.Trainer = AliasTrainer
    monkeypatch.setitem(sys.modules, "tmp_trainer_module_alias_mod", mod)

    trainer = load_trainer_plugin("tmp_trainer_module_alias_mod")
    out = trainer.train_step(7, {}, _ctx())
    assert out.metrics["loss"] == 7.0


def test_load_trainer_plugin_from_module_only_rejects_ambiguous(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = types.ModuleType("tmp_trainer_module_ambiguous_mod")

    class ATrainer:
        def setup(self, ctx: StepContext) -> None:
            _ = ctx

        def configure(self, ctx: StepContext) -> dict[str, object]:
            _ = ctx
            return {}

        def prepare_batch(self, raw_batch: object, state: dict[str, object], ctx: StepContext) -> object:
            _ = state
            _ = ctx
            return raw_batch

        def train_step(self, batch: object, state: dict[str, object], ctx: StepContext) -> StepResult:
            _ = state
            _ = ctx
            return StepResult(metrics={"loss": float(batch)})

        def state_dict(self, state: dict[str, object]) -> dict[str, object]:
            return state

        def load_state_dict(self, state: dict[str, object], payload: dict[str, object], ctx: StepContext) -> None:
            _ = ctx
            state.update(payload)

    class BTrainer(ATrainer):
        pass

    mod.ATrainer = ATrainer
    mod.BTrainer = BTrainer
    monkeypatch.setitem(sys.modules, "tmp_trainer_module_ambiguous_mod", mod)

    with pytest.raises(ValueError, match="multiple trainer candidates"):
        load_trainer_plugin("tmp_trainer_module_ambiguous_mod")
