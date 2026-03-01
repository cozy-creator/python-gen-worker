from __future__ import annotations

import json
from pathlib import Path
import sys
import types

import pytest

from gen_worker.trainer import StepContext, StepResult
from gen_worker.trainer.runtime import run_training_runtime_from_env


def _register_simple_trainer_module(monkeypatch: pytest.MonkeyPatch, module_name: str, class_name: str) -> str:
    mod = types.ModuleType(module_name)

    class _Trainer:
        def setup(self, ctx: StepContext) -> None:
            _ = ctx

        def configure(self, ctx: StepContext) -> dict[str, object]:
            _ = ctx
            return {"seen_resume": False}

        def prepare_batch(self, raw_batch: object, state: dict[str, object], ctx: StepContext) -> object:
            _ = state
            _ = ctx
            return raw_batch

        def train_step(self, batch: object, state: dict[str, object], ctx: StepContext) -> StepResult:
            _ = state
            _ = ctx
            return StepResult(metrics={"train/loss": float(batch)})

        def state_dict(self, state: dict[str, object]) -> dict[str, object]:
            return dict(state)

        def load_state_dict(self, state: dict[str, object], payload: dict[str, object], ctx: StepContext) -> None:
            _ = ctx
            state.update(payload)

    _Trainer.__name__ = class_name
    setattr(mod, class_name, _Trainer)
    monkeypatch.setitem(sys.modules, module_name, mod)
    return f"{module_name}:{class_name}"


def test_training_runtime_from_env_with_mock_batches(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    trainer_import = _register_simple_trainer_module(monkeypatch, "tmp_runtime_plugin_mod", "RuntimeTrainer")

    events = tmp_path / "events.jsonl"
    checkpoints_dir = tmp_path / "checkpoints"
    samples_dir = tmp_path / "samples"
    spec = {
        "trainer_api_version": "v1",
        "run_id": "run_runtime_1",
        "trainer": trainer_import,
        "max_steps": 3,
        "metric_every": 1,
        "checkpoint_every": 2,
        "sample_every": 2,
        "mock_batches": [1, 2, 3, 4],
    }
    spec_path = tmp_path / "trainer_job.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")

    monkeypatch.setenv("TRAINER_JOB_SPEC_PATH", str(spec_path))
    monkeypatch.setenv("TRAINER_EVENTS_PATH", str(events))
    monkeypatch.setenv("TRAINER_CHECKPOINTS_DIR", str(checkpoints_dir))
    monkeypatch.setenv("TRAINER_SAMPLES_DIR", str(samples_dir))

    assert run_training_runtime_from_env() == 0
    assert (checkpoints_dir / "step-00000002.json").exists()
    assert (checkpoints_dir / "final.json").exists()
    assert (samples_dir / "step-00000002.txt").exists()
    assert list(checkpoints_dir.glob("*.tmp.*")) == []

    lines = [json.loads(line) for line in events.read_text(encoding="utf-8").splitlines() if line.strip()]
    event_types = [row["event"] for row in lines]
    assert "started" in event_types
    assert "metric" in event_types
    assert "checkpoint" in event_types
    assert "sample" in event_types
    assert "completed" in event_types


def test_training_runtime_resume_from_latest_step(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    trainer_import = _register_simple_trainer_module(monkeypatch, "tmp_runtime_plugin_mod_resume", "ResumeTrainer")

    events = tmp_path / "events.jsonl"
    checkpoints_dir = tmp_path / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    (checkpoints_dir / "step-00000003.json").write_text(json.dumps({"state": {"seen_resume": True}}), encoding="utf-8")
    samples_dir = tmp_path / "samples"
    spec = {
        "trainer_api_version": "v1",
        "run_id": "run_runtime_resume",
        "trainer": trainer_import,
        "max_steps": 5,
        "metric_every": 1,
        "checkpoint_every": 2,
        "sample_every": 0,
        "resume_from_latest": True,
        "mock_batches": [10, 11, 12, 13],
    }
    spec_path = tmp_path / "trainer_job_resume.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")

    monkeypatch.setenv("TRAINER_JOB_SPEC_PATH", str(spec_path))
    monkeypatch.setenv("TRAINER_EVENTS_PATH", str(events))
    monkeypatch.setenv("TRAINER_CHECKPOINTS_DIR", str(checkpoints_dir))
    monkeypatch.setenv("TRAINER_SAMPLES_DIR", str(samples_dir))

    assert run_training_runtime_from_env() == 0
    assert (checkpoints_dir / "step-00000004.json").exists()
    assert (checkpoints_dir / "final.json").exists()

    lines = [json.loads(line) for line in events.read_text(encoding="utf-8").splitlines() if line.strip()]
    resumed = [row for row in lines if row.get("event") == "metric" and row.get("name") == "trainer/resumed_from_step"]
    assert resumed and int(resumed[0].get("step", 0)) == 3


def test_training_runtime_samples_contract_supports_t2i_and_instruct_edit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    trainer_import = _register_simple_trainer_module(monkeypatch, "tmp_runtime_plugin_mod_samples", "SamplesTrainer")

    events = tmp_path / "events.jsonl"
    checkpoints_dir = tmp_path / "checkpoints"
    samples_dir = tmp_path / "samples"
    spec = {
        "trainer_api_version": "v1",
        "run_id": "run_runtime_samples",
        "trainer": trainer_import,
        "max_steps": 1,
        "metric_every": 1,
        "checkpoint_every": 0,
        "sample_every": 1,
        "sample_seed": 77,
        "sample_prompts": [
            "a cinematic portrait",
            {
                "name": "edit-1",
                "task": "instruct-edit",
                "prompt": "make it a watercolor",
                "instruction": "apply watercolor style",
                "source_image": "local://source.png",
            },
        ],
        "mock_batches": [1],
    }
    spec_path = tmp_path / "trainer_job_samples.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")

    monkeypatch.setenv("TRAINER_JOB_SPEC_PATH", str(spec_path))
    monkeypatch.setenv("TRAINER_EVENTS_PATH", str(events))
    monkeypatch.setenv("TRAINER_CHECKPOINTS_DIR", str(checkpoints_dir))
    monkeypatch.setenv("TRAINER_SAMPLES_DIR", str(samples_dir))

    assert run_training_runtime_from_env() == 0
    sample_a = samples_dir / "step-00000001-00.json"
    sample_b = samples_dir / "step-00000001-01.json"
    assert sample_a.exists()
    assert sample_b.exists()

    payload_a = json.loads(sample_a.read_text(encoding="utf-8"))
    payload_b = json.loads(sample_b.read_text(encoding="utf-8"))
    assert payload_a["task"] == "t2i"
    assert payload_a["seed"] == 77
    assert payload_b["task"] == "instruct-edit"
    assert payload_b["instruction"] == "apply watercolor style"
    assert payload_b["source_image"] == "local://source.png"


def test_training_runtime_resume_skips_corrupt_higher_checkpoint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    trainer_import = _register_simple_trainer_module(
        monkeypatch, "tmp_runtime_plugin_mod_resume_corrupt", "ResumeCorruptTrainer"
    )

    events = tmp_path / "events.jsonl"
    checkpoints_dir = tmp_path / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    (checkpoints_dir / "step-00000003.json").write_text(json.dumps({"state": {"seen_resume": True}}), encoding="utf-8")
    (checkpoints_dir / "step-00000005.json").write_text("{not json", encoding="utf-8")
    samples_dir = tmp_path / "samples"
    spec = {
        "trainer_api_version": "v1",
        "run_id": "run_runtime_resume_corrupt",
        "trainer": trainer_import,
        "max_steps": 4,
        "metric_every": 1,
        "checkpoint_every": 0,
        "sample_every": 0,
        "resume_from_latest": True,
        "mock_batches": [9, 10, 11],
    }
    spec_path = tmp_path / "trainer_job_resume_corrupt.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")

    monkeypatch.setenv("TRAINER_JOB_SPEC_PATH", str(spec_path))
    monkeypatch.setenv("TRAINER_EVENTS_PATH", str(events))
    monkeypatch.setenv("TRAINER_CHECKPOINTS_DIR", str(checkpoints_dir))
    monkeypatch.setenv("TRAINER_SAMPLES_DIR", str(samples_dir))

    assert run_training_runtime_from_env() == 0
    lines = [json.loads(line) for line in events.read_text(encoding="utf-8").splitlines() if line.strip()]
    resumed = [row for row in lines if row.get("event") == "metric" and row.get("name") == "trainer/resumed_from_step"]
    assert resumed and int(resumed[0].get("step", 0)) == 3
