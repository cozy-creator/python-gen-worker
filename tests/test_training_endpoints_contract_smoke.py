from __future__ import annotations

import os
import sys
from pathlib import Path
from dataclasses import dataclass, field

import pytest

from gen_worker.trainer import StepContext, TrainingJobSpec, load_trainer_plugin, run_training_loop


class _Batch:
    def __init__(self, values: dict[str, list[object]]) -> None:
        self._values = values

    def to_pydict(self) -> dict[str, list[object]]:
        return self._values


class _T2IModel:
    def train_t2i_step(
        self, *, image_refs: list[str], prompts: list[str], hyperparams: dict[str, object]
    ) -> float:
        _ = hyperparams
        if len(image_refs) != len(prompts):
            raise ValueError("length mismatch")
        return 1.0


class _EditModel:
    def train_edit_step(
        self,
        *,
        source_refs: list[str],
        target_refs: list[str],
        instructions: list[str],
        mask_refs: list[str | None],
        hyperparams: dict[str, object],
    ) -> float:
        _ = hyperparams
        if not (len(source_refs) == len(target_refs) == len(instructions) == len(mask_refs)):
            raise ValueError("length mismatch")
        return 2.0


@dataclass
class _Reporter:
    metrics: list[tuple[str, float, int]] = field(default_factory=list)
    checkpoints: list[tuple[str, int]] = field(default_factory=list)
    samples: list[tuple[str, int]] = field(default_factory=list)
    completed_calls: list[tuple[int, str | None]] = field(default_factory=list)
    failed_calls: list[tuple[int, str]] = field(default_factory=list)

    def started(self, *, run_id: str) -> None:
        _ = run_id

    def metric(self, *, name: str, value: float, step: int) -> None:
        self.metrics.append((name, value, step))

    def checkpoint(self, *, path: str, step: int) -> None:
        self.checkpoints.append((path, step))

    def sample(self, *, path: str, step: int) -> None:
        self.samples.append((path, step))

    def completed(self, *, step: int, final_checkpoint: str | None) -> None:
        self.completed_calls.append((step, final_checkpoint))

    def failed(self, *, step: int, error: str) -> None:
        self.failed_calls.append((step, error))

    def is_canceled(self) -> bool:
        return False


@dataclass
class _Writer:
    checkpoints: list[tuple[int, dict[str, object]]] = field(default_factory=list)
    final_payloads: list[dict[str, object]] = field(default_factory=list)

    def write_checkpoint(
        self,
        *,
        step: int,
        state_payload: dict[str, object],
        trainer: object,
        state: object,
        ctx: StepContext,
    ) -> str:
        _ = trainer
        _ = state
        _ = ctx
        self.checkpoints.append((step, state_payload))
        return f"/tmp/ckpt-{step}.json"

    def write_samples(self, *, step: int, state: object, ctx: StepContext) -> list[str]:
        _ = state
        _ = ctx
        return [f"/tmp/sample-{step}.json"]

    def finalize(
        self,
        *,
        state_payload: dict[str, object],
        trainer: object,
        state: object,
        ctx: StepContext,
    ) -> str | None:
        _ = trainer
        _ = state
        _ = ctx
        self.final_payloads.append(state_payload)
        return "/tmp/final.json"


def _training_endpoints_root() -> Path:
    env_override = os.getenv("TRAINING_ENDPOINTS_ROOT")
    if env_override:
        return Path(env_override).resolve()
    return Path(__file__).resolve().parents[2] / "training-endpoints"


@pytest.mark.skipif(not _training_endpoints_root().exists(), reason="training-endpoints repo not present")
def test_training_endpoints_examples_run_in_gen_worker_loop() -> None:
    root = _training_endpoints_root()
    t2i_src = root / "t2i_three_prompts" / "src"
    edit_src = root / "img2img_edit_optional_prompt_mask" / "src"
    sys.path.insert(0, str(t2i_src))
    sys.path.insert(0, str(edit_src))
    try:
        t2i_trainer = load_trainer_plugin("t2i_three_prompts.main:ThreePromptT2ITrainer")
        t2i_job = TrainingJobSpec(run_id="t2i-run", max_steps=1, metric_every=1, checkpoint_every=1, sample_every=1)
        t2i_ctx = StepContext(
            job=t2i_job,
            model_handles={"model": _T2IModel()},
        )
        t2i_reporter = _Reporter()
        t2i_writer = _Writer()
        t2i_terminal = run_training_loop(
            job=t2i_job,
            ctx=t2i_ctx,
            trainer=t2i_trainer,
            batches=[_Batch({"image_ref": ["a"], "caption_short": ["short"]})],
            reporter=t2i_reporter,
            artifact_writer=t2i_writer,
        )
        assert t2i_terminal == 1
        assert any(name == "train/loss" for (name, _value, _step) in t2i_reporter.metrics)
        assert t2i_reporter.checkpoints and t2i_writer.final_payloads
        assert t2i_reporter.samples

        edit_trainer = load_trainer_plugin("img2img_edit_optional_prompt_mask.main:Img2ImgEditTrainer")
        edit_job = TrainingJobSpec(run_id="edit-run", max_steps=1, metric_every=1, checkpoint_every=1, sample_every=1)
        edit_ctx = StepContext(
            job=edit_job,
            model_handles={"model": _EditModel()},
        )
        edit_reporter = _Reporter()
        edit_writer = _Writer()
        edit_terminal = run_training_loop(
            job=edit_job,
            ctx=edit_ctx,
            trainer=edit_trainer,
            batches=[
                _Batch(
                    {
                        "source_image_ref": ["source"],
                        "target_image_ref": ["target"],
                        "edit_type": ["mosaic"],
                    }
                )
            ],
            reporter=edit_reporter,
            artifact_writer=edit_writer,
        )
        assert edit_terminal == 1
        assert any(name == "train/loss" for (name, _value, _step) in edit_reporter.metrics)
        assert edit_reporter.checkpoints and edit_writer.final_payloads
        assert edit_reporter.samples
    finally:
        if str(t2i_src) in sys.path:
            sys.path.remove(str(t2i_src))
        if str(edit_src) in sys.path:
            sys.path.remove(str(edit_src))
