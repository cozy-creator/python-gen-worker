from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from gen_worker.trainer import StepContext, TrainingJobSpec
from gen_worker.trainer.runtime import LocalArtifactWriter


class MinimalTrainer:
    def setup(self, ctx: StepContext) -> None:
        _ = ctx

    def configure(self, ctx: StepContext) -> dict[str, object]:
        _ = ctx
        return {}

    def prepare_batch(self, raw_batch: Any, state: dict[str, object], ctx: StepContext) -> Any:
        _ = state
        _ = ctx
        return raw_batch

    def train_step(self, batch: Any, state: dict[str, object], ctx: StepContext) -> Any:
        _ = batch
        _ = state
        _ = ctx
        return None

    def state_dict(self, state: dict[str, object]) -> dict[str, object]:
        return dict(state)

    def load_state_dict(self, state: dict[str, object], payload: dict[str, object], ctx: StepContext) -> None:
        _ = ctx
        state.update(payload)


def test_local_artifact_writer_writes_checkpoint_json_atomically(tmp_path: Path) -> None:
    writer = LocalArtifactWriter(
        checkpoints_dir=str(tmp_path / "checkpoints"),
        samples_dir=str(tmp_path / "samples"),
    )
    job = TrainingJobSpec(request_id="trainer-runtime-test", max_steps=1)
    ctx = StepContext(job=job)

    path = writer.write_checkpoint(
        step=1,
        state_payload={"step": 1},
        trainer=MinimalTrainer(),
        state={"step": 1},
        ctx=ctx,
    )

    assert path == str(tmp_path / "checkpoints" / "step-00000001.json")
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    assert payload == {
        "request_id": "trainer-runtime-test",
        "state": {"step": 1},
        "step": 1,
    }
