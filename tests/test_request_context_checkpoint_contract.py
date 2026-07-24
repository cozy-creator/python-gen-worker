from __future__ import annotations

import inspect

from gen_worker.convert.dataset import Dataset
from gen_worker.request_context import TrainingContext


def test_checkpoint_contract_exposes_only_persisted_metadata() -> None:
    parameters = inspect.signature(TrainingContext.save_checkpoint).parameters

    assert {"step_number", "epoch_number", "output_kind"} <= parameters.keys()
    assert {
        "produced_by_kind",
        "target_dtype",
        "flavor",
        "attributes",
    }.isdisjoint(parameters)


def test_dataset_contract_keeps_live_readers_only() -> None:
    assert hasattr(Dataset, "iter_examples")
    assert hasattr(Dataset, "as_dataloader")
    assert hasattr(Dataset, "iter_rows")
    assert not hasattr(Dataset, "as_hf_dataset")
    assert not hasattr(Dataset, "is_eval_set")
