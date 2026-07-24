from __future__ import annotations

import inspect

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
