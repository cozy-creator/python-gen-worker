from __future__ import annotations

from gen_worker.trainer import build_default_adamw_bundle, to_float_scalar


class _DetachableItem:
    def __init__(self, value: float) -> None:
        self._value = value

    def detach(self) -> "_DetachableItem":
        return self

    def item(self) -> float:
        return self._value


def test_to_float_scalar_handles_detach_item() -> None:
    assert to_float_scalar(_DetachableItem(3.5)) == 3.5


def test_to_float_scalar_handles_plain_number() -> None:
    assert to_float_scalar(2) == 2.0


def test_build_default_adamw_bundle_without_params_returns_empty_bundle() -> None:
    bundle = build_default_adamw_bundle(None, hyperparams={"learning_rate": 1e-4})
    assert bundle.optimizer is None
    assert bundle.scheduler is None
