from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol


@dataclass(frozen=True)
class ModelRuntimeConfig:
    device: str = "cuda:0"
    dtype: str = "bf16"


class ModelRuntimeLoader(Protocol):
    def load_components_to_device(self, model_dir: str, config: ModelRuntimeConfig) -> Mapping[str, Any]:
        ...


__all__ = ["ModelRuntimeConfig", "ModelRuntimeLoader"]
