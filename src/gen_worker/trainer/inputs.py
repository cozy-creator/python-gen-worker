from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class DownloadedInputs:
    model_dir: str
    dataset_dir: str
    resume_checkpoint: str | None = None


class InputDownloader(Protocol):
    def download_weights(self, ref: str) -> str:
        ...

    def download_dataset_parquet(self, dataset_ref: str) -> str:
        ...

    def download_resume_checkpoint(self, checkpoint_ref: str) -> str:
        ...


__all__ = ["DownloadedInputs", "InputDownloader"]
