from __future__ import annotations

from typing import Any, Mapping, Protocol


class ArtifactUploadError(RuntimeError):
    pass


class ArtifactUploader(Protocol):
    def upload_checkpoint(self, *, local_path: str, step: int) -> Mapping[str, Any]:
        ...

    def upload_sample(self, *, local_path: str, step: int) -> Mapping[str, Any]:
        ...

    def upload_metrics(self, *, metrics: Mapping[str, float], step: int) -> Mapping[str, Any]:
        ...


__all__ = ["ArtifactUploader", "ArtifactUploadError"]
