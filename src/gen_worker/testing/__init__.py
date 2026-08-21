"""Test helpers for authoring gen-worker endpoints."""

from __future__ import annotations

import os
import shutil
import tempfile
import weakref
from contextlib import contextmanager
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Tuple,
    Type,
    TypeVar,
)

import msgspec

from ..api.binding import ModelRef
from ..api.types import Asset, AudioAsset, ImageAsset, VideoAsset
from ..families.base import GenerationDefaults
from ..request_context import RequestContext

if TYPE_CHECKING:
    import numpy as np
    import torch
    from PIL import Image

C = TypeVar("C", bound="RequestContext[Any]")


class SavedArtifact(msgspec.Struct, frozen=True, kw_only=True):
    """One output the handler saved, as the real ``save_*`` path produced it."""

    kind: str
    ref: str
    asset: Asset
    call: Mapping[str, Any] = {}

    @property
    def path(self) -> Optional[Path]:
        """Where the recorder's output directory holds the encoded bytes."""
        return Path(self.asset.local_path) if self.asset.local_path else None

    def read_bytes(self) -> bytes:
        """The encoded payload — a real webp/wav/mp4, not a stub."""
        p = self.path
        if p is None:
            raise FileNotFoundError(f"{self.ref} has no local payload")
        return p.read_bytes()


class RecordedEvent(msgspec.Struct, frozen=True, kw_only=True):
    """One event the context emitted (``request.log``, ``request.progress``, ``request.checkpoint``), captured at the REAL emitter seam — which is why ``ctx.log`` and ``ctx.progress`` need no override to ..."""

    type: str
    payload: Mapping[str, Any] = {}


class Recorder:
    """Collects what a handler saved and emitted through a :func:`fake_context`."""

    def __init__(self, *, output_dir: Optional[str | os.PathLike[str]] = None) -> None:
        self.saved: List[SavedArtifact] = []
        self.events: List[RecordedEvent] = []
        if output_dir is None:
            created = tempfile.mkdtemp(prefix="gw-recorder-")
            weakref.finalize(self, shutil.rmtree, created, True)
            self.output_dir = Path(created)
        else:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def of_kind(self, kind: str) -> List[SavedArtifact]:
        return [a for a in self.saved if a.kind == kind]

    @property
    def images(self) -> List[SavedArtifact]:
        return self.of_kind("image")

    @property
    def audio(self) -> List[SavedArtifact]:
        return self.of_kind("audio")

    @property
    def videos(self) -> List[SavedArtifact]:
        return self.of_kind("video")

    @property
    def files(self) -> List[SavedArtifact]:
        return self.of_kind("file")

    @property
    def refs(self) -> List[str]:
        return [a.ref for a in self.saved]

    @property
    def logs(self) -> List[RecordedEvent]:
        return [e for e in self.events if e.type == "request.log"]

    @property
    def messages(self) -> List[str]:
        """``ctx.log`` message strings, in order — the list 23 suites were hand-rolling a ``log`` override to build."""
        return [str(e.payload.get("message", "")) for e in self.logs]

    @property
    def progress(self) -> List[RecordedEvent]:
        return [e for e in self.events if e.type == "request.progress"]

    def emit(self, event: Mapping[str, Any]) -> None:
        """Emitter callback — the same seam the worker installs in production."""
        self.events.append(
            RecordedEvent(
                type=str(event.get("type", "")),
                payload=dict(event.get("payload") or {}),
            )
        )

    def record(self, kind: str, asset: Asset, call: Mapping[str, Any]) -> None:
        self.saved.append(
            SavedArtifact(kind=kind, ref=asset.ref, asset=asset, call=dict(call))
        )


class _RecordingMixin:

    _gw_recorder: Recorder
    _gw_depth: int

    @contextmanager
    def _gw_outermost(self) -> Iterator[bool]:
        depth = self._gw_depth
        self._gw_depth = depth + 1
        try:
            yield depth == 0
        finally:
            self._gw_depth = depth

    def _gw_capture(
        self, kind: str, call: Mapping[str, Any], run: Callable[[], Any]
    ) -> Any:
        with self._gw_outermost() as outermost:
            asset = run()
        if outermost:
            self._gw_recorder.record(kind, asset, call)
        return asset

    def save_bytes(self, ref: str, data: bytes) -> Asset:
        return self._gw_capture(
            "bytes",
            {},
            lambda: super(_RecordingMixin, self).save_bytes(ref, data),  # type: ignore[misc]
        )

    def save_image(
        self, image: "Image.Image", ref: Optional[str] = None, **kwargs: Any
    ) -> ImageAsset:
        return self._gw_capture(
            "image",
            kwargs,
            lambda: super(_RecordingMixin, self).save_image(image, ref, **kwargs),  # type: ignore[misc]
        )

    def save_audio(
        self,
        audio: "np.ndarray[Any, Any] | torch.Tensor | bytes",
        ref: Optional[str] = None,
        **kwargs: Any,
    ) -> AudioAsset:
        return self._gw_capture(
            "audio",
            kwargs,
            lambda: super(_RecordingMixin, self).save_audio(audio, ref, **kwargs),  # type: ignore[misc]
        )

    def save_video(
        self,
        video: "bytes | str | os.PathLike[str]",
        ref: Optional[str] = None,
        **kwargs: Any,
    ) -> VideoAsset:
        return self._gw_capture(
            "video",
            kwargs,
            lambda: super(_RecordingMixin, self).save_video(video, ref, **kwargs),  # type: ignore[misc]
        )

    def save_file(
        self, ref: str, local_path: str | os.PathLike[str], **kwargs: Any
    ) -> Asset:
        return self._gw_capture(
            "file",
            kwargs,
            lambda: super(_RecordingMixin, self).save_file(ref, local_path, **kwargs),  # type: ignore[misc]
        )


_RECORDING_CLASSES: Dict[type, type] = {}


def _recording_class(cls: type) -> type:
    built = _RECORDING_CLASSES.get(cls)
    if built is None:
        built = type(f"Recording{cls.__name__}", (_RecordingMixin, cls), {})
        _RECORDING_CLASSES[cls] = built
    return built


def fake_context(
    *,
    request_id: str = "test-request",
    cls: Type[C] = RequestContext,  # type: ignore[assignment]
    recorder: Optional[Recorder] = None,
    **kwargs: Any,
) -> C:
    """Build a :class:`RequestContext` (or ``cls=JobContext``, the producer context every ``@job`` body and producer-kind handler receives) for a handler unit test."""
    if recorder is None:
        return cls(
            request_id=request_id,
            **kwargs,
        )

    caller_emitter: Optional[Callable[[Dict[str, Any]], None]] = kwargs.pop("emitter", None)

    def _emit(event: Dict[str, Any]) -> None:
        recorder.emit(event)
        if caller_emitter is not None:
            caller_emitter(event)

    kwargs.setdefault("local_output_dir", str(recorder.output_dir))
    ctx = _recording_class(cls)(
        request_id=request_id,
        emitter=_emit,
        **kwargs,
    )
    ctx._gw_recorder = recorder
    ctx._gw_depth = 0
    return ctx


__all__ = [
    "RecordedEvent",
    "Recorder",
    "SavedArtifact",
    "fake_context",
]
