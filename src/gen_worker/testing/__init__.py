"""Test helpers for authoring gen-worker endpoints.

Every ``Slot``-declared endpoint needs a ``ctx.slots["<name>"]`` stub to
unit-test its handler without a live hub, so that no endpoint hand-rolls its
own ``FakeCtx``. :func:`fake_context` builds a real
:class:`~gen_worker.request_context.RequestContext` (or a producer-kind
subclass) with ``ctx.slots`` pre-resolved from plain ``(ref, defaults)``
pairs::

    from gen_worker.testing import fake_context
    from gen_worker import HF
    from myendpoint.defaults import SdxlDefaults  # your endpoint's own vocabulary

    ctx = fake_context(slots={
        "pipeline": (HF("stabilityai/stable-diffusion-xl-base-1.0"), SdxlDefaults(steps=28)),
    })
    out = Generate().generate(ctx, TextToImage(prompt="a cat"))

Pass a :class:`Recorder` to assert what the handler SAVED and LOGGED
 — the outputs go through the SDK's real encode/stamp/write path
and land in an inspectable list::

    rec = Recorder()
    ctx = fake_context(slots={...}, recorder=rec)
    Generate().generate(ctx, TextToImage(prompt="a cat"))

    assert rec.refs == ["outputs/test-request/image.webp"]
    assert rec.images[0].read_bytes()[:4] == b"RIFF"     # a real webp encode
    assert "loaded 2 loras" in rec.messages

Do NOT hand-roll a ``_Ctx`` subclass overriding ``save_image``/``save_audio``
to capture calls: overriding them makes a suite green over an encode path it
never ran. The recorder captures the same facts on the way THROUGH it.

Not imported by ``gen_worker`` itself — production code has no reason to
import test helpers; import ``gen_worker.testing`` explicitly from test
modules.
"""

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
from ..api.slot import ResolvedSlot
from ..api.types import Asset, AudioAsset, ImageAsset, VideoAsset
from ..families.base import GenerationDefaults
from ..request_context import RequestContext

if TYPE_CHECKING:  # heavy optional deps — imported only for signature fidelity
    import numpy as np
    import torch
    from PIL import Image

C = TypeVar("C", bound="RequestContext[Any]")


def stub_slots(
    slots: Mapping[str, Tuple[ModelRef, GenerationDefaults]],
) -> Dict[str, "ResolvedSlot[GenerationDefaults]"]:
    """``{slot_name: (ref, defaults)}`` -> ``{slot_name: ResolvedSlot}`` —
    the same shape ``ctx.slots`` hands a handler in production, built
    directly instead of via the repo-metadata resolution chain."""
    return {name: ResolvedSlot(ref=ref, defaults=defaults) for name, (ref, defaults) in slots.items()}


class SavedArtifact(msgspec.Struct, frozen=True, kw_only=True):
    """One output the handler saved, as the real ``save_*`` path produced it.

    ``kind`` is which ``ctx.save_*`` the handler called (``image`` / ``audio``
    / ``video`` / ``file`` / ``bytes``); ``ref`` is the normalized output ref
    the SDK assigned (handlers often pass none); ``asset`` is the typed value
    the handler got back, carrying the REAL ``sha256`` and ``size_bytes``;
    ``call`` is the keyword arguments the handler passed (``format``,
    ``quality``, ``sample_rate``, ...) so a suite can assert the encode
    REQUEST as well as its result.
    """

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
    """One event the context emitted (``request.log``, ``request.progress``,
    ``request.checkpoint``), captured at the REAL emitter seam — which is why
    ``ctx.log`` and ``ctx.progress`` need no override to be observed."""

    type: str
    payload: Mapping[str, Any] = {}


class Recorder:
    """Collects what a handler saved and emitted through a
    :func:`fake_context`.

    Hold one per test and read it after the handler returns::

        rec = Recorder()
        ctx = fake_context(slots={...}, recorder=rec)
        out = Generate().generate(ctx, payload)
        assert rec.images[0].call["format"] == "png"

    The recorder owns a temporary output directory (removed when it is
    garbage-collected), which is what lets every ``save_*`` run its real
    encode, C2PA stamp and size-limit checks with no hub and no network.
    """

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

    # -- reads ------------------------------------------------------------

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
        """``ctx.log`` message strings, in order — the list 23 suites were
        hand-rolling a ``log`` override to build."""
        return [str(e.payload.get("message", "")) for e in self.logs]

    @property
    def progress(self) -> List[RecordedEvent]:
        return [e for e in self.events if e.type == "request.progress"]

    # -- writes (called by the recording context) -------------------------

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
    """Records each ``save_*`` AFTER the real implementation ran.

    Mixed in front of the requested context class by :func:`fake_context`, so
    every ``super()`` call below is the production method — the encode, the
    C2PA stamp, the output-size limit and the ref normalization all happen
    exactly as they do on a pod. Nested calls (``save_image`` -> ``save_bytes``)
    record once, at the outermost frame, under the kind the handler asked for.
    """

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
    """``RecordingRequestContext`` / ``RecordingConversionContext`` / ... —
    built once per context class so ``isinstance(ctx, cls)`` still holds."""
    built = _RECORDING_CLASSES.get(cls)
    if built is None:
        built = type(f"Recording{cls.__name__}", (_RecordingMixin, cls), {})
        _RECORDING_CLASSES[cls] = built
    return built


def fake_context(
    *,
    request_id: str = "test-request",
    slots: Mapping[str, Tuple[ModelRef, GenerationDefaults]] = {},
    cls: Type[C] = RequestContext,  # type: ignore[assignment]
    recorder: Optional[Recorder] = None,
    **kwargs: Any,
) -> C:
    """Build a :class:`RequestContext` (or ``cls=``, a producer-kind
    subclass: ``ConversionContext``/``DatasetContext``/``TrainingContext``)
    for a handler unit test, with ``ctx.slots`` pre-populated.

    ``slots`` maps slot name -> ``(ref, defaults)`` — exactly what a
    ``Slot``-declared endpoint's handler reads via
    ``ctx.slots["<name>"].ref`` / ``.defaults``. Every other
    :class:`RequestContext` constructor kwarg (``owner``, ``invoker_id``,
    ...) passes through via ``**kwargs``.

    ``recorder`` turns on RECORDING MODE: saves land in
    ``recorder.saved`` and events in ``recorder.events``, and outputs are
    written into the recorder's own directory instead of being uploaded — so
    the handler runs the SDK's real encode path with no hub and no network.
    An explicit ``local_output_dir=`` still wins, and a caller's ``emitter=``
    is chained rather than replaced.
    """
    if recorder is None:
        return cls(
            request_id=request_id,
            resolved_slots=stub_slots(slots),
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
        resolved_slots=stub_slots(slots),
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
    "stub_slots",
]
