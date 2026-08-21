from __future__ import annotations

import contextlib
import logging
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, Callable, Optional

from ..serving.context import _lane_torch_dtype


class ProjectedTreeAtTrace(RuntimeError):
    """An author loader failed on a tree whose tensors are POINTERS."""


def _projected_tree_diagnosis(
    tree: Path, exc: BaseException
) -> Optional[ProjectedTreeAtTrace]:

    try:
        from ..models import projection

        stubs = [
            (path.relative_to(tree).as_posix(), path.lstat().st_size, stub.size)
            for path in sorted(Path(tree).rglob("*"))
            if (path.is_file() or path.is_symlink())
            and (stub := projection.stub_at(path)) is not None
        ]
    except Exception:  # noqa: BLE001 — the original failure must survive
        return None
    if not stubs:
        return None
    shown = ", ".join(
        f"{rel} ({on_disk} B on disk, names {names} B)"
        for rel, on_disk, names in stubs[:3]
    )
    more = "" if len(stubs) <= 3 else f" (+{len(stubs) - 3} more)"
    return ProjectedTreeAtTrace(
        f"{type(exc).__name__}: {exc}\n"
        f"  ...and {tree} is a PROJECTED tree: {len(stubs)} of its tensor "
        f"containers are TFSSTUB1 pointer stubs, not weights — {shown}{more}. "
        f"The checkpoint is intact; its bytes are CAS objects and are not at "
        f"any file path. A loader that reads a container with the stock "
        f"safetensors reader sees the stub's first 8 bytes as a header length "
        f"and reports a corrupt checkpoint.\n"
        f"  Fix it in the LOADER, by pgw#1303's access ladder:\n"
        f"    tier 1  gen_worker.models.tensor_source.open_tensor_source(path, "
        f"why=...) — safe_open's exact shape, reads the CAS, copies nothing;\n"
        f"    tier 1  ...load_state_dict(path, why=...) — the "
        f"safetensors.torch.load_file replacement;\n"
        f"    tier 3  gen_worker.models.materialized_view.third_party_dir("
        f"path, why=...) — one real file, for third-party code that insists on "
        f"one (it is a no-op on a tree that is not projected).\n"
        f"  Worked example: serverless-endpoints anima `main.py` (se#817)."
    )


class TraceLoadContext:
    """What ``Model.load`` sees under ``gen-worker release derive``."""

    def __init__(
        self,
        *,
        lane: Any,
        checkpoint_dir: Path,
        model_type: Optional[type] = None,
        defaults_instance: Any = None,
    ) -> None:
        self.lane = lane
        self.checkpoint_dir = Path(checkpoint_dir)
        self.model_type = model_type
        self.defaults_instance = defaults_instance
        self._log = logging.getLogger("gen_worker.release.trace")
        self.marked_modules: list[Any] = []
        self._dtype_said: set[str] = set()

    def load(self, loader: Any) -> Any:
        """Hollow-materialize the CONFIG-ONLY tree through the author's loader."""
        from_pretrained = getattr(loader, "from_pretrained", None)
        if from_pretrained is None:
            raise TypeError(
                f"ctx.load() needs a loader with from_pretrained "
                f"(a diffusers/transformers class); got {loader!r}"
            )
        try:
            loaded = from_pretrained(self.checkpoint_dir)
        except Exception as exc:  # noqa: BLE001 — re-raised, never swallowed
            diagnosis = _projected_tree_diagnosis(self.checkpoint_dir, exc)
            if diagnosis is None:
                raise
            raise diagnosis from exc
        for lora_call in ("load_lora_weights", "set_adapters", "unload_lora_weights"):
            if hasattr(loaded, lora_call):
                setattr(loaded, lora_call, _noop)
        return loaded

    def component_dtype(self, tree: Any, subfolder: Any, module: Any = None) -> Any:
        """The precision ONE component loads at — **from the LANE DECLARATION**."""

        from ..serving.checkpoint_dtype import (
            _config_dtype,
            _tensor_dtype as _header_dtype,
        )

        directory = Path(str(tree))
        name = str(subfolder).strip("/") if subfolder else ""
        if name:
            directory = directory / name
        else:
            try:
                relative = directory.resolve().relative_to(
                    self.checkpoint_dir.resolve()
                )
            except (ValueError, OSError):
                relative = None
            if relative is not None and str(relative) not in (".", ""):
                name = str(relative).strip("/")

        declared = _lane_torch_dtype(self.lane)
        if declared is not None:
            self._say_checkpoint_disagrees(directory, name, declared)
            return declared

        own = _header_dtype(directory)
        if own is not None:
            return own
        config = _config_dtype(directory)
        if config is not None:
            return config

        return None

    def _say_checkpoint_disagrees(
        self, directory: Path, name: str, declared: Any
    ) -> None:

        from ..serving.checkpoint_dtype import _tensor_dtype

        key = str(directory)
        if key in self._dtype_said:
            return
        try:
            own = _tensor_dtype(directory)
        except Exception:  # noqa: BLE001 — a diagnostic never fails a derive
            return
        if own is None or own == declared:
            return
        self._dtype_said.add(key)
        self._log.warning(
            "derive: lane %s declares dtype %s and the mounted checkpoint's "
            "%s containers carry %s. The TRACE FOLLOWS THE LANE (pgw#1567): "
            "the store converts a tree through the layout contract before a "
            "pod mounts it, so the lane is what serves. This tree was not "
            "converted — the graphs are still right, but nothing here has "
            "verified the checkpoint against the contract it is keyed under.",
            getattr(self.lane, "contract", self.lane),
            declared,
            name or "root",
            own,
        )

    def compile(self, target: Any) -> Any:
        import torch

        if isinstance(target, torch.nn.Module):
            if all(existing is not target for existing in self.marked_modules):
                self.marked_modules.append(target)
            return target
        components = getattr(target, "components", None)
        if isinstance(components, Mapping):
            for component in components.values():
                if isinstance(component, torch.nn.Module):
                    self.compile(component)
            return target
        raise TypeError(
            f"ctx.compile() marks nn.Modules (or a pipeline-like object "
            f"whose .components carries them); got {type(target).__name__}"
        )

    def defaults(self) -> Any:
        """The enumerated defaults VARIANT for the class-header model type."""
        if self.defaults_instance is not None:
            return self.defaults_instance
        if self.model_type is None:
            raise TypeError(
                "ctx.defaults() reads the model type off the class header "
                "(class X(Model[SDXL], ...)); this model's base is "
                "unparameterized"
            )
        defaults_type = getattr(self.model_type, "Defaults", self.model_type)
        return defaults_type()


def _noop(*_args: Any, **_kwargs: Any) -> None:
    return None


class StepBudgetReached(Exception):
    """The trace has seen every shape this denoise loop produces."""


class TraceSurfaceUnavailable(RuntimeError):
    """An author touched a ctx member the derive genuinely cannot answer."""


class TraceRequestContext:
    """What entrypoints see under ``gen-worker release derive``."""

    def __init__(
        self,
        *,
        lane: Any,
        checkpoint_ref: str = "",
        step_budget: Optional[int] = None,
        checkpoint_dir: Optional[Path] = None,
        device: Any = None,
    ) -> None:
        self.lane = lane
        self.step_budget = step_budget
        self.checkpoint_ref = checkpoint_ref or "trace:config-only"
        self._log = logging.getLogger("gen_worker.release.trace")
        self._checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self._device = device
        self._warnings: list[str] = []
        self._adjustments: list[dict[str, str]] = []
        self.progress_events: list[tuple[Optional[float], Optional[str], dict[str, Any]]] = []
        self.unanswered_calls: list[tuple[str, str]] = []
        self._temporaries: list[Any] = []

    def clamp(
        self,
        field: str,
        requested: float,
        *,
        lo: Optional[float] = None,
        hi: Optional[float] = None,
        reason: str = "",
    ) -> float:
        """Same arithmetic as the serving ctx, and the same RECORD."""
        applied = float(requested)
        if lo is not None and applied < lo:
            applied = float(lo)
        if hi is not None and applied > hi:
            applied = float(hi)
        if applied != float(requested):
            self.adjusted(field, requested, applied, reason)
        return applied

    def raise_if_cancelled(self, message: str = "request cancelled") -> None:
        del message

    def warn(self, message: str) -> None:
        """Caller-visible advisory at serve; recorded AND logged at trace."""
        self._warnings.append(str(message))
        self._log.warning("trace: %s", message)

    def log(self, message: str, level: str = "info", **fields: Any) -> None:

        self._log.info(
            "trace: %s%s",
            message,
            f" {fields!r}" if fields else "",
        )

    def step_callback(
        self, num_inference_steps: int = 0, **kwargs: Any
    ) -> Callable[..., dict[str, Any]]:
        """A diffusers ``callback_on_step_end`` that enforces the step budget."""
        del num_inference_steps, kwargs
        seen = 0
        budget = self.step_budget

        def callback(
            _pipe: Any, _index: Any, _timestep: Any,
            callback_kwargs: Any = None, **_: Any,
        ) -> dict[str, Any]:
            del callback_kwargs
            nonlocal seen
            seen += 1
            if budget is not None and seen >= budget:
                raise StepBudgetReached
            return {}

        return callback

    def save_image(self, image: Any, *, format: str = "webp", **_: Any) -> Any:
        """A stub asset: nothing is encoded or uploaded at trace time."""
        del image
        from ..api.types import ImageAsset

        return ImageAsset(ref=f"trace://image.{format}")

    _outputs_discarded = True

    def save_video(self, video: Any, ref: Optional[str] = None, *, format: str = "mp4",
                   **_: Any) -> Any:
        del video
        from ..api.types import VideoAsset

        return VideoAsset(ref=ref or f"trace://video.{format}")

    def save_audio(self, audio: Any, ref: Optional[str] = None, *, format: str = "wav",
                   **_: Any) -> Any:
        del audio
        from ..api.types import AudioAsset

        return AudioAsset(ref=ref or f"trace://audio.{format}")

    def save_bytes(self, ref: str, data: bytes, **_: Any) -> Any:
        del data
        from ..api.types import Asset

        return Asset(ref=f"trace://{ref}")

    def save_file(self, ref: str, local_path: Any, *, create: bool = False, **_: Any) -> Any:
        del local_path, create
        from ..api.types import Asset

        return Asset(ref=f"trace://{ref}")

    def save_checkpoint(self, ref: str, local_path: Any, format: Optional[str] = None,
                        **_: Any) -> Any:
        del local_path, format
        from ..api.types import Asset

        return Asset(ref=f"trace://{ref}")

    def for_request(
        self,
        pipeline: Any,
        *,
        sampler: str = "",
        seed: Optional[int] = None,
        generator: Optional[Any] = None,
        scheduler_config: Optional[dict[str, Any]] = None,
        schedulers: Optional[Sequence[str]] = None,
    ) -> Any:
        """The per-request view, for REAL -- the guide's canonical line."""

        from ..view import for_request as _view_for_request

        gen = generator
        if gen is None and seed is not None:
            gen = self.generator(seed)
        return _view_for_request(
            pipeline, sampler=sampler, objective="", generator=gen,
            scheduler_config=scheduler_config, schedulers=schedulers,
        )

    @property
    def device(self) -> Any:
        """The device THIS TRACE is stamping, not the host's availability."""

        import torch

        if self._device is not None:
            return torch.device(self._device) if isinstance(self._device, str) else self._device
        return torch.device("cpu")

    def generator(self, seed: Optional[int] = None) -> Any:
        import torch

        gen = torch.Generator()
        if seed is not None:
            gen.manual_seed(int(seed))
        return gen

    def mktemp(self) -> Path:
        holder = tempfile.TemporaryDirectory(prefix="trace-ctx-")
        self._temporaries.append(holder)
        return Path(holder.name)

    def checkpoint_dir(self, *, key: str = "") -> Path:
        """The CONFIG-ONLY subset the derive resolved; there are no weights."""

        del key
        if self._checkpoint_dir is None:
            raise TraceSurfaceUnavailable(
                "ctx.checkpoint_dir() has no config-only tree in this trace: "
                "the derive was constructed without one. This is the derive's "
                "gap, not the endpoint's."
            )
        return self._checkpoint_dir

    @contextlib.contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Bracketing is real; only the timing sink is absent at trace."""

        del name
        yield

    def workflow_checkpoint(self, key: str, fn: Callable[[], Any]) -> Any:
        """No resume at trace, so the work always RUNS -- which is the point."""

        del key
        return fn()

    def position(self, phase: str = "") -> Optional[float]:
        del phase
        return None

    def progress(
        self,
        progress: Optional[float] = None,
        stage: Optional[str] = None,
        **extra: Any,
    ) -> None:
        self.progress_events.append((progress, stage, dict(extra)))

    def adjusted(self, field: str, requested: Any, applied: Any, reason: str = "") -> None:
        self._adjustments.append({
            "field": str(field), "requested": str(requested),
            "applied": str(applied), "reason": str(reason),
        })

    @property
    def adjustments(self) -> tuple[dict[str, str], ...]:
        return tuple(self._adjustments)

    @property
    def warnings(self) -> tuple[str, ...]:
        return tuple(self._warnings)

    @property
    def models(self) -> dict[str, str]:
        return {}

    @property
    def loras(self) -> dict[str, Any]:
        """No adapter is applied at trace -- TraceLoadContext no-ops the calls."""
        return {}

    @property
    def config(self) -> dict[str, Any]:
        return {}

    @property
    def cancelled(self) -> bool:
        return False

    @property
    def boot_warmup(self) -> bool:
        return False

    @property
    def publishes(self) -> bool:
        """Nothing is published from a trace; egress returns stub refs."""
        return False

    @property
    def emits_media(self) -> bool:
        """TRUE: the media egress path is code whose graphs must be observed."""
        return True

    @property
    def child_calls(self) -> bool:
        return False

    @property
    def handles(self) -> tuple[str, ...]:
        """The lane a trace IS, when it has one."""
        body = str(getattr(self.lane, "contract", "") or "")
        return (body,) if body else ()

    @property
    def request_id(self) -> str:
        return "trace"

    @property
    def execution_lane(self) -> str:
        return str(getattr(self.lane, "contract", "") or "")

    @property
    def hf_token(self) -> str:
        return ""

    @property
    def candidate(self) -> dict[str, Any]:
        return {}

    @property
    def candidate_path(self) -> Optional[str]:
        return None

    @property
    def source(self) -> dict[str, Any]:
        return {}

    @property
    def source_path(self) -> Optional[str]:
        return None

    @property
    def destination(self) -> dict[str, Any]:
        return {}

    @property
    def resume_from(self) -> dict[str, Any]:
        return {}

    @property
    def resume_from_path(self) -> Optional[str]:
        return None

    @property
    def text_encoder(self) -> dict[str, Any]:
        return {}

    @property
    def text_encoder_path(self) -> Optional[str]:
        return None

    @property
    def dataset_paths(self) -> dict[str, str]:
        return {}

    def resolve_dataset(self, ref: str, **_: Any) -> str:
        raise TraceSurfaceUnavailable(
            f"ctx.resolve_dataset({ref!r}) needs real dataset bytes and a trace "
            f"has none. A fabricated path would fail two frames later inside "
            f"the endpoint's own reader, naming the author's line instead of "
            f"this gap. Guard the call, or derive this endpoint with a "
            f"dataset-free enumeration arm."
        )

    def materialize_blob(self, digest: str, dest: Any, *, origin: str = "payload") -> Any:
        del dest, origin
        raise TraceSurfaceUnavailable(
            f"ctx.materialize_blob({digest!r}) needs real payload bytes and a "
            f"trace has none. An empty file at the destination would be worse "
            f"than this refusal: the endpoint would read it and fail obscurely."
        )

    def open_checkpoint_stream(self, ref: str, **_: Any) -> Any:
        raise TraceSurfaceUnavailable(
            f"ctx.open_checkpoint_stream({ref!r}) streams real checkpoint bytes "
            f"and the derive runs against a CONFIG-ONLY subset -- there is "
            f"nothing to stream. ctx.load() is the weights-free path."
        )

    def call_endpoint(
        self, endpoint: str, function: str, payload: dict[str, Any], **_: Any
    ) -> dict[str, Any]:
        """A peer endpoint the derive cannot reach: answered EMPTY, and STATED."""

        del payload
        self.unanswered_calls.append((str(endpoint), str(function)))
        self._log.warning(
            "trace: ctx.call_endpoint(%r, %r) answered EMPTY -- this trace "
            "cannot reach a peer endpoint, so graphs reached only through its "
            "result are unobserved and will mint on first live encounter.",
            endpoint, function,
        )
        return {}


__all__ = [
    "ProjectedTreeAtTrace",
    "StepBudgetReached",
    "TraceLoadContext",
    "TraceRequestContext",
    "TraceSurfaceUnavailable",
]
