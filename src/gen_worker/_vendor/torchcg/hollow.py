"""Weights-free pipeline instantiation for the publish-time derive (pgw#1370).

The 2026-08-18 ruling puts the derive at PUBLISH time, on CPU, inside the
release env, against CONFIG-ONLY checkpoint subsets -- never weights. Author
code still calls ``from_pretrained(...).to("cuda")`` as-is, so this module
supplies the session that makes that code run hollow:

* **Loader interception** -- ``diffusers.ModelMixin.from_pretrained`` and
  ``transformers.PreTrainedModel.from_pretrained`` are replaced for the
  session: the component is built from its CONFIG (parameters born on meta,
  buffers computed real -- they are pure functions of config and are what a
  literal digest reads), then every parameter becomes a FAKE tensor in the
  session's one ``FakeTensorMode``. Tokenizers, schedulers and processors
  load real from the config-only tree; they never carry weights.
* **Device neutrality** -- an explicit ``"cuda"`` in author code is remapped
  to ``"cpu"`` at the torch-function boundary. The publish derive is
  device-free by design: graph identity carries dtype and shape, and the sm
  enters at ARTIFACT level, never at graph level.
* **Fake egress** -- scalarization of a fake tensor (``.numpy()``,
  ``.item()``, ``.tolist()``, ``bool()``) answers canonical ZEROS instead of
  refusing. Observation is structure-only; a value-dependent branch in author
  code follows the zero surrogate path, which costs coverage on that branch,
  never correctness of an observed graph. Real tensors (schedulers, token
  ids) are untouched -- the whole-pipeline ambient-fake drive was measured
  non-viable precisely because scheduler ``.item()`` needs real values.

The session deliberately does NOT wrap the run in an ambient
``FakeTensorMode``: only tensors DERIVED from hollow parameters are fake, so
the author's control flow keeps its real values.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any


class HollowError(RuntimeError):
    """A component cannot be built hollow from code + config alone."""


@dataclass(frozen=True, slots=True)
class HollowSession:
    """The one fake mode every hollow parameter of a derive run belongs to."""

    fake_mode: Any


def _numpy_dtype(dtype: Any) -> Any:
    import numpy
    import torch

    named = {
        torch.float64: numpy.float64,
        torch.float32: numpy.float32,
        torch.float16: numpy.float16,
        torch.int64: numpy.int64,
        torch.int32: numpy.int32,
        torch.int16: numpy.int16,
        torch.int8: numpy.int8,
        torch.uint8: numpy.uint8,
        torch.bool: numpy.bool_,
    }
    # bfloat16 and below have no numpy spelling; float32 is the widening
    # numpy itself would pick after the author's own `.float()`.
    return named.get(dtype, numpy.float32)


def _zero_scalar(dtype: Any) -> Any:
    import torch

    if dtype is torch.bool:
        return False
    if dtype.is_floating_point or dtype.is_complex:
        return 0.0
    return 0


@contextlib.contextmanager
def _empty_parameters() -> Iterator[None]:
    """Construct modules with parameters on meta and buffers computed real.

    The ``register_parameter`` patch (the accelerate ``init_empty_weights``
    shape, owned here so the derive has no undeclared dependency) moves each
    parameter to meta AT REGISTRATION, before any initializer touches it, so
    a 10 GiB denoiser costs nothing to build. Buffers are left alone: they
    are computed by ``__init__`` from config, they are KB-to-MB scale, and
    their VALUES are what a literal-bearing trace digests.
    """

    import torch

    original = torch.nn.Module.register_parameter

    def register(module: Any, name: str, param: Any) -> None:
        original(module, name, param)
        if param is not None:
            registered = module._parameters[name]
            module._parameters[name] = torch.nn.Parameter(
                registered.to("meta"), requires_grad=False
            )

    torch.nn.Module.register_parameter = register  # type: ignore[method-assign, assignment]
    try:
        yield
    finally:
        torch.nn.Module.register_parameter = original  # type: ignore[method-assign]


def virtualize_parameters(module: Any, session: HollowSession, *, dtype: Any = None) -> None:
    """Every parameter becomes a fake CPU tensor in the session's mode.

    ``dtype`` restates the author's ``torch_dtype=`` ask: floating parameters
    take it, integer parameters keep their own -- exactly what a real
    ``from_pretrained(torch_dtype=...)`` load produces.
    """

    import torch

    for submodule in module.modules():
        for name, param in list(submodule._parameters.items()):
            if param is None:
                continue
            target = (
                dtype
                if dtype is not None and param.dtype.is_floating_point
                else param.dtype
            )
            with session.fake_mode:
                hollow = torch.empty(
                    tuple(int(d) for d in param.shape), dtype=target, device="cpu"
                )
            submodule._parameters[name] = torch.nn.Parameter(
                hollow, requires_grad=False
            )
        for name, buffer in list(submodule._buffers.items()):
            if buffer is not None and buffer.device.type == "meta":
                raise HollowError(
                    f"buffer {name!r} on {type(submodule).__name__} was built on "
                    f"meta: its value is a function of config and must be REAL "
                    f"for the trace's literal digest. A module that moves its "
                    f"buffers to meta cannot be derived hollow."
                )


def _hollow_diffusers(session: HollowSession) -> Any:
    def from_pretrained(cls: Any, /, pretrained_model_name_or_path: Any, **kwargs: Any) -> Any:
        subfolder = kwargs.get("subfolder")
        torch_dtype = kwargs.get("torch_dtype", kwargs.get("dtype"))
        try:
            config, _unused = cls.load_config(
                pretrained_model_name_or_path,
                subfolder=subfolder,
                return_unused_kwargs=True,
            )
            with _empty_parameters():
                model = cls.from_config(config)
        except Exception as exc:
            raise HollowError(
                f"hollow build of {cls.__name__} from "
                f"{str(pretrained_model_name_or_path)!r}"
                + (f" (subfolder {subfolder!r})" if subfolder else "")
                + f" failed: {type(exc).__name__}: {exc}"
            ) from exc
        virtualize_parameters(model, session, dtype=torch_dtype)
        model.eval()
        return model

    return classmethod(from_pretrained)


def _hollow_transformers(session: HollowSession) -> Any:
    def from_pretrained(cls: Any, /, pretrained_model_name_or_path: Any, **kwargs: Any) -> Any:
        subfolder = kwargs.get("subfolder") or ""
        torch_dtype = kwargs.get("torch_dtype", kwargs.get("dtype"))
        try:
            config = cls.config_class.from_pretrained(
                pretrained_model_name_or_path, subfolder=subfolder
            )
            with _empty_parameters():
                model = cls(config)
        except Exception as exc:
            raise HollowError(
                f"hollow build of {cls.__name__} from "
                f"{str(pretrained_model_name_or_path)!r}"
                + (f" (subfolder {subfolder!r})" if subfolder else "")
                + f" failed: {type(exc).__name__}: {exc}"
            ) from exc
        virtualize_parameters(model, session, dtype=torch_dtype)
        model.eval()
        return model

    return classmethod(from_pretrained)


@contextlib.contextmanager
def _loader_interception(session: HollowSession) -> Iterator[None]:
    """Patch the model-bearing loaders of whichever libraries are present.

    Library-level, never family-level: the patch knows ``ModelMixin`` and
    ``PreTrainedModel``, not what any endpoint builds with them. A library
    that is not installed is simply not patched -- pure-transformers authors
    and pure-diffusers authors both resolve.
    """

    patched: list[tuple[Any, Any]] = []
    try:
        import diffusers.models.modeling_utils as _dmu
    except ImportError:
        _dmu = None
    try:
        import transformers.modeling_utils as _tmu
    except ImportError:
        _tmu = None
    if _dmu is None and _tmu is None:
        raise HollowError(
            "hollow instantiation intercepts diffusers and/or transformers "
            "loaders, and neither library is importable in this env"
        )
    try:
        if _dmu is not None:
            patched.append((_dmu.ModelMixin, _dmu.ModelMixin.__dict__["from_pretrained"]))
            _dmu.ModelMixin.from_pretrained = _hollow_diffusers(session)
        if _tmu is not None:
            patched.append(
                (_tmu.PreTrainedModel, _tmu.PreTrainedModel.__dict__["from_pretrained"])
            )
            _tmu.PreTrainedModel.from_pretrained = _hollow_transformers(session)
        yield
    finally:
        for owner, original in patched:
            owner.from_pretrained = original


def _parse_move(args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> Any:
    """The dtype (if any) a ``Module.to``-style call asks for."""

    import torch

    dtype = kwargs.get("dtype")
    for value in args:
        if isinstance(value, torch.dtype):
            dtype = value
        elif isinstance(value, torch.Tensor):
            dtype = value.dtype
    return dtype


@contextlib.contextmanager
def _hollow_module_moves() -> Iterator[None]:
    """``Module.to`` / ``Module.cuda`` on a HOLLOW module never swaps.

    ``Module._apply`` force-routes fake parameters through
    ``torch.utils.swap_tensors``, which refuses any tensor the fake mode holds
    a weakref to -- so author code's ``.to("cuda")`` would crash on exactly
    the modules this session hollowed. For a hollow module the move is
    IDENTITY on device (the derive is device-neutral) and a re-virtualization
    on dtype; real modules keep torch's own behavior.
    """

    import torch
    from torch._subclasses.fake_tensor import FakeTensor

    original_to = torch.nn.Module.to
    original_cuda = torch.nn.Module.cuda

    def is_hollow(module: Any) -> bool:
        return any(isinstance(p, FakeTensor) for p in module.parameters())

    def hollow_to(module: Any, *args: Any, **kwargs: Any) -> Any:
        if not is_hollow(module):
            return original_to(module, *args, **kwargs)
        dtype = _parse_move(args, kwargs)
        if dtype is not None:
            for submodule in module.modules():
                for name, param in list(submodule._parameters.items()):
                    if param is None or not param.dtype.is_floating_point:
                        continue
                    mode = param.fake_mode
                    with mode:
                        hollow = torch.empty(
                            tuple(int(d) for d in param.shape),
                            dtype=dtype,
                            device="cpu",
                        )
                    submodule._parameters[name] = torch.nn.Parameter(
                        hollow, requires_grad=False
                    )
                for name, buffer in list(submodule._buffers.items()):
                    if buffer is not None and buffer.dtype.is_floating_point:
                        submodule._buffers[name] = buffer.to(dtype)
        return module

    def hollow_cuda(module: Any, *args: Any, **kwargs: Any) -> Any:
        if not is_hollow(module):
            return original_cuda(module, *args, **kwargs)
        return module

    torch.nn.Module.to = hollow_to  # type: ignore[method-assign, assignment]
    torch.nn.Module.cuda = hollow_cuda  # type: ignore[method-assign, assignment]
    try:
        yield
    finally:
        torch.nn.Module.to = original_to  # type: ignore[method-assign]
        torch.nn.Module.cuda = original_cuda  # type: ignore[method-assign]


def _remap_device(value: Any) -> Any:
    import torch

    if isinstance(value, torch.device):
        return torch.device("cpu") if value.type == "cuda" else value
    if isinstance(value, str) and (value == "cuda" or value.startswith("cuda:")):
        return "cpu"
    if isinstance(value, (list, tuple)):
        return type(value)(_remap_device(item) for item in value)
    if isinstance(value, dict):
        return {key: _remap_device(item) for key, item in value.items()}
    return value


@contextlib.contextmanager
def observation_shims() -> Iterator[None]:
    """The two torch-function shims the hollow drive runs under."""

    import torch
    from torch._subclasses.fake_tensor import FakeTensor
    from torch.overrides import TorchFunctionMode

    class DeviceNeutral(TorchFunctionMode):
        def __torch_function__(
            self, func: Any, types: Any, args: Any = (), kwargs: Any = None
        ) -> Any:
            if func is torch.Tensor.cuda:
                return args[0]
            return func(*_remap_device(tuple(args)), **_remap_device(dict(kwargs or {})))

    class FakeEgress(TorchFunctionMode):
        def __torch_function__(
            self, func: Any, types: Any, args: Any = (), kwargs: Any = None
        ) -> Any:
            kwargs = kwargs or {}
            host = args[0] if args else None
            if isinstance(host, FakeTensor):
                if func is torch.Tensor.numpy:
                    import numpy

                    return numpy.zeros(
                        tuple(int(d) for d in host.shape), dtype=_numpy_dtype(host.dtype)
                    )
                if func is torch.Tensor.item:
                    return _zero_scalar(host.dtype)
                if func is torch.Tensor.tolist:
                    import numpy

                    return numpy.zeros(
                        tuple(int(d) for d in host.shape), dtype=_numpy_dtype(host.dtype)
                    ).tolist()
                if func is torch.Tensor.__bool__:
                    return False
                if func in (torch.Tensor.__int__, torch.Tensor.__index__):
                    return 0
                if func is torch.Tensor.__float__:
                    return 0.0
            return func(*args, **kwargs)

    with DeviceNeutral(), FakeEgress():
        yield


@contextlib.contextmanager
def hollow_session() -> Iterator[HollowSession]:
    """Everything a weights-free derive runs inside: one context, one mode.

    Instantiation (the author's ``setup``) and observation (the author's
    handlers driven with sample payloads) both run in here; derivation may
    too -- the shims never touch non-fake tensors and ``torch.export``
    re-enters the session's own mode for hollow modules.
    """

    from torch._subclasses.fake_tensor import FakeTensorMode

    session = HollowSession(fake_mode=FakeTensorMode(allow_non_fake_inputs=True))
    with _loader_interception(session), _hollow_module_moves(), observation_shims():
        yield session


__all__ = [
    "HollowError",
    "HollowSession",
    "hollow_session",
    "observation_shims",
    "virtualize_parameters",
]
