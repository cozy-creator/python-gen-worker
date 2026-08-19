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
* **One stated trace device** -- author code's device asks are remapped to
  the SESSION's device at the torch-function boundary. **This is not device
  neutrality and never was** (pgw#1458): the trace stamps a concrete device
  into every node's meta, so a cpu-remapped derive emits a device-CPU
  artifact, and a cuda mint of it dies four layers down inside AOTInductor's
  own ``Expected: cpu, Got: cuda:0``. The device is established at TRACE time
  and cannot be re-homed downstream -- four escalating attempts to move it
  were each defeated by a different layer.

  What IS neutral is the ARCH. Device (``cpu`` vs ``cuda``) is not arch
  (``sm_86`` vs ``sm_90``): a fake-``cuda`` trace needs no silicon --
  measured, with ``torch.cuda.is_available() == False`` -- and one such trace
  serves every ``sm``, because the sm enters at ARTIFACT level through
  ``RuntimeCompatibility``. So the session states a device CLASS, the
  declaration records it (derived from the program), and a mismatched mint
  refuses by name before it compiles.

  Device uniformity must be TOTAL. Parameters, buffers AND lifted tensor
  constants all move; leaving any one real-on-cpu yields either
  ``FakeTensorDeviceMismatchError`` or -- worse, because it exports cleanly --
  a MIXED ``['cpu','cuda:0']`` placement that only AOTI rejects.
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
from dataclasses import dataclass, field
from typing import Any


class HollowError(RuntimeError):
    """A component cannot be built hollow from code + config alone."""


#: The device class a compiled-graph derive traces on unless told otherwise.
#: The fleet's compiled graphs are all cuda; a cpu trace is for tests and for
#: cpu-target endpoints, and it is a DIFFERENT graph class, not a fallback.
DEFAULT_TRACE_DEVICE = "cuda"


class TraceDeviceUnavailable(HollowError):
    """A cuda-stamped exported program cannot be read on this host.

    ``torch.export.save`` of a cuda trace works fine with no GPU visible, but
    ``torch.export.load`` does NOT: it rebuilds the state dict with
    ``torch.zeros(..., device="cuda")`` and dies with ``No CUDA GPUs are
    available``. Named so a CPU-side verification step says which side of the
    asymmetry it is on instead of surfacing torch's deserializer traceback.
    """


@dataclass(frozen=True, slots=True)
class HollowSession:
    """The one fake mode and the one trace DEVICE of a derive run.

    ``real_constants`` holds the true values of every tensor this session had
    to fake in order to reach device uniformity, keyed by module FQN. They are
    not decoration: ``graph_hash`` folds ``_literal_digest`` in, so a graph
    whose lifted constants were faked would key off a table of ZEROS, and two
    models differing only in their constant tables would collide on one graph
    identity. :meth:`restore_constants` puts the real values back into the
    exported program before it is serialized.
    """

    fake_mode: Any
    device: str = DEFAULT_TRACE_DEVICE
    real_constants: dict[str, Any] = field(default_factory=dict)

    @property
    def device_type(self) -> str:
        return str(self.device).split(":", 1)[0]

    @property
    def materializes_real_tensors(self) -> bool:
        """Can a REAL tensor live on this session's device, here, now?

        cpu always. cuda only with visible silicon -- and the publish derive
        deliberately runs on boxes without any, which is why the constants
        have to be faked and restored rather than moved.
        """

        if self.device_type == "cpu":
            return True
        import torch

        return bool(torch.cuda.is_available())

    def restore_constants(self, program: Any) -> Any:
        """Put every faked lifted constant's REAL value back on the program.

        Called after ``torch.export`` and before serialization. Without it the
        literal digest -- and therefore ``cg-graph-v1`` -- would be computed
        over zeros.
        """

        constants = getattr(program, "constants", None)
        if not constants or not self.real_constants:
            return program
        for name, value in self.real_constants.items():
            if name in constants:
                constants[name] = value
        return program


#: The pt2 archive members that record a per-tensor device, in the shape
#: ``{"config": {<fqn>: {"tensor_meta": {"device": {"type": ..., ...}}}}}``.
_PT2_DEVICE_MEMBERS = ("model_weights_config.json", "model_constants_config.json")


def exported_program_devices(path: Any) -> tuple[str, ...]:
    """The device CLASSES a serialized ``ExportedProgram`` is stamped with.

    Read from the pt2 archive's own per-tensor metadata WITHOUT loading the
    program, which is the whole point: loading a cuda-stamped program needs a
    visible CUDA device, so this is the only device check a CPU-side caller
    can perform on the bytes. Returns ``()`` when the archive records no
    device -- absent is a word here, never a silent "cpu".
    """

    import json
    import zipfile

    devices: set[str] = set()
    try:
        with zipfile.ZipFile(str(path)) as archive:
            for name in archive.namelist():
                if not name.endswith(_PT2_DEVICE_MEMBERS):
                    continue
                config = json.loads(archive.read(name).decode("utf-8")).get("config", {})
                for entry in config.values():
                    device = (entry.get("tensor_meta") or {}).get("device") or {}
                    kind = device.get("type")
                    if kind:
                        devices.add(str(kind))
    except (OSError, zipfile.BadZipFile, ValueError, AttributeError):
        return ()
    return tuple(sorted(devices))


def load_exported_program(path: Any) -> Any:
    """``torch.export.load``, refusing a device this host cannot rebuild.

    The asymmetry is permanent and it shapes the test story: SAVING a
    cuda-stamped program needs no GPU, LOADING one does -- ``_load_state_dict``
    rebuilds the state dict with ``torch.zeros(..., device="cuda")``, and
    torch reports that as a generic "error when deserializing", with the real
    cause emitted to the WARNING log and never chained onto the exception. So
    the check is POSITIVE and happens first: read the archive's own recorded
    devices and say the sentence. Everything downstream of the load is
    developer-box-and-pod-only (pgw has no GPU CI lane), and this is the line.
    """

    import torch

    devices = exported_program_devices(path)
    if "cuda" in devices and not torch.cuda.is_available():
        raise TraceDeviceUnavailable(
            f"exported program {str(path)!r} is stamped {list(devices)!r} and this "
            f"host has no visible CUDA device, so torch cannot rebuild its state "
            f"dict. Saving such a program needs no GPU; loading one does. A "
            f"CPU-side check must read the recorded placement "
            f"(exported_program_devices) instead of loading the blob."
        )
    return torch.export.load(str(path))


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


def _tensor_attributes(submodule: Any) -> list[tuple[str, Any]]:
    """Plain ``torch.Tensor`` attributes -- torch.export's LIFTED CONSTANTS.

    Neither parameters nor buffers, so neither walk finds them, and they are
    exactly what lands in ``ExportedProgram.constants`` and feeds the literal
    digest. Leaving one real-on-cpu under a cuda trace is the failure that
    exports CLEANLY and only AOTI rejects.
    """

    import torch

    return [
        (name, value)
        for name, value in list(vars(submodule).items())
        if isinstance(value, torch.Tensor) and not isinstance(value, torch.nn.Parameter)
    ]


def virtualize_parameters(module: Any, session: HollowSession, *, dtype: Any = None) -> None:
    """Move every tensor of ``module`` onto the session's ONE trace device.

    ``dtype`` restates the author's ``torch_dtype=`` ask: floating parameters
    take it, integer parameters keep their own -- exactly what a real
    ``from_pretrained(torch_dtype=...)`` load produces.

    Parameters are always faked (a 10 GiB denoiser must cost nothing).
    Buffers and lifted constants are faked only when the session's device
    cannot hold a real tensor here -- and then their REAL values are captured
    on the session, because the literal digest reads them.
    """

    import torch

    device = session.device
    real_ok = session.materializes_real_tensors
    for prefix, submodule in module.named_modules():
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
                    tuple(int(d) for d in param.shape), dtype=target, device=device
                )
            submodule._parameters[name] = torch.nn.Parameter(
                hollow, requires_grad=False
            )
        for name, buffer in list(submodule._buffers.items()):
            if buffer is None:
                continue
            if buffer.device.type == "meta":
                raise HollowError(
                    f"buffer {name!r} on {type(submodule).__name__} was built on "
                    f"meta: its value is a function of config and must be REAL "
                    f"for the trace's literal digest. A module that moves its "
                    f"buffers to meta cannot be derived hollow."
                )
            submodule._buffers[name] = _rehome(
                buffer, session, f"{prefix}.{name}" if prefix else name, real_ok
            )
        for name, value in _tensor_attributes(submodule):
            setattr(
                submodule,
                name,
                _rehome(value, session, f"{prefix}.{name}" if prefix else name, real_ok),
            )


def _rehome(value: Any, session: HollowSession, fqn: str, real_ok: bool) -> Any:
    """Put one value-bearing tensor on the trace device, keeping its value.

    Real move when the device can hold one; otherwise a fake stand-in plus a
    recorded real value, so ``restore_constants`` can put the truth back
    before the program is serialized. A tensor already on the device is
    untouched -- the cpu-trace path is bit-for-bit what it was.
    """

    if value.device.type == session.device_type:
        return value
    if real_ok:
        return value.to(session.device)
    import torch

    session.real_constants[fqn] = value.detach().cpu()
    with session.fake_mode:
        return torch.empty(
            tuple(int(d) for d in value.shape), dtype=value.dtype, device=session.device
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
def _hollow_module_moves(device: str) -> Iterator[None]:
    """``Module.to`` / ``Module.cuda`` on a HOLLOW module never swaps.

    ``Module._apply`` force-routes fake parameters through
    ``torch.utils.swap_tensors``, which refuses any tensor the fake mode holds
    a weakref to -- so author code's ``.to("cuda")`` would crash on exactly
    the modules this session hollowed. For a hollow module the move is
    IDENTITY on device -- the session already put every tensor on its ONE
    trace device, so there is nowhere else to go -- and a re-virtualization on
    dtype; real modules keep torch's own behavior.
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
                            device=device,
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


#: Device spellings author code uses that the session redirects to its own.
_DEVICE_TYPES = ("cuda", "cpu", "mps", "xpu")


def _remap_device(value: Any, device: str) -> Any:
    """Redirect any device an author names to the session's ONE trace device.

    v1 hardcoded cuda -> cpu, which is what made the derive quietly emit
    device-CPU artifacts while claiming to be device-neutral (pgw#1458).
    """

    import torch

    if isinstance(value, torch.device):
        return torch.device(device) if value.type != device.split(":", 1)[0] else value
    if isinstance(value, str) and any(
        value == name or value.startswith(f"{name}:") for name in _DEVICE_TYPES
    ):
        return device
    if isinstance(value, (list, tuple)):
        return type(value)(_remap_device(item, device) for item in value)
    if isinstance(value, dict):
        return {key: _remap_device(item, device) for key, item in value.items()}
    return value


@contextlib.contextmanager
def observation_shims(device: str = DEFAULT_TRACE_DEVICE) -> Iterator[None]:
    """The two torch-function shims the hollow drive runs under."""

    import torch
    from torch._subclasses.fake_tensor import FakeTensor
    from torch.overrides import TorchFunctionMode

    class OneTraceDevice(TorchFunctionMode):
        def __torch_function__(
            self, func: Any, types: Any, args: Any = (), kwargs: Any = None
        ) -> Any:
            if func in (torch.Tensor.cuda, torch.Tensor.cpu):
                return args[0]
            return func(
                *_remap_device(tuple(args), device),
                **_remap_device(dict(kwargs or {}), device),
            )

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

    with OneTraceDevice(), FakeEgress():
        yield


@contextlib.contextmanager
def hollow_session(device: str = DEFAULT_TRACE_DEVICE) -> Iterator[HollowSession]:
    """Everything a weights-free derive runs inside: one mode, one DEVICE.

    Instantiation (the author's ``setup``) and observation (the author's
    handlers driven with sample payloads) both run in here; derivation may
    too -- the shims never touch non-fake tensors and ``torch.export``
    re-enters the session's own mode for hollow modules.

    ``device`` is the trace DEVICE CLASS, and it is a real decision, not a
    formality: it is stamped into every node's meta and therefore into the
    graph's identity, and a mint for a different class refuses by name
    (``RuntimeCompatibility.key``). It needs no silicon -- a fake-cuda trace
    is measured working with ``torch.cuda.is_available() == False``.
    """

    from torch._subclasses.fake_tensor import FakeTensorMode

    session = HollowSession(
        fake_mode=FakeTensorMode(allow_non_fake_inputs=True), device=str(device)
    )
    with (
        _loader_interception(session),
        _hollow_module_moves(session.device),
        observation_shims(session.device),
    ):
        yield session


__all__ = [
    "DEFAULT_TRACE_DEVICE",
    "HollowError",
    "HollowSession",
    "TraceDeviceUnavailable",
    "hollow_session",
    "exported_program_devices",
    "load_exported_program",
    "observation_shims",
    "virtualize_parameters",
]
