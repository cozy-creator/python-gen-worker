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
* **Lazy pipeline completion** (tcg#65) -- ``diffusers.ModularPipeline.from_pretrained``
  returns a pipeline whose ``from_pretrained`` components are all ``None``:
  it builds SPECS and defers the build to whoever holds the object. At serve
  that caller exists; a derive has none, so it marked an empty pipeline and
  armed nothing. The wrapped loader finishes the build in-session, so every
  component is born hollow through the loaders above.
* **Two devices, named honestly** (tcg#64) -- the STATED trace device, which
  the graphs are stamped with, and the DRIVE device, where the run's real
  tensors actually live. They are the same device whenever this host can hold
  a real tensor there, and they differ on exactly one shape: a GPU-less box
  stating ``cuda``.

  Splitting them is what makes a normal trace just work. The stated device
  used to be forced onto the drive as well, so a GPU-less ``cuda`` derive had
  to fake every real tensor the author's code touched -- and a full diffusers
  pipeline does not survive that: ``encode_prompt`` moves real token ids to
  the execution device, the scheduler reads real sigmas with ``.item()``, and
  fakifying them either runs a real cuda kernel on a box with no GPU or trips
  fake-tensor data-dependence. The drive now runs on a device that EXISTS, so
  every one of those reads gets its real value, exactly as it does on a GPU.

  It costs nothing in identity because the drive does not decide identity.
  Discovery is two passes: the drive only records each marked module's call
  STRUCTURE (dtype and shape), and the graph is produced by a second,
  independent ``torch.export`` of that module alone. So the drive's device
  reaches the graph through one channel only -- the device stamped into the
  exported program -- and :func:`restate_program` restates it afterwards.
  Measured line-for-line: a cpu export restated onto cuda is byte-identical
  to a native fake-cuda export, and a GPU-less sd1.5 derive reproduces the
  graph keys of the GPU-traced one.

  The stated device is still a REAL decision (pgw#1458): it is stamped into
  every node's meta and therefore into the graph's identity, and a mint for a
  different class refuses by name (``RuntimeCompatibility.key``). What is
  gone is the HOST deciding it -- ``cuda`` needs no silicon.

  What was never device is the ARCH. Device (``cpu`` vs ``cuda``) is not arch
  (``sm_86`` vs ``sm_90``): one cuda trace serves every ``sm``, because the sm
  enters at ARTIFACT level through ``RuntimeCompatibility``.

  Device uniformity must be TOTAL on the stated side. Parameters, buffers AND
  lifted tensor constants all carry the stated device; leaving any one
  real-on-cpu yields a MIXED ``['cpu','cuda:0']`` placement that exports
  cleanly and only AOTI rejects.
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


#: The device class a compiled-graph derive traces on unless told otherwise.
#: The fleet's compiled graphs are all cuda, and this needs no GPU to state
#: (tcg#64). A cpu trace is for tests and for cpu-target endpoints, and it is a
#: DIFFERENT graph specialization, never a fallback for a missing device.
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
    """The one fake mode, the STATED trace device, and where the drive runs.

    ``device`` is what the graphs are stamped with and therefore part of their
    identity. :attr:`drive_device` is where the run's real tensors physically
    live; it is ``device`` whenever this host can hold one there and ``cpu``
    otherwise. A GPU-less ``cuda`` session is the whole point of the split.

    Nothing value-bearing is ever faked. The session that HAD to fake buffers
    and lifted constants -- and therefore had to carry a table of their real
    values to keep the literal digest honest -- was the session that forced
    the stated device onto the drive. It does not any more (tcg#64), so that
    table is gone rather than empty.
    """

    fake_mode: Any
    device: str = DEFAULT_TRACE_DEVICE

    @property
    def device_type(self) -> str:
        return str(self.device).split(":", 1)[0]

    @property
    def materializes_real_tensors(self) -> bool:
        """Can a REAL tensor live on this session's STATED device, here, now?

        cpu always. cuda only with visible silicon -- and the publish derive
        deliberately runs on boxes without any, which is what
        :attr:`drive_device` answers.
        """

        if self.device_type == "cpu":
            return True
        import torch

        return bool(torch.cuda.is_available())

    @property
    def drive_device(self) -> str:
        """Where the author's code actually runs -- a device that EXISTS.

        The single place the fallback is decided, so no caller can compute a
        second answer. Every real value the drive reads (``.item()`` on a
        sigma, token ids moved to the execution device, an ``encode_prompt``
        kernel) is a real value on this device.
        """

        return self.device if self.materializes_real_tensors else "cpu"

    @property
    def drive_device_type(self) -> str:
        return str(self.drive_device).split(":", 1)[0]

    def restate(self, program: Any) -> tuple[str, ...]:
        """Restate one exported program on this session's STATED device.

        A no-op when the drive already ran there. Otherwise it is the ONLY
        place the drive's device is corrected, and it happens before the graph
        is hashed, so the key a GPU-less box derives is the key a GPU box
        derives. Returns the value-bearing tensors that had to stay on the
        drive device (see :func:`restate_program`); ``()`` for a weights-free
        program, which is every program this session produces today.
        """

        if self.drive_device_type == self.device_type:
            return ()
        return restate_program(program, self.device)


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

    The loaded program is then restated onto its own GRAPH's device (tcg#64).
    That is a no-op for a weights-free program, which is fake all the way
    through and therefore uniform already. It matters for the one case that is
    not: a value-bearing buffer a GPU-less derive could not move stayed real
    on cpu under a cuda graph, and this -- running where the device exists --
    is where it lands.
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
    program = torch.export.load(str(path))
    graph_device = program_graph_device(program)
    if graph_device is not None:
        restate_program(program, graph_device)
    return program


def program_graph_device(program: Any) -> Any:
    """The device an exported program's GRAPH is stated on, or ``None``.

    The graph's device -- not the state dict's -- is the one that entered
    ``cg-graph-v1`` and the one AOTI compiles for.
    """

    import torch

    for node in program.graph_module.graph.nodes:
        value = node.meta.get("val")
        for item in value if isinstance(value, (list, tuple)) else [value]:
            if isinstance(item, torch.Tensor):
                return item.device
    return None


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
    """Move every tensor of ``module`` onto the session's DRIVE device.

    ``dtype`` restates the author's ``torch_dtype=`` ask: floating parameters
    take it, integer parameters keep their own -- exactly what a real
    ``from_pretrained(torch_dtype=...)`` load produces.

    Parameters are always faked (a 10 GiB denoiser must cost nothing).
    Buffers and lifted constants stay REAL -- the drive device is a device
    that exists, so there is never a reason to destroy a value here. The
    stated device is applied afterwards, to the exported program
    (:func:`restate_program`), which is the only thing that carries it.
    """

    import torch

    device = session.drive_device
    for _prefix, submodule in module.named_modules():
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
            submodule._buffers[name] = _rehome(buffer, device)
        for name, value in _tensor_attributes(submodule):
            setattr(submodule, name, _rehome(value, device))


def _rehome(value: Any, device: str) -> Any:
    """Put one value-bearing tensor on the drive device, keeping its value.

    A real move, always: the drive device can hold a real tensor by
    construction. A tensor already there is untouched.
    """

    if value.device.type == str(device).split(":", 1)[0]:
        return value
    return value.to(device)


def restate_program(program: Any, device: Any) -> tuple[str, ...]:
    """Restate an ``ExportedProgram``'s device, in place. Returns what stayed.

    The whole device content of an exported program is METADATA: the fake
    tensor in each node's ``meta['val']``, the ``device=`` a factory node
    burnt in, and the placement of the state dict and the constant table.
    None of it is computation, which is why this runs on a host that has no
    such device at all -- and why the graph a GPU-less box derives is the same
    graph, key for key, that a GPU box derives. Measured line-for-line against
    a native fake-cuda export.

    A weights-free program is FAKE all the way through, so the usual case
    moves everything and returns ``()``. A tensor carrying a real VALUE (a
    computed buffer, a lifted constant) can only be moved where a real tensor
    can live: on a host without the device it stays where it is and its name
    is returned. That is deliberate -- a value is the one thing a restate must
    never destroy, the literal digest reads it (``.cpu()``, so the placement
    does not change the key), and the mint re-homes it on a machine that has
    the device (:func:`load_exported_program`).
    """

    import torch
    from torch._subclasses.fake_tensor import FakeTensor

    # The restate runs INSIDE the session, so `OneTraceDevice` is live -- and
    # `torch.device` construction goes through `__torch_function__`, so
    # `torch.device("cuda", 0)` evaluated in here comes back `cpu:0`. The shim
    # is doing its job (redirect every device ask to the drive device); it
    # simply cannot tell the session's own repair from author code asking for
    # a device it must not get. Measured, and it fails SILENTLY: every move
    # became a no-op and the derive emitted cpu-stamped graphs while reporting
    # a cuda trace. So the whole restate is the one thing in the session that
    # opts out of the session's own redirection.
    with torch._C.DisableTorchFunction():
        return _restate(program, device, torch, FakeTensor)


def _restate(program: Any, device: Any, torch: Any, FakeTensor: Any) -> tuple[str, ...]:
    target = torch.device(device)
    if target.type != "cpu" and target.index is None:
        # torch normalizes a device-less `cuda` ask to `cuda:0` on the tensors
        # it creates, so a restate that left it un-indexed would produce a
        # `placement` of ('cuda',) where a native trace says ('cuda:0',).
        target = torch.device(target.type, 0)
    stranded: list[str] = []
    # A non-strict export does not copy the module's parameters: the same
    # FakeTensor object is the module's parameter, the program's state-dict
    # entry AND a placeholder's `meta['val']`. So re-homing by assigning
    # `fake_device` re-homes the LIVE MODULE, and the next observed call of the
    # same target then exports cuda parameters against cpu synthesized inputs
    # -- `FakeTensorDeviceMismatchError: cpu and cuda:0`, raised from a module
    # nobody moved. Substituting a fresh fake tensor keeps the restate a
    # property of the PROGRAM, which is what it is supposed to be. The memo
    # preserves aliasing: one input object still maps to one output object, so
    # a program that shared a tensor across nodes still shares it.
    moved: dict[int, Any] = {}

    def move(value: Any, fqn: str = "") -> Any:
        if not isinstance(value, torch.Tensor) or value.device.type == target.type:
            return value
        if isinstance(value, FakeTensor):
            known = moved.get(id(value))
            if known is None:
                with value.fake_mode:
                    known = torch.empty_strided(
                        value.shape, value.stride(), dtype=value.dtype, device=target
                    )
                known.requires_grad_(value.requires_grad)
                if getattr(value, "_is_param", False):
                    known._is_param = True
                moved[id(value)] = known
            return known
        try:
            return value.to(target)
        except (RuntimeError, AssertionError, AttributeError):
            stranded.append(fqn or f"<{value.device.type} node value>")
            return value

    for holder in ("_state_dict", "_constants"):
        table = getattr(program, holder, None)
        if not isinstance(table, dict):
            continue
        for name, value in list(table.items()):
            table[name] = move(value, name)

    for graph_module in program.graph_module.modules():
        if not isinstance(graph_module, torch.fx.GraphModule):
            continue
        for node in graph_module.graph.nodes:
            if "device" in node.kwargs:
                node.kwargs = {**node.kwargs, "device": target}
            if node.op == "call_function" and node.target is torch.ops.aten.to.device:
                arguments = list(node.args)
                arguments[1] = target
                node.args = tuple(arguments)
            for key in ("val", "example_value"):
                if key in node.meta:
                    node.meta[key] = torch.utils._pytree.tree_map(move, node.meta[key])

    return tuple(sorted(set(stranded)))


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


def attach_lazy_components(pipeline: Any, *, dtype: Any = None) -> tuple[str, ...]:
    """Build the components a LAZY loader declared and did not construct.

    A classic ``DiffusionPipeline.from_pretrained`` returns a pipeline
    CARRYING every component, so intercepting the component loaders is
    enough to make that load hollow. A lazy pipeline loader returns the same
    registry with every ``from_pretrained`` component ``None`` and defers the
    build to whoever holds the object -- at serve, the streaming engine that
    passes the components in. A derive has no such caller: it gets exactly
    what the loader returned and marks what the pipeline carries, which is
    nothing (pgw#1450 -- eleven declared names, eleven ``None`` attributes,
    "no DiT resolves on this pipeline").

    Duck-typed on the object's own surface, never on its class: an object
    that NAMES unbuilt components and EXPOSES the call that builds them gets
    that call made, here, inside the session -- so each component goes
    through the intercepted loaders and is born hollow. An object without
    that surface is left alone; nothing here knows any pipeline by name.

    Then the postcondition, because the lazy build swallows per-component
    failures into a log line (tcg#59's doctrine: intercept-or-REFUSE, never
    silently pass through). A component the pipeline says it can build and
    then has not built is named and refused -- it is the difference between
    a traced DiT and a derive that arms nothing.

    Returns the names attached, ``()`` when there was nothing to do.
    """

    build = getattr(pipeline, "load_components", None)
    if not callable(build):
        return ()
    unbuilt = _buildable_names(pipeline)
    if not unbuilt:
        return ()
    if dtype is None:
        build(names=list(unbuilt))
    else:
        build(names=list(unbuilt), torch_dtype=dtype)
    still_missing = _buildable_names(pipeline)
    if still_missing:
        raise HollowError(
            f"{type(pipeline).__name__} declares component(s) "
            f"{list(still_missing)} it can build and did not build them; the "
            f"pipeline carries None under those names, so a trace would mark "
            f"nothing and the derive would arm no graphs. The build reports "
            f"the per-component cause on the diffusers logger."
        )
    return unbuilt


def _buildable_names(pipeline: Any) -> tuple[str, ...]:
    """Declared components that are unset AND have somewhere to load from.

    A declared-but-absent optional (no source in the index) is legitimately
    ``None`` and is not a defect -- the same distinction the lazy build makes
    when it skips a spec with no source.
    """

    specs = getattr(pipeline, "_component_specs", None)
    if not isinstance(specs, Mapping):
        return ()
    names = []
    for name, spec in specs.items():
        if getattr(spec, "default_creation_method", None) != "from_pretrained":
            continue
        if not getattr(spec, "pretrained_model_name_or_path", None):
            continue
        if getattr(pipeline, name, None) is None:
            names.append(name)
    return tuple(names)


def _hollow_lazy_pipeline(original: Any) -> Any:
    """Wrap a pipeline loader that returns SPECS where a load returns objects.

    ``torch_dtype`` is consumed here rather than forwarded: a lazy pipeline
    constructor does not take it (it lands in ``**kwargs`` and is dropped
    with a warning, or refused by a strict author subclass), while the
    component build it defers is exactly where the dtype belongs. So the ask
    reaches the components, which is what a classic load does with it.
    """

    def from_pretrained(cls: Any, /, *args: Any, **kwargs: Any) -> Any:
        dtype = kwargs.pop("torch_dtype", None)
        if dtype is None:
            dtype = kwargs.pop("dtype", None)
        else:
            kwargs.pop("dtype", None)
        pipeline = original.__func__(cls, *args, **kwargs)
        attach_lazy_components(pipeline, dtype=dtype)
        return pipeline

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

    Library-level, never family-level: the patch knows ``ModelMixin``,
    ``PreTrainedModel`` and ``ModularPipeline``, not what any endpoint builds
    with them. A library that is not installed is simply not patched --
    pure-transformers authors and pure-diffusers authors both resolve.

    The first two are the WEIGHT-BEARING loaders and are REPLACED: they build
    from config and virtualize. The third is a LAZY pipeline loader and is
    WRAPPED: it does not read weights, it hands back a registry of unbuilt
    components, and the wrap makes it finish the job (:func:`attach_lazy_components`).
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
    try:
        import diffusers.modular_pipelines.modular_pipeline as _dmp
    except ImportError:
        _dmp = None
    if _dmu is None and _tmu is None:
        raise HollowError(
            "hollow instantiation intercepts diffusers and/or transformers "
            "loaders, and neither library is importable in this env"
        )
    try:
        if _dmp is not None:
            patched.append(
                (_dmp.ModularPipeline, _dmp.ModularPipeline.__dict__["from_pretrained"])
            )
            _dmp.ModularPipeline.from_pretrained = _hollow_lazy_pipeline(
                _dmp.ModularPipeline.__dict__["from_pretrained"]
            )
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

    #: Every op that takes a tensor OFF the device and onto the host. A fake
    #: tensor answers these with zeros; a real one has to be copied first.
    _HOST_EGRESS = (
        torch.Tensor.numpy,
        torch.Tensor.__array__,
        torch.Tensor.tolist,
        torch.Tensor.item,
    )

    class FakeEgress(TorchFunctionMode):
        def __torch_function__(
            self, func: Any, types: Any, args: Any = (), kwargs: Any = None
        ) -> Any:
            kwargs = kwargs or {}
            host = args[0] if args else None
            # tcg#57: the session put this tensor on the trace device — the
            # author never asked for it there — so the session owes the
            # `.cpu()` at host egress. Outside a session the same code works
            # because the tensor was on cpu to begin with; a redirection that
            # is visible to the author is a redirection that breaks them.
            #
            # Real, not hypothetical: diffusers' `EulerDiscreteScheduler.
            # set_timesteps` does `np.array(sigmas)`, which reaches
            # `Tensor.__array__` — note that is a DIFFERENT function from
            # `Tensor.numpy`, so listing only the latter (as this shim did)
            # misses the numpy-protocol path entirely.
            if (
                func in _HOST_EGRESS
                and isinstance(host, torch.Tensor)
                and not isinstance(host, FakeTensor)
                and host.device.type != "cpu"
            ):
                # The copy MUST run with torch-function dispatch disabled.
                # `OneTraceDevice` answers `Tensor.cpu` with `args[0]`
                # unchanged — deliberately, so author code cannot pull weights
                # off the trace device — which makes a plain `host.cpu()` here
                # a silent no-op that returns the cuda tensor and fails one
                # frame later. Disabling dispatch is what distinguishes "the
                # session moving its own tensor back" from "the author trying
                # to re-home one".
                with torch._C.DisableTorchFunction():
                    moved = host.detach().to("cpu")
                return func(moved, *args[1:], **kwargs)
            if isinstance(host, FakeTensor):
                # `__array__` alongside `numpy`: `np.array(t)` and `t.numpy()`
                # are the same egress through two different functions, and a
                # fake tensor has to answer both (tcg#57).
                if func in (torch.Tensor.numpy, torch.Tensor.__array__):
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

    # tcg#57: the DEFAULT device has to be the trace device too, or the session
    # has two devices while claiming one.
    #
    # `_remap_device` only rewrites a device the author NAMES. A factory call
    # that names none (`torch.linspace(0, 1, n)`) is not remapped and lands on
    # cpu — so inside a cuda session, unnamed factories produce cpu tensors
    # while named ones produce cuda tensors. Author code that round-trips a
    # device through a tensor it just made gets the two mixed:
    #
    #     sigmas = torch.linspace(...)                      # -> cpu (unnamed)
    #     torch.cat([sigmas, torch.zeros(1, device=sigmas.device)])
    #                              # `sigmas.device` is cpu, REMAPPED to cuda
    #                              # -> cat(cpu, cuda) -> RuntimeError
    #
    # That is real, not hypothetical: it is diffusers' `EulerDiscreteScheduler.
    # __init__`, so every SDXL derive died in `load()` before tracing anything.
    # Setting the default makes the unnamed case land where the named case is
    # already redirected, which is what "ONE trace device" was always supposed
    # to mean — and it makes `_remap_device` a no-op for correct author code
    # rather than the thing that breaks it.
    with torch.device(device), OneTraceDevice(), FakeEgress():
        yield


@contextlib.contextmanager
def hollow_session(device: str = DEFAULT_TRACE_DEVICE) -> Iterator[HollowSession]:
    """Everything a weights-free derive runs inside: one mode, one DEVICE.

    Instantiation (the author's ``setup``) and observation (the author's
    handlers driven with sample payloads) both run in here; derivation may
    too -- the shims never touch non-fake tensors and ``torch.export``
    re-enters the session's own mode for hollow modules.

    ``device`` is the STATED trace device class, and it is a real decision,
    not a formality: it is stamped into every node's meta and therefore into
    the graph's identity, and a mint for a different class refuses by name
    (``RuntimeCompatibility.key``). It needs no silicon (tcg#64): the drive
    runs on ``session.drive_device`` -- a device that exists -- and each
    exported program is restated onto the stated device before it is hashed.
    """

    from torch._subclasses.fake_tensor import FakeTensorMode

    session = HollowSession(
        fake_mode=FakeTensorMode(allow_non_fake_inputs=True), device=str(device)
    )
    with (
        _loader_interception(session),
        _hollow_module_moves(session.drive_device),
        observation_shims(session.drive_device),
    ):
        yield session


__all__ = [
    "DEFAULT_TRACE_DEVICE",
    "HollowError",
    "attach_lazy_components",
    "HollowSession",
    "TraceDeviceUnavailable",
    "hollow_session",
    "exported_program_devices",
    "load_exported_program",
    "observation_shims",
    "program_graph_device",
    "restate_program",
    "virtualize_parameters",
]
