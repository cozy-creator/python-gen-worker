from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from importlib import import_module
from pathlib import Path
from typing import Any, cast

from . import _wrapper_split
from .host_isa import HostISAError, _impose_host_policy

_COMPILER_OPTIONS: dict[str, object] = {
    # pgw#757 measured 32 -> 4 as effectively free while sharply reducing
    # contention with live serving. A later balanced 16-run 1-vs-4 cold AOTI
    # comparison put both means inside run-to-run noise (4 was 1.0% lower), so
    # there is no measured reason to fork the current worker's value. Four is
    # the one sealed policy; callers cannot select a compiler topology.
    "compile_threads": 4,
    "aot_inductor.package_constants_in_so": False,
    "aot_inductor.use_runtime_constant_folding": True,
    "aot_inductor.package": True,
}


class CompileError(RuntimeError):
    """AOTInductor did not produce a packageable code-only graph."""


def _compiler_options() -> dict[str, object]:
    """Return the sole compile policy accepted by pre-launch v1."""

    return dict(_COMPILER_OPTIONS)


def _export_context(program: object) -> tuple[object, object]:
    """Derive the sole FakeTensorMode and ShapeEnv carried by graph metadata."""

    try:
        pytree = import_module("torch.utils._pytree")
        fake_tensor = import_module("torch._subclasses.fake_tensor")
        fake_mode_type = vars(fake_tensor)["FakeTensorMode"]
    except (ImportError, KeyError) as exc:
        raise CompileError("PyTorch FakeTensor and pytree support is unavailable") from exc
    graph_module = getattr(program, "graph_module", None)
    graph = getattr(graph_module, "graph", None)
    modes: dict[int, object] = {}
    environments: dict[int, object] = {}
    for node in getattr(graph, "nodes", ()):
        metadata = getattr(node, "meta", None)
        if not isinstance(metadata, Mapping):
            continue
        for name in ("val", "example_value"):
            if name not in metadata:
                continue
            leaves = cast(Any, pytree).tree_leaves(metadata[name])
            for value in leaves:
                mode = getattr(value, "fake_mode", None)
                if mode is not None:
                    if not isinstance(mode, fake_mode_type):
                        raise CompileError("exported graph metadata carries a non-FakeTensor mode")
                    modes[id(mode)] = mode
                    environment = getattr(mode, "shape_env", None)
                    if environment is not None:
                        environments[id(environment)] = environment
                environment = getattr(getattr(value, "node", None), "shape_env", None)
                if environment is not None:
                    environments[id(environment)] = environment
                shape = getattr(value, "shape", None)
                if shape is not None:
                    for dimension in shape:
                        environment = getattr(getattr(dimension, "node", None), "shape_env", None)
                        if environment is not None:
                            environments[id(environment)] = environment
    if len(modes) != 1:
        raise CompileError(
            f"exported graph metadata must carry exactly one FakeTensorMode, found {len(modes)}"
        )
    if len(environments) != 1:
        raise CompileError(
            f"exported graph metadata must carry exactly one ShapeEnv, found {len(environments)}"
        )
    return next(iter(modes.values())), next(iter(environments.values()))


@contextmanager
def _compiling_under_export_context(program: object) -> Iterator[None]:
    """Restore the FakeTensor tracing context carried by one exported program."""

    mode, environment = _export_context(program)
    try:
        guards = import_module("torch._guards")
        tracing_context = vars(guards)["TracingContext"]
        tracing = vars(guards)["tracing"]
    except (ImportError, KeyError) as exc:
        raise CompileError("PyTorch FakeTensor tracing support is unavailable") from exc

    previous = getattr(mode, "shape_env", None)
    cast(Any, mode).shape_env = environment
    try:
        with cast(Any, tracing)(cast(Any, tracing_context)(mode)):
            yield
    finally:
        cast(Any, mode).shape_env = previous


def _aot_compile(
    graph_module: object,
    args: tuple[object, ...],
    kwargs: dict[str, object],
    *,
    options: Mapping[str, object],
) -> object:
    try:
        module = import_module("torch._inductor")
        compiler = vars(module)["aot_compile"]
    except (ImportError, KeyError) as exc:  # pragma: no cover - exercised with torch extra
        raise CompileError("PyTorch with AOTInductor is required to compile") from exc
    return cast(Any, compiler)(graph_module, args, kwargs, options=options)


def _compile_inputs(program: object) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Rebuild the call AOTInductor compiles against, from the program itself.

    pgw#1465. This used to read ``program.example_inputs``, and the derive now
    DROPS that field: a hollow trace's example inputs are fake tensors, torch
    serializes them by pickling ``_reconstruct_fake_tensor``, and torch's own
    loader then refuses that under ``weights_only=True`` -- so a blob that kept
    them could not be reloaded at all, on any device. Keeping them was not an
    option; reading them was the last place a compile depended on a field
    carried alongside the graph rather than derived from it.

    Everything needed is already in the program, exactly and by construction:

    * ``graph_signature.user_inputs`` is the FLAT call in order. A ``str``
      entry names a placeholder; anything else IS a specialized constant that
      the trace baked (``return_dict=False``), and it is carried verbatim
      because its value is part of the graph's identity.
    * each placeholder's ``meta['val']`` is its exact dtype, shape and DEVICE
      -- the same fake tensors the trace ran on, which is what a compile wants
      (pgw#1458: reading a device from anywhere else is how a mixed placement
      reaches AOTI).
    * ``call_spec.in_spec`` restores the nesting, so a pipeline's
      ``conditioning={"a": ..., "b": ...}`` comes back as the dict the
      author's ``forward`` declares rather than three loose tensors.

    Derived, so there is nothing here for a producer to get wrong -- the same
    move tcg#55 made for the graph interface.
    """

    from torch.utils import _pytree as pytree

    graph = getattr(getattr(program, "graph_module", None), "graph", None)
    signature = getattr(program, "graph_signature", None)
    in_spec = getattr(getattr(program, "call_spec", None), "in_spec", None)
    if graph is None or signature is None or in_spec is None:
        raise CompileError(
            "program cannot state its own call: one of graph_module.graph, "
            "graph_signature or call_spec.in_spec is absent"
        )
    values = {
        node.name: node.meta.get("val")
        for node in graph.nodes
        if getattr(node, "op", "") == "placeholder"
    }
    leaves: list[Any] = []
    for entry in getattr(signature, "user_inputs", ()) or ():
        if not isinstance(entry, str):
            # A specialized constant the trace baked in. Its VALUE is part of
            # the graph, so it is replayed exactly, never re-defaulted.
            leaves.append(entry)
            continue
        if entry not in values:
            raise CompileError(
                f"exported program names user input {entry!r} with no matching "
                f"placeholder; its graph and signature disagree"
            )
        leaves.append(values[entry])
    try:
        rebuilt = pytree.tree_unflatten(leaves, in_spec)
    except Exception as exc:
        raise CompileError(
            f"cannot rebuild the exported program's call from its own signature "
            f"({len(leaves)} flat input(s)) and in_spec: {type(exc).__name__}: {exc}"
        ) from exc
    try:
        args, kwargs = rebuilt
    except (TypeError, ValueError) as exc:
        raise CompileError(
            f"exported program in_spec does not unflatten to (args, kwargs); "
            f"got {type(rebuilt).__name__}"
        ) from exc
    return tuple(args), dict(kwargs or {})


def _compile_exported_program(program: object) -> tuple[object, ...]:
    """Compile one ExportedProgram to the loose files used by `package_aoti`.

    FakeTensor and ShapeEnv state is derived only from ``program``. Compile
    settings are fixed because every setting that can affect generated code
    must have exactly one v1 identity.
    """
    try:
        _impose_host_policy()
        _wrapper_split.install()
    except HostISAError as exc:
        raise CompileError(f"cannot establish host ISA policy: {exc}") from exc

    exported_module = getattr(program, "module", None)
    if not callable(exported_module):
        raise CompileError("program has no callable module(check_guards=False)")
    args, kwargs = _compile_inputs(program)

    try:
        with _compiling_under_export_context(program):
            result = _aot_compile(
                exported_module(check_guards=False),
                tuple(args),
                dict(kwargs or {}),
                options=_compiler_options(),
            )
    except Exception as exc:
        raise CompileError(f"AOTInductor compile failed: {type(exc).__name__}: {exc}") from exc
    if not isinstance(result, list) or not result:
        raise CompileError("AOTInductor did not return a non-empty loose-file list")
    return tuple(result)


def _package_aoti(output: str, files: Mapping[str, Sequence[object]]) -> object:
    try:
        module = import_module("torch._inductor.package")
        packager = vars(module)["package_aoti"]
    except (ImportError, KeyError) as exc:  # pragma: no cover - exercised with torch extra
        raise CompileError("PyTorch with AOTInductor is required to package") from exc
    return cast(Any, packager)(output, files)


def _package_compiled_files(
    name: str,
    files: Sequence[object],
    output: str | Path,
) -> Path:
    """Package one named graph specialization into a `.pt2` file."""

    name = str(name).strip()
    if not name:
        raise CompileError("graph specialization name must not be empty")
    if not files:
        raise CompileError("cannot package a graph specialization with no compiled files")
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = _package_aoti(str(target), {name: list(files)})
    except Exception as exc:
        raise CompileError(f"AOTInductor packaging failed: {type(exc).__name__}: {exc}") from exc
    return Path(str(result))
