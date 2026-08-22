"""Making the artifact: bind, compile, package.

Torch does the compiling. What this module owns is everything the compile must
be honest ABOUT -- the options it ran under, the host ISA it clamped to, the
shapes it actually specialized, and the refusal when any of those disagree with
what the artifact would claim.
"""

from __future__ import annotations

import json
import tempfile
import threading
from collections.abc import Iterator, Mapping
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, replace
from importlib import import_module
from pathlib import Path
from typing import Any, cast

from .identity import (
    CallIngress,
    _host_isa,
    constant_names,
    contiguous_handle,
    graph_hash,
    literal_names,
    placement,
    require_morphism,
)
from .refuse import BindError, DroppedOptimization, MintError, RangeNarrowed

# ---------------------------------------------------------------------------
# The compile policy -- five options, sealed
# ---------------------------------------------------------------------------

#: `always_keep_tensor_constants=True` is the tcg#80 fold policy. Read
#: `GraphLowering.get_attr`: a lifted constant takes the bindable
#: `add_tensor_constant` path when EITHER `use_runtime_constant_folding` or
#: `always_keep_tensor_constants` is set, and is otherwise inlined into
#: generated code whenever its shape is () or 1-D with <= 8 elements. The two
#: are not equivalent -- `use_runtime_constant_folding` additionally splits a
#: const-fold subgraph that runs on the first call and materializes a SECOND
#: full constant set by direct cudaMalloc outside the caching allocator
#: (+4782 MiB first-call transient on sdxl UNet-only). This buys the same
#: bindable table with no second set.
POLICY: dict[str, bool | int | str] = {
    "compile_threads": 4,
    "aot_inductor.package_constants_in_so": False,
    "always_keep_tensor_constants": True,
    "aot_inductor.use_runtime_constant_folding": False,
    "aot_inductor.package": True,
}

#: Options that change what inductor EMITS. Only these are keyed: a topology
#: option changes how fast the compile runs, never its output.
_CODEGEN = frozenset((
    "aot_inductor.package_constants_in_so",
    "always_keep_tensor_constants",
    "aot_inductor.use_runtime_constant_folding",
    "aot_inductor.package",
    "max_autotune",
    "max_autotune_gemm",
    "coordinate_descent_tuning",
))
_TOPOLOGY = frozenset(("compile_threads",))
_INERT = frozenset(("triton.cudagraphs",))

#: Options this build accepts the target silently dropping. Empty on purpose:
#: an unsanctioned drop is a refusal, because the artifact would key as if the
#: lever ran and compile as if it did not.
ACCEPT_DROPPED: frozenset[str] = frozenset()


def _gemm_templates_available(device_type: str) -> str:
    """"" when the target can use GEMM templates, else why it cannot.

    The verdict is READ from torch (`is_big_gpu`), never a copied SM threshold,
    and read UNCACHED: a `functools.cache`d verdict is a guard that cannot go
    red. Measured motivation: `max_autotune: True` on an RTX 4070 Laptop moved
    the key, cost 622.5 s against an 85.2 s baseline, and torch logged
    `Not enough SMs to use max_autotune_gemm mode` -- 36 SMs against its
    internal `min_sms = 68`.
    """

    if device_type != "cuda":
        return ""
    import torch

    try:
        utils = import_module("torch._inductor.utils")
        is_big_gpu = vars(utils)["is_big_gpu"]
        properties_type = vars(import_module("torch._inductor.utils"))["DeviceProperties"]
    except (ImportError, KeyError) as exc:
        raise MintError(f"torch does not expose its GEMM-template gate: {exc}") from exc
    index = torch.cuda.current_device() if torch.cuda.is_available() else 0
    device = torch.device("cuda", index)
    gate = getattr(is_big_gpu, "__wrapped__", is_big_gpu)
    if gate(device):
        return ""
    properties = properties_type.create(device)
    return (
        f"torch._inductor.utils.is_big_gpu refuses this device: "
        f"{properties.multi_processor_count} SMs at compute capability {properties.cc}"
    )


_PRECONDITIONS = {
    "max_autotune": _gemm_templates_available,
    "max_autotune_gemm": _gemm_templates_available,
}


def compile_policy(device_type: str = "cpu") -> dict[str, bool | int | str]:
    """The codegen options this target will ACTUALLY run under.

    Classify or refuse: an option in none of the three sets is a `MintError`,
    because an unclassified option is one nobody decided whether the key must
    carry. A requested lever the target would silently drop is a refusal, not a
    quiet downgrade.
    """

    unclassified = sorted(set(POLICY) - _CODEGEN - _TOPOLOGY - _INERT)
    if unclassified:
        raise MintError(
            f"compile options {unclassified!r} are classified neither codegen, "
            f"topology nor inert; nobody has decided whether the key carries them"
        )
    dropped = []
    for name, value in POLICY.items():
        precondition = _PRECONDITIONS.get(name)
        if precondition is None or not value:
            continue
        shortfall = precondition(device_type)
        if shortfall:
            dropped.append((name, shortfall))
    unsanctioned = sorted(n for n, _ in dropped if n not in ACCEPT_DROPPED)
    if unsanctioned:
        detail = "; ".join(f"{n}: {d}" for n, d in dropped)
        raise DroppedOptimization(
            f"compile policy requests {unsanctioned!r}, which this target will "
            f"SILENTLY DROP -- {detail}. The artifact would key differently and "
            f"compile identically-minus-the-lever, so the fleet re-mints for "
            f"nothing. Mint on a target that clears the precondition, or add the "
            f"option to ACCEPT_DROPPED so the key states what actually ran."
        )
    return {name: POLICY[name] for name in sorted(POLICY) if name in _CODEGEN}


def declared_input_layout() -> str:
    """The layout this build compiles its inputs against, READ from tensorfs."""

    return contiguous_handle()


# ---------------------------------------------------------------------------
# Host ISA -- the one env fact this package owns, and it fails closed
# ---------------------------------------------------------------------------

#: x86-64 microarchitecture levels, keyed by the names **`/proc/cpuinfo`
#: actually prints** -- which are NOT the names the psABI document uses. Linux
#: reports the LZCNT capability as `abm` and never emits `lzcnt`, so a table
#: written from the spec matches nothing and silently demotes every Intel host
#: one level: measured here on an i9-13950HX, which has the full v3 feature set
#: and was classified v2, costing AVX2 and FMA in every artifact it minted.
#: The `f16c`/`xsave`/`lahf_lm` members are part of the levels and are equally
#: easy to leave out; they are listed for the same reason.
_LOCK = threading.RLock()


def impose_host_policy() -> dict[str, str]:
    """Clamp inductor's target ISA, and PROVE the clamp took, process-wide.

    A config that reads back correctly on this thread and natively on another
    is worse than no clamp: the artifact then carries instructions the fleet
    cannot execute while every local check says it is fine.
    """

    machine, march, simdlen = _host_isa()
    with _LOCK:
        try:
            config = import_module("torch._inductor.config")
        except ImportError as exc:
            raise MintError("AOTInductor is required to impose host ISA") from exc
        # The DEFAULT first, then the value. torch's ConfigModule resolves a
        # read against a thread-local override layered over a process-wide
        # default, so assigning only the attribute clamps THIS thread and leaves
        # inductor's compile workers building for the native ISA -- which is the
        # exact failure the readback below exists to catch.
        entries = getattr(config, "_config", {})
        for name, value in (("cpp.march", march), ("cpp.simdlen", simdlen)):
            entry = entries.get(name)
            if entry is None:
                raise MintError(f"torch config states no process-wide {name!r} default")
            entry.default = value
        config.cpp.march = march
        config.cpp.simdlen = simdlen
        if (config.cpp.march, config.cpp.simdlen) != (march, simdlen):
            raise MintError(
                f"host ISA clamp reads {(config.cpp.march, config.cpp.simdlen)!r}, "
                f"expected {(march, simdlen)!r}"
            )
        seen: list[Any] = []
        thread = threading.Thread(
            target=lambda: seen.append((config.cpp.march, config.cpp.simdlen))
        )
        thread.start()
        thread.join(timeout=None)
        if not seen:
            raise MintError("fresh-thread ISA readback did not complete")
        if seen[0] != (march, simdlen):
            raise MintError(
                f"host ISA clamp is thread-local: a fresh thread reads {seen[0]!r}"
            )
    return {
        "machine": machine,
        "cpp_march": march or "",
        "cpp_simdlen": "" if simdlen is None else str(simdlen),
    }


# ---------------------------------------------------------------------------
# The static bind (tcg#88): the symbolic parent is the ONE trace
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GraphSpec:
    """One graph to mint: its identity, its program, and its call contract."""

    graph: str
    target: str
    program: Any
    ingress: CallIngress
    passes: tuple[str, ...] = ()


def strip_diagnostics(program: Any) -> Any:
    """Drop per-node provenance strings. Measured on sd15: 1.8 MB of a 3.1 MB
    graph JSON, ~60%. Nothing on the mint path reads them -- the graph hash
    renders ops, args and values, never `node.meta` provenance."""

    for node in program.graph_module.graph.nodes:
        for key in ("stack_trace", "nn_module_stack", "source_fn_stack", "torch_fn"):
            node.meta.pop(key, None)
    return program


def flatten_weight_subclasses(program: Any) -> tuple[str, ...]:
    """Replace unpicklable weight WRAPPERS with the tensor they wrap (pgw#1662).

    `torch.export.save` pickles `state_dict`. A traceable wrapper subclass —
    torchao's `Float8Tensor`, and every quantized wrapper after it — falls to
    the generic pickle path and drags in the inner `FakeTensorMode`'s weakref
    closure, which cannot be pickled. h3 dies on 1442 such entries.

    **What is wrong is the WRAPPER, not the weight**, and the first version of
    this got that wrong: it replaced each entry with a meta tensor, which
    pgw#1465's fence caught immediately — a meta state dict against a
    device-stamped graph is TWO DEVICE STORIES, and AOTI refuses it. The blob
    keeps one device story.

    So this unwraps instead of erasing. The inner tensor is the same device and
    the same virtuality; only the subclass that could not be pickled is gone.
    A plain fake tensor pickles fine, which is why every non-quantized model
    already banks cleanly. Returns the names it flattened, so a caller can SAY
    what it did rather than doing it silently.

    **It cannot move the graph key.** `graph_hash` renders the canonical GRAPH
    plus the ingress plus the literal digest, and `literal_names` is
    `constants - (parameters | buffers)` — the state dict is read for names,
    dtypes and shapes only, and none of those move here.
    """

    import torch
    from torch.utils._python_dispatch import is_traceable_wrapper_subclass

    state = getattr(program, "state_dict", None)
    if not state:
        return ()
    flattened: list[str] = []
    for name, value in list(state.items()):
        if not isinstance(value, torch.Tensor):
            continue
        if not is_traceable_wrapper_subclass(value):
            continue
        try:
            names, _ctx = value.__tensor_flatten__()
            inner = [getattr(value, str(held)) for held in names]
        except Exception:  # noqa: BLE001 - an unwalkable wrapper is left alone
            continue
        if len(inner) != 1 or not isinstance(inner[0], torch.Tensor):
            # Several held tensors is not one weight -- flattening would have
            # to invent which one the state dict entry IS. Leave it and let the
            # save refuse honestly rather than guess.
            continue
        # PARAMETER-NESS SURVIVES: `torch.export.save` refuses a state-dict
        # entry for a parameter that is not an `nn.Parameter`.
        state[name] = (
            torch.nn.Parameter(inner[0], requires_grad=False)
            if isinstance(value, torch.nn.Parameter)
            else inner[0]
        )
        flattened.append(str(name))
    return tuple(sorted(flattened))


def _fake_mode_of(graph_module: Any) -> Any:
    import torch

    modes = {
        id(value.fake_mode): value.fake_mode
        for value in graph_module.parameters()
        if isinstance(value, torch._subclasses.fake_tensor.FakeTensor)
    }
    if not modes:
        return nullcontext()
    if len(modes) > 1:
        raise BindError(
            "the parent program holds fake parameters from two fake modes; it "
            "cannot be re-exported under one"
        )
    return next(iter(modes.values()))


def respecialize(program: Any, ingress: CallIngress, *, strict: bool = False) -> Any:
    """Re-export the symbolic parent at ONE record's concrete shapes.

    ConstantArguments are replayed verbatim -- their values are part of the
    graph's identity. Tensors become zeros in the parent's own dtype and device.
    """

    import torch
    from torch.utils import _pytree as pytree

    rows = {row.exported_name: row for row in ingress.inputs}
    placeholders = {
        str(node.name): node.meta.get("val")
        for node in program.graph_module.graph.nodes
        if node.op == "placeholder"
    }
    leaves: list[Any] = []
    with _fake_mode_of(program.graph_module):
        for spec in program.graph_signature.input_specs:
            if spec.kind != torch.export.graph_signature.InputKind.USER_INPUT:
                continue
            if isinstance(spec.arg, torch.export.graph_signature.ConstantArgument):
                leaves.append(spec.arg.value)
                continue
            name = spec.arg.name
            row = rows.get(name)
            template = placeholders.get(name)
            if row is None or template is None:
                raise BindError(
                    f"the record's ingress names no input for the parent's "
                    f"placeholder {name!r}; the parent cannot restate this record"
                )
            if any(not isinstance(d, int) for d in row.shape):
                raise BindError(
                    f"ingress input {row.name!r} still carries a symbol in "
                    f"{row.shape!r}; a static bind needs concrete shapes"
                )
            leaves.append(
                torch.zeros(
                    tuple(int(d) for d in row.shape),
                    dtype=template.dtype,
                    device=template.device,
                )
            )
        args, kwargs = pytree.tree_unflatten(leaves, program.call_spec.in_spec)
        return torch.export.export(program.module(), args, kwargs, strict=strict)


def bind_static_spec(spec: GraphSpec) -> GraphSpec:
    """Re-derive a static record from the banked symbolic parent, or REFUSE.

    The derive exports one symbolic parent per structural variant and banks that
    parent under every bucket's identity (the CAS dedups the bytes to one blob),
    so a compile position asking for a static record can be handed a symbolic
    program. Re-specializing and re-hashing is what makes that safe: the bind is
    accepted only when it derives the identity that was asked for.
    """

    if not getattr(spec.program, "range_constraints", None):
        return spec
    if spec.ingress.symbols:
        # The record itself is DYNAMIC: the range program IS the input.
        return spec
    bound = respecialize(spec.program, spec.ingress)
    derived = graph_hash(bound, spec.ingress, passes=spec.passes)
    if derived != spec.graph:
        raise BindError(
            f"binding the symbolic parent to this record's shapes derives "
            f"{derived}, not the requested graph {spec.graph!r}. Either the "
            f"banked parent does not cover this record, or the record was hashed "
            f"with transform passes this seam cannot restate -- refusing rather "
            f"than minting under the wrong identity."
        )
    return replace(spec, program=bound)


# ---------------------------------------------------------------------------
# Range narrowing (tcg#78)
# ---------------------------------------------------------------------------


def declared_ranges(program: Any) -> dict[str, tuple[int | None, int | None]]:
    """The ranges the EXPORT froze into the program."""

    out: dict[str, tuple[int | None, int | None]] = {}
    for symbol, interval in (getattr(program, "range_constraints", {}) or {}).items():
        out[str(symbol)] = (
            _as_int(getattr(interval, "lower", None)),
            _as_int(getattr(interval, "upper", None)),
        )
    return out


def live_ranges(program: Any) -> dict[str, tuple[int | None, int | None]]:
    """What the ShapeEnv believes NOW -- after whatever the compiler concluded.

    A symbol in `replacements` reads as the SINGLETON `(v, v)`: a replacement is
    the strongest narrowing there is and must not read as "unchanged".
    """

    environment = _shape_env(program)
    if environment is None:
        return {}
    out: dict[str, tuple[int | None, int | None]] = {}
    for symbol, interval in (getattr(environment, "var_to_range", {}) or {}).items():
        out[str(symbol)] = (
            _as_int(getattr(interval, "lower", None)),
            _as_int(getattr(interval, "upper", None)),
        )
    for symbol, value in (getattr(environment, "replacements", {}) or {}).items():
        fixed = _as_int(value)
        if fixed is not None:
            out[str(symbol)] = (fixed, fixed)
    return out


def narrowed_symbols(
    declared: Mapping[str, tuple[int | None, int | None]],
    live: Mapping[str, tuple[int | None, int | None]],
) -> dict[str, tuple[tuple[int | None, int | None], tuple[int | None, int | None]]]:
    """Only NARROWING is reported: a widened range still serves every declared shape."""

    moved = {}
    for name, (low, high) in declared.items():
        span = live.get(name)
        if span is None:
            continue
        live_low, live_high = span
        if (live_low is not None and low is not None and live_low > low) or (
            live_high is not None and high is not None and live_high < high
        ):
            moved[name] = ((low, high), (live_low, live_high))
    return moved


def format_narrowing(moved: Mapping[str, Any]) -> str:
    return "; ".join(
        f"{name} declared [{d[0]}, {d[1]}] but the graph's guards allow only "
        f"[{live[0]}, {live[1]}]"
        for name, (d, live) in sorted(moved.items())
    )


def assert_ranges_hold(program: Any, graph: str) -> None:
    """Refuse an artifact that compiled NARROWER than it declares.

    Asked directly rather than hoped for: the old guard compared re-declared
    digests, which caught the singleton case only because a replaced symbol
    renders differently -- a narrowing that leaves two or more values moves no
    digest at all and shipped green.
    """

    moved = narrowed_symbols(declared_ranges(program), live_ranges(program))
    if not moved:
        return
    raise RangeNarrowed(
        f"graph {graph} compiled to a NARROWER range than it declares: "
        f"{format_narrowing(moved)}. The compiled artifact cannot serve every "
        f"shape the declaration admits, and the dispatcher guards on the "
        f"declaration -- re-derive with a range the graph's own guards support."
    )


def assert_device_uniform(program: Any, graph: str) -> None:
    """Device uniformity must be TOTAL, and it is TWO questions (tcg#89).

    A lifted constant left behind on another device is the failure that exports
    cleanly and only AOTI rejects. The old guard asked it by comparing a device
    STRING against a device TYPE, so it passed only where it could not fire --
    green on a cardless runner (which stamps `cuda`) and red on every machine
    with a card (which stamps `cuda:0`).

    Asked correctly it is two facts, and both are worth having separately: the
    TYPE must be one, because a graph is traced onto one device and cannot be
    re-homed afterwards; and the INDEX must be one, because a constant on
    `cuda:1` in a single-device trace is a real defect and nothing else looks
    for it.
    """

    spellings = placement(program)
    if not spellings:
        return
    types = {value.split(":", 1)[0] for value in spellings}
    if len(types) > 1:
        raise MintError(
            f"graph {graph} straddles device types {sorted(types)!r}: a graph is "
            f"traced onto ONE device and cannot be re-homed afterwards. "
            f"Virtualize the whole module onto one trace device."
        )
    indices = {value.split(":", 1)[1] for value in spellings if ":" in value}
    if len(indices) > 1:
        raise MintError(
            f"graph {graph} places constants on {sorted(spellings)!r}: one device "
            f"type but several INDICES. A single-device trace that leaves a "
            f"constant on another card produces an artifact AOTI rejects at load."
        )


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _shape_env(program: Any) -> Any:
    for node in program.graph_module.graph.nodes:
        for key in ("val", "example_value"):
            value = node.meta.get(key)
            for dimension in getattr(value, "shape", ()) or ():
                environment = getattr(getattr(dimension, "node", None), "shape_env", None)
                if environment is not None:
                    return environment
    return None


# ---------------------------------------------------------------------------
# The compile
# ---------------------------------------------------------------------------


def compile_inputs(program: Any) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Rebuild the call AOTInductor compiles against, FROM THE PROGRAM.

    Never from `example_inputs`: a hollow trace's example inputs are fake
    tensors, torch pickles them as `_reconstruct_fake_tensor`, and torch's own
    loader refuses that under `weights_only=True` -- a blob that kept them could
    not be reloaded at all, on any device.

    Everything needed is already in the program by construction:
    `graph_signature.user_inputs` is the flat call in order (a non-str entry IS
    a specialized constant the trace baked, replayed verbatim because its value
    is part of the graph's identity); each placeholder's `meta['val']` is its
    exact dtype, shape and DEVICE; `call_spec.in_spec` restores the nesting.
    """

    from torch.utils import _pytree as pytree

    graph = getattr(getattr(program, "graph_module", None), "graph", None)
    signature = getattr(program, "graph_signature", None)
    in_spec = getattr(getattr(program, "call_spec", None), "in_spec", None)
    if graph is None or signature is None or in_spec is None:
        raise MintError(
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
            leaves.append(entry)
            continue
        if entry not in values:
            raise MintError(
                f"exported program names user input {entry!r} with no matching "
                f"placeholder; its graph and signature disagree"
            )
        leaves.append(values[entry])
    try:
        args, kwargs = pytree.tree_unflatten(leaves, in_spec)
    except Exception as exc:
        raise MintError(
            f"cannot rebuild the exported program's call from its own signature "
            f"({len(leaves)} flat input(s)) and in_spec: {type(exc).__name__}: {exc}"
        ) from exc
    return tuple(args), dict(kwargs or {})


@contextmanager
def _export_context(program: Any) -> Iterator[None]:
    """Restore the FakeTensor tracing context the exported program carries.

    Exactly one mode and exactly one ShapeEnv, derived from node metadata. Two
    of either means the program was assembled from separate traces, and
    compiling it would silently pick one.
    """

    from torch._guards import TracingContext, tracing
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.utils import _pytree as pytree

    modes: dict[int, Any] = {}
    environments: dict[int, Any] = {}
    for node in program.graph_module.graph.nodes:
        for key in ("val", "example_value"):
            if key not in node.meta:
                continue
            for value in pytree.tree_leaves(node.meta[key]):
                mode = getattr(value, "fake_mode", None)
                if mode is not None:
                    if not isinstance(mode, FakeTensorMode):
                        raise MintError("graph metadata carries a non-FakeTensor mode")
                    modes[id(mode)] = mode
                    if getattr(mode, "shape_env", None) is not None:
                        environments[id(mode.shape_env)] = mode.shape_env
                for holder in (value, *(getattr(value, "shape", ()) or ())):
                    environment = getattr(getattr(holder, "node", None), "shape_env", None)
                    if environment is not None:
                        environments[id(environment)] = environment
    if len(modes) != 1:
        raise MintError(
            f"graph metadata must carry exactly one FakeTensorMode, found {len(modes)}"
        )
    if len(environments) != 1:
        raise MintError(
            f"graph metadata must carry exactly one ShapeEnv, found {len(environments)}"
        )
    mode = next(iter(modes.values()))
    previous = getattr(mode, "shape_env", None)
    cast(Any, mode).shape_env = next(iter(environments.values()))
    try:
        with tracing(TracingContext(mode)):
            yield
    finally:
        cast(Any, mode).shape_env = previous


def compile_package(program: Any, name: str, output: Path, device_type: str) -> Path:
    """AOTI-compile one exported program into a `.pt2` package."""

    impose_host_policy()
    args, kwargs = compile_inputs(program)
    options = dict(POLICY)
    compile_policy(device_type)  # refuses an unclassified or silently-dropped option
    try:
        with _export_context(program):
            files = vars(import_module("torch._inductor"))["aot_compile"](
                program.module(check_guards=False), args, kwargs, options=options
            )
    except Exception as exc:
        raise MintError(f"AOTInductor compile failed: {type(exc).__name__}: {exc}") from exc
    if not isinstance(files, list) or not files:
        raise MintError("AOTInductor did not return a non-empty loose-file list")
    try:
        vars(import_module("torch._inductor.package"))["package_aoti"](
            str(output), {name: list(files)}
        )
    except Exception as exc:
        raise MintError(f"AOTInductor packaging failed: {type(exc).__name__}: {exc}") from exc
    return output


def write_literals(program: Any, destination: Path) -> Path | None:
    """The trace-baked constant values, as safetensors. None when there are none."""

    names = literal_names(program)
    if not names:
        return None
    from safetensors.torch import save_file

    values = getattr(program, "constants", {}) or {}
    save_file(
        {name: values[name].detach().cpu().contiguous() for name in names}, str(destination)
    )
    return destination


def metadata(spec: GraphSpec, *, key: str, sm: str, env: Mapping[str, str],
             device_type: str) -> dict[str, Any]:
    return {
        "kind": "aot-inductor",
        "key": key,
        "graph": spec.graph,
        "name": spec.graph,
        "target": spec.target,
        "sm": sm,
        "env": dict(env),
        "compile_policy": compile_policy(device_type),
        "declared_input_layout": declared_input_layout(),
        "placement": list(placement(spec.program)),
        "ingress": spec.ingress.as_dict(),
        "passes": list(spec.passes),
        # Which constants come from the live module and which the trace baked.
        # STATED by the mint, which knows, rather than re-derived at adopt by
        # regexing the generated C++ wrapper for `constants_info_[i].kind` --
        # 489 lines of parser standing in for one fact the producer had.
        "constants": {
            "literal": list(literal_names(spec.program)),
            "state": [
                name
                for name in constant_names(spec.program)
                if name not in set(literal_names(spec.program))
            ],
        },
    }


@dataclass(frozen=True, slots=True)
class Minted:
    """One minted artifact and the identity it was minted under.

    The key is RETURNED rather than left for the caller to re-derive, and
    ``env`` is what was ACTUALLY keyed: `artifact_key` imposes the host ISA
    facts, so the block here may carry more than the caller passed in.
    """

    key: str
    path: Path
    env: Mapping[str, str]


def mint(
    spec: GraphSpec,
    *,
    sm: str,
    env: Mapping[str, str],
    device_type: str,
    destination: Path,
) -> Minted:
    """Bind, compile, package and stamp ONE graph into one artifact tarball.

    The order is load-bearing: the bind must re-derive the requested identity
    before anything is compiled, and the range check must run on the program
    that was ACTUALLY compiled (the compile installs ShapeEnv replacements, so
    asking earlier asks the wrong program).

    ``env``'s other members are the caller's to choose; the host ISA facts are
    imposed by :func:`~torchcg.identity.artifact_key` itself, so the key this
    returns may carry more than the caller passed in -- which is why it is
    RETURNED rather than left to be re-derived.
    """

    from .identity import artifact_key

    bound = bind_static_spec(spec)
    assert_device_uniform(bound.program, bound.graph)
    key = artifact_key(
        bound.graph,
        sm=sm,
        env=env,
        policy=compile_policy(device_type),
        layout=declared_input_layout(),
    )
    require_morphism(declared_input_layout())
    with tempfile.TemporaryDirectory() as scratch:
        workspace = Path(scratch)
        compile_package(bound.program, bound.graph, workspace / "model.pt2", device_type)
        assert_ranges_hold(bound.program, bound.graph)
        write_literals(bound.program, workspace / "constants.safetensors")
        (workspace / "metadata.json").write_text(
            json.dumps(
                metadata(bound, key=key.value, sm=sm, env=dict(key.env),
                         device_type=device_type),
                sort_keys=True,
                indent=2,
            )
        )
        from .store import pack

        pack(workspace, destination)
    return Minted(key=key.value, path=destination, env=dict(key.env))


__all__ = [
    "ACCEPT_DROPPED",
    "POLICY",
    "GraphSpec",
    "Minted",
    "assert_device_uniform",
    "assert_ranges_hold",
    "bind_static_spec",
    "compile_inputs",
    "compile_package",
    "compile_policy",
    "declared_input_layout",
    "declared_ranges",
    "format_narrowing",
    "impose_host_policy",
    "live_ranges",
    "metadata",
    "mint",
    "narrowed_symbols",
    "respecialize",
    "flatten_weight_subclasses",
    "strip_diagnostics",
    "write_literals",
]
