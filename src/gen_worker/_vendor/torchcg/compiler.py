from __future__ import annotations

import logging
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, cast

from . import _wrapper_split
from .host_isa import HostISAError, _impose_host_policy
from .identity import policy_axis_digest
from .layout import CONTIGUOUS, morphism_for_stride_order, require_morphism

logger = logging.getLogger(__name__)

_COMPILER_OPTIONS: dict[str, object] = {
    # pgw#757 measured 32 -> 4 as effectively free while sharply reducing
    # contention with live serving. A later balanced 16-run 1-vs-4 cold AOTI
    # comparison put both means inside run-to-run noise (4 was 1.0% lower), so
    # there is no measured reason to fork the current worker's value. Four is
    # the one sealed policy; callers cannot select a compiler topology.
    "compile_threads": 4,
    "aot_inductor.package_constants_in_so": False,
    # tcg#80. ONE copy of the weights on the card, and every state-dict
    # constant bindable. Read `GraphLowering.get_attr` (torch 2.13,
    # `_inductor/graph.py:1560-1585`): a lifted constant takes the
    # `add_tensor_constant` path -- a bindable table row -- when EITHER
    # `use_runtime_constant_folding` OR `always_keep_tensor_constants` is set,
    # and is otherwise INLINED INTO THE GENERATED CODE whenever its shape is
    # `()` or 1-D with <= 8 elements. So one of the two must be on, or a
    # checkpoint's values are compiled into the artifact and the weightless
    # CAS contract is a lie (pgw#1097 named that fence; sdxl bakes
    # `conv_out.bias`, 4 floats, with both off).
    #
    # The two are NOT equivalent, and that is the whole of tcg#80.
    # `use_runtime_constant_folding` additionally splits a const-fold subgraph
    # that runs on the FIRST call and materializes a SECOND full constant set
    # by direct `cudaMalloc`, outside the caching allocator
    # (`aoti_runtime/model_base.h:345`, `model_container.h:169-186`). On sdxl
    # UNet-only that is +4782 MiB of first-call transient on top of the eager
    # weights the artifact already points at by raw pointer -- ~10.4 GiB
    # against 7.8 GiB visible on an 8 GiB card, measured. It bought sdxl one
    # folded constant in 2423.
    #
    # `always_keep_tensor_constants` buys the same bindable table with NO
    # second set: both flags take the identical `add_tensor_constant` branch,
    # so the table's composition is unchanged and only the folded rows go
    # away. It is what pgw#1097 shipped first and reverted on a CI red that
    # its own head 3 later root-caused to a fixture authoring violation
    # (a plain-attribute table, refused identically under the flag it shipped
    # instead) -- so the revert's stated reason did not distinguish them.
    "always_keep_tensor_constants": True,
    "aot_inductor.use_runtime_constant_folding": False,
    "aot_inductor.package": True,
}

#: The options whose VALUE decides what the compiler EMITS. These are the
#: compile-policy axis of an artifact key: two artifacts built under different
#: values of any of them are different bytes and must not share an address.
_CODEGEN_OPTIONS = frozenset(
    (
        "aot_inductor.package_constants_in_so",
        "always_keep_tensor_constants",
        "aot_inductor.use_runtime_constant_folding",
        "aot_inductor.package",
        # Not shipped. Classified in advance because they ARE codegen and
        # tcg#82 priced them: an option nobody classified is refused, which is
        # correct, but these three have a precondition (below) and their
        # refusal should be about the CARD, not about the vocabulary.
        "max_autotune",
        "max_autotune_gemm",
        "coordinate_descent_tuning",
    )
)

#: Options that change only HOW the compile is executed, never what it emits.
#: Keying on these would split the artifact pool on a scheduling knob, which
#: is the pgw#1472 disease.
_TOPOLOGY_OPTIONS = frozenset(("compile_threads",))

#: Options torch ITSELF discards under AOTI, so setting one changes no emitted
#: byte. `triton.cudagraphs` is computed as `triton.cudagraphs and not
#: V.aot_compilation` in `get_cpp_wrapper_config` (torch 2.13,
#: `compile_fx.py:2246`) -- CUDA-graph capture is a SERVING lever around
#: `runner.__call__`, never a compile one. Classified rather than left
#: unclassified so that setting it is legal and INERT: keying on it would split
#: the artifact pool on a flag that cannot move the bytes, which is the exact
#: pgw#1472 disease from the other direction (tcg#82, tcg#85).
_INERT_OPTIONS = frozenset(("triton.cudagraphs",))

#: The declared byte layout of the tensors this mint compiles against, and the
#: sole v1 policy -- the same sealed shape `_COMPILER_OPTIONS` has, and the
#: same rule: the axis is READ from here, so moving this moves the key, the
#: metadata and the binder together (tcg#83). It is `torch.contiguous@1`
#: today, which is what every existing tree and every existing artifact is.
#: `research/layout-morphisms/0-DESIGN.md` §1: the compiler declares the layout
#: it wants and the platform delivers it; until the storage half lands
#: ([[tensorfs#155]]/[[pgw#1645]]) the honest declaration is the stored layout.
_DECLARED_INPUT_LAYOUT: str = CONTIGUOUS


class CompileError(RuntimeError):
    """AOTInductor did not produce a packageable code-only graph."""


class DroppedOptimization(CompileError):
    """The target will silently DROP an optimization this policy requests.

    tcg#85, named rather than generic because the remedy is specific: the mint
    is not broken, it is about to buy a re-key and a much longer compile for
    half a lever. Either mint on a target that clears the precondition, or
    sanction the drop in `_ACCEPT_DROPPED_OPTIONS` so the KEY describes what
    actually ran.
    """


def _compiler_options() -> dict[str, object]:
    """Return the sole compile policy accepted by pre-launch v1."""

    return dict(_COMPILER_OPTIONS)


def compile_policy() -> dict[str, object]:
    """The codegen-relevant compile options, READ from the sole policy.

    This is the source of truth every declared axis and every key derives
    from, so that a build cannot state a compile property it did not compile
    under (tcg#80: `package_constants_in_so` and `constant_folding_fenced`
    were written as literals by the same function that validated them, and
    two differently-configured mints therefore produced the SAME key and
    collided in the CAS).

    Every option must be classified. An unclassified one is a refusal rather
    than a silent omission: a new codegen option that nobody put in a set
    would otherwise leave the key blind to it, which is the defect this
    function exists to remove.
    """

    options = _compiler_options()
    unclassified = sorted(
        set(options) - _CODEGEN_OPTIONS - _TOPOLOGY_OPTIONS - _INERT_OPTIONS
    )
    if unclassified:
        raise CompileError(
            f"compile option(s) {unclassified!r} are neither codegen, topology "
            f"nor inert. Classify them: a codegen option belongs in the artifact "
            f"key, a topology option must not be in it, and an inert one is "
            f"discarded by torch itself (tcg#80, tcg#82)."
        )
    return {name: value for name, value in sorted(options.items()) if name in _CODEGEN_OPTIONS}


def device_type_of(sm: str) -> str:
    """The DEVICE CLASS one recorded ``sm`` names -- ``cpu`` or ``cuda``.

    ONE derivation with two callers: `RuntimeCompatibility.device_type`
    answers it from a live target, this answers it from a recorded string, and
    both must say the same thing about the same ``sm``. Torch-free, because
    `recipe` folds a key without importing torch.
    """

    return "cpu" if str(sm).strip().lower().startswith("cpu") else "cuda"


@dataclass(frozen=True, slots=True)
class OptionShortfall:
    """What a target will ACTUALLY do with one requested option.

    ``executed_value`` is the canonical value the option takes in the executed
    policy when the shortfall is sanctioned. It is NOT the option's absence:
    a partially-honoured option emits partially-different code, so it is a
    THIRD state and must key as one. `max_autotune` on a small card is the
    worked example -- torch drops the GEMM templates (`is_big_gpu`) and still
    benchmarks the pointwise kernels, so the bytes match neither the
    full-lever mint nor the no-lever mint, and a key that said "absent" would
    collide in the CAS with a no-lever artifact.
    """

    executed_value: str
    reason: str


def _gemm_template_precondition(device_type: str) -> OptionShortfall | None:
    """torch's OWN answer to "will Triton GEMM templates run here?".

    tcg#85, measured: `max_autotune: True` on this box's RTX 4070 Laptop moved
    the key, cost 622.5 s against an 85.2 s baseline, and torch logged
    `Not enough SMs to use max_autotune_gemm mode` -- 36 SMs against its
    internal `min_sms = 68`. The GEMM half of the lever, the only half that
    reaches the 88.6% of device time tcg#82 measured in extern cuBLAS, never
    ran. The policy said something the compile did not do.

    The verdict is READ from torch (`torch._inductor.utils.is_big_gpu`, the
    same function `_use_template_for_gpu` consults), never a copy of its
    threshold: a copied 68 drifts silently the first time torch moves it. The
    reason quotes the SM count torch itself measured.

    `is_big_gpu` is `functools.cache`d on the device index, and a cached
    verdict is a guard that cannot go red -- one mocked probe would poison
    every later question in the process. The uncached `__wrapped__` is called
    when torch still exposes it.
    """

    if device_type != "cuda":
        # CPU keeps Triton templates without an SM gate
        # (`utils.use_triton_template`), so there is nothing to drop.
        return None
    try:
        import torch
        from torch._inductor.runtime.hints import DeviceProperties
        from torch._inductor.utils import is_big_gpu
    except ImportError as exc:  # pragma: no cover - exercised with torch extra
        raise CompileError(
            "cannot read torch's max-autotune device gate; PyTorch with "
            f"AOTInductor is required to state a compile policy ({exc})"
        ) from exc
    index = torch.cuda.current_device() if torch.cuda.is_available() else 0
    device = torch.device("cuda", index)
    gate = cast(Any, getattr(is_big_gpu, "__wrapped__", is_big_gpu))
    if gate(device):
        return None
    properties = cast(Any, DeviceProperties).create(device)
    return OptionShortfall(
        "no-gemm-templates",
        f"torch._inductor.utils.is_big_gpu refuses this device: "
        f"{properties.multi_processor_count} SMs at compute capability "
        f"{properties.cc}. Triton GEMM templates -- the half of the lever that "
        f"reaches the extern cuBLAS calls -- will not be generated, while the "
        f"pointwise half still is",
    )


#: Every option whose effect depends on the TARGET, and the torch-read question
#: that decides it. An option absent here has no hardware precondition.
_OPTION_PRECONDITIONS: dict[str, Callable[[str], OptionShortfall | None]] = {
    "max_autotune": _gemm_template_precondition,
    "max_autotune_gemm": _gemm_template_precondition,
}

#: Options whose silent drop is SANCTIONED: the mint proceeds without them and
#: the key describes what ran. Empty, and it is meant to be: the escape hatch
#: exists so that wanting the pointwise/conv half of `max_autotune` on a small
#: card is a decision somebody wrote down, not a surprise (tcg#85).
_ACCEPT_DROPPED_OPTIONS: frozenset[str] = frozenset()


def dropped_options(device_type: str) -> dict[str, OptionShortfall]:
    """The requested options this target will not fully honour, and why.

    Only truthy options are asked about: `max_autotune: False` requests
    nothing, so there is nothing for a card to decline.
    """

    dropped: dict[str, OptionShortfall] = {}
    for name, value in compile_policy().items():
        precondition = _OPTION_PRECONDITIONS.get(name)
        if precondition is None or not value:
            continue
        shortfall = precondition(str(device_type))
        if shortfall is not None:
            dropped[name] = shortfall
    return dropped


def executed_compile_policy(device_type: str) -> dict[str, object]:
    """The codegen options that will ACTUALLY run on one target.

    tcg#85's core clause: **the key must be a function of the optimizations
    that EXECUTED, not the ones requested.** Otherwise two cards running the
    same emitted code disagree on its address, and one card running two
    different codes agrees on one.

    An option the target will drop is a refusal naming the option and torch's
    own reason -- unless the drop is sanctioned, in which case the option's
    value is rewritten HERE, before the digest, to the canonical name of what
    actually ran. It is NOT deleted: a partially-honoured option emits
    partially-different code, so its artifact must key as a third thing rather
    than colliding with the no-lever artifact.
    """

    policy = compile_policy()
    dropped = dropped_options(device_type)
    unsanctioned = sorted(set(dropped) - _ACCEPT_DROPPED_OPTIONS)
    if unsanctioned:
        detail = "; ".join(f"{name}: {dropped[name].reason}" for name in unsanctioned)
        raise DroppedOptimization(
            f"compile policy requests {unsanctioned!r}, which this target will "
            f"SILENTLY DROP -- {detail}. The artifact would key differently and "
            f"compile identically-minus-the-lever, so the fleet re-mints for "
            f"nothing. Mint on a target that clears the precondition, or add "
            f"the option to `_ACCEPT_DROPPED_OPTIONS` so the key states what "
            f"actually ran (tcg#85)."
        )
    executed = dict(policy)
    for name, shortfall in dropped.items():
        logger.warning(
            "compile option %r is SANCTIONED-PARTIAL on this target; the key "
            "states %r instead of what was requested: %s",
            name,
            shortfall.executed_value,
            shortfall.reason,
        )
        executed[name] = shortfall.executed_value
    return executed


def compile_policy_digest(device_type: str) -> str:
    """The compile-policy axis of a key this process would mint on one target."""

    return policy_axis_digest(executed_compile_policy(device_type))


def declared_input_layout() -> str:
    """The byte layout this mint compiles its inputs and constants against.

    READ from `_DECLARED_INPUT_LAYOUT`, the one place it is stated, and
    validated against the ratified catalog at every call -- an unclassified
    layout is a refusal, never a coercion to row-major (tcg#83). This is the
    same move tcg#80 made for the folding flags: the axis derives from the
    build, so a build cannot state a layout it did not compile under.
    """

    return require_morphism(_DECLARED_INPUT_LAYOUT).handle


def _option(policy: Mapping[str, object], name: str) -> bool:
    if name not in policy:
        raise CompileError(f"compile policy states no {name!r}")
    return bool(policy[name])


def policy_packages_constants(policy: Mapping[str, object]) -> bool:
    """Whether one STATED policy makes AOTI carry its own constant blob."""

    return _option(policy, "aot_inductor.package_constants_in_so")


def policy_fences_folding(policy: Mapping[str, object]) -> bool:
    """Whether one STATED policy leaves every state-dict constant bindable."""

    return _option(policy, "aot_inductor.use_runtime_constant_folding") or _option(
        policy, "always_keep_tensor_constants"
    )


def package_constants_in_so() -> bool:
    """Whether this policy makes AOTI carry its OWN copy of the constants.

    False is the weightless contract: the artifact is code, the constants are
    raw pointers into the live module's weights (`runner.load_constants(...,
    user_managed=True)`). True would make AOTI `cudaMalloc` a full constant
    blob of its own WHILE the eager module's weights stay resident and
    referenced -- the same double residency runtime folding causes, bought
    back under another name (tcg#80).
    """

    return policy_packages_constants(compile_policy())


def constant_folding_fenced() -> bool:
    """Whether this policy leaves EVERY state-dict constant BINDABLE.

    A restatement of torch's own gate (`GraphLowering.get_attr`): with both
    flags off, inductor inlines a lifted constant's VALUES into the generated
    kernel when its shape is `()` or 1-D with <= 8 elements, and the constant
    then appears in no table anyone could rebind. The artifact becomes
    specific to the checkpoint it was minted from, silently.

    Derived, never asserted: the property is whatever the policy makes true,
    and `artifact.validate_metadata` refuses an artifact that cannot state it
    (pgw#1097's fence, now readable off the build that produced it).
    """

    return policy_fences_folding(compile_policy())


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


@dataclass(frozen=True, slots=True)
class LayoutWish:
    """One constant whose layout inductor asked to CHANGE during this compile.

    ``name`` is inductor's own buffer name for the constant (``conv_weight``);
    the mint translates it to the artifact fqn through the package's own
    constant table, which is the second, independent witness.

    ``morphism`` is the ratified catalog handle when the wanted layout is one,
    and ``""`` when it is not -- an out-of-catalog wish is a CANDIDATE carrying
    only the stride order inductor asked for. Nothing here invents a name.
    """

    name: str
    stride_order: tuple[int, ...]
    morphism: str

    @property
    def ratified(self) -> bool:
        return bool(self.morphism)


@contextmanager
def _observing_layout_wishes(wishes: dict[str, LayoutWish]) -> Iterator[None]:
    """Record what inductor asks of each CONSTANT's layout, from inductor.

    ``ExternKernel.require_strides`` is the single choke point where inductor
    states the layout it wants for an input (`require_channels_last`,
    `require_stride_order` and `require_contiguous` all route through it), and
    it inserts a copy only when the strides do not already match. So a wish is
    exactly: a graph CONSTANT whose require_strides call returned a DIFFERENT
    buffer. Measured at micro scale on CPU, no card: a contiguous conv weight
    is copied under order [3, 0, 2, 1]; the same weight delivered channels_last
    is returned unchanged, and the per-call permute kernel is never generated.

    Read, never steered: the original is always called, its result is always
    returned, and any failure inside the observer leaves the compile alone.
    """

    try:
        module = import_module("torch._inductor.ir")
        virtualized = import_module("torch._inductor.virtualized")
        extern_kernel = vars(module)["ExternKernel"]
        graph_state = vars(virtualized)["V"]
    except (ImportError, KeyError):  # pragma: no cover - exercised with torch extra
        logger.debug("layout wishes: torch._inductor is unavailable")
        yield
        return

    original = extern_kernel.require_strides.__func__
    #: Constants two kernels want in DIFFERENT layouts. There is no single
    #: ideal layout for them, so they carry no wish -- an unsatisfiable wish is
    #: worse than none, because the whole point of the wishlist is that a
    #: re-mint against it deletes copies rather than moving them.
    conflicted: set[str] = set()

    def _record(node: object, requested: object, result: object) -> None:
        name = getattr(node, "get_name", lambda: None)()
        if not isinstance(name, str) or name not in (
            getattr(graph_state.graph, "constants", None) or {}
        ):
            return
        if getattr(result, "get_name", lambda: None)() == name:
            # No copy: the constant is ALREADY in the layout inductor wants.
            # That is the win this axis exists to make expressible, and it says
            # nothing about any OTHER kernel that did copy it, so it never
            # erases a wish already recorded.
            return
        order = tuple(int(value) for value in cast(Any, requested))
        known = wishes.get(name)
        if known is not None and known.stride_order != order:
            conflicted.add(name)
            return
        morphism = morphism_for_stride_order(order)
        wishes[name] = LayoutWish(name, order, morphism.handle if morphism else "")

    def _patched(
        cls: Any,
        x: Any,
        order: Any = None,
        exact_strides: Any = None,
        allow_padding: bool = False,
    ) -> Any:
        result = original(
            cls, x, order=order, exact_strides=exact_strides, allow_padding=allow_padding
        )
        try:
            requested = order
            if requested is None and exact_strides is not None:
                from torch._inductor.ir import get_stride_order

                requested = get_stride_order(exact_strides)
            if requested is not None:
                _record(x, requested, result)
        except Exception:  # pragma: no cover - an observer never breaks a mint
            logger.warning("layout wishes: could not read one request", exc_info=True)
        return result

    extern_kernel.require_strides = classmethod(_patched)
    try:
        yield
    finally:
        extern_kernel.require_strides = classmethod(original)
        for name in conflicted:
            wishes.pop(name, None)
            logger.info(
                "layout wishes: constant %r is wanted in two layouts; no single "
                "ideal layout exists for it, so it carries no wish",
                name,
            )


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


@dataclass(frozen=True, slots=True)
class CompiledFiles:
    """The loose files one compile produced, and what it wished for."""

    files: tuple[object, ...]
    wishes: tuple[LayoutWish, ...]


def _compile_exported_program(program: object) -> CompiledFiles:
    """Compile one ExportedProgram to the loose files used by `package_aoti`.

    FakeTensor and ShapeEnv state is derived only from ``program``. Compile
    settings are fixed because every setting that can affect generated code
    must have exactly one v1 identity.

    The layout wishes ride back out with the files: they are an OBSERVATION of
    this compile (tcg#83), read from inductor's own `require_strides`, and the
    only moment they exist is while it runs.
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

    wishes: dict[str, LayoutWish] = {}
    try:
        with _compiling_under_export_context(program), _observing_layout_wishes(wishes):
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
    return CompiledFiles(
        tuple(result),
        tuple(wishes[name] for name in sorted(wishes)),
    )


def layout_violations(program: object) -> list[str]:
    """The program tensors that are NOT in the layout this mint declares.

    tcg#83's honesty check, and the layout twin of tcg#85's clause: an artifact
    may not declare a layout it did not compile against. torchcg does not
    transform the bytes to make the declaration true -- delivering them is the
    platform's job (`research/layout-morphisms/0-DESIGN.md` §3/§4) -- so a
    disagreement is named here, before a compile is paid for.
    """

    morphism = require_morphism(_DECLARED_INPUT_LAYOUT)
    tensors: dict[str, Any] = {}
    for source in ("state_dict", "constants"):
        for name, value in (getattr(program, source, None) or {}).items():
            if hasattr(value, "dim") and hasattr(value, "stride"):
                tensors[str(name)] = value
    violations: list[str] = []
    for name in sorted(tensors):
        value = tensors[name]
        try:
            satisfied = morphism.matches(value)
        except (RuntimeError, TypeError, ValueError) as exc:
            raise CompileError(
                f"cannot read the layout of constant {name!r}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        if not satisfied:
            violations.append(f"{name} {morphism.describe(value)}")
    return violations


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
