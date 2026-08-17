"""Worker graph-class declaration, tracing, and mint supervision.

``torchcg`` owns compiled-graph identity, compiler policy,
package admission, artifact storage, and execution.  This module retains only
the worker facts that TCG cannot know: endpoint declarations, pipeline
composition, per-class tracing, child-pool orchestration, and mint telemetry.
Every compiled result is independently reusable and publishable; no worker
multi-graph package, local mint CLI, or alternate compiled-artifact format
exists.
"""

from __future__ import annotations
from .hostfacts import cuda_ready

import hashlib
import json
import logging
import os
import time

import msgspec
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any, Callable, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple)

from gen_worker._vendor.torchcg import (
    GRAPH_CLASS_BLOCK,
    CallIngress,
    IngressError,
    build_call_ingress,
    exported_input_name,
)
from gen_worker._vendor.torchcg.identity import toolchain_axis_digest

from . import activity as activity_mod
from . import aot_compile_pool, aot_serve, boot_phases
from .aot_inputs import (  # re-exported: the declaration layer's vocabulary
    ADAPTER_FORK,
    DynamicDim,
    ExportSpec,
    MintRefused,
)
from .compile_cache import _resolve_target
from dataclasses import replace
import inspect
from .models.memory import is_cuda_oom
from . import aot_inputs
from . import aot_declaration as _decl
from . import meta_instantiation
from .models import lora_lifted
from .models import structure_only

logger = logging.getLogger(__name__)

class MintResourceExhausted(RuntimeError):
    """This mint ran out of MEMORY. It is not a refusal, and the difference
    is the whole point (pgw#848).

    Deliberately NOT a subclass of :class:`MintRefused`. ``mint_child`` maps
    every ``MintRefused`` to ``EXIT_REFUSED``, which ``mint_process``
    documents as "typed, deterministic — terminal", so an OOM-killed entry
    child inherited a never-retry verdict: the one failure class a narrower
    pool would have fixed was the one class that could never try a narrower
    pool. Raising a sibling type instead lets the child's own
    ``_is_resource_error`` see it (via ``mint_resource_shortfall``) and exit
    ``EXIT_RESOURCE``, which the parent re-budgets and retries.

    ``peak_rss_bytes`` is the dead entry's MEASURED high-water, sampled by
    the parent while it lived — a child the OOM killer takes writes no
    report, so this is the only measurement of it that will ever exist.
    """

    #: Duck type read by ``mint_child._is_resource_error`` so that module
    #: does not have to import this one (it deliberately imports as little
    #: of the arming brain as it can).
    mint_resource_shortfall = True

    def __init__(
        self, detail: str, *, entry: str = "", basis: str = "",
        peak_rss_bytes: int = 0,
    ) -> None:
        super().__init__(detail)
        self.entry = entry
        self.basis = basis
        self.peak_rss_bytes = int(peak_rss_bytes)


def raise_if_device_oom(exc: BaseException, where: str) -> None:
    """Re-raise a CUDA OOM as the RESOURCE type instead of letting a broad
    ``except Exception`` launder it into a deterministic refusal (pgw#848).

    The mint path is full of broad catches that exist to name a failure. They
    are right about naming and wrong about classification: an out-of-memory is
    the one failure whose whole remedy is "try again with more room", and
    every one of these sites was converting it to the verdict that guarantees
    it never will be.
    """

    if not is_cuda_oom(exc):
        return
    peak = 0
    try:
        import torch

        if cuda_ready():
            peak = int(torch.cuda.max_memory_allocated())
    except Exception:  # noqa: BLE001 — a probe never changes an outcome
        peak = 0
    raise MintResourceExhausted(
        f"{where}: OUT OF DEVICE MEMORY ({type(exc).__name__}: {exc}). "
        f"This process peaked at {peak / (1 << 30):.2f} GiB. It is a resource "
        f"shortfall to be retried with more room, NOT a deterministic "
        f"refusal — and it is the ONLY VRAM signal this mint produces "
        f"(§4.33: the attempt is the measurement, nothing predicts it)",
        peak_rss_bytes=0) from exc


# ---------------------------------------------------------------------------
# The declared export contract
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def exported_input_names(
    name: str, containers: Optional[Mapping[str, int]] = None,
) -> Tuple[str, ...]:
    """The exported program's user-input name(s) a DECLARED input name means.

    THE ONE FLATTENING RULE (pgw#993). ``torch.export`` flattens a container
    argument into one positional user input per element, suffixed ``_0``,
    ``_1``, … — measured for every arity, ``N=1`` included, so a one-element
    container is ``x_0`` and never ``x``. A declared name therefore survives
    into the exported program only when it is NOT a container.

    Every consumer that maps declared names onto exported ones goes through
    here: :func:`dynamic_shapes_spec` (which mirrors the structure),
    :func:`declared_range_gaps` and :func:`lifted_input_gaps` (which resolve
    names against the program). Two independent spellings of one mapping is
    the defect class this replaces — z-image's ``Dim(carried_by=(("x", 2),))``
    was unsatisfiable by construction, refusing every mint with "declared
    dynamic dim names input 'x', which is not a user input of the exported
    program (inputs: [… 'x_0', 'x_1'])" while the export it gated succeeded.

    The arity comes from the SAME class row the example feed was built from
    (``aot_declaration.container_arities``), never re-guessed, and the spelling
    comes from TCG's ``exported_input_name`` — the ONE naming rule the ingress
    contract and the serve-side bind also read (pgw#994). A declared container
    is the leaf path ``(0,), (1,), …`` of its parameter, so this function is a
    special case of that rule rather than a second copy of it.
    """
    arity = (containers or {}).get(name)
    if arity is None:
        return (exported_input_name(name),)
    return tuple(exported_input_name(name, (index,))
                 for index in range(int(arity)))


def dynamic_shapes_spec(
    dims: Sequence[DynamicDim], input_names: Sequence[str],
    containers: Optional[Mapping[str, int]] = None,
) -> Dict[str, Any]:
    """``torch.export`` ``dynamic_shapes`` for the declared dims.

    Keyed by parameter NAME (the dict form), which is the only readable way to
    mirror a combined args+kwargs structure. Inputs with no declared dynamic
    axis map to ``None`` — fully static — so an input the spec forgot cannot
    accidentally acquire a symbolic dim.

    Rows carrying the same :attr:`DynamicDim.dim` share ONE torch symbol
    (pgw#812 D1). Minting a symbol per (input, axis) instead makes a declared
    multi-carrier dim into several INDEPENDENT symbols, and strict export
    refuses the declaration outright::

        Constraints violated (img_ids_1)! The values of
        img_ids_1 = L['img_ids'].size()[1] and
        hidden_states_1 = L['hidden_states'].size()[1] must always be equal.

    So the most careful declaration in the fleet — flux2's, which binds
    ``T_img`` to both carriers so the edit lane cannot silently pin
    ``img_ids`` — was the one that could not export at all. Rows with no
    declared name keep today's per-(input, axis) symbol, which is required by
    the hand-registered builder path (``aot_inputs.latent_hw_dims``: latent H
    and W are two independent axes of ONE input and must NOT share a symbol).
    """
    from torch.export import Dim

    by_input: Dict[str, Dict[int, Any]] = {}
    shared: Dict[Tuple[str, int, int, int], Any] = {}
    for d in dims:
        if d.min < 1 or d.max < d.min:
            raise MintRefused(
                f"declared dim {d.input_name}[{d.axis}] has an empty range "
                f"[{d.min}, {d.max}]")
        # Carriers only share when the declaration says they are the SAME dim
        # AND the bounds agree; a name reused at different bounds is two
        # symbols, not one silently-widened one.
        key = (d.dim, int(d.multiple_of), int(d.min), int(d.max))
        symbol = shared.get(key) if d.dim else None
        if symbol is None:
            base_name = d.dim or f"{d.input_name}_{d.axis}"
            if d.multiple_of > 1:
                if d.min % d.multiple_of or d.max % d.multiple_of:
                    raise MintRefused(
                        f"declared dim {d.input_name}[{d.axis}] bounds "
                        f"[{d.min}, {d.max}] are not multiples of "
                        f"{d.multiple_of}; export cannot express that guard")
                symbol = d.multiple_of * Dim(
                    f"{base_name}_u",
                    min=d.min // d.multiple_of, max=d.max // d.multiple_of)
            else:
                symbol = Dim(base_name, min=d.min, max=d.max)
            if d.dim:
                shared[key] = symbol
        by_input.setdefault(d.input_name, {})[d.axis] = symbol
    unknown = sorted(set(by_input) - set(input_names))
    if unknown:
        raise MintRefused(
            f"declared dynamic dims name inputs the target does not take: "
            f"{unknown!r} (target inputs: {list(input_names)!r})")
    # pgw#853: `dynamic_shapes` must MIRROR the example feed's container
    # structure or torch refuses by name. z-image measured the sentence:
    #
    #     Detected mismatch between the structure of `inputs` and
    #     `dynamic_shapes`: `inputs['x']` is a <class 'list'>, but
    #     `dynamic_shapes['x']` is a <class 'dict'>
    #
    # and the nested form torch wants — {'x': [{2: Dim, 3: Dim}]} — exports
    # fine, so this was an SDK gap, not a torch limitation. The arity comes
    # from the SAME class row the feed was built from
    # (`aot_declaration.container_arities`), never re-guessed here.
    out: Dict[str, Any] = {}
    for name in input_names:
        spec = by_input.get(name)
        # One entry per EXPORTED element; every element of a declared
        # container shares the container's declared axes, which is what makes
        # the elements one graph class rather than N. The element count comes
        # from `exported_input_names` — the same rule the declared-range and
        # lifted-input gates resolve names with (pgw#993).
        exported = exported_input_names(name, containers)
        out[name] = spec if exported == (name,) else [spec for _ in exported]
    return out


def export_program(
    module: Any,
    example_args: Tuple[Any, ...],
    example_kwargs: Mapping[str, Any],
    *,
    dynamic_shapes: Optional[Mapping[str, Any]] = None,
    strict: bool = True,
) -> Any:
    """``torch.export.export`` the target, refusing by name on failure."""
    import torch

    try:
        return torch.export.export(
            module, tuple(example_args), dict(example_kwargs),
            dynamic_shapes=dict(dynamic_shapes) if dynamic_shapes else None,
            strict=strict,
        )
    except Exception as exc:
        # pgw#848: a CUDA OOM here is NOT a refusal. This broad catch turned
        # "the child hit its own memory cap on entry 1 of 36" into
        # `MintRefused` -> `EXIT_REFUSED` -> never retried — so the mint that
        # a bigger cap would fix was the one mint that could never be given
        # one. Same defect as the entry pool's (item 4), at the other end.
        raise_if_device_oom(
            exc,
            f"torch.export(strict={strict}) for {type(module).__name__}")
        raise MintRefused(
            f"torch.export(strict={strict}) failed for "
            f"{type(module).__name__}: {type(exc).__name__}: {exc}"
        ) from exc


#: pgw#847: measured ONCE per mint process, and read by nobody.
#:
#: `torch.export.export` runs once per declared class row, SERIALLY, in this
#: parent (`_export_entry`) — sdxl is 36 entries at a banked `export_s` of
#: 37.8 s, so that loop is ~22 minutes of wall the pgw#809 pool never covered.
#: An exported graph's `graph_module.code` is byte-identical across shape rows
#: (the row lives in node metadata), and ONE export plus a per-row
#: `FakeTensorProp` reproduces the compiled artifact byte for byte — wrapper,
#: kernel and the linked `.so` — proven with `torch.export.export` monkeypatched
#: to raise. The saving is therefore `export_s - prop_s`, and `prop_s` on a real
#: family's graph has never been measured; off-pod probes bound it only to
#: 0.25-0.97x. This is that one number, so the next real mint settles it.
_PROP_PROBE_DONE = False


def _probe_prop_s(program: Any) -> Optional[float]:
    """Time `FakeTensorProp` over this program's own graph, once per process.

    Probe only: it mutates a FRESH `program.module()` (never the program), no
    decision reads it, and any failure records nothing rather than touching a
    mint. `.module()` is excluded from the timing because TCG's compiler path
    pays it too, so the comparison stays like-for-like.
    """
    global _PROP_PROBE_DONE
    if _PROP_PROBE_DONE:
        return None
    _PROP_PROBE_DONE = True
    try:
        import torch
        from torch._subclasses.fake_tensor import FakeTensorMode
        from torch.fx.passes.fake_tensor_prop import FakeTensorProp

        args, _kwargs = program.example_inputs
        gm = program.module(check_guards=False)
        t0 = time.monotonic()
        mode = FakeTensorMode(allow_non_fake_inputs=True)
        with mode:
            fake = tuple(
                mode.from_tensor(a) if isinstance(a, torch.Tensor) else a
                for a in args)
            FakeTensorProp(gm, mode=mode).propagate(*fake)
        return round(time.monotonic() - t0, 2)
    except Exception as exc:  # noqa: BLE001
        logger.info(
            "aot-mint: pgw#847 prop probe skipped: %s: %s",
            type(exc).__name__, exc)
        return None


def _placeholder_shapes(program: Any) -> Dict[str, Tuple[Any, ...]]:
    """``{user input name: shape tuple}`` from the exported placeholders."""
    signature = getattr(program, "graph_signature", None)
    user_inputs = [str(n) for n in getattr(signature, "user_inputs", ()) or ()]
    by_node: Dict[str, Any] = {}
    graph = getattr(getattr(program, "graph_module", None), "graph", None)
    for node in getattr(graph, "nodes", ()) or ():
        if getattr(node, "op", "") == "placeholder":
            by_node[str(node.name)] = node.meta.get("val")
    out: Dict[str, Tuple[Any, ...]] = {}
    for name in user_inputs:
        val = by_node.get(name)
        if val is not None:
            out[name] = tuple(getattr(val, "shape", ()) or ())
    return out


def _free_symbols(dim: Any) -> Tuple[Any, ...]:
    node = getattr(dim, "node", None)
    expr = getattr(node, "expr", None)
    return tuple(getattr(expr, "free_symbols", ()) or ()) if expr is not None else ()


def _shape_env(program: Any) -> Any:
    for shape in _placeholder_shapes(program).values():
        for dim in shape:
            env = getattr(getattr(dim, "node", None), "shape_env", None)
            if env is not None:
                return env
    return None


def declared_range_gaps(
    program: Any, dims: Sequence[DynamicDim],
    containers: Optional[Mapping[str, int]] = None,
) -> List[str]:
    """Named reasons the export did not honour the declared dynamic contract.

    An export that specialized — or silently PINNED — a dim we advertise as
    dynamic produces an artifact serving one shape while its metadata claims a
    range, and pgw#704 B2 measured that nothing at runtime refuses the
    difference. So this must fail the mint, not ship.

    Three checks, because presence of a range entry is NOT proof (ie#566 G3 —
    a measured FALSE PASS: wan ti2v-5b declared symbolic H/W, exported clean,
    passed a presence-only gate, and yet a static per-token input of 27,280
    pinned H*W to exactly one shape):

    1. **specialization** — the declared axis is a plain int in the exported
       placeholder, so the dim never became symbolic at all;
    2. **solved range** — the governing symbol's range in the exported program
       must COVER the declared ``[min, max]``. A collapsed (``lower == upper``)
       or narrowed range is a pin, and the artifact declares a narrower ENVELOPE
       than it advertises;
    3. **pinning guards** — an equality guard in the shape env mentioning a
       declared symbol. A dim that is genuinely a function of the declared
       extents forces the tracer to record ``Eq(h*w, N)``; a dim that merely
       shares a factor records nothing. This is the check the presence-only gate
       lacked, and it is evidence-based rather than arithmetic — see
       :func:`_pinning_guards` for why the arithmetic version was wrong.

    A dim carried by a ``repeat=`` container is checked on EVERY exported
    element (pgw#993): the declared name is resolved through
    :func:`exported_input_names`, the same flattening rule
    :func:`dynamic_shapes_spec` mirrors the structure with.
    """
    gaps: List[str] = []
    shapes = _placeholder_shapes(program)
    ranges = getattr(program, "range_constraints", {}) or {}
    declared_symbols: List[Any] = []
    for d in dims:
        if d.min == d.max:
            continue
        for input_name in exported_input_names(d.input_name, containers):
            shape = shapes.get(input_name)
            if shape is None:
                declared = (
                    repr(d.input_name) if input_name == d.input_name
                    else f"{d.input_name!r} (flattened element "
                         f"{input_name!r})")
                gaps.append(
                    f"declared dynamic dim names input {declared}, which is "
                    f"not a user input of the exported program "
                    f"(inputs: {sorted(shapes)!r})")
                continue
            if d.axis >= len(shape):
                gaps.append(
                    f"{input_name}[{d.axis}] is out of range for the exported "
                    f"shape {tuple(str(x) for x in shape)!r}")
                continue
            dim = shape[d.axis]
            text = str(dim)
            if text.lstrip("-").isdigit():
                gaps.append(
                    f"{input_name}[{d.axis}] exported as the STATIC value "
                    f"{text} but is declared dynamic [{d.min}, {d.max}] — "
                    f"export specialized a dim the declaration advertises as "
                    f"dynamic")
                continue
            syms = _free_symbols(dim)
            declared_symbols.extend(syms)
            covered = False
            # The SOLVED range of the axis's full expression, when the program
            # records one. This is what makes a UNIFIED relational axis (#739 /
            # ie#566 §5) gate-able: wan ti2v's per-token dim solves to
            # ``31*s25*s56`` with its own composite range entry, while the
            # per-symbol path below would compare a governing symbol's [20, 40]
            # against the declared [12400, 49600] and refuse a sound artifact.
            # Composite entries are in the axis's OWN units, so the declared
            # bounds compare directly, with no multiple-of scaling.
            expr = getattr(getattr(dim, "node", None), "expr", None)
            interval = ranges.get(expr) if expr is not None else None
            if interval is not None:
                try:
                    lo, hi = int(interval.lower), int(interval.upper)
                except (TypeError, ValueError, OverflowError):
                    lo = hi = -1
                if lo >= 0:
                    if lo == hi:
                        gaps.append(
                            f"{input_name}[{d.axis}] ({expr}) solved to the "
                            f"single value {lo} — the declared range "
                            f"[{d.min}, {d.max}] is advertised but the "
                            f"artifact admits ONE shape")
                    elif lo > d.min or hi < d.max:
                        gaps.append(
                            f"{input_name}[{d.axis}] ({expr}) solved to "
                            f"[{lo}, {hi}] which does not cover the declared "
                            f"[{d.min}, {d.max}] — the artifact declares a "
                            f"narrower envelope than it advertises")
                    continue
            for sym in syms:
                interval = ranges.get(sym)
                if interval is None:
                    continue
                try:
                    lo, hi = int(interval.lower), int(interval.upper)
                except (TypeError, ValueError, OverflowError):
                    continue
                if lo == hi:
                    gaps.append(
                        f"{input_name}[{d.axis}] symbol {sym} solved to the "
                        f"single value {lo} — the declared range "
                        f"[{d.min}, {d.max}] is advertised but the artifact "
                        f"admits ONE shape")
                    covered = True
                    break
                # The symbol may carry a multiple-of factor (8*s95), so compare
                # the DECLARED bounds against the symbol's own solved bounds
                # scaled by the factor the declaration states.
                factor = max(1, int(d.multiple_of or 1))
                want_lo, want_hi = d.min // factor, d.max // factor
                if lo > want_lo or hi < want_hi:
                    gaps.append(
                        f"{input_name}[{d.axis}] symbol {sym} solved to "
                        f"[{lo * factor}, {hi * factor}] which does not cover "
                        f"the declared [{d.min}, {d.max}] — the artifact "
                        f"declares a narrower envelope than it advertises")
                covered = True
                break
            if not covered and not syms:
                gaps.append(
                    f"{input_name}[{d.axis}] is symbolic ({text}) but carries "
                    f"no resolvable symbol; its admissible range is unprovable")

    gaps.extend(_pinning_guards(program, declared_symbols))
    return gaps


def _is_tautology(expr: Any) -> bool:
    """``True`` only when the guard is provably true for EVERY value (pgw#812 D2).

    A gate that cannot tell "pinned" from "trivially true" refuses correct
    mints, which is the expensive direction of the error. flux2 is refused
    today on::

        Eq(Mod(3072*s50 + 1572864, 48*s50 + 24576), 0)

    and ``3072*s + 1572864 == 64 * (48*s + 24576)``, so that is
    ``Mod(64*X, X) == 0`` — identically true, pinning nothing. The algebra
    comes from attention's ``unflatten``/``flatten`` over the concatenated
    image+text stream, so every export of the graph carries it.

    Only a PROOF admits: anything sympy cannot reduce to ``true`` keeps
    refusing, so the gate still fails closed on an unrecognised guard.
    """
    try:
        import sympy

        return sympy.simplify(expr) is sympy.true
    except Exception:  # noqa: BLE001 — an unprovable guard stays refused
        logger.debug("range gate: could not simplify guard %s", expr,
                     exc_info=True)
        return False


def _refuted(value: Any) -> bool:
    """``True`` when a dict source records this relation as PROVEN FALSE.

    ``ShapeEnv.axioms`` is a ``{relation: sympy.true | sympy.false}`` map, and
    ``symbolic_shapes.get_implications`` deposits ``Eq(a, b) => false`` — plus
    its commuted mirror — for every ``Ne(a, b)`` the graph PROVES. So a bare
    KEY of that map is a refutation as often as a pin: pgw#1077 measured six
    such keys (``Eq(Mod(1, s18*s57), 0)`` and friends) refusing a z-image mint
    whose declared symbols nothing pinned.

    Only a recognised false admits; an unrecognised value stays refused, the
    same fail-closed direction :func:`_is_tautology` takes.
    """
    if isinstance(value, bool):
        return value is False
    try:
        import sympy

        return value is sympy.false
    except Exception:  # noqa: BLE001 — an unreadable value stays refused
        return False


def _pinning_guards(program: Any, declared_symbols: Sequence[Any]) -> List[str]:
    """Equality guards that pin a declared-dynamic symbol (ie#566 G3).

    A dim that is GENUINELY a function of the declared extents forces the tracer
    to record an equality guard; a dim that merely happens to share a factor
    records nothing. Measured on this toolchain::

        genuine  x.reshape(b, c, h*w) + tokens   ->  guards ['Eq(s37*s46, 384)']
        coincidence  x.flatten(2).sum(-1) + tokens.sum()  ->  guards []

    In BOTH cases ``range_constraints`` still reports the full declared range,
    which is exactly why a presence check — and even a solved-range check —
    passes the pinned artifact. The guard is the only place the truth appears.

    This replaces an earlier divisibility heuristic that asked whether a static
    dim was a multiple of the product of the declared extents. That test could
    not distinguish the two cases above, and it FALSE-POSITIVED on sdxl, whose
    cross-attention width 2048 and pooled width 1280 are multiples of the
    declared extents' product by pure architectural coincidence — refusing every
    symbolic sdxl mint, including the one reproducing pgw#704's headline. The
    lesson is recorded here because the coincidence is easy to re-introduce:
    integer arithmetic on shapes is NOT evidence of dependence; a solved
    relation is.

    A truly free symbol carries no equality guard, so any ``Eq`` mentioning a
    declared symbol means that symbol is not free.
    """
    declared = {str(sym) for sym in declared_symbols}
    if not declared:
        return []
    env = _shape_env(program)
    if env is None:
        return []
    out: List[str] = []
    seen: set = set()
    for source in ("guards", "axioms"):
        entries = getattr(env, source, None) or ()
        if isinstance(entries, dict):
            # The VALUE is the truth the graph proved — reading keys alone
            # reports every proven inequality as a pin (pgw#1077).
            entries = [key for key, value in entries.items()
                       if not _refuted(value)]
        for entry in entries:
            expr = getattr(entry, "expr", entry)
            if expr is None:
                continue
            if getattr(expr, "func", None) is None or \
                    type(expr).__name__ not in ("Eq", "Equality"):
                continue
            names = {str(sym) for sym in
                     (getattr(expr, "free_symbols", ()) or ())}
            hit = sorted(names & declared)
            if not hit:
                continue
            # A pin has a CONSTANT on one side (``Eq(s37*s46, 384)``). An
            # equality between two symbolic sides (``Eq(s37, 8*s95)``) is the
            # DEFINITIONAL relation our own multiple-of factor introduces —
            # ``8 * Dim(...)`` — and refusing it would block every mint that
            # declares a divisibility, which is all of the image families.
            sides = list(getattr(expr, "args", ()) or ())
            if len(sides) != 2 or not any(
                not (getattr(side, "free_symbols", None) or set())
                for side in sides
            ):
                continue
            if _is_tautology(expr):
                continue
            text = str(expr)
            # ``Eq(a, b)`` and ``Eq(b, a)`` are one relation written twice —
            # get_implications records both — so report it once (pgw#1077).
            key = tuple(sorted(str(side) for side in sides))
            if key in seen:
                continue
            seen.add(key)
            out.append(
                f"the exported program carries the equality guard {text}, "
                f"which PINS the declared dynamic symbol(s) {hit!r} — some "
                f"other input's static shape is an algebraic function of the "
                f"declared extents, so the artifact serves exactly the traced "
                f"shape even though its declared range says otherwise "
                f"(ie#566 G3). Declare that input dynamic too, or make it "
                f"independent of the latent extent")
    return out


# ---------------------------------------------------------------------------
# Compile
# ---------------------------------------------------------------------------












# ---------------------------------------------------------------------------
# Mint-phase telemetry (#757's instrument-first doctrine; recorded per cell)
# ---------------------------------------------------------------------------

#: ``compilation_time_metrics`` keys summarized into named phases. Host C++
#: compile+link (``AotCodeCompiler.compile``) is the stage the JIT path
#: skips entirely — its wrapper is Python — and the primary 3.9x suspect.
_PHASE_KEYS: Dict[str, Tuple[str, ...]] = {
    "lowering_s": ("GraphLowering.run",),
    "codegen_s": ("GraphLowering.codegen",),
    "host_compile_s": ("AotCodeCompiler.compile",),
    "graph_passes_s": (
        "_recursive_pre_grad_passes",
        "_recursive_joint_graph_passes",
        "_recursive_post_grad_passes",
    ),
}

# ---------------------------------------------------------------------------
# The mint
# ---------------------------------------------------------------------------


@dataclass
class MintedArtifact:
    """ONE packed, gated, publishable ENTRY: one graph class."""

    key: str
    entry: str
    artifact: Path
    metadata: Dict[str, Any]
    aliases: Tuple[str, ...] = ()
    mint_phases: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MintResult:
    """Every entry this mint packed, plus its telemetry.

    pgw#1176: a mint no longer produces "a cell". It produces N independently
    keyed, independently publishable, independently armable artifacts and a
    derived MANIFEST digest. Nothing waits for the set to be complete —
    :attr:`entries` is whatever finished, which is what makes the mint's
    durability incremental (a crash costs the one in-flight entry, ~2 min,
    never a 1 h 37 m cell) and what makes a partially compiled family useful.
    """

    entries: Tuple[MintedArtifact, ...]
    manifest: str
    timings: Dict[str, float]
    family: str = ""

    @property
    def keys(self) -> Tuple[str, ...]:
        return tuple(sorted(row.key for row in self.entries))


@dataclass
class _MintedEntry:
    """One exported graph class handed to TCG for compilation."""

    name: str
    spec: ExportSpec
    module: Any
    owner: Any
    program: Any
    input_names: Tuple[str, ...]
    ingress: CallIngress
    timings: Dict[str, Any]


def adapter_arm(fork: Sequence[Tuple[str, Any]]) -> Optional[bool]:
    """The pgw#790 adapter arm a fork coordinate states, or ``None`` when the
    coordinate does not carry one (a bucket-0 family, or a target that carries
    no branch-capable module — wan's ``vae.decode`` in a bucket-128 cell)."""
    value = dict(fork).get(ADAPTER_FORK, None)
    return None if value is None else bool(value)


def declared_fork(fork: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
    """The coordinate MINUS the SDK-synthesized adapter arm — what the
    ENDPOINT declared, and therefore the only part its declaration can be
    asserted against."""
    return {str(n): v for n, v in fork if str(n) != ADAPTER_FORK}


def with_adapter_arm(plan: Any, arm: bool) -> Any:
    """One mint plan pinned to an adapter arm (a fork coordinate, so it names
    the entry and lands in the class hash like every other fork)."""

    return replace(
        plan,
        fork=tuple(sorted(
            tuple(plan.fork) + ((ADAPTER_FORK, bool(arm)),),
            key=lambda pair: str(pair[0]))))


def _entry_spec(spec: ExportSpec, plan: Any, decl: Any) -> ExportSpec:
    """The per-entry :class:`ExportSpec` one mint plan derives from the
    cell-level request."""

    specialization = dict(spec.specialization)
    specialization.setdefault(
        "shape_strategy", _decl.effective_shape_strategy(decl))
    specialization.setdefault("warm_changes_key", bool(decl.warm_changes_key))
    for name, value in plan.fork:
        specialization.setdefault(f"fork.{name}", value)
    fork, dims = _decl.entry_coordinates(plan)
    espec = replace(
        spec,
        target=str(plan.target),
        fork=fork,
        class_dims=dims,
        dynamic=tuple(plan.dynamic),
        specialization=specialization,
    )
    if adapter_arm(plan.fork) is False:
        # pgw#790's branchless class: no bucket, no lifted pair, nothing for
        # the adapter gates to assert. The CELL still declares its bucket —
        # the fork is what says this graph has no branch, and the specialization
        # block records both, so a reader can see which arm they are holding.
        espec = replace(espec, lora_bucket=0, lifted_inputs=(), lora_fqns=())
    return espec


def _run_declared_warm(module: Any, args: Tuple[Any, ...], entry: str) -> float:
    """Execute the declared mint-warm canon: one forward with the entry's
    own seed inputs BEFORE export (the warm-canon obligation — z-image's
    rope pre-warm measurably changes the graph, 4327 cold vs 4285 warmed
    nodes; a family declaring ``warm_changes_key=True`` that skips this
    mints the graph the fleet never serves, and the #699 double-mint
    byte-compare flaps on warm order). Returns the warm seconds."""
    import torch

    t0 = time.monotonic()
    try:
        with torch.no_grad():
            module(*args)
    except Exception as exc:
        raise_if_device_oom(exc, f"entry {entry!r}: declared mint-warm forward")
        raise MintRefused(
            f"entry {entry!r}: declared mint-warm forward failed "
            f"({type(exc).__name__}: {exc}) — warm_changes_key=True makes "
            f"the pre-warm a mint obligation, not a best effort") from exc
    return round(time.monotonic() - t0, 2)


#: pgw#1076: the short spelling of a floating-point dtype, matching the
#: vocabulary the weight-lane labels already use (`bf16`, `fp8-…`) so one cell's
#: `precision` reads the same whether it came from the lane or from this
#: measurement.
_PRECISION_LABELS: Dict[str, str] = {
    "torch.bfloat16": "bf16",
    "torch.float16": "fp16",
    "torch.float32": "fp32",
    "torch.float64": "fp64",
    "torch.float8_e4m3fn": "fp8-e4m3",
    "torch.float8_e4m3fnuz": "fp8-e4m3",
    "torch.float8_e5m2": "fp8-e5m2",
    "torch.float8_e5m2fnuz": "fp8-e5m2",
}






#: pgw#1208: the phase token an unexportable class reports under. One kind, so
#: the hub can COUNT them per family — a class that cannot be exported is a
#: standing authoring/toolchain fact, not a transient, and the fleet's answer to
#: "which classes never compile" should be a query rather than a log hunt.
KIND_ENTRY_EXPORT_UNSUPPORTED = "entry_export_unsupported"








def _export_entry(
    pipeline: Any,
    spec: ExportSpec,
    plan: Any,
    decl: Any,
) -> _MintedEntry:
    """Resolve, feed, warm, export, and gate ONE declared graph
    class. Every refusal is prefixed with the entry name — a multi-graph
    mint that cannot say WHICH class failed is the silent-failure path in
    a new hat (pgw#758).

    Compilation is exclusively TCG's responsibility in the dedicated compile
    child. Export stays here because it runs against the one live pipeline,
    on the one card, inside the one branch-arm toggle."""

    espec = _entry_spec(spec, plan, decl)
    entry = _decl.plan_entry_name(plan)
    timings: Dict[str, Any] = {}

    resolved = _resolve_target(pipeline, espec.target)
    if resolved is None:
        raise MintRefused(
            f"entry {entry!r}: pipeline {type(pipeline).__name__} has no "
            f"compile target {espec.target!r}")
    owner, attr, _fn = resolved
    if decl.forks:
        gaps = _decl.fork_gaps(
            decl, declared_fork(espec.fork), target=espec.target,
            pipeline=pipeline, module=owner)
        if gaps:
            raise MintRefused(
                f"entry {entry!r}: fork gate (pgw#739): " + "; ".join(gaps))
    module = owner if attr == "forward" else _CallableTarget(owner, attr)

    # The LoRA bucket is a CELL-level request but a PER-TARGET fact: adapters
    # ride the branch-capable denoisers, never the VAE (wan's vae.decode
    # entry is bucket-0 in the same cell as its bucket-128 transformer).
    # Scoped by COMPOSED truth (lora_lifted.branch_targets), not vocabulary —
    # and a branch-capable target whose lifting was not installed still fails
    # the lifted-input gate by name, never silently mints bucket-0.
    if espec.lora_bucket or espec.lifted_inputs or espec.lora_fqns:

        branch_owners = {
            id(m) for m in lora_lifted.branch_targets(pipeline).values()}
        if id(owner) not in branch_owners:
            espec = replace(
                espec, lora_bucket=0, lifted_inputs=(), lora_fqns=())
        elif espec.lifted_inputs \
                and lora_lifted.lifted_binding(owner) is None:
            # pgw#822: the ARM, asserted where it is needed rather than
            # assumed. Without this the miss surfaces as `_positionalize`
            # refusing the DECLARATION ("lora_a/lora_b are not parameters of
            # forward") — a true sentence about the wrong thing, which is
            # what sent the first real mint attempt looking at the endpoint's
            # contract instead of at the pipeline's preparation.
            raise MintRefused(
                f"entry {entry!r}: this class declares lifted adapter "
                f"input(s) {list(espec.lifted_inputs)!r} but "
                f"{type(owner).__name__} carries no lifted forward — the "
                f"branch containers are armed and the lifted signature is "
                f"not, so the module about to be exported cannot take the "
                f"adapter (lora_lifted.arm_lifted_lora_lanes installs both)")

    # pgw#1080: a structure-only target's tensors are FAKE and belong to ONE
    # fake mode. Everything that produces a tensor for this export — the
    # example feed, the declared warm, the export itself — has to happen
    # inside that mode or the pieces belong to different modes and
    # `aot_compile` refuses them. `None` on a real-weight mint, where this
    # whole block is the identity context it has always been.
    fake_mode = structure_only.fake_mode_of(owner)
    with structure_only.under(fake_mode):
        return _export_entry_body(
            pipeline, espec, plan, decl, entry=entry, owner=owner, attr=attr,
            module=module, timings=timings, fake_mode=fake_mode)


def _export_entry_body(
    pipeline: Any,
    espec: Any,
    plan: Any,
    decl: Any,
    *,
    entry: str,
    owner: Any,
    attr: str,
    module: Any,
    timings: Dict[str, Any],
    fake_mode: Any = None,
) -> _MintedEntry:
    """The body of :func:`_export_entry`, run inside the target's fake mode.

    Split out rather than indented so the structure-only window is one
    statement and the real-weight path reads exactly as it did.
    """
    builder = aot_inputs.builder_for(espec.family, espec.target)
    args, kwargs = builder(owner, espec)
    if kwargs:
        # All-positional example feeds are a MINT OBLIGATION (pgw#723
        # residuals, pod 9): the AOTI package's call convention mirrors the
        # traced args/kwargs split, and the serve marshal is POSITIONAL
        # (aot_serve.marshal_positional) — a kwarg-traced package arms, then
        # silently revokes to eager on its first call ("Ran into a kwarg
        # keyword mismatch: Got [] but expected ['lora_a','lora_b']",
        # measured). Refused HERE so the failure is a named mint refusal
        # instead of a vacuous eager-serving artifact.
        raise MintRefused(
            f"entry {entry!r}: example feed carries keyword argument(s) "
            f"{sorted(kwargs)!r} — all-positional feeds are a mint "
            f"obligation (pod 9, pgw#723 residuals): a kwarg-traced package "
            f"is uncallable by the positional serve marshal and fails only "
            f"at first serve, silently revoking to eager. Feed every input "
            f"positionally (signature defaults fill the gaps)")

    # The WARM CANON, EXECUTED (declared per family; previously keyed but
    # never acted on): sdxl declared False and skips; z-image's True runs.
    if bool(decl.warm_changes_key):
        timings["warm_s"] = _run_declared_warm(module, args, entry)

    input_names = _input_names(module, args, kwargs)
    # ONE arity map for this arm: the spec builder mirrors the container
    # structure with it and the gates below resolve declared names against
    # the exported ones with it (pgw#993).
    arities = _decl.container_arities(decl, espec, module)
    dynamic = dynamic_shapes_spec(
        espec.dynamic, input_names, arities) if espec.dynamic else None

    def _full_export() -> Any:
        try:
            return export_program(
                module, args, kwargs, dynamic_shapes=dynamic,
                strict=espec.strict)
        except MintRefused as exc:
            raise MintRefused(f"entry {entry!r}: {exc}") from exc

    t0 = time.monotonic()
    if fake_mode is None:
        program = _full_export()
    else:
        # pgw#1080 / ie#628: the TRACE half of the meta-instantiation gate. A
        # structure-only target allocates nothing, so any real tensor born in
        # this window is a model that materializes weights at CALL time — the
        # z-image rope class — and it is refused with the ENDPOINT's own
        # file:line rather than discovered as a mysterious pod OOM.
        try:
            with meta_instantiation.guard(
                    f"trace:{entry}", actionable_only=True) as census:
                program = _full_export()
            if not census.clean:
                # Real, but unattributable — torch's own machinery inside the
                # fake mode. Reported, never refused: see `guard`.
                logger.info(
                    "aot-mint: entry %s traced with %d unattributable real "
                    "allocation(s) (%s)", entry, len(census.events),
                    ", ".join(sorted({e.op for e in census.events})[:6]))
        except meta_instantiation.MetaMaterializationError as exc:
            raise MintRefused(f"entry {entry!r}: {exc}") from exc
    timings["export_s"] = round(time.monotonic() - t0, 2)
    # pgw#1000, telemetry only: the one number that sized "export once,
    # re-propagate per row". Kept after pgw#847 was deleted because it prices
    # the SAME question the export pool now answers with processes, and it is
    # free. Once per process, so a 36-entry mint pays it once.
    prop_probe = _probe_prop_s(program)
    if prop_probe is not None:
        timings["prop_probe_s"] = prop_probe

    gaps = declared_range_gaps(program, espec.dynamic, arities)
    if gaps:
        raise MintRefused(
            f"entry {entry!r}: declared-range gate: " + "; ".join(gaps))
    lifted_gaps = lifted_input_gaps(program, espec, arities)
    if lifted_gaps:
        raise MintRefused(
            f"entry {entry!r}: lifted-input gate: " + "; ".join(lifted_gaps))

    # pgw#725 G3, on the EXPORTEDPROGRAM and before any packing: the adapter
    # must be absent from the constant table AND present among the user inputs.
    # A missing pair is the same defect as a baked one (the branch was traced
    # away, so every request silently gets the base model), and packing renames
    # a plain-__dict__ adapter to _tensor_constant0 — which makes the
    # package-side scan a false PASS. Free here, unsound there.
    if espec.lora_bucket or espec.lifted_inputs or espec.lora_fqns:
        from .api.errors import ValidationError

        try:
            lora_lifted.assert_no_baked_adapter(
                program, label=f"{espec.family}/{espec.target}")
        except ValidationError as exc:
            raise MintRefused(
                f"entry {entry!r}: no-baked-adapter gate (#725 G3): "
                f"{exc}") from exc

    excluded = (
        lora_lifted.LIFTED_INPUT_NAMES
        if adapter_arm(espec.fork) is False
        else ()
    )
    try:
        ingress = build_call_ingress(
            program,
            input_names,
            args,
            kwargs,
            excluded_inputs=excluded,
        )
    except IngressError as exc:
        raise MintRefused(
            f"entry {entry!r}: TCG call-ingress declaration refused "
            f"({exc.reason}: {exc})"
        ) from exc

    return _MintedEntry(
        name=entry, spec=espec, module=module, owner=owner, program=program,
        input_names=input_names, ingress=ingress,
        timings=timings)




def adapter_arm_plans(
    plans: Sequence[Any], pipeline: Any, spec: ExportSpec,
) -> List[Tuple[Any, Optional[bool]]]:
    """``[(plan, arm)]`` — every branch-capable target's plan forked into an
    adapter-bearing and a branchless graph class (pgw#790).

    Scoped by COMPOSED truth, not vocabulary: a cell's non-branch targets
    (wan's ``vae.decode``) fork into nothing, because there is no adapter for
    them to carry and a second identical graph would be pure mint bill. A
    bucket-0 family forks into nothing either — its cell IS the branchless
    class already.

    Adapter-bearing rows come FIRST so the compile child's already-lifted
    pipeline is disarmed exactly once.
    """
    if not int(spec.lora_bucket or 0):
        return [(plan, None) for plan in plans]

    branch_owners = {
        id(m) for m in lora_lifted.branch_targets(pipeline).values()}
    rows: List[Tuple[Any, Optional[bool]]] = []
    for plan in plans:
        resolved = _resolve_target(pipeline, str(plan.target))
        owner = resolved[0] if resolved else None
        if owner is None or id(owner) not in branch_owners:
            rows.append((plan, None))
            continue
        rows.append((with_adapter_arm(plan, True), True))
        rows.append((with_adapter_arm(plan, False), False))
    rows.sort(key=lambda row: row[1] is False)   # stable: order within a group
    return rows


def declaration_module_gaps(
    pipeline: Any, spec: ExportSpec, decl: Any,
) -> List[str]:
    """Every declared input name a graph class's target module cannot take —
    the mint's own signature refusal, asked BEFORE anything is rented
    (pgw#822).

    pgw#822 cost a real L4 pod to learn a sentence a signature comparison
    could have produced locally: the ``lora64`` bucket declares two lifted
    adapter inputs and the module handed to ``torch.export`` did not take
    them. Nothing about that needed a GPU, a child process, or a weight read.

    PER CLASS, because the fork's two halves ask different questions
    (pgw#790): the ``adapter=true`` class is exported from the LIFTED
    forward, so ``lora_a``/``lora_b`` are admissible on a target the mint
    will lift; the ``adapter=false`` class is exported from the PLAIN module
    and declares no adapter at all (``_entry_spec`` zeroes its bucket). A
    single one-size check would either refuse the branchless class or admit a
    lifted declaration on a module that can never carry one.

    ``lora_a``/``lora_b`` are admitted on the strength of the target being
    LIFT-CAPABLE (``lora_lifted.branch_targets``), not of it being lifted
    right now: the caller is a serving parent, and the lift is installed by
    :func:`_arm_branches` inside the mint. That is the same COMPOSED-truth
    scoping ``_export_entry`` applies, so this check predicts the mint rather
    than describing the parent.

    Returns gap sentences; empty means every declared class fits its module.
    Never raises — an unreadable declaration is itself a gap, and the caller
    is deciding whether to mint, not whether to serve.
    """

    gaps: List[str] = []
    try:
        rows = adapter_arm_plans(_decl.cell_plans(decl), pipeline, spec)
        branch_owners = {
            id(m) for m in lora_lifted.branch_targets(pipeline).values()}
    except MintRefused as exc:
        # A refused declaration is PROVEN unmintable — the mint would raise
        # the same sentence, so say it now.
        return [f"the declaration's class set is unmintable ({exc})"]
    except Exception as exc:  # noqa: BLE001
        # Anything else means this check could not READ the composed
        # pipeline. A gate that declines when it cannot see is a new
        # silent-decline (the pgw#813/#815 class in a new hat) — it reports
        # what it can PROVE and abstains otherwise. The mint's own gates stay
        # load-bearing either way.
        logger.debug(
            "aot-mint: declaration/module check not applicable (%s: %s)",
            type(exc).__name__, exc)
        return []

    for plan, _arm in rows:
        entry = _decl.plan_entry_name(plan)
        try:
            espec = _entry_spec(spec, plan, decl)
            resolved = _resolve_target(pipeline, espec.target)
            if resolved is None:
                # Not this gate's question. "The pipeline carries no such
                # target" is a different condition with its own owners
                # (`compile_cache.has_compile_target` on the way in,
                # `_export_entry` on the way out); with no module there is no
                # signature to compare a declaration against, and claiming
                # one would make this check refuse every pipeline shape it
                # merely does not recognise.
                continue
            owner, _attr, _fn = resolved
            positional, keyword_only = _decl.call_signature(
                owner, espec.target, decl.family)
            takes = {p.name for p in positional}
            declared = {i.top_name for i in _decl.target_inputs(
                decl, espec.target)}
            declared |= {a.name for a in _decl.target_args(decl, espec.target)}
            if espec.lifted_inputs and id(owner) in branch_owners:
                # The mint lifts this target; the pair is admissible even
                # though the resident forward does not take it yet.
                declared |= set(espec.lifted_inputs)
                takes |= set(espec.lifted_inputs)
            elif espec.lifted_inputs:
                gaps.append(
                    f"entry {entry!r}: the class declares lifted adapter "
                    f"input(s) {list(espec.lifted_inputs)!r} but "
                    f"{type(owner).__name__} carries no branch-capable "
                    f"module, so no lifted forward can be installed on it")
                continue
        except MintRefused as exc:
            gaps.append(f"entry {entry!r}: {exc}")
            continue
        except Exception as exc:  # noqa: BLE001 — see the abstain note above
            logger.debug(
                "aot-mint: entry %r not checkable (%s: %s)",
                entry, type(exc).__name__, exc)
            continue
        blocked = sorted(declared & keyword_only)
        if blocked:
            gaps.append(
                f"entry {entry!r}: declared input(s) {blocked!r} are "
                f"KEYWORD-ONLY on {type(owner).__name__}."
                f"{_decl.target_attr(espec.target)} — the mint feeds every "
                f"input positionally")
        missing = sorted(declared - takes - keyword_only)
        if missing:
            gaps.append(
                f"entry {entry!r}: declared input(s) {missing!r} are not "
                f"parameters of {_decl.target_attr(espec.target)!r} on "
                f"{type(owner).__name__} (parameters: {sorted(takes)!r})")
    return gaps


def _disarm_branches(pipeline: Any) -> None:
    """Take the pipeline back to the BRANCHLESS graph family (pgw#790).

    Both halves are required: removing the lifted forward alone would leave
    the zeroed branch containers on every leaf, and the trace would still emit
    the branch — the exact arithmetic-over-zeros this fork exists to delete.
    """
    from .models import w8a8_lora

    lora_lifted.remove_lifted_lora_execution_lanes(pipeline)
    w8a8_lora.disable_branch_execution_lanes(pipeline)
    logger.info(
        "aot-mint: branch containers dropped — exporting the adapter=false "
        "graph class(es)")


def _arm_branches(pipeline: Any, bucket: int) -> None:
    """Put the pipeline on the LIFTED branch-bearing graph family — canonical
    branch containers first, lifted call signature second (pgw#822).

    The SAME two shipped calls, in the SAME order, as the serving arm
    (``models.provision.arm_aot``): a lifted class exports the denoiser's
    lifted forward, so the module must carry it before ``builder()`` binds
    ``lora_a``/``lora_b`` to positional slots. pgw#822 is what happens when
    it does not — the child armed the DYNAMO lane (containers only) and the
    declaration was refused against the bare ``UNet2DConditionModel``.

    Owned HERE rather than left to the caller because this function already
    owns the other half of the fork (:func:`_disarm_branches`): an arm state
    machine with one end enforced and the other a convention is how the
    convention gets skipped. Idempotent, so a caller that already armed the
    lifted execution lane pays nothing.

    Also the RE-arm after the branchless exports: the mint process may go on
    to serve or re-mint, and a pipeline left branchless would silently be a
    different graph family.
    """

    lora_lifted.arm_lifted_lora_execution_lanes(pipeline, int(bucket or 0))


# pgw#824: the phase tokens the in-mint progress callback reports under.
# Wire-shared with ``activity.PHASE_*`` and framed verbatim by ``mint_child``,
# so a reader groups a delegated mint's progress on the SAME strings whether it
# came from the child's frames or from the parent's own activity.
PHASE_TRACE_GRAPH = "trace_graph"
PHASE_INDUCTOR_COMPILE = "inductor_compile"
PHASE_SEAL_PUBLISH = "seal_publish"


@dataclass
class MintProgress:
    """The ONE record of where a mint IS — live and post-mortem.

    pgw#824 and pgw#825 arrived at the same question from opposite ends and
    must not answer it twice. pgw#825 needs the partial state (`minted`,
    `timings`, `width`, the mint's own clock) so a mint that ABORTS still
    reports the seconds it spent instead of a bare wall clock. pgw#824 needs
    those same positions pushed out LIVE, because a 20-minute export that
    reports nothing is indistinguishable from a hung one.

    One object carries both, so a live beat and an abort table can never
    disagree about which entry the mint was on: :meth:`beat` records the
    position it reports, and :func:`_attach_partial_phases` stamps that same
    position onto the aborted table as ``at``. That closes pgw#825's one
    remaining blind spot — its per-entry rows name the entries that FINISHED,
    and the entry a mint dies ON is the one a reader most needs named.

    Handed down and mutated in place; ``on_progress`` is optional and
    best-effort by construction — a raising callback never costs a mint.
    """

    on_progress: Optional[Callable[[str, int, int, str], None]] = None
    t_mint: Optional[float] = None
    timings: Dict[str, float] = field(default_factory=dict)
    minted: List[_MintedEntry] = field(default_factory=list)
    width: Optional[aot_compile_pool.PoolWidth] = None
    #: pgw#842: the pool's own ledger (pgw#830) once it has run, carried here
    #: so it reaches the phase table — and therefore the hub — instead of
    #: dying in the mint child's log with the pod.
    pool_ledger: Dict[str, Any] = field(default_factory=dict)
    #: The last position :meth:`beat` reported, verbatim.
    at: Dict[str, Any] = field(default_factory=dict)

    def beat(self, phase: str, step: int, total: int, note: str = "") -> None:
        """Report a position, and REMEMBER it.

        Recording precedes reporting: the position must survive into the abort
        table even when there is no ``on_progress`` sink and even when the sink
        raises, or the two halves of this object would tell different stories.
        """
        phase, step, total = str(phase), int(step), int(total)
        note = str(note)[:180]
        self.at = {
            "phase": phase, "step": step, "total": total, "note": note}
        if self.on_progress is None:
            return
        try:
            self.on_progress(phase, step, total, note)
        except Exception:  # noqa: BLE001 — telemetry never fails a mint
            logger.debug("aot-mint progress callback failed", exc_info=True)


def _attach_snapshot(
    progress: "MintProgress", phase_snapshot: Optional[Path],
) -> None:
    """Make every beat re-write the on-disk phase table and touch podguard.

    pgw#848: a mint that is KILLED still leaves the minutes it did spend behind
    it — a 36-class mint abandoned at 30 must not report "no cell produced".
    Wrapped around the caller's own sink rather than replacing it: both are
    best-effort and neither may cost a mint.

    ONE implementation, shared by :func:`mint` and :func:`mint_graph_classes`
    (pgw#1215). Two would be two answers to "what does an abandoned mint leave
    on disk", and the K-wide driver is precisely the one whose runs get
    abandoned.
    """
    if phase_snapshot is None:
        return
    inner = progress.on_progress
    snap = Path(phase_snapshot)

    def _beat(phase: str, step: int, total: int, note: str) -> None:
        try:
            write_phase_snapshot(snap, progress)
            # pgw#848: the SAME beat tells the pod-side reaper this mint is
            # progressing. It has to be a CHANGING token, so it carries the
            # position — which is the honest signal anyway.
            _touch_pod_progress(f"aot_mint {phase} {step}/{total} {note}")
        except Exception:  # noqa: BLE001 — telemetry never fails a mint
            logger.debug("aot-mint: phase snapshot failed", exc_info=True)
        if inner is not None:
            inner(phase, step, total, note)

    progress.on_progress = _beat




def partial_phase_table(
    progress: MintProgress, *, terminus: str = "aborted",
) -> Dict[str, Any]:
    """Everything this mint has measured SO FAR, as a phase table.

    Empty dict when there is nothing yet — "no measurement" and "zero" must
    not read the same.
    """
    minted = list(progress.minted)
    timings = dict(progress.timings)
    started = progress.t_mint
    where = dict(progress.at)
    if not minted and not timings and not where:
        return {}
    if started is not None:
        # The mint's OWN wall clock, so an aborted total is comparable
        # with a completed one rather than being a sum of entry seconds.
        timings["total_s"] = round(time.monotonic() - float(started), 2)
    table = _mint_phase_table(
        minted, timings, progress.width, progress.pool_ledger)
    table["terminus"] = terminus
    if where:
        # pgw#824 x pgw#825: the entries block names what FINISHED; this
        # names what the mint was ON. Without it an 18-entry mint that
        # dies in entry 12's export reports 11 rows and no twelfth, and
        # the row that matters is the missing one.
        table["at"] = where
    return table


#: pgw#848 long-fuse sweep: where the POD-SIDE reaper reads progress from.
#: Set by `podguard.arm()` on every pod it rents; absent everywhere else, so
#: this is inert off-pod and costs one `os.environ.get` per beat.
PODGUARD_STATE_ENV = "PODGUARD_STATE"


def _touch_pod_progress(note: str) -> None:
    """Tell the pod-side reaper this mint is still doing work.

    pgw#848. podguard's own docstring says both its layers "kill on liveness +
    progress-staleness" — Paul's rule, implemented — and the pod-side layer
    reads a token file that `podguard-progress` writes. **Nothing in the SDK
    has ever written it.** Zero references to podguard anywhere in gen_worker,
    so the pod-side progress path had no producer and the ONLY signal keeping
    a minting pod alive was podguard's own renewal thread.

    SCOPE, stated because it is narrower than it looks. `PODGUARD_STATE` is
    injected by `podguard.arm()`, which runs only when PODGUARD creates the
    pod. A HUB-created pod never passes through it, carries no watchdog and no
    state dir, and this call is therefore a no-op there — which is most pods.
    So this makes lane-rented pods
    progress-keyed and leaves hub-created pods on renter-liveness plus a fixed
    1800 s grace (`lease_seconds` 900 x `REAP_LEASE_MULTIPLE` 2.0).

    It did NOT cause pgw#846 attempt sixteen and would not have prevented it:
    that pod's verdict was UNREACHABLE (podguard cannot ssh a hub-created pod
    — no lane key), so `reap()` fell through to `box_stale=1950s > 1800s` and
    terminated on RENTER liveness alone, which an unstarted Keeper had frozen.
    One failure, not two. The structural fix for hub-created pods is for
    `reap()` to ask the HUB whether a pod is progressing — the signal already
    exists (`SelfMintActivityRunning`, refreshed every ~5 s by
    `mint_process._observe`) — rather than to ssh the pod at all.

    Best-effort and unconditional-safe: a mint must not fail because a
    telemetry file could not be written, and the whole call is inert when
    `PODGUARD_STATE` is unset (every non-podguard pod, and this box).
    """
    if podguard_status() != PODGUARD_ARMED:
        return
    state = os.environ.get(PODGUARD_STATE_ENV, "").strip()
    try:
        Path(state).mkdir(parents=True, exist_ok=True)
        # CONTENT, not mtime: the watchdog compares the token, so a value that
        # does not change reads as no progress even if the file is rewritten.
        (Path(state) / "progress").write_text(note)
    except OSError:
        logger.debug("aot-mint: could not touch podguard progress", exc_info=True)


#: The three states pgw#929 requires this adapter to be able to REPORT. The
#: distinction that matters is `not_present` vs `invalid`: a hub-created pod
#: legitimately has no podguard producer, whereas a path that is set but
#: unusable is a rented pod whose progress signal is silently going nowhere —
#: and a mint that looks unprogressing to a watchdog gets reaped.
PODGUARD_ARMED = "armed"
PODGUARD_NOT_PRESENT = "not_present"
PODGUARD_INVALID = "invalid"


def podguard_status() -> str:
    """Validate the external watchdog handoff and say which state it is in.

    pgw#929 keeps `PODGUARD_STATE` deliberately — it is the ONE env in the IPC
    bucket that is NOT a mechanical parent-to-child handoff this program could
    move to argv. Its producer is `podguard.arm()`, an external process that
    runs before this one exists on pods podguard rents, so there is no argv to
    put it on and no `Settings` that could own it. It is an adapter contract
    with an outside system, and it stays.

    What pgw#929 adds is that the adapter must be HONEST about its own state
    rather than treating "unset" and "broken" as the same silent no-op.
    """
    raw = os.environ.get(PODGUARD_STATE_ENV, "").strip()
    if not raw:
        return PODGUARD_NOT_PRESENT
    path = Path(raw)
    if not path.is_absolute():
        return PODGUARD_INVALID
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError:
        return PODGUARD_INVALID
    if not os.access(path, os.W_OK):
        return PODGUARD_INVALID
    return PODGUARD_ARMED




def write_phase_snapshot(path: Path, progress: MintProgress) -> None:
    """Put the partial table on DISK, atomically, as the mint runs.

    pgw#848. Attaching the table to an exception (below) only reaches a mint
    that gets to raise one. A mint that is KILLED — the parent abandons it,
    the OOM killer takes it, the pod goes — raises nothing and writes no
    report, and every measurement it made dies with the process.

    That is not hypothetical: attempt sixteen compiled for **29 minutes** and
    reported `status=abandoned total_s=1741.33 — no cell produced`. Zero
    `entry:` rows. No `pool` row. K, its binding constraint, the per-entry
    timings and the peaks were all measured and all discarded, and the
    K-and-binding answer had to be re-bought with another pod.

    A file the parent can read after a hard kill is the only shape that
    survives a signal, which is the same reason the pgw#848 resume design
    keys on content on disk rather than on the process staying alive.
    """
    table = partial_phase_table(progress, terminus="in_flight")
    if not table:
        return
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(table))
    os.replace(tmp, path)   # atomic: a reader never sees a half-written table









def class_manifest(
    entry_blocks: Mapping[str, Mapping[str, Any]], spec: ExportSpec,
) -> str:
    """A deterministic v1 telemetry view over TCG graph classes.

    Each member remains independently keyed and admitted. The manifest is a
    report of what one supervised mint observed, never an identity or trust
    unit.
    """

    del spec
    rows = [
        {
            "name": str(name),
            "class_hash": str(block.get("class_hash") or ""),
        }
        for name, block in sorted(entry_blocks.items())
    ]
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return "compiled-graph-manifest-v1-" + hashlib.sha256(payload).hexdigest()




def fold_held_graph_classes(
    held: Sequence[MintedArtifact], *, spec: ExportSpec,
) -> MintResult:
    """The result for a mint that has to compile NOTHING (pgw#1215 step 4).

    Coverage accretes on disk, so an attempt can legitimately find every
    declared graph class already packed — a supervisor restarted after a
    crash, or a retry whose only failing class succeeded on the pass before.
    Standing up a K-wide pool to prove that is more expensive than the answer.

    Runs the same two gates and the same fold :func:`mint_graph_classes` runs
    over its shares, because they are properties of the SET and not of who
    compiled it: a pgw#917 cluster is just as ambiguous when it is read off
    disk, and the declaration-wide coverage label must be the same digest
    either way.
    """
    metas = {str(row.entry): dict(row.metadata) for row in held}
    blocks: Dict[str, Dict[str, Any]] = {}
    for name, meta in metas.items():
        block = meta.get(GRAPH_CLASS_BLOCK)
        if not isinstance(block, dict):
            raise MintRefused(
                f"held graph class {name!r} carries no graph_class, so its "
                f"coverage cannot be folded")
        blocks[name] = dict(block)
    absorbed_by = canonicalize_packed_classes(blocks, metas)
    absorbed = {n for names in absorbed_by.values() for n in names}
    survivors = [row for row in held if str(row.entry) not in absorbed]
    manifest = class_manifest(
        {n: b for n, b in blocks.items() if n not in absorbed}, spec)
    return MintResult(
        entries=tuple(survivors), manifest=manifest,
        timings={"total_s": 0.0, "held_classes": float(len(held))},
        family=spec.family)


def mint_graph_classes(
    template: aot_compile_pool.EntryJob,
    *,
    workdir: Path,
    width: aot_compile_pool.PoolWidth,
    spec: ExportSpec,
    python: str = "",
    on_progress: Optional[Callable[[str, int, int, str], None]] = None,
    phase_snapshot: Optional[Path] = None,
    held: Sequence[MintedArtifact] = (),
    should_abandon: Optional[Callable[[], bool]] = None,
) -> MintResult:
    """Trace, compile and pack a family's declared graph classes K-wide, in
    CHILDREN, and return what they packed BESIDE what this pod already held.

    th#1834 Phase 3's two-tier shape (pgw#1215). The caller does NOT export:
    everything a child needs to build its own weight-free target rides on
    ``template`` (``function`` / ``modules`` / ``slots`` / ``cfg`` /
    ``posture`` / ``out_dir``), the pool stamps the share and the
    locations, and each child hands back an artifact that is already keyed and
    already carries its envelope. The ExportedProgram is never serialized.

    ``spec`` is the caller's own :class:`ExportSpec` for this family — the same
    object it would have handed :func:`mint`. It names the supervised mint but
    supplies no artifact identity: each child derives and stamps its exact TCG
    graph class from the pipeline it composed.

    Refusals are mapped onto the mint's own vocabulary, exactly as the old
    program-staging driver did: a MEMORY shortfall is
    :class:`MintResourceExhausted` (retryable at a narrower K) and everything
    else is :class:`MintRefused` (deterministic, terminal). Collapsing the two
    is how the ONE failure class a narrower pool would have fixed became the
    one class routed down the never-retry path (pgw#848).

    ``held`` (pgw#1215 step 4) is what an EARLIER attempt of this same mint
    already packed — ``template.have_classes`` is the set of names the
    children were told to skip, and these are the artifacts behind it. They
    join the result as ordinary entries and they join the manifest fold,
    because coverage is a property of the pod and not of one attempt. A
    retry's honest report is *"36 of 36 classes"*, not *"1 of 36"*.

    ⚠️ This function does NOT publish and does not arm. The supervisor's row
    loop owns local CAS -> verify -> arm -> async publish per graph class
    (pgw#1183); here the terminus is an artifact on disk with its key and its
    envelope already stamped by the child that traced it.
    """
    progress = MintProgress(on_progress=on_progress)
    progress.width = width
    _attach_snapshot(progress, phase_snapshot)
    t_mint = time.monotonic()
    progress.t_mint = t_mint
    pool = aot_compile_pool.EntryCompilePool(
        Path(workdir), width=width, python=python)
    progress.beat(
        PHASE_INDUCTOR_COMPILE, 0, width.workers,
        f"{width.workers} compile child(ren), one share each — {width.reason}")
    try:
        packed = pool.compile(
            template,
            on_share=lambda name, done, total: progress.beat(
                PHASE_INDUCTOR_COMPILE, done, total, name),
            should_abandon=should_abandon)
    except aot_compile_pool.EntryCompileAbandoned:
        # Not a failure and not this mint's fault: the ledger still has to
        # survive, because the next attempt sizes K off it (pgw#848).
        progress.pool_ledger = _pool_facts(pool)
        raise
    except aot_compile_pool.EntryCompileFailed as exc:
        # pgw#848: the pool's ledger and its MEASURED peak have to survive the
        # failure, because the aborted phase table is what the parent banks
        # and re-sizes K from. Without this the OOM'd attempt teaches the
        # retry nothing and attempt 2 runs the identical width.
        progress.pool_ledger = _pool_facts(pool)
        if exc.resource:
            raise MintResourceExhausted(
                str(exc), entry=exc.entry, basis=exc.basis,
                peak_rss_bytes=exc.peak_rss_bytes) from exc
        raise MintRefused(str(exc)) from exc
    progress.pool_ledger = _pool_facts(pool)

    timings = progress.timings
    timings["compile_all_s"] = round(time.monotonic() - t_mint, 2)
    timings["entry_workers"] = float(width.workers)
    timings["total_s"] = round(time.monotonic() - t_mint, 2)
    metas: Dict[str, Dict[str, Any]] = {}
    decoded_blocks: Dict[str, Dict[str, Any]] = {}
    for name in sorted(packed):
        row = packed[name]
        try:
            meta = dict(msgspec.json.decode(row.metadata.encode()))
        except (msgspec.DecodeError, ValueError) as exc:
            raise MintRefused(
                f"graph class {name!r}: the compile child returned an "
                f"unreadable envelope ({exc}) — an artifact whose metadata "
                f"this process cannot parse cannot be published") from exc
        graph_class = meta.get(GRAPH_CLASS_BLOCK)
        if (
            not isinstance(graph_class, dict)
            or str(graph_class.get("name") or "") != name
            or str(meta.get("compiled_graph_key") or "") != str(row.key)
        ):
            raise MintRefused(
                f"graph class {name!r}: TCG metadata does not restate the "
                "child's exact name and compiled_graph_key")
        metas[name] = meta
        decoded_blocks[name] = dict(graph_class)
    keys: Dict[str, str] = {name: str(packed[name].key) for name in metas}
    artifacts: Dict[str, Path] = {
        name: Path(packed[name].artifact) for name in metas}
    # What an earlier attempt already packed, joining as an ordinary entry. A
    # class that is BOTH held and freshly packed means the skip list did not
    # reach the child that owned it, which would publish two artifacts for one
    # class — refused by name rather than resolved by last-writer-wins.
    for carried in held:
        name = str(carried.entry)
        if name in metas:
            raise MintRefused(
                f"graph class {name!r} was compiled by this attempt AND is "
                f"already held from an earlier one — the skip list did not "
                f"reach the child that owns it, so one class would publish "
                f"two artifacts")
        held_meta = dict(carried.metadata)
        held_block = held_meta.get(GRAPH_CLASS_BLOCK)
        if not isinstance(held_block, dict):
            raise MintRefused(
                f"held graph class {name!r} carries no graph_class, so its "
                f"coverage cannot be folded")
        metas[name] = held_meta
        decoded_blocks[name] = dict(held_block)
        keys[name] = str(carried.key)
        artifacts[name] = Path(carried.artifact)

    # ── INGRESS MERGE (pgw#917 proper), then DEDUPE BY KEY ────────────────
    #
    # The parent is the only process that sees every share, so BOTH gates run
    # here and in this order. The ingress merge is the semantic one — two rows
    # the dispatch cannot tell apart are ONE class, and a cluster that is not
    # one class is a terminal refusal naming the axis; it keys them APART
    # (`class_hash` folds `class_dims`), so the by-key dedupe below never sees
    # the pair. The by-key dedupe is the narrower second net: a true same-key
    # duplicate, which publish would otherwise discover as a 409 on the pod
    # after both compiles were paid for.
    absorbed_by: Dict[str, List[str]] = {
        keep: list(merged) for keep, merged
        in canonicalize_packed_classes(decoded_blocks, metas).items()}
    #
    # A compile child holds ONE SHARE, so two declared classes that key
    # identically can land in different children and neither can see the
    # other. The pool returns a dict keyed by class NAME, so both survive the
    # collection intact and the collision is first discovered by the HUB — a
    # duplicate-key 409 on the second publish, on the pod, after both compiles
    # are paid for. The parent is the only process that sees every share, so
    # it collapses them HERE: one artifact per key, the absorbed names
    # recorded as that entry's aliases. The key is what publish uniques on, so
    # grouping by it makes the 409 impossible by construction rather than
    # merely unlikely.
    #
    # ⚠️ It is NOT pgw#917's ingress merge and must not be read as one: rows
    # that share an ingress contract while differing in `class_dims` key APART
    # (dims fold into `class_hash`), so this never sees them. That merge is
    # `canonicalize_packed_classes`, run above.
    #
    # An alias is not a `class_hash` fact (`aot_serve.class_hash` folds named
    # fields only), so recording one cannot re-key the survivor.
    merged_at_ingress = {n for names in absorbed_by.values() for n in names}
    survivor_of: Dict[str, str] = {}
    for name in sorted(metas):
        if name in merged_at_ingress:
            continue
        key = keys[name]
        keep = survivor_of.setdefault(key, name)
        if keep != name:
            absorbed_by.setdefault(keep, []).append(name)

    phase_table = _mint_phase_table(
        [], timings, width, progress.pool_ledger)
    entries: List[MintedArtifact] = []
    blocks: Dict[str, Dict[str, Any]] = {}
    absorbed = {n for names in absorbed_by.values() for n in names}
    for name in sorted(metas):
        if name in absorbed:
            continue
        block = decoded_blocks[name]
        merged = sorted(absorbed_by.get(name) or ())
        if merged:
            logger.info(
                "aot-mint: pgw#917 graph class %s absorbed %d class(es) "
                "(one ingress contract, or one key) (%s) -> %s",
                name, len(merged), ", ".join(merged), keys[name])
        blocks[name] = block
        entries.append(MintedArtifact(
            key=keys[name], entry=name,
            artifact=artifacts[name], metadata=metas[name],
            aliases=tuple(merged), mint_phases=phase_table))
    # The declaration-wide telemetry view is folded HERE because this is the
    # only process that sees every share. It never enters TCG's closed artifact
    # metadata. Fold over SURVIVORS: an absorbed class contributes the same
    # class_hash its survivor already contributes, so counting it as well would
    # make the report depend on how the declaration happened to be sharded.
    manifest = class_manifest(blocks, spec)
    logger.info(
        "aot-mint: pgw#1215 %d compile child(ren) packed %d graph class(es) "
        "in %.0fs (sum of child seconds %.0fs, peak child RSS %.1f GiB) -> "
        "manifest %s",
        width.workers, len(entries), time.monotonic() - t_mint,
        sum(pool.entry_seconds.values()), pool.peak_rss_bytes / 1024**3,
        manifest)
    return MintResult(
        entries=tuple(entries), manifest=manifest, timings=timings,
        family=spec.family)


#: The mint window `entry_device_peaks` measures. Named on the row rather than
#: implied, because an EXPORT high-water and an entry COMPILE high-water are
#: different questions about the same card and maxing them together would
#: produce a number describing neither (`_ExportFootprint`'s own docstring is
#: the worked example of that mistake).
DEVICE_PEAK_PHASE = "entry_compile"


def _device_peak_provenance() -> Dict[str, str]:
    """The conditions every row in this pool's device census was taken under.

    Stated ONCE beside the rows rather than repeated on each: they come from
    one child, on one card, under one toolchain, so per-row copies could only
    ever disagree with each other. Read in the CHILD deliberately — it is the
    process that ran on the card, and on a multi-GPU box the parent's probe can
    name a different device.

    ``weight_lane`` is NOT here: the parent owns it (it is what it already keys
    the RSS bank by) and adds it when banking. A fact belongs to whoever knows
    it first-hand.
    """
    from . import compile_cache as cc

    try:
        runtime = cc.runtime_key()
    except Exception:  # noqa: BLE001 — telemetry never fails a mint
        runtime = {}
    try:
        toolchain = toolchain_axis_digest(dict(cc.toolchain_digest()))
    except Exception:  # noqa: BLE001
        toolchain = ""
    try:
        version = cc.gen_worker_version()
    except Exception:  # noqa: BLE001
        version = ""
    return {
        # Both namings of the card: the SKU a human reads and the arch the
        # kernels were built for. A cell minted at the wrong arch is
        # unadoptable, so a reading must never be shared across arches.
        "card": str(runtime.get("sku") or ""),
        "sm": str(runtime.get("sm") or ""),
        # The SAME digest the cell key's toolchain axis uses, so a banked row
        # and the cell it was measured for agree on what this toolchain is.
        "toolchain": str(toolchain),
        "gen_worker": str(version),
        "phase": DEVICE_PEAK_PHASE,
    }


def _pool_facts(pool: aot_compile_pool.EntryCompilePool) -> Dict[str, Any]:
    """The pool block of the phase table, on BOTH termini (pgw#848).

    An aborted mint's pool facts are the ones a reader most needs — they are
    the inputs to the decision that has to change on the retry.
    """
    facts: Dict[str, Any] = {
        **pool.ledger.facts(),
        # Observed, not intended: the only load-independent evidence that the
        # pool actually overlapped rather than looping K-wide on paper.
        "peak_concurrency": int(pool.peak_concurrency),
        "peak_child_rss_bytes": int(pool.peak_rss_bytes),
        # pgw#877 #2: the entry children's own DEVICE high-water, which is what
        # the NEXT mint's per-entry ask is sized from. It rides the phase table
        # rather than a second event for the same reason the RSS figure does:
        # the phase table is what survives the mint child, and the mint child
        # is the process that dies.
        "peak_child_device_bytes": int(pool.peak_device_bytes),
        # pgw#1205: the same reading, PER GRAPH CLASS, with the conditions it
        # was taken under stated once beside it. `peak_child_device_bytes`
        # above is the max across a whole cell — one number for 18 classes,
        # which cannot answer the only question anyone asks of it. These rows
        # can, and they ride the SAME phase table, so the row the hub receives
        # and the row this machine banks are the same bytes rather than two
        # measurements that have to be reconciled.
        "entry_device_peaks": {
            name: {"allocated_bytes": int(a), "reserved_bytes": int(r)}
            for name, (a, r) in sorted(pool.entry_device_peaks.items())
        },
        "device_peak_provenance": _device_peak_provenance(),
    }
    if pool.oom_entry:
        facts["oom_entry"] = pool.oom_entry
        facts["oom_basis"] = pool.oom_basis
    return facts


@dataclass(frozen=True)
class _DeclaredArg:
    """One declared input as the INGRESS ASSERTION sees a tensor: a shape and
    a dtype, and nothing else. Nothing is allocated — the gate has to stay
    cheap enough to run before a single kernel is built."""

    shape: Tuple[int, ...]
    dtype: str




def _representative_calls(contract: CallIngress) -> Tuple[Dict[str, Any], ...]:
    """One call per corner of an entry's symbol hull — all symbols at their
    lower bound, all at their upper, all at the midpoint — deduplicated.

    A fully specialized entry (the sdxl case) yields exactly one call, which
    is the call its class row exists to serve. Shared by the two sites that
    ask the ambiguity question — the whole-declaration gate, which holds
    ``ExportedProgram``s, and :func:`canonicalize_packed_classes`, which holds
    only packed envelopes — so the sharded path and the serial path cannot
    drift into two ideas of which calls an entry admits.
    """

    def _at(pick: Callable[[int, int], int]) -> Dict[str, Any]:
        return {
            spec.name: _DeclaredArg(
                tuple(
                    dim if isinstance(dim, int)
                    else pick(*contract.symbol_bounds[dim])
                    for dim in spec.shape),
                spec.dtype,
            )
            for spec in contract.inputs
        }

    calls: List[Dict[str, Any]] = []
    for pick in (lambda lo, _hi: lo,
                 lambda _lo, hi: hi,
                 lambda lo, hi: (lo + hi) // 2):
        call = _at(pick)
        if call not in calls:
            calls.append(call)
    return tuple(calls)


def _admits(contract: Any, call: Mapping[str, Any]) -> bool:
    """Does this entry admit this call? Asked of
    :func:`aot_serve.assert_ingress` itself — the same function, on the same
    contract shape, that :meth:`aot_serve.EntryDispatch.select` runs on a pod.
    Keyword-fed, because that is how the negative contract (an EXCLUDED input
    the call carries) is visible at all."""
    try:
        aot_serve.assert_ingress(contract, (), call)
    except aot_serve.IngressContractError:
        return False
    return True




def _differing_axes(
    identities: Mapping[str, Mapping[str, Any]],
) -> Tuple[str, ...]:
    """The named identity axes on which a colliding cluster disagrees."""
    axes = sorted({axis for ident in identities.values() for axis in ident})
    out: List[str] = []
    for axis in axes:
        values = {
            json.dumps(ident.get(axis), sort_keys=True, default=str)
            for ident in identities.values()
        }
        if len(values) > 1:
            out.append(axis)
    return tuple(out)




def _packed_class_identity(
    block: Mapping[str, Any], meta: Mapping[str, Any],
) -> Dict[str, Any]:
    """TCG identity excluding the class-row coordinate being merged.

    The child returns TCG's closed graph-class declaration. The worker may
    merge two dispatch-equivalent rows only when every TCG identity fact other
    than ``name``/``class_dims``/their derived ``class_hash`` agrees, and the
    artifact compatibility axes agree too. Re-deriving a subset here would
    create a second identity contract beside TCG.
    """
    declaration = {
        str(key): value
        for key, value in block.items()
        if key not in {"name", "class_dims", "class_hash"}
    }
    return {
        **declaration,
        "sm": str(meta.get("sm") or ""),
        "toolchain": dict(meta.get("toolchain") or {}),
    }


def canonicalize_packed_classes(
    blocks: Mapping[str, Mapping[str, Any]],
    metas: Mapping[str, Mapping[str, Any]],
) -> Dict[str, List[str]]:
    """pgw#917 over PACKED envelopes: ``{survivor: [absorbed names]}``.

    :func:`canonicalize_dispatch_classes` asks the same question of
    ``_MintedEntry`` rows — before a kernel is built, which is the pgw#847
    compile saving — and it can only be asked there by a process holding the
    WHOLE declaration's ``ExportedProgram``s. Under th#1834 Phase 3 no such
    process exists: a compile child holds ONE SHARE, and the serving parent
    that supervises the shares may not trace at all (the th#1299 fence). So
    the sharded path lost the gate entirely, and the loss is SILENT — worse
    than the duplicate-key 409 an earlier reading of this predicted.
    Measured: a mergeable pair keys APART, because ``aot_serve.class_hash``
    folds ``class_dims`` and ``class_dims`` is the one axis such a pair
    differs on (``6decad0789e30a3a`` vs ``a185615c3fd880e4`` over a
    byte-identical block). Both rows compile, both publish, both arm, and
    :meth:`aot_serve.EntryDispatch.select` answers ``entry_ambiguous`` on
    every call they carry — 100 % eager on those coordinates, which is the
    4,200-refusal defect pgw#917 was filed to fix. The parent-side dedupe by
    KEY that landed with the keystone is a different, narrower invariant and
    never sees this pair.

    This is the same gate at the only seam the supervisor can reach: the
    packed envelope. It is asked of TCG's ``CallIngress.from_graph`` and
    ``aot_serve.assert_ingress`` — the serve path's own parser and its own
    admission — so it cannot drift from what dispatch will do, and its
    identity axes are ``_class_identity``'s, complete (see
    :func:`_packed_class_identity`). Merge when a colliding cluster differs
    only on the class-row coordinate; REFUSE, naming the members and the
    differing axes, when it does not.

    **What it does NOT recover, stated rather than implied:** the duplicate
    COMPILE. Both members of a mergeable cluster are already built by the
    time an envelope exists, so pgw#847's "36 of sdxl regional's 72 compiles
    bought nothing" still costs what it costs on this path. Recovering that
    needs the decision BEFORE the trace, and the predicate is a property of
    the traced program — so it belongs to whichever change gives the shards a
    cluster-preserving partition, not to this one. Correctness first: a merged
    entry serves those coordinates compiled today, where two published rows
    served them eager.
    """
    groups: Dict[Tuple[str, Any], List[str]] = {}
    for name in sorted(blocks):
        block = blocks[name]
        fork = {str(n): v for n, v in (block.get("fork") or [])}
        groups.setdefault(
            (str(block.get("target") or ""), fork.get(ADAPTER_FORK)),
            []).append(name)

    aliases: Dict[str, List[str]] = {}
    conflicts: List[str] = []
    for members in groups.values():
        if len(members) < 2:
            continue
        declared: Dict[str, Tuple[Any, Tuple[Dict[str, Any], ...]]] = {}
        for name in members:
            try:
                graph = blocks[name].get("graph")
                if not isinstance(graph, Mapping):
                    raise ValueError("graph class records no graph interface")
                contract = CallIngress.from_graph(graph)
            except IngressError as exc:
                # An unreadable declaration is not "probably fine": it is an
                # artifact whose dispatchability nobody can prove.
                raise MintRefused(
                    f"graph class {name!r}: dispatch-ambiguity gate cannot "
                    f"read the packed ingress contract, so this artifact "
                    f"cannot be shown to be dispatchable at all: {exc}"
                ) from exc
            declared[name] = (contract, _representative_calls(contract))

        cluster_of: Dict[str, str] = {name: name for name in declared}

        def _root(name: str, _of: Dict[str, str] = cluster_of) -> str:
            while _of[name] != name:
                _of[name] = _of[_of[name]]
                name = _of[name]
            return name

        for name, (_own, calls) in declared.items():
            for other, (contract, _c) in declared.items():
                if other == name or not any(
                    _admits(contract, call) for call in calls
                ):
                    continue
                a, b = _root(name), _root(other)
                if a != b:
                    cluster_of[max(a, b)] = min(a, b)
        clusters: Dict[str, List[str]] = {}
        for name in sorted(declared):
            clusters.setdefault(_root(name), []).append(name)

        for cluster in clusters.values():
            if len(cluster) < 2:
                continue
            identities = {
                name: _packed_class_identity(blocks[name], metas.get(name, {}))
                for name in cluster
            }
            axes = _differing_axes(identities)
            if axes:
                conflicts.append(
                    f"{sorted(cluster)[:4]!r} collide at ingress but are NOT "
                    f"one class — they differ on {list(axes)!r}")
                continue
            keep, *rest = sorted(cluster)
            aliases[keep] = rest

    if conflicts:
        raise MintRefused(
            f"dispatch-ambiguity gate: {len(conflicts)} cluster(s) of packed "
            f"graph classes are admitted by more than one entry of the same "
            f"dispatch, so every call they carry would be refused "
            f"'entry_ambiguous' and served EAGER — "
            + "; ".join(conflicts[:4]) + ". Rows that reduce to ONE "
            "dispatchable ingress contract are one entry and are merged "
            "automatically; these cannot be, because the named axes say they "
            "are different artifacts. Fix the declaration so every entry's "
            "ingress contract is uniquely admitting, rather than publishing a "
            "class the dispatch could never select")

    if aliases:
        for keep, merged in sorted(aliases.items()):
            logger.info(
                "aot-mint: pgw#917 canonicalized %d packed graph class(es) "
                "onto entry %r — identical ingress contract, target and code, "
                "so they are ONE dispatchable class: %s",
                len(merged), keep, merged)
        activity_mod.emit_event(
            "aot_class_canonicalized",
            f"{sum(len(v) for v in aliases.values())} of {len(blocks)} packed "
            f"graph classes reduce to an ingress contract a sibling already "
            f"declares; merged onto {len(aliases)} entry/entries as aliases "
            f"instead of publishing a class the dispatch could never select: "
            + "; ".join(
                f"{keep} <- {merged}"
                for keep, merged in sorted(aliases.items())[:4]),
            phase="entry_merged",
        )
    return aliases








@dataclass
class TracedClass:
    """One declared graph class, traced for its IDENTITY and nothing else."""

    name: str
    #: The entry-envelope fields that reach this class's ``class_hash``.
    block: Dict[str, Any]
    nodes: int
    #: Held only for the caller's probe window; drop it before the next class.
    program: Any
    #: How many classes the WHOLE declaration produces on this pipeline —
    #: carried on every row so a sharded caller can prove its shares are the
    #: whole set without enumerating it itself.
    declared: int = 0
    #: This row's own export timings. TCG compile/reuse timings are measured by
    #: the dedicated compile child rather than smuggled through the trace.
    timings: Dict[str, Any] = field(default_factory=dict)

    def release(self) -> None:
        """Drop everything but the KEYING facts.

        ``boot_trace_child`` and ``measure_child`` want the block, node count
        and timings; the exported program is megabytes and nothing downstream
        of a completed compile reads it.
        """
        self.program = None


def declared_class_rows(pipeline: Any, spec: ExportSpec, decl: Any) -> List[Any]:
    """The family's declared graph-class rows, adapter-forked, in the order the
    mint exports them (adapter-bearing first, then the branchless group).

    ONE enumeration. The fork depends on the COMPOSED pipeline
    (``lora_lifted.branch_targets``), not on the declaration, which is why no
    caller can enumerate this without a pipeline — and why the boot derivation
    shards by INDEX rather than by name.
    """
    rows = adapter_arm_plans(_decl.cell_plans(decl), pipeline, spec)
    rows.sort(key=lambda row: (
        row[1] is False, _decl.plan_entry_name(row[0])))
    return rows


def trace_for_key(
    pipeline: Any,
    spec: ExportSpec,
    decl: Any,
    *,
    share_index: int = 0,
    share_count: int = 1,
    have_classes: Sequence[str] = (),
) -> Iterator[TracedClass]:
    """Export the named declared graph classes and yield each one's KEYING
    facts — §4.27 step 1's unit of work (pgw#1089).

    This is the mint's export loop with the compile, the packaging and every
    package-side gate removed, and it lives HERE rather than in the boot module
    for two reasons the boot module cannot satisfy on its own:

    * the **branch-arm ordering rule** is pipeline state — the arm itself,
      adapter-bearing rows first, ONE ``_disarm_branches`` for the whole
      branchless group. A caller that ordered its rows differently would trace
      different graphs, and the rule belongs beside the loop that made it;
    * the trace itself is ``_export_entry``, whose refusals, fork gate,
      declared-range gate and lifted-input gate are the mint's. A boot-side
      re-implementation would be a second trace path, and the two would
      eventually key differently — which is the whole failure this derivation
      exists to make impossible.

    ``share_index``/``share_count`` select one trace child's share by INDEX
    into that order — ``rows[i::K]``, round-robin, so every child draws a mix
    of the (few, expensive) denoiser rows and the (many, cheap) rest rather
    than one child drawing the whole denoiser group. Sharding by index rather
    than by NAME is not a convenience: the adapter fork is decided by the
    COMPOSED pipeline, so no parent can enumerate the names to hand out.

    The FULL row count is yielded on every ``TracedClass`` (``declared``), so
    the parent can prove the shares reconstruct the whole class set without
    ever having enumerated it — a stronger check than comparing against a
    parent-side guess would have been.

    The yielded program remains in this process and is consumed directly by
    TCG. No ExportedProgram and no loose compiler file crosses a process
    boundary. A caller that only derives identity calls
    :meth:`TracedClass.release` immediately.
    """
    ordered = declared_class_rows(pipeline, spec, decl)
    declared = len(ordered)
    count = max(1, int(share_count))
    rows = ordered[max(0, int(share_index)) % count::count] if count > 1 \
        else ordered
    # pgw#1215 step 4: a class this pod already holds as a packed artifact is
    # dropped from the share BEFORE the export, so a retry pays neither the
    # trace nor the compile for it. `declared` is unchanged — it is the size
    # of the DECLARATION, and the pool's whole-set proof counts held classes
    # beside packed ones (`_assert_shares_whole(have=...)`). Filtering after
    # the shard rather than before it keeps `rows[i::K]` the same partition of
    # the same order, so a skipped class does not move its siblings between
    # children.
    have = {str(n) for n in have_classes}
    if have:
        rows = [row for row in rows if _decl.plan_entry_name(row[0]) not in have]
    # pgw#1132: the adapter-BEARING rows export from the LIFTED forward, and
    # arming it is this loop's job exactly as it is `mint_targets`' — the
    # callers of both (`boot_trace_child`, `mint_child`) arm the CONTAINER
    # half only, which is pgw#822 verbatim. Held only in the `finally` below,
    # the first adapter-bearing row of every bucket-bearing family refused and
    # the whole derivation died before a resolve was possible.
    _arm_branches(pipeline, int(spec.lora_bucket or 0))
    disarmed = False
    try:
        for plan, arm in rows:
            entry = _decl.plan_entry_name(plan)
            if arm is False and not disarmed:
                _disarm_branches(pipeline)
                disarmed = True
            # pgw#1087's `trace_for_key` row is emitted HERE — around the
            # export it measures — and never by the caller. A caller-side span
            # around a generator brackets the loop body, not the trace, which
            # is how a phase table ends up honest about its names and wrong
            # about its numbers.
            with boot_phases.span(
                boot_phases.PHASE_TRACE_FOR_KEY, function=entry,
            ) as span:
                row = _export_entry(pipeline, spec, plan, decl)
                try:
                    nodes = int(len(row.program.graph_module.graph.nodes))
                except Exception:  # noqa: BLE001 — never fails a trace
                    nodes = 0
                # pgw#1087's owed item: a class's trace cost is meaningless
                # without the graph size it paid for.
                span.note(f"nodes={nodes}")
            yield TracedClass(
                name=entry,
                block=keying_block(row.program, row.ingress, row.spec),
                nodes=nodes,
                program=row.program,
                declared=declared,
                timings=dict(row.timings or {}),
            )
    finally:
        if disarmed:
            _arm_branches(pipeline, int(spec.lora_bucket or 0))


def keying_block(
    program: Any, ingress: CallIngress, spec: ExportSpec,
) -> Dict[str, Any]:
    """Worker coordinates plus TCG's canonical exported graph interface.

    TCG derives graph witness, constant/literal identity, placement and range
    digest from ``program`` and the exact ``graph.pytree.ingress`` value. The
    worker carries only the declaration coordinates TCG cannot know.
    """
    return {
        "target": spec.target,
        "fork": [[str(n), v] for n, v in sorted(spec.fork)],
        "class_dims": [
            [str(n), int(v)] for n, v in sorted(spec.class_dims)],
        "graph": entry_graph_block(program, spec, ingress),
    }


def tcg_graph_class_spec(traced: TracedClass, export_spec: ExportSpec) -> Any:
    """Translate one worker export row into TCG's sole public declaration.

    Both the compile child and the boot-trace child call this function.  The
    former hands the result to ``Engine.compile``; the latter calls
    ``GraphClassSpec.declare`` while the exported program is still alive and
    memoizes only TCG's resulting class hash.  Keeping the translation here
    prevents boot lookup and mint from growing two worker-side descriptions of
    the same graph class.
    """
    from gen_worker._vendor.torchcg import GraphClassSpec

    if traced.program is None:
        raise ValueError(f"graph class {traced.name!r} carries no exported program")
    block = dict(traced.block or {})
    graph = block.get("graph")
    if not isinstance(graph, dict):
        raise ValueError(f"graph class {traced.name!r} carries no graph interface")
    try:
        fork = tuple((str(name), value) for name, value in block["fork"])
        class_dims = tuple(
            (str(name), int(value)) for name, value in block["class_dims"]
        )
        target = str(block["target"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"graph class {traced.name!r} carries incomplete coordinates: {exc}"
        ) from exc
    return GraphClassSpec(
        graph_class=str(traced.name),
        target=target,
        program=traced.program,
        graph=dict(graph),
        fork=fork,
        class_dims=class_dims,
        strict=bool(export_spec.strict),
        lora_bucket=int(export_spec.lora_bucket or 0),
    )


def _mint_phase_table(
    minted: Sequence[_MintedEntry],
    timings: Mapping[str, float],
    width: Optional[aot_compile_pool.PoolWidth] = None,
    pool_ledger: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """The per-mint phase table (#757's instrument-first deliverable): one
    readable record of where the mint's seconds went, per entry and in
    total. Compiler policy is owned and versioned by TCG; this worker records
    the TCG policy identity rather than restating mutable Inductor options.

    pgw#809 adds the ``pool`` block: the K this mint ran at AND the budget
    that chose it. Without it a mint's wall clock is uninterpretable —
    two mints of the same cell on two pods legitimately differ by 4x."""
    entries = {
        row.name: dict(row.timings) for row in minted}
    totals: Dict[str, float] = {
        "export_s": round(sum(
            float(row.timings.get("export_s") or 0) for row in minted), 2),
        "compile_s": round(sum(
            float(row.timings.get("compile_s") or 0) for row in minted), 2),
        "warm_s": round(sum(
            float(row.timings.get("warm_s") or 0) for row in minted), 2),
    }
    phase_totals: Dict[str, float] = {}
    overlay_totals: Dict[str, float] = {}
    for row in minted:
        for label, value in (row.timings.get("phases") or {}).items():
            phase_totals[label] = round(
                phase_totals.get(label, 0.0) + float(value), 3)
        # Kept OUT of `phases` on purpose (pgw#830): overlays nest inside
        # partition members, and summing them in was the original
        # attribution bug. Reported beside it, never inside it.
        for label, value in (row.timings.get("overlays") or {}).items():
            overlay_totals[label] = round(
                overlay_totals.get(label, 0.0) + float(value), 3)
    return {
        "v": 1,
        "n_entries": len(minted),
        "compiler_owner": "torchcg",
        "totals": {**totals, **{k: v for k, v in timings.items()}},
        "phases": phase_totals,
        "overlays": overlay_totals,
        "entries": entries,
        # pgw#842: the width's INPUTS and the pool's own ledger ride the same
        # block, because the two questions a slow mint raises — "why this K"
        # and "what did K buy" — are answered by different halves of it, and a
        # record that carries only the scalar K (as this one did through
        # attempts ten and eleven) can answer neither.
        "pool": {
            **(width.facts() if width is not None
               else {"entry_workers": 1, "binding": "serial"}),
            **dict(pool_ledger or {}),
        },
    }


MINT_PHASES_KIND = "aot_mint_phases"


def _entry_duration_s(timings: Mapping[str, Any]) -> float:
    """One graph-class entry's own compile cost: export + compile + warm.

    The package/declare/pack phases are per-MINT, not per-entry, so they belong
    to the roll-up only — summing the entry rows must never reproduce
    ``total_s`` or a reader would double-count.
    """
    return sum(
        float(timings.get(key) or 0.0)
        for key in ("export_s", "compile_s", "warm_s")
    )


def _emit_phase_event(spec: ExportSpec, table: Mapping[str, Any]) -> None:
    """The mint's own emission (see :func:`emit_phase_events`).

    The spec is read inside the guard: telemetry must never fail the compile
    it measures, and reading the lane label off a broken spec is still
    telemetry.
    """
    try:
        family, execution_lane = spec.family, spec.execution_lane_label() or "plain"
    except Exception:  # pragma: no cover — telemetry never fails a mint
        logger.debug("aot-mint: phase event emission failed", exc_info=True)
        return
    emit_phase_events(family=family, execution_lane=execution_lane, table=table)


#: pgw#842: the mint's WIDTH decision, as its own hub row.
POOL_PHASE = "pool"


def _emit_pool_event(
    *, family: str, execution_lane: str, table: Mapping[str, Any],
) -> None:
    """pgw#842: one event that says what K was, what chose it, and what it
    bought — the standing "no silent decisions" rule applied to the mint's
    only multiplicative lever.

    Attempts ten and eleven compiled the same 72-entry sdxl cell for the same
    seconds (1314.94 vs 1327.23) and took 347.94 s vs 554.78 s, because K was
    5 and then 3. Nothing hub-side recorded WHY: the width block existed in
    the phase table and was never emitted, and the pgw#830 pool ledger was
    emitted from the mint CHILD, which holds no orchestrator session (see
    ``mint_supervisor._emit_aot_phases``) — so both were pod-log-only and died
    with the pod. A width narrower than the pod could carry is a performance
    defect; it must be READABLE from one mint's record, not inferred by
    diffing two pods that no longer exist.
    """
    pool = dict(table.get("pool") or {})
    if not pool:
        return

    workers = int(pool.get("entry_workers") or 1)
    binding = str(pool.get("binding") or "unknown")
    under = int(pool.get("underwidth") or 0)
    wall_s = float(pool.get("pool_wall_s") or 0.0)
    head = (
        f"family={family} lane={execution_lane} entry_workers={workers} "
        f"binding={binding} underwidth={under}")
    if under > 0:
        # Named in the FIRST line, so a narrow pool is legible without
        # parsing the dict: this is the number that cost attempt eleven 59 %.
        head += (
            f" — the pool ran {under} worker(s) narrower than this cell could "
            f"use, held by {binding}")
    activity_mod.emit_event(
        MINT_PHASES_KIND, f"{head} pool={pool}",
        phase=POOL_PHASE,
        duration_ms=int(round(wall_s * 1000)),
    )


def emit_phase_events(
    *, family: str, execution_lane: str, table: Mapping[str, Any],
    terminus: str = "",
) -> None:
    """Typed telemetry event — the phase table must reach observability,
    never only a pod log.

    th#1322: the mint's TOTAL now rides the numeric ``duration_ms`` field, and
    each graph-class entry gets its own event under ``phase=entry:<name>``. The
    interpolated table stays in ``detail`` as the breakdown you read per event
    (exactly as ``stage_ms`` stays in the request payload), but every number a
    measurement lane groups, percentiles or trends on is now a column.

    Paul's question — "why is AOT mint so much slower than JIT mint?" — needs
    the per-entry rows: an AOT mint compiles N graph classes where a JIT mint
    compiles the graphs one real warm plan happens to trace, and the answer is
    which entries the extra minutes are in, not just that there are more.
    """
    try:

        totals = dict(table.get("totals") or {})
        execution_lane = execution_lane or "plain"
        total_s = float(totals.get("total_s") or 0.0)
        # pgw#825: the roll-up's PHASE is the mint's terminus. An aborted mint
        # measured real entries and must report them — under `aborted`, never
        # under `minted`, or a partial table would enter an AOT-vs-JIT
        # comparison as if a cell came out.
        roll_up = terminus or str(table.get("terminus") or "") \
            or activity_mod.PHASE_MINTED
        activity_mod.emit_event(
            MINT_PHASES_KIND,
            f"family={family} lane={execution_lane} status={roll_up} "
            f"n_entries={table.get('n_entries')} totals={totals} "
            f"phases={dict(table.get('phases') or {})} "
            f"overlays={dict(table.get('overlays') or {})} "
            f"autotune={dict(table.get('autotune') or {})}",
            phase=roll_up,
            duration_ms=int(round(total_s * 1000)),
        )
        _emit_pool_event(family=family, execution_lane=execution_lane, table=table)
        for name, timings in sorted((table.get("entries") or {}).items()):
            if not isinstance(timings, Mapping):
                continue
            entry_s = _entry_duration_s(timings)
            if entry_s <= 0:
                continue
            activity_mod.emit_event(
                MINT_PHASES_KIND,
                f"family={family} lane={execution_lane} entry={name} "
                f"timings={dict(timings)}",
                phase=f"entry:{name}",
                duration_ms=int(round(entry_s * 1000)),
            )
    except Exception:  # pragma: no cover — telemetry must never fail a mint
        logger.debug("aot-mint: phase event emission failed", exc_info=True)


def entry_graph_block(
    program: Any, spec: ExportSpec, ingress: CallIngress,
) -> Dict[str, Any]:
    """The worker-owned declaration inside TCG's graph-class identity.

    The exported program itself is the sole authority for graph witness,
    constants, literals and placement. TCG derives those facts directly when
    it constructs :class:`GraphClassSpec`; repeating any of them here would
    create two copies of one identity. The worker contributes only its frozen
    branch declaration and TCG's canonical call-ingress value.
    """
    return {
        "v": 3,
        "lifted_inputs": sorted(str(n) for n in spec.lifted_inputs),
        "pytree": _pytree_facts(program, ingress),
        "specialization": _specialization_facts(spec),
    }




def _pytree_facts(program: Any, ingress: CallIngress) -> Dict[str, Any]:
    """The flattened call spec the package expects.

    An AOTI package takes FLAT tensors while the pipeline calls the module with
    diffusers' nested kwargs (``added_cond_kwargs={...}``). The consumer must
    flatten exactly as export did or it feeds the wrong tensor to the wrong
    input, so the spec is declared rather than re-derived at serve time.
    """
    signature = getattr(program, "graph_signature", None)
    call_spec = getattr(program, "call_spec", None)
    return {
        "user_inputs": [
            str(n) for n in getattr(signature, "user_inputs", ()) or ()],
        "in_spec": _treespec_text(getattr(call_spec, "in_spec", None)),
        "out_spec": _treespec_text(getattr(call_spec, "out_spec", None)),
        "ingress": ingress.as_dict(),
    }


def _treespec_text(spec: Any) -> str:
    if spec is None:
        return ""
    try:
        import torch.utils._pytree as pytree

        return str(pytree.treespec_dumps(spec))
    except Exception:
        # A spec we cannot canonically serialize is recorded by repr rather than
        # dropped: an unrecordable spec must still differ from a DIFFERENT
        # unrecordable spec in the key.
        return repr(spec)








def _input_names(
    module: Any, args: Tuple[Any, ...], kwargs: Mapping[str, Any],
) -> Tuple[str, ...]:
    """The target's forward parameter names, in call order.

    Positional args are named from the signature so the ``dynamic_shapes`` dict
    form can key on the same names whether a caller passed a tensor
    positionally or by keyword.
    """

    forward = getattr(module, "forward", module)
    try:
        params = [
            p.name for p in inspect.signature(forward).parameters.values()
            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD,
                          p.KEYWORD_ONLY)
        ]
    except (TypeError, ValueError) as exc:
        raise MintRefused(
            f"cannot read the forward signature of {type(module).__name__}: "
            f"{exc}") from exc
    positional = params[:len(args)]
    keyword = [name for name in kwargs if name not in positional]
    return tuple(positional) + tuple(keyword)


def _specialization_facts(spec: ExportSpec) -> Dict[str, Any]:
    """The frozen-branch declaration, with the lane facts always present.

    The lane and bucket belong here as well as in the key axes: the
    declaration is what a human reads off a rejected cell, and "which lane was
    this traced under" is the first question.
    """
    facts: Dict[str, Any] = dict(spec.specialization)
    facts.setdefault("weight_lane", str(spec.weight_lane or ""))
    facts.setdefault("lora_bucket", int(spec.lora_bucket or 0))
    facts.setdefault("strict", bool(spec.strict))
    return facts


def lifted_input_gaps(
    program: Any, spec: ExportSpec,
    containers: Optional[Mapping[str, int]] = None,
) -> List[str]:
    """Named reasons the declared lifted inputs are not actually graph inputs.

    #725 option 2's guarantee is structural: the adapter cannot be baked
    because it is an INPUT. If export did not lift it, the guarantee is gone
    and the G1 constant-table check would pass on absence — the "missing FQN
    means the branch was constant-folded, the same bug in a different hat"
    case. So the presence of every declared lifted input is proven here, on
    the program, before a single second of AOTI compile is spent.

    Names resolve through :func:`exported_input_names` (pgw#993), so a lifted
    input declared as a ``repeat=`` container is looked up under the flattened
    names export actually emits rather than refusing a sound program.
    """
    if not spec.lifted_inputs:
        return []
    signature = getattr(program, "graph_signature", None)
    user_inputs = {str(n) for n in getattr(signature, "user_inputs", ()) or ()}
    gaps: List[str] = []
    for declared in spec.lifted_inputs:
        for name in exported_input_names(str(declared), containers):
            if name not in user_inputs:
                gaps.append(
                    f"declared lifted input {name!r} is not a user input of "
                    f"the exported program (inputs: {sorted(user_inputs)!r}) "
                    f"— the adapter would not be swappable (#725 option 2)"
                )
    return gaps


class _CallableTarget:
    """Adapts a non-``forward`` target (``vae.decode``) to a module for export.

    ``torch.export`` traces ``forward``; a target like ``vae.decode`` is a bound
    method on a module whose ``forward`` does something else. Wrapping keeps the
    owner's parameters reachable (so they lift as constants normally) while
    presenting the declared callable as the traced entrypoint.

    **The wrapper COPIES the bound method's signature (ie#566 G1b).** A bare
    ``forward(*args, **kwargs)`` erases the parameter names, and
    :func:`_input_names` reads exactly those names to build the
    ``dynamic_shapes`` dict — so an erased signature silently yields zero
    bindable inputs and EVERY declared dynamic dim on a dotted target fails to
    bind. Signature preservation is the whole reason this class can carry a
    dynamic contract at all.
    """

    def __init__(self, owner: Any, attr: str) -> None:

        import torch.nn as nn

        self._owner = owner
        self._attr = str(attr)
        bound = getattr(owner, self._attr)

        class _Wrapper(nn.Module):
            def __init__(self, owner: Any, attr: str) -> None:
                super().__init__()
                self.owner = owner
                self._attr = attr

            def forward(self, *args: Any, **kwargs: Any) -> Any:
                return getattr(self.owner, self._attr)(*args, **kwargs)

        # Per-instance class, so stamping cannot leak across targets. ``self``
        # must be prepended: the stamp lands on the UNBOUND function, and
        # ``inspect.signature`` of the BOUND ``forward`` strips its first
        # parameter — without the filler it would strip the target's real first
        # argument and shift every name by one (silently mis-binding every
        # declared dynamic dim, which is the ie#566 G1b bug in a subtler form).
        try:
            sig = inspect.signature(bound)
            _Wrapper.forward.__signature__ = sig.replace(  # type: ignore[attr-defined]
                parameters=[inspect.Parameter(
                    "self", inspect.Parameter.POSITIONAL_OR_KEYWORD)]
                + list(sig.parameters.values()))
        except (TypeError, ValueError) as exc:
            raise MintRefused(
                f"cannot read the signature of target {self._attr!r} on "
                f"{type(owner).__name__} ({exc}); without it no declared "
                f"dynamic dim could bind (ie#566 G1b)") from exc

        self._module = _Wrapper(owner, self._attr)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._module, name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._module(*args, **kwargs)
