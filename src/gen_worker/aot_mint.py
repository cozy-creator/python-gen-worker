"""The AOT mint — export + AOTInductor-package a family's WHOLE declared
class set as ONE multi-graph cell (pgw#704 GO, #723 mint path, #758 packaging).

    compose -> per declared class: torch.export.export -> aot_compile(code-only)
            -> per-entry gates -> package_aoti({entry: files}) -> pack -> publish

Paul's ruling (pgw#758): "generate and generate_turbo are separate functions,
they have separate graphs, but they are COMBINED TOGETHER INTO ONE FILE." One
mint invocation produces one cell per (family x lane x contract) carrying every
declared graph class as a NAMED ENTRY — which removes the one-artifact-per-pod
serving ceiling the pilot runbook accepted.

``aot_serve`` owns the ENVELOPE — metadata contract, ``pack``, ``verify`` (#721
S1 / #723 S1: ONE source of truth, imported by both lanes, never re-declared)
— and consumes the result. ``aot_package`` reads facts back out of a compiled
``.pt2`` (per entry) and holds the B1 gate. ``lora_lifted`` owns the
no-baked-adapter gate. This module drives PRODUCTION and nothing else.
Deliberately NOT folded into ``compile_cache``:
``trt_engine`` already established that a compiled-lane backend is its own
module riding the compile-cache rails, and the dynamo mint stays live and
fully-forced in parallel during rollout (#722: nothing retires before sdxl AOT
is live in prod).

The mint does not judge the lane (pgw#879, pgw#850)
---------------------------------------------------
Paul's ruling: *"here is the code, here is the lane, please compile this for
all graphs we have declared. That's it."* The lane is an INPUT — chosen once,
by the hub's resolution tree, and observed on the composed pipeline by
``loading.pipeline_weight_lane``. This module compiles what it is handed.

The refusal that used to live here (``lane_admitted`` / ``PARITY_LANES``, a
one-member allowlist holding ``w8a8``) is deleted because it was a SECOND
opinion about an already-resolved fact, and the two opinions composed into a
total block: tensorhub's lane table makes ``fp8-w8a8-dynamic``
compiled-only, so the hub withholds that lane from AUTO until a cell exists
(th#1123/th#1127), and only a pod already serving the lane can mint one. The
one lane the mint admitted was the one lane no pod could ever be on, and every
lane a pod COULD be on (``bf16-w16a16``, ``fp8-w8a16``, ``svdq-*-w4a4``) was
refused — quoting a 6.9-7.0% AOTI regression pgw#704 Q4 measured on sdxl's
lanes at families and lanes it never measured. Measured result: zero fleet
families reached the mint gate on AUTO, on any card (pgw#850).

The pgw#704 Q4 / #730 measurement is not discarded — it is a RANKING input to
the lane/execution choice (``+compiled`` vs ``+eager``), which lives in the
hub. A lane held on dynamo is simply never asked for a cell; if one IS asked
for, the mint compiles it.

Where minting runs (#724 REJECTED — Paul, 2026-07-28)
-----------------------------------------------------
Serving pods background-mint their own cells under the proven pgw#677
eager-first machinery — "I'd rather keep that, rather than a whole complex
separate compilation system; our compilation system would only ever just be
running the endpoint code we have already anyway." There is no dedicated mint
fleet. ``python -m gen_worker.aot_mint`` stays CLI-invokable for ops and
testing. Mint cost is INSTRUMENTED, not assumed: every mint records a
per-phase, per-entry ``mint_phases`` table (export / lowering / codegen /
triton / host C++ compile+link) plus the graph-class count and the autotune
posture, so an AOT-vs-JIT comparison is labeled data, never folklore (#757
consumes it).

What is exported
----------------
STRICT export, by default. pgw#704 confirmed the sdxl UNet passes strict AND
non-strict, static AND dynamic, so strict is chosen for what it buys: it
refuses python side effects instead of silently baking them, which is exactly
the failure class an artifact serving live traffic must not have. Non-strict
stays available per-mint for a family that needs it (z-image's ragged list
inputs, #729) and is RECORDED in the key, because a non-strict trace is a
different guarantee and must not share identity with a strict one.

The two gates that make an artifact publishable
-----------------------------------------------
**B1 code-only.** Compiled with ``package_constants_in_so=False``, then PROVEN
so by ``aot_package.code_only_violations`` — two independent structural proofs,
failing red and naming the tensors. See that module for why weights in a cell
would destroy the CAS distribution model.

**No baked adapter.** ``lora_lifted.assert_no_baked_adapter`` runs at pack time
on the **ExportedProgram**, not on the package: packing renames a
plain-``__dict__`` adapter to ``_tensor_constant0``, so a package-side FQN scan
is a false PASS on the plain-bf16 and fp8-hooks lanes (pgw#725, measured). The
ep-level gate is also free — it needs no packing at all.

Both gates run before pack, so a violating artifact never becomes a file that a
later step could upload.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any, Callable, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple)

from . import activity as activity_mod
from . import (
    aot_compile_pool, aot_export_parallel, aot_package,
    aot_serve, aot_wrapper_split, cell_key, graph_hash, kernel_path)
from .aot_contract import (  # re-exported: the declaration layer's vocabulary
    ADAPTER_FORK,
    DynamicDim,
    ExportSpec,
    MintRefused,
)
from .aot_preconditions import LIFTED_LORA_TORCH_FLOOR, torch_version_gap
from .compile_cache import (
    _resolve_target,
    toolchain_present,
)
from dataclasses import replace
import inspect
from .models.memory import is_cuda_oom
from . import aot_flatten
from . import aot_inputs
from . import aot_declaration as _decl
from .api.export_contract import export_declaration
from .models import lora_lifted
from . import compile_cache as cc
from . import env_seal
from . import config, worker_credential
from .fleet_cells import CellPublisher

logger = logging.getLogger(__name__)

#: The inductor config that makes the package code-only. Not a knob: B1.
CODE_ONLY_CONFIGS: Dict[str, Any] = {
    "aot_inductor.package_constants_in_so": False,
}


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

        if torch.cuda.is_available():
            peak = int(torch.cuda.max_memory_allocated())
    except Exception:  # noqa: BLE001 — a probe never changes an outcome
        peak = 0
    raise MintResourceExhausted(
        f"{where}: OUT OF DEVICE MEMORY ({type(exc).__name__}: {exc}). "
        f"This process peaked at {peak / (1 << 30):.2f} GiB against the cap "
        f"the parent set from `mint_budget.co_residency`; it is a resource "
        f"shortfall to be retried with more room, NOT a deterministic "
        f"refusal", peak_rss_bytes=0) from exc


# ---------------------------------------------------------------------------
# The declared export contract
# ---------------------------------------------------------------------------


def lifted_torch_gap(spec: ExportSpec) -> str:
    """'' when torch meets the lifted-LoRA floor (or the spec has no lifted
    fork declared), else the named refusal reason.

    pgw#996: the floor arithmetic lives in ``aot_preconditions`` because the
    BUILD gate asks the same question of the same image — a second spelling
    here is how a build proves one thing and a pod discovers another. This
    wrapper survives for the mint-request CLI, which can be handed a spec the
    build gate never saw.
    """
    if not (spec.lora_bucket or spec.lifted_inputs or spec.lora_fqns):
        return ""
    import torch

    return torch_version_gap(str(getattr(torch, "__version__", "") or ""))


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
    comes from ``aot_flatten.exported_name`` — the ONE naming rule the ingress
    contract and the serve-side bind also read (pgw#994). A declared container
    is the leaf path ``(0,), (1,), …`` of its parameter, so this function is a
    special case of that rule rather than a second copy of it.
    """
    arity = (containers or {}).get(name)
    if arity is None:
        return (aot_flatten.exported_name(name),)
    return tuple(aot_flatten.exported_name(name, (index,))
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
    mint. `.module()` is excluded from the timing because the current path
    pays it too (`compile_entry_files`), so the comparison stays like-for-like.
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
       or narrowed range is a pin, and the artifact admits less traffic than it
       advertises;
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
                            f"[{d.min}, {d.max}] — the artifact admits less "
                            f"traffic than it advertises")
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
                        f"admits less traffic than it advertises")
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
            entries = list(entries)
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
            if text in seen:
                continue
            seen.add(text)
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


#: Mint-path default worker count for inductor's parallel compile (#757,
#: MEASURED): 32 -> 4 is FREE (-2% wall clock) and is the recommended
#: default for background mints on serving pods — same speed, less CPU
#: contention with live serving. NOT seal-relevant: compile_threads is
#: outside cell identity per #757's re-key pre-verification, so this
#: default (and a caller override) never re-keys a cell. A caller value
#: wins; this is a default, not a clamp.
MINT_COMPILE_THREADS = 4


def _entry_configs(inductor_configs: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """The per-entry inductor config: caller options + the non-negotiable
    packaging flags. ``CODE_ONLY_CONFIGS`` is applied LAST so no caller-
    supplied config can re-enable constant baking — B1 is a fleet
    correctness requirement, not a default a caller may override. One cell's
    entries ALL compile under this one dict (a per-entry config drift would
    be an identity fact nothing records), and the resolved dict is recorded
    in the mint-phase telemetry."""
    configs: Dict[str, Any] = dict(inductor_configs or {})
    configs.setdefault("compile_threads", MINT_COMPILE_THREADS)
    overridden = sorted(set(configs) & set(CODE_ONLY_CONFIGS))
    if overridden:
        logger.warning(
            "aot-mint: ignoring caller inductor config %s — code-only is B1, "
            "not a knob", overridden)
    configs.update(CODE_ONLY_CONFIGS)
    # Emit loose files for package_aoti to combine, instead of a per-entry
    # archive: the multi-graph cell is ONE .pt2 (pgw#758).
    configs["aot_inductor.package"] = True
    return configs


def compile_entry_files(
    program: Any,
    entry: str,
    *,
    inductor_configs: Optional[Mapping[str, Any]] = None,
) -> List[Any]:
    """AOTI-compile one exported program into CODE-ONLY loose files.

    This is ``aoti_compile_and_package``'s own internal compile step
    (verified on the pin: ``aot_compile(ep.module(check_guards=False),
    *ep.example_inputs, options)``), deferred before packaging so N entries
    combine into one archive. Compilation is byte-identical to the
    single-model mint; only the packaging changes.

    pgw#793: the host C++ compile is 46% of an AOTI compile and is ONE g++
    invocation on ONE translation unit whose largest function is inductor's
    constants_info_ table spelled as 26k straight-line statements.
    :func:`aot_wrapper_split.install` regroups exactly that run of
    statements before g++ sees it — same statements, same order, verified by
    reconstruction, declining unmodified on any wrapper shape it does not
    recognise. It changes no compiler, flag, inductor config or library, so
    no cell is re-keyed.
    """
    from torch._inductor import aot_compile

    aot_wrapper_split.install()
    gm = program.module(check_guards=False)
    args, kwargs = program.example_inputs
    try:
        files = aot_compile(
            gm, tuple(args), dict(kwargs or {}),
            options=_entry_configs(inductor_configs))
    except Exception as exc:
        raise MintRefused(
            f"entry {entry!r}: aot_compile failed: "
            f"{type(exc).__name__}: {exc}") from exc
    if not isinstance(files, list):
        raise MintRefused(
            f"entry {entry!r}: aot_compile returned {type(files).__name__}, "
            f"not the loose-file list packaging needs "
            f"(aot_inductor.package was forced True)")
    return files


def package_cell(
    files_by_entry: Mapping[str, Sequence[Any]], package_path: Path,
) -> Path:
    """Combine every entry's compiled files into ONE ``.pt2`` of named
    models (``data/aotinductor/<entry>/`` each) — the pgw#758 cell."""
    from torch._inductor.package import package_aoti

    package_path = Path(package_path)
    package_path.parent.mkdir(parents=True, exist_ok=True)
    if not files_by_entry:
        raise MintRefused("cannot package a cell with no entries")
    try:
        out = package_aoti(
            str(package_path),
            {str(name): list(files) for name, files in files_by_entry.items()})
    except Exception as exc:
        raise MintRefused(
            f"package_aoti failed: {type(exc).__name__}: {exc}") from exc
    return Path(str(out))


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

#: pgw#1006. NOT a member of ``_PHASE_KEYS`` — these nest INSIDE ``codegen_s``
#: (``autotune_at_compile_time`` resolves True for AOTI, so the autotune block
#: runs during codegen), exactly as ``triton_s`` does. Named because it answers
#: two questions that were being read out of a residual: the whole ceiling of a
#: shared autotune cache, and whether the selected config moved between two
#: mints of one key — it is baked into the generated wrapper's grid expression
#: and ``num_warps``. Same keys as ``aot_compile_spans.OVERLAY_KEYS``.
_AUTOTUNE_KEYS: Tuple[str, ...] = (
    "CachingAutotuner.benchmark_all_configs",
    "CachingAutotuner.coordinate_descent_tuning",
    "CachingAutotuner.combo_sequential_autotune",
)


def _phase_snapshot() -> Dict[str, float]:
    try:
        import torch._dynamo.utils as du

        return {
            str(k): float(sum(v))
            for k, v in du.compilation_time_metrics.items()}
    except Exception:
        return {}


def _phase_delta(
    before: Mapping[str, float], after: Mapping[str, float],
) -> Dict[str, float]:
    """Named phase seconds spent between two snapshots. ``triton_s`` sums
    every async-compile/triton key so GPU kernel compilation is one
    labeled number; ``autotune_s`` (pgw#1006) does the same for the
    compile-time autotune benchmark. Both NEST inside the named phases above
    them and must not be summed with them. The remainder of inductor time is
    NOT invented — the coarse wall clocks around export/compile hold the
    totals."""
    raw = {
        k: round(float(after.get(k, 0.0)) - float(before.get(k, 0.0)), 3)
        for k in set(after) | set(before)
    }
    out: Dict[str, float] = {}
    for label, keys in _PHASE_KEYS.items():
        value = round(sum(raw.get(k, 0.0) for k in keys), 3)
        if value:
            out[label] = value
    triton = round(sum(
        v for k, v in raw.items()
        if ("async_compile" in k or "triton" in k.lower()) and v > 0), 3)
    if triton:
        out["triton_s"] = triton
    autotune = round(sum(raw.get(k, 0.0) for k in _AUTOTUNE_KEYS), 3)
    if autotune > 0:
        out["autotune_s"] = autotune
    return out


def autotune_posture(
    inductor_configs: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """The benchmark-driven kernel-selection posture of THIS mint —
    recorded so an AOT-vs-JIT cost comparison can rule the asymmetry in or
    out (#757: it would dominate everything else)."""
    posture: Dict[str, Any] = {}
    try:
        import torch._inductor.config as inductor_config

        for name in ("max_autotune", "max_autotune_pointwise",
                     "max_autotune_gemm", "search_autotune_cache"):
            if hasattr(inductor_config, name):
                posture[name] = bool(getattr(inductor_config, name))
    except Exception:
        pass
    for key, value in (inductor_configs or {}).items():
        if "autotune" in str(key):
            posture[f"override.{key}"] = value
    return posture


# ---------------------------------------------------------------------------
# The mint
# ---------------------------------------------------------------------------


@dataclass
class MintResult:
    """A packed, gated, publishable artifact plus its mint telemetry."""

    artifact: Path
    metadata: Dict[str, Any]
    timings: Dict[str, float]

    @property
    def cell_key(self) -> str:
        return str(self.metadata.get("cell_key") or "")


@dataclass
class _MintedEntry:
    """One exported+compiled graph class, pre-packaging."""

    name: str
    spec: ExportSpec
    module: Any
    owner: Any
    program: Any
    input_names: Tuple[str, ...]
    flat_leaves: Tuple[aot_flatten.Leaf, ...]
    files: List[Any]
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


def _export_entry(
    pipeline: Any,
    spec: ExportSpec,
    plan: Any,
    decl: Any,
    *,
    inductor_configs: Optional[Mapping[str, Any]] = None,
    compile_now: bool = True,
    rows: int = 0,
) -> _MintedEntry:
    """Resolve, feed, (warm,) export, gate, and compile ONE declared graph
    class. Every refusal is prefixed with the entry name — a multi-graph
    mint that cannot say WHICH class failed is the silent-failure path in
    a new hat (pgw#758).

    ``compile_now=False`` stops after the export-side gates and returns the
    entry with no files: pgw#809's pool then compiles every entry K-wide out
    of process. Export must stay here and stay SERIAL — it runs against the
    one live pipeline, on the one card, inside the one branch-arm toggle."""

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
    flat_leaves = flat_input_leaves(module, args, kwargs)
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
    program = _full_export()
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

    # pgw#825: BEFORE the compile, not after it. The packed bindability gate
    # runs on the artifact and is the proof of what shipped; this one asks the
    # same question of the exported program, where it costs milliseconds
    # instead of an entry's whole compile.
    unbindable = aot_package.unbindable_program_constants(
        program, _state_dict_keys(owner))
    if unbindable:
        raise MintRefused(
            f"entry {entry!r}: pre-compile bindability gate: "
            + "; ".join(unbindable))

    files: List[Any] = []
    if compile_now:
        before = _phase_snapshot()
        t0 = time.monotonic()
        files = compile_entry_files(
            program, entry, inductor_configs=inductor_configs)
        timings["compile_s"] = round(time.monotonic() - t0, 2)
        phases = _phase_delta(before, _phase_snapshot())
        if phases:
            timings["phases"] = phases

    return _MintedEntry(
        name=entry, spec=espec, module=module, owner=owner, program=program,
        input_names=input_names, flat_leaves=flat_leaves, files=files,
        timings=timings)


def bench_step(pipeline: Any, spec: ExportSpec) -> Callable[[], Any]:
    """A zero-argument callable running ONE forward of this family's DOMINANT
    declared graph class, in the PRODUCTION (compiled) posture (pgw#947).

    The dominant class is the declaration's FIRST target — the denoiser,
    which runs once per step while the VAE runs once per image — so "ms/step"
    means what the pgw#862/#863 benchmark tables mean by it. A cell-level spec
    that names a target (the operator CLI path) overrides that. Its inputs are
    the family's OWN declared example feed (``aot_inputs.builder_for``), i.e.
    the same representative shape the mint is about to export against, not a
    shape invented here.

    COMPILED, deliberately: pgw#863's whole finding is that the eager ranking
    and the compiled ranking disagree — inductor fuses the baseline lane's
    open elementwise chain and cannot fuse across our custom ops — and the
    compiled posture is the only one production serves from.
    """
    import torch

    decl = export_declaration(spec.family)
    if decl is None:
        raise MintRefused(
            f"family {spec.family!r} has no export declaration — nothing to "
            f"benchmark a kernel lane against")
    plans = list(_decl.cell_plans(decl))
    if not plans:
        raise MintRefused(f"family {spec.family!r} declares no graph classes")
    want = str(spec.target or "") or str(decl.targets[0])
    dominant = next(
        (p for p in plans if str(p.target) == want), plans[0])
    espec = _entry_spec(spec, dominant, decl)
    resolved = _resolve_target(pipeline, espec.target)
    if resolved is None:
        raise MintRefused(
            f"pipeline {type(pipeline).__name__} has no compile target "
            f"{espec.target!r} to benchmark")
    owner, attr, _fn = resolved
    module = owner if attr == "forward" else _CallableTarget(owner, attr)
    builder = aot_inputs.builder_for(espec.family, espec.target)
    args, kwargs = builder(owner, espec)
    compiled = torch.compile(module)

    def step() -> Any:
        with torch.no_grad():
            return compiled(*args, **kwargs)

    return step


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

    Adapter-bearing rows come FIRST so the composed pipeline, which arrives
    lifted from :func:`aot_inputs.compose`, is disarmed exactly once.
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
    convention gets skipped. Idempotent, so a caller that already composed
    lifted (:func:`aot_inputs.compose`) pays nothing.

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

    inductor_configs: Optional[Mapping[str, Any]] = None
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


def mint(
    pipeline: Any,
    spec: ExportSpec,
    out_dir: Path,
    *,
    inductor_configs: Optional[Mapping[str, Any]] = None,
    entry_workers: int = 0,
    entry_peak_rss_bytes: int = 0,
    entry_device_peak_bytes: int = 0,
    on_progress: Optional[Callable[[str, int, int, str], None]] = None,
    phase_snapshot: Optional[Path] = None,
    execution_lane_verdict: Optional[kernel_path.Verdict] = None,
) -> MintResult:
    """:func:`_mint_cell`, with the phase table attached to EVERY terminus.

    pgw#825: an aborted mint used to report a wall-clock total and nothing
    else — `compile_s`, `export_s` and `n_entries` all parsed to `-` — so a
    run that paid for real compiles and then refused produced no measurement
    at all, and the pod it happened on is gone by the time anyone reads the
    table. The live entry list and timings dict are handed down and mutated
    in place, so whatever completed before the terminus is reportable from
    here whether the mint returned, refused, or died.

    pgw#824 rides the SAME record (:class:`MintProgress`) rather than a second
    one: ``on_progress(phase, step, total, note)`` is where those in-flight
    positions are pushed out live, and the last one pushed is what an aborted
    table reports as ``at``.
    """
    # pgw#929: say the podguard state ONCE, before any compile. `invalid` means
    # this pod's progress signal goes nowhere, and a mint that looks
    # unprogressing to the watchdog gets reaped mid-compile — the failure the
    # `_touch_pod_progress` beats below exist to prevent. Reporting it only when
    # a beat fires would announce the problem after the expensive part has
    # started; reporting it here makes a broken handoff visible at boot.
    report_podguard_status()
    progress = MintProgress(
        inductor_configs=inductor_configs, on_progress=on_progress)
    if phase_snapshot is not None:
        # pgw#848: every beat re-writes the on-disk table, so a mint that is
        # KILLED still leaves 29 minutes of measurement behind it. Wrapped
        # around the caller's sink rather than replacing it: both are
        # best-effort and neither may cost a mint.
        inner = progress.on_progress
        snap = Path(phase_snapshot)

        def _beat(phase: str, step: int, total: int, note: str) -> None:
            try:
                write_phase_snapshot(snap, progress)
                # pgw#848: the SAME beat tells the pod-side reaper this mint
                # is progressing. It has to be a CHANGING token, so it carries
                # the position — which is the honest signal anyway.
                _touch_pod_progress(f"aot_mint {phase} {step}/{total} {note}")
            except Exception:  # noqa: BLE001 — telemetry never fails a mint
                logger.debug("aot-mint: phase snapshot failed", exc_info=True)
            if inner is not None:
                inner(phase, step, total, note)

        progress.on_progress = _beat
    try:
        return _mint_cell(
            pipeline, spec, out_dir,
            inductor_configs=inductor_configs,
            entry_workers=entry_workers,
            entry_peak_rss_bytes=entry_peak_rss_bytes,
            entry_device_peak_bytes=entry_device_peak_bytes,
            execution_lane_verdict=execution_lane_verdict,
            progress=progress)
    except BaseException as exc:
        _attach_partial_phases(exc, progress)
        raise


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
        minted, timings, progress.inductor_configs, progress.width,
        progress.pool_ledger)
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
    state dir, and this call is therefore a no-op there — which is most pods,
    and will include th#1359 forge pods. So this makes lane-rented pods
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


def report_podguard_status() -> str:
    """Log the podguard state once, so a broken handoff is visible at boot."""
    status = podguard_status()
    if status == PODGUARD_INVALID:
        logger.warning(
            "podguard=invalid: %s=%r is set but is not a usable absolute "
            "writable directory. This pod's mint progress signal goes nowhere "
            "and the watchdog will read it as unprogressing (pgw#929).",
            PODGUARD_STATE_ENV, os.environ.get(PODGUARD_STATE_ENV, ""))
    else:
        logger.info("podguard=%s", status)
    return status


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


def _attach_partial_phases(exc: BaseException, progress: MintProgress) -> None:
    """Hang the partial phase table off a failed mint's exception.

    Telemetry never changes an outcome, so every step is guarded: a mint that
    refuses must refuse with ITS sentence, not with a reporting error.
    """
    try:
        table = partial_phase_table(progress)
        if table:
            setattr(exc, "mint_phases", table)
    except Exception:  # pragma: no cover — telemetry never fails a mint
        logger.debug("aot-mint: partial phase table unavailable", exc_info=True)


class _ExportFootprint:
    """What ONE export costs a card, as distinct from what the whole export
    PHASE costs one.

    pgw#1000 — the measurement that kept export parallelism dark. The width
    rule needs "how much device memory does one export worker hold"; the only
    figure ever recorded was ``max_memory_allocated`` over the ENTIRE serial
    loop, taken against a resident pipeline:

        export_peak_device_bytes  16,558,897,664   (15.4 GiB, sdxl, attempt 26)

    That is a cumulative high-water across 36 rows *including* the resident
    module — so dividing free VRAM by it answers "how many whole mint children
    fit", which is 1, forever. ``aot_export_parallel.width_for`` then returned
    1 with ``binding='export-footprint-unmeasured'`` and the feature could
    never turn on. The number was not missing; it was the wrong number, and
    nothing said so.

    A row's DELTA over the resident baseline is the right one: it is what a
    worker adds to a card that already holds the module it traces. Export runs
    on fake tensors and launches no kernel, so it should be small — but
    "should be" is the claim this class exists to replace, and the fallback
    when it cannot be read is still a refusal to widen.

    Reported both ways, because they answer different questions and conflating
    them is the defect: ``per_export_device_bytes`` (the MAX row delta — sizes
    a pool) and ``export_peak_device_bytes`` (the phase high-water — sizes the
    mint child itself, and is what pgw#992's CardCensus already prices).
    """

    __slots__ = ("baseline", "rows", "readable", "census")

    def __init__(self, baseline: int, readable: bool,
                 census: Optional[Any] = None) -> None:
        self.baseline = baseline
        self.rows: List[int] = []
        self.readable = readable
        #: pgw#992's card census, taken HERE — before the first row traces.
        #: The budget it feeds is `total - co-tenant - own high-water`, and
        #: that only bounds anything if the census predates the growth it is
        #: meant to price. Taken at `decide()` time instead it would read the
        #: mint child's own grown footprint as the baseline and hand the
        #: export pool back everything the phase had already consumed.
        self.census = census

    @classmethod
    def open(cls) -> "_ExportFootprint":
        from . import aot_compile_pool

        census = aot_compile_pool.card_census()
        try:
            import torch as _t

            if not _t.cuda.is_available():
                return cls(0, False, census)
            return cls(int(_t.cuda.memory_reserved()), True, census)
        except Exception:  # noqa: BLE001 — a probe never changes an outcome
            return cls(0, False, census)

    @contextlib.contextmanager
    def row(self) -> Iterator[None]:
        """Measure ONE row's export. Never raises: the row's own exception
        propagates untouched and its measurement is simply absent."""
        if not self.readable:
            yield
            return
        try:
            import torch as _t

            _t.cuda.reset_peak_memory_stats()
            before = int(_t.cuda.memory_reserved())
        except Exception:  # noqa: BLE001
            yield
            return
        try:
            yield
        finally:
            try:
                import torch as _t2

                peak = int(_t2.cuda.max_memory_reserved())
                # Against the row's OWN entry level, not the phase baseline:
                # rows that leave allocations behind (a cached graph, a lifted
                # constant) would otherwise charge every later row for them.
                self.rows.append(max(0, peak - before))
            except Exception:  # noqa: BLE001
                pass

    def facts(self) -> Dict[str, float]:
        """Flat scalars for ``timings``. Absent keys where nothing was read —
        a measured zero and an unread card are different facts, and the width
        rule must be able to tell them apart."""
        if not self.readable or not self.rows:
            return {}
        return {
            "per_export_device_bytes": float(max(self.rows)),
            "per_export_device_rows": float(len(self.rows)),
            "export_resident_baseline_bytes": float(self.baseline),
        }


def _mint_cell(
    pipeline: Any,
    spec: ExportSpec,
    out_dir: Path,
    *,
    inductor_configs: Optional[Mapping[str, Any]] = None,
    entry_workers: int = 0,
    entry_peak_rss_bytes: int = 0,
    entry_device_peak_bytes: int = 0,
    execution_lane_verdict: Optional[kernel_path.Verdict] = None,
    progress: Optional[MintProgress] = None,
) -> MintResult:
    """Export + compile EVERY declared graph class and pack them as ONE
    multi-graph cell (pgw#758).

    ``lane_verdict`` (pgw#947) is the MEASURED serving-kernel lane for this
    card, produced by the mint driver before the pipeline was loaded — only
    the loader can swap the linears, so the A/B happens one level up
    (``mint_child.lane_verdict_for``). The discrete verdict is packed into
    the envelope so serving reads it instead of an SM tuple; the numbers ride
    the result metadata beside ``mint_phases``, never the artifact.

    ``entry_workers`` CAPS pgw#809's compile pool; it never widens it. The
    width is derived from the pod (see :func:`aot_compile_pool.entry_workers`),
    and this argument exists so an operator or a test can force the serial
    path (``1``) without pretending a 4-vCPU pod is an H100 host.

    ``progress`` (:class:`MintProgress`) is the mint's own position record —
    handed down by :func:`mint`, mutated in place, and read back by
    :func:`_attach_partial_phases` on any terminus. ``progress.beat()``
    reports progress WITHIN the mint (pgw#824). This function used to be one
    opaque call spanning the family's whole declared class set — sdxl declares
    18 — so the mint child framed ``trace_graph`` once and said nothing again
    until ``seal_publish``. A real export measured ~5 minutes of complete wire
    silence, and the pod's only liveness evidence was that its CPU was warm.
    "Entry 12 of 18" is the difference between a pod that is alive and a pod
    that is working, which is exactly the distinction the no-magic-timeouts
    doctrine runs on.

    Does NOT publish — :func:`publish` is a separate step so a mint can be
    inspected, byte-compared (#699 double-mint), or produced on a box with no
    hub credentials.
    """

    refusal = lifted_torch_gap(spec)
    if refusal:
        raise MintRefused(refusal)
    decl = export_declaration(spec.family)
    if decl is None:
        raise MintRefused(
            f"family {spec.family!r} has no registered export declaration — "
            f"a multi-graph cell derives its class set from the declaration "
            f"(pgw#739/#758); register one before minting")
    if decl.warm_changes_key is None:
        raise MintRefused(
            f"family {spec.family!r} declares no mint-warm canon "
            f"(warm_changes_key) — whether pre-warm changes the graph is a "
            f"measured per-family FACT (sdxl False, z-image True), not a "
            f"default")

    out_dir = Path(out_dir)
    work = out_dir / "work"
    work.mkdir(parents=True, exist_ok=True)
    # Handed to the caller BY REFERENCE and mutated in place: an aborted mint
    # reports the seconds it did spend (pgw#825) and the position it was on
    # (pgw#824), out of ONE record.
    progress = MintProgress() if progress is None else progress
    timings = progress.timings
    t_mint = time.monotonic()
    progress.t_mint = t_mint

    # pgw#846 (Paul's ruling): the exported cell is always WHOLE-GRAPH.
    # Regional (block-class) export is retired for production — a
    # `Compile(regional=True)` declaration keeps its dynamo/JIT meaning
    # (ie#381, compile_cache) and the AOT mint ignores it.
    rows = adapter_arm_plans(_decl.cell_plans(decl), pipeline, spec)
    # pgw#809: how wide this pod may compile. Derived from the pod's REAL
    # budget (cgroup-aware vCPUs minus serving headroom, and available host
    # RAM over the measured per-entry peak) — never os.cpu_count, never a
    # constant. K=1 IS the pre-#809 serial in-process path, which is the
    # honest answer on a narrow pod.
    entry_count = len(rows)
    # pgw#877: the DEVICE ask now has the same shape the HOST ask got in
    # pgw#848 — a measurement made on this pod by a previous mint of this
    # (family, lane), banked by the serving parent and handed down on the
    # request. 0 keeps the estimate, and keeps saying so.
    device_bytes, device_basis = _entry_device_bytes(
        spec, int(entry_device_peak_bytes or 0))
    width = aot_compile_pool.entry_workers(
        entry_count, limit=int(entry_workers or 0),
        device_bytes=device_bytes, device_basis=device_basis,
        # pgw#848: the HOST ask, measured on this pod by a previous mint of
        # this (family, lane) and banked by the serving parent. Until this
        # existed the argument was never passed at ALL, so `mem_workers`
        # divided available RAM by a 3 GiB constant on every mint the fleet
        # has run and `per_entry_rss_basis` said "default" forever. 0 keeps
        # the constant, and keeps saying so.
        peak_rss_bytes=int(entry_peak_rss_bytes or 0))
    parallel = width.workers > 1
    logger.info("aot-mint: entry compile width — %s", width.reason)
    if width.underwidth:
        # pgw#842: a pool narrower than the cell could use is a COST, and it
        # is the mint's only multiplicative lever. Say so at WARNING with the
        # readings behind it — the same facts ride the `pool` event.
        logger.warning(
            "aot-mint: pgw#842 entry pool runs %d worker(s) narrower than "
            "this cell could use (K=%d of %d), held by %s — inputs %s",
            width.underwidth, width.workers,
            min(entry_count, width.ceiling), width.binding, width.facts())
    progress.width = width

    minted = progress.minted
    disarmed = False
    # pgw#822: the adapter-BEARING classes are exported from the lifted
    # forward. Arming it is this function's job, not the caller's — see
    # `_arm_branches`.
    _arm_branches(pipeline, int(spec.lora_bucket or 0))
    t_export = time.monotonic()
    # pgw#868 A4: the EXPORT phase's own device high-water, which nothing has
    # ever measured. `aot_compile_child` resets and samples around the INDUCTOR
    # compile (`peak_device_bytes` in `EntryReport`), so that number is the
    # compile's; export runs HERE, serially, in the parent, and was never
    # sampled at all. The two are different questions and must not share a
    # figure: export traces with FAKE tensors and executes no kernel, so the
    # compile pool's `weights * 1.25 + 5 GiB` — whose activation and workspace
    # terms are INDUCTOR'S, and ~56 % of which was never observed — does not
    # describe it. That is exactly why `aot_export_parallel.width_for()`
    # returns 1 while this is unknown rather than guessing and OOMing a
    # 74-minute phase. A probe: it reads and clears a counter and decides
    # nothing (pgw#830 — instrument first, optimise never in the same change).
    try:
        import torch as _t
        if _t.cuda.is_available():
            _t.cuda.reset_peak_memory_stats()
    except Exception:  # noqa: BLE001 — a probe never changes an outcome
        pass
    # pgw#1000: the RESIDENT baseline, read before the first row traces. Every
    # per-row figure below is a DELTA over this, because what a hypothetical
    # export worker adds to a card is what it allocates on top of the module
    # it holds — not the module plus everything the parent already had.
    export_footprint = _ExportFootprint.open()
    progress.beat(
        PHASE_TRACE_GRAPH, 0, len(rows),
        f"{len(rows)} declared class row(s)")
    try:
        for index, (plan, arm) in enumerate(rows, start=1):
            if arm is False and not disarmed:
                # ONE toggle for the whole branchless group (the rows are
                # ordered adapter-bearing first): disable/enable reallocates
                # every leaf's branch container, and doing it per entry would
                # be N times the VRAM churn for the same graphs.
                _disarm_branches(pipeline)
                disarmed = True
            # Reported BEFORE the work, not after: a row that never returns
            # is the one a reader most needs named, and an after-the-fact tick
            # names only the rows that finished.
            progress.beat(
                PHASE_TRACE_GRAPH, index, len(rows),
                _decl.plan_entry_name(plan))
            with export_footprint.row():
                minted.append(_export_entry(
                    pipeline, spec, plan, decl,
                    inductor_configs=inductor_configs,
                    compile_now=not parallel, rows=len(rows)))
    finally:
        if disarmed:
            _arm_branches(pipeline, int(spec.lora_bucket or 0))
    # Asked of the EXPORTED programs: a cell whose entries cannot be told
    # apart at dispatch must cost seconds to refuse, not a full compile bill
    # (the pgw#825 discipline, one gate over). Exact on the parallel path,
    # which is every real mint — pgw#809 sizes width off the pod's own budget
    # and the pool has not built a kernel yet. A width-1 serial mint has
    # already compiled as it exported, so there it refuses late; correct
    # either way, cheap where it matters.
    #
    # pgw#917: and it MERGES before it refuses. Declared rows that reduce to
    # one ingress contract over one target with byte-identical code are one
    # dispatchable class — compiling each of them separately buys nothing and
    # makes the cell undispatchable, which is the same fact twice.
    minted, class_aliases = canonicalize_dispatch_classes(minted)
    timings["canonicalized_entries"] = float(
        sum(len(rows) for rows in class_aliases.values()))

    if parallel:
        timings["export_all_s"] = round(time.monotonic() - t_export, 2)
        # Sampled BEFORE the pool is built, so no inductor allocation can be
        # attributed to export. Reported even when zero (no CUDA / probe
        # failed), because a missing key and a measured zero are different
        # facts and the width rule must be able to tell them apart.
        try:
            import torch as _t
            if _t.cuda.is_available():
                timings["export_peak_device_bytes"] = float(
                    _t.cuda.max_memory_allocated())
                timings["export_peak_device_reserved_bytes"] = float(
                    _t.cuda.max_memory_reserved())
        except Exception:  # noqa: BLE001 — a probe never changes an outcome
            pass
        # pgw#1000: and the PER-ROW figure beside it. The two are different
        # questions — phase high-water sizes the mint child, one row's delta
        # sizes a pool — and the width rule spent its whole life dividing by
        # the first one, which is why it always answered 1.
        timings.update(export_footprint.facts())
        # pgw#868 A4: THE CONNECTION. The probe above is exactly
        # `aot_export_parallel.width_for(per_export_device_bytes=)`. Both were
        # built and neither ever called the other, so the flag was inert.
        # Recorded on EVERY mint, flag or no flag: the DECISION is the
        # observable, and a reader must be able to see what width export would
        # have run at — and which fact bound it — from a mint that changed
        # nothing.
        try:
            timings.update(aot_export_parallel.decide(
                rows, timings, census=export_footprint.census))
        except Exception:  # noqa: BLE001 — telemetry never fails a mint
            logger.debug("aot-mint: export-parallel decision failed",
                         exc_info=True)
        progress.beat(
            PHASE_INDUCTOR_COMPILE, 0, len(minted),
            f"{len(minted)} entries, {width.workers} wide")
        progress.pool_ledger = _compile_entries_parallel(
            minted, work, width, inductor_configs=inductor_configs,
            progress=progress,
            on_entry=lambda name, done, total: progress.beat(
                PHASE_INDUCTOR_COMPILE, done, total, name))
        # NOTE: `_compile_entries_parallel` refreshes `progress.pool_ledger`
        # on every completed entry (pgw#848), so the snapshot each beat writes
        # already carries a LIVE ledger — K, its binding, efficiency, peaks —
        # rather than only the width. An abandoned mint's row is the one that
        # needs it most.
    timings["entry_workers"] = float(width.workers)

    t0 = time.monotonic()
    progress.beat(
        PHASE_SEAL_PUBLISH, len(minted), len(minted),
        f"packaging {len(minted)} entries")
    package = package_cell(
        {row.name: row.files for row in minted}, work / aot_serve.PACKAGE_NAME)
    timings["package_s"] = round(time.monotonic() - t0, 2)

    t0 = time.monotonic()
    entry_blocks: Dict[str, Dict[str, Any]] = {}
    for row in minted:
        entry_blocks[row.name] = _gate_and_declare_entry(row, package)
        # pgw#917: the declared-class names this entry absorbed. Recorded so
        # the merge is auditable from the envelope alone — a reader asking
        # "where did class row X go" gets an answer instead of an absence.
        # NOT a `class_hash` fact (see `aot_serve.class_hash`, which folds
        # named fields only): an alias declares no traffic the surviving
        # entry's own contract does not already declare, so it must not
        # re-key an otherwise identical cell.
        merged = class_aliases.get(row.name) or ()
        if merged:
            entry_blocks[row.name]["aliases"] = [
                {"name": alias.name,
                 "class_dims": [
                     [str(n), int(v)] for n, v in sorted(alias.spec.class_dims)]}
                for alias in sorted(merged, key=lambda r: r.name)
            ]
    _write_literals(minted, package, work)

    try:
        meta = aot_serve.artifact_metadata(
            family=spec.family,
            precision=spec.precision,
            cell_key="",
            entries=entry_blocks,
            strict_export=bool(spec.strict),
            lora_bucket=int(spec.lora_bucket or 0),
            source_ref=spec.source_ref,
            source_digest=spec.source_digest,
        )
    except ValueError as exc:
        # The envelope validates the contract it is handed. A malformed one must
        # fail HERE, on the mint pod, not at serve time on a paying request.
        raise MintRefused(
            f"envelope refused the declared contract: {exc}") from exc
    meta.update(shared_identity_blocks(spec))
    if execution_lane_verdict is not None:
        # pgw#947: the DISCRETE verdict only. Milliseconds in metadata.json
        # would break the #699 double-mint byte-compare — the artifact
        # deliberately carries no wall clocks — and the margin threshold is
        # what makes the discrete answer reproducible across two mints.
        meta[kernel_path.META_KEY] = kernel_path.envelope_block(execution_lane_verdict)
    mode_drift = aot_package.strict_mode_drift(meta, spec.strict)
    if mode_drift:
        raise MintRefused("trace-mode drift: " + "; ".join(mode_drift))

    timings["declare_s"] = round(time.monotonic() - t0, 2)
    timings["total_s"] = round(time.monotonic() - t_mint, 2)
    phase_table = _mint_phase_table(
        minted, timings, inductor_configs, width, progress.pool_ledger)
    _emit_phase_event(spec, phase_table)

    meta["cell_key"] = key = cell_identity(meta, spec).digest
    t0 = time.monotonic()
    artifact = aot_serve.pack(work, out_dir / f"{key}.tar.gz", meta)
    timings["pack_s"] = round(time.monotonic() - t0, 2)
    # The phase table rides the RESULT (and the published checkpoint
    # metadata + the typed event), never the packed envelope: durations in
    # metadata.json would break the #699 double-mint byte-compare — the
    # artifact deliberately carries no timestamps and no wall clocks.
    meta["mint_phases"] = phase_table
    if execution_lane_verdict is not None:
        # The EVIDENCE: both lanes' ms/step and peak bytes, the margin, the
        # headroom terms, and the device it was all measured on. Same channel
        # as the phase table (published checkpoint metadata + the typed
        # event), so a verdict is auditable long after the pod is gone.
        meta[kernel_path.EVIDENCE_KEY] = kernel_path.evidence_block(
            execution_lane_verdict)

    logger.info(
        "aot-mint: %s lane=%s -> %s (%d entr%s across %d target(s), %.1f MB "
        "package, combined=%s, %s)",
        spec.family, spec.execution_lane_label() or "(plain)", key,
        len(minted), "y" if len(minted) == 1 else "ies",
        len({row.spec.target for row in minted}),
        package.stat().st_size / 1e6, meta.get("combined_graph_hash"),
        timings,
    )
    return MintResult(artifact=artifact, metadata=meta, timings=timings)


def _entry_device_bytes(
    spec: ExportSpec, banked_device_peak: int = 0,
) -> Tuple[int, str]:
    """One entry child's DEVICE ask, and the PROVENANCE of that number.

    pgw#877 #1/#2. Two sources, ranked, because they are not the same kind of
    thing:

    * ``banked_device_peak`` — what an entry child on THIS pod, for this
      (family, lane), was actually measured to peak at
      (``EntryReport.peak_device_reserved_bytes``), banked by the serving
      parent and handed down on ``MintRequest.entry_device_peak_bytes``. It
      travels on the WIRE and not through ``mint_budget``'s module globals,
      which is the entire fix: those globals are written in the serving parent
      and this function runs in the MINT CHILD, where they are empty by
      construction. Basis ``"measured"``.
    * ``mint_budget.co_residency().need_bytes`` — the fallback, and an
      ESTIMATE of a different process: the mint child's whole co-residency
      footprint (a full pipeline), used as one entry child's (one exported
      program plus inductor). ~56 % of it was never observed. Basis
      ``"estimated"``, which is what it used to call ``"measured"``.

    ``(0, "unmeasured")`` means unprobeable, and the width policy refuses to
    license concurrency on a card it cannot size against.
    """
    if banked_device_peak > 0:
        from . import mint_budget

        return mint_budget.entry_device_ask(int(banked_device_peak)), "measured"
    try:
        from . import mint_budget

        budget = mint_budget.co_residency(
            family=str(spec.family or ""),
            weight_lane=str(spec.execution_lane_label() or ""))
    except Exception:  # noqa: BLE001
        return 0, "unmeasured"
    if not budget.probed:
        return 0, "unmeasured"
    return int(budget.need_bytes), "estimated"


def _compile_entries_parallel(
    minted: List[_MintedEntry],
    work: Path,
    width: aot_compile_pool.PoolWidth,
    *,
    inductor_configs: Optional[Mapping[str, Any]] = None,
    on_entry: Optional[Callable[[str, int, int], None]] = None,
    progress: Optional["MintProgress"] = None,
) -> Dict[str, Any]:
    """pgw#809: fill every entry's ``files`` K-wide, out of process.

    Returns the pool's own ledger (pgw#830) so it reaches the phase table:
    the pool emits it as a typed event too, but that emission happens in the
    mint CHILD, which holds no orchestrator session — pgw#842.

    Mutates ``minted`` in place, and every entry MUST come back with files —
    a pool that quietly returned fewer entries than it was given would pack a
    short cell. Assembly is by entry NAME: ``package_cell`` reads
    ``{row.name: row.files}`` in the order ``minted`` already holds (the
    declaration's order), so completion order is not observable in the
    artifact.
    """
    # A SIBLING of work/, never inside it. pack() only copies a fixed member
    # set so debris there would be harmless today, but a pool workdir living
    # inside the directory that becomes the artifact is one refactor away from
    # putting job files and stderr tails into a cell.
    pool = aot_compile_pool.EntryCompilePool(
        work.parent / "entry-pool", width=width,
        inductor_configs=inductor_configs)
    t0 = time.monotonic()

    def _tick(name: str, done: int, total: int) -> None:
        # pgw#848: refresh the ledger BEFORE the beat, so the snapshot the
        # beat writes carries this entry's numbers. A mint killed at entry 30
        # of 36 then leaves 30 entries' worth of measurement on disk instead
        # of one bare "no cell produced" row.
        if progress is not None:
            progress.pool_ledger = _pool_facts(pool)
        if on_entry is not None:
            on_entry(name, done, total)

    try:
        by_entry = pool.compile(
            [(row.name, row.program) for row in minted], on_entry=_tick)
    except aot_compile_pool.EntryCompileFailed as exc:
        # pgw#848: the pool's ledger and its MEASURED peak have to survive the
        # failure, because the aborted phase table is what the parent banks
        # and re-sizes K from. Without this the OOM'd attempt teaches the
        # retry nothing and attempt 2 runs the identical width.
        if progress is not None:
            progress.pool_ledger = _pool_facts(pool)
        # Named, and the siblings are already torn down group-wide by the
        # pool. A mint that says only "a compile failed" over 18 entries is
        # the silent-failure path in a new hat (pgw#758).
        if exc.resource:
            raise MintResourceExhausted(
                str(exc), entry=exc.entry, basis=exc.basis,
                peak_rss_bytes=exc.peak_rss_bytes) from exc
        raise MintRefused(str(exc)) from exc
    wall = time.monotonic() - t0
    missing = [row.name for row in minted if row.name not in by_entry]
    if missing:
        raise MintRefused(
            f"entry compile pool returned {len(by_entry)} of {len(minted)} "
            f"entries — missing {missing!r}. Packing the rest would ship a "
            f"cell whose declared class set is a lie")
    for row in minted:
        row.files = list(by_entry[row.name])
        row.timings["compile_s"] = pool.entry_seconds.get(row.name, 0.0)
        # Measured in the child; folded in here so the roll-up reads the same
        # whether a cell was minted serially or K-wide.
        phases = pool.entry_phases.get(row.name) or {}
        if phases:
            row.timings["phases"] = dict(phases)
        # pgw#842: the overlays travel too. `child_seal_s` is a partition
        # member and its SPLIT is an overlay — and the split is the whole
        # answer to "what is the seal still costing": pgw#832 cut the library
        # hash to ~0.07 s (measured), while the child's `import torch`, which
        # `establish_config` owns, is the rest. Without the overlay a reader
        # sees only the sum and re-opens a closed question.
        overlays = pool.entry_overlays.get(row.name) or {}
        if overlays:
            row.timings["overlays"] = dict(overlays)
    logger.info(
        "aot-mint: pgw#809 pool compiled %d entr%s at K=%d in %.0fs "
        "(sum of entry seconds %.0fs, peak child RSS %.1f GiB)",
        len(minted), "y" if len(minted) == 1 else "ies", width.workers, wall,
        sum(pool.entry_seconds.values()), pool.peak_rss_bytes / 1024**3)
    return _pool_facts(pool)


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


def _entry_ingress_declaration(
    row: "_MintedEntry",
) -> Tuple[Any, Tuple[Dict[str, Any], ...], Dict[str, Any]]:
    """``(contract, representative calls, declaration meta)`` for one entry.

    The contract is built from ``aot_package.input_contract`` — the exact rows
    the packed cell will carry — and read back through
    :func:`aot_serve.contract_from_meta`, the serve path's OWN parser, so the
    gate cannot drift from what dispatch will actually do.

    The calls are that entry's own declared shapes: one per corner of its
    symbol hull (all symbols at their lower bound, all at their upper, all at
    the midpoint), deduplicated. A fully specialized entry — the sdxl case —
    yields exactly one call, which is the call its class row exists to serve.
    """
    inputs, symbols = aot_package.input_contract(row.program, row.flat_leaves)
    meta: Dict[str, Any] = {"inputs": inputs, "symbols": symbols}
    if adapter_arm(row.spec.fork) is False:
        # The NEGATIVE half of a branchless class's contract, exactly as
        # `_gate_and_declare_entry` will pack it (pgw#790). Without it here the
        # gate would ask a question the serve path never asks.

        meta["excluded_inputs"] = list(lora_lifted.LIFTED_INPUT_NAMES)
    contract = aot_serve.contract_from_meta(meta)

    def _at(pick: Callable[[int, int], int]) -> Dict[str, Any]:
        return {
            spec.name: _DeclaredArg(
                tuple(
                    dim if isinstance(dim, int)
                    else pick(*contract.symbols[dim])
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
    return contract, tuple(calls), meta


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


def _class_identity(
    row: "_MintedEntry", declaration: Mapping[str, Any],
) -> Dict[str, Any]:
    """Every axis except the class-row COORDINATE that two declared classes
    must share to be one logical class (pgw#917).

    The entry key and the ingress contract have to be the same object. When
    two rows collide at ingress the only question left is whether they are the
    same *thing* — same code, same target, same compatibility metadata — and
    each key here is one axis a human can be told about by name when they are
    not.

    ``graph_hash`` is the canonical form of the EXPORTED program (nodes,
    signature and declared ranges), so "byte-identical code" is asked of the
    artifact rather than assumed from the declaration. Read before a single
    kernel is compiled, which is the whole pgw#847 win: four rows that reduce
    to one dispatchable shape must cost one compile, not four.
    """
    return {
        "target": str(row.spec.target),
        "fork": [[str(n), v] for n, v in sorted(row.spec.fork)],
        "graph": graph_hash.graph_hash(row.program),
        "ingress": aot_serve.range_digest(declaration),
        "pytree": _pytree_facts(row.program),
        "literal_values": aot_package.literal_values_digest(row.program),
        "specialization": _specialization_facts(row.spec),
        "lifted_inputs": sorted(str(n) for n in row.spec.lifted_inputs),
        "precision": str(row.spec.precision or ""),
        "lora_bucket": int(row.spec.lora_bucket or 0),
        "strict": bool(row.spec.strict),
        "source_digest": str(row.spec.source_digest or ""),
    }


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


def canonicalize_dispatch_classes(
    minted: Sequence["_MintedEntry"],
) -> Tuple[List["_MintedEntry"], Dict[str, Tuple["_MintedEntry", ...]]]:
    """Collapse declared classes that are ONE dispatchable entry; refuse the
    ones that are not (pgw#917).

    :meth:`aot_serve.EntryDispatch.select` calls two entries admitting one
    call ``entry_ambiguous`` — "a declaration that cannot discriminate two
    graph classes by ingress, which is a defect to surface, never a coin to
    flip". It is a per-REQUEST refusal, so a cell with a colliding declaration
    arms, reports armed, and serves those coordinates 100 % eager: 4,200
    refused calls across gen-worker 0.89.0/0.90.0 on the standing stack, every
    single one ``entry_ambiguous``, zero of any other phase.

    **The collision is arithmetic, not a race.** sdxl's aspect rows at one
    megapixel bucket are area-preserving — 112x144 = 144x112 = 168x96 =
    96x168 = 16,128 — and a ``BasicTransformerBlock`` never sees ``H_lat`` and
    ``W_lat``, only the flattened sequence ``(B, H_lat*W_lat, C)``. The
    declaration keys entries on the pair; the ingress contract can only
    observe the product. So ambiguity is GUARANTEED for every area-preserving
    aspect family at a fixed bucket, which is exactly how the fleet's shape
    rows are generated.

    **Merge, don't only refuse.** Four rows that produce one ingress contract
    over one target with byte-identical code are one logical class: mint ONE
    entry and keep the declared-class names as aliases (36 of the regional
    shape's 72 compiles bought nothing — the direct pgw#847 shape-invariant
    win). Refusal is reserved for a collision whose members are NOT the same
    thing, and then it names the colliding pair AND the differing axis, which
    a bare "these two clash" never could.

    Grouped exactly the way the serve path groups — target, adapter arm —
    because those are the axes dispatch resolves BEFORE ingress, and two
    entries on different arms are meant to differ only by the lifted pair.

    Returns ``(entries to compile and package, aliases by surviving entry)``.
    """
    groups: Dict[Tuple[str, Any], List["_MintedEntry"]] = {}
    for row in sorted(minted, key=lambda r: r.name):
        fork = {str(n): v for n, v in tuple(row.spec.fork)}
        key = (str(row.spec.target), fork.get(ADAPTER_FORK))
        groups.setdefault(key, []).append(row)

    dropped: Dict[str, "_MintedEntry"] = {}
    aliases: Dict[str, Tuple["_MintedEntry", ...]] = {}
    conflicts: List[str] = []
    for rows in groups.values():
        if len(rows) < 2:
            continue
        by_name = {row.name: row for row in rows}
        declared: Dict[
            str, Tuple[Any, Tuple[Dict[str, Any], ...], Dict[str, Any]]] = {}
        for row in rows:
            try:
                declared[row.name] = _entry_ingress_declaration(row)
            except (aot_package.PackageIntrospectionError, ValueError) as exc:
                # An unreadable declaration is not "probably fine": it is a
                # cell whose dispatchability nobody can prove.
                raise MintRefused(
                    f"entry {row.name!r}: dispatch-ambiguity gate cannot read "
                    f"the declared ingress contract, so this cell cannot be "
                    f"shown to be dispatchable at all: {exc}") from exc
        # Union the mutual-admission relation into clusters. Asked through
        # `aot_serve.assert_ingress` itself, on the contract shape the pod
        # parses, so the gate cannot drift from what dispatch will do.
        cluster_of: Dict[str, str] = {name: name for name in declared}

        def _root(name: str) -> str:
            while cluster_of[name] != name:
                cluster_of[name] = cluster_of[cluster_of[name]]
                name = cluster_of[name]
            return name

        for name, (_own, calls, _m) in declared.items():
            for other, (contract, _c, _dm) in declared.items():
                if other == name or not any(
                    _admits(contract, call) for call in calls
                ):
                    continue
                a, b = _root(name), _root(other)
                if a != b:
                    cluster_of[min(a, b)] = min(a, b)
                    cluster_of[max(a, b)] = min(a, b)
        clusters: Dict[str, List[str]] = {}
        for name in sorted(declared):
            clusters.setdefault(_root(name), []).append(name)

        for members in clusters.values():
            if len(members) < 2:
                continue
            identities = {
                name: _class_identity(by_name[name], declared[name][2])
                for name in members
            }
            axes = _differing_axes(identities)
            if axes:
                conflicts.append(
                    f"{sorted(members)[:4]!r} collide at ingress but are NOT "
                    f"one class — they differ on {list(axes)!r}")
                continue
            keep, *rest = sorted(members)
            aliases[keep] = tuple(by_name[name] for name in rest)
            for name in rest:
                dropped[name] = by_name[name]

    if conflicts:
        raise MintRefused(
            f"dispatch-ambiguity gate: {len(conflicts)} cluster(s) of declared "
            f"class rows are admitted by more than one entry of the same "
            f"dispatch, so every call they carry would be refused "
            f"'entry_ambiguous' and served EAGER — "
            + "; ".join(conflicts[:4]) + ". Rows that reduce to ONE "
            "dispatchable ingress contract are one entry and are merged "
            "automatically; these cannot be, because the named axes say they "
            "are different artifacts. Fix the declaration so every entry's "
            "ingress contract is uniquely admitting, rather than compiling "
            "and publishing a class the cell can never select")

    if dropped:
        for keep, merged_rows in sorted(aliases.items()):
            logger.info(
                "aot-mint: pgw#917 canonicalized %d declared class row(s) onto "
                "entry %r — identical ingress contract, target and code, so "
                "they are ONE dispatchable class: %s",
                len(merged_rows), keep, [r.name for r in merged_rows])
        activity_mod.emit_event(
            "aot_class_canonicalized",
            f"{len(dropped)} of {len(minted)} declared class rows reduce to an "
            f"ingress contract a sibling already declares; merged onto "
            f"{len(aliases)} entry/entries as aliases instead of compiling a "
            f"class the dispatch could never select: "
            + "; ".join(
                f"{keep} <- {[r.name for r in merged_rows]}"
                for keep, merged_rows in sorted(aliases.items())[:4]),
            phase="entry_merged",
        )
    return [row for row in minted if row.name not in dropped], aliases


def _gate_and_declare_entry(
    row: _MintedEntry, package: Path,
) -> Dict[str, Any]:
    """Run every package-side gate for one entry and build its envelope
    block. Refusals name the entry AND the cause (pgw#758)."""
    entry = row.name
    violations = aot_package.code_only_violations(package, entry)
    if violations:
        raise MintRefused(
            f"entry {entry!r}: code-only gate (pgw#704 B1): "
            + "; ".join(violations))
    unbindable = aot_package.unbindable_constants(
        package, _state_dict_keys(row.owner), entry)
    if unbindable:
        raise MintRefused(
            f"entry {entry!r}: bindability gate: " + "; ".join(unbindable))
    # pgw#728: strict and non-strict traces lift DIFFERENT constant sets, so the
    # manifest must be proven to describe the package that ships beside it. Two
    # independent derivations (program vs generated wrapper) required to agree —
    # drift the env seal cannot see, because both modes run identically sealed.
    drift = aot_package.program_package_drift(row.program, package, entry)
    if drift:
        raise MintRefused(
            f"entry {entry!r}: constant-set drift: " + "; ".join(drift))
    fused = aot_package.eliminated_constants(row.program, package, entry)
    if fused:
        # Routine compiler fusion (measured on real sdxl: conv_out.bias folded
        # into the conv epilogue). Recorded, never fatal — but a surprising jump
        # in the count should be visible rather than silently discarded.
        logger.info(
            "aot-mint: %s: %d lifted constant(s) fused away by the compiler "
            "(e.g. %s)", entry, len(fused), fused[:3])
    try:
        inputs, symbols = aot_package.input_contract(
            row.program, row.flat_leaves)
        constants = aot_package.constants_manifest(package, entry)
    except aot_package.PackageIntrospectionError as exc:
        raise MintRefused(f"entry {entry!r}: declaration: {exc}") from exc
    block: Dict[str, Any] = {
        "target": row.spec.target,
        "fork": [[str(n), v] for n, v in sorted(row.spec.fork)],
        "class_dims": [
            [str(n), int(v)] for n, v in sorted(row.spec.class_dims)],
        "inputs": inputs,
        "symbols": symbols,
        "constants": constants,
        "graph": entry_graph_block(row.program, package, row.name, row.spec),
    }
    if adapter_arm(row.spec.fork) is False:
        # pgw#790: the NEGATIVE half of this class's contract. Without it the
        # branchless entry silently ADMITS an adapter-bearing call (a
        # name-keyed bind ignores inputs it does not declare), the dispatch
        # sees two admitting entries and refuses `entry_ambiguous` — and the
        # cell serves the whole attach lane eagerly. Declared, so the refusal
        # is the right one and it names the input.

        block["excluded_inputs"] = list(lora_lifted.LIFTED_INPUT_NAMES)
    return block


def _mint_phase_table(
    minted: Sequence[_MintedEntry],
    timings: Mapping[str, float],
    inductor_configs: Optional[Mapping[str, Any]],
    width: Optional[aot_compile_pool.PoolWidth] = None,
    pool_ledger: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """The per-mint phase table (#757's instrument-first deliverable): one
    readable record of where the mint's seconds went, per entry and in
    total, plus the two facts an AOT-vs-JIT comparison needs to be fair —
    the graph-class COUNT this mint compiled and the autotune posture.

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
    # The ONE resolved inductor config every entry compiled under, recorded
    # verbatim: #757's open seal-bypass concern is a per-call config the
    # seal cannot see, so whatever the mint passed is either identity-inert
    # (compile_threads, #757 pre-verified) or visible RIGHT HERE for the
    # audit that says otherwise.
    resolved = {
        key: value if isinstance(value, (bool, int, float, str, type(None)))
        else repr(value)
        for key, value in sorted(_entry_configs(inductor_configs).items())
    }
    return {
        "v": 1,
        "n_entries": len(minted),
        "autotune": autotune_posture(inductor_configs),
        "inductor_configs": resolved,
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
    ``mint_delegate._emit_aot_phases``) — so both were pod-log-only and died
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
    program: Any, package: Path, entry: str, spec: ExportSpec,
) -> Dict[str, Any]:
    """The per-entry graph-interface facts (fold into that entry's
    ``class_hash``): the declared constant FQN set, the lifted inputs, the
    pytree spec, and the python branches export FROZE at trace time.
    Constant BYTE SIZES are deliberately absent — they are a property of the
    resident weights, and a fine-tune of one family must keep sharing
    cells, which is the premise of family-scoped cells.

    pgw#857: that exclusion is right for a WEIGHT and wrong for a LITERAL, and
    both were excluded. A weight is rebound from the resident ``state_dict``
    at load, so two fine-tunes should share a cell. A literal ships INSIDE the
    artifact and is never rebound — *"nothing outside the artifact knows its
    value"* — so for a literal the VALUE IS THE ARTIFACT, and two checkpoints
    needing different literals were sharing a key. ``literal_values`` closes
    that and nothing else: state_dict-sourced constants are still keyed by
    NAME only, so fine-tune sharing is untouched.

    **Emitted ONLY when the program lifts a literal.** A family with none
    (sdxl: measured zero across five real mints) produces a byte-identical
    block and does not re-key — the discipline ``range_digest`` already uses
    for ``excluded``, and for the same reason: a field that says "unchanged"
    must not strand already-published cells."""
    block: Dict[str, Any] = {
        "v": 2,
        "constant_fqns": sorted(aot_package.constant_names(package, entry)),
        "fused_constants": sorted(
            aot_package.eliminated_constants(program, package, entry)),
        "lifted_inputs": sorted(str(n) for n in spec.lifted_inputs),
        "pytree": _pytree_facts(program),
        "specialization": _specialization_facts(spec),
    }
    literals = aot_package.literal_values_digest(program)
    if literals:
        block["literal_values"] = literals
    return block


def shared_identity_blocks(spec: ExportSpec) -> Dict[str, Any]:
    """The cell-level identity facts an exported cell must record.

    ``aot_serve.artifact_metadata`` takes ``cell_key`` as a STRING, so the
    envelope on its own would carry a stamp WITHOUT the axes the stamp
    summarizes — and ``cell_key``'s standing discipline is that a key is always
    recomputed FROM recorded facts, so a stamp can never disagree with them.
    These blocks are what make that recomputation possible for the new kind, and
    they ride the metadata additively (the envelope's parsers read named fields
    and are unaffected). Per-entry graph facts live in the ``entries`` blocks
    (:func:`entry_graph_block`) and reach the key through the combined hash.
    """

    return {
        "weight_lane": str(spec.weight_lane or ""),
        # pgw#846: an exported cell is always WHOLE-GRAPH (`mode` "").
        # `shell_digest` likewise: both keys are recorded at their whole-graph
        # values ("") so the v3 contract-facts shape — and therefore every
        # existing whole-graph cell key — is byte-identical to pre-#846.
        "mode": "",
        "shell_digest": "",
        "sm": str(cc.runtime_key().get("sm") or ""),
        env_seal.SEAL_KEY: env_seal.effective_seal(),
        "toolchain": dict(cc.toolchain_digest()),
        "code_closure": dict(cc.static_code_closure(tuple(spec.closure_roots))),
        "content_keys": dict(cc.content_keys()),
        "loaded_libs": dict(env_seal.frozen_library_digests()),
        "gen_worker": cc.gen_worker_version(),
    }


def _pytree_facts(program: Any) -> Dict[str, Any]:
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


def cell_identity(meta: Mapping[str, Any], spec: ExportSpec) -> cell_key.CellKey:
    """The cell key a multi-graph artifact's OWN recorded facts describe.

    Computed from the recorded blocks, never from separate probes, so the stamp
    can never disagree with the axes it summarizes — the discipline
    ``cell_key.from_artifact_metadata`` enforces for dynamo cells, mirrored for
    the new kind. ``cell_key.from_axes`` already accepts any ``kind`` VALUE (it
    validates axis NAMES), so no KEY_SCHEME bump: the axis set is unchanged and
    ``kind`` does the discriminating. No dynamo cell is stranded.

    The ``contract`` axis is the pgw#716 formula, IMPLEMENTED AS ANTICIPATED:
    the cell keys on the ``combined_graph_hash`` — first 16 hex of the sha256
    over the newline-joined SORTED per-class hashes — while the per-class
    hashes ride ``entries[*].class_hash`` so a mismatch NAMES the class. Each
    class hash folds that entry's ``range_digest`` (the #723 S3 requirement:
    three exports differing ONLY in declared range produced identical node-only
    digests) plus its coordinate and graph-interface block.

    CONTRACT-FACTS SHAPE CHANGE (v1 -> v2, pgw#758): this re-keys every
    published ``aot-inductor`` cell; single-graph format-1 cells are RETIRED —
    correct and expected under exact identity.

    CONTRACT-FACTS SHAPE CHANGE (v2 -> v3, pgw#817): ``shell_digest`` joined
    the facts for the (since-retired) regional kind. pgw#846 retires regional
    but deliberately KEEPS the v3 shape with ``shell_digest`` pinned ``""``
    and the ``mode`` axis ``""`` — the whole-graph key is byte-identical to
    what it was before and after pgw#817, so nothing re-keys.
    """

    sm = str(meta.get("sm") or "")
    if not sm:
        raise MintRefused(
            "cannot state the compute capability (sm) of this runtime; an "
            "exported cell has no identity without it — mint on the target GPU")
    entries = dict(meta.get("entries") or {})
    combined = str(meta.get("combined_graph_hash") or "")
    if not entries or not combined:
        raise MintRefused(
            "the envelope recorded no entries/combined_graph_hash; a "
            "multi-graph cell must not be keyed without its class set "
            "(pgw#716/#758)")
    unhashed = sorted(
        name for name, block in entries.items()
        if not str((block or {}).get("class_hash") or ""))
    if unhashed:
        raise MintRefused(
            f"entries {unhashed[:4]!r} carry no class_hash; a class the key "
            f"cannot name is a class a mismatch cannot name (pgw#716)")
    contract_facts: Dict[str, Any] = {
        "v": 3,
        "combined_graph_hash": combined,
        # pgw#846: always "" — kept in the v3 shape so the whole-graph
        # contract digest (and every derived cell identity) does not move.
        "shell_digest": "",
        "targets": sorted({
            str((block or {}).get("target") or "") for block in entries.values()}),
        "shapes": sorted([int(v) for v in row] for row in spec.shapes),
        "text_lens": sorted({int(v) for v in spec.text_lens}),
        "guidance": sorted(float(v) for v in spec.guidance_scales),
        "lora_bucket": int(spec.lora_bucket or 0),
        "strict": bool(spec.strict),
    }
    contract = cell_key.contract_digest(contract_facts)
    return cell_key.from_axes({
        "format": str(meta.get("format") or ""),
        "kind": aot_serve.ARTIFACT_KIND,
        "family": str(meta.get("family") or ""),
        "lane": spec.execution_lane_label(),
        # pgw#846: an exported cell is always whole-graph again; "" is the
        # optional-axis value `from_axes` omits, matching every pre-regional
        # whole-graph key.
        "mode": "",
        "sm": sm,
        "contract": contract,
        "env_seal": env_seal.seal_digest(dict(meta.get(env_seal.SEAL_KEY) or {})),
        "toolchain": cell_key.facts_digest(dict(meta.get("toolchain") or {})),
    })


def _state_dict_keys(module: Any) -> Tuple[str, ...]:
    """The names the SERVE arm will have to bind from — pgw#825.

    ``aot_serve.resident_constants``, never ``state_dict()``: the gate's whole
    value is that it predicts the arm, and the arm binds non-persistent
    buffers (the LoRA branch pair) that ``state_dict()`` does not report.
    """
    try:
        return tuple(str(k) for k in aot_serve.resident_constants(module))
    except Exception:
        return ()


def _write_literals(
    minted: Sequence[_MintedEntry], package: Path, content_dir: Path,
) -> None:
    """Pack the bytes of every entry's declared LITERAL constants beside the
    package, keys namespaced ``<entry>::<fqn>`` (pgw#758).

    A literal has no ``state_dict`` counterpart (folded scalars, sinusoidal
    tables, shape vectors), so a consumer cannot bind it from resident weights.
    Under B1 the ``.so`` does not carry it either — which is exactly the
    unbound-constant precondition for the worker-killing segfault. Shipping the
    bytes is therefore not an optimization; it is what makes a code-only
    artifact loadable at all.
    """
    tensors: Dict[str, Any] = {}
    missing: List[str] = []
    for row in minted:
        literals = aot_package.literal_constants(package, row.name)
        if not literals:
            continue
        values = dict(getattr(row.program, "constants", {}) or {})
        for constant in literals:
            tensor = values.get(constant.fqn) or values.get(constant.name)
            if tensor is None:
                missing.append(f"{row.name}{aot_serve.LITERAL_SEP}{constant.fqn}")
                continue
            tensors[f"{row.name}{aot_serve.LITERAL_SEP}{constant.fqn}"] = \
                tensor.detach().cpu().contiguous()
    if missing:
        raise MintRefused(
            f"{len(missing)} declared literal constant(s) have no value in "
            f"their exported program, e.g. {missing[:6]!r} — the cell could "
            f"never bind them and would segfault on first call (pgw#704 B1)")
    if not tensors:
        return
    from safetensors.torch import save_file

    save_file(tensors, str(Path(content_dir) / aot_serve.LITERALS_NAME))


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


def flat_input_leaves(
    module: Any, args: Tuple[Any, ...], kwargs: Mapping[str, Any],
) -> Tuple[aot_flatten.Leaf, ...]:
    """ONE leaf per EXPORTED user input — containers FLATTENED the way export
    flattens them, each carrying WHERE IN THE CALL it came from.

    MEASURED (pgw#790 lane, real sdxl UNet, torch 2.13): `aot_package.
    input_contract` zips the caller-side parameter names against the exported
    program's user inputs positionally, and a container argument occupies ONE
    parameter slot but produces N placeholders. sdxl's `added_cond_kwargs`
    ({text_embeds, time_ids}) therefore shifted every later name by one and the
    recorded contract came out as

        position 7  name 'added_cond_kwargs'                shape [2, 1280]
        position 8  name 'down_block_additional_residuals'  shape [2, 6]

    i.e. text_embeds and time_ids wearing the names of the parameters that
    follow them. At serve time `bind_call_inputs` then binds the pipeline's
    `added_cond_kwargs` DICT to a declared tensor input (`input_not_tensor`)
    and cannot find `down_block_additional_residuals` at all
    (`input_missing`) — every request refuses by name and the armed cell
    serves eager for life. That is the field symptom `bind_call_inputs`'
    own docstring records from pod ae2uc81yub0gyq; the nested-lookup patch
    treated the symptom, but a nested lookup cannot help when the NAMES it
    searches for are the wrong ones.

    THE NAMES ARE NOT ENOUGH (pgw#994). A name says which leaf this is; it
    does not say where the leaf LIVES in a call, and the serve side has only
    the call. So each leaf carries its identity — parameter, that parameter's
    position, and the path into it — and ``aot_serve.bind_call_inputs``
    replays that identity instead of guessing. The walk itself lives in
    ``aot_flatten``, which every consumer of the flat view reads, mint and
    serve alike: three separate spellings of this one mapping is exactly what
    pgw#790, pgw#993 and pgw#994 each were.

    ``_input_names`` is deliberately left alone: `dynamic_shapes_spec` keys on
    top-level PARAMETER names and mirrors containers structurally.
    """
    return aot_flatten.flatten_call(
        _input_names(module, args, kwargs), args, kwargs)


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


# ---------------------------------------------------------------------------
# Publish
# ---------------------------------------------------------------------------


def publish(result: MintResult, publisher: Any) -> str:
    """Publish a minted cell through a ``fleet_cells.CellPublisher``.

    Receipts are the HUB's business: it adds them at publish-finalize (#709),
    so the producer's whole obligation is a keyed ``metadata.json`` inside the
    tar — which :func:`mint` has already stamped and proven. Refuses before the
    wire when the artifact carries no key, since an unaddressable cell would be
    stored under a flavor nothing can request.
    """
    key = result.cell_key
    if not key:
        raise MintRefused("cannot publish an artifact with no cell_key")
    family = str(result.metadata.get("family") or "")
    if not family:
        raise MintRefused("cannot publish an artifact with no family")
    # th#1355: the mint pod already measured this (timings["total_s"]), so the
    # cell's own cell_store row records what it cost to build instead of the
    # cost living only in an activity event that carries no cell key.
    mint_duration_ms = max(0, int(round(float(result.timings.get("total_s") or 0.0) * 1000)))
    return str(publisher.publish(
        family, result.artifact, dict(result.metadata), mint_duration_ms))


# ---------------------------------------------------------------------------
# CLI — python -m gen_worker.aot_mint
# ---------------------------------------------------------------------------


def _load_spec(path: Path) -> Tuple[ExportSpec, Dict[str, Any]]:
    """A cell-level :class:`ExportSpec` from a JSON mint request.

    The request is a file rather than a pile of flags because a mint request is
    a CONTRACT (lane, precision, provenance, frozen specialization) that wants
    review and version control, not 20 argv strings. It names a FAMILY, never
    a target/fork/class coordinate: the cell covers the whole declared class
    set (pgw#758), so a coordinate-shaped request is refused by name rather
    than silently minting a subset of the contract the key advertises.
    """
    body = json.loads(Path(path).read_text())
    subset_fields = sorted(
        k for k in ("target", "fork", "class_dims", "dynamic")
        if body.get(k))
    if subset_fields:
        raise MintRefused(
            f"mint request {path} names {subset_fields!r} — a multi-graph "
            f"cell covers the family's WHOLE declared class set (pgw#758); "
            f"coordinates and dynamic rows derive from the declaration, "
            f"never from the request")
    spec = ExportSpec(
        family=str(body.get("family") or ""),
        target="",
        weight_lane=str(body.get("weight_lane") or ""),
        precision=str(body.get("precision") or "bf16"),
        lora_bucket=int(body.get("lora_bucket") or 0),
        shapes=tuple(tuple(int(v) for v in row) for row in body.get("shapes") or ()),
        batch=int(body.get("batch") or 0),
        text_lens=tuple(int(v) for v in body.get("text_lens") or ()),
        guidance_scales=tuple(float(v) for v in body.get("guidance_scales") or ()),
        specialization=dict(body.get("specialization") or {}),
        lora_fqns=tuple(str(v) for v in body.get("lora_fqns") or ()),
        lifted_inputs=tuple(str(v) for v in body.get("lifted_inputs") or ()),
        strict=bool(body.get("strict", True)),
        source_ref=str(body.get("source_ref") or ""),
        source_digest=str(body.get("source_digest") or ""),
        closure_roots=tuple(str(v) for v in body.get("closure_roots") or ()),
    )
    if not spec.family:
        raise MintRefused(f"mint request {path} must name 'family'")
    return spec, body


def main(argv: Optional[Sequence[str]] = None) -> int:
    """``python -m gen_worker.aot_mint <request.json> --out <dir>`` — produce
    one multi-graph cell (ops/testing entry point; production mints run in a
    serving pod's background under the pgw#677 eager-first machinery — #724
    was REJECTED, there is no dedicated mint fleet).

    Exit 0 minted (and published when asked), 2 a named mint refusal, 3 a bad
    invocation. Inspect-only by default: ``--publish`` is opt-in so a mint can
    be produced and byte-compared before anything reaches the hub.
    """
    parser = argparse.ArgumentParser(
        prog="gen_worker.aot_mint",
        description="Export + AOTI-package a family's declared class set "
                    "as one multi-graph cell.")
    parser.add_argument("request", type=Path, help="mint request JSON")
    parser.add_argument("--out", type=Path, required=True,
                        help="output directory for the packed artifact")
    parser.add_argument("--model", type=str, default="",
                        help="model path/ref to compose (default: the "
                             "request's source_ref)")
    parser.add_argument("--publish", action="store_true",
                        help="publish through the fleet CellPublisher")
    parser.add_argument("--require-toolchain", action="store_true",
                        default=True,
                        help="refuse without a C toolchain (default on)")
    args = parser.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        spec, body = _load_spec(args.request)
    except Exception as exc:
        print(f"BAD REQUEST {args.request}: {exc}", file=sys.stderr)
        return 3
    try:
        # pgw#739/#758: load the endpoint's declaration module (registers the
        # family's export contract). The cell's class set, coordinates, and
        # dynamic contracts all derive from it — the request only ever names
        # the family and the lane facts.
        from . import aot_declaration

        aot_declaration.load_declaration(body, request_path=args.request)
    except MintRefused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    if args.require_toolchain and not toolchain_present():
        print(
            "REFUSED: no C toolchain (cc/gcc) — an AOTI mint needs the "
            "compile-job image, not a prod worker image", file=sys.stderr)
        return 2

    model = args.model or spec.source_ref
    if not model:
        print("BAD REQUEST: no --model and the request has no source_ref",
              file=sys.stderr)
        return 3
    try:
        pipeline, _build_inputs = compose_for_mint(model, spec, body)
        result = mint(pipeline, spec, Path(args.out))
    except MintRefused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({
        "artifact": str(result.artifact),
        "cell_key": result.cell_key,
        "entries": sorted((result.metadata.get("entries") or {})),
        "timings": result.timings,
    }, indent=1))

    if args.publish:
        try:
            checkpoint = publish(result, _publisher_from_settings())
        except MintRefused as exc:
            print(f"REFUSED: {exc}", file=sys.stderr)
            return 2
        print(f"published checkpoint {checkpoint}")
    return 0


def compose_for_mint(
    model: str, spec: ExportSpec, request: Mapping[str, Any],
) -> Tuple[Any, Callable[[Any], Tuple[Tuple[Any, ...], Mapping[str, Any]]]]:
    """Compose the pipeline for a mint and return its example-input builder.

    Delegates to the family's own registered input contract rather than growing
    a per-family ``if`` ladder here: each family already owns how its denoiser
    is called (SDXL's ``added_cond_kwargs``, z-image's ragged lists, #729), and
    that knowledge must live with the family, not in the mint driver.
    """

    return aot_inputs.compose(model, spec, request)


def _publisher_from_settings() -> Any:
    """A ``fleet_cells.CellPublisher`` for a mint pod.

    A serving worker builds its publisher from the HelloAck ``file_base_url``
    and its rotating JWT (``executor._cell_publisher``). A mint pod has no
    orchestrator session, so the hub base comes from the injected tensorhub URL
    — the repo-commit API lives on the same combined-binary host — and the JWT
    from settings. Refuses by name when either is absent rather than attempting
    an unauthenticated publish.
    """

    settings = config.current()
    base_url = str(
        settings.tensorhub_public_url or settings.tensorhub_url or "").strip()
    # th#1423: `worker_credential.current()` only answers once someone HANDS it
    # the boot token, and only `entrypoint` / the procsplit parent do — never
    # this CLI. So pgw#876 §2's stated effect (WORKER_JWT visible here) was not
    # actually reached in this process; the fallback was doing all the work.
    worker_credential.install_bootstrap(settings)

    def credential() -> str:
        # th#1423: this was `lambda: token`, a token CAPTURED at construction.
        # A mint runs for tens of minutes and the credential's TTL does not
        # pause for it — the publisher must read the freshest one at USE time,
        # which is the whole reason `CellPublisher` takes a provider.
        return str(worker_credential.current()
                   or getattr(settings, "tensorhub_token", "") or "").strip()

    if not base_url or not credential():
        raise MintRefused(
            "cannot publish: TENSORHUB_PUBLIC_URL/TENSORHUB_URL and "
            "WORKER_JWT/TENSORHUB_TOKEN must both be set on a mint pod (the "
            "artifact was produced and is on disk)")
    publisher = CellPublisher(
        base_url=base_url,
        worker_jwt=credential,
        image_digest=str(getattr(settings, "worker_image_digest", "") or ""),
    )
    if not publisher.enabled():
        raise MintRefused("fleet CellPublisher is not enabled")
    return publisher


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "CODE_ONLY_CONFIGS",
    "DynamicDim",
    "ExportSpec",
    "MintRefused",
    "MintResourceExhausted",
    "PODGUARD_STATE_ENV",
    "partial_phase_table",
    "raise_if_device_oom",
    "write_phase_snapshot",
    "MINT_COMPILE_THREADS",
    "MintResult",
    "autotune_posture",
    "bench_step",
    "cell_identity",
    "compile_entry_files",
    "compose_for_mint",
    "declaration_module_gaps",
    "declared_range_gaps",
    "dynamic_shapes_spec",
    "emit_phase_events",
    "entry_graph_block",
    "export_program",
    "exported_input_names",
    "shared_identity_blocks",
    "LIFTED_LORA_TORCH_FLOOR",
    "lifted_input_gaps",
    "lifted_torch_gap",
    "main",
    "mint",
    "package_cell",
    "publish",
]
