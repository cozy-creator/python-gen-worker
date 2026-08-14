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
a compiled-lane backend is its own module riding the compile-cache rails
(``trt_engine`` established the pattern before TensorRT was deleted in
pgw#1187), and the dynamo mint stays live and
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
import hashlib
import json
import logging
import os
import sys
import time

import msgspec
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any, Callable, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple)

from . import activity as activity_mod
from . import (
    aot_compile_pool, aot_package,
    aot_serve, aot_wrapper_split, boot_phases, cell_key, graph_hash,
    kernel_path)
from .aot_contract import (  # re-exported: the declaration layer's vocabulary
    ADAPTER_FORK,
    DynamicDim,
    ExportSpec,
    MintRefused,
)
from .aot_preconditions import LIFTED_LORA_TORCH_FLOOR
from .compile_cache import (
    _resolve_target,
    cxx_toolchain_present,
)
from dataclasses import replace
import inspect
from .models.memory import is_cuda_oom
from . import aot_flatten
from . import aot_inputs
from . import aot_declaration as _decl
from .api.export_contract import export_declaration
from . import meta_instantiation
from .models import lora_lifted
from .models import structure_only
from . import compile_cache as cc
from . import env_seal
from . import config, worker_credential
from .fleet_cells import CellPublisher

logger = logging.getLogger(__name__)

#: The inductor config that makes the package code-only. Not a knob: B1.
CODE_ONLY_CONFIGS: Dict[str, Any] = {
    "aot_inductor.package_constants_in_so": False,
}

#: The inductor config that keeps every lifted weight BINDABLE. Not a knob
#: either, and for the same reason B1 is not one — this is what makes one cell
#: legally serve every fine-tune of a family (pgw#1097, pgw#857).
#:
#: Off (the torch default), ``GraphLowering.get_attr`` inlines a constant whose
#: SHAPE meets either of two rules — 0-dim, or ``len(shape) == 1 and
#: shape[0] <= 8`` — by rendering its VALUES into the generated kernel. Those
#: values are the minting checkpoint's, and the constant then appears in no
#: table anyone could rebind. MEASURED on torch 2.13.0 (pgw#1097): a 4-element
#: conv bias, an 8-element group-norm scale and a 0-dim learned scalar all left
#: the bindable set; with this flag all three stay. It is also what the fleet's
#: one recorded real-weight elimination was — sdxl's ``unet.conv_out.bias``,
#: 4 floats — which the tree had filed as routine conv-epilogue fusion.
#:
#: **Why the RUNTIME-FOLDING split and not ``always_keep_tensor_constants``.**
#: Both restore bindability, and the cheaper-looking flag is the wrong one —
#: CI proved it. ``always_keep_tensor_constants`` also retains ANONYMOUS graph
#: literals (``_tensor_constant0``) as ORDINARY constants, and a literal the
#: recorded program never lifted is precisely the ``program_package_drift``
#: refusal (pgw#704 B1): "the package declares a constant the program never
#: lifted — nothing would bind it and the first call would segfault". Measured
#: on a plain-attribute table built inside ``forward`` (the pgw#857 authoring
#: violation, `test_aot_multigraph_pgw758.WarmSensitive`): every mint of that
#: shape REFUSED. The runtime split has no such problem because its outputs are
#: ``FoldedConstant``/``SOURCE_COMPUTED`` rows, which that gate ALREADY exempts
#: — a carve-out pgw#1080 added for this exact flag, and which no equivalent
#: exists for the other one.
#:
#: The price is honest and is paid on purpose: the split materializes a
#: ``_FOLDED_CONST_*`` tensor per folded op at load — measured 106,496 bytes on
#: a 1.1 MB micro decoder (permuted copies of its linear weights, ~10% of model
#: size). **Unmeasured at sdxl scale and owed**; do not extrapolate the 10%,
#: since which ops fold is graph-shaped. It buys the property that one cell may
#: legally serve every fine-tune of a family, which is the whole cell economy.
#:
#: Weightless mints (pgw#1080) needed this same flag for a DIFFERENT reason —
#: their values are fake — so the two motives now converge on one config and
#: there is no longer a weightless special case.
CONSTANT_BINDING_CONFIGS: Dict[str, Any] = {
    "aot_inductor.use_runtime_constant_folding": True,
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


#: Mint-path default worker count for inductor's parallel compile (#757,
#: MEASURED): 32 -> 4 is FREE (-2% wall clock) and is the recommended
#: default for background mints on serving pods — same speed, less CPU
#: contention with live serving. NOT seal-relevant: compile_threads is
#: outside cell identity per #757's re-key pre-verification, so this
#: default (and a caller override) never re-keys a cell. A caller value
#: wins; this is a default, not a clamp.
MINT_COMPILE_THREADS = 4


def _entry_configs(
    inductor_configs: Optional[Mapping[str, Any]], *, weightless: bool = False,
) -> Dict[str, Any]:
    """The per-entry inductor config: caller options + the non-negotiable
    packaging flags. ``CODE_ONLY_CONFIGS`` and ``CONSTANT_BINDING_CONFIGS``
    are applied LAST so no caller-supplied config can re-enable constant
    baking or weight inlining — B1 and the folding fence are fleet
    correctness requirements, not defaults a caller may override. One cell's
    entries ALL compile under this one dict (a per-entry config drift would
    be an identity fact nothing records), and the resolved dict is recorded
    in the mint-phase telemetry."""
    configs: Dict[str, Any] = dict(inductor_configs or {})
    configs.setdefault("compile_threads", MINT_COMPILE_THREADS)
    non_negotiable = {**CODE_ONLY_CONFIGS, **CONSTANT_BINDING_CONFIGS}
    overridden = sorted(set(configs) & set(non_negotiable))
    if overridden:
        logger.warning(
            "aot-mint: ignoring caller inductor config %s — code-only (B1) "
            "and the folding fence (pgw#1097) are not knobs", overridden)
    configs.update(non_negotiable)
    # Emit loose files for package_aoti to combine, instead of a per-entry
    # archive: the multi-graph cell is ONE .pt2 (pgw#758).
    configs["aot_inductor.package"] = True
    # pgw#1080's weightless motive and pgw#1097's real-weight motive converge
    # on ONE config, so `weightless` no longer selects anything here. It stays
    # in the signature because callers pass it and because the two motives are
    # worth keeping distinct in the record: weightless mints must defer the
    # fold because the values they would bake are FAKE (micro decoder's
    # `norm.weight`/`norm.bias` vanished; the adopted cell scored cosine 0.13);
    # real-weight mints must defer it because the values they would bake are
    # one CHECKPOINT'S, which no other fine-tune could rebind.
    del weightless
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
        # pgw#1080: a program exported from a structure-only target carries
        # fake parameters, and `aot_compile` asserts every input belongs to
        # ONE fake mode — so it runs inside that program's own mode. Identity
        # context for a real-weight program.
        with structure_only.compiling_under(program):
            files = aot_compile(
                gm, tuple(args), dict(kwargs or {}),
                options=_entry_configs(
                    inductor_configs,
                    weightless=structure_only.fake_mode_of_program(
                        program) is not None))
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


def _entry_dir_name(entry: str) -> str:
    """A filesystem-safe directory name for one entry's packing area.

    Entry names carry ``/``, ``=`` and ``,`` (``denoiser/h=16,w=16``), so they
    are not path components. The digest keeps two entries apart without
    inventing a second naming scheme anyone has to keep in sync — nothing
    reads this name back; the ARTIFACT is addressed by its ``cg-key-v1`` key.
    """
    return hashlib.sha256(str(entry).encode("utf-8")).hexdigest()[:16]


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
class MintedArtifact:
    """ONE packed, gated, publishable ENTRY: one graph class."""

    key: str
    entry: str
    artifact: Path
    metadata: Dict[str, Any]


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

    @property
    def keys(self) -> Tuple[str, ...]:
        return tuple(sorted(row.key for row in self.entries))


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


def module_precision(module: Any) -> str:
    """pgw#1076: what floating-point dtype a module ACTUALLY holds.

    Weighted by element count over parameters and buffers, because a module is
    "an fp32 module with one bf16 norm buffer" and not "half and half". More
    than one float dtype is reported as ``mixed(a+b)``, dominant first — a
    mixture is a fact worth naming, and naming it is strictly better than
    picking a winner and calling the cell that.

    ``""`` when nothing is measurable (no tensors, no torch, a callable
    target that is not a module). An absent fact beats an invented one; that is
    the whole issue.
    """
    counts: Dict[str, int] = {}
    for attr in ("parameters", "buffers"):
        get = getattr(module, attr, None)
        if not callable(get):
            continue
        try:
            tensors = list(get(recurse=True))
        except Exception:  # noqa: BLE001 — a label never fails a mint
            continue
        for tensor in tensors:
            try:
                if not bool(tensor.is_floating_point()):
                    continue
                raw = str(tensor.dtype)
                counts[_PRECISION_LABELS.get(raw, raw.replace("torch.", ""))] = (
                    counts.get(
                        _PRECISION_LABELS.get(raw, raw.replace("torch.", "")), 0)
                    + int(tensor.numel()))
            except Exception:  # noqa: BLE001
                continue
    if not counts:
        return ""
    if len(counts) == 1:
        return next(iter(counts))
    return "mixed(" + "+".join(
        sorted(counts, key=lambda label: (-counts[label], label))) + ")"


def _measured_precision(pipeline: Any, rows: Sequence[Tuple[Any, Any]]) -> str:
    """The cell-wide precision stamp, measured over the modules this mint will
    actually trace (pgw#1076).

    Every distinct declared target contributes; disagreement between targets is
    reported as a mixture rather than resolved, for the same reason a mixture
    inside one module is. Unresolvable targets contribute nothing — this runs
    BEFORE the export's own target gate, and a label must never be the thing
    that refuses a mint.
    """
    labels: Dict[str, None] = {}
    for plan, _arm in rows:
        target = str(getattr(plan, "target", "") or "")
        if not target:
            continue
        resolved = _resolve_target(pipeline, target)
        if resolved is None:
            continue
        label = module_precision(resolved[0])
        if label:
            labels[label] = None
    if not labels:
        return ""
    if len(labels) == 1:
        return next(iter(labels))
    return "mixed(" + "+".join(sorted(labels)) + ")"


#: pgw#1208: the phase token an unexportable class reports under. One kind, so
#: the hub can COUNT them per family — a class that cannot be exported is a
#: standing authoring/toolchain fact, not a transient, and the fleet's answer to
#: "which classes never compile" should be a query rather than a log hunt.
KIND_ENTRY_EXPORT_UNSUPPORTED = "entry_export_unsupported"


def _export_skippable(exc: BaseException) -> bool:
    """Whether ONE class's export failure may be skipped, leaving the rest.

    Skippable means DETERMINISTIC, LOCAL, and TORCH'S OWN: this graph class
    contains a construct `torch.export` refuses, and it will refuse identically
    on every retry, on every pod, forever. Skipping it costs one class and
    saves the other thirty-five.

    NOT skippable, and the distinction is the whole safety of this:

    * a RESOURCE shortfall (CUDA OOM, host memory) — it says nothing about the
      class, it says the pod is out of room, and the mint must abort so the
      parent can retry narrower. Skipping here would silently publish a partial
      cell whose missing classes are an artifact of memory pressure and would
      have exported fine on the retry.
    * a ``BaseException`` that is not an ``Exception`` (KeyboardInterrupt,
      SystemExit) — a shutdown is not a property of the graph.
    * ``MintResourceExhausted`` and anything else carrying the duck-typed
      ``mint_resource_shortfall`` marker, for the same reason as the first.
    """
    from .models.memory import is_cuda_oom

    if not isinstance(exc, Exception):
        return False
    if is_cuda_oom(exc):
        return False
    if getattr(exc, "mint_resource_shortfall", False) is True:
        return False
    # ...and it must be TORCH'S OWN export refusal, not any deterministic
    # exception that happened to surface during the export step.
    #
    # `_export_entry` resolves, feeds, WARMS, exports, GATES and compiles. Only
    # the export's own "I cannot trace this construct" is a property of the
    # graph class. The others are not, and skipping them would be the silent
    # failure this issue exists to remove, wearing a new hat:
    #
    #   * a declared WARM that blows up says the class cannot RUN — pgw#758
    #     made that a named refusal deliberately, and a cell whose classes were
    #     never warm-proven must not publish;
    #   * a MintRefused from our own gates (the folding fence, ingress
    #     admission, identity) is a CORRECTNESS verdict, and a correctness gate
    #     that can be skipped is not a gate;
    #   * a plain exception from the module's own forward is a broken forward,
    #     not an unsupported construct.
    #
    # So the test is the exception's ORIGIN: torch's dynamo/export namespace.
    module = type(exc).__module__ or ""
    return module.startswith(("torch._dynamo", "torch._export", "torch.export",
                              "torch.fx.experimental"))


def _unsupported_construct(exc: BaseException) -> str:
    """The CONSTRUCT that refused, named, in one line.

    A skipped class is only actionable if the reason names the thing an author
    or a toolchain bump has to change. Dynamo says it precisely and then buries
    it in a page of traceback, so the useful lines are lifted out: its own
    ``Explanation:`` and ``Developer debug context:`` when present, else the
    first line of the message. Never the whole traceback — this rides an event.
    """
    text = str(exc).strip()
    picked = [
        line.strip() for line in text.splitlines()
        if line.strip().startswith(("Explanation:", "Developer debug context:"))
    ]
    head = text.splitlines()[0].strip() if text else type(exc).__name__
    out = f"{type(exc).__name__}: {head}"
    if picked:
        out += " — " + " ".join(picked[:2])
    return out[:600]


def _emit_entry_export_unsupported(
    entry: str, detail: str, *, family: str = "",
) -> None:
    """Tell the hub which class was skipped and why. Never fails a mint."""
    try:
        activity_mod.emit_event(
            KIND_ENTRY_EXPORT_UNSUPPORTED,
            f"family={family} entry={entry}: torch.export refused this graph "
            f"class and it is SKIPPED — {detail}. The remaining classes still "
            f"mint and publish; this class serves eager.",
            phase=PHASE_TRACE_GRAPH,
        )
    except Exception:  # noqa: BLE001 — telemetry never fails a mint
        logger.debug("aot-mint: entry-skip event dropped", exc_info=True)


def _export_entry(
    pipeline: Any,
    spec: ExportSpec,
    plan: Any,
    decl: Any,
    *,
    inductor_configs: Optional[Mapping[str, Any]] = None,
    compile_now: bool = True,
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
            module=module, timings=timings, fake_mode=fake_mode,
            inductor_configs=inductor_configs, compile_now=compile_now)


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
    inductor_configs: Optional[Mapping[str, Any]] = None,
    compile_now: bool = True,
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


#: pgw#1167 verdicts. UNRECONCILED is a FIRST-CLASS state, not a silent pass:
#: the whole point is that a gate which cannot fire is indistinguishable from
#: one that passed, so the two are named apart and both are reported.
LATENT_RECONCILED = "latent_basis_reconciled"
LATENT_UNRECONCILED_UNDECLARED = "latent_basis_unreconciled_no_declared_basis"
LATENT_UNRECONCILED_NO_VAE = "latent_basis_unreconciled_no_vae"


def observed_latent_basis(pipeline: Any) -> Optional[int]:
    """The pipeline's REAL latent divisor, or ``None`` when unobservable.

    diffusers computes it as ``2 ** (len(vae.config.block_out_channels) - 1)
    if vae else 8`` — so a pipeline with NO vae reports **8**, a default nobody
    chose and indistinguishable from a real observation of 8. Believing the
    attribute alone would reproduce pgw#1058's silent-dtype-default defect one
    field over, so the vae is required before the number is believed.
    """
    if getattr(pipeline, "vae", None) is None:
        return None
    value = getattr(pipeline, "vae_scale_factor", None)
    try:
        basis = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return basis if basis > 0 else None


def reconcile_latent_basis(pipeline: Any, spec: ExportSpec) -> str:
    """Refuse a mint whose declared latent divisor is not the pipeline's.

    pgw#1167. The class rows are derived by dividing declared PIXEL shapes by a
    latent divisor the author passes to ``derive.cfg_image_classes``. Nothing
    checked that divisor against the checkpoint, so a wrong one produced a
    whole cell of correctly-shaped, permanently unusable artifacts — silent,
    and paid for at full mint price.

    It is checked HERE because here is the first place both facts exist and the
    last place before the export is paid for: ``mint()`` is reached by the
    production child (``mint_child``) AND the operator CLI, whereas
    ``aot_export_spec`` is on the pod route only and would have exempted the
    operator entirely.

    The divisor cannot be VALIDATED at declaration time (no pipeline) nor
    DERIVED from one (§4.27/pgw#1089 requires ``ck1`` to derive from code alone,
    before any weight is resident, and the class rows feed the key). So it is
    declared once, carried, and reconciled — and when either side is missing
    the verdict is UNRECONCILED, never a pass.
    """
    # This gate must never be the thing that kills a mint. Its only two
    # outcomes are a NAMED refusal for a proven mismatch and an explicit
    # UNRECONCILED; anything it cannot read is the latter, so a caller with no
    # spec (the telemetry paths hand `mint` placeholder arguments) degrades
    # instead of raising an AttributeError from inside a correctness check.
    family = str(getattr(spec, "family", "") or "")
    declaration = export_declaration(family) if family else None
    declared = getattr(declaration, "latent_basis", None) if declaration else None
    if declared is None:
        logger.info(
            "aot-mint: %s — the class rows carry no derived latent basis, so "
            "the pipeline's divisor is not reconciled against anything",
            LATENT_UNRECONCILED_UNDECLARED)
        return LATENT_UNRECONCILED_UNDECLARED

    observed = observed_latent_basis(pipeline)
    if observed is None:
        logger.info(
            "aot-mint: %s — this pipeline exposes no vae, and diffusers' "
            "`else 8` fallback is a default nobody chose; declared basis %d is "
            "left unreconciled rather than compared against it",
            LATENT_UNRECONCILED_NO_VAE, declared)
        return LATENT_UNRECONCILED_NO_VAE

    if declared != observed:
        raise MintRefused(
            f"latent_basis_mismatch: the declaration derived its graph classes "
            f"at a latent divisor of {declared}, and the composed pipeline's "
            f"vae divides by {observed}. Every declared latent extent is "
            f"therefore wrong for THIS composition, and the cell would mint "
            f"correctly-shaped artifacts that serve nothing. This declaration "
            f"does not match this composition — check the `latent_scale=` "
            f"passed to the class deriver, and any component override that "
            f"swaps the vae for one of a different architecture")
    return LATENT_RECONCILED


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


def mint(
    pipeline: Any,
    spec: ExportSpec,
    out_dir: Path,
    *,
    inductor_configs: Optional[Mapping[str, Any]] = None,
    entry_workers: int = 0,
    compiled_graph_peak_rss_bytes: int = 0,
    on_progress: Optional[Callable[[str, int, int, str], None]] = None,
    phase_snapshot: Optional[Path] = None,
    execution_lane_verdict: Optional[kernel_path.Verdict] = None,
    release_residents: bool = False,
) -> MintResult:
    """:func:`_mint_cell`, with the phase table attached to EVERY terminus.

    ``release_residents`` (pgw#1053) is the CALLER's statement that it has no
    further use for ``pipeline`` after the last row exports — the mint then
    projects the pipeline and the retained programs to code-only and hands
    the device memory back to the compile pool's budget. The mint child and
    the operator CLI say True (their pipeline dies with the process); library
    callers and tests that keep using the pipeline afterwards default False.
    A lifecycle fact stated by the owner, not a tuning knob.

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
    # pgw#719/pgw#1049: the tripwire, before ANY trace — drifted settings
    # refuse by name here; they can no longer move the (declaration-derived)
    # seal, so this is the only place ambient mutation can surface.
    env_seal.assert_seal_unchanged("aot_mint")
    # pgw#1167: before ANY export, while a wrong latent divisor still costs
    # seconds instead of a whole cell of unusable artifacts.
    reconcile_latent_basis(pipeline, spec)
    progress = MintProgress(
        inductor_configs=inductor_configs, on_progress=on_progress)
    _attach_snapshot(progress, phase_snapshot)
    try:
        return _mint_cell(
            pipeline, spec, out_dir,
            inductor_configs=inductor_configs,
            entry_workers=entry_workers,
            compiled_graph_peak_rss_bytes=compiled_graph_peak_rss_bytes,
            execution_lane_verdict=execution_lane_verdict,
            release_residents=release_residents,
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
    fit", which is 1, forever. The width rule (``aot_export_parallel``,
    deleted by pgw#1030 — export stayed serial and nothing consumed its
    decision) then returned 1 with ``binding='export-footprint-unmeasured'``.
    The number was not missing; it was the wrong number, and nothing said so.

    A row's DELTA over the resident baseline is the right one: it is what a
    worker adds to a card that already holds the module it traces. Export runs
    on fake tensors and launches no kernel, so it should be small — but
    "should be" is the claim this class exists to replace, and the fallback
    when it cannot be read is still a refusal to widen.

    Reported both ways, because they answer different questions and conflating
    them is the defect: ``per_export_device_bytes`` (the MAX row delta) and
    ``export_peak_device_bytes`` (the phase high-water). pgw#1175: both are
    TELEMETRY. Nothing divides a card by either.
    """

    __slots__ = ("baseline", "rows", "readable")

    def __init__(self, baseline: int, readable: bool) -> None:
        self.baseline = baseline
        self.rows: List[int] = []
        self.readable = readable

    @classmethod
    def open(cls) -> "_ExportFootprint":
        try:
            import torch as _t

            if not _t.cuda.is_available():
                return cls(0, False)
            return cls(int(_t.cuda.memory_reserved()), True)
        except Exception:  # noqa: BLE001 — a probe never changes an outcome
            return cls(0, False)

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
        a measured zero and an unread card are different facts."""
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
    compiled_graph_peak_rss_bytes: int = 0,
    execution_lane_verdict: Optional[kernel_path.Verdict] = None,
    release_residents: bool = False,
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
    # pgw#1076: the `precision` stamp is a MEASUREMENT, and until here nobody
    # made it — `ExportSpec.precision` defaulted to "bf16", so a micro-conv
    # cell with fp32 weights, fp32 inputs and an fp32 traced graph packaged
    # `metadata.json precision: "bf16"` and every arm line printed
    # `precision=bf16`. A reader debugging a 1.2e-3 GPU parity delta reads
    # that as "the mint cast to bf16" and spends a cycle disproving it (the
    # real cause was TF32 conv kernels). A caller that KNOWS the lane — every
    # real family, via `weight_lane` — keeps its own word; only an ABSENT
    # stamp is derived, and an underivable one stays absent.
    if not str(spec.precision or "").strip():
        measured = _measured_precision(pipeline, rows)
        logger.info(
            "aot-mint: pgw#1076 precision measured from the traced modules: "
            "%r (no lane declared; %d declared class row(s))",
            measured or "<unmeasurable>", len(rows))
        spec = replace(spec, precision=measured)
    # pgw#1215: this function is the SERIAL path and nothing else. It holds a
    # live pipeline on a live card, so it exports and compiles in its own
    # address space, one declared class at a time — which is exactly what the
    # keystone made every path do. The K-wide path did the opposite: it
    # exported here and shipped the ExportedProgram to a compile child, and
    # the `torch.export.save`/`load` pair that took cost a 36.04 s median per
    # class (pgw#1216). K-wide now means K compile CHILDREN that each trace
    # their own share — driven by :func:`mint_graph_classes`, which needs no
    # pipeline in the parent at all, and therefore cannot be driven from here.

    minted = progress.minted
    #: pgw#1208: classes this mint could not export, each with the construct
    #: that refused. They are NOT packed and NOT published — they are recorded,
    #: so a cell that covers 35 of 36 classes says which one it does not and
    #: why, instead of the whole mint disappearing behind one refusal.
    skipped: List[Tuple[str, str]] = []
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
    # describe it. A probe: it reads and clears a counter and decides
    # nothing (pgw#830 — instrument first, optimise never in the same change).
    # Still honest under pgw#1052's overlap: the compile children are separate
    # OS processes, so their device use never lands in THIS allocator's
    # counters — the export figures stay the export's.
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

    def _rows_source() -> Iterator[_MintedEntry]:
        """Export every declared row IN ORDER — serial, in this process,
        inside the one branch-arm toggle (pgw#790). One body for the serial
        and the overlapped paths: pgw#1052 changes the HANDOFF time of each
        exported row, never the export order or its gates."""
        disarmed = False
        try:
            for index, (plan, arm) in enumerate(rows, start=1):
                if arm is False and not disarmed:
                    # ONE toggle for the whole branchless group (the rows are
                    # ordered adapter-bearing first): disable/enable
                    # reallocates every leaf's branch container, and doing it
                    # per entry would be N times the VRAM churn for the same
                    # graphs.
                    _disarm_branches(pipeline)
                    disarmed = True
                # Reported BEFORE the work, not after: a row that never
                # returns is the one a reader most needs named, and an
                # after-the-fact tick names only the rows that finished.
                progress.beat(
                    PHASE_TRACE_GRAPH, index, len(rows),
                    _decl.plan_entry_name(plan))
                name = _decl.plan_entry_name(plan)
                try:
                    with export_footprint.row():
                        entry = _export_entry(
                            pipeline, spec, plan, decl,
                            inductor_configs=inductor_configs,
                            compile_now=True)
                except BaseException as exc:  # noqa: BLE001 — classified below
                    # pgw#1208: ONE class that cannot export must not cost the
                    # other 35. Before this, a single deterministic refusal
                    # anywhere in the row loop aborted the whole mint and threw
                    # away every class that had already exported clean — the
                    # per-entry atom's own philosophy (pgw#718: the entry is
                    # the unit of identity and of publish) applied everywhere
                    # EXCEPT to the loop that produces the entries.
                    #
                    # Fail-closed stays fail-closed AT THE ENTRY: this class
                    # proved nothing, so it is never packed and never
                    # published, and serving covers it eager by the same
                    # mechanism that covers any shape outside a cell's declared
                    # envelope (pgw#844 — refused BY NAME, served eager, still
                    # armed). What changes is only the blast radius.
                    if not _export_skippable(exc):
                        raise
                    detail = _unsupported_construct(exc)
                    skipped.append((name, detail))
                    logger.warning(
                        "aot-mint: entry %s cannot be exported and is SKIPPED "
                        "— %s. The remaining classes still mint; this one "
                        "serves eager.", name, detail)
                    _emit_entry_export_unsupported(
                        name, detail, family=str(spec.family or ""))
                    progress.beat(
                        PHASE_TRACE_GRAPH, index, len(rows),
                        f"{name}: SKIPPED ({detail})")
                    continue
                minted.append(entry)
                yield entry
        finally:
            if disarmed:
                _arm_branches(pipeline, int(spec.lora_bucket or 0))

    def _close_export_phase() -> None:
        timings["export_all_s"] = round(time.monotonic() - t_export, 2)
        # Sampled the moment the LAST row exports — before pgw#1053's release
        # can reset the counters. Reported even when zero (no CUDA / probe
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

    for _entry in _rows_source():
        pass
    _close_export_phase()
    if release_residents:
        t0 = time.monotonic()
        timings.update(release_mint_residents(pipeline, minted))
        timings["residents_release_s"] = round(time.monotonic() - t0, 2)
    # Asked of the EXPORTED programs: a cell whose entries cannot be told
    # apart at dispatch must cost seconds to refuse, not a full compile bill
    # (the pgw#825 discipline, one gate over). A serial mint has already
    # compiled as it exported, so here it refuses late.
    minted, class_aliases = canonicalize_dispatch_classes(minted)
    # pgw#1208: what this cell does NOT cover, and why. Recorded on the mint's
    # own timings so it reaches the phase table (and therefore the hub) beside
    # the classes that did export — a partial cell must be able to say which
    # classes are missing, or "35 of 36" is indistinguishable from "36 of 36".
    #
    # A cell with NO entries still refuses: `_pack` raises `cannot package a
    # cell with no entries`, so "every class was skipped" fails closed on the
    # path it already failed closed on rather than through a second check here.
    timings["skipped_entries"] = float(len(skipped))
    if skipped:
        logger.warning(
            "aot-mint: %d of %d declared class(es) could not be exported and "
            "are absent from this cell: %s", len(skipped), len(rows),
            "; ".join(f"{n} ({d})" for n, d in skipped[:4]))
    timings["canonicalized_entries"] = float(
        sum(len(rows) for rows in class_aliases.values()))

    # ── PACK PER ENTRY (pgw#1176) ──────────────────────────────────────────
    # Lifted to `pack_graph_classes` (pgw#1215) so the compile child can pack
    # its own share. Ordering, refusals and timings keys are unchanged.
    return pack_graph_classes(
        minted,
        spec=spec,
        work=work,
        out_dir=out_dir,
        class_aliases=class_aliases,
        timings=timings,
        t_mint=t_mint,
        inductor_configs=inductor_configs,
        pool_ledger=progress.pool_ledger,
        execution_lane_verdict=execution_lane_verdict,
        progress=progress,
    )



def class_manifest(
    entry_blocks: Mapping[str, Mapping[str, Any]], spec: ExportSpec,
) -> str:
    """The declaration-wide coverage LABEL over a set of entry blocks.

    ONE fold, so the label a whole-declaration mint stamps and the label
    :func:`mint_graph_classes` assembles from K shares are the same
    computation over the same stamped ``class_hash`` values. Telemetry, never
    identity: nothing resolves it and nothing downloads it — the hub folds
    compile-health rows under ``(manifest, sm, toolchain)`` with it.
    """
    return cell_key.manifest_digest(
        aot_serve.stamp_entry(
            name, block, strict=bool(spec.strict),
            lora_bucket=int(spec.lora_bucket or 0)).get("class_hash") or ""
        for name, block in entry_blocks.items())


def pack_graph_classes(
    minted: Sequence["_MintedEntry"],
    *,
    spec: ExportSpec,
    work: Path,
    out_dir: Path,
    class_aliases: Mapping[str, Sequence[Any]],
    timings: Dict[str, Any],
    t_mint: float,
    inductor_configs: Optional[Mapping[str, Any]] = None,
    width: Optional[aot_compile_pool.PoolWidth] = None,
    pool_ledger: Optional[Mapping[str, Any]] = None,
    execution_lane_verdict: Optional[kernel_path.Verdict] = None,
    progress: Optional["MintProgress"] = None,
    manifest: Optional[str] = None,
) -> MintResult:
    """Pack every compiled graph class into its own artifact — the mint's tail.

    Lifted out of :func:`_mint_cell` UNCHANGED (pgw#1215, th#1834 Phase 3 step
    2a). It moves because the process-layer rewrite has to call it from the
    COMPILE CHILD: once a child traces and compiles its own share in one
    address space it holds the ``_MintedEntry`` rows and must turn them into
    artifacts itself. Left inline, the child would have to re-implement
    packaging — and two packagers is exactly the divergence per-graph-class
    identity exists to rule out.

    It takes ``_MintedEntry`` rows because that is already what
    :func:`_export_entry` returns and what :func:`trace_for_key` builds
    internally. The seam is not invented here; the two halves have always
    spoken this type, which is why the extraction is a move and not a design.

    Byte-identical to the inline version: same order, same refusals, the same
    ``timings`` keys written at the same points, the same phase event.
    ``progress`` is optional only so a caller with no beat can pack; every
    other argument is required because the artifact's identity depends on it.

    ``manifest`` (pgw#1215) states WHOSE coverage the packed label describes.
    ``None`` — the caller holds the whole declaration — computes it over the
    rows in hand, which is what every caller before the keystone did and what
    the serial path still does. A compile child holds ONE SHARE and passes
    ``""``: it cannot state a declaration-wide coverage label, and the
    honest answer to "how much of the declaration does this cover" is silence
    rather than a share-local digest that reads like a whole one. The publish
    path already says so in as many words (``fleet_cells._identity_axes``:
    *"Empty is HONEST for an entry minted by a pod that has not folded its
    whole declaration, so it is not a publish refusal"*), and the label is
    telemetry — it reaches no key. :func:`mint_graph_classes` folds the real
    one across every share and stamps it on the result it returns.
    """
    progress = MintProgress() if progress is None else progress
    # ── PACK PER ENTRY (pgw#1176) ──────────────────────────────────────────
    #
    # One artifact per graph class, not one per cell. The multi-entry
    # `model.pt2` is DELETED: it made identity, adoption, durability,
    # verification and arming the same 36-entry unit, so a class that failed
    # anywhere destroyed the whole mint (th#1825 lost 1 h 37 m to a segfault on
    # the last entry). Each entry is packed, keyed, and finished on its own —
    # so a crash costs the one in flight, and the entries that already exist
    # are publishable and armable immediately.
    #
    # NOTE ON PROCESS GRANULARITY, because "per entry" invites the wrong
    # reading: this is the granularity of the ARTIFACT, not of the compile
    # CHILD. pgw#1177 measured ~39 s of `env_seal.establish()` per fresh child
    # — ~23 minutes across 36 entries — so a child that compiles several
    # entries in sequence is strictly better and nothing here forbids it.
    # FAIL-CLOSED SURVIVES THE SPLIT, and pgw#1208's row is what caught that it
    # nearly did not. `package_cell` used to raise "cannot package a cell with
    # no entries" because it was called ONCE with the whole set; moving it
    # inside the per-entry loop meant an all-skipped mint simply never entered
    # the loop and returned an EMPTY MintResult — a silent success reporting
    # that a mint produced nothing. Absence is a verdict (pgw#939), and the
    # verdict for "every declared class refused" is a refusal, not an empty
    # set. Per-entry fail-closed means each class refuses on its own; it does
    # not mean the mint stops having a terminal outcome.
    if not minted:
        raise MintRefused(
            "no entries: every declared graph class refused before packaging, "
            "so this mint produced nothing to key, publish or arm")
    t0 = time.monotonic()
    progress.beat(
        PHASE_SEAL_PUBLISH, len(minted), len(minted),
        f"packaging {len(minted)} entries")
    entry_blocks: Dict[str, Dict[str, Any]] = {}
    packages: Dict[str, Path] = {}
    for row in minted:
        entry_work = work / "entries" / _entry_dir_name(row.name)
        entry_work.mkdir(parents=True, exist_ok=True)
        package = package_cell(
            {row.name: row.files}, entry_work / aot_serve.PACKAGE_NAME)
        packages[row.name] = package
        entry_blocks[row.name] = _gate_and_declare_entry(row, package)
        # pgw#917: the declared-class names this entry absorbed. Recorded so
        # the merge is auditable from the artifact alone — a reader asking
        # "where did class row X go" gets an answer instead of an absence.
        # NOT a `class_hash` fact (see `aot_serve.class_hash`, which folds
        # named fields only): an alias declares no envelope the surviving
        # entry's own contract does not already declare, so it must not
        # re-key an otherwise identical entry.
        merged = class_aliases.get(row.name) or ()
        if merged:
            entry_blocks[row.name]["aliases"] = [
                {"name": alias.name,
                 "class_dims": [
                     [str(n), int(v)] for n, v in sorted(alias.spec.class_dims)]}
                for alias in sorted(merged, key=lambda r: r.name)
            ]
        _write_literals([row], package, entry_work)
    timings["package_s"] = round(time.monotonic() - t0, 2)

    t0 = time.monotonic()
    shared = shared_identity_blocks(spec)
    # The declaration-wide coverage LABEL, computed once over every class this
    # mint produced. Telemetry, never identity: nothing resolves it, nothing
    # downloads it, and the hub folds compile-health rows under
    # (manifest, sm, toolchain) with it.
    manifest = class_manifest(entry_blocks, spec) if manifest is None \
        else str(manifest)
    timings["declare_s"] = round(time.monotonic() - t0, 2)
    timings["total_s"] = round(time.monotonic() - t_mint, 2)
    phase_table = _mint_phase_table(
        minted, timings, inductor_configs, width, pool_ledger)
    _emit_phase_event(spec, phase_table)

    t0 = time.monotonic()
    packed: List[MintedArtifact] = []
    for name, block in entry_blocks.items():
        try:
            meta = aot_serve.entry_metadata(
                family=spec.family,
                precision=spec.precision,
                cell_key="",
                name=name,
                entry=block,
                strict_export=bool(spec.strict),
                lora_bucket=int(spec.lora_bucket or 0),
                source_ref=spec.source_ref,
                source_digest=spec.source_digest,
                manifest_digest=manifest,
            )
        except ValueError as exc:
            # The artifact-metadata envelope validates the contract it is
            # handed (the OTHER "envelope" — the declared serving region — is
            # what the range gates above police). A malformed one must fail
            # HERE, on the mint pod, not at serve time on a paying request.
            raise MintRefused(
                f"envelope refused the declared contract: {exc}") from exc
        meta.update(shared)
        if execution_lane_verdict is not None:
            # pgw#947: the DISCRETE verdict only. Milliseconds in
            # metadata.json would break the #699 double-mint byte-compare —
            # the artifact deliberately carries no wall clocks — and the
            # margin threshold is what makes the discrete answer reproducible
            # across two mints.
            meta[kernel_path.META_KEY] = kernel_path.envelope_block(
                execution_lane_verdict)
        meta["cell_key"] = key = cell_identity(meta).digest
        artifact = aot_serve.pack(
            packages[name].parent, out_dir / f"{key}.tar.gz", meta)
        # The phase table rides the RESULT (and the published checkpoint
        # metadata + the typed event), never the packed envelope: durations in
        # metadata.json would break the #699 double-mint byte-compare — the
        # artifact deliberately carries no timestamps and no wall clocks.
        meta["mint_phases"] = phase_table
        if execution_lane_verdict is not None:
            # The EVIDENCE: both lanes' ms/step and peak bytes, the margin,
            # the headroom terms, and the device it was all measured on.
            meta[kernel_path.EVIDENCE_KEY] = kernel_path.evidence_block(
                execution_lane_verdict)
        packed.append(MintedArtifact(
            key=key, entry=name, artifact=artifact, metadata=meta))
    timings["pack_s"] = round(time.monotonic() - t0, 2)

    logger.info(
        "aot-mint: %s lane=%s -> manifest %s (%d entr%s across %d target(s), "
        "%.1f MB packed, %s)",
        spec.family, spec.execution_lane_label() or "(plain)", manifest,
        len(packed), "y" if len(packed) == 1 else "ies",
        len({row.spec.target for row in minted}),
        sum(row.artifact.stat().st_size for row in packed) / 1e6,
        timings,
    )
    return MintResult(
        entries=tuple(packed), manifest=manifest, timings=timings)


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
        block = meta.get(cell_key.ENTRY_BLOCK_KEY)
        if not isinstance(block, dict):
            raise MintRefused(
                f"held graph class {name!r} carries no entry block, so its "
                f"coverage cannot be folded")
        blocks[name] = block
    absorbed_by = canonicalize_packed_classes(blocks, metas)
    absorbed = {n for names in absorbed_by.values() for n in names}
    survivors = [row for row in held if str(row.entry) not in absorbed]
    manifest = class_manifest(
        {n: b for n, b in blocks.items() if n not in absorbed}, spec)
    for row in survivors:
        row.metadata["manifest_digest"] = manifest
    return MintResult(
        entries=tuple(survivors), manifest=manifest,
        timings={"total_s": 0.0, "held_classes": float(len(held))})


def mint_graph_classes(
    template: aot_compile_pool.EntryJob,
    *,
    workdir: Path,
    width: aot_compile_pool.PoolWidth,
    spec: ExportSpec,
    inductor_configs: Optional[Mapping[str, Any]] = None,
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
    ``execution_lane`` / ``out_dir``), the pool stamps the share and the
    locations, and each child hands back an artifact that is already keyed and
    already carries its envelope. The ExportedProgram is never serialized.

    ``spec`` is the caller's own :class:`ExportSpec` for this family — the same
    object it would have handed :func:`mint`. It is read for exactly one thing
    here: the ``strict``/``lora_bucket`` axes the shared class-hash fold needs
    (:func:`class_manifest`). Nothing about the graphs is derived from it in
    this process; each child derives its own from the pipeline it composed.

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
    progress = MintProgress(
        inductor_configs=inductor_configs, on_progress=on_progress)
    progress.width = width
    _attach_snapshot(progress, phase_snapshot)
    t_mint = time.monotonic()
    progress.t_mint = t_mint
    pool = aot_compile_pool.EntryCompilePool(
        Path(workdir), width=width, inductor_configs=inductor_configs,
        python=python)
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
        block = meta.get(cell_key.ENTRY_BLOCK_KEY)
        if not isinstance(block, dict):
            raise MintRefused(
                f"graph class {name!r}: the packed envelope carries no entry "
                f"block, so its coverage cannot be folded")
        metas[name] = meta
        decoded_blocks[name] = block
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
        held_block = held_meta.get(cell_key.ENTRY_BLOCK_KEY)
        if not isinstance(held_block, dict):
            raise MintRefused(
                f"held graph class {name!r} carries no entry block, so its "
                f"coverage cannot be folded")
        metas[name] = held_meta
        decoded_blocks[name] = held_block
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

    entries: List[MintedArtifact] = []
    blocks: Dict[str, Dict[str, Any]] = {}
    absorbed = {n for names in absorbed_by.values() for n in names}
    for name in sorted(metas):
        if name in absorbed:
            continue
        block = decoded_blocks[name]
        merged = sorted(absorbed_by.get(name) or ())
        if merged:
            # Recorded so the merge is auditable from the result alone — a
            # reader asking "where did class row X go" gets an answer instead
            # of an absence. Same shape `pack_graph_classes` writes.
            block["aliases"] = [
                {"name": alias,
                 "class_dims": [
                     [str(n), int(v)] for n, v in sorted(
                         decoded_blocks[alias].get("class_dims") or ())]}
                for alias in merged
            ]
            logger.info(
                "aot-mint: pgw#917 graph class %s absorbed %d class(es) "
                "(one ingress contract, or one key) (%s) -> %s",
                name, len(merged), ", ".join(merged), keys[name])
        blocks[name] = block
        entries.append(MintedArtifact(
            key=keys[name], entry=name,
            artifact=artifacts[name], metadata=metas[name]))
    # The declaration-wide coverage label, folded HERE because this is the
    # only process that sees every share. Each child stamped its artifact's
    # own `manifest_digest` EMPTY rather than a share-local digest that would
    # read like a whole-declaration one (see `pack_graph_classes`). Folded
    # over the SURVIVORS: an absorbed class contributes the same `class_hash`
    # its survivor already contributes, so folding it as well would make the
    # label depend on how the declaration happened to be sharded.
    manifest = class_manifest(blocks, spec)
    phase_table = _mint_phase_table(
        [], timings, inductor_configs, width, progress.pool_ledger)
    for artifact in entries:
        # `mint_phases` rides the RESULT, never the packed envelope (the
        # artifact deliberately carries no wall clocks), so the whole-mint
        # table replaces the share-local one each child attached.
        artifact.metadata["mint_phases"] = phase_table
        # ...and so does the folded coverage label. The bytes INSIDE each
        # artifact keep the empty stamp its child honestly wrote; this is the
        # result-side view, which is what the publish path reads.
        artifact.metadata["manifest_digest"] = manifest
    logger.info(
        "aot-mint: pgw#1215 %d compile child(ren) packed %d graph class(es) "
        "in %.0fs (sum of child seconds %.0fs, peak child RSS %.1f GiB) -> "
        "manifest %s",
        width.workers, len(entries), time.monotonic() - t_mint,
        sum(pool.entry_seconds.values()), pool.peak_rss_bytes / 1024**3,
        manifest)
    return MintResult(
        entries=tuple(entries), manifest=manifest, timings=timings)


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
    from . import cell_key
    from . import compile_cache as cc

    try:
        runtime = cc.runtime_key()
    except Exception:  # noqa: BLE001 — telemetry never fails a mint
        runtime = {}
    try:
        toolchain = cell_key.toolchain_axis_digest(dict(cc.toolchain_digest()))
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
    return contract, _representative_calls(contract), meta


def _representative_calls(contract: Any) -> Tuple[Dict[str, Any], ...]:
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


#: The identity axes :func:`_class_identity` folds, addressed in the PACKED
#: envelope instead of in the ``_MintedEntry`` it was projected from. Every
#: axis is present — the fold loses nothing — which is what makes the sharded
#: path's merge exactly as strict as the whole-declaration path's:
#:
#: ===================  ==========================================
#: ``_class_identity``  packed envelope
#: ===================  ==========================================
#: ``target``           ``entry.target``
#: ``fork``             ``entry.fork``
#: ``graph``            ``entry.graph_witness`` (the same
#:                      ``graph_hash.graph_hash`` of the program)
#: ``ingress``          ``entry.range_digest``
#: ``pytree``           ``entry.graph.pytree``
#: ``literal_values``   ``entry.graph.literals``
#: ``specialization``   ``entry.graph.specialization``
#: ``lifted_inputs``    ``entry.graph.lifted_inputs``
#: ``precision``        ``metadata.precision``
#: ``lora_bucket``      ``metadata.lora_bucket``
#: ``strict``           ``metadata.strict_export``
#: ``source_digest``    ``metadata.source_digest``
#: ===================  ==========================================
#:
#: ``class_dims`` is deliberately ABSENT: it is the class-row COORDINATE, the
#: one axis two mergeable rows are allowed to differ on.
def _packed_class_identity(
    block: Mapping[str, Any], meta: Mapping[str, Any],
) -> Dict[str, Any]:
    graph = dict(block.get("graph") or {})
    return {
        "target": str(block.get("target") or ""),
        "fork": [[str(n), v] for n, v in (block.get("fork") or [])],
        "graph": str(block.get("graph_witness") or ""),
        "ingress": str(block.get("range_digest") or ""),
        "pytree": graph.get("pytree"),
        "literal_values": graph.get("literals"),
        "specialization": graph.get("specialization"),
        "lifted_inputs": sorted(
            str(n) for n in (graph.get("lifted_inputs") or ())),
        "precision": str(meta.get("precision") or ""),
        "lora_bucket": int(meta.get("lora_bucket") or 0),
        "strict": bool(meta.get("strict_export")),
        "source_digest": str(meta.get("source_digest") or ""),
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
    packed envelope. It is asked of ``aot_serve.contract_from_meta`` and
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
                contract = aot_serve.contract_from_meta(blocks[name])
            except ValueError as exc:
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


def release_mint_residents(
    pipeline: Any, minted: Sequence["_MintedEntry"] = (),
) -> Dict[str, float]:
    """pgw#1053: hand the mint parent's dead residents back to the card.

    From the moment the LAST row exports, neither the composed pipeline nor
    the retained programs' weight aliases do anything for the rest of the mint
    — measured at 16.2 GiB held through the entire 97-minute compile phase on
    the L40S (attempt 30), which with the pgw#992 budget is most of what held
    K at 2. This is the MINT PARENT'S OWN copy: the serving worker's eager
    pipeline lives in a different process and is untouched here — a serving pod
    keeps eager resident per Paul's ruling (GPU hot, eager minimally
    disrupted), and after §4.28 every pod is a serving pod.

    Projection, not deletion. Every retained ``ExportedProgram`` keeps its
    graph, signature, placeholders and LITERAL values (a literal ships inside
    the artifact and is keyed by VALUE — pgw#857), while its state_dict-
    sourced constants — device aliases of the pipeline weights — become meta
    tensors of the same shape and dtype. Every parent-side gate that still
    runs (``program_package_drift``, ``unbindable_constants``,
    ``input_contract``, ``entry_graph_block``, ``_write_literals``, the
    drain-time pgw#917 pass) reads names, shapes and literal bytes only, so
    NO gate is dropped; each runs against the code-only projection. The
    compile children read the STAGED programs from disk, written before this
    runs, byte for byte — nothing about the artifact can move (pgw#846).

    Public since pgw#1215: the K-wide path releases from ``mint_child``, which
    holds a pipeline it will never export from once the compile children trace
    their own shares. ``minted`` defaults to empty for exactly that caller —
    there are no retained programs in that process to project, only the
    pipeline.

    Best-effort in every direction: a tensor or module that refuses the
    projection is skipped, and the release reports what it actually freed. A
    partially released card still hands back what came back. Superseded by
    construction once pgw#1056's fake-weight mint lands — there is then no
    resident to release. pgw#1175 deleted the pool-side regrant it used to
    feed; what remains is a genuine release of memory to the tenant.
    """
    facts: Dict[str, float] = {}
    try:
        import gc

        import torch
    except Exception:  # noqa: BLE001 — no torch, nothing resident
        return facts
    cuda = False
    before = 0
    try:
        cuda = torch.cuda.is_available()
        if cuda:
            before = int(torch.cuda.memory_reserved())
    except Exception:  # noqa: BLE001 — a probe never changes an outcome
        cuda = False

    def _meta(value: Any) -> Any:
        try:
            if isinstance(value, torch.Tensor) and value.device.type != "meta":
                return value.to("meta")
        except Exception:  # noqa: BLE001 — an unmovable tensor stays
            pass
        return value

    for row in minted:
        program = row.program
        weights = set(aot_package.program_state_dict_fqns(program))
        state = getattr(program, "state_dict", None)
        if isinstance(state, dict):
            for fqn in list(state):
                state[fqn] = _meta(state[fqn])
        constants = getattr(program, "constants", None)
        if isinstance(constants, dict):
            # Only the state_dict-sourced entries: everything else is a
            # LITERAL whose bytes still have to reach `_write_literals`.
            for fqn in list(constants):
                if fqn in weights:
                    constants[fqn] = _meta(constants[fqn])
    modules: List[Any] = []
    candidates: List[Any] = [pipeline]
    try:
        candidates.extend(vars(pipeline).values())
    except TypeError:
        pass
    candidates.extend(row.owner for row in minted)
    for obj in candidates:
        if isinstance(obj, torch.nn.Module) \
                and all(obj is not m for m in modules):
            modules.append(obj)
    refused = 0
    for module in modules:
        try:
            module.to("meta")
        except Exception:  # noqa: BLE001 — best effort, reported below
            refused += 1
            logger.debug(
                "aot-mint: pgw#1053 module %s refused the meta projection; "
                "its storage stays resident", type(module).__name__,
                exc_info=True)
    gc.collect()
    facts["residents_release_modules"] = float(len(modules) - refused)
    if cuda:
        try:
            torch.cuda.empty_cache()
            after = int(torch.cuda.memory_reserved())
            facts["residents_released_bytes"] = float(max(0, before - after))
            # The TRUE mint peak, reported before the reset erases it. §4.33
            # target for a full sdxl mint is ~8 GiB; this is the figure that
            # says whether we are there. It is telemetry — nothing sizes
            # anything from it.
            facts["peak_vram_before_release_bytes"] = float(
                torch.cuda.max_memory_allocated())
            torch.cuda.reset_peak_memory_stats()
        except Exception:  # noqa: BLE001 — a probe never changes an outcome
            pass
    logger.info(
        "aot-mint: pgw#1053 mint residents released — %d module(s) projected "
        "to meta, %.2f GiB handed back",
        len(modules) - refused,
        facts.get("residents_released_bytes", 0.0) / 1024**3)
    return facts


def _structure_only_drift_hint(row: _MintedEntry) -> str:
    """Name the AUTHORING cause a drift refusal usually has.

    pgw#1097 WIDENED this from structure-only to EVERY mint, because the cause
    stopped being structure-only. With the folding fence on, nothing is
    inlined, so a plain-attribute table reaches the artifact's constant table
    under AOTInductor's own name (``_tensor_constant0``) while the exported
    program lifted it under its ATTRIBUTE PATH (``_table``) — measured — and
    the two no longer reconcile. That was always fatal for a literal large
    enough to be declared; the fence merely removes the size threshold
    (0-dim, or 1-D with <=8 elements) that used to hide it by baking the
    values instead. The fix is the same one the tensor-binding contract has
    always asked for, so the message points at it.

    Measured (pgw#1080, micro-rope RED control): a table built lazily inside
    ``forward`` under ``with torch.device("cpu")`` is FAKE during a
    fake-mode export — so the meta-instantiation gate cannot see it — and
    lands as an anonymous ``_tensor_constant0`` the exported program never
    lifted. The drift gate refuses it, correctly and deterministically, but
    "the package declares a constant the program never lifted" names a
    symptom. This names the class of cause, so the author gets a place to
    look instead of a compiler sentence.
    """
    hint = (
        ". The usual cause is a tensor BUILT INSIDE `forward`, or held as a "
        "PLAIN ATTRIBUTE, instead of registered at __init__: export lifts it "
        "under its attribute path while the compiled artifact names it "
        "`_tensor_constant0`, so nothing can reconcile the two and nothing "
        "could bind it. Register derived tables with `register_buffer` and no "
        "device pin (ie#630's `rope_buffers` is the worked example; pgw#857 "
        "is the contract, pgw#1097 is why it is now load-bearing at every "
        "size rather than only above the inlining threshold)")
    if structure_only.is_structure_only(row.owner):
        return (
            ". This mint built its target from code + config (pgw#1080)" + hint)
    return hint


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
            f"entry {entry!r}: constant-set drift: " + "; ".join(drift)
            + _structure_only_drift_hint(row))
    # pgw#1097, THE FOLDING FENCE. `CONSTANT_BINDING_CONFIGS` is what prevents
    # a weight's values from being compiled in; this is what PROVES it, per
    # entry, against the artifact's own table. Without the proof the setting is
    # a hope: an inlining route torch adds tomorrow would ship a cell that
    # serves the minting checkpoint's tensor to every other fine-tune, and the
    # only thing to notice would be the adopt-side parity floor refusing that
    # cell on every checkpoint but one — cell sharing dying quietly.
    folded = aot_package.folded_weights(
        row.program, package, _state_dict_keys(row.owner), entry)
    if folded:
        raise MintRefused(
            f"entry {entry!r}: folding fence (pgw#1097): " + "; ".join(folded))
    fused = aot_package.eliminated_constants(row.program, package, entry)
    if fused:
        # What is LEFT here is anonymous graph literals, never a weight — the
        # fence above has already refused those. Recorded, never fatal, but a
        # surprising jump in the count should be visible rather than silently
        # discarded.
        logger.info(
            "aot-mint: %s: %d lifted literal(s) folded away by the compiler "
            "(e.g. %s)", entry, len(fused), fused[:3])
    try:
        block = keying_block(row.program, row.flat_leaves, row.spec)
        constants = aot_package.constants_manifest(package, entry)
        # pgw#1058: the manifest rows this envelope will carry, proven against
        # the artifact's OWN generated input guards before anything can
        # publish. Same doctrine as the constant manifest: two independent
        # readings (program vs generated wrapper) that must agree, so a label
        # that drifted from its artifact fails closed HERE, not as an opaque
        # 36/36 admission miss on every adopting pod.
        admission = aot_package.admission_drift(package, entry, block["inputs"])
    except aot_package.PackageIntrospectionError as exc:
        raise MintRefused(f"entry {entry!r}: declaration: {exc}") from exc
    if admission:
        raise MintRefused(
            f"entry {entry!r}: admission drift (pgw#1058): "
            + "; ".join(admission[:6]))
    # The manifest is RECORDED, never keyed (`aot_serve.class_hash` folds
    # target/fork/class_dims/range_digest/graph/strict/lora_bucket and nothing
    # else) — which is why the boot-side derivation can state an entry's
    # identity while carrying an empty one.
    block["constants"] = constants
    return block


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
    #: This row's own ``export_s`` / ``compile_s``, straight off
    #: ``_export_entry``. The key path ignores them; the pgw#1134 measure-only
    #: child is here for exactly these numbers.
    timings: Dict[str, Any] = field(default_factory=dict)
    #: Loose inductor files, when the caller asked for the compile. The caller
    #: owns them — the measure-only child counts and deletes them, and the key
    #: path never asks for a compile so it never sees any.
    files: Tuple[str, ...] = ()
    #: pgw#1215: the full ``_MintedEntry`` this row was projected from, held so
    #: a caller that COMPILED can also PACK. ``pack_graph_classes`` takes these
    #: rows — it is the one packager, and a compile child that had only the
    #: projection would have to re-implement packaging from a program it was
    #: handed a copy of. Two packagers is the divergence per-graph-class
    #: identity exists to rule out.
    #:
    #: It is the largest object a caller holds. A key-only caller drops it (and
    #: the program) with :meth:`release` the moment the block is read.
    row: Any = None

    def release(self) -> None:
        """Drop everything but the KEYING facts.

        ``boot_trace_child`` and ``measure_child`` want the block, the node
        count and the timings; the program and the minted row are megabytes
        apiece and nothing downstream of them reads either. One method rather
        than an assignment at each call site, because there are now two things
        to drop and a caller that dropped one of them would look correct.
        """
        self.program = None
        self.row = None


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
    compile_now: bool = False,
    inductor_configs: Optional[Mapping[str, Any]] = None,
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

    ``compile_now`` (pgw#1134) runs the INDUCTOR half of each row as well —
    ``_export_entry``'s own compile, the one a mint runs — and is what the
    measure-only child needs: an export-only trace never exercises the
    whole-graph planner an OOM blocker is about, so a measurement taken
    without it answers a different question than the one asked. It rides THIS
    loop rather than a second one for the reason the docstring above gives:
    two loops trace two graphs, and a measurement of a graph the mint does not
    export is worth nothing. The key path never passes it, and a compiled row
    hands its loose files to the caller (``TracedClass.files``) — this
    function keeps none of them, and nothing here packages anything — but the
    caller can: since pgw#1215 every yielded row carries the ``_MintedEntry``
    it was projected from (``TracedClass.row``), which is what
    :func:`pack_graph_classes` takes. That is the whole of th#1834 Phase 3's
    keystone: the process that traces a class holds everything needed to
    compile AND pack it, so no ``ExportedProgram`` ever crosses a process
    boundary. A caller that wants only the key calls ``TracedClass.release()``.
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
                row = _export_entry(
                    pipeline, spec, plan, decl, compile_now=bool(compile_now),
                    inductor_configs=inductor_configs)
                try:
                    nodes = int(len(row.program.graph_module.graph.nodes))
                except Exception:  # noqa: BLE001 — never fails a trace
                    nodes = 0
                # pgw#1087's owed item: a class's trace cost is meaningless
                # without the graph size it paid for.
                span.note(f"nodes={nodes}")
            yield TracedClass(
                name=entry,
                block=keying_block(row.program, row.flat_leaves, row.spec),
                nodes=nodes,
                program=row.program,
                declared=declared,
                timings=dict(row.timings or {}),
                files=tuple(str(f) for f in (row.files or ())),
                row=row,
            )
    finally:
        if disarmed:
            _arm_branches(pipeline, int(spec.lora_bucket or 0))


def keying_block(
    program: Any, flat_leaves: Sequence[Any], spec: ExportSpec,
) -> Dict[str, Any]:
    """The entry-envelope fields that reach an entry's ``class_hash`` — built
    from the EXPORTED PROGRAM and the declaration, and from nothing else.

    ONE construction, shared by the mint (:func:`_entry_block`, which adds the
    package-side ``constants`` manifest afterwards) and by the boot-side
    derivation (``boot_key``, which has no package and carries the manifest
    empty). Two constructions of the same block would be exactly the
    attempt-28 phantom in a new hat: a declared-facts key beside a traced-facts
    key under one axis name.

    ``constants`` is present-but-empty rather than absent because
    ``aot_serve.entries_from_meta`` validates every block as a full contract,
    and an entry that cannot be parsed cannot be keyed.

    ``graph_witness`` (pgw#1031) is the node-level digest of the traced
    program (``graph_hash.graph_hash``). It is recorded as a top-level SIBLING
    of ``graph`` AND folded into the key: since pgw#1031 (option a, Paul-ruled)
    ``aot_serve.class_hash`` (facts v3) folds it, so the ``graph`` axis is the
    traced COMPUTATION, not merely the traced ingress. It was measured that the
    interface alone could not separate two bodies — 2026-08-10, ``micro-pad32``
    and ``micro-pad32-branchy`` produced a byte-identical keying block
    (identical signature, symbol ranges, pytree spec, constant FQNs and
    declared envelope) from 112- and 102-node graphs, one key, two artifacts.
    The witness closes that at the key: two bodies key apart, a collision is a
    MISS (eager + mint). It stays a top-level field for the adopt backstop
    (``aot_identity.verify_graph_witness``, defense-in-depth), which refuses a
    materialized cell whose recorded witness is not this pod's graph.
    """
    inputs, symbols = aot_package.input_contract(program, flat_leaves)
    block: Dict[str, Any] = {
        "target": spec.target,
        "fork": [[str(n), v] for n, v in sorted(spec.fork)],
        "class_dims": [
            [str(n), int(v)] for n, v in sorted(spec.class_dims)],
        "inputs": inputs,
        "symbols": symbols,
        "constants": [],
        "graph": entry_graph_block(program, spec),
        "graph_witness": graph_hash.graph_hash(program),
    }
    placement = graph_hash.device_placement(program)
    if len(placement) > 1:
        # pgw#1113 / pgw#819: the program's own device map put its modules on
        # more than one card, and inductor bakes that placement into the
        # artifact. Recorded — and keyed, in `aot_serve.class_hash` — ONLY
        # here, on the `excluded_inputs` precedent above: a single-device
        # program states nothing, so no published cell's block moves and no
        # published cell re-keys.
        block["placement"] = list(placement)
    if adapter_arm(spec.fork) is False:
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


def entry_graph_block(program: Any, spec: ExportSpec) -> Dict[str, Any]:
    """The per-entry graph-interface facts (fold into that entry's
    ``class_hash``): the lifted constant FQN set, the lifted inputs, the
    pytree spec, and the python branches export FROZE at trace time.
    Constant BYTE SIZES are deliberately absent — they are a property of the
    resident weights, and a fine-tune of one family must keep sharing
    cells, which is the premise of family-scoped cells.

    **v3 (pgw#1089): every fact here comes from the EXPORTED PROGRAM, never
    from the compiled package.** v2 read ``constant_fqns`` off the packaged
    artifact and carried ``fused_constants`` (the constants the compiler folded
    away), so an entry's identity could not be stated until after its compile.
    Two consequences, one of which was already live:

    * **A weightless mint and a real-weight mint of the IDENTICAL graph keyed
      differently.** pgw#1080 compiles structure-only entries with
      ``aot_inductor.use_runtime_constant_folding`` (it must — a compile-time
      fold bakes fake values), which keeps every parameter bindable and adds
      ``_FOLDED_CONST_*`` rows. Both package-side sets therefore move, so the
      same traced graph produced two different ``ck1`` keys depending on how
      its mint happened to obtain its weights. That is precisely the fused-axis
      failure the membership axiom forbids.
    * **The key could not be derived before the artifact existed**, which makes
      §4.27 step 1 (derive on boot, from code alone) impossible by
      construction.

    Both facts are a FUNCTION of (graph x toolchain x sm) — the same program
    compiled by the same toolchain on the same architecture folds the same
    constants — so they carry zero information the key does not already hold,
    and the axiom admits nothing that does. They are not lost: the mint still
    PROVES them (``aot_package.program_package_drift`` refuses a package whose
    constant set disagrees with its program; ``eliminated_constants`` is still
    logged per entry). Proven, not keyed.

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
        "v": 3,
        "constant_fqns": sorted(aot_package.program_constant_fqns(program)),
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
        # pgw#1059: `mode` and `shell_digest` are DELETED (both pinned ""
        # since regional died, #846 — recording a constant that says
        # "unchanged" was only ever keyed byte-compatibility with the fused
        # v3 contract facts, which die in the same redefinition).
        "sm": str(cc.runtime_key().get("sm") or ""),
        # The seal dict stays RECORDED (the observable statement of the
        # declaration this cell was minted under; its digest is a published
        # wire fact the hub's ArtifactIdentity requires) — but it is no
        # longer a key axis: the declaration + loaded-libs digests fold into
        # the `toolchain` block below (pgw#1059 amendment 4).
        env_seal.SEAL_KEY: env_seal.effective_seal(),
        "toolchain": dict(cc.toolchain_digest()),
        # pgw#1046/pgw#1059: the DECLARED ENVELOPE — the `envelope` axis's
        # whole input, recorded so an exported cell can restate its own key
        # from the artifact alone (`cell_key.from_exported_artifact_metadata`;
        # the publish path recomputes the same axes before a byte moves).
        # Canonical form is `cell_key.envelope_facts`; the behavior-posture
        # `overlay` slot (amendment 5) is absent because the overlay menu is
        # empty.
        cell_key.EXPORT_ENVELOPE_KEY: {
            "shapes": sorted([int(v) for v in row] for row in spec.shapes),
            "text_lens": sorted({int(v) for v in spec.text_lens}),
            "guidance": sorted(float(v) for v in spec.guidance_scales),
        },
        # pgw#1034: no ``code_closure``. pgw#990 took source content out of the
        # key; the memo it was demoted to is `compile_cache`'s own block, read
        # by the local re-trace off ITS copy. This one had zero readers and cost
        # a per-mint AST walk of the whole import closure plus declare size.
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


def cell_identity(meta: Mapping[str, Any]) -> cell_key.CellKey:
    """The ``cg-key-v1`` key ONE entry artifact's OWN recorded facts describe.

    The computation is :func:`cell_key.from_entry_metadata` — ONE
    implementation, so the key the mint stamps and the axes the publish path
    declares (``fleet_cells._identity_axes``) are the same object rather than
    two derivations that can drift. Every input is a RECORDED block, which is
    what makes the recomputation possible off the artifact alone.

    A missing fact is a :class:`MintRefused` here because at mint time it means
    this pod cannot name its own product; the same absence at publish time is a
    publish refusal, not a fallback.
    """
    try:
        return cell_key.from_entry_metadata(meta)
    except cell_key.CellKeyError as exc:
        raise MintRefused(str(exc)) from exc


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


def publish(result: MintResult, publisher: Any) -> Dict[str, str]:
    """Publish every compiled graph a mint produced. ``{key -> checkpoint}``.

    ONE attested intent for the whole mint (pgw#1224 / th#1842 PR #1121), then
    one transfer per entry under that entry's OWN token. This is the caller the
    batch wire exists for: it holds every artifact of a mint at once, so a
    36-class sdxl mint pays one axis attestation instead of 36. The transfers
    stay per entry because the grants are — a token for entry 5 cannot publish
    entry 6's bytes.

    Failures are NOT swallowed and not collected: the first refusal raises. A
    caller that wants best-effort per-entry publishing drives the two halves
    itself — one ``publish_intent`` for the batch, then ``publish_granted`` per
    entry — and decides what a partial set means. (``publish_entry`` used to be
    that seam and is DELETED with the per-entry intent it wrapped: under the
    batch wire it would have issued one attested intent per artifact, which is
    the cost this change exists to remove.) What changed under pgw#1176 is that
    a partial set is now a coherent outcome rather than a broken cell.

    Receipts are the HUB's business: it adds them at publish-finalize (#709),
    so the producer's whole obligation is a keyed ``metadata.json`` inside the
    tar — which :func:`mint` has already stamped and proven. An artifact with
    no key is refused before the wire, since an unaddressable entry would be
    stored under a flavor nothing can request.
    """
    from . import fleet_cells

    rows = list(result.entries)
    if not rows:
        return {}
    # th#1355: the mint pod already measured this (timings["total_s"]), so the
    # cell's own cell_store row records what it cost to build instead of the
    # cost living only in an activity event that carries no cell key.
    mint_duration_ms = max(0, int(round(
        float(result.timings.get("total_s") or 0.0) * 1000)))
    family = ""
    entries = []
    sku = gen_worker = ""
    for row in rows:
        if not row.key:
            raise MintRefused("cannot publish an artifact with no cell_key")
        row_family = str(row.metadata.get("family") or "")
        if not row_family:
            raise MintRefused("cannot publish an artifact with no family")
        if family and row_family != family:
            # The family is the batch's namespace and the hub attests it once.
            # A mint spanning two would have to be two intents, and silently
            # publishing the second under the first's declaration is how a row
            # lands in a namespace nobody asked for.
            raise MintRefused(
                f"this mint's entries name two families ({family!r} and "
                f"{row_family!r}); one intent declares one family")
        family = row_family
        entry, row_sku, row_gen_worker = fleet_cells.intent_entry(
            family, dict(row.metadata), mint_duration_ms)
        sku = sku or row_sku
        gen_worker = gen_worker or row_gen_worker
        entries.append(entry)
    batch = publisher.publish_intent(
        family, entries, sku=sku, gen_worker=gen_worker)
    return {
        row.key: str(publisher.publish_granted(
            family, row.artifact, dict(row.metadata),
            batch.grant_for(row.key), repo=batch.repo))
        for row in rows
    }


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
        # pgw#1076: NO default. An absent `precision` is derived from the
        # modules the mint actually traces (`_measured_precision`); a
        # fabricated "bf16" is a measurement nobody made.
        precision=str(body.get("precision") or ""),
        lora_bucket=int(body.get("lora_bucket") or 0),
        shapes=tuple(tuple(int(v) for v in row) for row in body.get("shapes") or ()),
        text_lens=tuple(int(v) for v in body.get("text_lens") or ()),
        guidance_scales=tuple(float(v) for v in body.get("guidance_scales") or ()),
        specialization=dict(body.get("specialization") or {}),
        lora_fqns=tuple(str(v) for v in body.get("lora_fqns") or ()),
        lifted_inputs=tuple(str(v) for v in body.get("lifted_inputs") or ()),
        strict=bool(body.get("strict", True)),
        source_ref=str(body.get("source_ref") or ""),
        source_digest=str(body.get("source_digest") or ""),
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
                        help="refuse without a C++ AOTI toolchain (default on)")
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
    if args.require_toolchain and not cxx_toolchain_present():
        # pgw#900: an AOTI mint links a real `.so` through inductor's C++
        # wrapper, so a C-only image (`cc`/`gcc` but no working C++) passes
        # `toolchain_present()` yet dies 336 s later at the linker
        # (`InvalidCxxCompiler`, measured on L4 0.84.0). Gate on the SAME
        # predicate the mint child uses (`cxx_toolchain_present`), not the
        # dynamo-lane `toolchain_present`, so the CLI refuses at second zero.
        print(
            "REFUSED: no C++ AOTI toolchain — inductor cannot link a kernel "
            "on this image. An AOTI mint needs the compile-job image, not a "
            "prod worker image", file=sys.stderr)
        return 2

    model = args.model or spec.source_ref
    if not model:
        print("BAD REQUEST: no --model and the request has no source_ref",
              file=sys.stderr)
        return 3
    try:
        pipeline, _build_inputs = compose_for_mint(model, spec, body)
        # pgw#1053: an operator mint's pipeline is composed for the mint and
        # dies with the process — surrender it once the last row exports.
        result = mint(pipeline, spec, Path(args.out), release_residents=True)
    except MintRefused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({
        "manifest": result.manifest,
        "entries": [
            {"entry": row.entry, "key": row.key, "artifact": str(row.artifact)}
            for row in sorted(result.entries, key=lambda r: r.entry)
        ],
        "timings": result.timings,
    }, indent=1))

    if args.publish:
        try:
            checkpoints = publish(result, _publisher_from_settings())
        except MintRefused as exc:
            print(f"REFUSED: {exc}", file=sys.stderr)
            return 2
        for key, checkpoint in sorted(checkpoints.items()):
            print(f"published {key} -> checkpoint {checkpoint}")
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
    "MintedArtifact",
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
    "keying_block",
    "trace_for_key",
    "declared_class_rows",
    "TracedClass",
    "export_program",
    "exported_input_names",
    "shared_identity_blocks",
    "LIFTED_LORA_TORCH_FLOOR",
    "lifted_input_gaps",
    "main",
    "mint",
    "package_cell",
    "publish",
]
