"""Worker serving policy around TCG-owned compiled graph artifacts.

TCG's Engine is the sole artifact import, resolve, extraction, runner and
constant-binding authority. This module adds the worker policy that belongs
outside that engine: input ingress checks and normalization, per-class
dispatch, eager fallback, sticky de-arm, shape-growth reporting and live
serve-state introspection.

An arm resolves and creates a runner at the same destination, binds the
resident module tensors, and only then mutates the live module. Each compiled
graph class is independently visible through :func:`entry_states`; no
worker-owned archive, extraction tree, package loader or compatibility store
exists here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple,
)

from gen_worker._vendor.torchcg import (
    GRAPH_CLASS_BLOCK,
    CallIngress,
    CallInput,
    CompiledGraphRunner,
    ConstantBindingError,
    IngressError,
    StoreOutcome,
    is_compiled_graph_key,
)

from . import activity as activity_mod
from . import aot_identity
from . import local_cell_store
from .cell_adopt import AdoptOutcome
from . import serve_posture
from . import shape_growth
from .compile_cache import (
    AdoptError,
    CompiledExecutionLaneUnavailableError,
    _resolve_target,
    parse_cell_ref,
)
from .models import lora_lifted
from .models.cache_paths import open_worker_engine, tensorhub_cas_dir
from .models.memory import is_cuda_oom

logger = logging.getLogger(__name__)
#: pgw#791: an input the artifact was compiled for as 16-byte aligned arrived
#: unaligned (or non-contiguous) and this ingress realigned it. Typed and
#: hub-visible because the ALTERNATIVE is what shipped: AOTInductor's own
#: ``run_impl`` copies the input on EVERY call and says so on the worker's
#: stderr, which is unreachable on hub-spawned pods — a fleet paying the tax
#: was indistinguishable from one that was not. Coalesced: once per
#: (entry, input, reason).
REALIGN_EVENT = "aot_input_realigned"
#: pgw#1074: a rank-0 input arrived in an INTEGER dtype where the graph is
#: specialized on float32/float64, and this ingress recast it. Same shape and
#: same doctrine as :data:`REALIGN_EVENT` — the ingress normalizes the feed to
#: the artifact's contract and SAYS so, once per (entry, input, dtype pair).
RECAST_EVENT = "aot_input_recast"
#: AOTInductor generates its aligned-input fast path at 16 bytes
#: (``torch._inductor.codegen.aoti_runtime``'s ALIGNMENT). An input whose
#: ``data_ptr()`` is not a multiple of this — diffusers hands the denoiser
#: ``timesteps[i]``, a scalar VIEW at an odd element offset — makes the
#: runner clone it per call. Not a knob: it is the compiler's constant.
AOTI_ALIGNMENT = 16
#: THE compiled-graph artifact metadata/package version. v1 = ONE graph class
#: per artifact: TCG metadata carries one ``graph_class`` block, never an
#: ``entries`` map.
#:
#: DESIGN-RULINGS §1.38b (Paul, 2026-08-13): *"we are pre-launch, so we should
#: be on v1 for everything, including our compiled-graph format."* The version
#: records the FIRST CONTRACT WE SHIP, not the number of abandoned internal
#: prototypes — carrying the old `ARTIFACT_FORMAT = 3` forward would turn
#: pre-launch history into a permanent public constraint for no benefit. The
#: surviving compiled-graph boundaries each begin at their own v1 and are
#: INDEPENDENT: `cg-key-v1` (key grammar), this (artifact metadata/package),
#: the manifest schema, the local CAS record schema, and the hub's
#: resolve/publish route. A future artifact v2 does not make the key grammar
#: v2.
#:
#: HARD CUT, not compatibility: formats 1/2/3 of the deleted cell/bundle
#: implementations have no reader here. There is no 3->1 mapping and no
#: accepted set: TCG Engine validates the artifact format while importing and
#: resolving it, so this worker cannot consume a retired package by accident.
#: The window is real and was measured before this shipped: the old store held
#: 0 rows and no `cg-key-v1` object existed anywhere durable.
#:
#: The name is QUALIFIED on purpose. §1.38b: *"use qualified names such as
#: `compiled_graph_format`, never the generic `format` that let pgw#1230
#: compare the AOT package schema with an unrelated torch-inductor
#: semantic-cache epoch."* That collision cost a fully compiled 4-class mint.
COMPILED_GRAPH_FORMAT = 1

#: The metadata KEY the value above is stamped under, and the arm axis name.
#: One symbol so the stamp and the comparison cannot drift into two spellings
#: — the pgw#1230 failure mode, one level up.
COMPILED_GRAPH_FORMAT_KEY = "compiled_graph_format"
_MARKER_ATTR = "_cozy_aot"

#: The hardware/toolchain axes an ``.pt2`` is genuinely pinned to (pgw#765).
#: ``sm`` is the GPU identity: AOTInductor itself keys on
#: ``AOTI_COMPUTE_CAPABILITY`` — capability, never the marketing name
#: (``codecache.get_device_information``). ``sku`` is deliberately ABSENT:
#: Paul's ruling ("AOT cells are locked into the sm_x version, not the actual
#: GPU"), the pgw#691 collapse that removed it from cell identity on
#: byte-identical evidence, and the pgw#754 ISA clamp that made the host half
#: portable. It stays in metadata for observability — never as a refusal.
IDENTITY_AXES: Tuple[str, ...] = ("sm", "torch", "cuda")


# ---------------------------------------------------------------------------
# Typed refusals
# ---------------------------------------------------------------------------


class ConstantsUnboundError(RuntimeError):
    """A code-only artifact was asked to run before its constants were
    proven bound (pgw#704 B1).

    This exception EXISTS so the segfault does not: reaching
    ``AOTICompiledModel.__call__`` with unbound constants takes down the
    whole worker process, which no ``except`` can recover. Raised by name,
    before any call crosses into the compiled extension.
    """

    def __init__(self, reason: str, detail: str = "") -> None:
        self.reason = reason
        super().__init__(detail or reason)


class IngressContractError(RuntimeError):
    """A call's shapes/dtypes fall outside the artifact's DECLARED contract
    (pgw#704 B2).

    The exported graph will happily run it and return numerically
    unvalidated output. Named refusal instead; the caller serves the request
    eagerly, so the request succeeds and the deviation is observable.
    """

    def __init__(self, reason: str, detail: str = "") -> None:
        self.reason = reason
        super().__init__(detail or reason)


def is_aot_ref(ref: str, family: str = "") -> bool:
    """Whether ``ref`` names a TCG compiled graph, optionally for ``family``."""
    ref_family, key = parse_cell_ref(ref)
    return bool(
        ref_family
        and (not family or ref_family == family)
        and is_compiled_graph_key(key)
    )


# ---------------------------------------------------------------------------
# The declared contract — shapes, symbols, constants
# ---------------------------------------------------------------------------


def entry_from_meta(meta: Mapping[str, Any]) -> Dict[str, Any]:
    """Return one graph class in the shape used by the numerics probe.

    pgw#1176: the plural ``entries_from_meta`` is GONE with the multi-entry
    artifact. A caller that wants several entries holds several artifacts.
    """
    graph_class = meta.get(GRAPH_CLASS_BLOCK)
    if not isinstance(graph_class, Mapping):
        raise ValueError("compiled graph metadata has no graph_class")
    graph = graph_class.get("graph")
    if not isinstance(graph, Mapping):
        raise ValueError("compiled graph class has no graph contract")
    return {
        **dict(graph),
        "name": str(graph_class.get("name") or ""),
        "target": str(graph_class.get("target") or ""),
        "fork": list(graph_class.get("fork") or ()),
        "class_dims": list(graph_class.get("class_dims") or ()),
    }


# ---------------------------------------------------------------------------
# B2 — the ingress range assertion
# ---------------------------------------------------------------------------


def _dtype_name(value: Any) -> str:
    """``torch.bfloat16`` -> ``bfloat16``; anything else -> ''."""
    raw = str(getattr(value, "dtype", "") or "")
    if not raw:
        return ""
    return raw.split(".")[-1]


def bind_call_inputs(
    contract: CallIngress, args: Sequence[Any], kwargs: Mapping[str, Any],
) -> Dict[str, Any]:
    """Match one call's actual arguments to the declared inputs by REPLAYING
    each input's recorded identity in the call. Missing non-optional input =>
    named refusal.

    THE RULE (pgw#994): an input is found at ``kwargs[param]`` — or at
    ``args[param_position]`` — followed by its ``path`` into that argument.
    TCG built and identity-hashed that mapping from the exported call, and the
    same :class:`CallIngress` value resolves it here.

    WHAT THIS REPLACES, because both halves were luck. The old resolution
    tried ``kwargs[name]``, then ``args[position]``, then a SEARCH inside any
    mapping-valued kwarg for ``name``.

    * ``position`` counts FLATTENED graph inputs, but ``args`` is the call
      BEFORE flattening. sdxl escaped because diffusers passes its dict by
      keyword and the positional branch never fired. z-image, whose ``x`` is
      a ``list[Tensor]``, could not escape: ``x.0`` bound the whole list,
      ``x.1`` bound the NEXT argument, and the last input then refused
      ``input_missing`` — measured off-GPU, and it is what pgw#994 filed.
    * the nested search found ``text_embeds`` in ``added_cond_kwargs``
      because no other kwarg happened to carry that key. It is deleted here:
      the contract now SAYS where the value is, so nothing needs to hunt for
      it, and a dict that carries a colliding key can no longer decide a bind.
    """
    try:
        return dict(contract.bind(args, kwargs))
    except IngressError as exc:
        raise IngressContractError(exc.reason, str(exc)) from exc


def marshal_positional(
    contract: CallIngress,
    args: Sequence[Any],
    kwargs: Mapping[str, Any],
) -> List[Any]:
    """Flatten one call into the package's POSITIONAL signature.

    MEASURED on the first-light pod (pgw#721 S8 / #723): an AOTI package
    takes **positional inputs only** — splatting the pipeline's kwargs at it
    dies with ``Ran into a kwarg keyword mismatch: Got [...] but expected
    []`` before any compiled code runs. Diffusers, meanwhile, calls a
    denoiser with a mix of positional and (nested) keyword arguments, so the
    serve path cannot pass the call through untouched: it has to flatten to
    the exact order export recorded, or it feeds the right tensor to the
    wrong graph input.

    THE COMPLEMENT, measured on the pgw#723 residuals pod (2.13.0+cu130):
    the package's call convention mirrors the traced EXAMPLE's args/kwargs
    split. An input fed as a KWARG at export bakes a kwarg-demanding
    ``in_spec`` — the package then refuses a positional call with the same
    error in reverse (``Got [] but expected ['lora_a', 'lora_b']``), the
    wrap treats it as an artifact failure, and the lane silently revokes to
    eager on the FIRST call. So "positional inputs only" is a MINT
    obligation, not a torch invariant: example inputs must be fed
    all-positionally (lifted adapter pair included) so this positional
    marshal matches the package.

    ``position`` in the declared contract is that order. Every declared
    input must be present — a package has a FIXED flat arity, so a missing
    one cannot be skipped without silently shifting every later argument
    into the wrong slot. That is a named refusal, not a best effort.
    """
    try:
        return list(contract.feeds(args, kwargs))
    except IngressError as exc:
        raise IngressContractError(exc.reason, str(exc)) from exc


def excluded_inputs_present(
    contract: CallIngress, kwargs: Mapping[str, Any],
) -> Tuple[str, ...]:
    """The contract's EXCLUDED inputs this call actually carries (pgw#790).

    Keyword and nested-mapping only: an excluded input has no position in this
    graph's signature, so a positional index would name a different argument
    entirely. A ``None`` value does not count as carrying it — that is how a
    diffusers pipeline says "absent".

    This still SEARCHES, and deliberately (pgw#994, which deleted the search
    in :func:`bind_call_inputs`). The two questions are not the same one: a
    declared input has a contract row, so its location is recorded and can be
    replayed; an EXCLUDED input has no row by definition, and the question
    asked of it is "does this call carry such a value anywhere" — for which
    there is nothing to replay. A diffusers pipeline can hand an adapter down
    nested (``cross_attention_kwargs``), and a branchless class that missed it
    would silently serve the base model.
    """
    if not contract.excluded_inputs:
        return ()
    found: List[str] = []
    for name in contract.excluded_inputs:
        value = kwargs.get(name, None)
        if value is None:
            value = next(
                (v[name] for v in kwargs.values()
                 if isinstance(v, Mapping) and v.get(name) is not None),
                None)
        if value is not None:
            found.append(name)
    return tuple(found)


#: pgw#1074: the ONLY dtype normalization this ingress performs. Integer ->
#: float32/float64 on a rank-0 input, and nothing else.
#:
#: float32 represents every integer up to 2**24 exactly, so the recast is
#: value-preserving over any diffusion timestep domain, and it is EXACTLY the
#: op the traced graph's own first node performs on this input (diffusers
#: ``get_timestep_embedding``: ``timesteps[:, None].float()``) — so a recast
#: feed produces bit-identical output to the eager call that presented the
#: integer. bfloat16 and float16 are deliberately NOT targets: bf16 has 8
#: mantissa bits and would round timestep 999 to 1000, which is a numeric
#: change, not a normalization.
RECAST_TARGETS = ("float32", "float64")
_INTEGER_DTYPES = ("int8", "int16", "int32", "int64", "uint8")


def recast_gap(spec: CallInput, value: Any) -> str:
    """The named dtype normalization this feed needs, or ``''`` (pgw#1074).

    ``''`` means either "already in contract" or "must be REFUSED" — this
    function never widens the contract, it only names the one normalization
    the ingress is allowed to perform, and :func:`ingress_report` refuses
    every other dtype disagreement exactly as before.

    **Why this exists.** The dtype a diffusers denoiser is handed for its
    scalar timestep is a per-request SAMPLER fact, not a family fact. Measured
    over ``gen_worker.view.SAMPLERS`` on the fleet's own diffusers (pgw#1074):
    ``euler``/``euler_a``/``euler_trailing``/``heun``/``flow_euler`` present
    **float32**; ``ddim``/``ddim_trailing``/``ddpm``/``deis``/``dpmpp_2m*``/
    ``lcm``/``unipc`` present **int64** (``set_timesteps`` ends in
    ``.to(dtype=torch.int64)``). sdxl's cell was minted float32 (ie#627, and
    correct — it is what the graph is specialized on), so its turbo arm
    (``euler_trailing``) served from the cell and its base arm, on an int64
    sampler, was refused ``no_entry_admits`` by an entry that covered it in
    every other respect. No single declared dtype can be right for a family
    whose sampler is per-request VIEW state, and the sampler is deliberately
    not a compile axis — so the normalization belongs at the boundary that
    knows the contract, once, named and counted.
    """
    if spec.shape:  # rank-0 only: the timestep class, not a tensor of values
        return ""
    if spec.dtype not in RECAST_TARGETS:
        return ""
    got = _dtype_name(value)
    if got not in _INTEGER_DTYPES:
        return ""
    if getattr(value, "shape", None) is None or tuple(value.shape):
        return ""
    return f"{got}_to_{spec.dtype}"


def alignment_gap(value: Any) -> str:
    """'' when a feed satisfies the artifact's aligned-input contract, else
    the named reason (pgw#791).

    AOTInductor compiles the fast path for 16-byte-aligned, contiguous
    inputs. When the run-time pointer is not aligned its ``run_impl`` copies
    the tensor to an aligned buffer ON EVERY CALL and reports it with a C++
    ``TORCH_WARN`` — i.e. on the worker's stderr, which hub-spawned pods do
    not expose. Measured on an RTX 4090 (WARM-INFERENCE-MATRIX §2c): the
    request residual over 28x(per-forward) is 196 ms for the armed AOT
    artifact against 77 ms for the equivalent dynamo cell, and diffusers'
    ``timesteps[i]`` scalar view is the offending input. So the check the
    serve path never performed cost more than the artifact kind was worth.
    """
    ptr = getattr(value, "data_ptr", None)
    if not callable(ptr):
        return ""
    is_contig = getattr(value, "is_contiguous", None)
    if callable(is_contig) and not is_contig():
        return "non_contiguous"
    try:
        if int(ptr()) % AOTI_ALIGNMENT:
            return "unaligned_16b"
    except (RuntimeError, ValueError):  # pragma: no cover — meta/fake tensors
        return ""
    return ""


class FeedAligner:
    """Owned aligned staging buffers for one runner's declared inputs.

    The fix pgw#791 asks for is "align once at ingress rather than copying
    per call", and for a value that CHANGES every call (the timestep does)
    the honest reading is: allocate once, copy the value. The allocation —
    which is what the runner repeats, along with a stderr write and an ATen
    dispatch — happens exactly once per (input, shape, dtype, device); after
    that the feed is a pointer-stable, correctly aligned buffer the call
    writes into. Pointer stability is also what cudagraph static inputs want,
    so this cannot cost the lane a capture later.
    """

    __slots__ = ("_buffers",)

    def __init__(self) -> None:
        self._buffers: Dict[str, Any] = {}

    def staged(self, name: str, value: Any, dtype: str = "") -> Any:
        """A 16-byte-aligned contiguous copy of ``value`` in an owned buffer.

        ``dtype`` (pgw#1074) stages into the artifact's DECLARED dtype instead
        of the feed's own. The conversion is ``Tensor.copy_``'s — one device
        kernel into the buffer this ingress already had to write, so the
        recast costs nothing beyond the staging copy and never synchronises.
        """
        import torch

        want = getattr(torch, dtype) if dtype else value.dtype
        buf = self._buffers.get(name)
        if (buf is None or buf.dtype is not want
                or buf.device != value.device
                or tuple(buf.shape) != tuple(value.shape)):
            buf = torch.empty(
                tuple(value.shape), dtype=want, device=value.device)
            if int(buf.data_ptr()) % AOTI_ALIGNMENT:
                # torch's caching allocator hands out 512-byte-aligned blocks
                # (CPU: 64). If that ever stops being true, realigning by
                # allocation is not a fix and this must fail loudly rather
                # than quietly hand the runner another unaligned pointer.
                raise IngressContractError(
                    "realign_unavailable",
                    f"a freshly allocated buffer for input {name!r} is itself "
                    f"not {AOTI_ALIGNMENT}-byte aligned "
                    f"(ptr%{AOTI_ALIGNMENT}="
                    f"{int(buf.data_ptr()) % AOTI_ALIGNMENT}); the artifact's "
                    f"aligned-input contract cannot be satisfied here")
            self._buffers[name] = buf
        buf.copy_(value)
        return buf

    def buffered(self) -> Tuple[str, ...]:
        return tuple(sorted(self._buffers))


def aligned_feeds(
    contract: CallIngress,
    feeds: Sequence[Any],
    aligner: FeedAligner,
    report: Optional[Callable[[str, str, str], None]] = None,
) -> List[Any]:
    """``feeds`` normalized to the artifact's contract (pgw#791 + pgw#1074).

    Two normalizations, one staging buffer: the declared dtype
    (:func:`recast_gap`) and the declared 16-byte alignment
    (:func:`alignment_gap`). A recast implies a staged copy, so it subsumes
    the alignment pass rather than running after it.

    ``feeds`` is :func:`marshal_positional`'s output, so it is one value per
    declared input in ``position`` order — the same order this walks, which is
    what lets a reason NAME the input instead of an index.
    """
    specs = sorted(contract.inputs, key=lambda s: s.position)
    if len(specs) != len(feeds):
        raise IngressContractError(
            "feed_arity_mismatch",
            f"{len(feeds)} marshalled feed(s) for {len(specs)} declared "
            f"input(s); an aligned-ingress pass that cannot name its inputs "
            f"would realign the wrong slot")
    out = list(feeds)
    for idx, (spec, value) in enumerate(zip(specs, feeds)):
        recast = recast_gap(spec, value)
        reason = recast or alignment_gap(value)
        if not reason:
            continue
        out[idx] = aligner.staged(spec.name, value, spec.dtype if recast else "")
        if report is not None:
            report(spec.name, reason, RECAST_EVENT if recast else REALIGN_EVENT)
    return out


#: pgw#1074: how far each refusal reason puts an entry from the call. Dims
#: LAST, so an entry that matches every declared dimension sorts to the front
#: of a refusal listing whatever its remaining complaint is. The rungs are
#: ordinal only — nothing reads their absolute values.
MISS_RUNGS: Mapping[str, int] = {
    # The call fits this graph's shape and disagrees about one scalar fact.
    "dtype_mismatch": 1,
    "input_not_tensor": 2,
    # A branch/adapter routing disagreement: same shape family, wrong class.
    "input_excluded": 3,
    # Shape disagreements — the call does not fit this graph at all.
    "static_dim_mismatch": 4,
    "range_violation": 4,
    "symbol_inconsistent": 4,
    "rank_mismatch": 5,
    # The call does not even carry the input; nothing else was measurable.
    "input_missing": 6,
}
_MISS_RUNG_DEFAULT = 9
#: How many non-closest entries a refusal names individually before it
#: switches to a per-reason count. The closest entry is ALWAYS named in full
#: and always first: this detail is truncated by the hub at ~573 chars, and
#: pgw#1074 is what happens when the one informative entry falls past that.
_MISS_SAMPLE = 3


@dataclass(frozen=True)
class IngressMiss:
    """ONE reason one entry refuses one call (pgw#1074).

    :attr:`rung` is how FAR that reason puts the entry from the call, and it
    exists because a refusal listing has to be ORDERED by something. The
    ordering is dims-first: an entry the call matches in every declared
    dimension and misses only on dtype is the entry a reader is looking for,
    and listing entries in iteration order buried exactly that one (36 tried,
    6 listed, the dims-matching one not among them — the pgw#1074 filing).
    """

    reason: str
    detail: str
    input: str = ""
    rung: int = _MISS_RUNG_DEFAULT


def _rung(reason: str) -> int:
    return MISS_RUNGS.get(reason, _MISS_RUNG_DEFAULT)


def miss_distance(misses: Sequence[IngressMiss]) -> Tuple[int, ...]:
    """Sort key over one entry's misses — lower is CLOSER to the call.

    The sorted rung tuple, so that (a) an entry whose only complaint is a
    shallow one beats an entry with a deep one, and (b) among entries with
    the same shallowest complaint, the one with FEWER complaints wins (a
    prefix sorts before its extension). Deterministic and total.
    """
    return tuple(sorted(_rung(m.reason) for m in misses))


def ingress_report(
    contract: CallIngress,
    args: Sequence[Any],
    kwargs: Mapping[str, Any],
    *,
    first_only: bool = False,
) -> Tuple[Tuple[IngressMiss, ...], Dict[str, int]]:
    """EVERY way this call misses this contract, plus the symbol bindings.

    The one implementation of the pgw#704 B2 check. :func:`assert_ingress`
    raises this function's FIRST miss and :meth:`EntryDispatch.select` ranks
    whole entries by all of them, so an admission decision and the sentence
    that explains it can never be computed by two different rules.

    ``first_only`` returns as soon as one miss is found. It is an early EXIT
    from this same walk, never a second rule — every ADMISSION decision takes
    it (an admitted call has no misses, so the two are identical there), and
    the exhaustive walk is paid only on the refusal path, which is already
    falling back to eager. A 36-entry cell is asked this per denoise step.

    Misses are collected in declaration order, per input in
    dtype -> rank -> dims order, which is the order the raising check used
    before this became a collecting one.
    """
    present = excluded_inputs_present(contract, kwargs)
    if present:
        return ((IngressMiss(
            "input_excluded",
            f"this graph class REFUSES input(s) {list(present)!r}: the call "
            f"carries them, so it must be served by the class that declares "
            f"them (pgw#790 — a branchless class fed an adapter would return "
            f"the base model and look correct)",
            str(present[0])),), {})
    try:
        bound = bind_call_inputs(contract, args, kwargs)
    except IngressContractError as exc:
        # An input the call does not carry at all: nothing further about this
        # entry can be measured, so it is one miss and the deepest rung.
        return ((IngressMiss(exc.reason, str(exc)),), {})
    misses: List[IngressMiss] = []
    symbols: Dict[str, int] = {}
    owner: Dict[str, str] = {}
    bounds = contract.symbol_bounds
    for spec in contract.inputs:
        if first_only and misses:
            break
        if spec.name not in bound:
            continue
        value = bound[spec.name]
        shape = getattr(value, "shape", None)
        if shape is None:
            misses.append(IngressMiss(
                "input_not_tensor",
                f"declared input {spec.name!r} is a "
                f"{type(value).__name__} with no shape", spec.name))
            continue
        got_dtype = _dtype_name(value)
        if got_dtype != spec.dtype and not recast_gap(spec, value):
            misses.append(IngressMiss(
                "dtype_mismatch",
                f"input {spec.name!r} dtype {got_dtype or '<unknown>'} != "
                f"declared {spec.dtype}", spec.name))
        actual = tuple(int(d) for d in shape)
        if len(actual) != len(spec.shape):
            misses.append(IngressMiss(
                "rank_mismatch",
                f"input {spec.name!r} rank {len(actual)} != declared "
                f"{len(spec.shape)} (declared shape {list(spec.shape)!r})",
                spec.name))
            continue
        for pos, (declared, got) in enumerate(zip(spec.shape, actual)):
            if isinstance(declared, int):
                if got != declared:
                    misses.append(IngressMiss(
                        "static_dim_mismatch",
                        f"input {spec.name!r} dim {pos} = {got} != "
                        f"statically specialized {declared}", spec.name))
                continue
            lo, hi = bounds[declared]
            if not (lo <= got <= hi):
                misses.append(IngressMiss(
                    "range_violation",
                    f"input {spec.name!r} dim {pos} (symbol {declared!r}) = "
                    f"{got} outside declared range [{lo}, {hi}]", spec.name))
                continue
            prior = symbols.get(declared)
            if prior is not None and prior != got:
                misses.append(IngressMiss(
                    "symbol_inconsistent",
                    f"symbol {declared!r} = {got} on input {spec.name!r} dim "
                    f"{pos} but {prior} on input {owner[declared]!r}",
                    spec.name))
                continue
            symbols[declared] = got
            owner.setdefault(declared, spec.name)
    return (tuple(misses[:1]) if first_only else tuple(misses)), symbols


def assert_ingress(
    contract: CallIngress,
    args: Sequence[Any],
    kwargs: Mapping[str, Any],
) -> Dict[str, int]:
    """Assert one call against the artifact's DECLARED contract (pgw#704 B2).

    Returns the resolved symbol bindings on success; raises
    :class:`IngressContractError`, naming the input, the dim, the symbol,
    the value and the bound, on the FIRST violation :func:`ingress_report`
    finds. Checks, per input:

    * present (unless declared optional);
    * dtype EXACT — an exported graph is specialized on dtype — except for
      the one named normalization :func:`recast_gap` performs;
    * rank EXACT;
    * static dims EXACT — a specialized dim is not a range;
    * symbolic dims inside the declared inclusive range;
    * **symbol CONSISTENCY** — one symbol appearing in two shapes must take
      the same value. ``range_constraints`` cannot express this, but the
      graph requires it, so a mismatch is outside the declared envelope even
      when both values are individually in range.
    """
    misses, symbols = ingress_report(contract, args, kwargs, first_only=True)
    if misses:
        raise IngressContractError(misses[0].reason, misses[0].detail)
    return symbols


# ---------------------------------------------------------------------------
# B1 — the constants-bound gate
# ---------------------------------------------------------------------------


def resident_constants(module: Any) -> Dict[str, Any]:
    """Every resident tensor a ``state_dict``-sourced constant can bind to.

    ``module.state_dict()`` is NOT that set, and the difference is pgw#825:
    it omits NON-PERSISTENT buffers. The canonical LoRA branch pair is
    exactly that — ``w8a8_lora.alloc_branch_buffers`` registers
    ``lora_a``/``lora_b`` with ``persistent=False`` so a checkpoint never
    carries a zeroed adapter — yet ``torch.export`` lifts them as BUFFER
    inputs and AOTInductor declares them ``ConstantType::Buffer`` under their
    real FQN, i.e. ``source=state_dict``. A bind table built from
    ``state_dict()`` alone therefore declares 20 constants per sdxl
    ``BasicTransformerBlock`` that no lookup could ever resolve.

    ONE definition, used by the mint's bindability gate and by both arm
    sites, because a gate that models the arm differently from the arm is
    the pgw#816/#822 class: it either refuses a cell that would have served
    or admits one that cannot.

    Binding is by FQN and :func:`resolve_constants` reads only the DECLARED
    names, so the extra keys are inert for any artifact that does not want
    them.
    """
    out: Dict[str, Any] = dict(module.state_dict())
    try:
        buffers = list(module.named_buffers())
    except Exception:  # pragma: no cover — duck-typed owners in unit rigs
        return out
    for name, buf in buffers:
        if buf is not None:
            out.setdefault(str(name), buf)
    return out


@dataclass
class TCGEntryRunner:
    """Worker ingress and fallback policy around one TCG-owned runner."""

    runner: CompiledGraphRunner
    contract: CallIngress
    module_name: str
    entry: str
    family: str
    refusals: Dict[str, int] = field(default_factory=dict)
    realigned: Dict[str, int] = field(default_factory=dict)
    aligner: FeedAligner = field(default_factory=FeedAligner)

    @property
    def calls(self) -> int:
        return int(self.runner.calls)

    @property
    def bound(self) -> bool:
        return bool(self.runner.bound)

    @property
    def user_managed(self) -> bool:
        return True

    def declared_fqns(self) -> Tuple[str, ...]:
        return tuple(self.runner.declared_fqns)

    def excludes(self, names: Sequence[str]) -> bool:
        wanted = {str(name) for name in names}
        return bool(wanted) and wanted <= set(self.contract.excluded_inputs)

    def assert_ready(self) -> None:
        if not self.runner.bound:
            raise ConstantsUnboundError(
                "constants_unbound",
                f"refusing to invoke compiled graph {self.entry!r} before "
                "TCG completed its exact constant bind",
            )

    def _report_normalized(self, name: str, reason: str, event: str) -> None:
        key = f"{name}/{reason}"
        seen = self.realigned.get(key, 0)
        self.realigned[key] = seen + 1
        if seen:
            return
        activity_mod.emit_event(
            event,
            f"family={self.family} graph_class={self.entry} "
            f"target={self.module_name} input={name}: {reason}",
            phase=reason,
        )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.assert_ready()
        try:
            assert_ingress(self.contract, args, kwargs)
            feeds = aligned_feeds(
                self.contract,
                marshal_positional(self.contract, args, kwargs),
                self.aligner,
                self._report_normalized,
            )
        except IngressContractError as exc:
            self.refusals[exc.reason] = self.refusals.get(exc.reason, 0) + 1
            raise
        return self.runner(*feeds)


EntryRunner = TCGEntryRunner


def no_entry_detail(
    tried: int,
    missed: Sequence[Tuple[Tuple[int, ...], str, Tuple[IngressMiss, ...]]],
) -> str:
    """The ``no_entry_admits`` sentence, CLOSEST ENTRY FIRST (pgw#1074).

    The refusal this replaces said "36 tried" and then listed six in iteration
    order — and the one entry whose dims matched the call was not among them,
    so diagnosing a live refusal meant pulling the published cell apart off-pod
    to find out what the covering entry actually objected to. A refusal that
    hides the one relevant miss is the pgw#1058 lesson repeating one layer up,
    in the diagnostics.

    So: rank by :func:`miss_distance`, name the closest entry and its FULL
    reason first (it survives any downstream truncation), then account for
    every other entry by reason COUNT rather than by naming an arbitrary few.
    Nothing is silently dropped — ``tried`` and the counts always add up.
    """
    if not missed:
        return (
            f"request out of declared envelope: no packaged entry admits "
            f"this call ({tried} tried), so the request is served EAGER "
            f"and named at ingress")
    ranked = sorted(missed, key=lambda row: (row[0], row[1]))
    _distance, closest, misses = ranked[0]
    dims_ok = all(_rung(m.reason) < MISS_RUNGS["static_dim_mismatch"]
                  for m in misses)
    head = (
        f"request out of declared envelope — no packaged entry admits this "
        f"call, served EAGER ({tried} tried); CLOSEST entry "
        f"{closest!r}"
        f"{' — every declared dim MATCHES' if dims_ok else ''}: "
        + "; ".join(f"{m.reason} ({m.detail})" for m in misses[:2]))
    rest = ranked[1:]
    if not rest:
        return head
    # ONE count per entry, under its own closest reason, so the counts sum to
    # exactly the number of entries tried and "36 tried" can be checked
    # against the sentence that follows it.
    tally: Dict[str, int] = {}
    for _d, _name, other in rest:
        primary = min(other, key=lambda m: _rung(m.reason)).reason
        tally[primary] = tally.get(primary, 0) + 1
    named = "; ".join(
        f"{name}: {min(misses_, key=lambda m: _rung(m.reason)).reason}"
        for _d, name, misses_ in rest[:_MISS_SAMPLE])
    counted = ", ".join(
        f"{reason} x{count}" for reason, count in
        sorted(tally.items(), key=lambda kv: (-kv[1], kv[0])))
    return f"{head}. Other {len(rest)} entries [{counted}] — next: {named}"


@dataclass
class EntryDispatch:
    """The REGISTRY of armed entries serving ONE target, behind one call site
    (pgw#758, re-based per entry by pgw#1176).

    Dispatch is the declared contract itself: the call routes to the entry
    whose ingress contract ADMITS it. No admitting entry is a named
    per-request refusal (the caller serves eagerly); more than one is
    ``entry_ambiguous`` — the declaration failed to discriminate two graph
    classes by ingress, which is a defect to surface, never a coin to flip.

    pgw#1176: entries JOIN as they arm and LEAVE when they de-arm. The tuple
    that used to be built once, from a complete cell, was the arming half of
    the wrong atom. A registry has a further property the tuple could not:
    a subset is a legitimate STEADY STATE, not merely a stage on the way to
    coverage. pgw#1177 measured why that matters — ~0.75 GiB of device memory
    per resident AOTI container — so a pod that arms its hot classes and
    leaves cold ones eager holds a handful of containers rather than 36, on
    purpose and permanently.

    ``declared`` is every class name this pod's DECLARATION traces to,
    whether armed yet or not. It is what lets a miss distinguish "declared,
    pending compile" (silent eager, count the hotness) from "undeclared
    shape" (a real shape-growth report).
    """

    runners: Tuple[Tuple[str, EntryRunner], ...] = ()
    declared: Tuple[str, ...] = ()
    #: entry name -> the reason it left. A de-armed entry is REMEMBERED, not
    #: forgotten: §4.31's de-arm is sticky for the boot, and a re-arm of a
    #: class that failed for cause would be the thing that rule forbids.
    de_armed: Dict[str, str] = field(default_factory=dict)
    #: The entry :meth:`select` last routed to — the only way the fail-soft
    #: wrapper one frame up can name the graph class that raised.
    last_selected: str = ""
    #: Calls served by entries that have since DE-ARMED. Banked rather than
    #: discarded: those executions happened, and `execution_count` is the
    #: adoption proof's evidence — dropping them when a sibling de-arms would
    #: make a pod that served 5,000 compiled requests and then lost one class
    #: read as a pod that never served compiled at all.
    retired_calls: int = 0

    def add(self, name: str, runner: EntryRunner) -> None:
        """Register one armed entry. Replaces an entry of the same name (a
        re-arm), and refuses one this dispatch de-armed for cause."""
        label = str(name)
        if label in self.de_armed:
            raise AdoptError(
                "entry_de_armed",
                f"entry {label!r} was de-armed this boot "
                f"({self.de_armed[label]}); §4.31's de-arm is sticky, so it "
                f"must not be re-armed without a new process")
        rows = [(n, r) for n, r in self.runners if n != label]
        rows.append((label, runner))
        self.runners = tuple(sorted(rows, key=lambda row: row[0]))

    def remove(self, name: str, reason: str) -> bool:
        """De-arm ONE entry, sticky for the boot. True when it was armed.

        This is the whole replacement for the old cell-wide revoke: a
        cell-attributable failure in one graph class costs that class, and
        every sibling keeps serving compiled.
        """
        label = str(name)
        before = len(self.runners)
        self.retired_calls += sum(
            int(r.calls) for n, r in self.runners if n == label)
        self.runners = tuple(
            (n, r) for n, r in self.runners if n != label)
        self.de_armed[label] = str(reason or "unstated")
        return len(self.runners) != before

    @property
    def pending(self) -> Tuple[str, ...]:
        """Declared classes that are neither armed nor de-armed — the ones a
        background compile has not reached yet. Serving them eager is
        correct, expected, and not a defect."""
        armed = {n for n, _r in self.runners}
        return tuple(
            n for n in self.declared
            if n not in armed and n not in self.de_armed)

    def assert_ready(self) -> None:
        """B1 for every entry — an unbound one is a segfault, not a miss."""
        for _name, runner in self.runners:
            runner.assert_ready()

    @property
    def bound(self) -> bool:
        return bool(self.runners) and all(
            runner.bound for _n, runner in self.runners)

    @property
    def user_managed(self) -> bool:
        """True only when EVERY entry bound by reference — one copying entry
        is one copy of the block's weights per instance."""
        return bool(self.runners) and all(
            runner.user_managed for _n, runner in self.runners)

    def declared_fqns(self) -> Tuple[str, ...]:
        names: List[str] = []
        for _n, runner in self.runners:
            names.extend(runner.declared_fqns())
        return tuple(sorted(set(names)))

    def select(
        self, args: Sequence[Any], kwargs: Mapping[str, Any],
    ) -> Tuple[str, EntryRunner]:
        admitted: List[Tuple[str, EntryRunner]] = []
        missed: List[Tuple[str, EntryRunner]] = []
        for name, runner in self.runners:
            misses, _symbols = ingress_report(
                runner.contract, args, kwargs, first_only=True)
            if misses:
                missed.append((name, runner))
                continue
            admitted.append((name, runner))
        if not admitted:
            # Only now is the exhaustive walk worth its cost: the call is
            # already headed for the eager fallback, and the sentence it
            # leaves behind is the whole diagnosis anyone will ever get.
            ranked = [
                (miss_distance(rep), name, rep) for name, rep in (
                    (name, ingress_report(runner.contract, args, kwargs)[0])
                    for name, runner in missed)]
            pending = self.pending
            if pending:
                # pgw#1176: under accretion the commonest reason nothing
                # admits a call is that its class has not been compiled YET.
                # That is not a shape gap and must not be reported as one —
                # the growth path would submit a class the declaration
                # already contains.
                raise IngressContractError(
                    "entry_pending_compile",
                    f"no ARMED entry admits this call ({len(self.runners)} "
                    f"armed, {len(pending)} declared classes still pending "
                    f"compile) — served EAGER while the background compile "
                    f"reaches them: {list(pending)[:4]!r}. "
                    + no_entry_detail(len(self.runners), ranked))
            raise IngressContractError(
                "no_entry_admits", no_entry_detail(len(self.runners), ranked))
        if len(admitted) > 1:
            names = sorted(name for name, _ in admitted)
            raise IngressContractError(
                "entry_ambiguous",
                f"{len(admitted)} entries admit this call ({names[:6]!r}) — "
                f"the declaration does not discriminate these graph classes "
                f"by ingress contract")
        return admitted[0]

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        name, runner = self.select(args, kwargs)
        # Recorded BEFORE the call so a raising entry can be de-armed BY NAME
        # (§4.31 per entry): the wrapper catches the exception a frame up and
        # has no other way to know which of N graph classes produced it.
        self.last_selected = name
        return runner(*args, **kwargs)

    @property
    def calls(self) -> int:
        return int(self.retired_calls) + sum(
            runner.calls for _n, runner in self.runners)

    def refusal_counts(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for _n, runner in self.runners:
            for reason, count in runner.refusals.items():
                out[reason] = out.get(reason, 0) + count
        return out

    def realignment_counts(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for _n, runner in self.runners:
            for key, count in runner.realigned.items():
                out[key] = out.get(key, 0) + count
        return out

    def entry_calls(self) -> Dict[str, int]:
        """Per-entry served-call counts — which graph class actually served."""
        return {name: runner.calls for name, runner in self.runners}

    def excludes(self, names: Sequence[str]) -> bool:
        """True when some packaged entry REFUSES every one of ``names``.

        The adapter-free routing question, asked of the declaration rather
        than of a flag: is there a class in this cell that serves calls
        WITHOUT these inputs? (pgw#790)
        """
        wanted = set(str(n) for n in names)
        if not wanted:
            return False
        return any(wanted <= set(runner.contract.excluded_inputs)
                   for _n, runner in self.runners)


# ---------------------------------------------------------------------------
# Serve — module swap behind a fail-soft guard
# ---------------------------------------------------------------------------


def ingress_class_name(
    target: str, args: Sequence[Any], kwargs: Mapping[str, Any],
) -> str:
    """The DECLARED CLASS one refused call names (pgw#916).

    A shape alone is not a growable unit of work — a mint is asked for a
    class, and a class is the whole ingress coordinate of one target: every
    declared input, in position order, with its dtype and extents.  Rendering
    it here (rather than logging "a shape was refused") is what lets the
    growth path submit the exact thing that is missing, and what lets two
    pods agree they are missing the SAME thing.

    Deliberately independent of any artifact: the whole point is that the
    call arrived at a class no packaged entry declares, so there is no entry
    block to read it off.
    """
    parts: List[str] = []

    def render(name: str, value: Any) -> None:
        shape = getattr(value, "shape", None)
        dtype = getattr(value, "dtype", None)
        if shape is None:
            parts.append(f"{name}={value!r}"[:64])
            return
        extents = ",".join(str(int(d)) for d in tuple(shape))
        token = str(dtype or "").replace("torch.", "")
        parts.append(f"{name}={token}[{extents}]")

    for index, value in enumerate(args):
        render(f"#{index}", value)
    for name in sorted(kwargs):
        value = kwargs[name]
        if value is None or isinstance(value, (bool, int, float, str)):
            parts.append(f"{name}={value!r}")
            continue
        render(name, value)
    return f"{target}/{','.join(parts)}"


def lifted_call_kwargs(module: Any) -> Dict[str, Any]:
    """The lifted-adapter call kwargs for one denoiser, or ``{}`` (pgw#725).

    Under input-lifting the rank bucket arrives as two flat tensors in the
    CALL rather than as module state, so nothing can be baked and the
    no-baked-adapter gate degenerates to a signature check. Both kwargs are
    MANDATORY when a binding exists: ``bind_views`` refuses a ``None`` half
    by name, so the EAGER fallback needs them exactly as much as the
    artifact does — which is why the wrapper merges them once, up front, for
    both paths.

    The tensors are returned by reference and never copied. The adapter
    machinery mutates them IN PLACE through views, so a swap needs no
    artifact interaction at all and the call arguments stay pointer-stable
    (what cudagraph static inputs require).

    ``bucket=0`` is its own branchless graph class with no lifted signature,
    so it has no binding and gets no kwargs.
    """
    binding = lora_lifted.lifted_binding(module)
    return binding.call_kwargs() if binding is not None else {}


def adapter_call_kwargs(
    module: Any, runner: "EntryDispatch",
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """``(artifact kwargs, eager kwargs)`` for one call's adapter inputs.

    pgw#790's routing rule, in one place. The two differ in exactly one case:
    the denoiser carries a lifted adapter, NO adapter is currently active, and
    the cell packages a class that REFUSES the adapter inputs. Then the
    artifact call omits the pair so the BRANCHLESS class admits it, while the
    eager fallback still receives it — ``bind_views`` refuses a ``None`` half
    by name, so the eager path needs the pair exactly as much as a
    branch-bearing artifact does.

    Nothing here is a new piece of state. "Is an adapter active" is
    ``w8a8_lora``'s own ``_cozy_lora_active``, written by the per-request
    attach/clear path (``utils.lora``) since gw#627; this only reads it. And
    it decides only WHICH pre-compiled class serves the call — never the
    shape of a program, never a baked constant. Both classes return the same
    tensor for adapter-free traffic (a zeroed B adds exactly 0), so the
    routing is a cost decision whose correctness does not depend on it.
    """
    lifted = lifted_call_kwargs(module)
    if not lifted:
        return {}, {}
    if lora_lifted.adapter_active(module):
        return lifted, lifted
    if not runner.excludes(lora_lifted.LIFTED_INPUT_NAMES):
        # One branch-bearing class only: it declares the pair as MANDATORY,
        # so withholding it would refuse every request by name.
        return lifted, lifted
    return {}, lifted


def assert_lifted_contract(module: Any, contract: CallIngress) -> None:
    """The module's lifted state and the artifact's signature must AGREE.

    Either mismatch is the pgw#704 S9 defect in one direction or the other:
    an artifact that declares the adapter inputs but a module with no
    binding has no way to supply a mandatory input, while a lifted module
    served by an artifact that does NOT declare them means the branch was
    traced away — every request silently gets the base model. Caught at arm
    time, by name, instead of at the first call.
    """
    declared = {spec.name for spec in contract.inputs}
    wanted = set(lora_lifted.LIFTED_INPUT_NAMES)
    lifted = lora_lifted.lifted_binding(module) is not None
    if lifted and wanted <= set(contract.excluded_inputs):
        # pgw#790: the BRANCHLESS arm of an adapter-forked cell. It says so
        # explicitly — the adapter inputs are refused, not forgotten — so the
        # dispatch can never route adapter-bearing traffic to it and "the
        # branch was traced away" is impossible to reach silently.
        return
    if lifted and not wanted <= declared:
        raise AdoptError(
            "lifted_inputs_undeclared",
            f"module carries a lifted LoRA adapter but the artifact declares "
            f"no {sorted(wanted - declared)!r} input(s); the branch was traced "
            "away, so every request would silently serve the base model")
    if not lifted and wanted & declared:
        raise AdoptError(
            "lifted_inputs_unbindable",
            f"artifact declares lifted adapter input(s) "
            f"{sorted(wanted & declared)!r} but the module has no lifted "
            "binding to supply them (bucket=0 is a different graph class)")


def wrap_module(
    module: Any,
    runner: "EntryDispatch",
    meta: Dict[str, Any],
    *,
    attr: str = "forward",
    target: str = "",
    eager_forward: Optional[Callable[..., Any]] = None,
) -> None:
    """Swap ``module.<attr>`` for the cell's dispatch behind a fail-soft
    guard.

    The first artifact ERROR
    synchronously revokes scheduler-visible compiled proof and permanently
    routes to eager; the module object (config, dtype, device, weights)
    stays untouched, and its weights remain the constant-binding source.
    ``runner`` is one target's :class:`EntryDispatch`; ``attr`` generalizes
    the swap beyond ``forward`` for dotted targets like ``vae.decode``.

    An :class:`IngressContractError` is NOT such an error. It is a named,
    counted, per-request contract refusal — the request serves eagerly and
    the artifact stays armed for traffic inside the declared envelope,
    because one
    out-of-range request (or an entry-dispatch miss/ambiguity) says nothing
    about the artifact's health.

    pgw#725 LoRA seam: when the denoiser carries a lifted adapter, both
    ``lora_a``/``lora_b`` are MANDATORY call kwargs and this wrapper is the
    call site that supplies them — see :func:`lifted_call_kwargs`.
    """
    attr = str(attr or "forward")
    label = target or f"{meta.get('family')}.{attr}"
    original = eager_forward or getattr(module, attr)
    state: Dict[str, Any] = {
        "failed": False,
        "successful_calls": 0,
        "ingress_refusals": 0,
        "last_refusal": "",
        "original": original,
        "attr": attr,
        "target": label,
        "failure_callback": None,
        "refusal_callback": None,
        "revocation_error": "",
        "runner": runner,
    }

    def _de_arm(reason: str, detail: str) -> None:
        """§4.31 PER ENTRY (pgw#1176): a cell-attributable failure de-arms the
        GRAPH CLASS that produced it, sticky for the boot, and every sibling
        keeps serving compiled. The target's compiled lane is revoked only
        when the registry empties — that is the moment, and the only moment,
        at which this target genuinely stopped serving compiled.
        """
        name = str(getattr(runner, "last_selected", "") or "")
        remove = getattr(runner, "remove", None)
        if name and callable(remove):
            remove(name, reason)
            activity_mod.emit_event(
                "aot_entry_de_armed",
                f"target={label}: {detail}",
                phase=reason,
                family=str(meta.get("family") or ""),
                compiled_graph_key=str(meta.get("compiled_graph_key") or ""),
                graph_class=name,
            )
            siblings = tuple(getattr(runner, "runners", ()) or ())
            if siblings:  # siblings still serve
                logger.warning(
                    "aot-serve: %s de-armed entry %s (%s); %d sibling "
                    "entr%s still armed", label, name, reason, len(siblings),
                    "y" if len(siblings) == 1 else "ies")
                return
        state["failed"] = True
        _revoke(state, detail)

    def aot_forward(*args: Any, **kwargs: Any) -> Any:
        if state["revocation_error"]:
            raise CompiledExecutionLaneUnavailableError(state["revocation_error"])
        # The lifted pair is resolved PER CALL, not captured at wrap time:
        # the LoRA lane may install/remove lifting independently of arming,
        # and a stale capture would either starve the graph of a mandatory
        # input or feed one to a forward that no longer takes it.
        # pgw#790: an adapter-FREE call omits the pair when the cell packages
        # a branchless class, so the dispatch routes it to the graph that does
        # not spend 32-45% of its compiled forward on arithmetic over zeros.
        # The eager fallback always receives it.
        artifact_lora, eager_lora = adapter_call_kwargs(module, runner)
        eager_kwargs = {**kwargs, **eager_lora}
        kwargs = {**kwargs, **artifact_lora}
        if serve_posture.eager_only():
            # pgw#1142 / §4.32 item 4: an operator ordered eager. This is THE
            # reversibility seam — the artifact is not unwrapped and `state`
            # is not touched, so releasing the order resumes compiled serving
            # on the very next call with no re-arm, no re-materialize and no
            # re-mint. Unwrapping here would have been the same posture for
            # one boot and a lie afterwards.
            #
            # Ordered BEFORE the `failed` check only for cost; the two are
            # independent and stay independent — releasing the order never
            # resurrects a cell de-armed for cause (§4.31), because that
            # de-arm is evidence and this is policy.
            return original(*args, **eager_kwargs)
        if state["failed"]:
            return original(*args, **eager_kwargs)
        try:
            out = runner(*args, **kwargs)
            # Envelope parity (live-named on the 0.76.x rerun line,
            # 2026-07-29): the exported graph returns the RAW tensor, but a
            # diffusers pipeline calls `unet(..., return_dict=False)[0]` —
            # indexing a bare tensor silently slices the batch dim and the
            # crash surfaces downstream as a broadcast error. Restore the
            # caller's declared envelope: return_dict=False means a 1-tuple.
            if kwargs.get("return_dict") is False and not isinstance(
                    out, tuple):
                return (out,)
            return out
        except IngressContractError as exc:
            # B2: the exported graph would have run this and returned
            # unvalidated output. Named refusal + eager service.
            state["ingress_refusals"] = int(state["ingress_refusals"]) + 1
            state["last_refusal"] = f"{exc.reason}: {exc}"
            logger.warning(
                "aot-serve: %s REFUSED input outside the declared envelope (%s: %s); "
                "serving this request eager, artifact stays armed",
                label, exc.reason, exc)
            activity_mod.emit_event(
                "aot_ingress_refused",
                f"family={meta.get('family')} target={label}: {exc}",
                phase=exc.reason,
            )
            report_ingress_refusal(state, exc.reason, str(exc))
            if exc.reason == "entry_pending_compile":
                # pgw#1176: the class IS declared; the background compile has
                # not reached it. Submitting a shape gap here would ask the
                # growth path to add a class the declaration already carries.
                return original(*args, **eager_kwargs)
            # pgw#916: the refusal is also a SHAPE GAP — the armed cell does
            # not cover this declared class, and on the AOT arm nothing was
            # ever going to grow it (hot_swap.enable returns False without a
            # dynamo router, so the executor's three growth call sites are
            # no-ops on every AOT arm). Named, counted, and submitted through
            # the one arm-agnostic growth module.
            shape_growth.report_and_submit(shape_growth.ShapeGap(
                arm=shape_growth.ARM_AOT,
                family=str(meta.get("family") or ""),
                target=label,
                declared_class=ingress_class_name(label, args, kwargs),
                reason=exc.reason,
                detail=str(exc)[:400],
                compiled_graph_key=str(meta.get("compiled_graph_key") or ""),
            ))
            return original(*args, **eager_kwargs)
        except ConstantsUnboundError as exc:
            # Reaching here means the arm order was violated. The gate did
            # its job (no segfault); THIS graph class is structurally
            # unusable.
            logger.error(
                "aot-serve: %s invoked with unbound constants (%s); eager for "
                "the rest of this process", label, exc)
            activity_mod.emit_event(
                "aot_constants_unbound",
                f"family={meta.get('family')} target={label}: {exc}",
                phase=exc.reason,
            )
            _de_arm(exc.reason, f"constants unbound: {exc}")
            return original(*args, **eager_kwargs)
        except Exception as exc:  # noqa: BLE001 — ANY artifact problem => eager
            if is_cuda_oom(exc):
                # pgw#1141: ATTRIBUTION. The serve-first doctrine makes the
                # first real request the proof, so what that request blames
                # decides whether a good cell survives — and allocator
                # exhaustion is a fact about the CARD at this instant (a
                # sibling load, a concurrent rotation), not about the artifact.
                # Condemning the cell for it would retire a correct one on the
                # first busy moment and re-mint it on the replacement pod.
                # Serve THIS request eager, stay armed, say so.
                logger.warning(
                    "aot-serve: %s hit CUDA OOM (%s); serving this request "
                    "eager, artifact stays armed — allocator pressure is not "
                    "the cell's fault", label, exc)
                activity_mod.emit_event(
                    "aot_serve_oom",
                    f"family={meta.get('family')} target={label}: "
                    f"{type(exc).__name__}: {exc}",
                    phase="cuda_oom",
                )
                return original(*args, **eager_kwargs)
            entry_name = str(getattr(runner, "last_selected", "") or "")
            detail = (
                f"AOTI artifact {label}"
                + (f" entry {entry_name}" if entry_name else "")
                + f" failed: {type(exc).__name__}: {exc}")
            _de_arm("artifact_failed", detail)
            logger.warning(
                "aot-serve: %s failed (%s: %s)", label,
                type(exc).__name__, exc)
            return original(*args, **eager_kwargs)

    def _counted_forward(*args: Any, **kwargs: Any) -> Any:
        out = aot_forward(*args, **kwargs)
        state["successful_calls"] = int(runner.calls)
        return out

    setattr(module, attr, _counted_forward)
    setattr(module, _MARKER_ATTR, {
        "meta": {k: meta.get(k) for k in ("sku", "torch", "precision")},
        "state": state,
    })


def _revoke(state: Dict[str, Any], detail: str) -> None:
    """Run the scheduler-state revocation callback for a failed artifact."""
    callback = state.get("failure_callback")
    if not callable(callback):
        return
    try:
        callback(detail)
    except Exception as callback_exc:
        state["revocation_error"] = (
            f"compiled-state revocation failed: "
            f"{type(callback_exc).__name__}: {callback_exc}")
        logger.exception("aot-serve: %s", state["revocation_error"])
        raise CompiledExecutionLaneUnavailableError(state["revocation_error"]) from callback_exc


def report_ingress_refusal(state: Dict[str, Any], reason: str, detail: str) -> None:
    """Hand ONE per-request ingress refusal to the arming brain (pgw#844).

    The refusal already rides ``aot_ingress_refused`` as a countable typed
    event, but that event names no REQUEST — so the request row that was
    served eager by an armed compiled lane still reported
    ``serving_mode=aot_cell, fallback_reason=""``, i.e. an eager latency
    sample counted as compiled. That is the exact contamination
    :mod:`serving_mode` exists to prevent, and arming a partially
    dispatchable cell (which pgw#844 now does) is what makes it common.

    Never raises: telemetry must not be able to un-serve a request that the
    eager fallback is about to answer correctly.
    """
    callback = state.get("refusal_callback")
    if not callable(callback):
        return
    try:
        callback(str(reason or ""), str(detail or ""))
    except Exception:  # noqa: BLE001 — a reporting failure is not a serve failure
        logger.debug("aot-serve: ingress-refusal report failed", exc_info=True)


def set_ingress_refusal_callback(pipeline: Any, callback: Any) -> bool:
    """Bind per-request eager-fallback accounting to every wrapped target."""
    states = _marker_states(pipeline)
    if not states:
        return False
    for state in states:
        state["refusal_callback"] = callback
    return True


def _target_owner(pipeline: Any, target: str) -> Tuple[Any, str]:
    """``(owner module, attribute)`` one entry target names on a pipeline:
    ``"unet"`` -> ``(pipeline.unet, "forward")``; ``"vae.decode"`` ->
    ``(pipeline.vae, "decode")``; ``"vae.decoder"`` ->
    ``(pipeline.vae.decoder, "forward")``.

    pgw#967: this delegates to ``compile_cache._resolve_target`` — ONE
    resolver, because the two disagreed on a dotted target whose leaf is a
    MODULE. This function partitioned on the FIRST dot and treated the rest
    as a plain attribute, so ``vae.decoder`` resolved to
    ``(pipeline.vae, "decoder")`` and the arm would have replaced a
    SUBMODULE with a wrapped function, while the mint resolved the same
    string to the decoder's ``forward`` and exported that. A cell whose
    entries are traced from one callable and armed onto another is the
    silent-wrongness class every gate in this stack exists to prevent, and
    nothing would have caught it: ``vae.decode`` (a bound method) is the
    only dotted target any family has ever named, and for it the two agreed.
    """
    resolved = _resolve_target(pipeline, str(target))
    if resolved is None:
        raise AdoptError(
            "no_target", f"pipeline has no callable target {target!r}")
    module, attr, _fn = resolved
    return module, attr


#: The classified refusal a bind that ran out of device memory produces. Same
#: token the two deleted `mint_budget.adopt_headroom` gates used, so every
#: downstream reader (the `cell_adopt_declined` event, `fleet_cells`' abort
#: classification, the hub's phase column) keeps its vocabulary — what changed
#: is that it is now emitted on EVIDENCE rather than on an estimate.
ADOPT_OOM_REASON = "insufficient_adopt_vram"


def _marker(pipeline: Any) -> Dict[str, Any]:
    """The pipeline's arm marker, created empty on first use.

    pgw#1176: the marker is the pod's per-entry serve-state record, not a
    snapshot of one cell. It carries the wrapped targets, the by-reference
    constant pools that must outlive their runners, the literal tables, and
    one row per entry ever armed on this object.
    """
    marker = getattr(pipeline, _MARKER_ATTR, None)
    if not isinstance(marker, dict):
        marker = {
            "meta": {}, "targets": {},
            "bound_constants": {"pools": {}, "literals": {}},
            "entries": {},
        }
        setattr(pipeline, _MARKER_ATTR, marker)
    return marker


def _dispatch_for(marker: Dict[str, Any], target: str) -> Optional[EntryDispatch]:
    row = (marker.get("targets") or {}).get(target) or {}
    runner = (row.get("state") or {}).get("runner")
    return runner if isinstance(runner, EntryDispatch) else None


def _tcg_destination(cache_dir: Optional[Path], compiled_graph_key: str) -> Path:
    root = tensorhub_cas_dir() if cache_dir is None else Path(cache_dir)
    return root / "compiled-graph-runtime" / compiled_graph_key


def arm_compiled_graph(
    pipeline: Any,
    cfg: Any,
    compiled_graph_key: str,
    cache_dir: Optional[Path] = None,
    *,
    declared: Sequence[str] = (),
) -> Dict[str, Any]:
    """Resolve, bind, then register one exact TCG graph class.

    TCG is the only artifact/store authority. The first resolve establishes
    the immutable extraction directory and admitted metadata; ``runner`` is
    asked for the same key and the same directory, so its internal resolve
    verifies/reuses that extraction instead of creating a second one. No live
    module or dispatch state changes until TCG's exact constant bind succeeds.
    """

    key = str(compiled_graph_key or "").strip()
    if not is_compiled_graph_key(key):
        raise AdoptError(
            "compiled_graph_key_invalid", f"not a compiled-graph key: {key!r}"
        )
    # pgw#1283 criterion 4 — THE WORKER'S OWN QUARANTINE, asked before the CAS
    # is. §1.3.4 keeps a refused cell "quarantined-local for forensics"; before
    # the store cutover those bytes lived only in `local_cell_store`, so no
    # load path could reach them. They are now in the very CAS this function
    # resolves from, and TCG has no concept of a worker parity/arm refusal, so
    # without this the runner loads a cell this worker already refused.
    #
    # Only the QUARANTINED verdict refuses. An unverified row is a cell that is
    # durable but not yet proven — §1.5 stores before the gate runs, and it is
    # this very arm that proves it — and a key this worker never recorded is
    # every hub-delivered cell there is.
    if local_cell_store.is_quarantined(key):
        raise AdoptError(
            "compiled_graph_worker_quarantined",
            f"this worker quarantined {key!r} at its own gate (§1.3.4); its "
            f"bytes are kept for forensics and are never armed",
        )
    engine = open_worker_engine(cache_dir)
    destination = _tcg_destination(cache_dir, key)
    compiled_graph = engine.resolve(key, destination)
    if compiled_graph is None:
        raise AdoptError(
            "compiled_graph_unavailable", f"TCG could not resolve {key!r}"
        )
    runner = engine.runner(key, destination)
    if runner is None:
        raise AdoptError(
            "compiled_graph_unavailable", f"TCG could not load {key!r}"
        )
    metadata = dict(compiled_graph.metadata)
    graph_class = metadata.get(GRAPH_CLASS_BLOCK)
    if not isinstance(graph_class, Mapping):
        raise AdoptError(
            "contract_invalid",
            "TCG admitted a compiled graph with no graph_class declaration",
        )
    graph = graph_class.get("graph")
    if not isinstance(graph, Mapping):
        raise AdoptError(
            "contract_invalid", "TCG graph_class records no worker ingress contract"
        )
    name = str(graph_class.get("name") or "").strip()
    target = str(graph_class.get("target") or "").strip()
    if not name or not target:
        raise AdoptError(
            "contract_invalid", "TCG graph_class must name both graph class and target"
        )

    family = str(getattr(cfg, "family", "") or "")
    module, attr = _target_owner(pipeline, target)
    try:
        contract = CallIngress.from_graph(graph)
        assert_lifted_contract(module, contract)
        device = str(getattr(module, "device", "") or "cuda")
        runner.bind(resident_constants(module), device=device)
    except ConstantBindingError as exc:
        reason = ADOPT_OOM_REASON if exc.reason == "out_of_memory" else exc.reason
        raise AdoptError(reason, f"graph class {name!r}: {exc}") from exc

    marker = _marker(pipeline)
    dispatch = _dispatch_for(marker, target)
    first_for_target = dispatch is None
    if dispatch is None:
        dispatch = EntryDispatch(declared=tuple(str(item) for item in declared))
    elif declared:
        dispatch.declared = tuple(str(item) for item in declared)
    entry_runner = TCGEntryRunner(runner, contract, target, name, family)
    dispatch.add(name, entry_runner)

    serve_meta: Dict[str, Any] = {
        **metadata,
        "family": family,
        "compiled_graph_key": key,
    }
    if first_for_target:
        wrap_module(
            module,
            dispatch,
            serve_meta,
            attr=attr,
            target=target,
            eager_forward=None,
        )
        module_marker = getattr(module, _MARKER_ATTR, {})
        marker["targets"][target] = {
            "module": module,
            "attr": attr,
            "state": module_marker.get("state", {}),
        }
    marker["meta"] = serve_meta
    marker["entries"][name] = {
        "compiled_graph_key": key,
        "target": target,
        "class_hash": str(graph_class.get("class_hash") or ""),
    }
    logger.info(
        "aot-serve: armed TCG graph class %s on %s (%d constants, key=%s)",
        name,
        target,
        len(runner.declared_fqns),
        key,
    )
    return serve_meta


def disarm_entry(pipeline: Any, name: str, reason: str) -> bool:
    """De-arm ONE graph class, sticky for the boot. True when it was armed.

    The per-entry half of §4.31, reachable from outside a serve call: the
    mint's parity gate refuses an entry here rather than un-arming a cell,
    and an LRU eviction of a cold container is the same operation.
    """
    marker = getattr(pipeline, _MARKER_ATTR, None)
    if not isinstance(marker, dict):
        return False
    dropped = False
    for target, row in dict(marker.get("targets") or {}).items():
        dispatch = _dispatch_for(marker, target)
        if dispatch is None or str(name) not in dict(dispatch.runners):
            continue
        dropped = dispatch.remove(str(name), str(reason)) or dropped
        if not dispatch.runners:
            # The target no longer serves anything compiled. Restore its eager
            # callable rather than leaving a wrapper that only ever falls back.
            state = row.get("state") or {}
            state["failed"] = True
    marker.get("entries", {}).pop(str(name), None)
    marker["bound_constants"]["literals"].pop(str(name), None)
    return dropped


def entry_states(pipeline: Any) -> Dict[str, Dict[str, Any]]:
    """Per-entry SERVE STATE — what this pod actually serves, per graph class.

    pgw#1176 §1.4: the pod never claims "cell X armed". It reports, per entry,
    ``armed`` / ``de_armed(reason)`` / ``pending`` — so there is no unit left
    that can advertise more than it serves. This is the record the per-entry
    hub events (§1.7) are built from.
    """
    marker = getattr(pipeline, _MARKER_ATTR, None) or {}
    out: Dict[str, Dict[str, Any]] = {}
    for target in (marker.get("targets") or {}):
        dispatch = _dispatch_for(marker, target)
        if dispatch is None:
            continue
        for name, runner in dispatch.runners:
            out[name] = {
                "state": "armed", "target": target, "calls": int(runner.calls)}
        for name, why in dispatch.de_armed.items():
            out[name] = {"state": "de_armed", "target": target, "reason": why}
        for name in dispatch.pending:
            out[name] = {"state": "pending", "target": target}
    return out


def _adopt_identity(artifact: Path) -> str:
    """Best-effort exact TCG key for a typed adopt event."""
    try:
        from gen_worker._vendor.torchcg.artifact import read_metadata

        meta = read_metadata(artifact)
    except Exception:  # noqa: BLE001 - identity is diagnostic on a refusal
        return f"artifact={artifact.name}"
    return f"key={meta.get('compiled_graph_key')}"


def _import_and_arm(
    pipeline: Any,
    cfg: Any,
    artifact: Path,
    cache_dir: Optional[Path],
    *,
    expected: "Optional[aot_identity.ExpectedIdentity]",
    declared: Sequence[str],
) -> Dict[str, Any]:
    from gen_worker._vendor.torchcg.artifact import read_metadata

    transfer_staging = (
        artifact.parent.name == ".incoming"
        and artifact.parent.parent.name == "compiled-graph-transfer"
    )
    try:
        metadata = read_metadata(artifact)
        key = str(metadata.get("compiled_graph_key") or "").strip()
        if not is_compiled_graph_key(key):
            raise AdoptError(
                "compiled_graph_key_invalid",
                f"artifact names no canonical compiled-graph key: {key!r}",
            )
        if expected is not None and expected.compiled_graph_key != key:
            raise AdoptError(
                "compiled_graph_key_mismatch",
                f"the arm named {expected.compiled_graph_key!r}, artifact names {key!r}",
            )
        publication = open_worker_engine(cache_dir).import_artifact(key, artifact)
        if publication.outcome == StoreOutcome.DIVERGENT:
            raise AdoptError(
                "compiled_graph_divergent",
                f"local TCG already binds {key!r} to different admitted bytes",
            )
        return arm_compiled_graph(
            pipeline, cfg, key, cache_dir, declared=declared
        )
    finally:
        if transfer_staging:
            artifact.unlink(missing_ok=True)


def enable(
    pipeline: Any,
    cfg: Any,
    cache_dir: Optional[Path] = None,
    artifact: Optional[Path] = None,
    *,
    expected: "Optional[aot_identity.ExpectedIdentity]" = None,
    declared: Sequence[str] = (),
) -> AdoptOutcome:
    """Consumer entry point: verify + load + bind + register ONE entry.

    Falsy (staying eager) on ANY miss — the caller's ordinary miss policy
    (fleet self-mint / eager / typed refusal) takes over. Truthy IS the HIT:
    ``fleet_cells`` treats it as a genuine match and
    skips the self-mint.

    pgw#1176: a miss here is a miss for ONE graph class. Nothing about it
    un-arms a sibling, and the caller's retry/mint policy is per class too.

    pgw#923: the outcome is RETURNED rather than narrated. The classified
    refusal reason used to leave this function only as the ``phase`` of a
    free-text ``aot_adopt`` event, which is why the adoption that actually
    happens on every boot had no measured row anywhere — the caller could not
    see what it had just been told.
    """
    if artifact is None:
        return AdoptOutcome.miss("no_artifact")
    try:
        meta = _import_and_arm(
            pipeline,
            cfg,
            Path(artifact),
            cache_dir,
            expected=expected,
            declared=declared,
        )
    except Exception as exc:
        reason = str(getattr(exc, "reason", "") or "") or type(exc).__name__
        identity = _adopt_identity(Path(artifact))
        logger.warning(
            "aot-serve: entry unusable (%s: %s); this class serves eager",
            reason, exc)
        return AdoptOutcome.miss(
            reason, f"{identity}: {type(exc).__name__}: {exc}", identity)
    entry = dict(meta.get(GRAPH_CLASS_BLOCK) or {})
    armed = len(armed_entries(pipeline))
    logger.info(
        "aot-serve: armed %s entry %s (sku=%s torch=%s precision=%s, "
        "constants bound BY REFERENCE from resident weights); %d armed",
        meta.get("family"), entry.get("name"),
        meta.get("sm"), "TCG", "code-only", armed)
    return AdoptOutcome.hit(
        f"family={meta.get('family')} key={meta.get('compiled_graph_key')} "
        f"entry={entry.get('name')} armed={armed} sm={meta.get('sm')}")


def _marker_states(subject: Any) -> List[Dict[str, Any]]:
    """Every wrapped target's state dict on a marker — PIPELINE or MODULE.

    pgw#1176, CORRECTED. I first deleted the single-``state`` branch here as
    "a legacy shape only tests build". That was half right and the wrong
    half: a bare ``state`` on a PIPELINE is indeed a shape nothing produces
    (and no fixture builds one any more), but this function is also called
    with a MODULE, and a bare ``state`` is exactly what :func:`wrap_module`
    writes there — on every arm, in production. Deleting it made
    `execution_count(module)` answer 0 for a module that had served.

    So the branch is not legacy and stays; what it reads is named. The two
    markers are different objects and always were: :func:`wrap_module` writes
    ``state`` on the MODULE it swapped, while :func:`arm_compiled_graph`
    writes ``targets`` on the PIPELINE that owns it.
    """
    marker = getattr(subject, _MARKER_ATTR, None) or {}
    rows = marker.get("targets")
    if isinstance(rows, dict):
        return [row.get("state") or {} for row in rows.values()]
    state = marker.get("state")
    return [state] if isinstance(state, dict) and state else []


def execution_count(pipeline: Any) -> int:
    """Successful artifact calls observed on this exact wrapped pipeline,
    summed over every wrapped target."""
    total = 0
    for state in _marker_states(pipeline):
        runner = state.get("runner")
        if runner is not None:
            total += int(getattr(runner, "calls", 0))
        else:
            total += int(state.get("successful_calls", 0))
    return total


def ingress_refusals(pipeline: Any) -> int:
    """Out-of-contract calls refused by name on this pipeline (B2),
    summed over every wrapped target."""
    return sum(
        int(state.get("ingress_refusals", 0))
        for state in _marker_states(pipeline))


def realigned_inputs(pipeline: Any) -> Dict[str, int]:
    """``"<input>/<reason>" -> count`` of ingress realignments (pgw#791).

    Zero is the contract holding. Non-zero is the tax MEASURED rather than
    inferred from a stderr line nobody can read — and it is paid at ingress
    into an owned buffer, not by the runner per call.

    pgw#1035 audited this for deletion and KEPT it. It has no fleet reader —
    but neither do its two siblings :func:`ingress_refusals` and
    :func:`served_entry_calls` (the mint rig drives the latter), and deleting
    one of three sibling measurements would make the #791 tax unobservable
    while leaving the machinery that computes it. This is the built-but-UNWIRED
    class, whose fix is a caller — a JobMetrics or activity field carrying all
    three — not a diff that hides the gap.
    """
    out: Dict[str, int] = {}
    for state in _marker_states(pipeline):
        runner = state.get("runner")
        counts = getattr(runner, "realignment_counts", None)
        rows = counts() if callable(counts) else getattr(runner, "realigned", {})
        for key, count in dict(rows or {}).items():
            out[key] = out.get(key, 0) + int(count)
    return out


def served_entry_calls(pipeline: Any) -> Dict[str, int]:
    """``entry -> served calls`` across every wrapped target (pgw#790).

    Which GRAPH CLASS served, not just that something did: an adapter-forked
    cell is only doing its job when adapter-free traffic lands on the
    branchless entry.
    """
    out: Dict[str, int] = {}
    for state in _marker_states(pipeline):
        runner = state.get("runner")
        getter = getattr(runner, "entry_calls", None)
        rows = getter() if callable(getter) else {}
        for name, count in dict(rows or {}).items():
            out[name] = out.get(name, 0) + int(count)
    return out


def proven_since(pipeline: Any, before: int) -> bool:
    """The exported lane's ADOPTION PROOF (pgw#735).

    An exported artifact performs no FX cache lookup, so the dynamo lane's
    ``cache_hit_count > 0`` proof can never pass for it — scoring a ``.pt2``
    that way disproves every honest adoption. Its own evidence is: it EXECUTED
    since ``before``, and it is STILL armed (an artifact that ran and then
    revoked on a B1/B2 refusal has not proven anything). Both halves are
    load-bearing, and neither is a synthesized counter: this is the one path
    whose entire job is to detect a lie about serving compiled.
    """
    return execution_count(pipeline) > int(before) and is_armed(pipeline)


def armed_targets(pipeline: Any) -> Dict[str, Dict[str, Any]]:
    """Every wrapped target of an armed pipeline, keyed by target name.

    Each row carries ``module``, ``attr`` and the wrap ``state`` — whose
    ``original`` is the EAGER callable the cell replaced and whose ``runner``
    is that target's :class:`EntryDispatch`. The pgw#868 numerics probe needs
    exactly those two to run the cell and its own reference on one feed, and a
    private marker read from another module would be a second interpretation
    of this one's format. ``{}`` when nothing is armed.
    """
    marker = getattr(pipeline, _MARKER_ATTR, None) or {}
    rows = marker.get("targets")
    if not isinstance(rows, dict):
        return {}
    return {str(name): dict(row) for name, row in rows.items()}


def armed_metadata(pipeline: Any) -> Dict[str, Any]:
    """The metadata the ARM itself used, off the live marker.

    The authority for anything asked about an armed cell: a caller that
    re-unpacked the artifact could be reading a different file than the one
    :func:`load_and_wrap` staged, verified and bound.
    """
    marker = getattr(pipeline, _MARKER_ATTR, None) or {}
    meta = marker.get("meta")
    return dict(meta) if isinstance(meta, dict) else {}


def armed_entries(pipeline: Any) -> Dict[str, str]:
    """``entry name -> entry key`` for every graph class ARMED right now.

    pgw#1176: this — not a cell-level boolean — is what the pod may claim. A
    subset is a legitimate steady state, so "how many, and which" is the only
    honest answer to "what does this pod serve compiled".
    """
    marker = getattr(pipeline, _MARKER_ATTR, None) or {}
    rows = dict(marker.get("entries") or {})
    out: Dict[str, str] = {}
    for target in (marker.get("targets") or {}):
        dispatch = _dispatch_for(marker, target)
        if dispatch is None:
            continue
        for name, _runner in dispatch.runners:
            out[name] = str(
                (rows.get(name) or {}).get("compiled_graph_key") or ""
            )
    return out


def is_armed(pipeline: Any) -> bool:
    """Whether this pipeline is serving ANY compiled graph class right now.

    pgw#1176 DELETED the every-target rule ("one revoked target means the
    cell no longer serves the contract its key advertises"). That rule was
    the arming half of the wrong atom: a key that advertised 36 classes made
    partial service a lie, so the guard was locally correct — and it is what
    forbade the incremental compile-and-adopt this design exists to deliver.
    A key now advertises ONE class, an entry arms whole or not at all, and a
    de-armed entry costs itself. Mixed compiled/eager service is not a
    degraded state; it is the design's normal one, and it is numerically as
    proven as full coverage because every armed entry passed the same mint
    parity gate against the same eager reference.

    Ask :func:`armed_entries` / :func:`entry_states` when you need to know
    WHAT is served rather than WHETHER anything is.
    """
    return bool(armed_entries(pipeline))


def holds_exported_cell(pipeline: Any) -> bool:
    """Whether an AOTI cell is WRAPPED onto this object — armed or revoked.

    pgw#1141b: the LANE question, asked of the object instead of a ref string.
    "Is this the exported lane?" decides which failure detector applies — the
    dynamo lane's per-class cache-hit ledger (which an AOTI artifact can never
    move, so it reads every honest adoption as a disproof) or §4.31's
    serve-first rule. Answering it through :func:`is_aot_ref` made the answer
    depend on whether some earlier caller had announced the key to this
    process; the wrap is the fact itself.

    Distinct from :func:`is_armed` on purpose: a cell whose guard revoked it
    still HOLDS this pipeline's eager originals, and the install path has to
    tell "revoked exported cell" (never advertise it) apart from "no cell here
    at all" (an ordinary dynamo/eager object).
    """
    return bool(_marker_states(pipeline))


def set_guard_failure_callback(pipeline: Any, callback: Any) -> bool:
    """Bind scheduler-state revocation to every wrapped target's guard."""
    states = _marker_states(pipeline)
    if not states:
        return False
    for state in states:
        state["failure_callback"] = callback
    return True


def unwrap(pipeline: Any) -> bool:
    """Restore every wrapped target's eager callable — rotation/eviction
    and the unproven-adoption rollback both go through here."""
    marker = getattr(pipeline, _MARKER_ATTR, None) or {}
    # One pipeline-marker shape, written by `arm_compiled_graph`. The
    # `module`/`state` fallback that stood here read a shape no production
    # path produces — see :func:`_marker_states`.
    targets = marker.get("targets")
    rows: List[Dict[str, Any]] = (
        list(targets.values()) if isinstance(targets, dict) else [])
    restored = False
    for row in rows:
        module = row.get("module")
        state = row.get("state") or {}
        original = state.get("original")
        if module is None or not callable(original):
            continue
        setattr(module, str(row.get("attr") or "forward"), original)
        try:
            delattr(module, _MARKER_ATTR)
        except AttributeError:
            pass
        restored = True
    if restored:
        try:
            delattr(pipeline, _MARKER_ATTR)
        except AttributeError:
            pass
    return restored


__all__ = [
    "AdoptOutcome",
    "COMPILED_GRAPH_FORMAT",
    "COMPILED_GRAPH_FORMAT_KEY",
    "ConstantsUnboundError",
    "EntryDispatch",
    "IDENTITY_AXES",
    "IngressContractError",
    "TCGEntryRunner",
    "aligned_feeds",
    "armed_entries",
    "armed_metadata",
    "armed_targets",
    "arm_compiled_graph",
    "assert_lifted_contract",
    "assert_ingress",
    "bind_call_inputs",
    "disarm_entry",
    "enable",
    "entry_from_meta",
    "entry_states",
    "execution_count",
    "holds_exported_cell",
    "ingress_class_name",
    "ingress_refusals",
    "is_aot_ref",
    "is_armed",
    "lifted_call_kwargs",
    "marshal_positional",
    "proven_since",
    "realigned_inputs",
    "resident_constants",
    "served_entry_calls",
    "set_guard_failure_callback",
    "set_ingress_refusal_callback",
    "report_ingress_refusal",
    "unwrap",
    "wrap_module",
]
