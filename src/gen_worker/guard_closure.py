"""Guard-closure ADVISORY audit + boundary canonicalization.

Dynamo's guard set for a compiled graph IS the exhaustive list of everything
that graph depends on, so closure against the declared compile contract is
machine-checkable. Three pieces, all SDK-generic (parameterized ONLY by the
declared ``CompileCell`` contract — never per-endpoint or per-family code):

1. **Extraction** (:func:`extract`): post-mint, walk every compiled graph's live
   guard tree. The robust torch 2.13 surface:
   ``torch._dynamo.eval_frame._debug_get_cache_entry_list(code)`` ->
   ``entry.guard_manager.root`` -> recursive ``get_child_managers()`` (each
   carries ``get_source()``) + ``get_leaf_guards()``. A repr parse of the
   ``TREE_GUARD_MANAGER`` dump is the fallback when the structured walk is
   unavailable. An entry whose guards cannot be read records an
   :data:`UNPROVEN` row, never a silent pass.

2. **Closure classifier** (:func:`classify`): the RelationalGuard family
   (aliasing, symbolic shapes) is judged by TYPE first — torch attaches it to
   input managers, never predictably to one root. Every other guard is
   classified by its SOURCE ROOT: module-rooted (``L['self']…``) guards are the
   weights/structure identity (covered by the family + graph_signature +
   weight_contract key axes) and global-rooted (``G[…]``) guards are the code
   identity (covered by the gen_worker/diffusers/transformers/image_digest
   axes); neither can vary per request. Cross-request variance enters ONLY
   through call inputs and ambient process state, so those two roots form a
   CLOSED WORLD: every input/ambient guard type must be explicitly covered by
   the contract or canonicalized away at the ingress — anything else is a LEAK.

   **The classifier has NO VETO** (Paul's ruling). It protects against WASTED
   REUSE, not incorrectness: dynamo re-evaluates its own guards on every call at
   the consumer, so a cell depending on unpinned state fails its guards THERE —
   and with fail-on-recompile armed that raises, serves eager, and reports a
   countable ``guard_miss``. So :func:`closure_manifest` classifies,
   records, emits a typed ``guard_leak`` event, and CONTINUES the mint. Hard
   refusals survive only where a defect is PROVEN rather than inferred: zero
   compiled graphs, and posture drift (:func:`assert_posture`).

   **Never rebuild this classifier.** torch's own ``guard_filter_fn`` /
   ``GuardFilterEntry`` is the supported structured hook (typed entries, no C++
   manager walk and no repr parsing), and upstream maintains a versioned
   ``UNSUPPORTED_SERIALIZATION_GUARD_TYPES``
   classification. torch.export / AOTI removes the question entirely, and this
   module is DELETED when the last family migrates to AOT. If a JIT-resident
   family ever needs the check, implement it as a thin call into
   ``guard_filter_fn`` plus upstream's serializability list.

3. **Boundary canonicalization** (:func:`canonical_ingress`): one wrapper at the
   single compiled-graph ingress (installed by ``compile_cache.apply`` for every
   target, mint and consumer alike, so traced guards and serving inputs see the
   SAME canonical form). Tensor strides are pinned to the canonical contiguous
   layout — ``.contiguous()`` alone is NOT the pin: a size-1 dim makes
   ``is_contiguous()`` true under an arbitrary stride and ``.contiguous()`` a
   no-op while TENSOR_MATCH still guards the exact stride tuple, so the residue
   case rebuilds via ``as_strided``. Tensor dtypes are memo-asserted per
   argument path — a drift raises a NAMED :class:`GuardBoundaryError` instead of
   a silent recompile. Scalar policy is REPORTED by the classifier; the ingress
   never rewrites scalar values.

The full guard dump rides the cell as ``metadata.json``'s ``guard_manifest``
block (deterministic: comments stripped, ASLR ids scrubbed, rows sorted). The
audit surface (:func:`audit_armed`, :func:`consolidate` + ``python -m
gen_worker.guard_closure``) is the N-cold-pod zero-miss closure check: exit 0 =
closed and consistent, 2 = leaks, 3 = cross-pod divergence.

Note: ``_debug_get_cache_entry_list`` keys on the class-shared ``__code__``, so
a warm process's dump can include same-family sibling entries; classification is
source-based, so siblings classify identically.
"""

from __future__ import annotations

import ast
import functools
import logging
import re
from dataclasses import dataclass
from typing import (Any, Callable, Dict, Iterable, List, Mapping, Optional,
                    Sequence, Tuple)

from . import torch_capability

logger = logging.getLogger(__name__)

# pgw#958 (§1.27(g)): restarted at 1 alongside KEY_SCHEME / SEAL_VERSION,
# with the pre-existing cell corpus purged in the same cut. The manifest
# shape is the one the old v3 reached (posture process-seal block, advisory
# gate, "unproven" rows).
MANIFEST_VERSION = 1
GATE_KEY = "gate"
GATE_ADVISORY = "advisory"

# Verdicts. LEAK and UNPROVEN are REPORTED, never fatal.
LEAK = "LEAK"
UNPROVEN = "unproven"                    # guards unreadable on this torch build
RUNTIME_STATE = "runtime-state"          # ambient process state (cell-key runtime axes)
CODE_IDENTITY = "code-identity"          # G[...] roots (version/image axes)
MODULE_STRUCTURE = "module-structure"    # L['self'] roots (family/graph/weight axes)
CONTRACT_SHAPE = "contract-shape"        # input tensor shape/stride/dtype/device
CONTRACT_SCALAR = "contract-scalar"      # input scalar pinned by the contract
STRUCTURAL = "structural"                # type/len/None/bool topology of the call
CODE_CONSTANT = "code-constant"          # string constants from the call path
CANONICALIZED = "canonicalized-stride"   # stride axes the ingress pins

_COVERED = (
    RUNTIME_STATE, CODE_IDENTITY, MODULE_STRUCTURE, CONTRACT_SHAPE,
    CONTRACT_SCALAR, STRUCTURAL, CODE_CONSTANT, CANONICALIZED,
)

# The vocabulary below is built on the C++ leaf CLASS names the structured
# walk yields (`type(leaf).__name__` over the 31 concrete LeafGuard
# subclasses in torch._C._dynamo.guards on 2.13.0 — the repr-parse fallback
# prints the same names). Python GuardBuilder create_fn names survive only
# where they harmlessly alias a covered fact; they never reach the walk
# (pgw#691 audit).

# RelationalGuard family (C++): torch attaches these to the INPUT guard
# managers — one row per participating value, never the root — so they are
# judged by TYPE before any root dispatch. Root-first dispatch made
# NO_TENSOR_ALIASING an unconditional LEAK on every graph with >=2 tensor
# inputs (pgw#691 P0). Cross-input (non-)aliasing/overlap is fixed by the
# endpoint call topology: the compiled target receives freshly materialized,
# distinct tensors each request, so the relation cannot vary per call.
_RELATIONAL_ALIASING = frozenset({
    "NO_TENSOR_ALIASING", "OBJECT_ALIASING", "STORAGE_OVERLAPPING",
    "DUPLICATE_INPUT",  # python GuardBuilder alias of OBJECT_ALIASING
})
# Ambient (root-attached) guard types covered by process/runtime identity.
# DETERMINISTIC_ALGORITHMS/GRAD_MODE/TORCH_FUNCTION_STATE/FUNCTORCH_STACK_
# MATCH/DUAL_LEVEL are python-side aliases whose facts GLOBAL_STATE and
# DUAL_LEVEL_MATCH (the C++ names) carry on 2.13.
_AMBIENT_COVERED = frozenset({
    "GLOBAL_STATE", "TORCH_FUNCTION_MODE_STACK", "DEFAULT_DEVICE",
    "DUAL_LEVEL_MATCH", "DETERMINISTIC_ALGORITHMS", "GRAD_MODE",
    "TORCH_FUNCTION_STATE", "FUNCTORCH_STACK_MATCH", "DUAL_LEVEL",
})
# Input-rooted guard types that assert call TOPOLOGY (types, lengths,
# presence), which the endpoint call path fixes deterministically.
# FAKE_SCRIPT_TYPE_MATCH asserts a torchscript-facing input type — same
# topology fact as TYPE_MATCH.
_STRUCTURAL_TYPES = frozenset({
    "TYPE_MATCH", "DIMENSION_DYNAMIC_MARKING_GUARD", "HASATTR", "NO_HASATTR",
    "LENGTH_CHECK", "DICT_LENGTH", "DICT_CONTAINS", "SET_CONTAINS",
    "MAPPING_KEYS_MATCH", "TUPLE_ITERATOR_LEN", "FAKE_SCRIPT_TYPE_MATCH",
    "RANGE_ITERATOR_MATCH", "NONE_MATCH", "NOT_NONE", "NOT_NONE_MATCH",
    "TRUE_MATCH", "FALSE_MATCH", "DISPATCH_KEY_SET_MATCH", "FLOAT_IS_NAN",
    "COMPLEX_IS_NAN", "SEQUENCE_LENGTH",
})
# Input-rooted object-identity guard types: an object-valued argument whose
# identity dynamo guards (enum member, callable, config dict) can only be
# bound by the endpoint's fixed call path, so it cannot vary per request;
# across processes an identity mismatch is a MISS (the pgw#680 guard-miss
# path recompiles), never wrongness — CODE_CONSTANT, not a leak.
_IDENTITY_CONSTANT_TYPES = frozenset({"ID_MATCH", "DICT_VERSION"})

_CANONICAL_DEPTH = 4

# `___check_obj_id(x, 129760138209744)` / `___check_type_id(x, 21008688)`
# embed process memory addresses (ASLR) — scrub for cross-pod determinism.
_ID_SCRUB_RE = re.compile(r"(___check_(?:obj|type)_id\(.*?,\s*)\d+(\))")
_COMMENT_RE = re.compile(r"\s{2,}#.*$", re.DOTALL)
_TENSOR_MATCH_RE = re.compile(r"size=\[([^\]]*)\], stride=\[([^\]]*)\]")
_SOURCE_LOCAL_RE = re.compile(r"^L\['([^']+)'\]")
# torch 2.13 emits DERIVED guard sources that CONTAIN a local root
# without starting with it — `dict(type(L['self']).__mro__[1].__dict__)` for a
# base-class @property (diffusers ConfigMixin.config, read in every model's
# forward) and `type(L['self']._modules['x']).__dict__['forward'].__defaults__`
# for a submodule forward with a default argument. Prefix-matching alone sent
# both to the "other" root, which LEAKS before the type dispatch that already
# covers them — refusing every self-mint fleet-wide.
_SOURCE_EMBEDDED_RE = re.compile(r"L\['([^']+)'\]")
_SOURCE_EMBEDDED_GLOBAL_RE = re.compile(r"\bG[\['.]")


class GuardClosureError(RuntimeError):
    """The mint produced no readable compiled graphs, or a stored manifest
    is unreadable. NOTE: a classification LEAK no longer raises —
    it is recorded in the manifest and emitted as a ``guard_leak`` event."""


class GuardBoundaryError(RuntimeError):
    """An input crossed the compiled-graph ingress outside the canonical
    boundary (dtype drift). Named — never a silent recompile. Consumer
    guards catch it into the pgw#672 explicit-eager degrade; mint arms
    (``guard=False``) let it fail the mint red."""


class PostureError(GuardClosureError):
    """The process posture differs from the canonical serving posture or
    from a cell's sealed posture. Named per fact — a posture
    drift refuses the mint/arm loudly instead of surfacing later as an
    undiagnosable ambient guard miss."""


@dataclass(frozen=True)
class GuardRecord:
    """One classified dynamo guard."""

    guard_type: str
    source: str  # "" for root/ambient guards
    expr: str    # comment-stripped, id-scrubbed
    verdict: str
    axis: str    # covering contract/runtime axis, or the leak reason

    def row(self) -> Dict[str, str]:
        return {
            "type": self.guard_type, "source": self.source,
            "expr": self.expr, "verdict": self.verdict, "axis": self.axis,
        }


@dataclass(frozen=True)
class GraphGuards:
    """The complete guard set of one compiled graph (one cache entry)."""

    target: str
    code: str
    entry: int
    guards: Tuple[GuardRecord, ...]


@dataclass(frozen=True)
class ClosureReport:
    """The audit view for one armed pipeline / minted cell."""

    graphs: Tuple[GraphGuards, ...]

    @property
    def leaks(self) -> Tuple[str, ...]:
        out: List[str] = []
        for g in self.graphs:
            for r in g.guards:
                if r.verdict == LEAK:
                    out.append(
                        f"target={g.target} {r.guard_type} "
                        f"{r.source or '<ambient>'}: {r.expr} ({r.axis})")
        return tuple(out)

    @property
    def unproven(self) -> Tuple[str, ...]:
        """Entries whose guards could not be read. Reported, never
        fatal: this is a fact about the torch build's guard debug surface,
        not about the mint."""
        return tuple(
            f"target={g.target} entry={g.entry}: {r.expr}"
            for g in self.graphs for r in g.guards if r.verdict == UNPROVEN)

    @property
    def closed(self) -> bool:
        return bool(self.graphs) and not self.leaks and not self.unproven

    def verdict_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for g in self.graphs:
            for r in g.guards:
                counts[r.verdict] = counts.get(r.verdict, 0) + 1
        return dict(sorted(counts.items()))

    def manifest(self) -> Dict[str, Any]:
        """Deterministic JSON manifest — the durable per-cell guard dump."""
        return {
            "v": MANIFEST_VERSION,
            "graphs": [
                {
                    "target": g.target, "code": g.code, "entry": g.entry,
                    "guards": [r.row() for r in g.guards],
                }
                for g in self.graphs
            ],
            "verdicts": self.verdict_counts(),
            "leaks": list(self.leaks),
            "unproven": list(self.unproven),
            GATE_KEY: GATE_ADVISORY,
        }

    def text(self) -> str:
        lines = [
            f"guard closure: {len(self.graphs)} graph(s), "
            f"verdicts={self.verdict_counts()}",
        ]
        for leak in self.leaks:
            lines.append(f"  LEAK {leak}")
        for row in self.unproven:
            lines.append(f"  UNPROVEN {row}")
        if not self.graphs:
            lines.append("  (no compiled graphs extractable)")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Contract pins
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ContractPins:
    """The scalar/dynamic vocabulary one declared contract admits into
    guards. 0/1 are torch's own free specializations; everything else must
    be a declared axis value."""

    ints: frozenset
    floats: frozenset
    has_dynamic: bool
    # Names the traced callable CLOSES OVER. A freevar renders exactly like a
    # call input in a guard source, so the classifier needs the
    # callable's own vocabulary to tell them apart.
    freevars: frozenset = frozenset()


def contract_pins(
    cfg: Any, freevars: Iterable = (),
) -> ContractPins:
    ints = {0, 1}
    for row in tuple(getattr(cfg, "shapes", ()) or ()):
        ints.update(int(v) for v in row)
    ints.update(int(v) for v in tuple(getattr(cfg, "text_lens", ()) or ()))
    text_len = getattr(cfg, "text_len", None)
    if text_len is not None:
        ints.add(int(text_len))
    bucket = int(getattr(cfg, "lora_bucket", 0) or 0)
    if bucket:
        ints.add(bucket)
    dynamic = tuple(getattr(cfg, "dynamic", ()) or ())
    for d in dynamic:
        ints.add(int(getattr(d, "min", 0)))
        ints.add(int(getattr(d, "max", 0)))
    floats = {0.0, 1.0}
    floats.update(float(v) for v in tuple(getattr(cfg, "guidance_scales", ()) or ()))
    return ContractPins(
        ints=frozenset(ints), floats=frozenset(floats),
        has_dynamic=bool(dynamic), freevars=frozenset(freevars))


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def _normalize(expr: str) -> str:
    out = _COMMENT_RE.sub("", str(expr)).strip()
    return _ID_SCRUB_RE.sub(r"\g<1><id>\g<2>", out)


def _walk_manager(manager: Any, source: str, out: List[Tuple[str, str, str]]) -> None:
    for leaf in manager.get_leaf_guards():
        parts_attr = getattr(leaf, "verbose_code_parts", None)
        parts = list(parts_attr() if callable(parts_attr) else (parts_attr or ()))
        if not parts:
            parts = [type(leaf).__name__]
        for part in parts:
            out.append((type(leaf).__name__, source, str(part)))
    for child in manager.get_child_managers():
        get_source = getattr(child, "get_source", None)
        child_source = str(get_source()) if callable(get_source) else source
        _walk_manager(child, child_source or source, out)


_TREE_LEAF_RE = re.compile(r"^\|(?: \|)* \+- ([A-Z0-9_]+): (.*)$")
_TREE_MGR_RE = re.compile(r"^\|(?: \|)* \+- \w*GuardManager: source=([^,]+),")


def _parse_tree_repr(dump: str) -> List[Tuple[str, str, str]]:
    """Fallback: parse the ``TREE_GUARD_MANAGER`` repr into (type, source,
    raw expr) rows. Depth-tracked so leaves attach to the nearest manager."""
    out: List[Tuple[str, str, str]] = []
    source_by_depth: Dict[int, str] = {}
    for line in dump.splitlines():
        depth = line.count("| ")
        mgr = _TREE_MGR_RE.match(line)
        if mgr:
            source_by_depth[depth] = mgr.group(1).strip()
            continue
        leaf = _TREE_LEAF_RE.match(line)
        if leaf:
            source = ""
            for d in sorted(source_by_depth):
                if d < depth:
                    source = source_by_depth[d]
            out.append((leaf.group(1), source, leaf.group(2)))
    return out


def _code_of(fn: Any) -> Optional[Any]:
    return getattr(getattr(fn, "__func__", fn), "__code__", None)


def _entry_guard_rows(entry: Any) -> List[Tuple[str, str, str]]:
    manager = getattr(entry, "guard_manager", None)
    if manager is None:
        raise GuardClosureError(
            "dynamo cache entry exposes no guard_manager — the guard debug "
            "surface changed; closure is unprovable on this torch")
    root = getattr(manager, "root", None)
    if root is not None:
        try:
            rows: List[Tuple[str, str, str]] = []
            _walk_manager(root, "", rows)
            if rows:
                return rows
        except Exception:
            logger.warning(
                "guard-closure: structured guard walk failed; falling back "
                "to the repr parse", exc_info=True)
    rows = _parse_tree_repr(str(manager))
    if not rows:
        raise GuardClosureError(
            "no guards extractable from a live cache entry (structured walk "
            "and repr parse both empty) — closure is unprovable")
    return rows


def extract_target_guards(
    fn: Any, target: str, cfg: Any,
) -> List[GraphGuards]:
    """Classified guard sets for every live compiled graph on ``fn``."""
    code = _code_of(fn)
    if code is None:
        return []
    from torch._dynamo.eval_frame import _debug_get_cache_entry_list

    pins = contract_pins(cfg, getattr(code, "co_freevars", ()) or ())
    graphs: List[GraphGuards] = []
    for index, entry in enumerate(_debug_get_cache_entry_list(code)):
        try:
            rows = _entry_guard_rows(entry)
        except Exception as exc:  # noqa: BLE001 — recorded, never fatal
            # The guards of a LIVE compiled graph were unreadable.
            # That is a torch-surface fact, not a mint defect, so it is
            # recorded as an UNPROVEN row and the mint continues.
            logger.warning(
                "guard-closure: %s entry %d guards unreadable (%s: %s) — "
                "recorded UNPROVEN, mint continues",
                target, index, type(exc).__name__, exc)
            records = [GuardRecord(
                guard_type="EXTRACTION", source="",
                expr=f"{type(exc).__name__}: {exc}", verdict=UNPROVEN,
                axis="guard debug surface unreadable on this torch build")]
        else:
            records = sorted(
                {
                    _classify_row(guard_type, source, raw, pins)
                    for guard_type, source, raw in rows
                },
                key=lambda r: (r.guard_type, r.source, r.expr),
            )
        graphs.append(GraphGuards(
            target=str(target), code=str(getattr(code, "co_qualname", code.co_name)),
            entry=index, guards=tuple(records),
        ))
    return graphs


def extract(pipeline: Any, cfg: Any) -> ClosureReport:
    """Classified guard sets for every compiled graph on an armed pipeline
    (whole-graph targets via the marker's originals; regional targets via
    the repeated-block forwards — same enumeration the in-memory compile
    probe trusts)."""
    # CYCLE (see manifest_digest): compile_cache reaches back here via registry.
    from . import compile_cache as cc

    marker = getattr(pipeline, cc._MARKER_ATTR, None) or {}
    graphs: List[GraphGuards] = []
    seen_codes: set = set()
    for _owner, attr, fn in marker.get("originals") or ():
        code = _code_of(fn)
        if code is None or id(code) in seen_codes:
            continue
        seen_codes.add(id(code))
        graphs.extend(extract_target_guards(fn, str(attr), cfg))
    for mod in marker.get("regional_mods") or ():
        try:
            children = list(mod.modules())
        except Exception:
            logger.warning(
                "guard-closure: could not enumerate regional submodules of "
                "%s", type(mod).__name__, exc_info=True)
            continue
        for child in children:
            fwd = getattr(type(child), "forward", None)
            code = _code_of(fwd)
            if code is None or id(code) in seen_codes:
                continue
            seen_codes.add(id(code))
            graphs.extend(extract_target_guards(
                fwd, f"{type(child).__name__}.forward", cfg))
    graphs.sort(key=lambda g: (g.target, g.code, g.entry))
    return ClosureReport(graphs=tuple(graphs))


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def _source_root(source: str, freevars: frozenset = frozenset()) -> str:
    """The ROOT a guard source is rooted at: self / input / freevar / global /
    ambient, or "other" when nothing recognizable is embedded.

    resolves the root EMBEDDED anywhere in a derived source, not just
    a prefix. torch 2.13 wraps class/code-structure facts in expressions like
    ``dict(type(L['self']).__mro__[1].__dict__)`` — semantically self-rooted,
    and covered by the type dispatch — but a prefix match sees "other" and
    LEAKS. The closed world is unchanged: a source with no recognizable root is
    still "other", still a leak.
    """
    s = str(source or "").strip()
    if not s:
        return "ambient"
    if s.startswith("G[") or s.startswith("G."):
        return "global"
    m = _SOURCE_LOCAL_RE.match(s)
    if m is None:
        # Derived source: find the root it is DERIVED FROM. `self` wins when
        # present — a fact about self's class/code structure stays a
        # module-structure fact however it is spelled.
        names = _SOURCE_EMBEDDED_RE.findall(s)
        if "self" in names:
            return "self"
        if names:
            return "freevar" if names[0] in freevars else "input"
        if _SOURCE_EMBEDDED_GLOBAL_RE.search(s):
            return "global"
        return "other"
    name = m.group(1)
    if name == "self":
        return "self"
    # a closure FREEVAR renders exactly
    # like a call input, so `L['outer_scale'] == 0.3` was judged an input and
    # leaked with the wrong name. Production roots are bound `forward` methods
    # with no freevars; naming it correctly keeps the check and fixes the
    # diagnosis.
    return "freevar" if name in freevars else "input"


def _contiguous_strides(shape: Sequence) -> Tuple[int, ...]:
    out: List[int] = []
    step = 1
    for dim in reversed([int(v) for v in shape]):
        out.append(step)
        step *= max(dim, 1)
    return tuple(reversed(out))


def _classify_tensor_match(expr: str) -> Tuple[str, str]:
    m = _TENSOR_MATCH_RE.search(expr)
    if m is None:
        return CONTRACT_SHAPE, "shapes/dtype/device (contract + lane)"
    size_txt = [v.strip() for v in m.group(1).split(",") if v.strip()]
    stride_txt = [v.strip() for v in m.group(2).split(",") if v.strip()]
    if any(v == "None" for v in size_txt):
        # Declared-dynamic dims record None; dependent strides follow suit.
        return CONTRACT_SHAPE, "declared dynamic dims"
    try:
        size = [int(v) for v in size_txt]
        stride = tuple(int(v) for v in stride_txt if v != "None")
    except ValueError:
        return CONTRACT_SHAPE, "shapes/dtype/device (contract + lane)"
    want = _contiguous_strides(size)
    have = tuple(
        int(v) if v != "None" else want[i]
        for i, v in enumerate(stride_txt)
    ) if len(stride_txt) == len(size) else stride
    if have != want:
        return LEAK, (
            f"non-canonical stride {list(have)} for size {size} survived "
            "the ingress pin")
    return CANONICALIZED, "ingress stride pin + contract shapes"


def _scalar_verdict(value: Any, pins: ContractPins) -> Tuple[str, str]:
    if value is None or isinstance(value, bool):
        return STRUCTURAL, "call-path constant"
    if isinstance(value, (str, bytes)):
        return CODE_CONSTANT, "call-path string constant"
    if isinstance(value, int):
        if value in pins.ints:
            return CONTRACT_SCALAR, "declared contract int"
        return LEAK, f"int {value} is not a declared contract pin"
    if isinstance(value, float):
        if value in pins.floats or (value.is_integer() and int(value) in pins.ints):
            return CONTRACT_SCALAR, "declared contract float"
        return LEAK, f"float {value} is not a declared contract pin"
    if isinstance(value, (tuple, list)):
        for v in value:
            verdict, axis = _scalar_verdict(v, pins)
            if verdict == LEAK:
                return LEAK, f"element {axis}"
        return CONTRACT_SCALAR, "declared contract sequence"
    return LEAK, f"unclassifiable literal {type(value).__name__}"


# `L['dt'] == torch.float32` / `L['mf'] == torch.contiguous_format` — the
# RHS a torch singleton renders in verbose parts. `device(type='cuda',
# index=0)` is how a torch.device literal renders.
_TORCH_ATTR_RE = re.compile(r"^torch\.([A-Za-z_][A-Za-z0-9_]*)$")
_TORCH_DEVICE_RE = re.compile(r"^(?:torch\.)?device\(type=")


def _torch_object_verdict(rhs: str) -> Optional[Tuple[str, str]]:
    """Verdict for an EQUALS_MATCH RHS that is a torch singleton rather than
    a python literal (pgw#691 P1: diffusers compares dtypes constantly, and
    ``ast.literal_eval`` reported them as false-positive LEAKs). Dtype at the
    compiled boundary is already pinned twice over — the ``weight_lane`` key
    axis and :func:`canonical_ingress`'s per-path dtype memo (which raises a
    NAMED :class:`GuardBoundaryError` on drift); layout/memory_format ride
    the same canonical-form pin, device rides the runtime key axes."""
    if _TORCH_DEVICE_RE.match(rhs):
        return CONTRACT_SHAPE, "runtime device axes (sm/cuda/torch)"
    m = _TORCH_ATTR_RE.match(rhs)
    if m is None:
        return None
    try:
        import torch
    except ImportError:  # classification without torch: stay conservative
        return None
    obj = getattr(torch, m.group(1), None)
    if isinstance(obj, (torch.dtype, torch.layout, torch.memory_format)):
        return CONTRACT_SHAPE, "weight_lane + ingress dtype memo"
    return None


def _classify_equals(expr: str, pins: ContractPins) -> Tuple[str, str]:
    _lhs, sep, rhs = expr.partition(" == ")
    if not sep:
        return LEAK, "unparseable EQUALS_MATCH expression"
    # classify() receives the RAW verbose part; python literals survive the
    # trailing "  # file:line" comment (the tokenizer eats it) but the
    # torch-object regex must see a clean RHS.
    rhs = _COMMENT_RE.sub("", rhs).strip()
    try:
        value = ast.literal_eval(rhs)
    except Exception:
        torch_verdict = _torch_object_verdict(rhs)
        if torch_verdict is not None:
            return torch_verdict
        return LEAK, f"unparseable EQUALS_MATCH literal {rhs[:60]!r}"
    return _scalar_verdict(value, pins)


def classify(
    guard_type: str, source: str, expr: str, pins: ContractPins,
) -> Tuple[str, str]:
    """(verdict, covering axis / leak reason) for one guard.

    The RelationalGuard family is judged by TYPE first — torch attaches it
    to the input managers (and SYMBOLIC_SHAPE_GUARD to synthetic tuple
    sources), never predictably to one root. Everything else dispatches on
    the source root: module- and global-rooted guards are covered by
    construction (weights + code identity, pinned by the cell-key axes);
    input-rooted and ambient guards are the closed world — unknown types are
    LEAKS, never waved on.
    """
    if guard_type in _RELATIONAL_ALIASING:
        return STRUCTURAL, "cross-input aliasing fixed by call topology"
    if guard_type == "SYMBOLIC_SHAPE_GUARD":
        # torch 2.13's shape-env relations (install_symbolic_shape_guard,
        # enable_cpp_symbolic_shape_guards) — also a RelationalGuard,
        # observed with synthetic sources like "(0, L['x'].size()[0])".
        if pins.has_dynamic:
            return CONTRACT_SHAPE, "declared dynamic-dim range"
        return LEAK, "shape-env relation without declared dynamic dims"
    root = _source_root(source, pins.freevars)
    if root == "self":
        return MODULE_STRUCTURE, "family + graph_signature + weight_contract"
    if root == "global":
        return CODE_IDENTITY, "gen_worker/diffusers/transformers/image versions"
    if root == "ambient":
        if guard_type in _AMBIENT_COVERED:
            return RUNTIME_STATE, "process runtime state (cell-key runtime axes)"
        if guard_type == "LAMBDA_GUARD":
            if ("init_ambient_guards" in expr
                    or "top_saved_tensors_hooks" in expr):
                return RUNTIME_STATE, "torch ambient state"
            if ".size()" in expr or ".stride()" in expr:
                if pins.has_dynamic:
                    return CONTRACT_SHAPE, "declared dynamic-dim range"
                return LEAK, "shape-env relation without declared dynamic dims"
        return LEAK, f"unclassified ambient guard {guard_type}"
    if root == "other":
        return LEAK, f"unrecognized guard source {source!r}"
    # input- and freevar-rooted share ONE dispatch: a wrapper's
    # captured code object is code (its identity/structure is pinned by
    # code_closure + toolchain), while a captured runtime SCALAR is not — and
    # _scalar_verdict already draws exactly that line. What the freevar root
    # adds is the DIAGNOSIS: a leaking freevar must not be reported as a call
    # input, because the fix is a declaration, not a different request.
    verdict, axis = _classify_input_rooted(guard_type, expr, pins)
    if verdict == LEAK and root == "freevar":
        return LEAK, (
            f"{axis} — closure freevar {source!r}, captured by the traced "
            "code rather than passed by the caller")
    return verdict, axis


def _classify_input_rooted(
    guard_type: str, expr: str, pins: ContractPins,
) -> Tuple[str, str]:
    if guard_type == "TENSOR_MATCH":
        return _classify_tensor_match(expr)
    if guard_type in _STRUCTURAL_TYPES:
        return STRUCTURAL, "call topology (deterministic per endpoint code)"
    if guard_type in _IDENTITY_CONSTANT_TYPES:
        return CODE_CONSTANT, "object identity of a call-path constant"
    if guard_type == "EQUALS_MATCH":
        return _classify_equals(expr, pins)
    if guard_type == "LAMBDA_GUARD" and (".size()" in expr or ".stride()" in expr):
        if pins.has_dynamic:
            return CONTRACT_SHAPE, "declared dynamic-dim range"
        return LEAK, "shape-env relation without declared dynamic dims"
    return LEAK, f"input-dependent guard {guard_type} outside the contract"


def _classify_row(
    guard_type: str, source: str, raw_expr: str, pins: ContractPins,
) -> GuardRecord:
    verdict, axis = classify(guard_type, source, raw_expr, pins)
    return GuardRecord(
        guard_type=str(guard_type), source=str(source or ""),
        expr=_normalize(raw_expr), verdict=verdict, axis=axis)


# ---------------------------------------------------------------------------
# Process-posture seal
# ---------------------------------------------------------------------------

# The ONE canonical serving posture. Every fact is ambient process state a
# dynamo guard observes (GLOBAL_STATE, TORCH_FUNCTION_MODE_STACK,
# DEFAULT_DEVICE): mint and consumer must present the SAME posture or the
# minted guards can never HIT — and a drift must refuse the arm with a
# named reason, never surface as a downstream guard miss. String-valued
# for JSON determinism.
CANONICAL_POSTURE: Dict[str, str] = {
    "grad_enabled": "True",
    "inference_mode": "False",
    "autocast_cpu": "False",
    "autocast_cuda": "False",
    "torch_function_stack": "0",
    "default_device": "cpu",
    "deterministic_algorithms": "False",
    "deterministic_warn_only": "False",
}


def posture_snapshot() -> Dict[str, str]:
    """The live process posture in canonical string form. On a torchless
    worker the only honest fact is the absence itself."""
    torch = torch_capability.torch_or_none()
    if torch is None:
        return {"torch": torch_capability.ABSENT}

    return {
        "grad_enabled": str(torch.is_grad_enabled()),
        "inference_mode": str(torch.is_inference_mode_enabled()),
        "autocast_cpu": str(torch.is_autocast_enabled("cpu")),
        "autocast_cuda": str(torch.is_autocast_enabled("cuda")),
        "torch_function_stack": str(torch._C._len_torch_function_stack()),
        "default_device": str(torch.get_default_device()),
        "deterministic_algorithms": str(
            torch.are_deterministic_algorithms_enabled()),
        "deterministic_warn_only": str(
            torch.is_deterministic_algorithms_warn_only_enabled()),
    }


def _posture_diff(sealed: Mapping[str, Any], live: Mapping[str, str]) -> List[str]:
    out: List[str] = []
    for fact in sorted(set(sealed) | set(live)):
        want = str(sealed.get(fact, "<absent>"))
        have = str(live.get(fact, "<absent>"))
        if want != have:
            out.append(f"{fact}: sealed {want!r} != process {have!r}")
    return out


def establish_posture() -> Dict[str, str]:
    """Set the canonical posture explicitly (boot entry). Settable facts are
    SET (grad mode, deterministic algos); ambient contexts that library code
    must not pop from under the embedding process (an active autocast, a
    foreign torch-function mode, a moved default device) REFUSE with a named
    :class:`PostureError` instead.

    a torchless worker has no grad mode, no autocast stack and no
    default device to set or refuse. It seals the absence and boots."""
    torch = torch_capability.torch_or_none()
    if torch is None:
        return {"torch": torch_capability.ABSENT}

    torch.set_grad_enabled(True)
    torch.use_deterministic_algorithms(False)
    diffs = _posture_diff(CANONICAL_POSTURE, posture_snapshot())
    if diffs:
        raise PostureError(
            "process posture is not canonical at establish: "
            + "; ".join(diffs))
    return dict(CANONICAL_POSTURE)


# ---------------------------------------------------------------------------
# The advisory closure audit (pgw#756: no veto) + armed-cell audit
# ---------------------------------------------------------------------------


def audit_armed(pipeline: Any, cfg: Any) -> ClosureReport:
    """The auditable diagnostics view of an armed pipeline's guard closure."""
    return extract(pipeline, cfg)


# ---------------------------------------------------------------------------
# Boundary canonicalization (the single compiled-graph ingress)
# ---------------------------------------------------------------------------


def canonical_strides(shape: Sequence) -> Tuple[int, ...]:
    """The canonical contiguous stride tuple torch mints for fresh tensors."""
    return _contiguous_strides(shape)


def _canonical_tensor(t: Any, path: str, label: str, dtypes: Dict[str, str]) -> Any:
    want = _contiguous_strides(t.shape)
    if tuple(t.stride()) != want:
        t = t.contiguous()
        if tuple(t.stride()) != want:
            # a size-1 dim keeps is_contiguous() true under an
            # arbitrary stride, making .contiguous() a no-op while
            # TENSOR_MATCH guards the exact stride tuple. The layout is
            # already contiguous, so restating the strides is safe.
            t = t.as_strided(tuple(int(v) for v in t.shape), want)
    seen = dtypes.setdefault(path, str(t.dtype))
    if seen != str(t.dtype):
        raise GuardBoundaryError(
            f"compiled ingress {label}: {path} arrived as {t.dtype} but "
            f"this boundary first observed {seen} — undeclared dtype "
            "drift at the compiled graph boundary")
    return t


def _canonical_value(
    value: Any, path: str, label: str, dtypes: Dict[str, str],
    torch_mod: Any, depth: int = 0,
) -> Any:
    if isinstance(value, torch_mod.Tensor):
        return _canonical_tensor(value, path, label, dtypes)
    if depth >= _CANONICAL_DEPTH:
        return value
    if isinstance(value, tuple):
        return tuple(
            _canonical_value(v, f"{path}[{i}]", label, dtypes, torch_mod, depth + 1)
            for i, v in enumerate(value))
    if isinstance(value, list):
        return [
            _canonical_value(v, f"{path}[{i}]", label, dtypes, torch_mod, depth + 1)
            for i, v in enumerate(value)]
    if isinstance(value, dict):
        return {
            k: _canonical_value(v, f"{path}[{k!r}]", label, dtypes, torch_mod, depth + 1)
            for k, v in value.items()}
    return value


def canonical_ingress(fn: Callable[..., Any], label: str) -> Callable[..., Any]:
    """Wrap the compiled callable so every entry crosses one canonical
    boundary: contiguous-canonical strides (stride-perturbed inputs HIT the
    minted graph instead of recompiling) and per-path dtype asserts.
    Installed symmetrically for mint capture and consumer serving, so the
    traced guards describe exactly the form serving presents."""
    dtypes: Dict[str, str] = {}

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        import torch

        canon_args = tuple(
            _canonical_value(a, f"args[{i}]", label, dtypes, torch)
            for i, a in enumerate(args))
        canon_kwargs = {
            k: _canonical_value(v, f"kwargs[{k!r}]", label, dtypes, torch)
            for k, v in kwargs.items()}
        return fn(*canon_args, **canon_kwargs)

    setattr(wrapper, "_cozy_canonical_ingress", label)
    return wrapper


__all__ = [
    "CANONICALIZED",
    "CANONICAL_POSTURE",
    "CODE_CONSTANT",
    "CODE_IDENTITY",
    "CONTRACT_SCALAR",
    "CONTRACT_SHAPE",
    "ClosureReport",
    "GATE_ADVISORY",
    "GATE_KEY",
    "ContractPins",
    "GraphGuards",
    "GuardBoundaryError",
    "GuardClosureError",
    "GuardRecord",
    "LEAK",
    "MANIFEST_VERSION",
    "MODULE_STRUCTURE",
    "PostureError",
    "RUNTIME_STATE",
    "STRUCTURAL",
    "UNPROVEN",
    "audit_armed",
    "canonical_ingress",
    "canonical_strides",
    "classify",
    "contract_pins",
    "establish_posture",
    "extract",
    "extract_target_guards",
    "posture_snapshot",
]
