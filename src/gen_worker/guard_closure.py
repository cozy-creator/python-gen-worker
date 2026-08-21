"""Guard-closure ADVISORY audit + boundary canonicalization."""

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

MANIFEST_VERSION = 1
GATE_KEY = "gate"
GATE_ADVISORY = "advisory"

LEAK = "LEAK"
UNPROVEN = "unproven"
RUNTIME_STATE = "runtime-state"
CODE_IDENTITY = "code-identity"
MODULE_STRUCTURE = "module-structure"
CONTRACT_SHAPE = "contract-shape"
CONTRACT_SCALAR = "contract-scalar"
STRUCTURAL = "structural"
CODE_CONSTANT = "code-constant"
CANONICALIZED = "canonicalized-stride"

_COVERED = (
    RUNTIME_STATE, CODE_IDENTITY, MODULE_STRUCTURE, CONTRACT_SHAPE,
    CONTRACT_SCALAR, STRUCTURAL, CODE_CONSTANT, CANONICALIZED,
)

_RELATIONAL_ALIASING = frozenset({
    "NO_TENSOR_ALIASING", "OBJECT_ALIASING", "STORAGE_OVERLAPPING",
    "DUPLICATE_INPUT",
})
_AMBIENT_COVERED = frozenset({
    "GLOBAL_STATE", "TORCH_FUNCTION_MODE_STACK", "DEFAULT_DEVICE",
    "DUAL_LEVEL_MATCH", "DETERMINISTIC_ALGORITHMS", "GRAD_MODE",
    "TORCH_FUNCTION_STATE", "FUNCTORCH_STACK_MATCH", "DUAL_LEVEL",
})
_STRUCTURAL_TYPES = frozenset({
    "TYPE_MATCH", "DIMENSION_DYNAMIC_MARKING_GUARD", "HASATTR", "NO_HASATTR",
    "LENGTH_CHECK", "DICT_LENGTH", "DICT_CONTAINS", "SET_CONTAINS",
    "MAPPING_KEYS_MATCH", "TUPLE_ITERATOR_LEN", "FAKE_SCRIPT_TYPE_MATCH",
    "RANGE_ITERATOR_MATCH", "NONE_MATCH", "NOT_NONE", "NOT_NONE_MATCH",
    "TRUE_MATCH", "FALSE_MATCH", "DISPATCH_KEY_SET_MATCH", "FLOAT_IS_NAN",
    "COMPLEX_IS_NAN", "SEQUENCE_LENGTH",
})
_IDENTITY_CONSTANT_TYPES = frozenset({"ID_MATCH", "DICT_VERSION"})

_CANONICAL_DEPTH = 4

_ID_SCRUB_RE = re.compile(r"(___check_(?:obj|type)_id\(.*?,\s*)\d+(\))")
_COMMENT_RE = re.compile(r"\s{2,}#.*$", re.DOTALL)
_TENSOR_MATCH_RE = re.compile(r"size=\[([^\]]*)\], stride=\[([^\]]*)\]")
_SOURCE_LOCAL_RE = re.compile(r"^L\['([^']+)'\]")
_SOURCE_EMBEDDED_RE = re.compile(r"L\['([^']+)'\]")
_SOURCE_EMBEDDED_GLOBAL_RE = re.compile(r"\bG[\['.]")


class GuardClosureError(RuntimeError):
    """The mint produced no readable compiled graphs, or a stored manifest is unreadable."""


class GuardBoundaryError(RuntimeError):
    """An input crossed the compiled-graph ingress outside the canonical boundary (dtype drift)."""


class PostureError(GuardClosureError):
    """The process posture differs from the canonical serving posture or from a compiled graph's sealed posture."""


@dataclass(frozen=True)
class GuardRecord:
    """One classified dynamo guard."""

    guard_type: str
    source: str
    expr: str
    verdict: str
    axis: str

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
    """The audit view for one armed pipeline / minted compiled graph."""

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
        """Entries whose guards could not be read."""
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
        """Deterministic JSON manifest — the durable per-compiled graph guard dump."""
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


@dataclass(frozen=True)
class ContractPins:
    """The scalar/dynamic vocabulary one declared contract admits into guards."""

    ints: frozenset
    floats: frozenset
    has_dynamic: bool
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
    dynamic = tuple(getattr(cfg, "dynamic", ()) or ())
    for d in dynamic:
        ints.add(int(getattr(d, "min", 0)))
        ints.add(int(getattr(d, "max", 0)))
    floats = {0.0, 1.0}
    floats.update(float(v) for v in tuple(getattr(cfg, "guidance_scales", ()) or ()))
    return ContractPins(
        ints=frozenset(ints), floats=frozenset(floats),
        has_dynamic=bool(dynamic), freevars=frozenset(freevars))


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


def _source_root(source: str, freevars: frozenset = frozenset()) -> str:
    s = str(source or "").strip()
    if not s:
        return "ambient"
    if s.startswith("G[") or s.startswith("G."):
        return "global"
    m = _SOURCE_LOCAL_RE.match(s)
    if m is None:
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


_TORCH_ATTR_RE = re.compile(r"^torch\.([A-Za-z_][A-Za-z0-9_]*)$")
_TORCH_DEVICE_RE = re.compile(r"^(?:torch\.)?device\(type=")


def _torch_object_verdict(rhs: str) -> Optional[Tuple[str, str]]:
    if _TORCH_DEVICE_RE.match(rhs):
        return CONTRACT_SHAPE, "runtime device axes (sm/cuda/torch)"
    m = _TORCH_ATTR_RE.match(rhs)
    if m is None:
        return None
    try:
        import torch
    except ImportError:
        return None
    obj = getattr(torch, m.group(1), None)
    if isinstance(obj, (torch.dtype, torch.layout, torch.memory_format)):
        return CONTRACT_SHAPE, "weight_lane + ingress dtype memo"
    return None


def _classify_equals(expr: str, pins: ContractPins) -> Tuple[str, str]:
    _lhs, sep, rhs = expr.partition(" == ")
    if not sep:
        return LEAK, "unparseable EQUALS_MATCH expression"
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
    """(verdict, covering axis / leak reason) for one guard."""
    if guard_type in _RELATIONAL_ALIASING:
        return STRUCTURAL, "cross-input aliasing fixed by call topology"
    if guard_type == "SYMBOLIC_SHAPE_GUARD":
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
            return RUNTIME_STATE, "process runtime state (compiled graph-key runtime axes)"
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
    """The live process posture in canonical string form."""
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
    """Set the canonical posture explicitly (boot entry)."""
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


def canonical_strides(shape: Sequence) -> Tuple[int, ...]:
    """The canonical contiguous stride tuple torch mints for fresh tensors."""
    return _contiguous_strides(shape)


def _canonical_tensor(t: Any, path: str, label: str, dtypes: Dict[str, str]) -> Any:
    want = _contiguous_strides(t.shape)
    if tuple(t.stride()) != want:
        t = t.contiguous()
        if tuple(t.stride()) != want:
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
    """Wrap the compiled callable so every entry crosses one canonical boundary: contiguous-canonical strides (stride-perturbed inputs HIT the minted graph instead of recompiling) and per-path dtype asserts."""
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
    "canonical_ingress",
    "canonical_strides",
    "classify",
    "contract_pins",
    "establish_posture",
    "extract_target_guards",
    "posture_snapshot",
]
