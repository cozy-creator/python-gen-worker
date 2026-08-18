"""RuntimeFormula — declared compute-time formula terms.

``runtime=RuntimeFormula("a + b*num_inference_steps + c*num_inference_steps*megapixels")``
on ``@endpoint``: the author declares the SHAPE of compute time as a sum of
terms, each a learned constant times an expression over payload field names.
The platform learns the constants per physics compiled graph; the worker evaluates the
term expressions on the executed payload and reports them on the observation
back-channel (``JobMetrics.runtime_terms``).

The grammar and term-key canonicalization mirror tensorhub's
``internal/formula`` package exactly — the formula STRING is the wire
contract and both sides must mint identical term keys.

Grammar (ASCII only; ``WS`` is space, tab, or newline)::

    runtime := rterm ('+' rterm)*
    rterm   := IDENT ('*' term)?
    expr    := term (('+'|'-') term)*
    term    := unary (('*'|'/') unary)*
    unary   := '-'? primary
    primary := NUMBER | IDENT | '(' expr ')'
    IDENT   := [A-Za-z_] [A-Za-z0-9_]*, excluding ``_RESERVED_IDENTIFIERS``
    NUMBER  := ('0' | [1-9][0-9]*) ('.' [0-9]*)? ([eE] [+-]? [0-9]+)?
             | '.' [0-9]+ ([eE] [+-]? [0-9]+)?

There are no comments, calls, comparisons, Unicode identifiers, or numeric
separators. Top-level ``+`` joins learned-coefficient terms.

A COUNTABLE CONTAINER is a payload variable, mirroring the hub's
``internal/price`` rule: an identifier naming a LIST or a MAP field binds to
that container's ITEM COUNT. There is deliberately no ``len()``
— the grammar is the wire contract, calls are excluded from it on both sides,
and a term key the hub cannot canonicalize identically is not a term. One
vocabulary: ``"a + b*num_inference_steps + c*references"`` prices and predicts
the same reference-heavy request with the same string.
"""

from __future__ import annotations

import ast
import math
import re
import types
import typing
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Set, Tuple

import msgspec

__all__ = [
    "COMPUTE_TIME_LIMITS",
    "FormulaLimits",
    "FormulaParseError",
    "FormulaRefusal",
    "RuntimeFormula",
]

_NUMERIC_TYPES = (int, float, bool)
#: The container kinds that carry an item count. Exactly the hub's
#: ``price.numericValue`` set (a JSON array or object) — `str`/`bytes` are
#: Sized and are NOT countable containers: a prompt's length is not a declared
#: cost dimension on either side, and admitting it here would price one.
_COUNTABLE_TYPES = (list, tuple, set, frozenset, dict)
_FORMULA_SPACE = " \t\n"
_RESERVED_IDENTIFIERS = frozenset({
    "False", "None", "True", "and", "as", "assert", "async", "await",
    "break", "class", "continue", "def", "del", "elif", "else", "except",
    "finally", "for", "from", "global", "if", "import", "in", "is",
    "lambda", "nonlocal", "not", "or", "pass", "raise", "return", "try",
    "while", "with", "yield",
})
_TOKEN = re.compile(
    r"(?P<space>[ \t\n]+)|"
    r"(?P<number>(?:(?:0|[1-9][0-9]*)(?:\.[0-9]*)?|\.[0-9]+)"
    r"(?:[eE][+-]?[0-9]+)?)|"
    r"(?P<ident>[A-Za-z_][A-Za-z0-9_]*)|"
    r"(?P<operator>[+\-*/()])"
)


@dataclass(frozen=True)
class FormulaLimits:
    """The parser/evaluator budget shared with Tensorhub."""

    max_depth: int
    max_len: int
    max_nodes: int
    max_abs_result: float
    fail_closed: bool = False


# Safety ceilings, mirrored byte-for-byte in behavior rather than learned
# scheduling constants: ordinary formulas are only a few terms, while these
# make worst-case parser work fixed and exclude results such as 1e300.
COMPUTE_TIME_LIMITS = FormulaLimits(
    max_depth=32,
    max_len=4096,
    max_nodes=256,
    max_abs_result=1e15,
)


class FormulaParseError(ValueError):
    """The source is malformed or exceeds its parser budget."""


class FormulaRefusal(ValueError):
    """A fail-closed evaluation could not produce an allowed value."""


class _EvaluationError(Exception):
    pass


class _Term:
    __slots__ = ("constant", "key", "factor")

    def __init__(self, constant: str, key: str, factor: Optional[ast.expr]) -> None:
        self.constant = constant
        self.key = key          # canonical feature key ("1" for the intercept)
        self.factor = factor    # None for the bare-constant intercept


class RuntimeFormula:
    """Parsed, validated compute-time formula declaration."""

    def __init__(
        self, source: str, *, limits: FormulaLimits = COMPUTE_TIME_LIMITS,
    ) -> None:
        if not isinstance(source, str):
            raise FormulaParseError(
                "RuntimeFormula: source must be a non-empty string"
            )
        _validate_source(source, limits)
        self.source = source.strip(_FORMULA_SPACE)
        if not self.source:
            raise FormulaParseError(
                "RuntimeFormula: source must be a non-empty string"
            )
        self._limits = limits
        self.terms: List[_Term] = _parse_terms(self.source, limits)
        self.fields: Tuple[str, ...] = tuple(sorted({
            name for t in self.terms if t.factor is not None
            for name in _names(t.factor)
        }))

    # -- declaration-time validation ------------------------------------

    def validate_for_payload(
        self, payload_type: type, owner: str, *,
        defaults_type: Optional[type] = None,
    ) -> None:
        """Every referenced identifier must be a payload field whose
        EFFECTIVE value is numeric: either the field carries a numeric/bool
        wire default, or the handler's derived defaults schema
        (``ctx: RequestContext[D]``) declares a same-named field the catalog
        recipe resolves — the ``Optional[...] = None -> ctx.defaults`` pattern.
        No constant may collide with a field name."""
        try:
            field_map = {f.name: f for f in msgspec.structs.fields(payload_type)}
        except Exception as exc:  # not a Struct — walker validates elsewhere
            raise ValueError(
                f"{owner}: runtime= formula needs a msgspec.Struct payload ({exc})"
            ) from exc
        defaults_fields: Set[str] = set()
        if defaults_type is not None:
            try:
                defaults_fields = {
                    f.name for f in msgspec.structs.fields(defaults_type)
                }
            except Exception:
                defaults_fields = set()
        for t in self.terms:
            if t.constant in field_map:
                raise ValueError(
                    f"{owner}: runtime formula constant {t.constant!r} collides "
                    f"with a payload field name"
                )
        for name in self.fields:
            f = field_map.get(name)
            if f is None:
                raise ValueError(
                    f"{owner}: runtime formula field {name!r} is not a payload field"
                )
            if _is_countable_container(getattr(f, "type", None)):
                # A list/map field IS a numeric term — its item count. It needs
                # no numeric default: an omitted or null container is zero
                # items, which is a reading of the field rather than a guess.
                continue
            default = f.default
            if default is msgspec.NODEFAULT and f.default_factory is not msgspec.NODEFAULT:
                default = f.default_factory()
            if isinstance(default, _NUMERIC_TYPES):
                continue
            if name in defaults_fields:
                # Resolved-effective evaluation: the catalog recipe
                # (ctx.defaults.<name>) is the reference value when the wire
                # field is omitted/None.
                continue
            raise ValueError(
                f"{owner}: runtime formula field {name!r} needs a numeric/bool "
                "wire default, or a same-named field on the handler's derived "
                "defaults schema (the resolved recipe is the reference value "
                "— pgw#654 gap #4)"
            )

    # -- worker-side evaluation ------------------------------------------

    def term_values(self, values: Mapping[str, Any]) -> Optional[Dict[str, float]]:
        """Evaluate each term's factor -> {term key: value}. None when any
        referenced field is missing/non-finite (the hub then falls back to
        its own evaluation)."""
        out: Dict[str, float] = {}
        for t in self.terms:
            if t.factor is None:
                if 1.0 > self._limits.max_abs_result:
                    self._evaluation_failure(
                        "runtime formula: intercept exceeds the absolute result limit"
                    )
                    return None
                out[t.key] = 1.0
                continue
            try:
                v = _eval(t.factor, values)
                if abs(v) > self._limits.max_abs_result:
                    raise _EvaluationError(
                        f"runtime formula: term {t.key!r} result {v:g} exceeds "
                        f"absolute limit {self._limits.max_abs_result:g}"
                    )
            except _EvaluationError as exc:
                self._evaluation_failure(str(exc), exc)
                return None
            out[t.key] = v
        return out

    def _evaluation_failure(
        self, message: str, cause: Optional[BaseException] = None,
    ) -> None:
        if self._limits.fail_closed:
            raise FormulaRefusal(message) from cause

    def term_values_from_struct(
        self, payload: Any, defaults: Any = None,
    ) -> Optional[Dict[str, float]]:
        """Evaluate on RESOLVED EFFECTIVE values: an explicit payload value
        wins; a ``None``/missing wire field falls
        back to the same-named field of ``defaults`` (the catalog-resolved
        recipe object, ``ctx.defaults``)."""
        containers: Optional[Set[str]] = None   # resolved only if a None shows up
        values: Dict[str, float] = {}
        for name in self.fields:
            raw = getattr(payload, name, None)
            if raw is None and defaults is not None:
                raw = getattr(defaults, name, None)
            if isinstance(raw, bool):
                values[name] = 1.0 if raw else 0.0
            elif isinstance(raw, _COUNTABLE_TYPES):
                # The countable-container reading, identical to the hub's
                # `price.PayloadVars`.
                values[name] = float(len(raw))
            elif isinstance(raw, (int, float)):
                values[name] = raw
            elif raw is None:
                # A DECLARED container that arrived null/absent is zero items.
                # Only the declaration makes this safe: a missing NUMBER stays
                # missing, so the formula still declines rather than reading a
                # zero into a term the request never stated.
                if containers is None:
                    containers = _countable_fields(type(payload))
                if name in containers:
                    values[name] = 0.0
        return self.term_values(values)

    def __repr__(self) -> str:  # pragma: no cover
        return f"RuntimeFormula({self.source!r})"


# ---------------------------------------------------------------------------
# Parsing (mirrors tensorhub internal/formula)
# ---------------------------------------------------------------------------


def _validate_source(source: str, limits: FormulaLimits) -> None:
    if (
        limits.max_depth <= 0
        or limits.max_len <= 0
        or limits.max_nodes <= 0
        or limits.max_abs_result <= 0
        or not math.isfinite(limits.max_abs_result)
    ):
        raise FormulaParseError("runtime formula: limits must be finite and positive")
    if (
        limits.max_depth > COMPUTE_TIME_LIMITS.max_depth
        or limits.max_len > COMPUTE_TIME_LIMITS.max_len
        or limits.max_nodes > COMPUTE_TIME_LIMITS.max_nodes
        or limits.max_abs_result > COMPUTE_TIME_LIMITS.max_abs_result
    ):
        raise FormulaParseError("runtime formula: limits exceed the platform ceiling")
    source_len = len(source.encode("utf-8"))
    if source_len > limits.max_len:
        raise FormulaParseError(
            f"runtime formula: source is {source_len} bytes; limit is {limits.max_len}"
        )
    depth = 0
    for offset, char in enumerate(source):
        if char == "(":
            depth += 1
            if depth > limits.max_depth:
                raise FormulaParseError(
                    "runtime formula: nesting depth exceeds "
                    f"{limits.max_depth} at offset {offset}"
                )
        elif char == ")":
            depth -= 1
            if depth < 0:
                raise FormulaParseError(
                    f"runtime formula: unexpected ')' at offset {offset}"
                )
    if depth:
        raise FormulaParseError("runtime formula: unbalanced parentheses")


def _validate_lexical_source(source: str) -> None:
    pos = 0
    previous = ""
    while pos < len(source):
        token = _TOKEN.match(source, pos)
        if token is None:
            raise FormulaParseError(
                f"runtime formula: unexpected {source[pos]!r} at offset {pos}"
            )
        kind = token.lastgroup
        text = token.group()
        pos = token.end()
        if kind == "space":
            continue
        if kind in ("number", "ident"):
            if previous == "value":
                raise FormulaParseError(
                    f"runtime formula: adjacent values at offset {token.start()}"
                )
            if kind == "ident" and text in _RESERVED_IDENTIFIERS:
                raise FormulaParseError(
                    f"runtime formula: reserved identifier {text!r}"
                )
            previous = "value"
            continue
        if text == "-" and previous in ("", "(", "+", "-", "*", "/"):
            following = pos
            while following < len(source) and source[following] in _FORMULA_SPACE:
                following += 1
            if following < len(source) and source[following] == "-":
                raise FormulaParseError(
                    f"runtime formula: repeated unary '-' at offset {token.start()}"
                )
        previous = "value" if text == ")" else text


def _parse_terms(source: str, limits: FormulaLimits) -> List[_Term]:
    _validate_lexical_source(source)
    try:
        tree = ast.parse(
            "".join(" " if char in _FORMULA_SPACE else char for char in source),
            mode="eval",
        )
        _check_nodes(tree.body)
        term_exprs = _split_sum(tree.body)
    except (SyntaxError, RecursionError) as exc:
        raise FormulaParseError(f"runtime formula: {exc}") from exc
    except ValueError as exc:
        raise FormulaParseError(str(exc)) from exc
    terms: List[_Term] = []
    seen: Set[str] = set()
    nodes = 0
    for expr in term_exprs:
        try:
            constant, factor = _split_leading_constant(expr)
        except ValueError as exc:
            raise FormulaParseError(str(exc)) from exc
        nodes += 1  # the learned coefficient is one logical node
        if factor is not None:
            nodes += _logical_nodes(factor)
        if nodes > limits.max_nodes:
            raise FormulaParseError(
                f"runtime formula: node count exceeds {limits.max_nodes}"
            )
        if constant in seen:
            raise FormulaParseError(
                f"runtime formula: constant {constant!r} used in more than one term"
            )
        seen.add(constant)
        key = "1" if factor is None else _canonical(factor)
        if factor is not None and not _names(factor):
            raise FormulaParseError(
                f"runtime formula term with constant {constant!r}: factor "
                f"references no payload field"
            )
        terms.append(_Term(constant, key, factor))
    return terms


def _is_countable_container(annotation: Any) -> bool:
    """Is this declared field type a countable container?

    ``list[X]``, ``dict[K, V]``, ``tuple[...]``, ``set[X]`` and their
    ``Optional``/union spellings. A union counts only when EVERY non-``None``
    member is countable — ``Union[list[X], int]`` is two different readings of
    one identifier and is refused as a plain non-numeric field would be.
    """
    if annotation is None:
        return False
    origin = typing.get_origin(annotation)
    if origin in (typing.Union, types.UnionType):
        members = [a for a in typing.get_args(annotation) if a is not type(None)]
        return bool(members) and all(_is_countable_container(a) for a in members)
    target = origin if origin is not None else annotation
    return isinstance(target, type) and issubclass(target, _COUNTABLE_TYPES)


def _countable_fields(payload_type: Any) -> Set[str]:
    """Names of the struct's countable-container fields; empty for anything
    that is not a msgspec Struct (the value-shape reading still applies)."""
    try:
        return {
            f.name for f in msgspec.structs.fields(payload_type)
            if _is_countable_container(getattr(f, "type", None))
        }
    except Exception:
        return set()


def _check_nodes(node: ast.expr) -> None:
    allowed = (
        ast.BinOp, ast.UnaryOp, ast.Name, ast.Constant,
        ast.Add, ast.Sub, ast.Mult, ast.Div, ast.USub, ast.Load,
    )
    for n in ast.walk(node):
        if not isinstance(n, allowed):
            raise ValueError(
                f"runtime formula: {type(n).__name__} is not allowed "
                f"(arithmetic over payload fields only)"
            )
        if isinstance(n, ast.BinOp) and not isinstance(n.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            raise ValueError("runtime formula: only + - * / are allowed")
        if isinstance(n, ast.UnaryOp) and not isinstance(n.op, ast.USub):
            raise ValueError("runtime formula: only unary '-' is allowed")
        if isinstance(n, ast.Constant):
            value = n.value
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("runtime formula: literals must be numeric")
            try:
                literal = float(value)
            except (OverflowError, ValueError) as exc:
                raise ValueError("runtime formula: literal is not finite") from exc
            if not math.isfinite(literal):
                raise ValueError("runtime formula: literal is not finite")


def _split_sum(node: ast.expr) -> List[ast.expr]:
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _split_sum(node.left) + _split_sum(node.right)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Sub):
        raise ValueError(
            "runtime formula: top-level '-' not allowed (constants are "
            "learned signed; write '+' terms only)"
        )
    return [node]


def _split_leading_constant(term: ast.expr) -> Tuple[str, Optional[ast.expr]]:
    """A term is ``constIdent`` or ``constIdent * factor [* factor...]``.
    The leftmost factor of the product chain must be a bare Name."""
    if isinstance(term, ast.Name):
        return term.id, None
    if isinstance(term, ast.BinOp) and isinstance(term.op, (ast.Mult, ast.Div)):
        # Walk to the leftmost leaf of the product chain.
        chain: List[Tuple[ast.BinOp, str]] = []
        node: ast.expr = term
        while isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Mult, ast.Div)):
            chain.append((node, "l"))
            node = node.left
        if not isinstance(node, ast.Name):
            raise ValueError(
                "runtime formula: each term must start with a constant "
                "identifier (e.g. \"b*steps\")"
            )
        # The constant is multiplied (never divided) into the term: the
        # innermost BinOp holding the Name on its left must be Mult.
        innermost = chain[-1][0]
        if not isinstance(innermost.op, ast.Mult):
            raise ValueError(
                f"runtime formula: constant {node.id!r} must be followed by '*'"
            )
        factor = _rebuild_without_leading(term)
        return node.id, factor
    raise ValueError(
        "runtime formula: each term must start with a constant identifier"
    )


def _rebuild_without_leading(term: ast.BinOp) -> ast.expr:
    """Drop the leftmost leaf (the constant) from a product chain:
    ``c*a*b/d`` -> ``a*b/d``."""
    if isinstance(term.left, ast.Name):
        return term.right
    assert isinstance(term.left, ast.BinOp)
    new_left = _rebuild_without_leading(term.left)
    return ast.BinOp(left=new_left, op=term.op, right=term.right)


def _names(node: ast.expr) -> Set[str]:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def _logical_nodes(node: ast.expr) -> int:
    return sum(
        isinstance(n, (ast.BinOp, ast.UnaryOp, ast.Name, ast.Constant))
        for n in ast.walk(node)
    )


def _prec(op: ast.operator) -> int:
    return 2 if isinstance(op, (ast.Mult, ast.Div)) else 1


def _canonical(node: ast.expr) -> str:
    """Whitespace-free canonical serialization — byte-identical to
    tensorhub internal/formula's Expr.Canonical for the same source."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return _canonical_number(float(node.value))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = _canonical(node.operand)
        if isinstance(node.operand, ast.BinOp) and _prec(node.operand.op) < 2:
            return f"-({inner})"
        return f"-{inner}"
    if isinstance(node, ast.BinOp):
        op_char = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/"}[type(node.op)]
        left = _canonical(node.left)
        right = _canonical(node.right)
        if isinstance(node.left, ast.BinOp) and _prec(node.left.op) < _prec(node.op):
            left = f"({left})"
        if isinstance(node.right, ast.BinOp) and (
            _prec(node.right.op) < _prec(node.op)
            or (_prec(node.right.op) == _prec(node.op) and op_char in "-/")
        ):
            right = f"({right})"
        return f"{left}{op_char}{right}"
    raise ValueError(f"runtime formula: cannot canonicalize {type(node).__name__}")


def _canonical_number(value: float) -> str:
    if value == math.trunc(value) and abs(value) < 1e15:
        return str(int(value))
    mantissa, exponent = format(value, ".16e").split("e")
    mantissa = mantissa.rstrip("0").rstrip(".")
    exp = int(exponent)
    return mantissa if exp == 0 else f"{mantissa}e{exp}"


def _eval(node: ast.expr, values: Mapping[str, Any]) -> float:
    if isinstance(node, ast.Name):
        v = values.get(node.id)
        if v is None or not isinstance(v, (int, float)) or isinstance(v, bool):
            if isinstance(v, bool):
                return 1.0 if v else 0.0
            raise _EvaluationError(
                f"runtime formula: missing numeric value for field {node.id!r}"
            )
        try:
            f = float(v)
        except (OverflowError, ValueError) as exc:
            raise _EvaluationError(
                f"runtime formula: field {node.id!r} is not finite"
            ) from exc
        if not math.isfinite(f):
            raise _EvaluationError(
                f"runtime formula: field {node.id!r} is not finite"
            )
        return f
    if isinstance(node, ast.Constant):
        if not isinstance(node.value, (int, float)):
            raise _EvaluationError("runtime formula: literal is not numeric")
        value = float(node.value)
        if not math.isfinite(value):
            raise _EvaluationError("runtime formula: literal is not finite")
        return value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        v = _eval(node.operand, values)
        out = -v
        if not math.isfinite(out):
            raise _EvaluationError(
                "runtime formula: non-finite result after negation"
            )
        return out
    if isinstance(node, ast.BinOp):
        left = _eval(node.left, values)
        right = _eval(node.right, values)
        if isinstance(node.op, ast.Add):
            out = left + right
        elif isinstance(node.op, ast.Sub):
            out = left - right
        elif isinstance(node.op, ast.Mult):
            out = left * right
        elif isinstance(node.op, ast.Div):
            if right == 0:
                raise _EvaluationError("runtime formula: division by zero")
            out = left / right
        else:  # pragma: no cover
            raise _EvaluationError("runtime formula: operator is not supported")
        if not math.isfinite(out):
            raise _EvaluationError("runtime formula: non-finite result")
        return out
    raise _EvaluationError("runtime formula: expression node is not supported")
