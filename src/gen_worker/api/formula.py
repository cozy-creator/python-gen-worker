"""RuntimeFormula — declared compute-time formula terms (th#1051).

``runtime=RuntimeFormula("a + b*num_inference_steps + c*num_inference_steps*megapixels")``
on ``@endpoint``: the author declares the SHAPE of compute time as a sum of
terms, each a learned constant times an expression over payload field names.
The platform learns the constants per physics cell; the worker evaluates the
term expressions on the executed payload and reports them on the observation
back-channel (``JobMetrics.runtime_terms``).

The grammar and term-key canonicalization mirror tensorhub's
``internal/formula`` package exactly — the formula STRING is the wire
contract and both sides must mint identical term keys.

Grammar: ``+ - * /``, parentheses, numeric literals, identifiers. Top level
must be a '+'-joined sum; each term starts with a bare constant identifier
(the learned coefficient), optionally ``* <payload expression>``.
"""

from __future__ import annotations

import ast
import math
from typing import Any, Dict, List, Mapping, Optional, Set, Tuple

import msgspec

__all__ = ["RuntimeFormula"]

_NUMERIC_TYPES = (int, float, bool)


class _Term:
    __slots__ = ("constant", "key", "factor")

    def __init__(self, constant: str, key: str, factor: Optional[ast.expr]) -> None:
        self.constant = constant
        self.key = key          # canonical feature key ("1" for the intercept)
        self.factor = factor    # None for the bare-constant intercept


class RuntimeFormula:
    """Parsed, validated compute-time formula declaration."""

    def __init__(self, source: str) -> None:
        if not isinstance(source, str) or not source.strip():
            raise ValueError("RuntimeFormula: source must be a non-empty string")
        self.source = source.strip()
        self.terms: List[_Term] = _parse_terms(self.source)
        self.fields: Tuple[str, ...] = tuple(sorted({
            name for t in self.terms if t.factor is not None
            for name in _names(t.factor)
        }))

    # -- declaration-time validation ------------------------------------

    def validate_for_payload(self, payload_type: type, owner: str) -> None:
        """Every payload identifier must be a numeric/bool payload field WITH
        a declared default (the defaults are the reference payload); no
        constant may collide with a field name."""
        try:
            field_map = {f.name: f for f in msgspec.structs.fields(payload_type)}
        except Exception as exc:  # not a Struct — walker validates elsewhere
            raise ValueError(
                f"{owner}: runtime= formula needs a msgspec.Struct payload ({exc})"
            ) from exc
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
            default = f.default
            if default is msgspec.NODEFAULT and f.default_factory is not msgspec.NODEFAULT:
                default = f.default_factory()
            if default is msgspec.NODEFAULT or not isinstance(default, _NUMERIC_TYPES):
                raise ValueError(
                    f"{owner}: runtime formula field {name!r} needs a numeric/bool "
                    f"default (the defaults are the reference payload)"
                )

    # -- worker-side evaluation ------------------------------------------

    def term_values(self, values: Mapping[str, Any]) -> Optional[Dict[str, float]]:
        """Evaluate each term's factor -> {term key: value}. None when any
        referenced field is missing/non-finite (the hub then falls back to
        its own evaluation)."""
        out: Dict[str, float] = {}
        for t in self.terms:
            if t.factor is None:
                out[t.key] = 1.0
                continue
            v = _eval(t.factor, values)
            if v is None:
                return None
            out[t.key] = v
        return out

    def term_values_from_struct(self, payload: Any) -> Optional[Dict[str, float]]:
        values: Dict[str, float] = {}
        for name in self.fields:
            raw = getattr(payload, name, None)
            if isinstance(raw, bool):
                values[name] = 1.0 if raw else 0.0
            elif isinstance(raw, (int, float)):
                values[name] = float(raw)
        return self.term_values(values)

    def __repr__(self) -> str:  # pragma: no cover
        return f"RuntimeFormula({self.source!r})"


# ---------------------------------------------------------------------------
# Parsing (mirrors tensorhub internal/formula)
# ---------------------------------------------------------------------------


def _parse_terms(source: str) -> List[_Term]:
    try:
        tree = ast.parse(source, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"runtime formula: {exc}") from exc
    _check_nodes(tree.body)
    term_exprs = _split_sum(tree.body)
    terms: List[_Term] = []
    seen: Set[str] = set()
    for expr in term_exprs:
        constant, factor = _split_leading_constant(expr)
        if constant in seen:
            raise ValueError(
                f"runtime formula: constant {constant!r} used in more than one term"
            )
        seen.add(constant)
        key = "1" if factor is None else _canonical(factor)
        if factor is not None and not _names(factor):
            raise ValueError(
                f"runtime formula term with constant {constant!r}: factor "
                f"references no payload field"
            )
        terms.append(_Term(constant, key, factor))
    return terms


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
        if isinstance(n, ast.Constant) and (
            isinstance(n.value, bool) or not isinstance(n.value, (int, float))
        ):
            raise ValueError("runtime formula: literals must be numeric")


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


def _prec(op: ast.operator) -> int:
    return 2 if isinstance(op, (ast.Mult, ast.Div)) else 1


def _canonical(node: ast.expr) -> str:
    """Whitespace-free canonical serialization — byte-identical to
    tensorhub internal/formula's Expr.Canonical for the same source."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        v = float(node.value)
        if v == math.trunc(v) and abs(v) < 1e15:
            return str(int(v))
        return repr(v)
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


def _eval(node: ast.expr, values: Mapping[str, Any]) -> Optional[float]:
    if isinstance(node, ast.Name):
        v = values.get(node.id)
        if v is None or not isinstance(v, (int, float)) or isinstance(v, bool):
            if isinstance(v, bool):
                return 1.0 if v else 0.0
            return None
        f = float(v)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(node, ast.Constant):
        if not isinstance(node.value, (int, float)):
            return None
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        v = _eval(node.operand, values)
        return None if v is None else -v
    if isinstance(node, ast.BinOp):
        left = _eval(node.left, values)
        right = _eval(node.right, values)
        if left is None or right is None:
            return None
        if isinstance(node.op, ast.Add):
            out = left + right
        elif isinstance(node.op, ast.Sub):
            out = left - right
        elif isinstance(node.op, ast.Mult):
            out = left * right
        elif isinstance(node.op, ast.Div):
            if right == 0:
                return None
            out = left / right
        else:  # pragma: no cover
            return None
        return None if (math.isnan(out) or math.isinf(out)) else out
    return None  # pragma: no cover
