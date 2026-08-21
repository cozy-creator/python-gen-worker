from __future__ import annotations

import ast
from typing import Any, Iterable

HARDWARE_READS: tuple[str, ...] = (
    "get_device_capability",
    "is_bf16_supported",
    "get_device_name",
    "mem_get_info",
    "get_device_properties",
)

NUMERICS_CALLS: tuple[str, ...] = (
    "quantize_",
    "apply_fp8_storage",
    "restructure_fp8_storage",
    "sanitize_w8a8_state_dict",
    "sanitize_w4a4_state_dict",
    "swap_w8a8_linears",
    "swap_w4a4_linears",
    "quantize_dit",
    "quantize_conditioner",
)

DTYPE_KEYWORDS: tuple[str, ...] = ("torch_dtype", "storage_dtype")


class Finding:
    """One branch a `load()` body should not contain, and where it is."""

    __slots__ = ("kind", "name", "lineno")

    def __init__(self, kind: str, name: str, lineno: int) -> None:
        self.kind = kind
        self.name = name
        self.lineno = lineno

    def __repr__(self) -> str:  # pragma: no cover — diagnostics
        return f"Finding({self.kind!r}, {self.name!r}, line {self.lineno})"

    def __eq__(self, other: object) -> bool:
        return (isinstance(other, Finding) and self.kind == other.kind
                and self.name == other.name and self.lineno == other.lineno)

    def __hash__(self) -> int:
        return hash((self.kind, self.name, self.lineno))

    def line(self) -> str:
        return f"{self.kind}:{self.name}@{self.lineno}"


KIND_HARDWARE = "hardware_read"
KIND_NUMERICS = "numerics_call"
KIND_DTYPE = "dtype_keyword"

KINDS = (KIND_HARDWARE, KIND_NUMERICS, KIND_DTYPE)


def load_branches_on_hardware(definition: Any) -> tuple[Finding, ...]:
    """Every hardware/numerics branch in one parsed ``load`` DEFINITION."""
    if not isinstance(definition, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return ()
    return _branches_in(definition)


def _branches_in(definition: ast.AST) -> tuple[Finding, ...]:
    found: list[Finding] = []
    for node in ast.walk(definition):
        if not isinstance(node, ast.Call):
            continue
        name = _called_name(node.func)
        if name in HARDWARE_READS:
            found.append(Finding(KIND_HARDWARE, name, node.lineno))
        elif name in NUMERICS_CALLS:
            found.append(Finding(KIND_NUMERICS, name, node.lineno))
        for keyword in node.keywords:
            if keyword.arg in DTYPE_KEYWORDS:
                found.append(Finding(KIND_DTYPE, str(keyword.arg), node.lineno))
    return tuple(sorted(set(found), key=lambda f: (f.lineno, f.kind, f.name)))


def _called_name(func: ast.expr) -> str:
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return ""


def load_definitions(tree: ast.AST) -> Iterable[ast.FunctionDef]:
    """Every ``def load(self, ctx)`` in a parsed module."""
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name != "load":
            continue
        args = node.args.args
        if len(args) != 2 or args[0].arg != "self":
            continue
        yield node  # type: ignore[misc]


def scan_source(source: str) -> tuple[Finding, ...]:
    """Every finding across every ``load`` in one module's source text."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return ()
    found: list[Finding] = []
    for definition in load_definitions(tree):
        found.extend(load_branches_on_hardware(definition))
    return tuple(found)


def scan_module(source: str) -> tuple[Finding, ...]:
    """Every finding in EVERY function of one module — the real fence."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return ()
    found: list[Finding] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            found.extend(_branches_in(node))
    return tuple(sorted(set(found), key=lambda f: (f.lineno, f.kind, f.name)))


def refusal(class_name: str, findings: tuple[Finding, ...]) -> str:
    """The message a declaration-time refusal or a repo gate prints."""
    rows = ", ".join(f.line() for f in findings)
    return (
        f"{class_name}.load() branches on hardware or numerics ({rows}). "
        f"Lane selection is PLATFORM machinery (pgw#1606): the boot ladder "
        f"already ranked this model's declared lanes against this card, this "
        f"host's kernel gates and what the deploy staged, and "
        f"`ctx.load_pipeline(...)` materializes the modules for the lane it "
        f"chose. An endpoint that asks the card its own question will answer "
        f"it differently from the ladder, and the two answers are invisible "
        f"until the numerics are already wrong"
    )


__all__ = [
    "DTYPE_KEYWORDS",
    "Finding",
    "HARDWARE_READS",
    "KINDS",
    "KIND_DTYPE",
    "KIND_HARDWARE",
    "KIND_NUMERICS",
    "NUMERICS_CALLS",
    "load_branches_on_hardware",
    "load_definitions",
    "refusal",
    "scan_module",
    "scan_source",
]
