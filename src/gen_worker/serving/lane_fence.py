"""The ONE reader for "does this ``load()`` branch on hardware or numerics?"

pgw#1606 acceptance (d): *"a fence test finds zero dtype/quant branches in
endpoint ``load()`` bodies."* Lane selection is platform machinery; an endpoint
that reads `get_device_capability()` or calls `torchao.quantize_` has taken the
decision back, and it will take it differently from the ladder.

**Split out exactly as `load_marks_compile` was (pgw#1581/se#809), for the same
reason.** A reader reachable only through `inspect.getsource` can only be used
by code that already IMPORTED the endpoint — and an endpoint module imports
torch. Every repo-side gate over the fleet is therefore torch-free and
AST-based. Without a public parsed-node entry point, the endpoints repo has to
write a SECOND walker, and the last time that happened the two disagreed about
how many endpoints compiled (2 vs 13). One reader, two entry points.

Parsed, never grepped. The string `torch.cuda.get_device_capability` appears in
comments explaining why a model deliberately does NOT branch on hardware, and a
substring check would flag exactly the classes that documented themselves best.

The findings this returns are the CURRENT fleet's, verbatim from the audit:

  * `minimax-h3` — `w8a8_capable()`/`fa3_capable()` read the capability tuple
    and `quantize_dit()` runs `torchao.quantize_` at setup;
  * `joycaption` — `load()` branches `is_bf16_supported()` to choose a dtype;
  * `anima` — an endpoint-owned shim calling `sanitize_w8a8_state_dict`;
  * `ltx-video-2.3` — `apply_fp8_storage(...)` from endpoint code;
  * `wan-2.2` — an sm90 branch selecting an fp8 attention kernel.
"""

from __future__ import annotations

import ast
from typing import Any, Iterable

#: Attribute calls that ask the CARD a question. The platform owns this
#: question — `serving/lane_host.host_card_facts()` is the one place it is
#: asked, and its answer is what the ladder ranks against.
HARDWARE_READS: tuple[str, ...] = (
    "get_device_capability",
    "is_bf16_supported",
    "get_device_name",
    "mem_get_info",
    "get_device_properties",
)

#: Calls that change what the WEIGHTS are. Every one of these is a lane
#: decision wearing a function call, and every one has a platform home:
#: `lane_materialize.materialize` for the swaps, the ladder for the choice.
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

#: Keywords that hand a dtype to a loader from author code. `torch_dtype=` is
#: the spelling `ctx.load` deleted (pgw#1380: "no `torch_dtype=` — the lane
#: contract IS the dtype"), and it keeps coming back through pipeline shims.
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
    """Every hardware/numerics branch in one parsed ``load`` DEFINITION.

    Takes an `ast.FunctionDef`/`AsyncFunctionDef` — the same shape
    `load_marks_compile` takes — so a torch-free repo-side gate and pgw's own
    declaration-time refusal share this one walker.

    Empty tuple means the body is clean, which is what every endpoint should
    read after the fleet migration.
    """
    if not isinstance(definition, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return ()
    return _branches_in(definition)


def _branches_in(definition: ast.AST) -> tuple[Finding, ...]:
    """The one walker. Deduplicated and ordered by position, so a gate's
    output is stable and a diff of it reads as "this line moved" rather than
    "everything changed"."""
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
    """Every ``def load(self, ctx)`` in a parsed module.

    Matched on the RULED signature (pgw#1382) rather than on the name alone, so
    an unrelated helper called `load` in an endpoint module is not swept in.
    "Two arguments" is not enough on its own — a module-level
    ``def load(path, dtype)`` is also two — so the first must be ``self``.
    The SECOND name is free: `load(self, ctx)` and `load(self, context)` are
    the same method, and pinning the parameter's spelling would let a rename
    turn the fence off.
    """
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
    """Every finding across every ``load`` in one module's source text.

    Torch-free and import-free: the caller needs only the bytes on disk, which
    is what makes this usable from the endpoints repo's own CI.

    ⚠️ **Narrow on purpose, and NOT the fence.** pgw#1606 acceptance (d) is
    worded *"zero dtype/quant branches in endpoint `load()` bodies"*, and
    measured against the fleet that wording is very nearly VACUOUS: of the 23
    branches the audit found across five endpoints, **21 live one function
    outside `load()`** — in `minimax-h3`'s own `serve_recipe` module, in
    `anima`'s pipeline shim, in `ltx`'s and `wan`'s helpers. Four of those five
    endpoints pass this scan today with every branch intact. Use
    :func:`scan_module` for the fence; this one exists because "what is in
    `load()` itself" is still the number acceptance (d) names.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return ()
    found: list[Finding] = []
    for definition in load_definitions(tree):
        found.extend(load_branches_on_hardware(definition))
    return tuple(found)


def scan_module(source: str) -> tuple[Finding, ...]:
    """Every finding in EVERY function of one module — the real fence.

    A branch does not stop being a lane decision by being extracted into a
    helper. `minimax-h3` reads the capability tuple in `w8a8_capable()` and
    calls it from `setup()`; `anima` casts inside its pipeline class's own
    `from_pretrained`. A fence scoped to `load()` scores both as clean, which
    is the shape of guard this workspace has been bitten by before: it goes
    green by construction and nobody re-reads its scope.
    """
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
