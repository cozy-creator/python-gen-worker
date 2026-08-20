#!/usr/bin/env python3
"""NO RAW AOTI PACKAGE LOAD IN `src/gen_worker` (pgw#1460).

torchcg mints WEIGHTLESS artifacts by SEALED POLICY: `compiler.py` compiles
with `aot_inductor.package_constants_in_so: False` and `artifact.py`'s
`validate_metadata` REFUSES any artifact whose metadata says otherwise. There
is no other kind. So a bare

    torch._inductor.aoti_load_package(path)          # or
    torch._inductor.package.load_package(path)

yields a callable with an EMPTY CONSTANT BUFFER, and one that takes flat
positional feeds while its caller has the author's nested `(*args, **kwargs)`.
Arming it is wrong numerics or a hard AOTI error, on the exact path the
adopt-and-reuse program is judged on.

pgw#1460 is what happened without this fence: BOTH v2 adopt loaders --
`serving/serve_adoption.py` and a byte-identical copy in `serving/__main__.py`
-- were exactly that call, accepting the `GraphRecord` and discarding it,
while every test that covered them injected a stub loader and asserted only
that a path existed. The gap survived a full rewrite because the shape is the
obvious thing to write and nothing said otherwise.

**The sanctioned path is `torchcg.serve.aoti_loader` (tcg#58)**, which is also
the DEFAULT `AdoptSession` loader -- so the correct amount of code to write at
a call site is none. `CompiledGraphRunner` refuses direct construction and
refuses invocation before a complete bind, which makes every OTHER way into
AOTI loud; the raw package load was the one silent bypass, and this closes it.

**Scope: first-party `src/gen_worker` only.** `_vendor/torchcg` IS the
sanctioned implementation and necessarily calls the loader -- fencing it would
fence the fix. Tests are out of scope too: a test that loads a package
directly to prove the refusal (torchcg has one) is doing the right thing.

**Read as TEXT, not as AST call nodes**, and deliberately: the whole failure
mode is that the spelling is obvious, so a deferred `getattr` form is exactly
what a working-around author would reach for. Prose that EXPLAINS the banned
spelling is not a violation -- docstrings and comments are stripped before the
scan (`ast.unparse` of a docstring-blanked tree also drops comments), because
a gate that punishes the explanation makes deleting the explanation the
cheapest way to green.

Run::

    python scripts/lint_raw_aoti_load.py
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _lint_side  # noqa: E402
SRC = REPO / "src" / "gen_worker"
VENDOR = SRC / "_vendor"

#: The spellings that hand back an unbound AOTI package. `load_package` is
#: matched bare because `torch._inductor.package.load_package` is routinely
#: reached through an `import_module`/`vars()` indirection -- which is what
#: torchcg's own gated loader does, in the file this fence exempts.
BANNED = ("aoti_load_package", "load_package")

#: The one sanctioned reference, by name, so the error message can point at it
#: instead of describing it.
SANCTIONED = "torchcg.serve.aoti_loader"


def _code_only(source: str, filename: str) -> str:
    """The module's CODE, with every docstring and comment removed."""

    tree = ast.parse(source, filename=filename)
    for node in ast.walk(tree):
        if not isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            continue
        body = getattr(node, "body", None)
        if not body:
            continue
        first = body[0]
        if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
            if isinstance(first.value.value, str):
                first.value.value = ""
    return ast.unparse(tree)


def main() -> int:
    violations: list[str] = []
    scanned = 0
    for file in sorted(SRC.rglob("*.py")):
        if VENDOR in file.parents:
            continue
        scanned += 1
        code = _code_only(file.read_text(), str(file))
        for line_number, line in enumerate(code.splitlines(), start=1):
            for spelling in BANNED:
                if spelling in line:
                    violations.append(
                        f"{file.relative_to(REPO)}: names `{spelling}` in code. "
                        f"torchcg artifacts are WEIGHTLESS by sealed policy, so "
                        f"a raw package load arms an EMPTY constant buffer and "
                        f"takes flat feeds the caller does not have (pgw#1460). "
                        f"Use `{SANCTIONED}` -- which is already the default "
                        f"`AdoptSession` loader, so the right amount of code "
                        f"here is none. (code line {line_number}: {line.strip()})"
                    )
    if not scanned:
        # A fence that scans nothing reports clean. Say so instead.
        print(
            f"raw-AOTI-load fence: scanned NOTHING under {SRC} — the tree "
            f"moved and this fence is asserting over an empty set.",
            file=sys.stderr,
        )
        return 1
    if violations:
        _lint_side.report(violations, "pgw#1460 raw AOTI package loads")
    if violations:
        print(
            f"\n{len(violations)} raw AOTI package load(s).", file=sys.stderr
        )
        return 1
    print(f"raw-AOTI-load fence: clean ({scanned} first-party module(s) scanned)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
