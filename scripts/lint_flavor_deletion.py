#!/usr/bin/env python3
"""A18 / DESIGN-RULINGS §1.32(d): the flavor dies entirely, and what is left of
it may not GROW while it waits for its cross-repo leg.

WHY A LINT AND NOT ONLY A TEST. `tests/test_flavor_deletion_pgw1148.py` is a
good fence and it runs — but it runs in `tests`, which pgw#1264 took off the
merge path, so nothing in the REQUIRED context (`fast gates`) has an opinion
about the flavor axis. A rename that regrows a reader merges green and is found
on a pod. This runs where the merge is decided.

WHAT SURVIVES, AND WHY IT IS NOT CUT HERE. `ProducedFlavor.flavor` is a
PRODUCTION AUTHOR SURFACE: `training-endpoints/conversion` constructs it at 14
sites and `scripts/author_surface_allowlist.txt` pins `publish_flavors` to
`conversion/src/conversion/_common.py:9`. The token is not decoration — it is
the input to `classify_flavor_token`, which derives
`checkpoints.metadata["placement"]["precision_class"]`, the hub's strongest
evidence for a stored precision class where no tensor-layout contract is proven
(tensorhub `precision.StoredPrecisionOf`). Deleting the field from this repo
alone would either fail every te producer with a `TypeError` on a rented pod, or
— worse — silently drop the precision class of every svdq/fp8/nvfp4 row whose
producer states it only through the token. So the residue stays, NAMED, with its
removal condition, and this fence holds it at exactly the size it is today.

THE REMOVAL CONDITION, so this file states it rather than a tracker page: every
te producer whose token classifies to a non-base class declares
`precision_class` in its own attribute bag (te already does this once, at
`conversion/src/conversion/quant/modelopt.py:1262`). When that lands,
`classify_flavor_token`, the `label` read and `ProducedFlavor.flavor` all go,
`RESIDUE` below becomes empty, and this fence becomes a pure deletion check.

Three rules, all AST — a grep cannot tell `X.flavor` from `flavor_label(...)`
from `cgroup_flavor`, and this axis is a four-way homonym (the compile-cell
`#`-fragment is the one that must NOT move).

  1. DELETED names stay deleted. Nothing in `src/` binds or reads a name the
     deletion removed.
  2. A `flavor`-shaped FIELD is declared at exactly one place, the named residue.
  3. The token is READ at exactly the declared sites. A new reader is red even
     though it compiles, because a second reader is how a deleted axis comes
     back.

Usage:

    python scripts/lint_flavor_deletion.py [PATH ...]
    python scripts/lint_flavor_deletion.py --selftest
"""

from __future__ import annotations

import ast
import sys
import tempfile
from pathlib import Path
from typing import Iterator, List, Tuple

REPO = Path(__file__).resolve().parents[1]
DEFAULT_ROOTS = (REPO / "src" / "gen_worker",)

#: Names §1.32(d) / pgw#1148 / th#1803 deleted. Binding or reading one in `src/`
#: means the axis grew back under an old spelling.
DELETED_NAMES = (
    "flavor_token",
    "pick_family_fp8_flavor",
    "maybe_rebind_family_fp8",
    "select_gguf",
    "maybe_rebind_gguf",
    "fetch_gguf_snapshot",
    "WorkerResolvedFlavor",
    "sibling_flavors",
    "default_flavor",
)

#: Field names that name the axis. `flavors` is deliberately absent: it is the
#: parameter of `publish_flavors`, a LIST of produced artifacts, not a selector.
FIELD_NAMES = ("flavor", "default_flavor", "sibling_flavors")

#: The A18 residue, at the size it is allowed to be. `<relpath>: <what>`.
#: Every row dies with the te leg described in this module's docstring; a row
#: that no longer matches anything is red too, so the list cannot outlive it.
RESIDUE_FIELD = "convert/produced.py"          # ProducedFlavor.flavor
RESIDUE_READERS = ("convert/publish.py",)      # the one `label = ...` derivation
RESIDUE_CLASSIFIERS = ("convert/publish.py",)  # the one classify_flavor_token call

CLASSIFIER = "classify_flavor_token"


def _iter_py(roots: Tuple[Path, ...]) -> Iterator[Path]:
    for root in roots:
        if root.is_file():
            yield root
            continue
        for p in sorted(root.rglob("*.py")):
            if "__pycache__" not in p.parts:
                yield p


def _rel(path: Path, roots: Tuple[Path, ...]) -> str:
    for root in roots:
        base = root if root.is_dir() else root.parent
        if path.is_relative_to(base):
            return path.relative_to(base).as_posix()
    return path.name


def scan(roots: Tuple[Path, ...]) -> List[str]:
    findings: List[str] = []
    seen_field: List[str] = []
    seen_readers: List[str] = []
    seen_classifiers: List[str] = []

    for path in _iter_py(roots):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue
        rel = _rel(path, roots)
        # The classifier's own definition module is not a reader of it.
        is_classifier_def = any(
            isinstance(n, ast.FunctionDef) and n.name == CLASSIFIER
            for n in ast.walk(tree))

        for node in ast.walk(tree):
            # Rule 1 — a deleted name, bound or read.
            name = None
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = node.name
            elif isinstance(node, ast.Name):
                name = node.id
            elif isinstance(node, ast.Attribute):
                name = node.attr
            if name in DELETED_NAMES:
                findings.append(
                    f"{rel}:{node.lineno}: {name!r} is DELETED (§1.32(d)) and "
                    "must not be bound or read")

            # Rule 2 — a field that names the axis.
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) \
                    and node.target.id in FIELD_NAMES:
                if rel == RESIDUE_FIELD:
                    seen_field.append(rel)
                else:
                    findings.append(
                        f"{rel}:{node.lineno}: a new {node.target.id!r} field — "
                        "the flavor is not an axis; state the bytes with `dtype` "
                        "+ `artifact_contract`, or the class with `precision_class`")

            # Rule 3 — a read of the token, or a call to the classifier.
            if isinstance(node, ast.Attribute) and node.attr in FIELD_NAMES:
                if rel in RESIDUE_READERS:
                    seen_readers.append(rel)
                else:
                    findings.append(
                        f"{rel}:{node.lineno}: a new read of `.{node.attr}` — the "
                        "producer-local label has ONE reader and is losing that "
                        "one; see this script's docstring for the te leg")
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                    and node.func.id == CLASSIFIER and not is_classifier_def:
                if rel in RESIDUE_CLASSIFIERS:
                    seen_classifiers.append(rel)
                else:
                    findings.append(
                        f"{rel}:{node.lineno}: a new {CLASSIFIER} caller — it is a "
                        "package-internal choke point that dies with the token; "
                        "nothing new may grow on it")

    # A residue row matching nothing is stale, and a stale row reads as a fence.
    if not seen_field:
        findings.append(
            f"RESIDUE_FIELD names {RESIDUE_FIELD!r}, which declares no flavor "
            "field any more — the te leg landed: delete the row (and probably "
            "this whole rule)")
    for row in RESIDUE_READERS:
        if row not in seen_readers:
            findings.append(
                f"RESIDUE_READERS names {row!r}, which no longer reads the token "
                "— delete the row")
    for row in RESIDUE_CLASSIFIERS:
        if row not in seen_classifiers:
            findings.append(
                f"RESIDUE_CLASSIFIERS names {row!r}, which no longer calls "
                f"{CLASSIFIER} — delete the row")
    return findings


def _selftest() -> int:
    """Each rule must go RED on a planted regression and GREEN on the homonyms.

    The homonyms are the point: this axis has four spellings and only one of
    them is the deleted selector. A fence that cannot tell them apart would be
    reverted by the first lane that touched the compile cache.
    """
    cases: List[Tuple[str, str, bool]] = [
        # (name, source, expect_red)
        ("deleted_name", "def select_gguf(x):\n    return x\n", True),
        ("deleted_attr", "def f(r):\n    return r.sibling_flavors\n", True),
        ("new_field", "import msgspec\n\n\nclass S(msgspec.Struct):\n    flavor: str = ''\n", True),
        ("new_reader", "def f(x):\n    return x.flavor\n", True),
        ("new_classifier", "def f(t):\n    return classify_flavor_token(t)\n", True),
        # Homonyms that must stay GREEN.
        ("cell_label", "def flavor_label(sku, torch_version):\n    return sku\n", False),
        ("cgroup", "def f(d):\n    d['cgroup_flavor'] = 'v2'\n", False),
        ("publish_list", "def publish_flavors(ctx, flavors):\n    return list(flavors)\n", False),
        ("param_named_flavor", "def g(flavor):\n    return flavor\n", False),
    ]
    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        for name, src, expect_red in cases:
            root = Path(tmp) / name
            root.mkdir()
            # Satisfy the residue rows so only the planted rule can speak.
            (root / RESIDUE_FIELD).parent.mkdir(parents=True, exist_ok=True)
            (root / RESIDUE_FIELD).write_text(
                "import msgspec\n\n\nclass ProducedFlavor(msgspec.Struct):\n"
                "    flavor: str = ''\n")
            (root / RESIDUE_READERS[0]).write_text(
                "def publish_flavors(ctx, flavors):\n"
                "    for f in flavors:\n"
                "        label = f.flavor\n"
                f"        yield {CLASSIFIER}(label)\n")
            (root / "planted.py").write_text(src)
            findings = [f for f in scan((root,)) if f.startswith("planted.py")]
            if bool(findings) != expect_red:
                ok = False
                verb = "did not go red" if expect_red else "went red"
                print(f"SELFTEST FAILED: {name} {verb}: {findings}", file=sys.stderr)

        # The staleness rule: a residue row that matches nothing is red.
        root = Path(tmp) / "stale"
        (root / "convert").mkdir(parents=True)
        (root / RESIDUE_FIELD).write_text("X = 1\n")
        (root / RESIDUE_READERS[0]).write_text("Y = 2\n")
        stale = scan((root,))
        if len(stale) != 3:
            ok = False
            print(f"SELFTEST FAILED: a fully-landed te leg should red all three "
                  f"residue rows, got {stale}", file=sys.stderr)
    if not ok:
        return 1
    print("lint_flavor_deletion selftest: red on all five regressions, green on "
          "all four homonyms, red on a stale residue row")
    return 0


def main(argv: List[str]) -> int:
    if "--selftest" in argv:
        return _selftest()
    roots = tuple(Path(a).resolve() for a in argv) or DEFAULT_ROOTS
    findings = scan(roots)
    if findings:
        print("A18 / §1.32(d): the flavor is deleted and its residue may not "
              "grow. See scripts/lint_flavor_deletion.py for the residue's "
              "removal condition (a training-endpoints leg).\n", file=sys.stderr)
        for f in findings:
            print(f, file=sys.stderr)
        print(f"\n{len(findings)} finding(s)", file=sys.stderr)
        return 1
    print("lint_flavor_deletion: the flavor axis is deleted; the A18 residue is "
          f"{len(RESIDUE_READERS)} reader, 1 field, "
          f"{len(RESIDUE_CLASSIFIERS)} classifier call — unchanged")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
