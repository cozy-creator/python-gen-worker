#!/usr/bin/env python3
"""A18 / DESIGN-RULINGS §1.32(d): the flavor is deleted, and the residue is ZERO.

WHY A LINT AND NOT ONLY A TEST. `tests/test_flavor_deletion_pgw1148.py` is a
good fence and it runs — but it runs in `tests`, which pgw#1264 took off the
merge path, so nothing in the REQUIRED context (`fast gates`) would have an
opinion about the flavor axis. A rename that regrows a reader merges green and
is found on a pod. This runs where the merge is decided.

THE RESIDUE IS ZERO (pgw#1319). `ProducedFlavor.flavor`, the
`label = flavor.flavor or ...` read and `classify_flavor_token` are DELETED;
`precision_class` is a DECLARATION at every producer that publishes a non-base
row, and `convert.publish` REFUSES sub-16-bit bytes that declare no class
rather than guessing one from a label. So this file holds no allowance at a
size: it asserts the axis names nothing at all, with NO exemption.

Three rules, all AST — a grep cannot tell `X.flavor` from `flavor_label(...)`
from `cgroup_flavor`, and this axis is a four-way homonym (the compiled graph
`#`-fragment is the one that must NOT move).

  1. DELETED names stay deleted. Nothing in `src/` binds or reads a name the
     deletion removed — `classify_flavor_token` among them now.
  2. NO `flavor`-shaped FIELD is declared anywhere. There is no residue row to
     be excused by.
  3. The token is READ nowhere. A new reader is red even though it compiles,
     because a second reader is how a deleted axis comes back.

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
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _lint_side  # noqa: E402
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
    "classify_flavor_token",
)

#: Field names that name the axis. `flavors` is deliberately absent: it is the
#: parameter of `publish_flavors`, a LIST of produced artifacts, not a selector.
FIELD_NAMES = ("flavor", "default_flavor", "sibling_flavors")

#: The A18 residue, now EMPTY. Kept as named constants rather than deleted so
#: the rules below read as "at exactly these sites, and there are none" — and
#: so that a future lane re-adding one has to say so in this file.
RESIDUE_FIELD = ""       # no module may declare a flavor field
RESIDUE_READERS: Tuple[str, ...] = ()      # no module may read the token
RESIDUE_CLASSIFIERS: Tuple[str, ...] = ()  # the classifier is deleted

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
                        f"{rel}:{node.lineno}: a read of `.{node.attr}` — the "
                        "producer-local label is DELETED and has no readers; "
                        "the class is declared, not inferred from a label")
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                    and node.func.id == CLASSIFIER and not is_classifier_def:
                if rel in RESIDUE_CLASSIFIERS:
                    seen_classifiers.append(rel)
                else:
                    findings.append(
                        f"{rel}:{node.lineno}: a {CLASSIFIER} caller — the "
                        "classifier is DELETED; declare `precision_class` from a "
                        "structural fact instead of re-deriving it from a label")

    # The residue is EMPTY, so there is no staleness rule left: every match
    # above is already a finding. These stay asserted rather than dropped so a
    # lane that re-adds a row has to make the allowance visible here.
    assert not seen_field and not seen_readers and not seen_classifiers, (
        "the A18 residue is empty; a match here means a residue row was "
        "re-added without saying so")
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
        ("graph_label", "def flavor_label(sku, torch_version):\n    return sku\n", False),
        ("cgroup", "def f(d):\n    d['cgroup_flavor'] = 'v2'\n", False),
        ("publish_list", "def publish_flavors(ctx, flavors):\n    return list(flavors)\n", False),
        ("param_named_flavor", "def g(flavor):\n    return flavor\n", False),
    ]
    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        for name, src, expect_red in cases:
            root = Path(tmp) / name
            root.mkdir()
            (root / "planted.py").write_text(src)
            findings = [f for f in scan((root,)) if f.startswith("planted.py")]
            if bool(findings) != expect_red:
                ok = False
                verb = "did not go red" if expect_red else "went red"
                print(f"SELFTEST FAILED: {name} {verb}: {findings}", file=sys.stderr)

        # The flip pgw#1319 makes: the residue is ZERO, so the paths that
        # used to be excused are excused no longer. Regrow all three there and
        # every one is red — the arm that would have gone quiet if the cut had
        # deleted the code but left the allowance behind.
        root = Path(tmp) / "regrown"
        (root / "convert").mkdir(parents=True)
        (root / "convert" / "produced.py").write_text(
            "import msgspec\n\n\nclass ProducedFlavor(msgspec.Struct):\n"
            "    flavor: str = ''\n")
        (root / "convert" / "publish.py").write_text(
            "def publish_flavors(ctx, flavors):\n"
            "    for f in flavors:\n"
            "        label = f.flavor\n"
            f"        yield {CLASSIFIER}(label)\n")
        regrown = scan((root,))
        # field + `.flavor` read + classifier CALL + the classifier's name read
        # (it is a DELETED name now, so rule 1 speaks too).
        if len(regrown) != 4:
            ok = False
            print("SELFTEST FAILED: regrowing the former residue in its own "
                  f"files must red every rule, got {regrown}", file=sys.stderr)
    if not ok:
        return 1
    print("lint_flavor_deletion selftest: red on all five regressions, green on "
          "all four homonyms, red on a regrown residue in its own former "
          "files (the residue is ZERO)")
    return 0


def main(argv: List[str]) -> int:
    if "--selftest" in argv:
        return _selftest()
    roots = tuple(Path(a).resolve() for a in argv) or DEFAULT_ROOTS
    findings = scan(roots)
    if findings:
        print("A18 / §1.32(d): the flavor is deleted and names nothing. The "
              "precision class is DECLARED by the producer (`precision_class`, "
              "checked against models.ladder.PRECISION_CLASSES); it is never "
              "inferred from a label.\n", file=sys.stderr)
        _lint_side.report(findings, "A18 flavor-token residue")
        print(f"\n{len(findings)} finding(s)", file=sys.stderr)
        return 1
    print("lint_flavor_deletion: the flavor axis is deleted; the A18 residue is "
          f"{len(RESIDUE_READERS)} readers, "
          f"{1 if RESIDUE_FIELD else 0} fields, "
          f"{len(RESIDUE_CLASSIFIERS)} classifier calls — the token names nothing")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
