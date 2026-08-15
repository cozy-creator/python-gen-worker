#!/usr/bin/env python3
"""th#1987 / HARDCUT A9: no `owner/repo:tag` literal survives in this tree.

The tag production is DELETED from the hub grammar. A surviving `:tag` pin does
not fail at the line that wrote it — `gen_worker.models.refs` refuses it, so it
fails on a rented pod, mid-mint, at the resolve. This sweep finds them where
they are typed.

WHAT IT LOOKS FOR is a repo-ref-shaped literal: `<owner>/<repo>:<tag>` inside a
quoted string, in Python, JSON, TOML, YAML and Markdown. The owner segment may
not contain a `.` — that is what keeps docker/OCI refs (`docker.io/x:1`,
`ghcr.io/y:v2`) out; git tags, image tags and `sTAGe` substrings never match the
shape at all.

THERE IS NO ALLOWLIST, and the design point is that it needs none. The two
places a `:tag` string legitimately survives are places that PROVE IT IS
REFUSED, and each is recognised by that proof rather than by its path:

* a conformance vector carrying `"error": true` — the corpus's own statement
  that the parser must reject it; and
* a source line carrying a `# refused:` comment, which says the same thing at
  the line where a test feeds it to the parser.

A path allowlist would let a real pin hide behind a filename. Proof-of-refusal
cannot: delete the refusal and the line goes red.

Usage:

    python scripts/lint_repo_ref_pins.py [PATH ...]
    python scripts/lint_repo_ref_pins.py --selftest

Defaults to `src/`, `tests/`, `tests_v2/`, `examples/`, `docs/`, `benchmarks/`
and `scripts/`.
"""

from __future__ import annotations

import json
import re
import sys
import tempfile
from pathlib import Path
from typing import Iterator, List, Tuple

REPO = Path(__file__).resolve().parents[1]

DEFAULT_ROOTS = (
    REPO / "src", REPO / "tests", REPO / "tests_v2", REPO / "examples",
    REPO / "docs", REPO / "benchmarks", REPO / "scripts",
)

SUFFIXES = {".py", ".json", ".toml", ".yaml", ".yml", ".md", ".txt"}

#: `<owner>/<repo>:<tag>` inside a quote. The owner segment forbids `.` so a
#: registry host (`docker.io/...`, `ghcr.io/...`, `quay.io/...`) is not a repo
#: ref; the tag segment forbids `/` so a path with a colon is not one either.
PIN_RE = re.compile(
    r"""["'](?P<owner>[A-Za-z0-9][A-Za-z0-9_-]*)"""
    r"""/(?P<repo>[A-Za-z0-9][A-Za-z0-9._-]*)"""
    r""":(?P<tag>[A-Za-z0-9][A-Za-z0-9._-]*)["'#@]"""
)

#: The line-level proof that a `:tag` string is an INPUT to a refusal.
REFUSAL_MARKER = "# refused:"


def _iter_files(roots: Tuple[Path, ...]) -> Iterator[Path]:
    for root in roots:
        if root.is_file():
            yield root
            continue
        for p in sorted(root.rglob("*")):
            if p.is_file() and p.suffix in SUFFIXES and "__pycache__" not in p.parts:
                yield p


def _refused_json_vectors(path: Path) -> set:
    """Every `ref` a JSON conformance corpus declares must be REFUSED."""
    try:
        doc = json.loads(path.read_text())
    except Exception:  # noqa: BLE001 — not a corpus, so it proves nothing
        return set()
    out = set()
    vectors = doc.get("vectors") if isinstance(doc, dict) else None
    for vec in vectors or []:
        if isinstance(vec, dict) and vec.get("error") and isinstance(vec.get("ref"), str):
            out.add(vec["ref"])
    return out


def scan(paths: Tuple[Path, ...]) -> List[str]:
    findings: List[str] = []
    for path in _iter_files(paths):
        try:
            text = path.read_text()
        except (OSError, UnicodeDecodeError):
            continue
        if ":" not in text:
            continue
        refused = _refused_json_vectors(path) if path.suffix == ".json" else set()
        for lineno, line in enumerate(text.splitlines(), 1):
            if REFUSAL_MARKER in line:
                continue
            for m in PIN_RE.finditer(line):
                pin = f"{m.group('owner')}/{m.group('repo')}:{m.group('tag')}"
                if any(r.startswith(pin) for r in refused):
                    continue
                rel = path.relative_to(REPO) if path.is_relative_to(REPO) else path
                findings.append(f"{rel}:{lineno}: retired tag pin {pin!r}")
    return findings


_SELFTEST_PIN = "acme/model" + ":prod"


def _selftest() -> int:
    """The fence must go RED on a planted pin and GREEN on a proven refusal."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "pin.py").write_text(f'REF = "{_SELFTEST_PIN}"\n')
        (root / "proven.py").write_text(
            f'BAD = "{_SELFTEST_PIN}"  {REFUSAL_MARKER} th#1987\n')
        (root / "corpus.json").write_text(json.dumps(
            {"vectors": [{"ref": _SELFTEST_PIN, "error": True}]}))
        (root / "docker.py").write_text('IMAGE = "docker.io/tensorhub:0.1"\n')
        (root / "clean.py").write_text('REF = "acme/model@prod"\n')

        red = scan((root / "pin.py",))
        if len(red) != 1:
            print(f"SELFTEST FAILED: a planted pin was not caught: {red}", file=sys.stderr)
            return 1
        for name in ("proven.py", "corpus.json", "docker.py", "clean.py"):
            green = scan((root / name,))
            if green:
                print(f"SELFTEST FAILED: {name} flagged: {green}", file=sys.stderr)
                return 1
    print("lint_repo_ref_pins selftest: the fence goes red on a pin and green on a proof")
    return 0


def main(argv: List[str]) -> int:
    if "--selftest" in argv:
        return _selftest()
    roots = tuple(Path(a).resolve() for a in argv) or DEFAULT_ROOTS
    findings = scan(roots)
    if findings:
        print("th#1987: the `:tag` ref production is DELETED, and these literals "
              "still name it.\nWrite `owner/repo@<release>` (cut a release, then "
              "attach artifacts to it); a `:tag` ref is refused by "
              "gen_worker.models.refs.parse_model_ref on the pod, not here.\n",
              file=sys.stderr)
        for f in findings:
            print(f, file=sys.stderr)
        print(f"\n{len(findings)} retired tag pin(s)", file=sys.stderr)
        return 1
    print("lint_repo_ref_pins: no retired `:tag` repo-ref literal survives")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
