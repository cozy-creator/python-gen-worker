#!/usr/bin/env python3
"""th#1987 / HARDCUT A9: no `owner/repo:tag` literal survives in this tree.

The tag production is DELETED from the hub grammar. A surviving `:tag` pin does
not fail at the line that wrote it — `gen_worker.models.refs` refuses it, so it
fails on a rented pod, mid-mint, at the resolve. This sweep finds them where
they are typed.

WHAT IT LOOKS FOR is a repo-ref-shaped literal: `<owner>/<repo>:<tag>` inside a
quoted string, in Python, JSON, TOML, YAML and Markdown. The owner segment may
not contain a `.`, which keeps a registry-qualified OCI ref (`docker.io/x:1`,
`ghcr.io/y:v2`) out.

**Image tags DO match the shape**: a Docker Hub image in a user namespace —
`pytorch/pytorch:2.13.0-cuda13.0-cudnn9-runtime` — is `<owner>/<repo>:<tag>`
exactly, with no dot in the owner (pgw#1347). The shape cannot separate a hub
model ref from a container image ref, so the separation has to be STATED.

THERE IS NO PATH ALLOWLIST, and the design point is that it needs none. The
three places a `:tag` string legitimately survives are recognised by what they
PROVE, at the line, rather than by which file they are in:

* a conformance vector carrying `"error": true` — the corpus's own statement
  that the parser must reject it;
* a source line carrying a `# refused:` comment, which says the same thing at
  the line where a test feeds it to the parser; and
* a source line carrying an `# oci-image:` comment, which states that the
  literal is a CONTAINER IMAGE consumed by a container runtime and never
  reaches `gen_worker.models.refs.parse_model_ref`. Write the reason after the
  marker.

A path allowlist would let a real pin hide behind a filename. A line-level
claim cannot: it sits on the literal, it is visible in review, and deleting it
turns the line red.

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
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _lint_side  # noqa: E402

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

#: The line-level claim that a `:tag` string is a CONTAINER IMAGE, not a hub
#: model ref — a distinction the shape cannot make (see the module docstring).
OCI_MARKER = "# oci-image:"


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
            if REFUSAL_MARKER in line or OCI_MARKER in line:
                continue
            for m in PIN_RE.finditer(line):
                pin = f"{m.group('owner')}/{m.group('repo')}:{m.group('tag')}"
                if any(r.startswith(pin) for r in refused):
                    continue
                rel = path.relative_to(REPO) if path.is_relative_to(REPO) else path
                findings.append(f"{rel}:{lineno}: retired tag pin {pin!r}")
    return findings


_SELFTEST_PIN = "acme/model" + ":prod"
#: Split the same way, and for the same reason: a fixture written as one
#: literal would make this file its own first finding.
_SELFTEST_IMAGE = "pytorch/pytorch" + ":2.13.0-cuda13.0"


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
        # pgw#1347: a namespaced Docker Hub image is ref-SHAPED. Unmarked it is
        # red; marked it is green — the marker is the whole difference.
        (root / "image_bare.py").write_text(f'IMAGE = "{_SELFTEST_IMAGE}"\n')
        (root / "image_marked.py").write_text(
            f'IMAGE = "{_SELFTEST_IMAGE}"  {OCI_MARKER} RunPod pulls this\n')

        for name in ("pin.py", "image_bare.py"):
            red = scan((root / name,))
            if len(red) != 1:
                print(f"SELFTEST FAILED: a planted pin in {name} was not caught: {red}",
                      file=sys.stderr)
                return 1
        for name in ("proven.py", "corpus.json", "docker.py", "clean.py", "image_marked.py"):
            green = scan((root / name,))
            if green:
                print(f"SELFTEST FAILED: {name} flagged: {green}", file=sys.stderr)
                return 1
    print("lint_repo_ref_pins selftest: red on a pin and on a bare image, "
          "green on a refusal proof and on a marked container image")
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
        _lint_side.report(findings, "th#1987 retired `:tag` pins")
        print(f"\n{len(findings)} retired tag pin(s)", file=sys.stderr)
        return 1
    print("lint_repo_ref_pins: no retired `:tag` repo-ref literal survives")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
