#!/usr/bin/env python3
"""tcg#56: the retired `graph class` vocabulary does not survive in this tree.

Paul, 2026-08-19: *"yeah I like graph-specialization better. So each model has a
'graph' and then has many 'graph specializations' perhaps like 14 of them."* A
target has ONE graph; it has MANY graph SPECIALIZATIONS. `GraphClassSpec`,
`GraphClassDeclaration`, `GRAPH_CLASS_BLOCK`, `class_hash`, `class_dims` and the
prose "graph class" are gone across torchcg (`8e8c5ca6`), this repo (`e9b62bc2`,
`53142b36`) and tensorhub (`deed5ed4`). There is no shim and no alias.

WHY A FENCE AND NOT A ONE-TIME SWEEP, and this one is not hypothetical: **it had
already half-landed within hours.** `lint_retired_sdk_spellings.py` says a rename
this wide half-lands; tcg#56 proved it twice on the same day. A sibling lane's
NEW file (`cli/endpoint_lock.py`, pgw#1466) landed after the rename carrying
"a DIFFERENT graph class" in a docstring, and this repo's own `ci.yaml` comment
still described the lint's guarded value by its dead spelling. Neither is code,
both send the next reader to a vocabulary the fleet no longer speaks, and no
existing gate could see either.

WHAT IT LOOKS FOR is a retired SPELLING, by regex, in the swept files. It is a
TEXT sweep, not an AST one, because prose is the failure mode here — a docstring
saying "a different graph class" is exactly as wrong as an identifier would be.

WHAT IT DELIBERATELY DOES NOT LOOK FOR is the bare word `class`. That word is
load-bearing and CORRECT throughout this tree and its peers, and a sweep keyed on
it would have to be reverted rather than fixed. Measured during tcg#56, every one
of these is a live non-graph use:

* `pipeline_class` — a *Python* class (`"QwenImagePipeline"`);
* `reason_class`, `ConfigClassMask`, `changed_config_classes`, `pending_classes`
  — proto fields in the same file as the renamed one;
* `ClassCounts` / `u.Class` in tensorhub's `tensorlayout` and `layoutsweep` —
  TENSOR-LAYOUT classes (dtype/packing), a different domain entirely;
* `PartitionAdvisoryLockClassID` — an advisory-lock namespace;
* `classifier`, "classes of failure", `cuda_probe` failure classes;
* `class Foo:`, `dataclass`, `classmethod`, `subclass`.

So the patterns below are anchored on `graph`-qualified forms and on the two
field names that were genuinely renamed, never on the word alone.

EXCLUSIONS, all structural rather than an allowlist:

* this file, which DEFINES the retired spellings and so must spell them;
* `src/gen_worker/_vendor/` — byte-identical upstream snapshots (`_lint_scope`);
* `src/gen_worker/pb/` and `proto/` — generated protobuf and the wire contract
  it is generated from. The proto field WAS renamed (`graph_specialization`,
  number 20 unchanged), but both files legitimately carry the dead spelling in
  comments recording the rename, and `PROTO_DIGEST` records it as provenance;
* `CHANGELOG.md` and `changelog.d/` — released text is a record of what was true
  when it was written (`lint_retired_sdk_spellings.py`'s rule, same reason);
* `v1_deleted.py` — a TOMBSTONE must keep the dead symbol's exact spelling or it
  stops matching the import it exists to catch. `"GraphClass"` there is load
  bearing: an author writing `from gen_worker import GraphClass` gets a typed
  message naming the successor instead of a bare ImportError.

Everything else is recognised by a PROOF AT THE LINE: a `graph-class-spelling:`
marker with a reason after the colon. A bare marker is not a proof.

Usage:

    python scripts/lint_retired_graph_class_spellings.py [PATH ...]
    python scripts/lint_retired_graph_class_spellings.py --selftest
"""

from __future__ import annotations

import re
import sys
import tempfile
from pathlib import Path
from typing import Iterator

REPO = Path(__file__).resolve().parents[1]

#: spelling -> what replaced it. The message names the successor, because a
#: refusal that does not say what to write instead just costs a grep.
RETIRED: dict[str, str] = {
    r"\bGraphClass\w*": "GraphSpecialization... (GraphSpecialization, GraphSpecializationDeclaration, GraphSpecializationCandidate)",
    r"\bgraph_class\w*": "graph_specialization (and GRAPH_SPECIALIZATION_BLOCK for the metadata-block constant)",
    r"\bGRAPH_CLASS_BLOCK\b": "GRAPH_SPECIALIZATION_BLOCK",
    r"\bclass_hash\b": "specialization_hash",
    r"\bclass_dims\b": "specialization_dims",
    r"\bclass_count\b": "specialization_count",
    r"\bClassReport\b": "SpecializationReport",
    r"\bclass_ambiguous\b": "specialization_ambiguous",
    r"\bno_class_admits\b": "no_specialization_admits",
    r"\bone_class_admits\b": "one_specialization_admits",
    r"graph[- ]class(es)?\b": "graph specialization(s)",
    r"graph[- ]Class(es)?\b": "graph specialization(s)",
}

MARKER = "graph-class-spelling:"

#: Paths whose dead spellings are the RECORD, not a defect. Structural, not an
#: allowlist of individual findings — see the module docstring for each reason.
EXCLUDED_PARTS = (
    "_vendor",
    "__pycache__",
    ".git",
)
EXCLUDED_SUFFIXES = ("lint_retired_graph_class_spellings.py",)
EXCLUDED_RELATIVE = (
    "src/gen_worker/pb",
    "proto",
    "CHANGELOG.md",
    "changelog.d",
    "src/gen_worker/v1_deleted.py",
)


def _excluded(path: Path) -> bool:
    if any(part in EXCLUDED_PARTS for part in path.parts):
        return True
    if str(path).endswith(EXCLUDED_SUFFIXES):
        return True
    try:
        rel = path.resolve().relative_to(REPO).as_posix()
    except ValueError:
        return False
    return any(rel == e or rel.startswith(e + "/") for e in EXCLUDED_RELATIVE)


def scan_text(text: str, name: str) -> list[tuple[int, str, str]]:
    """Every retired spelling in `text`, minus lines carrying a proof."""
    hits: list[tuple[int, str, str]] = []
    for lineno, line in enumerate(text.splitlines(), 1):
        if MARKER in line and line.split(MARKER, 1)[1].strip():
            continue
        for pattern, successor in RETIRED.items():
            m = re.search(pattern, line)
            if m:
                hits.append((lineno, m.group(0), successor))
                break
    return hits


def _walk(roots: list[Path]) -> Iterator[Path]:
    for root in roots:
        if root.is_file():
            if not _excluded(root):
                yield root
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix in {".py", ".md", ".toml", ".yaml", ".yml", ".json"}:
                if not _excluded(path):
                    yield path


def selftest() -> int:
    """A sweep that can only ever print 'clean' guards nothing."""
    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        bad = Path(tmp) / "bad.py"
        bad.write_text(
            '"""A cpu-traced graph is a DIFFERENT graph class."""\n'
            "x = meta['class_hash']\n"
        )
        hits = scan_text(bad.read_text(), str(bad))
        if len(hits) != 2:
            print(f"SELFTEST FAIL: expected 2 findings, got {hits}")
            ok = False

        good = Path(tmp) / "good.py"
        good.write_text(
            '"""A cpu-traced graph is a DIFFERENT graph specialization."""\n'
            "x = meta['specialization_hash']\n"
        )
        if scan_text(good.read_text(), str(good)):
            print("SELFTEST FAIL: the current vocabulary must not be a finding")
            ok = False

        # The word `class` alone is CORRECT throughout this tree. If this arm
        # ever goes red the patterns have over-reached and the fix is the
        # pattern, never the source.
        live = Path(tmp) / "live.py"
        live.write_text(
            "pipeline_class = 'QwenImagePipeline'\n"
            "reason_class = 'cuda_unavailable'\n"
            "CARDLESS_PROBE_CLASSES = frozenset()\n"
            "class Foo:\n"
            "    @classmethod\n"
            "    def bar(cls) -> None: ...\n"
            "from dataclasses import dataclass\n"
            "counts = res.ClassCounts\n"
        )
        stray = scan_text(live.read_text(), str(live))
        if stray:
            print(f"SELFTEST FAIL: live non-graph uses of 'class' were flagged: {stray}")
            ok = False

        exempt = Path(tmp) / "exempt.py"
        exempt.write_text(
            "# graph_class was the v3 spelling  # graph-class-spelling: prose recording the rename\n"
        )
        if scan_text(exempt.read_text(), str(exempt)):
            print("SELFTEST FAIL: a classified line must not be a finding")
            ok = False

        bare = Path(tmp) / "bare.py"
        bare.write_text("x = meta['class_hash']  # graph-class-spelling:\n")
        if not scan_text(bare.read_text(), str(bare)):
            print("SELFTEST FAIL: a marker with no reason is not a proof")
            ok = False

    print("selftest: OK" if ok else "selftest: FAILED")
    return 0 if ok else 1


def main(argv: list[str]) -> int:
    if "--selftest" in argv:
        return selftest()
    roots = [Path(a) for a in argv] or [
        REPO / "src" / "gen_worker",
        REPO / "tests",
        REPO / "scripts",
        REPO / "docs",
    ]
    findings = 0
    for path in _walk(roots):
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for lineno, spelling, successor in scan_text(text, str(path)):
            rel = path.resolve()
            try:
                rel = rel.relative_to(REPO)
            except ValueError:
                pass
            print(f"{rel}:{lineno}: retired spelling {spelling!r} -> write {successor}")
            findings += 1
    if findings:
        print(
            f"\nlint_retired_graph_class_spellings: {findings} finding(s). tcg#56 renamed the "
            "compiled-graph vocabulary: a target has ONE graph and MANY graph SPECIALIZATIONS.\n"
            f"If a line must keep a dead spelling, prove it at the line with `# {MARKER} <reason>`."
        )
        return 1
    print("tcg#56: clean — no retired graph-class spelling survives")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
