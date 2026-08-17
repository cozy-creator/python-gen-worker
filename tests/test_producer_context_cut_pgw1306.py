"""pgw#1306 — the three producer context classes are GONE, and stay gone.

pgw#1294 merged `ConversionContext` / `DatasetContext` / `TrainingContext` into
`JobContext` and left the three names as thin aliases with a sentence naming
th#2052 as their executioner. **th#2052 is a tensorhub commit and cannot delete
Python**, so the sentence had no repo that could carry it out. This module is
the carried-out sentence plus the fence that keeps it carried out.

WHY A TEXT FENCE AND NOT A SYMBOL CHECK (deliverable 2, as filed). An alias can
come back through a re-export — `from .request_context import ConversionContext`
in `__init__.py` — and an `assert not hasattr(gen_worker, ...)` written against
the package would still pass if a submodule kept the name and something else
imported it from there. So the fence reads SOURCE TEXT across the tree.

WHY NO TYPED WIRE REFUSAL. Nothing retired here ever crossed a wire. `kind` is
an AUTHOR declaration read from local source — `@endpoint(kind=...)` at import
time, validated against a closed set in `discovery.validation._KNOWN_KINDS` —
and `execution_hints["kind"]` is outbound-only. A worker never learns a kind
from a request, so no request shaped by an older wheel can arrive carrying one
of these names. The peer-side story is the wheel boundary, not the wire: a
package pinned to 0.122.0 keeps the aliases in its OWN wheel and is unaffected;
one that repins gets an ImportError at import time, which is immediate, loud
and located. That is a hard cut, and it is recorded here rather than softened
with a shim — a module `__getattr__` naming the three strings would defeat the
fence above.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

#: The three names, whole-word.
RETIRED = ("ConversionContext", "DatasetContext", "TrainingContext")
_PATTERN = re.compile(r"\b(" + "|".join(RETIRED) + r")\b")

#: Scanned roots. `CHANGELOG.md` and `changelog.d/` are deliberately absent:
#: they are an append-only HISTORY of released versions, and rewriting what
#: 0.4.x shipped to match today's tree would be a lie. Everything else that can
#: carry an import IS here — `.github/` included, because a workflow smoke-import
#: (`python -c "from gen_worker import ..."`) is exactly the shape that breaks a
#: build without any source file naming the symbol. Verified zero there at the
#: cut; it stays scanned so it stays zero.
SCAN_ROOTS = (
    "src", "tests", "tests_v2", "scripts", "docs", "examples", ".github",
)
SCAN_FILES = ("README.md", "pyproject.toml", "conftest.py")

#: TOMBSTONES — the only places allowed to spell a retired name, each with the
#: number of lines that may do so. Declared, not blanket, and the count is the
#: point: a tombstone that grows is a re-import hiding behind prose, and a
#: declaration whose count no longer matches is refused in `test_tombstones_...`
#: below. (pgw#1302 step 6 paid for this rule the other way round — a
#: declaration that no longer describes anything is as bad as an undeclared
#: hit.) A tombstone earns its exemption by saying what replaced the name; do
#: not add one to keep an ordinary reference alive.
TOMBSTONES = {
    # The fence itself must spell what it forbids.
    "tests/test_producer_context_cut_pgw1306.py": 7,
    # The deletion site, where the successor and the reason live.
    "src/gen_worker/request_context/__init__.py": 1,
    # Author-facing: "do not reintroduce a per-kind context" needs the names.
    "docs/endpoint-authoring.md": 2,
}

#: Suffixes worth reading. A wheel-name grep over .png is noise.
SUFFIXES = {".py", ".pyi", ".md", ".toml", ".txt", ".yaml", ".yml", ".cfg", ".json"}


def _tracked_files() -> list[Path]:
    """Ask git, not the filesystem: a stale `.venv`, `__pycache__` or a
    sibling's scratch file under the tree is not this repo's source.

    `--others --exclude-standard` includes files that are NOT YET COMMITTED, so
    a new file reintroducing a retired name goes red in the author's own run
    rather than one commit later."""
    out = subprocess.run(
        ["git", "-C", str(REPO), "ls-files", "-z", "--cached", "--others",
         "--exclude-standard", "--", *SCAN_ROOTS, *SCAN_FILES],
        capture_output=True, text=True, check=True,
    ).stdout
    paths = sorted({p for p in out.split("\0") if p})
    assert paths, "git ls-files returned nothing — the fence would be vacuous"
    return [REPO / p for p in paths]


def test_the_scan_actually_reads_files() -> None:
    """The fence's own red arm: prove the corpus is non-empty and that the
    pattern can MATCH, so a green below means "absent" and not "never looked".
    A fence that cannot spell the symbol it guards counts zero and means
    nothing (the lesson pgw#1302 step 6 paid for)."""
    files = _tracked_files()
    py = [f for f in files if f.suffix == ".py"]
    assert len(py) > 200, f"only {len(py)} python files scanned"
    assert _PATTERN.search("a ConversionContext here")
    assert _PATTERN.search("DatasetContext")
    assert _PATTERN.search("TrainingContext,")
    # Whole-word: the pattern must not fire on an unrelated longer identifier.
    assert not _PATTERN.search("MyConversionContextFactory")


def _scan() -> dict[str, list[str]]:
    """rel path -> the lines in it that name a retired context."""
    found: dict[str, list[str]] = {}
    for path in _tracked_files():
        if path.suffix not in SUFFIXES:
            continue
        rel = path.relative_to(REPO).as_posix()
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            if _PATTERN.search(line):
                found.setdefault(rel, []).append(f"{rel}:{lineno}: {line.strip()}")
    return found


def test_no_source_file_names_a_retired_producer_context() -> None:
    hits = [
        line
        for rel, lines in _scan().items()
        if rel not in TOMBSTONES
        for line in lines
    ]
    assert not hits, (
        "pgw#1306 retired ConversionContext / DatasetContext / TrainingContext. "
        "`JobContext` is the one producer context, and a body's write authority "
        "comes from its @job/@endpoint declaration, never from its kind:\n  "
        + "\n  ".join(hits)
    )


def test_tombstones_describe_exactly_what_they_claim() -> None:
    """A declaration is a claim about the tree, so it can be WRONG in two
    directions and both are red: a tombstone that grew is a reference hiding
    behind an exemption, and one that shrank to zero is an exemption for
    nothing. Neither is caught by the scan above, which only ever looks
    outside them."""
    found = _scan()
    problems: list[str] = []
    for rel, expected in sorted(TOMBSTONES.items()):
        actual = len(found.get(rel, []))
        if actual != expected:
            problems.append(
                f"{rel}: declared {expected} tombstone line(s), found {actual}"
                + ("  (delete the declaration)" if actual == 0 else "")
            )
    assert not problems, "\n  ".join(["stale tombstone declarations:", *problems])


def test_the_names_are_not_importable_from_anywhere_they_used_to_be() -> None:
    import gen_worker
    from gen_worker import request_context

    for name in RETIRED:
        assert not hasattr(gen_worker, name), name
        assert not hasattr(request_context, name), name
        assert name not in gen_worker.__all__, name
        with pytest.raises(ImportError):
            exec(f"from gen_worker import {name}", {})
        with pytest.raises(ImportError):
            exec(f"from gen_worker.request_context import {name}", {})


def test_job_context_is_still_exported_under_its_own_name() -> None:
    """The other half of the cut: deleting three names must not have taken the
    successor with them."""
    import gen_worker
    from gen_worker.request_context import JobContext

    assert gen_worker.JobContext is JobContext
    assert "JobContext" in gen_worker.__all__


def test_no_kind_selects_a_different_producer_context() -> None:
    """The PROPERTY the three classes used to carry, asserted directly rather
    than by their absence: the executor's kind map answers `producer?` and
    nothing else, so every producer kind resolves to one class."""
    from gen_worker.executor import _CONTEXT_BY_KIND
    from gen_worker.request_context import JobContext, RequestContext

    assert _CONTEXT_BY_KIND["inference"] is RequestContext
    producer = {k: v for k, v in _CONTEXT_BY_KIND.items() if k != "inference"}
    assert set(producer) == {"conversion", "dataset", "training", "eval"}
    assert set(producer.values()) == {JobContext}


def test_write_authority_comes_from_the_declaration_not_the_kind() -> None:
    """`publishes` is what gates the publisher surface. Two contexts of the
    SAME class and (implicitly) the same kind disagree about writing, because
    the declaration is what differs — which is the whole point of the cut."""
    from gen_worker.request_context import JobContext

    declared = JobContext(request_id="r-1306-yes", publishes=True)
    undeclared = JobContext(request_id="r-1306-no", publishes=False)
    assert type(declared) is type(undeclared)
    assert declared.publishes is True
    assert undeclared.publishes is False
