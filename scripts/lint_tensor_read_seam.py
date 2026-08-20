#!/usr/bin/env python3
"""pgw#1550: ONE tensorfs-aware reader turns checkpoint files into tensors.

THE INCIDENT. For ~21 h no endpoint in the fleet completed a serve. Root cause:
gen-worker had TWO construction paths for the tensorfs-aware streaming loader —
the local CLI host asked for it, the serverless worker's ``ServeLoop`` never
did — so every pod fell through to a naive file reader, met a ~128 B
``TFSSTUB1`` pointer stub where it expected a safetensors header, and refused.
Every local test passed, because every local test went through the caller that
asked. Paul's verdict: *"this is why you need fewer code paths / DRY
principles."*

WHAT THIS FENCE ASSERTS. A projected tensorfs tree is symlinks for non-tensor
files plus pointer stubs for tensor containers; the bytes are chunked into the
CAS and are unreachable by any path-based read. So a raw tensor-file read in
first-party serving code is not a style question — it is that outage, ready to
happen again. Every such read must go through
``gen_worker.models.tensor_source`` (``open_tensor_source`` /
``load_state_dict``), which keeps ``safe_open``'s shape and moves the source.

WHAT IT DOES NOT ASSERT. ``from_pretrained`` is not banned: it is the eager
bridge for a tree with NO chunk store behind it, and ``ctx.load`` already
refuses to hand it a projected one. This fence is about the RAW readers —
``safetensors.safe_open``, ``safetensors.torch.load_file``, ``torch.load`` —
which have no such gate and cannot acquire one from outside.

SCOPE. ``src/gen_worker`` minus ``_vendor``/``pb`` (``_lint_scope``), and the
endpoint packages of the sibling ``serverless-endpoints`` checkout when it is
present — an endpoint that opens a container itself is the same defect wearing
a different repo, and pgw#1550's audit found several.

EXEMPTION IS A PROOF AT THE LINE, never a path allowlist::

    tensors = load_file(path)  # tensor-seam-exempt: <why this cannot be a stub>

A reason is mandatory. A bare marker is a violation, because "somebody wrote a
marker" is not evidence and an unexplained one is what a future reader deletes
the wrong half of.

Usage:
    scripts/lint_tensor_read_seam.py [ROOT ...]
    scripts/lint_tensor_read_seam.py --selftest
"""

from __future__ import annotations

import ast
import os
import sys
import tempfile
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lint_scope import is_unowned  # noqa: E402
import _lint_side  # noqa: E402

SRC = REPO / "src" / "gen_worker"
def _sibling_endpoints() -> Path:
    """The `serverless-endpoints` checkout beside this one, wherever we run.

    NOT `REPO.parent / ...`: in a worktree under `~/cozy/.worktrees/<repo>/<br>`
    that resolves two directories below the workspace and silently finds
    nothing — the "absent scan reads as clean" failure this fence has its own
    empty-scan rail for. Walk up instead, and report absence rather than
    scoring it green.
    """
    for base in (REPO, *REPO.parents):
        candidate = base.parent / "serverless-endpoints"
        # `is_dir()` is not enough: `~/cozy/.worktrees/serverless-endpoints`
        # is a real directory holding BRANCH worktrees, and stopping there
        # found zero endpoints and scanned nothing while reporting green.
        # Identify the checkout by what it contains, not by its name.
        if candidate.is_dir() and any(candidate.glob("*/endpoint.toml")):
            return candidate
    return REPO.parent / "serverless-endpoints"


#: The sibling endpoint checkout. Absent in CI's pgw-only clone, which is why
#: its absence is reported rather than silently scoring green.
ENDPOINTS = _sibling_endpoints()

MARKER = "tensor-seam-exempt:"

#: The seam itself, and the modules whose whole job is to be the thing behind
#: it. These are STRUCTURAL exemptions: `tensor_source` IS the sanctioned
#: reader, `safetensors_header` reads the 8-byte length that decides stub vs
#: file, and `projection` reads the stub. A guard that refused them would be
#: refusing its own remedy.
SANCTIONED = frozenset({
    SRC / "models" / "tensor_source.py",
    SRC / "models" / "safetensors_header.py",
    SRC / "models" / "projection.py",
    SRC / "models" / "materialized_view.py",
})

#: The only two modules that may name `LoadContext(` — the class's own module,
#: and pgw#1549's single production factory. See `_load_context_sites`.
CONTEXT_BUILDERS = frozenset({
    SRC / "serving" / "context.py",
    SRC / "serving" / "worker_context.py",
})

#: Directories whose job is to PRODUCE checkpoints from upstream source trees
#: (ingest -> normalize -> contract -> CAS). They read files that are files by
#: construction: a conversion input has not been projected yet, and its output
#: is being written by the same process. The serving fleet never enters here.
PRODUCER_DIRS = (SRC / "convert",)

#: `module.attr` spellings that read tensors from a path.
DOTTED = {
    ("safetensors", "safe_open"),
    ("safetensors", "torch", "load_file"),
    ("torch", "load"),
    ("st", "load_file"),
}
#: Bare names, when imported from a tensor library. Tracked per-file: a local
#: function called `load_file` is not this.
BARE = {"safe_open", "load_file", "st_load_file"}
TENSOR_MODULES = ("safetensors", "safetensors.torch", "torch")


def _dotted_name(node: ast.AST) -> Tuple[str, ...]:
    parts: List[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return tuple(reversed(parts))
    return ()


def _exempt(lines: Sequence[str], lineno: int) -> Tuple[bool, str]:
    """Is this line proved, and with what reason. Looks at the line and the
    three above it, because a long call wraps and the proof belongs with the
    explanation rather than crammed onto the closing paren."""
    for index in range(max(0, lineno - 4), min(len(lines), lineno)):
        text = lines[index]
        if MARKER in text:
            reason = text.split(MARKER, 1)[1].strip().rstrip("\"')")
            return True, reason
    return False, ""


def _imported_tensor_names(tree: ast.AST) -> set:
    """Bare names in this file that were imported FROM a tensor library."""
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "") in TENSOR_MODULES:
            for alias in node.names:
                name = alias.asname or alias.name
                if alias.name in BARE or name in BARE:
                    found.add(name)
    return found


def scan_file(path: Path) -> List[str]:
    try:
        source = path.read_text()
    except (OSError, UnicodeDecodeError):
        return []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    lines = source.splitlines()
    names = _imported_tensor_names(tree)
    problems: List[str] = _load_context_sites(path, tree, lines)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        spelling = ""
        if isinstance(node.func, ast.Attribute):
            dotted = _dotted_name(node.func)
            if dotted in DOTTED or (len(dotted) >= 2 and dotted[-2:] in DOTTED):
                spelling = ".".join(dotted)
            # NOT a suffix match on `.load`: `torch.export.load` reads an
            # ExportedProgram, not a checkpoint, and has its own fence
            # (`lint_raw_aoti_load.py`). A guard that claims a neighbouring
            # domain teaches people to ignore it.
        elif isinstance(node.func, ast.Name) and node.func.id in names:
            spelling = node.func.id
        if not spelling:
            continue
        proved, reason = _exempt(lines, node.lineno)
        if proved and reason:
            continue
        shown = _shown(path)
        if proved:
            problems.append(
                f"{shown}:{node.lineno}: `{spelling}` carries a bare "
                f"`{MARKER}` with NO reason. The marker is not the proof — "
                f"state why this path cannot hold a TFSSTUB1 stub."
            )
            continue
        problems.append(
            f"{shown}:{node.lineno}: `{spelling}` reads tensors from a FILE "
            f"PATH. On a projected tensorfs tree that path holds a ~128 B "
            f"TFSSTUB1 pointer stub and the real bytes are in the CAS: this "
            f"reads the stub's first 8 bytes as a header length and blames "
            f"the checkpoint. Use gen_worker.models.tensor_source "
            f"(`open_tensor_source` / `load_state_dict`), or prove the path "
            f"with `# {MARKER} <why>`."
        )
    return problems


def _shown(path: Path) -> Path:
    """The path as a finding spells it — repo-relative under a root
    `_lint_side._PATH_RE` recognises, so the finding is ATTRIBUTABLE.

    An endpoint file is spelled `src/serverless-endpoints/<endpoint>/...`: the
    `src/` prefix is what makes attribution work at all, and the repo name is
    what tells a reader which tree to open.
    """
    try:
        return path.relative_to(REPO)
    except ValueError:
        pass
    try:
        return Path("src") / "serverless-endpoints" / path.relative_to(ENDPOINTS)
    except ValueError:
        return Path("src") / path.name


#: Not source: installed environments, vendored trees, build output.
SKIP_DIRS = frozenset({".venv", "venv", "vendor", "node_modules",
                       "__pycache__", "build", "dist", ".git"})


def _load_context_sites(path: Path, tree: ast.AST, lines: Sequence[str]) -> List[str]:
    """pgw#1549: production code builds a ``LoadContext`` in ONE place.

    The outage was not caused by a bad load context. It was caused by there
    being TWO of them — `EndpointHost` and `ServeLoop` each assembling one out
    of their own copy of the worker's decisions — so a decision added to one
    silently missed the other, three times running (pgw#1452's placement
    device, pgw#1380's engine, pgw#1543's repair). Consolidating them fixes
    today; this arm is what stops a third host from re-creating the split,
    which is the only reason the consolidation is worth anything.

    Tests build them freely: a bare `LoadContext(...)` is inert by design
    (it names no device and binds no engine) and pgw#1452's arm 2 asserts
    exactly that, so the fence is on `src/` only.
    """
    if path in CONTEXT_BUILDERS or not str(path).startswith(str(SRC)):
        return []
    problems: List[str] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "LoadContext"):
            continue
        proved, reason = _exempt(lines, node.lineno)
        if proved and reason:
            continue
        problems.append(
            f"{_shown(path)}:{node.lineno}: builds a `LoadContext` directly. "
            f"Production has ONE (pgw#1549): "
            f"`gen_worker.serving.worker_context.worker_load_context`, which "
            f"carries the worker's placement device, io mode and weight "
            f"budget. A second builder is how pgw#1452's fix reached the local "
            f"CLI and never reached a pod — every eagerly bridged pipeline on "
            f"the fleet ran on the CPU because `ServeLoop` named no device."
        )
    return problems


def _python_files(root: Path) -> Iterable[Path]:
    """Every first-party ``.py`` under ``root``.

    The skip test is RELATIVE to ``root``, never over the absolute path: this
    repo is routinely checked out at `~/cozy/.worktrees/<repo>/<branch>`, and
    an absolute-path test there matched every file in the tree and scanned
    NOTHING — a fence reporting clean because it could not see. The empty-scan
    rail below caught it, which is exactly what that rail is for.
    """
    for path in sorted(root.rglob("*.py")):
        if SKIP_DIRS & set(path.relative_to(root).parts):
            continue
        yield path


def scan(roots: Sequence[Path]) -> Tuple[List[str], int]:
    problems: List[str] = []
    scanned = 0
    for root in roots:
        if not root.exists():
            continue
        for path in _python_files(root):
            if path in SANCTIONED:
                continue
            if root == SRC and is_unowned(path, SRC):
                continue
            if any(str(path).startswith(str(d)) for d in PRODUCER_DIRS):
                continue
            scanned += 1
            problems.extend(scan_file(path))
    return problems, scanned


def _endpoint_roots() -> List[Path]:
    if not ENDPOINTS.is_dir():
        return []
    return [d / "src" for d in sorted(ENDPOINTS.iterdir())
            if (d / "endpoint.toml").is_file() and (d / "src").is_dir()]


def _selftest() -> int:
    """Plant the violation and require the guard to SEE it. A fence whose red
    has never been observed is a fence nobody has tested."""
    failures: List[str] = []
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        (root / "bad.py").write_text(
            "from safetensors.torch import load_file\n"
            "def go(p):\n"
            "    return load_file(str(p))\n"
        )
        # The planted call is ASSEMBLED, never written literally: HARDCUT E5's
        # `lint_pickle_readers.py` scans this file too and would (correctly)
        # refuse a verbatim torch deserializer sitting in a guard's source.
        # Two fences over one repo must not make each other's fixtures illegal.
        deserializer = "torch." + "load"
        (root / "bad_dotted.py").write_text(
            "import torch\n"
            "def go(p):\n"
            f"    return {deserializer}(p, map_location='cpu')\n"
        )
        (root / "bare_marker.py").write_text(
            "from safetensors import safe_open\n"
            "def go(p):\n"
            "    return safe_open(p)  # tensor-seam-exempt:\n"
        )
        (root / "good_seam.py").write_text(
            "from gen_worker.models.tensor_source import load_state_dict\n"
            "def go(p):\n"
            "    return load_state_dict(p, why='the sanctioned reader')\n"
        )
        (root / "good_proved.py").write_text(
            "from safetensors import safe_open\n"
            "def go(p):\n"
            "    # tensor-seam-exempt: this path is written by this process\n"
            "    return safe_open(p)\n"
        )
        found, scanned = scan([root])
        if scanned != 5:
            failures.append(f"scanned {scanned} file(s), expected 5")
        named = " ".join(found)
        for wanted in ("bad.py:3", "bad_dotted.py:3", "bare_marker.py:3"):
            if wanted not in named:
                failures.append(f"MISSED the planted violation at {wanted}")
        for unwanted in ("good_seam.py", "good_proved.py"):
            if unwanted in named:
                failures.append(f"refused the legitimate form in {unwanted}")
        if "NO reason" not in named:
            failures.append("a bare marker was accepted as a proof")

    # The one-load-context arm fires only under `src/gen_worker`, so its
    # planted violation has to live there — the `test_guard_attribution.py`
    # doctrine (plant it where the guard actually looks, or the selftest
    # proves a code path production never takes).
    planted = SRC / f"_seam_selftest_{os.getpid()}.py"
    try:
        planted.write_text(
            "from .serving.context import LoadContext\n"
            "def go(b):\n"
            "    return LoadContext(binding=b)\n"
        )
        found, _ = scan([SRC])
        if not any(planted.name in line and "ONE" in line for line in found):
            failures.append(
                "MISSED a second LoadContext builder planted under src/")
    finally:
        planted.unlink(missing_ok=True)
    if failures:
        for line in failures:
            print(f"SELFTEST FAILED: {line}", file=sys.stderr)
        return 1
    print("lint_tensor_read_seam --selftest: the fence goes red on a planted "
          "violation and green on the seam")
    return 0


def main(argv: Sequence[str]) -> int:
    if "--selftest" in argv:
        return _selftest()
    roots = [Path(a).resolve() for a in argv if not a.startswith("-")]
    if not roots:
        roots = [SRC, *_endpoint_roots()]
        if not ENDPOINTS.is_dir():
            # NOT a silent pass. The endpoint half of this fence is the half
            # that found live violations; a run that could not look at it must
            # say so, or "green" means "did not check".
            print(
                f"lint_tensor_read_seam: NOTE — {ENDPOINTS} is not present, so "
                f"only gen_worker was scanned. The endpoint half of this fence "
                f"did not run.",
                file=sys.stderr,
            )
    problems, scanned = scan(roots)
    if not scanned:
        print(
            f"lint_tensor_read_seam: scanned NOTHING under "
            f"{', '.join(str(r) for r in roots)} — the tree moved and this "
            f"fence is asserting over an empty set.",
            file=sys.stderr,
        )
        return 1
    if problems:
        _lint_side.report(problems, "pgw#1550 tensor-read seam")
        print(
            f"\nlint_tensor_read_seam: {len(problems)} raw tensor-file "
            f"read(s) outside the seam ({scanned} file(s) scanned). The ~21 h "
            f"fleet outage of 2026-08-19 was one of these.",
            file=sys.stderr,
        )
        return 1
    print(f"lint_tensor_read_seam: clean ({scanned} file(s) scanned)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
