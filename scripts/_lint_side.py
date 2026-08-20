"""pgw#1521: which SIDE a guard finding falls on — THIS DIFF, or the base.

`fast gates` is this repo's only required context and it bundles ~25 unrelated
guards, so at the merge button these two are indistinguishable:

* *"`fast gates` is red because of somebody else's problem on master"* — the
  legitimate reason to reach for `--admin`; and
* *"`fast gates` is red because a guard is refusing MY OWN new file"* — which is
  the guard doing the single job it exists for.

A bypass granted for the first silently includes the second. **Nothing new needs
computing to tell them apart**: every one of these guards already names the file
it is refusing, and the set of files this diff touches is known to CI. This
module joins the two and prints the answer on the failing line itself.

ONE HOME for that fact, the `_lint_scope.py` shape: a guard that resolves its own
base makes a change of CI trigger a hunt through every scanner.

WHERE THE DIFF COMES FROM, in order:

1. ``$PGW1521_DIFF_FILES`` — a file this module wrote earlier in the job
   (``--emit``). CI resolves the base ONCE, before the guards run, because on a
   shallow PR checkout resolving it costs a fetch.
2. Otherwise, locally: ``git diff --name-only <merge-base origin/master HEAD>``
   plus uncommitted and untracked files, because a guard refusing a file you have
   not committed yet is very definitely refusing YOURS.
3. Otherwise **UNKNOWN**, which is printed as UNKNOWN and never as "not yours".
   An unattributed finding is the state this module exists to end, not a pass.

WHICH WAY IT ROUNDS. A finding naming several files is attributed to the diff if
ANY of them is in it. The two errors are not symmetric: calling master's red
YOURS costs a minute of reading, calling YOUR red master's buys an `--admin` past
your own violation. So ambiguity resolves toward YOURS, always.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Set, Tuple

REPO = Path(__file__).resolve().parents[1]

#: Set by the `fast gates` emit step to the file `--emit` wrote.
DIFF_FILES_ENV = "PGW1521_DIFF_FILES"

#: Written as the first line of that file. `UNKNOWN` means the base could not be
#: resolved — distinct from a resolved base with an empty diff.
_BASE_PREFIX = "# base "
_UNKNOWN = "UNKNOWN"

#: Repo-relative paths as the guards spell them in their findings. Anchored on a
#: known top-level directory (or one of the two root files the guards name) so
#: prose that mentions `config.py` is not mistaken for a path — an unanchored
#: match would round toward PRE-EXISTING, which is the unsafe direction.
_PATH_RE = re.compile(
    r"\b((?:(?:src|tests|tests_v2|scripts|examples|docs|proto|benchmarks|\.github)"
    r"/[A-Za-z0-9_./+-]*[A-Za-z0-9_]|uv\.lock|pyproject\.toml))"
)

YOURS = "YOUR DIFF"
BASE = "PRE-EXISTING"
UNKNOWN = "SIDE UNKNOWN"

_cache: Optional[Tuple[Optional[Set[str]], str]] = None


# ---------------------------------------------------------------------------
# resolving the diff
# ---------------------------------------------------------------------------

def _git(*args: str, cwd: Path = REPO) -> Optional[str]:
    try:
        out = subprocess.run(
            ("git", *args), cwd=cwd, capture_output=True, text=True, timeout=120
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout if out.returncode == 0 else None


def _event_base() -> Optional[str]:
    """The base sha GitHub already put in the event payload, if any."""
    event = os.environ.get("GITHUB_EVENT_NAME", "")
    path = os.environ.get("GITHUB_EVENT_PATH", "")
    if not path or not Path(path).is_file():
        return None
    try:
        payload = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None
    if event == "pull_request":
        return (payload.get("pull_request") or {}).get("base", {}).get("sha")
    if event == "merge_group":
        return (payload.get("merge_group") or {}).get("base_sha")
    return None


def _have(sha: str) -> bool:
    return _git("cat-file", "-e", f"{sha}^{{commit}}") is not None


def resolve_base() -> Optional[str]:
    """The commit this diff is measured against, or None when unresolvable."""
    sha = _event_base()
    if sha:
        if not _have(sha):
            # A PR checkout is shallow at the merge commit; the base is one fetch
            # away and the diff needs only its tree.
            _git("fetch", "--no-tags", "--depth=1", "origin", sha)
        return sha if _have(sha) else None
    for ref in ("origin/master", "master"):
        if _git("rev-parse", "--verify", "--quiet", ref):
            merge_base = _git("merge-base", ref, "HEAD")
            if merge_base and merge_base.strip():
                return merge_base.strip()
    return None


def _diff_against(base: str) -> Set[str]:
    """Every path this working tree changes relative to `base`."""
    paths: Set[str] = set()
    for out in (
        _git("diff", "--name-only", base, "HEAD"),
        _git("diff", "--name-only", "HEAD"),
        _git("ls-files", "--others", "--exclude-standard"),
    ):
        for line in (out or "").splitlines():
            if line.strip():
                paths.add(line.strip())
    return paths


def _read_emitted(path: Path) -> Tuple[Optional[Set[str]], str]:
    try:
        lines = path.read_text().splitlines()
    except OSError as exc:
        return None, f"cannot read {DIFF_FILES_ENV}={path} ({exc})"
    base = _UNKNOWN
    paths: Set[str] = set()
    for line in lines:
        if line.startswith(_BASE_PREFIX):
            base = line[len(_BASE_PREFIX):].strip() or _UNKNOWN
        elif line.strip() and not line.startswith("#"):
            paths.add(line.strip())
    if base == _UNKNOWN:
        return None, "CI could not resolve this run's base commit"
    return paths, base


def diff_files() -> Tuple[Optional[Set[str]], str]:
    """``(paths, base)``; `paths` is None when the side cannot be known.

    `base` is a short description for the rollup line — a sha, or why not.
    """
    global _cache
    if _cache is None:
        emitted = os.environ.get(DIFF_FILES_ENV)
        if emitted:
            _cache = _read_emitted(Path(emitted))
        else:
            base = resolve_base()
            if base is None:
                _cache = (None, "no base commit resolvable from here")
            else:
                _cache = (_diff_against(base), base[:12])
    return _cache


def reset_cache() -> None:
    """For the selftest; the diff does not move inside one guard run."""
    global _cache
    _cache = None


# ---------------------------------------------------------------------------
# attributing findings
# ---------------------------------------------------------------------------

def paths_in(text: str) -> List[str]:
    """The repo-relative paths a finding names."""
    return [m.rstrip(".,:;") for m in _PATH_RE.findall(text)]


def side_of(text: str) -> str:
    """`YOURS` / `BASE` / `UNKNOWN` for one finding."""
    paths, _ = diff_files()
    if paths is None:
        return UNKNOWN
    named = paths_in(text)
    if not named:
        return UNKNOWN
    return YOURS if any(p in paths for p in named) else BASE


def annotate(problems: Iterable[str]) -> List[str]:
    """Each finding prefixed with the side it falls on."""
    return [f"[{side_of(p)}] {p}" for p in problems]


def verdict(problems: Sequence[str], guard: str) -> str:
    """The one line the merge button needs."""
    sides = [side_of(p) for p in problems]
    _, base = diff_files()
    yours = sides.count(YOURS)
    theirs = sides.count(BASE)
    unknown = sides.count(UNKNOWN)
    head = (
        f"pgw#1521 attribution — {guard}: {len(problems)} finding(s), "
        f"{yours} in files THIS DIFF touches, {theirs} pre-existing on the base"
    )
    if unknown:
        head += f", {unknown} unattributed"
    if yours:
        return (
            f"{head} (base {base}). "
            f"AT LEAST ONE OF THESE IS YOURS — an `--admin` here bypasses your "
            f"own violation, not somebody else's master-red."
        )
    if unknown or theirs == 0:
        return (
            f"{head} (base {base}). "
            f"UNKNOWN is not 'not yours' — attribute before bypassing."
        )
    return (
        f"{head} (base {base}). "
        f"None of this is in your diff: a TARGETED `--admin` is defensible, "
        f"and it does not fix the base — say both on the PR."
    )


def report(problems: Sequence[str], guard: str, stream=sys.stderr) -> None:
    """Print findings with their side, then the verdict. Callers exit 1."""
    for line in annotate(problems):
        print(line, file=stream)
    print(verdict(problems, guard), file=stream)


# ---------------------------------------------------------------------------
# emit — CI resolves the base once, ahead of the guards
# ---------------------------------------------------------------------------

def emit(target: Path) -> int:
    base = resolve_base()
    if base is None:
        target.write_text(f"{_BASE_PREFIX}{_UNKNOWN}\n")
        print(
            "pgw#1521: no base commit resolvable — guard findings will print "
            "as unattributed rather than as somebody else's problem",
            file=sys.stderr,
        )
        return 0
    paths = sorted(_diff_against(base))
    target.write_text(f"{_BASE_PREFIX}{base}\n" + "".join(f"{p}\n" for p in paths))
    print(f"pgw#1521: {len(paths)} file(s) in this diff against {base[:12]}")
    return 0


# ---------------------------------------------------------------------------
# selftest — a mechanism that has never been seen work is not known to work
# ---------------------------------------------------------------------------

def _selftest() -> int:
    import tempfile

    failures = 0

    def expect(label: str, got: object, want: object) -> None:
        nonlocal failures
        if got != want:
            failures += 1
            print(f"SELFTEST FAIL — {label}: got {got!r}, want {want!r}",
                  file=sys.stderr)

    expect("a path is found in a finding",
           paths_in("tests/test_a_pgw1.py: a NEW test module"),
           ["tests/test_a_pgw1.py"])
    expect("a line:col suffix does not swallow the path",
           paths_in("src/gen_worker/cli/compile.py:294: reads config"),
           ["src/gen_worker/cli/compile.py"])
    expect("several paths in one finding are all found",
           paths_in("src/gen_worker/a.py and tests/b.py disagree"),
           ["src/gen_worker/a.py", "tests/b.py"])
    expect("prose that names no file yields nothing",
           paths_in("the baseline lists 124 modules; the mark is 120"), [])

    with tempfile.TemporaryDirectory() as raw:
        emitted = Path(raw) / "diff.txt"

        # a resolved base with the file in it -> YOURS
        emitted.write_text("# base deadbeef\ntests/test_a_pgw1.py\n")
        os.environ[DIFF_FILES_ENV] = str(emitted)
        reset_cache()
        expect("a file in the diff is YOURS",
               side_of("tests/test_a_pgw1.py: refused"), YOURS)
        expect("a file NOT in the diff is the base's",
               side_of("tests/test_b_pgw2.py: refused"), BASE)
        expect("ANY named path in the diff makes the finding yours",
               side_of("tests/test_b_pgw2.py mismatches tests/test_a_pgw1.py"),
               YOURS)
        expect("a finding naming no file is unattributed",
               side_of("the high-water mark is wrong"), UNKNOWN)
        v = verdict(["tests/test_a_pgw1.py: refused"], "g")
        expect("a finding in the diff says so in the verdict",
               "AT LEAST ONE OF THESE IS YOURS" in v, True)
        v = verdict(["tests/test_b_pgw2.py: refused"], "g")
        expect("a base-only red offers the targeted bypass",
               "TARGETED" in v, True)

        # a resolved base with an EMPTY diff is not the same as UNKNOWN
        emitted.write_text("# base deadbeef\n")
        reset_cache()
        expect("an empty diff still attributes to the base",
               side_of("tests/test_a_pgw1.py: refused"), BASE)

        # an unresolvable base is UNKNOWN, and never reads as 'not yours'
        emitted.write_text(f"{_BASE_PREFIX}{_UNKNOWN}\n")
        reset_cache()
        expect("an unresolved base is UNKNOWN, not BASE",
               side_of("tests/test_a_pgw1.py: refused"), UNKNOWN)
        expect("and the verdict refuses to call it somebody else's",
               "UNKNOWN is not 'not yours'" in
               verdict(["tests/test_a_pgw1.py: refused"], "g"), True)

        del os.environ[DIFF_FILES_ENV]
        reset_cache()

    if failures:
        print(f"{failures} selftest case(s) failed", file=sys.stderr)
        return 1
    print("_lint_side selftest OK")
    return 0


def main(argv: List[str]) -> int:
    if "--selftest" in argv:
        return _selftest()
    if "--emit" in argv:
        return emit(Path(argv[argv.index("--emit") + 1]))
    paths, base = diff_files()
    if paths is None:
        print(f"pgw#1521: side UNKNOWN — {base}", file=sys.stderr)
        return 0
    print(f"pgw#1521: {len(paths)} file(s) in this diff against {base}")
    for path in sorted(paths):
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
