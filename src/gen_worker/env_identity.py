"""ONE definition of "what environment is this", for every party (pgw#1472).

A release document's `closure` is the env half of the compile identity. The
publish-time derive stamps it, the mint child folds it into the ck1 key
alongside `env_seal`, and a serving pod must RESTATE it or adoption refuses.
pgw#1367's trace-once-at-publish architecture makes those three DIFFERENT
processes by design, so the value has exactly one job: **be restatable to the
same string by any of them.** Two spellings of it fragment the whole
[release x sm] serving table.

**Ruled: the closure is over the INSTALLED SET AS OBSERVED BY A RUNNING
PROCESS**, computed by :func:`env_closure` / :func:`env_closure_hash` and by
nothing else. It is the only definition every party CAN restate: the derive
runs inside the release image, the mint child runs inside it, the serving pod
runs inside it. A lockfile closure is restatable by no running process at all —
which is the defect this module exists to have ended.

This reverses the 2026-08-19 coordinator ruling (*"lockfile at both ends"*).
Three measured facts decided it:

1. **A lockfile is not readable as a closure at all in the general case.** pgw's
   own `uv.lock` resolves `torch` to BOTH `2.13.0` and `2.13.0+cu130` — uv forks
   the resolution per index marker. There is no single "the lock's package set".
2. **`env_seal` — the OTHER axis of the same compile key — is already
   installed-observed** (it digests the torch/triton tree on disk). Making
   `closure` a lockfile hash would leave one key folding two contradictory
   answers to "which environment".
3. **Production already does it.** `derive_runner.go` passes no `--lockfile`, so
   the fleet stamps `installed_closure()` inside the release image and the pod
   restates it from that same image. The migration cost of the alternative was
   moving a currently-green fleet; the cost of this one is zero.

A lockfile is still a useful DIFFERENT thing, and it is spelled differently
here on purpose (the th#2137 lesson: never two names for one authority):
:func:`lockfile_packages` reads it, :func:`closure_drift` compares it against
the installed set as a **typed drift signal that is never a gate**, normalized
on both sides so it reports real divergence instead of PEP 503 spelling and PEP
440 local segments. Nothing in this module calls a lockfile "env identity".
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

#: What `uv` writes beside a project. Named once so every reader of a lockfile
#: — all of them diagnostics — spells the filename the same way.
LOCKFILE_NAME = "uv.lock"

_SEPARATORS = re.compile(r"[-_.]+")


class EnvIdentityError(ValueError):
    """The installed set cannot be observed, or a lockfile cannot be read."""


# --- THE definition --------------------------------------------------------


def env_closure() -> dict[str, str]:
    """`{name: version}` of the distributions THIS process can import.

    The one measurement of "which environment am I". Called by the derive that
    stamps a document, by the boot that restates it, and by the reuse key that
    decides whether a saved trace still describes this env — so that all three
    are one function and cannot drift apart.
    """

    from ._vendor.torchcg.graph_identity import GraphIdentityError, installed_closure

    try:
        return installed_closure()
    except GraphIdentityError as exc:
        raise EnvIdentityError(f"cannot observe this process's env: {exc}") from exc


def env_closure_hash() -> str:
    """The 64-hex env closure of this process — the stamped/restated value.

    `closure_hash` canonicalizes names the way package indexes compare them, so
    this is stable across the `PyYAML`/`pyyaml` spelling that `importlib` and
    `uv` disagree on.
    """

    from ._vendor.torchcg.graph_identity import GraphIdentityError, closure_hash

    try:
        return closure_hash(env_closure())
    except GraphIdentityError as exc:
        raise EnvIdentityError(f"cannot state this process's env identity: {exc}") from exc


# --- lockfiles: a different thing, deliberately named differently ----------


def normalize_name(name: str) -> str:
    """The PEP 503 normalized distribution name.

    `PyYAML`, `pyyaml` and `Py.YAML` are ONE distribution. Comparing them raw
    is not strictness, it is a bug with no upside — it makes a drift report
    fire on ten packages that are identical.
    """

    return _SEPARATORS.sub("-", str(name)).lower()


def normalize_version(version: str) -> str:
    """The version without its PEP 440 LOCAL segment (`2.13.0+cu129` ->
    `2.13.0`).

    Stripping it is a real loss and it is stated rather than hidden: `+cu129`
    against `+cu130` is exactly the kind of difference a compiled artifact
    cares about. But a lockfile CANNOT express it — uv records the version and
    the wheel URL separately — so a comparison that keeps it can only ever
    report a difference that is not one. Identity does not go through here: it
    is `env_closure_hash`, which keeps the local segment because an installed
    distribution can state it.
    """

    return str(version).split("+", 1)[0]


def lockfile_packages(lockfile: Path | str) -> dict[str, str]:
    """`{name: version}` for every package a `uv.lock` resolves — a DIAGNOSTIC.

    Explicitly NOT an env identity and never hashed into one. A lock enumerates
    the resolution for every marker, so on any real multi-index project it does
    not even name one set: pgw's own lock resolves `torch` to `2.13.0` under one
    marker and `2.13.0+cu130` under another. That fork is reported here, per
    package, instead of being collapsed into a number nobody could restate.
    """

    path = Path(lockfile)
    try:
        parsed = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise EnvIdentityError(f"cannot read lockfile {path}: {exc}") from exc
    entries: dict[str, str] = {}
    forked: dict[str, set[str]] = {}
    for package in parsed.get("package", ()):
        name = package.get("name")
        version = package.get("version")
        if not isinstance(name, str) or not isinstance(version, str):
            continue
        known = entries.get(name)
        if known is not None and known != version:
            forked.setdefault(name, {known}).add(version)
        entries[name] = version
    if not entries:
        raise EnvIdentityError(f"lockfile {path} states no resolved packages")
    for name, versions in forked.items():
        # Recorded, not raised: this reader feeds a drift report, and a report
        # that dies on the very property that killed lockfile-as-identity is
        # useless exactly where it is most informative.
        entries[name] = "/".join(sorted(versions))
    return entries


def lockfile_beside(endpoint_dir: Path | str) -> Path | None:
    """The endpoint's own `uv.lock`, or `None` — never a guess elsewhere."""

    candidate = Path(endpoint_dir) / LOCKFILE_NAME
    return candidate if candidate.is_file() else None


@dataclass(frozen=True, slots=True)
class DriftRow:
    """One package on which the installed set and a lockfile differ."""

    name: str
    #: ``missing`` (locked, not installed), ``extra`` (installed, not locked),
    #: ``version`` (both, different versions). A closed vocabulary.
    kind: str
    stated: str = ""
    installed: str = ""

    def __str__(self) -> str:  # pragma: no cover - diagnostics
        if self.kind == "version":
            return f"{self.name} {self.stated} != {self.installed}"
        if self.kind == "missing":
            return f"{self.name} {self.stated} stated, not installed"
        return f"{self.name} {self.installed} installed, not stated"


def closure_drift(
    installed: Mapping[str, str], stated: Mapping[str, str]
) -> tuple[DriftRow, ...]:
    """How far the running env is from what a lockfile resolves. NEVER a gate.

    NORMALIZED on both sides, so every row is a real divergence rather than a
    spelling. Sorted by name so two runs of the same pair produce the same
    report. Adoption does not consult this: identity is `env_closure_hash` and
    this is the signal that tells an operator the image drifted from its lock.
    """

    live = {normalize_name(n): normalize_version(v) for n, v in installed.items()}
    want = {normalize_name(n): normalize_version(v) for n, v in stated.items()}
    rows: list[DriftRow] = []
    for name in sorted(set(want) - set(live)):
        rows.append(DriftRow(name, "missing", stated=want[name]))
    for name in sorted(set(live) - set(want)):
        rows.append(DriftRow(name, "extra", installed=live[name]))
    for name in sorted(set(live) & set(want)):
        if live[name] != want[name]:
            rows.append(
                DriftRow(name, "version", stated=want[name], installed=live[name])
            )
    return tuple(rows)


def describe_drift(rows: tuple[DriftRow, ...], *, limit: int = 8) -> str:
    """A one-line drift summary. ``""`` when the sets agree.

    Deliberately says the COUNT before the examples: a report that shows six
    rows out of sixty reads as "six problems".
    """

    if not rows:
        return ""
    counts = {kind: sum(1 for r in rows if r.kind == kind)
              for kind in ("missing", "extra", "version")}
    head = ", ".join(f"{n} {k}" for k, n in counts.items() if n)
    shown = "; ".join(str(row) for row in rows[:limit])
    more = f" (+{len(rows) - limit} more)" if len(rows) > limit else ""
    return f"{len(rows)} package(s) differ ({head}): {shown}{more}"


def describe_lockfile_drift(lockfile: Path | str) -> str:
    """The drift line a boot or a lock run prints. Never raises, never gates."""

    try:
        stated = lockfile_packages(lockfile)
    except EnvIdentityError as exc:
        return f"lockfile drift unavailable: {exc}"
    try:
        installed = env_closure()
    except EnvIdentityError as exc:  # pragma: no cover - a dead env
        return f"lockfile drift unavailable: {exc}"
    summary = describe_drift(closure_drift(installed, stated))
    return (
        f"installed-vs-{Path(lockfile)} ({len(stated)} locked, "
        f"{len(installed)} installed): {summary or 'none'}"
    )


__all__ = [
    "LOCKFILE_NAME",
    "DriftRow",
    "EnvIdentityError",
    "closure_drift",
    "describe_drift",
    "describe_lockfile_drift",
    "env_closure",
    "env_closure_hash",
    "lockfile_beside",
    "lockfile_packages",
    "normalize_name",
    "normalize_version",
]
