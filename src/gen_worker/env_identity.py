"""ONE definition of "what environment is this", for both ends (pgw#1472).

A release document's `closure` is a hash of a package set, and the serving side
must be able to RESTATE it or adoption refuses. So the two ends have to name
the same set with the same spelling — and until this module existed they did
not:

* `gen-worker lock` stamps the endpoint's `uv.lock` (`release/derive.py`);
* a serving process restates `importlib.metadata` (`installed_closure()`).

Those are incomparable, and not by a little. Measured against
`serverless-endpoints/sd15/uv.lock` and its own venv, THREE independent reasons
they can never be equal:

1. **Name spelling.** uv normalizes to PEP 503; `importlib.metadata` reports the
   DECLARED name. Ten packages on that one endpoint agree only after
   normalization: `pyyaml`/`PyYAML`, `huggingface-hub`/`huggingface_hub`,
   `typing-extensions`/`typing_extensions`, and seven more.
2. **The local version segment.** A lock says `torch = "2.13.0"`; every
   installed CUDA wheel says `2.13.0+cu129`. A lock cannot state it and an
   installed distribution always has one.
3. **Platform-conditional entries.** `colorama` is in that lock and will never
   install on Linux — a lock enumerates the resolution for every marker, an
   install enumerates one platform.

**Ruled (2026-08-19): the LOCKFILE is the identity at both ends.** The
weights-free derive can only ever state the lock, and the lock is the one
artifact mint and serve pods share by construction. This module is that single
source, so a second copy cannot drift from it.

**And the comparison installed-vs-lock is a DRIFT SIGNAL, never a gate.** It is
normalized on both sides precisely so it reports real divergence instead of
spelling — a signal that fires on `PyYAML` vs `pyyaml` is measuring itself.
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

#: What `uv` writes beside a project, and therefore what `gen-worker lock`
#: hashed. Named once so the CLI and the serve runner cannot disagree.
LOCKFILE_NAME = "uv.lock"

_SEPARATORS = re.compile(r"[-_.]+")


class EnvIdentityError(ValueError):
    """A lockfile cannot be read, or states no resolved packages."""


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
    report a difference that is not one. The local segment is recoverable from
    the artifact's own requirements manifest, which is where the compiled side
    checks it.
    """

    return str(version).split("+", 1)[0]


def closure_entries_from_lockfile(lockfile: Path | str) -> dict[str, str]:
    """`{name: version}` for every package a `uv.lock` resolves.

    Verbatim names and versions: this is the hashed identity, so normalization
    must NOT happen here — it would change every closure hash ever stamped.
    Normalization belongs to the DRIFT comparison, which is not an identity.
    """

    path = Path(lockfile)
    try:
        parsed = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise EnvIdentityError(f"cannot read lockfile {path}: {exc}") from exc
    entries: dict[str, str] = {}
    for package in parsed.get("package", ()):
        name = package.get("name")
        version = package.get("version")
        if not isinstance(name, str) or not isinstance(version, str):
            continue
        known = entries.get(name)
        if known is not None and known != version:
            raise EnvIdentityError(
                f"lockfile {path} resolves {name!r} to both {known!r} and "
                f"{version!r}; a release env has ONE resolved closure"
            )
        entries[name] = version
    if not entries:
        raise EnvIdentityError(f"lockfile {path} states no resolved packages")
    return entries


def lockfile_beside(endpoint_dir: Path | str) -> Path | None:
    """The endpoint's own `uv.lock`, or `None` — never a guess elsewhere.

    This is the SAME file `gen-worker lock` hashes for the document it writes,
    which is the whole point: a locally derived document and a local serve now
    read one file rather than two sources.
    """

    candidate = Path(endpoint_dir) / LOCKFILE_NAME
    return candidate if candidate.is_file() else None


@dataclass(frozen=True, slots=True)
class DriftRow:
    """One package on which the installed set and the stated closure differ."""

    name: str
    #: ``missing`` (stated, not installed), ``extra`` (installed, not stated),
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
    """How far the running env is from the one the release states.

    NORMALIZED on both sides, so every row is a real divergence. Sorted by name
    so two runs of the same pair produce the same report.
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


__all__ = [
    "LOCKFILE_NAME",
    "DriftRow",
    "EnvIdentityError",
    "closure_drift",
    "closure_entries_from_lockfile",
    "describe_drift",
    "lockfile_beside",
    "normalize_name",
    "normalize_version",
]
