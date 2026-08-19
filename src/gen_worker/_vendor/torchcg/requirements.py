"""Requirements manifests and the miss-path ladder (pgw#1368).

The manifest is written by the MINT from what it actually linked -- an
include-set of libraries with versions, the compiled sm, the CUDA driver
floor, and autotune provenance. Its role is AUDIT, not matching: the normal
path adopts only its own release's exact env (the image pin collapses the
compatibility question), and the manifest is what lets a worker refuse LOUDLY
when pod env != release metadata -- a build-system bug surfacing, never a
compat decision.

Ranking exists only for the miss-path fallback ladder and is kept minimal by
ruling: native sm before binary-compatible sm (same major, host minor >=
compiled minor), native autotune before heuristic, deterministic tie-break on
artifact digest. The normal path never ranks -- exact env, one answer.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .graph_identity import EnvIdentity, GraphIdentityError, is_graph_hash

_MANIFEST_FIELDS = frozenset(
    ("v", "include_set", "sm_compiled", "cuda_floor", "autotuned_on")
)
# CUDA convention: the final digit is the minor version (sm_89 = 8.9,
# sm_120 = 12.0), so the major greedily takes the rest.
_SM_NUMERIC_RE = re.compile(r"sm_([0-9]{1,2})([0-9])\Z")


class EnvironmentMismatch(RuntimeError):
    """The running environment is not the environment the metadata names.

    On the normal path this firing means the build system produced a pod
    whose env differs from its release stamp. It is a refusal to serve a
    wrong answer quietly, so the message names every divergence.
    """


class RequirementsError(ValueError):
    """A requirements manifest is incomplete or noncanonical."""


def _canonical_pairs(
    entries: Sequence[tuple[str, str]] | Mapping[str, str],
) -> tuple[tuple[str, str], ...]:
    pairs = entries.items() if isinstance(entries, Mapping) else entries
    clean: dict[str, str] = {}
    for name, version in pairs:
        if not isinstance(name, str) or not name.strip():
            raise RequirementsError("include-set names must be non-empty strings")
        if not isinstance(version, str) or not version.strip():
            raise RequirementsError(f"include-set entry {name!r} states no version")
        canonical = re.sub(r"[-_.]+", "-", name.strip().lower())
        known = clean.get(canonical)
        if known is not None and known != version.strip():
            raise RequirementsError(
                f"include-set entry {canonical!r} appears twice with different versions"
            )
        clean[canonical] = version.strip()
    return tuple(sorted(clean.items()))


def _sm_parts(sm: str) -> tuple[int, int] | None:
    match = _SM_NUMERIC_RE.fullmatch(sm)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


@dataclass(frozen=True, slots=True)
class RequirementsManifest:
    """What one mint ACTUALLY linked, stated by the minter, never inferred."""

    include_set: tuple[tuple[str, str], ...]
    sm_compiled: str
    cuda_floor: str | None = None
    autotuned_on: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "include_set", _canonical_pairs(self.include_set))
        if not self.include_set:
            raise RequirementsError(
                "a mint links SOMETHING; an empty include-set is a recording bug"
            )
        if not isinstance(self.sm_compiled, str) or not self.sm_compiled.strip():
            raise RequirementsError("manifest must state the compiled sm")
        for field in ("cuda_floor", "autotuned_on"):
            value = getattr(self, field)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise RequirementsError(f"manifest {field} must be None or non-empty")

    def as_dict(self) -> dict[str, Any]:
        return {
            "v": 1,
            "include_set": [[name, version] for name, version in self.include_set],
            "sm_compiled": self.sm_compiled,
            "cuda_floor": self.cuda_floor,
            "autotuned_on": self.autotuned_on,
        }

    def encode(self) -> bytes:
        return json.dumps(
            self.as_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")

    @classmethod
    def decode(cls, raw: Mapping[str, Any]) -> RequirementsManifest:
        if not isinstance(raw, Mapping) or set(raw) != _MANIFEST_FIELDS:
            raise RequirementsError(
                f"manifest fields must be exactly {sorted(_MANIFEST_FIELDS)!r}"
            )
        if raw.get("v") != 1:
            raise RequirementsError("manifest v must be 1")
        rows = raw.get("include_set")
        if not isinstance(rows, list) or any(
            not isinstance(row, list) or len(row) != 2 for row in rows
        ):
            raise RequirementsError("manifest include_set must be [name, version] rows")
        return cls(
            include_set=tuple((row[0], row[1]) for row in rows),
            sm_compiled=raw["sm_compiled"],
            cuda_floor=raw["cuda_floor"],
            autotuned_on=raw["autotuned_on"],
        )

    def assert_environment(self, installed: Mapping[str, str], *, sm: str) -> None:
        """The audit assertion: every linked library is present, exactly.

        EXACT versions, deliberately: a lib bump only arrives via a new image
        = new release = fresh derive+mint, so any divergence here is the
        build system contradicting itself, and the right behavior is a loud
        typed refusal naming everything that diverged.
        """

        present = dict(_canonical_pairs(installed))
        divergences: list[str] = []
        for name, version in self.include_set:
            found = present.get(name)
            if found is None:
                divergences.append(f"{name}: linked {version}, not installed")
            elif found != version:
                divergences.append(f"{name}: linked {version}, installed {found}")
        if not _sm_runnable(self.sm_compiled, sm):
            divergences.append(f"sm: compiled {self.sm_compiled}, running {sm}")
        if divergences:
            raise EnvironmentMismatch(
                "pod environment does not match the artifact's requirements "
                "manifest (build-system bug surfacing): " + "; ".join(divergences)
            )


def assert_exact_env(
    stamped: EnvIdentity, *, stack: Mapping[str, str] | Sequence[tuple[str, str]], sm: str
) -> None:
    """Boot-time audit that this pod's env == the artifact env (pgw#1367).

    ``stack`` is THIS process's compile stack, read from the endpoint's own
    ``uv.lock`` -- the same file and the same rows the derive stamped, so the
    two sides are one representation and a difference is a real difference
    (pgw#1489 killed the installed-set restatement that could never agree).

    Every divergence is named per package: ``torch 2.13.0 != 2.14.0`` is
    actionable where two differing digests were not.
    """

    try:
        observed = EnvIdentity(stack=stack, sm=sm)
    except GraphIdentityError as exc:
        raise EnvironmentMismatch(f"cannot state this pod's env identity: {exc}") from exc
    if observed == stamped:
        return
    mine = dict(observed.stack)
    theirs = dict(stamped.stack)
    details: list[str] = []
    for name in sorted(set(mine) | set(theirs)):
        if name not in theirs:
            details.append(f"{name} {mine[name]} here, absent from the artifact env")
        elif name not in mine:
            details.append(f"{name} {theirs[name]} stamped, absent here")
        elif mine[name] != theirs[name]:
            details.append(f"{name} {mine[name]} != stamped {theirs[name]}")
    if observed.sm != stamped.sm:
        details.append(f"sm {observed.sm} != stamped {stamped.sm}")
    raise EnvironmentMismatch(
        "this env is not the compile stack the artifacts were derived under "
        "(a rebuild is the fix, never a cast): " + "; ".join(details)
    )


@dataclass(frozen=True, slots=True)
class ArtifactCandidate:
    """One published artifact row a miss-path ladder may consider."""

    graph: str
    env: EnvIdentity
    digest: str
    manifest: RequirementsManifest

    def __post_init__(self) -> None:
        if not is_graph_hash(self.graph):
            raise RequirementsError("candidate graph must be a cg-graph-v1 hash")
        if not isinstance(self.digest, str) or not self.digest.strip():
            raise RequirementsError("candidate must carry its artifact digest")


def _sm_runnable(compiled: str, host: str) -> bool:
    if compiled == host:
        return True
    compiled_parts = _sm_parts(compiled)
    host_parts = _sm_parts(host)
    if compiled_parts is None or host_parts is None:
        return False
    return compiled_parts[0] == host_parts[0] and host_parts[1] >= compiled_parts[1]


def rank(
    candidates: Sequence[ArtifactCandidate], *, sm: str
) -> tuple[ArtifactCandidate, ...]:
    """Order miss-path candidates; the normal path never calls this.

    Runnable set: same sm, or same major with host minor >= compiled minor.
    Order: native sm, then native autotune, then higher compiled minor
    (closer to native), then artifact digest for a deterministic tail.
    """

    runnable = [row for row in candidates if _sm_runnable(row.manifest.sm_compiled, sm)]

    def order(row: ArtifactCandidate) -> tuple[int, int, int, str]:
        parts = _sm_parts(row.manifest.sm_compiled)
        return (
            0 if row.manifest.sm_compiled == sm else 1,
            0 if row.manifest.autotuned_on == sm else 1,
            -(parts[1] if parts is not None else 0),
            row.digest,
        )

    return tuple(sorted(runnable, key=order))


__all__ = [
    "ArtifactCandidate",
    "EnvironmentMismatch",
    "RequirementsError",
    "RequirementsManifest",
    "assert_exact_env",
    "rank",
]
