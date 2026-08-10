"""What the SYNTHESIZED Dockerfile does, enumerated — and asked of custom ones.

Tensorhub builds an endpoint image one of two ways. Dockerfile-LESS families get
a Dockerfile the hub SYNTHESIZES, and every platform invariant is established
there by writing a layer. A family that ships its own Dockerfile gets none of
those layers: its file is author-owned content and the platform never injects
into it (pgw#1017's settled contract). So on that path the platform can only
VERIFY — and each invariant it forgets to verify is a guarantee that quietly
holds on one path and not the other.

That gap produced four filed instances in this repo, all the same shape:

* pgw#1016 — the docs taught a BuildKit cache mount the publish validator
  refuses, because no first-party family exercised the doc.
* pgw#1017 — the AOT host toolchain (``g++``) was installed by the synthesized
  file only, so a custom-Dockerfile family could never AOT-mint.
* pgw#1017 GAP A — the composed ``/usr/local/cuda`` root: worse than the first
  two, because nothing refused ANYWHERE. The pod booted, loaded, exported — the
  expensive part — and died at ``CUDA_HOME``.
* pgw#1068 — a family shipping its own Dockerfile missed the ``cuda_root`` step
  live, and was fixed one-off. The fleet was then swept BY HAND, eight
  Dockerfiles at a time.

Four one-off patches and a manual sweep are not a mechanism. This module is the
mechanism: ONE enumerated registry of the steps a hand-written Dockerfile owes,
each row naming where the platform itself refuses if the author skips it, and a
checker any repo can run over its own endpoints::

    python -m gen_worker.build_guarantees path/to/endpoint [more...]

The SDK owns the registry's correctness; the author owns invoking the check —
the same division as ``python -m gen_worker.cuda_root``, and for the same
reason: three spellings of one platform requirement drift the moment a base
image changes.

**Scope, stated rather than implied.** This is a STATIC pre-flight over a source
tree. It is not the authority — ``aot_preconditions`` (in-image, at build) and
tensorhub's post-build image inspection are, and every row here names which one.
Its value is where it runs: in CI, on a diff, before a build exists, for $0.00.
The audit's own standing instruction is what it encodes — *enumerate what the
synthesized file DOES, line by line, not what it INSTALLS* — which is why two of
these five rows are not packages at all.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Pattern, Sequence, Tuple

#: Applicability: this step is owed by every hand-written Dockerfile.
ALWAYS = "always"
#: Applicability: owed only when an endpoint in the tree declares ``compile=``.
DECLARES_AOT_EXPORT = "aot_export"

#: A ``verified_by`` value naming an in-image ``aot_preconditions`` row. The
#: registry test resolves the suffix against that module and asserts the row is
#: reachable from ``static_mint_preconditions`` — a guarantee whose verifier is
#: unwired is the very defect this registry exists to stop.
PRECONDITION = "precondition:"

#: `compile=Compile(...)` on an `@endpoint` is the AOT declaration, and matching
#: it is exactly how pgw#1068's fleet sweep was done by hand. Conservative by
#: construction: a family that hides the object behind an alias reads as no-AOT
#: here and is still caught in-image by `aot_preconditions`, which asks the
#: registered declaration rather than the source text.
RE_AOT_DECLARATION = re.compile(r"compile\s*=\s*Compile\s*\(")

# tensorhub/internal/builder/validation.go: reBuildKitCacheMount
RE_BUILDKIT_CACHE_MOUNT = re.compile(r"(?i)--mount\s*=\s*type\s*=\s*cache\b")


@dataclass(frozen=True)
class Guarantee:
    """One step the synthesized Dockerfile takes, and the custom path's answer.

    ``require``/``forbid`` are what a hand-written Dockerfile must carry or must
    not; exactly one is set. ``verified_by`` names the platform's own refusal —
    a row that cannot name one is a guarantee nothing enforces, and the registry
    test refuses it.
    """

    id: str
    synthesized: str
    applies: str
    line: str
    verified_by: str
    cost: str
    require: Optional[Pattern[str]] = None
    forbid: Optional[Pattern[str]] = None
    #: The hub matches the RAW BYTES of a Dockerfile for its bans, comments
    #: included, so a `forbid` row must read comments too. A `require` row must
    #: not: a comment mentioning a step is not the step.
    reads_comments: bool = False


REGISTRY: Tuple[Guarantee, ...] = (
    Guarantee(
        id="discovery_lock",
        synthesized="generate_dockerfile.go — the discovery RUN step",
        applies=ALWAYS,
        require=re.compile(r"gen_worker\.discovery\b"),
        line="RUN mkdir -p /app/.tensorhub \\\n"
             "    && python -m gen_worker.discovery > /app/.tensorhub/endpoint.lock",
        verified_by="tensorhub executor.go — endpoint.lock read out of the image",
        cost="the build is refused after the whole image is built and pushed, "
             "rather than on the diff that dropped the step",
    ),
    Guarantee(
        id="worker_entrypoint",
        synthesized="generate_dockerfile.go — the ENTRYPOINT line",
        applies=ALWAYS,
        require=re.compile(r"gen_worker\.entrypoint\b"),
        line='ENTRYPOINT ["python", "-m", "gen_worker.entrypoint"]',
        verified_by="tensorhub executor.go — matchesWorkerEntrypoint, from the "
                    "image config",
        cost="same as discovery_lock: a post-build refusal for a one-line omission",
    ),
    Guarantee(
        id="cxx_toolchain",
        synthesized="generate_dockerfile.go — the apt line (pgw#823)",
        applies=DECLARES_AOT_EXPORT,
        # `\bg\+\+\b` does NOT match "g++ " — `+` is a non-word character, so
        # there is no word boundary after it. A trailing `\b` here silently
        # refuses every Dockerfile that DOES install the compiler.
        require=re.compile(r"\bg\+\+(?!\w)|\bbuild-essential\b"),
        line="RUN apt-get update \\\n"
             " && apt-get install -y --no-install-recommends "
             "ca-certificates curl g++ \\\n"
             " && rm -rf /var/lib/apt/lists/*",
        verified_by=PRECONDITION + "CHECK_CXX_TOOLCHAIN",
        cost="the in-image gate refuses the build for $0.00 — this row only "
             "moves that refusal earlier, onto the diff (pgw#1017)",
    ),
    Guarantee(
        id="cuda_root",
        synthesized="generate_dockerfile.go — cudaRootLine (pgw#823, ~30 lines "
                    "of shell composing a directory)",
        applies=DECLARES_AOT_EXPORT,
        require=re.compile(r"gen_worker\.cuda_root\b"),
        line="RUN python -m gen_worker.cuda_root",
        verified_by=PRECONDITION + "CHECK_CUDA_ROOT",
        cost="the most expensive row in this table and the reason it exists: "
             "before pgw#1017's precondition NOTHING refused, and the pod paid "
             "for boot, load and export before dying at CUDA_HOME. The step is "
             "a documented no-op on a CPU image, so it is owed unconditionally "
             "by an AOT-declaring family rather than guessed from the base tag",
    ),
    Guarantee(
        id="buildkit_cache_mount",
        synthesized="generate_dockerfile.go never emits one",
        applies=ALWAYS,
        forbid=RE_BUILDKIT_CACHE_MOUNT,
        reads_comments=True,
        line="install uncached instead — `uv pip install --no-cache`",
        verified_by="tensorhub validation.go — ErrBuildKitCache, 400 "
                    "invalid_tarball at publish",
        cost="the publish is refused, and pgw#1016 is the proof the doc itself "
             "can teach the refusal: a constant cache id is one mutable "
             "directory shared across tenant builds",
    ),
)


@dataclass(frozen=True)
class Finding:
    """One guarantee a hand-written Dockerfile does not satisfy."""

    guarantee: str
    path: str
    message: str

    def __str__(self) -> str:
        return f"{self.path}: build guarantee {self.guarantee}: {self.message}"


def guarantee(guarantee_id: str) -> Guarantee:
    for row in REGISTRY:
        if row.id == guarantee_id:
            return row
    raise KeyError(guarantee_id)


def verifier_text(row: Guarantee) -> str:
    """``verified_by``, rendered for the author rather than for the registry.

    A ``precondition:`` row stores the CONSTANT's name so the registry test can
    resolve it and prove the gate emits it. What an author needs is the string
    they will read in a failing build, so it is resolved here — one table, two
    audiences, no second spelling of the check name.
    """
    if not row.verified_by.startswith(PRECONDITION):
        return row.verified_by
    from . import aot_preconditions as ap

    check = getattr(ap, row.verified_by[len(PRECONDITION):])
    return (f"the in-image build gate — `aot precondition {check}`, refused by "
            f"`python -m gen_worker.discovery` inside your own image")


def _instructions(text: str) -> str:
    """The Dockerfile with comment lines removed.

    A `require` row asks what the file DOES. A comment explaining a step — or
    explaining why a step is missing, which is how pgw#1017's example read for
    two days — is not the step.
    """
    return "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("#")
    )


def declares_aot_export(source_dir: Path) -> List[str]:
    """Files under ``source_dir`` declaring ``compile=Compile(...)``."""
    hits: List[str] = []
    for path in sorted(source_dir.rglob("*.py")):
        rel = path.relative_to(source_dir)
        if any(part in {".venv", "venv", "build", "dist", "__pycache__", "tests"}
               for part in rel.parts):
            continue
        try:
            if RE_AOT_DECLARATION.search(path.read_text(errors="replace")):
                hits.append(rel.as_posix())
        except OSError:  # pragma: no cover — unreadable file in a source tree
            continue
    return hits


def applicable(row: Guarantee, *, aot: bool) -> bool:
    return row.applies == ALWAYS or (row.applies == DECLARES_AOT_EXPORT and aot)


def check_dockerfile(text: str, *, aot: bool, path: str = "Dockerfile",
                     why_aot: str = "") -> List[Finding]:
    """Every guarantee ``text`` owes and does not satisfy."""
    findings: List[Finding] = []
    instructions = _instructions(text)
    for row in REGISTRY:
        if not applicable(row, aot=aot):
            continue
        haystack = text if row.reads_comments else instructions
        if row.require is not None and not row.require.search(haystack):
            findings.append(Finding(row.id, path, _missing(row, why_aot)))
        elif row.forbid is not None and row.forbid.search(haystack):
            findings.append(Finding(row.id, path, _present(row)))
    return findings


def _missing(row: Guarantee, why_aot: str) -> str:
    trigger = ""
    if row.applies == DECLARES_AOT_EXPORT:
        trigger = (f" This is owed because an endpoint in this tree declares an "
                   f"AOT export{' (' + why_aot + ')' if why_aot else ''}; a "
                   f"family that declares none owes nothing here.")
    return (f"a hand-written Dockerfile must carry this step; the synthesized "
            f"path establishes it at {row.synthesized}.{trigger} Add:\n"
            f"{_indent(row.line)}\n"
            f"Platform refusal if you do not: {verifier_text(row)}. Cost of "
            f"skipping it: {row.cost}")


def _present(row: Guarantee) -> str:
    return (f"this Dockerfile carries something the platform refuses "
            f"({verifier_text(row)}); the synthesized path {row.synthesized}. "
            f"Instead: {row.line}. Cost of skipping this check: {row.cost}")


def _indent(block: str) -> str:
    return "\n".join("    " + line for line in block.splitlines())


def check_endpoint(source_dir: Path) -> List[Finding]:
    """Check one endpoint source tree.

    A tree with NO Dockerfile is silently fine: it takes the synthesized path,
    where every row in the registry is established by a layer the hub writes.
    That asymmetry is the whole subject of this module.
    """
    if source_dir.is_file() and source_dir.name == "Dockerfile":
        source_dir = source_dir.parent
    dockerfile = source_dir / "Dockerfile"
    if not dockerfile.is_file():
        return []
    decls = declares_aot_export(source_dir)
    return check_dockerfile(
        dockerfile.read_text(errors="replace"),
        aot=bool(decls),
        path=dockerfile.as_posix(),
        why_aot=", ".join(decls[:3]),
    )


def check_paths(paths: Iterable[Path]) -> List[Finding]:
    findings: List[Finding] = []
    for path in paths:
        findings.extend(check_endpoint(path))
    return findings


def describe_registry() -> str:
    out = []
    for row in REGISTRY:
        out.append(f"{row.id}  [{row.applies}]")
        out.append(f"    synthesized: {row.synthesized}")
        out.append(f"    verified by: {verifier_text(row)}")
    return "\n".join(out)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m gen_worker.build_guarantees",
        description=("Check a hand-written Dockerfile against the steps the "
                     "hub's synthesized Dockerfile takes for you. Run it in "
                     "CI for every endpoint that ships its own Dockerfile."))
    parser.add_argument("paths", nargs="*", type=Path,
                        help="endpoint source directories (default: .)")
    parser.add_argument("--list", action="store_true",
                        help="print the registry and exit")
    args = parser.parse_args(argv)

    if args.list:
        print(describe_registry())
        return 0

    paths = list(args.paths) or [Path(".")]
    for path in paths:
        if not path.exists():
            print(f"error: {path}: no such path", file=sys.stderr)
            return 2
        # Say what was checked. A check that passes because it looked at
        # nothing reads exactly like a check that passed.
        target = path.parent if path.name == "Dockerfile" else path
        if not (target / "Dockerfile").is_file():
            print(f"{path}: no Dockerfile — this endpoint takes the "
                  f"synthesized path, where the hub establishes every one of "
                  f"these steps itself. Nothing to check.", file=sys.stderr)

    findings = check_paths(paths)
    for finding in findings:
        print(f"error: {finding}", file=sys.stderr)
    if findings:
        print(f"\n{len(findings)} build guarantee(s) unmet. These are steps the "
              f"hub takes for Dockerfile-less endpoints and cannot take for "
              f"yours — your Dockerfile is author-owned content and the "
              f"platform never injects into it.", file=sys.stderr)
        return 1
    return 0


__all__ = [
    "ALWAYS",
    "DECLARES_AOT_EXPORT",
    "PRECONDITION",
    "REGISTRY",
    "RE_BUILDKIT_CACHE_MOUNT",
    "Finding",
    "Guarantee",
    "applicable",
    "check_dockerfile",
    "check_endpoint",
    "check_paths",
    "declares_aot_export",
    "describe_registry",
    "guarantee",
    "verifier_text",
]


if __name__ == "__main__":
    raise SystemExit(main())
