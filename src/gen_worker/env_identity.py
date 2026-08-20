"""ONE definition of "what environment is this", for both ends (pgw#1472).

**The env half of an artifact key is the COMPILE STACK** — torch, triton and
the `nvidia-*` libraries — read off the endpoint's own `uv.lock` (Paul,
2026-08-19, DESIGN-RULINGS addendum 4 as corrected; pgw#1489). Nothing else
in a lockfile can change what inductor emits, so nothing else is allowed to
split the artifact pool.

What this module used to do, and why it stopped: it stated the WHOLE resolved
package set as the identity, and the serving side restated its own
`importlib.metadata` set to check it. Two representations of one environment,
structurally unable to agree — pgw#1472 measured three independent reasons
(PEP 503 spelling, the `+cu129` local segment a lock cannot express, and
platform-conditional rows like `colorama`) and 43-package diffs between envs
that serve identically. Both halves are gone. The lock is read once, the
compile stack is selected from it by `torchcg.compile_stack`, and that same
selection is what a serving process compares against. One representation.

Everything else a pod must satisfy is ADMISSION metadata checked at adopt,
never a key input: the ELF-derived driver range (pgw#1471) and the measured
peak-VRAM stamp (tcg#62) ride the artifact's requirements manifest.
"""

from __future__ import annotations

import re
import tomllib
from collections import deque
from pathlib import Path
from typing import Any, Mapping

from ._vendor.torchcg.graph_identity import (
    GraphIdentityError,
    compile_stack,
    is_compile_relevant,
)

#: What `uv` writes beside a project, and therefore what every end reads.
#: Named once so the CLI, the derive and the serve runner cannot disagree.
LOCKFILE_NAME = "uv.lock"

#: What a CUDA bucket is CALLED: `cu126`, `cu130`, `cu1300`.
_BUCKET_RE = re.compile(r"cu[0-9]{3,4}")


class EnvIdentityError(ValueError):
    """A lockfile cannot be read, or states no compile stack."""


def _packages(lockfile: Path | str) -> list[dict[str, Any]]:
    path = Path(lockfile)
    try:
        parsed = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise EnvIdentityError(f"cannot read lockfile {path}: {exc}") from exc
    rows = [row for row in parsed.get("package", ()) if isinstance(row, dict)]
    if not rows:
        raise EnvIdentityError(f"lockfile {path} states no resolved packages")
    return rows


def lock_entries(lockfile: Path | str, *, bucket: str = "") -> dict[str, str]:
    """`{name: version}` for every package a `uv.lock` resolves.

    Verbatim: uv already writes PEP 503 names, and re-normalizing a key input
    is the drift-papering that pgw#1489 deleted. NOT a key on its own — the
    key input is :func:`compile_stack_from_lockfile`, which selects from this.
    Refuses a multi-flavor lock by name, because "the version of nvidia-cublas"
    is not a question that lock answers (see :func:`cuda_buckets`).
    """

    path = Path(lockfile)
    forked: dict[str, list[str]] = {}
    for package in _packages(path):
        name = package.get("name")
        version = package.get("version")
        if not isinstance(name, str) or not isinstance(version, str):
            continue
        versions = forked.setdefault(name, [])
        if version not in versions:
            versions.append(version)
    entries: dict[str, str] = {}
    for name, versions in forked.items():
        if len(versions) == 1:
            entries[name] = versions[0]
            continue
        entries[name] = _resolve_fork(path, name, versions, bucket)
    return entries


def _resolve_fork(path: Path, name: str, versions: list[str], bucket: str) -> str:
    """One name, several resolutions: the CUDA line decides, or nobody does.

    uv FORKS a resolution per index marker, so a lock can legitimately state
    `torch` at both `2.13.0` and `2.13.0+cu130` — pgw's own lock does, and a
    reader that raises on it fails closed on the repo that most needs it
    (measured, pgw#1472). The fork is a CUDA fork: its branches differ by the
    PEP 440 local segment, which is the bucket. So the host's own bucket picks
    its branch, exactly as it picks a flavored lock's extra.

    A fork this cannot attribute to a CUDA line is refused with both versions
    named. Guessing would key artifacts to an environment nobody has.
    """

    if bucket:
        matched = [v for v in versions if v.partition("+")[2] == bucket]
        if len(matched) == 1:
            return matched[0]
    if not is_compile_relevant(name):
        # Outside the compile stack a fork cannot reach the key; the first
        # resolution is as good as any, and `compile_stack` drops it anyway.
        return sorted(versions)[0]
    raise EnvIdentityError(
        f"lockfile {path} resolves {name!r} {len(versions)} ways "
        f"({', '.join(sorted(versions))}) and this env "
        + (f"is {bucket!r}, which matches none of them"
           if bucket else "states no CUDA bucket to pick one")
    )


def _root(packages: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The lock's own project row — where the flavor extras are declared."""

    for row in packages:
        source = row.get("source")
        if isinstance(source, dict) and ("virtual" in source or "editable" in source):
            return row
    return None


def cuda_buckets(lockfile: Path | str) -> tuple[str, ...]:
    """The CUDA buckets this lock resolves, e.g. ``("cu126", "cu130")``.

    Empty when the lock states ONE resolution — the pre-flavor shape, and the
    shape every endpoint has until its author adopts conflicting extras. A
    bucket is an author-declared extra whose pin is the compile stack's own
    torch, so this reads the lock rather than pattern-matching a name.
    """

    root = _root(_packages(lockfile))
    if root is None:
        return ()
    extras = root.get("optional-dependencies")
    if not isinstance(extras, dict):
        return ()
    # A bucket is an extra NAMED for a CUDA line (`cu130`) that pins a
    # compile-stack package. The name test is not decoration: pgw's own
    # pyproject has an extra called `torch`, and an extra is only a bucket when
    # the author meant it as one.
    return tuple(
        sorted(
            name
            for name, pins in extras.items()
            if _BUCKET_RE.fullmatch(str(name))
            and isinstance(pins, list)
            and any(
                isinstance(pin, dict) and is_compile_relevant(str(pin.get("name") or ""))
                for pin in pins
            )
        )
    )


def compile_stack_from_lockfile(
    lockfile: Path | str, *, bucket: str = ""
) -> tuple[tuple[str, str], ...]:
    """The endpoint's compile stack: the env half of every artifact key.

    THE key input, and the only one this module produces. Reads the rows the
    endpoint's author locked, never what happens to be installed.

    ``bucket`` names the CUDA line this env materialized (``uv sync --extra
    cu130``). It is required exactly when the lock resolves more than one —
    the author locks every flavor in ONE file (conflicting extras) and the
    host picks by driver at bootstrap, so a lock with three buckets cannot
    answer "which nvidia-cublas" without being told. It is IGNORED for a
    single-resolution lock, which has nothing to pick.

    The bucket is not a separate key component, deliberately: a flavored
    torch states it IN its version (``2.8.0+cu126``) and every nvidia pin
    below it differs per bucket, so the versions ARE the bucket. Keying it
    twice is the second-representation defect this issue exists to delete.
    """

    path = Path(lockfile)
    packages = _packages(path)
    buckets = cuda_buckets(path)
    try:
        if not buckets:
            return compile_stack(lock_entries(path, bucket=bucket))
        if not bucket:
            raise EnvIdentityError(
                f"lockfile {path} locks {len(buckets)} CUDA buckets "
                f"({', '.join(buckets)}); state the one this env materialized"
            )
        if bucket not in buckets:
            raise EnvIdentityError(
                f"lockfile {path} locks {', '.join(buckets)}; this env is "
                f"{bucket!r}, which its author never locked"
            )
        return compile_stack(_bucket_entries(path, packages, bucket))
    except GraphIdentityError as exc:
        raise EnvIdentityError(f"lockfile {path}: {exc}") from exc


def _bucket_entries(
    path: Path, packages: list[dict[str, Any]], bucket: str
) -> dict[str, str]:
    """Every compile-stack row reachable from one bucket's own torch pin.

    A flavored lock pins the bucket's torch in the project's extra, and THAT
    package row pins its own nvidia set — so the resolution is read out of the
    lock's dependency edges rather than guessed from version numbers.
    """

    by_name_version: dict[tuple[str, str], dict[str, Any]] = {}
    by_name: dict[str, list[dict[str, Any]]] = {}
    for row in packages:
        name, version = row.get("name"), row.get("version")
        if isinstance(name, str) and isinstance(version, str):
            by_name_version[(name, version)] = row
            by_name.setdefault(name, []).append(row)

    root = _root(packages)
    pins = (root or {}).get("optional-dependencies", {}).get(bucket, [])
    # BREADTH-FIRST from the bucket's own torch, and the SHALLOWEST edge wins.
    # Depth matters: a package shared between buckets (cudnn is locked once)
    # carries edges to ONE bucket's cublas, and a depth-first walk would let
    # that stale edge overwrite the pin torch itself states. The bucket's
    # answer is the one its torch names.
    frontier = deque(
        pin for pin in pins
        if isinstance(pin, dict) and is_compile_relevant(str(pin.get("name") or ""))
    )
    found: dict[str, str] = {}
    while frontier:
        pin = frontier.popleft()
        name = str(pin.get("name") or "")
        version = pin.get("version")
        if not isinstance(version, str):
            rows = by_name.get(name, [])
            if len(rows) != 1:
                raise EnvIdentityError(
                    f"lockfile {path}: {name!r} is named without a version by "
                    f"bucket {bucket!r} and resolves {len(rows)} ways; the "
                    f"lock cannot say which one this bucket links"
                )
            version = str(rows[0].get("version") or "")
        if name in found:
            continue
        found[name] = version
        entry: dict[str, Any] = by_name_version.get((name, version)) or {}
        for dependency in entry.get("dependencies", []) or []:
            if isinstance(dependency, dict) and is_compile_relevant(
                str(dependency.get("name") or "")
            ):
                frontier.append(dependency)
    return found


def _torch_cuda_line() -> str:
    """``torch.version.cuda`` WITHOUT executing the torch package (pgw#1546).

    ``torch/version.py`` is a generated constants module with no imports of
    its own, and the value is static per install — paying the full torch
    import (~1.5 s) to read one string made every warm ``compile`` carry it.
    A torch already in ``sys.modules`` is used as-is; the full import survives
    as the fallback for any install whose layout differs.
    """
    import importlib.util
    import sys

    if "torch" not in sys.modules:
        try:
            spec = importlib.util.find_spec("torch")
            for location in list(getattr(spec, "submodule_search_locations", None) or []):
                path = Path(location) / "version.py"
                if not path.is_file():
                    continue
                probe = importlib.util.spec_from_file_location(
                    "_gen_worker_torch_version_probe", path
                )
                if probe is None or probe.loader is None:
                    continue
                module = importlib.util.module_from_spec(probe)
                probe.loader.exec_module(module)
                return str(getattr(module, "cuda", "") or "")
        except Exception:  # noqa: BLE001 - fall through to the real import
            pass
    try:
        import torch

        return str(getattr(torch.version, "cuda", "") or "")
    except Exception:  # noqa: BLE001 - absence is an answer
        return ""


def cuda_bucket() -> str:
    """This host's CUDA bucket, from the torch it actually materialized.

    One of the two variables a HOST contributes (the other is sm). Read off
    the installed torch because that is what `uv sync --extra` produced — the
    host does not resolve anything, it reports which flavor it received.
    ``""`` when there is no CUDA torch here, which a single-resolution lock
    does not care about.
    """

    line = _torch_cuda_line()
    parts = line.split(".")
    if len(parts) < 2 or not parts[0].isdigit():
        return ""
    return f"cu{parts[0]}{parts[1]}"


def lockfile_beside(endpoint_dir: Path | str) -> Path | None:
    """The endpoint's own `uv.lock`, or `None` — never a guess elsewhere.

    This is the SAME file `gen-worker lock` reads for the document it writes,
    which is the whole point: a derive and a serve read one file, not two
    sources that have to be reconciled.
    """

    candidate = Path(endpoint_dir) / LOCKFILE_NAME
    return candidate if candidate.is_file() else None


def installed_stack_drift(stated: Mapping[str, str] | tuple[tuple[str, str], ...]) -> tuple[str, ...]:
    """DIAGNOSTIC ONLY: how the venv's compile stack differs from the lock's.

    Nothing gates on this and nothing keys on it — that is the pgw#1489 line.
    It exists because
    one divergence here is genuinely fatal at RUN time (an artifact compiled
    against the lock's torch, loaded into a venv with a different one), and a
    warning naming the package beats a segfault. It compares the compile
    stack only, so it cannot fire on a docs extra, and it strips the `+cu129`
    local segment a lockfile cannot express.
    """

    import importlib.metadata

    want = dict(stated)
    rows: list[str] = []
    for name, version in sorted(want.items()):
        try:
            found = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            rows.append(f"{name} {version} locked, not installed")
            continue
        if found.split("+", 1)[0] != version.split("+", 1)[0]:
            rows.append(f"{name} {version} locked, {found} installed")
    return tuple(rows)


__all__ = [
    "LOCKFILE_NAME",
    "EnvIdentityError",
    "compile_stack_from_lockfile",
    "installed_stack_drift",
    "is_compile_relevant",
    "lock_entries",
    "lockfile_beside",
]
