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

LOCKFILE_NAME = "uv.lock"

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
    """`{name: version}` for every package a `uv.lock` resolves."""

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

    if bucket:
        matched = [v for v in versions if v.partition("+")[2] == bucket]
        if len(matched) == 1:
            return matched[0]
    if not is_compile_relevant(name):
        return sorted(versions)[0]
    raise EnvIdentityError(
        f"lockfile {path} resolves {name!r} {len(versions)} ways "
        f"({', '.join(sorted(versions))}) and this env "
        + (f"is {bucket!r}, which matches none of them"
           if bucket else "states no CUDA bucket to pick one")
    )


def _root(packages: list[dict[str, Any]]) -> dict[str, Any] | None:

    for row in packages:
        source = row.get("source")
        if isinstance(source, dict) and ("virtual" in source or "editable" in source):
            return row
    return None


def cuda_buckets(lockfile: Path | str) -> tuple[str, ...]:
    """The CUDA buckets this lock resolves, e.g."""

    root = _root(_packages(lockfile))
    if root is None:
        return ()
    extras = root.get("optional-dependencies")
    if not isinstance(extras, dict):
        return ()
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
    """The endpoint's compile stack: the env half of every artifact key."""

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

    by_name_version: dict[tuple[str, str], dict[str, Any]] = {}
    by_name: dict[str, list[dict[str, Any]]] = {}
    for row in packages:
        name, version = row.get("name"), row.get("version")
        if isinstance(name, str) and isinstance(version, str):
            by_name_version[(name, version)] = row
            by_name.setdefault(name, []).append(row)

    root = _root(packages)
    pins = (root or {}).get("optional-dependencies", {}).get(bucket, [])
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
    """This host's CUDA bucket, from the torch it actually materialized."""

    line = _torch_cuda_line()
    parts = line.split(".")
    if len(parts) < 2 or not parts[0].isdigit():
        return ""
    return f"cu{parts[0]}{parts[1]}"


def lockfile_beside(endpoint_dir: Path | str) -> Path | None:
    """The endpoint's own `uv.lock`, or `None` — never a guess elsewhere."""

    candidate = Path(endpoint_dir) / LOCKFILE_NAME
    return candidate if candidate.is_file() else None


def installed_stack_drift(stated: Mapping[str, str] | tuple[tuple[str, str], ...]) -> tuple[str, ...]:
    """DIAGNOSTIC ONLY: how the venv's compile stack differs from the lock's."""

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
