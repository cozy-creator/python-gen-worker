"""The box-shared stores the verbs share, and how a ref becomes a tree."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

DEFAULT_WEIGHTS_CAS = Path.home() / ".cache" / "tensorhub" / "cas"

DEFAULT_GRAPH_CAS = Path.home() / ".cache" / "cozy" / "graph-cas"

DEFAULT_ARTIFACTS = Path.home() / ".cache" / "cozy" / "compiled-graphs"


class WorkspaceError(RuntimeError):
    """A store or a ref could not be resolved to something on disk."""


def _settings() -> Any:
    from .. import config

    return config.current_or(config.Settings())


def weights_cas_root() -> Path:
    return Path(_settings().weights_cas_root or DEFAULT_WEIGHTS_CAS)


def graph_cas_root() -> Path:
    return Path(_settings().graph_cas_root or DEFAULT_GRAPH_CAS)


def artifacts_root() -> Path:
    return Path(_settings().artifacts_root or DEFAULT_ARTIFACTS)


def host_sm() -> str:
    """This host's CUDA compute capability as ``sm_XY``, or ``""``."""
    import subprocess

    try:
        answer = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip().splitlines()
        if answer:
            major, _, minor = answer[0].strip().partition(".")
            if major.isdigit() and minor.isdigit():
                return f"sm_{int(major)}{int(minor)}"
    except Exception:  # noqa: BLE001 - fall through to the torch reading
        pass
    try:
        import torch

        if not torch.cuda.is_available():
            return ""
        capability = torch.cuda.get_device_capability()
        return f"sm_{capability[0]}{capability[1]}"
    except Exception:  # noqa: BLE001 - absence is an answer, not an error
        return ""


def trace_device() -> str:
    """The device CLASS a trace states."""
    from ..release.derive import _trace_device

    return str(_trace_device())


@dataclass(frozen=True)
class CheckpointRef:
    """``owner/name@rev`` — parsed, never re-spelled by hand."""

    owner: str
    name: str
    rev: str

    @property
    def repo(self) -> str:
        return f"{self.owner}/{self.name}"

    def __str__(self) -> str:
        return f"{self.owner}/{self.name}@{self.rev}"


def parse_checkpoint_ref(raw: str) -> CheckpointRef:
    """``tensorhub/sd15-base@latest`` -> a CheckpointRef."""
    text = raw.strip()
    if "@" not in text:
        raise WorkspaceError(
            f"checkpoint ref {raw!r} pins nothing: it needs an explicit "
            f"@revision (e.g. {text}@latest). A bare owner/repo resolves to "
            f"whatever is current, which makes every lock derived from it "
            f"unreproducible."
        )
    repo, _, rev = text.partition("@")
    if repo.count("/") != 1 or not all(repo.split("/")) or not rev:
        raise WorkspaceError(
            f"checkpoint ref {raw!r} is not owner/name@revision"
        )
    owner, name = repo.split("/")
    return CheckpointRef(owner=owner, name=name, rev=rev)


def resolve_checkpoint(ref: CheckpointRef, *, cas_root: Optional[Path] = None) -> Path:
    """The materialized snapshot tree for ``ref``, or a refusal that says why."""
    root = Path(cas_root) if cas_root is not None else weights_cas_root()
    if not root.is_dir():
        raise WorkspaceError(
            f"no weight CAS at {root} (set COZY_WEIGHTS_CAS to point at one)"
        )
    ref_file = root / "refs" / ref.owner / ref.name / ref.rev
    if not ref_file.is_file():
        available = sorted(
            p.relative_to(root / "refs").as_posix()
            for p in (root / "refs").rglob("*")
            if p.is_file()
        )
        raise WorkspaceError(
            f"{ref} is not in the weight CAS at {root}.\n"
            f"  refs present: {', '.join(available) if available else '(none)'}\n"
            f"  run `gen-worker download {ref}` to materialize it."
        )
    snapshot_id = ref_file.read_text(encoding="utf-8").strip()
    if not snapshot_id:
        raise WorkspaceError(f"{ref_file} is empty — the store is corrupt")
    tree = root / "snapshots" / snapshot_id
    if not tree.is_dir():
        raise WorkspaceError(
            f"{ref} points at snapshot {snapshot_id}, which is not "
            f"materialized at {tree}. The ref outlived its tree; re-fetch."
        )
    return tree


def tree_bytes(tree: Path) -> int:
    """Total size of a materialized tree."""
    return sum(p.stat().st_size for p in tree.rglob("*") if p.is_file())


def local_graph_store(root: Path) -> Any:
    """torchcg's ``LocalGraphStore`` over the graph CAS at ``root``."""
    from .._vendor.torchcg.store import LocalGraphStore

    return LocalGraphStore(local_cas(root))


def local_cas(root: Path) -> Any:
    """A tensorfs ``LocalCAS`` at ``root``, created if absent."""
    from .._vendor.tensorfs import LocalCAS

    root.mkdir(parents=True, exist_ok=True)
    return LocalCAS(root)


__all__ = [
    "DEFAULT_ARTIFACTS",
    "DEFAULT_GRAPH_CAS",
    "DEFAULT_WEIGHTS_CAS",
    "CheckpointRef",
    "WorkspaceError",
    "artifacts_root",
    "graph_cas_root",
    "host_sm",
    "local_cas",
    "parse_checkpoint_ref",
    "resolve_checkpoint",
    "trace_device",
    "tree_bytes",
    "weights_cas_root",
]
