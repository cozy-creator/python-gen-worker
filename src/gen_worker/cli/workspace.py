"""The box-shared stores the four verbs share, and how a ref becomes a tree.

pgw#1466. ``lock``, ``sync``, ``serve`` and ``run`` are four front doors onto
one set of stores; this module is where "where do weights live" and "where do
graphs live" are each answered ONCE. A verb that computed its own answer would
be a verb that can disagree with its siblings — `sync` pre-warming into a
directory `serve` never reads is the failure mode, and it is silent.

Two stores, deliberately separate:

* the **weight CAS** — the box-shared tensorfs store. Big, blake3-addressed,
  shared with every other tool on this machine, and NOT ours to invent a layout
  for: we read the one that is already there.
* the **graph CAS** — exported-program blobs from the trace, and the compiled
  artifacts minted from them. Ours, small, per-user.

Weights and graphs are split because their lifetimes differ by orders of
magnitude: a weight tree is worth 6 GB and survives every code change, a graph
blob is invalidated by a torch bump. One CAS would make a GC policy that is
right for one of them wrong for the other.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

#: The box-shared tensorfs weight CAS. This path is not a preference — it is
#: where the store on this machine already is, populated by the hub client.
#: Overridable via the `COZY_WEIGHTS_CAS` config key (Settings field
#: `weights_cas_root`), never guessed.
DEFAULT_WEIGHTS_CAS = Path.home() / ".cache" / "tensorhub" / "cas"

#: Exported programs (from `lock`) and compiled artifacts (from `sync`).
#: Config key `COZY_GRAPH_CAS` (Settings field `graph_cas_root`).
DEFAULT_GRAPH_CAS = Path.home() / ".cache" / "cozy" / "graph-cas"


class WorkspaceError(RuntimeError):
    """A store or a ref could not be resolved to something on disk."""


def _settings() -> Any:
    """Installed Settings, or a zero-config default.

    §1.18: config is loaded ONCE by the pipeline and passed; a module reading
    `os.environ` itself is a second loader that can disagree with the first.
    `current_or` takes the fallback AS A VALUE so the zero-config case is
    visible here rather than hidden in an env read — which matters because this
    module is imported both by the CLI (which installs Settings at process
    entry) and by scripts that never bring a worker up.
    """
    from .. import config

    return config.current_or(config.Settings())


def weights_cas_root() -> Path:
    return Path(_settings().weights_cas_root or DEFAULT_WEIGHTS_CAS)


def graph_cas_root() -> Path:
    return Path(_settings().graph_cas_root or DEFAULT_GRAPH_CAS)


def trace_device() -> str:
    """The device CLASS a trace states. ``derive._trace_device`` IS the law.

    Imported rather than reimplemented: the CLI answering this differently
    from the derive would write a lock whose ``trace_device`` disagrees with
    the document inside it, and the skip check would then hit forever or miss
    forever. There is one answer to this question and it lives in the derive.

    Since tcg#64 the answer does not depend on the host at all -- `lock` is an
    author-time command and a trace needs no GPU.
    """
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
    """``tensorhub/sd15-base@latest`` -> a CheckpointRef.

    A bare ``owner/repo`` pins NOTHING — the revision it happens to resolve to
    today is not the revision it resolves to tomorrow, and a lock keyed on an
    unpinned ref is a lock that silently describes a different model later. So
    an absent ``@rev`` is refused rather than defaulted.
    """
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
    """The materialized snapshot tree for ``ref``, or a refusal that says why.

    Reads the on-disk store layout directly (``refs/<owner>/<name>/<rev>``
    holding a snapshot id, ``snapshots/<id>`` holding the tree) rather than going
    through a client, because that layout is what exists on this box and the
    verbs need to work against it TODAY. The hub-fetch path that populates a cold
    store is a separate concern; this function's job is to answer "is it here",
    and to be unambiguous when the answer is no.
    """
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
            f"  run `gen-worker sync` to materialize it."
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
    """Total size of a materialized tree. The residency budget's input.

    Walks rather than trusting a manifest because a partially-materialized tree
    is exactly the case where a manifest's number and reality disagree, and
    sizing residency off the manifest would admit a model that does not fit.
    """
    return sum(p.stat().st_size for p in tree.rglob("*") if p.is_file())


def local_cas(root: Path) -> Any:
    """A tensorfs ``LocalCAS`` at ``root``, created if absent.

    The graph CAS is ours to create; the weight CAS is not (see module
    docstring), so only callers holding the graph root should use this.
    """
    from .._vendor.tensorfs import LocalCAS

    root.mkdir(parents=True, exist_ok=True)
    return LocalCAS(root)


__all__ = [
    "DEFAULT_GRAPH_CAS",
    "DEFAULT_WEIGHTS_CAS",
    "CheckpointRef",
    "WorkspaceError",
    "graph_cas_root",
    "local_cas",
    "parse_checkpoint_ref",
    "resolve_checkpoint",
    "trace_device",
    "tree_bytes",
    "weights_cas_root",
]
