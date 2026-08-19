"""tcg#58: the bytes -> callable half of adoption, which nobody had written.

``AdoptSession`` fetches an artifact and hands it to an :data:`~torchcg.adopt.
ArtifactLoader` to become the target module's forward. Every consumer wrote
that loader as a bare ``aoti_load_package(path)`` -- and a bare load is
UNSERVABLE here, for two independent reasons that are both sealed policy:

* **The artifacts are WEIGHTLESS.** ``compiler.py`` compiles with
  ``aot_inductor.package_constants_in_so = False`` and ``artifact.py``'s
  ``validate_metadata`` REFUSES any artifact that says otherwise, so a raw
  handle has an empty constant buffer. There is no other kind of artifact.
* **The package takes FLAT POSITIONAL feeds.** The dispatcher calls the loaded
  object with the author's own nested ``(*args, **kwargs)``. Mapping those onto
  the exported flat arity is exactly what :class:`~torchcg.ingress.CallIngress`
  exists for, and it is why the loader is handed the record at all.

So the loader is not a place a consumer improvises: this module IS it.
:func:`aoti_loader` is the whole sanctioned path from fetched bytes to an armed
callable -- materialize + verify, load through the gated
:class:`~torchcg.runner.CompiledGraphRunner`, bind the live module's resident
constants by reference, and feed through the record's ingress.

**The constants come from the LIVE MODULE, and that is why the loader is given
one.** A serving pod adopts INTO author code that is already resident: the
eager module stays the serve host (a hole, an ambiguous pair, or an unmatched
call structure all fall back to its own forward), so its parameters are the
one copy of the checkpoint on the device and the compiled graph binds them BY
REFERENCE rather than materializing a second set.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .artifact import (
    LITERALS_NAME,
    PACKAGE_NAME,
    ArtifactError,
    _verify_materialized,
    unpack_artifact,
)
from .document import GraphRecord
from .identity import GRAPH_SPECIALIZATION_BLOCK
from .runner import CompiledGraphRunner, ConstantBindingError

#: Where :func:`aoti_loader` unpacks a packed artifact when the caller states
#: no workspace: beside the artifact, named after it. Deterministic on purpose
#: -- a second boot over the same ``artifacts_dir`` re-verifies the directory
#: it already wrote instead of unpacking the same bytes again.
UNPACKED_SUFFIX = ".unpacked"


@dataclass(slots=True)
class MaterializedGraph:
    """One verified artifact on disk, in the shape the runner reads.

    The v1 store hands the runner a ``StoredCompiledGraph``; the v2 adopt path
    has only a fetched artifact at a path. Both satisfy
    :class:`~torchcg.runner.VerifiedGraph`, so there is ONE gated loader rather
    than a second one for adoption.
    """

    key: str
    directory: Path
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def package(self) -> Path:
        return self.directory / PACKAGE_NAME

    @property
    def literals(self) -> Path | None:
        path = self.directory / LITERALS_NAME
        return path if path.is_file() else None


def materialize(artifact: str | Path, workspace: str | Path | None = None) -> MaterializedGraph:
    """Verify one artifact -- packed archive or unpacked directory -- on disk.

    Both shapes are real and reach the same loader from different producers:
    ``GraphStore.fetch_artifact`` copies out the PACKED blob, while a runtime
    mint's ``Engine.compile`` leaves the resolved artifact UNPACKED at its
    destination. Verification is the same either way and it is not optional --
    ``_verify_materialized`` is what proves the metadata, the package and any
    literal payload agree before AOTI is handed anything.
    """

    source = Path(artifact)
    if source.is_dir():
        directory = source
        metadata = _verify_materialized(directory)
    else:
        directory = (
            Path(workspace)
            if workspace is not None
            else source.with_name(source.name + UNPACKED_SUFFIX)
        )
        if directory.is_dir():
            # Already unpacked by an earlier boot over the same artifacts_dir.
            # Accepted only because it VERIFIES -- the same doctrine the store's
            # occupied-destination check runs under, never "it exists, trust it".
            metadata = _verify_materialized(directory)
        else:
            metadata = unpack_artifact(source, directory)
    key = str(metadata.get("compiled_graph_key") or "")
    if not key:
        raise ArtifactError(f"artifact at {source} states no compiled_graph_key")
    return MaterializedGraph(key=key, directory=directory, metadata=metadata)


def resident_constants(module: Any) -> dict[str, Any]:
    """Every resident tensor a ``state_dict``-sourced constant can bind to.

    ``module.state_dict()`` is NOT that set: it omits NON-PERSISTENT buffers,
    which ``torch.export`` still lifts as buffer inputs and which AOTInductor
    still declares under their real FQN, i.e. ``source=state_dict``. A bind
    table built from ``state_dict()`` alone therefore declares constants no
    lookup could ever resolve. Binding is by FQN against the DECLARED names, so
    the extra keys are inert for any artifact that does not want them.
    """

    values: dict[str, Any] = dict(module.state_dict())
    for name, buffer in module.named_buffers():
        if buffer is not None:
            values.setdefault(str(name), buffer)
    return values


def module_device(module: Any) -> str:
    """Where this module's own tensors live -- the only honest bind device.

    Read off the tensors rather than off a ``.device`` attribute: the attribute
    is a library convenience some modules do not have and others compute, while
    the constants being bound BY REFERENCE are these exact tensors.
    """

    for tensor in module.state_dict().values():
        device = getattr(tensor, "device", None)
        if device is not None:
            return str(device)
    for _, buffer in module.named_buffers():
        if buffer is not None:
            return str(buffer.device)
    raise ConstantBindingError(
        "module_has_no_tensors",
        f"{type(module).__name__} carries no parameter or buffer, so there is "
        f"no device its compiled graph's constants could be bound from",
    )


def _declared_placement(metadata: Mapping[str, object]) -> tuple[str, ...]:
    block = metadata.get(GRAPH_SPECIALIZATION_BLOCK)
    if not isinstance(block, Mapping):
        return ()
    placement = block.get("placement")
    if not isinstance(placement, list):
        return ()
    return tuple(str(row) for row in placement)


def _assert_placement(graph: MaterializedGraph, device: str) -> None:
    """The artifact's own recorded placement must be where the module IS.

    An INDEPENDENT cross-check, which is the only kind worth writing: the
    placement was recorded by the mint from the program it compiled, and the
    device is read from the live module's tensors. Disagreement means the bind
    would install host pointers into a device graph (or the reverse) -- AOTI
    accepts that and the first call returns garbage or faults, so it is caught
    here, by name, and the adopt turns it into a hole instead of a wrong image.
    """

    declared = _declared_placement(graph.metadata)
    if not declared:
        return
    import torch

    want = torch.device(device)
    for row in declared:
        try:
            candidate = torch.device(row)
        except (RuntimeError, ValueError, TypeError):
            continue
        if candidate.type != want.type:
            continue
        if candidate.index is None or want.index is None or candidate.index == want.index:
            return
    raise ConstantBindingError(
        "placement_mismatch",
        f"compiled graph {graph.key} was compiled for placement "
        f"{list(declared)!r} but the module holding its constants is on "
        f"{device!r}; binding across that gap produces a graph that reads the "
        f"wrong memory",
    )


def aoti_loader(
    artifact: str | Path,
    record: GraphRecord,
    module: Any,
    *,
    workspace: str | Path | None = None,
) -> Callable[..., Any]:
    """THE loader: fetched artifact + record + live module -> armed forward.

    The returned callable is what ``AdoptSession`` installs behind the target
    module's dispatcher. It takes the author's own call -- nested kwargs and
    all -- maps it through the record's ingress to the flat positional feeds
    the package was exported with, and invokes the bound runner.

    Nothing here falls back. A refusal is a typed
    :class:`~torchcg.runner.ConstantBindingError` and ``AdoptSession`` is what
    decides that the graph becomes a hole and the module keeps serving eager;
    a loader that swallowed its own failure would arm a graph that cannot run.
    """

    graph = materialize(artifact, workspace)
    runner = CompiledGraphRunner._from_verified_graph(graph)
    device = module_device(module)
    _assert_placement(graph, device)
    runner.bind(resident_constants(module), device=device)
    return CompiledGraphCall(runner=runner, record=record)


@dataclass(slots=True)
class CompiledGraphCall:
    """One armed graph, called the way the author calls the module it replaced.

    A dataclass rather than a closure so a serving pod can be ASKED what it
    armed: ``runner.calls``, ``runner.bound_fqns`` and ``record.graph`` are the
    adopt evidence, and a closure would have hidden all three.
    """

    runner: CompiledGraphRunner
    record: GraphRecord

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        # STRAIGHT THROUGH (tcg#61). The package carries the export's own
        # `in_spec` and flattens with it, so the author's call is exactly what
        # it wants. `CallIngress` still owns DISPATCH -- `_matches` is what
        # decides this graph may answer this call at all -- but it does not
        # own the calling convention, and the version of this line that
        # thought it did died on the first real artifact with
        # `Ran into a kwarg keyword mismatch`.
        return self.runner(*args, **kwargs)


__all__ = [
    "CompiledGraphCall",
    "MaterializedGraph",
    "UNPACKED_SUFFIX",
    "aoti_loader",
    "materialize",
    "module_device",
    "resident_constants",
]
