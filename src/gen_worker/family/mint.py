"""A family DECLARATION mints its own graph classes. No endpoint, no weights.

pgw#1331's first two bullets ask for the VAE decoder and the text encoders to be
minted as graph classes. Declaring them made them exportable; this module is
what makes them MINTABLE — the bridge from a :class:`~gen_worker.family.spec.
GraphFamily` to a packed AOTI artifact, through the mint lane's own inner seam.

**Why a bridge and not a transpiler.** The production mint lane
(``aot_mint.mint_graph_classes``) drives a compile POOL that re-composes a
tenant's pipeline in fresh interpreters from ``modules`` + ``function`` +
``MintSlot``s, and derives its example feeds from a registered
``api.export_contract.Compile``. A family already carries everything that
machinery reconstructs — ``Runner.build`` is the module, ``Runner.example`` is
the call, ``Bucket``/``layouts`` are the coordinates — so making a family
pretend to be an endpoint would be a second declaration of the same facts, and
pgw#824's rule is that two lists of the same literals drift. This calls the
seam BELOW the pool instead: ``export_program`` -> ``build_call_ingress`` ->
``keying_block`` -> ``TracedClass`` -> ``Engine.compile``, which is the same
sequence ``aot_compile_child`` runs and the same one ``measure_child`` already
drives directly.

**The trace is the declaration's own.** :func:`gen_worker.family.export.
trace_variant` produces the program, and the snapshot in the catalog is
produced from that same function — so the mint's ingress digest and the
committed export's ingress digest are the same number by construction, not by
comparison. That is torchcg G16's direction: the recipe ASSERTS against the
declaration, it never becomes the source.

**No checkpoint is downloaded, and that is the ruling, not a shortcut.**
Cell identity is graph x sm x toolchain — checkpoint-free (DESIGN-RULINGS
§4.27), which is exactly what lets one compiled cell serve sixteen fine-tunes.
The declaration builds the ARCHITECTURE inside a fake-tensor mode, the program
carries fake constants, and the real weights arrive at ARM time by manifest FQN
out of the constant store (pgw#1329's ``arm_compiled_graph_from_store``). So
minting a family costs a card and a compile, never a 24 GB download.

**This module is MINT MACHINERY** and is declared as such in
:data:`gen_worker.serve.role.MINT_MACHINERY`: an adopt-only pod that could
reach it could compile, which is the one thing that role is defined by not
being able to do (§4.28). The family SURFACE it mints for is on the serve path;
this is not.

Run it with ``gen-worker family mint`` — **on a pod, never on a shared box**.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..aot_inputs import ExportSpec
from .errors import FamilyError, FamilyRefusal
from .export import TracedVariant, trace_variant
from .spec import GraphFamily, Runner


def graph_class_name(runner: str, bucket: Mapping[str, int], layout: str) -> str:
    """The handle one (runner, bucket, layout) is minted under.

    A HANDLE, never an identity: what keys the artifact is the class hash the
    trace produces (torchcg ``cg-key-v1``), and this name only has to be stable,
    readable in a phase table, and unique within a family. Bucket axes are
    sorted so the same variant spells the same name on any machine.
    """

    rows = "".join(f".{name}{int(value)}" for name, value in sorted(bucket.items()))
    return f"{runner}{rows}.{layout}"


def export_spec_for(
    family: GraphFamily, runner: Runner, bucket: Mapping[str, int], layout: str
) -> ExportSpec:
    """The worker coordinates one declared variant is keyed under.

    The mapping is one-for-one and there is nothing to invent: the family is the
    family, the RUNNER is the target (one runner, one traced entry point), and
    the BUCKET is ``class_dims`` — which is precisely the field whose values
    separate two classes of the same target inside one artifact family.

    ``layout`` rides ``specialization`` rather than being left implicit. The
    dtype it implies is already inside the traced graph, so two layouts key
    apart regardless; recording it makes the key's reason legible in the
    envelope instead of only derivable from it.
    """

    return ExportSpec(
        family=family.name,
        target=runner.name,
        class_dims=tuple(sorted((str(name), int(value)) for name, value in bucket.items())),
        specialization={"layout": str(layout)},
        strict=True,
    )


@dataclass(frozen=True, slots=True)
class MintedVariant:
    """One packed artifact, and the declaration coordinate it answers."""

    runner: str
    bucket: tuple[tuple[str, int], ...]
    layout: str
    graph_class: str
    key: str
    artifact: Path
    metadata: Mapping[str, Any]
    ingress_digest: str
    compile_s: float
    reuse_s: float

    @property
    def reused(self) -> bool:
        """Whether the CAS already held this class. Not a failure — the point."""

        return self.compile_s == 0.0


def variants_of(
    family: GraphFamily, *, only: Sequence[str] = ()
) -> tuple[tuple[Runner, dict[str, int], str], ...]:
    """Every (runner, bucket, layout) to mint, in the declaration's own order.

    ``only`` narrows to named runners — the shape a pod uses when it is minting
    the cheap classes first and the 12-billion-parameter one last, or when a
    previous attempt already banked half the set. A name the family does not
    declare is REFUSED rather than silently minting nothing.
    """

    rows = family.variants()
    if not only:
        return rows
    wanted = {str(name) for name in only}
    unknown = sorted(wanted - {runner.name for runner in family.runners})
    if unknown:
        raise FamilyError(
            FamilyRefusal.FAMILY_INVALID,
            f"family {family.name!r} declares no runner {unknown[0]!r}; it has "
            f"{sorted(runner.name for runner in family.runners)!r}",
        )
    return tuple(row for row in rows if row[0].name in wanted)


def traced_classes(
    family: GraphFamily, *, only: Sequence[str] = ()
) -> Iterator[tuple[TracedVariant, ExportSpec, Any]]:
    """Trace every selected variant, yielding one at a time.

    A GENERATOR, and that is the memory bound: an exported program for a
    12-billion-parameter denoiser is large even with fake constants, and the
    caller compiles and releases each row before the next is traced. Building
    the whole list first is how a mint pod runs out of host RAM on the family
    it was bought for.
    """

    from .. import aot_mint

    for runner, bucket, layout in variants_of(family, only=only):
        traced = trace_variant(runner, bucket, layout)
        spec = export_spec_for(family, runner, bucket, layout)
        row = aot_mint.TracedClass(
            name=graph_class_name(runner.name, bucket, layout),
            block=aot_mint.keying_block(traced.program, traced.ingress, spec),
            nodes=len(list(traced.program.graph_module.graph.nodes)),
            program=traced.program,
            declared=len(variants_of(family)),
        )
        yield traced, spec, row


def mint_family(
    family: GraphFamily,
    *,
    out_dir: Path,
    work: Path,
    cache_root: Path | None = None,
    only: Sequence[str] = (),
    on_class: Any = None,
) -> tuple[MintedVariant, ...]:
    """Compile and pack every declared graph class of ``family``.

    This is the whole bridge. It opens the worker's own TCG engine — the same
    one ``aot_compile_child`` opens, with the same sealed toolchain facts, so
    the artifacts it produces are keyed identically to the ones the endpoint
    mint lane produces and adopt by the same route.

    Refuses on a plain :class:`~gen_worker.family.spec.Family`: an eager-only
    family has no graph classes, and minting one would mean inventing them.
    """

    if not isinstance(family, GraphFamily):
        raise FamilyError(
            FamilyRefusal.FAMILY_INVALID,
            f"family {getattr(family, 'name', family)!r} is eager-only and declares no "
            "graph classes; declare a GraphFamily() before asking for a mint",
        )
    from .. import aot_compile_child

    out_dir.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    engine, runtime = aot_compile_child._tcg_runtime(cache_root)
    minted: list[MintedVariant] = []
    for traced, spec, row in traced_classes(family, only=only):
        if on_class is not None:
            on_class(row.name, len(minted), row.declared)
        result = aot_compile_child.compile_traced_class(
            row, spec, engine, runtime, work=work, out_dir=out_dir
        )
        packed = result.packed
        minted.append(
            MintedVariant(
                runner=traced.runner,
                bucket=tuple(sorted(traced.bucket.items())),
                layout=traced.layout,
                graph_class=str(packed.name),
                key=str(packed.key),
                artifact=Path(str(packed.artifact)),
                metadata=json.loads(str(packed.metadata)),
                ingress_digest=str(traced.ingress.digest()),
                compile_s=float(result.compile_s),
                reuse_s=float(result.reuse_s),
            )
        )
    return tuple(minted)


def assert_matches_export(
    minted: Sequence[MintedVariant], export: Any
) -> None:
    """Every minted class answers a variant the COMMITTED export declares.

    The drift assertion torchcg G16 asks for, in the direction it asks for it:
    the export is the source and the mint is checked against it. A mint whose
    ingress digest differs from the committed one is an artifact that will arm
    under a signature no handler was type-checked against, which is the failure
    the whole binding scheme exists to make impossible.
    """

    for row in minted:
        declared = export.runner(row.runner).variant(dict(row.bucket), row.layout)
        if str(declared.ingress_digest) != row.ingress_digest:
            raise FamilyError(
                FamilyRefusal.EXPORT_FAILED,
                f"minted class {row.graph_class!r} has ingress digest "
                f"{row.ingress_digest!r}; the committed export declares "
                f"{str(declared.ingress_digest)!r}. Regenerate the export and the "
                "bindings from the declaration, then mint again — the mint never "
                "becomes the source.",
            )


__all__ = [
    "MintedVariant",
    "assert_matches_export",
    "export_spec_for",
    "graph_class_name",
    "mint_family",
    "traced_classes",
    "variants_of",
]
