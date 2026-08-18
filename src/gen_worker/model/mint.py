"""A family DECLARATION mints its own graph classes. No endpoint, no weights.

pgw#1331's first two bullets ask for the VAE decoder and the text encoders to be
minted as graph classes. Declaring them made them exportable; this module is
what makes them MINTABLE — the bridge from a :class:`~gen_worker.model.spec.
GraphModelSpec` to a packed AOTI artifact, through the mint lane's own inner seam.

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

**The trace is the declaration's own.** :func:`gen_worker.model.export.
trace_variant` produces the program, and the snapshot in the catalog is
produced from that same function — so the mint's ingress digest and the
committed export's ingress digest are the same number by construction, not by
comparison. That is torchcg G16's direction: the recipe ASSERTS against the
declaration, it never becomes the source.

**No checkpoint is downloaded, and that is the ruling, not a shortcut.**
Compiled graph identity is graph x sm x toolchain — checkpoint-free (DESIGN-RULINGS
§4.27), which is exactly what lets one compiled graph serve sixteen fine-tunes.
The declaration builds the ARCHITECTURE inside a fake-tensor mode, the program
carries fake constants, and the real weights arrive at ARM time by manifest FQN
out of the constant store (pgw#1329's ``arm_compiled_graph_from_store``). So
minting a family costs a card and a compile, never a 24 GB download.

**This module is MINT MACHINERY** and is declared as such in
:data:`gen_worker.serve.role.MINT_MACHINERY`: an adopt-only pod that could
reach it could compile, which is the one thing that role is defined by not
being able to do (§4.28). The family SURFACE it mints for is on the serve path;
this is not.

Run it with ``gen-worker model mint`` — **on a pod, never on a shared box**.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..aot_inputs import ExportSpec
from .errors import ModelError, ModelRefusal
from .export import TracedVariant, trace_variant
from .spec import GraphModelSpec, Runner


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
    family: GraphModelSpec, runner: Runner, bucket: Mapping[str, int], layout: str
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
    family: GraphModelSpec,
    *,
    only: Sequence[str] = (),
    buckets: Mapping[str, Sequence[int]] | None = None,
) -> tuple[tuple[Runner, dict[str, int], str], ...]:
    """Every (runner, bucket, layout) to mint, in the declaration's own order.

    ``only`` narrows to named runners — the shape a pod uses when it is minting
    the cheap classes first and the 12-billion-parameter one last, or when a
    previous attempt already banked half the set. A name the family does not
    declare is REFUSED rather than silently minting nothing.

    ``buckets`` narrows the AXIS VALUES, and it exists because pgw#1346 B2 made
    the gap expensive: SDXL's shape axis carries the endpoint's nine real
    buckets, so a ``--runner denoiser`` mint is NINE compiles. A gauntlet row
    proving one card serves one shape wants one of them, and paying 9x for it
    is money spent on classes the row will never adopt. Same refusal
    discipline: an axis the family does not declare, or a value outside its
    closed set, is refused rather than silently minting nothing.
    """

    rows = family.variants()
    if only:
        wanted = {str(name) for name in only}
        unknown = sorted(wanted - {runner.name for runner in family.runners})
        if unknown:
            raise ModelError(
                ModelRefusal.FAMILY_INVALID,
                f"family {family.name!r} declares no runner {unknown[0]!r}; it has "
                f"{sorted(runner.name for runner in family.runners)!r}",
            )
        rows = tuple(row for row in rows if row[0].name in wanted)
    if not buckets:
        return rows
    declared = family.axis_values
    for axis, values in buckets.items():
        if axis not in declared:
            raise ModelError(
                ModelRefusal.BUCKET_AXIS_INVALID,
                f"family {family.name!r} declares no bucket axis {axis!r}; it has "
                f"{sorted(declared)!r}",
            )
        outside = sorted(set(values) - set(declared[axis]))
        if outside:
            raise ModelError(
                ModelRefusal.BUCKET_AXIS_INVALID,
                f"bucket axis {axis!r} has no value {outside[0]!r}; it declares "
                f"{list(declared[axis])!r}",
            )
    return tuple(
        row
        for row in rows
        # A runner that does not DECLARE the narrowed axis is kept: SDXL's text
        # towers have no shape axis, and dropping them would turn "mint one
        # shape" into "mint no text encoder", which is a different request.
        if all(
            axis not in row[1] or row[1][axis] in set(values)
            for axis, values in buckets.items()
        )
    )


def traced_classes(
    family: GraphModelSpec,
    *,
    only: Sequence[str] = (),
    buckets: Mapping[str, Sequence[int]] | None = None,
) -> Iterator[tuple[TracedVariant, ExportSpec, Any]]:
    """Trace every selected variant, yielding one at a time.

    A GENERATOR, and that is the memory bound: an exported program for a
    12-billion-parameter denoiser is large even with fake constants, and the
    caller compiles and releases each row before the next is traced. Building
    the whole list first is how a mint pod runs out of host RAM on the family
    it was bought for.
    """

    from .. import aot_mint

    selected = variants_of(family, only=only, buckets=buckets)
    # The count THIS RUN will compile, not the family's whole declared set: the
    # phase table's "[3/20]" is a progress bar, and a narrowed mint that
    # counted to 20 while compiling 3 reads as a mint that silently skipped 17.
    declared = len(selected)
    for runner, bucket, layout in selected:
        traced = trace_variant(runner, bucket, layout)
        spec = export_spec_for(family, runner, bucket, layout)
        row = aot_mint.TracedClass(
            name=graph_class_name(runner.name, bucket, layout),
            block=aot_mint.keying_block(traced.program, traced.ingress, spec),
            nodes=len(list(traced.program.graph_module.graph.nodes)),
            program=traced.program,
            declared=declared,
        )
        yield traced, spec, row


def mint_model(
    family: GraphModelSpec,
    *,
    out_dir: Path,
    work: Path,
    cache_root: Path | None = None,
    only: Sequence[str] = (),
    buckets: Mapping[str, Sequence[int]] | None = None,
    on_class: Any = None,
) -> tuple[MintedVariant, ...]:
    """Compile and pack every declared graph class of ``family``.

    This is the whole bridge. It opens the worker's own TCG engine — the same
    one ``aot_compile_child`` opens, with the same sealed toolchain facts, so
    the artifacts it produces are keyed identically to the ones the endpoint
    mint lane produces and adopt by the same route.

    Refuses on a plain :class:`~gen_worker.model.spec.ModelSpec`: an eager-only
    family has no graph classes, and minting one would mean inventing them.
    """

    if not isinstance(family, GraphModelSpec):
        raise ModelError(
            ModelRefusal.FAMILY_INVALID,
            f"family {getattr(family, 'name', family)!r} is eager-only and declares no "
            "graph classes; declare a GraphModelSpec() before asking for a mint",
        )
    from .. import aot_compile_child

    out_dir.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    engine, runtime = aot_compile_child._tcg_runtime(cache_root)
    minted: list[MintedVariant] = []
    for traced, spec, row in traced_classes(family, only=only, buckets=buckets):
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
            raise ModelError(
                ModelRefusal.EXPORT_FAILED,
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
    "mint_model",
    "traced_classes",
    "variants_of",
]
