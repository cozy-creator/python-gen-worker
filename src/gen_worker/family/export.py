"""Declaration-time export: a family declaration becomes ``family_export_v1``.

This is the step torchcg G16 puts BEFORE codegen and before any mint. It runs
``torch.export`` over each declared (runner, bucket, layout) with **fake**
tensors, reads the resulting ``CallIngress`` off the exported program, and
writes the snapshot the binding generator consumes.

Three properties are the whole point, and each one is a thing that used to be
impossible:

1. **No GPU, no weights, no compile.** Every parameter and buffer is created
   inside a ``FakeTensorMode``, so a family of any size costs shape arithmetic
   and nothing else. A new family therefore type-checks in CI, on a PR, before
   anybody rents a card — which is what makes the catalog reviewable.
2. **No network.** Nothing here fetches a checkpoint or a config; the
   declaration's own ``build`` constructs the architecture. Identity is
   checkpoint-free (DESIGN-RULINGS §4.27), so weights would add nothing to it.
3. **The export is the SOURCE, not a check.** The bindings a handler compiles
   against are generated from this document. The mint-emitted ``recipe_v1``
   asserts against it later (``drift.assert_recipe``), which is only meaningful
   because the direction runs this way.

What this module deliberately does NOT do: compile, quantize, place anything on
a device, or decide which bucket serves a live call. The first three are the
mint's (and DESIGN-RULINGS §4.30 puts them on the machine that will use the
cell); the fourth is ``ingress_selection_v1``'s.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator, Mapping
from typing import Any

from .._vendor.torchcg.ingress import CallIngress, IngressError, build_call_ingress
from .._vendor.torchcg.recipe import (
    BucketAxisName,
    FamilyName,
    IngressDigest,
    LayoutContract,
    ParameterName,
    RunnerName,
)
from .errors import FamilyError, FamilyRefusal
from .snapshot import (
    ExportedLoop,
    ExportedOutput,
    ExportedParameter,
    ExportedRunner,
    ExportedScheduler,
    ExportedStage,
    ExportedVariant,
    FamilyExport,
    TunedRef,
)
from .spec import CallExample, GraphFamily, Runner


@contextlib.contextmanager
def fake_structure() -> Iterator[Any]:
    """A ``FakeTensorMode`` whose factory calls allocate nothing.

    Modules are built INSIDE it, so a 12-billion-parameter denoiser costs the
    same as a toy: ``torch.empty``/``torch.zeros``/``nn.Linear`` are all
    intercepted and produce metadata. That is what keeps this runnable on a
    CI box and inside the machine rules — a fake-tensor trace is not a mint.

    The mode carries its own ``ShapeEnv`` because ``torch.export`` needs one to
    allocate symbols for a dynamic axis, and because ``aot_compile`` asserts
    every input belongs to ONE mode — the same discipline
    ``gen_worker.models.structure_only.virtualize`` keeps for the mint path.

    The process-wide default dtype is SAVED AND RESTORED around the trace. A
    declaration builds its architecture at its own compute dtype, and the only
    way to say so to a diffusers constructor is ``torch.set_default_dtype`` —
    ``module.to(dtype)`` refuses on fake parameters ("couldn't swap
    Conv2d.weight"). Leaving that set would make one family's build silently
    decide the next one's dtype, and every export after the first in a catalog
    run would be a different graph for a reason nothing recorded.
    """

    try:
        import torch
        from torch._subclasses.fake_tensor import FakeTensorMode
        from torch.fx.experimental.symbolic_shapes import ShapeEnv
    except ImportError as exc:  # pragma: no cover - torch is an extra
        raise FamilyError(
            FamilyRefusal.EXPORT_FAILED,
            "a declaration-time export needs PyTorch; install the `torch` extra",
        ) from exc
    mode = FakeTensorMode(allow_non_fake_inputs=True, shape_env=ShapeEnv())
    restore = torch.get_default_dtype()
    try:
        with mode, torch.device("cpu"):
            yield mode
    finally:
        torch.set_default_dtype(restore)


def export_variant(
    runner: Runner,
    bucket: Mapping[str, int],
    layout: str,
) -> ExportedVariant:
    """Export ONE (runner, bucket, layout) and read its call ingress.

    The module and the example call are built inside one fake mode, so the
    program's placeholders carry fake metadata and nothing on the machine is
    allocated. Failures are named rather than propagated raw: an author reading
    ``export_failed: runner 'denoiser' at bucket ...`` knows which of a
    family's thirty-six rows refused.
    """

    with fake_structure():
        try:
            module = runner.build(layout)
        except Exception as exc:  # noqa: BLE001 - every build failure is one refusal
            raise FamilyError(
                FamilyRefusal.EXPORT_FAILED,
                f"runner {runner.name!r} build({str(layout)!r}) failed: "
                f"{type(exc).__name__}: {exc}",
            ) from exc
        example = runner.example(bucket, layout)
        if not isinstance(example, CallExample):
            raise FamilyError(
                FamilyRefusal.CLASS_INVALID,
                f"runner {runner.name!r} example() must return a CallExample, got "
                f"{type(example).__name__}",
            )
        program = _export(runner, bucket, layout, module, example)
        try:
            ingress = build_call_ingress(
                program,
                example.params,
                example.args,
                example.kwargs,
                excluded_inputs=example.excluded,
            )
        except IngressError as exc:
            raise FamilyError(
                FamilyRefusal.EXPORT_FAILED,
                f"runner {runner.name!r} at bucket {dict(bucket)!r} layout {str(layout)!r} "
                f"produced no call ingress: {exc}",
            ) from exc
        outputs = _outputs(runner, program)
    return ExportedVariant(
        bucket=tuple(sorted((BucketAxisName(str(k)), int(v)) for k, v in bucket.items())),
        layout=LayoutContract(str(layout)),
        ingress=ingress,
        ingress_digest=IngressDigest(ingress.digest()),
        outputs=outputs,
    )


def _outputs(runner: Runner, program: Any) -> tuple[ExportedOutput, ...]:
    """The tensors this variant returns, read off the exported program.

    Read from the graph's ``output`` node rather than from a run, so nothing is
    executed and nothing is allocated. A non-tensor return (a dataclass field, a
    Python scalar folded into the graph) is REFUSED rather than skipped: a fake
    backing that silently dropped one would hand a handler a shorter tuple than
    the compiled backing does, which is a difference the handler cannot see
    until it indexes past the end.
    """

    graph = getattr(getattr(program, "graph_module", None), "graph", None)
    node = next(
        (item for item in reversed(list(getattr(graph, "nodes", ()) or ())) if item.op == "output"),
        None,
    )
    if node is None:
        raise FamilyError(
            FamilyRefusal.EXPORT_FAILED,
            f"runner {runner.name!r} exported a graph with no output node",
        )
    values = node.args[0] if node.args else ()
    if not isinstance(values, (list, tuple)):
        values = (values,)
    rows: list[ExportedOutput] = []
    for index, item in enumerate(values):
        value = getattr(item, "meta", {}).get("val") if hasattr(item, "meta") else None
        dtype = str(getattr(value, "dtype", "") or "").removeprefix("torch.")
        shape = getattr(value, "shape", None)
        if not dtype or shape is None:
            raise FamilyError(
                FamilyRefusal.EXPORT_FAILED,
                f"runner {runner.name!r} output {index} is not a tensor the export can "
                "describe; a graph class returns tensors",
            )
        rows.append(
            ExportedOutput(
                dtype=dtype,
                shape=tuple(
                    int(str(dim)) if str(dim).lstrip("-").isdigit() else str(dim)
                    for dim in shape
                ),
            )
        )
    return tuple(rows)


def _export(
    runner: Runner,
    bucket: Mapping[str, int],
    layout: str,
    module: Any,
    example: CallExample,
) -> Any:
    import torch

    try:
        return torch.export.export(
            module,
            tuple(example.args),
            dict(example.kwargs),
            dynamic_shapes=None if example.dynamic is None else dict(example.dynamic),
            strict=True,
        )
    except Exception as exc:  # noqa: BLE001 - a refusal, not a crash to propagate
        raise FamilyError(
            FamilyRefusal.EXPORT_FAILED,
            f"torch.export failed for runner {runner.name!r} at bucket {dict(bucket)!r} "
            f"layout {str(layout)!r}: {type(exc).__name__}: {exc}",
        ) from exc


def _tuned_ref(schema: type) -> TunedRef:
    return TunedRef(module=schema.__module__, qualname=schema.__qualname__)


def export_family(family: GraphFamily) -> FamilyExport:
    """Export every declared variant of ``family`` into one snapshot.

    Variants are exported in the declaration's own canonical order (runner,
    then layout, then bucket), so the document — and therefore its digest and
    the generated binding — is a function of the declaration alone. Two
    machines running this on the same source produce the same bytes.
    """

    runners: list[ExportedRunner] = []
    for runner in family.runners:
        variants: list[ExportedVariant] = []
        for layout in runner.layouts:
            for bucket in runner.buckets(family.axis_values):
                variants.append(export_variant(runner, bucket, layout))
        runners.append(
            ExportedRunner(
                name=RunnerName(runner.name),
                axes=tuple(BucketAxisName(name) for name in runner.axes),
                variants=tuple(sorted(variants, key=lambda item: item.selector)),
            )
        )
    loop = family.staged_loop
    scheduler = family.scheduler
    return FamilyExport(
        family=FamilyName(family.name),
        buckets=tuple(
            (BucketAxisName(bucket.name), bucket.values) for bucket in family.buckets
        ),
        runners=tuple(runners),
        loop=ExportedLoop(
            kind=loop.kind,
            session_state=loop.session_state,
            stages=tuple(
                ExportedStage(
                    runner=RunnerName(stage.runner),
                    repeat=stage.kind,
                    parameter=ParameterName(stage.repeat) if stage.repeat else None,
                )
                for stage in loop.stages
            ),
        ),
        tuned=_tuned_ref(family.tuned),
        parameters=tuple(
            ExportedParameter(
                name=ParameterName(parameter.name),
                minimum=parameter.minimum,
                maximum=parameter.maximum,
            )
            for parameter in family.parameters
        ),
        scheduler=None
        if scheduler is None
        else ExportedScheduler(
            name=scheduler.name,
            parameters=tuple(
                (ParameterName(name), value) for name, value in scheduler.parameters.items()
            ),
        ),
        lora_tuned=None if family.lora_tuned is None else _tuned_ref(family.lora_tuned),
    )


def ingress_of(export: FamilyExport, runner: str, bucket: Mapping[str, int],
               layout: str | None = None) -> CallIngress:
    """The exact declared ingress for one variant — an exact lookup, never a rank."""

    return export.runner(runner).variant(bucket, layout).ingress


__all__ = [
    "export_family",
    "export_variant",
    "fake_structure",
    "ingress_of",
]
