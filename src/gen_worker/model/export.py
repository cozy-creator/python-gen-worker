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
compiled graph); the fourth is ``ingress_selection_v1``'s.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
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
from .errors import ModelError, ModelRefusal
from .snapshot import (
    EagerExport,
    ExportedLoop,
    ExportedOutput,
    ExportedParameter,
    ExportedRunner,
    ExportedScheduler,
    ExportedStage,
    ExportedVariant,
    ModelExport,
    TunedRef,
)
from .spec import CallExample, GraphModelSpec, ModelSpec, Runner


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
        raise ModelError(
            ModelRefusal.EXPORT_FAILED,
            "a declaration-time export needs PyTorch; install the `torch` extra",
        ) from exc
    mode = FakeTensorMode(allow_non_fake_inputs=True, shape_env=ShapeEnv())
    restore = torch.get_default_dtype()
    try:
        with mode, torch.device("cpu"):
            yield mode
    finally:
        torch.set_default_dtype(restore)


@dataclass(frozen=True, slots=True)
class TracedVariant:
    """One traced (runner, bucket, layout), with the PROGRAM still attached.

    :func:`export_variant` throws the program away — the snapshot only needs the
    ingress and the output shapes. The mint needs the program itself, and it
    must be the SAME program: an ingress digest derived from a second, subtly
    different trace would make ``drift.assert_recipe`` compare two things that
    were never one (torchcg G16). So there is one trace function and two
    readings of its result, rather than two tracers that agree by inspection.
    """

    runner: str
    bucket: dict[str, int]
    layout: str
    program: Any
    ingress: CallIngress
    outputs: tuple[ExportedOutput, ...]
    example: CallExample

    def snapshot(self) -> ExportedVariant:
        """This trace as the document row it contributes."""

        return ExportedVariant(
            bucket=tuple(
                sorted((BucketAxisName(str(k)), int(v)) for k, v in self.bucket.items())
            ),
            layout=LayoutContract(str(self.layout)),
            ingress=self.ingress,
            ingress_digest=IngressDigest(self.ingress.digest()),
            outputs=self.outputs,
        )


def trace_variant(
    runner: Runner,
    bucket: Mapping[str, int],
    layout: str,
) -> TracedVariant:
    """Trace ONE (runner, bucket, layout) and read its call ingress.

    The module and the example call are built inside one fake mode, so the
    program's placeholders carry fake metadata and nothing on the machine is
    allocated. Failures are named rather than propagated raw: an author reading
    ``export_failed: runner 'denoiser' at bucket ...`` knows which of a
    family's thirty-six rows refused.

    The program is returned STILL ALIVE and the fake mode is closed around it,
    which is the same order the mint lane uses (``aot_mint._export_entry``
    exports inside ``structure_only.fake_mode_of`` and compiles outside it).
    """

    with fake_structure():
        try:
            module = runner.build(layout)
        except Exception as exc:  # noqa: BLE001 - every build failure is one refusal
            raise ModelError(
                ModelRefusal.EXPORT_FAILED,
                f"runner {runner.name!r} build({str(layout)!r}) failed: "
                f"{type(exc).__name__}: {exc}",
            ) from exc
        example = runner.example(bucket, layout)
        if not isinstance(example, CallExample):
            raise ModelError(
                ModelRefusal.CLASS_INVALID,
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
            raise ModelError(
                ModelRefusal.EXPORT_FAILED,
                f"runner {runner.name!r} at bucket {dict(bucket)!r} layout {str(layout)!r} "
                f"produced no call ingress: {exc}",
            ) from exc
        outputs = _outputs(runner, program)
    return TracedVariant(
        runner=runner.name,
        bucket={str(name): int(value) for name, value in bucket.items()},
        layout=str(layout),
        program=program,
        ingress=ingress,
        outputs=outputs,
        example=example,
    )


def export_variant(
    runner: Runner,
    bucket: Mapping[str, int],
    layout: str,
) -> ExportedVariant:
    """One traced variant, as the snapshot row it contributes."""

    return trace_variant(runner, bucket, layout).snapshot()


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
        raise ModelError(
            ModelRefusal.EXPORT_FAILED,
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
            raise ModelError(
                ModelRefusal.EXPORT_FAILED,
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
        raise ModelError(
            ModelRefusal.EXPORT_FAILED,
            f"torch.export failed for runner {runner.name!r} at bucket {dict(bucket)!r} "
            f"layout {str(layout)!r}: {type(exc).__name__}: {exc}",
        ) from exc


def _tuned_ref(schema: type) -> TunedRef:
    return TunedRef(module=schema.__module__, qualname=schema.__qualname__)


def _require_tuned(family: GraphModelSpec) -> type:
    if family.tuned is None:  # pragma: no cover - GraphModelSpec._validate refuses it
        raise ModelError(
            ModelRefusal.TUNED_INVALID,
            f"graph model {family.name!r} carries no tuned schema to export",
        )
    return family.tuned


def export_model(family: GraphModelSpec) -> ModelExport:
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
    return ModelExport(
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
        # A GraphModelSpec always carries one — `GraphModelSpec._validate`
        # refuses `tuned=None`, because the optional tier is the eager one and
        # a graph model's parameters are exactly what a tuned schema names.
        tuned=_tuned_ref(_require_tuned(family)),
        parameters=tuple(
            ExportedParameter(
                name=ParameterName(parameter.name),
                minimum=parameter.minimum,
                maximum=parameter.maximum,
            )
            for parameter in family.parameters
        ),
        # `GraphModelSpec._validate_schedulers` already sorted the set by
        # sampler, which is the order the export's own invariant wants.
        schedulers=tuple(
            ExportedScheduler(
                sampler=sampler,
                name=scheduler.name,
                parameters=tuple(
                    (ParameterName(name), value) for name, value in scheduler.parameters.items()
                ),
            )
            for sampler, scheduler in family.schedulers.items()
        ),
        lora_tuned=None if family.lora_tuned is None else _tuned_ref(family.lora_tuned),
    )


def export_eager_model(model: ModelSpec) -> EagerExport:
    """Export an EAGER declaration into ``eager_model_v1`` (pgw#1346 B5).

    Nothing is traced, because there is nothing to trace: the F3 ruling makes
    an external-binary or non-PyTorch model a permanent eager citizen, and it
    has no graph to export. So this needs no torch, no fake-tensor mode and no
    model library at all — it is a projection of the declaration's own fields,
    which is what lets the eager half of the catalog be authored, committed and
    fenced by a two-minute job.

    A :class:`GraphModelSpec` is REFUSED rather than narrowed to its eager
    fields: it has runners, and silently exporting the smaller document would
    generate bindings with no typed callables for a family that has them.
    """

    if isinstance(model, GraphModelSpec):
        raise ModelError(
            ModelRefusal.FAMILY_INVALID,
            f"model {model.name!r} is a GraphModelSpec; export it with export_model() so "
            "its runners reach the bindings. eager_model_v1 carries no graph classes.",
        )
    requirements: list[tuple[str, str]] = []
    for handle, declared in sorted(model.layout_requirements.items()):
        if declared.recommended_terms().declared():
            raise ModelError(
                ModelRefusal.FAMILY_INVALID,
                f"model {model.name!r} declares a RECOMMENDED requirement for "
                f"{handle!r}. eager_model_v1 carries the compact MINIMUM, which is the "
                "only level with a single-string spelling that round-trips; grow the "
                "document when a declaration genuinely needs the second level.",
            )
        requirements.append((handle, declared.render()))
    return EagerExport(
        family=FamilyName(model.name),
        tuned=None if model.tuned is None else _tuned_ref(model.tuned),
        lora_tuned=None if model.lora_tuned is None else _tuned_ref(model.lora_tuned),
        layouts=tuple(
            (component, tuple(handles))
            for component, handles in sorted((model.layouts or {}).items())
        ),
        layouts_undeclarable=model.layouts_undeclarable,
        layout_requirements=tuple(requirements),
    )


def ingress_of(export: ModelExport, runner: str, bucket: Mapping[str, int],
               layout: str | None = None) -> CallIngress:
    """The exact declared ingress for one variant — an exact lookup, never a rank."""

    return export.runner(runner).variant(bucket, layout).ingress


__all__ = [
    "TracedVariant",
    "export_eager_model",
    "export_model",
    "export_variant",
    "fake_structure",
    "ingress_of",
    "trace_variant",
]
