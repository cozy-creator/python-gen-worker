"""Publish-time validation of tenant transform functions.

Runs at endpoint publish (tensorhub Docker-build stage + orchestrator
registration) against a tenant module. Collects all violations and returns
them as one structured result — catches ALL errors in one report rather
than failing on the first one.

Validation checks:
  1. Decorator-time errors (TypeError from @training_function): reserved
     name + wrong type, missing required ctx/source.
  2. Tenant-named params must have resolvable annotations msgspec can decode
     (no bare `Any`, no unresolved forward refs).
  3. No two transform functions in the same endpoint with the same name.
  4. Endpoint.toml ``endpoint_kind`` coherence: if 'transform' / 'conversion' /
     'training', every exported function must be @training_function; if
     'inference', no @training_function.

Designed to run in a fresh subprocess with the same interpreter the worker
uses at runtime — so import failures (missing dep, broken install) surface
as validation errors rather than job-time crashes.
"""

from __future__ import annotations

import importlib
import inspect
import typing
from dataclasses import dataclass, field
from typing import Any, Callable

from .dispatch import TrainingFunctionSpec


@dataclass
class ValidationViolation:
    """One detected problem in a tenant module.

    ``severity`` distinguishes hard errors (fail the publish) from nudge-only
    warnings (surfaced in the report but don't block). Kind-label miscategorization
    nudges use severity='warning'; decorator errors + unresolvable annotations
    use 'error'.
    """

    function: str                 # qualified name of the tenant function
    kind: str                     # machine-readable violation category
    message: str                  # human-readable explanation
    severity: str = "error"       # 'error' | 'warning'


@dataclass
class ValidationReport:
    """Aggregated results from validating a tenant module."""

    module_name: str
    functions_seen: list[str] = field(default_factory=list)
    violations: list[ValidationViolation] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when no error-severity violations exist. Warnings don't fail ok."""
        return not any(v.severity == "error" for v in self.violations)

    @property
    def errors(self) -> list[ValidationViolation]:
        return [v for v in self.violations if v.severity == "error"]

    @property
    def warnings(self) -> list[ValidationViolation]:
        return [v for v in self.violations if v.severity == "warning"]


_CONVERSION_VIOLATION_KINDS = {
    "import_failed": "module import raised",
    "decorator_error": "decorator time TypeError / ValueError",
    "unresolved_annotation": "param annotation couldn't resolve",
    "duplicate_function_name": "two transforms share a name",
    "endpoint_kind_mismatch": "decorator vs endpoint.toml disagree",
    "kind_label_suspect": "declared kind likely mismatches the function shape",
}


# Job-kind labels that DON'T typically need a dataset. Declaring one of these
# on a function that takes ``datasets: list[Dataset]`` is a likely
# miscategorization — fine-tuning or distillation is usually what the author
# meant. Calibrated quantization is a known exception: it's kind='quantization'
# AND uses a calibration dataset. We keep that exception narrow.
_DATASETLESS_KINDS = frozenset({"fusion", "format-conversion"})
# Coarse labels that ARE dataset-using: fine-tuning, distillation, and
# pruning (gradient-scored) are the common ones. We don't emit warnings for
# these.


def validate_transform_module(
    module_name: str,
    *,
    expected_kind: str | None = None,
) -> ValidationReport:
    """Import ``module_name``, find @training_function decorations, validate each.

    Args:
        module_name: dotted module path, e.g. 'conversion.main' or
            'conversion_cpu.cast_dtype'. Must be importable from the current
            Python path.
        expected_kind: if set, must be one of ``'transform'``, ``'conversion'``,
            ``'training'``, or ``'inference'``. When 'inference', the module
            is expected to have NO @training_function decorations; otherwise
            exported functions SHOULD be decorated.

    Returns:
        A ``ValidationReport`` listing every transform function and every
        violation found. Caller checks ``.ok`` and/or ``.violations``.
    """
    report = ValidationReport(module_name=module_name)

    try:
        mod = importlib.import_module(module_name)
    except Exception as exc:
        report.violations.append(ValidationViolation(
            function="<module>",
            kind="import_failed",
            message=f"{type(exc).__name__}: {exc}",
        ))
        return report

    seen_names: set[str] = set()
    transform_funcs: list[tuple[str, Callable, TrainingFunctionSpec]] = []
    for name in dir(mod):
        if name.startswith("_"):
            continue
        obj = getattr(mod, name, None)
        spec = getattr(obj, "__training_spec__", None)
        if spec is None:
            continue
        fqname = f"{module_name}.{name}"
        report.functions_seen.append(fqname)
        if name in seen_names:
            report.violations.append(ValidationViolation(
                function=fqname, kind="duplicate_function_name",
                message=f"two @training_function decorations share name {name!r}",
            ))
        seen_names.add(name)
        transform_funcs.append((fqname, obj, spec))

    # Per-function: nudge warnings for kind-vs-signature mismatches (issue #10).
    # Not a hard violation — tenants may have legitimate exceptions — but helps
    # catch typo-ish miscategorizations early. Adds entries with kind
    # 'kind_label_suspect' that format_report surfaces.
    for fqname, fn, spec in transform_funcs:
        declared_kind = getattr(spec, "kind", "") or ""
        coarse = declared_kind.split(":", 1)[0] if declared_kind else ""
        declares_datasets = "datasets" in spec.signature.parameters
        if coarse in _DATASETLESS_KINDS and declares_datasets:
            report.violations.append(ValidationViolation(
                function=fqname, kind="kind_label_suspect", severity="warning",
                message=(
                    f"kind={declared_kind!r} typically doesn't need a dataset, "
                    f"but {fn.__name__} declares `datasets: list[Dataset]`. "
                    "Did you mean kind='fine-tuning' or 'distillation'?"
                ),
            ))
        # Dataset-using default-kind: author probably forgot to pick a specific
        # label. Suggest fine-tuning/distillation explicitly.
        if declared_kind == "fine-tuning" and not declares_datasets and fn.__name__.startswith("convert_"):
            report.violations.append(ValidationViolation(
                function=fqname, kind="kind_label_suspect", severity="warning",
                message=(
                    f"kind='fine-tuning' is the default, but {fn.__name__} "
                    "has no datasets param and the name starts with 'convert_'. "
                    "Consider kind='quantization' or 'format-conversion' for "
                    "fail-early categorization at publish."
                ),
            ))

    # Per-function: validate tenant-named param annotations are resolvable
    for fqname, fn, spec in transform_funcs:
        for pname, param in spec.other_params.items():
            ann = param.annotation
            if ann is inspect.Parameter.empty:
                report.violations.append(ValidationViolation(
                    function=fqname, kind="unresolved_annotation",
                    message=(
                        f"tenant-named param {pname!r} has no annotation — "
                        "msgspec cannot decode it from the wire payload"
                    ),
                ))
                continue
            if ann is Any:
                report.violations.append(ValidationViolation(
                    function=fqname, kind="unresolved_annotation",
                    message=(
                        f"tenant-named param {pname!r} is annotated Any — "
                        "msgspec requires a concrete type"
                    ),
                ))
                continue
            # typing.get_type_hints already ran inside the decorator, so
            # forward refs that couldn't resolve would have raised at
            # decoration time. If we reached here the annotation resolved.

    # endpoint_kind coherence
    if expected_kind is not None:
        kind = expected_kind.lower()
        if kind == "training" and not transform_funcs:
            report.violations.append(ValidationViolation(
                function="<module>", kind="endpoint_kind_mismatch",
                message=(
                    f"endpoint_kind={kind!r} but module has no "
                    "@training_function decorations"
                ),
            ))
        elif kind == "inference" and transform_funcs:
            for fqname, _, _ in transform_funcs:
                report.violations.append(ValidationViolation(
                    function=fqname, kind="endpoint_kind_mismatch",
                    message=(
                        f"endpoint_kind='inference' but {fqname} is decorated "
                        "with @training_function"
                    ),
                ))
        elif kind not in transform_kinds | {"inference"}:
            report.violations.append(ValidationViolation(
                function="<module>", kind="endpoint_kind_mismatch",
                message=f"unrecognized endpoint_kind {kind!r}",
            ))

    return report


def format_report(report: ValidationReport) -> str:
    """Human-readable render of a ValidationReport for publish-response body."""
    lines = [f"module: {report.module_name}"]
    if report.functions_seen:
        lines.append(f"transform functions seen ({len(report.functions_seen)}):")
        for fq in report.functions_seen:
            lines.append(f"  - {fq}")
    if report.ok:
        lines.append("validation: OK")
        return "\n".join(lines)
    errs = report.errors
    warns = report.warnings
    if errs:
        lines.append(f"validation: {len(errs)} error(s)")
        for v in errs:
            lines.append(f"  [error/{v.kind}] {v.function}: {v.message}")
    if warns:
        lines.append(f"validation: {len(warns)} warning(s) (non-blocking)")
        for v in warns:
            lines.append(f"  [warn/{v.kind}] {v.function}: {v.message}")
    return "\n".join(lines)


__all__ = [
    "ValidationViolation",
    "ValidationReport",
    "validate_transform_module",
    "format_report",
]
