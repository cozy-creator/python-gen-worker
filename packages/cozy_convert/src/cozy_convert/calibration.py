"""Calibration-policy metadata + enforcement helper.

Training functions that quantize weights fall into three buckets:

- ``"required"`` — the recipe produces broken-but-not-erroring weights
  without a calibration forward pass. Example: ``int4_awq``, ``w4a8_awq``.
  No dataset → refuse the job up-front (unless the caller explicitly
  opts into a tiny smoke-test pool via ``allow_dummy``).

- ``"beneficial"`` — the recipe works without calibration but quality
  suffers measurably when skipped. Example: modelopt ``fp8`` / ``int8`` /
  ``nvfp4`` — activation scales default to per-tensor max without a
  forward pass. Default behavior is "use calibration"; callers who value
  speed over quality can set ``skip_calibration=True`` on their spec.

- ``"unsupported"`` — the recipe is weight-only and never consumes
  calibration data. Example: torchao ``int4_wo`` / ``int8_wo`` / ``fp8_wo``,
  bitsandbytes ``nf4`` / ``fp4``. If the caller passes a dataset it's a
  mistake (wasted generation + wrong expectations about output quality);
  fail loudly rather than silently discard.

Tenants declare policy as a ``{scheme: CalibrationPolicy}`` dict passed
to ``@conversion(calibration=...)``. At runtime, the tenant calls
``resolve_calibration_action(policy, ...)`` once per spec entry to decide
whether to calibrate, skip, or run dummy (or raise on invalid combos).

The metadata is ALSO introspectable — discovery bakes it into
``endpoint.lock`` so the orchestrator / UI can tell callers "this function
at scheme=X needs a calibration dataset" before minting capability tokens.

The worker library only decides whether calibration is required, optional, or
unsupported for a requested scheme; callers own how datasets are selected.
"""

from __future__ import annotations

from typing import Literal

CalibrationPolicy = Literal["required", "beneficial", "unsupported"]

VALID_POLICIES: frozenset[str] = frozenset({"required", "beneficial", "unsupported"})


CalibrationAction = Literal["calibrate", "skip", "dummy"]
"""What the tenant should do at runtime:

- ``"calibrate"`` — run the calibration forward loop against the dataset.
- ``"skip"`` — weight-only path; no forward_loop.
- ``"dummy"`` — calibration with the built-in smoke pool. Used ONLY in
  tests / CI; production submits never reach this branch because callers
  don't set ``allow_dummy=True``.
"""


def validate_policy_map(fn_name: str, policy_map) -> dict[str, CalibrationPolicy]:
    """Validate a tenant-supplied calibration policy.

    Accepts either:
      - a single policy string (e.g. ``"unsupported"``) — applies to every
        scheme this function handles. Stored as ``{"*": <policy>}``.
      - a ``{scheme: policy}`` dict for per-scheme differences.

    Raises TypeError when any value isn't one of ``VALID_POLICIES``.
    """
    if isinstance(policy_map, str):
        if policy_map not in VALID_POLICIES:
            raise TypeError(
                f"{fn_name}: @conversion calibration={policy_map!r} "
                f"is not valid. Use one of {sorted(VALID_POLICIES)}."
            )
        return {"*": policy_map}  # type: ignore[dict-item,return-value]
    if not isinstance(policy_map, dict):
        raise TypeError(
            f"{fn_name}: @conversion calibration= must be a string or "
            f"dict[scheme_name, policy]; got {type(policy_map).__name__}"
        )
    for scheme, policy in policy_map.items():
        if not isinstance(scheme, str) or not scheme:
            raise TypeError(
                f"{fn_name}: @conversion calibration= keys must be "
                f"non-empty strings; got {scheme!r}"
            )
        if policy not in VALID_POLICIES:
            raise TypeError(
                f"{fn_name}: @conversion calibration[{scheme!r}]="
                f"{policy!r} is not valid. Use one of {sorted(VALID_POLICIES)}."
            )
    return dict(policy_map)  # type: ignore[return-value]


def lookup_policy(
    policy_map: dict[str, CalibrationPolicy], scheme: str
) -> CalibrationPolicy | None:
    """Resolve the calibration policy for ``scheme``. Falls back to the
    ``"*"`` wildcard entry when no scheme-specific policy is registered.
    Returns ``None`` when nothing matches.
    """
    if scheme in policy_map:
        return policy_map[scheme]
    if "*" in policy_map:
        return policy_map["*"]
    return None


def resolve_calibration_action(
    policy: CalibrationPolicy,
    *,
    has_dataset: bool,
    skip_calibration: bool = False,
    allow_dummy: bool = False,
    scheme: str = "",
) -> CalibrationAction:
    """Decide whether to calibrate, skip, or use a dummy pool for one scheme.

    The 3×2×2×2 truth table boils down to:

    ============== =========== =================== =========== =================
    policy         has_dataset skip_calibration    allow_dummy result
    ============== =========== =================== =========== =================
    required       yes         any                 any         calibrate
    required       no          any                 yes         dummy (warn)
    required       no          any                 no          ValueError
    beneficial     yes         true                any         skip (warn)
    beneficial     yes         false               any         calibrate
    beneficial     no          any                 yes         dummy (warn)
    beneficial     no          true                no          skip (warn)
    beneficial     no          false               no          ValueError
    unsupported    yes         any                 any         ValueError
    unsupported    no          any                 any         skip
    ============== =========== =================== =========== =================

    The ``beneficial + no dataset + skip_calibration=False`` row used to fall
    through to a silent weight-only run. We now hard-fail there: the invoker
    asked for a scheme whose policy declares calibration *helpful* but didn't
    supply a dataset and didn't explicitly opt out. Falling through silently
    shipped uncalibrated weights to invokers who didn't realize the difference.
    Force the choice by supplying a calibration dataset, or set
    ``skip_calibration=True`` on the spec.

    Tenants typically take the returned action + any ``WARN-``-tagged notes
    they want to surface. A raise means the caller must fix the request.
    """
    import logging

    _log = logging.getLogger(__name__)
    label = f"[{scheme}]" if scheme else ""

    if policy == "required":
        if has_dataset:
            return "calibrate"
        if allow_dummy:
            _log.warning(
                "calibration%s: allow_dummy=True — running with built-in smoke "
                "pool. DO NOT SHIP.", label,
            )
            return "dummy"
        raise ValueError(
            f"calibration{label}: scheme requires a calibration dataset "
            f"(policy='required'). Supply a calibration dataset, or set "
            f"allow_dummy_calibration=True on the spec for smoke tests. "
            f"See docs/calibration-dataset-schema.md for the dataset shape."
        )

    if policy == "beneficial":
        if has_dataset:
            if skip_calibration:
                _log.warning(
                    "calibration%s: skip_calibration=True — running weight-only "
                    "even though a dataset was supplied. Expect measurable "
                    "quality drop vs a calibrated run.", label,
                )
                return "skip"
            return "calibrate"
        if allow_dummy:
            _log.warning(
                "calibration%s: no dataset supplied but allow_dummy=True — "
                "running with built-in smoke pool. DO NOT SHIP.", label,
            )
            return "dummy"
        if skip_calibration:
            _log.warning(
                "calibration%s: skip_calibration=True — running weight-only. "
                "Expect measurable quality drop vs a calibrated run.", label,
            )
            return "skip"
        raise ValueError(
            f"calibration{label}: scheme has policy='beneficial' but no "
            f"calibration dataset was supplied. Default is calibrate — "
            f"silently falling back to weight-only would ship uncalibrated "
            f"weights to invokers who didn't realize the difference. Pass "
            f"a calibration dataset, or set skip_calibration=True on the spec "
            f"to opt out explicitly."
        )

    if policy == "unsupported":
        if has_dataset:
            raise ValueError(
                f"calibration{label}: scheme is weight-only (policy="
                f"'unsupported') — a calibration dataset is not used. Drop "
                f"the calibration dataset, or switch to a calibrated "
                f"quantization recipe such as int4_awq / w4a8_awq."
            )
        return "skip"

    raise ValueError(f"calibration{label}: unknown policy {policy!r}")


__all__ = [
    "CalibrationAction",
    "CalibrationPolicy",
    "VALID_POLICIES",
    "resolve_calibration_action",
    "validate_policy_map",
]
