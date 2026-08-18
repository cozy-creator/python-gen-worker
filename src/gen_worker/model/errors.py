"""The closed refusal set for the typed ModelSpec SDK.

Every refusal this package can produce is a member of :class:`ModelRefusal`.
It is closed for the same reason torchcg's ``RecipeRefusal`` is: a second
implementation (``gen-worker-rs``'s ``build.rs``, pgw#1326 Stage 3) must be
able to report the SAME reason for the same declaration, and a reason invented
locally is a rung nobody else agrees on.

The split against torchcg is deliberate. ``RecipeRefusal`` names what is wrong
with a MINT-EMITTED document; this names what is wrong with an AUTHOR'S
DECLARATION, which is a different audience at a different time — declaration
refusals fire at module import on the author's own machine, with no GPU, no
pod and no compile behind them.
"""

from __future__ import annotations

from enum import StrEnum


class ModelRefusal(StrEnum):
    """Every reason the ModelSpec SDK refuses a declaration, export or binding."""

    #: A family, runner, bucket-axis, parameter or scheduler name is not a
    #: legal generated symbol (torchcg's ``IDENTIFIER_GRAMMAR``), or is on the
    #: frozen Python/Rust reserved list.
    IDENTIFIER_INVALID = "identifier_invalid"
    #: Two declarations claim one family name.
    FAMILY_DUPLICATE = "family_duplicate"
    #: The declaration's own field set is malformed.
    FAMILY_INVALID = "family_invalid"
    #: A bucket axis declares no values, unsorted values, or non-positive ones.
    BUCKET_AXIS_INVALID = "bucket_axis_invalid"
    #: A runner buckets on an axis the family never declared, or pins a value
    #: the axis does not carry.
    BUCKET_INVALID = "bucket_invalid"
    #: A runner has no variant for some (bucket, layout) combination, so a
    #: generated ``Literal`` would not be exhaustive (torchcg G6/G15).
    BUCKET_COVERAGE_INCOMPLETE = "bucket_coverage_incomplete"
    #: A graph-class declaration is malformed (no build, no example, ...).
    CLASS_INVALID = "class_invalid"
    #: The loop stages a runner the family does not declare, or leaves one
    #: unstaged, or gives a host-owned loop a repeat count.
    LOOP_INVALID = "loop_invalid"
    #: A counted stage reads an undeclared parameter, or a declared parameter
    #: is read by no stage.
    PARAMETER_INVALID = "parameter_invalid"
    #: The scheduler block is not a name plus finite JSON scalars.
    SCHEDULER_INVALID = "scheduler_invalid"
    #: A checkpoint is stamped with a SAMPLER this family's declaration does
    #: not carry a scheduler for (pgw#1346 K10). Never a fallback: serving the
    #: family's other schedule would silently render a different image than the
    #: recipe asked for, which is the whole reason the set is exhaustive.
    SCHEDULER_UNDECLARED = "scheduler_undeclared"
    #: The declaration-time export could not produce a ``CallIngress``.
    EXPORT_FAILED = "export_failed"
    #: Two variants of one runner project onto different call signatures
    #: (torchcg G2): one runner is one binding, always.
    SIGNATURE_DISAGREEMENT = "signature_disagreement"
    #: An export snapshot states a version this reader does not implement.
    SNAPSHOT_VERSION_UNSUPPORTED = "snapshot_version_unsupported"
    #: An export snapshot's fields are malformed or noncanonical.
    SNAPSHOT_INVALID = "snapshot_invalid"
    #: A committed binding module is not what regenerating its snapshot emits.
    BINDING_STALE = "binding_stale"
    #: The mint-emitted recipe disagrees with the declaration the bindings were
    #: generated against (torchcg G16 / ``declaration_drift``).
    DECLARATION_DRIFT = "declaration_drift"
    #: A tuned-values schema is not a closed, frozen struct.
    TUNED_INVALID = "tuned_invalid"
    #: A product-grid axis maps onto a bucket the family does not declare (B5).
    GRID_INVALID = "grid_invalid"
    #: An instance was asked for a backing it does not carry, or a family class
    #: was constructed without one.
    BACKING_MISSING = "backing_missing"
    #: A runner call does not satisfy its declared ingress.
    CALL_INVALID = "call_invalid"
    #: A decode session was used outside its context manager, or twice.
    SESSION_INVALID = "session_invalid"


class ModelError(ValueError):
    """A family declaration, export, binding or instance is not admissible.

    Carries the closed :class:`ModelRefusal` reason as ``.reason`` so callers
    branch on the reason rather than on message text — the same shape
    ``torchcg.RecipeError`` uses, for the same reason.
    """

    def __init__(self, reason: ModelRefusal, detail: str) -> None:
        self.reason = reason
        super().__init__(f"{reason.value}: {detail}")


__all__ = ["ModelError", "ModelRefusal"]
