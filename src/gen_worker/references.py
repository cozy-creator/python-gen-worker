"""Edit/compose reference handling (ie#600) — mode, named references, caps.

An edit model is a GENERATOR THAT TAKES REFERENCES. The pipeline sees one
undifferentiated ordered list, so "which image is being edited" exists only
as OUR declaration, and it is exactly what :mod:`gen_worker.geometry` needs
to know whose framing to preserve. Hence two functions, never one inferred
mode::

    edit(image, references=[...])   # image carries the framing
    compose(references=[...])       # output geometry is free

The library owns the MECHANISM; the family declares the DATA
(:class:`EditFamily`) — the same split :class:`~gen_worker.geometry.FamilyGeometry`
uses. Only four things vary per family: the reference budget, the rendered
positional label (**or none**), whether an edit-target slot exists, and how
many references the pipeline appends behind our back.

Named references (``{pose}``) are a REQUEST-BOUNDARY REWRITE into the label
the family was conditioned on. They are never a model input: no pipeline in
the fleet has a per-image name channel.

**The label domain admits "none", and that is load-bearing (ie#603).** The
three families have different index channels:

* Qwen-Image-Edit: ``Picture N:`` is literally in the token stream, plus a
  per-reference RoPE block. Rendered label ``Picture {n}``.
* FLUX.2 Klein: one channel only (the ``t_coords`` RoPE offset); the text
  encoder is text-only. ``image {n}`` is a BFL-documented USER convention
  whose training basis is unpublished.
* HiDream-O1: no index channel at all, and its report §2.4 publishes an
  entity-description caption recipe — the opposite of an ordinal. Rendering
  ``image 2`` there would invent a convention against a published
  counter-indication, so its label is ``None`` and ``{name}`` is REFUSED.

DreamOmni2 (arXiv 2510.06679 §3.2) is why this matters: *"in DiT, positional
encoding alone cannot accurately distinguish the index of reference images"*.
Positional prose does not bind for free.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Mapping, Optional, Sequence, Union

import msgspec

from .api.errors import ValidationError
from .geometry import FamilyGeometry, FitMode, FitPlan, OutputSize, fit_to_native

__all__ = [
    "EditFamily",
    "Reference",
    "PromptRewrite",
    "EditRequest",
    "ReferenceRefusal",
    "NAME_GRAMMAR",
    "canonical_name",
    "normalize_references",
    "rewrite_prompt",
    "plan_edit",
    "fit_edit_request",
    "condition_tokens",
]

#: ``{name}`` grammar: lowercase canonical, 2-32 chars. Lowercase-canonical so
#: ``{Face}`` and ``{face}`` cannot be two references. Runway's 3-16 is
#: needlessly tight.
NAME_GRAMMAR = re.compile(r"^[a-z][a-z0-9_]{1,31}$")

# One scan handles escapes and placeholders together: a two-brace run is a
# literal brace, a single brace opens a placeholder. Scanning for `{name}`
# alone would mis-read `{{face}}`.
_TOKEN = re.compile(r"\{\{|\}\}|\{([^{}]*)\}|\{|\}")

ReferenceInput = Union[Sequence[Any], Mapping[str, Any]]


class ReferenceRefusal(ValidationError):
    """A typed, fail-closed refusal at the request boundary.

    ``str(exc)`` is ``"<code>: <detail>"`` so the code survives the
    ``safe_message`` wire hop into the request's ``error.code`` — the
    :class:`~gen_worker.api.errors.PayloadRefError` shape. Every refusal here
    happens BEFORE upload, billing or dispatch: a silently-unsubstituted
    ``{face}`` reaching the text encoder is the worst outcome, a plausible
    render that ignored the user's intent.
    """

    def __init__(self, detail: str, *, code: str, name: str = "") -> None:
        self.code = code
        self.name = str(name or "")
        super().__init__(f"{code}: {detail}")


def canonical_name(raw: str) -> str:
    """NFC-normalize + casefold a reference name for lookup and dedup."""
    return unicodedata.normalize("NFC", str(raw or "")).strip().casefold()


class EditFamily(msgspec.Struct, frozen=True, kw_only=True):
    """One family's reference facts. DATA, declared by the endpoint package.

    ``max_references`` is **OUR cost/shape budget, never a quality limit**. We
    self-host: no vendor number binds us, and no published count is a
    capability statement (ie#605 traced every one of ours to a competitor's
    product decision or a test-set artifact). It stays FINITE for two reasons
    that ARE ours — each ``N`` is a distinct compiled graph (ie#550) and VRAM.
    ``budget_basis`` must say so in the family's own words; it is rendered into
    the over-cap refusal, so the caller reads the real reason.

    ``reference_label`` is a ``str.format``-style template taking ``n`` (the
    1-based position in the marshalled list), or ``None`` for a family with no
    index channel — see the module docstring. ``None`` makes ``{name}``
    references a refusal rather than an invented ordinal.

    ``pipeline_appended_references`` is the count the PIPELINE adds behind our
    back for a given call shape (HiDream's ``create_layout_reference_images``
    appends a synthetic layout canvas when ``layout_bboxes`` is set). It is
    passed per request, not declared here, because it depends on the payload —
    this field is only the family's declared maximum, for documentation.
    """

    name: str
    #: Total references the model may see, INCLUDING the edit target and any
    #: pipeline-appended synthetic reference (see :func:`plan_edit`).
    max_references: int
    #: Why THIS number, in our terms. Required — an uncited cap is how three
    #: foreign product decisions became our schema (ie#605).
    budget_basis: str
    #: ``"Picture {n}"`` / ``"image {n}"`` / ``None`` (no index channel).
    reference_label: Optional[str] = None
    #: Provenance of the label. Required when a label exists.
    label_basis: str = ""
    #: False for a family that has no edit-target slot (compose-only).
    has_edit_target: bool = True
    #: Fixed latent tokens each reference adds, when it IS fixed. ``None`` for
    #: a family that resizes references as the count grows (HiDream).
    marginal_latent_tokens: Optional[int] = None
    #: Text-stream tokens each reference adds (Qwen's ~188 vision tokens).
    marginal_text_tokens: int = 0
    #: Largest number of synthetic references the pipeline may append.
    pipeline_appended_references: int = 0

    def __post_init__(self) -> None:
        if not str(self.name or "").strip():
            raise ValueError("EditFamily requires a name")
        if self.max_references < 1:
            raise ValueError(f"{self.name}: max_references must be >= 1")
        if not str(self.budget_basis or "").strip():
            raise ValueError(
                f"{self.name}: budget_basis is required — a reference cap must "
                "state OUR cost/shape reason, never a vendor's number (ie#605)"
            )
        if self.reference_label is not None:
            if "{n}" not in self.reference_label:
                raise ValueError(
                    f"{self.name}: reference_label {self.reference_label!r} must "
                    "contain '{n}', the 1-based marshalled position"
                )
            if not str(self.label_basis or "").strip():
                raise ValueError(
                    f"{self.name}: a rendered label needs label_basis — whether it "
                    "is proven in the token stream or a vendor-documented "
                    "convention is exactly what a reader needs (ie#603)"
                )

    @property
    def labels_positions(self) -> bool:
        """Whether ``{name}`` can be rewritten at all for this family."""
        return self.reference_label is not None

    def render_label(self, position: int) -> str:
        if self.reference_label is None:
            raise ReferenceRefusal(
                f"{self.name} has no positional reference label, so a name cannot "
                "be rendered",
                code="positional_labels_unsupported",
            )
        return self.reference_label.format(n=int(position))


class Reference(msgspec.Struct, frozen=True, kw_only=True):
    """One reference, with the position the model will see it at."""

    #: Canonical name, or ``None`` in list form.
    name: Optional[str]
    asset: Any
    #: 1-based position in the marshalled list (the edit target is 1).
    position: int


class PromptRewrite(msgspec.Struct, frozen=True, kw_only=True):
    """Both prompts, because a run must be reproducible from its record.

    Every edit-quality artifact we own binds by INSTRUCTION (te#156) — bind
    those to ``original``, and stamp ``rendered`` as provenance.
    """

    original: str
    rendered: str
    #: ``(name, position)`` pairs actually substituted, in prompt order.
    substitutions: tuple[tuple[str, int], ...] = ()
    #: Declared names the prompt never mentions. Legitimate (an untagged style
    #: reference is a real workflow) — the endpoint should emit a warning
    #: event, not refuse.
    unreferenced: tuple[str, ...] = ()

    @property
    def rewritten(self) -> bool:
        return self.rendered != self.original


class EditRequest(msgspec.Struct, frozen=True, kw_only=True):
    """One normalized edit/compose request: what the model actually receives."""

    family: EditFamily
    mode: FitMode
    prompt: PromptRewrite
    #: The marshalled ordered list. In ``edit`` mode index 0 is the target.
    images: tuple[Any, ...]
    references: tuple[Reference, ...]
    #: Synthetic references the pipeline will append, charged to the cap.
    pipeline_appends: int = 0

    @property
    def named(self) -> bool:
        return any(r.name is not None for r in self.references)

    @property
    def total_references(self) -> int:
        return len(self.images) + int(self.pipeline_appends)

    @property
    def primary(self) -> int:
        """Index of the framing-carrying image; ``0`` in edit, unused in compose."""
        return 0


def normalize_references(
    references: Optional[ReferenceInput],
    *,
    offset: int = 0,
) -> tuple[Reference, ...]:
    """List OR name->asset map to ONE ordered list — all the model has.

    List form keeps the caller's order. **Map form is ordered by sorted key**,
    matching what gen-worker's asset walker and the hub's ``*`` map-values
    traversal already do, so ordering agrees with the transport layer instead
    of inventing a second convention.

    ``offset`` is how many images precede these in the marshalled list (1 when
    an edit target occupies position 1).
    """
    if references is None:
        return ()
    if isinstance(references, Mapping):
        canonical: dict[str, Any] = {}
        for raw_name, asset in references.items():
            name = canonical_name(raw_name)
            if not NAME_GRAMMAR.match(name):
                raise ReferenceRefusal(
                    f"reference name {raw_name!r} is not a legal name: 2-32 chars "
                    "matching [a-z][a-z0-9_]*, case-insensitive",
                    code="malformed_reference_name", name=str(raw_name),
                )
            if name in canonical:
                raise ReferenceRefusal(
                    f"reference name {name!r} is declared twice: names are "
                    f"case-insensitive and NFC-normalized, so {raw_name!r} "
                    "collides with an earlier key",
                    code="duplicate_reference_name", name=name,
                )
            canonical[name] = asset
        return tuple(
            Reference(name=name, asset=canonical[name], position=offset + i + 1)
            for i, name in enumerate(sorted(canonical))
        )
    if isinstance(references, (str, bytes)):
        raise ReferenceRefusal(
            "references must be a list of images or a name->image map",
            code="malformed_references",
        )
    return tuple(
        Reference(name=None, asset=asset, position=offset + i + 1)
        for i, asset in enumerate(references)
    )


def rewrite_prompt(
    prompt: str, references: Sequence[Reference], family: EditFamily
) -> PromptRewrite:
    """Rewrite ``{name}`` into the family's rendered positional label.

    ``{{`` and ``}}`` are literal braces — required, or a prompt containing
    JSON or code becomes unsendable. Any OTHER unescaped brace is significant:
    a lone ``{`` or a ``{not a name}`` is a refusal, never a passthrough,
    because a typo'd placeholder that ships silently produces a confident
    render of the wrong thing.
    """
    original = str(prompt or "")
    declared = {r.name: r for r in references if r.name is not None}

    substitutions: list[tuple[str, int]] = []
    out: list[str] = []
    cursor = 0
    for match in _TOKEN.finditer(original):
        out.append(original[cursor : match.start()])
        cursor = match.end()
        token = match.group(0)
        if token == "{{":
            out.append("{")
            continue
        if token == "}}":
            out.append("}")
            continue
        if token in ("{", "}"):
            raise ReferenceRefusal(
                f"unbalanced {token!r} at offset {match.start()}: braces are "
                "reference placeholders; write '{{' or '}}' for a literal brace",
                code="malformed_reference_name",
            )
        name = canonical_name(match.group(1))
        if not NAME_GRAMMAR.match(name):
            raise ReferenceRefusal(
                f"{{{match.group(1)}}} is not a reference placeholder: names are "
                "2-32 chars matching [a-z][a-z0-9_]*. Write '{{' and '}}' for "
                "literal braces",
                code="malformed_reference_name", name=match.group(1),
            )
        if not declared:
            raise ReferenceRefusal(
                f"the prompt names reference {{{name}}} but the payload sent "
                "references as a LIST, which declares no names — send a "
                "name->image map, or write the position in ordinary prose",
                code="named_references_not_declared", name=name,
            )
        if not family.labels_positions:
            raise ReferenceRefusal(
                f"{family.name} has no positional reference label to rewrite "
                f"{{{name}}} into: its condition tokens carry no index channel and "
                "its published caption recipe is entity descriptions, not ordinals "
                "(ie#603). Name the subjects in the prompt and send references as "
                "a list",
                code="positional_labels_unsupported", name=name,
            )
        reference = declared.get(name)
        if reference is None:
            known = ", ".join(sorted(declared)) or "(none)"
            raise ReferenceRefusal(
                f"the prompt names reference {{{name}}}, which the payload does not "
                f"declare; declared names are: {known}",
                code="unknown_reference_name", name=name,
            )
        out.append(family.render_label(reference.position))
        substitutions.append((name, reference.position))
    out.append(original[cursor:])

    used = {name for name, _ in substitutions}
    return PromptRewrite(
        original=original,
        rendered="".join(out),
        substitutions=tuple(substitutions),
        unreferenced=tuple(sorted(set(declared) - used)),
    )


def plan_edit(
    *,
    family: EditFamily,
    mode: FitMode,
    prompt: str,
    image: Any = None,
    references: Optional[ReferenceInput] = None,
    pipeline_appends: int = 0,
) -> EditRequest:
    """Normalize one edit/compose request. Refuses before dispatch, never truncates.

    ``pipeline_appends`` is charged to the cap, so the count is what the MODEL
    sees: HiDream's ``create_layout_reference_images`` appends a synthetic
    layout canvas when ``layout_bboxes`` is set, and a pre-append cap would let
    that push the pipeline across a ``get_sizes`` bucket and silently shrink
    every reference. The cap exists for compiled-shape count and VRAM — both
    properties of what reaches the model — so it is counted POST-append.
    """
    mode = FitMode(mode)
    if mode is FitMode.EDIT:
        if not family.has_edit_target:
            raise ReferenceRefusal(
                f"{family.name} has no edit-target slot; use compose",
                code="edit_mode_unsupported",
            )
        if image is None:
            raise ReferenceRefusal(
                "edit requires the image being edited: it is what carries the "
                "framing the output preserves (ie#599), and the payload cannot "
                "reveal which reference that would be",
                code="edit_target_missing",
            )
        offset = 1
    else:
        if image is not None:
            raise ReferenceRefusal(
                "compose takes references only: there is no image being edited, "
                "so output geometry is a free parameter",
                code="compose_takes_no_edit_target",
            )
        offset = 0

    normalized = normalize_references(references, offset=offset)
    if mode is FitMode.COMPOSE and not normalized:
        raise ReferenceRefusal(
            "compose requires at least one reference",
            code="references_missing",
        )

    appends = max(0, int(pipeline_appends))
    total = offset + len(normalized) + appends
    if total > family.max_references:
        detail = (
            f"this request sends {offset + len(normalized)} images"
            + (f" and the pipeline appends {appends} more" if appends else "")
            + f", for {total} references; {family.name} accepts at most "
            f"{family.max_references}. {family.budget_basis}"
        )
        raise ReferenceRefusal(detail, code="too_many_references")

    rewrite = rewrite_prompt(prompt, normalized, family)
    images = tuple([image] if offset else []) + tuple(r.asset for r in normalized)
    return EditRequest(
        family=family,
        mode=mode,
        prompt=rewrite,
        images=images,
        references=normalized,
        pipeline_appends=appends,
    )


def fit_edit_request(
    request: EditRequest,
    images: Sequence[Any],
    geometry: FamilyGeometry,
    *,
    output_size: OutputSize = OutputSize.MATCH_INPUT,
    preset: Optional[tuple[int, int]] = None,
) -> FitPlan:
    """The other half of the one stage: fit this request's images to native.

    ``images`` are the decoded PIL images in ``request.images`` order — the
    normalization above is asset-shaped and never opens a file. ``edit`` fits
    the target's framing; ``compose`` refuses to inherit ``references[0]``'s
    aspect and requires an explicit output bucket (ie#599's contract).
    """
    if len(images) != len(request.images):
        raise ValidationError(
            f"fit_edit_request: got {len(images)} decoded images for "
            f"{len(request.images)} marshalled references"
        )
    return fit_to_native(
        images,
        geometry,
        mode=request.mode,
        output_size=output_size,
        preset=preset,
        primary=request.primary,
    )


def condition_tokens(family: EditFamily, count: int) -> Optional[int]:
    """Conditioning tokens ``count`` references add, when the family has a
    constant marginal cost. ``None`` when it does not (HiDream resizes every
    reference as K grows, so its marginal cost is sub-linear by construction).

    The attention term is quadratic in total sequence length, so this is the
    input to a per-reference price, not the price itself: Qwen's fixed
    +4096 latent +~188 text per reference makes 6 references ~12x the 1-reference
    attention term at 1024².
    """
    if family.marginal_latent_tokens is None:
        return None
    n = max(0, int(count))
    return n * (family.marginal_latent_tokens + family.marginal_text_tokens)
