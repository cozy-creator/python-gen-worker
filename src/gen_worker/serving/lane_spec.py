"""The LANE DECLARATION SURFACE — one place the author writes demand,
residency and fork axes (pgw#1599, implementing pgw#1597 plan 1 + pgw#1598 §2).

What the author writes, and nothing else::

    class SdxlModel(
        Model[SDXL],
        lanes={("sdxl.diffusers@1", "plain.bf16@1"): lane(
            request=const(GiB(1.2)) + per_mp_batch(MiB(220)),
            resident=("vae",),
        )},
        structural={"timestep_dtype": Structural(
            field="scheduler",
            classes={"int64": "dpmpp_2m_karras", "float32": "euler"},
            measured="pgw#1572: set_timesteps(20) per served scheduler — 5 int64 / 3 float32",
        )},
        shapes={"aspect": STATIC},
    ):

Paul's rule, stated once and enforced here: **the author declares only what
only the author knows** — demand SCALING, fork AXES, "my VAE decode will
thrash if streamed" — and the platform DERIVES everything derivable (weight
bytes from the manifest, the capability floor from the QUANT RULE,
launch-residency from the ``ctx.compile`` marks, coefficients from
measurement). A VRAM STRING is none of those things, which is why it is gone.

**pgw#1621 re-keyed what a lane is NAMED BY, and nothing else here.** The key
was a tensorfs v1 ``Contract`` object (``contracts.SDXL_DIFFUSERS_BF16``); it
is now the ``(topology, quant)`` stamp pair, which is what tensor-layout v2
makes identity. The per-lane ``request=`` formula, the residency override, the
fork axes and the refusals are untouched — se#816's surface SHAPE survives the
cut intact. The old spellings survive as DISPLAY names on the catalog record
and are never parsed.

Three things live here and only here:

* :func:`lane` — the per-lane declaration: its demand formula and the
  optional additive residency override.
* :class:`Structural` — a STRUCTURAL fork axis: same contract, different
  traced PROGRAM (the scheduler timestep dtype is the measured instance).
* :data:`STATIC` / :data:`DYNAMIC` — the per-shape-axis choice: baked buckets
  or a dynamic dim. Per model, never a global derive flag.

The demand TERM ALGEBRA is :mod:`gen_worker.demand`. This module validates a
declaration; pgw#1600 evaluates it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import msgspec

from ..demand import Demand

__all__ = [
    "DYNAMIC",
    "STATIC",
    "DECLARABLE_SHAPE_AXES",
    "PERMANENTLY_STATIC_SHAPE_AXES",
    "DeclaredLane",
    "LaneSpec",
    "LaneDeclarationError",
    "Structural",
    "lane",
    "parse_shapes",
    "parse_structural",
]


class LaneDeclarationError(TypeError):
    """A lane / fork-axis declaration is not one the platform can read."""


class LaneSpec(msgspec.Struct, frozen=True, kw_only=True):
    """What ONE lane declares: its demand formula, and what must stay resident."""

    request: Demand
    resident: tuple[str, ...] = ()

    def as_document(self) -> dict[str, Any]:

        row: dict[str, Any] = {"request": self.request.as_document()}
        if self.resident:
            row["resident"] = list(self.resident)
        return row


class DeclaredLane(msgspec.Struct, frozen=True, kw_only=True):
    """ONE declared lane, fully READ at class-definition time.

    The read surface every consumer shares (pgw#1606's boot-time lane
    selection ladder, derive, placement) so that nothing re-parses a stamp
    and nothing re-derives a floor:

    * ``layout`` — the ``(topology, quant)`` STAMP PAIR the author named, as a
      :class:`~gen_worker.models.tensor_layout_contract.LayoutId`. This is the
      lane's IDENTITY, and it is compared FIELD-WISE: a whole-string compare
      cannot express "topology differs, quant matches", which is the DERIVABLE
      rung the hub's admission is built on.
    * ``topology`` / ``quant`` — the two halves, spelled out, so a consumer
      that wants one axis does not have to take the pair apart.
    * ``contract_id`` — the WIRE rendering, ``"<topology>+<quant>"``. th#1809's
      spelling, shared with the hub's ``tensorfs.LayoutID.String`` and with the
      derived-artifact CAS address. A drift here is a fork, not a cosmetic bug.
    * ``dtype`` — the QUANT RULE's declared dtype, READ from the ratified rule
      document. Under v1 this was a top-level field on a per-lane document that
      could simply be missing, which is why there was a declaration-time refusal
      and a waiver set for it. A v2 rule cannot lack one: the corpus is
      conformance-checked and `declared_dtype` is required, so the dtype-less
      lane is now inexpressible rather than guarded.
    * ``min_sm`` — the rule's ``capability_floor_sm``, read from the same
      document. An 8-bit lane needs 8-bit kernels because of what it IS. It is
      per LANE, which is exactly why one hand-written number could never be
      right for a bf16/fp8/nvfp4 class — and it is now per RULE, which is why
      it cannot be lost by spelling the dtype differently.
    * ``display_name`` — the v1 lane-handle spelling this pair used to be known
      by (``"sdxl.diffusers-bf16@1"``), or ``""``. **NEVER PARSED, GATES
      NOTHING.** It exists so a refusal message names the lane the way the
      operator still thinks of it.
    * ``spec`` — this lane's demand formula and residency override.

    Selection among lanes is PLATFORM machinery (pgw#1606) and never appears
    in endpoint code; declaration ORDER here is the author's writing order
    and carries no priority.
    """

    layout: Any
    topology: str
    quant: str
    dtype: str
    min_sm: int
    display_name: str = ""
    spec: LaneSpec

    @property
    def contract_id(self) -> str:
        """The wire rendering of the pair — ``"<topology>+<quant>"``."""
        return f"{self.topology}+{self.quant}"

    @property
    def request(self) -> Demand:
        return self.spec.request

    @property
    def resident(self) -> tuple[str, ...]:
        return self.spec.resident


def lane(
    *,
    request: Demand,
    resident: Sequence[str] | None = None,
) -> LaneSpec:
    """Declare one lane."""

    if not isinstance(request, Demand):
        raise LaneDeclarationError(
            f"lane(request=): a demand FORMULA is required, got "
            f"{type(request).__name__}. The VRAM string is deleted — there is "
            f"no single number a lane needs, because demand scales with the "
            f"request (Paul, 2026-08-20). Write the terms the model actually "
            f"scales on, e.g. "
            f"`request=const(GiB(1.2)) + per_mp_batch(MiB(220))` "
            f"(`from gen_worker.demand import const, per_mp_batch, GiB, MiB`)."
        )
    if not request.terms:
        raise LaneDeclarationError(
            "lane(request=): the formula declares no terms. A lane that "
            "needs a fixed floor and nothing else still says so: "
            "`request=const(GiB(1.2))`."
        )
    if resident is None:
        components: tuple[str, ...] = ()
    elif isinstance(resident, str):
        raise LaneDeclarationError(
            "lane(resident=): a SEQUENCE of component names, got a bare "
            'string. Write `resident=("vae",)` — the trailing comma is the '
            "difference between one component and its letters."
        )
    else:
        components = tuple(str(name).strip() for name in resident)
        if any(not name for name in components):
            raise LaneDeclarationError(
                "lane(resident=): a component name cannot be empty; name the "
                "attribute the pipeline holds it under (`vae`, `text_encoder`)."
            )
        if len(set(components)) != len(components):
            raise LaneDeclarationError(
                f"lane(resident=): duplicate component names in {components!r}"
            )
    return LaneSpec(request=request, resident=components)


class Structural(msgspec.Struct, frozen=True, kw_only=True):
    """A STRUCTURAL fork axis — same contract, a DIFFERENT traced program."""

    field: str
    classes: dict[str, Any]
    measured: str

    def variants(self) -> tuple[tuple[str, Any], ...]:
        """``((variant name, representative value), …)`` in declared order."""

        return tuple(self.classes.items())

    def as_document(self, axis: str) -> dict[str, Any]:

        return {
            "axis": axis,
            "declared": [name for name, _ in self.variants()],
            "from": self.field,
            "representatives": [value for _, value in self.variants()],
            "measured": self.measured,
        }


def parse_structural(
    where: str, structural: Mapping[str, Any] | None
) -> dict[str, Structural]:
    """Validate a class-level ``structural=`` map."""

    if structural is None:
        return {}
    if not isinstance(structural, Mapping):
        raise LaneDeclarationError(
            f"{where}: structural= is a mapping of AXIS NAME -> Structural(...), "
            f"got {type(structural).__name__}"
        )
    parsed: dict[str, Structural] = {}
    for axis, declaration in structural.items():
        site = f"{where}: structural[{axis!r}]"
        if not isinstance(axis, str) or not axis.strip():
            raise LaneDeclarationError(
                f"{where}: a structural axis NAME is a non-empty string, got "
                f"{axis!r}"
            )
        if not isinstance(declaration, Structural):
            raise LaneDeclarationError(
                f"{site}: expected Structural(field=…, classes=…, measured=…), "
                f"got {type(declaration).__name__}"
            )
        if not isinstance(declaration.field, str) or not declaration.field.strip():
            raise LaneDeclarationError(
                f"{site}: Structural(field=) names the PAYLOAD FIELD whose "
                f'values fork the program (e.g. field="scheduler"), got '
                f"{declaration.field!r}"
            )
        if not isinstance(declaration.classes, Mapping):
            raise LaneDeclarationError(
                f"{site}: Structural(classes=) maps each variant NAME to ONE "
                f"representative field value, got "
                f"{type(declaration.classes).__name__}"
            )
        ordered = tuple(declaration.classes.items())
        if len(ordered) < 2:
            raise LaneDeclarationError(
                f"{site}: a fork axis has at least TWO variant classes; "
                f"{len(ordered)} was declared. An axis whose values all "
                f"produce the SAME program is not a fork — delete the "
                f"declaration rather than paying a trace for it."
            )
        seen: set[str] = set()
        for name, value in ordered:
            if not isinstance(name, str) or not name.strip():
                raise LaneDeclarationError(
                    f"{site}: a variant class NAME is a non-empty string "
                    f"(it lands in the lock and in the release document), got "
                    f"{name!r}"
                )
            key = repr(value)
            if key in seen:
                raise LaneDeclarationError(
                    f"{site}: variant {name!r} repeats representative "
                    f"{value!r}. Two classes with the same representative "
                    f"trace the same program, which means they are one class."
                )
            seen.add(key)
        if not declaration.measured.strip():
            raise LaneDeclarationError(
                f"{site}: Structural(measured=) is MANDATORY — state HOW the "
                f"variant classes were measured (pgw#1572 measured sdxl's by "
                f"instantiating each served scheduler on CPU and reading "
                f"`timesteps.dtype`). A declared fork with no measurement "
                f"behind it is a guess wearing a declaration's clothes."
            )
        parsed[axis] = declaration
    return parsed


STATIC = "static"
DYNAMIC = "dynamic"

_SHAPE_CHOICES = (STATIC, DYNAMIC)

DECLARABLE_SHAPE_AXES: tuple[str, ...] = ("aspect",)

PERMANENTLY_STATIC_SHAPE_AXES: tuple[str, ...] = ("batch",)


def parse_shapes(
    where: str, shapes: Mapping[str, str] | None, *, marks_compile: bool
) -> dict[str, str]:
    """Validate a class-level ``shapes=`` map into ``{axis: static|dynamic}``."""

    if shapes is None:
        if marks_compile:
            raise LaneDeclarationError(
                f"{where}: shapes= is REQUIRED on a model that marks a compile "
                f"target. Declare each shape axis STATIC (baked buckets) or "
                f'DYNAMIC (a symbolic dim): `shapes={{"aspect": STATIC}}`. '
                f"The two choices produce DIFFERENT closed key sets (sdxl: 36 "
                f"keys static vs ~4 dynamic-aspect) and the platform does not "
                f"get to presume which one you meant — pgw#1548 is measuring "
                f"the cost per model, and the answer is yours to write down. "
                f"Declarable axes: {list(DECLARABLE_SHAPE_AXES)!r}."
            )
        return {}
    if not isinstance(shapes, Mapping):
        raise LaneDeclarationError(
            f"{where}: shapes= is a mapping of AXIS NAME -> STATIC|DYNAMIC, "
            f"got {type(shapes).__name__}"
        )
    parsed: dict[str, str] = {}
    for axis, choice in shapes.items():
        if axis in PERMANENTLY_STATIC_SHAPE_AXES:
            raise LaneDeclarationError(
                f"{where}: shapes[{axis!r}] is not declarable — CFG/batch is a "
                f"shape fork that is PERMANENTLY STATIC (Paul, 2026-08-20). "
                f"Two measured grounds, pgw#1548: batch-dynamic removed ZERO "
                f"specializations on the real endpoint, and batch-dynamic "
                f"records FAIL TO MINT (tcg#78, deterministic, n=3). The x2 "
                f"cfg-on/cfg-off bucket is a fixed part of every image model's "
                f"bucket set; delete the row."
            )
        if axis not in DECLARABLE_SHAPE_AXES:
            raise LaneDeclarationError(
                f"{where}: shapes[{axis!r}] is not a shape axis this platform "
                f"knows. Declarable: {list(DECLARABLE_SHAPE_AXES)!r}; "
                f"permanently static: {list(PERMANENTLY_STATIC_SHAPE_AXES)!r}. "
                f"An axis that changes the traced PROGRAM rather than its "
                f"shapes is a STRUCTURAL fork — declare it in structural=."
            )
        if choice not in _SHAPE_CHOICES:
            raise LaneDeclarationError(
                f"{where}: shapes[{axis!r}] = {choice!r}; a shape axis is "
                f"{STATIC!r} (baked buckets, one artifact each, one shared "
                f"traced program) or {DYNAMIC!r} (one artifact over a "
                f"symbolic dim)."
            )
        parsed[axis] = choice
    missing = [axis for axis in DECLARABLE_SHAPE_AXES if axis not in parsed]
    if missing:
        raise LaneDeclarationError(
            f"{where}: shapes= declares no choice for {missing!r}. Every "
            f"declarable shape axis is stated — an omitted axis is the "
            f"presumed default this surface exists to delete."
        )
    return parsed
