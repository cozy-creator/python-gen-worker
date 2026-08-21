"""The LANE DECLARATION SURFACE — one place the author writes demand,
residency and fork axes (pgw#1599, implementing pgw#1597 plan 1 + pgw#1598 §2).

What the author writes, and nothing else::

    class SdxlModel(
        Model[SDXL],
        lanes={contracts.SDXL_DIFFUSERS_BF16: lane(
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
bytes from the manifest, the capability floor from the contract dtype,
launch-residency from the ``ctx.compile`` marks, coefficients from
measurement). A VRAM STRING is none of those things, which is why it is gone.

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
    """A lane / fork-axis declaration is not one the platform can read.

    A subclass of ``TypeError`` for the same reason ``ModelDeclarationError``
    is: a class header is code, and a header that does not state a valid
    declaration is a definition-time defect, not a runtime one.
    """


# ── the lane declaration ────────────────────────────────────────────────────


class LaneSpec(msgspec.Struct, frozen=True, kw_only=True):
    """What ONE lane declares: its demand formula, and what must stay resident.

    ``request`` is this lane's OWN formula. Per-lane and not per-model
    deliberately (se#816): fp8 halves the weight bytes AND shrinks the
    activation coefficients, so one formula for a bf16/fp8/nvfp4 class would
    be wrong for two of its three lanes.

    ``resident`` is an OPTIONAL, ADDITIVE override (pgw#1598 amendment 7).
    Residency classes are INFERRED by default — compile-marked components are
    launch-resident all-or-nothing, everything else is leaf-streamable — and
    this names ONLY the judgment-call ADDITIONS: an uncompiled dense-burst
    component (the VAE) whose streaming would thrash. Wrong in either
    direction stays SAFE (too-streamed is slower, too-resident spends grant),
    so it is a performance statement and never a correctness one. Most
    endpoints omit it, and an empty tuple means "add nothing", never "nothing
    is resident".
    """

    request: Demand
    resident: tuple[str, ...] = ()

    def as_document(self) -> dict[str, Any]:
        """The release-document shape (pgw#1600 serializes the evaluation)."""

        row: dict[str, Any] = {"request": self.request.as_document()}
        if self.resident:
            row["resident"] = list(self.resident)
        return row


class DeclaredLane(msgspec.Struct, frozen=True, kw_only=True):
    """ONE declared lane, fully READ at class-definition time.

    The read surface every consumer shares (pgw#1606's boot-time lane
    selection ladder, derive, placement) so that nothing re-parses a stamp
    and nothing re-derives a floor:

    * ``contract`` — the tensorfs ``Contract`` object the author named.
    * ``contract_id`` — its stable handle, e.g. ``sdxl.diffusers-bf16@1``.
    * ``dtype`` — the contract's OWN load dtype, READ (a dtype-less lane is
      refused at declaration, never discovered at load on a rented pod).
    * ``min_sm`` — DERIVED via ``capability_floor_for_dtype``. An 8-bit lane
      needs 8-bit kernels because of what it IS; a hand-written floor is a
      second producer of one fact and is a declaration-time refusal. It is
      per LANE, which is exactly why one hand-written number could never be
      right for a bf16/fp8/nvfp4 class.
    * ``spec`` — this lane's demand formula and residency override.

    Selection among lanes is PLATFORM machinery (pgw#1606) and never appears
    in endpoint code; declaration ORDER here is the author's writing order
    and carries no priority.
    """

    contract: Any
    contract_id: str
    dtype: Any
    min_sm: int
    spec: LaneSpec

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
    """Declare one lane. The mapping VALUE of ``lanes={contract: lane(...)}``.

    ``request=`` is REQUIRED and is the whole point of the surface: the old
    value was a VRAM STRING (``"vram7g"``) that stated one number for every
    request a lane would ever serve, and Paul's ruling is that there is no
    such number — a 4 MP image and a 1 MP image do not demand the same bytes,
    and an H3 video demands them quadratically in the frame count.
    """

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


# ── structural fork axes ────────────────────────────────────────────────────


class Structural(msgspec.Struct, frozen=True, kw_only=True):
    """A STRUCTURAL fork axis — same contract, a DIFFERENT traced program.

    The measured instance is the scheduler timestep dtype (pgw#1572): of the
    schedulers sdxl serves, 5 feed the UNet an ``int64`` timestep and 3 feed
    ``float32``, which is a different PROGRAM and therefore a different
    graph — and because nothing declared it, 5 of sdxl's 8 served scheduler
    configs fell to loud eager. Every one was a key-closure violation the
    leak detector was reporting correctly and nobody could act on, because
    there was no way to SAY the axis existed.

    ``field`` names the entrypoint payload field whose values fork the
    program. ``classes`` maps each variant's NAME to ONE representative
    value — the derive traces the representatives, not the cross-product, so
    an axis with 8 values and 2 variant classes costs 2 traces and covers
    8/8. ``measured`` is the author's evidence and is MANDATORY: a declared
    fork with no measurement behind it is the derived-dressed-as-measured
    defect this whole design exists to end.

    The axis is AUTHOR-OWNED, all the way down (Paul, 2026-08-20). The
    platform's job is exactly four things — enumeration, closure-checking,
    pricing, leak detection — and it never invents an axis.
    """

    field: str
    classes: dict[str, Any]
    measured: str

    def variants(self) -> tuple[tuple[str, Any], ...]:
        """``((variant name, representative value), …)`` in declared order."""

        return tuple(self.classes.items())

    def as_document(self, axis: str) -> dict[str, Any]:
        """The release-document row (pgw#1572's proposed shape)."""

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


# ── shape fork axes ─────────────────────────────────────────────────────────

#: Baked buckets — one artifact per bucket, all sharing one traced program.
STATIC = "static"
#: One artifact over a symbolic dim.
DYNAMIC = "dynamic"

_SHAPE_CHOICES = (STATIC, DYNAMIC)

#: Shape axes the author CHOOSES between static and dynamic, per model
#: (pgw#1597: *"we need to see how much this costs us, in inference time.
#: This will be a per-model decision, for whoever implements the model"*).
DECLARABLE_SHAPE_AXES: tuple[str, ...] = ("aspect",)

#: Shape axes with NO choice to declare. CFG/batch is a shape fork that is
#: PERMANENTLY STATIC (Paul, 2026-08-20: *"CFG stays a fork axes
#: permanently"*), on two measured grounds: batch-dynamic removed ZERO
#: specializations on the real endpoint, and batch-dynamic records FAIL TO
#: MINT (tcg#78, deterministic, n=3). The ruling stands even if tcg#78 is
#: fixed — the zero-reduction result alone kills the axis.
PERMANENTLY_STATIC_SHAPE_AXES: tuple[str, ...] = ("batch",)


def parse_shapes(
    where: str, shapes: Mapping[str, str] | None, *, marks_compile: bool
) -> dict[str, str]:
    """Validate a class-level ``shapes=`` map into ``{axis: static|dynamic}``.

    REQUIRED of a class that marks a compile target, and refused on one that
    does not: the choice only means something where a graph is minted, and
    presuming a default is exactly what pgw#1599 acceptance (d) forbids —
    the two declarations yield different CLOSED key sets (sdxl static: 9
    aspect x 2 batch x 2 structural = 36; dynamic-aspect: ~4) and neither is
    the platform's to assume.
    """

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
    # DELIBERATELY NOT REFUSED on a class the AST reads as unmarked. The
    # reader (`load_marks_compile`, se#809) sees a literal `ctx.compile(...)`
    # in THIS class's `load` and nothing else, and a real fixture marks
    # through a helper (`self.engine.compile_dit(ctx)`) — the mark is genuine,
    # the graphs are genuine, and the AST cannot see it. Refusing there would
    # have made a correct endpoint undeclarable to buy a tidiness check, so
    # the asymmetry stands: REQUIRED where a mark is provable, PERMITTED
    # where it is not. A shapes= on a model that truly compiles nothing is
    # inert noise; a refusal on one that compiles is a wall.
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
