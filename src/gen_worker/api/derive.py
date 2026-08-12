"""Declaration inference + the migration safety gate (pgw#1107).

pgw#1107 collapses ``aot_declaration.py`` and the inline ``@endpoint(
compile=Compile(...))`` into ONE declaration on the class decorator. Two
pieces of that live here — both pure Python, no model, no trace, no inductor,
so they run anywhere including this box:

1. :func:`cfg_image_classes` — the MECHANICAL half of "the SDK derives
   ``classes``". For a CFG-batched image UNet/DiT whose legal request set is
   ``aspect rows x the two CFG regimes`` (sdxl, z-image), the graph-class
   cross-product is a pure function of the author's kept ``shapes`` + the
   latent scale + ``text_len``. The author stops hand-writing the
   ``_..._graph_classes()`` helper; the loop, the ``//vae_scale`` latent math
   and the ``dict.fromkeys`` transposed-row dedupe move here, byte-identical.

   It is deliberately NOT a universal deriver. qwen (ceil-div token grid),
   flux2 (token-COUNT collapse), wan (3-D latent + relational ``N_tok``) and
   ltx (audio-token formula x two-stage x keyframe axis x frame-legality x
   H100/B200 tier) embed model-specific token math that IS the endpoint's own
   legality oracle (pgw#669). For those the class SET is family-declared; only
   the collapse-onto-the-decorator and the gate below apply.

2. :func:`contract_delta` / :func:`assert_faithful` — the MIGRATION SAFETY
   GATE. ``Compile.contract_axes()`` is exactly what feeds the cell key's
   contract digest (pgw#647), so two declarations with an identical
   ``contract_axes()`` produce an identical contract digest and — the traced
   graph being a pure function of the same declaration under fixed model code
   and toolchain — an identical ``combined_graph_hash``. Deleting the standing
   ``aot_declaration.py`` in favour of a minimal decorator is therefore safe
   iff the two declarations' contract axes match. A non-empty delta is a STOP,
   not a merge: it is the surprise re-key (or the silently-lost hard-won fact)
   the gate exists to catch before a family being minted RIGHT NOW is disturbed.

3. :func:`blocker_delta` / :func:`assert_blockers` — the REFUSAL half of that
   gate (pgw#1115). The one fact a fold can drop that neither of the above
   sees is a family's declared mint blockers, because before the fold the
   family that had them kept them OUTSIDE its ``Compile`` (a module-level
   table read by a refusing thunk, retired in pgw#1107). Dropping one does not
   re-key anything; it simply starts minting against an open design question.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Sequence, Tuple

from .decorators import Compile
from .export_contract import GraphClass, open_blockers

__all__ = [
    "DeclarationMismatch",
    "assert_blockers",
    "assert_faithful",
    "blocker_delta",
    "cfg_image_classes",
    "class_set_delta",
    "contract_delta",
]

#: The sdxl/z-image regime table: CFG on traces ONE batch-2 forward, turbo /
#: no-CFG pins the batch-1 graph. Each arm is its own graph class of one cell
#: (ie#345, gw#627). ``(fork_value, traced_batch)`` — the two families that
#: batch CFG into a single forward share this exact pair.
CFG_BATCH_REGIMES: Tuple[Tuple[bool, int], ...] = ((True, 2), (False, 1))


class DerivedClasses(tuple):
    """The deriver's class rows, carrying the divisor they were derived AT.

    A tuple subclass so every existing call site is unaffected — it IS the
    tuple they already got, and `Compile(classes=…)` still receives a sequence
    of :class:`GraphClass`. The extra attribute exists only to survive the trip
    from the deriver to `Compile.__post_init__`, which transfers it to
    `Compile.latent_basis` and then rebuilds `classes` as a plain tuple.

    So this is TRANSPORT, not storage: nothing downstream reads it, and it
    deliberately does not persist on the declaration. The alternative — a
    cell-wide scalar stamped onto every row and read back off `rows[0]` — is
    the shape that produced a P0 filed on a false premise, because a label
    written beside a thing cannot be told from a label describing it, and it
    silently answers a question nothing asks (what if two rows disagree?).
    """

    latent_basis: int

    def __new__(cls, rows: Sequence[GraphClass], *, latent_basis: int) -> "DerivedClasses":
        self = super().__new__(cls, tuple(rows))
        self.latent_basis = int(latent_basis)
        return self


def cfg_image_classes(
    *,
    shapes: Sequence[Sequence[int]],
    latent_scale: int,
    text_len: int,
    batch_dim: str = "B",
    height_dim: str = "H_lat",
    width_dim: str = "W_lat",
    text_dim: str = "T_txt",
    cfg_fork: str = "cfg",
    regimes: Sequence[Tuple[bool, int]] = CFG_BATCH_REGIMES,
) -> Tuple[GraphClass, ...]:
    """The ``aspect rows x CFG regimes`` graph-class cross-product.

    Reproduces sdxl's ``_sdxl_graph_classes`` and z-image's
    ``_z_image_graph_classes`` byte-identically (same iteration order, same
    ``//latent_scale`` latent math, same ``dict.fromkeys`` dedupe of
    transposed rows). ``shapes`` is the author's kept ``(w, h)`` aspect table
    — single-sourced from the payload preset enum, never re-declared.

    The dim NAMES differ per family (sdxl ``B``/``T_txt`` vs z-image
    ``N``/``T_cap``) and are parameters, not literals, so one deriver serves
    both. The fork→batch effect is the ``regimes`` pair: it is CFG-specific
    (only these two families batch CFG into one forward — qwen/flux2/wan run
    CFG as two sequential batch-1 calls, so they do NOT use this deriver).
    """
    if latent_scale <= 0:
        raise ValueError(f"latent_scale must be positive, got {latent_scale!r}")
    if int(text_len) <= 0:
        raise ValueError(
            f"cfg_image_classes needs a positive text_len, got {text_len!r}")
    out: List[GraphClass] = []
    for row in shapes:
        w, h = int(row[0]), int(row[1])
        for cfg, batch in regimes:
            out.append(GraphClass(
                dims={
                    batch_dim: batch,
                    height_dim: h // latent_scale,
                    width_dim: w // latent_scale,
                    text_dim: int(text_len),
                },
                fork={cfg_fork: cfg},
            ))
    # pgw#1167: the divisor rides out with the rows it produced, so the mint
    # can reconcile it against the pipeline instead of the author declaring it
    # a second time.
    return DerivedClasses(
        tuple(dict.fromkeys(out)), latent_basis=int(latent_scale))


# ---------------------------------------------------------------------------
# The migration safety gate
# ---------------------------------------------------------------------------

class DeclarationMismatch(AssertionError):
    """A migrated declaration's cell-key contract differs from the standing
    one — deleting the standing file would re-key or drop a fact. STOP."""


def _canonical(value: Any) -> str:
    """Order-stable JSON for a ``contract_axes()`` value, so the comparison
    sees a byte difference, never a dict-ordering artefact."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False)


def contract_delta(standing: Compile, migrated: Compile) -> Dict[str, Tuple[Any, Any]]:
    """``{axis: (standing_value, migrated_value)}`` for every cell-key contract
    axis the two declarations disagree on; ``{}`` iff identical.

    ``{}`` is the migration's green light: identical ``contract_axes()`` ⟹
    identical contract digest ⟹ identical ``combined_graph_hash`` (the trace
    is a pure function of the declaration under fixed code + toolchain), so
    the standing ``aot_declaration.py`` can be deleted with no re-key. Any
    entry is a STOP.
    """
    a = standing.contract_axes()
    b = migrated.contract_axes()
    delta: Dict[str, Tuple[Any, Any]] = {}
    for key in sorted(set(a) | set(b)):
        av, bv = a.get(key, _MISSING), b.get(key, _MISSING)
        if _canonical(_repr(av)) != _canonical(_repr(bv)):
            delta[key] = (av, bv)
    return delta


class _Missing:
    def __repr__(self) -> str:  # pragma: no cover - diagnostic only
        return "<absent>"


_MISSING = _Missing()


def _repr(value: Any) -> Any:
    """``contract_axes()`` values are already JSON-able except the sentinel."""
    return None if value is _MISSING else value


#: Must-survive facts that are NOT part of ``contract_axes()`` (so a change
#: does not re-key) but ARE hard-won measured decisions the migration must not
#: drop: the numerics ADOPT band (pgw#812/#814) and the compile-vs-eager SPEED
#: bar the hub's publish-time validation gate judges against (pgw#1149 /
#: th#1811). ``shape_strategy``, ``warm_changes_key`` and ``eager`` are already
#: contract axes and covered by :func:`contract_delta`; these four are the gap.
#:
#: The speed pair is here for the same reason the floor is: dropping it loosens
#: a gate without re-keying anything, so :func:`contract_delta` alone waves it
#: through — and a release that arrives at the hub with no bar is not refused,
#: it is judged `bar_undeclared` and simply stops being promotable.
OVERRIDE_FACTS: Tuple[str, ...] = (
    "numerics_floor", "numerics_warn", "speed_metric", "min_speedup")


def override_delta(standing: Compile, migrated: Compile) -> Dict[str, Tuple[Any, Any]]:
    """``{field: (standing, migrated)}`` for must-survive overrides that live
    OUTSIDE ``contract_axes()`` — dropping one loosens the adopt gate without
    re-keying, so :func:`contract_delta` alone would wave it through. ``{}`` iff
    every such fact is preserved."""
    delta: Dict[str, Tuple[Any, Any]] = {}
    for field in OVERRIDE_FACTS:
        av, bv = getattr(standing, field), getattr(migrated, field)
        if av != bv:
            delta[field] = (av, bv)
    return delta


def blocker_delta(standing: Compile, migrated: Compile) -> Tuple[str, ...]:
    """The OPEN blocker ids the standing declaration carries and the migrated
    one does not (pgw#1115). ``()`` iff no refusal was lost.

    Deliberately DIRECTIONAL, unlike every other delta here. A fold may ADD a
    blocker — that is a family recording a question it had not written down —
    but it may never drop one, because a dropped blocker is a family that
    starts minting against an open design question and says nothing. A
    RESOLVED-in-the-migrated-declaration id counts as dropped: resolving is a
    reviewable edit to the standing declaration, not a side effect of a move.
    """
    before = {b.id for b in open_blockers(standing)}
    after = {b.id for b in open_blockers(migrated)}
    return tuple(sorted(before - after))


def assert_blockers(
    decl: Compile, *, ids: Sequence[str], family: str = "",
) -> None:
    """Raise :class:`DeclarationMismatch` unless ``decl``'s OPEN blocker ids
    are EXACTLY ``ids`` (pgw#1115) — the per-family testable guard.

    :func:`blocker_delta` can only compare two declarations, so it is blind
    where the standing declaration kept its blockers OUTSIDE the ``Compile``
    (a module-level table read by a refusing thunk — ltx-video-2.3's pre-fold
    shape, and precisely the shape the fold had to carry across). This states
    the expectation in the family's OWN test instead, so the assertion survives
    the file the blockers used to live in.

    ``ids=()`` asserts the family is mintable, and is just as load-bearing:
    it goes red the day somebody adds a blocker without telling the family's
    tests.
    """
    want = tuple(sorted(str(i).strip() for i in ids))
    got = tuple(sorted(b.id for b in open_blockers(decl)))
    if want == got:
        return
    who = f" for {family!r}" if family else ""
    missing = sorted(set(want) - set(got))
    extra = sorted(set(got) - set(want))
    lines = [f"declared mint blockers{who} are not the asserted set:"]
    if missing:
        lines.append(
            f"  MISSING (the declaration no longer refuses on): {missing} — a "
            f"refusal that disappeared without being resolved lets this family "
            f"mint against an open design question")
    if extra:
        lines.append(
            f"  UNEXPECTED (the declaration now refuses on): {extra} — the "
            f"family's tests have not been told")
    raise DeclarationMismatch("\n".join(lines))


def class_set_delta(
    standing: Sequence[GraphClass], migrated: Sequence[GraphClass],
) -> Dict[str, Any]:
    """A focused diff of just the graph-class tuples — order-sensitive, since
    ``contract_axes()`` serialises ``classes`` as an ordered list. ``{}`` iff
    the derived class tuple is byte-identical to the standing one.
    """
    sa = [c.as_row() for c in standing]
    sb = [c.as_row() for c in migrated]
    if _canonical(sa) == _canonical(sb):
        return {}
    out: Dict[str, Any] = {"count": (len(sa), len(sb))}
    if len(sa) == len(sb):
        out["rows"] = [
            {"index": i, "standing": x, "migrated": y}
            for i, (x, y) in enumerate(zip(sa, sb))
            if _canonical(x) != _canonical(y)
        ]
    else:
        out["standing_only"] = [x for x in sa if x not in sb]
        out["migrated_only"] = [y for y in sb if y not in sa]
    return out


def assert_faithful(
    standing: Compile, migrated: Compile, *, family: str = "",
) -> None:
    """Raise :class:`DeclarationMismatch` unless the migrated declaration's
    cell-key contract is byte-identical to the standing one. The reusable
    per-family migration gate: run it in the endpoint's test with the standing
    ``aot_declaration`` Compile and the new decorator's ``compile`` BEFORE the
    file is deleted.
    """
    cdelta = contract_delta(standing, migrated)
    odelta = override_delta(standing, migrated)
    bdelta = blocker_delta(standing, migrated)
    if not cdelta and not odelta and not bdelta:
        return
    who = f" for {family!r}" if family else ""
    lines = [f"migrated declaration{who} is not faithful to the standing one:"]
    if bdelta:
        lines.append(
            f"  mint REFUSAL dropped ({len(bdelta)} blocker(s)): {list(bdelta)} "
            f"— the migrated declaration would mint against open design "
            f"question(s) the standing one refuses on (pgw#1115)")
    if cdelta:
        lines.append(
            f"  cell-key contract re-keys ({len(cdelta)} axis(es) differ):")
        for axis, (av, bv) in cdelta.items():
            lines.append(f"    - {axis}: standing={av!r} migrated={bv!r}")
    if odelta:
        lines.append(
            f"  must-survive override dropped ({len(odelta)} field(s)):")
        for field, (av, bv) in odelta.items():
            lines.append(f"    - {field}: standing={av!r} migrated={bv!r}")
    raise DeclarationMismatch("\n".join(lines))
