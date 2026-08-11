"""Worker-owned compile-cell identity (gw#581, th#883; redefined by pgw#1059).

ONE compatibility brain: the worker computes the exact key of the compile
cell from the artifact's OWN recorded facts, with this module — and every
consumer of cell identity uses the same code: the production mint
(``aot_mint.mint``) stamps the key it actually produced, and the publish
path (``fleet_cells._identity_axes``) recomputes the same key from the same
facts before a byte moves.

THE MEMBERSHIP AXIOM (Paul, 2026-08-09, pgw#1059 amendment 6): **the key
contains exactly what determines the compiled artifact — nothing else.**
"Don't key on parameters that don't require us to recompile." That single
test admits four axes and no more:

    graph         ``combined_graph_hash`` — first 16 hex of sha256 over the
                  newline-joined SORTED per-entry ``class_hash`` values
                  (pgw#716). Each class hash folds the entry's target, fork
                  coordinate, class dims, declared-range digest,
                  graph-INTERFACE block, the node-level ``graph_witness`` body
                  digest, trace mode and lora bucket (``aot_serve.class_hash``
                  / ``aot_serve.combined_graph_hash`` — stamped by
                  ``aot_serve.artifact_metadata``, proven at admission by
                  ``aot_serve.verify_contract``, READ — never re-derived —
                  here).

                  **This axis is the traced COMPUTATION** (pgw#1031, option a,
                  Paul-ruled): since class_hash facts v3 the per-node digest
                  (``graph_hash.graph_hash``) folds in, so two endpoints whose
                  declarations agree while their bodies differ key APART. Before
                  v3 the axis folded the graph INTERFACE only and they collided
                  — measured 2026-08-10, ``micro-pad32`` vs
                  ``micro-pad32-branchy``: 112- and 102-node graphs,
                  byte-identical keying block, one key, two artifacts; post-fix
                  two keys. A residual collision (a witness-blind or hash-broken
                  entry) is still caught belt-and-braces: every entry records a
                  ``graph_witness`` top-level sibling and the adopt path refuses
                  a cell whose witness is not the graph this pod traced
                  (``aot_identity.verify_graph_witness``).
    envelope      the DECLARED serving region (flight-envelope sense): the
                  shape ladder x text lens x guidance classes the author
                  declared, plus the (currently empty) behavior-posture
                  overlay slot (amendment 5). Changes when the author
                  widens/narrows served traffic without pretending the
                  computation changed. ONE derivation:
                  :func:`envelope_digest` over the recorded
                  ``declared_envelope`` block.
    sm            compute capability (sm_89, ...) — the GPU architecture.
    toolchain     "the compiler stack AS WE CONFIGURE IT" (amendment 4):
                  CONTENT digest of the compile stack (pgw#710 dist-info
                  RECORDs + bundled ptxas/nvdisasm binaries) PLUS the
                  settings DECLARATION digest (pgw#1049 seal v4) and the
                  boot-frozen loaded-library digest. Content, never version
                  strings.

Axes deliberately NOT in the key, each because it fails the axiom:

* ``kind`` / ``format`` — single-valued since pgw#1010 (``aot-inductor`` /
  format 2): zero information, and ``kind``'s dual digest spaces already
  killed pull-by-key once (pgw#1032/#1033). They stay METADATA facts that
  the compat gates (``aot_serve.verify_declared``) refuse on by name.
* ``family`` / ``lane`` — store metadata + discovery scoping only (Paul:
  "identity is the computation; family is namespace/metadata only"). Lanes
  that genuinely differ differ in ``graph`` already (w8a8 ops, lora_a/
  lora_b lifted inputs); two lanes tracing identical graphs SHOULD share.
* ``env_seal`` — LEFT the key (amendment 4): with pgw#1049's single
  settings authority the declaration is ONE value fleet-wide, and a
  constant axis carries zero bits. Settings are compiler flags, so the
  declaration digest folds into ``toolchain`` — a deliberate settings
  change still re-keys, through the axis it honestly belongs to. The
  seal's non-identity roles survive unchanged as GATES: boot verify
  (``env_seal.establish`` fail-closed) and the pre-trace tripwire
  (``env_seal.assert_seal_unchanged``). The seal dict is still RECORDED on
  every artifact and its digest still rides the published identity-axis
  map (the hub's ``ArtifactIdentity.env_seal_digest`` requires it — a wire
  fact, not a key axis, exactly like ``graph_contract``).
* ``code_closure`` — pgw#990: a memo, never identity.
* ``sku`` / ``cuda_driver`` / version strings / ``image_digest`` —
  observability (pgw#691, gw#577, pgw#700).

There is no pre-trace ("arm") cell key any more. The pgw#1033/#1042
two-digest-space design — a COMPUTED ``kind="inductor"`` key from declared
facts beside the STAMPED ``aot-inductor`` key from traced facts — is
retired with the ``contract`` axis that made it necessary: attempt 28 read
the two as one diverging key, which is exactly the fused-axis failure
pgw#1059 splits away. A mint obligation is named by
``fleet_cells.arm_identity`` (NOT a cell key, never ck-prefixed), and the
cozy-local store verdict compares recorded facts directly
(``compile_cache.local_cell_mismatch``).

A wrong key can only produce a MISS (eager + demand + forge), never a
refusal: any failure to arm a self-requested cell is by construction a
selection-logic bug that must surface loudly (``cell_selection_bug``),
never a silent eager fallback.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Tuple

# pgw#958 (DESIGN-RULINGS §1.27(g)) and pgw#1059 amendment 1: the counter
# stays at 1 — Paul 2026-08-09: "stick with version-1 for now since we're
# still pre-launch". pgw#1059 REDEFINES ck1 (contract -> graph x envelope,
# metadata axes dropped, env_seal folded into toolchain) and the disposable
# pre-redefinition corpus is PURGED (pgw#868 runbook step) — the same
# precedent as the §1.27(g) ck1..ck6 counter reset. The scheme-prefix
# machinery STAYS: post-launch, the identical change would be ck2 with
# strand-by-name. New identity FACTS ride the content digests (envelope
# facts v, toolchain entries), never new axes and never a new scheme number.
KEY_SCHEME = "ck1"
_PREFIX = KEY_SCHEME + "-"
# The key digest doubles as the store flavor token, whose shared grammar
# (th#597 C5: [a-z0-9][a-z0-9._-]{0,63}, Go+Py identical) caps tokens at 64
# chars: 56 hex chars of SHA-256 (224 bits) keeps the whole key at 60.
_DIGEST_HEX = 56

# THE four axes, all required — see THE MEMBERSHIP AXIOM in the module
# docstring. There are no optional axes: an exported cell that cannot state
# one of these has no identity. Adding a name here is adding an axis, which
# the axiom forbids unless the new fact provably alters the compiled
# artifact AND cannot ride an existing axis's fact block
# (tests/test_cell_key_pgw1059.py enforces the set).
_REQUIRED = ("graph", "envelope", "sm", "toolchain")
_OPTIONAL: tuple = ()

#: The `kind` METADATA value of an exported .pt2 cell — the only publishable
#: kind since pgw#1010, and since pgw#1059 a compat-gate fact rather than a
#: key axis (`aot_serve.verify_declared` refuses on it by name).
EXPORTED_KIND = "aot-inductor"

#: pgw#1046/pgw#1059: the cell-metadata block recording an exported cell's
#: DECLARED ENVELOPE — the declared serving region (shapes / text_lens /
#: guidance, plus the empty-for-now behavior-posture ``overlay`` slot,
#: amendment 5). It is the ``envelope`` axis's whole input; recorded on the
#: artifact so the exported-cell key is recomputable from the artifact alone
#: (which is what lets publish state full identity axes,
#: ``fleet_cells._identity_axes``).
#:
#: Naming note (pgw#1059 §F): this is the DECLARED envelope — the
#: Paul-ratified axis noun. The OTHER "envelope" in this codebase is the
#: cell-metadata blob itself (``kernel_path.envelope_block``, the hub error
#: envelope). Where both appear, write "the cell-metadata envelope" vs "the
#: declared envelope".
EXPORT_ENVELOPE_KEY = "declared_envelope"


class CellKeyError(ValueError):
    """The artifact (or runtime) cannot state a required key axis."""


@dataclass(frozen=True)
class CellKey:
    """A computed cell identity: canonical axes + their digest."""

    axes: tuple  # sorted ((name, value), ...)

    def axes_dict(self) -> Dict[str, str]:
        return dict(self.axes)

    def canonical(self) -> str:
        return json.dumps(
            self.axes_dict(), sort_keys=True, separators=(",", ":"),
            ensure_ascii=True,
        )

    @property
    def digest(self) -> str:
        h = hashlib.sha256(self.canonical().encode()).hexdigest()
        return _PREFIX + h[:_DIGEST_HEX]


def is_key(value: str) -> bool:
    """True when ``value`` has cell-key SHAPE: ``ck`` + 1-2 scheme digits +
    ``-`` + 56 lowercase hex.

    Scheme-AGNOSTIC, byte-for-byte the grammar tensorhub's
    ``compilecache.IsCellKey`` enforces, and for the same reason it gives
    (th#1183): pinning the current scheme here turns every other-scheme cell
    into ``unreadable_cell_key``, which is both a lie and a filter no axis
    justifies. A cell of an older scheme is admitted to the candidate list and
    then ruled on by the axes that actually decide whether this runtime can
    execute it — the declared envelope, the identity axes and the numerics
    gate — not by the label on it.
    """
    v = str(value or "")
    rest = v[2:] if v.startswith("ck") else ""
    if not rest:
        return False
    digits = 0
    while digits < len(rest) and rest[digits].isdigit():
        digits += 1
    if not 1 <= digits <= 2 or digits >= len(rest) or rest[digits] != "-":
        return False
    hexpart = rest[digits + 1:]
    return (
        len(hexpart) == _DIGEST_HEX
        and all(c in "0123456789abcdef" for c in hexpart)
    )


def from_axes(axes: Mapping[str, str]) -> CellKey:
    """Canonicalize an axes mapping into a :class:`CellKey`.

    Unknown axes are rejected TYPED — including every pre-pgw#1059 axis name
    (``contract``, ``env_seal``, ``kind``, ``format``, ``family``, ``lane``):
    a new axis is forbidden by the membership axiom, and a stale caller
    shipping a dropped one must fail here, not silently widen the key.
    """
    clean: Dict[str, str] = {}
    for name, value in axes.items():
        text = str(value or "").strip()
        if name not in _REQUIRED and name not in _OPTIONAL:
            raise CellKeyError(
                f"unknown cell-key axis {name!r}: the key is exactly "
                f"{list(_REQUIRED)!r} — the membership axiom (pgw#1059 "
                "amendment 6, Paul: \"don't key on parameters that don't "
                "require us to recompile\") admits an axis only when the "
                "fact provably alters the compiled artifact and cannot ride "
                "an existing axis's fact block")
        if text:
            clean[name] = text
    missing = [name for name in _REQUIRED if not clean.get(name)]
    if missing:
        raise CellKeyError(
            f"cell key requires axes {missing!r} (got {sorted(clean)!r})")
    return CellKey(axes=tuple(sorted(clean.items())))


def facts_digest(facts: Mapping[str, Any]) -> str:
    """16-hex canonical digest of one recorded fact block (the toolchain
    axis; also the declared-compile-contract digest surfaces) — computed
    identically from live probes and recorded metadata, so a stamp can never
    disagree with the facts it summarizes."""
    encoded = json.dumps(
        dict(facts), sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def envelope_facts(block: Mapping[str, Any]) -> Dict[str, Any]:
    """The canonical form of one DECLARED-ENVELOPE block — the single
    canonicalizer behind the ``envelope`` axis (used by the mint's recorded
    block, the publish recomputation, and the pre-mint obligation identity,
    so no two consumers can canonicalize the same declaration differently).

    ``overlay`` is the behavior-posture slot (pgw#1059 amendment 5): a
    typed, allowlisted author-settings overlay digested into the envelope
    when one is declared. The menu is EMPTY today — no entry has passed the
    §4.25 justification gate — so the slot is omitted whenever falsy, which
    keeps every current declaration's canonical form free of a field that
    says "unchanged" (the ``excluded``/``literal_values`` discipline).
    """
    facts: Dict[str, Any] = {
        "v": 1,
        "shapes": sorted(
            [int(v) for v in row] for row in (block.get("shapes") or ())),
        "text_lens": sorted({int(v) for v in (block.get("text_lens") or ())}),
        "guidance": sorted(float(v) for v in (block.get("guidance") or ())),
    }
    overlay = block.get("overlay")
    if overlay:
        facts["overlay"] = {
            str(k): str(v) for k, v in sorted(dict(overlay).items())}
    return facts


def envelope_digest(block: Mapping[str, Any]) -> str:
    """The ``envelope`` key-axis value for one declared-envelope block."""
    return facts_digest(envelope_facts(block))


# --- the SUBJECT: what an obligation is FOR (pgw#1113) ---------------------
#
# THE ASYMMETRY, stated once, here, because both consumers below depend on it:
#
#   The CELL KEY is the computation and must not OVER-split (the membership
#   axiom at the top of this module).  The ARM TOKEN and the boot-key memo are
#   an OBLIGATION and a CACHE LOOKUP, and must not UNDER-split.  Over-splitting
#   an obligation costs one re-mint; under-splitting one binds a pipeline to a
#   cell nobody proved is its computation.
#
# So the subject is deliberately NOT a key axis and must never become one: one
# cell legally serves every checkpoint whose graph it is (weight VALUES are
# never hashed — see graph_hash's module docstring), and keying on the
# checkpoint would put every fine-tune in its own key space.  What the subject
# does is stop two DIFFERENT checkpoints sharing one pending mint, one
# local-store memo entry or one boot-key memo row on the strength of an
# assumption nothing checked.


@dataclass(frozen=True)
class SlotSubject:
    """WHICH checkpoint one setup slot resolved to.

    ``refs`` is the base wire ref plus every pgw#617 component-override ref,
    in the order ``api.binding.binding_wire_refs`` produces them;
    ``snapshot_digest`` is the materialized tree's content digest when the
    resolver stated one (it is "" for a slot resolved without one, which is a
    narrower statement, never a different subject).
    """

    slot: str
    refs: Tuple[str, ...] = ()
    snapshot_digest: str = ""


def subject_facts(subjects: Iterable[SlotSubject]) -> Dict[str, Any]:
    """The canonical SUBJECT block for one arm/trace — sorted by slot, so two
    callers that resolved the same slots in different orders state one fact."""
    return {
        "v": 1,
        "slots": [
            [sub.slot, list(sub.refs), sub.snapshot_digest]
            for sub in sorted(tuple(subjects), key=lambda s: s.slot)
        ],
    }


def subject_digest(subjects: Iterable[SlotSubject]) -> str:
    """16-hex digest of the resolved subject, or ``""`` when the caller could
    state none.

    ``""`` is the honest answer for a pipeline whose slot resolution this
    process never saw (an endpoint that builds its own pipeline out of a
    path-valued slot, reached through ``arm_compile()``); it is exactly the
    pre-pgw#1113 posture for that path and it under-splits, so every caller
    that CAN state a subject states one.
    """
    subs = tuple(subjects)
    if not subs:
        return ""
    return facts_digest(subject_facts(subs))


def from_exported_artifact_metadata(meta: Mapping[str, Any]) -> CellKey:
    """The key an EXPORTED (``aot-inductor``) cell's OWN recorded facts
    describe — THE single implementation of the cell key: ``aot_mint``
    stamps what this returns and the publish path recomputes it, so the
    axes a cell is published under cannot drift from the axes its key was
    minted from.

    Every axis is read from a recorded block, never from a probe and never
    from the ``cell_key`` stamp (pgw#1046 recorded the declared envelope for
    exactly this reason). Raises :class:`CellKeyError` when a fact is
    missing: a cell that cannot name an axis has no identity, and must not
    be published under a partial one.

    ``graph`` is READ from ``combined_graph_hash``, never re-derived here —
    the one derivation lives in ``aot_serve`` (``class_hash`` /
    ``combined_graph_hash``, stamped by ``artifact_metadata``), and
    admission (``aot_serve.verify_contract``) recomputes it from the staged
    bytes so a forged stamp is refused before it can arm. The per-class
    hashes ride ``entries[*].class_hash`` so a mismatch NAMES the class
    (pgw#716).

    Pre-pgw#1059 artifacts fail here STRUCTURALLY: they record
    ``declared_traffic`` (not ``declared_envelope``), so the envelope axis
    is unrecoverable and the refusal below fires — a pre-redefinition cell
    can never restate a post-redefinition identity, which is what makes the
    dev-corpus purge hygiene rather than a correctness precondition.
    """
    kind = str(meta.get("kind") or "")
    if kind != EXPORTED_KIND:
        raise CellKeyError(
            f"artifact kind {kind!r} has no cell-key identity: only exported "
            f"{EXPORTED_KIND!r} cells are keyed (pgw#1010/pgw#1059 — JIT is "
            "intake, local torch-inductor-cache artifacts compare facts via "
            "compile_cache.local_cell_mismatch)")
    sm = str(meta.get("sm") or "")
    if not sm:
        raise CellKeyError(
            "cannot state the compute capability (sm) of this runtime; an "
            "exported cell has no identity without it — mint on the target GPU")
    entries = dict(meta.get("entries") or {})
    combined = str(meta.get("combined_graph_hash") or "")
    if not entries or not combined:
        raise CellKeyError(
            "the cell-metadata envelope recorded no entries/"
            "combined_graph_hash; a multi-graph cell must not be keyed "
            "without its class set (pgw#716/#758)")
    unhashed = sorted(
        name for name, block in entries.items()
        if not str((block or {}).get("class_hash") or ""))
    if unhashed:
        raise CellKeyError(
            f"entries {unhashed[:4]!r} carry no class_hash; a class the key "
            f"cannot name is a class a mismatch cannot name (pgw#716)")
    envelope = meta.get(EXPORT_ENVELOPE_KEY)
    if not isinstance(envelope, dict) or not envelope:
        raise CellKeyError(
            f"artifact records no {EXPORT_ENVELOPE_KEY!r} block; its declared "
            "envelope is unrecoverable, so the 'envelope' axis cannot be "
            "restated from the artifact (pgw#1046/pgw#1059)")
    toolchain = meta.get("toolchain")
    if not isinstance(toolchain, dict) or not toolchain:
        raise CellKeyError(
            "artifact records no toolchain block; no recipe identity")
    return from_axes({
        "graph": combined,
        "envelope": envelope_digest(envelope),
        "sm": sm,
        "toolchain": facts_digest(toolchain),
    })


__all__ = [
    "KEY_SCHEME",
    "EXPORTED_KIND",
    "EXPORT_ENVELOPE_KEY",
    "CellKey",
    "CellKeyError",
    "SlotSubject",
    "subject_digest",
    "subject_facts",
    "envelope_digest",
    "envelope_facts",
    "facts_digest",
    "from_exported_artifact_metadata",
    "from_axes",
    "is_key",
]
