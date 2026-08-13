"""Worker-owned compiled-ENTRY identity (gw#581, th#883; pgw#1059; pgw#1176).

ONE compatibility brain: the worker computes the exact key of one compiled
GRAPH CLASS from that artifact's OWN recorded facts, with this module — and
every consumer of identity uses the same code: the production mint
(``aot_mint.mint``) stamps the key it actually produced, and the publish
path (``fleet_cells._identity_axes``) recomputes the same key from the same
facts before a byte moves.

THE ATOM IS ONE GRAPH CLASS (pgw#1176, Paul-directed 2026-08-12). What used
to be keyed here was a 36-entry all-or-nothing transaction whose identity,
adoption, durability, verification, arming and advertisement were the SAME
unit; weight-free compile removed every reason for that unit to be big and
the design never renegotiated. The unit of identity, storage, transfer,
verification, arming and de-arming is now the **entry**: one graph class,
compiled whole. "The cell" survives only as a derived, artifact-less
CONTRACT MANIFEST — the set of entry keys one declaration traces to. A
manifest is a view; it is never downloaded, verified or armed, and
:func:`manifest_digest` is a telemetry/coverage label, never identity.

THE MEMBERSHIP AXIOM (Paul, 2026-08-09, pgw#1059 amendment 6): **the key
contains exactly what determines the compiled artifact — nothing else.**
"Don't key on parameters that don't require us to recompile." Applied at the
honest granularity that test admits THREE axes and no more:

    graph         this entry's own ``class_hash`` — 16 hex folding the
                  entry's target, fork coordinate, class dims, declared-range
                  digest, graph-INTERFACE block, the node-level
                  ``graph_witness`` body digest, trace mode and lora bucket
                  (``aot_serve.class_hash`` — stamped by
                  ``aot_serve.entry_metadata``, proven at admission by
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
    sm            compute capability (sm_89, ...) — the GPU architecture.
    toolchain     "the compiler stack AS WE CONFIGURE IT" (amendment 4):
                  CONTENT digest of the compile stack (pgw#710 dist-info
                  RECORDs + bundled ptxas/nvdisasm binaries) PLUS the
                  settings DECLARATION digest (pgw#1049 seal v4) and the
                  boot-frozen loaded-library digest. Content, never version
                  strings. MEMBERSHIP is the COMPILER, not the model
                  libraries (pgw#1050) — ONE derivation:
                  :func:`toolchain_axis_digest` over the recorded
                  ``toolchain`` block, which is what
                  :func:`toolchain_facts` says the axis is.

Axes deliberately NOT in the key, each because it fails the axiom:

* ``envelope`` — **EVICTED by pgw#1176, and it is the whole point.**
  ``envelope_facts`` digests the UNION of shapes / text_lens / guidance
  ACROSS THE BUNDLE — a property of the collection, not of any computation
  — so adding one aspect ratio re-keyed all 36 sdxl entries although 35 of
  them trace byte-identically (measured on origin/master @ ``4dfdcd60``:
  two byte-identical classes moved ``ck1-c4c134db…`` -> ``ck1-48512ea3…``
  for one extra shape row). Per entry the shape facts that genuinely affect
  tracing are ALREADY inputs to that class's ``class_hash``, through
  ``range_digest`` and ``class_dims``. The one real edge is honest under
  this split rather than lost by it: widening a DYNAMIC dim's range does
  change the traced graph, and it re-keys — through that class's own graph
  hash, not through a union digest that punishes its siblings. The declared
  envelope survives as a MANIFEST fact (:func:`envelope_facts` — the
  declaration that produced the key set), never as identity.
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
* the MODEL LIBRARIES — ``diffusers`` / ``transformers`` / ``peft``
  (pgw#1050). They ran inside ``toolchain`` until 2026-08-11, and that was
  the axiom's other failure mode: an OVER-split. They are pure-python and
  run at TRACE time only, so everything they can do to a cell arrives as
  the traced computation — ops, literal args, tensor meta, symbolic ranges,
  the graph signature and both pytree specs, all of which the ``graph``
  axis hashes node-for-node since pgw#1031. The two channels that could
  route around the graph are both closed by construction, not by this
  axis: weight/constant VALUES cannot reach the artifact (the B1 code-only
  gate and the pgw#1097 folding fence, both DECLARED and refused
  pre-download by ``aot_serve.verify_declared``), and torch settings a
  library mutates behind our back trip ``env_seal.assert_seal_unchanged``
  pre-trace and are refused by name. So a model-library bump either moves
  the graph — and re-keys through ``graph``, which is the honest axis — or
  cannot change the artifact at all. Keeping them double-counted the first
  case and re-minted the whole fleet for the second.

There is no pre-trace ("arm") cell key any more. The pgw#1033/#1042
two-digest-space design — a COMPUTED ``kind="inductor"`` key from declared
facts beside the STAMPED ``aot-inductor`` key from traced facts — is
retired with the ``contract`` axis that made it necessary: attempt 28 read
the two as one diverging key, which is exactly the fused-axis failure
pgw#1059 splits away. A mint obligation is named by
``fleet_cells.arm_identity`` (NOT an entry key, never ek-prefixed), and the
cozy-local store verdict compares recorded facts directly
(``compile_cache.local_cell_mismatch``).

A wrong key can only produce a MISS (eager, then a background self-mint),
never a refusal: any failure to arm a self-requested cell is by construction a
selection-logic bug that must surface loudly (``cell_selection_bug``),
never a silent eager fallback.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Tuple

from gen_worker.refgrammar import MAX_FRAGMENT_LEN

# pgw#1176: ``ck1`` (the 36-entry CELL key) is REPLACED by the per-entry key
# below (one graph class). This is a §1.34 third-category change — a name persisted as an
# ADDRESS — so it is a COORDINATED FLEET RE-KEY, never a quiet rename: every
# stored cell is orphaned by it, and the cut carries scheme + per-entry store
# schema + resolve/publish wire + pack format + the
# ``Arm.graph_contract_digest`` proto change + the corpus purge together.
# Priced today: the adoptable corpus is ONE toy family, so the re-key costs
# one re-mint of a 3-class toy — effectively free, and never this cheap
# again. The scheme-prefix machinery STAYS: post-launch, the identical change
# would be a new scheme token with strand-by-name. New identity FACTS ride the
# content digests (toolchain entries), never new axes and never a new scheme.
#
# pgw#1213: the scheme token is ``cg-key-v1`` (Paul, final). It landed before
# the ``v0.114.0`` tag existed and before anything published, so no artifact
# was ever addressed by ``ek1`` — the 0.114.0 wheel ships ``cg-key-v1``. The
# token CONTAINS hyphens, so no reader may split a key on ``-``: the grammar
# is scheme + ``-`` + the fixed-width hex digest, matched from the RIGHT
# (``is_key``).
KEY_SCHEME = "cg-key-v1"
_PREFIX = KEY_SCHEME + "-"
# The key doubles as the store flavor token, whose shared grammar (th#597 C5:
# [a-z0-9][a-z0-9._-]*) caps a token at MAX_FRAGMENT_LEN. 56 hex chars of
# SHA-256 (224 bits) + the 10-char ``cg-key-v1-`` prefix is 66, which is why
# th#1897 moved that cap from 64 to 96.
#
# BOTH SIDES ARE AT 96: tensorhub's `refgrammar.MaxFragmentLen` and
# `gen_worker.refgrammar.MAX_FRAGMENT_LEN`. "Go+Py identical" used to be an
# assertion in this comment and nothing else, which is exactly how the two
# halves drifted — the parity is enforced now, by the 96/97 boundary vectors
# in both vendored corpora and by scripts/grammar-vector-drift.sh, which
# compares the peer's bytes to ours.
_DIGEST_HEX = 56

# THE three axes, all required — see THE MEMBERSHIP AXIOM in the module
# docstring. There are no optional axes: an exported entry that cannot state
# one of these has no identity. Adding a name here is adding an axis, which
# the axiom forbids unless the new fact provably alters the compiled
# artifact AND cannot ride an existing axis's fact block
# (tests/test_cell_key_pgw1059.py enforces the set).
_REQUIRED = ("graph", "sm", "toolchain")
_OPTIONAL: tuple = ()

#: The same three, public, because the PUBLISH WIRE has to name them: every
#: batched entry restates all three (pgw#1224 / th#1842 PR #1121), so the
#: enumeration the wire enforces and the enumeration the key is built from must
#: be ONE list. Two copies of an axis set is how a wire starts accepting a row
#: that cannot restate its own key.
KEY_AXES: Tuple[str, ...] = _REQUIRED

#: The `kind` METADATA value of an exported .pt2 entry — the only publishable
#: kind since pgw#1010, and since pgw#1059 a compat-gate fact rather than a
#: key axis (`aot_serve.verify_declared` refuses on it by name).
EXPORTED_KIND = "aot-inductor"

#: The MANIFEST block recording one declaration's DECLARED ENVELOPE — the
#: declared serving region (shapes / text_lens / guidance, plus the
#: empty-for-now behavior-posture ``overlay`` slot).
#:
#: pgw#1176: this is a MANIFEST fact and NEVER a key axis. It rides the
#: derived contract manifest (the declaration that produced the key set), not
#: the per-entry artifact, because it is a property of the COLLECTION: the
#: union of every class's shapes. Keying on it re-minted 35 unchanged classes
#: every time an author added an aspect ratio.
#:
#: Naming note (pgw#1059 §F): this is the DECLARED envelope. The OTHER
#: "envelope" in this codebase is the artifact-metadata blob itself
#: (``kernel_path.envelope_block``, the hub error envelope). Where both
#: appear, write "the artifact-metadata envelope" vs "the declared envelope".
EXPORT_ENVELOPE_KEY = "declared_envelope"

#: The per-entry artifact's block naming the ONE graph class it carries.
ENTRY_BLOCK_KEY = "entry"


class CellKeyError(ValueError):
    """The artifact (or runtime) cannot state a required key axis."""


@dataclass(frozen=True)
class CellKey:
    """A computed ENTRY identity: canonical axes + their digest.

    One instance = one compiled graph class. The name is kept (rather than
    churned to ``EntryKey``) because every consumer of this type wants "the
    identity of the compiled thing", and pgw#1176 changed WHAT the compiled
    thing is, not what identity means.
    """

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


#: THE compiled-graph key grammar: a fragment-charset scheme, ``-``, then the
#: fixed-width lowercase-hex digest, anchored to the end. ``\Z`` and not ``$``
#: — ``$`` also matches before a trailing newline, and a key with a newline in
#: it is not a key. Bounded by :data:`MAX_FRAGMENT_LEN` separately, because the
#: length is the ref grammar's rule and the shape is this one's.
_KEY_RE = re.compile(r"[a-z0-9][a-z0-9._-]*-[0-9a-f]{%d}\Z" % _DIGEST_HEX)


def is_key(value: str) -> bool:
    """True when ``value`` is a compiled-graph key: ``<scheme>-<56 lowercase
    hex>``, whole token at most :data:`MAX_FRAGMENT_LEN` bytes of the ref
    fragment charset.

    THE CONTRACT, not a local opinion: this must answer identically to
    tensorhub's ``compilecache.IsCompiledGraphKey`` on every vector in
    ``tests/testdata/compiled_graph_key_vectors.json``, which is vendored
    byte-identically in both repos and fenced by
    ``scripts/grammar-vector-drift.sh``. th#1897 exists because there was no
    such file: both implementations were internally consistent, they disagreed
    about ``cg-key-v1``, and neither CI could see it — the disagreement was
    observable only on a GPU pod, 45 minutes into a compile, at the publish
    gate. Change this function and you are changing the corpus, in both repos,
    in one window.

    THE DIGEST IS THE SUFFIX, so **never split, partition or scan for the
    first** ``-``: the scheme may contain hyphens (``cg-key-v1``) and may even
    contain a hex run (the corpus carries that regression vector). The scheme
    is whatever precedes the anchored digest.

    Scheme-AGNOSTIC (th#1183): the grammar refuses SHAPE, never scheme. A key
    of a newer fleet's scheme is admitted to the candidate list and then ruled
    on by the axes that actually decide whether this runtime can execute it —
    the identity axes and the ingress contract — not by the label on it. This
    is the half of the contract that lets hub and fleet ship in different
    windows at all.
    """
    v = str(value or "")
    if len(v) > MAX_FRAGMENT_LEN:
        return False
    return _KEY_RE.match(v) is not None


def _refuse_key_shaped(where: str, name: str, value: str) -> None:
    """A KEY where a DIGEST belongs is a category error, not a bad value.

    pgw#1176: `cg-key-v1-` keys and the 16-hex fact digests are all `str`, so
    nothing in the type system distinguishes them — that is the honest reason
    a whole class of confusion type-checks cleanly. An opaque `NewType` cannot
    close it either at the places that matter: `is_key` is a BOUNDARY
    VALIDATOR whose entire job is to rule on untrusted strings, so it must
    take `str`.

    What IS closeable is the inverse direction, cheaply and here: an axis
    value, a class hash and a manifest input are all fact digests, and a
    key-shaped value in any of them means a caller passed the identity where
    the ingredient belongs. Refuse it by name at the constructor rather than
    letting it hash into a key nobody can restate.
    """
    if is_key(value):
        raise CellKeyError(
            f"{where}: {name}={value!r} is an ENTRY KEY where a fact digest "
            f"belongs. A key is the OUTPUT of this computation, never an "
            f"input to it — passing one here would hash an identity into "
            f"another identity and produce a key no artifact can restate.")


def from_axes(axes: Mapping[str, str]) -> CellKey:
    """Canonicalize an axes mapping into a :class:`CellKey`.

    Unknown axes are rejected TYPED — including every dropped axis name
    (``envelope`` since pgw#1176; ``contract``, ``env_seal``, ``kind``,
    ``format``, ``family``, ``lane`` since pgw#1059): a new axis is forbidden
    by the membership axiom, and a stale caller shipping a dropped one must
    fail here, not silently widen the key.
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
            _refuse_key_shaped("cell key axes", name, text)
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
    canonicalizer of the declared serving region, so no two consumers can
    canonicalize the same declaration differently.

    pgw#1176: a MANIFEST fact, never a key axis. It digests the UNION of the
    ladder across the whole declaration, which is a property of the
    collection; per entry the tracing-relevant half of it is already inside
    ``class_hash`` via ``range_digest`` and ``class_dims``.

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


#: Components a recorded ``toolchain`` block may carry that are NOT the
#: ``toolchain`` axis (pgw#1050). The eviction is a MEMBERSHIP change and
#: only that: identification of everything that stays is still the CONTENT
#: digest of the installed wheel's RECORD, never a version string (the
#: pgw#1050 amendment's own constraint — in a pinned-PyPI fleet versions and
#: content change at the same rate, so version strings would buy no churn
#: reduction and would accept the rebuilt/patched-wheel divergences content
#: catches).
#:
#: Read the module docstring for WHY these three are not members. Deny-list
#: rather than allow-list on purpose: a component nobody has classified must
#: stay IN the key, because the axiom's expensive failure is the over-split
#: and the axiom's UNSAFE failure is the under-split.
_NOT_TOOLCHAIN: Tuple[str, ...] = ("diffusers", "transformers", "peft")


def toolchain_facts(block: Mapping[str, Any]) -> Dict[str, str]:
    """The canonical form of one recorded TOOLCHAIN block — the single
    statement of what the ``toolchain`` axis IS.

    Applied at BOTH ends, exactly as :func:`envelope_facts` is: the producer
    (``compile_cache.toolchain_digest``) collects the components, and every
    reader that restates the axis from an artifact's recorded block
    (the publish recompute, the boot key, the arm handback, the wire
    identity) reads membership from here. Two ends that decide membership
    separately are two derivations of one axis, which is the failure
    ``test_cell_key_pgw1059``'s fence exists to prevent.
    """
    return {
        str(name): str(value) for name, value in block.items()
        if str(name) not in _NOT_TOOLCHAIN
    }


def toolchain_axis_digest(block: Mapping[str, Any]) -> str:
    """The ``toolchain`` key-axis value for one recorded toolchain block."""
    return facts_digest(toolchain_facts(block))


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


def from_entry_metadata(meta: Mapping[str, Any]) -> CellKey:
    """The key an EXPORTED (``aot-inductor``) ENTRY's OWN recorded facts
    describe — THE single implementation of entry identity: ``aot_mint``
    stamps what this returns and the publish path recomputes it, so the axes
    an entry is published under cannot drift from the axes its key was
    minted from.

    Every axis is read from a recorded block, never from a probe and never
    from the ``cell_key`` stamp. Raises :class:`CellKeyError` when a fact is
    missing: an entry that cannot name an axis has no identity, and must not
    be published under a partial one.

    ``graph`` is READ from the entry's ``class_hash``, never re-derived here
    — the one derivation lives in ``aot_serve.class_hash``, stamped by
    ``aot_serve.entry_metadata``, and admission
    (``aot_serve.verify_contract``) recomputes it from the staged bytes so a
    forged stamp is refused before it can arm.

    Pre-pgw#1176 artifacts fail here STRUCTURALLY: they record an ``entries``
    MAP and a ``combined_graph_hash`` rather than one ``entry`` block, so a
    36-entry cell can never restate a per-entry identity. That is what makes
    the ck1 corpus purge hygiene rather than a correctness precondition.
    """
    kind = str(meta.get("kind") or "")
    if kind != EXPORTED_KIND:
        raise CellKeyError(
            f"artifact kind {kind!r} has no entry-key identity: only exported "
            f"{EXPORTED_KIND!r} entries are keyed (pgw#1010/pgw#1059 — JIT is "
            "intake, local torch-inductor-cache artifacts compare facts via "
            "compile_cache.local_cell_mismatch)")
    sm = str(meta.get("sm") or "")
    if not sm:
        raise CellKeyError(
            "cannot state the compute capability (sm) of this runtime; an "
            "exported entry has no identity without it — mint on the target GPU")
    entry = meta.get(ENTRY_BLOCK_KEY)
    if not isinstance(entry, Mapping) or not entry:
        raise CellKeyError(
            f"artifact records no {ENTRY_BLOCK_KEY!r} block; the atom is ONE "
            "graph class (pgw#1176) and an artifact that cannot name its "
            "class has no identity")
    graph = str(entry.get("class_hash") or "")
    if not graph:
        raise CellKeyError(
            f"entry {str(entry.get('name') or '')!r} carries no class_hash; a "
            "class the key cannot name is a class a mismatch cannot name "
            "(pgw#716)")
    _refuse_key_shaped(
        f"entry {str(entry.get('name') or '')!r}", "class_hash", graph)
    toolchain = meta.get("toolchain")
    if not isinstance(toolchain, dict) or not toolchain:
        raise CellKeyError(
            "artifact records no toolchain block; no recipe identity")
    return from_axes({
        "graph": graph,
        "sm": sm,
        "toolchain": toolchain_axis_digest(toolchain),
    })


# --- the CONTRACT MANIFEST: a derived view, never an artifact ---------------


def manifest_digest(class_hashes: Iterable[str]) -> str:
    """The coverage LABEL of one declaration's class set — 16 hex of sha256
    over the newline-joined SORTED per-class hashes.

    This is ``combined_graph_hash``'s arithmetic, VERBATIM (pgw#716: sorted
    by the hash string itself, single ``\\n`` joins, no trailing newline,
    UTF-8 bytes) and NOT its job. pgw#1176 demotes it from identity to a
    telemetry/coverage label:

    * it is what the hub folds compile-health rows under — one row per
      ``(manifest_digest, sm, toolchain)``, coverage n/m, last failure per
      class (§1.7). It is deliberately sm- and toolchain-FREE so that tuple
      is not degenerate;
    * nothing resolves, downloads, verifies or arms it. An artifact is one
      entry, addressed by :func:`from_entry_metadata`.

    A manifest is a VIEW. The moment it becomes something a pod downloads or
    a hub hands back as one row, the wrong atom is back.
    """
    rows = [str(h) for h in class_hashes]
    for row in rows:
        _refuse_key_shaped("manifest digest", "class_hash", row)
    joined = "\n".join(sorted(rows))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]


__all__ = [
    "KEY_AXES",
    "KEY_SCHEME",
    "EXPORTED_KIND",
    "EXPORT_ENVELOPE_KEY",
    "ENTRY_BLOCK_KEY",
    "CellKey",
    "CellKeyError",
    "SlotSubject",
    "subject_digest",
    "subject_facts",
    "envelope_digest",
    "envelope_facts",
    "facts_digest",
    "manifest_digest",
    "toolchain_axis_digest",
    "toolchain_facts",
    "from_entry_metadata",
    "from_axes",
    "is_key",
]
