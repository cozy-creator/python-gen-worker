"""The VENDORED tensor-layout v2 corpus is what it claims to be.

**What this file used to be, and why none of it could survive verbatim.**
pgw#1391 pinned a pure-Python PORT of tensorfs's v1 contract validator against
upstream's released `spec/v1/contract-vectors` corpus — every golden digest,
every typed refusal — and resolved the library vectors against the 30 documents
this repo vendored, so a bad re-vendor failed here. pgw#1621 deleted the port,
the corpus and the documents: `contract.py`, `contracts.py` and
`spec/v1/contracts` exist at NO tensorfs rev on master. **A conformance oracle
whose subject is deleted is not a gate that got weaker; it is a gate with
nothing left to check.** So this file states the v2 property instead, and
`tests/testdata/contract-vectors/` went with it — a vector corpus nothing reads
is worse than absent, because it reads as authoritative to whoever finds it.

**Under v2 the vendored corpus is pinned TWO ways, and both are asserted.**

1. **FILE BYTES** — `_vendor/VENDORED.toml`'s `[packages.tensorfs.files]`
   sha256 table, enforced as a SET EQUALITY by
   `test_vendored_snapshot_pgw1310.py::test_every_vendored_file_matches_its_recorded_digest`.
   That catches a hand-patch and a partial re-vendor. What it cannot catch is a
   corpus that is internally inconsistent — the bytes would be recorded
   faithfully either way — so it is not the whole story, and this file asserts
   the `spec/v2/**` files are actually IN that inventory rather than sitting on
   disk unfenced (which is the failure v1's third-document drift actually took).

2. **SEMANTIC IDENTITY** — `spec/v2/vectors/digests.json`, which pins each
   topology / quant-rule / morphism document's canonical digest. Its own
   docstring says *"A reader that holds a corpus checks it against this."*
   Nothing in this repo did. `test_the_corpus_agrees_with_its_own_digest_vectors`
   is that reader.

   ⚠️ **WHAT THAT DOES AND DOES NOT PROVE.** The digest is over **Go's**
   canonical rendering, and the canonicalizer lives in Go on purpose (a second
   one would let two spellings of a rule disagree about its identity), so
   nothing here RECOMPUTES it. What is proved: every document the corpus holds
   has a row in `digests.json` and every row has a document — set equality in
   BOTH directions, so neither a vendored-but-unpinned document nor a
   pinned-but-missing one can pass — and each document's own embedded `digest`
   field equals its row, which is a real cross-check because those are two
   independently written places. What is NOT proved: that either number is the
   digest Go would compute. Only upstream's own test can say that, and the
   sibling-drift check below is how this repo borrows it.

**The unratified-candidate ledger has no v2 successor.** v1's generated
documents carried "NOT RATIFIED" in their own `description` and this file
enumerated them, expecting to go red one at a time. No v2 document carries such
a marker: ratification is upstream's `CORPUS.tsv` gate now, and every vendored
document is past it. `test_no_document_smuggles_an_unratified_marker` asserts
that absence, so if the practice ever returns it returns visibly.
"""

from __future__ import annotations

import json
import subprocess
import tomllib
from pathlib import Path

import pytest

from gen_worker.models.tensor_layout_contract import (
    LayoutDeclarationError,
    display_names,
    known_quant_rules,
    known_topologies,
    parse_lane_stamp,
    topologies,
)

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "src" / "gen_worker" / "_vendor"
SPEC_V2 = VENDOR / "tensorfs" / "spec" / "v2"

#: The three catalogued document kinds, and the key each is filed under in
#: `vectors/digests.json`. Kept as a pair so a fourth kind cannot be added
#: upstream and quietly go unchecked here: the set equality below is per kind
#: AND the kind set itself is asserted.
KINDS: tuple[tuple[str, str], ...] = (
    ("rules", "rules"),
    ("topologies", "topologies"),
    ("morphisms", "morphisms"),
)

#: Non-vacuity floors. Read as "the corpus moved or the glob broke", never as a
#: target: an empty glob makes every set-equality below trivially true, which is
#: the failure mode this whole file is about.
#: Bumped 21 -> 28 by the tensorfs#152 (`ac9c9d4`) re-vendor IN THE SAME
#: CHANGE, which is what this constant asks for. `rules` did NOT move, so
#: `lane_ladder._RULE_BODY` needed no new row — the fence in
#: `test_lane_dtype_fence_pgw1606` would have failed if it had.
EXPECTED_COUNTS = {"rules": 8, "topologies": 28, "morphisms": 5}


def _documents(kind: str) -> dict[str, dict]:
    """handle -> parsed document, for one kind."""
    out: dict[str, dict] = {}
    for path in sorted((SPEC_V2 / kind).glob("*.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        out[f"{doc['name']}@{doc['version']}"] = doc
    return out


VECTORS = json.loads((SPEC_V2 / "vectors" / "digests.json").read_text())


# ── non-vacuity first ────────────────────────────────────────────────────────


def test_the_vendored_corpus_is_actually_there() -> None:
    """Read the COUNT, not the verdict."""
    for kind, _ in KINDS:
        docs = _documents(kind)
        assert len(docs) == EXPECTED_COUNTS[kind], (
            f"{kind}: found {len(docs)} documents under {SPEC_V2 / kind}, "
            f"expected {EXPECTED_COUNTS[kind]}. The vendored corpus moved, the "
            f"glob broke, or upstream ratified a new document — in the last "
            f"case bump this number IN THE SAME CHANGE that re-vendors, and "
            f"check `lane_ladder._RULE_BODY` grew a row "
            f"(test_lane_dtype_fence_pgw1606 fails if it did not)."
        )

    # ...and the module-level readers agree with the raw glob, so the API this
    # repo actually uses cannot drift away from the files on disk.
    assert set(known_quant_rules()) == set(_documents("rules"))
    assert set(known_topologies()) == set(_documents("topologies"))


# ── pin 2: semantic identity ─────────────────────────────────────────────────


def test_the_corpus_agrees_with_its_own_digest_vectors() -> None:
    """`vectors/digests.json` is the corpus's own identity ledger. Check it.

    See this module's docstring for exactly what this proves: set equality in
    both directions, plus each document's embedded `digest` against its row.
    It does NOT recompute Go's canonicalisation and does not pretend to.
    """
    assert {k for k, _ in KINDS} <= set(VECTORS), (
        f"digests.json lost a document kind: it carries {sorted(VECTORS)}"
    )
    unknown_kinds = set(VECTORS) - {k for k, _ in KINDS} - {"_"}
    assert not unknown_kinds, (
        f"digests.json pins a document kind this reader does not check: "
        f"{sorted(unknown_kinds)}. Add it to KINDS — an unchecked kind is a "
        f"corpus half nothing verifies."
    )

    for kind, vector_key in KINDS:
        docs = _documents(kind)
        pinned = dict(VECTORS[vector_key])

        # BOTH directions. A vendored-but-unpinned document is a layout with
        # no ratified identity; a pinned-but-missing one is a stamp the hub
        # can name and this image cannot read.
        assert set(docs) == set(pinned), (
            f"{kind}: vendored and pinned sets differ — "
            f"vendored only {sorted(set(docs) - set(pinned))}, "
            f"pinned only {sorted(set(pinned) - set(docs))}"
        )

        # ...and the two independently-written copies of each digest agree.
        for handle, doc in docs.items():
            assert doc.get("digest"), f"{handle} carries no digest field"
            assert doc["digest"] == pinned[handle], (
                f"{handle}: the document says {doc['digest'][:16]}… and "
                f"digests.json says {pinned[handle][:16]}…. One of the two was "
                f"hand-edited; a change to a digest is a change to what a "
                f"stamp MEANS and must be a VERSION BUMP upstream, never an "
                f"edit."
            )


def test_no_document_smuggles_an_unratified_marker() -> None:
    """v1's generated candidates said "NOT RATIFIED" in their own description
    and this file enumerated them. No v2 document does — ratification is
    upstream's `CORPUS.tsv` gate. If the practice returns, it returns here."""
    offenders = []
    for kind, _ in KINDS:
        for handle, doc in _documents(kind).items():
            if "NOT RATIFIED" in str(doc.get("description", "")).upper():
                offenders.append(handle)
    assert not offenders, (
        f"these vendored documents declare themselves unratified: {offenders}. "
        f"A generated candidate is shaped exactly like a ratified one and the "
        f"bind gate cannot tell them apart — it will refuse a real checkpoint, "
        f"or admit a wrong one, with equal confidence. Either ratify them "
        f"upstream or stop vendoring them."
    )


# ── pin 1's coverage: the spec files are inside the byte inventory ───────────


def test_every_spec_v2_file_is_inside_the_single_digest_inventory() -> None:
    """pgw#1575's property, re-aimed at v2.

    The digest fence itself lives in `test_vendored_snapshot_pgw1310.py` and is
    a SET EQUALITY over `rglob("*")`, so this is not a second fence — it is the
    assertion that the corpus is in the inventory's SCOPE at all, which is the
    failure v1's third-document drift actually took (files on disk, unfenced).
    """
    manifest = tomllib.loads((VENDOR / "VENDORED.toml").read_text())
    spec = manifest["packages"]["tensorfs"]
    assert len(spec["rev"]) == 40
    recorded = set(spec["files"])

    root = VENDOR / "tensorfs"
    on_disk = {
        p.relative_to(root).as_posix()
        for p in (SPEC_V2).rglob("*")
        if p.is_file() and "__pycache__" not in p.parts
    }
    assert len(on_disk) >= sum(EXPECTED_COUNTS.values()) + 2, len(on_disk)
    assert on_disk <= recorded, (
        f"spec/v2 files are on disk and NOT in [packages.tensorfs.files]: "
        f"{sorted(on_disk - recorded)}. They ship unfenced — a hand-edit would "
        f"pass every gate."
    )
    # And the v1 plane really is gone rather than merely unused.
    assert not any(name.startswith("_contracts/") for name in recorded)
    assert "contract.py" not in recorded and "contracts.py" not in recorded
    assert not (root / "_contracts").exists()
    assert "layout2.py" in recorded


def test_the_v1_vocabulary_is_UNIMPORTABLE_not_merely_unused() -> None:
    """A module that still imports is a module something can still reach."""
    for dead in ("gen_worker._vendor.tensorfs.contract",
                 "gen_worker._vendor.tensorfs.contracts"):
        with pytest.raises(ModuleNotFoundError):
            __import__(dead)

    import gen_worker._vendor.tensorfs as vendored

    for symbol in ("Contract", "ContractError", "MissingDtype", "contracts"):
        assert not hasattr(vendored, symbol), (
            f"{symbol} is re-exported from the vendored package again; the v1 "
            f"contract plane is deleted UPSTREAM (spec/v1/contracts exists at "
            f"no tensorfs rev on master)."
        )


# ── the DRIFT check against the sibling checkout, at the PINNED rev ──────────


def _sibling_tensorfs() -> Path:
    """The neighbouring tensorfs checkout, from a worktree OR the main clone.

    The old spelling was `ROOT.parent.parent.parent / "tensorfs"`, which is
    correct for exactly one layout. From a worktree
    (`~/cozy/.worktrees/python-gen-worker/<branch>`) it lands on
    `~/cozy/tensorfs` and the drift check runs; from the MAIN checkout
    (`~/cozy/python-gen-worker`) the same three hops land on `/home/tensorfs`,
    which does not exist, so it SKIPS. An instrument that is invisible exactly
    where reviewers and CI run it reports green by not looking.
    """
    for ancestor in ROOT.parents:
        candidate = ancestor / "tensorfs"
        if (candidate / ".git").exists():
            return candidate
    return ROOT.parent / "tensorfs"  # a stable non-existent path -> honest skip


UPSTREAM = _sibling_tensorfs()


def _upstream_blob(path: str) -> bytes | None:
    """One upstream file AT the pinned `rev`, or None if unavailable.

    THE REV, NOT THE SIBLING'S WORKING TREE, and the difference is the whole
    point of pinning one. Reading `UPSTREAM / <path>` would assert that this
    repo is always at tensorfs TIP, which contradicts the pin existing at all
    and goes red for reasons that have nothing to do with the change under
    test. A snapshot is faithful to the REV IT RECORDS; that is what is
    asserted.
    """
    if not (UPSTREAM / ".git").exists():
        return None
    manifest = tomllib.loads((VENDOR / "VENDORED.toml").read_text())
    rev = manifest["packages"]["tensorfs"]["rev"]
    done = subprocess.run(
        ["git", "-C", str(UPSTREAM), "show", f"{rev}:{path}"],
        capture_output=True,
    )
    # An unfetched rev is "cannot check here", never "drifted": the sibling is
    # a convenience, and CI has no sibling at all.
    return done.stdout if done.returncode == 0 else None


def test_the_vendored_corpus_matches_the_pinned_upstream_rev() -> None:
    """The vendored corpus is a SNAPSHOT; when the real repo is beside us,
    prove the snapshot has not drifted from the rev it claims.

    This is also the only thing in this repo that can vouch for Go's digests:
    if these bytes are upstream's bytes at the pinned rev, then upstream's own
    conformance tests ran against them.
    """
    if _upstream_blob("spec/v2/README.md") is None:
        pytest.skip("no sibling tensorfs checkout carrying the pinned rev")

    checked = 0
    for path in sorted(SPEC_V2.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        relative = "spec/v2/" + path.relative_to(SPEC_V2).as_posix()
        theirs = _upstream_blob(relative)
        assert theirs is not None, (
            f"{relative} is vendored and does not exist at the pinned rev — "
            f"either it was hand-added here, or the rev is wrong."
        )
        assert path.read_bytes() == theirs, f"{relative} drifted from upstream"
        checked += 1

    # A loop that checked nothing is not a passing drift check. Read the
    # COUNT, not the verdict.
    assert checked >= sum(EXPECTED_COUNTS.values()) + 2, (
        f"the drift check only compared {checked} files"
    )


# ── the vocabulary this repo actually consumes ───────────────────────────────


def test_every_display_name_names_a_PAIR_THAT_EXISTS() -> None:
    """A display name is prose for humans and gates nothing — but a row whose
    pair is not in the corpus is a dangling name a refusal would print at an
    operator, so the KEYS are checked even though the values are not."""
    names = display_names()
    assert names, "the display-name table is empty — every check below is vacuous"
    for pair, name in names.items():
        # The KEY must round-trip through the real parser, which is what
        # refuses a half that is not in the corpus.
        stamp = parse_lane_stamp(pair, where="display-names.json")
        assert stamp.render() == pair
        assert name.strip(), f"{pair} has a blank display name"
    # Counted AFTER the per-row check, so a dangling row reports as a dangling
    # row rather than as a count that moved.
    # 18 -> 24 with tensorfs#152: seven lanes gained a successor (flux1,
    # stable-audio, trellis2, qwen-image, internvl-u, krea-2, rife) and one
    # pair was already display-named. Growing this is fine; saying so is the point.
    assert len(names) == 24, (
        f"the ratified display-name table has {len(names)} rows; it had 18 "
        f"when this was written. Growing it is fine — say so here."
    )


def test_a_display_name_is_NEVER_a_handle_and_musicgen_is_the_proof() -> None:
    """The trap that makes string surgery on a display name a defect.

    `musicgen.transformers@1+plain.f16@1` is displayed as
    `musicgen.transformers-fp16@1` — **`fp16` in the name, `f16` in the rule
    handle**. Deriving one from the other by suffix-stripping produces
    `plain.fp16@1`, which is in no corpus and refuses; deriving it the other
    way produces a display name nobody has ever seen. There is no alias
    resolution anywhere on purpose, and this is why.
    """
    names = display_names()
    assert names["musicgen.transformers@1+plain.f16@1"] == (
        "musicgen.transformers-fp16@1"), (
        "musicgen's display name no longer disagrees with its rule handle. "
        "That is fine upstream — but this test is the standing PROOF that a "
        "display name is not a handle, so find the next row where the two "
        "spellings differ and pin that one instead. If no row differs any "
        "more, string surgery would appear to work, which is the trap."
    )
    assert "plain.fp16@1" not in known_quant_rules()

    # The old spelling resolves to NOTHING as a lane key — it only ever
    # appears inside the refusal, as a hint pointing at the real pair.
    with pytest.raises(LayoutDeclarationError) as caught:
        parse_lane_stamp("musicgen.transformers-fp16@1", where="test")
    assert "musicgen.transformers@1+plain.f16@1" in str(caught.value)


def test_no_vendored_topology_is_claimed_by_two_model_types() -> None:
    """tensorfs#124's warning, re-aimed at the v2 topology half.

    `model_type_for_contract` returns the FIRST matching type, so a tie would
    be resolved by declaration ORDER — a silent, order-dependent
    classification. pgw#1621 narrowed the match to the TOPOLOGY half of the
    pair, which is what makes this checkable at all: a fingerprint that could
    see the quant half would change when nothing about the architecture did.
    """
    from fnmatch import fnmatchcase

    from gen_worker.models.model_types import MODEL_TYPES, model_type_for_contract

    ties = {}
    for topology in known_topologies():
        hits = [
            mt.name for mt in MODEL_TYPES
            if any(fnmatchcase(topology, pattern) for pattern in mt.contracts)
        ]
        if len(hits) > 1:
            ties[topology] = hits
    assert not ties, f"topologies claimed by more than one model type: {ties}"

    # ...and the classifier really is reading the topology half: the SAME
    # topology under two different quant rules classifies identically, and an
    # unrecognised topology answers None (unclassified is LEGAL and VISIBLE).
    for quant in ("plain.bf16@1", "cozy.fp8-rowwise@1", "cozy.nvfp4-flat@1"):
        got = model_type_for_contract(f"sdxl.diffusers@1+{quant}")
        assert got is not None and got.name == "sdxl", quant
    assert model_type_for_contract("nope.nothing@1+plain.bf16@1") is None


def _fixture_lane():
    """A minimal but REAL lane declaration (pgw#1599). These tests are about
    the STAMP half of the header, so the demand formula is the simplest legal
    one."""
    from gen_worker import lane
    from gen_worker.demand import GiB, const

    return lane(request=const(GiB(1)))


def test_a_lane_naming_no_ratified_document_refuses_at_DECLARATION() -> None:
    """The seam the DISCOVERY guarantee rides on: discovery imports the author
    module, so `__init_subclass__` is what refuses — before any author code
    runs, and before a pod ever fetches bytes.

    Both halves are checked, separately, and each refusal names its own
    remedy: a topology is EXTRACTED from banked headers, a rule is AUTHORED
    and ratified. A single "not registered" for both would send an author to
    the wrong place half the time.
    """
    from gen_worker.models import SDXL
    from gen_worker.serving import Model, ModelDeclarationError

    # The absent handle is DERIVED, never hardcoded. This case used to name
    # `flux1.diffusers@1` — and tensorfs#152 banked flux1, so the test stopped
    # testing anything and said "DID NOT RAISE". A name chosen because nobody
    # has ratified it yet is a name that can be ratified; asserting the absence
    # first is what keeps the case honest whatever the corpus grows to.
    absent_topology = "nonesuch-family.diffusers@1"
    assert absent_topology not in known_topologies(), (
        f"{absent_topology} is vendored now — pick another absent handle; the "
        f"point of this case is a handle the corpus does NOT carry"
    )

    with pytest.raises(ModelDeclarationError, match="topology .* not in the vendored"):

        class BadTopology(
            Model[SDXL], lanes={(absent_topology, "plain.bf16@1"): _fixture_lane()}
        ):
            def load(self, ctx: object) -> None: ...

    # Same discipline for the rule half. `cozy.q4-k@1` is the natural example
    # (a GGUF block quant has no ratified v2 rule) and is exactly the kind of
    # handle that gets authored one day.
    absent_rule = "cozy.q4-k@1"
    assert absent_rule not in known_quant_rules(), (
        f"{absent_rule} is ratified now — pick another absent rule handle"
    )

    with pytest.raises(ModelDeclarationError, match="quant .* not in the vendored"):

        class BadRule(
            Model[SDXL],
            lanes={("sdxl.diffusers@1", absent_rule): _fixture_lane()},
        ):
            def load(self, ctx: object) -> None: ...

    # `requires=` is deleted and the refusal names the spelling that replaces
    # it — a silently-ignored floor is exactly what this class of bug was.
    with pytest.raises(ModelDeclarationError, match="requires.*is DELETED"):

        class Floored(Model[SDXL], requires={"sdxl.diffusers@1": "vram12g"}):
            def load(self, ctx: object) -> None: ...


def test_flux1_and_klein_are_TWO_topologies_that_v1_could_not_have_separated() -> None:
    """tensorfs#124's finding, and what v2 does about it.

    v1 shipped `flux1.diffusers-bf16@1` and `flux2-klein.diffusers-bf16@1` as
    two documents precisely because the quieter hazard was BORROWING: measured
    upstream, `flux2-klein` explained 308 of a FLUX.1 transformer's 1160
    tensors with no dtype or rank refusal, so it won every FLUX.1 file
    outright. The v1 schema is SHAPELESS, so a large shared key set looked
    exactly like a match.

    This test used to assert that NO flux1 topology existed and to say it would
    go red when the headers were banked. tensorfs#152 banked them and it went
    red — and the honest successor is not to delete it, because the
    anti-borrowing property is now MEASURABLE where v1 could only hope for it.

    Everything below is measured off the vendored corpus rather than recalled.
    """
    from gen_worker.models import Flux1, Flux2Klein
    from gen_worker.serving import Model, model_declared_lanes

    tops = topologies()
    flux1, klein = tops["flux1.diffusers@1"], tops["flux2-klein.diffusers@1"]

    def named(one: object) -> set[str]:
        return {k for tensors in one.values() for k in tensors}  # type: ignore[union-attr]

    shared = named(flux1) & named(klein)
    assert len(shared) == 348, (
        f"the flux1/klein shared key set is {len(shared)}, was 348. That number "
        f"IS the v1 hazard — it is how much of FLUX.1 a shapeless klein document "
        f"could explain. If it moved, one of the two was re-extracted."
    )

    # THE SEPARATION, three independent ways. Any one would do; all three are
    # asserted because v1 had none of them.
    assert sorted(flux1) != sorted(klein)
    assert len(named(flux1)) != len(named(klein))
    differing = [
        k for k in shared
        for a in (next(c[k] for c in flux1.values() if k in c),)
        for b in (next(c[k] for c in klein.values() if k in c),)
        if a != b
    ]
    assert len(differing) == 6, (
        f"{len(differing)} shared keys differ in SHAPE, expected 6. This is the "
        f"refusal v1 could not express at all: same name, same rank, different "
        f"dimensions."
    )

    # And each declares independently, naming its own pair.
    class KleinModel(
        Model[Flux2Klein],
        lanes={("flux2-klein.diffusers@1", "plain.bf16@1"): _fixture_lane()},
    ):
        def load(self, ctx: object) -> None: ...

    class Flux1Model(
        Model[Flux1],
        lanes={("flux1.diffusers@1", "plain.bf16@1"): _fixture_lane()},
    ):
        def load(self, ctx: object) -> None: ...

    assert [r.contract_id for r in model_declared_lanes(KleinModel)] == [
        "flux2-klein.diffusers@1+plain.bf16@1"
    ]
    assert [r.contract_id for r in model_declared_lanes(Flux1Model)] == [
        "flux1.diffusers@1+plain.bf16@1"
    ]
