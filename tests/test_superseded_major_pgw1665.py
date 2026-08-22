"""pgw#1665 — a SUPERSEDED major RESOLVES at the declaration boundary; a major
AHEAD of the vendored corpus refuses TYPED.

**The production case, stated once.** tensorfs#153 does not add `@2` beside
`@1`; it REPLACES the document. Nine topology records, one quant rule and one
morphism moved that way, and `sdxl.diffusers` is one of them — half the lane
contract id of the lane the tcg#90 acceptance run served. Every endpoint on the
platform writes `lanes={("sdxl.diffusers@1", "plain.bf16@1"): lane(...)}` and
every committed `endpoint.lock` carries that spelling, so on a literal
membership test the whole fleet refuses at CLASS DEFINITION — before author code
runs, before a pod fetches a byte, and identically under `gen-worker lock
--check`, which reaches the corpus by importing the author module.

This is th#2301's hub-side `tensorlayout.ResolveDeclaredContract`, applied to
the other end of the same wire, and the arms below are its arms: superseded
resolves, ahead refuses typed, an unknown NAME still refuses as an unknown name,
an exact member is never "upgraded", and the DECLARED SPELLING SURVIVES —
resolution happens at corpus consultation, never by rewriting what the author
wrote.

**The fixtures are not re-spelled, deliberately.** `release_fixtures/
lane_contracts.py` declares `sd15.diffusers@1`, which is one of the nine. A
fixture declaring `@1` against an `@2` corpus IS the production case; changing
it to `@2` would delete the only end-to-end evidence that the fleet's real
declarations still derive.
"""

from __future__ import annotations

import json
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

import pytest

from gen_worker.models import tensor_layout_contract as tlc
from gen_worker.models.tensor_layout_contract import (
    AXIS_QUANT,
    AXIS_TOPOLOGY,
    ContractDescriptorUnprovenError,
    ContractVersionAheadError,
    LayoutDeclarationError,
    known_quant_rules,
    known_topologies,
    parse_lane_stamp,
    resolve_declared_handle,
)

if TYPE_CHECKING:
    from gen_worker.discovery.decode_set import DecodeSet

#: THE MIGRATION, ENUMERATED BY NAME. `201c32e1` -> `07f9615`: these records
#: exist ONLY at `@2` now — the `@1` file is deleted, not beside it — so every
#: one of them is a live `@1` declaration that must resolve.
SUPERSEDED_TOPOLOGIES: tuple[str, ...] = (
    "ernie.diffusers",
    "ltx2-upsampler.diffusers",
    "sd15.diffusers",
    "sd2.diffusers",
    "sdxl-inpainting.diffusers",
    "sdxl.clip-g-fused",
    "sdxl.diffusers",
    "wan22.diffusers",
    "z-image.diffusers",
)

#: The ONE quant rule that moved, and the reason th#2301 exists hub-side: 358 of
#: 358 released images declare it at `@1`.
SUPERSEDED_RULE = "cozy.fp8-rowwise"

#: Records that gained an `@2` while KEEPING their `@1`. These must NOT resolve
#: — an exact member is answered exactly, or a declaration silently changes
#: meaning under a corpus bump.
ADDITIVE_AT_V2: tuple[str, ...] = ("anima.net", "minimax-h3.diffusers")

#: Net-new names at `@1`. `sensenova-u1.mot@1` is tensorfs#161 and is what
#: pgw#1664 is blocked on; asserting it here is what makes the rev choice
#: falsifiable rather than a claim in a PR body.
NEW_AT_V1: tuple[str, ...] = (
    "flux1-schnell.diffusers@1",
    "flux2-klein-9b.diffusers@1",
    "sensenova-u1.mot@1",
)


def test_the_migration_is_exactly_these_records() -> None:
    """The enumeration, asserted against the corpus on disk.

    A COUNT cannot say this: `rules` holds eight documents before and after,
    because `cozy.fp8-rowwise` moved IN PLACE. Only naming the versions can.
    """
    tops, rules = set(known_topologies()), set(known_quant_rules())

    for name in SUPERSEDED_TOPOLOGIES:
        assert f"{name}@2" in tops, f"{name}@2 is not vendored"
        assert f"{name}@1" not in tops, (
            f"{name}@1 is back in the corpus. This test's whole subject is that "
            f"tensorfs#153 REPLACED it — if upstream restored it, the migration "
            f"changed shape and the resolution below is answering a question "
            f"nobody is asking any more."
        )
    assert f"{SUPERSEDED_RULE}@2" in rules and f"{SUPERSEDED_RULE}@1" not in rules

    for name in ADDITIVE_AT_V2:
        assert {f"{name}@1", f"{name}@2"} <= tops, (
            f"{name} no longer carries BOTH majors, so the 'exact match is "
            f"never upgraded' arm below has lost its subject."
        )

    for handle in NEW_AT_V1:
        assert handle in tops, f"{handle} is not vendored"

    # The morphism half moved too. Nothing in this repo parses a morphism
    # handle, so it gets no resolution — but it is part of the migration and an
    # unstated half is how the next lane mis-measures the blast radius.
    morphisms = {
        p.name for p in
        (Path(tlc._SPEC_V2) / "morphisms").glob("sdxl.clip-g-split-to-fused.*")
    }
    assert morphisms == {"sdxl.clip-g-split-to-fused.v2.json"}, morphisms


# ── RESOLUTION ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("name", SUPERSEDED_TOPOLOGIES)
def test_a_superseded_topology_major_resolves_by_name(name: str) -> None:
    assert resolve_declared_handle(
        f"{name}@1", axis=AXIS_TOPOLOGY, where="test") == f"{name}@2"


def test_a_superseded_quant_major_resolves_and_carries_the_rules_facts() -> None:
    """The fp8 lane: 44 references in this repo, 358 released images hub-side."""
    declared = f"{SUPERSEDED_RULE}@1"
    assert resolve_declared_handle(
        declared, axis=AXIS_QUANT, where="test") == f"{SUPERSEDED_RULE}@2"
    # ...and the facts a lane derives from the rule come out of the RESOLVED
    # document rather than raising. These two reads are what a serving pod does.
    assert tlc.rule_dtype(declared) == tlc.rule_dtype(f"{SUPERSEDED_RULE}@2")
    assert tlc.capability_floor_for_rule(declared) == tlc.capability_floor_for_rule(
        f"{SUPERSEDED_RULE}@2")


@pytest.mark.parametrize("name", ADDITIVE_AT_V2)
def test_an_exact_member_is_never_upgraded(name: str) -> None:
    """`anima.net@1` still EXISTS. Answering `@2` for it would silently change
    what a declaration means — the failure resolution exists to prevent, arriving
    from the other direction."""
    assert resolve_declared_handle(
        f"{name}@1", axis=AXIS_TOPOLOGY, where="test") == f"{name}@1"


def test_the_declared_spelling_survives_resolution_end_to_end() -> None:
    """The property th#2301 states as "nothing is rewritten and nothing is
    migrated": the stored row keeps what the image DECLARED.

    It matters here for a reason it does not hub-side — the rendered
    `contract_id` is a derivation input of the graph identity, so a worker that
    rewrote `@1` to `@2` would re-key every artifact of every endpoint on a
    corpus bump. Resolution is a lookup, not a rename.
    """
    from gen_worker import lane
    from gen_worker.demand import GiB, const
    from gen_worker.models import SDXL
    from gen_worker.serving import Model, model_declared_lanes

    class Declares1(
        Model[SDXL],
        lanes={("sdxl.diffusers@1", f"{SUPERSEDED_RULE}@1"): lane(
            request=const(GiB(1)))},
    ):
        def load(self, ctx: object) -> None: ...

    rows = model_declared_lanes(Declares1)
    assert [r.contract_id for r in rows] == [
        f"sdxl.diffusers@1+{SUPERSEDED_RULE}@1"]
    assert rows[0].topology == "sdxl.diffusers@1"
    assert rows[0].quant == f"{SUPERSEDED_RULE}@1"
    # ...while the DERIVED facts came from the @2 documents, which is the whole
    # point: the declaration is preserved and the corpus is consulted.
    assert rows[0].dtype == "float8_e4m3fn"
    assert rows[0].min_sm == tlc.capability_floor_for_rule(f"{SUPERSEDED_RULE}@2")
    # The display name is prose, and it survives the bump too: the table is
    # re-keyed to @2 pairs, so a literal lookup would have blanked it.
    assert rows[0].display_name


def test_parse_lane_stamp_admits_the_fleets_spelling_both_ways() -> None:
    """The tuple an author writes and the rendered form a document carries."""
    pair = ("sdxl.diffusers@1", "plain.bf16@1")
    assert parse_lane_stamp(pair, where="t").render() == (
        "sdxl.diffusers@1+plain.bf16@1")
    assert parse_lane_stamp(
        "sdxl.diffusers@1+plain.bf16@1", where="t").render() == (
        "sdxl.diffusers@1+plain.bf16@1")


# ── THE REFUSALS ──────────────────────────────────────────────────────────────


def test_a_major_AHEAD_of_the_corpus_refuses_TYPED() -> None:
    """The line th#2295's wire resolution did not draw.

    Superseded and ahead are OPPOSITE operator actions — re-declare the
    endpoint, versus re-vendor the worker — so they are two errors, and reading
    a version this build has never seen as the one it holds would decode
    plausible numbers instead of failing.
    """
    with pytest.raises(ContractVersionAheadError) as caught:
        resolve_declared_handle("sdxl.diffusers@3", axis=AXIS_TOPOLOGY, where="t")
    assert "sdxl.diffusers@2" in str(caught.value)

    with pytest.raises(ContractVersionAheadError):
        resolve_declared_handle(
            f"{SUPERSEDED_RULE}@9", axis=AXIS_QUANT, where="t")

    # ...and it arrives at the DECLARATION, through the real class machinery.
    from gen_worker import lane
    from gen_worker.demand import GiB, const
    from gen_worker.models import SDXL
    from gen_worker.serving import Model, ModelDeclarationError

    with pytest.raises(ModelDeclarationError, match="behind the declarer"):

        class Ahead(
            Model[SDXL],
            lanes={("sdxl.diffusers@3", "plain.bf16@1"): lane(
                request=const(GiB(1)))},
        ):
            def load(self, ctx: object) -> None: ...


def test_an_unknown_NAME_still_refuses_as_an_unknown_name() -> None:
    """Resolution must not turn an invented handle into a version complaint.

    `None` here is what keeps each call site's own remedy — a topology is
    EXTRACTED from banked headers, a rule is AUTHORED and ratified — attached to
    the refusal an author actually reads.
    """
    assert resolve_declared_handle(
        "nonesuch-family.diffusers@1", axis=AXIS_TOPOLOGY, where="t") is None
    assert resolve_declared_handle(
        "cozy.q4-k@1", axis=AXIS_QUANT, where="t") is None
    # A handle that is not a handle is not a version question either.
    assert resolve_declared_handle("not a handle", axis=AXIS_QUANT, where="t") is None

    with pytest.raises(LayoutDeclarationError) as caught:
        parse_lane_stamp(("nonesuch-family.diffusers@1", "plain.bf16@1"), where="t")
    assert "not in the vendored v2 corpus" in str(caught.value)
    assert not isinstance(caught.value, ContractVersionAheadError)


def test_a_byte_DIFFERENT_descriptor_under_the_same_name_REFUSES(
    tmp_path: Path,
) -> None:
    """Resolution is the ONE place this repo substitutes an identity the author
    did not write — so the identity it substitutes must be the RATIFIED one.

    A document whose bytes moved under a name that did not is exactly what must
    not resolve silently: the resolved handle reads identically in every log
    line while meaning something else. The two independently-written copies of
    the digest (the document's own field, and `vectors/digests.json`) are what
    make that provable without a Go canonicalizer.
    """
    fake = tmp_path / "v2"
    shutil.copytree(Path(tlc._SPEC_V2), fake)
    target = fake / "topologies" / "sdxl.diffusers.v2.json"
    doc = json.loads(target.read_text())
    # The NAME and VERSION are untouched. Only the descriptor moves — one
    # tensor's logical shape — which is precisely the substitution that must not
    # pass, and it moves the document's own digest away from the ledger row.
    doc["digest"] = "0" * 64
    target.write_text(json.dumps(doc))

    with _repointed(fake):
        # The exact spelling still resolves: an unchanged declaration is fenced
        # byte-for-byte by `[packages.tensorfs.files]` and does not pay this.
        assert resolve_declared_handle(
            "sdxl.diffusers@2", axis=AXIS_TOPOLOGY, where="t") == "sdxl.diffusers@2"

        with pytest.raises(ContractDescriptorUnprovenError) as caught:
            resolve_declared_handle(
                "sdxl.diffusers@1", axis=AXIS_TOPOLOGY, where="t")
        assert "cannot be proven" in str(caught.value)

        # And a sibling record with untouched bytes still resolves, so the
        # refusal is about THAT document, not a corpus-wide bail-out.
        assert resolve_declared_handle(
            "sd15.diffusers@1", axis=AXIS_TOPOLOGY, where="t") == "sd15.diffusers@2"


def test_the_RED_ARM_resolution_disabled_reproduces_the_fleet_wide_refusal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The measurement this whole change exists for, forced red on demand.

    With resolution reduced to the literal membership test it replaced, the
    fleet's own spelling refuses at class definition. An arm that cannot go red
    proves nothing, so this one disables the mechanism rather than asserting
    around it.
    """
    from gen_worker import lane
    from gen_worker.demand import GiB, const
    from gen_worker.models import SDXL
    from gen_worker.serving import Model, ModelDeclarationError

    def literal(handle: object, *, axis: str, where: str) -> str | None:
        text = str(handle or "").strip()
        return text if text in tlc._corpus_members(axis) else None

    monkeypatch.setattr(tlc, "resolve_declared_handle", literal)

    with pytest.raises(ModelDeclarationError, match="not in the vendored v2 corpus"):

        class Fleet(
            Model[SDXL],
            lanes={("sdxl.diffusers@1", "plain.bf16@1"): lane(
                request=const(GiB(1)))},
        ):
            def load(self, ctx: object) -> None: ...

    with pytest.raises(LayoutDeclarationError):
        tlc.rule_dtype(f"{SUPERSEDED_RULE}@1")


@contextmanager
def _repointed(root: Path) -> Iterator[None]:
    """Point the module's corpus readers at another tree, caches cleared.

    Every reader is `lru_cache`d over a module-level path, so swapping the path
    without clearing them would test the OLD corpus and pass for the wrong
    reason — the exact shape of mistake that had the tcg#90 acceptance run
    executing new code over the old corpus. Cleared on the way IN and on the way
    OUT, so no later test in the session inherits a poisoned cache either.
    """
    cached = (
        tlc.quant_rules, tlc.topologies, tlc.display_names,
        tlc._corpus_digests, tlc._ledger_digests,
    )
    original = tlc._SPEC_V2
    for fn in cached:
        fn.cache_clear()
    tlc._SPEC_V2 = root
    try:
        yield
    finally:
        tlc._SPEC_V2 = original
        for fn in cached:
            fn.cache_clear()


# ── THE BIND DOOR, worker-side ────────────────────────────────────────────────


def _decode_set(*rules: str) -> "DecodeSet":
    from gen_worker.discovery.decode_set import DecodeEntry, DecodeSet

    return DecodeSet(
        derivation="pgw#1665 arm",
        entries=tuple(
            DecodeEntry(rule=r, decoder=f"test:{r}", serves=("bf16-w16a16",),
                        composes_lora=False)
            for r in rules
        ),
        unregistered=(),
        excluded_modules=(),
    )


def test_a_bound_variant_at_2_is_decodable_by_an_image_declaring_1() -> None:
    """The door th#2301 opened hub-side, arriving on the worker.

    The hub's resweep is re-deriving checkpoints onto `cozy.fp8-rowwise@2` one
    row at a time (`wan22-t2v-a14b@serve-fp8` already reads `@2`), while every
    released image's decode set declares `@1` — this repo's own
    `models/w8a8.py` does. On a literal comparison each moved row becomes
    `decode_set_rule_undeclared` on a rented pod, at load time, for a tree the
    image reads perfectly well.
    """
    from gen_worker.discovery.decode_set import (
        RuleNotDecodableError,
        require_decodable,
    )

    image_declares_v1 = _decode_set(f"{SUPERSEDED_RULE}@1", "plain.bf16@1")
    require_decodable(f"{SUPERSEDED_RULE}@2", decode_set=image_declares_v1,
                     where="bind")
    # ...and symmetrically, an image that has re-vendored still reads a catalog
    # row the resweep has not reached.
    require_decodable(f"{SUPERSEDED_RULE}@1",
                     decode_set=_decode_set(f"{SUPERSEDED_RULE}@2"),
                     where="bind")

    # THE REFUSAL IS INTACT. A rule this image genuinely cannot decode still
    # refuses — resolution widened the comparison, it did not open the gate.
    with pytest.raises(RuleNotDecodableError):
        require_decodable("cozy.nvfp4-flat@1", decode_set=image_declares_v1,
                         where="bind")
    # An unresolvable spelling compares as a literal, exactly as before, in
    # both directions.
    with pytest.raises(RuleNotDecodableError):
        require_decodable("cozy.q4-k@1", decode_set=image_declares_v1,
                         where="bind")
    require_decodable("cozy.q4-k@1", decode_set=_decode_set("cozy.q4-k@1"),
                     where="bind")


def test_the_RED_ARM_a_literal_decode_set_comparison_shuts_the_bind_door(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With `declared_contract_key` reduced to the spelling it replaced, the
    resweep's own catalog row refuses against the fleet's own image."""
    from gen_worker.discovery import decode_set as ds_mod
    from gen_worker.discovery.decode_set import (
        RuleNotDecodableError,
        require_decodable,
    )

    monkeypatch.setattr(
        tlc, "declared_contract_key",
        lambda handle, *, axis: str(handle or "").strip())
    assert ds_mod is not None  # the import site is inside the function under test

    with pytest.raises(RuleNotDecodableError):
        require_decodable(f"{SUPERSEDED_RULE}@2",
                         decode_set=_decode_set(f"{SUPERSEDED_RULE}@1"),
                         where="bind")
