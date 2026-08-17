"""pgw#1113 — the obligation names its subject, the memo names what it traced,
the advertisement names what armed, and the key names the placement.
THE GOVERNING PRINCIPLE, which is why these five changes are one change:

    The CELL KEY is the computation and must not OVER-split (the membership
    axiom, ``tcg.identity``).  The ARM TOKEN is a mint obligation and a cache
    lookup and must not UNDER-split.  Over-splitting an obligation costs one
    re-mint; under-splitting one binds a pipeline to a cell nobody proved is
    its computation.

The tree applied the first axiom to both, so every key downstream of the cell
key named a (family, lane, runtime) triple and nothing about WHAT was being
compiled. Each test below is red on the parent commit.

Nothing here needs a card, a pod or a mint: every claim in this issue is about
what a digest is a function of, and that is answerable on a CPU in
milliseconds.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest

from harness.slot_facts import TEST_FACTS as _TEST_FACTS

from gen_worker import boot_key, fleet_cells, graph_facts, keyset, local_cell_store
from gen_worker.keyset import document as keyset_doc, store as keyset_store


def _row(class_hash: str) -> Any:
    """One closure row, as the mint lane emits it (pgw#1327)."""
    return keyset_doc.closure_row(
        family="q", function="fn", tcg_version=keyset.tcg_version(),
        classes={"a": keyset_doc.GraphClassRow(
            graph_class="a", class_hash=class_hash,
            ingress_digest="9" * 32, target="unet")})
from gen_worker.api.binding import Hub
from gen_worker.child_contract import CompileSpec, MintSlot

# --------------------------------------------------------------------------
# fixtures — two checkpoints of one family, and one declaration over them
# --------------------------------------------------------------------------

BASE = Hub("qwen/qwen-image")
EDIT = Hub("qwen/qwen-image-edit-2511")


class _Cfg:
    """A ``registry.CompileCell`` duck — the declaration both slots share."""

    family = "qwen-image"
    targets = ("transformer",)
    shapes = ((1024, 1024),)
    text_lens = (77,)
    guidance_scales = (4.0,)
    dynamic = ()
    regional = False
    lora_bucket = 0


class _Pipe:
    pass


@pytest.fixture(autouse=True)
def _stable_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    """One fixed runtime, so every token difference below is a SUBJECT
    difference and cannot be an environment one."""
    monkeypatch.setattr(
        fleet_cells.cc, "runtime_key",
        lambda: {"sm": "sm_89", "sku": "l4", "torch": "t", "triton": "",
                 "cuda": "", "image_digest": ""})
    monkeypatch.setattr(
        fleet_cells.cc, "toolchain_digest", lambda: (("torch", "x" * 16),))
    monkeypatch.setattr(
        fleet_cells.env_seal, "effective_seal", lambda: {"v": 4})


def _token(pipe: Any, cfg: Any = None) -> str:
    return fleet_cells.arm_identity(
        "qwen-image", "", 0, cfg or _Cfg(),
        subject=fleet_cells.pipeline_arm_subject(pipe)).token


# --------------------------------------------------------------------------
# 1. the arm token gains the SUBJECT
# --------------------------------------------------------------------------


def test_two_slots_bound_to_two_checkpoints_owe_two_mints() -> None:
    """The qwen edit shape. Two slots of ONE class, one ``Compile``, two
    checkpoints — and ``zero_cond_t`` (present on Edit-2511, absent on
    Qwen-Image) makes the two graphs structurally different forever.

    Before pgw#1113 both computed ONE token: one pending, one child, one
    local-store memo row, and the first slot to arm handed its cell to the
    other — "correct by backstop, not by design", the backstop being
    ``_bind_compile_guard`` returning False on an unarmed pipe.
    """
    t2i, edit = _Pipe(), _Pipe()
    fleet_cells.stamp_arm_subject(t2i, "pipeline", ["qwen/qwen-image"])
    fleet_cells.stamp_arm_subject(edit, "edit", ["qwen/qwen-image-edit-2511"])
    assert _token(t2i) != _token(edit)


def test_two_endpoint_classes_sharing_one_compile_owe_two_mints() -> None:
    """The flux.2-klein-4b shape: ``Generate`` and ``GenerateTurbo``, each
    with its own root ``pipeline`` slot bound to a DIFFERENT checkpoint at
    deploy, both passing the same ``_KLEIN_COMPILE``.

    Benign today only because the two checkpoints happen to share an
    architecture — which nothing checks. Cross-checkpoint sharing must be
    EARNED by the graph, never ASSUMED by the obligation token.
    """
    base, turbo = _Pipe(), _Pipe()
    fleet_cells.stamp_arm_subject(base, "pipeline", ["flux2/klein-4b"])
    fleet_cells.stamp_arm_subject(turbo, "pipeline", ["flux2/klein-4b-turbo"])
    assert _token(base) != _token(turbo)


def test_the_same_slot_at_the_same_checkpoint_still_shares_one_mint() -> None:
    """The other half, and the reason the subject is not simply "the pipe":
    two pipelines that ARE the same thing to compile must still buy one
    compile. Over-splitting is cheap, not free."""
    one, two = _Pipe(), _Pipe()
    for pipe in (one, two):
        fleet_cells.stamp_arm_subject(
            pipe, "pipeline", ["qwen/qwen-image"], "sha256:aa")
    assert _token(one) == _token(two)


def test_a_component_override_is_part_of_the_subject() -> None:
    """pgw#617 rebinds a single component. The composition it produces is a
    different set of bytes and can be a different graph, so it is a different
    obligation."""
    plain, overridden = _Pipe(), _Pipe()
    fleet_cells.stamp_arm_subject(plain, "pipeline", ["qwen/qwen-image"])
    fleet_cells.stamp_arm_subject(
        overridden, "pipeline", ["qwen/qwen-image", "qwen/other-vae"])
    assert _token(plain) != _token(overridden)


def test_a_snapshot_digest_move_is_a_different_obligation() -> None:
    old, new = _Pipe(), _Pipe()
    fleet_cells.stamp_arm_subject(old, "pipeline", ["q/img"], "sha256:aa")
    fleet_cells.stamp_arm_subject(new, "pipeline", ["q/img"], "sha256:bb")
    assert _token(old) != _token(new)


@pytest.mark.parametrize("field,value", [
    ("targets", ("transformer", "vae.decode")),
    ("dynamic", (type("D", (), {"dim": 1, "min": 1, "max": 4})(),)),
    ("regional", True),
])
def test_the_token_states_the_declaration_facts_it_could_not_see(
    field: str, value: Any,
) -> None:
    """``declared_compile_facts`` has all four (targets/dynamic/regional plus
    the shapes the envelope already carried) and the token saw none of the
    three. Two declarations that compile different programs must not share one
    obligation."""
    pipe = _Pipe()
    fleet_cells.stamp_arm_subject(pipe, "pipeline", ["q/img"])
    other = type("_Other", (_Cfg,), {field: value})
    assert _token(pipe) != _token(pipe, other())


def test_an_unstamped_pipeline_states_no_subject_and_says_so() -> None:
    """An endpoint that builds its own pipeline out of a path-valued slot and
    calls ``arm_compile()`` — nothing in this process saw a resolution for it.
    ``""`` is the honest answer and it UNDER-splits, which is why the executor
    stamps every subject it can (including, for that path, all of them)."""
    assert fleet_cells.pipeline_arm_subject(_Pipe()) == ()
    assert graph_facts.subject_digest(()) == ""
    facts = fleet_cells.arm_identity("f", "", 0, _Cfg()).facts_dict()
    assert facts["subject"] == ""


def test_the_stamp_accumulates_and_is_order_free() -> None:
    a, b = _Pipe(), _Pipe()
    fleet_cells.stamp_arm_subject(a, "pipeline", ["r1"])
    fleet_cells.stamp_arm_subject(a, "refiner", ["r2"])
    fleet_cells.stamp_arm_subject(b, "refiner", ["r2"])
    fleet_cells.stamp_arm_subject(b, "pipeline", ["r1"])
    assert _token(a) == _token(b)


def test_the_arm_facts_split_into_environment_and_obligation() -> None:
    """The two halves are different axioms, so they are different tuples.

    ``ARM_ENVIRONMENT_FACTS`` is what a delegated child RECORDS on the cell it
    hands back, and is therefore comparable across the process boundary.
    ``ARM_OBLIGATION_FACTS`` is not on the cell and must not be. For the
    subject half (pgw#1113) that is an axiom: the key is the computation, so
    one cell legally serves every checkpoint whose graph it is, and demanding
    the cell restate its minting checkpoint would refuse exactly the reuse the
    membership axiom exists to allow. For ``family``/``lane``/``env_seal``
    (pgw#1340) it is a fact about TCG's closed artifact vocabulary, which has
    no field for any of them — and comparing them anyway is what made every
    `sd15` self-mint refuse after a 25-minute compile.
    """
    assert set(fleet_cells.ARM_FACTS) == (
        set(fleet_cells.ARM_ENVIRONMENT_FACTS)
        | set(fleet_cells.ARM_OBLIGATION_FACTS))
    assert not (set(fleet_cells.ARM_ENVIRONMENT_FACTS)
                & set(fleet_cells.ARM_OBLIGATION_FACTS))
    assert set(fleet_cells.ARM_SUBJECT_FACTS) <= set(
        fleet_cells.ARM_OBLIGATION_FACTS)
    # The environment half is exactly what a cell can state — the invariant
    # pgw#1340's preflight refuses a mint on, asserted here on the constants.
    assert fleet_cells.unstateable_arm_axes() == ()
    identity = fleet_cells.arm_identity("f", "", 0, _Cfg())
    assert set(identity.facts_dict()) == set(fleet_cells.ARM_FACTS)
    assert "graph" not in identity.facts_dict()


# --------------------------------------------------------------------------
# 2. the boot-key memo is no longer checkpoint-blind
# --------------------------------------------------------------------------


def _spec() -> CompileSpec:
    return CompileSpec(
        family="qwen-image", targets=("transformer",),
        shapes=((1024, 1024),), text_lens=(77,), guidance_scales=(4.0,))


def _slots(ref: Any, path: str = "/cas/a") -> Dict[str, MintSlot]:
    return {"pipeline": MintSlot(ref=ref, path=path, facts=_TEST_FACTS)}


def test_a_rebinding_forces_a_memo_MISS() -> None:
    """THE finding this issue most wants read.

    ``closure_digest`` folded the SDK+endpoint code content and the
    declaration — and not the resolved slot refs the traces are actually run
    against. A memo HIT skips those traces, so folding the slots is what keeps
    a different checkpoint from reusing stale TCG class hashes.
    """
    cfg = _spec()
    base = keyset.closure_digest(
        "qwen-image", cfg, function="generate", slots=_slots(BASE))
    edit = keyset.closure_digest(
        "qwen-image", cfg, function="generate", slots=_slots(EDIT))
    assert base != edit


def test_the_memo_row_of_one_checkpoint_does_not_answer_for_another(
    tmp_path: Path,
) -> None:
    """The same claim at the memo, which is where the wrong answer is served:
    one host with a volume, one family, unchanged code and declaration, and a
    redeploy that rebinds the slot."""
    cfg = _spec()
    was = keyset.closure_digest("q", cfg, function="fn", slots=_slots(BASE))
    keyset_store.write_closure(tmp_path, was, _row("1" * 16))
    assert keyset_store.class_hashes(was, cache_dir=tmp_path), (
        "the cache must answer itself")

    now = keyset.closure_digest("q", cfg, function="fn", slots=_slots(EDIT))
    assert keyset_store.class_hashes(now, cache_dir=tmp_path) == {}, (
        "a rebound slot must MISS — a hit here returns the previous "
        "checkpoint's witnesses and the witness floor can only agree with them")


def test_the_slot_PATH_is_not_part_of_the_memo_key() -> None:
    """Where the bytes were materialized is a location on this machine, never
    an identity. Folding it in would miss the memo on every fresh tmpdir,
    which is how a correctness fix turns into "the memo never hits"."""
    cfg = _spec()
    here = keyset.closure_digest("q", cfg, slots=_slots(BASE, "/cas/a"))
    there = keyset.closure_digest("q", cfg, slots=_slots(BASE, "/tmp/b"))
    assert here == there


def test_a_memo_file_from_the_previous_schema_is_discarded_whole(
    tmp_path: Path,
) -> None:
    """Memo invalidation is EXPECTED here (one re-trace per host after a
    rebinding) and it must be typed: a v3 row was filed under a
    checkpoint-blind digest, so it answers a different question and is never
    read as an answer to this one."""
    digest = keyset.parse_closure_digest("ab" * 16)
    path = tmp_path / keyset.KEYSET_FILENAME
    path.write_text(json.dumps({
        "schema": keyset.KEYSET_SCHEMA,
        "version": keyset.KEYSET_VERSION - 1,
        "closures": {str(digest): {"blocks": {"a": "{}"}}},
    }))
    assert keyset_store.class_hashes(digest, cache_dir=tmp_path) == {}
    # pgw#1327: the version rides the CLOSURE DIGEST input as well as the file
    # header, so a stale row is unaddressable AND its file is discarded whole.
    assert keyset.CLOSURE_VERSION >= 6


# --------------------------------------------------------------------------
# 3. the local-store memo: superseded arm tokens are swept, not left to rot
# --------------------------------------------------------------------------


def test_the_arm_token_scheme_is_its_fact_set(tmp_path: Path) -> None:
    """The scheme digit IS the schema. ``arm1`` could not state a subject, so
    an ``arm1-`` memo row is an answer to a question no reader asks — and the
    cost of that, one re-mint per family per machine, is spent explicitly and
    counted rather than discovered later as a store of unreadable files."""
    memo_dir = local_cell_store.cells_root(tmp_path) / local_cell_store.MEMO_DIRNAME
    memo_dir.mkdir(parents=True)
    stale = memo_dir / ("arm1-" + "a" * 56 + ".json")
    current = memo_dir / (fleet_cells.ARM_SCHEME + "-" + "b" * fleet_cells.ARM_DIGEST_HEX + ".json")
    for entry in (stale, current):
        entry.write_text(json.dumps({"compiled_graph_key": "cg-key-v1-" + "c" * 56}))

    dropped = local_cell_store.sweep_superseded_memos(
        fleet_cells.ARM_SCHEME, tmp_path)
    assert dropped == 1
    assert not stale.exists() and current.exists()
    assert fleet_cells.ARM_SCHEME != "arm1"
