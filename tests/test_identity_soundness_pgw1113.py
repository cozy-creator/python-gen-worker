"""pgw#1113 — the obligation names its subject, the memo names what it traced,
the advertisement names what armed, and the key names the placement.
THE GOVERNING PRINCIPLE, which is why these five changes are one change:

    The CELL KEY is the computation and must not OVER-split (the membership
    axiom, ``cell_key.py``).  The ARM TOKEN is a mint obligation and a cache
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

from gen_worker import boot_key, child_contract, fleet_cells
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
    assert child_contract.subject_digest(()) == ""
    facts = fleet_cells.arm_identity("f", "", 0, _Cfg()).facts_dict()
    assert facts["subject"] == ""


def test_the_stamp_accumulates_and_is_order_free() -> None:
    a, b = _Pipe(), _Pipe()
    fleet_cells.stamp_arm_subject(a, "pipeline", ["r1"])
    fleet_cells.stamp_arm_subject(a, "refiner", ["r2"])
    fleet_cells.stamp_arm_subject(b, "refiner", ["r2"])
    fleet_cells.stamp_arm_subject(b, "pipeline", ["r1"])
    assert _token(a) == _token(b)


def test_the_arm_facts_split_into_environment_and_subject() -> None:
    """The two halves are different axioms, so they are different tuples.

    ``ARM_ENVIRONMENT_FACTS`` is what a delegated child RECORDS on the cell it
    hands back, and is therefore comparable across the process boundary.
    ``ARM_SUBJECT_FACTS`` is not on the cell and must not be: the key is the
    computation, so one cell legally serves every checkpoint whose graph it
    is. Demanding the cell restate its minting checkpoint would refuse exactly
    the reuse the membership axiom exists to allow.
    """
    assert set(fleet_cells.ARM_FACTS) == (
        set(fleet_cells.ARM_ENVIRONMENT_FACTS)
        | set(fleet_cells.ARM_SUBJECT_FACTS))
    assert not (set(fleet_cells.ARM_ENVIRONMENT_FACTS)
                & set(fleet_cells.ARM_SUBJECT_FACTS))
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
    return {"pipeline": MintSlot(ref=ref, path=path)}


def test_a_rebinding_forces_a_memo_MISS() -> None:
    """THE finding this issue most wants read.

    ``closure_digest`` folded the SDK+endpoint code content and the
    declaration — and not the resolved slot refs the traces are actually run
    against. A memo HIT skips the traces and returns TCG's memoized class hash,
    which is the graph axis of the exact key ``boot_adopt`` resolves. Without
    the slot refs in the closure, a rebinding could ask for the previous
    checkpoint's graph without retracing.

    Folding the slots in is what makes that check capable of failing.
    """
    cfg = _spec()
    base = boot_key.closure_digest(
        "qwen-image", cfg, function="generate", slots=_slots(BASE))
    edit = boot_key.closure_digest(
        "qwen-image", cfg, function="generate", slots=_slots(EDIT))
    assert base != edit


def test_the_memo_row_of_one_checkpoint_does_not_answer_for_another(
    tmp_path: Path,
) -> None:
    """The same claim at the memo, which is where the wrong answer is served:
    one host with a volume, one family, unchanged code and declaration, and a
    redeploy that rebinds the slot."""
    cfg = _spec()
    was = boot_key.closure_digest("q", cfg, function="fn", slots=_slots(BASE))
    boot_key.write_memo(tmp_path, was, {"a": {"class_hash": "1" * 16}})
    assert boot_key.read_memo(tmp_path, was), "the memo must answer itself"

    now = boot_key.closure_digest("q", cfg, function="fn", slots=_slots(EDIT))
    assert boot_key.read_memo(tmp_path, now) == {}, (
        "a rebound slot must MISS — a hit here returns the previous "
        "checkpoint's witnesses and the witness floor can only agree with them")


def test_the_slot_PATH_is_not_part_of_the_memo_key() -> None:
    """Where the bytes were materialized is a location on this machine, never
    an identity. Folding it in would miss the memo on every fresh tmpdir,
    which is how a correctness fix turns into "the memo never hits"."""
    cfg = _spec()
    here = boot_key.closure_digest("q", cfg, slots=_slots(BASE, "/cas/a"))
    there = boot_key.closure_digest("q", cfg, slots=_slots(BASE, "/tmp/b"))
    assert here == there


def test_a_memo_file_from_the_previous_schema_is_discarded_whole(
    tmp_path: Path,
) -> None:
    """Memo invalidation is EXPECTED here (one re-trace per host after a
    rebinding) and it must be typed: a v3 row was filed under a
    checkpoint-blind digest, so it answers a different question and is never
    read as an answer to this one."""
    path = tmp_path / boot_key.MEMO_FILENAME
    path.write_text(json.dumps({
        "v": boot_key.MEMO_VERSION - 1,
        "closures": {"whatever": {"blocks": {"a": "{}"}}},
    }))
    assert boot_key.read_memo(tmp_path, "whatever") == {}
    assert boot_key.MEMO_VERSION >= 4
