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

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping

import pytest

from gen_worker import boot_key, cell_key, fleet_cells, local_cell_store
from gen_worker import mint_process as mp
from gen_worker.api.binding import Hub
from gen_worker.mint_process import CompileCellSpec

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
    assert cell_key.subject_digest(()) == ""
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


def test_a_subject_difference_is_not_a_handback_divergence() -> None:
    """A cell records no subject, so the handback seam must not ask for one —
    otherwise every delegated mint would refuse itself."""
    pipe = _Pipe()
    fleet_cells.stamp_arm_subject(pipe, "pipeline", ["q/img"], "sha256:aa")
    arm = fleet_cells.arm_identity(
        "qwen-image", "", 0, _Cfg(),
        subject=fleet_cells.pipeline_arm_subject(pipe))
    facts = arm.facts_dict()
    meta: Dict[str, Any] = {
        "family": facts["family"],
        "format": facts["format"],
        "weight_lane": "",
        "lora_bucket": 0,
        "sm": facts["sm"],
        cell_key.EXPORT_ENVELOPE_KEY: fleet_cells.declared_envelope_block(
            _Cfg()),
        "toolchain": dict(fleet_cells.cc.toolchain_digest()),
    }
    meta[fleet_cells.env_seal.SEAL_KEY] = fleet_cells.env_seal.effective_seal()
    assert fleet_cells.arm_axis_divergence(arm, meta) == ""


# --------------------------------------------------------------------------
# 2. the boot-key memo is no longer checkpoint-blind
# --------------------------------------------------------------------------


def _spec() -> CompileCellSpec:
    return CompileCellSpec(
        family="qwen-image", targets=("transformer",),
        shapes=((1024, 1024),), text_lens=(77,), guidance_scales=(4.0,))


def _slots(ref: Any, path: str = "/cas/a") -> Dict[str, mp.MintSlot]:
    return {"pipeline": mp.MintSlot(ref=ref, path=path)}


def test_a_rebinding_forces_a_memo_MISS() -> None:
    """THE finding this issue most wants read.

    ``closure_digest`` folded the SDK+endpoint code content and the
    declaration — and not the resolved slot refs the traces are actually run
    against. A memo HIT skips the traces and returns the MEMO's own witnesses
    (``graph_witnesses_of(memoized)``), which ``boot_adopt`` then verifies the
    pulled cell against. On that path pgw#1031's graph-witness floor — the
    fail-closed backstop for a wrong cell by key — was comparing a cell
    against a stale record of a DIFFERENT checkpoint's graph, so it could only
    agree. It was structurally unable to fire on the one path that most needs
    it.

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
    boot_key.write_memo(tmp_path, was, {
        "a": {"class_hash": "1" * 16, "graph_witness": "w" * 16}})
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
    current = memo_dir / (fleet_cells.ARM_SCHEME + "-" + "b" * 56 + ".json")
    for entry in (stale, current):
        entry.write_text(json.dumps({"cell_key": "ek1-" + "c" * 56}))

    dropped = local_cell_store.sweep_superseded_memos(
        fleet_cells.ARM_SCHEME, tmp_path)
    assert dropped == 1
    assert not stale.exists() and current.exists()
    assert fleet_cells.ARM_SCHEME != "arm1"


# --------------------------------------------------------------------------
# 4. the placement keying fact — and the NO-RE-KEY proof
# --------------------------------------------------------------------------


def _class_hash_before_pgw1113(
    entry: Mapping[str, Any], *, strict: bool, lora_bucket: int,
) -> str:
    """``aot_serve.class_hash`` VERBATIM as of the parent commit.

    Kept as a literal second implementation on purpose — the only way to prove
    a keying change re-keys nothing is to compute both keys, and the honest
    "before" is the old formula, not the new one with the new field left out.
    """
    facts = {
        "v": 3,
        "target": str(entry.get("target") or ""),
        "fork": [[str(n), v] for n, v in (entry.get("fork") or [])],
        "class_dims": [
            [str(n), int(v)] for n, v in (entry.get("class_dims") or [])],
        "range_digest": str(entry.get("range_digest") or ""),
        "graph": dict(entry.get("graph") or {}),
        "graph_witness": str(entry.get("graph_witness") or ""),
        "strict": bool(strict),
        "lora_bucket": int(lora_bucket or 0),
    }
    blob = json.dumps(facts, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def _entry(**extra: Any) -> Dict[str, Any]:
    block: Dict[str, Any] = {
        "target": "transformer",
        "fork": [["adapter", False]],
        "class_dims": [["height", 1024], ["width", 1024]],
        "range_digest": "d" * 32,
        "graph": {"specialization": {"strict": True, "lora_bucket": 0}},
        "graph_witness": "e" * 16,
    }
    block.update(extra)
    return block


@pytest.mark.parametrize("extra", [
    {},                                    # every cell published to date
    {"placement": ["cuda:0"]},             # a single-device cell that states it
    {"placement": []},                     # stated, and empty
])
def test_no_live_cell_re_keys(extra: Dict[str, Any]) -> None:
    """pgw#1113 claims NOTHING re-keys a live cell, as a deliberate property
    rather than luck: every new fact is omitted at the value every published
    cell holds. This is that claim, checked rather than asserted.

    A single-device placement is TRIVIAL, so the canonical form is
    byte-identical to the form with no placement at all — the ``excluded`` /
    ``param`` / ``overlay`` precedent, and the reason ``v`` does not move.
    """
    from gen_worker import aot_serve

    entry = _entry(**extra)
    assert aot_serve.class_hash(entry, strict=True, lora_bucket=0) == (
        _class_hash_before_pgw1113(entry, strict=True, lora_bucket=0))


def test_a_multi_device_placement_keys_APART() -> None:
    """pgw#819: a cell minted on a ``gpu_count=2, parallel="internal"`` pod —
    where the pipeline's own device map split the modules across
    ``cuda:0``/``cuda:1`` and inductor baked that placement into the graph —
    published under a key byte-identical to the single-GPU one, in BOTH
    directions, and the hub deduped them."""
    from gen_worker import aot_serve

    narrow = aot_serve.class_hash(_entry(), strict=True, lora_bucket=0)
    wide = aot_serve.class_hash(
        _entry(placement=["cuda:0", "cuda:1"]), strict=True, lora_bucket=0)
    assert narrow != wide
    # …and the order it was observed in is not information.
    assert wide == aot_serve.class_hash(
        _entry(placement=["cuda:1", "cuda:0"]), strict=True, lora_bucket=0)


def test_the_graph_hash_still_scrubs_the_device_index() -> None:
    """The placement rides its OWN fact precisely so the canonical graph form
    does not have to change. Un-scrubbing the index there would re-key every
    published cell to record a fact all of them state trivially — and it is
    scrubbed by deliberate design (*"placement is the sm axis, not graph
    identity"*)."""
    import inspect

    from gen_worker import graph_hash

    body = inspect.getsource(graph_hash._render_scalar)
    assert 'text.split(":", 1)[0]' in body


def _fx_module(torch: Any, first: str, second: str) -> Any:
    """A two-node graph whose nodes name two devices. No allocation, no CUDA
    context, no card — ``torch.device("cuda:1")`` is just a value."""
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    a = graph.call_function(
        torch.zeros, (2,), {"device": torch.device(first)})
    b = graph.call_function(
        torch.zeros, (2,), {"device": torch.device(second)})
    graph.output((x, a, b))
    return torch.fx.GraphModule(torch.nn.Module(), graph)


def test_device_placement_reads_exactly_what_the_graph_hash_scrubs() -> None:
    """pgw#819, demonstrated on a CPU in milliseconds.

    The two graphs differ ONLY in the device index of one node — a 1-card
    program and a 2-card one. Their canonical graph forms are byte-identical
    (the index is scrubbed by design), which is why no key axis could tell
    them apart and both directions of the adoption were silent. The placement
    observer sees the difference the canonical form deliberately does not.
    """
    torch = pytest.importorskip("torch")

    from gen_worker import graph_hash

    narrow = _fx_module(torch, "cuda:0", "cuda:0")
    wide = _fx_module(torch, "cuda:0", "cuda:1")

    assert graph_hash.graph_hash(narrow) == graph_hash.graph_hash(wide)
    assert graph_hash.device_placement(narrow) == ("cuda:0",)
    assert graph_hash.device_placement(wide) == ("cuda:0", "cuda:1")
