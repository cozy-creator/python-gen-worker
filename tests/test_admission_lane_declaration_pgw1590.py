"""Serving admission charges the LANE'S DECLARATION, not the stored tree.

pgw#1590, filed from a real H100 pod: ``NeverFits`` refused
``minimax.h3-dit-diffusers@1`` as needing 180,063,706,300 bytes on a card with
84,368,556,032 — the exact shape this fleet served on single H100s two weeks
earlier. The 180 GB is the WHOLE minimax-h3 repo at its stored bf16 precision
plus 25%; the lane loads one part of it and `quantize_()`s that part to w8a8
inside ``setup()``, which no manifest can see and which the endpoint's own
``lanes={contracts.MINIMAX_H3_DIT_DIFFUSERS: "vram78g"}`` header states.

INTEGRATION, and the arithmetic is never faked: a real ``LocalCAS`` holding a
real ``RepositoryManifest`` at minimax-h3's measured per-shard byte sizes, the
production ``SnapshotSizer`` reading it through ``disk_gc.tree_bytes``, the
production ``ResidencyManager``, and the declaration read off a real
``Model`` subclass by the production ``placement.declared_vram_bytes``. The
only stand-in is the resolver seam (WHICH tree), which is not part of the
number under test.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, cast

import pytest

from gen_worker._vendor.tensorfs import (
    CASRef,
    FileEntry,
    LocalCAS,
    RepositoryManifest,
)
from gen_worker._vendor.tensorfs import contracts
from gen_worker.models import MiniMaxH3
from gen_worker.models.projection import REF_PREFIX, SNAPSHOTS_DIR
from gen_worker.serving.model import Model
from gen_worker.serving.placement import declared_vram_bytes
from gen_worker.serving.residency import (
    NeverFits,
    ResidencyManager,
    Tier,
    admission_charge,
)
from gen_worker.worker import SnapshotSizer

GIB = 1024 ** 3

# ── the measured article ─────────────────────────────────────────────────────
#
# Every number here was READ, not chosen: the two DiT component_fetch shards
# and the whole-repo total come from the pgw#1590 refusal on pod
# `ys9d2mu6opwb3a` (request 68eace0e-a9ba-4df6-ad1f-ada73f40c79a), and the
# budget is that H100's own `mem_get_info` headroom.
H3_DIT_SHARD_A = 66_726_413_676
H3_DIT_SHARD_B = 66_280_505_604
H3_TREE_BYTES = 144_050_965_040
H3_REST = H3_TREE_BYTES - H3_DIT_SHARD_A - H3_DIT_SHARD_B
H100_BUDGET = 84_368_556_032

#: What the refusal charged: the whole tree + the /4 activation estimate.
H3_TREE_CHARGE = H3_TREE_BYTES + H3_TREE_BYTES // 4

#: What the endpoint DECLARES (`serverless-endpoints/minimax-h3`, main.py:292).
H3_FLOOR_BYTES = 78 * GIB


class H3Model(Model[MiniMaxH3], lanes={contracts.MINIMAX_H3_DIT_DIFFUSERS: "vram78g"}):
    """The declaration under test, spelled exactly as the shipped endpoint
    spells it — a real contract object and a real floor, so this test reads
    the same class attribute production reads."""


H3_LANE = contracts.MINIMAX_H3_DIT_DIFFUSERS
H3_KEY = ("tensorhub/minimax-h3@serve-narrowed", "H3Model/minimax.h3-dit-diffusers@1")


# ── a real projected snapshot, without 144 GB of disk ────────────────────────


def _digest(name: str) -> CASRef:
    return CASRef.parse("sha256:" + hashlib.sha256(name.encode()).hexdigest())


def project(base: Path, name: str, files: Dict[str, int]) -> Path:
    """A REAL tensorfs projection: a real ``LocalCAS``, a real manifest stored
    and pinned under the real ref name, and the tree at the path
    ``resolve_projection`` looks for.

    The manifest is the article — ``disk_gc.tree_bytes`` sizes a projected
    tree from its manifest and never from the files, which is precisely why a
    stubbed 144 GB repo can be sized on a laptop and precisely how the
    production sizer got its 144,050,965,040.
    """

    cas = LocalCAS(base)
    manifest = RepositoryManifest(
        files=tuple(
            FileEntry(path=path, size_bytes=size, digest=_digest(path))
            for path, size in sorted(files.items())
        )
    )
    ref = cas.store_manifest(manifest)
    cas.compare_and_swap_ref(REF_PREFIX + name, ref, expected=None)
    tree = base / SNAPSHOTS_DIR / name
    tree.mkdir(parents=True, exist_ok=True)
    (tree / "config.json").write_text(json.dumps({"_class_name": "H3Pipeline"}))
    return tree


H3_TREE_LAYOUT = {
    # The DiT lane's own bytes: the two component_fetch shards.
    "transformer/diffusion_pytorch_model-00001-of-00002.safetensors": H3_DIT_SHARD_A,
    "transformer/diffusion_pytorch_model-00002-of-00002.safetensors": H3_DIT_SHARD_B,
    # Everything the DiT lane is NOT: vae, text encoder, audio vae, the
    # tokenizer/scheduler/processor configs. Charged to the card today.
    "text_encoder/model.safetensors": H3_REST - 2_000_000_000,
    "vae/diffusion_pytorch_model.safetensors": 1_500_000_000,
    "audio_vae/diffusion_pytorch_model.safetensors": 500_000_000,
}


class _Resolver:
    """The resolver SEAM — which tree, never how big. Shaped like
    ``HubBindingResolver`` for ``SnapshotSizer``'s one call."""

    def __init__(self, trees: Dict[str, Path]) -> None:
        self._trees = trees

    def tree_for(self, model_cls: type, checkpoint_ref: str) -> Path:
        return self._trees[checkpoint_ref]


class Backend:
    """Records every move; the residency engine never touches tensors."""

    def __init__(self, journal: List[str]) -> None:
        self.journal = journal

    def load(self) -> None:
        self.journal.append("load")

    def demote_to_host(self) -> None:
        self.journal.append("demote")

    def promote_to_device(self) -> None:
        self.journal.append("promote")

    def drop(self) -> None:
        self.journal.append("drop")


@pytest.fixture()
def h3_manager(tmp_path: Path) -> Any:
    tree = project(tmp_path / "store", "minimax-h3", H3_TREE_LAYOUT)
    sizer = SnapshotSizer(cast(Any, _Resolver({H3_KEY[0]: tree})))
    return sizer, lambda budget: ResidencyManager(budget, sizer)


# ── the sizer still tells the truth about the tree ───────────────────────────


def test_the_production_sizer_reads_the_measured_tree_from_a_real_manifest(
    h3_manager: Any,
) -> None:
    """Non-vacuity: the tree number this test is about is the one the pod
    computed, produced by the same code, from a manifest — not a constant
    typed into the test."""
    sizer, _ = h3_manager
    assert sizer.resident_bytes(*H3_KEY) == H3_TREE_BYTES
    assert sizer.resident_bytes(*H3_KEY) + sizer.activation_headroom_bytes(
        *H3_KEY
    ) == H3_TREE_CHARGE == 180_063_706_300


# ── the regression ───────────────────────────────────────────────────────────


def test_the_h3_dit_lane_is_refused_when_its_declaration_is_not_read(
    h3_manager: Any,
) -> None:
    """THE BUG, reproduced byte-for-byte on the real objects. Charging the
    stored tree — which is what a lane with no readable declaration gets —
    reproduces the pod's exact refusal string."""
    _, manager_for = h3_manager
    manager = manager_for(H100_BUDGET)
    journal: List[str] = []
    with pytest.raises(NeverFits) as excinfo:
        manager.lease(*H3_KEY, lambda: Backend(journal))
    message = str(excinfo.value)
    assert "needs 180063706300 bytes resident" in message
    assert "144050965040 weights + 36012741260 activation headroom" in message
    assert "the whole VRAM budget is 84368556032" in message
    assert journal == []  # refused at admission: no byte moved


def test_the_h3_dit_lane_is_admitted_on_a_single_h100_from_its_own_declaration(
    h3_manager: Any,
) -> None:
    """THE FIX, at the production numbers.

    ``vram78g`` -> 83,751,862,272 bytes against this H100's 84,368,556,032 of
    headroom: it fits, with 616,693,760 bytes (588 MiB) to spare. That margin
    is tight ON PURPOSE and it is the endpoint's own arithmetic, not this
    test's: the DiT is 133,006,919,280 bf16 bytes, w8a8 halves the weight
    tensors to ~66.5 GB, and ~17 GB of activations lands the serve at ~83 GB —
    which is why the header says 78 GiB and not 40.

    The fleet is the witness that this is a measurement and not a hope: pods
    yntiu7c3pd70bk, adux79i8nvy2j1, ac939lahn39c63, gqxy5elsartscu,
    ouu3szyqer3lss and lb27wbsay43z72 each hydrated and SERVED this shape on
    one NVIDIA H100 80GB HBM3 on 2026-08-14.
    """
    _, manager_for = h3_manager
    manager = manager_for(H100_BUDGET)
    journal: List[str] = []

    declared = declared_vram_bytes(H3Model, H3_LANE)
    assert declared == H3_FLOOR_BYTES == 83_751_862_272
    assert declared < H100_BUDGET
    assert H100_BUDGET - declared == 616_693_760

    with manager.lease(
        *H3_KEY, lambda: Backend(journal), declared_vram_bytes=declared
    ):
        assert manager.tier_of(*H3_KEY) is Tier.VRAM
    assert journal == ["load"]

    # The reservation IS the declaration — activations included, no 25% on top.
    reserved, _host = manager.reserved_bytes()
    assert reserved == H3_FLOOR_BYTES
    # pgw#1497's seam agrees with what admission actually reserved, rather
    # than re-asking the sizer and answering 144 GB.
    assert manager.weight_budget_bytes(*H3_KEY) == H3_FLOOR_BYTES


def test_a_declared_floor_is_not_blanket_optimism_a_smaller_card_still_refuses(
    h3_manager: Any,
) -> None:
    """The declaration lowers the charge; it does not delete the gate. An
    L40S/A6000-class 48 GiB card cannot hold this lane and is still refused
    typed, before any byte moves — an OOM on a rented card is worse than a
    refusal, which is why the cap is a DECLARED number and never a guess."""
    _, manager_for = h3_manager
    manager = manager_for(48 * GIB)
    journal: List[str] = []
    with pytest.raises(NeverFits) as excinfo:
        manager.lease(
            *H3_KEY,
            lambda: Backend(journal),
            declared_vram_bytes=declared_vram_bytes(H3Model, H3_LANE),
        )
    message = str(excinfo.value)
    assert "needs 83751862272 bytes resident" in message
    assert "LANE'S OWN DECLARATION" in message
    assert journal == []


# ── the properties that keep every other lane where it was ───────────────────


def test_no_declaration_leaves_the_conservative_charge_byte_identical() -> None:
    """The fallback is untouched. A lane that declares nothing is charged the
    whole stored tree plus 25%, exactly as before — and the refusal now says
    what to declare instead of leaving the reader to guess."""
    charge = admission_charge(H3_TREE_BYTES, H3_TREE_BYTES // 4, 0)
    assert charge.weight_bytes == H3_TREE_BYTES
    assert charge.headroom_bytes == H3_TREE_BYTES // 4
    assert charge.total == H3_TREE_CHARGE
    assert "DECLARES NO VRAM FLOOR" in charge.basis
    assert 'lanes={contract: "vramNNg"}' in charge.basis


@pytest.mark.parametrize(
    "floor_gb",
    # The fleet's real declared floors, one per shape class
    # (`serverless-endpoints/*/src/*/main.py`): sd15 vram6g, sdxl vram7g,
    # z-image vram32g, flux.1-dev vram38g, ltx-2 vram78g, minimax-h3 vram78g.
    [6.0, 7.0, 8.0, 12.0, 22.0, 24.0, 30.0, 32.0, 36.0, 38.0, 44.0, 78.0],
)
@pytest.mark.parametrize(
    "tree_gb", [0.5, 2.0, 4.3, 6.9, 13.9, 16.0, 24.0, 60.0, 134.0]
)
def test_a_declared_floor_can_only_ever_lower_the_charge(
    floor_gb: float, tree_gb: float
) -> None:
    """THE SAFETY PROPERTY, over every (fleet floor x plausible tree) pair:
    the charge is never larger than it was, so no lane admitted today can be
    refused tomorrow. This is what makes the change safe to land for the whole
    fleet on the evidence of one endpoint."""
    weights = int(tree_gb * GIB)
    headroom = weights // 4
    before = weights + headroom
    after = admission_charge(weights, headroom, int(floor_gb * GIB))
    assert after.total <= before
    assert after.total == min(before, int(floor_gb * GIB))


def test_a_floor_above_the_tree_leaves_the_charge_exactly_where_it_was() -> None:
    """z-image's shape and the reason sd15/sdxl/z-image do not move: a lane
    whose floor is generous relative to its tree keeps the tree number,
    weights-and-headroom split included. The cap is `min`, so it is inert
    until the tree crosses the declaration."""
    weights, floor = 16 * GIB, 32 * GIB
    charge = admission_charge(weights, weights // 4, floor)
    assert (charge.weight_bytes, charge.headroom_bytes) == (weights, weights // 4)
    assert "does not cap this charge" in charge.basis


# ── the whole serve path, class header to reservation ────────────────────────


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "serving_v2_endpoint"
FIXTURE_REF = "org/oversized@1"


class _LoopResolver:
    """One object for both seams the serve path resolves through: the deploy
    binding ServeLoop asks for, and the tree SnapshotSizer sizes."""

    def __init__(self, tree: Path) -> None:
        self.tree = tree

    def resolve(self, model_cls: type, checkpoint_ref: str) -> Any:
        from gen_worker.serving import DeployBinding

        return DeployBinding(
            checkpoint_ref=checkpoint_ref,
            checkpoint_dir=self.tree,
            model="sdxl",
            defaults={},
        )

    def default_pick(self, model_cls: type, slot_name: str) -> str:
        return ""

    def tree_for(self, model_cls: type, checkpoint_ref: str) -> Path:
        return self.tree


def test_the_declaration_reaches_admission_through_the_real_serve_path(
    tmp_path: Path,
) -> None:
    """END TO END on the production dispatcher, because the defect was never
    in the arithmetic — it was that the ONE caller holding the model class did
    not hand its declaration to the ONE object doing the arithmetic.

    A real endpoint (``load_endpoint``), its real ``lanes={contract:
    "vram12g"}`` header, a real ``ServeLoop``, a real ``ResidencyManager`` over
    the real ``SnapshotSizer``, and a real projected manifest carrying
    minimax-h3's measured 144,050,965,040 bytes. Under the tree charge this
    request is a ``JOB_STATUS_FATAL``; under the declaration it serves.
    """
    from gen_worker.serving import load_endpoint
    from gen_worker.serving.serve_loop import ServeLoop

    tree = project(tmp_path / "store", "oversized", H3_TREE_LAYOUT)
    resolver = _LoopResolver(tree)
    manager = ResidencyManager(H100_BUDGET, SnapshotSizer(cast(Any, resolver)))
    loop = ServeLoop(
        load_endpoint(FIXTURE_DIR),
        residency=manager,
        resolver=resolver,
        lane_contract="sdxl.diffusers-bf16@1",
        output_dir=tmp_path / "outputs",
    )

    outcome = loop.invoke(
        "generate",
        {"model": FIXTURE_REF, "input": {"prompt": "a lighthouse", "seed": 3}},
        request_id="pgw1590",
    )
    assert outcome.result.model == FIXTURE_REF

    lane_key = "SdxlModel/sdxl.diffusers-bf16@1"
    assert manager.tier_of(FIXTURE_REF, lane_key) is Tier.VRAM
    # Charged the header's `vram12g`, not the 180,063,706,300 the tree implies.
    reserved, _host = manager.reserved_bytes()
    assert reserved == 12 * GIB
    assert manager.weight_budget_bytes(FIXTURE_REF, lane_key) == 12 * GIB


def test_the_serve_path_still_refuses_when_the_declaration_cannot_help(
    tmp_path: Path,
) -> None:
    """The same real path, on a card under the endpoint's own floor: 8 GiB
    against a `vram12g` lane. Still ``NeverFits``, still before the author's
    class is constructed — the fix is a smaller HONEST charge, not a smaller
    gate."""
    from gen_worker.serving import load_endpoint
    from gen_worker.serving.serve_loop import ServeLoop

    tree = project(tmp_path / "store", "oversized", H3_TREE_LAYOUT)
    resolver = _LoopResolver(tree)
    manager = ResidencyManager(8 * GIB, SnapshotSizer(cast(Any, resolver)))
    loop = ServeLoop(
        load_endpoint(FIXTURE_DIR),
        residency=manager,
        resolver=resolver,
        lane_contract="sdxl.diffusers-bf16@1",
        output_dir=tmp_path / "outputs",
    )
    with pytest.raises(NeverFits) as excinfo:
        loop.invoke(
            "generate",
            {"model": FIXTURE_REF, "input": {"prompt": "x"}},
            request_id="pgw1590-small",
        )
    assert "refuse at admission" in str(excinfo.value)
    assert manager.tier_of(FIXTURE_REF, "SdxlModel/sdxl.diffusers-bf16@1") is Tier.ABSENT


def test_the_crossover_is_exact_and_has_no_gap() -> None:
    """Where the two arms meet, stated byte-exactly rather than in prose: at
    the floor the charge is the tree's (they are equal), one byte over it is
    the floor's."""
    floor = 6 * GIB
    for total, expect_declared in ((floor - 1, False), (floor, False), (floor + 1, True)):
        weights = total - total // 5  # total = weights + weights//4, near enough
        headroom = total - weights
        charge = admission_charge(weights, headroom, floor)
        assert charge.total == min(total, floor)
        assert ("LANE'S OWN DECLARATION" in charge.basis) is expect_declared
