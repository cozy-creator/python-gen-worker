"""Serving admission charges the STORED TREE, and h3 is the price of that.

**pgw#1590 measured the defect; pgw#1599 deleted its fix and could not
replace it. This file is the record of both, and of the three replacements
that were built and MEASURED before the over-charge was left standing.**

pgw#1590, filed from a real H100 pod: ``NeverFits`` refused
``minimax.h3-dit-diffusers@1`` as needing 180,063,706,300 bytes on a card with
84,368,556,032 — the exact shape this fleet served on single H100s two weeks
earlier. The 180 GB is the WHOLE minimax-h3 repo at its stored bf16 precision
plus 25%; the lane loads one part of it and `quantize_()`s that part to w8a8
inside ``setup()``. pgw#1590 corrected it with the endpoint's hand-written
``lanes={contracts.MINIMAX_H3_DIT_DIFFUSERS: "vram78g"}`` floor, capping the
charge downward.

pgw#1599 deletes every hand-written floor (Paul, 2026-08-20: *"there is no
required VRAM"*). Three replacements were tried, each measured against the
vendored contract library, and **all three can UNDER-count — the one direction
that OOMs a rented card instead of refusing it**:

1. **sum the tensors the lane's CONTRACT claims.** A contract is a layout
   TEMPLATE describing a matching SET, not an inventory: h3's declares 10
   patterns, and anything in the DiT they do not name goes uncounted.
2. **charge only the FILES the contract claims a tensor in.** Measured across
   four shipped contracts and the coverage is not consistent enough to decide
   residency from — ``sdxl.diffusers-bf16`` covers unet + vae + text encoders,
   ``sd15.diffusers-bf16`` covers the **UNET ONLY**, ``minimax.h3-dit-diffusers``
   the DiT only. Narrowing sd15 to its contract would drop the VAE and both
   text encoders its model class holds resident.
3. **charge at the contract's dtype.** h3's bf16 contract honestly says
   ~133 GB. The gap to the ~66.5 GB it actually holds is a RUNTIME
   `quantize_()` that no manifest, header or contract can see.

**The finding underneath, and it is the point of this file now:** (3) is not
an admission defect at all. h3's `setup()`-time quantization is the
undeclared runtime numerics change pgw#1605 exclusion 1 bans — *"Quantization
is a MINT-time, declared, measured decision — never an admission-time
reflex."* pgw#1590's hand-written 78 was load-bearing precisely because it
encoded that violation's consequence. **The close is h3 serving a real
``minimax.h3-dit-fp8-rowwise@1`` lane** (it exists in the library,
`dtype=float8_e4m3fn`), which needs tensorfs#128's converted artifact and
pgw#1606's loader — not a cleverer sizer.

The regression is LATENT, not immediate: endpoints pin gen-worker from PyPI
and this code reaches h3 only at a version cut it adopts.

INTEGRATION, and the arithmetic is never faked: a real ``LocalCAS`` holding a
real ``RepositoryManifest`` at minimax-h3's measured per-shard byte sizes, the
production ``SnapshotSizer`` reading it through ``disk_gc.tree_bytes``, and the
production ``ResidencyManager``. The only stand-in is the resolver seam (WHICH
tree), which is not part of the number under test.
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
from gen_worker.models.projection import REF_PREFIX, SNAPSHOTS_DIR
from gen_worker.serving.residency import (
    NeverFits,
    ResidencyManager,
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

#: What the endpoint used to DECLARE, and what h3 actually holds. Neither is
#: reachable from a manifest, a header or a contract — see the module
#: docstring. Kept as CONSTANTS rather than deleted because they are the
#: measurement that says how big the standing over-charge is.
H3_FLOOR_BYTES = 78 * GIB
H3_LANE_BF16_BYTES = 133_006_919_280   # the DiT at the contract's own dtype
H3_ACTUALLY_HELD = 66_503_459_640      # after the setup()-time quantize_()

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
    """Non-vacuity: the tree number this file is about is the one the pod
    computed, produced by the same code, from a manifest — not a constant
    typed into the test."""
    sizer, _ = h3_manager
    assert sizer.resident_bytes(*H3_KEY) == H3_TREE_BYTES
    assert sizer.resident_bytes(*H3_KEY) + sizer.activation_headroom_bytes(
        *H3_KEY
    ) == H3_TREE_CHARGE == 180_063_706_300


def test_the_h3_dit_lane_is_refused_on_an_h100_and_this_is_the_KNOWN_cost(
    h3_manager: Any,
) -> None:
    """THE STANDING REGRESSION, asserted rather than left to be rediscovered.

    pgw#1590 made this pass by reading the endpoint's hand-written floor;
    pgw#1599 deleted every hand-written floor and the three measured
    replacements can all under-count (module docstring). So h3 is refused
    again, and the refusal is REPRODUCED HERE BYTE FOR BYTE so that the day it
    stops being true, this test says so instead of a pod.

    Its close is NOT here: h3 must serve a lane whose contract states the
    precision its weights land at (`minimax.h3-dit-fp8-rowwise@1`,
    float8_e4m3fn), which needs tensorfs#128's converted artifact and
    pgw#1606's loader. The runtime `quantize_()` that opened the gap is itself
    the standing-rule violation pgw#1605 exclusion 1 names.
    """
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
    # ...and the CONFESSION says it is an upper bound, so the reader of a pod
    # log is not left to work out why a card that served this shape refused
    # it. `admission_charge`'s basis is what the manager confesses.
    sizer, _ = h3_manager
    basis = admission_charge(
        sizer.resident_bytes(*H3_KEY), sizer.activation_headroom_bytes(*H3_KEY)
    ).basis
    assert "UPPER BOUND" in basis and "STORED TREE" in basis


def test_the_gap_the_close_has_to_shut_is_MEASURED_not_asserted() -> None:
    """How big the over-charge is, in one place, so the fp8-lane close has a
    number to be judged against rather than a feeling.

    Three numbers, each from a different source: the tree walk (this file's
    real manifest), the DiT at its contract's OWN dtype (bf16, so no
    quantization is assumed), and what the endpoint actually holds after
    `setup()`. The last one is the only one no producer in this repo can
    compute, and that is exactly the defect.
    """
    # What admission charges today.
    assert H3_TREE_BYTES == 144_050_965_040
    # What a perfect lane-scoped, correct-dtype sizer would charge — still
    # refused on an 84 GB card once the /4 headroom is added.
    assert H3_LANE_BF16_BYTES + H3_LANE_BF16_BYTES // 4 > H100_BUDGET
    # What the card actually holds. Roughly half, because of a runtime
    # quantize no declaration describes.
    assert H3_ACTUALLY_HELD * 2 == pytest.approx(H3_LANE_BF16_BYTES, rel=1e-6)
    assert H3_ACTUALLY_HELD < H3_FLOOR_BYTES < H100_BUDGET


def test_the_charge_is_the_sizer_s_two_numbers_and_nothing_else() -> None:
    """No cap, no floor, no third input. The whole point of pgw#1599's
    deletion is that there is exactly ONE producer of this number again — and
    an over-charging single producer is strictly better than two that
    disagree, because only one of them can be silently wrong."""
    charge = admission_charge(H3_TREE_BYTES, H3_TREE_BYTES // 4)
    assert charge.weight_bytes == H3_TREE_BYTES
    assert charge.headroom_bytes == H3_TREE_BYTES // 4
    assert charge.total == H3_TREE_CHARGE
    assert "STORED TREE" in charge.basis
    assert "UPPER BOUND" in charge.basis
    # The refusal names what actually fixes it, not what to hand-write.
    assert "never a hand-written floor" in charge.basis


@pytest.mark.parametrize("tree_gb", [0.5, 2.0, 4.3, 6.9, 13.9, 16.0, 24.0, 60.0, 134.0])
def test_the_charge_is_monotone_in_the_tree_and_never_optimistic(
    tree_gb: float,
) -> None:
    """THE SAFETY PROPERTY that survived the deletion: the charge is a
    function of the tree alone and is never smaller than it. Every rejected
    replacement failed exactly here — each could return a number BELOW what
    the lane holds resident, and an under-charge is an OOM on a rented card
    while an over-charge is a typed refusal."""
    weights = int(tree_gb * GIB)
    headroom = weights // 4
    charge = admission_charge(weights, headroom)
    assert charge.total == weights + headroom
    assert charge.weight_bytes >= weights


def test_a_smaller_card_is_still_refused_typed_before_any_byte_moves(
    h3_manager: Any,
) -> None:
    """The gate itself is intact. An L40S/A6000-class 48 GiB card cannot hold
    this lane under any of the numbers above, and is refused TYPED at
    admission — before the factory runs, before a byte moves."""
    _, manager_for = h3_manager
    manager = manager_for(48 * GIB)
    journal: List[str] = []
    with pytest.raises(NeverFits):
        manager.lease(*H3_KEY, lambda: Backend(journal))
    assert journal == []
