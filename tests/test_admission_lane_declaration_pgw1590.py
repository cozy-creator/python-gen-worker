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
``lanes={h3_dit_lane: "vram78g"}`` floor, capping the
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
``minimax-h3.diffusers@1+cozy.fp8-rowwise@1`` lane** (both halves are ratified
in the v2 corpus and the rule declares `float8_e4m3fn`), which needs
tensorfs#128's converted artifact and pgw#1606's loader — not a cleverer sizer.

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
from gen_worker.models.projection import REF_PREFIX, SNAPSHOTS_DIR
from gen_worker.serving.residency import (
    NeverFits,
    ResidencyManager,
    admission_charge,
)
from gen_worker.worker import SnapshotSizer

GIB = 1024 ** 3

H3_DIT_SHARD_A = 66_726_413_676
H3_DIT_SHARD_B = 66_280_505_604
H3_TREE_BYTES = 144_050_965_040
H3_REST = H3_TREE_BYTES - H3_DIT_SHARD_A - H3_DIT_SHARD_B
H100_BUDGET = 84_368_556_032

H3_TREE_CHARGE = H3_TREE_BYTES + H3_TREE_BYTES // 4

H3_FLOOR_BYTES = 78 * GIB
H3_LANE_BF16_BYTES = 133_006_919_280
H3_ACTUALLY_HELD = 66_503_459_640

#: pgw#1621: the lane is the v2 stamp pair, rendered `<topology>+<quant>`.
#: The v1 Contract OBJECT this file used to import is deleted with the v1
#: corpus; the h3 DiT lane is `minimax-h3.diffusers@1` composed with
#: `plain.bf16@1`, which is the bf16 tree the 180 GB refusal was charged on.
H3_LANE = ("minimax-h3.diffusers@1", "plain.bf16@1")
H3_LANE_ID = "minimax-h3.diffusers@1+plain.bf16@1"
H3_KEY = ("tensorhub/minimax-h3@serve-narrowed", f"H3Model/{H3_LANE_ID}")


def _digest(name: str) -> CASRef:
    return CASRef.parse("sha256:" + hashlib.sha256(name.encode()).hexdigest())


def project(base: Path, name: str, files: Dict[str, int]) -> Path:
    """A REAL tensorfs projection: a real ``LocalCAS``, a real manifest stored and pinned under the real ref name, and the tree at the path ``resolve_projection`` looks for."""

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
    "transformer/diffusion_pytorch_model-00001-of-00002.safetensors": H3_DIT_SHARD_A,
    "transformer/diffusion_pytorch_model-00002-of-00002.safetensors": H3_DIT_SHARD_B,
    "text_encoder/model.safetensors": H3_REST - 2_000_000_000,
    "vae/diffusion_pytorch_model.safetensors": 1_500_000_000,
    "audio_vae/diffusion_pytorch_model.safetensors": 500_000_000,
}


class _Resolver:

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


def test_the_production_sizer_reads_the_measured_tree_from_a_real_manifest(
    h3_manager: Any,
) -> None:
    """Non-vacuity: the tree number this file is about is the one the pod computed, produced by the same code, from a manifest — not a constant typed into the test."""
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
    precision its weights land at (`minimax-h3.diffusers@1+cozy.fp8-rowwise@1`,
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
    assert journal == []
    sizer, _ = h3_manager
    basis = admission_charge(
        sizer.resident_bytes(*H3_KEY), sizer.activation_headroom_bytes(*H3_KEY)
    ).basis
    assert "UPPER BOUND" in basis and "STORED TREE" in basis


def test_the_gap_the_close_has_to_shut_is_MEASURED_not_asserted() -> None:
    """How big the over-charge is, in one place, so the fp8-lane close has a number to be judged against rather than a feeling."""
    assert H3_TREE_BYTES == 144_050_965_040
    assert H3_LANE_BF16_BYTES + H3_LANE_BF16_BYTES // 4 > H100_BUDGET
    assert H3_ACTUALLY_HELD * 2 == pytest.approx(H3_LANE_BF16_BYTES, rel=1e-6)
    assert H3_ACTUALLY_HELD < H3_FLOOR_BYTES < H100_BUDGET


def test_the_charge_is_the_sizer_s_two_numbers_and_nothing_else() -> None:
    """No cap, no floor, no third input."""
    charge = admission_charge(H3_TREE_BYTES, H3_TREE_BYTES // 4)
    assert charge.weight_bytes == H3_TREE_BYTES
    assert charge.headroom_bytes == H3_TREE_BYTES // 4
    assert charge.total == H3_TREE_CHARGE
    assert "STORED TREE" in charge.basis
    assert "UPPER BOUND" in charge.basis
    assert "never a hand-written floor" in charge.basis


@pytest.mark.parametrize("tree_gb", [0.5, 2.0, 4.3, 6.9, 13.9, 16.0, 24.0, 60.0, 134.0])
def test_the_charge_is_monotone_in_the_tree_and_never_optimistic(
    tree_gb: float,
) -> None:
    """THE SAFETY PROPERTY that survived the deletion: the charge is a function of the tree alone and is never smaller than it."""
    weights = int(tree_gb * GIB)
    headroom = weights // 4
    charge = admission_charge(weights, headroom)
    assert charge.total == weights + headroom
    assert charge.weight_bytes >= weights


def test_a_smaller_card_is_still_refused_typed_before_any_byte_moves(
    h3_manager: Any,
) -> None:
    """The gate itself is intact."""
    _, manager_for = h3_manager
    manager = manager_for(48 * GIB)
    journal: List[str] = []
    with pytest.raises(NeverFits):
        manager.lease(*H3_KEY, lambda: Backend(journal))
    assert journal == []
