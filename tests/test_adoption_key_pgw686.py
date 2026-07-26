"""pgw#686: requested cell key and published cell key must resolve the base
lane identically, or a published cell is never requested by ANY worker and
every cold pod re-mints — the ie#546 16-checkpoint burst stampede (10 pods,
9 simultaneous mints, 0 adoptions, 1,608 hub `cell not attached` resolves).

The constants below are the LIVE burst evidence, byte-exact (hub
`cozyhub-e2e112-34f3e64fe2`, sdxl release `b266454e92a9000c4c0c13f4`,
gen-worker 0.67.1, L4/sm_89, 2026-07-26):

* the two keys every worker ADVERTISED (and the hub tried to resolve 974x
  each, `status=tag_not_found`) come from the speculative/probed base lanes
  ``""`` and ``"fp8-hooks"``;
* the ONE key both L4 workers PUBLISHED (`cell_store` row, obligation
  discharged) is the same identity on the ``"w8a8"`` base lane — the lane
  `w8a8_lora.branch_lane` sees on the denoiser (`_cozy_w8a8_mode`) but
  `loading.pipeline_weight_lane` cannot.

Every other axis is identical — proven by reproducing all three digests from
one axes dict varying ONLY the lane.
"""

from __future__ import annotations

from typing import Any, Dict, Iterator

import pytest

from gen_worker import cell_key as ck
from gen_worker import compile_cache as cc
from gen_worker.models import w8a8_lora

# --- the burst's recorded artifact metadata (cell_store checkpoint
# 9167d89b..., trimmed to the axes from_artifact_metadata consumes) ---------

_SHAPE_CONTRACT: Dict[str, Any] = {
    "v": 3,
    "shapes": [
        [640, 1536], [768, 1344], [832, 1216], [896, 1152], [1024, 1024],
        [1152, 896], [1216, 832], [1344, 768], [1536, 640],
    ],
    "dynamic": [],
    "targets": ["unet"],
    "guidance": [0.0, 5.0],
    "regional": False,
    "lora_bucket": 64,
    "text_lens": [77],
}

_IMAGE_DIGEST = (
    "sha256:3f02f1051c572597fda3626b0ad85d284a5c1f945847b354b449d96bcb499071"
)

_BURST_META: Dict[str, Any] = {
    "format": 2,
    "kind": "torch-inductor-cache",
    "family": "sdxl",
    "weight_lane": "w8a8-lora64",
    "lora_bucket": 64,
    "compile_mode": "whole",
    "sku": "l4",
    "sm": "sm_89",
    "cuda": "13.0",
    "torch": "2.13.0+cu130",
    "triton": "3.7.1",
    "gen_worker": "0.67.1",
    "image_digest": _IMAGE_DIGEST,
    "libs": {"diffusers": "0.39.0", "transformers": "5.13.1"},
    "shape_contract": _SHAPE_CONTRACT,
}

# What the hub saw, verbatim.
PUBLISHED = "ck2-30b872ea452a3447ac368d0108de302c087b0a3b4b244ebeac10f15f"
REQUESTED_PLAIN = (
    "ck2-1688dc35245507a65d1a0fd2087adaae35f5885d5a71f05144a80710")
REQUESTED_FP8_HOOKS = (
    "ck2-3156506b08a75f15202355242f7509abd409cfaabdd9467391ffe815")

_RT = {
    "sku": "l4", "sm": "sm_89", "cuda": "13.0", "cuda_driver": "13000",
    "torch": "2.13.0+cu130", "triton": "3.7.1",
    "image_digest": _IMAGE_DIGEST,
}


@pytest.fixture()
def burst_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin every probe to the burst pod's recorded axes."""
    monkeypatch.setattr(cc, "runtime_key", lambda: dict(_RT))
    monkeypatch.setattr(cc, "gen_worker_version", lambda: "0.67.1")
    monkeypatch.setattr(
        cc, "_lib_versions",
        lambda: {"diffusers": "0.39.0", "transformers": "5.13.1"})
    monkeypatch.setenv("WORKER_IMAGE_DIGEST", _IMAGE_DIGEST)


def _requested(weight_lane: str) -> str:
    return ck.compute(
        "sdxl", weight_lane, 64,
        contract=ck.contract_digest(_SHAPE_CONTRACT),
        regional=False,
    ).digest


class _Denoiser:
    """A denoiser on the w8a8 GEMM lane, as the fp8-w8a8 flavor loader
    leaves it (`_cozy_w8a8_mode`), with no pipeline-level lane stamp."""

    _cozy_w8a8_mode = "pertensor"
    _cozy_fp8_storage_applied: bool

    def named_modules(self) -> Iterator[Any]:  # branch-capable duck
        return iter(())


class _Pipe:
    # Dynamic lane markers real pipelines carry (declared for mypy).
    _cozy_weight_lane: str
    _cozy_lora_base_lane: str

    def __init__(self) -> None:
        self.unet = _Denoiser()


# --- the defect, reproduced from the live evidence -------------------------


def test_burst_divergence_reproduced_byte_exact(burst_runtime: None) -> None:
    """All three observed keys are ONE identity varying only the lane axis:
    the advertised lanes name keys nobody publishes; the published lane
    names a key nobody advertises. Adoption was structurally impossible."""
    assert ck.from_artifact_metadata(_BURST_META).digest == PUBLISHED
    assert _requested("") == REQUESTED_PLAIN
    assert _requested("fp8-hooks") == REQUESTED_FP8_HOOKS
    assert _requested("w8a8") == PUBLISHED
    assert _requested("w8a8-lora64") == PUBLISHED  # canonical-lane fold


def test_raw_pipeline_probe_is_blind_to_w8a8_mode(burst_runtime: None) -> None:
    """The mechanism: loading.pipeline_weight_lane cannot see the denoiser's
    w8a8 GEMM mode, so a key computed from it can never equal the key the
    mint stamps (stamp_lane's branch_lane fallback sees it)."""
    from gen_worker.models.loading import pipeline_weight_lane

    pipe = _Pipe()
    assert pipeline_weight_lane(pipe) == ""
    assert w8a8_lora.branch_lane(pipe.unet) == "w8a8"
    # The pre-fix advertised key IS the burst's never-published key.
    assert _requested(pipeline_weight_lane(pipe)) == REQUESTED_PLAIN


# --- the fix: one base-lane resolution for every cell-identity surface -----


def test_cell_base_lane_sees_w8a8_mode(burst_runtime: None) -> None:
    pipe = _Pipe()
    assert w8a8_lora.effective_base_lane(pipe) == "w8a8"
    assert cc.cell_base_lane(pipe) == "w8a8"
    # An identical worker now requests exactly the key the mint published.
    assert _requested(cc.cell_base_lane(pipe)) == PUBLISHED


def test_cell_base_lane_precedence() -> None:
    # Explicit pipeline stamp wins.
    pipe = _Pipe()
    pipe._cozy_weight_lane = "w8a8-lora64"
    assert cc.cell_base_lane(pipe) == "w8a8-lora64"
    # fp8-hooks marker (w8a16 storage lane) wins over the denoiser fallback.
    pipe2 = _Pipe()
    pipe2.unet._cozy_fp8_storage_applied = True
    assert cc.cell_base_lane(pipe2) == "fp8-hooks"
    # Plain pipeline stays plain.
    class _PlainDenoiser:
        def named_modules(self) -> Iterator[Any]:
            return iter(())

    class _PlainPipe:
        def __init__(self) -> None:
            self.unet = _PlainDenoiser()

    assert cc.cell_base_lane(_PlainPipe()) == ""


def test_stamp_lane_memoizes_the_same_base() -> None:
    """Parity by construction: the base stamp_lane memoizes at mint time is
    exactly what effective_base_lane resolved before the mint — so the
    requested key (advertise) and the stamped key (publish) share one lane."""
    pipe = _Pipe()
    expected = w8a8_lora.effective_base_lane(pipe)
    w8a8_lora.stamp_lane(pipe, {"unet": pipe.unet})
    assert pipe._cozy_lora_base_lane == expected == "w8a8"
    # bucket 0 on the stub: the stamp restores the branchless base lane.
    assert pipe._cozy_weight_lane == "w8a8"


def test_lookup_speculation_covers_the_w8a8_lane() -> None:
    """Pre-load pull-by-key speculation must include the w8a8 base lane, or
    a cold worker can never pull the very cell its own boot will mint."""
    from gen_worker import executor

    assert "w8a8" in executor._SPECULATIVE_CELL_BASE_LANES
    assert "" in executor._SPECULATIVE_CELL_BASE_LANES
    assert "fp8-hooks" in executor._SPECULATIVE_CELL_BASE_LANES
