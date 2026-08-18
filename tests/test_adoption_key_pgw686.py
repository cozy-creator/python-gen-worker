"""pgw#686: requested compiled graph key and published compiled graph key must resolve the base
lane identically, or a published compiled graph is never requested by ANY worker and
every cold pod re-mints — the ie#546 16-checkpoint burst stampede (10 pods,
9 simultaneous mints, 0 adoptions, 1,608 hub `compiled graph not attached` resolves).

The constants below are the LIVE burst evidence, byte-exact (hub
`cozyhub-e2e112-34f3e64fe2`, sdxl release `b266454e92a9000c4c0c13f4`,
gen-worker 0.67.1, L4/sm_89, 2026-07-26):

* the two keys every worker ADVERTISED (and the hub tried to resolve 974x
  each, `status=tag_not_found`) come from the speculative/probed base lanes
  ``""`` and ``"fp8-hooks"``;
* the ONE key both L4 workers PUBLISHED (`compiled_graph_store` row, obligation
  discharged) is the same identity on the ``"w8a8"`` base lane — the lane
  `w8a8_lora.branch_lane` sees on the denoiser (`_cozy_w8a8_mode`) but
  `loading.pipeline_weight_lane` cannot.

Every other axis is identical — proven by reproducing all three digests from
one axes dict varying ONLY the lane.
"""

from __future__ import annotations

from typing import Any, Dict, Iterator

import pytest

from gen_worker._vendor.torchcg import identity as ck
from gen_worker._vendor.torchcg import is_compiled_graph_key
from gen_worker import compile_cache as cc
from gen_worker import env_seal
from gen_worker.models import w8a8_lora

# --- the burst's recorded artifact metadata (compiled_graph_store checkpoint
# 9167d89b..., trimmed to the axes from_artifact_metadata consumes) ---------

_SHAPE_CONTRACT: Dict[str, Any] = {
    "v": 1,
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
    "declared_compile_contract": _SHAPE_CONTRACT,
    # Reconstruction: the recorded burst pre-dates env sealing and recipe
    # identity; fixed representative blocks keep the lane-only-divergence
    # relations provable under exact identity (burst_runtime pins effective_seal /
    # toolchain_digest / static_code_closure to THESE dicts, exactly as it
    # pins every other runtime probe).
    "env_seal": {
        "seal_v": 1,
        "posture": {"grad_enabled": "True"},
        "config": {"cudnn_benchmark": "False"},
        "inductor": "0" * 16,
    },
    "toolchain": {"torch": "1" * 16, "triton": "2" * 16},
    "code_closure": {"gen_worker/compile_cache.py": "3" * 16},
}

# What the hub saw, verbatim — ck2-era historical evidence. The pgw#691 sku
# collapse bumped the scheme to ck3 (every digest changed), so these bytes
# are RECORDS, no longer reproducible: the invariants below are asserted as
# key RELATIONS from one axes dict, and the old keys must be dead (a clean
# MISS, never a half-match).
CK2_PUBLISHED = "ck2-30b872ea452a3447ac368d0108de302c087b0a3b4b244ebeac10f15f"
CK2_REQUESTED_PLAIN = (
    "ck2-1688dc35245507a65d1a0fd2087adaae35f5885d5a71f05144a80710")
CK2_REQUESTED_FP8_HOOKS = (
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
    monkeypatch.setattr(
        env_seal, "effective_seal",
        lambda: dict(_BURST_META["env_seal"]))
    monkeypatch.setattr(
        cc, "toolchain_digest",
        lambda: tuple(sorted(_BURST_META["toolchain"].items())))
    monkeypatch.setattr(
        cc, "static_code_closure",
        lambda roots=(): tuple(sorted(_BURST_META["code_closure"].items())))
    monkeypatch.setenv("WORKER_IMAGE_DIGEST", _IMAGE_DIGEST)


class _BurstCfg:
    """Duck of the burst release's declaration (registry.CompileContract)."""

    shapes = tuple(tuple(r) for r in _SHAPE_CONTRACT["shapes"])
    targets = ("unet",)
    text_lens = (77,)
    text_len = None
    dynamic = ()
    regional = False
    lora_bucket = 64
    guidance_scales = (0.0, 5.0)


def _lane(weight_lane: str) -> str:
    """The compiled graph-identity LANE label a worker on ``weight_lane`` would ask for.

    pgw#1181: the adoption verdict this used to call —
    `compile_cache.local_compiled_graph_mismatch`, the fact-by-fact check a delivered
    `torch-inductor-cache` compiled graph was admitted by — is deleted with that format.
    The burst's mechanism does not need it: the divergence WAS the lane, and
    the lane label is the axis every compiled graph-identity surface is keyed on.
    A worker asking on the wrong lane asks for a different compiled graph — which is the
    same conclusion the verdict used to reach one step later, reached by
    construction instead of by comparison."""
    return cc.execution_lane_label(weight_lane, _BurstCfg.lora_bucket)


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


def test_burst_divergence_reproduced_execution_lane_only(burst_runtime: None) -> None:
    """All three observed identities were ONE identity varying only the
    lane: the advertised lanes named compiled graphs nobody publishes; the published
    lane named a compiled graph nobody advertised. Adoption was structurally
    impossible. Re-asserted post-pgw#1059 on the fact-by-fact verdict: the
    lane fact alone flips the verdict, and the refusal NAMES the lane."""
    published = _lane("w8a8")
    assert _lane("w8a8-lora64") == published  # canonical-lane fold
    for wrong in ("", "fp8-hooks"):
        assert _lane(wrong) != published, wrong
    # pgw#691/pgw#958: the recorded ck2 burst keys are dead — a
    # torch-inductor-cache artifact has no key identity at all any more
    # (pgw#1059), so an old key can only MISS.
    #
    # th#1897 puts the miss back where it can be decided by one repo: a `ck`
    # key IS key-shaped — the shared grammar refuses shape, never scheme, so a
    # newer fleet's key stays addressable by an older hub — and it names
    # nothing, because no current derivation restates its axes. The orphan
    # misses at the comparison, not at the parse.
    for old in (CK2_PUBLISHED, CK2_REQUESTED_PLAIN, CK2_REQUESTED_FP8_HOOKS):
        assert is_compiled_graph_key(old)
    with pytest.raises(ck.IdentityError, match="only an aot-inductor artifact has compiled-graph identity"):
        ck.from_artifact_metadata(_BURST_META)


# --- the fix: one base-lane resolution for every compiled graph-identity surface -----


def test_compiled_graph_base_execution_lane_sees_w8a8_mode(burst_runtime: None) -> None:
    pipe = _Pipe()
    assert w8a8_lora.effective_base_execution_lane(pipe) == "w8a8"
    assert cc.compiled_graph_base_execution_lane(pipe) == "w8a8"
    # An identical worker now asks on exactly the lane the mint published on
    # — one lane derivation on both sides, which is the whole of pgw#686.
    assert _lane(cc.compiled_graph_base_execution_lane(pipe)) == _lane("w8a8")


def test_compiled_graph_base_execution_lane_precedence() -> None:
    # Explicit pipeline stamp wins.
    pipe = _Pipe()
    pipe._cozy_weight_lane = "w8a8-lora64"
    assert cc.compiled_graph_base_execution_lane(pipe) == "w8a8-lora64"
    # fp8-hooks marker (w8a16 storage lane) wins over the denoiser fallback.
    pipe2 = _Pipe()
    pipe2.unet._cozy_fp8_storage_applied = True
    assert cc.compiled_graph_base_execution_lane(pipe2) == "fp8-hooks"
    # Plain pipeline stays plain.
    class _PlainDenoiser:
        def named_modules(self) -> Iterator[Any]:
            return iter(())

    class _PlainPipe:
        def __init__(self) -> None:
            self.unet = _PlainDenoiser()

    assert cc.compiled_graph_base_execution_lane(_PlainPipe()) == ""


def test_stamp_execution_lane_memoizes_the_same_base() -> None:
    """Parity by construction: the base stamp_lane memoizes at mint time is
    exactly what effective_base_lane resolved before the mint — so the
    requested key (advertise) and the stamped key (publish) share one lane."""
    pipe = _Pipe()
    expected = w8a8_lora.effective_base_execution_lane(pipe)
    w8a8_lora.stamp_execution_lane(pipe, {"unet": pipe.unet})
    assert pipe._cozy_lora_base_lane == expected == "w8a8"
    # bucket 0 on the stub: the stamp restores the branchless base lane.
    assert pipe._cozy_weight_lane == "w8a8"
