"""pgw#1042 — the delegated handback seam.

Attempt 28's 36/36 sdxl mint published nothing, behind two defects this file
pins:

1. The child's returned cell carried a key the parent could not relate to its
   own (`ck1-8f498f43…` opened, `ck1-886ffbcc…` returned). The two keys live
   in DISJOINT spaces by formula (pgw#1032/#1033) — but every axis they share
   must be byte-identical across the process boundary, and one measurably was
   not: `torch._inductor.aot_compile` mutates the global
   `aot_inductor.metadata` config entry as a side effect, so a child that has
   compiled seals a different `env_seal` than its own boot (and than every
   other process on the pod). The seal digest now excludes that entry, and
   `adopt_delegated_mint` refuses BY AXIS NAME when any shared axis diverges.

2. The artifact failed `update_constant_buffer_func_(... ) API call failed at
   model_container_runner.cpp:289` on the parent runtime. Reproduced locally
   byte-for-byte: the per-entry copying bind allocates the target's FULL
   constant set once per entry, so a 36-entry sdxl cell demanded ~N x 2.6 GB
   at arm and the failing cudaMalloc surfaced as that anonymous C++ error.
   Entries now bind BY REFERENCE against one marker-owned pool per target,
   and any residual AOTI failure is a typed `injection_failed` refusal
   carrying live device-memory context.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pytest

from gen_worker import aot_serve, cell_key, env_seal, fleet_cells


def _seal_dict() -> Dict[str, Any]:
    return {
        "seal_v": env_seal.SEAL_VERSION,
        "posture": {"grad": "off"},
        "config": {"float32_matmul_precision": "high"},
        "inductor": "aaaa000011112222",
        "loaded_libs": "bbbb333344445555",
    }


def _arm_key(seal: Dict[str, Any], toolchain: Dict[str, Any]) -> cell_key.CellKey:
    return cell_key.from_axes({
        "format": "2",
        "kind": "inductor",
        "family": "micro-diffusion",
        "lane": "w8a8-lora64",
        "sm": "sm_89",
        "contract": "1234567890abcdef",
        "env_seal": env_seal.seal_digest(seal),
        "toolchain": cell_key.facts_digest(toolchain),
    })


def _envelope(seal: Dict[str, Any], toolchain: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "cell_key": "ck1-" + "e" * 56,
        "kind": "aot-inductor",
        "format": "2",
        "family": "micro-diffusion",
        "weight_lane": "w8a8",
        "lora_bucket": 64,
        "sm": "sm_89",
        env_seal.SEAL_KEY: dict(seal),
        "toolchain": dict(toolchain),
    }


TOOLCHAIN = {"libtorch.so": "cafe0123"}


def test_shared_axes_agree_is_empty() -> None:
    seal = _seal_dict()
    assert fleet_cells.arm_axis_divergence(
        _arm_key(seal, TOOLCHAIN), _envelope(seal, TOOLCHAIN)) == ""


def test_env_seal_divergence_is_named() -> None:
    """The pod class: the child's recorded seal differs from the parent's."""
    parent_seal = _seal_dict()
    child_seal = dict(parent_seal, inductor="ffff999988887777")
    got = fleet_cells.arm_axis_divergence(
        _arm_key(parent_seal, TOOLCHAIN), _envelope(child_seal, TOOLCHAIN))
    assert got.startswith("env_seal: ")
    assert env_seal.seal_digest(child_seal) in got
    assert env_seal.seal_digest(parent_seal) in got


@pytest.mark.parametrize("field,value,axis", [
    ("sm", "sm_80", "sm"),
    ("family", "other-family", "family"),
    ("lora_bucket", 0, "lane"),
    ("format", "1", "format"),
])
def test_every_shared_axis_is_guarded(field: str, value: Any, axis: str) -> None:
    seal = _seal_dict()
    meta = _envelope(seal, TOOLCHAIN)
    meta[field] = value
    got = fleet_cells.arm_axis_divergence(_arm_key(seal, TOOLCHAIN), meta)
    assert got.startswith(f"{axis}: "), got


def test_adopt_refuses_typed_before_any_arm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A diverging child cell must fail `key_axis_divergence` at the seam —
    never reach the arm, whose C++ error was the pod's whole diagnostic."""
    seal = _seal_dict()
    child_seal = dict(seal, inductor="ffff999988887777")
    meta = _envelope(child_seal, TOOLCHAIN)

    from gen_worker import artifact_meta
    from gen_worker.models import provision

    monkeypatch.setattr(
        artifact_meta, "try_read_metadata", lambda _p: dict(meta))

    def _no_arm(*_a: Any, **_k: Any) -> Any:
        raise AssertionError("arm_aot must not run on a diverged cell")

    monkeypatch.setattr(provision, "arm_aot", _no_arm)

    mint_root = tmp_path / "mint-root"
    mint_root.mkdir()
    artifact = tmp_path / "cell.tar.gz"
    artifact.write_bytes(b"not-a-real-cell")
    arm = _arm_key(seal, TOOLCHAIN)
    pending = fleet_cells.PendingSelfMint(
        family="micro-diffusion", cell_key=arm.digest,
        ref=f"x#{arm.digest}", cfg=object(), target=tmp_path / "adopted.tar.gz",
        mint_root=mint_root, publisher=None, cache_dir=tmp_path / "cache",
        arm_key=arm)

    assert fleet_cells.adopt_delegated_mint(object(), pending, artifact) is None
    reason, detail = fleet_cells.adopt_refusal(pending)
    assert reason == "key_axis_divergence"
    assert "env_seal: " in detail
    assert meta["cell_key"] in detail


def test_inductor_digest_ignores_aot_compile_metadata() -> None:
    """torch's aot_compile writes machine facts into global config as a side
    effect; the seal must not move with it (the measured pgw#1042 axis)."""
    torch = pytest.importorskip("torch")
    import torch._inductor.config as inductor_config

    before = env_seal.inductor_config_digest()
    prior = dict(getattr(inductor_config.aot_inductor, "metadata", {}) or {})
    try:
        inductor_config.aot_inductor.metadata = {
            "AOTI_DEVICE_KEY": "cuda", "AOTI_CPU_ISA": "AVX2",
            "AOTI_COMPUTE_CAPABILITY": "89"}
        assert env_seal.inductor_config_digest() == before
    finally:
        inductor_config.aot_inductor.metadata = prior
    assert torch is not None


class _FakeTensor:
    def __init__(self, n: int = 4) -> None:
        self._n = n

    def numel(self) -> int:
        return self._n

    def element_size(self) -> int:
        return 2

    def detach(self) -> "_FakeTensor":
        return self

    def clone(self) -> "_FakeTensor":
        return _FakeTensor(self._n)


class _RefusingPackage:
    """A package whose C++ update fails the way the pod's did."""

    def get_constant_fqns(self) -> list:
        return ["lin.weight"]

    def load_constants(self, *_a: Any, **_k: Any) -> None:
        raise RuntimeError(
            "update_constant_buffer_func_( container_handle_, "
            "(AOTInductorConstantMapHandle)&const_map, use_inactive, "
            "check_full_update) API call failed at "
            "model_container_runner.cpp, line 289")


def test_cpp_bind_failure_is_typed_injection_failed() -> None:
    """The pod's anonymous `API call failed` becomes a NAMED refusal with the
    entry and live device-memory context — pgw#999's rule, one frame deeper."""
    spec = aot_serve.ConstantSpec(
        fqn="lin.weight", source=aot_serve.SOURCE_STATE_DICT,
        dtype="torch.bfloat16", shape=(2, 2))
    runner = aot_serve.ArtifactRunner(
        package=_RefusingPackage(),
        contract=None,  # bind never reads it
        constants=(spec,), entry="transformer/cfg=true")
    with pytest.raises(aot_serve.ConstantsUnboundError) as exc:
        runner.bind({"lin.weight": _FakeTensor()}, {})
    assert exc.value.reason == "injection_failed"
    assert "transformer/cfg=true" in str(exc.value)
    assert not runner.bound


def test_target_constant_pool_is_one_clone_per_fqn() -> None:
    """N entries share ONE owned copy — the N x constants VRAM demand that
    OOM'd the 36-entry sdxl arm must not be reconstructible."""
    torch = pytest.importorskip("torch")
    spec = aot_serve.ConstantSpec(
        fqn="w", source=aot_serve.SOURCE_STATE_DICT,
        dtype="torch.float32", shape=(4,))
    lit = aot_serve.ConstantSpec(
        fqn="tbl", source=aot_serve.SOURCE_LITERAL,
        dtype="torch.float32", shape=(4,))
    resident = torch.arange(4, dtype=torch.float32)
    pool = aot_serve.target_constant_pool(
        [(spec, lit), (spec,), (spec,)], {"w": resident})
    assert set(pool) == {"w"}  # literals ride their own payload
    assert pool["w"] is not resident  # owned, not aliased (pgw#812 D3)
    assert torch.equal(pool["w"], resident)
