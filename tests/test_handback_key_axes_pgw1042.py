"""pgw#1042 — the delegated handback seam.

A full 36/36 sdxl mint publishing nothing, behind two defects this file pins:

1. The child's returned cell carried a key the parent could not relate to its
   own. The obligation identity is not a key at all (`arm1-…`,
   `fleet_cells.ArmIdentity`) — but every pre-trace FACT the two sides share
   must be byte-identical across the process boundary, and one was not:
   `torch._inductor.aot_compile` mutates the global `aot_inductor.metadata`
   config entry as a side effect, so a child that has compiled seals a
   different `env_seal` than its own boot (and than every other process on
   the pod). The seal digest now excludes that entry, and
   `adopt_delegated_mint` refuses BY FACT NAME when any shared fact diverges.

2. The artifact failed `update_constant_buffer_func_(... ) API call failed at
   model_container_runner.cpp:289` on the parent runtime: the per-entry copying
   bind allocates the target's FULL constant set once per entry, so a 36-entry
   sdxl cell demanded ~N x 2.6 GB
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


_DECLARED_ENVELOPE = {
    "shapes": [[64, 64]], "text_lens": [7], "guidance": [1.0]}


def _arm_key(seal: Dict[str, Any], toolchain: Dict[str, Any]) -> fleet_cells.ArmIdentity:
    return fleet_cells.ArmIdentity(facts=tuple(sorted({
        "family": "micro-diffusion",
        "format": str(aot_serve.ARTIFACT_FORMAT),
        "lane": "w8a8-lora64",
        "sm": "sm_89",
        "envelope": cell_key.envelope_digest(_DECLARED_ENVELOPE),
        "env_seal": env_seal.seal_digest(seal),
        "toolchain": cell_key.facts_digest(toolchain),
    }.items())))


def _envelope(seal: Dict[str, Any], toolchain: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "cell_key": "cg-key-v1-" + "e" * 56,
        "kind": "aot-inductor",
        "format": str(aot_serve.ARTIFACT_FORMAT),
        "family": "micro-diffusion",
        "weight_lane": "w8a8",
        "lora_bucket": 64,
        "sm": "sm_89",
        cell_key.EXPORT_ENVELOPE_KEY: dict(_DECLARED_ENVELOPE),
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


def test_the_envelope_is_deliberately_NOT_compared_at_this_seam() -> None:
    """INVERTED by pgw#1176, and stated rather than merely dropped (pgw#939:
    absence is a verdict, never a skipped check).

    This used to be a sixth `test_every_shared_axis_is_guarded` case asserting
    a diverging `EXPORT_ENVELOPE_KEY` is refused `envelope: ...`. `envelope`
    left `ARM_ENVIRONMENT_FACTS` with the key: a per-entry artifact records no
    declared envelope — that is a manifest fact about the whole declaration,
    not about one graph class — so comparing it here would test a value the
    child can no longer state, and refuse every handback by construction.

    Left as a silent parametrize deletion this would read as an oversight; as
    a row it is the ruling, and it goes red if anyone puts the axis back
    without revisiting the reason."""
    seal = _seal_dict()
    assert "envelope" not in fleet_cells.ARM_ENVIRONMENT_FACTS

    meta = _envelope(seal, TOOLCHAIN)
    meta[cell_key.EXPORT_ENVELOPE_KEY] = {
        "shapes": [[128, 128]], "text_lens": [7], "guidance": [1.0]}
    assert fleet_cells.arm_axis_divergence(
        _arm_key(seal, TOOLCHAIN), meta) == ""


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

    # The adopt reads through `read_metadata` now — an envelope it
    # CANNOT read is its own refusal (`cell_envelope_unreadable`) rather than
    # a `None` that flows on into a gate it silently disables. This test is
    # about the DIVERGENCE verdict, so it supplies a readable envelope.
    monkeypatch.setattr(
        artifact_meta, "read_metadata", lambda _p: dict(meta))

    def _no_arm(*_a: Any, **_k: Any) -> Any:
        raise AssertionError("arm_aot must not run on a diverged cell")

    monkeypatch.setattr(provision, "arm_aot", _no_arm)

    mint_root = tmp_path / "mint-root"
    mint_root.mkdir()
    artifact = tmp_path / "cell.tar.gz"
    artifact.write_bytes(b"not-a-real-cell")
    arm = _arm_key(seal, TOOLCHAIN)
    pending = fleet_cells.PendingSelfMint(
        family="micro-diffusion", arm_token=arm.token,
        ref=f"x#{arm.token}", cfg=object(), target=tmp_path / "adopted.tar.gz",
        mint_root=mint_root, publisher=None, cache_dir=tmp_path / "cache",
        arm_key=arm)

    assert fleet_cells.adopt_delegated_mint(object(), pending, [artifact]) is None
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


def test_target_constant_pool_is_one_REFERENCE_per_fqn() -> None:
    """N entries share ONE pool slot per FQN — the N x constants VRAM demand
    that OOM'd the 36-entry sdxl arm must not be reconstructible.

    INVERTED by pgw#1177. This row asserted `pool["w"] is not resident`
    ("owned, not aliased", pgw#812 D3). That clone was the ONLY copy in the
    system — one full duplicate of the target's weights held for the life of
    the arm, ~5.1 GiB on sdxl's single `unet` target — in direct contradiction
    of §4.33 step 4, *"the compiled entries bind constants BY REFERENCE
    against the resident weights; there is no second copy of the model"*. The
    dedup this row actually guards is unchanged; only the ownership claim
    inverted.
    """
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
    assert pool["w"] is resident  # BY REFERENCE — no second copy (§4.33 §4)


def test_a_NON_contiguous_resident_is_the_one_thing_still_copied() -> None:
    """The surviving guard from the inversion above. Dropping the blanket
    clone kept exactly one thing it also did: an AOTI container binds a RAW
    POINTER, so a non-contiguous resident cannot be bound by reference and is
    copied individually — priced per tensor rather than by cloning the whole
    pool for its sake. Without this row the inversion would read as "nothing
    is ever copied", which would be a segfault precondition."""
    torch = pytest.importorskip("torch")
    spec = aot_serve.ConstantSpec(
        fqn="w", source=aot_serve.SOURCE_STATE_DICT,
        dtype="torch.float32", shape=(4,))
    resident = torch.arange(8, dtype=torch.float32)[::2]
    assert not resident.is_contiguous()

    pool = aot_serve.target_constant_pool([(spec,)], {"w": resident})
    assert pool["w"] is not resident
    assert pool["w"].is_contiguous()
    assert torch.equal(pool["w"], resident)
