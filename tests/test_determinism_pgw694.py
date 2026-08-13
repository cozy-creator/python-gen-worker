"""pgw#694 determinism hardening: posture seal (#695), env seal
(#696), composition fingerprint (#697), cubin gate (#698).

Red-verified against real torch state and real file trees — no mocks of
torch posture, no fabricated guard rows where a real API exists. CPU only.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any, Dict, Iterator

import pytest

torch = pytest.importorskip("torch")

from gen_worker import compile_cache as cc
from gen_worker import guard_closure as gc
from gen_worker.registry import CompileCell


def _env_seal() -> Any:
    """Function-scoped so the pre-fix red run COLLECTS: on a pre-pgw#696
    tree the module is absent and each seal test fails red individually."""
    from gen_worker import env_seal

    return env_seal


@pytest.fixture(autouse=True)
def _fresh_dynamo() -> Iterator[None]:
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


@pytest.fixture(autouse=True)
def _restore_global_matmul_flags() -> Iterator[None]:
    """The canonical imposition is deliberately process-global; the SUITE
    must not leak it across files (entries compiled under one TF32 state
    GlobalStateGuard-miss under another — the flux hit-counter tests)."""
    precision = torch.get_float32_matmul_precision()
    matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    benchmark = torch.backends.cudnn.benchmark
    yield
    torch.set_float32_matmul_precision(precision)
    torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
    torch.backends.cudnn.allow_tf32 = cudnn_tf32
    torch.backends.cudnn.benchmark = benchmark


def _cfg(**overrides: Any) -> CompileCell:
    base: Dict[str, Any] = dict(
        shapes=((64, 64),), targets=("transformer",), family="toyfam",
        regional=False, text_len=None, dynamic=(), lora_bucket=0,
        guidance_scales=(), text_lens=(),
    )
    base.update(overrides)
    return CompileCell(**base)


# ---------------------------------------------------------------------------
# #695: process-posture seal
# ---------------------------------------------------------------------------


def test_pytest_process_posture_is_canonical() -> None:
    """The canonical table IS this box's honest resting state — establish
    must be a no-op assertion here, not a mutation."""
    assert gc.posture_snapshot() == gc.CANONICAL_POSTURE
    assert gc.establish_posture() == gc.CANONICAL_POSTURE


# pgw#1181 REMOVED 11 rows whose subject is the `torch-inductor-cache` format:
# `assert_posture` and the two posture rows that round-tripped a seal through a
# cell's metadata, the three `consolidate` rows, the cubin/PTX `pack`
# completeness rows (3), `verify`'s sku and silent-axes rows (2), and the local
# store's seal-drift verdict. The manifest they compared is written by
# `closure_manifest`, the pack they exercised is `compile_cache.pack`, and both
# are deleted with the format whose last writer died in pgw#1178. The posture
# fact itself survives and is proven where it is produced —
# `establish_posture` in this file and in `test_torchless_boot_pgw788`.



# ---------------------------------------------------------------------------
# #696: env seal joins the key
# ---------------------------------------------------------------------------


def test_hostile_torch_env_is_erased_not_fatal(monkeypatch: Any) -> None:
    """pgw#718 erase-and-impose (supersedes the #696 allowlist gate): an
    unknown toggle in a behavior namespace is DELETED — never a refusal,
    never a silently different kernel. Plumbing survives."""
    monkeypatch.setenv("TORCHINDUCTOR_MAX_AUTOTUNE", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    erased = _env_seal().scrub_env()
    assert "TORCHINDUCTOR_MAX_AUTOTUNE" in erased
    import os

    assert "TORCHINDUCTOR_MAX_AUTOTUNE" not in os.environ
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "0"  # plumbing kept


def test_establish_config_imposes_the_serving_posture() -> None:
    """The canonical table IS the pgw#654 serving posture (TF32 on):
    establish must impose it from ANY prior state and verify the read-back
    — code decides the flags, never a library default or an env var."""
    from gen_worker import settings_authority as sa

    before = torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cudnn.allow_tf32 = False  # non-canonical prior state
        effective = sa.impose_torch()
        assert effective["cudnn_allow_tf32"] == "True"
        assert torch.backends.cudnn.allow_tf32 is True
        assert effective["float32_matmul_precision"] == "high"
        assert sa.torch_readback() == effective
    finally:
        torch.backends.cudnn.allow_tf32 = before
        sa.impose_torch()


def test_seal_digest_tracks_the_declaration_not_live_flags() -> None:
    """pgw#1049 REVERSES this test's original claim: the seal digests the
    DECLARATION, so a behind-our-back flag flip can no longer move it — it
    trips the pgw#719 wire instead. A DECLARED change (a knob) still moves
    the digest, which is what keeps config in the key."""
    from gen_worker import settings_authority as sa

    baseline = _env_seal().seal_digest(_env_seal().effective_seal())
    before = torch.backends.cudnn.benchmark
    try:
        torch.backends.cudnn.benchmark = not before
        assert _env_seal().seal_digest(_env_seal().effective_seal()) == baseline
    finally:
        torch.backends.cudnn.benchmark = before
    knobbed = dict(_env_seal().effective_seal())
    knobbed["config"] = sa.validated_table({"cudnn_benchmark": "True"})
    assert _env_seal().seal_digest(knobbed) != baseline


# ---------------------------------------------------------------------------
# #697: composition fingerprint
# ---------------------------------------------------------------------------


class _Tree(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin1 = torch.nn.Linear(8, 8)
        self.lin2 = torch.nn.Linear(8, 8)

    def forward(self, x: Any) -> Any:
        return self.lin2(self.lin1(x))


class _Pipe:
    def __init__(self) -> None:
        self.transformer = _Tree()


def test_a_dtype_flip_in_one_submodule_moves_the_graph_signature() -> None:
    """The pgw#683 class: one submodule left in Half inside a bf16 tree.

    `composition_fingerprint` and `contract_drift` were the
    `torch-inductor-cache` cell's own adoption fence and are deleted with the
    format. The FACT they rested on is `execution_contract`, which is alive and
    is folded into `ck1`'s contract axis — so on the surviving lane the drifted
    consumer does not get a named refusal, it gets a different key and never
    resolves the cell at all. That is strictly stronger, and this row states
    the fact underneath it: the signature moves on the flip and does NOT move
    on a fine-tune of the same composition."""
    minted = _Pipe()
    cfg = _cfg()
    signature, _contract = cc.execution_contract(minted, cfg)

    consumer = _Pipe()
    consumer.transformer.lin2.half()  # the drift
    drifted, _ = cc.execution_contract(consumer, cfg)
    assert drifted != signature

    # Identical composition (different VALUES = a fine-tune) is the same graph.
    finetune = _Pipe()
    with torch.no_grad():
        finetune.transformer.lin1.weight.mul_(2.0)
    same, _ = cc.execution_contract(finetune, cfg)
    assert same == signature


def test_hook_presence_is_a_composition_fact() -> None:
    """fp8 layerwise-cast is hook-driven: a hooked module tree is
    a DIFFERENT composition than a bare one. Red pre-fix: identical
    signatures with and without hooks."""
    bare, hooked = _Pipe(), _Pipe()
    handle = hooked.transformer.lin1.register_forward_hook(
        lambda mod, args, out: out)
    try:
        cfg = _cfg()
        sig_bare, _ = cc.execution_contract(bare, cfg)
        sig_hooked, _ = cc.execution_contract(hooked, cfg)
        assert sig_bare != sig_hooked
        # The per-module `composition_fingerprint` rows that used to
        # localise this to `transformer:lin1` were the deleted format's
        # adoption fence. The hook-presence FACT they digested is the same one
        # `execution_contract` hashes (`_module_hooks`), which is why the
        # signatures above differ at all.
    finally:
        handle.remove()


# ---------------------------------------------------------------------------
# #698: cubin-completeness gate at pack
# ---------------------------------------------------------------------------


def _elf_cubin_bytes(arch: int) -> bytes:
    """A minimal ELF64 header whose e_flags carry ``arch`` in the low byte
    — the real layout nvidia cubins use (file-format check, no GPU)."""
    header = bytearray(64)
    header[0:4] = b"\x7fELF"
    header[4] = 2  # ELF64
    header[5] = 1  # little-endian
    struct.pack_into("<I", header, 0x30, arch)
    return bytes(header)


def _capture(tmp_path: Path, *, ptx: bool, cubin_arch: int = 0) -> Path:
    root = tmp_path / "capture"
    fx = root / "inductor" / "fxgraph" / "aa" / "bb"
    fx.mkdir(parents=True)
    (fx / "entry").write_bytes(b"fx")
    kdir = root / "triton" / "k1"
    kdir.mkdir(parents=True)
    if ptx:
        (kdir / "kern.ptx").write_text(".version 8.0")
    if cubin_arch:
        (kdir / "kern.cubin").write_bytes(_elf_cubin_bytes(cubin_arch))
    return root


# ---------------------------------------------------------------------------
# cache-design review fixes (ML-systems + build-systems reviews)
# ---------------------------------------------------------------------------


def test_fx_system_shim_normalizes_device_name(monkeypatch: Any) -> None:
    """P0 (review 6.1, VERIFIED on a real B200 cell: system_info[device] =
    {'name': 'NVIDIA B200'}): the inner FX key hashes the GPU MARKETING
    name, so cross-SKU same-sm adoption missed 100% inside torch's own
    lookup. The shim rewrites the name to the sm token with the hash
    recomputed via torch's own strategy — two SKUs of one arch become one
    inner key."""
    from torch._inductor.codecache import SYSTEM_CACHE_KEY_STRATEGY

    a40 = {"device": {"name": "NVIDIA A40"},
           "version": {"triton": "tk", "cuda": "13.0"}, "hash": "orig"}
    rtx = {"device": {"name": "NVIDIA GeForce RTX 3090"},
           "version": {"triton": "tk", "cuda": "13.0"}, "hash": "orig"}
    na = cc._normalize_system_info(dict(a40), "sm_86")
    nb = cc._normalize_system_info(dict(rtx), "sm_86")
    assert na == nb  # THE portability claim: one arch, one inner key
    assert na["device"]["name"] == "sm_86"
    assert na["hash"] != "orig"
    assert na["hash"] == SYSTEM_CACHE_KEY_STRATEGY.key_from_json(
        {"device": na["device"], "version": na["version"]})
    # CPU / cuda-less shape passes through untouched.
    cpu = {"hash": "bare"}
    assert cc._normalize_system_info(dict(cpu), "sm_86") == cpu
    # No sm token (non-CUDA runtime): no rewrite.
    assert cc._normalize_system_info(dict(a40), "") == a40


def test_fx_system_shim_installs_idempotently(monkeypatch: Any) -> None:
    from torch._inductor import codecache

    fake = {"device": {"name": "NVIDIA A40"},
            "version": {"triton": "tk", "cuda": "13.0"}, "hash": "orig"}
    monkeypatch.setattr(
        codecache.CacheBase, "get_system", staticmethod(lambda: dict(fake)))
    monkeypatch.setattr(cc, "runtime_key", lambda: {
        "sku": "a40", "sm": "sm_86", "cuda": "13.0", "cuda_driver": "",
        "torch": "t", "triton": "3", "image_digest": "",
    })
    cc._install_fx_system_shim()
    try:
        got = codecache.CacheBase.get_system()
        assert got["device"]["name"] == "sm_86"
        marked = codecache.CacheBase.get_system
        cc._install_fx_system_shim()  # second install is a no-op
        assert codecache.CacheBase.get_system is marked
    finally:
        # monkeypatch restores the fake; the shim wrapped the fake only.
        pass


def test_upstream_get_system_shape_is_pinned() -> None:
    """Version-pin (pgw#705 doctrine): the shim rewrites a structure torch
    does not contract. If a torch bump changes get_system's shape, this
    fails LOUDLY before the shim ships against it."""
    from pathlib import Path as _P

    from torch._inductor import codecache

    source = _P(codecache.__file__).read_text()
    assert "def get_system()" in source
    assert "get_device_properties" in source
    assert "device_properties.name" in source  # the SKU-name pin we rewrite
    assert "SYSTEM_CACHE_KEY_STRATEGY" in source
    # The upstream precedent the shim cites: AOTI keys on capability.
    assert "AOTI_COMPUTE_CAPABILITY" in source


def test_semantic_cache_tag_binds_semantic_axes_only() -> None:
    """Review 6.3: the tag digests format|kind|family|lane|mode|contract —
    a foreign semantic identity can never consume delivered entries; the
    environment axes stay OUT so pgw#700 equivalence adoption survives."""
    import torch.compiler.config as compiler_config

    pipe_a, pipe_b = _Pipe(), _Pipe()
    tag_same = cc._semantic_cache_tag(pipe_a, _cfg())
    assert tag_same == cc._semantic_cache_tag(pipe_b, _cfg())
    assert tag_same != cc._semantic_cache_tag(pipe_a, _cfg(family="other"))
    assert tag_same != cc._semantic_cache_tag(
        pipe_a, _cfg(shapes=((32, 32),)))  # contract rides the tag
    before = compiler_config.cache_key_tag
    try:
        cc._set_semantic_cache_tag(pipe_a, _cfg())
        assert compiler_config.cache_key_tag == tag_same
    finally:
        compiler_config.cache_key_tag = before


def test_pytorch_and_triton_namespaces_are_scrubbed(monkeypatch: Any) -> None:
    """pgw#718: every behavior namespace is erased wholesale — PYTORCH_*
    (which the original TORCH*-only gate missed) and TRITON_*
    (TRITON_PTXAS_PATH = silently different cubins) included. The 0.70.3
    fleet-killer class (informational PYTORCH_VERSION refused at boot) is
    impossible by construction: erased, never fatal."""
    seal = _env_seal()
    for name in ("PYTORCH_FOO_TOGGLE", "TRITON_PTXAS_PATH",
                 "PYTORCH_VERSION", "CUBLAS_WORKSPACE_CONFIG"):
        monkeypatch.setenv(name, "x")
    erased = seal.scrub_env()
    import os

    for name in ("PYTORCH_FOO_TOGGLE", "TRITON_PTXAS_PATH",
                 "PYTORCH_VERSION", "CUBLAS_WORKSPACE_CONFIG"):
        assert name in erased and name not in os.environ


def test_the_seal_carries_no_operator_settable_recall_salt(
    monkeypatch: Any,
) -> None:
    """pgw#1034 deleted the ``epoch`` fact (``COZY_CELL_EPOCH``).

    A recall is a recorded operator intent with an actor and a reason; an env
    var on a pod is neither, which is why the config-reads allowlist ruled the
    read a VIOLATION and named the hub's ``cell_revocations`` as the
    real home. The seal is now unmovable from the process environment: the
    ONLY way to disown a cell generation is a ``SEAL_VERSION`` bump in this
    file, which is a diff someone signs.
    """
    seal = _env_seal()
    base = seal.effective_seal()
    assert base["seal_v"] == seal.SEAL_VERSION
    assert "epoch" not in base
    assert not hasattr(seal, "EPOCH_ENV")

    baseline = seal.seal_digest(base)
    monkeypatch.setenv("COZY_CELL_EPOCH", "1")
    assert seal.seal_digest(seal.effective_seal()) == baseline

