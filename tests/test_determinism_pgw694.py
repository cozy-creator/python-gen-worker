"""pgw#694 determinism hardening: posture seal (#695), env seal + ck4
(#696), composition fingerprint (#697), cubin gate (#698).

Red-verified against real torch state and real file trees — no mocks of
torch posture, no fabricated guard rows where a real API exists. CPU only.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any, Dict, Iterator

import pytest

torch = pytest.importorskip("torch")

from gen_worker import cell_key as ck
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


def test_assert_posture_names_every_drifted_fact() -> None:
    with torch.no_grad():
        with pytest.raises(gc.PostureError) as excinfo:
            gc.assert_posture(gc.CANONICAL_POSTURE, label="arm")
    message = str(excinfo.value)
    assert "grad_enabled" in message
    assert "sealed 'True'" in message and "process 'False'" in message
    assert "arm" in message
    # Clean state passes.
    gc.assert_posture(gc.CANONICAL_POSTURE, label="arm")


def test_mint_manifest_carries_the_posture_seal() -> None:
    """assert_closure seals the live posture into the manifest (v2)."""
    import threading

    def forward(self: Any, x: Any) -> Any:
        return self.lin(x) + 1

    class _Mod(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = torch.nn.Linear(8, 8)

    _Mod.forward = forward  # type: ignore[method-assign]

    class _Pipe:
        pass

    pipe = _Pipe()
    mod = _Mod()
    pipe.transformer = mod  # type: ignore[attr-defined]
    setattr(pipe, cc._MARKER_ATTR, {
        "targets": ["transformer"], "shapes": [(64, 64)], "cache": True,
        "originals": [(mod, "forward", mod.forward)], "regional_mods": [],
        "failure_signal": {
            "callback": None, "lock": threading.Lock(),
            "successful_calls": 0, "cache_hits": 0, "cache_misses": 0,
            "router": None,
        },
    })
    compiled = torch.compile(mod.forward, backend="eager", dynamic=None)
    compiled(torch.randn(2, 4, 8))

    manifest = gc.assert_closure(pipe, _cfg(), label="toyfam")
    assert manifest[gc.POSTURE_KEY] == gc.CANONICAL_POSTURE
    assert gc.MANIFEST_VERSION == 2 and manifest["v"] == 2

    # A mint attempted in a non-canonical posture fails red, named.
    with torch.no_grad():
        with pytest.raises(gc.PostureError, match="grad_enabled"):
            gc.assert_closure(pipe, _cfg(), label="toyfam")


def test_arm_refuses_on_posture_drift_named() -> None:
    """artifact_drift converts a sealed-vs-live posture mismatch into a
    named refusal at arm time — never a downstream guard miss."""
    base = {
        "compile_mode": "whole", "shapes": [[64, 64]],
        "targets": ["transformer"], "guidance_scales": [],
    }
    meta = dict(base)
    meta[gc.MANIFEST_KEY] = {
        "v": 2, "graphs": [], "leaks": [],
        gc.POSTURE_KEY: dict(gc.CANONICAL_POSTURE),
    }

    class _Pipe:
        pass

    assert cc.artifact_drift(meta, _Pipe(), _cfg()) == ""
    with torch.no_grad():
        drift = cc.artifact_drift(meta, _Pipe(), _cfg())
    assert "posture" in drift and "grad_enabled" in drift

    # A manifest-bearing cell WITHOUT a posture seal is a pre-#695 mint.
    legacy = dict(base)
    legacy[gc.MANIFEST_KEY] = {"v": 1, "graphs": [], "leaks": []}
    assert "no posture seal" in cc.artifact_drift(legacy, _Pipe(), _cfg())


def test_consolidate_flags_posture_divergence() -> None:
    base = {"v": 2, "graphs": [], "leaks": [],
            gc.POSTURE_KEY: dict(gc.CANONICAL_POSTURE)}
    drifted = json.loads(json.dumps(base))
    drifted[gc.POSTURE_KEY]["grad_enabled"] = "False"
    audit = gc.consolidate({"pod-a": base, "pod-b": drifted})
    assert not audit.ok
    assert any("posture/grad_enabled" in d for d in audit.divergence)
    # Identical postures stay clean; a missing seal on ONE pod diverges.
    assert gc.consolidate({"pod-a": base, "pod-b": base}).divergence == ()
    unsealed = {"v": 1, "graphs": [], "leaks": []}
    audit = gc.consolidate({"pod-a": base, "pod-b": unsealed})
    assert any("posture seal absent" in d and "pod-b" in d
               for d in audit.divergence)


# ---------------------------------------------------------------------------
# #696: env seal + ck4
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
    before = torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cudnn.allow_tf32 = False  # non-canonical prior state
        effective = _env_seal().establish_config()
        assert effective["cudnn_allow_tf32"] == "True"
        assert torch.backends.cudnn.allow_tf32 is True
        assert effective["float32_matmul_precision"] == "high"
        assert _env_seal().effective_config() == effective
    finally:
        torch.backends.cudnn.allow_tf32 = before
        _env_seal().establish_config()


def test_seal_digest_tracks_config_flags() -> None:
    """Flipping one sealed flag changes the seal digest — and therefore the
    ck4 key (red pre-fix: no env_seal axis, keys blind to config)."""
    baseline = _env_seal().seal_digest(_env_seal().effective_seal())
    before = torch.backends.cudnn.benchmark
    try:
        torch.backends.cudnn.benchmark = not before
        assert _env_seal().seal_digest(_env_seal().effective_seal()) != baseline
    finally:
        torch.backends.cudnn.benchmark = before
    assert _env_seal().seal_digest(_env_seal().effective_seal()) == baseline


def test_ck5_requires_and_recomputes_the_env_seal(monkeypatch: Any) -> None:
    monkeypatch.setattr(cc, "runtime_key", lambda: {
        "sku": "l4", "sm": "sm_89", "cuda": "13.0", "cuda_driver": "13000",
        "torch": "2.13.0+cu130", "triton": "3.7.1", "image_digest": "",
    })
    monkeypatch.setattr(cc, "gen_worker_version", lambda: "0.74.0")
    monkeypatch.setattr(cc, "_lib_versions", lambda: {})
    monkeypatch.delenv("WORKER_IMAGE_DIGEST", raising=False)

    facts = cc.declared_contract_facts(_cfg())
    want = ck.compute("toyfam", contract=ck.contract_digest(facts))
    assert want.digest.startswith("ck5-")
    assert not ck.is_key("ck3-" + "a" * 56)  # older schemes are dead

    meta = cc.artifact_metadata(
        family="toyfam", shapes=((64, 64),), targets=("transformer",),
        shape_contract=facts,
    )
    # The metadata records the seal dict verbatim; the key axis is
    # RECOMPUTED from it and matches the live compute.
    assert meta[_env_seal().SEAL_KEY]["seal_v"] == _env_seal().SEAL_VERSION
    assert meta["cell_key"] == want.digest
    assert ck.mismatch(meta, want) == ""

    # A cell sealed under a DIFFERENT environment gets a different key,
    # and the mismatch names the env_seal axis.
    drifted = json.loads(json.dumps(meta))
    drifted[_env_seal().SEAL_KEY]["config"]["cudnn_benchmark"] = "True"
    assert ck.from_artifact_metadata(drifted).digest != want.digest
    assert ck.mismatch(drifted, want).startswith("env_seal:")

    # No env_seal block = pre-ck4 cell = no identity.
    unsealed = {k: v for k, v in meta.items() if k != _env_seal().SEAL_KEY}
    with pytest.raises(ck.CellKeyError, match="env_seal"):
        ck.from_artifact_metadata(unsealed)


def test_verify_no_longer_pins_sku(monkeypatch: Any) -> None:
    """ck3 completion (pgw#691): sku left the identity axes, but verify()
    still hard-required it — a same-sm cell minted on a different SKU
    refused to arm, structurally undoing the collapse."""
    monkeypatch.setattr(cc, "runtime_key", lambda: {
        "sku": "rtx-3090", "sm": "sm_86", "cuda": "13.0",
        "cuda_driver": "13000", "torch": "2.13.0+cu130", "triton": "3.7.1",
        "image_digest": "",
    })
    monkeypatch.setattr(cc, "gen_worker_version", lambda: "0.74.0")
    monkeypatch.setattr(cc, "_lib_versions", lambda: {})
    meta = {
        "format": cc.ARTIFACT_FORMAT, "sku": "a40", "sm": "sm_86",
        "cuda": "13.0", "torch": "2.13.0+cu130", "triton": "3.7.1",
        "gen_worker": "0.74.0", "libs": {},
    }
    assert cc.verify(dict(meta)) == ""  # sku differs, sm matches: arms
    assert "sm" in cc.verify(dict(meta, sm="sm_89"))  # sm still refuses


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


def test_fingerprint_names_the_drifted_module_on_dtype_flip() -> None:
    """The pgw#683 class: one submodule left in Half inside a bf16-tree.
    Pre-fix the refusal was two digest prefixes; now it names the module."""
    minted = _Pipe()
    cfg = _cfg()
    signature, contract = cc.execution_contract(minted, cfg)
    meta = cc.artifact_metadata(
        family="toyfam", shapes=cfg.shapes, targets=cfg.targets,
        graph_signature=signature, weight_contract=contract,
        shape_contract=cc.declared_contract_facts(cfg),
        composition=cc.composition_fingerprint(minted, cfg),
    )

    consumer = _Pipe()
    consumer.transformer.lin2.half()  # the drift
    drift = cc.contract_drift(meta, consumer, cfg)
    assert "module composition" in drift
    assert "transformer:lin2" in drift
    assert "transformer:lin1" not in drift

    # Identical composition (different VALUES = a fine-tune) arms.
    finetune = _Pipe()
    with torch.no_grad():
        finetune.transformer.lin1.weight.mul_(2.0)
    assert cc.contract_drift(meta, finetune, cfg) == ""


def test_hook_presence_is_a_composition_fact() -> None:
    """fp8 layerwise-cast is hook-driven (ie#381): a hooked module tree is
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
        rows_bare = dict(cc.composition_fingerprint(bare, cfg))
        rows_hooked = dict(cc.composition_fingerprint(hooked, cfg))
        assert rows_bare["transformer:lin1"] != rows_hooked["transformer:lin1"]
        assert rows_bare["transformer:lin2"] == rows_hooked["transformer:lin2"]
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


def test_ptx_only_kernel_fails_the_pack_naming_it(tmp_path: Path) -> None:
    root = _capture(tmp_path, ptx=True)
    with pytest.raises(RuntimeError) as excinfo:
        cc.pack(root, tmp_path / "cell.tar.gz", {"sm": "sm_89"})
    message = str(excinfo.value)
    assert "PTX" in message and "kern.ptx" in message and "pgw#698" in message
    assert not (tmp_path / "cell.tar.gz").exists()


def test_sm_exact_cubin_packs_and_wrong_arch_refuses(tmp_path: Path) -> None:
    good = _capture(tmp_path, ptx=True, cubin_arch=89)
    out = tmp_path / "cell.tar.gz"
    cc.pack(good, out, {"sm": "sm_89"})
    assert out.exists()

    bad = _capture(tmp_path / "bad", ptx=True, cubin_arch=86)
    with pytest.raises(RuntimeError) as excinfo:
        cc.pack(bad, tmp_path / "bad.tar.gz", {"sm": "sm_89"})
    assert "sm_86" in str(excinfo.value) and "sm_89" in str(excinfo.value)


def test_inductor_only_capture_still_packs(tmp_path: Path) -> None:
    """No triton kernels at all (the existing fixture shape) stays valid."""
    root = _capture(tmp_path, ptx=False)
    out = tmp_path / "cell.tar.gz"
    cc.pack(root, out, {"sm": "sm_89"})
    assert out.exists()


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


def test_verify_refuses_silent_axes_named(monkeypatch: Any) -> None:
    """P1 (review 6.9 — the JAX #27814 wrong-hit shape): a cell SILENT on
    sm/cuda/image_digest/libs was accepted by `if want and ...`. Strict
    now: absent axis = named refusal."""
    monkeypatch.setattr(cc, "runtime_key", lambda: {
        "sku": "l4", "sm": "sm_89", "cuda": "13.0", "cuda_driver": "",
        "torch": "2.13.0+cu130", "triton": "3.7.1",
        "image_digest": "sha256:live",
    })
    monkeypatch.setattr(cc, "gen_worker_version", lambda: "0.74.0")
    monkeypatch.setattr(cc, "_lib_versions", lambda: {"diffusers": "0.39.0"})
    complete = {
        "format": cc.ARTIFACT_FORMAT, "torch": "2.13.0+cu130",
        "triton": "3.7.1", "sm": "sm_89", "cuda": "13.0",
        "image_digest": "sha256:live", "gen_worker": "0.74.0",
        "libs": {"diffusers": "0.39.0"},
    }
    assert cc.verify(dict(complete)) == ""
    for axis in ("sm", "cuda", "image_digest"):
        silent = {k: v for k, v in complete.items() if k != axis}
        assert axis in cc.verify(silent), axis
    silent_libs = dict(complete, libs={})
    assert "diffusers" in cc.verify(silent_libs)


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


def test_epoch_salt_disowns_a_generation(monkeypatch: Any) -> None:
    """R2 (Bazel Action.salt / ccache HASH_PREFIX): bumping COZY_CELL_EPOCH
    changes every env_seal digest — one config change disowns a poisoned
    mint generation, no scheme bump."""
    seal = _env_seal()
    monkeypatch.delenv(seal.EPOCH_ENV, raising=False)
    base = seal.effective_seal()
    assert base["seal_v"] == 3 and base["epoch"] == "0"
    baseline = seal.seal_digest(base)
    monkeypatch.setenv(seal.EPOCH_ENV, "1")
    assert seal.seal_digest(seal.effective_seal()) != baseline


def test_content_keys_recorded_in_metadata() -> None:
    """Review 6.5: torch/triton CONTENT digests ride metadata as the
    pgw#700 equivalence precondition (a patched wheel under an unchanged
    version string becomes visible)."""
    keys = dict(cc.content_keys())
    assert set(keys) == {"torch", "triton"}
    assert all(len(v) == 16 for v in keys.values() if v)
    assert keys["torch"], "torch_key must be computable on this wheel"
    meta = cc.artifact_metadata(
        family="toyfam", shapes=((64, 64),), targets=("transformer",))
    assert meta["content_keys"] == keys
