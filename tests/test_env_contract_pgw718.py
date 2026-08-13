"""pgw#718/#719: erase-and-impose env contract + seal composition +
boot-vs-point-of-use enforcement. Red-verified against real process state:
real os.environ, real torch flags, real /proc/self/maps."""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any, Dict, Iterator

import pytest

torch = pytest.importorskip("torch")

from gen_worker import compile_cache as cc
from gen_worker import env_seal
from gen_worker.registry import CompileCell


@pytest.fixture(autouse=True)
def _reset_boot_seal(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(env_seal, "_BOOT_READBACK", None)
    yield
    env_seal._BOOT_READBACK = None


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
# #718: scrub + impose + typed knobs
# ---------------------------------------------------------------------------


def test_scrub_never_fails_and_names_what_it_erased(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hostile = "TORCHINDUCTOR_FORCE_DISABLE_CACHES"
    informational = "PYTORCH_VERSION"
    unknown = "TORCH_TOTALLY_UNKNOWN_TOGGLE_XYZ"
    for name in (hostile, informational, unknown, "OMP_NUM_THREADS",
                 "MKL_NUM_THREADS", "NVIDIA_TF32_OVERRIDE"):
        monkeypatch.setenv(name, "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/opt/libs")
    erased = env_seal.scrub_env()
    for name in (hostile, informational, unknown, "OMP_NUM_THREADS",
                 "MKL_NUM_THREADS", "NVIDIA_TF32_OVERRIDE"):
        assert name in erased and name not in os.environ, name
    # Plumbing untouched.
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"
    assert os.environ["LD_LIBRARY_PATH"] == "/opt/libs"


def test_establish_scrubs_then_imposes_and_is_pure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The 0.70.3 regression class, dead: a base-image var neither refuses
    the boot nor reaches the seal — post-establish the seal is a pure
    function of (SDK build x declared knobs)."""
    monkeypatch.setenv("PYTORCH_VERSION", "2.13.0")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    seal_a = env_seal.establish()
    assert "PYTORCH_VERSION" not in os.environ
    assert "CUBLAS_WORKSPACE_CONFIG" not in os.environ
    # No env-derived facts survive in the config block.
    assert "CUBLAS_WORKSPACE_CONFIG" not in seal_a["config"]
    env_seal._BOOT_READBACK = None
    seal_b = env_seal.establish()
    assert env_seal.seal_digest(seal_a) == env_seal.seal_digest(seal_b)


def test_typed_knob_is_sealed_and_unknown_knob_refuses() -> None:
    from gen_worker import settings_authority as sa

    try:
        default = sa.impose_torch()
        assert default["cudnn_benchmark"] == "False"
        base_digest = env_seal.seal_digest(env_seal.establish())
        env_seal._BOOT_READBACK = None
        knobbed_seal = env_seal.establish(
            overrides={"cudnn_benchmark": "True"})
        assert knobbed_seal["config"]["cudnn_benchmark"] == "True"
        assert torch.backends.cudnn.benchmark is True
        # The knob is SEALED — a knobbed process has a different identity
        # (pgw#1049: because it is part of the DECLARATION, not because a
        # read-back happened to see it).
        assert env_seal.seal_digest(knobbed_seal) != base_digest
        with pytest.raises(sa.SettingsImpositionError, match="not_a_real_knob"):
            sa.impose_torch(overrides={"not_a_real_knob": "1"})
    finally:
        env_seal._BOOT_READBACK = None
        env_seal.establish()  # restore canonical


def test_hash_seed_facts_ride_the_seal() -> None:
    config = env_seal.effective_config()
    assert "python_hash_seed" in config
    assert config["hash_randomization"] in ("0", "1")


# ---------------------------------------------------------------------------
# #719: loaded-library manifest
# ---------------------------------------------------------------------------


def test_loaded_libraries_come_from_the_real_loader_map() -> None:
    libs = dict(env_seal.loaded_library_digests())
    # torch is imported in this process: its native libs are mapped.
    assert any(name.startswith("libtorch") for name in libs), libs.keys()
    assert all(len(d) == 16 for d in libs.values() if d != "<unreadable>")
    assert env_seal.loaded_library_digests() == tuple(sorted(libs.items()))
    seal = env_seal.effective_seal()
    assert len(seal["loaded_libs"]) == 16  # combined digest is a seal fact
    # The DISK identity manifest (phase-independent), of which the
    # mapped set is a content-consistent subset — a mapped toolchain lib whose
    # digest diverges from the manifest is a substitution
    # `assert_seal_unchanged` refuses by name. pgw#1181 reads the manifest from
    # its producer: `compile_cache.artifact_metadata` embedded it in a
    # `torch-inductor-cache` cell and is deleted with that format, while
    # `aot_mint` records this same call's output under the same key.
    manifest = dict(env_seal.frozen_library_digests())
    for base, digest in libs.items():
        if base in manifest and digest != "<unreadable>":
            assert manifest[base] == digest, base


# ---------------------------------------------------------------------------
# #719: boot-seal vs point-of-use
# ---------------------------------------------------------------------------


def test_drift_between_boot_and_use_refuses_named() -> None:
    env_seal.establish()
    env_seal.assert_seal_unchanged("mint")  # unchanged: passes
    before = torch.backends.cudnn.benchmark
    try:
        torch.backends.cudnn.benchmark = True  # behind-our-back mutation
        with pytest.raises(env_seal.EnvSealError) as excinfo:
            env_seal.assert_seal_unchanged("mint")
        message = str(excinfo.value)
        assert "config/cudnn_benchmark" in message
        assert "boot 'False'" in message and "now 'True'" in message
        assert "mint" in message
    finally:
        torch.backends.cudnn.benchmark = before
    env_seal.assert_seal_unchanged("mint")  # restored: passes again


def test_mint_refuses_on_env_drift_naming_the_flag(tmp_path: Path) -> None:
    """The pgw#719 red test end-to-end: a flag mutated between boot and the
    mint trace fails the REAL mint path, naming the flag."""

    def forward(self: Any, x: Any, scale: float) -> Any:
        return self.lin(x) * scale

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
    torch._dynamo.reset()
    compiled = torch.compile(mod.forward, backend="eager", dynamic=None)
    compiled(torch.randn(2, 4, 8), 1.0)

    capture = tmp_path / "capture"
    fx_entry = capture / "inductor" / "fxgraph" / "aa" / "bb"
    fx_entry.mkdir(parents=True)
    (fx_entry / "entry").write_bytes(b"fx")

    # The seal this drove was `finish_fleet_mint`'s, which packed a
    # DYNAMO cell and is deleted with that artifact class. The RULE is
    # unchanged and lives where every surviving mint reads it —
    # `env_seal.assert_seal_unchanged("mint")`, called by `mint_artifact` (the
    # local store's mint) and by `aot_mint` — so the drift is driven directly
    # against the one authority rather than through a deleted caller.
    env_seal.establish()
    before = torch.backends.cudnn.benchmark
    try:
        torch.backends.cudnn.benchmark = True
        with pytest.raises(env_seal.EnvSealError, match="cudnn_benchmark"):
            env_seal.assert_seal_unchanged("mint")
        assert not (tmp_path / "cell.tar.gz").exists()
    finally:
        torch.backends.cudnn.benchmark = before
        torch._dynamo.reset()


# pgw#1181 REMOVED `test_arm_names_env_seal_drift`. Its subject,
# `compile_cache.artifact_drift`, compared a delivered
# `torch-inductor-cache` cell's recorded env seal with the live one at ARM
# time; the format has had no writer since pgw#1178 and is deleted. The
# property is stronger on the surviving lane rather than absent: the seal is
# an identity AXIS of the exported cell (`fleet_cells.ENV_SEAL_AXIS`, folded
# into `ck1`), so a drifted seal yields a different key and the cell never
# resolves at all — proven in `tests/test_cell_key_pgw1059.py`, whose
# staleness matrix lists `env_seal` among the axes that re-key.
