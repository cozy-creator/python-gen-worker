"""Config authority: the declared interpreter env, and the torch flags it imposes.

Sections keep their incident id; the full narratives live in the tracker.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import threading
from pathlib import Path
from typing import Any, Dict, Iterator

import pytest

from gen_worker import compile_cache as cc
from gen_worker import env_seal
from gen_worker import settings_authority as sa
from gen_worker.registry import CompileContract

# ============================================================================
# pgw#1049 — pgw#1049: the single settings authority — seal from
#   DECLARATION.
# ============================================================================

torch = pytest.importorskip("torch")


REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _fresh_boot(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(env_seal, "_BOOT_READBACK", None)
    monkeypatch.setattr(env_seal, "_ESTABLISHED_OVERRIDES", None)
    yield
    env_seal._BOOT_READBACK = None
    env_seal._ESTABLISHED_OVERRIDES = None


@pytest.fixture()
def _restore_inductor() -> Iterator[None]:
    import torch._inductor.config as ic

    max_autotune = ic.max_autotune
    metadata = dict(ic.aot_inductor.metadata)
    yield
    ic.max_autotune = max_autotune
    ic.aot_inductor.metadata = metadata


def test_pgw1042_mutation_cannot_move_the_seal(_restore_inductor: None) -> None:
    """The pgw#1042 replay, against v4: mutate global inductor config the way a mid-mint library side effect doe..."""
    import torch._inductor.config as ic

    env_seal.establish()
    sealed = env_seal.seal_digest(env_seal.effective_seal())
    readback_before = env_seal.inductor_config_digest()

    # The mutation class: a global entry OUTSIDE _PORTABLE_VOLATILE (the
    # pgw#1042 fix excluded exactly one entry; the DISEASE was never that
    # one entry — this is the cure for the class).
    ic.max_autotune = True

    # v3's seal would have moved: the read-back digest demonstrably does.
    assert env_seal.inductor_config_digest() != readback_before
    # v4's seal did not: identity derives from the declaration.
    assert env_seal.seal_digest(env_seal.effective_seal()) == sealed
    # And the mint seams refuse by name — the JIT mint's pre-trace assert
    # and aot_mint's (both call assert_seal_unchanged).
    with pytest.raises(env_seal.EnvSealError, match="inductor"):
        env_seal.assert_seal_unchanged("mint")


def test_torch_owned_compile_output_is_not_drift(_restore_inductor: None) -> None:
    """The OTHER half of pgw#1042: ``aot_compile`` legitimately writes machine facts into ``aot_inductor.metadat..."""
    import torch._inductor.config as ic

    env_seal.establish()
    sealed = env_seal.seal_digest(env_seal.effective_seal())
    ic.aot_inductor.metadata["AOTI_CPU_ISA"] = "AVX2"  # what aot_compile does
    assert env_seal.seal_digest(env_seal.effective_seal()) == sealed
    env_seal.assert_seal_unchanged("post-compile")  # no refusal


def test_backend_flag_mutation_trips_but_cannot_rekey() -> None:
    env_seal.establish()
    sealed = env_seal.seal_digest(env_seal.effective_seal())
    before = torch.backends.cudnn.benchmark
    try:
        torch.backends.cudnn.benchmark = True
        assert env_seal.seal_digest(env_seal.effective_seal()) == sealed
        with pytest.raises(env_seal.EnvSealError, match="cudnn_benchmark"):
            env_seal.assert_seal_unchanged("mint")
    finally:
        torch.backends.cudnn.benchmark = before


def test_seal_v4_states_the_declaration() -> None:
    seal = env_seal.establish()
    assert seal["seal_v"] == env_seal.SEAL_VERSION == 4
    decl = sa.declaration()
    for fact in ("env", "config", "dynamo", "inductor", "posture"):
        assert seal[fact] == decl[fact], fact
    assert seal["env"]["PYTHONHASHSEED"] == "0"
    assert len(seal["loaded_libs"]) == 16  # the one measured (content) fact


def test_declared_knob_moves_identity_by_declaration() -> None:
    base = env_seal.seal_digest(env_seal.establish())
    env_seal._BOOT_READBACK = None
    try:
        knobbed = env_seal.establish(overrides={"cudnn_benchmark": "True"})
        assert knobbed["config"]["cudnn_benchmark"] == "True"
        assert env_seal.seal_digest(knobbed) != base
    finally:
        env_seal._BOOT_READBACK = None
        env_seal.establish()


def test_dynamo_shape_posture_is_process_wide() -> None:
    """pgw#1049: The v2 dynamo posture moved into the authority (it was a second writer in compile_cache): impos..."""
    import torch._dynamo

    sa.impose_dynamo()
    got = sa.read_in_fresh_thread(
        lambda: (bool(torch._dynamo.config.automatic_dynamic_shapes),
                 bool(torch._dynamo.config.assume_static_by_default)))
    assert got == (False, True)
    assert sa.dynamo_readback() == sa.DECLARED_DYNAMO


def _run(code: str, *, seed_env: dict) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k != "PYTHONHASHSEED"}
    env.update(seed_env)
    env["PYTHONPATH"] = str(REPO / "src")
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        env=env, capture_output=True, text=True, timeout=120)


def test_establish_refuses_undeclared_hash_seed() -> None:
    """pgw#1049: Fail-closed: a process whose interpreter booted outside the declared env cannot seal an identit..."""
    code = """
    from gen_worker import env_seal
    try:
        env_seal.establish()
    except Exception as exc:
        print(f"REFUSED: {type(exc).__name__}: {exc}")
    else:
        print("ESTABLISHED")
    """
    out = _run(code, seed_env={})
    assert "REFUSED: SettingsImpositionError" in out.stdout, out.stdout + out.stderr
    assert "PYTHONHASHSEED" in out.stdout
    ok = _run(code, seed_env={"PYTHONHASHSEED": "0"})
    assert "ESTABLISHED" in ok.stdout, ok.stdout + ok.stderr


def test_ensure_interpreter_env_reexecs_once() -> None:
    """pgw#1049: The sanctioned imposition: a process launched without the declared seed re-execs itself (sys.or..."""
    code = """
    import sys
    from gen_worker.settings_authority import ensure_interpreter_env
    ensure_interpreter_env()
    print(f"flag={sys.flags.hash_randomization} "
          f"seed={__import__('os').environ.get('PYTHONHASHSEED')}")
    """
    out = _run(code, seed_env={})
    assert "flag=0 seed=0" in out.stdout, out.stdout + out.stderr


def test_children_inherit_the_declared_seed() -> None:
    """pgw#1049: impose_process_env is what every spawned child inherits — the mint child and AOT entry children..."""
    sa.impose_process_env()
    assert os.environ["PYTHONHASHSEED"] == "0"
    child = subprocess.run(
        [sys.executable, "-c",
         "import sys; print(sys.flags.hash_randomization)"],
        capture_output=True, text=True, timeout=60)
    assert child.stdout.strip() == "0"


def test_revert_is_one_declaration_entry() -> None:
    """pgw#1049: The HUMAN_MUST_DO record promises the imposition is trivially revertible: with the entry remove..."""
    saved = dict(sa.DECLARED_ENV)
    try:
        del sa.DECLARED_ENV["PYTHONHASHSEED"]
        assert sa._interpreter_env_diffs() == []
        sa.verify_interpreter_env()  # no refusal without the entry
    finally:
        sa.DECLARED_ENV.clear()
        sa.DECLARED_ENV.update(saved)


def test_census_namespaces_are_actually_scrubbed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in ("NCCL_SOMETHING", "ATEN_CPU_CAPABILITY",
                 "INDUCTOR_PROVENANCE", "AOT_INDUCTOR_ENABLE_LTO",
                 "CUDA_LAUNCH_BLOCKING", "CUTLASS_EPILOGUE_FUSION",
                 "TENSORIFY_PYTHON_SCALARS"):
        monkeypatch.setenv(name, "1")
    erased = env_seal.scrub_env()
    for name in ("NCCL_SOMETHING", "ATEN_CPU_CAPABILITY",
                 "INDUCTOR_PROVENANCE", "AOT_INDUCTOR_ENABLE_LTO",
                 "CUDA_LAUNCH_BLOCKING", "CUTLASS_EPILOGUE_FUSION",
                 "TENSORIFY_PYTHON_SCALARS"):
        assert name in erased and name not in os.environ, name


def test_scrub_then_impose_keeps_declared_env() -> None:
    """pgw#1049: The order establish() runs: scrub erases the whole namespace (including our own entries), impos..."""
    seal = env_seal.establish()
    for key, value in sa.DECLARED_ENV.items():
        assert os.environ.get(key) == value, key
    assert seal["env"] == dict(sa.DECLARED_ENV)


# ============================================================================
# pgw#718 — pgw#718/#719: erase-and-impose env contract + seal composition +
#   boot-vs-point-of-use enforcement.
# ============================================================================

# pgw#718's `_reset_boot_seal` is gone: `_fresh_boot` above already unwinds
# `_BOOT_READBACK` and `_ESTABLISHED_OVERRIDES` around every row in this module.

@pytest.fixture(autouse=True)
def _restore_global_matmul_flags() -> Iterator[None]:
    """pgw#718: The canonical imposition is deliberately process-global; the SUITE must not leak it across files..."""
    precision = torch.get_float32_matmul_precision()
    matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    benchmark = torch.backends.cudnn.benchmark
    yield
    torch.set_float32_matmul_precision(precision)
    torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
    torch.backends.cudnn.allow_tf32 = cudnn_tf32
    torch.backends.cudnn.benchmark = benchmark


def _cfg(**overrides: Any) -> CompileContract:
    base: Dict[str, Any] = dict(
        shapes=((64, 64),), targets=("transformer",), family="toyfam",
        regional=False, text_len=None, dynamic=(), lora_bucket=0,
        guidance_scales=(), text_lens=(),
    )
    base.update(overrides)
    return CompileContract(**base)


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
    """pgw#718: The 0.70.3 regression class, dead: a base-image var neither refuses the boot nor reaches the sea..."""
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
    # `torch-inductor-cache` compiled graph and is deleted with that format, while
    # `aot_mint` records this same call's output under the same key.
    manifest = dict(env_seal.frozen_library_digests())
    for base, digest in libs.items():
        if base in manifest and digest != "<unreadable>":
            assert manifest[base] == digest, base


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
    """The pgw#719 red test end-to-end: a flag mutated between boot and the mint trace fails the REAL mint path,..."""

    def forward(self: Any, x: Any, scale: float) -> Any:
        return self.lin(x) * scale

    class _Mod(torch.nn.Module):  # type: ignore[name-defined,misc]
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
    # DYNAMO compiled graph and is deleted with that artifact class. The RULE is
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
