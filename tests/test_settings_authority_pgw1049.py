"""pgw#1049: the single settings authority — seal from DECLARATION."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Iterator

import pytest

torch = pytest.importorskip("torch")

from gen_worker import env_seal
from gen_worker import settings_authority as sa

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _fresh_boot(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(env_seal, "_ESTABLISHED_OVERRIDES", None)
    yield
    env_seal._ESTABLISHED_OVERRIDES = None


# ---------------------------------------------------------------------------
# The declaration IS the seal
# ---------------------------------------------------------------------------


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
    try:
        knobbed = env_seal.establish(overrides={"cudnn_benchmark": "True"})
        assert knobbed["config"]["cudnn_benchmark"] == "True"
        assert env_seal.seal_digest(knobbed) != base
    finally:
        env_seal.establish()


def test_dynamo_shape_posture_is_process_wide() -> None:
    """The v2 dynamo posture moved into the authority (it was a second
    writer in compile_cache): imposed at the default layer, so a FRESH
    thread — where warm compiles actually run — reads the declared values."""
    import torch._dynamo

    sa.impose_dynamo()
    got = sa.read_in_fresh_thread(
        lambda: (bool(torch._dynamo.config.automatic_dynamic_shapes),
                 bool(torch._dynamo.config.assume_static_by_default)))
    assert got == (False, True)
    assert {
        name: str(getattr(torch._dynamo.config, name))
        for name in sa.DECLARED_DYNAMO
    } == sa.DECLARED_DYNAMO


# ---------------------------------------------------------------------------
# PYTHONHASHSEED: imposed, verified fail-closed (HUMAN_MUST_DO executed)
# ---------------------------------------------------------------------------


def _run(code: str, *, seed_env: dict) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k != "PYTHONHASHSEED"}
    env.update(seed_env)
    env["PYTHONPATH"] = str(REPO / "src")
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        env=env, capture_output=True, text=True, timeout=120)


def test_establish_refuses_undeclared_hash_seed() -> None:
    """Fail-closed: a process whose interpreter booted outside the declared
    env cannot seal an identity that claims it."""
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
    """The sanctioned imposition: a process launched without the declared
    seed re-execs itself (sys.orig_argv) and comes back with hash
    randomization OFF — the entrypoint/conftest path."""
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
    """impose_process_env is what every spawned child inherits — the mint
    child and AOT entry children boot with the seed already at interpreter
    start, so their establish() verifies instead of re-exec'ing."""
    sa.impose_process_env()
    assert os.environ["PYTHONHASHSEED"] == "0"
    child = subprocess.run(
        [sys.executable, "-c",
         "import sys; print(sys.flags.hash_randomization)"],
        capture_output=True, text=True, timeout=60)
    assert child.stdout.strip() == "0"


def test_revert_is_one_declaration_entry() -> None:
    """The HUMAN_MUST_DO record promises the imposition is trivially
    revertible: with the entry removed, ensure/verify have nothing to check
    and impose nothing — no second site knows about the seed."""
    saved = dict(sa.DECLARED_ENV)
    try:
        del sa.DECLARED_ENV["PYTHONHASHSEED"]
        assert sa._interpreter_env_diffs() == []
        sa.verify_interpreter_env()  # no refusal without the entry
    finally:
        sa.DECLARED_ENV.clear()
        sa.DECLARED_ENV.update(saved)


# ---------------------------------------------------------------------------
# Scrub extensions (census-driven)
# ---------------------------------------------------------------------------


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
    """The order establish() runs: scrub erases the whole namespace
    (including our own entries), impose puts the DECLARED values back — so
    the allocator conf is REAL (the old entrypoint setdefault died at the
    scrub and expandable_segments was silently off, found by pgw#1049)."""
    seal = env_seal.establish()
    for key, value in sa.DECLARED_ENV.items():
        assert os.environ.get(key) == value, key
    assert seal["env"] == dict(sa.DECLARED_ENV)
