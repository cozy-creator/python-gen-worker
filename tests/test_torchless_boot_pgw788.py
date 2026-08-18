"""A worker with NO torch boots, and seals the absence as a fact.

`env_seal.establish()` is called by `entrypoint.py` on every boot regardless of
`accelerator`, and a bare torch import anywhere under it (`establish_config` ->
`host_isa.impose()` -> `guard_closure.establish_posture()`) kills every
torchless CPU endpoint at `phase=env_seal`, before it advertises a single
function.

torch is import-BLOCKED with a `sys.meta_path` finder rather than uninstalled:
same failure at the same call sites, no second venv. The finder is installed and
removed per test, and `torch_capability` deliberately does not memoize its
verdict, so nothing leaks into the rest of the session.

The other half of the guarantee is asserted here too: with torch PRESENT the
seal must be byte-identical to the pre-fix seal, because the seal is a compiled graph-key
axis and a changed shape would strand every published compiled graph.
"""

from __future__ import annotations

import importlib.abc
import sys

import pytest

from gen_worker import env_seal, guard_closure, host_isa, torch_capability


class _BlockTorch(importlib.abc.MetaPathFinder):
    """Make `import torch` (and any submodule) raise ImportError."""

    def find_spec(self, fullname, path=None, target=None):  # type: ignore[no-untyped-def]
        if fullname == "torch" or fullname.startswith("torch."):
            raise ImportError(f"pgw#788 test: {fullname} is not installed")
        return None


@pytest.fixture()
def torchless(monkeypatch: pytest.MonkeyPatch):
    """A process that behaves exactly like a torchless image at boot."""
    finder = _BlockTorch()
    saved = dict(sys.modules)
    for name in [m for m in sys.modules if m == "torch" or m.startswith("torch.")]:
        monkeypatch.delitem(sys.modules, name, raising=False)
    sys.meta_path.insert(0, finder)
    try:
        with pytest.raises(ImportError):
            __import__("torch")
        yield
    finally:
        sys.meta_path.remove(finder)
        sys.modules.update(saved)


# ---------------------------------------------------------------------------
# RED before the fix: every one of these raised ImportError.
# ---------------------------------------------------------------------------


def test_torchless_worker_completes_the_boot_seal(torchless, monkeypatch):
    monkeypatch.setattr(env_seal, "_BOOT_READBACK", None, raising=False)
    seal = env_seal.establish()

    assert seal["config"]["torch"] == torch_capability.ABSENT
    # torchless declares NO codegen clamp — an empty declared
    # inductor block, same as a non-x86 host (absence rides `config`).
    assert seal["inductor"] == {}
    assert seal["posture"] == {"torch": torch_capability.ABSENT}
    # A seal that states the absence is still a KEY: it digests, and the
    # boot-vs-point-of-use check agrees with itself.
    assert len(env_seal.seal_digest(seal)) == 16
    env_seal.assert_seal_unchanged("pgw788")


def test_torchless_config_carries_no_torch_flag_it_cannot_read(torchless):
    cfg = env_seal.effective_config()
    for flag in ("float32_matmul_precision", "cuda_matmul_allow_tf32",
                 "cudnn_allow_tf32", "cudnn_benchmark"):
        assert flag not in cfg, f"{flag} cannot be read without torch"
    # The interpreter-level facts are torch-free and must survive.
    assert "python_hash_seed" in cfg and "hash_randomization" in cfg


def test_torchless_isa_clamp_and_posture_no_op(torchless):
    assert host_isa.impose() == {}
    assert host_isa.effective() == {}
    assert guard_closure.establish_posture() == {
        "torch": torch_capability.ABSENT}
    assert guard_closure.posture_snapshot() == {
        "torch": torch_capability.ABSENT}


def test_torchless_posture_assertion_agrees_with_its_own_seal(torchless):
    """pgw#1181: `assert_posture` is deleted — it compared the live process
    against a COMPILED GRAPH's recorded posture seal, and the only writer of that seal
    was `closure_manifest`, which went with the `torch-inductor-cache` format.
    The property this row is named for survives without it: what a torchless
    worker establishes is exactly what it later observes, so a seal taken at
    boot and a snapshot taken afterwards agree fact for fact."""
    sealed = guard_closure.establish_posture()
    assert sealed == guard_closure.posture_snapshot()


def test_declared_knob_on_a_torchless_worker_refuses_by_name(torchless):
    """Every canonical knob is a torch flag. Silently ignoring one would fork
    compiled graph identity, so a torchless worker that declares one refuses."""
    from gen_worker import settings_authority as sa

    with pytest.raises(sa.SettingsImpositionError) as exc:
        sa.impose_torch({"cudnn_benchmark": "True"})
    assert "TORCHLESS" in str(exc.value)
    assert "cudnn_benchmark" in str(exc.value)


def test_unknown_knob_still_refuses_without_torch(torchless):
    """The knob-name contract is torch-free and must not be skipped."""
    from gen_worker import settings_authority as sa

    with pytest.raises(sa.SettingsImpositionError) as exc:
        sa.impose_torch({"not_a_knob": "1"})
    assert "not_a_knob" in str(exc.value)


def test_capability_probe_is_not_memoized_across_the_boundary(torchless):
    """The verdict must be re-derived, or one import-blocking test poisons the
    whole session (and, worse, a real worker could cache a stale answer)."""
    assert torch_capability.torch_or_none() is None
    assert torch_capability.present() is False


def test_boot_modules_import_without_torch(torchless):
    """entrypoint is what calls establish(); it must be importable torchless."""
    for name in ("gen_worker.env_seal", "gen_worker.host_isa",
                 "gen_worker.guard_closure", "gen_worker.worker_main"):
        __import__(name)


# ---------------------------------------------------------------------------
# The other direction: torch present is UNTOUCHED (compiled graph identity is a key).
# ---------------------------------------------------------------------------


def test_torch_present_seal_shape_is_unchanged():
    torch = torch_capability.torch_or_none()
    if torch is None:
        pytest.skip("this leg asserts the torch-present shape")
    cfg = env_seal.effective_config()
    assert "torch" not in cfg, (
        "a torch-present worker must seal exactly the pre-pgw#788 config keys — "
        "an extra key rewrites every compiled graph key in the fleet")
    for flag in ("float32_matmul_precision", "cuda_matmul_allow_tf32",
                 "cudnn_allow_tf32", "cudnn_benchmark"):
        assert flag in cfg
    assert env_seal.inductor_config_digest() != torch_capability.ABSENT
    assert guard_closure.posture_snapshot() == guard_closure.CANONICAL_POSTURE
