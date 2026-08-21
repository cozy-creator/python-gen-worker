"""A worker with NO torch boots, and seals the absence as a fact."""

from __future__ import annotations

import importlib.abc
import sys

import pytest

from gen_worker import env_seal, guard_closure, host_isa, torch_capability


class _BlockTorch(importlib.abc.MetaPathFinder):

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


def test_torchless_worker_completes_the_boot_seal(torchless, monkeypatch):
    monkeypatch.setattr(env_seal, "_BOOT_READBACK", None, raising=False)
    seal = env_seal.establish()

    assert seal["config"]["torch"] == torch_capability.ABSENT
    assert seal["inductor"] == {}
    assert seal["posture"] == {"torch": torch_capability.ABSENT}
    assert len(env_seal.seal_digest(seal)) == 16
    env_seal.assert_seal_unchanged("pgw788")


def test_torchless_config_carries_no_torch_flag_it_cannot_read(torchless):
    cfg = env_seal.effective_config()
    for flag in ("float32_matmul_precision", "cuda_matmul_allow_tf32",
                 "cudnn_allow_tf32", "cudnn_benchmark"):
        assert flag not in cfg, f"{flag} cannot be read without torch"
    assert "python_hash_seed" in cfg and "hash_randomization" in cfg


def test_torchless_isa_clamp_and_posture_no_op(torchless):
    assert host_isa.impose() == {}
    assert host_isa.effective() == {}
    assert guard_closure.establish_posture() == {
        "torch": torch_capability.ABSENT}
    assert guard_closure.posture_snapshot() == {
        "torch": torch_capability.ABSENT}


def test_torchless_posture_assertion_agrees_with_its_own_seal(torchless):
    sealed = guard_closure.establish_posture()
    assert sealed == guard_closure.posture_snapshot()


def test_declared_knob_on_a_torchless_worker_refuses_by_name(torchless):
    """Every canonical knob is a torch flag."""
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
    """The verdict must be re-derived, or one import-blocking test poisons the whole session (and, worse, a real worker could cache a stale answer)."""
    assert torch_capability.torch_or_none() is None
    assert torch_capability.present() is False


def test_boot_modules_import_without_torch(torchless):
    """entrypoint is what calls establish(); it must be importable torchless."""
    import gen_worker

    decorator = gen_worker.entrypoint
    for name in ("gen_worker.env_seal", "gen_worker.host_isa",
                 "gen_worker.guard_closure", "gen_worker.entrypoint"):
        __import__(name)
    gen_worker.entrypoint = decorator


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
