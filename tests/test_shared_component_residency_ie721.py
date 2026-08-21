from __future__ import annotations

from typing import Any

import pytest

from gen_worker.models import memory as mem


class _Mod:

    def __init__(self, name: str) -> None:
        self.name = name
        self.moved_to: list[str] = []

    def parameters(self):
        return iter(())

    def to(self, device: str) -> _Mod:
        self.moved_to.append(device)
        return self


class _Cfg:
    def __init__(self, force_upcast: bool) -> None:
        self.force_upcast = force_upcast


class _VAE(_Mod):
    def __init__(self, *, force_upcast: bool) -> None:
        super().__init__("vae")
        self.config = _Cfg(force_upcast)


class _Pipe:
    _exclude_from_cpu_offload: list[str]

    def __init__(self, *, fragile_vae: bool) -> None:
        self.text_encoder = _Mod("text_encoder")
        self.transformer = _Mod("transformer")
        self.vae = _VAE(force_upcast=fragile_vae)
        self.components = {
            "text_encoder": self.text_encoder,
            "transformer": self.transformer,
            "vae": self.vae,
        }
        self.moved_to: list[str] = []

    def to(self, device: str) -> _Pipe:
        self.moved_to.append(device)
        return self


def test_marking_only_touches_things_an_offload_rung_could_hook() -> None:
    mod = _Mod("text_encoder")
    n = mem.mark_shared_components(
        {"text_encoder": mod, "path": "/tmp/x", "nothing": None},
    )
    assert n == 1, "a path string and a None are not hookable modules"
    assert getattr(mod, mem.SHARED_COMPONENT_ATTR) is True


def test_an_unmarked_pipeline_reports_no_shared_components() -> None:
    assert mem.shared_component_names(_Pipe(fragile_vae=False)) == []


def test_shared_names_are_read_off_the_live_objects() -> None:
    pipe = _Pipe(fragile_vae=False)
    mem.mark_shared_components({"text_encoder": pipe.text_encoder})
    assert mem.shared_component_names(pipe) == ["text_encoder"]


def test_shared_but_NOT_dtype_fragile_is_still_unhookable() -> None:
    """Arm 1: sever dtype-fragility."""
    pipe = _Pipe(fragile_vae=False)
    mem.mark_shared_components({"text_encoder": pipe.text_encoder})
    assert mem._dtype_fragile_vae(pipe) is None, "fragility is severed in this arm"
    assert mem.unhookable_components(pipe) == ["text_encoder"]


def test_dtype_fragile_but_NOT_shared_is_still_unhookable() -> None:
    """Arm 2: sever sharing."""
    pipe = _Pipe(fragile_vae=True)
    assert mem.shared_component_names(pipe) == [], "sharing is severed in this arm"
    assert mem.unhookable_components(pipe) == ["vae"]


def test_both_terms_together_produce_the_union_without_duplicates() -> None:
    pipe = _Pipe(fragile_vae=True)
    mem.mark_shared_components({"text_encoder": pipe.text_encoder, "vae": pipe.vae})
    assert mem.unhookable_components(pipe) == ["text_encoder", "vae"]


@pytest.mark.parametrize("fragile", [False, True])
def test_hook_rungs_exclude_shared_components(fragile: bool) -> None:
    pipe = _Pipe(fragile_vae=fragile)
    mem.mark_shared_components({"text_encoder": pipe.text_encoder})
    applied: dict = {}
    mem._pin_unhookable_components(pipe, applied, mem._LOG)

    excl = list(getattr(pipe, "_exclude_from_cpu_offload", []))
    assert "text_encoder" in excl, (
        "a shared component left in the offload hooks is the ie#480 finding-12 "
        "crash: it goes to the host and its co-resident consumers do not"
    )
    assert applied.get("shared_resident") is True
    assert applied.get("vae_resident", False) is fragile
    if fragile:
        assert "vae" in excl


def test_hook_rungs_are_a_no_op_when_nothing_is_unhookable() -> None:
    pipe = _Pipe(fragile_vae=False)
    applied: dict = {}
    mem._pin_unhookable_components(pipe, applied, mem._LOG)
    assert not hasattr(pipe, "_exclude_from_cpu_offload")
    assert applied == {}


def test_an_existing_exclusion_list_is_extended_not_replaced() -> None:
    pipe = _Pipe(fragile_vae=False)
    pipe._exclude_from_cpu_offload = ["safety_checker"]
    mem.mark_shared_components({"text_encoder": pipe.text_encoder})
    mem._pin_unhookable_components(pipe, {}, mem._LOG)
    assert set(pipe._exclude_from_cpu_offload) == {"safety_checker", "text_encoder"}


class _GroupPipe(_Pipe):
    def __init__(self, *, fragile_vae: bool) -> None:
        super().__init__(fragile_vae=fragile_vae)
        self.group_kwargs: dict[str, Any] = {}

    def enable_group_offload(self, **kwargs: Any) -> None:
        self.group_kwargs = kwargs


@pytest.mark.parametrize("fragile", [False, True])
def test_group_rung_excludes_and_then_onloads(
    monkeypatch: pytest.MonkeyPatch, fragile: bool,
) -> None:
    pipe = _GroupPipe(fragile_vae=fragile)
    mem.mark_shared_components({"text_encoder": pipe.text_encoder})
    _force_cuda(monkeypatch)

    applied: dict = {}
    ok = mem._apply_group_offload(pipe, applied, offload_to_disk_path=None)

    assert ok is True
    excluded: list[str] = list(pipe.group_kwargs.get("exclude_modules") or [])
    assert "text_encoder" in excluded, "the group rung must exclude shared modules too"
    assert applied.get("shared_resident") is True
    assert pipe.text_encoder.moved_to == ["cuda"], (
        "excluded-from-hooks and left-on-the-host is the same crash by another route"
    )
    if fragile:
        assert "vae" in excluded and pipe.vae.moved_to == ["cuda"]


def test_group_rung_untouched_when_nothing_is_shared_or_fragile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The gw#441 baseline: no exclusions means no `exclude_modules` kwarg at all, so this change cannot alter placement for a pipeline it does not concern."""
    pipe = _GroupPipe(fragile_vae=False)
    _force_cuda(monkeypatch)
    applied: dict = {}
    assert mem._apply_group_offload(pipe, applied, offload_to_disk_path=None) is True
    assert "exclude_modules" not in pipe.group_kwargs
    assert applied.get("shared_resident") is None
    assert pipe.text_encoder.moved_to == []


class _FakeTorch:

    class cuda:
        @staticmethod
        def is_available() -> bool:
            return True

    @staticmethod
    def device(name: str) -> str:
        return name


def _force_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    monkeypatch.setitem(sys.modules, "torch", _FakeTorch)
