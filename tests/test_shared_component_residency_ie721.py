"""ie#721 — a content-shared component may not be moved to the host.

THE DEFECT, AND WHY IT ARRIVED. A module reaching a pipeline through
`provision.load_slot`'s `components=` was loaded ONCE and aliased into every
lane sharing its content address (gw#479). An offload rung that moves it to the
host strands every co-resident consumer on the device, and the failure is a
fatal `mat1 is on cuda:0, mat2 on cpu` in the MIDDLE of a generate — measured on
krea-2, qwen-image, z-image and hidream-o1-image (ie#480 finding 12).

Until th#1867 nothing hit that path because five endpoints declared
`Resources.strict_vram`, refusing every CPU-touching rung. That declaration
deserved to go — it was an author's card-size claim in softer words — but it was
ALSO the only enforcer of an invariant the codebase still states in prose:
`provision.load_slot`'s docstring says the joint multi-lane fit decision exists
so a lane never "starves a sibling lane into an offload placement the
shared-component invariant refuses." The refusal it names had been deleted.

So this suite is the invariant's new enforcer, and it takes its input from a
MEASURED fact — this object was injected as a shared component — rather than
from an author's word. Nothing here declares how big a card must be.

TWO INDEPENDENTLY SUFFICIENT TERMS, PROVEN SEPARATELY. `unhookable_components`
is the UNION of dtype-fragility (gw#441/gw#469) and content-sharing
(gw#479/ie#721). A proof that severs only one says nothing about the other, so
every arm below is run in three states: shared-and-not-fragile,
fragile-and-not-shared, and both.
"""

from __future__ import annotations

from typing import Any

import pytest

from gen_worker.models import memory as mem


class _Mod:
    """A component the offload rungs can see and move."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.moved_to: list[str] = []

    def parameters(self):  # marked only on things that can be hooked
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
    # diffusers sets this on the pipeline; declared so the exclusion assertions
    # below type-check as the real attribute rather than as a dynamic one.
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


# ---------------------------------------------------------------------------
# The mark is a fact about what happened, not a declaration.
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# THE UNION — each term severed independently.
# ---------------------------------------------------------------------------

def test_shared_but_NOT_dtype_fragile_is_still_unhookable() -> None:
    """Arm 1: sever dtype-fragility. Sharing alone must be sufficient."""
    pipe = _Pipe(fragile_vae=False)
    mem.mark_shared_components({"text_encoder": pipe.text_encoder})
    assert mem._dtype_fragile_vae(pipe) is None, "fragility is severed in this arm"
    assert mem.unhookable_components(pipe) == ["text_encoder"]


def test_dtype_fragile_but_NOT_shared_is_still_unhookable() -> None:
    """Arm 2: sever sharing. Fragility alone must be sufficient — this is the
    gw#441/gw#469 behaviour ie#721 must not regress."""
    pipe = _Pipe(fragile_vae=True)
    assert mem.shared_component_names(pipe) == [], "sharing is severed in this arm"
    assert mem.unhookable_components(pipe) == ["vae"]


def test_both_terms_together_produce_the_union_without_duplicates() -> None:
    pipe = _Pipe(fragile_vae=True)
    mem.mark_shared_components({"text_encoder": pipe.text_encoder, "vae": pipe.vae})
    assert mem.unhookable_components(pipe) == ["text_encoder", "vae"]


# ---------------------------------------------------------------------------
# The hook-based rungs (model_offload / sequential) honour the exclusion.
# ---------------------------------------------------------------------------

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
    # The other term is reported independently, never folded into this one.
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


# ---------------------------------------------------------------------------
# The group rung honours the SAME union, and puts them back on the device.
# ---------------------------------------------------------------------------

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
    # Excluding a module from the group hooks does NOT place it — the caller
    # owes it the device, exactly as the fragile-VAE path always has.
    assert pipe.text_encoder.moved_to == ["cuda"], (
        "excluded-from-hooks and left-on-the-host is the same crash by another route"
    )
    if fragile:
        assert "vae" in excluded and pipe.vae.moved_to == ["cuda"]


def test_group_rung_untouched_when_nothing_is_shared_or_fragile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The gw#441 baseline: no exclusions means no `exclude_modules` kwarg at
    all, so this change cannot alter placement for a pipeline it does not
    concern."""
    pipe = _GroupPipe(fragile_vae=False)
    _force_cuda(monkeypatch)
    applied: dict = {}
    assert mem._apply_group_offload(pipe, applied, offload_to_disk_path=None) is True
    assert "exclude_modules" not in pipe.group_kwargs
    assert applied.get("shared_resident") is None
    assert pipe.text_encoder.moved_to == []


class _FakeTorch:
    """The minimum `_apply_group_offload` reads: `cuda.is_available()` and
    `device()`. Nothing here allocates, computes, or touches a GPU."""

    class cuda:  # mirrors torch's namespace, hence the lowercase
        @staticmethod
        def is_available() -> bool:
            return True

    @staticmethod
    def device(name: str) -> str:
        return name


def _force_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the group rung believe it has a card, WITHOUT torch and WITHOUT a GPU.

    `pytest.importorskip("torch")` — this repo's usual idiom — would SKIP these
    arms in the torch-free lane, and a skipped arm is the false-clean this whole
    issue's neighbourhood keeps producing: the guard would report green having
    proven nothing about the group rung. Injecting a fake makes them run
    everywhere, deterministically, on any machine. `_apply_group_offload`
    imports torch INSIDE the function, so a `sys.modules` entry is all it takes.
    """
    import sys

    monkeypatch.setitem(sys.modules, "torch", _FakeTorch)
