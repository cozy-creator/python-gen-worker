"""pgw#747: an auxiliary bare-typed slot must not inherit the FUNCTION's family.

Discovery stamped EVERY slot with the function's architecture family, including
a slot the endpoint declared as a bare type with no ref. A frame interpolator
and an upscaler are family-agnostic by construction — they consume decoded RGB
frames — so the hub's th#586 gate demanded they classify as `wan-2.2-i2v-a14b`,
could not, and failed the whole manifest closed. Both wan-2.2's `fps` preset and
ltx-video-2.3's `resolution` preset were undeployable on exactly this.

Red-verified against the pre-fix registry: `interpolator` comes out `"example"`.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest


@pytest.fixture()
def tmp_pkg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.syspath_prepend(str(tmp_path))
    return tmp_path


def _write(pkg: Path, main_src: str) -> None:
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "main.py").write_text(textwrap.dedent(main_src))


_TWO_SLOT = """
    import msgspec
    from gen_worker import HF, RequestContext, Resources, Slot, endpoint
    from _example_family import ExampleDefaults

    class In_(msgspec.Struct):
        prompt: str = ""

    class Out_(msgspec.Struct):
        y: str

    class Interpolator:  # family-agnostic: it only sees decoded frames
        pass

    @endpoint(
        models={
            "pipeline": Slot(
                object,
                default_checkpoint=HF("example/base"),
            ),
            # Bare type, no ref: the endpoint declares WHAT loads it and
            # nothing about architecture.
            "interpolator": Slot(Interpolator),
        },
        resources=Resources(gpu=True),
    )
    class Gen:
        def setup(self, pipeline: object, interpolator: Interpolator) -> None: ...
        def generate(self, ctx: RequestContext[ExampleDefaults], data: In_) -> Out_:
            return Out_(y="ok")
"""


def _function(tmp_pkg: Path, name: str, src: str) -> dict:
    from gen_worker.discovery.discover import discover_functions

    _write(tmp_pkg / name, src)
    (fn,) = discover_functions(tmp_pkg, main_module=f"{name}.main")
    return fn


def _slots(tmp_pkg: Path, name: str, src: str) -> dict[str, dict]:
    fn = _function(tmp_pkg, name, src)
    return {s["name"]: s for s in fn["slots"]}


def test_bare_typed_auxiliary_slot_emits_no_family(tmp_pkg: Path) -> None:
    """The pipeline slot keeps the function's family; the auxiliary one has
    none to keep. An ABSENT key is the contract — the hub's gate no-ops on it,
    which is why nothing is needed hub-side."""
    slots = _slots(tmp_pkg, "ep_pgw747_two", _TWO_SLOT)

    assert slots["pipeline"]["family"] == "example"
    assert "family" not in slots["interpolator"], slots["interpolator"]


def test_auxiliary_slot_that_names_a_ref_still_inherits(tmp_pkg: Path) -> None:
    """Unchanged for a slot that DID opt in: naming a catalog ref is an
    architecture claim, and the gate should keep policing overlays on it."""
    slots = _slots(
        tmp_pkg, "ep_pgw747_ref",
        _TWO_SLOT.replace(
            '"interpolator": Slot(Interpolator),',
            '"interpolator": Slot(Interpolator, '
            'default_checkpoint=HF("example/aux")),'),
    )

    assert slots["pipeline"]["family"] == "example"
    assert slots["interpolator"]["family"] == "example"


def test_explicit_family_keeps_non_root_defaultless_model_governed(
    tmp_pkg: Path,
) -> None:
    """pgw#766/th#1566: a deploy-bound non-root model slot has no ref by
    design, so its explicit family must win over pgw#747's auxiliary-slot
    inference."""
    slots = _slots(
        tmp_pkg, "ep_pgw766_explicit_model",
        _TWO_SLOT.replace(
            '"interpolator": Slot(Interpolator),',
            '"interpolator": Slot(Interpolator, family="example"),'),
    )

    assert slots["interpolator"]["family"] == "example"


def test_explicit_empty_family_keeps_ref_bearing_auxiliary_agnostic(
    tmp_pkg: Path,
) -> None:
    """Explicit agnosticism also wins: a shared auxiliary may name its
    bootstrap mirror without claiming the handler's architecture family."""
    fn = _function(
        tmp_pkg, "ep_pgw766_explicit_aux",
        _TWO_SLOT.replace(
            '"interpolator": Slot(Interpolator),',
            '"interpolator": Slot(Interpolator, family="", '
            'default_checkpoint=HF("example/aux")),'),
    )
    slots = {s["name"]: s for s in fn["slots"]}

    assert "family" not in slots["interpolator"], slots["interpolator"]
    assert "family" not in fn["bindings"]["interpolator"]


def test_single_defaultless_slot_keeps_its_family(tmp_pkg: Path) -> None:
    """The guard the naive "no ref => no family" rule would have broken.

    A deliberately DEFAULTLESS root slot is a real model slot bound at deploy
    time through `?bindings=` (krea-2 ships exactly this shape). Emptying its
    family would silently drop the LoRA-overlay policing pgw#523 added, so
    only a NON-ROOT ref-less slot is treated as family-agnostic.
    """
    slots = _slots(tmp_pkg, "ep_pgw747_defaultless", """
        import msgspec
        from gen_worker import RequestContext, Resources, Slot, endpoint
        from _example_family import ExampleDefaults

        class In_(msgspec.Struct):
            prompt: str = ""

        class Out_(msgspec.Struct):
            y: str

        @endpoint(models={"pipeline": Slot(object)},
                  resources=Resources(gpu=True))
        class Gen:
            def setup(self, pipeline: object) -> None: ...
            def generate(self, ctx: RequestContext[ExampleDefaults],
                         data: In_) -> Out_:
                return Out_(y="ok")
    """)

    assert slots["pipeline"]["family"] == "example"
