"""pgw#1370: the lanes-only @endpoint surface (code + lanes, nothing else)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from gen_worker.api.decorators import ATTR, endpoint


def _lane(name: str = "bf16", compile: tuple = ("unet",)) -> SimpleNamespace:
    return SimpleNamespace(
        name=name, compile=compile, contract="plain.bf16@1", dtype=None
    )


class _In:
    pass


def test_lanes_class_records_the_declaration() -> None:
    @endpoint(lanes=(_lane(),))
    class Ep:
        def setup(self, ctx) -> None: ...
        def generate(self, ctx, payload: _In) -> _In: ...

    decl = getattr(Ep, ATTR)
    assert len(decl.lanes) == 1 and decl.lanes[0].name == "bf16"
    assert [attr for attr, _ in Ep.__gen_worker_handlers__] == ["generate"]


def test_lanes_do_not_compose_with_the_catalog_surface() -> None:
    from gen_worker import HF

    with pytest.raises(ValueError, match="does not compose"):
        @endpoint(lanes=(_lane(),), model=HF("org/repo"))
        class Ep:
            def setup(self, ctx) -> None: ...
            def generate(self, ctx, payload: _In) -> _In: ...


def test_lanes_require_a_class() -> None:
    with pytest.raises(TypeError, match="class"):
        @endpoint(lanes=(_lane(),))
        def handler(ctx, payload: _In) -> _In: ...


def test_duplicate_lane_names_refuse() -> None:
    with pytest.raises(ValueError, match="unique"):
        @endpoint(lanes=(_lane("a"), _lane("a")))
        class Ep:
            def generate(self, ctx, payload: _In) -> _In: ...


def test_a_lane_without_a_contract_refuses() -> None:
    bare = SimpleNamespace(name="bf16", compile=("unet",), contract="")
    with pytest.raises(ValueError, match="contract"):
        @endpoint(lanes=(bare,))
        class Ep:
            def generate(self, ctx, payload: _In) -> _In: ...
