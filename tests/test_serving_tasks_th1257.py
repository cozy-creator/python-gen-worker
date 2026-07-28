"""th#1257: ``@worker_function(tasks=...)`` — the SERVING-TASK declaration
that the hub's lane-verdict store keys its human quality sign-off on.

A quantized transformer that holds up on text-to-image can degrade badly on
an image-edit path that must preserve an input image, so an fp8 approval is
scoped per task. flux.2-klein-9b serves both off ONE set of weights, which is
why the task cannot be inferred from the model and has to be declared here.

Real discovery/registry code, no mocking (test_objectives_pgw654 idiom).
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


def test_tasks_round_trip_into_manifest(tmp_pkg: Path) -> None:
    """The klein-9b shape: one class, one set of weights, generate AND edit."""
    from gen_worker.discovery.discover import discover_functions

    pkg = tmp_pkg / "ep_tasks"
    _write(pkg, """
        import msgspec
        from gen_worker import RequestContext, endpoint, worker_function

        class In_(msgspec.Struct):
            prompt: str = ""

        class Out_(msgspec.Struct):
            y: str

        @endpoint
        class Klein:
            @worker_function(tasks=("text_to_image",), distilled=False)
            def generate(self, ctx: RequestContext, data: In_) -> Out_:
                return Out_(y="t2i")

            @worker_function(tasks=("image_edit",), distilled=False)
            def edit(self, ctx: RequestContext, data: In_) -> Out_:
                return Out_(y="edit")

            @worker_function(tasks=("text_to_image",), distilled=True)
            def generate_turbo(self, ctx: RequestContext, data: In_) -> Out_:
                return Out_(y="t2i-turbo")

            def utility(self, ctx: RequestContext, data: In_) -> Out_:
                return Out_(y="undeclared")
    """)

    fns = {f["name"]: f for f in discover_functions(tmp_pkg, main_module="ep_tasks.main")}
    assert fns["generate"]["tasks"] == ["text_to_image"]
    assert fns["edit"]["tasks"] == ["image_edit"]
    # The turbo sibling is the SAME task on a different variant — the hub
    # scopes those to separate verdict rows via the distilled fact.
    assert fns["generate-turbo"]["tasks"] == ["text_to_image"]
    assert fns["generate-turbo"]["distilled"] is True
    assert fns["generate"]["distilled"] is False
    # Undeclared: the key is omitted, and the hub then resolves every quant
    # lane unmeasured rather than guessing which task was approved.
    assert "tasks" not in fns["utility"]


def test_merged_handler_declares_several_tasks(tmp_pkg: Path) -> None:
    """One input-routing generate() serving t2i and edit must declare both;
    the hub requires EVERY declared task to be approved for the lane."""
    from gen_worker.discovery.discover import discover_functions

    pkg = tmp_pkg / "ep_merged"
    _write(pkg, """
        import msgspec
        from gen_worker import RequestContext, endpoint, worker_function

        class In_(msgspec.Struct):
            prompt: str = ""

        class Out_(msgspec.Struct):
            y: str

        @endpoint
        @worker_function(tasks=("text_to_image", "image_edit"), objectives=("flow",))
        def generate(ctx: RequestContext, data: In_) -> Out_:
            return Out_(y="routed")
    """)

    fns = {f["name"]: f for f in discover_functions(tmp_pkg, main_module="ep_merged.main")}
    assert fns["generate"]["tasks"] == ["text_to_image", "image_edit"]
    assert fns["generate"]["objectives"] == ["flow"]


def test_task_spellings_and_validation() -> None:
    from gen_worker.api.decorators import WF_ATTR, worker_function

    @worker_function(tasks=("image-edit",))
    def hyphenated(self, ctx, data): ...

    # Separator folding only — the declaration site stays canonical so two
    # spellings can never become two verdict rows.
    assert getattr(hyphenated, WF_ATTR).tasks == ("image_edit",)

    with pytest.raises(ValueError, match="unknown task"):
        @worker_function(tasks=("t2i",))
        def abbreviated(self, ctx, data): ...

    with pytest.raises(ValueError, match="must not be empty"):
        @worker_function(tasks=())
        def empty(self, ctx, data): ...

    with pytest.raises(ValueError, match="repeats a value"):
        @worker_function(tasks=("image_edit", "image_edit"))
        def dupe(self, ctx, data): ...
