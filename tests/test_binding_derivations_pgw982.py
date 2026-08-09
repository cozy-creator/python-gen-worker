"""pgw#982: the rotation preload driver no longer reaches through the
executor's underscores for the executor's own binding derivations.

`preload` staged what `executor._binding_wire_refs` / `_component_overrides`
said a binding resolved to, through two FUNCTION-BODY imports that existed only
to dodge the `executor -> preload` cycle. That is two consumers of one
derivation with no contract holding them together: change how the executor
derives overrides and the rotation driver silently stages a different
checkpoint, with nothing failing until the preloaded one is wrong.

The derivations describe a BINDING, so they live on `api.binding` beside the
field they read, and both consumers name the same public function. The cycle
that forced the deferral is gone with them.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

from gen_worker import executor as executor_mod
from gen_worker import preload as preload_mod
from gen_worker.api.binding import (
    HF,
    Hub,
    ModelRef,
    binding_wire_refs,
    component_overrides,
    wire_ref,
)


def _with_overrides(*pairs: tuple[str, str]) -> ModelRef:
    """A tensorhub binding carrying component substitutions — the shape the
    executor mints at dispatch (``structs.replace(binding, component_overrides=…)``);
    the ``Hub()`` factory has no keyword for it."""
    return ModelRef(
        source="tensorhub", path="acme/qwen", component_overrides=pairs)


def _deferred_imports(module) -> list[tuple[str, tuple[str, ...]]]:
    """Every `from X import ...` that runs inside a function body."""
    tree = ast.parse(Path(inspect.getsourcefile(module)).read_text())
    out: list[tuple[str, tuple[str, ...]]] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for inner in ast.walk(node):
            if isinstance(inner, ast.ImportFrom) and inner.module:
                out.append((
                    "." * inner.level + inner.module,
                    tuple(a.name for a in inner.names),
                ))
    return out


def test_preload_never_reaches_into_the_executor_at_runtime():
    reaches = [
        (mod, names) for mod, names in _deferred_imports(preload_mod)
        if mod.endswith("executor")
    ]
    assert reaches == [], (
        "preload still imports from executor inside a function body: "
        f"{reaches!r}"
    )


def test_the_derivations_are_public_and_have_one_definition():
    # The private executor copies are gone, not merely wrapped.
    assert not hasattr(executor_mod, "_binding_wire_refs")
    assert not hasattr(executor_mod, "_component_overrides")
    # And both consumers resolve to the SAME object, which is what makes a
    # change to one derivation reach the other.
    assert executor_mod.component_overrides is component_overrides
    assert executor_mod.binding_wire_refs is binding_wire_refs
    assert preload_mod.component_overrides is component_overrides
    assert preload_mod.binding_wire_refs is binding_wire_refs


def test_component_overrides_reads_the_bindings_own_normalized_field():
    b = _with_overrides(("vae", "acme/vae"), ("denoiser", "acme/dit"))
    # ModelRef sorts and cleans the field itself; the accessor does not
    # re-derive it.
    assert component_overrides(b) == b.component_overrides
    assert component_overrides(b) == (("denoiser", "acme/dit"), ("vae", "acme/vae"))
    assert component_overrides(Hub("acme/qwen")) == ()


def test_binding_wire_refs_is_the_base_ref_plus_every_override():
    b = _with_overrides(("vae", "acme/vae"), ("denoiser", "acme/dit"))
    assert binding_wire_refs(b) == [wire_ref(b), "acme/dit", "acme/vae"]
    # A source with no override axis materializes exactly one ref.
    hf = HF("black-forest-labs/FLUX.1-dev")
    assert binding_wire_refs(hf) == [wire_ref(hf)]


def test_a_stand_in_binding_is_answered_rather_than_raising():
    """Specs assembled from the wire and test doubles reach these too."""

    class _Stub:
        source = "tensorhub"
        path = "acme/qwen"
        tag = "prod"
        flavor = ""

    assert component_overrides(_Stub()) == ()


def test_import_binding_module_alone_pulls_in_neither_executor_nor_preload():
    """The derivations must be reachable without the compile/serve stack —
    that is the whole point of moving them off the executor."""
    import subprocess
    import sys

    proc = subprocess.run(
        [sys.executable, "-c",
         "import sys; import gen_worker.api.binding as b; "
         "assert b.binding_wire_refs and b.component_overrides; "
         "print(int('gen_worker.executor' in sys.modules), "
         "int('gen_worker.preload' in sys.modules))"],
        capture_output=True, text=True, check=True,
    )
    assert proc.stdout.split() == ["0", "0"], proc.stdout


def test_the_public_names_are_exported():
    import gen_worker.api.binding as binding_mod

    assert "binding_wire_refs" in binding_mod.__all__
    assert "component_overrides" in binding_mod.__all__
