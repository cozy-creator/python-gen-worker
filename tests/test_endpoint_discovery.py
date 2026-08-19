"""What v2 discovery can READ off an endpoint's `load()` — and what it cannot.

`pipeline_class` is the hub's required field on every model slot, recovered by
parsing `ctx.load(<PlainName>)`. These tests fix which spellings resolve, which
do not, and where the boundary sits.

The three arms below are the ones measured against `e0725c71` when the defect
was found. They are kept as fixtures because each answers a different question,
and the control is what makes the other two mean anything:

    ArmControl       module-level class          -> resolves   (the control)
    ArmLazyImport    `from x import C` in load()  -> fixed here
    ArmStubbedDep    module-top import of a STUB  -> fixed here (internvl lane)
    ArmSelfLoading   no `ctx.load` at all         -> still ""   (needs pgw#1421's
                                                                 typed self_loading
                                                                 marker, a Paul
                                                                 ruling; NOT this)

The two fixed shapes have ONE cause: the reader recovered the class by
importing and type-checking when the AST already carried the answer. So the
fallback is keyed on "the runtime lookup did not yield a TYPE", which covers a
name that is unbound (deferred import) and a name bound to a MODULE OBJECT (a
stubbed heavy dep) with a single branch.

An endpoint whose upstream runtime is source-built cannot import it at module
top — the package is not on PyPI and compiles CUDA extensions, so a discovery
runner has nothing to import — and the SDK's own convention tells the author to
defer it. Before this fix that convention made `_pipeline_class` return "" and
`_pipeline_class_or_refuse` hard-fail the publish, so the rule that keeps
discovery importable was the rule that made discovery refuse.
"""

from __future__ import annotations

import types

import pytest

from gen_worker.discovery.entrypoints_v2 import (
    EntrypointDiscoveryError,
    _pipeline_class,
    _pipeline_class_or_refuse,
)


class RealPipeline:
    """Importable at discovery — the control arm's subject."""


# The internvl-U condition, reproduced exactly: a module-top ImportFrom in the
# source (which is what `_import_sites` reads) whose RUNTIME binding is a
# MODULE OBJECT rather than a class — which is what the stub finder produces
# off-image for a package listed in `discovery_heavy_deps`. The try/except is
# the stub finder's stand-in, so this test needs none of its machinery.
try:  # pragma: no cover - `internvlu` is never installed here
    from internvlu import InternVLUPipeline
except ImportError:
    InternVLUPipeline = types.ModuleType("internvlu.InternVLUPipeline")


class ArmControl:
    def load(self, ctx):  # type: ignore[no-untyped-def]
        self.pipe = ctx.load(RealPipeline)


class ArmLazyImport:
    """What `trellis-3d` and `hunyuan3d-2.1` must write: the upstream package
    is source-built and unimportable on a discovery runner."""

    def load(self, ctx):  # type: ignore[no-untyped-def]
        from trellis2.pipelines import Trellis2ImageTo3DPipeline

        self.pipe = ctx.load(Trellis2ImageTo3DPipeline)


class ArmLazyAliased:
    def load(self, ctx):  # type: ignore[no-untyped-def]
        from trellis2.pipelines import Trellis2ImageTo3DPipeline as Pipe

        self.pipe = ctx.load(Pipe)


class ArmStubbedDep:
    """The internvl-U shape, contributed by that lane. A module-top import of a
    package listed in `discovery_heavy_deps`: off-image the stub finder binds a
    MODULE OBJECT where a class is expected, so `isinstance(target, type)` is
    False even though the import is module-top and sanctioned. In-image publish
    was fine; every off-image gate broke — the caller that reaches the refusal
    is `serverless-endpoints/scripts/lint_discovery.py:121`.
    """

    def load(self, ctx):  # type: ignore[no-untyped-def]
        self.pipe = ctx.load(InternVLUPipeline)


class ArmSelfLoading:
    """The v1 `Slot(str)` escape hatch. NOT addressed here, on purpose."""

    def load(self, ctx):  # type: ignore[no-untyped-def]
        from trellis2.pipelines import Trellis2ImageTo3DPipeline

        self.pipe = Trellis2ImageTo3DPipeline.from_pretrained(ctx.checkpoint_dir)


class ArmNoLoad:
    pass


def test_the_control_arm_resolves_through_the_module() -> None:
    """If this ever fails the other arms prove nothing — a probe that cannot
    succeed is not measuring the thing it names."""
    got = _pipeline_class(ArmControl)
    assert got.endswith(".RealPipeline")
    # Resolution (1) returns the REAL class object's identity, so it reports
    # the defining module rather than the import site's spelling.
    assert got == f"{RealPipeline.__module__}.{RealPipeline.__qualname__}"


# pgw#1431: a deferred import bound no module-level name, so discovery refused.
def test_a_deferred_import_resolves_to_its_dotted_path() -> None:
    """pgw#1431, the fix. Nothing is imported to answer this — `trellis2` is
    not installed in this environment, and that is the point."""
    with pytest.raises(ImportError):
        import trellis2  # noqa: F401

    assert (
        _pipeline_class(ArmLazyImport)
        == "trellis2.pipelines.Trellis2ImageTo3DPipeline"
    )


def test_a_deferred_import_under_an_alias_resolves_to_the_real_name() -> None:
    """`as Pipe` must report the imported name, not the local alias — the hub
    reads this string to name a class it will build."""
    assert (
        _pipeline_class(ArmLazyAliased)
        == "trellis2.pipelines.Trellis2ImageTo3DPipeline"
    )


# pgw#1431: a module-top import of a STUBBED heavy dep binds a module, not a class.
def test_a_stubbed_heavy_dep_resolves_statically_despite_the_vendored_package() -> None:
    """The internvl-U arm. `getattr(module, "InternVLUPipeline")` succeeds here
    and returns a MODULE, so the isinstance check correctly rejects it — and
    the static ImportFrom resolution then answers from the AST."""
    assert not isinstance(InternVLUPipeline, type)
    assert _pipeline_class(ArmStubbedDep) == "internvlu.InternVLUPipeline"


# pgw#1431: the boundary — self-loading needs its own marker (pgw#1421 ruling).
def test_a_self_loading_model_still_reads_as_absent() -> None:
    """The BOUNDARY of this fix, asserted so it cannot quietly widen.

    `ArmSelfLoading` never calls `ctx.load`, so there is no declaration to
    read and "" is the honest answer. Guessing the class from the
    `from_pretrained` receiver would publish a `pipeline_class` naming
    something the worker never builds through `ctx.load` — the se#757
    silent-lie shape. Self-loading needs its own typed marker (a Paul ruling,
    filed with pgw#1421), not a cleverer parser.
    """
    assert _pipeline_class(ArmSelfLoading) == ""


def test_a_model_with_no_load_reads_as_absent() -> None:
    assert _pipeline_class(ArmNoLoad) == ""


def test_an_absent_pipeline_class_still_refuses_at_publish() -> None:
    """The refusal this fix removes for the lazy-import case must remain fully
    armed for every other case — widening the parser must not disarm the gate."""
    rows = [{"name": "generate", "slots": [{"name": "model", "kind": "model", "pipeline_class": ""}]}]
    with pytest.raises(EntrypointDiscoveryError, match="could not read the pipeline class"):
        _pipeline_class_or_refuse(rows)


def test_an_adapter_slot_is_still_exempt() -> None:
    rows = [{"name": "generate", "slots": [{"name": "loras", "kind": "adapter"}]}]
    _pipeline_class_or_refuse(rows)  # does not raise
