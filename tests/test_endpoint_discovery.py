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


# --------------------------------------------------------------------------- #
# pgw#1431 fix (b): the `self_loading=` marker — the v2 successor to v1's       #
# `Slot(str)` escape hatch, for pipelines `ctx.load` structurally cannot drive. #
# --------------------------------------------------------------------------- #


from gen_worker import lane  # noqa: E402
from gen_worker._vendor.tensorfs import contracts as _tfs_contracts  # noqa: E402
from gen_worker.demand import GiB, const  # noqa: E402
from gen_worker.models import Trellis2  # noqa: E402
from gen_worker.serving.model import Model  # noqa: E402

#: pgw#1599: `lanes=` is REQUIRED on every model class, so every specimen in
#: this file names Trellis2's real document. That is orthogonal to what these
#: tests are about (`self_loading=`), which is the point of the last test here.
_TRELLIS_LANES = {_tfs_contracts.TRELLIS2_DIT_BF16: lane(request=const(GiB(1)))}


class PlainWithLoad(Model[Trellis2], lanes=_TRELLIS_LANES):
    """Unmarked, with a readable ctx.load — the control for the marker."""

    def load(self, ctx):  # type: ignore[no-untyped-def]
        self.pipe = ctx.load(RealPipeline)


class MarkedAndReadable(
    Model[Trellis2],
    lanes=_TRELLIS_LANES,
    self_loading="claims ctx.load cannot drive it",
):
    """Declares the marker AND calls ctx.load — a contradiction, and the thing
    `_model_slot` must refuse rather than silently prefer one of."""

    def load(self, ctx):  # type: ignore[no-untyped-def]
        self.pipe = ctx.load(RealPipeline)


class _Slot:
    """Only what `_model_slot` reads off a discovered parameter."""

    def __init__(self, name, annotation):  # type: ignore[no-untyped-def]
        self.name = name
        self.annotation = annotation


def _marked_model(reason="bespoke pipeline.json loader; ctx.load drives neither path"):  # type: ignore[no-untyped-def]
    from gen_worker.models import Trellis2
    from gen_worker.serving.model import Model

    namespace = {"Model": Model, "Trellis2": Trellis2, "LANES": _TRELLIS_LANES}
    exec(  # noqa: S102 - the class header IS the thing under test
        "class Marked(Model[Trellis2], lanes=LANES, self_loading=%r):\n"
        "    def load(self, ctx):\n"
        "        self.pipe = object()\n" % reason,
        namespace,
    )
    return namespace["Marked"]


def test_a_marked_slot_states_its_reason_instead_of_a_pipeline_class() -> None:
    """The two keys are mutually exclusive in the manifest, exactly as
    `layouts`/`layouts_undeclarable` are one level down: a slot either names
    its class or says why it has none."""
    from gen_worker.discovery.entrypoints_v2 import _model_slot

    emitted = _model_slot(_Slot("model", _marked_model()))
    assert emitted["self_loading"].startswith("bespoke pipeline.json loader")
    assert "pipeline_class" not in emitted


def test_an_unmarked_slot_still_emits_a_pipeline_class() -> None:
    from gen_worker.discovery.entrypoints_v2 import _model_slot

    emitted = _model_slot(_Slot("model", PlainWithLoad))
    assert emitted["pipeline_class"].endswith(".RealPipeline")
    assert "self_loading" not in emitted


def test_declaring_the_marker_AND_a_readable_ctx_load_is_a_refusal() -> None:
    """Both cannot be true at once. Without this refusal the marker is a way to
    silence a class discovery could have read perfectly well — the se#757
    silent-lie shape wearing a new hat."""
    from gen_worker.discovery.entrypoints_v2 import _model_slot

    with pytest.raises(EntrypointDiscoveryError, match="those contradict"):
        _model_slot(_Slot("model", MarkedAndReadable))


def test_a_marked_slot_passes_the_publish_gate_and_an_unmarked_one_does_not() -> None:
    """The ArmSelfLoading boundary becomes the green path FOR MARKED SLOTS
    ONLY. Widening the marker must not disarm the gate for everyone else."""
    from gen_worker.discovery.entrypoints_v2 import _pipeline_class_or_refuse

    marked = [{"name": "generate", "slots": [{"name": "model", "kind": "model", "self_loading": "bespoke loader"}]}]
    _pipeline_class_or_refuse(marked)  # does not raise

    unmarked = [{"name": "generate", "slots": [{"name": "model", "kind": "model", "pipeline_class": ""}]}]
    with pytest.raises(EntrypointDiscoveryError, match="could not read the pipeline class"):
        _pipeline_class_or_refuse(unmarked)


def test_the_marker_demands_a_reason() -> None:
    """Verbatim the rule `Slot(layouts_undeclarable=)` enforces one level down:
    an escape hatch with no stated reason is the silence the rung replaces."""
    from gen_worker.models import Trellis2
    from gen_worker.serving.model import Model, ModelDeclarationError

    namespace = {"Model": Model, "Trellis2": Trellis2, "LANES": _TRELLIS_LANES}
    for bad in ("", "   ", 123):
        with pytest.raises(ModelDeclarationError):
            exec(  # noqa: S102
                "class B(Model[Trellis2], lanes=LANES, self_loading=%r):\n"
                "    pass" % (bad,),
                dict(namespace),
            )


def test_the_marker_is_ORTHOGONAL_to_lanes() -> None:
    """A self-loading model still has weights, still has a lane, still has a
    demand formula. `trellis-3d` declares both; coupling them would strand its
    declaration the way pgw#1423 strands hunyuan's.

    pgw#1599: the VRAM FLOOR half of this test is gone with the strings — a
    lane's memory statement is now its demand FORMULA, and the placement row
    carries only the floor DERIVED from the contract dtype."""
    from gen_worker._vendor.tensorfs import contracts

    from gen_worker.models import Trellis2
    from gen_worker.serving.model import Model, model_declared_lanes

    namespace = {
        "Model": Model, "Trellis2": Trellis2, "contracts": contracts,
        "lane": lane, "const": const, "GiB": GiB,
    }
    exec(  # noqa: S102
        "class Both(Model[Trellis2],\n"
        "           lanes={contracts.TRELLIS2_DIT_BF16: lane(\n"
        "               request=const(GiB(2)))},\n"
        "           self_loading='bespoke loader'):\n"
        "    pass",
        namespace,
    )
    (declared,) = model_declared_lanes(namespace["Both"])
    assert declared.contract_id == "trellis2.dit-bf16@1"
    assert declared.request.coefficients() == {"const": GiB(2)}
    assert namespace["Both"].__cozy_self_loading__ == "bespoke loader"
