"""What v2 discovery can READ off an endpoint's `load()` — and what it cannot."""

from __future__ import annotations

import linecache
import types

import pytest

from gen_worker.discovery.entrypoints_v2 import (
    EntrypointDiscoveryError,
    _pipeline_class,
    _pipeline_class_or_refuse,
)


class RealPipeline:
    """Importable at discovery — the control arm's subject."""


try:  # pragma: no cover - `internvlu` is never installed here
    from internvlu import InternVLUPipeline
except ImportError:
    InternVLUPipeline = types.ModuleType("internvlu.InternVLUPipeline")


class ArmControl:
    def load(self, ctx):  # type: ignore[no-untyped-def]
        self.pipe = ctx.load(RealPipeline)


class ArmLazyImport:
    """What `trellis-3d` and `hunyuan3d-2.1` must write: the upstream package is source-built and unimportable on a discovery runner."""

    def load(self, ctx):  # type: ignore[no-untyped-def]
        from trellis2.pipelines import Trellis2ImageTo3DPipeline

        self.pipe = ctx.load(Trellis2ImageTo3DPipeline)


class ArmLazyAliased:
    def load(self, ctx):  # type: ignore[no-untyped-def]
        from trellis2.pipelines import Trellis2ImageTo3DPipeline as Pipe

        self.pipe = ctx.load(Pipe)


class ArmStubbedDep:
    """The internvl-U shape, contributed by that lane."""

    def load(self, ctx):  # type: ignore[no-untyped-def]
        self.pipe = ctx.load(InternVLUPipeline)


class ArmSelfLoading:
    """The v1 `Slot(str)` escape hatch."""

    def load(self, ctx):  # type: ignore[no-untyped-def]
        from trellis2.pipelines import Trellis2ImageTo3DPipeline

        self.pipe = Trellis2ImageTo3DPipeline.from_pretrained(ctx.checkpoint_dir)


class ArmNoLoad:
    pass


def test_the_control_arm_resolves_through_the_module() -> None:
    """If this ever fails the other arms prove nothing — a probe that cannot succeed is not measuring the thing it names."""
    got = _pipeline_class(ArmControl)
    assert got.endswith(".RealPipeline")
    assert got == f"{RealPipeline.__module__}.{RealPipeline.__qualname__}"


def test_a_deferred_import_resolves_to_its_dotted_path() -> None:
    with pytest.raises(ImportError):
        import trellis2  # noqa: F401

    assert (
        _pipeline_class(ArmLazyImport)
        == "trellis2.pipelines.Trellis2ImageTo3DPipeline"
    )


def test_a_deferred_import_under_an_alias_resolves_to_the_real_name() -> None:
    """`as Pipe` must report the imported name, not the local alias — the hub reads this string to name a class it will build."""
    assert (
        _pipeline_class(ArmLazyAliased)
        == "trellis2.pipelines.Trellis2ImageTo3DPipeline"
    )


def test_a_stubbed_heavy_dep_resolves_statically_despite_the_vendored_package() -> None:
    """The internvl-U arm."""
    assert not isinstance(InternVLUPipeline, type)
    assert _pipeline_class(ArmStubbedDep) == "internvlu.InternVLUPipeline"


def test_a_self_loading_model_still_reads_as_absent() -> None:
    """The BOUNDARY of this fix, asserted so it cannot quietly widen."""
    assert _pipeline_class(ArmSelfLoading) == ""


def test_a_model_with_no_load_reads_as_absent() -> None:
    assert _pipeline_class(ArmNoLoad) == ""


def test_an_absent_pipeline_class_still_refuses_at_publish() -> None:
    """The refusal this fix removes for the lazy-import case must remain fully armed for every other case — widening the parser must not disarm the gate."""
    rows = [{"name": "generate", "slots": [{"name": "model", "kind": "model", "pipeline_class": ""}]}]
    with pytest.raises(EntrypointDiscoveryError, match="could not read the pipeline class"):
        _pipeline_class_or_refuse(rows)


def test_an_adapter_slot_is_still_exempt() -> None:
    rows = [{"name": "generate", "slots": [{"name": "loras", "kind": "adapter"}]}]
    _pipeline_class_or_refuse(rows)


from gen_worker import lane  # noqa: E402
from gen_worker.demand import GiB, const  # noqa: E402
from gen_worker.models import Trellis2  # noqa: E402
from gen_worker.serving.model import Model  # noqa: E402

#: pgw#1599: `lanes=` is REQUIRED on every model class, so every specimen in
#: this file names a real lane. pgw#1621: a lane is the `(topology, quant)`
#: stamp pair and BOTH halves must be in the vendored `spec/v2` corpus — which
#: carries no `trellis2.*` topology, so `trellis2.dit-bf16@1` (a v1 document,
#: deleted with the v1 corpus) has no successor to name. These specimens keep
#: `Model[Trellis2]` and declare a ratified pair: the model TYPE and the lane
#: are independent declarations, which is exactly what the last test in this
#: file asserts, and what these tests are about is `self_loading=`.
_SPECIMEN_LANE = ("sd15.diffusers@1", "plain.bf16@1")
_SPECIMEN_LANE_ID = "sd15.diffusers@1+plain.bf16@1"
_TRELLIS_LANES = {_SPECIMEN_LANE: lane(request=const(GiB(1)))}


class PlainWithLoad(Model[Trellis2], lanes=_TRELLIS_LANES):
    """Unmarked, with a readable ctx.load — the control for the marker."""

    def load(self, ctx):  # type: ignore[no-untyped-def]
        self.pipe = ctx.load(RealPipeline)


class MarkedAndReadable(
    Model[Trellis2],
    lanes=_TRELLIS_LANES,
    self_loading="claims ctx.load cannot drive it",
):
    """Declares the marker AND calls ctx.load — a contradiction, and the thing `_model_slot` must refuse rather than silently prefer one of."""

    def load(self, ctx):  # type: ignore[no-untyped-def]
        self.pipe = ctx.load(RealPipeline)


class _Slot:

    def __init__(self, name, annotation):  # type: ignore[no-untyped-def]
        self.name = name
        self.annotation = annotation


def _marked_model(reason="bespoke pipeline.json loader; ctx.load drives neither path"):  # type: ignore[no-untyped-def]
    from gen_worker.models import Trellis2
    from gen_worker.serving.model import Model

    namespace = {"Model": Model, "Trellis2": Trellis2, "LANES": _TRELLIS_LANES}
    source = (
        "class Marked(Model[Trellis2], lanes=LANES, self_loading=%r):\n"
        "    def load(self, ctx):\n"
        "        self.pipe = object()\n" % reason
    )
    # pgw#1655: the header is READ, so it has to stay readable. `exec` of a
    # bare string leaves `load` with no source at all — `inspect.getsource`
    # raises, compile subjecthood cannot be stated, and a class header the
    # platform cannot read is a refusal, not an eager declaration. Seeding
    # linecache under a synthetic filename is what a real module gets for
    # free.
    filename = "<pgw1655-marked-model>"
    linecache.cache[filename] = (
        len(source), None, source.splitlines(keepends=True), filename
    )
    exec(  # noqa: S102 - the class header IS the thing under test
        compile(source, filename, "exec"), namespace
    )
    return namespace["Marked"]


def test_a_marked_slot_states_its_reason_instead_of_a_pipeline_class() -> None:
    """The two keys are mutually exclusive in the manifest, exactly as `layouts`/`layouts_undeclarable` are one level down: a slot either names its class or says why it has none."""
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
    """Both cannot be true at once."""
    from gen_worker.discovery.entrypoints_v2 import _model_slot

    with pytest.raises(EntrypointDiscoveryError, match="those contradict"):
        _model_slot(_Slot("model", MarkedAndReadable))


def test_a_marked_slot_passes_the_publish_gate_and_an_unmarked_one_does_not() -> None:
    """The ArmSelfLoading boundary becomes the green path FOR MARKED SLOTS ONLY."""
    from gen_worker.discovery.entrypoints_v2 import _pipeline_class_or_refuse

    marked = [{"name": "generate", "slots": [{"name": "model", "kind": "model", "self_loading": "bespoke loader"}]}]
    _pipeline_class_or_refuse(marked)

    unmarked = [{"name": "generate", "slots": [{"name": "model", "kind": "model", "pipeline_class": ""}]}]
    with pytest.raises(EntrypointDiscoveryError, match="could not read the pipeline class"):
        _pipeline_class_or_refuse(unmarked)


def test_the_marker_demands_a_reason() -> None:
    """Verbatim the rule `Slot(layouts_undeclarable=)` enforces one level down: an escape hatch with no stated reason is the silence the rung replaces."""
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
    """A self-loading model still has weights, still has a lane, still has a demand formula."""
    from gen_worker.models import Trellis2
    from gen_worker.serving.model import Model, model_declared_lanes

    namespace = {
        "Model": Model, "Trellis2": Trellis2, "LANE": _SPECIMEN_LANE,
        "lane": lane, "const": const, "GiB": GiB,
    }
    exec(  # noqa: S102
        "class Both(Model[Trellis2],\n"
        "           lanes={LANE: lane(request=const(GiB(2)))},\n"
        "           self_loading='bespoke loader'):\n"
        "    pass",
        namespace,
    )
    (declared,) = model_declared_lanes(namespace["Both"])
    assert declared.contract_id == _SPECIMEN_LANE_ID
    assert declared.request.coefficients() == {"const": GiB(2)}
    assert namespace["Both"].__cozy_self_loading__ == "bespoke loader"
