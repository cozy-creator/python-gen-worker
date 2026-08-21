from __future__ import annotations

import importlib
import inspect
import sys
from pathlib import Path
from types import ModuleType
from typing import Iterator

import pytest

from gen_worker.api.errors import ChildCallRefusedError, LaneNotDeclaredError
from gen_worker.discovery.entrypoints_v2 import discover_entrypoints
from gen_worker.models.execution_lanes import known_execution_lane_bodies
from gen_worker.serving.context import RequestContext
from gen_worker.serving.entrypoints import (
    ENTRYPOINT_ATTR,
    EntrypointDeclarationError,
    EntrypointSpec,
    entrypoint,
)

FIXTURES = Path(__file__).resolve().parent / "release_fixtures"
MODULE = "declaring_endpoint"


KWARG_DECLARATIONS: dict[str, str | None] = {
    "resources": "resources",
    "kind": "kind",
    "publishes": "publishes",
    "env": "env",
    "emits_media": "emits_media",
    "child_calls": "child_calls",
    "handles": "handles",
    "streams": "delta_output_schema",
}

ANNOTATION_DECLARATIONS: dict[str, str] = {
    "ExpectedOutput": "expected_outputs",
    "PromptRole": "moderation",
}

EXPECTED_OUTPUT_PLAN_KEYS = frozenset(
    {"field", "type", "mime_type", "count", "width", "height", "aspect_ratio"}
)

EXPECTED_OUTPUT_TYPES = frozenset({"image", "video", "audio", "file", "other"})


@pytest.fixture(scope="module")
def declaring() -> Iterator[ModuleType]:
    sys.path.insert(0, str(FIXTURES))
    try:
        yield importlib.import_module(MODULE)
    finally:
        sys.path.remove(str(FIXTURES))


@pytest.fixture(scope="module")
def rows(declaring: ModuleType) -> dict[str, dict]:
    return {row["name"]: row for row in discover_entrypoints(MODULE)}


def spec(fn: object) -> EntrypointSpec:
    stamped: EntrypointSpec = getattr(fn, ENTRYPOINT_ATTR)
    return stamped


def _probe(source: str, name: str) -> None:
    module = type(sys)(name)
    module.__dict__["__name__"] = name
    sys.modules[name] = module
    try:
        exec(compile(source, name, "exec"), module.__dict__)
    finally:
        sys.modules.pop(name, None)


def test_every_entrypoint_kwarg_is_classified() -> None:
    """THE FENCE THAT WOULD HAVE CAUGHT ALL FOUR."""
    parameters = set(inspect.signature(entrypoint).parameters) - {"fn"}
    missing = sorted(parameters - set(KWARG_DECLARATIONS))
    orphaned = sorted(set(KWARG_DECLARATIONS) - parameters)
    assert not missing, (
        f"@entrypoint grew {missing} with no ledger row. Add it to "
        "KWARG_DECLARATIONS naming the functions[] key it emits — or None with "
        "the MEASUREMENT showing the hub does not read it."
    )
    assert not orphaned, (
        f"KWARG_DECLARATIONS names {orphaned}, which @entrypoint no longer "
        "takes. If the deletion was deliberate it owes a v1_deleted.py row "
        "naming the successor; if it was not, this is the pgw#1579 shape again."
    )


def test_the_fully_declaring_row_carries_every_declaration(
    rows: dict[str, dict],
) -> None:
    """Every ledger entry with a manifest key reaches a row that declared it."""
    row = rows["everything"]
    for declaration, key in KWARG_DECLARATIONS.items():
        if key is None:
            continue
        assert key in row, (
            f"@entrypoint({declaration}=…) was declared and the manifest row "
            f"carries no {key!r}. The hub decodes it; an absent key is the "
            "fail-closed reading, so this endpoint would be declined the "
            "capability at RUNTIME with nothing failing at publish."
        )
    for marker, key in ANNOTATION_DECLARATIONS.items():
        if marker == "PromptRole":
            continue
        assert key in row, f"{marker} annotations reached no {key!r} key"


def test_the_declared_row_key_set_is_pinned(rows: dict[str, dict]) -> None:
    """Both directions."""
    assert set(rows["everything"]) == {
        "name", "python_name", "module", "declared_module", "class_name",
        "kind", "input_schema", "payload_schema_sha256", "output_schema",
        "output_schema_sha256", "incremental_output", "delta_output_schema",
        "delta_output_schema_sha256", "expected_outputs",
        "slots", "resources", "publishes", "env", "emits_media",
        "child_calls", "handles",
    }


def test_the_undeclared_row_stays_byte_identical(rows: dict[str, dict]) -> None:
    """The other end of the pin: conditional emission, so a row that declares nothing is unchanged by any of this."""
    assert set(rows["describe"]) == {
        "name", "python_name", "module", "declared_module", "class_name",
        "kind", "input_schema", "payload_schema_sha256", "output_schema",
        "output_schema_sha256", "incremental_output", "slots",
    }


def test_child_calls_reaches_the_spec_and_the_row(
    declaring: ModuleType, rows: dict[str, dict]
) -> None:
    """The dj-pipeline blocker."""
    assert spec(declaring.make_video).child_calls is True
    assert rows["make_video"]["child_calls"] is True
    assert rows["make_video"]["slots"] == []
    assert "child_calls" not in rows["describe"]
    assert spec(declaring.describe).child_calls is False


def test_child_calls_must_be_a_bool() -> None:
    with pytest.raises(EntrypointDeclarationError, match="child_calls= is a bool"):
        _probe(
            "import msgspec\n"
            "from gen_worker import RequestContext, entrypoint\n"
            "class I(msgspec.Struct): pass\n"
            "class O(msgspec.Struct): pass\n"
            "@entrypoint(child_calls='yes')\n"
            "def f(ctx: RequestContext, payload: I) -> O: ...\n",
            "pgw1579_child_calls_str",
        )


def test_an_undeclared_body_is_refused_the_child_call_surface() -> None:
    """The SDK half of one-fact-two-enforcers, and the CODE is the hub's own."""
    ctx: RequestContext = RequestContext("req-undeclared")
    with pytest.raises(ChildCallRefusedError) as excinfo:
        ctx._callout_client()
    assert excinfo.value.code == "child_calls_not_declared"


def test_a_declared_body_reaches_the_child_call_surface() -> None:
    """Past the declaration gate it fails on the PLATFORM fact instead — proof the gate is the declaration and not the environment."""
    from gen_worker.api.errors import ChildCallError

    ctx: RequestContext = RequestContext("req-declared", child_calls=True)
    with pytest.raises(ChildCallError, match="no platform base URL"):
        ctx._callout_client()


def test_the_callout_refusals_name_the_successor_decorator() -> None:
    from gen_worker import callout

    source = Path(callout.__file__).read_text()
    assert "@endpoint(" not in source


def test_handles_reaches_the_spec_and_the_row(
    declaring: ModuleType, rows: dict[str, dict]
) -> None:
    assert spec(declaring.render).handles == ("fp8-w8a8-dynamic", "bf16-w16a16")
    assert rows["render"]["handles"] == ["fp8-w8a8-dynamic", "bf16-w16a16"]
    assert "handles" not in rows["describe"]


def test_handles_tokens_come_from_the_hubs_own_table(rows: dict[str, dict]) -> None:
    """Not a second vocabulary: ``known_execution_lane_bodies`` is the SDK twin of ``precision.KnownExecutionLaneBodies``, which is what ``normalizeManifestHandles`` validates against."""
    known = set(known_execution_lane_bodies())
    for name in ("render", "everything"):
        assert set(rows[name]["handles"]) <= known


@pytest.mark.parametrize(
    "declared, match",
    [
        ("'fp8-w8a8-dynamic'", "iterate into"),
        ("('fp8-w8a8-dynamic+compiled',)", "carries an execution axis"),
        ("('fp8',)", "not a known lane body"),
        ("('bf16-w16a16', 'bf16-w16a16')", "repeats"),
    ],
)
def test_a_bad_handles_declaration_is_refused(declared: str, match: str) -> None:
    """Every one of these is a refusal the HUB would raise instead — after the image bake and the registry push."""
    with pytest.raises(EntrypointDeclarationError, match=match):
        _probe(
            "import msgspec\n"
            "from gen_worker import RequestContext, entrypoint\n"
            "class I(msgspec.Struct): pass\n"
            "class O(msgspec.Struct): pass\n"
            f"@entrypoint(handles={declared})\n"
            "def f(ctx: RequestContext, payload: I) -> O: ...\n",
            f"pgw1580_handles_{abs(hash(declared))}",
        )


def test_an_undeclared_body_is_refused_the_executing_lane() -> None:
    ctx: RequestContext = RequestContext("req-undeclared")
    with pytest.raises(LaneNotDeclaredError):
        ctx.execution_lane


def test_a_declared_body_reads_the_executing_lane() -> None:
    ctx: RequestContext = RequestContext("req-declared", handles=("bf16-w16a16",))
    assert ctx.execution_lane == "bf16-w16a16+eager"
    ctx._set_execution_lane("fp8-w8a8-dynamic+compiled")
    assert ctx.execution_lane == "fp8-w8a8-dynamic+compiled"


def test_ctx_lane_inside_load_is_not_gated() -> None:
    """The scope line, stated as a test so it is not re-litigated by guess."""
    from gen_worker.serving.context import LoadContext

    assert not hasattr(LoadContext, "_require_lane_declaration")
    assert "handles" not in inspect.signature(LoadContext.__init__).parameters


def test_expected_outputs_reproduces_the_measured_anima_row(
    rows: dict[str, dict],
) -> None:
    """THE PRODUCTION REGRESSION, pinned to the bytes that were measured."""
    plan = rows["render"]["expected_outputs"]
    assert plan[0] == {
        "field": "image",
        "type": "image",
        "count": 1,
        "aspect_ratio": "input.aspect_ratio",
    }


def test_expected_outputs_carries_refs_and_lists(rows: dict[str, dict]) -> None:
    """The multi-output shape: the hub resolves each expression against the real request payload, so a ref must survive as a ref."""
    plan = {item["field"]: item for item in rows["render"]["expected_outputs"]}
    assert plan["thumbnails"] == {
        "field": "thumbnails",
        "type": "image",
        "count": "input.num_images",
        "width": "input.width",
        "height": "input.height",
        "mime_type": "image/webp",
    }


def test_expected_outputs_matches_the_hubs_plan_struct(rows: dict[str, dict]) -> None:
    """Shape parity with ``ExpectedOutputPlan``, both directions."""
    for name in ("make_video", "render", "everything"):
        for item in rows[name]["expected_outputs"]:
            assert set(item) <= EXPECTED_OUTPUT_PLAN_KEYS, set(item)
            assert {"field", "type"} <= set(item)
            assert item["type"] in EXPECTED_OUTPUT_TYPES


def test_duration_s_is_validated_but_never_emitted(rows: dict[str, dict]) -> None:
    """v1 emitted it and no hub reader has ever decoded it — it is absent from both ``ExpectedOutputPlan`` structs and from ``expectedOutputsFromPlans``."""
    plan = rows["make_video"]["expected_outputs"]
    assert plan == [{
        "field": "video",
        "type": "video",
        "count": 1,
        "mime_type": "video/mp4",
    }]

    with pytest.raises(ValueError, match="unknown payload field"):
        _probe(
            "from typing import Annotated\n"
            "import msgspec\n"
            "from gen_worker import ExpectedOutput, RequestContext, VideoAsset\n"
            "from gen_worker import entrypoint\n"
            "from gen_worker.discovery.entrypoints_v2 import _entrypoint_row\n"
            "from gen_worker.serving.entrypoints import ENTRYPOINT_ATTR\n"
            "class I(msgspec.Struct): pass\n"
            "class O(msgspec.Struct):\n"
            "    v: Annotated[VideoAsset, ExpectedOutput("
            "media_type='video', duration_s='input.seconds')]\n"
            "@entrypoint\n"
            "def f(ctx: RequestContext, payload: I) -> O: ...\n"
            "_entrypoint_row(getattr(f, ENTRYPOINT_ATTR))\n",
            "pgw1580_bad_duration_ref",
        )


def test_an_unresolvable_output_struct_refuses_instead_of_emptying() -> None:
    from gen_worker.discovery.expected_outputs import expected_outputs

    module = type(sys)("pgw1580_unresolvable")
    module.__dict__["__name__"] = "pgw1580_unresolvable"
    sys.modules["pgw1580_unresolvable"] = module
    try:
        exec(compile(
            "from __future__ import annotations\n"
            "import msgspec\n"
            "class I(msgspec.Struct): pass\n"
            "class O(msgspec.Struct):\n"
            "    v: NeverImported\n",
            "pgw1580_unresolvable", "exec"), module.__dict__)
        with pytest.raises(ValueError, match="silently EMPTY"):
            expected_outputs(module.__dict__["I"], module.__dict__["O"])
    finally:
        sys.modules.pop("pgw1580_unresolvable", None)


def test_a_row_with_no_markers_omits_the_key(rows: dict[str, dict]) -> None:
    assert "expected_outputs" not in rows["describe"]


def test_the_serve_loop_stamps_both_declarations(declaring: ModuleType) -> None:
    """A declaration the running body never sees is a declaration that does nothing."""
    from gen_worker.serving.serve_loop import ServeLoop

    host = object.__new__(ServeLoop)
    host._context_kwargs = {}
    host._output_dir = None
    host._hf_token = ""

    workflow = host._make_context("r1", None, spec(declaring.make_video))
    assert workflow.child_calls is True
    assert workflow.handles == ()

    brancher = host._make_context("r2", None, spec(declaring.render))
    assert brancher.child_calls is False
    assert brancher.handles == ("fp8-w8a8-dynamic", "bf16-w16a16")
    assert brancher.execution_lane

    plain = host._make_context("r3", None, spec(declaring.describe))
    assert plain.child_calls is False
    with pytest.raises(LaneNotDeclaredError):
        plain.execution_lane


def test_the_local_host_stamps_them_too(declaring: ModuleType) -> None:
    """The CLI and the daemon build a context BEFORE they know the function, so they stamp at dispatch."""
    ctx: RequestContext = RequestContext("req-local")
    ctx._declare_from_spec(spec(declaring.everything))
    assert ctx.child_calls is True
    assert ctx.handles == ("fp8-w8a8-dynamic",)


def test_every_fixture_row_survives_discovery(rows: dict[str, dict]) -> None:
    """``discover_entrypoints`` runs the pipeline-class refusal and the duplicate-name check over these rows, so reaching here at all is the publish path accepting every declaration above."""
    assert set(rows) == {"describe", "everything", "make_video", "render"}
    assert all(row["module"] == MODULE for row in rows.values())
