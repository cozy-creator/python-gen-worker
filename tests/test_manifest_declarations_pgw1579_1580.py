"""THE DECLARATION LEDGER FENCE — every author-side declaration, and the
manifest key it owes (pgw#1579 / pgw#1580).

WHY THIS FILE EXISTS AND WHY IT IS GENERAL. The pgw#1373 v1 hardcut deleted
four author-side declarations while the hub kept decoding and ACTING on them,
and not one test failed. Each was found later by an endpoint breaking on it:
``child_calls`` at a workflow endpoint's first child call (pgw#1579),
``incremental_output`` at a streaming port (pgw#1576), ``expected_outputs``
LIVE IN PRODUCTION on five promoted endpoints (pgw#1580), and ``handles`` by
the enumeration audit that went looking for the third.

A test per field would have caught none of them, because the fields were gone
before anyone wrote the test. So the fences below are keyed on the SHAPE of the
mistake rather than on the field:

* :func:`test_every_entrypoint_kwarg_is_classified` reads
  ``inspect.signature(entrypoint)`` and requires every kwarg to appear in
  :data:`KWARG_DECLARATIONS`. A kwarg ADDED without wiring an emission fails
  here; a kwarg DELETED leaves an orphan row and fails here too.
* :func:`test_the_fully_declaring_row_carries_every_declaration` requires each
  ledger entry with a manifest key to appear on a row that declared it.
  Deleting an emission while keeping the kwarg fails here — the exact pgw#1579
  shape.
* :func:`test_the_declared_row_key_set_is_pinned` and its undeclared twin pin
  both ends of the manifest row, so a key that appears or vanishes must be
  argued for in this file.

The rule they encode is written down in ``gen_worker/v1_deleted.py``: a field
the platform decodes may only be dropped WITH a tombstone row naming its
successor. Every field that had one was a decision; every field that did not
was an accident.
"""

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


#: THE LEDGER: every ``@entrypoint`` kwarg -> the ``functions[]`` key it must
#: produce on a row that declares it, or ``None`` for one deliberately kept
#: SDK-side. Adding a kwarg without a row here fails
#: :func:`test_every_entrypoint_kwarg_is_classified`; a ``None`` must carry its
#: reason in the comment beside it, because "the hub does not read it" is a
#: measurement, not an assumption.
KWARG_DECLARATIONS: dict[str, str | None] = {
    "resources": "resources",
    "kind": "kind",
    "publishes": "publishes",
    "env": "env",
    # TRI-STATE and JOB-KIND ONLY (th#2177): on the REQUEST path the hub grants
    # `upload_media` unconditionally and has no column to store it in, so an
    # inference row emits nothing. Keyed here because a JOB-kind row does carry
    # it — `test_producer_declarations.py` owns that pair.
    "emits_media": "emits_media",
    # pgw#1579: `scheduler_dispatch.go` mints the `invoke_child` grant only
    # `if subj.ChildCallsDeclared`.
    "child_calls": "child_calls",
    # pgw#1580: `normalizeManifestHandles` validates it and the resolver
    # hydrates `FunctionMetadata.Handles` for selection.
    "handles": "handles",
    # pgw#1576: `manifestFunction.DeltaOutputSchema`, emitted off
    # `EntrypointSpec.delta_type`. It also flips `incremental_output`, which is
    # emitted UNCONDITIONALLY (the cardinality fact is stated, never omitted),
    # so the droppable-channel key is the one that proves the declaration
    # travelled. `tests/test_streaming_entrypoints.py` owns its
    # semantics; this row is here so the LEDGER stays complete — an
    # unclassified kwarg is the whole failure mode.
    "streams": "delta_output_schema",
}

#: Declarations that are NOT kwargs — read off the author's TYPES instead.
#: They need the same fence for the same reason: `ExpectedOutput` survived the
#: hardcut as an exported, documented, still-written annotation whose emission
#: was gone, which is why nothing complained for five promoted endpoints.
ANNOTATION_DECLARATIONS: dict[str, str] = {
    # `Annotated[..., ExpectedOutput(...)]` on the RETURN struct.
    "ExpectedOutput": "expected_outputs",
    # `Annotated[str, PromptRole(...)]` / typed Asset fields on the PAYLOAD.
    "PromptRole": "moderation",
}

#: ``release.ExpectedOutputPlan`` / ``builder.ExpectedOutputPlan``, field for
#: field. Read off the Go structs, not guessed. Anything outside this set is a
#: key the hub silently drops — a mirror with no reader, which is what th#2087
#: fences against and why ``duration_s`` is validated but never emitted.
EXPECTED_OUTPUT_PLAN_KEYS = frozenset(
    {"field", "type", "mime_type", "count", "width", "height", "aspect_ratio"}
)

#: ``normalizeExpectedOutputType``'s whole vocabulary; anything else becomes
#: ``other`` hub-side, so emitting a sixth word would be a silent downgrade.
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


# -- the general fences ------------------------------------------------------


def test_every_entrypoint_kwarg_is_classified() -> None:
    """THE FENCE THAT WOULD HAVE CAUGHT ALL FOUR.

    Read off the decorator's own signature, so it cannot go stale: every kwarg
    must be classified in :data:`KWARG_DECLARATIONS` as either producing a
    manifest key or deliberately not. A hardcut that deletes a kwarg orphans a
    row here; a lane that adds one without wiring the emission has nowhere to
    put it. Either way the next reader is stopped in this file, next to the
    rule, instead of by a pod nine months later.
    """
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
    """Every ledger entry with a manifest key reaches a row that declared it.

    This is the assertion the three P0s failed. It is written over the LEDGER
    rather than over a list of fields, so a future field is covered by adding
    one line above rather than by remembering to write a test.
    """
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
            continue  # `everything`'s payload declares no prompt/media field
        assert key in row, f"{marker} annotations reached no {key!r} key"


def test_the_declared_row_key_set_is_pinned(rows: dict[str, dict]) -> None:
    """Both directions. A DROPPED emission fails here even if someone deletes
    the ledger row with it; a NEW key has to be argued for in this file."""
    assert set(rows["everything"]) == {
        "name", "python_name", "module", "declared_module", "class_name",
        "kind", "input_schema", "payload_schema_sha256", "output_schema",
        "output_schema_sha256", "incremental_output", "delta_output_schema",
        "delta_output_schema_sha256", "expected_outputs",
        "slots", "resources", "publishes", "env", "emits_media",
        "child_calls", "handles",
    }


def test_the_undeclared_row_stays_byte_identical(rows: dict[str, dict]) -> None:
    """The other end of the pin: conditional emission, so a row that declares
    nothing is unchanged by any of this. The hub's own struct tags are
    ``omitempty`` and absent IS undeclared."""
    assert set(rows["describe"]) == {
        "name", "python_name", "module", "declared_module", "class_name",
        "kind", "input_schema", "payload_schema_sha256", "output_schema",
        "output_schema_sha256", "incremental_output", "slots",
    }


# -- child_calls (pgw#1579) --------------------------------------------------


def test_child_calls_reaches_the_spec_and_the_row(
    declaring: ModuleType, rows: dict[str, dict]
) -> None:
    """The dj-pipeline blocker. Weightless and slotless, because a workflow
    endpoint has no weights — the declaration must not depend on a model."""
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
    """The SDK half of one-fact-two-enforcers, and the CODE is the hub's own.

    Without this the refusal arrived from the hub as a 403 mid-request, after
    the child request had already been attempted, quoting a decorator the SDK
    had deleted."""
    ctx: RequestContext = RequestContext("req-undeclared")
    with pytest.raises(ChildCallRefusedError) as excinfo:
        ctx._callout_client()
    assert excinfo.value.code == "child_calls_not_declared"


def test_a_declared_body_reaches_the_child_call_surface() -> None:
    """Past the declaration gate it fails on the PLATFORM fact instead — proof
    the gate is the declaration and not the environment."""
    from gen_worker.api.errors import ChildCallError

    ctx: RequestContext = RequestContext("req-declared", child_calls=True)
    with pytest.raises(ChildCallError, match="no platform base URL"):
        ctx._callout_client()


def test_the_callout_refusals_name_the_successor_decorator() -> None:
    """pgw#1579's third layer: the remedy text pointed at ``@endpoint``, which
    the hardcut deleted. An unfollowable remedy is worse than none."""
    from gen_worker import callout

    source = Path(callout.__file__).read_text()
    assert "@endpoint(" not in source


# -- handles (pgw#1580) ------------------------------------------------------


def test_handles_reaches_the_spec_and_the_row(
    declaring: ModuleType, rows: dict[str, dict]
) -> None:
    assert spec(declaring.render).handles == ("fp8-w8a8-dynamic", "bf16-w16a16")
    assert rows["render"]["handles"] == ["fp8-w8a8-dynamic", "bf16-w16a16"]
    assert "handles" not in rows["describe"]


def test_handles_tokens_come_from_the_hubs_own_table(rows: dict[str, dict]) -> None:
    """Not a second vocabulary: ``known_execution_lane_bodies`` is the SDK twin
    of ``precision.KnownExecutionLaneBodies``, which is what
    ``normalizeManifestHandles`` validates against."""
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
    """Every one of these is a refusal the HUB would raise instead — after the
    image bake and the registry push. Raising at decoration names the author's
    own line."""
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
    """The asymmetry pgw#1580 asked about, CLOSED: reading the lane IS the
    divergence, so an undeclared read is refused rather than handed a plausible
    default nobody checked."""
    ctx: RequestContext = RequestContext("req-undeclared")
    with pytest.raises(LaneNotDeclaredError):
        ctx.execution_lane


def test_a_declared_body_reads_the_executing_lane() -> None:
    ctx: RequestContext = RequestContext("req-declared", handles=("bf16-w16a16",))
    assert ctx.execution_lane == "bf16-w16a16+eager"
    ctx._set_execution_lane("fp8-w8a8-dynamic+compiled")
    assert ctx.execution_lane == "fp8-w8a8-dynamic+compiled"


def test_ctx_lane_inside_load_is_not_gated() -> None:
    """The scope line, stated as a test so it is not re-litigated by guess.

    ``LoadContext.lane`` is the deploy's PICK and ``ctx.lane.dtype`` is how
    every model loads at all — the ordinary path, not a divergence — and it is
    MODEL scope while ``handles=`` is FUNCTION scope, so one model serving two
    entrypoints could not be gated coherently. Only the request-time read is
    the declared branch."""
    from gen_worker.serving.context import LoadContext

    assert not hasattr(LoadContext, "_require_lane_declaration")
    assert "handles" not in inspect.signature(LoadContext.__init__).parameters


# -- expected_outputs (pgw#1580) ---------------------------------------------


def test_expected_outputs_reproduces_the_measured_anima_row(
    rows: dict[str, dict],
) -> None:
    """THE PRODUCTION REGRESSION, pinned to the bytes that were measured.

    ``GET /api/v1/endpoints/tensorhub/anima`` served
    ``[{"type":"image","count":1,"field":"image",
    "aspect_ratio":"input.aspect_ratio"}]`` on v1 ``0.3.28`` and ``null`` on v2
    ``0.4.3``. The fixture's ``image`` field carries anima's annotation
    character for character, so this asserts the v1 row is back."""
    plan = rows["render"]["expected_outputs"]
    assert plan[0] == {
        "field": "image",
        "type": "image",
        "count": 1,
        "aspect_ratio": "input.aspect_ratio",
    }


def test_expected_outputs_carries_refs_and_lists(rows: dict[str, dict]) -> None:
    """The multi-output shape: the hub resolves each expression against the
    real request payload, so a ref must survive as a ref.

    The marker sits on the LIST, and the path is the annotated field's own —
    ``thumbnails``, not ``thumbnails[]``. v1's walk stopped at the ``Annotated``
    node without descending, and the cardinality is ``count`` rather than the
    container anyway."""
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
    """Shape parity with ``ExpectedOutputPlan``, both directions. A key outside
    the struct is silently dropped by the hub's decoder — a mirror with no
    reader — and a ``type`` outside the vocabulary becomes ``other``."""
    for name in ("make_video", "render", "everything"):
        for item in rows[name]["expected_outputs"]:
            assert set(item) <= EXPECTED_OUTPUT_PLAN_KEYS, set(item)
            assert {"field", "type"} <= set(item)
            assert item["type"] in EXPECTED_OUTPUT_TYPES


def test_duration_s_is_validated_but_never_emitted(rows: dict[str, dict]) -> None:
    """v1 emitted it and no hub reader has ever decoded it — it is absent from
    both ``ExpectedOutputPlan`` structs and from ``expectedOutputsFromPlans``.
    So it is checked (a wrong ref still fails the build) and dropped. The
    marker's own docstring already routes settlement at the probed
    ``VideoAsset.duration_s``."""
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
    """The pgw#1418 lesson, one field over. v1 swallowed a hint-resolution
    failure and fell back to ``__annotations__`` — strings, under ``from
    __future__ import annotations`` — so the walk found no markers and the plan
    came out EMPTY, indistinguishable from an endpoint that declared none."""
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


# -- the declarations reach the RUNNING body ---------------------------------


def test_the_serve_loop_stamps_both_declarations(declaring: ModuleType) -> None:
    """A declaration the running body never sees is a declaration that does
    nothing. This is the wire from the spec to ``ctx``, and it is the same wire
    ``publishes`` already had."""
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
    assert brancher.execution_lane  # does not raise

    plain = host._make_context("r3", None, spec(declaring.describe))
    assert plain.child_calls is False
    with pytest.raises(LaneNotDeclaredError):
        plain.execution_lane


def test_the_local_host_stamps_them_too(declaring: ModuleType) -> None:
    """The CLI and the daemon build a context BEFORE they know the function, so
    they stamp at dispatch. A declaration only the serverless dispatcher
    honours is one an author cannot test."""
    ctx: RequestContext = RequestContext("req-local")
    ctx._declare_from_spec(spec(declaring.everything))
    assert ctx.child_calls is True
    assert ctx.handles == ("fp8-w8a8-dynamic",)


# -- the row is still publishable -------------------------------------------


def test_every_fixture_row_survives_discovery(rows: dict[str, dict]) -> None:
    """``discover_entrypoints`` runs the pipeline-class refusal and the
    duplicate-name check over these rows, so reaching here at all is the
    publish path accepting every declaration above."""
    assert set(rows) == {"describe", "everything", "make_video", "render"}
    assert all(row["module"] == MODULE for row in rows.values())
