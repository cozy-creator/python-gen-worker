"""pgw#1307 batch B — the arms deferred as "own lanes", executed.

One test per cut, each red before its change:

  * arm (3)  the serving contract's SCOPE, ported from the hub's one rule, and
             the empty-status arm that made the backstop more permissive than
             the authority it backstops;
  * arm (7)  the fabricated ``legacy-drain-`` goal id on a projection the hub
             reads — unreachable, because ``ensure_intent`` is total;
  * arm (10) the duplicate-function-name error naming the WALKED module rather
             than the declared one;
  * arm (13) NOT a cut: the string-code error envelope is live on master's
             hub, so this pins the shapes that keep it load-bearing;
  * addendum a keyless cell record refuses instead of adopting its directory.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import msgspec
import pytest

from gen_worker import RequestContext, Slot, endpoint, worker_function  # noqa: E402


class _In(msgspec.Struct, frozen=True):
    x: str = ""
    model: str = ""


class _Out(msgspec.Struct, frozen=True):
    y: str = ""


# ---------------------------------------------------------------------------
# arm (3): the serving contract governs the slot the pick lands on
# ---------------------------------------------------------------------------


def test_serving_contract_scope_mirrors_the_hubs_one_rule() -> None:
    """`modelfamily.ServingContractGoverns`, term for term.

    A selectable slot is always governed; a function that has one elsewhere
    governs only that one; otherwise the slots that declared a family.
    """
    from gen_worker.api.slot import serving_contract_governs_slot as governs

    assert governs(
        selected_by="model", family="", function_has_selectable_slot=True)
    assert not governs(
        selected_by="", family="example", function_has_selectable_slot=True)
    assert governs(
        selected_by="", family="example", function_has_selectable_slot=False)
    assert not governs(
        selected_by="", family="", function_has_selectable_slot=False)


def test_an_ungoverned_slot_is_never_asked_for_evidence() -> None:
    """The contract governs the slot the pick lands on, not its siblings.

    A function with a selectable slot governs only that one — so an auxiliary
    slot (an interpolator, an upscaler) resolving to a checkpoint nothing has
    classified must resolve cleanly. Before the scope was ported, that stamp
    failed the WHOLE warmup for any function declaring `distilled=`.
    """
    from gen_worker import dispatch
    from gen_worker.api.binding import HF
    from gen_worker.registry import extract_specs
    from gen_worker.warmup import resolved_slots_kwargs

    @endpoint(
        models={
            "pipeline": Slot(
                object, family="example", selected_by="model",
                default_checkpoint=HF("acme/plain-xl")),
            "interpolator": Slot(
                object, family="example",
                default_checkpoint=HF("acme/rife")),
        }
    )
    class Gen:
        def setup(self, pipeline: object, interpolator: object) -> None:
            self.pipeline = pipeline
            self.interpolator = interpolator

        @worker_function(distilled=False)
        def render(self, ctx: RequestContext, data: _In) -> _Out:
            return _Out(y="ok")

    spec = extract_specs(Gen)[0]
    slots = {
        "pipeline": dispatch.SlotOrder(
            ref="acme/plain-xl", distilled=False,
            distilled_status="classified"),
        "interpolator": dispatch.SlotOrder(
            ref="acme/rife", distilled=False, distilled_status="unclassified"),
    }

    result = resolved_slots_kwargs(spec, slots)

    assert "interpolator" not in result["slot_errors"], result["slot_errors"]
    assert "interpolator" in result["resolved_slots"]
    # …and the slot the pick DOES land on is still governed.
    assert "pipeline" in result["resolved_slots"]


# ---------------------------------------------------------------------------
# arm (7): no fabricated goal id reaches a projection the hub reads
# ---------------------------------------------------------------------------


def test_ensure_intent_is_total_so_a_drain_never_fabricates_a_goal_id() -> None:
    from gen_worker.lifecycle_intents import IntentRegistry
    from gen_worker.pb import worker_scheduler_pb2 as pb

    reg = IntentRegistry("release-1", ["render"])
    # No hub-authored drain command has ever arrived: the worker-local carrier
    # is what must answer, and it must be a real minted intent.
    assert reg.ensure_intent(pb.DESIRED_INTENT_KIND_DRAIN)

    reg.set_drain(pb.DRAIN_LIFECYCLE_STATUS_DRAINING)
    snapshot = reg.snapshot()

    assert snapshot.drain.goal_id
    assert not snapshot.drain.goal_id.startswith("legacy-drain-")
    assert snapshot.drain.intent_id in {s.intent_id for s in snapshot.intents}


def test_the_legacy_drain_spelling_exists_nowhere_in_the_package() -> None:
    """The source fence: the arm was UNREACHABLE, so only a text check can
    tell the cut from the state before it."""
    import gen_worker

    root = Path(gen_worker.__file__).parent
    offenders = [
        str(p) for p in root.rglob("*.py")
        if "legacy-drain-" in p.read_text(encoding="utf-8", errors="ignore")
    ]
    assert offenders == []


# ---------------------------------------------------------------------------
# arm (10): the duplicate-name error names the DEFINING module
# ---------------------------------------------------------------------------


def test_duplicate_name_error_names_the_declared_module() -> None:
    from gen_worker.discovery.discover import _assert_unique_function_names

    functions = [
        {"name": "generate", "class_name": "A",
         "module": "pkg", "declared_module": "pkg.handlers.a"},
        {"name": "generate", "class_name": "B",
         "module": "pkg", "declared_module": "pkg.handlers.b"},
    ]
    with pytest.raises(ValueError) as exc:
        _assert_unique_function_names(functions)

    message = str(exc.value)
    assert "pkg.handlers.a" in message
    assert "pkg.handlers.b" in message


# ---------------------------------------------------------------------------
# arm (13): NOT residue — the string-code envelope is live on master's hub
# ---------------------------------------------------------------------------


class _Resp:
    def __init__(self, status: int, body: Any) -> None:
        self.status_code = status
        self.headers = {"Content-Type": "application/json"}
        self._body = body

    def json(self) -> Any:
        return self._body


@pytest.mark.parametrize(
    "status,body",
    [
        # internal/api/endpoint_bindings.go — bindingWriteError.write
        (422, {"error": "binding_incompatible", "message": "family mismatch",
               "issues": []}),
        # internal/orchestrator/http/model_overrides.go — modelOverrideErrorBody
        (400, {"error": "model_override_rejected", "code": "reserved_input_field",
               "message": "resolved_models is reserved", "param": "models"}),
        # internal/api/capability_upload_budget.go
        (403, {"error": "forbidden", "code": "no_matching_grant",
               "message": "no capability grant authorizes this upload"}),
        (429, {"error": "grant_exhausted", "code": "grant_exhausted",
               "message": "capability grant budget exhausted"}),
    ],
)
def test_live_string_code_envelopes_are_hub_verdicts(status: int, body: Any) -> None:
    """These four routes on master's hub emit the code as a STRING. Deleting
    the shape-2 classifier as "old-hub residue" reinstates pgw#743 on every
    one of them — a refusal retried until it becomes a different, later error.
    """
    from gen_worker.http_origin import is_definite_hub_answer, response_is_from_hub

    resp = _Resp(status, body)
    assert response_is_from_hub(resp)
    assert is_definite_hub_answer(resp)


# ---------------------------------------------------------------------------
# addendum (cl#63's sibling): a keyless record refuses
# ---------------------------------------------------------------------------


def test_a_keyless_cell_record_refuses_instead_of_adopting_its_directory(
    tmp_path: Path,
) -> None:
    from gen_worker import local_cell_store as store

    root = tmp_path / "cache"
    key = "ck1_" + "a" * 40
    cell = store.cells_root(root) / key
    cell.mkdir(parents=True)
    (cell / store.RECORD_NAME).write_text(json.dumps({
        "family": "example", "bytes": 12, "stored_at": 1.0,
        "verdict": store.VERDICT_ADMITTED, "sink": store.SINK_OWED,
    }))

    listed = store.stored_cells(root)

    assert [c.key for c in listed] == []
    # And the identity never reaches an upload scan either.
    assert store.cells_owed_to_sink(root) == []


def test_a_keyed_record_still_lists(tmp_path: Path) -> None:
    """The refusal above must not blank a healthy store."""
    from gen_worker import local_cell_store as store

    root = tmp_path / "cache"
    key = "ck1_" + "b" * 40
    cell = store.cells_root(root) / key
    cell.mkdir(parents=True)
    (cell / store.RECORD_NAME).write_text(json.dumps({
        "compiled_graph_key": key, "family": "example", "bytes": 12,
        "stored_at": 1.0, "verdict": store.VERDICT_ADMITTED,
        "sink": store.SINK_OWED,
    }))

    assert [c.key for c in store.stored_cells(root)] == [key]
    assert [c.key for c in store.cells_owed_to_sink(root)] == [key]
