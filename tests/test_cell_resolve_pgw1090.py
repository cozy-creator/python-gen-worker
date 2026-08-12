"""pgw#1090 (§4.29) — the worker half of ``POST /v1/worker/cells/resolve``.

Written against th#1750's ANSWER CONTRACT (hub side merged ``26275ff8``), so a
hub-side field rename reds here rather than on a pod. Every row is off-wire: the
broker is stubbed, which is what the ``procsplit`` seam exists to make possible.
"""

from __future__ import annotations

import json
from typing import Any, Dict

import pytest

from gen_worker import aot_identity, cell_key as ck, cell_resolve, env_seal
from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.plan import AttemptRef, PlanFactory
from gen_worker.procsplit import actions as actions_mod

KEY = "ck1-" + "ab" * 28
OTHER_KEY = "ck1-" + "cd" * 28


class _Resp:
    def __init__(self, status: int, body: Any) -> None:
        self.status_code = status
        self.text = json.dumps(body) if not isinstance(body, str) else body

    def json(self) -> Any:
        return json.loads(self.text) if self.text else {}


def _hit_body(**over: Any) -> Dict[str, Any]:
    body = {
        "found": True,
        "family": "sdxl",
        "cell_key": KEY,
        "cell_ref": f"root/family-sdxl#{KEY}",
        "checkpoint_id": "ckpt-1",
        "content_digest": "sha256:" + "11" * 32,
        "artifact_path": "cell.tar.gz",
        "size_bytes": 4096,
        "publisher_org": "org-a",
        "publisher_tier": "platform",
        "graph_contract": "c0ffee0000000000",
        "toolchain_digest": "toolch0000000000",
        "env_seal_digest": "seal000000000000",
        "identity_axes": {"graph": "c0ffee0000000000", "sm": "sm_89"},
        "sm": "sm_89",
        "sku": "l40s",
        "lane": "bf16",
        "receipt": "eyJ.aGVhZA.c2ln",
        "transport": {
            "snapshot_digest": "sha256:" + "22" * 32,
            "files": [{
                "path": "cell.tar.gz",
                "size_bytes": 4096,
                "digest": "sha256:" + "11" * 32,
                "url": "https://cas.example/cell.tar.gz",
            }],
        },
    }
    body.update(over)
    return body


@pytest.fixture
def stub(monkeypatch):
    """Capture what the client SENDS and control what it receives."""
    sent: Dict[str, Any] = {}

    def _request(method: str, path: str, **kw: Any) -> Any:
        sent.update({"method": method, "path": path, **kw})
        return sent.pop("_resp", None) or _resp["r"]

    _resp: Dict[str, Any] = {"r": _Resp(200, {"found": False})}
    monkeypatch.setattr(cell_resolve.broker, "request", _request)
    return sent, _resp


# ---------------------------------------------------------------------------
# The body carries the key and NOTHING else
# ---------------------------------------------------------------------------


def test_the_body_carries_family_and_cell_key_and_nothing_else(stub) -> None:
    """A body naming any entitlement input is a NAMED 400 hub-side
    (``cell_resolve_client_supplied_field``) — refused, never ignored. So a
    client that grew a field would not be tolerated, it would refuse the whole
    boot. Pinned here and in the action table, which are the two places the
    shape is stated."""
    sent, resp = stub
    resp["r"] = _Resp(200, {"found": False})
    cell_resolve.resolve("sdxl", KEY, base_url="https://hub", bearer="t")

    assert sent["method"] == "POST"
    assert sent["path"] == cell_resolve.RESOLVE_PATH == "/v1/worker/cells/resolve"
    assert sent["json"] == {"family": "sdxl", "cell_key": KEY}
    assert sent["timeout"] == cell_resolve.RESOLVE_TIMEOUT_S


def test_the_action_table_admits_exactly_the_two_fields() -> None:
    action = actions_mod.ACTIONS["cells.resolve"]
    assert action.method == "POST"
    assert action.body == frozenset({"family", "cell_key"})
    assert action.path.match("/v1/worker/cells/resolve")
    assert not action.path.match("/v1/worker/cells/resolve/extra")
    assert action.timeout_s > 0        # scripts/lint_http_timeouts.py
    assert not action.scoped_to_job    # a boot has no attempt (§4.16)


def test_the_resolve_action_is_not_a_publish_action() -> None:
    """``PUBLISH_ACTIONS`` gates the two actions that WRITE into a shared
    family namespace. Resolve reads; a probe pod must still be able to adopt."""
    assert "cells.resolve" not in actions_mod.PUBLISH_ACTIONS


# ---------------------------------------------------------------------------
# MISS is one shape; refusals are typed and are NOT misses
# ---------------------------------------------------------------------------


def test_a_miss_is_none(stub) -> None:
    _sent, resp = stub
    resp["r"] = _Resp(200, {"found": False})
    assert cell_resolve.resolve("sdxl", KEY) is None


@pytest.mark.parametrize("code,status", [
    ("cell_resolve_ambiguous", 409),
    ("cell_resolve_incomplete", 409),
    ("cell_resolve_transport_unavailable", 503),
    ("cell_resolve_client_supplied_field", 400),
])
def test_a_typed_refusal_is_never_read_as_a_miss(stub, code, status) -> None:
    """A pod that read ``cell_resolve_incomplete`` as "no cell" would go pay
    for a full cold mint believing the hub holds nothing, which is false and
    expensive. Every one of the hub's four refusal classes raises."""
    _sent, resp = stub
    resp["r"] = _Resp(status, {"code": code, "message": "because"})
    with pytest.raises(cell_resolve.CellResolveRefused) as err:
        cell_resolve.resolve("sdxl", KEY)
    assert err.value.code == code
    assert err.value.status == status
    assert code in cell_resolve.REFUSAL_CODES


def test_an_answer_naming_a_different_key_is_refused_not_adopted(stub) -> None:
    _sent, resp = stub
    resp["r"] = _Resp(200, _hit_body(cell_key=OTHER_KEY))
    with pytest.raises(cell_resolve.CellResolveRefused) as err:
        cell_resolve.resolve("sdxl", KEY)
    assert err.value.code == "cell_resolve_key_mismatch"


#: ``cell_key`` is caught one gate earlier — an empty key is not the key that
#: was asked for, so the mismatch gate names the seam that lied first.
_EARLIER_GATE = {"cell_key": "cell_resolve_key_mismatch"}


@pytest.mark.parametrize("field", [f for f, _ in cell_resolve._REQUIRED])
def test_an_incomplete_answer_is_refused_before_the_cell_is_paid_for(
    stub, field,
) -> None:
    """A 200 hit that leaves an admission field unnamed is REFUSED here.

    The Plan route cannot reach the identity gate incomplete — ``plan._artifact``
    ``_require``s every counterpart — so before pgw#1152 this route was the only
    one that could, and the gate that would have caught it runs after
    ``materialize`` has already downloaded the whole cell.
    """
    _sent, resp = stub
    resp["r"] = _Resp(200, _hit_body(**{field: ""}))
    with pytest.raises(cell_resolve.CellResolveRefused) as err:
        cell_resolve.resolve("sdxl", KEY)
    assert err.value.code == _EARLIER_GATE.get(field, "cell_resolve_incomplete")
    assert field in err.value.detail or field == "cell_key"


def test_every_expectation_axis_is_required_of_the_answer() -> None:
    """An axis on ``ExpectedIdentity`` that the answer need not name is an
    expectation this route can state as empty — and
    ``verify_declared_identity`` refuses an empty expectation only AFTER the
    bytes are paid for. Derived from the struct, so a new axis reds here."""
    required = {f for f, _ in cell_resolve._REQUIRED}
    # the answer spells the graph axis without the `_digest` suffix; every
    # other axis is copied by name in `ExpectedIdentity.named_by`
    answer_name = {"graph_contract_digest": "graph_contract"}
    for axis in aot_identity.ExpectedIdentity.__struct_fields__:
        assert answer_name.get(axis, axis) in required, axis


def test_an_incomplete_answer_is_a_typed_refusal_not_a_miss() -> None:
    """A pod that read it as "no cell" would go pay for a whole cold mint
    believing the hub holds nothing, which is false and expensive."""
    assert "cell_resolve_incomplete" in cell_resolve.REFUSAL_CODES


def test_a_non_key_is_refused_before_the_hub_is_dialled(stub) -> None:
    sent, _resp = stub
    for bad in ("", "sdxl", "arm1-" + "ab" * 28, "ck1-short"):
        with pytest.raises(cell_resolve.CellResolveRefused):
            cell_resolve.resolve("sdxl", bad)
    assert not sent  # nothing was sent


def test_a_missing_family_is_refused_before_the_hub_is_dialled(stub) -> None:
    sent, _resp = stub
    with pytest.raises(cell_resolve.CellResolveRefused):
        cell_resolve.resolve("", KEY)
    assert not sent


# ---------------------------------------------------------------------------
# A hit feeds the EXISTING admission machinery, never a second brain
# ---------------------------------------------------------------------------


def test_a_hit_builds_the_same_expected_identity_the_plan_path_builds(
    stub,
) -> None:
    """``ExpectedIdentity`` is the type ``aot_identity.expected_from_plan``
    produces, so an artifact reaching ``verify_declared_identity`` cannot tell
    whether it was named by a Plan or pulled by key. That is the property that
    keeps this from being a second admission brain."""
    _sent, resp = stub
    resp["r"] = _Resp(200, _hit_body())
    cell = cell_resolve.resolve("sdxl", KEY)
    assert cell is not None

    expected = cell.expected_identity()

    assert isinstance(expected, aot_identity.ExpectedIdentity)
    assert expected.cell_key == KEY
    assert expected.toolchain_digest == "toolch0000000000"
    assert expected.env_seal_digest == "seal000000000000"
    assert expected.graph_contract_digest == "c0ffee0000000000"
    assert expected.publisher_org == "org-a"

    # And it is COMPARABLE: an artifact whose recorded facts match passes the
    # existing gate, and one that does not is refused BY AXIS.
    meta = {
        "cell_key": KEY,
        "toolchain": {"torch": "x"},
        "env_seal": {"a": 1},
        "combined_graph_hash": "c0ffee0000000000",
    }
    meta["toolchain_digest_expected"] = ck.facts_digest(meta["toolchain"])
    reason = aot_identity.verify_declared_identity(
        meta,
        aot_identity.ExpectedIdentity(
            cell_key=KEY,
            toolchain_digest=ck.facts_digest(meta["toolchain"]),
            env_seal_digest=env_seal.seal_digest(meta["env_seal"]),
            graph_contract_digest="c0ffee0000000000",
            publisher_org="org-a"))
    assert reason == ""


def _plan_naming_the_same_artifact() -> Any:
    """A Plan whose arm names the artifact ``_hit_body`` describes."""
    arm = pb.Arm(
        graph_contract_digest="c0ffee0000000000",
        shape=pb.ARM_SHAPE_BRANCHLESS,
        backend=pb.STEADY_BACKEND_AOT_CELL,
    )
    arm.artifact.CopyFrom(pb.ArtifactIdentity(
        cell_ref=f"root/family-sdxl#{KEY}",
        content_digest="sha256:" + "11" * 32,
        cell_key=KEY,
        publisher_org="org-a",
        toolchain_digest="toolch0000000000",
        env_seal_digest="seal000000000000",
    ))
    spec = pb.ExecutionSpec(
        digest="sha256:" + "a" * 64,
        spec_version=1,
        release=pb.EndpointRelease(
            org="org-a", endpoint="sdxl", release_id="r1",
            image_digest="sha256:img", code_closure_id="clo_01"),
        function_name="txt2img",
        numerical_lane=pb.ExecutionLane(weights=pb.WEIGHT_LANE_BF16),
        arm=arm,
        topology=pb.Topology(accelerator="cuda", gpu_count=1, execution_groups=1),
        components=pb.ComponentManifest(slots=[pb.SlotBinding(
            slot="unet", ref="org-a/sdxl@v1", snapshot_digest="sha256:d")]),
    )
    return PlanFactory.from_execution_spec(AttemptRef("req", 1), spec)


def test_both_expectation_sources_produce_an_EQUAL_object(stub) -> None:
    """Not "the same type" — the same VALUE, compared whole.

    pgw#1150's defect was a field that reached one map and not the other while
    both still produced the right type. Since pgw#1152 there is one map
    (``ExpectedIdentity.named_by``), so this row is what proves the two sources
    have not drifted apart again: a new axis copied on one side only reds here
    without anyone remembering to extend an assertion list.
    """
    _sent, resp = stub
    resp["r"] = _Resp(200, _hit_body())
    cell = cell_resolve.resolve("sdxl", KEY)
    assert cell is not None

    from_plan = aot_identity.expected_from_plan(_plan_naming_the_same_artifact())
    assert from_plan is not None
    assert cell.expected_identity() == from_plan


def test_the_receipt_rides_the_answer_and_is_never_re_fetched(stub) -> None:
    """th#1680: ``handleWorkerCellReceipt`` scopes by ENDPOINT while resolve
    scopes by ORG, so a second fetch for the same cell could 403 what resolve
    just offered. The receipt therefore has to come from the answer — and this
    module must contain no receipt fetch at all."""
    _sent, resp = stub
    resp["r"] = _Resp(200, _hit_body())
    cell = cell_resolve.resolve("sdxl", KEY)
    assert cell is not None and cell.receipt == "eyJ.aGVhZA.c2ln"

    import inspect

    source = inspect.getsource(cell_resolve)
    assert "cells/receipt" not in source
    assert "receipt_for" not in source


def test_the_transport_is_shaped_for_the_existing_delivery_path(stub) -> None:
    """``materialize_named_artifact`` reads the grant by ATTRIBUTE. The answer's
    transport must present the same attributes or the resolve feeds a parallel
    downloader instead of the one with the digest checks in it."""
    _sent, resp = stub
    resp["r"] = _Resp(200, _hit_body())
    cell = cell_resolve.resolve("sdxl", KEY)
    assert cell is not None

    files = list(cell.transport.files)
    assert len(files) == 1
    entry = files[0]
    for attr in ("path", "size_bytes", "digest", "url", "chunks",
                 "chunk_size_bytes"):
        assert hasattr(entry, attr), attr
    assert str(entry.path).endswith(".tar.gz")


def test_a_chunked_transport_carries_the_chunk_attributes(stub) -> None:
    _sent, resp = stub
    body = _hit_body()
    body["transport"]["files"][0]["chunks"] = [
        {"sha256": "aa" * 32, "url": "https://cas/0", "len": 2048},
        {"sha256": "bb" * 32, "url": "https://cas/1", "len": 2048},
    ]
    body["transport"]["files"][0]["chunk_size_bytes"] = 2048
    resp["r"] = _Resp(200, body)
    cell = cell_resolve.resolve("sdxl", KEY)
    assert cell is not None
    chunks = list(cell.transport.files[0].chunks)
    assert [c.len for c in chunks] == [2048, 2048]
    assert chunks[0].sha256 == "aa" * 32
    assert cell.transport.files[0].chunk_size_bytes == 2048


def test_materialize_delegates_and_adds_nothing(monkeypatch, stub) -> None:
    """Deliberately a two-line function: a second downloader is a second place
    for "verified" to mean something slightly different."""
    _sent, resp = stub
    resp["r"] = _Resp(200, _hit_body())
    cell = cell_resolve.resolve("sdxl", KEY)
    assert cell is not None

    from gen_worker import aot_delivery

    seen: Dict[str, Any] = {}

    def _materialize(ref, digest, presigned, *, cache_dir, what):
        seen.update(ref=ref, digest=digest, presigned=presigned,
                    cache_dir=cache_dir, what=what)
        from pathlib import Path

        return Path("/tmp/cell.tar.gz")

    monkeypatch.setattr(
        aot_delivery, "materialize_named_artifact", _materialize)
    out = cell_resolve.materialize(cell, cache_dir=None)
    assert str(out).endswith("cell.tar.gz")
    assert seen["ref"] == cell.cell_ref
    assert seen["digest"] == cell.content_digest
    assert seen["presigned"] is cell.transport


# ---------------------------------------------------------------------------
# The boot-adopt orchestration: every failure degrades, none is fatal
# ---------------------------------------------------------------------------


def _attempt(monkeypatch, tmp_path, *, derive=None, resolve=None,
             materialize=None) -> Any:
    from gen_worker import boot_adopt, boot_key

    if derive is not None:
        monkeypatch.setattr(boot_key, "derive", derive)
    if resolve is not None:
        monkeypatch.setattr(cell_resolve, "resolve", resolve)
    if materialize is not None:
        monkeypatch.setattr(cell_resolve, "materialize", materialize)

    class _Cfg:
        family = "sdxl"
        targets = ("unet",)
        shapes = ((1024, 1024),)
        text_lens = (77,)
        guidance_scales = (7.5,)
        lora_bucket = 0

    return boot_adopt.attempt(
        function="txt2img", modules=("m",), cfg=_Cfg(), slots={},
        declared_hint=3,
        envelope={"shapes": [[1024, 1024]], "text_lens": [77],
                  "guidance": [7.5]},
        work_root=tmp_path)


def _derived(digest: str = KEY) -> Any:
    from gen_worker import boot_key, cell_key as ck

    return boot_key.DerivedKey(
        key=ck.from_axes({
            "graph": "c0ffee0000000000", "envelope": "e" * 16,
            "sm": "sm_89", "toolchain": "t" * 16}),
        class_hashes={"a": "0" * 16}, combined="c0ffee0000000000",
        workers=2, width_reason="test", traced=1, memo="miss", wall_ms=1234)


def test_an_underivable_key_degrades_with_the_reason(
    monkeypatch, tmp_path,
) -> None:
    from gen_worker import boot_key

    def _boom(**_kw):
        raise boot_key.BootKeyUnavailable(
            "structure_unsupported", "MicroEscapeDenoiser has no from_config")

    out = _attempt(monkeypatch, tmp_path, derive=_boom)
    assert not out.adopted
    assert out.reason == "structure_unsupported"
    assert "from_config" in out.detail


def test_a_hub_refusal_degrades_but_keeps_its_own_token(
    monkeypatch, tmp_path,
) -> None:
    def _refuse(*_a, **_k):
        raise cell_resolve.CellResolveRefused(
            "cell_resolve_ambiguous", "two rows", status=409)

    out = _attempt(
        monkeypatch, tmp_path,
        derive=lambda **_kw: _derived(), resolve=_refuse)
    assert not out.adopted
    assert out.reason == "cell_resolve_ambiguous"
    assert out.derived_key.startswith("ck1-")
    assert out.derive_ms == 1234


def test_a_miss_degrades_as_miss_not_as_a_failure(monkeypatch, tmp_path) -> None:
    out = _attempt(
        monkeypatch, tmp_path,
        derive=lambda **_kw: _derived(), resolve=lambda *_a, **_k: None)
    assert not out.adopted
    assert out.reason == "miss"
    assert out.derived_key.startswith("ck1-")


def test_a_failed_materialize_degrades_and_is_never_fatal(
    monkeypatch, tmp_path,
) -> None:
    from gen_worker import aot_delivery

    def _boom(*_a, **_k):
        raise aot_delivery.NamedArtifactUnavailable(
            "content_digest_mismatch", "bytes refused")

    class _Cell:
        publisher_org = "org-a"
        cell_ref = "root/family-sdxl#" + KEY
        publisher_tier = "platform"

    out = _attempt(
        monkeypatch, tmp_path,
        derive=lambda **_kw: _derived(),
        resolve=lambda *_a, **_k: _Cell(),
        materialize=_boom)
    assert not out.adopted
    assert out.reason == "materialize_failed"
    assert "content_digest_mismatch" in out.detail


def test_an_unexpected_exception_anywhere_still_degrades(
    monkeypatch, tmp_path,
) -> None:
    """The one thing this must never do is prevent a pod from serving."""
    def _boom(**_kw):
        raise RuntimeError("something nobody predicted")

    out = _attempt(monkeypatch, tmp_path, derive=_boom)
    assert not out.adopted
    assert out.reason == "derive_failed"
    assert "RuntimeError" in out.detail
