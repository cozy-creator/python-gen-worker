"""pgw#1090 (§4.29) — the worker half of ``POST /v1/worker/compiled graphs/resolve``.

Written against th#1750's ANSWER CONTRACT (hub side merged ``26275ff8``), so a
hub-side field rename reds here rather than on a pod. Every row is off-wire: the
broker is stubbed, which is what the ``procsplit`` seam exists to make possible.
"""

from __future__ import annotations

import json
from typing import Any, Dict

import pytest

from gen_worker import aot_identity, compiled_graph_key as ck, compiled_graph_resolve, env_seal
from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.procsplit import actions as actions_mod

KEY = "ek1-" + "ab" * 28
OTHER_KEY = "ek1-" + "cd" * 28


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
        "compiled_graph_key": KEY,
        "compiled_graph_ref": f"root/family-sdxl#{KEY}",
        "checkpoint_id": "ckpt-1",
        "content_digest": "sha256:" + "11" * 32,
        "artifact_path": "compiled_graph.tar.gz",
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
                "path": "compiled_graph.tar.gz",
                "size_bytes": 4096,
                "digest": "sha256:" + "11" * 32,
                "url": "https://cas.example/compiled_graph.tar.gz",
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
    monkeypatch.setattr(compiled_graph_resolve.broker, "request", _request)
    return sent, _resp


# ---------------------------------------------------------------------------
# The body carries the key and NOTHING else
# ---------------------------------------------------------------------------


def test_the_body_carries_family_and_compiled_graph_key_and_nothing_else(stub) -> None:
    """A body naming any entitlement input is a NAMED 400 hub-side
    (``compiled_graph_resolve_client_supplied_field``) — refused, never ignored. So a
    client that grew a field would not be tolerated, it would refuse the whole
    boot. Pinned here and in the action table, which are the two places the
    shape is stated."""
    sent, resp = stub
    resp["r"] = _Resp(200, {"found": False})
    compiled_graph_resolve.resolve("sdxl", KEY, base_url="https://hub", bearer="t")

    assert sent["method"] == "POST"
    assert sent["path"] == compiled_graph_resolve.RESOLVE_PATH == "/v1/worker/compiled_graphs/resolve"
    assert sent["json"] == {"family": "sdxl", "compiled_graph_key": KEY}
    assert sent["timeout"] == compiled_graph_resolve.RESOLVE_TIMEOUT_S


def test_the_action_table_admits_exactly_the_two_fields() -> None:
    action = actions_mod.ACTIONS["compiled_graphs.resolve"]
    assert action.method == "POST"
    assert action.body == frozenset({"family", "compiled_graph_key"})
    assert action.path.match("/v1/worker/compiled_graphs/resolve")
    assert not action.path.match("/v1/worker/compiled_graphs/resolve/extra")
    assert action.timeout_s > 0        # scripts/lint_http_timeouts.py
    assert not action.scoped_to_job    # a boot has no attempt (§4.16)


def test_the_resolve_action_is_not_a_publish_action() -> None:
    """``PUBLISH_ACTIONS`` gates the two actions that WRITE into a shared
    family namespace. Resolve reads; a probe pod must still be able to adopt."""
    assert "compiled_graphs.resolve" not in actions_mod.PUBLISH_ACTIONS


# ---------------------------------------------------------------------------
# MISS is one shape; refusals are typed and are NOT misses
# ---------------------------------------------------------------------------


def test_a_miss_is_none(stub) -> None:
    _sent, resp = stub
    resp["r"] = _Resp(200, {"found": False})
    assert compiled_graph_resolve.resolve("sdxl", KEY) is None


@pytest.mark.parametrize("code,status", [
    ("compiled_graph_resolve_ambiguous", 409),
    ("compiled_graph_resolve_incomplete", 409),
    ("compiled_graph_resolve_transport_unavailable", 503),
    ("compiled_graph_resolve_client_supplied_field", 400),
])
def test_a_typed_refusal_is_never_read_as_a_miss(stub, code, status) -> None:
    """A pod that read ``compiled_graph_resolve_incomplete`` as "no compiled graph" would go pay
    for a full cold mint believing the hub holds nothing, which is false and
    expensive. Every one of the hub's four refusal classes raises."""
    _sent, resp = stub
    resp["r"] = _Resp(status, {"code": code, "message": "because"})
    with pytest.raises(compiled_graph_resolve.CompiledGraphResolveRefused) as err:
        compiled_graph_resolve.resolve("sdxl", KEY)
    assert err.value.code == code
    assert err.value.status == status
    assert code in compiled_graph_resolve.REFUSAL_CODES


def test_an_answer_naming_a_different_key_is_refused_not_adopted(stub) -> None:
    _sent, resp = stub
    resp["r"] = _Resp(200, _hit_body(compiled_graph_key=OTHER_KEY))
    with pytest.raises(compiled_graph_resolve.CompiledGraphResolveRefused) as err:
        compiled_graph_resolve.resolve("sdxl", KEY)
    assert err.value.code == "compiled_graph_resolve_key_mismatch"


#: ``compiled_graph_key`` is caught one gate earlier — an empty key is not the key that
#: was asked for, so the mismatch gate names the seam that lied first.
_EARLIER_GATE = {"compiled_graph_key": "compiled_graph_resolve_key_mismatch"}


@pytest.mark.parametrize("field", [f for f, _ in compiled_graph_resolve._REQUIRED])
def test_an_incomplete_answer_is_refused_before_the_compiled_graph_is_paid_for(
    stub, field,
) -> None:
    """A 200 hit that leaves an admission field unnamed is REFUSED here.

    The gate that would otherwise catch it runs after ``materialize`` has
    already downloaded the whole compiled graph, so an unnamed admission field has to
    refuse HERE or it is paid for first.
    """
    _sent, resp = stub
    resp["r"] = _Resp(200, _hit_body(**{field: ""}))
    with pytest.raises(compiled_graph_resolve.CompiledGraphResolveRefused) as err:
        compiled_graph_resolve.resolve("sdxl", KEY)
    assert err.value.code == _EARLIER_GATE.get(field, "compiled_graph_resolve_incomplete")
    assert field in err.value.detail or field == "compiled_graph_key"


def test_every_expectation_axis_is_required_of_the_answer() -> None:
    """An axis on ``ExpectedIdentity`` that the answer need not name is an
    expectation this route can state as empty — and
    ``verify_declared_identity`` refuses an empty expectation only AFTER the
    bytes are paid for. Derived from the struct, so a new axis reds here."""
    required = {f for f, _ in compiled_graph_resolve._REQUIRED}
    # the answer spells the graph axis without the `_digest` suffix; every
    # other axis is copied by name in `ExpectedIdentity.named_by`
    answer_name = {"graph_contract_digest": "graph_contract"}
    for axis in aot_identity.ExpectedIdentity.__struct_fields__:
        assert answer_name.get(axis, axis) in required, axis


def test_an_incomplete_answer_is_a_typed_refusal_not_a_miss() -> None:
    """A pod that read it as "no compiled graph" would go pay for a whole cold mint
    believing the hub holds nothing, which is false and expensive."""
    assert "compiled_graph_resolve_incomplete" in compiled_graph_resolve.REFUSAL_CODES


def test_a_non_key_is_refused_before_the_hub_is_dialled(stub) -> None:
    sent, _resp = stub
    # `ck1-short` is refused for its LENGTH, which is why the WELL-FORMED ck1
    # key sits beside it: pgw#1176 re-keyed the grammar, and a `ck1` key names
    # a 36-compiled graph all-or-nothing compiled graph this runtime cannot arm at all. It must
    # fail HERE, at the comparison, rather than late inside a per-compiled graph path —
    # so the prefix, not the shape, has to be what refuses it.
    for bad in ("", "sdxl", "arm1-" + "ab" * 28, "ck1-short",
                "ck1-" + "0" * 56):
        with pytest.raises(compiled_graph_resolve.CompiledGraphResolveRefused):
            compiled_graph_resolve.resolve("sdxl", bad)
    assert not sent  # nothing was sent


def test_a_missing_family_is_refused_before_the_hub_is_dialled(stub) -> None:
    sent, _resp = stub
    with pytest.raises(compiled_graph_resolve.CompiledGraphResolveRefused):
        compiled_graph_resolve.resolve("", KEY)
    assert not sent


# ---------------------------------------------------------------------------
# A hit feeds the EXISTING admission machinery, never a second brain
# ---------------------------------------------------------------------------


def test_a_hit_builds_the_expected_identity_the_arm_gate_COMPARES(
    stub,
) -> None:
    """``ExpectedIdentity`` is what ``verify_declared_identity`` compares, and
    a resolve answer produces it through THE one map — which is what keeps this
    from being a second admission brain.

    Until pgw#1206 D this row read "the same object the Plan path builds"; the
    Plan head is gone and this is the only naming source left, so the guard is
    the pinned VALUE below plus the ``named_by`` equality: a new axis copied
    wrong, or silently defaulted, reds here."""
    _sent, resp = stub
    resp["r"] = _Resp(200, _hit_body())
    compiled_graph = compiled_graph_resolve.resolve("sdxl", KEY)
    assert compiled_graph is not None

    expected = compiled_graph.expected_identity()

    assert isinstance(expected, aot_identity.ExpectedIdentity)
    assert expected.compiled_graph_key == KEY
    assert expected.toolchain_digest == "toolch0000000000"
    assert expected.env_seal_digest == "seal000000000000"
    assert expected.graph_contract_digest == "c0ffee0000000000"
    assert expected.publisher_org == "org-a"
    # ...and it is THE map that built it, not a second one written here.
    assert expected == aot_identity.ExpectedIdentity.named_by(
        compiled_graph, "c0ffee0000000000")

    # And it is COMPARABLE: an artifact whose recorded facts match passes the
    # existing gate, and one that does not is refused BY AXIS.
    meta = {
        "compiled_graph_key": KEY,
        "toolchain": {"torch": "x"},
        "env_seal": {"a": 1},
        # pgw#1176: the declaration-wide coverage label. Same arithmetic as
        # `combined_graph_hash`, demoted from identity — identity is
        # `compiled_graph_key`, per compiled graph.
        "manifest_digest": "c0ffee0000000000",
    }
    meta["toolchain_digest_expected"] = ck.facts_digest(meta["toolchain"])
    reason = aot_identity.verify_declared_identity(
        meta,
        aot_identity.ExpectedIdentity(
            compiled_graph_key=KEY,
            toolchain_digest=ck.facts_digest(meta["toolchain"]),
            env_seal_digest=env_seal.seal_digest(meta["env_seal"]),
            graph_contract_digest="c0ffee0000000000",
            publisher_org="org-a"))
    assert reason == ""


def test_the_receipt_rides_the_answer_and_is_never_re_fetched(stub) -> None:
    """th#1680: ``handleWorkerCompiledGraphReceipt`` scopes by ENDPOINT while resolve
    scopes by ORG, so a second fetch for the same compiled graph could 403 what resolve
    just offered. The receipt therefore has to come from the answer — and this
    module must contain no receipt fetch at all."""
    _sent, resp = stub
    resp["r"] = _Resp(200, _hit_body())
    compiled_graph = compiled_graph_resolve.resolve("sdxl", KEY)
    assert compiled_graph is not None and compiled_graph.receipt == "eyJ.aGVhZA.c2ln"

    import inspect

    source = inspect.getsource(compiled_graph_resolve)
    assert "compiled_graphs/receipt" not in source
    assert "receipt_for" not in source


def test_the_transport_is_shaped_for_the_existing_delivery_path(stub) -> None:
    """``materialize_named_artifact`` reads the grant by ATTRIBUTE. The answer's
    transport must present the same attributes or the resolve feeds a parallel
    downloader instead of the one with the digest checks in it."""
    _sent, resp = stub
    resp["r"] = _Resp(200, _hit_body())
    compiled_graph = compiled_graph_resolve.resolve("sdxl", KEY)
    assert compiled_graph is not None

    files = list(compiled_graph.transport.files)
    assert len(files) == 1
    compiled_graph = files[0]
    for attr in ("path", "size_bytes", "digest", "url", "chunks",
                 "chunk_size_bytes"):
        assert hasattr(compiled_graph, attr), attr
    assert str(compiled_graph.path).endswith(".tar.gz")


def test_a_chunked_transport_carries_the_chunk_attributes(stub) -> None:
    _sent, resp = stub
    body = _hit_body()
    body["transport"]["files"][0]["chunks"] = [
        {"sha256": "aa" * 32, "url": "https://cas/0", "len": 2048},
        {"sha256": "bb" * 32, "url": "https://cas/1", "len": 2048},
    ]
    body["transport"]["files"][0]["chunk_size_bytes"] = 2048
    resp["r"] = _Resp(200, body)
    compiled_graph = compiled_graph_resolve.resolve("sdxl", KEY)
    assert compiled_graph is not None
    chunks = list(compiled_graph.transport.files[0].chunks)
    assert [c.len for c in chunks] == [2048, 2048]
    assert chunks[0].sha256 == "aa" * 32
    assert compiled_graph.transport.files[0].chunk_size_bytes == 2048


def test_materialize_delegates_and_adds_nothing(monkeypatch, stub) -> None:
    """Deliberately a two-line function: a second downloader is a second place
    for "verified" to mean something slightly different."""
    _sent, resp = stub
    resp["r"] = _Resp(200, _hit_body())
    compiled_graph = compiled_graph_resolve.resolve("sdxl", KEY)
    assert compiled_graph is not None

    from gen_worker import aot_delivery

    seen: Dict[str, Any] = {}

    def _materialize(ref, digest, presigned, *, cache_dir, what):
        seen.update(ref=ref, digest=digest, presigned=presigned,
                    cache_dir=cache_dir, what=what)
        from pathlib import Path

        return Path("/tmp/compiled_graph.tar.gz")

    monkeypatch.setattr(
        aot_delivery, "materialize_named_artifact", _materialize)
    out = compiled_graph_resolve.materialize(compiled_graph, cache_dir=None)
    assert str(out).endswith("compiled_graph.tar.gz")
    assert seen["ref"] == compiled_graph.compiled_graph_ref
    assert seen["digest"] == compiled_graph.content_digest
    assert seen["presigned"] is compiled_graph.transport


# ---------------------------------------------------------------------------
# The boot-adopt orchestration: every failure degrades, none is fatal
# ---------------------------------------------------------------------------


def _attempt(monkeypatch, tmp_path, *, derive=None, resolve=None,
             materialize=None) -> Any:
    from gen_worker import boot_adopt, boot_key

    if derive is not None:
        monkeypatch.setattr(boot_key, "derive", derive)
    if resolve is not None:
        monkeypatch.setattr(compiled_graph_resolve, "resolve", resolve)
    if materialize is not None:
        monkeypatch.setattr(compiled_graph_resolve, "materialize", materialize)

    class _Cfg:
        family = "sdxl"
        targets = ("unet",)
        shapes = ((1024, 1024),)
        text_lens = (77,)
        guidance_scales = (7.5,)
        lora_bucket = 0

    # pgw#1176: `attempt` returns ONE outcome per declared graph class, and a
    # derivation failure is a single-element tuple. Every declaration below
    # traces one class, so the unpack ASSERTS that arity for all five callers
    # rather than indexing past a set nobody checked.
    (out,) = boot_adopt.attempt(
        function="txt2img", modules=("m",), cfg=_Cfg(), slots={},
        declared_hint=3,
        envelope={"shapes": [[1024, 1024]], "text_lens": [77],
                  "guidance": [7.5]},
        work_root=tmp_path)
    return out


def _derived(digest: str = KEY) -> Any:
    from gen_worker import boot_key, compiled_graph_key as ck

    return boot_key.DerivedKey(
        # pgw#1176: a boot derives a KEY SET. These declarations trace to one
        # class, so the set has one member and callers take it from `keys`.
        compiled_graph_keys={"a": ck.from_axes({
            "graph": "c0ffee0000000000",
            "sm": "sm_89", "toolchain": "t" * 16}).digest},
        class_hashes={"a": "c0ffee0000000000"},
        manifest=ck.manifest_digest(["c0ffee0000000000"]),
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
        raise compiled_graph_resolve.CompiledGraphResolveRefused(
            "compiled_graph_resolve_ambiguous", "two rows", status=409)

    out = _attempt(
        monkeypatch, tmp_path,
        derive=lambda **_kw: _derived(), resolve=_refuse)
    assert not out.adopted
    assert out.reason == "compiled_graph_resolve_ambiguous"
    assert out.derived_key.startswith("ek1-")
    assert out.derive_ms == 1234


def test_a_miss_degrades_as_miss_not_as_a_failure(monkeypatch, tmp_path) -> None:
    out = _attempt(
        monkeypatch, tmp_path,
        derive=lambda **_kw: _derived(), resolve=lambda *_a, **_k: None)
    assert not out.adopted
    assert out.reason == "miss"
    assert out.derived_key.startswith("ek1-")


def test_a_failed_materialize_degrades_and_is_never_fatal(
    monkeypatch, tmp_path,
) -> None:
    from gen_worker import aot_delivery

    def _boom(*_a, **_k):
        raise aot_delivery.NamedArtifactUnavailable(
            "content_digest_mismatch", "bytes refused")

    class _CompiledGraph:
        publisher_org = "org-a"
        compiled_graph_ref = "root/family-sdxl#" + KEY
        publisher_tier = "platform"

    out = _attempt(
        monkeypatch, tmp_path,
        derive=lambda **_kw: _derived(),
        resolve=lambda *_a, **_k: _CompiledGraph(),
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
