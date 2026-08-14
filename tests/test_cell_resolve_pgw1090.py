"""pgw#1090 (§4.29), batched by pgw#1224 — the worker half of
``POST /v1/worker/compiled-graphs/resolve``.

Written against the hub's ANSWER CONTRACT (th#1750 merged ``26275ff8``;
th#1842 PR #1118 made it a batch), so a hub-side field rename reds here rather
than on a pod. Every row is off-wire: the broker is stubbed, which is what the
``procsplit`` seam exists to make possible.

The subject is now ``{family, keys[]}`` -> one answer per key, in request
order, each independently signed. The single-key wire is GONE; the rows that
drove it drive one-key BATCHES, and the properties that only a batch can have
— arity, order, per-answer signing, and a per-key fault not sinking the batch —
have rows of their own below.
"""

from __future__ import annotations

import json
from typing import Any, Dict

import pytest

from gen_worker import cell_resolve
from gen_worker.procsplit import actions as actions_mod

KEY = "cg-key-v1-" + "ab" * 28
OTHER_KEY = "cg-key-v1-" + "cd" * 28
THIRD_KEY = "cg-key-v1-" + "ef" * 28
_VERIFIED_RECEIPT = object()


class _Resp:
    def __init__(self, status: int, body: Any) -> None:
        self.status_code = status
        self.text = json.dumps(body) if not isinstance(body, str) else body

    def json(self) -> Any:
        return json.loads(self.text) if self.text else {}


def _hit_body(**over: Any) -> Dict[str, Any]:
    body = {
        "status": "hit",
        "found": True,
        "family": "sdxl",
        "compiled_graph_key": KEY,
        "compiled_graph_ref": f"root/family-sdxl#{KEY}",
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


def _batch(*answers: Any, **top: Any) -> Dict[str, Any]:
    """The hub's envelope. ``answers`` is the contract; the counts are logs."""
    body: Dict[str, Any] = {
        "object": "compiled_graph_resolve_batch",
        "family": "sdxl",
        "answers": list(answers),
        "hits": sum(1 for a in answers if a.get("status") == "hit"),
        "misses": sum(1 for a in answers if a.get("status") == "miss"),
    }
    body.update(top)
    return body


def _miss(key: str = KEY) -> Dict[str, Any]:
    return {"compiled_graph_key": key, "status": "miss", "found": False}


def _one(family: str = "sdxl", key: str = KEY, **kw: Any) -> Any:
    """One key through the BATCH wire — the shape a single-key caller has."""
    return cell_resolve.resolve_batch(family, [key], **kw)[0]


@pytest.fixture
def stub(monkeypatch):
    """Capture what the client SENDS and control what it receives."""
    sent: Dict[str, Any] = {}

    def _request(method: str, path: str, **kw: Any) -> Any:
        sent.update({"method": method, "path": path, **kw})
        return sent.pop("_resp", None) or _resp["r"]

    _resp: Dict[str, Any] = {"r": _Resp(200, _batch(_miss()))}
    monkeypatch.setattr(cell_resolve.broker, "request", _request)
    monkeypatch.setattr(
        cell_resolve.receipts,
        "verify_receipt",
        lambda *_a, **_kw: _VERIFIED_RECEIPT,
    )
    return sent, _resp


# ---------------------------------------------------------------------------
# The body carries the key and NOTHING else
# ---------------------------------------------------------------------------


def test_the_body_carries_family_and_keys_and_nothing_else(stub) -> None:
    """A body naming any entitlement input is a NAMED 400 hub-side
    (``cell_resolve_client_supplied_field``) — refused, never ignored. So a
    client that grew a field would not be tolerated, it would refuse the whole
    boot. Pinned here and in the action table, which are the two places the
    shape is stated.

    pgw#1224: the field is ``keys[]`` and the single-key ``cell_key`` is GONE.
    A hard cut needs both halves to be checkable, and this is the worker half."""
    sent, resp = stub
    resp["r"] = _Resp(200, _batch(_miss(), _miss(OTHER_KEY)))
    cell_resolve.resolve_batch(
        "sdxl", [KEY, OTHER_KEY], base_url="https://hub", bearer="t")

    assert sent["method"] == "POST"
    assert sent["path"] == cell_resolve.RESOLVE_PATH == (
        "/v1/worker/compiled-graphs/resolve"
    )
    assert sent["json"] == {"family": "sdxl", "keys": [KEY, OTHER_KEY]}
    assert sent["timeout"] == cell_resolve.RESOLVE_TIMEOUT_S


def test_the_single_key_wire_is_gone(stub) -> None:
    """HARD CUT. A client that could still speak ``{family, cell_key}`` would
    make the hub's own hard cut unprovable — the two halves have to land as one
    fleet, and a surviving alias is how "both sides landed" becomes false while
    every test stays green."""
    assert not hasattr(cell_resolve, "resolve")
    assert "cell_key" not in actions_mod.ACTIONS["compiled_graphs.resolve"].body


def test_the_action_table_admits_exactly_the_two_fields() -> None:
    action = actions_mod.ACTIONS["compiled_graphs.resolve"]
    assert action.method == "POST"
    assert action.body == frozenset({"family", "keys"})
    assert action.path.match("/v1/worker/compiled-graphs/resolve")
    assert not action.path.match("/v1/worker/compiled-graphs/resolve/extra")
    assert action.timeout_s > 0        # scripts/lint_http_timeouts.py
    assert not action.scoped_to_job    # a boot has no attempt (§4.16)


def test_the_resolve_action_is_not_a_publish_action() -> None:
    """``PUBLISH_ACTIONS`` gates the two actions that WRITE into a shared
    family namespace. Resolve reads; a probe pod must still be able to adopt."""
    assert "compiled_graphs.resolve" not in actions_mod.PUBLISH_ACTIONS


# ---------------------------------------------------------------------------
# MISS is one shape; refusals are typed and are NOT misses
# ---------------------------------------------------------------------------


def test_a_miss_is_an_answer_never_an_omission(stub) -> None:
    """A miss is a full answer in its own slot. An OMITTED answer would be
    indistinguishable from a truncated response, and the pod answers a miss by
    paying for a full cold mint — so a dropped answer is not a smaller reply,
    it is a wrong one that costs money."""
    _sent, resp = stub
    resp["r"] = _Resp(200, _batch(_miss()))
    answer = _one()
    assert answer.compiled_graph is None and answer.miss
    assert answer.compiled_graph_key == KEY
    assert answer.refusal_code == ""


@pytest.mark.parametrize("code,status", [
    ("compiled_graph_resolve_client_supplied_field", 400),
    ("compiled_graph_resolve_too_many_keys", 400),
])
def test_a_whole_batch_refusal_still_raises(stub, code, status) -> None:
    """The refusals that survive as whole-request are the ones that are
    properties of the CALLER or the REQUEST — they are wrong for every key, and
    answering them per key would report one hub fault as 256 misses."""
    _sent, resp = stub
    resp["r"] = _Resp(status, {"code": code, "message": "because"})
    with pytest.raises(cell_resolve.CompiledGraphResolveRefused) as err:
        _one()
    assert err.value.code == code
    assert err.value.status == status
    assert code in cell_resolve.VERDICT_CODES


@pytest.mark.parametrize("status,code", [
    ("incomplete", "compiled_graph_resolve_incomplete"),
    ("transport_unavailable", "compiled_graph_resolve_transport_unavailable"),
])
def test_a_per_key_fault_is_an_answer_and_keeps_its_own_token(
    stub, status, code,
) -> None:
    """These three were WHOLE-REQUEST refusals; under the atom one duplicated
    row would strand an entire boot's adoption — 35 good answers thrown away.
    They are per-answer STATUSES now, carrying the SAME code so the caller's
    vocabulary did not move. And none of them is a miss: a pod that read one as
    "no cell" would go pay for a full cold mint over a row the hub HAS."""
    _sent, resp = stub
    resp["r"] = _Resp(200, _batch(
        {"compiled_graph_key": KEY, "status": status, "found": False,
         "detail": "because"}))
    answer = _one()
    assert answer.compiled_graph is None
    assert not answer.miss
    assert answer.refusal_code == code
    assert code in cell_resolve.VERDICT_CODES


def test_an_unknown_status_is_refused_and_never_read_as_a_miss(stub) -> None:
    """RED (default an unrecognised status to miss): the day the hub learns to
    name a new fault, every pod silently self-mints over it."""
    _sent, resp = stub
    resp["r"] = _Resp(200, _batch(
        {"compiled_graph_key": KEY, "status": "something_new", "found": False}))
    with pytest.raises(cell_resolve.CompiledGraphResolveRefused) as err:
        _one()
    assert err.value.code == "compiled_graph_resolve_unknown_status"


def test_an_answer_naming_a_different_key_is_refused_not_adopted(stub) -> None:
    """Answers are consumed POSITIONALLY, so an answer whose echo does not
    match its slot is refused rather than adopted: a transposed batch would arm
    every class with a sibling's kernels."""
    _sent, resp = stub
    resp["r"] = _Resp(200, _batch(_hit_body(compiled_graph_key=OTHER_KEY)))
    with pytest.raises(cell_resolve.CompiledGraphResolveRefused) as err:
        _one()
    assert err.value.code == "compiled_graph_resolve_answer_out_of_order"


#: ``compiled_graph_key`` is caught one gate earlier — an empty echo is not the key that
#: was asked for, so the ORDER gate names the seam that lied first.
_EARLIER_GATE = {
    "compiled_graph_key": "compiled_graph_resolve_answer_out_of_order"
}


@pytest.mark.parametrize("field", [f for f, _ in cell_resolve._REQUIRED])
def test_an_incomplete_answer_is_refused_before_the_cell_is_paid_for(
    stub, field,
) -> None:
    """A hit that leaves an admission field unnamed is REFUSED here.

    The gate that would otherwise catch it runs after ``materialize`` has
    already downloaded the whole cell, so an unnamed admission field has to
    refuse HERE or it is paid for first. PER ANSWER: it costs this graph class
    and not its 35 siblings.
    """
    _sent, resp = stub
    resp["r"] = _Resp(200, _batch(_hit_body(**{field: ""})))
    if field == "compiled_graph_key":
        with pytest.raises(cell_resolve.CompiledGraphResolveRefused) as err:
            _one()
        assert err.value.code == _EARLIER_GATE[field]
        return
    answer = _one()
    assert answer.compiled_graph is None
    assert answer.refusal_code == "compiled_graph_resolve_incomplete"
    assert field in answer.detail


def test_every_pre_transport_requirement_has_one_downstream_consumer() -> None:
    """Resolve requires only address, bytes and the signed TCG identity.

    Graph/toolchain/seal projections are deliberately absent: the receipt
    carries TCG's closed identity axes and delivery imports against its exact
    compiled-graph key.
    """
    assert {field for field, _why in cell_resolve._REQUIRED} == {
        "compiled_graph_key",
        "compiled_graph_ref",
        "content_digest",
        "receipt",
    }


def test_an_incomplete_answer_is_a_typed_refusal_not_a_miss() -> None:
    """A pod that read it as "no cell" would go pay for a whole cold mint
    believing the hub holds nothing, which is false and expensive."""
    assert "compiled_graph_resolve_incomplete" in cell_resolve.VERDICT_CODES


def test_a_non_key_is_refused_before_the_hub_is_dialled(stub) -> None:
    sent, _resp = stub
    # Every entry here is refused for its SHAPE, which after th#1897 is the
    # only thing either side of the wire refuses: the grammar never rules on
    # scheme, so that a newer fleet's key stays addressable by an older hub.
    # A WELL-FORMED `ck1-<56 hex>` therefore no longer belongs in this list —
    # it parses, and then misses on the axes, which is where the re-key is
    # actually enforced. `arm1-` needs the arm token's own 64-hex width to be
    # a non-key at all.
    for bad in ("", "sdxl", "arm1-" + "ab" * 32, "ck1-short",
                "cg-key-v1-" + "0" * 55):
        with pytest.raises(cell_resolve.CompiledGraphResolveRefused):
            _one("sdxl", bad)
    # ...and ONE bad key refuses the WHOLE batch rather than being answered as
    # a miss the pod would self-mint over — the hub does the same.
    with pytest.raises(cell_resolve.CompiledGraphResolveRefused):
        cell_resolve.resolve_batch("sdxl", [KEY, "ck1-short"])
    assert not sent  # nothing was sent


def test_a_missing_family_is_refused_before_the_hub_is_dialled(stub) -> None:
    sent, _resp = stub
    with pytest.raises(cell_resolve.CompiledGraphResolveRefused):
        _one("", KEY)
    # ...and so is an empty key set: a batch that asks nothing is a caller bug,
    # not an answer of no misses.
    with pytest.raises(cell_resolve.CompiledGraphResolveRefused):
        cell_resolve.resolve_batch("sdxl", [])
    assert not sent


# ---------------------------------------------------------------------------
# A hit feeds the EXISTING admission machinery, never a second brain
# ---------------------------------------------------------------------------


def test_the_receipt_rides_the_answer_and_is_never_re_fetched(stub) -> None:
    """th#1680: ``handleWorkerCellReceipt`` scopes by ENDPOINT while resolve
    scopes by ORG, so a second fetch for the same cell could 403 what resolve
    just offered. The receipt therefore has to come from the answer — and this
    module must contain no receipt fetch at all."""
    _sent, resp = stub
    resp["r"] = _Resp(200, _batch(_hit_body()))
    compiled_graph = _one().compiled_graph
    assert compiled_graph is not None
    assert compiled_graph.receipt is _VERIFIED_RECEIPT

    import inspect

    source = inspect.getsource(cell_resolve)
    assert "cells/receipt" not in source
    assert "receipt_for" not in source


def test_the_transport_is_shaped_for_the_existing_delivery_path(stub) -> None:
    """``materialize_named_artifact`` reads the grant by ATTRIBUTE. The answer's
    transport must present the same attributes or the resolve feeds a parallel
    downloader instead of the one with the digest checks in it."""
    _sent, resp = stub
    resp["r"] = _Resp(200, _batch(_hit_body()))
    compiled_graph = _one().compiled_graph
    assert compiled_graph is not None

    files = list(compiled_graph.transport.files)
    assert len(files) == 1
    entry = files[0]
    for attr in ("path", "size_bytes", "digest", "url", "chunks"):
        assert hasattr(entry, attr), attr
    assert not hasattr(entry, "chunk_size_bytes")
    assert str(entry.path).endswith(".tar.gz")


def test_a_chunked_transport_carries_the_chunk_attributes(stub) -> None:
    _sent, resp = stub
    body = _hit_body()
    body["transport"]["files"][0]["chunks"] = [
        {"sha256": "aa" * 32, "url": "https://cas/0", "len": 2048},
        {"sha256": "bb" * 32, "url": "https://cas/1", "len": 2048},
    ]
    body["transport"]["files"][0]["chunk_size_bytes"] = 2048
    resp["r"] = _Resp(200, _batch(body))
    compiled_graph = _one().compiled_graph
    assert compiled_graph is not None
    chunks = list(compiled_graph.transport.files[0].chunks)
    assert [c.len for c in chunks] == [2048, 2048]
    assert chunks[0].sha256 == "aa" * 32
    assert not hasattr(compiled_graph.transport.files[0], "chunk_size_bytes")


def test_materialize_delegates_and_adds_nothing(monkeypatch, stub) -> None:
    """Deliberately a two-line function: a second downloader is a second place
    for "verified" to mean something slightly different."""
    _sent, resp = stub
    resp["r"] = _Resp(200, _batch(_hit_body()))
    compiled_graph = _one().compiled_graph
    assert compiled_graph is not None

    from gen_worker import aot_delivery

    seen: Dict[str, Any] = {}

    def _materialize(
        key, family, ref, digest, presigned, *, receipt, cache_dir, what
    ):
        seen.update(
            key=key,
            family=family,
            ref=ref,
            digest=digest,
            presigned=presigned,
            receipt=receipt,
            cache_dir=cache_dir,
            what=what,
        )
        from pathlib import Path

        return Path("/tmp/cell.tar.gz")

    monkeypatch.setattr(
        aot_delivery, "materialize_named_artifact", _materialize)
    out = cell_resolve.materialize(compiled_graph, cache_dir=None)
    assert str(out).endswith("cell.tar.gz")
    assert seen["key"] == compiled_graph.compiled_graph_key
    assert seen["family"] == compiled_graph.family
    assert seen["ref"] == compiled_graph.compiled_graph_ref
    assert seen["digest"] == compiled_graph.content_digest
    assert seen["presigned"] is compiled_graph.transport
    assert seen["receipt"] is _VERIFIED_RECEIPT


# ---------------------------------------------------------------------------
# The boot-adopt orchestration: every failure degrades, none is fatal
# ---------------------------------------------------------------------------


def _attempt(monkeypatch, tmp_path, *, derive=None, resolve_batch=None,
             materialize=None) -> Any:
    from gen_worker import boot_adopt, boot_key

    if derive is not None:
        monkeypatch.setattr(boot_key, "derive", derive)
    if resolve_batch is not None:
        monkeypatch.setattr(cell_resolve, "resolve_batch", resolve_batch)
    if materialize is not None:
        monkeypatch.setattr(cell_resolve, "materialize", materialize)

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


def _attempt_all(monkeypatch, tmp_path, *, derive=None, resolve_batch=None,
                 materialize=None) -> Any:
    """Every outcome, for the declarations that trace to more than one class."""
    from gen_worker import boot_adopt, boot_key

    if derive is not None:
        monkeypatch.setattr(boot_key, "derive", derive)
    if resolve_batch is not None:
        monkeypatch.setattr(cell_resolve, "resolve_batch", resolve_batch)
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


def _derived_multi() -> Any:
    """A declaration that traced THREE graph classes — the shape the batch wire
    exists for, and the one a single-key loop would bill three round trips."""
    from torch_compiled_graphs.identity import from_axes

    from gen_worker import boot_key

    axes = [{"graph": g, "sm": "sm_89", "toolchain": "t" * 16}
            for g in ("c0ffee0000000000", "beef000000000000",
                      "dead000000000000")]
    return boot_key.DerivedKey(
        entry_keys={f"e{i}": str(from_axes(a))
                    for i, a in enumerate(axes)},
        workers=2, width_reason="test", traced=3, memo="miss", wall_ms=1234)


def _derived(digest: str = KEY) -> Any:
    from torch_compiled_graphs.identity import from_axes

    from gen_worker import boot_key

    return boot_key.DerivedKey(
        # pgw#1176: a boot derives a KEY SET. These declarations trace to one
        # class, so the set has one member and callers take it from `keys`.
        entry_keys={"a": str(from_axes({
            "graph": "c0ffee0000000000",
            "sm": "sm_89", "toolchain": "t" * 16}))},
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
        raise cell_resolve.CompiledGraphResolveRefused(
            "compiled_graph_resolve_client_supplied_field",
            "caller supplied an entitlement field",
            status=400,
        )

    out = _attempt(
        monkeypatch, tmp_path,
        derive=lambda **_kw: _derived(), resolve_batch=_refuse)
    assert not out.adopted
    assert out.reason == "compiled_graph_resolve_client_supplied_field"
    assert out.derived_key.startswith("cg-key-v1-")
    assert out.derive_ms == 1234


def test_a_miss_degrades_as_miss_not_as_a_failure(monkeypatch, tmp_path) -> None:
    out = _attempt(
        monkeypatch, tmp_path,
        derive=lambda **_kw: _derived(),
        resolve_batch=lambda _f, keys, **_k: tuple(
            cell_resolve.ResolveAnswer(compiled_graph_key=k, status="miss")
            for k in keys))
    assert not out.adopted
    assert out.reason == "miss"
    assert out.derived_key.startswith("cg-key-v1-")


def test_a_failed_materialize_degrades_and_is_never_fatal(
    monkeypatch, tmp_path,
) -> None:
    from gen_worker import aot_delivery

    def _boom(*_a, **_k):
        raise aot_delivery.NamedArtifactUnavailable(
            "content_digest_mismatch", "bytes refused")

    class _Cell:
        publisher_org = "org-a"
        compiled_graph_ref = "root/family-sdxl#" + KEY
        publisher_tier = "platform"

    out = _attempt(
        monkeypatch, tmp_path,
        derive=lambda **_kw: _derived(),
        resolve_batch=lambda _f, keys, **_k: tuple(
            cell_resolve.ResolveAnswer(
                compiled_graph_key=k, status="hit", compiled_graph=_Cell())
            for k in keys),
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


# ---------------------------------------------------------------------------
# pgw#1224 — the properties only a BATCH can have. Each row names the red it
# would go if the client stopped checking it.
# ---------------------------------------------------------------------------


def test_one_answer_per_key_in_request_order(stub) -> None:
    """The whole contract in one row: N keys in, N answers out, positionally
    aligned, and the caller may zip them without checking."""
    keys = [KEY, OTHER_KEY, THIRD_KEY]
    _sent, resp = stub
    resp["r"] = _Resp(200, _batch(
        _hit_body(), _miss(OTHER_KEY),
        _hit_body(compiled_graph_key=THIRD_KEY, receipt="eyJ.dGhpcmQ.c2ln")))
    answers = cell_resolve.resolve_batch("sdxl", keys)
    assert [a.compiled_graph_key for a in answers] == keys
    assert [a.status for a in answers] == ["hit", "miss", "hit"]


def test_a_short_answer_is_refused_not_read_as_misses(stub) -> None:
    """RED (omit misses instead of answering them): ``asked 3 keys, got 1
    answers``. An omission is indistinguishable from a truncation, and the pod
    answers each phantom miss by paying for a full cold mint."""
    _sent, resp = stub
    resp["r"] = _Resp(200, _batch(_hit_body()))
    with pytest.raises(cell_resolve.CompiledGraphResolveRefused) as err:
        cell_resolve.resolve_batch("sdxl", [KEY, OTHER_KEY, THIRD_KEY])
    assert err.value.code == "compiled_graph_resolve_short_answer"
    assert "asked 3 keys, got 1 answers" in err.value.detail


def test_a_long_answer_is_refused_too(stub) -> None:
    """Symmetric, and for the same reason: an answer set that is not the
    question's shape has not answered the question."""
    _sent, resp = stub
    resp["r"] = _Resp(200, _batch(_hit_body(), _miss(OTHER_KEY)))
    with pytest.raises(cell_resolve.CompiledGraphResolveRefused) as err:
        cell_resolve.resolve_batch("sdxl", [KEY])
    assert err.value.code == "compiled_graph_resolve_short_answer"


def test_a_reply_with_no_answers_at_all_is_refused(stub) -> None:
    _sent, resp = stub
    resp["r"] = _Resp(200, {"object": "compiled_graph_resolve_batch", "family": "sdxl"})
    with pytest.raises(cell_resolve.CompiledGraphResolveRefused) as err:
        cell_resolve.resolve_batch("sdxl", [KEY])
    assert err.value.code == "compiled_graph_resolve_short_answer"


def test_transposed_answers_are_refused(stub) -> None:
    """RED (trust position without checking the echo, or the echo without the
    position): a batch whose answers were swapped would arm each graph class
    with a sibling's kernels, and the key is an ingress identity — only the
    graph-witness floor would stand in the way, after the bytes were paid for."""
    _sent, resp = stub
    resp["r"] = _Resp(200, _batch(
        _hit_body(compiled_graph_key=OTHER_KEY), _hit_body()
    ))
    with pytest.raises(cell_resolve.CompiledGraphResolveRefused) as err:
        cell_resolve.resolve_batch("sdxl", [KEY, OTHER_KEY])
    assert err.value.code == "compiled_graph_resolve_answer_out_of_order"
    assert "answers[0]" in err.value.detail


def test_a_batch_level_signature_is_refused_never_ignored(stub) -> None:
    """RED (ignore a top-level signature): an ignored signature field is one a
    later reader assumes was checked. There is deliberately NO signature over
    the batch — a batch-level one would make the COLLECTION the unit of trust,
    the exact mistake th#1834 is undoing one layer down."""
    _sent, resp = stub
    for field in ("signature", "receipt", "jws", "answers_signature"):
        resp["r"] = _Resp(200, _batch(_hit_body(), **{field: "eyJ.x.y"}))
        with pytest.raises(cell_resolve.CompiledGraphResolveRefused) as err:
            cell_resolve.resolve_batch("sdxl", [KEY])
        assert err.value.code == "compiled_graph_resolve_batch_signature"
        assert field in err.value.detail


def test_two_answers_may_not_share_one_receipt(stub) -> None:
    """EVERY ANSWER IS INDEPENDENTLY SIGNED. One receipt covering two answers
    is a batch-level signature wearing a per-answer field name: the receipt is
    the hub's signature over ONE compiled graph, so a shared one means at least
    one answer is vouched for by the other's proof.

    RED (verify signatures per BATCH rather than per answer): both answers are
    adopted, and the second arms bytes nothing signed for it."""
    _sent, resp = stub
    resp["r"] = _Resp(200, _batch(
        _hit_body(), _hit_body(compiled_graph_key=OTHER_KEY)))
    with pytest.raises(cell_resolve.CompiledGraphResolveRefused) as err:
        cell_resolve.resolve_batch("sdxl", [KEY, OTHER_KEY])
    assert err.value.code == "compiled_graph_resolve_shared_receipt"


def test_distinctly_signed_answers_are_both_adopted(stub) -> None:
    """The other side of the row above, so the guard cannot pass by refusing
    everything: two answers, two receipts, both hits."""
    _sent, resp = stub
    resp["r"] = _Resp(200, _batch(
        _hit_body(),
        _hit_body(
            compiled_graph_key=OTHER_KEY,
            receipt="eyJ.b3RoZXI.c2ln",
        ),
    ))
    answers = cell_resolve.resolve_batch("sdxl", [KEY, OTHER_KEY])
    assert [a.status for a in answers] == ["hit", "hit"]
    assert answers[0].compiled_graph is not None
    assert answers[1].compiled_graph is not None


def test_an_unsigned_hit_is_refused_before_the_bytes_are_paid_for(stub) -> None:
    """A hit with no receipt states an admission expectation nothing can check,
    and the gate that would catch it runs after the whole artifact is
    downloaded."""
    _sent, resp = stub
    resp["r"] = _Resp(200, _batch(_hit_body(receipt="")))
    answer = _one()
    assert answer.compiled_graph is None
    assert answer.refusal_code == "compiled_graph_resolve_incomplete"
    assert "receipt" in answer.detail


def test_one_bad_answer_does_not_sink_its_siblings(stub) -> None:
    """THE point of per-answer statuses. Under the atom a boot resolves one key
    per graph class, so a whole-request 409 on one duplicated row would throw
    away 35 good answers and re-mint them."""
    _sent, resp = stub
    resp["r"] = _Resp(200, _batch(
        _hit_body(),
        {
            "compiled_graph_key": OTHER_KEY,
            "status": "transport_unavailable",
            "found": False,
            "detail": "origin timed out",
        },
        _hit_body(compiled_graph_key=THIRD_KEY, receipt="eyJ.dGhpcmQ.c2ln")))
    answers = cell_resolve.resolve_batch("sdxl", [KEY, OTHER_KEY, THIRD_KEY])
    assert [a.status for a in answers] == ["hit", "transport_unavailable", "hit"]
    assert sum(1 for a in answers if a.hit) == 2
    assert answers[1].refusal_code == "compiled_graph_resolve_transport_unavailable"


def test_a_duplicate_key_is_refused_before_the_hub_is_dialled(stub) -> None:
    """Answers are positional; a collapsed duplicate would shift every later
    answer against its request. The hub refuses it by name and so does this,
    one round trip earlier."""
    sent, _resp = stub
    with pytest.raises(cell_resolve.CompiledGraphResolveRefused) as err:
        cell_resolve.resolve_batch("sdxl", [KEY, OTHER_KEY, KEY])
    assert err.value.code == "compiled_graph_resolve_duplicate_key"
    assert not sent


def test_over_the_bound_is_refused_before_the_hub_is_dialled(stub) -> None:
    """The bound and the explicit miss are ONE design: over it the hub refuses
    the WHOLE batch by name, so splitting late would cost every key's answer to
    learn the number."""
    sent, _resp = stub
    too_many = ["cg-key-v1-" + f"{i:056x}"
                for i in range(cell_resolve.MAX_RESOLVE_KEYS + 1)]
    with pytest.raises(cell_resolve.CompiledGraphResolveRefused) as err:
        cell_resolve.resolve_batch("sdxl", too_many)
    assert err.value.code == "compiled_graph_resolve_too_many_keys"
    assert not sent
    # ...and exactly the bound is fine, so the guard is a bound and not a ban.
    _resp["r"] = _Resp(200, _batch(*[_miss(k) for k in too_many[:-1]]))
    assert len(cell_resolve.resolve_batch("sdxl", too_many[:-1])) == \
        cell_resolve.MAX_RESOLVE_KEYS


def test_a_boot_resolves_its_whole_manifest_in_ONE_call(
    monkeypatch, tmp_path,
) -> None:
    """The saving the batch wire exists for, asserted as a COUNT: a 3-class
    declaration costs one round trip, not three. RED (loop per key): calls == 3.
    """
    calls: list = []

    def _batched(family, keys, **_kw):
        calls.append(tuple(keys))
        return tuple(
            cell_resolve.ResolveAnswer(compiled_graph_key=k, status="miss")
            for k in keys)

    outs = _attempt_all(
        monkeypatch, tmp_path,
        derive=lambda **_kw: _derived_multi(), resolve_batch=_batched)
    assert len(calls) == 1
    assert len(calls[0]) == 3
    assert len(outs) == 3
    assert {o.reason for o in outs} == {"miss"}
