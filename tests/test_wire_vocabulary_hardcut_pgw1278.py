"""pgw#1278 (HARDCUT-CHECKLIST B3): the compiled-graph wire hard cut, pgw half.

Two arms, and both are needed. The GREP FENCE proves the dead spelling is
absent from ``src/``; the LIVE SEAM proves the live spelling is what actually
crosses a socket. A fence alone greens on a tree that deleted the feature, and
a seam test alone greens on a tree that kept a compat alias — which is the
thing a hard cut forbids.

The allowlist is EMPTY on purpose. ``proto/`` and ``src/gen_worker/pb/`` are
excluded because they are tensorhub's canonical contract and its generated
output: ``ActivityUpdate`` field 19 is still spelled ``cell_key`` there and
moves in the proto lane (th#1947), not here. ``gen_worker.activity`` translates
it on exactly one line.

Anything the proto ENUMERATES moves with the proto, so it is absent from the
table below: the ``jit_cell``/``aot_cell`` serving modes, the four ``cell_*``
boot phases, and ``compile_cell_failed`` — the last a CLOSED
``FnUnavailable.reason`` vocabulary, already fenced by
``test_fn_unavailable_vocabulary_th1563``'s reason census, which caught this lane renaming it.
"""

from __future__ import annotations

import pathlib
import sys
from typing import Dict, List, Tuple

import pytest

from gen_worker import cell_resolve, fleet_cells as fc, receipts
from gen_worker.procsplit import actions

from harness.cell_hub import local_cell_hub

_REPO = pathlib.Path(__file__).resolve().parents[1]
_SRC = _REPO / "src"
#: tensorhub's canonical proto and its generated output — a different contract
#: with a different owner. Everything else in `src/` is this repo's vocabulary.
# pgw#1310: which subtrees a guard may not judge has ONE home.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
from _lint_scope import is_unowned  # noqa: E402

#: Every dead spelling, with the successor that replaced it. A pair rather than
#: a bare list so the failure names what to write instead.
DEAD_SPELLINGS: Tuple[Tuple[str, str], ...] = (
    ("/v1/worker/cells/", "/v1/worker/compiled-graphs/"),
    ("cell-receipt-v2", "compiled-graph-receipt-v1"),
    ("cell-receipt-v1", "compiled-graph-receipt-v1"),
    ('"cells.resolve"', '"compiled_graphs.resolve"'),
    ('"cells.receipt"', '"compiled_graphs.receipt"'),
    ('"cells.revocations"', '"compiled_graphs.revocations"'),
    ('"cells.publish_intent"', '"compiled_graphs.publish_intent"'),
    ('"cells.publish_complete"', '"compiled_graphs.publish_complete"'),
    ("cell_receipt_refused", "compiled_graph_receipt_refused"),
    ("cell_resolve_ambiguous", "compiled_graph_resolve_ambiguous"),
    ("cell_resolve_incomplete", "compiled_graph_resolve_incomplete"),
    ("cell_resolve_transport_unavailable",
     "compiled_graph_resolve_transport_unavailable"),
    ("cell_resolve_client_supplied_field",
     "compiled_graph_resolve_client_supplied_field"),
    ("cell_publish_untrusted_tier", "compiled_graph_publish_untrusted_tier"),
    ("local_cell_armed", "local_compiled_graph_armed"),
    ("local_cell_refused", "local_compiled_graph_refused"),
    ("local_cell_stored", "local_compiled_graph_stored"),
    ("local_cell_store_failed", "local_compiled_graph_store_failed"),
    ("CELL_PUBLISHER_TIER_", "COMPILED_GRAPH_PUBLISHER_TIER_"),
)


def _src_files() -> List[pathlib.Path]:
    out = []
    for p in sorted(_SRC.rglob("*")):
        if not p.is_file() or p.suffix not in {".py", ".pyi", ".json", ".txt"}:
            continue
        if is_unowned(p, _SRC):
            continue
        out.append(p)
    assert out, "the source scan found nothing — the fence would be vacuous"
    return out


@pytest.mark.parametrize("dead,live", DEAD_SPELLINGS,
                         ids=[d.strip('"/') for d, _ in DEAD_SPELLINGS])
def test_the_dead_wire_spelling_is_absent_from_src(dead: str, live: str) -> None:
    hits = []
    for p in _src_files():
        try:
            text = p.read_text()
        except UnicodeDecodeError:
            continue
        for n, line in enumerate(text.splitlines(), 1):
            if dead in line:
                hits.append(f"{p.relative_to(_REPO)}:{n}: {line.strip()[:110]}")
    assert not hits, (
        f"{dead!r} is DEAD — write {live!r}. This is a hard cut: no alias, no "
        "dual-speak, no compat shim.\n  " + "\n  ".join(hits))


def test_the_bare_compiled_graph_key_token_is_the_only_spelling() -> None:
    """``cell_key`` survives ONLY as the proto field's own name — and only
    where the text is talking about the proto. Anywhere else it is a second
    name for a value this repo already names."""
    hits = []
    for p in _src_files():
        try:
            text = p.read_text()
        except UnicodeDecodeError:
            continue
        for n, line in enumerate(text.splitlines(), 1):
            stripped = line.replace("requested_cell_key", "")
            if "cell_key" not in stripped:
                continue
            if p.name == "activity.py":
                continue  # the one translation seam, asserted below
            hits.append(f"{p.relative_to(_REPO)}:{n}: {line.strip()[:110]}")
    assert not hits, (
        "`cell_key` is the PROTO field's spelling only; this repo's vocabulary "
        "is `compiled_graph_key`.\n  " + "\n  ".join(hits))


def test_the_proto_translation_lives_on_exactly_one_line() -> None:
    """The proto seam is allowed to say ``cell_key`` — once. Two lines means a
    second name has grown back inside the worker."""
    text = (_SRC / "gen_worker" / "activity.py").read_text()
    code = [ln for ln in text.splitlines()
            if "cell_key=" in ln and not ln.lstrip().startswith("#")]
    assert code == ["        cell_key=str(compiled_graph_key or \"\")[:200],"], code


def test_the_live_constants_hold_the_compiled_graph_spelling() -> None:
    """The fence cannot be satisfied by deleting the thing it fences."""
    assert cell_resolve.RESOLVE_PATH == "/v1/worker/compiled-graphs/resolve"
    assert receipts.RECEIPT_PATH == "/v1/worker/compiled-graphs/receipt"
    assert receipts.REVOCATIONS_PATH == "/v1/worker/compiled-graphs/revocations"
    assert receipts.RECEIPT_VERSION == "compiled-graph-receipt-v1"
    assert actions.PUBLISH_ACTIONS == frozenset({
        "compiled_graphs.publish_intent", "compiled_graphs.publish_complete"})
    names = {a.name for a in actions.ACTIONS.values()}
    assert {"compiled_graphs.resolve", "compiled_graphs.receipt",
            "compiled_graphs.revocations", "compiled_graphs.publish_intent",
            "compiled_graphs.publish_complete"} <= names


def test_the_publisher_speaks_the_new_route_over_a_real_socket() -> None:
    """The live arm. A real ``CellPublisher`` against a real loopback hub: the
    path the hub SAW and the body key it read are the cut's actual product."""
    entry = fc.PublishEntry(
        compiled_graph_key="cg-key-v1-" + "a" * 56,
        identity_axes={"graph": "c0ffee0000000000", "sm": "sm_89",
                       "toolchain": "toolch0000000000"},
        mint_duration_ms=1234,
    )
    with local_cell_hub() as hub:
        pub = fc.CellPublisher(base_url=hub.base, worker_jwt=lambda: "worker-jwt",
                               image_digest="sha256:" + "1" * 64)
        batch = pub.publish_intent("sdxl", [entry], sku="l4", gen_worker="0.116.0")

    assert [g.compiled_graph_key for g in batch.grants] == [entry.compiled_graph_key]
    assert hub.routes() == ["/v1/worker/compiled-graphs/publish-intent"]
    (intent,) = hub.intents
    (wire_entry,) = intent["entries"]
    assert "compiled_graph_key" in wire_entry and "cell_key" not in wire_entry


class _Resp:
    def __init__(self, body: Dict[str, object]) -> None:
        self.status_code = 200
        self.text = ""
        self._body = body

    def json(self) -> Dict[str, object]:
        return self._body


def test_the_resolve_answer_is_read_under_the_new_wire_key(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """The consuming half of row 5. An answer that identifies itself the OLD
    way cannot restate the key it answers, so it is refused BY NAME — a hub
    that half-lands its side is loud rather than silently sending this pod to a
    full cold mint for every key it thought it had answered."""
    key = "cg-key-v1-" + "b" * 56
    answer: Dict[str, object] = {
        "status": "hit", "found": True, "family": "sdxl",
        "cell_key": key,  # the dead spelling
        "cell_ref": f"root/family-sdxl#{key}",
        "content_digest": "sha256:" + "11" * 32,
        "publisher_org": "org-a", "graph_contract": "c0ffee0000000000",
        "toolchain_digest": "toolch0000000000",
        "env_seal_digest": "seal000000000000",
        "receipt": "eyJ.aGVhZA.c2ln",
    }
    sent: Dict[str, object] = {}

    def _request(method: str, path: str, **kw: object) -> _Resp:
        sent["method"], sent["path"] = method, path
        return _Resp({"object": "compiled_graph_resolve_batch", "family": "sdxl",
                      "answers": [answer]})

    monkeypatch.setattr("gen_worker.cell_resolve.broker.request", _request)
    with pytest.raises(cell_resolve.CellResolveRefused) as exc:
        cell_resolve.resolve_batch("sdxl", [key], base_url="https://hub")

    assert sent["path"] == "/v1/worker/compiled-graphs/resolve"
    assert exc.value.code == "compiled_graph_resolve_answer_out_of_order"
