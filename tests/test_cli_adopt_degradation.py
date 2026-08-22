"""Boot-time adoption degrades; it does not die."""

from __future__ import annotations

from pathlib import Path

import pytest

from gen_worker.cli.daemon import BootSpec, _adoption_source


def _store_with_unreadable_document(root: Path, module: str) -> None:
    from gen_worker._vendor.tensorfs import LocalCAS
    from gen_worker.graphs.store import _document_ref

    root.mkdir(parents=True, exist_ok=True)
    cas = LocalCAS(root)
    ref = cas.put_bytes(b'{"v": 1, "this is a v1 document": true}')
    cas.compare_and_swap_ref(_document_ref(module), ref, expected=None)


def test_an_unreadable_graph_document_serves_eager_instead_of_killing_boot(
    tmp_path, capsys
):
    store = tmp_path / "graph-cas"
    _store_with_unreadable_document(store, "toy.main")

    spec = BootSpec(endpoint_dir=tmp_path, graph_store=store, sm="sm_89")
    adopted_store, document = _adoption_source(spec, "toy.main")

    assert document is None, (
        "an undecodable document must read as a MISS — the eager bridge"
    )
    assert adopted_store is not None, (
        "the store stays bound: it is still WRITABLE, so a re-mint can refill "
        "THIS store rather than requiring it be deleted first"
    )
    said = capsys.readouterr().err
    assert "SERVING EAGER" in said
    assert "gen-worker compile" in said


def test_the_discard_itself_is_announced_and_names_what_went_away(
    tmp_path, caplog
):
    """WHICH of the two "nothing usable" states it was is still stated."""
    import logging

    store = tmp_path / "graph-cas"
    _store_with_unreadable_document(store, "toy.main")
    spec = BootSpec(endpoint_dir=tmp_path, graph_store=store, sm="sm_89")

    with caplog.at_level(
        logging.WARNING, logger="gen_worker.graphs.store"
    ):
        _adoption_source(spec, "toy.main")

    said = "\n".join(
        r.getMessage()
        for r in caplog.records
        if r.name == "gen_worker.graphs.store"
    )
    assert "DISCARDED" in said
    assert "toy.main" in said
    assert "sha256:" in said
    assert "unreadable by this build" in said


def test_the_shape_actually_found_on_disk_is_the_version_mismatch(
    tmp_path, caplog
):
    """A document whose ONLY defect is its version, with a VALID field set.

    Worth its own case because the fixture above fails on the field set and
    would pass a reader that only rejected unknown fields. This one can only
    fail on the version, which is the shape a re-vendor actually produces.

    The version used is one AHEAD of this build's, not behind it (pgw#1621 —
    the shape va#3 hit was `v=2` against a `v=3` reader, and tcg#79 bumped the
    reader to 4). Forward is the direction a rolling fleet actually produces:
    a pod on the older image reading the document a newer pod already wrote.
    And "one ahead" is a RELATION a format bump cannot silently satisfy,
    whereas a pinned historical number drifts further from the fence with
    every bump until it is red for archaeological reasons rather than this
    one. The relation is asserted below so the drift is what goes red.
    """
    import logging

    from gen_worker._vendor.tensorfs import LocalCAS
    from gen_worker.graphs.document import DOCUMENT_FORMAT
    from gen_worker.graphs.store import _document_ref

    AHEAD = 5
    assert AHEAD == DOCUMENT_FORMAT + 1, (
        "this fixture must stay exactly ONE format ahead of the reader. A "
        "re-vendor that leaves it level makes the document READABLE and this "
        "case degenerates into the empty-store case above — passing while "
        "measuring nothing."
    )

    store = tmp_path / "graph-cas"
    store.mkdir(parents=True)
    cas = LocalCAS(store)
    ref = cas.put_bytes(b'{"lanes":[],"stack":[],"v":%d}' % AHEAD)
    cas.compare_and_swap_ref(_document_ref("toy.main"), ref, expected=None)

    spec = BootSpec(endpoint_dir=tmp_path, graph_store=store, sm="sm_89")
    with caplog.at_level(
        logging.WARNING, logger="gen_worker.graphs.store"
    ):
        _, document = _adoption_source(spec, "toy.main")

    assert document is None
    said = "\n".join(r.getMessage() for r in caplog.records)
    assert f"document v must be {DOCUMENT_FORMAT}" in said
    assert cas.read_ref(_document_ref("toy.main")) is None


def test_a_store_that_still_raises_keeps_the_typed_refusal(
    tmp_path, capsys, monkeypatch
):
    """The `StoreError` branch is NOT dead -- it guards the other backends."""
    import gen_worker.graphs.store as store_module
    from gen_worker.graphs.store import StoreError

    class RaisingStore:
        def get_graphs(self, name: str) -> None:
            raise StoreError(f"graph-set document {name!r} is unreadable: nope")

    store = tmp_path / "graph-cas"
    store.mkdir(parents=True)
    spec = BootSpec(endpoint_dir=tmp_path, graph_store=store, sm="sm_89")

    monkeypatch.setattr(store_module, "LocalGraphStore", lambda _cas: RaisingStore())

    adopted_store, document = _adoption_source(spec, "toy.main")

    assert document is None
    assert adopted_store is not None
    said = capsys.readouterr().err
    assert "graph_store_unreadable" in said
    assert "SERVING EAGER" in said
    assert "gen-worker compile" in said


def test_an_unreadable_document_lands_where_an_empty_store_lands(tmp_path):
    empty = tmp_path / "empty-cas"
    empty.mkdir(parents=True)
    _, empty_doc = _adoption_source(
        BootSpec(endpoint_dir=tmp_path, graph_store=empty, sm="sm_89"), "toy.main"
    )

    stale = tmp_path / "stale-cas"
    _store_with_unreadable_document(stale, "toy.main")
    _, stale_doc = _adoption_source(
        BootSpec(endpoint_dir=tmp_path, graph_store=stale, sm="sm_89"), "toy.main"
    )

    assert stale_doc is empty_doc is None


def test_a_clean_empty_store_is_still_a_miss(tmp_path):
    """A store with no document at all: the ordinary cold start."""
    store = tmp_path / "graph-cas"
    store.mkdir(parents=True)
    spec = BootSpec(endpoint_dir=tmp_path, graph_store=store, sm="sm_89")
    _, document = _adoption_source(spec, "toy.main")
    assert document is None


def test_adopting_without_an_sm_still_refuses(tmp_path):
    from gen_worker.cli.daemon import BootError

    store = tmp_path / "graph-cas"
    store.mkdir(parents=True)
    spec = BootSpec(endpoint_dir=tmp_path, graph_store=store, sm="")
    with pytest.raises(BootError) as caught:
        _adoption_source(spec, "toy.main")
    assert "--sm is required" in str(caught.value)
