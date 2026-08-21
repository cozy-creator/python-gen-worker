"""pgw#1603: parallel derive items and the parent-banking store.

Three facts, each falsifiable on the REAL derive path (fixture endpoint,
real diffusers modules, config-only tree):

1. the parallel degree must not change the DOCUMENT — workers=2 (spawned
   processes) and workers=1 (in-process) produce byte-identical bytes;
2. the item enumeration is the structural-variant count, never the bucket
   count;
3. a STATIC-declared shape fan banks SYMBOLIC PARENT programs — the store
   holds fewer blobs than records, every stored program is range-bearing,
   and the compile seam re-derives each record's exact identity from it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("torch")
pytest.importorskip("diffusers")
pytest.importorskip("transformers")
import torch  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "release_fixtures"

LOCK = (
    'version = 1\n'
    '\n[[package]]\nname = "torch"\nversion = "2.13.0"\n'
    '\n[[package]]\nname = "triton"\nversion = "3.7.1"\n'
    '\n[[package]]\nname = "nvidia-cublas"\nversion = "13.1.1.3"\n'
    '\n[[package]]\nname = "diffusers"\nversion = "0.39.0"\n'
)


@pytest.fixture(scope="module")
def config_only_tree(tmp_path_factory: pytest.TempPathFactory) -> Path:
    sys.path.insert(0, str(FIXTURES))
    try:
        import tiny_tree
    finally:
        sys.path.remove(str(FIXTURES))
    tree: Path = tiny_tree.save_config_only(tmp_path_factory.mktemp("par-tree"))
    return tree


def _derive(tree: Path, room: Path, workers: int) -> tuple[Any, Path]:
    import importlib

    from gen_worker.discovery.discover import prime_sys_path
    from gen_worker.release.derive import derive_release

    prime_sys_path(FIXTURES)
    module = importlib.import_module("static_axes_endpoint")
    lockfile = room / "uv.lock"
    lockfile.write_text(LOCK)
    cas = room / "cas"
    cas.mkdir()
    result = derive_release(
        module,
        checkpoint_dir=tree,
        lockfile=lockfile,
        graph_cas=cas,
        trace_workers=workers,
    )
    return result, cas


def test_the_parallel_degree_does_not_change_the_document(
    config_only_tree: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    sequential, _ = _derive(
        config_only_tree, tmp_path_factory.mktemp("seq"), workers=1
    )
    parallel, _ = _derive(
        config_only_tree, tmp_path_factory.mktemp("par"), workers=2
    )
    assert parallel.document == sequential.document
    assert parallel.digest == sequential.digest


def test_items_are_structural_variants_never_buckets(
    config_only_tree: Path,
) -> None:
    import importlib

    from gen_worker.discovery.discover import prime_sys_path
    from gen_worker.release.derive import derive_items

    prime_sys_path(FIXTURES)
    module = importlib.import_module("static_axes_endpoint")
    items = derive_items(module)
    # One class, one lane, no Defaults-cfg twin, no structural= axes: ONE
    # item covers the whole 3-aspect x 2-guidance payload fan.
    assert len(items) == 1


def test_static_buckets_bank_symbolic_parents(
    config_only_tree: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    from gen_worker._vendor.tensorfs import LocalCAS
    from gen_worker._vendor.torchcg.bind import respecialize
    from gen_worker._vendor.torchcg.graph_identity import graph_hash
    from gen_worker._vendor.torchcg.ingress import CallIngress
    from gen_worker._vendor.torchcg.store import LocalGraphStore
    from gen_worker.serving import weightless_program

    result, cas = _derive(
        config_only_tree, tmp_path_factory.mktemp("bank"), workers=1
    )
    document = json.loads(result.document)
    (lane,) = document["graphs"]["lanes"]
    records = lane["graphs"]
    assert len(records) >= 2, "the fixture fans over aspects and guidance"
    # Fewer BLOBS than records: the content-addressed store deduped the
    # shared parent bytes.
    blobs = {
        path
        for path in (cas / "objects").rglob("*")
        if path.is_file()
    }
    assert len(blobs) < len(records)

    store = LocalGraphStore(LocalCAS(cas))
    weightless_program.install()
    scratch = tmp_path_factory.mktemp("programs")
    for index, record in enumerate(records):
        blob = store.fetch_program(record["graph"], scratch / f"{index}.pt2")
        assert blob is not None
        program = torch.export.load(str(blob))
        assert program.range_constraints, (
            "a static bucket must resolve to its SYMBOLIC parent's bytes"
        )
        ingress = CallIngress.decode(record["ingress"])
        assert not ingress.symbols
        rebound = respecialize(program, ingress)
        assert graph_hash(rebound, ingress) == record["graph"]
