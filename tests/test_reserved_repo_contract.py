"""The reserved-repo payload contract: `source`, its three siblings, datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import pytest

import projection_fixture as fixture
from gen_worker._vendor.tensorfs import LocalCAS
from gen_worker.api.errors import ValidationError
from gen_worker.models import projection
from gen_worker.models.hub_client import WorkerResolvedRepo, WorkerResolvedRepoFile
from gen_worker.models.refs import normalize_model_ref
from gen_worker.serving.loader import load_endpoint_module
from gen_worker.serving.residency import ResidencyManager
from gen_worker.serving.serve_loop import ServeLoop

FIXTURES = Path(__file__).resolve().parent / "release_fixtures"
MODULE = "conversion_endpoint"

RAW_REF = "tensorhub/e2e1890-minilm-l6-v2@v1"

_SHARD_BYTES = 1 << 18


class _NeverResolver:
    def resolve(self, model_cls: type, checkpoint_ref: str) -> Any:
        raise AssertionError("a weightless producer resolved a binding")

    def default_pick(self, model_cls: type, slot_name: str) -> str:
        raise AssertionError("a weightless producer asked for a default pick")


class _NeverSizer:
    def resident_bytes(self, checkpoint_ref: str, lane: str) -> int:
        raise AssertionError("a weightless producer sized a residency slot")

    def activation_headroom_bytes(self, checkpoint_ref: str, lane: str) -> int:
        raise AssertionError("a weightless producer reserved activation bytes")


@pytest.fixture(autouse=True)
def _module_path() -> Iterator[None]:
    import sys

    sys.path.insert(0, str(FIXTURES))
    try:
        yield
    finally:
        sys.path.remove(str(FIXTURES))


def _loop() -> ServeLoop:
    return ServeLoop(
        load_endpoint_module(MODULE),
        residency=ResidencyManager(1 << 30, _NeverSizer()),
        resolver=_NeverResolver(),
    )


def _write_model(source: Path) -> None:
    source.mkdir(parents=True, exist_ok=True)
    (source / "config.json").write_text(json.dumps({"model_type": "bert"}))
    (source / "tokenizer_config.json").write_text(json.dumps({"model_max_length": 512}))
    (source / "model.safetensors").write_bytes(
        fixture.safetensors_bytes(
            {"encoder.weight": ("F32", (_SHARD_BYTES // 4,),
                                fixture.varied(_SHARD_BYTES, 13))}
        )
    )


@pytest.fixture()
def cas(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[WorkerResolvedRepo]:
    """The pod's CAS, with every file of the source model resident in it, and the resolved manifest the hub would have shipped on `RunJob.snapshots`."""
    source = tmp_path / "source-model"
    _write_model(source)
    root = tmp_path / "tensorhub-cache"
    (root / "cas").mkdir(parents=True)
    from gen_worker.config import process as config_process

    monkeypatch.setenv("TENSORHUB_CACHE_DIR", str(root))
    config_process.reload_for_test()

    store = LocalCAS(root / "cas")
    files: list[WorkerResolvedRepoFile] = []
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        body = path.read_bytes()
        files.append(
            WorkerResolvedRepoFile(
                path.relative_to(source).as_posix(),
                len(body),
                "http://127.0.0.1:1/must-not-fetch",
                digest=str(store.put_bytes(body)),
            )
        )
    try:
        yield WorkerResolvedRepo(snapshot_digest="sha256:" + "a" * 64, files=files)
    finally:
        config_process.reset_for_test()


def _snapshots(resolved: WorkerResolvedRepo) -> dict[str, WorkerResolvedRepo]:
    return {normalize_model_ref(RAW_REF): resolved}


def _payload() -> dict[str, Any]:
    return {
        "input": {
            "source": {"ref": RAW_REF, "attributes": {"family": "bert"}},
            "dtypes": ["bf16"],
            "destination": {"repo": "acme/out", "release": "v1"},
        }
    }


def test_the_serve_path_materializes_the_reserved_source(
    cas: WorkerResolvedRepo,
) -> None:
    """THE regression, in the shape it was measured."""
    outcome = _loop().invoke(
        "cast_dtype",
        _payload(),
        request_id="pgw1475-drive",
        attempt=1,
        snapshots=_snapshots(cas),
    )
    tree = Path(outcome.result.source_path)
    assert tree.is_dir()
    assert projection.resolve_projection(tree) is not None
    assert projection.stub_at(tree / "model.safetensors") is not None
    assert outcome.result.tensor_bytes >= _SHARD_BYTES


def test_the_reserved_info_struct_is_stamped_beside_the_path(
    cas: WorkerResolvedRepo,
) -> None:
    """`source_from_ctx` builds its `Source` from `ctx.source` as well as `ctx.source_path` — `info["ref"]` and `info["attributes"]`."""
    outcome = _loop().invoke(
        "cast_dtype", _payload(), request_id="pgw1475-info", attempt=1,
        snapshots=_snapshots(cas),
    )
    assert outcome.result.source_ref == RAW_REF
    assert outcome.result.source_attributes == {"family": "bert"}
    assert outcome.result.destination_repo == "acme/out"


def test_the_regression_itself_without_the_wire(
    cas: WorkerResolvedRepo, monkeypatch: pytest.MonkeyPatch
) -> None:
    """RED-PROOF: remove the call the fix added and the measured failure comes back VERBATIM — the message from the standing-stack A/B, character for character."""
    from gen_worker.serving import serve_loop as loop_mod

    monkeypatch.setattr(
        loop_mod, "materialize_reserved_inputs", lambda *a, **kw: None
    )
    with pytest.raises(
        ValidationError,
        match=r"this function requires the reserved `source` payload field",
    ):
        _loop().invoke(
            "cast_dtype", _payload(), request_id="pgw1475-red", attempt=1,
            snapshots=_snapshots(cas),
        )


def test_the_survivor_arm_is_unaffected_either_way(
    cas: WorkerResolvedRepo, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The population boundary, asserted rather than asserted-about."""
    payload = {"input": {"upstream_url": "https://hf.co/acme/model"}}
    with_wire = _loop().invoke(
        "ingest", payload, request_id="pgw1475-clone", attempt=1,
        snapshots=_snapshots(cas),
    )
    assert with_wire.result.source_path_is_none is True

    from gen_worker.serving import serve_loop as loop_mod

    monkeypatch.setattr(
        loop_mod, "materialize_reserved_inputs", lambda *a, **kw: None
    )
    without = _loop().invoke(
        "ingest", payload, request_id="pgw1475-clone-red", attempt=1,
        snapshots=_snapshots(cas),
    )
    assert without.result.source_path_is_none is True


def test_an_empty_ref_on_a_declared_struct_REFUSES(
    cas: WorkerResolvedRepo,
) -> None:
    """Never a silent skip: an empty `ref` on a struct the payload DECLARED is the caller's error, and a skip would hand the body `None` and reproduce the very failure this issue is about, one layer down."""
    payload = _payload()
    payload["input"]["source"]["ref"] = "   "
    with pytest.raises(ValidationError, match=r"payload\.source\.ref must be"):
        _loop().invoke(
            "cast_dtype", payload, request_id="pgw1475-empty", attempt=1,
            snapshots=_snapshots(cas),
        )


def test_an_unpinned_tensorhub_ref_REFUSES_typed(
    cas: WorkerResolvedRepo,
) -> None:
    """The dispatch shipping no snapshot for a tensorhub ref is a deterministic local condition, and the download layer says so by name rather than burning retries on it."""
    from gen_worker.models.errors import MissingSnapshotError

    with pytest.raises(MissingSnapshotError, match="needs an orchestrator"):
        _loop().invoke(
            "cast_dtype", _payload(), request_id="pgw1475-unpinned", attempt=1,
            snapshots={},
        )

