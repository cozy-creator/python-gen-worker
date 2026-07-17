"""gw#390 TRT engine artifacts: key/label/verify, deterministic pack/unpack,
value-identity refit maps, and the executor's dual-kind dispatch (boot +
hot-adopt) — everything that runs without a GPU or tensorrt installed."""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
from pathlib import Path

import numpy as np
import pytest

from gen_worker import compile_cache as cc
from gen_worker import trt_engine as te
from gen_worker.pb import worker_scheduler_pb2 as pb

from test_executor_adopt import (  # noqa: F401 — shared harness
    CACHE_REF,
    DIGEST_A,
    FAMILY,
    OP_A,
    _adopt,
    _events,
    _spec,
    _wire_executor,
)

TRT_REF = f"_system/family-{FAMILY}#trt-rtx-4090-trt10.16-fp16"

RUNTIME = {"sku": "rtx-4090", "trt": "10.16.0.14", "cuda": "12.8"}


def _meta(**over):
    m = {
        "format": te.ARTIFACT_FORMAT, "kind": "trt-engine", **RUNTIME,
        "family": FAMILY, "module": "unet", "precision": "fp16", "batch": 2,
        "shapes": [[1024, 1024]], "inputs": [{"name": "sample"}],
        "source_ref": "", "source_digest": "",
    }
    m.update(over)
    return m


def _tar(tmp_path: Path, meta=None, name="trt-rtx-4090-trt10.16-fp16.tar.gz") -> Path:
    work = tmp_path / "work"
    work.mkdir(exist_ok=True)
    (work / te.ENGINE_NAME).write_bytes(b"\x00engine-bytes")
    (work / te.REFIT_MAP_NAME).write_text("[]")
    return te.pack(work, tmp_path / name, meta or _meta())


def _tree_snapshot(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*")) if path.is_file()
    }


# ---------------------------------------------------------------------------
# key / label / refs
# ---------------------------------------------------------------------------


def test_labels_and_refs():
    assert te.trt_maj_min("10.16.0.14") == "10.16"
    assert te.trt_maj_min("") == ""
    assert te.flavor_label("rtx-4090", "10.16.0.14", "fp16") == "trt-rtx-4090-trt10.16-fp16"
    assert te.flavor_label("", "10.16.0", "fp16") == ""
    assert te.is_engine_ref(TRT_REF)
    assert te.is_engine_ref(TRT_REF, FAMILY)
    assert not te.is_engine_ref(TRT_REF, "sdxl")
    assert not te.is_engine_ref(CACHE_REF)  # inductor flavor is not an engine
    assert not cc.is_cache_ref(TRT_REF)  # and vice versa — disjoint kinds
    assert not te.is_engine_ref("acme/model#trt-x")


def test_verify_exact_key(monkeypatch):
    monkeypatch.setattr(te, "runtime_key", lambda: dict(RUNTIME))
    assert te.verify(_meta()) == ""
    assert te.verify(_meta(), family=FAMILY) == ""
    assert "family" in te.verify(_meta(), family="sdxl")
    # FULL trt version must match — maj.min agreement is not enough.
    assert "trt" in te.verify(_meta(trt="10.16.0.99"))
    assert "sku" in te.verify(_meta(sku="h100-80gb-hbm3"))
    assert "cuda" in te.verify(_meta(cuda="13.0"))
    assert "kind" in te.verify(_meta(kind="torch-inductor-cache"))
    assert "format" in te.verify(_meta(format=99))


def test_verify_requires_tensorrt(monkeypatch):
    monkeypatch.setattr(te, "runtime_key", lambda: {"sku": "rtx-4090", "trt": "", "cuda": "12.8"})
    assert "tensorrt not installed" in te.verify(_meta())


# ---------------------------------------------------------------------------
# pack / unpack
# ---------------------------------------------------------------------------


def test_pack_deterministic_and_unpack_roundtrip(tmp_path):
    a = _tar(tmp_path, name="a.tar.gz")
    b = _tar(tmp_path, name="b.tar.gz")
    assert a.read_bytes() == b.read_bytes()

    meta = te.unpack(a, tmp_path / "out")
    assert meta["kind"] == "trt-engine"
    assert (tmp_path / "out" / te.ENGINE_NAME).read_bytes() == b"\x00engine-bytes"
    assert json.loads((tmp_path / "out" / te.REFIT_MAP_NAME).read_text()) == []
    assert te.unpack_metadata(a)["sku"] == "rtx-4090"


def test_unpack_rejects_unexpected_members(tmp_path):
    import gzip
    import io
    import tarfile

    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb") as gz:
        with tarfile.open(fileobj=gz, mode="w") as tar:
            data = b"evil"
            ti = tarfile.TarInfo("../escape.sh")
            ti.size = len(data)
            tar.addfile(ti, io.BytesIO(data))
    p = tmp_path / "evil.tar.gz"
    p.write_bytes(buf.getvalue())
    with pytest.raises(ValueError, match="unexpected member"):
        te.unpack(p, tmp_path / "out")


def test_rejected_stage_leaves_cache_byte_identical(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "sentinel").write_bytes(b"live-cache-bytes")
    before = _tree_snapshot(cache)
    artifact = _tar(tmp_path, meta=_meta(trt="10.16.0.99"))
    monkeypatch.setattr(te, "runtime_key", lambda: dict(RUNTIME))

    with pytest.raises(cc.AdoptError) as exc:
        te.stage_artifact(artifact, FAMILY, cache)
    assert exc.value.reason == "key_mismatch"
    assert _tree_snapshot(cache) == before
    assert list(cache.glob("trt-engine-stage-*")) == []


def test_concurrent_stages_are_isolated_and_never_publish_live_files(
    tmp_path, monkeypatch,
):
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "sentinel").write_bytes(b"unchanged")
    artifact = _tar(tmp_path)
    monkeypatch.setattr(te, "runtime_key", lambda: dict(RUNTIME))

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        staged = list(pool.map(
            lambda _index: te.stage_artifact(artifact, FAMILY, cache), range(4),
        ))
    try:
        assert len({item.root for item in staged}) == 4
        assert all((item.root / te.ENGINE_NAME).read_bytes() == b"\x00engine-bytes"
                   for item in staged)
        assert not (cache / "trt-engine" / te.ENGINE_NAME).exists()
        assert (cache / "sentinel").read_bytes() == b"unchanged"
    finally:
        for item in staged:
            item.close()
    assert list(cache.glob("trt-engine-stage-*")) == []


def test_failure_after_staging_cleans_isolated_tree_before_live_mutation(
    tmp_path, monkeypatch,
):
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "sentinel").write_bytes(b"unchanged")
    artifact = _tar(tmp_path)
    monkeypatch.setattr(te, "runtime_key", lambda: dict(RUNTIME))
    monkeypatch.setattr(
        te, "_load_engine", lambda _path: (_ for _ in ()).throw(
            RuntimeError("process died while deserializing")),
    )

    class _Module:
        def forward(self, value):
            return value

        def state_dict(self):
            return {}

    class _Pipeline:
        def __init__(self):
            self.unet = _Module()

    pipeline = _Pipeline()
    original = pipeline.unet.forward
    with pytest.raises(RuntimeError, match="process died"):
        te.load_and_wrap(pipeline, _spec().compile, artifact, cache)
    assert pipeline.unet.forward == original
    assert not hasattr(pipeline, te._MARKER_ATTR)
    assert (cache / "sentinel").read_bytes() == b"unchanged"
    assert list(cache.glob("trt-engine-stage-*")) == []


def test_different_cell_rearm_replaces_failed_wrapper_after_staged_proof(
    tmp_path, monkeypatch,
):
    """A new exact TRT cell replaces, rather than reuses, a failed guard."""
    cache = tmp_path / "cache"
    artifact = _tar(
        tmp_path,
        meta=_meta(source_digest="new-cell"),
        name="new-cell.tar.gz",
    )
    calls = {"eager": 0, "old": 0, "new": 0, "loads": 0}

    class _Module:
        device = "cuda"

        def forward(self, *_args, **_kwargs):
            calls["eager"] += 1
            return "eager"

        def state_dict(self):
            return {}

    class _Pipeline:
        def __init__(self):
            self.unet = _Module()

    class _OldRunner:
        def __call__(self, _feeds):
            calls["old"] += 1
            raise RuntimeError("old cell failed")

    class _NewRunner:
        def __init__(self, _engine, _meta, *, device):
            assert device == "cuda"

        def __call__(self, _feeds):
            calls["new"] += 1
            return "new-engine"

    def _load_engine(_path):
        calls["loads"] += 1
        return object()

    monkeypatch.setattr(te, "runtime_key", lambda: dict(RUNTIME))
    monkeypatch.setattr(te, "_unet_feeds", lambda *args, **kwargs: {})
    monkeypatch.setattr(te, "_load_engine", _load_engine)
    monkeypatch.setattr(te, "_refit_engine", lambda _engine, _weights: None)
    monkeypatch.setattr(te, "TrtModuleRunner", _NewRunner)

    pipeline = _Pipeline()
    old_meta = _meta(source_digest="old-cell")
    te.wrap_module(pipeline.unet, _OldRunner(), old_meta)
    old_module_marker = getattr(pipeline.unet, te._MARKER_ATTR)
    old_state = old_module_marker["state"]
    setattr(pipeline, te._MARKER_ATTR, {
        "meta": old_meta,
        "state": old_state,
        "module": pipeline.unet,
    })

    # Trip the old cell once. Its eager fallback remains the only valid
    # unwrapped callable, but the failed marker cannot satisfy a new cell.
    assert pipeline.unet.forward(object(), return_dict=False) == "eager"
    assert old_state["failed"] is True

    meta = te.load_and_wrap(pipeline, _spec().compile, artifact, cache)
    new_state = getattr(pipeline, te._MARKER_ATTR)["state"]
    assert meta["source_digest"] == "new-cell"
    assert new_state is not old_state
    assert pipeline.unet.forward(object(), return_dict=False) == ("new-engine",)
    assert calls == {"eager": 1, "old": 1, "new": 1, "loads": 1}
    assert te.execution_count(pipeline) == 1
    assert list(cache.glob("trt-engine-stage-*")) == []


def test_find_artifact(tmp_path):
    a = _tar(tmp_path)
    assert te.find_artifact(a) == a
    assert te.find_artifact(tmp_path) == a
    assert te.find_artifact(tmp_path / "nope") is None


def test_trt_guard_revocation_failure_never_falls_back(monkeypatch):
    calls = {"eager": 0, "runner": 0, "callback": 0}

    class _Module:
        def forward(self, *_args, **_kwargs):
            calls["eager"] += 1
            return "eager"

    class _Runner:
        def __call__(self, _feeds):
            calls["runner"] += 1
            raise RuntimeError("engine failed")

    monkeypatch.setattr(te, "_unet_feeds", lambda *args, **kwargs: {})
    module = _Module()
    te.wrap_module(module, _Runner(), _meta())
    state = getattr(module, te._MARKER_ATTR)["state"]

    def revoke(_detail):
        calls["callback"] += 1
        raise RuntimeError("state path unavailable")

    state["failure_callback"] = revoke
    with pytest.raises(cc.CompiledLaneUnavailableError, match="revocation failed"):
        module.forward(object())
    with pytest.raises(cc.CompiledLaneUnavailableError, match="revocation failed"):
        module.forward(object())
    assert calls == {"eager": 0, "runner": 1, "callback": 1}


# ---------------------------------------------------------------------------
# refit map — value-identity matching
# ---------------------------------------------------------------------------


class _T:
    def __init__(self, arr):
        self._arr = np.asarray(arr)

    def cpu(self):
        return self

    def numpy(self):
        return self._arr


def test_build_refit_map_matches_by_value_and_transpose():
    w1 = np.arange(12, dtype=np.float16).reshape(3, 4)
    w2 = np.arange(6, dtype=np.float16).reshape(6)
    sd = {"blocks.0.weight": _T(w1), "blocks.0.bias": _T(w2)}
    inits = {
        "onnx::MatMul_123": w1.T.copy(),  # exporter transposed it
        "blocks.0.bias": w2.copy(),       # exporter kept the name
        "mystery": np.ones(5, dtype=np.float16),
    }
    entries, unmatched = te.build_refit_map(inits, sd)
    by_name = {e["name"]: e for e in entries}
    assert by_name["onnx::MatMul_123"] == {
        "name": "onnx::MatMul_123", "key": "blocks.0.weight", "transform": "transpose"}
    assert by_name["blocks.0.bias"]["transform"] == ""
    assert unmatched == ["mystery"]


def test_refit_weights_materializes_const_entries():
    import base64

    arr = np.arange(6, dtype=np.float16).reshape(2, 3)
    out = te.refit_weights({}, [{
        "name": "/unet/x/Constant_1_output_0", "key": "", "transform": "const",
        "dtype": str(arr.dtype), "shape": list(arr.shape),
        "data_b64": base64.b64encode(arr.tobytes()).decode(),
    }])
    assert np.array_equal(out["/unet/x/Constant_1_output_0"], arr)
    assert out["/unet/x/Constant_1_output_0"].dtype == np.float16


def test_refit_weights_applies_transform_and_fails_closed():
    w = np.arange(12, dtype=np.float16).reshape(3, 4)
    sd = {"k": _T(w)}
    out = te.refit_weights(sd, [{"name": "n", "key": "k", "transform": "transpose"}])
    assert out["n"].shape == (4, 3)
    assert np.array_equal(out["n"], w.T)
    with pytest.raises(cc.AdoptError) as exc:
        te.refit_weights(sd, [{"name": "n", "key": "gone", "transform": ""}])
    assert exc.value.reason == "refit_missing_key"


# ---------------------------------------------------------------------------
# executor dispatch — boot attach + hot adopt route trt refs to trt_engine
# ---------------------------------------------------------------------------


def test_fetch_compile_snapshot_deterministically_prefers_trt(tmp_path):
    spec = _spec()
    ex, _sent = _wire_executor(spec, tmp_path)
    trt_dir = tmp_path / "trt-snap"
    trt_dir.mkdir()
    _tar(trt_dir)
    inductor_dir = tmp_path / "ind-snap"
    inductor_dir.mkdir()
    (inductor_dir / "inductor-rtx-4090-torch2.9.tar.gz").write_bytes(b"x")

    async def _ensure(ref, snapshot=None, *, binding=None):
        return trt_dir if te.is_engine_ref(ref) else inductor_dir

    ex.store.ensure_local = _ensure  # type: ignore[method-assign]
    for refs in ((CACHE_REF, TRT_REF), (TRT_REF, CACHE_REF)):
        snaps = {ref: pb.Snapshot(digest=DIGEST_A) for ref in refs}
        got = asyncio.run(ex._fetch_compile_snapshot(spec, snaps))
        assert got is not None
        assert got.path.parent == trt_dir
        assert got.ref == TRT_REF
        assert got.snapshot_digest == DIGEST_A


def test_adopt_trt_ref_wraps_and_reports(tmp_path, monkeypatch):
    (tmp_path / "snap").mkdir()
    _tar(tmp_path / "snap")
    spec = _spec()
    ex, sent = _wire_executor(spec, tmp_path)

    calls = []

    def _fake_load_and_wrap(pipeline, cfg, artifact, cache_dir=None):
        calls.append((pipeline, cfg, Path(artifact)))
        setattr(pipeline, te._MARKER_ATTR, {
            "meta": _meta(),
            "state": {"failure_callback": None},
            "module": pipeline.transformer,
        })
        return _meta()

    monkeypatch.setattr(te, "load_and_wrap", _fake_load_and_wrap)
    counts = iter((0, 1))
    monkeypatch.setattr(te, "execution_count", lambda _pipeline: next(counts))
    from test_executor_adopt import _Endpoint

    warmups_before = _Endpoint.warmups
    _adopt(ex, ref=TRT_REF)
    assert len(calls) == 1
    assert calls[0][2].name.endswith(".tar.gz")
    adopted = _events(sent, pb.MODEL_STATE_ADOPTED)
    assert len(adopted) == 1 and adopted[0].ref == TRT_REF
    assert adopted[0].operation_id == OP_A
    assert _Endpoint.warmups == warmups_before + 1  # one warmup after the swap


def test_adopt_trt_key_mismatch_stays_eager(tmp_path, monkeypatch):
    (tmp_path / "snap").mkdir()
    _tar(tmp_path / "snap")
    spec = _spec()
    ex, sent = _wire_executor(spec, tmp_path)

    def _mismatch(pipeline, cfg, artifact, cache_dir=None):
        raise cc.AdoptError("key_mismatch", "trt '10.16.0.14' != runtime '10.15.2.1'")

    monkeypatch.setattr(te, "load_and_wrap", _mismatch)
    _adopt(ex, ref=TRT_REF)
    failed = _events(sent, pb.MODEL_STATE_FAILED)
    assert len(failed) == 1 and failed[0].error == "adopt_failed:key_mismatch"
    assert failed[0].snapshot_digest == DIGEST_A
    assert failed[0].operation_id == OP_A
    assert not _events(sent, pb.MODEL_STATE_ADOPTED)


def test_adopt_trt_warmup_must_execute_engine_or_roll_back(tmp_path, monkeypatch):
    (tmp_path / "snap").mkdir()
    _tar(tmp_path / "snap")
    spec = _spec()
    ex, sent = _wire_executor(spec, tmp_path)
    monkeypatch.setattr(te, "load_and_wrap", lambda *args, **kwargs: _meta())
    monkeypatch.setattr(te, "execution_count", lambda _pipeline: 0)
    unwrapped = []
    monkeypatch.setattr(te, "unwrap", lambda pipeline: unwrapped.append(pipeline) or True)

    _adopt(ex, ref=TRT_REF)
    failed = _events(sent, pb.MODEL_STATE_FAILED)
    assert len(failed) == 1
    assert failed[0].error == "adopt_failed:engine_not_executed"
    assert unwrapped
    target = ex.compile_targets()[0]
    assert target.active_compile_ref == ""
    assert target.active_compile_snapshot_digest == ""


def test_enable_compiled_dispatches_on_artifact_kind(tmp_path, monkeypatch):
    spec = _spec()
    ex, _sent = _wire_executor(spec, tmp_path)
    trt_artifact = _tar(tmp_path)

    seen = {}

    def _trt_enable(p, c, d, a):
        seen["trt"] = Path(a)
        return True

    def _cc_enable(p, c, d, a):
        seen["cc"] = a
        return True

    monkeypatch.setattr(te, "enable", _trt_enable)
    monkeypatch.setattr(cc, "enable", _cc_enable)

    assert ex._enable_compiled(object(), spec.compile, trt_artifact)
    assert seen == {"trt": trt_artifact}

    seen.clear()
    cap = tmp_path / "cap"
    (cap / "inductor").mkdir(parents=True)
    (cap / "inductor" / "x").write_text("x")
    (cap / "triton").mkdir()
    ind = cc.pack(cap, tmp_path / "ind.tar.gz", cc.artifact_metadata(family=FAMILY))
    assert ex._enable_compiled(object(), spec.compile, ind)
    assert seen == {"cc": ind}
