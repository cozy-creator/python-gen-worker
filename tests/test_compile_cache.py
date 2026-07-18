"""#384: per-SKU torch.compile cache artifacts — key, packaging, safety policy."""

from __future__ import annotations

import io
import tarfile
import threading

import msgspec
import pytest

from gen_worker import Compile, Resources, endpoint
from gen_worker import compile_cache as cc
from gen_worker.registry import collect_from_namespace


# ---------------------------------------------------------------------------
# key
# ---------------------------------------------------------------------------


def test_sku_slug():
    assert cc.sku_slug("NVIDIA GeForce RTX 4090") == "rtx-4090"
    assert cc.sku_slug("NVIDIA H100 80GB HBM3") == "h100-80gb-hbm3"
    assert cc.sku_slug("NVIDIA RTX 5090") == "rtx-5090"
    assert cc.sku_slug("") == ""


def test_flavor_label():
    assert cc.flavor_label("rtx-4090", "2.9.1+cu128") == "inductor-rtx-4090-torch2.9"
    assert cc.flavor_label("h100-80gb-hbm3", "2.11.0") == "inductor-h100-80gb-hbm3-torch2.11"


def test_cell_lane_is_exact_and_checkpoint_free():
    assert cc.cell_lane(
        "_system/family-sdxl#inductor-rtx-4090-torch2.13-w8a8"
    ) == "w8a8"
    assert cc.cell_lane(
        "_system/family-sdxl#inductor-rtx-4090-torch2.13"
    ) == ""
    assert cc.cell_lane("owner/checkpoint#fp8-w8a8") == ""


@pytest.mark.parametrize(
    ("lane", "bucket"),
    [
        ("", 0),
        ("fp8-hooks", 0),
        ("w8a16", 0),
        ("w8a8", 0),
        ("lora16", 16),
        ("fp8-hooks-lora32", 32),
        ("w8a16-lora64", 64),
        ("w8a8-lora128", 128),
    ],
)
def test_compile_target_lane_vocabulary_matches_tensorhub(lane, bucket):
    assert cc.compile_target_lane_error(lane, bucket) == ""


@pytest.mark.parametrize(
    ("lane", "bucket"),
    [
        ("w8a8-row", 0),
        ("fp8", 0),
        ("w8a8-lora8", 8),
        ("w8a8-lora32-sparse", 32),
        ("w8a8-lora32", 16),
        ("w8a8", 32),
    ],
)
def test_compile_target_lane_vocabulary_rejects_impossible_states(lane, bucket):
    assert cc.compile_target_lane_error(lane, bucket)


def test_verify_mismatches():
    meta = cc.artifact_metadata(family="sd15", shapes=[(768, 768)], targets=["transformer"])
    assert cc.verify(meta, family="sd15") == ""
    assert cc.verify(meta, family="") == ""  # consumer without a family: key-only check

    other = dict(meta, torch="0.0.0")
    assert "torch" in cc.verify(other, family="sd15")

    other = dict(meta, sku="not-this-gpu")
    assert "sku" in cc.verify(other, family="sd15")

    assert "family" in cc.verify(meta, family="sdxl")

    other = dict(meta, format=99)
    assert "format" in cc.verify(other, family="sd15")

    # gw#391: producer gen-worker version is part of the key — a cell built
    # by other gen-worker code must never be adopted (graph drift => FX miss).
    assert meta["gen_worker"] == cc.gen_worker_version() != ""
    other = dict(meta, gen_worker="0.9.2-not-this")
    assert "gen_worker" in cc.verify(other, family="sd15")
    other = dict(meta)
    del other["gen_worker"]
    assert "gen_worker" in cc.verify(other, family="sd15")

    for field in ("sm", "cuda", "image_digest"):
        other = dict(meta)
        other[field] = "definitely-not-this-runtime"
        reason = cc.verify(other, family="sd15")
        # the named-axis contract (gw#577): refusals carry axis + both values
        assert field in reason and "definitely-not-this-runtime" in reason


def test_verify_ignores_cuda_driver_host_lottery():
    """gw#577: a cell minted on one host must deliver to another same-SKU
    same-image host running a different driver build. Inductor/triton
    artifacts are keyed by torch/triton/cuda-runtime/SM arch (triton's disk
    key is source+ptxas-version+arch, ptxas ships in the wheel); the host
    libcuda build keys nothing — pinning it made fleet delivery a lottery
    (2 of 3 fresh B200 pods refused a proven cell, ie#495 flip rollback)."""
    meta = cc.artifact_metadata(
        family="sd15", shapes=[(768, 768)], targets=["transformer"])
    other = dict(meta, cuda_driver="13010")
    assert cc.verify(other, family="sd15") == ""


def test_execution_contract_uses_structure_not_checkpoint_values():
    torch = pytest.importorskip("torch")

    class _Pipe:
        def __init__(self, hidden: int, fill: float) -> None:
            self.transformer = torch.nn.Sequential(
                torch.nn.Linear(hidden, hidden), torch.nn.SiLU(),
            )
            with torch.no_grad():
                self.transformer[0].weight.fill_(fill)

    cfg = Compile(shapes=((1024, 1024),), targets=("transformer",), family="sdxl")
    a_sig, a_weights = cc.execution_contract(_Pipe(16, 1.0), cfg)
    b_sig, b_weights = cc.execution_contract(_Pipe(16, 9.0), cfg)
    c_sig, _ = cc.execution_contract(_Pipe(32, 1.0), cfg)
    assert a_sig == b_sig
    assert a_weights == b_weights == {"lane": ""}
    assert c_sig != a_sig


def test_execution_contract_records_dynamic_w8a8_exclusions():
    torch = pytest.importorskip("torch")

    class _Scaled(torch.nn.Module):
        _cozy_w8a8_linear = True

        def __init__(self) -> None:
            super().__init__()
            self.in_features = self.out_features = 16
            self.register_buffer("weight", torch.empty(
                16, 16, dtype=getattr(torch, "float8_e4m3fn")))
            self.register_buffer("weight_scale", torch.ones(16, 1))
            self.input_scale = None

        def forward(self, x):
            return x

    class _Target(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fast = _Scaled()
            self.sensitive = torch.nn.Linear(16, 16)

    class _Pipe:
        def __init__(self) -> None:
            self.transformer = _Target()
            self._cozy_weight_lane = "w8a8"

    cfg = Compile(shapes=((1024, 1024),), targets=("transformer",), family="sdxl")
    _signature, contract = cc.execution_contract(_Pipe(), cfg)
    assert contract["operator"] == "torch._scaled_mm"
    assert contract["activation_scaling"] == ["dynamic-per-row"]
    assert [r["path"] for r in contract["quantized"]] == ["transformer:fast"]
    assert [r["path"] for r in contract["excluded"]] == ["transformer:sensitive"]


def test_execution_contract_digest_covers_every_runtime_graph_axis():
    torch = pytest.importorskip("torch")

    class _Scaled(torch.nn.Module):
        _cozy_w8a8_linear = True

        def __init__(self, *, per_tensor: bool, fill: float) -> None:
            super().__init__()
            self.in_features = self.out_features = 8
            self.register_buffer("weight", torch.full(
                (8, 8), fill, dtype=getattr(torch, "float8_e4m3fn")))
            self.register_buffer("weight_scale", torch.ones(8, 1))
            self.input_scale = None
            self.gemm_mode = "pertensor" if per_tensor else "rowwise"

        def forward(self, value):
            return value

    class _Target(torch.nn.Module):
        def __init__(self, *, per_tensor: bool, fill: float) -> None:
            super().__init__()
            self.proj = _Scaled(per_tensor=per_tensor, fill=fill)

    class _Pipe:
        def __init__(
            self, *, per_tensor: bool = False, fill: float = 1.0,
            low_vram: str = "",
        ) -> None:
            self.transformer = _Target(per_tensor=per_tensor, fill=fill)
            self._cozy_weight_lane = "w8a8"
            self._cozy_low_vram_mode = low_vram

    base_cfg = Compile(
        shapes=((768, 768),), targets=("transformer",), family="flux2-klein-4b")
    base = cc.execution_contract_digest(_Pipe(fill=1.0), base_cfg)

    # Checkpoint values are deliberately excluded: compatible fine-tunes
    # share a family cell.
    assert cc.execution_contract_digest(_Pipe(fill=9.0), base_cfg) == base
    # Every consumer compatibility axis changes the fence identity.
    assert cc.execution_contract_digest(_Pipe(per_tensor=True), base_cfg) != base
    assert cc.execution_contract_digest(
        _Pipe(), Compile(
            shapes=((1024, 1024),), targets=("transformer",),
            family="flux2-klein-4b")) != base
    assert cc.execution_contract_digest(
        _Pipe(), Compile(
            shapes=((768, 768),), targets=("transformer",),
            family="flux2-klein-4b", regional=True)) != base
    assert cc.execution_contract_digest(_Pipe(low_vram="model_offload"), base_cfg) != base


def test_w8a8_guard_never_retries_eager():
    calls = {"eager": 0}

    def eager(value):
        calls["eager"] += 1
        return value

    def broken(_value):
        raise RuntimeError("graph miss")

    guarded = cc._guarded(eager, broken, "transformer", fail_closed=True)
    with pytest.raises(cc.CompiledLaneUnavailableError, match="graph miss"):
        guarded(1)
    with pytest.raises(cc.CompiledLaneUnavailableError, match="graph miss"):
        guarded(2)
    assert calls["eager"] == 0


def test_guard_revocation_failure_latches_fail_closed_for_optional_lane():
    calls = {"compiled": 0, "eager": 0, "callback": 0}

    def eager(value):
        calls["eager"] += 1
        return value

    def broken(_value):
        calls["compiled"] += 1
        raise RuntimeError("graph failed")

    def revoke(_detail):
        calls["callback"] += 1
        raise RuntimeError("state path unavailable")

    signal = {"callback": revoke}
    guarded = cc._guarded(eager, broken, "transformer", failure_signal=signal)
    with pytest.raises(
        cc.CompiledLaneUnavailableError, match="revocation failed",
    ):
        guarded(1)
    with pytest.raises(
        cc.CompiledLaneUnavailableError, match="revocation failed",
    ):
        guarded(2)
    assert calls == {"compiled": 1, "eager": 0, "callback": 1}


def test_guard_records_cache_proof_on_the_exact_wrapped_object(monkeypatch):
    counters = iter((
        {"fxgraph_cache_hit": 10, "fxgraph_cache_miss": 2},
        {"fxgraph_cache_hit": 13, "fxgraph_cache_miss": 3},
    ))
    monkeypatch.setattr(cc, "inductor_counters", lambda: next(counters))
    signal = {
        "callback": None,
        "lock": threading.Lock(),
        "successful_calls": 0,
        "cache_hits": 0,
        "cache_misses": 0,
    }

    guarded = cc._guarded(
        lambda value: value,
        lambda value: value + 1,
        "transformer",
        failure_signal=signal,
    )
    assert guarded(4) == 5
    assert guarded(5) == 6
    assert signal["successful_calls"] == 2
    assert signal["cache_hits"] == 3
    assert signal["cache_misses"] == 1


def test_mode_drift():
    class _P:
        pass

    p = _P()
    meta = cc.artifact_metadata(family="sd15", shapes=[(768, 768)],
                                targets=["transformer"], low_vram_mode="off")
    assert meta["low_vram_mode"] == "off"
    assert cc.mode_drift({}, p) == ""  # producer recorded no mode: unenforced
    assert "low_vram_mode" in cc.mode_drift(meta, p)  # unprepped pipeline
    p._cozy_low_vram_mode = "off"
    assert cc.mode_drift(meta, p) == ""
    p._cozy_low_vram_mode = "vae_only"
    assert "low_vram_mode" in cc.mode_drift(meta, p)


def test_unwrap_restores_eager():
    class _Mod:
        def forward(self, x):  # pragma: no cover
            return x

    class _Pipe2:
        def __init__(self):
            self.transformer = _Mod()

    pipe = _Pipe2()
    original = pipe.transformer.forward
    # simulate an armed pipeline the way apply() records it
    pipe._cozy_compile = {
        "targets": ["transformer"],
        "shapes": [(768, 768)],
        "cache": True,
        "originals": [(pipe.transformer, "forward", original)],
    }
    pipe.transformer.forward = lambda x: x  # the "compiled" wrap
    assert cc.unwrap(pipe) is True
    assert pipe.transformer.forward == original
    assert getattr(pipe, "_cozy_compile", None) is None
    assert cc.unwrap(pipe) is False  # idempotent


def test_regional_clear_and_guard():
    """ie#381 regional mode: blocks are compiled in place (nn.Module.compile
    sets _compiled_call_impl); rollback must CLEAR them, and the guard's
    first failure does so before retrying eager."""

    class _Block:
        _compiled_call_impl = None

    class _Mod:
        def __init__(self):
            self.b1, self.b2 = _Block(), _Block()

        def modules(self):
            return [self, self.b1, self.b2]

        def forward(self, x):
            if getattr(self.b1, "_compiled_call_impl", None) is not None:
                raise RuntimeError("compiled block exploded")
            return x + 1

    mod = _Mod()
    mod.b1._compiled_call_impl = object()  # "compiled"
    mod.b2._compiled_call_impl = object()
    guarded = cc._guarded_regional(mod, mod.forward, "transformer")
    assert guarded(1) == 2  # failure -> cleared -> eager retry succeeds
    assert mod.b1._compiled_call_impl is None
    assert mod.b2._compiled_call_impl is None
    assert guarded(2) == 3  # stays eager


def test_unwrap_clears_regional_mods():
    class _Block:
        _compiled_call_impl = object()

    class _Mod:
        def __init__(self):
            self.block = _Block()

        def modules(self):
            return [self, self.block]

    class _Pipe3:
        pass

    pipe = _Pipe3()
    mod = _Mod()
    pipe._cozy_compile = {
        "targets": ["transformer"], "shapes": [(960, 544, 241)],
        "cache": True, "originals": [], "regional_mods": [mod],
    }
    assert cc.unwrap(pipe) is True
    assert mod.block._compiled_call_impl is None


def test_artifact_metadata_records_compile_mode():
    meta = cc.artifact_metadata(
        family="ltx-2.3", shapes=[(960, 544, 241)],
        targets=["transformer"], compile_mode="regional",
    )
    assert meta["compile_mode"] == "regional"
    default = cc.artifact_metadata(family="sd15", shapes=[(768, 768)], targets=["transformer"])
    assert default["compile_mode"] == "whole"


def test_system_repo():
    assert cc.system_repo("sd15") == "_system/family-sd15"
    with pytest.raises(ValueError):
        cc.system_repo("")


# ---------------------------------------------------------------------------
# pack / unpack
# ---------------------------------------------------------------------------


def _capture_tree(root):
    (root / "inductor" / "ab").mkdir(parents=True)
    (root / "inductor" / "ab" / "graph.py").write_text("code")
    (root / "inductor" / "stale.lock").write_text("")  # junk: excluded
    (root / "triton" / "kern").mkdir(parents=True)
    (root / "triton" / "kern" / "kernel.cubin").write_bytes(b"\x00\x01")
    return root


def _tree_snapshot(root):
    root = root.resolve()
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*")) if path.is_file()
    }


def test_pack_is_deterministic_and_roundtrips(tmp_path):
    src = _capture_tree(tmp_path / "cap")
    meta = cc.artifact_metadata(family="sd15", source_ref="owner/model", shapes=[(768, 768)], targets=["transformer"])

    a = cc.pack(src, tmp_path / "a.tar.gz", meta)
    b = cc.pack(src, tmp_path / "b.tar.gz", meta)
    assert a.read_bytes() == b.read_bytes()

    dest = tmp_path / "seed"
    got = cc.unpack(a, dest)
    assert got["family"] == "sd15"
    assert got["source_ref"] == "owner/model"
    assert (dest / "inductor" / "ab" / "graph.py").read_text() == "code"
    assert (dest / "triton" / "kern" / "kernel.cubin").read_bytes() == b"\x00\x01"
    assert not (dest / "inductor" / "stale.lock").exists()

    # merge: a second artifact lands next to the first
    (src / "inductor" / "cd").mkdir()
    (src / "inductor" / "cd" / "more.py").write_text("x")
    c = cc.pack(src, tmp_path / "c.tar.gz", meta)
    cc.unpack(c, dest)
    assert (dest / "inductor" / "ab" / "graph.py").exists()
    assert (dest / "inductor" / "cd" / "more.py").exists()


def test_unpack_rejects_traversal(tmp_path):
    evil = tmp_path / "evil.tar.gz"
    import gzip

    with open(evil, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz:
        with tarfile.open(fileobj=gz, mode="w") as tar:
            data = b"{}"
            ti = tarfile.TarInfo(cc.METADATA_NAME)
            ti.size = len(data)
            tar.addfile(ti, io.BytesIO(data))
            ti = tarfile.TarInfo("inductor/../../escape.py")
            ti.size = 1
            tar.addfile(ti, io.BytesIO(b"x"))
    with pytest.raises(ValueError, match="unsafe"):
        cc.unpack(evil, tmp_path / "seed")


def test_unpack_requires_metadata(tmp_path):
    bad = tmp_path / "bad.tar.gz"
    with tarfile.open(bad, mode="w:gz") as tar:
        ti = tarfile.TarInfo("inductor/x")
        ti.size = 1
        tar.addfile(ti, io.BytesIO(b"x"))
    with pytest.raises(ValueError, match="metadata"):
        cc.unpack(bad, tmp_path / "seed")


def test_failed_seed_never_mutates_live_cache(tmp_path):
    cache_dir = tmp_path / "cache"
    live = _capture_tree(cache_dir / "compile-cache")
    before = _tree_snapshot(live)

    src = _capture_tree(tmp_path / "candidate")
    (src / "inductor" / "ab" / "graph.py").write_text("must-not-land")
    mismatch = cc.artifact_metadata(
        family="sd15", shapes=[(768, 768)], targets=["transformer"])
    mismatch["torch"] = "not-this-runtime"
    wrong = cc.pack(src, tmp_path / "wrong.tar.gz", mismatch)
    with pytest.raises(cc.AdoptError, match="torch") as exc:
        cc.seed_artifact(wrong, "sd15", cache_dir)
    assert exc.value.reason == "key_mismatch"
    assert _tree_snapshot(live) == before

    unsafe = tmp_path / "unsafe.tar.gz"
    valid = cc.artifact_metadata(
        family="sd15", shapes=[(768, 768)], targets=["transformer"])
    with tarfile.open(unsafe, mode="w:gz") as tar:
        metadata = msgspec.json.encode(valid)
        info = tarfile.TarInfo(cc.METADATA_NAME)
        info.size = len(metadata)
        tar.addfile(info, io.BytesIO(metadata))
        info = tarfile.TarInfo("inductor/ab/graph.py")
        info.size = len(b"must-not-land")
        tar.addfile(info, io.BytesIO(b"must-not-land"))
        info = tarfile.TarInfo("../inductor/escape.py")
        info.size = 1
        tar.addfile(info, io.BytesIO(b"x"))
    with pytest.raises(cc.AdoptError, match="unsafe") as exc:
        cc.seed_artifact(unsafe, "sd15", cache_dir)
    assert exc.value.reason == "artifact_invalid"
    assert _tree_snapshot(live) == before

    corrupt = tmp_path / "corrupt.tar.gz"
    corrupt.write_bytes(b"not a tar archive")
    with pytest.raises(cc.AdoptError) as exc:
        cc.seed_artifact(corrupt, "sd15", cache_dir)
    assert exc.value.reason == "artifact_invalid"
    assert _tree_snapshot(live) == before


@pytest.mark.parametrize("mismatch", ["lane", "contract", "mode"])
def test_pipeline_mismatch_never_activates_staged_cache(
    tmp_path, monkeypatch, mismatch,
):
    class _Target:
        def forward(self, value):
            return value

    class _Pipeline:
        def __init__(self):
            self.transformer = _Target()

    pipe = _Pipeline()
    cfg = Compile(
        shapes=((768, 768),), family="sd15", targets=("transformer",),
    )
    signature, contract = cc.execution_contract(pipe, cfg)
    meta = cc.artifact_metadata(
        family="sd15", shapes=cfg.shapes, targets=cfg.targets,
        graph_signature=signature, weight_contract=contract,
    )
    if mismatch == "lane":
        meta["weight_lane"] = "fp8-hooks"
    elif mismatch == "contract":
        meta["graph_signature"] = "different-module-graph"
    else:
        meta["compile_mode"] = "regional"
    # A pre-key cell (no computable cell-key axes): gw#581 SELF-REQUESTED
    # cells escalate drift to CellSelectionBugError instead — own test below.
    # verify() skips an unrecorded sm; the key requires it, so this cell can
    # never be classified as this runtime's own request.
    meta.pop("sm", None)
    meta.pop("cell_key", None)

    cache_dir = tmp_path / "cache"
    live = _capture_tree(cache_dir / "compile-cache")
    before = _tree_snapshot(live)
    source = _capture_tree(tmp_path / "candidate")
    (source / "inductor" / "ab" / "graph.py").write_text("must-not-land")
    artifact = cc.pack(source, tmp_path / "cell.tar.gz", meta)
    monkeypatch.setattr(cc, "apply", lambda *args, **kwargs: False)

    assert cc.enable(pipe, cfg, cache_dir, artifact) is False
    assert _tree_snapshot(live) == before


# ---------------------------------------------------------------------------
# gw#577: fleet delivery axes + named refusal reasons
# ---------------------------------------------------------------------------


def _module_tree_pipe(wrapper_name: str):
    torch = pytest.importorskip("torch")

    class _Tree:
        def __init__(self):
            self.transformer = torch.nn.Sequential(
                torch.nn.Linear(16, 16), torch.nn.SiLU(),
            )

    return type(wrapper_name, (_Tree,), {})()


def test_execution_contract_ignores_pipeline_wrapper_class():
    """gw#577 axis (c): conversion traces via generic DiffusionPipeline
    (-> LTX2Pipeline) while serving wraps the SAME module tree in
    LTX2ConditionPipeline. torch.compile wraps target callables — the
    pipeline class never enters a traced graph — so the signature must key
    on the traced module structure only (dual-load probe: identical trees,
    sig 2625baca vs e0f356f5 under the old class-name hash)."""
    cfg = Compile(shapes=((1024, 1024),), targets=("transformer",), family="fam")
    conv_sig, conv_wc = cc.execution_contract(
        _module_tree_pipe("LTX2Pipeline"), cfg)
    serve_sig, serve_wc = cc.execution_contract(
        _module_tree_pipe("LTX2ConditionPipeline"), cfg)
    assert conv_sig == serve_sig
    assert conv_wc == serve_wc
    # genuine structural drift still produces a different signature
    torch = pytest.importorskip("torch")
    other = _module_tree_pipe("LTX2Pipeline")
    other.transformer = torch.nn.Sequential(torch.nn.Linear(32, 32))
    assert cc.execution_contract(other, cfg)[0] != conv_sig


def test_contract_drift_names_signature_values():
    """A genuine graph mismatch refuses with BOTH digests in the reason —
    never the bare 'module graph signature mismatch' that cost a raw
    dual-load probe pod to diagnose."""
    pipe = _module_tree_pipe("AnyPipeline")
    cfg = Compile(shapes=((1024, 1024),), targets=("transformer",), family="fam")
    sig, wc = cc.execution_contract(pipe, cfg)
    meta = cc.artifact_metadata(
        family="fam", shapes=cfg.shapes, targets=cfg.targets,
        graph_signature="0" * 64, weight_contract=wc)
    reason = cc.contract_drift(meta, pipe, cfg)
    assert "module graph signature" in reason
    assert "0" * 12 in reason and sig[:12] in reason


def test_w8a8_identity_gate_drops_cuda_driver_keeps_rest(monkeypatch):
    """W8A8 cells still require sm/cuda/image_digest identity, but not the
    host-lottery cuda_driver axis (gw#577)."""
    pytest.importorskip("torch")
    real_key = cc.runtime_key

    def prod_key():
        key = dict(real_key())
        key.update(sm="sm_100", cuda="13.0", cuda_driver="13000",
                   image_digest="sha256:feedface")
        return key

    monkeypatch.setattr(cc, "runtime_key", prod_key)
    pipe = _module_tree_pipe("Serving")
    pipe.transformer[0]._cozy_w8a8_linear = True
    pipe.transformer[0].gemm_mode = "rowwise"
    pipe._cozy_weight_lane = "w8a8"
    cfg = Compile(shapes=((1024, 1024),), targets=("transformer",), family="fam")
    sig, wc = cc.execution_contract(pipe, cfg)
    meta = cc.artifact_metadata(
        family="fam", shapes=cfg.shapes, targets=cfg.targets,
        weight_lane="w8a8", graph_signature=sig, weight_contract=wc)
    meta["cuda_driver"] = ""  # cell minted without the axis: fine
    assert cc.contract_drift(meta, pipe, cfg) == ""
    # a cell recording a DIFFERENT driver build still delivers (verify skips)
    meta["cuda_driver"] = "13010"
    assert cc.verify(meta, family="fam") == ""
    assert cc.contract_drift(meta, pipe, cfg) == ""
    # the honest axes stay fail-closed, with named values
    for field in ("sm", "cuda", "image_digest"):
        broken = dict(meta)
        broken[field] = ""
        assert field in cc.contract_drift(broken, pipe, cfg)


def test_w8a8_enable_refusal_carries_exact_reason(tmp_path, monkeypatch):
    """gw#577 axis (a): the raised CompiledLaneUnavailableError is the ONLY
    wire-visible diagnostic on a serve pod — it must carry the exact
    mismatched axis and values, per refusal cause."""
    pytest.importorskip("torch")
    pipe = _module_tree_pipe("Serving")
    pipe._cozy_weight_lane = "w8a8"
    cfg = Compile(shapes=((768, 768),), targets=("transformer",), family="fam")
    cache_dir = tmp_path / "cache"

    # no artifact delivered
    with pytest.raises(cc.CompiledLaneUnavailableError, match="no cell artifact delivered"):
        cc.enable(pipe, cfg, cache_dir, artifact=None)

    # key mismatch: the axis and both values appear in the raise
    sig, wc = cc.execution_contract(pipe, cfg)
    meta = cc.artifact_metadata(
        family="fam", shapes=cfg.shapes, targets=cfg.targets,
        weight_lane="w8a8", graph_signature=sig, weight_contract=wc)
    meta["torch"] = "0.0.0+fake"
    source = _capture_tree(tmp_path / "cand")
    artifact = cc.pack(source, tmp_path / "cell.tar.gz", meta)
    with pytest.raises(cc.CompiledLaneUnavailableError) as exc:
        cc.enable(pipe, cfg, cache_dir, artifact=artifact)
    assert "torch" in str(exc.value) and "0.0.0+fake" in str(exc.value)

    # contract drift on a FOREIGN cell: the drift verdict appears in the
    # raise. (A self-keyed drifted cell is the th#883 cell_selection_bug
    # class instead — tests/test_cell_key.py.)
    meta = cc.artifact_metadata(
        family="fam", shapes=cfg.shapes, targets=cfg.targets,
        weight_lane="w8a8", graph_signature="1" * 64, weight_contract=wc)
    meta.pop("sm", None)
    meta.pop("cell_key", None)
    source2 = _capture_tree(tmp_path / "cand2")
    artifact2 = cc.pack(source2, tmp_path / "cell2.tar.gz", meta)
    with pytest.raises(cc.CompiledLaneUnavailableError) as exc:
        cc.enable(pipe, cfg, cache_dir, artifact=artifact2)
    assert "module graph signature" in str(exc.value)
    assert "1" * 12 in str(exc.value)


def test_build_refuses_w8a8_mint_without_serving_image_digest(tmp_path):
    """gw#577 finding (b): a w8a8 cell stamped with the PRODUCER image's
    digest can never be adopted by the fleet — refuse before any work."""
    with pytest.raises(RuntimeError, match="serving_image_digest"):
        cc.build(
            tmp_path, tmp_path / "out", shapes=[(768, 768)],
            family="fam", storage_dtype="fp8-w8a8",
        )


def test_cache_collision_and_merge_failure_leave_live_tree_unchanged(
    tmp_path, monkeypatch,
):
    cache_dir = tmp_path / "cache"
    live = _capture_tree(cache_dir / "compile-cache")
    before = _tree_snapshot(live)
    meta = cc.artifact_metadata(
        family="sd15", shapes=[(768, 768)], targets=["transformer"])

    collision_src = _capture_tree(tmp_path / "collision")
    (collision_src / "inductor" / "ab" / "graph.py").write_text("different")
    collision = cc.pack(collision_src, tmp_path / "collision.tar.gz", meta)
    with pytest.raises(cc.AdoptError) as exc:
        cc.seed_artifact(collision, "sd15", cache_dir)
    assert exc.value.reason == "cache_collision"
    assert _tree_snapshot(live) == before

    additive_src = tmp_path / "additive"
    (additive_src / "inductor" / "new-a").mkdir(parents=True)
    (additive_src / "inductor" / "new-a" / "a.py").write_text("a")
    (additive_src / "inductor" / "new-b").mkdir(parents=True)
    (additive_src / "inductor" / "new-b" / "b.py").write_text("b")
    additive = cc.pack(additive_src, tmp_path / "additive.tar.gz", meta)
    real_replace = cc.os.replace
    additions = 0

    def fail_second_addition(source, target):
        nonlocal additions
        if "new-" in str(target):
            additions += 1
            if additions == 2:
                raise OSError("injected merge failure")
        real_replace(source, target)

    monkeypatch.setattr(cc.os, "replace", fail_second_addition)
    with pytest.raises(cc.AdoptError) as exc:
        cc.seed_artifact(additive, "sd15", cache_dir)
    assert exc.value.reason == "activation_failed"
    assert _tree_snapshot(live) == before


def test_seed_activation_blocks_concurrent_cold_arming(
    tmp_path, monkeypatch,
):
    cache_dir = tmp_path / "cache"
    src = _capture_tree(tmp_path / "candidate")
    meta = cc.artifact_metadata(
        family="sd15", shapes=[(768, 768)], targets=["transformer"])
    artifact = cc.pack(src, tmp_path / "cell.tar.gz", meta)
    merging = threading.Event()
    release = threading.Event()
    observed = []
    errors = []
    real_merge = cc._merge_staged_cache

    def blocked_merge(staged, live):
        merging.set()
        assert release.wait(5)
        real_merge(staged, live)

    def observed_apply(_pipeline, _cfg, *, cache_ready):
        observed.append((cache_ready, _tree_snapshot(cache_dir / "compile-cache")))
        return True

    monkeypatch.setattr(cc, "_merge_staged_cache", blocked_merge)
    monkeypatch.setattr(cc, "apply", observed_apply)

    def seed():
        try:
            cc.seed_artifact(artifact, "sd15", cache_dir)
        except BaseException as exc:  # surfaced in the main test thread
            errors.append(exc)

    def cold_arm():
        try:
            cc.enable(
                _FakePipe(), Compile(shapes=((768, 768),), family="sd15"),
                cache_dir,
            )
        except BaseException as exc:  # surfaced in the main test thread
            errors.append(exc)

    seed_thread = threading.Thread(target=seed)
    seed_thread.start()
    assert merging.wait(5)
    arm_thread = threading.Thread(target=cold_arm)
    arm_thread.start()
    arm_thread.join(0.1)
    assert arm_thread.is_alive()
    assert observed == []
    release.set()
    seed_thread.join(5)
    arm_thread.join(5)
    assert not seed_thread.is_alive() and not arm_thread.is_alive()
    assert errors == []
    assert observed == [(False, {
        "inductor/ab/graph.py": b"code",
        "triton/kern/kernel.cubin": b"\x00\x01",
    })]


def test_capture_env_sets_cache_dirs(tmp_path, monkeypatch):
    monkeypatch.delenv("TORCHINDUCTOR_CACHE_DIR", raising=False)
    monkeypatch.delenv("TRITON_CACHE_DIR", raising=False)
    import os

    cc.capture_env(tmp_path / "cap")
    assert os.environ["TORCHINDUCTOR_CACHE_DIR"] == str(tmp_path / "cap" / "inductor")
    assert os.environ["TRITON_CACHE_DIR"] == str(tmp_path / "cap" / "triton")


# ---------------------------------------------------------------------------
# safety policy
# ---------------------------------------------------------------------------


class _FakePipe:
    pass


def test_apply_stays_eager_without_cache():
    """No verified artifact and no explicit cold opt-in => untouched pipeline."""
    pipe = _FakePipe()
    cfg = Compile(shapes=((768, 768),))
    assert cc.apply(pipe, cfg, cache_ready=False) is False
    assert getattr(pipe, "_cozy_compile", None) is None


def test_prepare_without_sources_is_none(tmp_path):
    assert cc.prepare("sd15", cache_dir=tmp_path) is None


def test_prepare_rejects_key_mismatch(tmp_path):
    src = _capture_tree(tmp_path / "cap")
    meta = cc.artifact_metadata(family="sd15", shapes=[(768, 768)], targets=["transformer"])
    meta["torch"] = "0.0.0"  # never matches this runtime
    art = cc.pack(src, tmp_path / "art.tar.gz", meta)
    assert cc.prepare(
        "sd15", cache_dir=tmp_path / "cache", artifact=art,
    ) is None


# ---------------------------------------------------------------------------
# declaration plumbing
# ---------------------------------------------------------------------------


def test_compile_struct_validation():
    c = Compile(shapes=[[768, 768], (1024, 1024)], family=" sd15 ")
    assert c.shapes == ((768, 768), (1024, 1024))
    assert c.targets == ("transformer", "vae.decode")
    assert c.family == "sd15"
    assert c.guidance_scales == ()
    assert c.regional is False
    assert Compile(shapes=((960, 544, 241),), targets=("transformer",),
                   family="ltx-2.3", regional=True).regional is True
    assert Compile(
        shapes=((1024, 1024),), guidance_scales=[5, 0],
    ).guidance_scales == (5.0, 0.0)
    with pytest.raises(ValueError):
        Compile(shapes=())
    with pytest.raises(ValueError):
        Compile(shapes=((0, 768),))
    with pytest.raises(ValueError):
        Compile(shapes=((768, 768),), targets=())
    with pytest.raises(ValueError, match="finite non-negative"):
        Compile(shapes=((768, 768),), guidance_scales=(float("nan"),))
    with pytest.raises(ValueError, match="finite non-negative"):
        Compile(shapes=((768, 768),), guidance_scales=(-1.0,))
    with pytest.raises(ValueError, match="duplicates"):
        Compile(shapes=((768, 768),), guidance_scales=(5.0, 5.0))


def test_compile_struct_video_shapes():
    """(w, h, frames) rows — video graphs key on the frame axis (ie#381)."""
    c = Compile(
        family="ltx-2.3",
        shapes=[[1280, 704, 121], (960, 544, 241), (1920, 1088, 241)],
        targets=("transformer",),
    )
    assert c.shapes == ((1280, 704, 121), (960, 544, 241), (1920, 1088, 241))
    # mixed image + video rows are fine (one endpoint, two modality functions)
    m = Compile(shapes=((768, 768), (1280, 704, 121)))
    assert m.shapes == ((768, 768), (1280, 704, 121))
    with pytest.raises(ValueError):
        Compile(shapes=((1280, 704, 0),))
    with pytest.raises(ValueError):
        Compile(shapes=((1280,),))
    with pytest.raises(ValueError):
        Compile(shapes=((1280, 704, 121, 24),))


def test_artifact_metadata_video_shapes_and_storage_dtype():
    meta = cc.artifact_metadata(
        family="ltx-2.3",
        shapes=[(1280, 704, 121), (1920, 1088, 241)],
        targets=["transformer"],
        storage_dtype="fp8",
    )
    assert meta["shapes"] == [[1280, 704, 121], [1920, 1088, 241]]
    assert meta["storage_dtype"] == "fp8"
    assert cc.verify(meta, family="ltx-2.3") == ""


def test_guidance_regimes_are_artifact_contract_axis():
    torch = pytest.importorskip("torch")

    class _Pipe:
        def __init__(self) -> None:
            self.transformer = torch.nn.Linear(16, 16)

    cfg = Compile(
        family="sdxl", shapes=((1024, 1024),), targets=("transformer",),
        guidance_scales=(5.0, 0.0),
    )
    meta = cc.artifact_metadata(
        family="sdxl", shapes=cfg.shapes, targets=cfg.targets,
        guidance_scales=cfg.guidance_scales,
    )
    assert meta["guidance_scales"] == [5.0, 0.0]
    assert cc.contract_drift(meta, _Pipe(), cfg) == ""
    assert "guidance_scales" in cc.contract_drift(
        dict(meta, guidance_scales=[5.0]), _Pipe(), cfg,
    )


def test_warm_call_captures_cfg_and_no_cfg(monkeypatch):
    torch = pytest.importorskip("torch")
    calls = []

    class _Generator:
        def __init__(self, device=None) -> None:
            self.device = device

        def manual_seed(self, _seed):
            return self

    class _Pipe:
        def __call__(self, *, guidance_scale=7.5, **_kwargs):
            calls.append(guidance_scale)

    monkeypatch.setattr(torch, "Generator", _Generator)
    cc._warm_call(
        _Pipe(), (1024, 1024), steps=2, prompt="warm", decode=False,
        guidance_scales=(5.0, 0.0),
    )
    assert calls == [5.0, 0.0]


class In(msgspec.Struct):
    prompt: str = ""


class Out(msgspec.Struct):
    ok: bool = True


def test_endpoint_compile_reaches_spec():
    import types

    @endpoint(
        resources=Resources(vram_gb=4),
        compile=Compile(shapes=((768, 768),), guidance_scales=(5.0, 0.0)),
    )
    class Ep:
        def setup(self) -> None:
            pass

        def gen(self, ctx, p: In) -> Out:
            return Out()

    mod = types.SimpleNamespace(Ep=Ep)
    specs = collect_from_namespace(mod)
    assert len(specs) == 1
    assert specs[0].compile is not None
    assert specs[0].compile.shapes == ((768, 768),)
    assert specs[0].compile.guidance_scales == (5.0, 0.0)

    with pytest.raises(TypeError, match="compile="):
        @endpoint(compile="yes")  # type: ignore[arg-type]
        def bad(ctx, p: In) -> Out:
            return Out()


def test_flavor_label_carries_weight_lane_gw534() -> None:
    from gen_worker.compile_cache import flavor_label, is_cache_ref, lane_token

    assert flavor_label("rtx-4090", "2.9.1+cu128") == "inductor-rtx-4090-torch2.9"
    assert flavor_label("h100-80gb-hbm3", "2.13.0+cu130", "w8a8") == (
        "inductor-h100-80gb-hbm3-torch2.13-w8a8")
    assert flavor_label("rtx-4090", "2.9.1", "fp8-hooks") == (
        "inductor-rtx-4090-torch2.9-w8a16")
    assert lane_token("") == "" and lane_token("w8a8") == "w8a8"
    assert is_cache_ref("_system/family-qwen-image#inductor-h100-80gb-hbm3-torch2.13-w8a8")


def test_resolve_pipeline_class_gw586() -> None:
    """gw#586 call-path parity: a mint may name the SERVING pipeline class;
    unknown names refuse loudly — a silent generic fallback would trace the
    wrong call path and publish a cell no serving lookup can hit."""
    from gen_worker.compile_cache import resolve_pipeline_class

    cls = resolve_pipeline_class("DiffusionPipeline")
    assert callable(getattr(cls, "from_pretrained", None))

    with pytest.raises(RuntimeError, match="wrong call path"):
        resolve_pipeline_class("NoSuchPipelineClass")
    with pytest.raises(RuntimeError, match="non-empty"):
        resolve_pipeline_class("   ")
    # A diffusers attribute that is not a loadable pipeline class refuses too.
    with pytest.raises(RuntimeError, match="wrong call path"):
        resolve_pipeline_class("__version__")
