"""#384: per-SKU torch.compile cache artifacts — key, packaging, safety policy."""

from __future__ import annotations

import io
import tarfile

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

    for field in ("sm", "cuda", "cuda_driver", "image_digest"):
        other = dict(meta)
        other[field] = "definitely-not-this-runtime"
        assert field in cc.verify(other, family="sd15")


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


def test_compile_struct_regional_flag():
    c = Compile(shapes=((960, 544, 241),), targets=("transformer",),
                family="ltx-2.3", regional=True)
    assert c.regional is True
    assert Compile(shapes=((768, 768),)).regional is False


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


def test_apply_stays_eager_without_cache(monkeypatch):
    """No verified artifact and no explicit cold opt-in => untouched pipeline."""
    monkeypatch.delenv(cc.ENV_ALLOW_COLD, raising=False)
    pipe = _FakePipe()
    cfg = Compile(shapes=((768, 768),))
    assert cc.apply(pipe, cfg, cache_ready=False) is False
    assert getattr(pipe, "_cozy_compile", None) is None


def test_prepare_without_sources_is_none(monkeypatch, tmp_path):
    monkeypatch.delenv(cc.ENV_CACHE_PATH, raising=False)
    monkeypatch.delenv(cc.ENV_CACHE_URL, raising=False)
    assert cc.prepare("sd15", cache_dir=tmp_path) is None


def test_prepare_rejects_key_mismatch(monkeypatch, tmp_path):
    src = _capture_tree(tmp_path / "cap")
    meta = cc.artifact_metadata(family="sd15", shapes=[(768, 768)], targets=["transformer"])
    meta["torch"] = "0.0.0"  # never matches this runtime
    art = cc.pack(src, tmp_path / "art.tar.gz", meta)
    monkeypatch.setenv(cc.ENV_CACHE_PATH, str(art))
    assert cc.prepare("sd15", cache_dir=tmp_path / "cache") is None


# ---------------------------------------------------------------------------
# declaration plumbing
# ---------------------------------------------------------------------------


def test_compile_struct_validation():
    c = Compile(shapes=[[768, 768], (1024, 1024)], family=" sd15 ")
    assert c.shapes == ((768, 768), (1024, 1024))
    assert c.targets == ("transformer", "vae.decode")
    assert c.family == "sd15"
    assert c.guidance_scales == ()
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
