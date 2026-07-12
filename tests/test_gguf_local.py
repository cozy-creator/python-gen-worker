"""cl#27: local-only GGUF rungs — ladder pick, composed snapshot, loader lane.

Real codepaths, no torch/network: the ladder walk is pure, composition works
on typed manifests, and detection reads real files on disk.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from gen_worker.api.binding import Hub
from gen_worker.models.gguf_local import (
    GGUF_MARKER,
    compose_resolved,
    composed_digest,
    read_marker,
    write_marker,
)
from gen_worker.models.hub_client import (
    WorkerResolvedFlavor,
    WorkerResolvedRepo,
    WorkerResolvedRepoFile,
)
from gen_worker.models.ladder import (
    FlavorRow,
    LadderModel,
    gguf_fit_bounds,
    gguf_qtype,
    is_denoiser_weight_path,
    is_te_weight_path,
    ladder_model_from_resolved,
    resolve_local_bindings,
    resolve_local_gguf,
)

GB = 1_000_000_000


# ---------------------------------------------------------------------------
# token + path predicates
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("token,qtype", [
    ("gguf-q4_k_m", "q4_k_m"),
    ("gguf-q8_0", "q8_0"),
    ("GGUF-Q8_0", "q8_0"),
    ("gguf-iq4_xs", ""),  # IQ-quants never load in diffusers
    ("gguf-bf16", ""),
    ("gguf", ""),
    ("fp8", ""),
    ("", ""),
])
def test_gguf_qtype(token: str, qtype: str) -> None:
    assert gguf_qtype(token) == qtype


@pytest.mark.parametrize("path,is_weight", [
    ("transformer/diffusion_pytorch_model-00001-of-00004.safetensors", True),
    ("transformer/diffusion_pytorch_model.safetensors.index.json", True),
    ("unet/diffusion_pytorch_model.safetensors", True),
    ("transformer/config.json", False),
    ("text_encoder/model.safetensors", False),
    ("vae/diffusion_pytorch_model.safetensors", False),
    ("model_index.json", False),
    ("transformer", False),
])
def test_is_denoiser_weight_path(path: str, is_weight: bool) -> None:
    assert is_denoiser_weight_path(path) == is_weight


# ---------------------------------------------------------------------------
# the local gguf rung walk
# ---------------------------------------------------------------------------

def _klein_like(*, non_denoiser_gb: float = 1.2, tree: bool = True) -> LadderModel:
    return LadderModel(
        base_size_gb=19.0,
        flavors=(
            FlavorRow(token="gguf-q8_0", size_gb=9.8),
            FlavorRow(token="gguf-q4_k_m", size_gb=5.6),
        ),
        non_denoiser_gb=non_denoiser_gb,
        diffusers_tree=tree,
    )


def test_gguf_pick_quality_descending() -> None:
    m = _klein_like()
    # Plenty of room: q8_0 (higher quality) wins.
    res = resolve_local_gguf(m, gpu_sm=89, free_vram_gb=16.0)
    assert res is not None and res.flavor == "gguf-q8_0"
    # 8GB card: q8_0 (9.8 + 1.2) misses, q4_k_m (5.6 + 1.2) fits.
    res = resolve_local_gguf(m, gpu_sm=89, free_vram_gb=7.4)
    assert res is not None and res.flavor == "gguf-q4_k_m"
    # Nothing fits.
    assert resolve_local_gguf(m, gpu_sm=89, free_vram_gb=4.0) is None


def test_gguf_fit_uses_real_size_plus_non_denoiser() -> None:
    m = _klein_like(non_denoiser_gb=3.0)
    # q4_k_m needs 5.6 + 3.0 = 8.6 — an 8.0 card refuses it now.
    assert resolve_local_gguf(m, gpu_sm=89, free_vram_gb=8.0) is None
    assert resolve_local_gguf(m, gpu_sm=89, free_vram_gb=8.7) is not None


def test_blackwell_never_gguf() -> None:
    # fp4 cores (SM >= 100): nvfp4/svdq-fp4 territory — GGUF never picked.
    m = _klein_like()
    for sm in (100, 120, 121):
        assert resolve_local_gguf(m, gpu_sm=sm, free_vram_gb=16.0) is None


def test_gguf_requires_diffusers_tree() -> None:
    assert resolve_local_gguf(
        _klein_like(tree=False), gpu_sm=89, free_vram_gb=16.0) is None


def _resolved_klein(*, fp8: bool = False) -> WorkerResolvedRepo:
    files = [
        WorkerResolvedRepoFile(path="model_index.json", size_bytes=1000, blake3="aa", url="u"),
        WorkerResolvedRepoFile(
            path="transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
            size_bytes=9 * GB, blake3="bb", url="u"),
        WorkerResolvedRepoFile(
            path="transformer/diffusion_pytorch_model-00002-of-00002.safetensors",
            size_bytes=8 * GB, blake3="cc", url="u"),
        WorkerResolvedRepoFile(path="transformer/config.json", size_bytes=1000, blake3="dd", url="u"),
        WorkerResolvedRepoFile(path="text_encoder/model.safetensors", size_bytes=1 * GB, blake3="ee", url="u"),
        WorkerResolvedRepoFile(path="vae/diffusion_pytorch_model.safetensors", size_bytes=200_000_000, blake3="ff", url="u"),
    ]
    sibs = [
        WorkerResolvedFlavor(flavor="gguf-q8_0", size_bytes=int(9.8 * GB)),
        WorkerResolvedFlavor(flavor="gguf-q4_k_m", size_bytes=int(5.6 * GB)),
    ]
    if fp8:
        sibs.append(WorkerResolvedFlavor(flavor="fp8", size_bytes=9 * GB))
    return WorkerResolvedRepo(
        snapshot_digest="basedigest000", files=files,
        size_bytes=sum(f.size_bytes for f in files), sibling_flavors=sibs,
    )


def test_ladder_model_from_resolved_computes_non_denoiser() -> None:
    m = ladder_model_from_resolved(_resolved_klein())
    assert m.diffusers_tree is True
    # everything except the two transformer weight shards
    assert m.non_denoiser_gb == pytest.approx(1.2, abs=0.01)
    assert m.te_gb == pytest.approx(1.0, abs=0.01)
    assert {r.token for r in m.flavors} == {"gguf-q8_0", "gguf-q4_k_m"}


def test_te_weight_path() -> None:
    assert is_te_weight_path("text_encoder/model-00001-of-00004.safetensors")
    assert is_te_weight_path("text_encoder_2/model.safetensors")
    assert not is_te_weight_path("vae/diffusion_pytorch_model.safetensors")
    assert not is_te_weight_path("text_encoder")


def test_gguf_fit_bounds_klein_shapes() -> None:
    # Real klein-9b shape: 5.91 gguf, 16.4 TE, ~0.4 other overhead.
    m = LadderModel(non_denoiser_gb=16.8, te_gb=16.4, diffusers_tree=True,
                    flavors=(FlavorRow(token="gguf-q4_k_m", size_gb=5.91),))
    resident, offloaded = gguf_fit_bounds(m, 5.91)
    assert resident == pytest.approx(5.91 + 0.4 + 8.2, abs=0.01)
    assert offloaded == pytest.approx(5.91 + 0.4, abs=0.01)
    # 8GB card: resident bound misses, TE-offload bound serves it — the
    # 9B-on-8GB story needs the offload relief valve.
    assert resolve_local_gguf(m, gpu_sm=89, free_vram_gb=7.6) is not None
    assert resolve_local_gguf(
        m, gpu_sm=89, free_vram_gb=7.6, allow_te_offload=False) is None
    # klein-4b shape fits FULLY resident on the same card (TE fp8-halved).
    m4 = LadderModel(non_denoiser_gb=8.3, te_gb=8.05, diffusers_tree=True,
                     flavors=(FlavorRow(token="gguf-q4_k_m", size_gb=2.60),))
    got = resolve_local_gguf(m4, gpu_sm=89, free_vram_gb=7.6,
                             allow_te_offload=False)
    assert got is not None and got.flavor == "gguf-q4_k_m" 


def _caps(sm: int) -> SimpleNamespace:
    return SimpleNamespace(gpu_sm=sm, installed_libs=[])


def test_local_bindings_rebind_to_gguf_on_8gb() -> None:
    # sm89, 7.4 GB free: no native rung fits (base 18.2, fp8 cast 13.6) ->
    # gguf-q4_k_m (5.6 + 1.2) is the pick.
    out = resolve_local_bindings(
        {"pipeline": Hub("tensorhub/flux2-klein-9b")},
        caps=_caps(89), free_vram_gb=7.4,
        resolver=lambda thref: _resolved_klein(),
    )
    assert out["pipeline"].flavor == "gguf-q4_k_m"
    assert out["pipeline"].storage_dtype == ""


def test_local_bindings_prefer_native_fp8_over_gguf() -> None:
    # A stored #fp8 flavor that fits outranks every gguf rung.
    out = resolve_local_bindings(
        {"pipeline": Hub("tensorhub/flux2-klein-9b")},
        caps=_caps(89), free_vram_gb=10.0,
        resolver=lambda thref: _resolved_klein(fp8=True),
    )
    assert out["pipeline"].flavor == "fp8"


def test_local_bindings_no_gguf_on_blackwell() -> None:
    out = resolve_local_bindings(
        {"pipeline": Hub("tensorhub/flux2-klein-9b")},
        caps=_caps(120), free_vram_gb=7.4,
        resolver=lambda thref: _resolved_klein(),
    )
    # nothing fits natively and gguf is refused -> declared binding kept
    assert out["pipeline"].flavor == ""


def test_local_bindings_author_pin_never_laddered() -> None:
    pinned = Hub("tensorhub/flux2-klein-9b", flavor="fp8")
    out = resolve_local_bindings(
        {"pipeline": pinned},
        caps=_caps(89), free_vram_gb=7.4,
        resolver=lambda thref: _resolved_klein(),
    )
    assert out["pipeline"] is pinned


# ---------------------------------------------------------------------------
# composed snapshot manifest
# ---------------------------------------------------------------------------

def _flavor_resolved() -> WorkerResolvedRepo:
    return WorkerResolvedRepo(
        snapshot_digest="flavordigest111",
        files=[
            WorkerResolvedRepoFile(
                path="flux2-klein-9b-Q4_K_M.gguf",
                size_bytes=int(5.6 * GB), blake3="99", url="u"),
            WorkerResolvedRepoFile(path="README.md", size_bytes=10, blake3="98", url="u"),
        ],
    )


def test_compose_resolved_merges_and_drops_denoiser() -> None:
    composed, gguf_rel = compose_resolved(_resolved_klein(), _flavor_resolved())
    paths = {f.path for f in composed.files}
    assert gguf_rel == "flux2-klein-9b-Q4_K_M.gguf"
    assert "flux2-klein-9b-Q4_K_M.gguf" in paths
    assert "model_index.json" in paths
    assert "transformer/config.json" in paths
    assert not any(is_denoiser_weight_path(p) for p in paths)
    assert composed.snapshot_digest == composed_digest("flavordigest111", "basedigest000")
    assert composed.snapshot_digest not in ("flavordigest111", "basedigest000")


def test_compose_resolved_rejects_bad_shapes() -> None:
    base = _resolved_klein()
    flavor = _flavor_resolved()
    two_ggufs = WorkerResolvedRepo(
        snapshot_digest="x",
        files=flavor.files + [WorkerResolvedRepoFile(
            path="second.gguf", size_bytes=1, blake3="97", url="u")],
    )
    with pytest.raises(ValueError, match="exactly 1"):
        compose_resolved(base, two_ggufs)
    no_index = WorkerResolvedRepo(
        snapshot_digest="y",
        files=[f for f in base.files if f.path != "model_index.json"],
    )
    with pytest.raises(ValueError, match="model_index"):
        compose_resolved(no_index, flavor)


# ---------------------------------------------------------------------------
# on-disk detection (marker + structural fallback)
# ---------------------------------------------------------------------------

def _make_composed_dir(tmp_path: Path, *, marker: bool = True) -> Path:
    snap = tmp_path / "snap"
    (snap / "transformer").mkdir(parents=True)
    (snap / "model_index.json").write_text(json.dumps({
        "_class_name": "Flux2KleinPipeline",
        "transformer": ["diffusers", "Flux2Transformer2DModel"],
        "vae": ["diffusers", "AutoencoderKL"],
    }))
    (snap / "transformer" / "config.json").write_text("{}")
    (snap / "flux2-klein-9b-Q4_K_M.gguf").write_bytes(b"GGUF")
    if marker:
        write_marker(snap, flavor="gguf-q4_k_m",
                     gguf_relpath="flux2-klein-9b-Q4_K_M.gguf")
    return snap


def test_marker_roundtrip(tmp_path: Path) -> None:
    snap = _make_composed_dir(tmp_path)
    m = read_marker(snap)
    assert m == {
        "flavor": "gguf-q4_k_m", "qtype": "q4_k_m",
        "gguf_path": "flux2-klein-9b-Q4_K_M.gguf",
    }
    assert (snap / GGUF_MARKER).exists()


def test_detect_gguf_snapshot(tmp_path: Path) -> None:
    from gen_worker.models.loading import detect_gguf_snapshot

    snap = _make_composed_dir(tmp_path)
    got = detect_gguf_snapshot(snap)
    assert got is not None
    assert got[0].name == "flux2-klein-9b-Q4_K_M.gguf" and got[1] == "q4_k_m"

    # structural fallback: marker lost, sole gguf + model_index still detect
    (snap / GGUF_MARKER).unlink()
    got = detect_gguf_snapshot(snap)
    assert got is not None and got[1] == "q4_k_m"

    # a plain diffusers tree (no gguf) is not the lane
    (snap / "flux2-klein-9b-Q4_K_M.gguf").unlink()
    assert detect_gguf_snapshot(snap) is None


def test_detect_requires_model_index(tmp_path: Path) -> None:
    from gen_worker.models.loading import detect_gguf_snapshot

    d = tmp_path / "flavoronly"
    d.mkdir()
    (d / "model-Q8_0.gguf").write_bytes(b"GGUF")
    # a bare denoiser-only flavor snapshot is NOT loadable composition
    assert detect_gguf_snapshot(d) is None


# ---------------------------------------------------------------------------
# listing probe (fail-open, override only when the ladder would pick gguf)
# ---------------------------------------------------------------------------

def test_gguf_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    from gen_worker.cli.listing import _gguf_probe
    from gen_worker.models import hub_client

    monkeypatch.setattr(hub_client, "resolve_repo",
                        lambda thref, **kw: _resolved_klein())
    caps = SimpleNamespace(gpu_sm=89, installed_libs=[])
    got = _gguf_probe(Hub("tensorhub/flux2-klein-9b"), caps, 7.4)
    assert got is not None and got["flavor"] == "gguf-q4_k_m"
    assert got["est_gb"] == pytest.approx(6.3, abs=0.05)
    assert got["te_offload"] is False

    # Blackwell: never
    assert _gguf_probe(Hub("tensorhub/flux2-klein-9b"),
                       SimpleNamespace(gpu_sm=120, installed_libs=[]), 7.4) is None
    # author-pinned flavor: never
    assert _gguf_probe(Hub("tensorhub/flux2-klein-9b", flavor="fp8"), caps, 7.4) is None
    # resolver failure: fail-open
    def _boom(thref, **kw):
        raise RuntimeError("down")
    monkeypatch.setattr(hub_client, "resolve_repo", _boom)
    assert _gguf_probe(Hub("tensorhub/flux2-klein-9b"), caps, 7.4) is None
