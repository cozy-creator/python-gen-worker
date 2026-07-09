"""SVDQuant/nunchaku loader mode (gw#415): artifact detection, the pin
matrix, the svdq fit rungs, and the load_from_pretrained routing.

Real nunchaku/diffusers/CUDA are absent in CI — kernels and the live serve
band are the gw#415 5090 acceptance run."""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

import pytest

from gen_worker import Hub, Resources
from gen_worker.models.hub_policy import (
    FIT_FITS,
    FIT_INCOMPATIBLE,
    FIT_SVDQ_FP4,
    FIT_SVDQ_INT4,
    TensorhubWorkerCapabilities,
    select_variant,
    svdq_flavor_kind,
    variant_fit,
)
from gen_worker.models.svdq import (
    SvdqStackError,
    check_svdq_stack_versions,
    detect_svdq_artifact,
    svdq_precision_for_sm,
)


# --------------------------------------------------------------------------
# fixtures — synthetic nunchaku safetensors
# --------------------------------------------------------------------------

def _write_svdq_safetensors(
    path: Path,
    *,
    model_class: str = "NunchakuZImageTransformer2DModel",
    weight_dtype: str = "fp4_e2m1_all",
    rank: int = 128,
) -> None:
    qc = {
        "method": "svdquant",
        "weight": {"dtype": weight_dtype, "group_size": 16},
        "activation": {"dtype": weight_dtype, "group_size": 16},
        "rank": rank,
    }
    header = {
        "__metadata__": {
            "model_class": model_class,
            "quantization_config": json.dumps(qc),
            "config": json.dumps({"_class_name": "ZImageTransformer2DModel"}),
        },
        "w": {"dtype": "BF16", "shape": [4], "data_offsets": [0, 8]},
    }
    blob = json.dumps(header).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        f.write(b"\x00" * 8)


def _write_plain_safetensors(path: Path) -> None:
    header = {"w": {"dtype": "BF16", "shape": [4], "data_offsets": [0, 8]}}
    blob = json.dumps(header).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        f.write(b"\x00" * 8)


def _svdq_tree(tmp_path: Path, *, weight_dtype: str = "fp4_e2m1_all") -> Path:
    root = tmp_path / "snap"
    root.mkdir(parents=True, exist_ok=True)
    (root / "model_index.json").write_text(json.dumps({
        "_class_name": "ZImagePipeline",
        "transformer": ["diffusers", "ZImageTransformer2DModel"],
    }))
    _write_svdq_safetensors(
        root / "transformer" / "svdq-fp4_r128-z-image-turbo.safetensors",
        weight_dtype=weight_dtype,
    )
    _write_plain_safetensors(root / "vae" / "diffusion_pytorch_model.safetensors")
    return root


# --------------------------------------------------------------------------
# detection
# --------------------------------------------------------------------------

def test_detect_svdq_artifact_fp4_tree(tmp_path: Path) -> None:
    root = _svdq_tree(tmp_path)
    art = detect_svdq_artifact(root)
    assert art is not None
    assert art.component == "transformer"
    assert art.precision == "fp4"
    assert art.rank == 128
    assert art.model_class == "NunchakuZImageTransformer2DModel"


def test_detect_svdq_artifact_int4_file(tmp_path: Path) -> None:
    f = tmp_path / "svdq-int4_r128.safetensors"
    _write_svdq_safetensors(f, weight_dtype="int4")
    art = detect_svdq_artifact(f)
    assert art is not None
    assert art.component == ""
    assert art.precision == "int4"


def test_detect_svdq_artifact_negative(tmp_path: Path) -> None:
    root = tmp_path / "plain"
    _write_plain_safetensors(root / "transformer" / "diffusion_pytorch_model.safetensors")
    (root / "model_index.json").write_text("{}")
    assert detect_svdq_artifact(root) is None


# --------------------------------------------------------------------------
# pin matrix
# --------------------------------------------------------------------------

def test_pin_matrix_accepts_verified_stack() -> None:
    check_svdq_stack_versions(
        nunchaku_version="1.2.1+cu13.0torch2.11",
        diffusers_version="0.36.0",
        torch_version="2.11.0+cu130",
        cuda_version="13.0",
    )


@pytest.mark.parametrize("diffusers_version", ["0.38.0.dev0", "0.39.0", "0.35.1"])
def test_pin_matrix_rejects_diffusers_window(diffusers_version: str) -> None:
    with pytest.raises(SvdqStackError, match="diffusers"):
        check_svdq_stack_versions(
            nunchaku_version="1.2.1", diffusers_version=diffusers_version,
        )


def test_pin_matrix_rejects_unknown_nunchaku() -> None:
    with pytest.raises(SvdqStackError, match="pin matrix"):
        check_svdq_stack_versions(
            nunchaku_version="1.4.0", diffusers_version="0.36.0",
        )


def test_pin_matrix_rejects_wheel_torch_mismatch() -> None:
    with pytest.raises(SvdqStackError, match="torch"):
        check_svdq_stack_versions(
            nunchaku_version="1.2.1+cu12.8torch2.10",
            diffusers_version="0.36.0",
            torch_version="2.11.0+cu128",
            cuda_version="12.8",
        )


def test_pin_matrix_rejects_wheel_cuda_mismatch() -> None:
    with pytest.raises(SvdqStackError, match="CUDA"):
        check_svdq_stack_versions(
            nunchaku_version="1.2.1+cu13.0torch2.11",
            diffusers_version="0.36.0",
            torch_version="2.11.0+cu128",
            cuda_version="12.8",
        )


def test_precision_for_sm_windows() -> None:
    assert svdq_precision_for_sm(120) == "fp4"
    assert svdq_precision_for_sm(121) == "fp4"
    assert svdq_precision_for_sm(89) == "int4"
    assert svdq_precision_for_sm(75) == "int4"
    # Hopper + datacenter Blackwell are deliberately OUT (TRT lane).
    assert svdq_precision_for_sm(90) == ""
    assert svdq_precision_for_sm(100) == ""


# --------------------------------------------------------------------------
# fit rungs
# --------------------------------------------------------------------------

def _caps(sm: int, libs: list[str] | None = None) -> TensorhubWorkerCapabilities:
    return TensorhubWorkerCapabilities(
        cuda_version="13.0", gpu_sm=sm, torch_version="2.11.0",
        installed_libs=libs if libs is not None else ["nunchaku"],
    )


_FP4_BINDING = Hub("Tongyi-MAI/Z-Image-Turbo", flavor="svdq-fp4-r128")
_INT4_BINDING = Hub("Tongyi-MAI/Z-Image-Turbo", flavor="svdq-int4-r128")
_PLAIN_BINDING = Hub("Tongyi-MAI/Z-Image-Turbo")


def test_svdq_flavor_kind() -> None:
    assert svdq_flavor_kind(_FP4_BINDING) == "fp4"
    assert svdq_flavor_kind(_INT4_BINDING) == "int4"
    assert svdq_flavor_kind(_PLAIN_BINDING) == ""
    assert svdq_flavor_kind(None) == ""


@pytest.fixture
def stack_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "gen_worker.models.svdq.svdq_stack_reason", lambda: None,
    )


def test_variant_fit_svdq_fp4_on_blackwell(stack_ok: None) -> None:
    fit, reason = variant_fit(
        Resources(vram_gb=14), _caps(120), 30.0, binding=_FP4_BINDING,
    )
    assert fit == FIT_SVDQ_FP4
    assert "svdq-fp4" in reason


def test_variant_fit_svdq_fp4_rejected_off_blackwell(stack_ok: None) -> None:
    for sm in (89, 90, 100):
        fit, reason = variant_fit(
            Resources(vram_gb=14), _caps(sm), 30.0, binding=_FP4_BINDING,
        )
        assert fit == FIT_INCOMPATIBLE, sm
        assert "SM" in reason


def test_variant_fit_svdq_int4_window(stack_ok: None) -> None:
    fit, _ = variant_fit(
        Resources(vram_gb=10), _caps(89), 20.0, binding=_INT4_BINDING,
    )
    assert fit == FIT_SVDQ_INT4
    fit, _ = variant_fit(
        Resources(vram_gb=10), _caps(120), 20.0, binding=_INT4_BINDING,
    )
    assert fit == FIT_INCOMPATIBLE


def test_variant_fit_svdq_pin_violation_is_typed_incompatible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "gen_worker.models.svdq.svdq_stack_reason",
        lambda: "nunchaku 1.2.1 requires diffusers>=0.36,<0.37",
    )
    fit, reason = variant_fit(
        Resources(vram_gb=14), _caps(120), 30.0, binding=_FP4_BINDING,
    )
    assert fit == FIT_INCOMPATIBLE
    assert "diffusers" in reason


def test_variant_fit_svdq_requires_nunchaku_library(stack_ok: None) -> None:
    fit, reason = variant_fit(
        Resources(vram_gb=14, libraries=("nunchaku",)),
        _caps(120, libs=[]), 30.0, binding=_FP4_BINDING,
    )
    assert fit == FIT_INCOMPATIBLE
    assert "nunchaku" in reason


# --------------------------------------------------------------------------
# selection ordering (QUANTIZATION-POLICY fit ladder)
# --------------------------------------------------------------------------

def _rows() -> list[tuple[str, Any, Any]]:
    return [
        ("generate_turbo_bf16", Resources(vram_gb=24), _PLAIN_BINDING),
        ("generate_turbo_fp8", Resources(vram_gb=15), Hub(
            "Tongyi-MAI/Z-Image-Turbo", flavor="fp8")),
        ("generate_turbo_svdq_fp4", Resources(vram_gb=14), _FP4_BINDING),
    ]


def test_select_variant_svdq_fp4_outranks_everything_fitting(stack_ok: None) -> None:
    choice = select_variant(_rows(), _caps(120), 30.0)
    assert choice is not None
    assert choice.name == "generate_turbo_svdq_fp4"
    assert choice.fit == FIT_SVDQ_FP4


def test_select_variant_falls_back_when_pin_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "gen_worker.models.svdq.svdq_stack_reason", lambda: "stack broken",
    )
    choice = select_variant(_rows(), _caps(120), 30.0)
    assert choice is not None
    assert choice.name == "generate_turbo_bf16"
    assert choice.fit == FIT_FITS


def test_select_variant_int4_is_fit_rung_not_speed_rung(stack_ok: None) -> None:
    rows = [
        ("generate_bf16", Resources(vram_gb=24), _PLAIN_BINDING),
        ("generate_fp8", Resources(vram_gb=15), Hub(
            "Tongyi-MAI/Z-Image-Turbo", flavor="fp8")),
        ("generate_svdq_int4", Resources(vram_gb=10), _INT4_BINDING),
    ]
    caps = _caps(89)
    # fp8 fits -> fp8 wins (int4 never outranks a fitting full-precision row).
    choice = select_variant(rows, caps, 16.0)
    assert choice is not None and choice.name == "generate_fp8"
    # nothing but the int4 flavor fits -> int4 rung fires.
    choice = select_variant(rows, caps, 12.0)
    assert choice is not None
    assert choice.name == "generate_svdq_int4"
    assert choice.fit == FIT_SVDQ_INT4


def test_select_variant_non_blackwell_ignores_fp4_row(stack_ok: None) -> None:
    choice = select_variant(_rows(), _caps(89), 30.0)
    assert choice is not None
    assert choice.name == "generate_turbo_bf16"


# --------------------------------------------------------------------------
# load routing
# --------------------------------------------------------------------------

def test_load_from_pretrained_routes_svdq_lane(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gen_worker.models import svdq as svdq_mod
    from gen_worker.models.loading import load_from_pretrained

    root = _svdq_tree(tmp_path)
    calls: list[Any] = []

    def _fake_load(cls: Any, path: Path, art: Any) -> str:
        calls.append((cls, Path(path), art))
        return "svdq-pipe"

    monkeypatch.setattr(svdq_mod, "load_svdq_pipeline", _fake_load)

    class _Pipe:
        @classmethod
        def from_pretrained(cls, *a: Any, **kw: Any) -> None:
            raise AssertionError("svdq snapshot must not take the plain lane")

    out = load_from_pretrained(_Pipe, root, dtype="bf16")
    assert out == "svdq-pipe"
    assert calls and calls[0][2].precision == "fp4"


def test_load_from_pretrained_svdq_without_nunchaku_is_typed(
    tmp_path: Path,
) -> None:
    """No nunchaku installed (CI): the svdq lane fails with SvdqStackError,
    never a mid-denoise crash or a silent fall-through to bf16."""
    from gen_worker.models.loading import load_from_pretrained

    root = _svdq_tree(tmp_path)

    class _Pipe:
        @classmethod
        def from_pretrained(cls, *a: Any, **kw: Any) -> None:
            raise AssertionError("must not fall through")

    with pytest.raises(SvdqStackError):
        load_from_pretrained(_Pipe, root, dtype="bf16")


# --------------------------------------------------------------------------
# convert-side flavor tree
# --------------------------------------------------------------------------

def test_build_svdq_flavor_tree(tmp_path: Path) -> None:
    from gen_worker.convert.svdq import build_svdq_flavor_tree

    base = tmp_path / "base"
    (base / "transformer").mkdir(parents=True)
    (base / "model_index.json").write_text(json.dumps({
        "_class_name": "ZImagePipeline",
        "transformer": ["diffusers", "ZImageTransformer2DModel"],
    }))
    (base / "transformer" / "config.json").write_text("{}")
    _write_plain_safetensors(base / "transformer" / "diffusion_pytorch_model.safetensors")
    _write_plain_safetensors(base / "vae" / "diffusion_pytorch_model.safetensors")
    (base / "scheduler").mkdir()
    (base / "scheduler" / "scheduler_config.json").write_text("{}")

    svdq_file = tmp_path / "svdq-fp4_r128-z-image-turbo.safetensors"
    _write_svdq_safetensors(svdq_file)

    out, attrs = build_svdq_flavor_tree(base, svdq_file, tmp_path / "out")
    assert attrs["flavor"] == "svdq-fp4-r128"
    assert attrs["quantization_method"] == "svdquant"
    assert attrs["quantization_library"] == "nunchaku"
    # the tree: base minus plain denoiser weights, plus the nunchaku file
    assert (out / "model_index.json").exists()
    assert (out / "vae" / "diffusion_pytorch_model.safetensors").exists()
    assert (out / "scheduler" / "scheduler_config.json").exists()
    assert (out / "transformer" / svdq_file.name).exists()
    assert not (out / "transformer" / "diffusion_pytorch_model.safetensors").exists()
    # and it is itself detectable + loadable by the serve-side sniffer
    art = detect_svdq_artifact(out)
    assert art is not None and art.component == "transformer"


def test_build_svdq_flavor_tree_rejects_plain_file(tmp_path: Path) -> None:
    from gen_worker.convert.svdq import build_svdq_flavor_tree

    base = tmp_path / "base"
    (base / "transformer").mkdir(parents=True)
    (base / "model_index.json").write_text("{}")
    plain = tmp_path / "plain.safetensors"
    _write_plain_safetensors(plain)
    with pytest.raises(ValueError, match="not a nunchaku"):
        build_svdq_flavor_tree(base, plain, tmp_path / "out")
