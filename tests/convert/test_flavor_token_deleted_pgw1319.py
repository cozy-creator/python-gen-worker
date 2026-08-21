from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fake_hub import _FakeHub

from gen_worker.convert.produced import ProducedFlavor
from gen_worker.convert.publish import PrecisionClassRefusal, publish_flavors

RELEASE = "2026.08"

PRODUCER_SHAPES: dict[str, tuple[dict[str, str], dict[str, Any] | None]] = {
    "modelopt-fp8": (
        {"precision_class": "fp8", "quantization_method": "w8a8",
         "quantization_library": "modelopt", "dtype": "fp8"},
        {"precision_class": "fp8"}),
    "modelopt-nvfp4": (
        {"precision_class": "nvfp4-w4a4", "quantization_method": "nvfp4",
         "quantization_library": "modelopt"},
        {"precision_class": "nvfp4-w4a4"}),
    "svdq-fp4": (
        {"precision_class": "svdq-fp4", "svdq_precision": "fp4",
         "svdq_rank": "128", "quantization_library": "nunchaku"},
        {"precision_class": "svdq-fp4"}),
    "svdq-int4": (
        {"precision_class": "svdq-int4", "svdq_precision": "int4",
         "svdq_rank": "32", "quantization_library": "nunchaku"},
        {"precision_class": "svdq-int4"}),
    "h3-svdq-fp4": (
        {"precision_class": "svdq-fp4", "svdq_precision": "fp4",
         "h3_native": "true"},
        {"precision_class": "svdq-fp4"}),
    "cast-w8a8": (
        {"precision_class": "fp8", "dtype": "fp8",
         "quantization_method": "w8a8"},
        {"precision_class": "fp8"}),
    "cast-fp8": ({"precision_class": "fp8", "dtype": "fp8"},
                 {"precision_class": "fp8"}),
    "coherent-fp8": (
        {"precision_class": "fp8", "dtype": "fp8",
         "coherent_checkpoint": "true"},
        {"precision_class": "fp8"}),
    "encoder-trunc-fp8": (
        {"precision_class": "fp8", "dtype": "fp8", "kept_layers": "20"},
        {"precision_class": "fp8"}),
    "fuse-fp8": (
        {"precision_class": "fp8", "fuse_component": "transformer",
         "fuse_scale": "1.0"},
        {"precision_class": "fp8"}),
    "adaln-full": (
        {"precision_class": "fp8", "component_set": "transformer",
         "adaln_projections": "present"},
        {"precision_class": "fp8"}),
    "adaln-baked": (
        {"precision_class": "fp8", "component_set": "transformer",
         "modulation_tables": "present"},
        {"precision_class": "fp8"}),
    "fuse-bf16": ({"fuse_component": "transformer"}, None),
    "prompt-corpus": ({"corpus_rows": "512"}, None),
    "lora-card": ({"output_kind": "lora", "lora_rank": "32"}, None),
}


class _Ctx:
    def __init__(self, base_url: str) -> None:
        self._file_api_base_url = base_url
        self._worker_capability_token = "cap-token"
        self.owner = "acme"

    def log(self, message: str, **fields: Any) -> None:
        pass


def _opaque_tree(tmp_path: Path, name: str = "out") -> Path:
    out = tmp_path / name
    out.mkdir()
    (out / "diffusion.safetensors").write_bytes(b"\x11" * 2048)
    return out


def _fp8_tree(tmp_path: Path, name: str, *, dtype: Any) -> Path:
    torch = pytest.importorskip("torch")
    from safetensors.torch import save_file

    out = tmp_path / name
    (out / "transformer").mkdir(parents=True)
    save_file({"proj_out.weight": torch.zeros(8, 8).to(dtype)},
              str(out / "transformer" / "diffusion_pytorch_model.safetensors"))
    return out


def _publish(fake_hub: Any, tree: Path, attrs: dict[str, str]) -> dict:
    ctx = _Ctx(f"http://127.0.0.1:{fake_hub.server_port}")
    publish_flavors(
        ctx,
        [ProducedFlavor(path=tree, attributes=dict(attrs))],
        destination_repo="acme/qwen-image",
        release=RELEASE,
    )
    return dict(_FakeHub.state["publish_request"].get("metadata") or {})


@pytest.mark.parametrize("shape", sorted(PRODUCER_SHAPES))
def test_every_producer_shape_publishes_the_class_it_declared(
    fake_hub: Any, tmp_path: Path, shape: str
) -> None:
    """Read off the declare request the fake hub actually receives."""
    attrs, expected = PRODUCER_SHAPES[shape]
    meta = _publish(fake_hub, _opaque_tree(tmp_path), attrs)

    if expected is None:
        assert "placement" not in meta, (
            f"{shape} is a base row and must publish unstamped; a block here "
            "restates the hub's own ClassBase fallback")
        return
    assert meta.get("placement") == expected, (
        f"{shape} declared {attrs.get('precision_class')!r} and published "
        f"{meta.get('placement')!r}")
    assert set(meta["placement"]) == {"precision_class"}


def test_the_token_is_not_published_as_metadata_under_any_spelling(
    fake_hub: Any, tmp_path: Path
) -> None:
    """A18's end state: the word names nothing at the wire either."""
    attrs, _ = PRODUCER_SHAPES["svdq-fp4"]
    meta = _publish(fake_hub, _opaque_tree(tmp_path), attrs)
    assert "flavor" not in meta
    assert "flavor" not in _FakeHub.state["publish_request"]


def test_narrow_bytes_with_no_declaration_are_REFUSED_not_published_unstamped(
    fake_hub: Any, tmp_path: Path
) -> None:
    """The regression the te legs exist to prevent, asserted from pgw's side."""
    torch = pytest.importorskip("torch")
    tree = _fp8_tree(tmp_path, "undeclared", dtype=torch.float8_e4m3fn)

    with pytest.raises(PrecisionClassRefusal) as exc:
        _publish(fake_hub, tree, {"dtype": "fp8"})

    msg = str(exc.value)
    assert "transformer=fp8" in msg, msg
    assert "precision_class" in msg, msg


def test_the_same_narrow_tree_publishes_once_its_class_is_declared(
    fake_hub: Any, tmp_path: Path
) -> None:
    """The refusal is about the SILENCE, not about the bytes: declare the class and the identical tree publishes, carrying it."""
    torch = pytest.importorskip("torch")
    tree = _fp8_tree(tmp_path, "declared", dtype=torch.float8_e4m3fn)

    meta = _publish(fake_hub, tree, {"dtype": "fp8", "precision_class": "fp8"})

    assert meta["placement"] == {"precision_class": "fp8"}
    assert meta["component_dtypes"] == {"transformer": "fp8"}


def test_a_base_tree_needs_no_declaration(fake_hub: Any, tmp_path: Path) -> None:
    """bf16 is 16 bits and every ladder rung is narrower, so the backstop stays silent for the rows that are correctly unstamped — otherwise it would refuse every base publish on the platform."""
    torch = pytest.importorskip("torch")
    tree = _fp8_tree(tmp_path, "base", dtype=torch.bfloat16)

    meta = _publish(fake_hub, tree, {"dtype": "bf16"})

    assert "placement" not in meta
    assert meta["component_dtypes"] == {"transformer": "bf16"}


@pytest.mark.parametrize("bad", ["fp8-w8a8", "svdq-fp4-r128", "int4", "q4_k_m"])
def test_a_class_outside_the_vocabulary_is_REFUSED(
    fake_hub: Any, tmp_path: Path, bad: str
) -> None:
    with pytest.raises(PrecisionClassRefusal) as exc:
        _publish(fake_hub, _opaque_tree(tmp_path), {"precision_class": bad})
    assert "not a class tensorhub reads" in str(exc.value)


def test_a_class_is_matched_case_and_whitespace_insensitively(
    fake_hub: Any, tmp_path: Path
) -> None:
    """The declaration is a producer's literal; `"FP8"` is the same statement as `"fp8"` and refusing it would be a spelling gate, not a class gate."""
    meta = _publish(fake_hub, _opaque_tree(tmp_path), {"precision_class": " FP8"})
    assert meta["placement"] == {"precision_class": "fp8"}


def test_the_publish_leg_names_the_artifact_not_a_flavor(
    fake_hub: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The leg's label was the token."""
    import gen_worker.activity as activity

    seen: list[str] = []
    monkeypatch.setattr(
        activity, "emit_event",
        lambda kind, detail, **kw: seen.append(f"{kind}: {detail}"))

    _publish(fake_hub, _opaque_tree(tmp_path, "svdq-tree"), {})

    legs = [s for s in seen if s.startswith("convert_publish:")]
    assert legs, "the publish emitted no legs at all"
    assert all("artifact=svdq-tree" in leg for leg in legs), legs
    assert not any("flavor=" in leg for leg in legs), legs
