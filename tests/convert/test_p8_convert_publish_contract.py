"""P8 (th#960/pgw#609 design table): convert/publish contract, hermetic
(fake tensorhub HTTP server, no torch/GPU, no real weight downloads).

  * pgw#589/th#901: an explicit dtype mismatch on a publish_as_is
    (dense-safetensors) source casts for real — never silent passthrough
    under a correct-looking label; a matching dtype is genuinely zero-work;
    non-cast-eligible strategies (gguf) still refuse loud.
  * pgw#593: classifier variant-tag selection against a real-world-shaped
    filename corpus (a table, not a one-off regression case).
  * pgw#566 (test-first — fix open): kohya-SGM SDXL adapter normalization
    must pass ``unet_config`` through **kwargs-only converters (the real
    diffusers ``StableDiffusionXLPipeline.lora_state_dict`` shape) so the
    SGM block remap actually runs — currently it only checks for a NAMED
    parameter and silently skips it.
  * pgw#569: not covered here — the W8A8 source verifier's one-ULP gate
    (writer.py's ``verify_w8a8_byte_gate``) operates on real safetensors
    artifact files with no factored-out standalone comparator; building a
    hermetic fixture for it needs a real (if tiny) w8a8 production
    round-trip, out of scope for this pass. Flagged as an open follow-up in
    the th#960 tracker checkpoint rather than faked or skipped silently.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import pytest

from gen_worker.convert.classifier import RepoRefusal, classify_repo
from gen_worker.convert.clone import run_clone
from gen_worker.convert.ingest import IngestedSource
from gen_worker.models.w8a8_lora import normalize_adapter_state_dict

from fake_hub import _FakeHub


class _Ctx:
    def __init__(self, server: Any) -> None:
        self._file_api_base_url = f"http://127.0.0.1:{server.server_port}"
        self._worker_capability_token = "cap-token"
        self.owner = "acme"
        self.request_id = "req-1"
        self.destination = {"repo": "acme/fallback"}


def _transformers_source(dest_dir: Path, *, dtype: str = "fp32") -> IngestedSource:
    dest_dir.mkdir(parents=True, exist_ok=True)
    (dest_dir / "config.json").write_text('{"architectures": ["FakeBackbone"]}')
    (dest_dir / "model.safetensors").write_bytes(b"\x00" * 64)
    return IngestedSource(
        provider="huggingface", source_ref="org/hidream-like", source_revision="sha-1",
        dir=dest_dir, layout="singlefile", model_family="hidream", model_family_variant="",
        classification=SimpleNamespace(strategy="transformers"),
        attrs={"dtype": dtype, "file_layout": "singlefile"},
        metadata={"source_provider": "huggingface"},
        repo_spec={"kind": "model", "library_name": "transformers"},
    )


def _install_fake_ingest(monkeypatch: pytest.MonkeyPatch, source: IngestedSource) -> None:
    def fake_ingest(source_ref, dest_dir, **kwargs):
        return source

    monkeypatch.setattr("gen_worker.convert.clone.ingest_huggingface", fake_ingest)


def _stub_build_flavor_tree(monkeypatch: pytest.MonkeyPatch, calls: list) -> None:
    def fake_build_flavor_tree(source, spec, out_dir, *, quantize_components=None):
        calls.append({"dtype": spec.dtype, "file_layout": spec.file_layout})
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "model.safetensors").write_bytes(b"\x01" * 32)
        attrs = {"dtype": spec.dtype, "file_layout": spec.file_layout, "file_type": spec.file_type}
        return out_dir, attrs

    monkeypatch.setattr("gen_worker.convert.clone.build_flavor_tree", fake_build_flavor_tree)


# ---------------------------------------------------------------------------
# pgw#589/th#901: dtype passthrough honesty.
# ---------------------------------------------------------------------------


def test_explicit_dtype_mismatch_casts_not_silent_passthrough(
    fake_hub, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _FakeHub.state["finalize_calls"] = 1
    monkeypatch.setenv("COZY_CONVERT_WORKDIR", str(tmp_path / "work"))
    source = _transformers_source(tmp_path / "source", dtype="fp32")
    _install_fake_ingest(monkeypatch, source)
    calls: list = []
    _stub_build_flavor_tree(monkeypatch, calls)

    result = run_clone(
        _Ctx(fake_hub), provider="huggingface", source_ref="org/hidream-like",
        destination_repo="acme/dest",
        outputs=[{"dtype": "bf16", "file_layout": "diffusers", "file_type": "safetensors"}],
    )

    assert not result.failed_flavors, result.failed_flavors
    assert result.published[0]["flavor"] == "bf16"
    assert len(calls) == 1 and calls[0]["dtype"] == "bf16"  # a real cast ran


def test_matching_dtype_is_genuinely_zero_work(
    fake_hub, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _FakeHub.state["finalize_calls"] = 1
    monkeypatch.setenv("COZY_CONVERT_WORKDIR", str(tmp_path / "work"))
    source = _transformers_source(tmp_path / "source", dtype="fp32")
    _install_fake_ingest(monkeypatch, source)
    calls: list = []
    _stub_build_flavor_tree(monkeypatch, calls)

    result = run_clone(
        _Ctx(fake_hub), provider="huggingface", source_ref="org/hidream-like",
        destination_repo="acme/dest",
        outputs=[{"dtype": "fp32", "file_layout": "diffusers", "file_type": "safetensors"}],
    )
    assert not result.failed_flavors
    assert result.published[0]["flavor"] == "fp32"
    assert calls == []


def test_non_cast_eligible_strategy_refuses_mismatch_loudly(
    fake_hub, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """gguf is a binary quant container, not a dense-safetensors tree — the
    th#901 fix must not weaken its existing loud refusal into a silent
    passthrough."""
    _FakeHub.state["finalize_calls"] = 1
    monkeypatch.setenv("COZY_CONVERT_WORKDIR", str(tmp_path / "work"))
    dest = tmp_path / "source"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "model.q4_k_m.gguf").write_bytes(b"\x00" * 32)
    source = IngestedSource(
        provider="huggingface", source_ref="org/gguf-src", source_revision="sha-1",
        dir=dest, layout="singlefile", model_family="", model_family_variant="",
        classification=SimpleNamespace(strategy="gguf"),
        attrs={"dtype": "q4_k_m", "file_layout": "singlefile"}, metadata={}, repo_spec={},
    )
    _install_fake_ingest(monkeypatch, source)
    calls: list = []
    _stub_build_flavor_tree(monkeypatch, calls)

    with pytest.raises(RuntimeError, match="no publishable flavor"):
        run_clone(
            _Ctx(fake_hub), provider="huggingface", source_ref="org/gguf-src",
            destination_repo="acme/dest",
            outputs=[{"dtype": "bf16", "file_layout": "singlefile", "file_type": "gguf"}],
        )
    assert calls == [], "a refused dtype mismatch must never reach the cast path"


# ---------------------------------------------------------------------------
# pgw#593: classifier corpus (real-world-shaped filenames).
# ---------------------------------------------------------------------------


_CLASSIFIER_CORPUS = [
    (
        "ltx-2.3-video-listing",
        ["ltx-2.3-13b-fp8.safetensors", "ltx-2.3-13b-bf16.safetensors", "README.md"],
        "aio_singlefile",
    ),
    (
        "sdxl-diffusers-component",
        [
            "unet/diffusion_pytorch_model.fp16.safetensors",
            "unet/config.json", "vae/diffusion_pytorch_model.fp16.safetensors",
            "vae/config.json", "text_encoder/model.fp16.safetensors",
            "text_encoder/config.json", "model_index.json",
        ],
        "diffusers",
    ),
    (
        "kohya-lora-safetensors",
        ["my_lora.safetensors", "README.md"],
        "native_lora",
    ),
    (
        "gguf-multi-quant",
        ["model.Q4_K_M.gguf", "model.Q8_0.gguf", "model.Q2_K.gguf", "README.md"],
        "gguf",
    ),
]


@pytest.mark.parametrize("label,files,expected_strategy", _CLASSIFIER_CORPUS, ids=[c[0] for c in _CLASSIFIER_CORPUS])
def test_classifier_corpus_strategy_selection(label: str, files: list, expected_strategy: str) -> None:
    sizes: Dict[str, int] = {}
    if expected_strategy == "native_lora":
        sizes = {"my_lora.safetensors": 40 * 1024 * 1024}
        c = classify_repo(files, sizes=sizes, safetensors_metadata={"ss_network_module": "networks.lora"})
    else:
        c = classify_repo(files)
    assert c.strategy == expected_strategy, (label, c.strategy)


def test_classifier_refuses_oversized_unclassifiable_repo() -> None:
    with pytest.raises(RepoRefusal):
        classify_repo(["random_blob.bin"] * 3, sizes={"random_blob.bin": 200 * 1024**3})


# ---------------------------------------------------------------------------
# pgw#566 (test-first, fix open): kohya-SGM SDXL adapter normalization.
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason="pgw#566 open: normalize_adapter_state_dict only passes unet_config "
           "to converters with a NAMED unet_config parameter (inspect.signature "
           "check); diffusers' real StableDiffusionXLPipeline.lora_state_dict "
           "takes **kwargs only, so unet_config is silently never passed and "
           "the SGM block remap never runs — the live nerijs/pixel-art-xl "
           "r32 repro (2166 unresolved keys). Fix direction: pass unet_config "
           "through **kwargs-accepting converters too.",
)
def test_normalize_passes_unet_config_through_kwargs_only_converters() -> None:
    captured: Dict[str, Any] = {}

    class _KwargsOnlySDXLPipe:
        """Pipeline-shaped stub reproducing diffusers' REAL SDXL signature
        shape (no NAMED unet_config parameter) — not a mock of gen_worker
        code, a stand-in third-party pipeline like this repo's other
        _StubPipeline/_RamPressurePipeline fixtures."""

        def __init__(self) -> None:
            self.unet = SimpleNamespace(config={"down_block_types": ()})

        @staticmethod
        def lora_state_dict(sd: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
            captured.update(kwargs)
            return dict(sd)

    normalize_adapter_state_dict(_KwargsOnlySDXLPipe(), {"a": 1}, ref="test")
    assert "unet_config" in captured, (
        "normalize_adapter_state_dict must pass unet_config into "
        "**kwargs-only converters, not just ones with a NAMED parameter"
    )
