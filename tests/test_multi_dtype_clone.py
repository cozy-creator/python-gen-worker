"""Regression tests for the multi-dtype clone path.

Covers:
  - GGUF default is now F16 (precision-first) — already in test_hf_classifier.
  - Multi-dtype request via `select_for_classification_multi` returns N
    SelectionResult entries with distinct `dtype` attrs.
  - `download_huggingface_repo_files` consumes `dtype_outputs` and emits
    a `selections` list in the return dict.
  - Concrete-dtype validator rejects fuzzy bit-width tokens at the
    tensorhub server boundary (mirrored in `concrete_dtype_enum.go`).
  - IngestResult carries `classifier_attrs_per_checkpoint` when N>1.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest

from gen_worker.conversion.dtype_vocab import (
    CONCRETE_DTYPES,
    is_concrete_dtype,
    is_fuzzy_bitwidth,
    resolve_fuzzy_to_concrete,
)
from gen_worker.conversion.hf_classifier import (
    SelectionResult,
    select_for_classification_multi,
)


# -------------------------------------------------------------------
# Concrete-dtype enum invariants (server side mirrors this set).
# -------------------------------------------------------------------

def test_concrete_dtype_enum_includes_gguf_quants() -> None:
    """The GGUF quant tokens must round-trip through the concrete-dtype
    set so destination checkpoint rows can carry them as-is."""
    for q in ("q4_k_m", "q5_k_m", "q8_0", "f16", "bf16", "f32", "q2_k"):
        assert q in CONCRETE_DTYPES, f"{q!r} missing from CONCRETE_DTYPES"


def test_concrete_dtype_enum_rejects_fuzzy_bitwidth() -> None:
    """`4bit` etc. must NOT be in the concrete set — they're client-side
    only and resolve to concrete values before tensorhub sees them."""
    for f in ("4bit", "8bit", "16bit", "32bit", "2bit"):
        assert f not in CONCRETE_DTYPES
        assert is_fuzzy_bitwidth(f)
        assert not is_concrete_dtype(f)


def test_fuzzy_resolution_per_source_kind_gguf_4bit() -> None:
    """Fuzzy `4bit` against a GGUF source resolves to the
    precision-first available q4 variant."""
    available = ["q4_k_m", "q4_k_s", "q4_0"]
    got = resolve_fuzzy_to_concrete("4bit", source_kind="gguf", available_dtypes=available)
    assert got == "q4_k_m"


def test_fuzzy_resolution_falls_through_when_no_match() -> None:
    """When the source kind doesn't ship any concrete dtype matching
    the fuzzy token, the resolver returns None — caller falls through
    to QUANTIZE_FROM_BF16 or returns DTYPE_UNAVAILABLE."""
    # transformers source ships only bf16; `8bit` (fp8:e4m3) has no match
    got = resolve_fuzzy_to_concrete(
        "8bit", source_kind="transformers", available_dtypes=["bf16"],
    )
    assert got is None


# -------------------------------------------------------------------
# select_for_classification_multi shape
# -------------------------------------------------------------------

def _make_classification(strategy: str = "gguf"):
    from gen_worker.conversion.hf_classifier import RepoClassification

    return RepoClassification(
        strategy=strategy,
        runtime_library="llama-cpp" if strategy == "gguf" else "transformers",
        subtype="",
        refusal=None,
        detection_reason="test",
    )


def _make_inputs_with_gguf_quants(quants: list[str]):
    from gen_worker.conversion.hf_classifier import ClassificationInputs

    files = [f"Llama-3.2-1B-Instruct-{q.upper()}.gguf" for q in quants]
    files.append("README.md")
    files.append("tokenizer.json")
    return ClassificationInputs(file_paths=files)


def test_multi_returns_one_result_per_quant() -> None:
    classification = _make_classification("gguf")
    inputs = _make_inputs_with_gguf_quants(["q4_k_m", "q8_0", "f16"])
    out = select_for_classification_multi(
        classification, inputs,
        gguf_quants=["q4_k_m", "q8_0", "f16"],
    )
    assert len(out) == 3
    # Each entry must have its dtype stamped on attrs.
    dtypes_seen = sorted(s.attrs.get("dtype", "").lower() for s in out)
    assert dtypes_seen == ["f16", "q4_k_m", "q8_0"]
    # Selected paths should contain exactly the matching quant file +
    # always-include extras.
    for s in out:
        assert any(p.endswith(".gguf") for p in s.selected_paths), (
            f"no .gguf in selection for {s.attrs.get('dtype')}"
        )


def test_multi_singleton_path_when_empty() -> None:
    """No requested dtypes → single-element list matching the default
    selector behavior (highest-precision available)."""
    classification = _make_classification("gguf")
    inputs = _make_inputs_with_gguf_quants(["q4_k_m", "f16"])
    out = select_for_classification_multi(classification, inputs, gguf_quants=[])
    assert len(out) == 1
    # Default GGUF order is precision-first: F16 wins over Q4_K_M.
    assert out[0].attrs.get("dtype", "").lower() == "f16"


def test_multi_for_transformers_falls_through_to_singleton() -> None:
    """Phase 1 only does multi-dtype for GGUF. Transformers/diffusers
    return a singleton from `select_for_classification_multi` regardless
    of how many dtypes the caller requests; the caller is expected to
    fall through to convert/quantize for the unrealized dtypes."""
    classification = _make_classification("transformers")
    from gen_worker.conversion.hf_classifier import ClassificationInputs

    inputs = ClassificationInputs(
        file_paths=["config.json", "model.safetensors", "tokenizer.json"],
        config_json={"model_type": "llama", "architectures": ["LlamaForCausalLM"]},
    )
    out = select_for_classification_multi(
        classification, inputs,
        dtype_prefs=["bf16", "fp8:e4m3"],
    )
    assert len(out) == 1


# -------------------------------------------------------------------
# download_huggingface_repo_files multi-dtype shape via dtype_outputs
# -------------------------------------------------------------------

def _stub_huggingface_hub() -> None:
    """Pre-stub the `huggingface_hub` module so `from huggingface_hub
    import hf_hub_download` succeeds without a real network dep."""
    if "huggingface_hub" not in sys.modules:
        placeholder = types.ModuleType("huggingface_hub")
        placeholder.hf_hub_download = lambda **_: ""  # type: ignore[attr-defined]
        sys.modules["huggingface_hub"] = placeholder


def _fake_classification_inputs(repo_id, revision, all_files_list, *, token=None, work_dir=None):  # noqa: ARG001
    """Mock for ingest._fetch_classification_inputs that returns a real
    ClassificationInputs from the listing rows."""
    from gen_worker.conversion.hf_classifier import ClassificationInputs

    paths = [str(item.get("path") or "") for item in all_files_list]
    return ClassificationInputs(file_paths=paths)


def test_download_returns_selections_list_for_multi_gguf(tmp_path: Path) -> None:
    """When dtype_outputs has >1 entries against a GGUF source, the
    return dict must include a `selections` list with one entry per
    resolved concrete dtype."""
    _stub_huggingface_hub()
    from gen_worker.conversion import ingest

    quants = ["q4_k_m", "q8_0", "f16"]
    listing = {
        "source_repo": "bartowski/Llama-3.2-1B-Instruct-GGUF",
        "source_revision": "main",
        "files": [
            {"path": f"Llama-3.2-1B-Instruct-{q.upper()}.gguf", "size_bytes": 1024}
            for q in quants
        ] + [{"path": "README.md", "size_bytes": 1024}],
    }

    class _Class:
        strategy = "gguf"
        runtime_library = "llama-cpp"
        subtype = ""
        refusal = None
        detection_reason = "test"

    class _Tracker:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def __call__(self, **kw):
            target = Path(kw["local_dir"]) / kw["filename"]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"x" * 1024)
            self.calls.append(kw["filename"])
            return str(target)

    tracker = _Tracker()
    with (
        patch("gen_worker.conversion.ingest.list_huggingface_repo_files", lambda *_a, **_kw: listing),
        patch("gen_worker.conversion.ingest._fetch_classification_inputs", _fake_classification_inputs),
        patch("gen_worker.conversion.ingest.classify_huggingface_repo", lambda *_a, **_kw: _Class()),
        patch("huggingface_hub.hf_hub_download", side_effect=tracker),
    ):
        result = ingest.download_huggingface_repo_files(
            source_repo="bartowski/Llama-3.2-1B-Instruct-GGUF",
            output_dir=tmp_path,
            dtype_outputs=quants,
        )

    selections = result.get("selections")
    assert isinstance(selections, list), f"selections must be list, got {type(selections)}"
    assert len(selections) == 3, f"expected 3 selections, got {len(selections)}"

    seen_dtypes = sorted(
        str((s.get("attrs") or {}).get("dtype") or "").lower()
        for s in selections
    )
    assert seen_dtypes == ["f16", "q4_k_m", "q8_0"]

    # File set: each requested quant downloaded once + tokenizer/README
    # always-included. Calls must be union'd not duplicated per-dtype.
    gguf_calls = sorted(c for c in tracker.calls if c.endswith(".gguf"))
    assert sorted(set(gguf_calls)) == gguf_calls, "duplicate downloads"
    assert len(gguf_calls) == 3


def test_download_singleton_path_when_dtype_outputs_omitted(tmp_path: Path) -> None:
    """When dtype_outputs is omitted/empty, the return dict's `selections`
    field is an empty list — single-checkpoint backwards-compatible."""
    _stub_huggingface_hub()
    from gen_worker.conversion import ingest

    listing = {
        "source_repo": "test/repo",
        "source_revision": "main",
        "files": [
            {"path": "Llama-3.2-1B-Instruct-Q4_K_M.gguf", "size_bytes": 1024},
            {"path": "Llama-3.2-1B-Instruct-F16.gguf", "size_bytes": 1024},
        ],
    }

    class _Class:
        strategy = "gguf"
        runtime_library = "llama-cpp"
        subtype = ""
        refusal = None
        detection_reason = "test"

    def _hf_dl(**kw):
        target = Path(kw["local_dir"]) / kw["filename"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"x" * 1024)
        return str(target)

    with (
        patch("gen_worker.conversion.ingest.list_huggingface_repo_files", lambda *_a, **_kw: listing),
        patch("gen_worker.conversion.ingest._fetch_classification_inputs", _fake_classification_inputs),
        patch("gen_worker.conversion.ingest.classify_huggingface_repo", lambda *_a, **_kw: _Class()),
        patch("huggingface_hub.hf_hub_download", side_effect=_hf_dl),
    ):
        result = ingest.download_huggingface_repo_files(
            source_repo="test/repo",
            output_dir=tmp_path,
        )

    selections = result.get("selections")
    assert selections == [], f"selections must be empty list when single, got {selections!r}"
    # Default precision-first: F16 wins.
    assert str(result.get("attrs", {}).get("dtype", "")).lower() == "f16"


# -------------------------------------------------------------------
# IngestResult.classifier_attrs_per_checkpoint shape
# -------------------------------------------------------------------

def test_ingest_result_carries_per_checkpoint_attrs() -> None:
    from gen_worker.conversion.core_types import IngestResult

    ir = IngestResult(
        classifier_attrs={"runtime_library": "llama-cpp"},
        classifier_attrs_per_checkpoint=[
            {"dtype": "q4_k_m", "runtime_library": "llama-cpp"},
            {"dtype": "f16", "runtime_library": "llama-cpp"},
        ],
    )
    assert len(ir.classifier_attrs_per_checkpoint) == 2
    dtypes = sorted(a["dtype"] for a in ir.classifier_attrs_per_checkpoint)
    assert dtypes == ["f16", "q4_k_m"]


def test_ingest_result_per_checkpoint_defaults_empty() -> None:
    """When the source is single-dtype, classifier_attrs_per_checkpoint
    is [] — the legacy single-flavor path uses `classifier_attrs`."""
    from gen_worker.conversion.core_types import IngestResult

    ir = IngestResult(classifier_attrs={"runtime_library": "transformers"})
    assert ir.classifier_attrs_per_checkpoint == []
