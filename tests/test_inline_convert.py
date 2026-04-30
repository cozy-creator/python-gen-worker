"""Tests for gen_worker.conversion.inline_convert (e2e progress.json #73).

Covers the dispatch table:
  - Direct passthrough is the caller's responsibility (we don't test it here)
  - Inline-supported: cast (bf16/fp16/fp32), torchao (fp8/int8/int4_wo), GGUF
  - Calibration-required: int4:awq / int4:gptq / nvfp4 → InlineConversionNotPossible

Heavy lifters (streaming_dtype_cast / torchao / convert_hf_to_gguf) are
exercised in their own module-level tests; here we just verify the
dispatcher routes correctly and the suggested_command shape stays stable.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gen_worker.conversion.inline_convert import (
    InlineConversionNotPossible,
    InlineConversionResult,
    is_calibration_required,
    is_inline_supported,
    run_inline_conversion,
    suggested_separate_job,
)


class TestSupportTables:
    """Boolean table tests for is_inline_supported / is_calibration_required."""

    @pytest.mark.parametrize("dtype", ["bf16", "fp16", "fp32", "f16", "f32"])
    def test_cast_dtypes_supported_safetensors(self, dtype: str) -> None:
        assert is_inline_supported(dtype, target_file_type="safetensors") is True
        assert is_calibration_required(dtype) is False

    @pytest.mark.parametrize("dtype", ["fp8:e4m3", "fp8:e5m2", "fp8", "int8", "int4"])
    def test_torchao_weight_only_supported(self, dtype: str) -> None:
        assert is_inline_supported(dtype, target_file_type="safetensors") is True
        assert is_calibration_required(dtype) is False

    @pytest.mark.parametrize("dtype", ["int4:awq", "int4:gptq", "nvfp4", "int8:awq", "int8:gptq"])
    def test_calibrated_quants_refused(self, dtype: str) -> None:
        assert is_inline_supported(dtype, target_file_type="safetensors") is False
        assert is_calibration_required(dtype) is True

    @pytest.mark.parametrize("dtype", ["q4_k_m", "q5_k_m", "q6_k", "q8_0", "f16", "bf16"])
    def test_gguf_encodings_supported_with_gguf_target(self, dtype: str) -> None:
        assert is_inline_supported(dtype, target_file_type="gguf") is True

    def test_gguf_encoding_with_safetensors_target_unsupported(self) -> None:
        # q4_k_m is a GGUF encoding; routing it to a safetensors target
        # is nonsense and should not be supported.
        assert is_inline_supported("q4_k_m", target_file_type="safetensors") is False

    def test_unknown_dtype_unsupported(self) -> None:
        assert is_inline_supported("nonsense_dtype") is False
        assert is_calibration_required("nonsense_dtype") is False

    def test_empty_dtype_unsupported(self) -> None:
        assert is_inline_supported("") is False
        assert is_calibration_required("") is False

    def test_normalization_handles_whitespace_and_case(self) -> None:
        assert is_inline_supported("  BF16  ") is True
        assert is_inline_supported("FP8:E4M3") is True
        assert is_calibration_required("INT4:AWQ") is True


class TestSuggestedCommand:
    """The CLI renders these as one-paragraph hints."""

    def test_int4_awq_routes_to_modelopt(self) -> None:
        cmd = suggested_separate_job(
            "int4:awq", source_ref="user/repo-src", destination_ref="user/repo-dst",
        )
        assert "modelopt_quantization" in cmd
        assert "scheme=int4_awq" in cmd
        assert "--source-ref user/repo-src" in cmd
        assert "--destination-ref user/repo-dst" in cmd
        assert "conversion-gpu" in cmd  # calibrated quants need GPU

    def test_nvfp4_routes_to_modelopt(self) -> None:
        cmd = suggested_separate_job(
            "nvfp4", source_ref="x/y", destination_ref="x/y",
        )
        assert "modelopt_quantization" in cmd
        assert "scheme=nvfp4" in cmd

    def test_non_calibrated_returns_empty(self) -> None:
        # Non-calibrated dtypes don't need a separate job — they can run inline.
        assert suggested_separate_job("bf16", source_ref="a", destination_ref="b") == ""
        assert suggested_separate_job("fp8:e4m3", source_ref="a", destination_ref="b") == ""

    def test_missing_refs_use_placeholders(self) -> None:
        cmd = suggested_separate_job("int4:awq", source_ref="", destination_ref="")
        assert "<source>" in cmd
        assert "<destination>" in cmd


class TestDispatchRefuses:
    """Calibrated dtypes raise InlineConversionNotPossible upfront."""

    @pytest.mark.parametrize("dtype", ["int4:awq", "int4:gptq", "nvfp4"])
    def test_refuses_calibrated_with_clear_hint(self, dtype: str, tmp_path: Path) -> None:
        with pytest.raises(InlineConversionNotPossible) as excinfo:
            run_inline_conversion(
                source_path=tmp_path / "model.safetensors",
                out_dir=tmp_path / "out",
                target_dtype=dtype,
                source_ref="src/repo",
                destination_ref="dst/repo",
            )
        exc = excinfo.value
        assert exc.target_dtype == dtype
        assert "calibration dataset" in exc.reason
        assert exc.suggested_command != ""
        # Reason gets folded into the str() so the operator log is informative.
        assert "calibration" in str(exc).lower()

    def test_unknown_dtype_refuses_without_command_hint(self, tmp_path: Path) -> None:
        with pytest.raises(InlineConversionNotPossible) as excinfo:
            run_inline_conversion(
                source_path=tmp_path / "model.safetensors",
                out_dir=tmp_path / "out",
                target_dtype="totally_made_up_dtype",
            )
        # Unknown dtypes don't have a known separate-job command path, so the
        # suggested_command is empty; the reason still describes the problem.
        assert excinfo.value.suggested_command == ""
        assert "not recognized" in excinfo.value.reason

    def test_missing_target_dtype_raises_value_error(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            run_inline_conversion(
                source_path=tmp_path / "x.safetensors",
                out_dir=tmp_path / "out",
                target_dtype="",
            )


class TestExceptionStr:
    """The runtime str() of InlineConversionNotPossible is what lands in error logs."""

    def test_includes_suggested_command_when_present(self) -> None:
        exc = InlineConversionNotPossible(
            reason="needs calibration",
            suggested_command="e2e convert ...",
            target_dtype="int4:awq",
        )
        s = str(exc)
        assert "needs calibration" in s
        assert "e2e convert ..." in s

    def test_omits_arrow_when_no_suggested_command(self) -> None:
        exc = InlineConversionNotPossible(
            reason="weird path",
            target_dtype="x",
        )
        # When suggested_command is empty, the str() is just the reason —
        # no trailing "→ run: " noise.
        assert str(exc) == "weird path"


class TestResultDataclass:
    """InlineConversionResult is a passive carrier — verify defaults / shape."""

    def test_defaults(self) -> None:
        r = InlineConversionResult()
        assert r.output_paths == []
        assert r.index_path is None
        assert r.target_dtype == ""
        assert r.target_file_type == "safetensors"
        assert r.attributes == {}
        assert r.summary == {}
