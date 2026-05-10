"""Tests for gen_worker.conversion.inline_convert.

Covers the dispatch table:
  - Direct passthrough is the caller's responsibility (we don't test it here)
  - Inline-supported: cast (bf16/fp16/fp32), torchao (fp8/int8/int4_wo), GGUF
  - Calibration-required: int4:awq / int4:gptq / nvfp4 → InlineConversionNotPossible

Heavy lifters (streaming dtype cast / torchao / GGUF conversion) are exercised
in their own module-level tests; here we verify the dispatcher routes correctly
and returns structured refusal metadata without depending on deployed endpoints.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gen_worker.conversion.inline_convert import (
    InlineConversionNotPossible,
    InlineConversionResult,
    deferred_conversion_requirement,
    is_calibration_required,
    is_inline_supported,
    run_inline_conversion,
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


class TestDeferredRequirement:
    """Calibrated refusals expose structured metadata, not endpoint commands."""

    def test_int4_awq_describes_calibrated_requirement(self) -> None:
        req = deferred_conversion_requirement("int4:awq")
        assert req is not None
        assert req.kind == "calibrated_quantization"
        assert req.scheme == "int4_awq"
        assert req.requires_calibration is True
        assert req.requires_gpu is True
        assert req.runtime == "modelopt"

    def test_nvfp4_describes_calibrated_requirement(self) -> None:
        req = deferred_conversion_requirement("nvfp4")
        assert req is not None
        assert req.scheme == "nvfp4"
        assert req.as_dict()["requires_calibration"] is True

    def test_non_calibrated_has_no_deferred_requirement(self) -> None:
        assert deferred_conversion_requirement("bf16") is None
        assert deferred_conversion_requirement("fp8:e4m3") is None


class TestDispatchRefuses:
    """Calibrated dtypes raise InlineConversionNotPossible upfront."""

    @pytest.mark.parametrize("dtype", ["int4:awq", "int4:gptq", "nvfp4"])
    def test_refuses_calibrated_with_structured_requirement(self, dtype: str, tmp_path: Path) -> None:
        with pytest.raises(InlineConversionNotPossible) as excinfo:
            run_inline_conversion(
                source_path=tmp_path / "model.safetensors",
                out_dir=tmp_path / "out",
                target_dtype=dtype,
            )
        exc = excinfo.value
        assert exc.target_dtype == dtype
        assert "calibration dataset" in exc.reason
        assert exc.deferred_requirement is not None
        assert exc.deferred_requirement.requires_calibration is True
        # Reason gets folded into the str() so the operator log is informative.
        assert "calibration" in str(exc).lower()

    def test_unknown_dtype_refuses_without_command_hint(self, tmp_path: Path) -> None:
        with pytest.raises(InlineConversionNotPossible) as excinfo:
            run_inline_conversion(
                source_path=tmp_path / "model.safetensors",
                out_dir=tmp_path / "out",
                target_dtype="totally_made_up_dtype",
            )
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

    def test_uses_reason_only(self) -> None:
        exc = InlineConversionNotPossible(
            reason="needs calibration",
            target_dtype="int4:awq",
        )
        assert str(exc) == "needs calibration"

    def test_omits_command_rendering(self) -> None:
        exc = InlineConversionNotPossible(
            reason="weird path",
            target_dtype="x",
        )
        # The string is just the reason; command rendering belongs to callers.
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
