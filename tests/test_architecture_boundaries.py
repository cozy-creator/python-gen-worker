"""Boundary checks for gen-worker as a reusable worker library."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "gen_worker"


FORBIDDEN_SOURCE_TOKENS = (
    "e2e",
    "progress.json",
    "conversion-endpoints",
    "training-endpoints",
    "inference-endpoints",
    "conversion-cpu",
    "conversion-gpu",
    "modelopt_quantization",
    "torchao_quantization",
    "bitsandbytes_quantization",
    "suggested_command",
    "suggested_separate_job",
    "endpoint-name",
    "root/calibration",
    "calibration-default",
    "/home/fidika",
    "~/cozy",
)

ALLOWED_MODULE_ENTRYPOINTS = {
    Path("src/gen_worker/discovery/__main__.py"),
    Path("src/gen_worker/entrypoint.py"),
}


def _source_files() -> list[Path]:
    ignored_parts = {"__pycache__"}
    return [
        path
        for path in SRC.rglob("*.py")
        if not ignored_parts.intersection(path.parts)
    ]


def test_worker_library_has_no_deployment_or_endpoint_coupling() -> None:
    """gen_worker may assume Tensorhub protocols, not local endpoint deployments."""
    failures: list[str] = []
    for path in _source_files():
        text = path.read_text(encoding="utf-8")
        lowered = text.lower()
        for token in FORBIDDEN_SOURCE_TOKENS:
            if token.lower() in lowered:
                rel = path.relative_to(ROOT)
                failures.append(f"{rel}: contains {token!r}")

    assert failures == []


def test_package_does_not_publish_console_scripts() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8").lower()
    assert "[project.scripts]" not in pyproject
    assert "console_scripts" not in pyproject


def test_package_only_has_runtime_and_discovery_module_entrypoints() -> None:
    failures: list[str] = []
    for path in _source_files():
        text = path.read_text(encoding="utf-8")
        rel = path.relative_to(ROOT)
        has_module_entrypoint = (
            'if __name__ == "__main__"' in text
            or "if __name__ == '__main__'" in text
        )
        if has_module_entrypoint and rel not in ALLOWED_MODULE_ENTRYPOINTS:
            failures.append(f"{rel}: unexpected module entrypoint")
        if "argparse" in text:
            failures.append(f"{rel}: command-line parser dependency")

    assert failures == []


def test_repo_does_not_carry_standalone_python_command_scripts() -> None:
    scripts_dir = ROOT / "scripts"
    if not scripts_dir.exists():
        return

    assert sorted(path.name for path in scripts_dir.glob("*.py")) == []


def test_system_boundaries_are_documented() -> None:
    doc = (ROOT / "docs" / "system-boundaries.md").read_text(encoding="utf-8")
    assert "generic primitives and metadata" in doc
    assert "product conversion functions and calibrated" in doc
    assert "does not implement modelopt execution" in doc
    assert "does not publish console scripts" in doc
    assert "Worker-Facing Import Surfaces" in doc


def test_internal_safetensors_plumbing_is_not_top_level_worker_api() -> None:
    init_py = (SRC / "conversion" / "__init__.py").read_text(encoding="utf-8")
    assert "materialize_safetensors_input" not in init_py
    assert "persist_safetensors_output" not in init_py


def test_top_level_worker_api_stays_small() -> None:
    import gen_worker

    expected = {
        "inference_function",
        "realtime_function",
        "ResourceRequirements",
        "ModelRef",
        "ModelRefSource",
        "RequestContext",
        "RealtimeSocket",
        "AuthError",
        "CanceledError",
        "RetryableError",
        "ResourceError",
        "ValidationError",
        "FatalError",
        "OutputTooLargeError",
        "RefCompatibilitySurprise",
        "WorkerError",
        "Asset",
        "Compute",
        "Tensors",
        "LoraSpec",
        "Clamp",
        "iter_transformers_text_deltas",
        "load_loras",
        "apply_low_vram_config",
        "with_oom_retry",
        "clone",
    }
    assert set(gen_worker.__all__) == expected

    forbidden = {
        "ModelCache",
        "ModelDownloader",
        "CozyHubDownloader",
        "PipelineLoader",
        "EndpointValidationResult",
        "validate_endpoint",
        "training_function",
        "StepContext",
        "StepResult",
        "SourceRepo",
        "DestinationRepo",
        "DatasetRef",
        "OutputSpec",
        "materialize_safetensors_input",
        "persist_safetensors_output",
    }
    assert forbidden.isdisjoint(set(gen_worker.__all__))


def test_conversion_worker_api_stays_authoring_focused() -> None:
    import gen_worker.conversion as conversion

    expected = {
        "Component",
        "ConversionContext",
        "TrainingFunctionSpec",
        "Dataset",
        "FileLayout",
        "ProducedFlavor",
        "Source",
        "StreamingWriter",
        "training_function",
        "CalibrationAction",
        "CalibrationPolicy",
        "resolve_calibration_action",
    }
    assert set(conversion.__all__) == expected

    forbidden = {
        "ConversionArtifact",
        "ConversionOutput",
        "IngestResult",
        "tensors_with",
        "ValidationReport",
        "ValidationViolation",
        "validate_transform_module",
        "materialize_safetensors_input",
        "persist_safetensors_output",
    }
    assert forbidden.isdisjoint(set(conversion.__all__))


def test_api_package_does_not_reexport_parser_internals() -> None:
    import gen_worker.api as api

    assert "parse_injection" not in api.__all__
    assert "InjectionSpec" not in api.__all__
