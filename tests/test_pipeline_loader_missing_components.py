from __future__ import annotations

from pathlib import Path

from gen_worker.pipeline_loader import missing_component_overrides_for_from_pretrained


class _PipelineWithSafetyChecker:
    def __init__(
        self,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker: bool = True,
        **kwargs,
    ) -> None:
        pass


class _PipelineWithoutThoseParams:
    def __init__(self, foo=None, **kwargs) -> None:
        pass


def test_missing_components_adds_overrides_when_dirs_missing(tmp_path: Path) -> None:
    # No safety_checker/feature_extractor directories exist.
    overrides = missing_component_overrides_for_from_pretrained(_PipelineWithSafetyChecker, tmp_path)
    assert overrides["safety_checker"] is None
    assert overrides["feature_extractor"] is None
    assert overrides["requires_safety_checker"] is False


def test_missing_components_noop_if_pipeline_does_not_support(tmp_path: Path) -> None:
    overrides = missing_component_overrides_for_from_pretrained(_PipelineWithoutThoseParams, tmp_path)
    assert overrides == {}

