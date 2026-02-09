from __future__ import annotations

import json
from pathlib import Path

import pytest

from gen_worker.cozy_pipeline_spec import (
    COZY_PIPELINE_FILENAME,
    COZY_PIPELINE_LOCK_FILENAME,
    DIFFUSERS_MODEL_INDEX_FILENAME,
    cozy_custom_pipeline_arg,
    ensure_diffusers_model_index_json,
    generate_diffusers_model_index_from_cozy,
    load_cozy_pipeline_spec,
)


def _write(p: Path, s: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(s, encoding="utf-8")


def test_load_cozy_pipeline_spec_prefers_lockfile(tmp_path: Path) -> None:
    _write(
        tmp_path / COZY_PIPELINE_FILENAME,
        """
apiVersion: v1
kind: DiffusersPipeline
pipe:
  library: diffusers
  class: StableDiffusionPipeline
components: {}
""".lstrip(),
    )
    _write(
        tmp_path / COZY_PIPELINE_LOCK_FILENAME,
        """
apiVersion: v1
kind: DiffusersPipeline
pipe:
  library: diffusers
  class: StableDiffusionXLPipeline
components: {}
""".lstrip(),
    )

    spec = load_cozy_pipeline_spec(tmp_path)
    assert spec is not None
    assert spec.source_path.name == COZY_PIPELINE_LOCK_FILENAME
    assert spec.pipe_class == "StableDiffusionXLPipeline"


def test_generate_model_index_from_cozy_pipeline_spec(tmp_path: Path) -> None:
    _write(
        tmp_path / COZY_PIPELINE_LOCK_FILENAME,
        """
apiVersion: v1
kind: DiffusersPipeline
pipe:
  library: diffusers
  class: StableDiffusionPipeline
components:
  unet:
    path: unet
    library: diffusers
    class: UNet2DConditionModel
  scheduler:
    # path omitted; defaults to key name
    library: diffusers
    class: PNDMScheduler
""".lstrip(),
    )
    spec = load_cozy_pipeline_spec(tmp_path)
    assert spec is not None

    out = generate_diffusers_model_index_from_cozy(spec)
    assert out["_class_name"] == "StableDiffusionPipeline"
    assert out["unet"] == ["diffusers", "UNet2DConditionModel"]
    assert out["scheduler"] == ["diffusers", "PNDMScheduler"]


def test_ensure_model_index_json_from_cozy_pipeline(tmp_path: Path) -> None:
    _write(
        tmp_path / COZY_PIPELINE_LOCK_FILENAME,
        """
apiVersion: v1
kind: DiffusersPipeline
pipe:
  library: diffusers
  class: StableDiffusionPipeline
components: {}
""".lstrip(),
    )

    mi = ensure_diffusers_model_index_json(tmp_path)
    assert mi is not None
    assert mi.name == DIFFUSERS_MODEL_INDEX_FILENAME
    parsed = json.loads((tmp_path / DIFFUSERS_MODEL_INDEX_FILENAME).read_text(encoding="utf-8"))
    assert parsed["_class_name"] == "StableDiffusionPipeline"

    # Ensure we don't overwrite an existing model_index.json.
    (tmp_path / DIFFUSERS_MODEL_INDEX_FILENAME).write_text('{"_class_name":"X"}\n', encoding="utf-8")
    mi2 = ensure_diffusers_model_index_json(tmp_path)
    assert mi2 is not None
    assert json.loads(mi2.read_text(encoding="utf-8"))["_class_name"] == "X"


def test_custom_pipeline_arg_validates_pipeline_py(tmp_path: Path) -> None:
    _write(
        tmp_path / COZY_PIPELINE_LOCK_FILENAME,
        """
apiVersion: v1
kind: DiffusersPipeline
pipe:
  library: diffusers
  class: StableDiffusionPipeline
  custom_pipeline_path: custom_pipe
components: {}
""".lstrip(),
    )
    (tmp_path / "custom_pipe").mkdir(parents=True, exist_ok=True)
    (tmp_path / "custom_pipe" / "pipeline.py").write_text("# ok\n", encoding="utf-8")

    spec = load_cozy_pipeline_spec(tmp_path)
    assert spec is not None
    arg = cozy_custom_pipeline_arg(tmp_path, spec)
    assert arg is not None
    assert arg.endswith("/custom_pipe")


def test_custom_pipeline_arg_rejects_escape(tmp_path: Path) -> None:
    _write(
        tmp_path / COZY_PIPELINE_LOCK_FILENAME,
        """
apiVersion: v1
kind: DiffusersPipeline
pipe:
  library: diffusers
  class: StableDiffusionPipeline
  custom_pipeline_path: ../../etc
components: {}
""".lstrip(),
    )
    spec = load_cozy_pipeline_spec(tmp_path)
    assert spec is not None
    with pytest.raises(ValueError):
        _ = cozy_custom_pipeline_arg(tmp_path, spec)

