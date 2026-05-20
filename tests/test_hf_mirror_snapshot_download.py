from __future__ import annotations

import sys
import types
from pathlib import Path

from gen_worker.conversion import ingest
from gen_worker.conversion.hf_classifier import RepoClassification, SelectionResult


def test_hf_mirror_uses_snapshot_download_scheduler(monkeypatch, tmp_path: Path) -> None:
    calls: list[dict[str, object]] = []
    mirror_root = tmp_path / "mirror"
    output_dir = tmp_path / "out"

    monkeypatch.setattr(ingest, "_HF_MIRROR_ROOT", mirror_root)
    monkeypatch.setattr(
        ingest,
        "list_huggingface_repo_files",
        lambda *args, **kwargs: {
            "source_repo": "org/model",
            "source_revision": "abc123",
            "files": [
                {"path": "config.json", "size_bytes": 2},
                {"path": "weights/model.safetensors", "size_bytes": 5},
                {"path": "ignored.txt", "size_bytes": 7},
            ],
        },
    )
    monkeypatch.setattr(ingest, "_fetch_classification_inputs", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        ingest,
        "classify_huggingface_repo",
        lambda *_args, **_kwargs: RepoClassification(
            strategy="transformers",
            runtime_library="transformers",
            detection_reason="test",
        ),
    )
    monkeypatch.setattr(
        ingest,
        "select_for_classification",
        lambda *_args, **_kwargs: SelectionResult(
            selected_paths=["config.json", "weights/model.safetensors"],
            skipped_paths=["ignored.txt"],
            attrs={"format": "safetensors"},
        ),
    )

    def fake_snapshot_download(**kwargs: object) -> str:
        calls.append(dict(kwargs))
        local_dir = Path(str(kwargs["local_dir"]))
        (local_dir / "weights").mkdir(parents=True, exist_ok=True)
        (local_dir / "config.json").write_text("{}", encoding="utf-8")
        (local_dir / "weights" / "model.safetensors").write_bytes(b"12345")
        return str(local_dir)

    fake_hf = types.SimpleNamespace(snapshot_download=fake_snapshot_download)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf)

    result = ingest.download_huggingface_repo_files("org/model", output_dir)

    assert len(calls) == 1
    assert calls[0]["repo_id"] == "org/model"
    assert calls[0]["revision"] == "abc123"
    assert calls[0]["allow_patterns"] == ["config.json", "weights/model.safetensors"]
    assert "max_workers" not in calls[0]
    assert (output_dir / "config.json").read_text("utf-8") == "{}"
    assert (output_dir / "weights" / "model.safetensors").read_bytes() == b"12345"
    assert result["file_count"] == 2
    assert result["total_bytes"] == 7
