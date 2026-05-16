"""Regression tests for the HF local-snapshot fast-path (gen-worker 0.7.19).

The fast-path in `HuggingFaceHubDownloader.download` used to bubble a
`RuntimeError` whenever the local cache was partial — `snapshot_download(
local_files_only=True)` returns the snapshot folder as long as the
`refs/<branch>` file exists, even when LFS blobs haven't been
materialized. The walked file list then fed `plan_diffusers_download` /
`finalize_diffusers_download`, which raise on shards that aren't on disk
yet ("weight shard referenced by ... not found in repo: ..."). That
error short-circuited the API-path retry below and crashed the worker.

The fix wraps the fast-path in try/except + fall-through so a partial
cache silently retries via the API path. These tests pin that
behaviour.

Tests inject a fake `huggingface_hub` module into ``sys.modules`` so
the lazy `from huggingface_hub import ...` inside ``download()`` picks
up our stubs — the real package is not required in the test env.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any, Optional, Sequence

import pytest

from gen_worker.models.hf_downloader import (
    HuggingFaceDownloadResult,
    HuggingFaceHubDownloader,
)
from gen_worker.models.refs import HuggingFaceRef


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _write_partial_cache(root: Path) -> None:
    """Create a fake HF snapshot dir with only model_index.json and a
    text_encoder weight index — the shards it references are NOT on disk,
    matching the real partial-cache failure mode.
    """
    root.mkdir(parents=True, exist_ok=True)
    (root / "model_index.json").write_text(
        json.dumps({
            "_class_name": "StableDiffusionXLPipeline",
            "text_encoder": ["transformers", "CLIPTextModel"],
            "tokenizer": ["transformers", "CLIPTokenizer"],
        })
    )
    text_enc = root / "text_encoder"
    text_enc.mkdir(parents=True, exist_ok=True)
    (text_enc / "model.safetensors.index.json").write_text(
        json.dumps({
            "weight_map": {
                "h.0.weight": "model-00001-of-00002.safetensors",
                "h.1.weight": "model-00002-of-00002.safetensors",
            }
        })
    )
    # Tokenizer dir is required by the planner for tokenizer components.
    tok = root / "tokenizer"
    tok.mkdir(parents=True, exist_ok=True)
    (tok / "tokenizer.json").write_text("{}")


def _write_complete_cache(root: Path) -> None:
    """Create a fully populated fake snapshot dir matching the planner's
    minimal selection (single-file model_index + tokenizer + a single
    unsharded fp16 weight)."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "model_index.json").write_text(
        json.dumps({
            "_class_name": "StableDiffusionXLPipeline",
            "text_encoder": ["transformers", "CLIPTextModel"],
            "tokenizer": ["transformers", "CLIPTokenizer"],
        })
    )
    text_enc = root / "text_encoder"
    text_enc.mkdir(parents=True, exist_ok=True)
    (text_enc / "model.fp16.safetensors").write_text("x" * 16)
    tok = root / "tokenizer"
    tok.mkdir(parents=True, exist_ok=True)
    (tok / "tokenizer.json").write_text("{}")


def _install_fake_hf_hub(
    monkeypatch: pytest.MonkeyPatch,
    *,
    snapshot_download: Any,
    hf_hub_download: Any,
    hf_api_factory: Any,
    hf_hub_url: Optional[Any] = None,
) -> None:
    """Install a stub `huggingface_hub` module into ``sys.modules`` so the
    lazy import inside ``HuggingFaceHubDownloader.download`` picks it up.
    """
    fake = types.ModuleType("huggingface_hub")
    fake.snapshot_download = snapshot_download
    fake.hf_hub_download = hf_hub_download
    fake.HfApi = hf_api_factory
    if hf_hub_url is not None:
        fake.hf_hub_url = hf_hub_url
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake)


# --------------------------------------------------------------------------- #
# Tests                                                                       #
# --------------------------------------------------------------------------- #


def test_partial_cache_falls_through_to_api_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A partial local cache must NOT raise — the fast-path is best-effort.
    The downloader must fall through and complete via the API path."""
    partial = tmp_path / "partial"
    _write_partial_cache(partial)

    completed = tmp_path / "completed"
    completed.mkdir(parents=True, exist_ok=True)

    api_path_calls: list[dict[str, Any]] = []

    def fake_snapshot_download(**kwargs: Any) -> str:
        if kwargs.get("local_files_only"):
            return str(partial)
        api_path_calls.append(kwargs)
        return str(completed)

    def fake_hf_hub_download(**kwargs: Any) -> str:
        name = kwargs.get("filename", "")
        scratch = tmp_path / "api_downloads"
        scratch.mkdir(parents=True, exist_ok=True)
        p = scratch / name.replace("/", "__")
        if name == "model_index.json":
            p.write_text(json.dumps({
                "_class_name": "StableDiffusionXLPipeline",
                "text_encoder": ["transformers", "CLIPTextModel"],
                "tokenizer": ["transformers", "CLIPTokenizer"],
            }))
        elif name.endswith(".index.json"):
            p.write_text(json.dumps({
                "weight_map": {
                    "h.0.weight": "model-00001-of-00002.safetensors",
                    "h.1.weight": "model-00002-of-00002.safetensors",
                }
            }))
        else:
            p.write_text("{}")
        return str(p)

    class FakeHfApi:
        def __init__(self, token: Any = None) -> None:  # noqa: ARG002
            pass

        def list_repo_files(
            self,
            *,
            repo_id: str,  # noqa: ARG002
            repo_type: str,  # noqa: ARG002
            revision: Any = None,  # noqa: ARG002
        ) -> Sequence[str]:
            return [
                "model_index.json",
                "text_encoder/model.safetensors.index.json",
                "text_encoder/model-00001-of-00002.safetensors",
                "text_encoder/model-00002-of-00002.safetensors",
                "tokenizer/tokenizer.json",
            ]

    _install_fake_hf_hub(
        monkeypatch,
        snapshot_download=fake_snapshot_download,
        hf_hub_download=fake_hf_hub_download,
        hf_api_factory=FakeHfApi,
    )

    downloader = HuggingFaceHubDownloader(hf_home=str(tmp_path / "cache"))
    result = downloader.download(HuggingFaceRef(repo_id="acme/test-model"))

    assert isinstance(result, HuggingFaceDownloadResult)
    # Must have routed through the API path, NOT returned the partial dir.
    assert result.local_dir == completed
    assert len(api_path_calls) == 1
    # Bonus: the API path's allow_patterns must include both shards (so
    # the eventual real snapshot_download would actually fetch them).
    allow = set(api_path_calls[0].get("allow_patterns") or [])
    assert "text_encoder/model-00001-of-00002.safetensors" in allow
    assert "text_encoder/model-00002-of-00002.safetensors" in allow


def test_complete_cache_fast_path_skips_api_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fully-populated cache must hit the fast-path with zero API calls."""
    snapshot = tmp_path / "snapshot"
    _write_complete_cache(snapshot)

    def fake_snapshot_download(**kwargs: Any) -> str:
        if kwargs.get("local_files_only"):
            return str(snapshot)
        raise AssertionError(
            "API snapshot_download must not be called when the local cache is complete"
        )

    def fake_hf_hub_download(**kwargs: Any) -> str:  # noqa: ARG001
        raise AssertionError(
            "hf_hub_download must not be called when the local cache is complete"
        )

    class FakeHfApi:
        def __init__(self, token: Any = None) -> None:  # noqa: ARG002
            pass

        def list_repo_files(self, **_: Any) -> Sequence[str]:
            raise AssertionError(
                "list_repo_files must not be called when the local cache is complete"
            )

    _install_fake_hf_hub(
        monkeypatch,
        snapshot_download=fake_snapshot_download,
        hf_hub_download=fake_hf_hub_download,
        hf_api_factory=FakeHfApi,
    )

    downloader = HuggingFaceHubDownloader(hf_home=str(tmp_path / "cache"))
    result = downloader.download(HuggingFaceRef(repo_id="acme/test-model"))

    assert isinstance(result, HuggingFaceDownloadResult)
    assert result.local_dir == snapshot


def test_finalize_raises_on_missing_shard_is_intentional() -> None:
    """Sanity test: the planner's raise-on-missing-shard behaviour is
    intentional — the FIX lives in the caller's error handling, not the
    planner's strictness. If this ever stops raising, revisit the
    fall-through logic in hf_downloader.download.
    """
    from gen_worker.models.hf_selection import (
        HFSelectionPlan,
        finalize_diffusers_download,
    )

    plan = HFSelectionPlan(
        selected_files={"model_index.json"},
        required_weight_index_files={"text_encoder/model.safetensors.index.json"},
    )
    repo_files = [
        "model_index.json",
        "text_encoder/model.safetensors.index.json",
        # NOTE: shards intentionally missing.
    ]
    weight_index_json_by_file = {
        "text_encoder/model.safetensors.index.json": {
            "weight_map": {"h.0.weight": "model-00001-of-00002.safetensors"}
        }
    }

    with pytest.raises(RuntimeError, match="weight shard referenced by"):
        finalize_diffusers_download(
            plan=plan,
            repo_files=repo_files,
            weight_index_json_by_file=weight_index_json_by_file,
        )
