"""Live download test for authenticated URLs (e.g. CivitAI).

Run with:
    CIVIT_AI_TOKEN=<your-token> pytest tests/test_download_with_token.py -s -v
"""
from __future__ import annotations

import os
import tempfile

import pytest

from gen_worker.worker import Worker


CIVITAI_URL = "https://civitai.com/api/download/models/2508727?type=Model&format=SafeTensor"
MIN_EXPECTED_BYTES = 1024 * 1024  # at least 1 MB — rules out an HTML error page


@pytest.fixture()
def worker(tmp_path):
    return Worker.__new__(Worker)


def test_download_civitai_with_token(worker, tmp_path):
    token = os.environ.get("CIVIT_AI_TOKEN")
    if not token:
        pytest.skip("CIVIT_AI_TOKEN not set")

    dst = str(tmp_path / "lora.safetensors")
    size, sha, mime = worker._download_url_to_file(CIVITAI_URL, dst, max_bytes=500 * 1024 * 1024, token=token)

    assert os.path.exists(dst), "file was not created"
    assert size >= MIN_EXPECTED_BYTES, f"file too small ({size} bytes) — likely got an error page"
    assert sha, "sha256 missing"
    os.remove(dst)
    


def test_download_civitai_without_token_raises(worker, tmp_path):
    dst = str(tmp_path / "lora_no_token.safetensors")
    with pytest.raises(RuntimeError, match="failed to download url"):
        worker._download_url_to_file(CIVITAI_URL, dst, max_bytes=500 * 1024 * 1024)
