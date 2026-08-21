"""The INGEST edge refuses pickle-format weights, and says which party is wrong."""

from __future__ import annotations

from pathlib import Path

import pytest

from gen_worker.models import download as dl
from gen_worker.models.errors import PickleWeightRefused


def test_a_pickle_only_civitai_version_blames_the_right_party(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The refusal must name OUR policy and the offending file, not read as "civitai is broken"."""
    payload = {"files": [{"id": 1, "name": "model.ckpt", "primary": True,
                          "downloadUrl": "https://example.invalid/x",
                          "sizeKB": 1, "hashes": {"SHA256": "ab" * 32}}]}
    monkeypatch.setattr(dl, "fetch_civitai_model_version", lambda *a, **k: payload)
    with pytest.raises(PickleWeightRefused) as excinfo:
        dl.download_civitai(1234, Path("/nonexistent/should-not-be-reached"))
    message = str(excinfo.value)
    assert "model.ckpt" in message
    assert "pickle" in message.lower()


def test_a_safetensors_civitai_version_is_not_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The control."""
    payload = {"files": [{"id": 1, "name": "model.safetensors", "primary": True,
                          "downloadUrl": "https://example.invalid/x",
                          "sizeKB": 1, "hashes": {"SHA256": "ab" * 32}}]}
    monkeypatch.setattr(dl, "fetch_civitai_model_version", lambda *a, **k: payload)
    with pytest.raises(Exception) as excinfo:
        dl.download_civitai(1234, tmp_path / "out", api_key="")
    assert not isinstance(excinfo.value, PickleWeightRefused)


@pytest.mark.parametrize("ext", [".bin", ".ckpt", ".pt", ".pth", ".pkl", ".pickle"])
def test_every_listed_pickle_extension_is_recognised(ext: str) -> None:
    """The extension set is ONE list (``models/errors.PICKLE_WEIGHT_EXTENSIONS``) precisely because five drifting copies once let ``weights.pkl`` through."""
    from gen_worker.models.errors import first_pickle_weight_path

    assert first_pickle_weight_path([f"unet/model{ext}"]) == f"unet/model{ext}"


def test_a_directory_named_like_a_pickle_does_not_mask_a_real_one() -> None:
    """Basename matching, asserted: a ``foo.bin/`` component must not answer for the shard inside it."""
    from gen_worker.models.errors import first_pickle_weight_path

    assert first_pickle_weight_path(["foo.bin/model.safetensors"]) == ""
    assert first_pickle_weight_path(
        ["foo.bin/model.safetensors", "unet/w.pt"]) == "unet/w.pt"
