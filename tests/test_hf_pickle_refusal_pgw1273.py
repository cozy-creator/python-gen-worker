"""pgw#1273 — the HF lane must refuse pickle weights before a byte moves.

These drive the REAL selection functions and the real `download_hf` body. The
only fakes are the two huggingface_hub network calls (`HfApi.list_repo_files`
and `snapshot_download`) — a test that faked the selection would be testing
nothing, since the selection IS the defect.

Red against master: `select_hf_files` picks `pytorch_model.bin` when a
component ships no safetensors (`pool = st or group`), `download_hf` has no
deny-list, and `use_safetensors=True` appears nowhere in the tree — so the file
reaches `from_pretrained` and is unpickled in the process holding hub
credentials.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from gen_worker.models import download as dl
from gen_worker.models.errors import (
    PICKLE_WEIGHT_EXTENSIONS,
    PickleWeightRefused,
    first_pickle_weight_path,
)
from gen_worker.models.refs import HuggingFaceRef


# --- the selection defect, stated as data ---------------------------------

# A diffusers-shaped repo whose UNet ships only a pickle. This is the shape
# `_pick`'s `pool = st or group` fallback selects, and it is common upstream.
PICKLE_ONLY_UNET = [
    "model_index.json",
    "unet/config.json",
    "unet/diffusion_pytorch_model.bin",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
]

ALL_SAFETENSORS = [
    "model_index.json",
    "unet/config.json",
    "unet/diffusion_pytorch_model.safetensors",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
]


def test_selection_still_picks_the_pickle_when_a_component_has_none():
    """Pin the actual behaviour the refusal defends against.

    This asserts the SELECTOR is unchanged — the fix is a refusal, not a
    silent narrowing. Narrowing would fetch a component with no weights at
    all, which fails later and further away.
    """
    selected = dl.select_hf_files(PICKLE_ONLY_UNET)
    assert selected is not None
    assert "unet/diffusion_pytorch_model.bin" in selected, (
        "if this stops holding, the selector changed and the refusal below is "
        "no longer proving what it claims to prove"
    )


# --- the refusal ----------------------------------------------------------

def _fake_hf(monkeypatch, repo_files):
    """Replace ONLY the huggingface_hub module the code resolves at call time.

    `download_hf` does `hub = hf()` then pulls `HfApi`/`snapshot_download` off
    it, so this substitutes at that seam and leaves every line of selection,
    narrowing and refusal running as production.
    """
    class _Api:
        def __init__(self, *a, **k):
            pass

        def list_repo_files(self, **_k):
            return list(repo_files)

        def list_repo_tree(self, **_k):
            return []

    calls: list[dict] = []

    def _snapshot_download(**kwargs):
        calls.append(kwargs)
        return "/nonexistent/should-not-be-reached"

    class _Hub:
        HfApi = _Api
        snapshot_download = staticmethod(_snapshot_download)

    # download_hf does a function-local `from ..net import hf`, so the seam is
    # on gen_worker.net, not on this module.
    import gen_worker.net as net

    monkeypatch.setattr(net, "hf", lambda: _Hub)
    return calls


def test_download_hf_refuses_a_pickle_before_downloading(monkeypatch):
    calls = _fake_hf(monkeypatch, PICKLE_ONLY_UNET)
    ref = HuggingFaceRef(repo_id="org/pickled", revision="main")

    with pytest.raises(PickleWeightRefused) as excinfo:
        dl.download_hf(ref)

    assert "diffusion_pytorch_model.bin" in str(excinfo.value)
    assert calls == [], (
        "snapshot_download was invoked — the refusal must land BEFORE a byte "
        "moves, not after"
    )


def test_download_hf_admits_an_all_safetensors_repo(monkeypatch):
    """The control. A refusal that refuses everything proves nothing."""
    calls = _fake_hf(monkeypatch, ALL_SAFETENSORS)
    ref = HuggingFaceRef(repo_id="org/clean", revision="main")

    dl.download_hf(ref)
    assert len(calls) == 1, "the clean repo must still download"


def test_download_hf_refuses_a_pickle_reached_through_files_patterns(monkeypatch):
    """The `files=` path bypasses select_hf_files entirely, so it needs the
    same chokepoint — an explicit pattern is not consent to unpickle."""
    calls = _fake_hf(monkeypatch, PICKLE_ONLY_UNET)
    ref = HuggingFaceRef(repo_id="org/pickled", revision="main")

    with pytest.raises(PickleWeightRefused):
        dl.download_hf(ref, allow_patterns=["unet/*"])
    assert calls == []


# --- the shared list ------------------------------------------------------

def test_one_home_for_the_extension_set():
    """Five copies of this list existed and had already drifted — convert's
    carried four entries where the others carried six, so a `.pkl`-only
    component yielded zero tensors instead of raising."""
    assert set(PICKLE_WEIGHT_EXTENSIONS) == {
        ".bin", ".ckpt", ".pt", ".pth", ".pkl", ".pickle"
    }

    # cozy_snapshot must consume the shared list, not redeclare one.
    import gen_worker.models.cozy_snapshot as cs

    assert cs.PICKLE_WEIGHT_EXTENSIONS is PICKLE_WEIGHT_EXTENSIONS
    assert cs.first_pickle_weight_path is first_pickle_weight_path


@pytest.mark.parametrize("ext", PICKLE_WEIGHT_EXTENSIONS)
def test_every_listed_extension_is_refused(ext, monkeypatch):
    repo = ["model_index.json", "unet/config.json", f"unet/weights{ext}"]
    calls = _fake_hf(monkeypatch, repo)
    with pytest.raises(PickleWeightRefused):
        dl.download_hf(HuggingFaceRef(repo_id="org/x", revision="main"))
    assert calls == []


def test_basename_matching_does_not_mask_a_later_pickle():
    """A directory whose name ends in an extension must not shadow a real
    weight further along the path."""
    assert first_pickle_weight_path(["notes.pt/readme.md"]) == ""
    assert first_pickle_weight_path(["notes.pt/readme.md", "a/b.bin"]) == "a/b.bin"


def test_refuse_pickles_on_disk_walks_symlinks(tmp_path: Path) -> None:
    """The HF cache materializes snapshots as symlinks into blobs/, so a
    check that only counted regular files would see an empty directory."""
    blob = tmp_path / "blobs" / "deadbeef"
    blob.parent.mkdir(parents=True)
    blob.write_bytes(b"not really a pickle")
    snap = tmp_path / "snapshots" / "rev" / "unet"
    snap.mkdir(parents=True)
    (snap / "diffusion_pytorch_model.bin").symlink_to(blob)

    with pytest.raises(PickleWeightRefused):
        dl._refuse_pickles_on_disk(tmp_path / "snapshots", "org/x (cached)")

    clean = tmp_path / "clean"
    clean.mkdir()
    (clean / "model.safetensors").write_bytes(b"ok")
    dl._refuse_pickles_on_disk(clean, "org/y (cached)")
