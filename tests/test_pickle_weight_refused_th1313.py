from __future__ import annotations

import pytest

from gen_worker.models.cozy_snapshot import (
    PICKLE_WEIGHT_EXTENSIONS,
    _validate_resolved,
    first_pickle_weight_path,
)
from gen_worker.models.errors import PickleWeightRefused
from gen_worker.models.hub_client import WorkerResolvedRepo, WorkerResolvedRepoFile
from gen_worker.models.refs import TensorhubRef

DIGEST = "b" * 64


def _resolved(*paths: str) -> WorkerResolvedRepo:
    return WorkerResolvedRepo(
        snapshot_digest="sha256:" + DIGEST,
        files=[
            WorkerResolvedRepoFile(
                path=p, size_bytes=16, digest="sha256:" + "a" * 64,
                url="https://example.invalid/" + p,
            )
            for p in paths
        ],
    )


def _ref() -> TensorhubRef:
    return TensorhubRef(owner="acme", repo="victim", release="prod")


@pytest.mark.parametrize("bad", [
    "pytorch_model.bin",
    "v1-5-pruned.ckpt",
    "model.pt",
    "weights.pth",
    "state.pkl",
    "state.pickle",
    "unet/diffusion_pytorch_model.bin",
    "UNet/Diffusion_Pytorch_Model.BIN",
])
def test_resolve_refuses_every_pickle_format(bad: str) -> None:
    with pytest.raises(PickleWeightRefused) as exc:
        _validate_resolved(_ref(), _resolved("model.safetensors", bad))
    assert bad.rsplit("/", 1)[-1].lower() in str(exc.value).lower()


def test_resolve_accepts_a_clean_snapshot() -> None:
    """False-positive guard: names that merely resemble the banned extensions must still resolve, and a normal safetensors tree must be untouched."""
    res = _validate_resolved(_ref(), _resolved(
        "model_index.json",
        "unet/diffusion_pytorch_model.safetensors",
        "unet/diffusion_pytorch_model.safetensors.index.json",
        "tokenizer/spiece.model",
        "flavor.gguf",
        "weights.bins",
        "notpt",
        "archive.ptx",
    ))
    assert len(res.files) == 8
    assert res.snapshot_digest == "sha256:" + DIGEST


def test_extension_set_matches_the_hub_ban() -> None:
    """The worker's list must not drift from tensorhub's catalog.PickleWeightExtensions — a worker that accepts what the hub refuses is the hole this closes."""
    assert PICKLE_WEIGHT_EXTENSIONS == (".bin", ".ckpt", ".pt", ".pth", ".pkl", ".pickle")
    assert first_pickle_weight_path(["x/model.safetensors"]) == ""
