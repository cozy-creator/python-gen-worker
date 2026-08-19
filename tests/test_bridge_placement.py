"""The eager bridge places weights where the WORKER said, or nowhere.

pgw#1452, measured on a real CUDA box: `ctx.load`'s eager arm built a pipeline
with `from_pretrained` and returned it unplaced, so every checkpoint tree with
no chunk store behind it — a bare download, a local run, a fixture, which is
the whole cozy-local substrate — ran on the CPU. Nothing failed. `sd1.5` simply
took 13 s/step on a card that does it in well under one, and every timing taken
through the bridge was a CPU timing.

The contract was never ambiguous; it sits four lines above the defect in
`ctx.load`'s own docstring — *"No `.to("cuda")` (weights land on device) …
PLACEMENT stays a worker decision and this code names no device"* — and
`host.py` says the same thing where it picks the device. The streaming arm
honoured it (`engine_for(..., device=device)`); the eager arm never received it.

Both arms are asserted here, because the interesting half is the SECOND:

1. handed a device, the bridge places every component on it;
2. handed NOTHING, the bridge places nothing — a bare `LoadContext(...)` and
   the derive path must not acquire a device because this fix exists.

The CUDA arm skips without a device; arm 2 runs everywhere, so the "names no
device of its own" half of the contract is measured on every runner.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")

from diffusers import StableDiffusionPipeline  # noqa: E402

from gen_worker.serving.context import DeployBinding, LoadContext  # noqa: E402

from harness.nvml import nvml_is_healthy  # noqa: E402

#: The smallest real diffusers pipeline that exists. Not a hand-built double:
#: the defect was in what `from_pretrained` returns, so a fake that skips
#: `from_pretrained` could not have caught it.
FIXTURE = "hf-internal-testing/tiny-stable-diffusion-pipe"


def _local_snapshot() -> Path:
    """The fixture's cached tree, or skip. Never a download: this suite must
    not reach the network to answer a question about device placement.

    The cache is walked directly rather than through
    `snapshot_download(local_files_only=True)`, which refuses a tree that
    diffusers loads perfectly well: it demands EVERY file the repo lists,
    including `README.md`, `.gitattributes` and the flax weights, and raises
    "incomplete" without them. Gating on it skips this suite on machines that
    can run it — a false skip, which reads as coverage and is exactly what the
    pgw#966 census exists to stop. The real precondition is the one asserted
    here: a snapshot dir with a `model_index.json` in it.
    """
    from huggingface_hub.constants import HF_HUB_CACHE

    repo = "models--" + FIXTURE.replace("/", "--")
    snapshots = sorted(Path(HF_HUB_CACHE).glob(f"{repo}/snapshots/*"))
    for snapshot in snapshots:
        if (snapshot / "model_index.json").is_file():
            return snapshot
    pytest.skip(f"{FIXTURE} is not in the local HF cache ({HF_HUB_CACHE})")


def _devices(pipe: Any) -> Dict[str, str]:
    return {
        name: str(next(module.parameters()).device)
        for name, module in (
            ("unet", pipe.unet),
            ("vae", pipe.vae),
            ("text_encoder", pipe.text_encoder),
        )
    }


def _load(device: str) -> Any:
    binding = DeployBinding(
        checkpoint_ref="ckpt:tiny@fixture", checkpoint_dir=_local_snapshot()
    )
    return LoadContext(binding=binding, device=device).load(StableDiffusionPipeline)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
@pytest.mark.skipif(
    not nvml_is_healthy(),
    # Kept under the census normalizer's 120-char key truncation (conftest
    # `_norm`): a longer reason is cut mid-phrase, leaving a trailing space the
    # census FILE's own `.strip()` removes — a disposition that never matches.
    reason="this host's NVML is version-mismatched; torch's P2P check raises inside pipeline.to(cuda)",
)
def test_the_bridge_places_the_pipeline_where_the_worker_said() -> None:
    """RED before pgw#1452: every component came back on `cpu` with a CUDA
    device present and idle."""
    devices = _devices(_load("cuda"))
    assert set(devices.values()) == {"cuda:0"}, (
        f"the worker decided `cuda` and the bridge left components elsewhere: "
        f"{devices} — this is the pgw#1452 defect, where a bare-tree boot "
        f"silently serves from system RAM"
    )


def test_the_bridge_names_no_device_of_its_own() -> None:
    """The other half of the contract, and the reason the fix is not a
    `.to("cuda")` in the bridge: handed no decision, it must place nothing.

    A bridge that reached for CUDA on its own would take the placement
    decision away from the worker — and would break the derive path, which
    builds on `meta` precisely because nothing is supposed to move.
    """
    devices = _devices(_load(""))
    assert set(devices.values()) == {"cpu"}, (
        f"no placement decision was handed down, so the bridge must place "
        f"nothing; components landed on {devices}"
    )
