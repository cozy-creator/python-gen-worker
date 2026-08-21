"""The eager bridge places weights where the WORKER said, or nowhere."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")

from diffusers import StableDiffusionPipeline  # noqa: E402

from gen_worker.serving.context import DeployBinding, LoadContext  # noqa: E402

from harness.nvml import nvml_is_healthy  # noqa: E402

FIXTURE = "hf-internal-testing/tiny-stable-diffusion-pipe"


def _local_snapshot() -> Path:
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
    reason="this host's NVML is version-mismatched; torch's P2P check raises inside pipeline.to(cuda)",
)
def test_the_bridge_places_the_pipeline_where_the_worker_said() -> None:
    devices = _devices(_load("cuda"))
    assert set(devices.values()) == {"cuda:0"}, (
        f"the worker decided `cuda` and the bridge left components elsewhere: "
        f"{devices} — this is the pgw#1452 defect, where a bare-tree boot "
        f"silently serves from system RAM"
    )


def test_the_bridge_names_no_device_of_its_own() -> None:
    """The other half of the contract, and the reason the fix is not a `.to("cuda")` in the bridge: handed no decision, it must place nothing."""
    devices = _devices(_load(""))
    assert set(devices.values()) == {"cpu"}, (
        f"no placement decision was handed down, so the bridge must place "
        f"nothing; components landed on {devices}"
    )


def test_the_host_default_device_is_not_a_literal() -> None:
    import inspect

    from gen_worker.serving.host import EndpointHost

    default = inspect.signature(EndpointHost).parameters["device"].default
    assert default == "", (
        f"EndpointHost defaults `device` to {default!r} — a literal device "
        f"name is a claim about the host that nothing measured"
    )


def test_the_probed_device_agrees_with_this_hosts_real_card() -> None:
    """The probe is the strong one: `hostfacts.cuda_state` allocates, runs an op, synchronizes and frees, so a card that is present but will not answer reads as unreadable rather than as usable."""
    from gen_worker.hostfacts import cuda_state
    from gen_worker.serving.placement import serving_device

    assert serving_device() == ("cuda" if cuda_state().present else "cpu")


def test_a_cardless_host_falls_back_to_cpu_and_says_so() -> None:
    """Run in a REAL cardless process rather than with a patched probe: `CUDA_VISIBLE_DEVICES=""` is how a CPU-only box actually presents, and the fallback has to be loud there — cozy-local on a CPU-only ..."""
    import os
    import subprocess
    import sys

    script = (
        "import logging,sys;"
        "logging.basicConfig(level=logging.WARNING,stream=sys.stderr);"
        "from gen_worker.serving.placement import serving_device;"
        "print(serving_device())"
    )
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    env["PYTHONPATH"] = os.pathsep.join(
        [p for p in sys.path if p] + [env.get("PYTHONPATH", "")]
    )
    done = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, env=env,
    )
    assert done.returncode == 0, done.stderr[-2000:]
    assert done.stdout.strip().splitlines()[-1] == "cpu", done.stdout
    assert "PLACEMENT" in done.stderr and "pgw#1452" in done.stderr, (
        f"the CPU fallback must name itself; stderr was {done.stderr[-2000:]!r}"
    )
