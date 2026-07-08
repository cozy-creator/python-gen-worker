"""gw#393 local end-to-end: REAL diffusers pipeline on CPU, real safetensors
LoRA file, full executor RunJob path (wire -> ensure_local -> parse/validate ->
load_lora_weights/set_adapters -> handler -> unload).

Proves, with pixel equality:
  1. an applied adapter changes the output deterministically,
  2. unload restores the baseline exactly (pipeline byte-clean per request),
  3. a repeat request serves the parsed state dict from the RAM cache.

Requires torch + diffusers (+ transformers/peft); skipped otherwise. Run:
  uv run --extra dev pytest tests/test_lora_e2e_local.py  (in a torch venv)
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional

import msgspec
import pytest

torch = pytest.importorskip("torch")
diffusers = pytest.importorskip("diffusers")
pytest.importorskip("peft")

from gen_worker.api.binding import Hub, wire_ref  # noqa: E402
from gen_worker.executor import Executor  # noqa: E402
from gen_worker.pb import worker_scheduler_pb2 as pb  # noqa: E402
from gen_worker.registry import EndpointSpec  # noqa: E402

TINY_SD = "hf-internal-testing/tiny-stable-diffusion-torch"
MODEL_REF = "acme/tiny-sd"
LORA_REF = "user1/tiny-lora"


class _In(msgspec.Struct):
    seed: int = 0


class _Out(msgspec.Struct):
    checksum: str = ""


class _Endpoint:
    pipe: Any = None

    def setup(self, pipeline: str) -> None:  # pragma: no cover
        pass

    def run(self, ctx, payload: _In) -> _Out:
        import hashlib

        out = type(self).pipe(
            "a photo of a cat",
            num_inference_steps=2,
            guidance_scale=7.5,
            generator=torch.Generator("cpu").manual_seed(payload.seed),
            output_type="np",
        ).images[0]
        return _Out(checksum=hashlib.sha256(out.tobytes()).hexdigest())


def _kohya_lora_file(pipe: Any, path: Path, *, rank: int = 4, seed: int = 7) -> None:
    """Build a real kohya-format LoRA safetensors targeting the pipe's unet
    attention projections — the format civitai LoRAs arrive in."""
    from safetensors.torch import save_file

    g = torch.Generator("cpu").manual_seed(seed)
    sd: Dict[str, torch.Tensor] = {}
    n = 0
    for name, mod in pipe.unet.named_modules():
        if not isinstance(mod, torch.nn.Linear):
            continue
        if not name.endswith(("to_q", "to_k", "to_v")):
            continue
        kohya = "lora_unet_" + name.replace(".", "_")
        sd[f"{kohya}.lora_down.weight"] = torch.randn(
            rank, mod.in_features, generator=g) * 0.5
        sd[f"{kohya}.lora_up.weight"] = torch.randn(
            mod.out_features, rank, generator=g) * 0.5
        sd[f"{kohya}.alpha"] = torch.tensor(float(rank))
        n += 1
        if n >= 4:
            break
    assert n > 0, "tiny unet exposed no attention Linears"
    save_file(sd, str(path))


@pytest.fixture(scope="module")
def tiny_pipe():
    pipe = diffusers.StableDiffusionPipeline.from_pretrained(TINY_SD)
    pipe.set_progress_bar_config(disable=True)
    return pipe


class _Harness:
    def __init__(self, tmp_path: Path, pipe: Any) -> None:
        self.sent: List[pb.WorkerMessage] = []
        self.pipe = pipe

        async def _send(msg: pb.WorkerMessage) -> None:
            self.sent.append(msg)

        spec = EndpointSpec(
            name="txt2img", method=_Endpoint.run, kind="inference",
            payload_type=_In, output_mode="single", cls=_Endpoint,
            attr_name="run", models={"pipeline": Hub(MODEL_REF)},
        )
        self.executor = Executor([spec], _send)
        _Endpoint.pipe = pipe
        rec = self.executor._classes[spec.instance_key]
        rec.instance = _Endpoint()
        rec.ready = True
        self.executor.store.residency.track_ram(wire_ref(spec.models["pipeline"]), pipe)

        lora_dir = tmp_path / "lora-snap"
        lora_dir.mkdir(parents=True, exist_ok=True)
        _kohya_lora_file(pipe, lora_dir / "adapter.safetensors")

        async def _fake_ensure_local(ref, snapshot=None, *, binding=None) -> Path:
            assert ref == LORA_REF, f"unexpected ensure_local({ref})"
            self.executor.store.residency.track_disk(ref, lora_dir)
            return lora_dir

        self.executor.store.ensure_local = _fake_ensure_local  # type: ignore[method-assign]

    async def run(self, request_id: str, *, lora_weight: Optional[float] = None) -> str:
        loras = [] if lora_weight is None else [
            pb.LoraOverlay(ref=LORA_REF, weight=lora_weight)
        ]
        run = pb.RunJob(
            request_id=request_id, attempt=1, function_name="txt2img",
            input_payload=msgspec.msgpack.encode(_In(seed=0)),
            models=[pb.ModelBinding(slot="pipeline", ref=MODEL_REF, loras=loras)],
        )
        run.snapshots[LORA_REF].digest = "sha256:tiny-lora-v1"
        await self.executor.handle_run_job(run)
        job = self.executor.jobs[(request_id, 1)]
        await job.task
        results = [m.job_result for m in self.sent if m.WhichOneof("msg") == "job_result"]
        res = results[-1]
        assert res.status == pb.JOB_STATUS_OK, res.safe_message
        return msgspec.msgpack.decode(res.inline, type=_Out).checksum


def test_adapter_changes_output_and_unload_restores_baseline(tmp_path, tiny_pipe) -> None:
    async def _run() -> None:
        h = _Harness(tmp_path, tiny_pipe)

        baseline = await h.run("r-base-1")
        with_lora = await h.run("r-lora-1", lora_weight=1.0)
        after = await h.run("r-base-2")
        repeat = await h.run("r-lora-2", lora_weight=1.0)

        assert with_lora != baseline, "adapter had no visual effect"
        assert after == baseline, "unload did not restore the baseline pipeline"
        assert repeat == with_lora, "adapter application is not deterministic"
        # Repeat adapter request came from the RAM cache (no re-parse).
        assert h.executor._adapter_cache.hits == 1
        assert h.executor._adapter_cache.misses == 1

        half = await h.run("r-lora-3", lora_weight=0.5)
        assert half not in (baseline, with_lora), "weight scaling had no effect"

    asyncio.run(_run())


def test_stuffed_adapter_rejected_end_to_end(tmp_path, tiny_pipe) -> None:
    from safetensors.torch import save_file

    async def _run() -> None:
        h = _Harness(tmp_path, tiny_pipe)
        bad_dir = tmp_path / "bad-snap"
        bad_dir.mkdir()
        save_file(
            {"unet.conv_in.weight": torch.zeros(3, 3)},
            str(bad_dir / "adapter.safetensors"),
        )

        async def _bad_ensure(ref, snapshot=None, *, binding=None) -> Path:
            return bad_dir

        h.executor.store.ensure_local = _bad_ensure  # type: ignore[method-assign]
        run = pb.RunJob(
            request_id="r-bad", attempt=1, function_name="txt2img",
            input_payload=msgspec.msgpack.encode(_In()),
            models=[pb.ModelBinding(
                slot="pipeline", ref=MODEL_REF,
                loras=[pb.LoraOverlay(ref="evil/stuffed", weight=1.0)],
            )],
        )
        await h.executor.handle_run_job(run)
        await h.executor.jobs[("r-bad", 1)].task
        results = [m.job_result for m in h.sent if m.WhichOneof("msg") == "job_result"]
        assert results[-1].status == pb.JOB_STATUS_INVALID
        assert "non-LoRA key" in results[-1].safe_message

    asyncio.run(_run())
