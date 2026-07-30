"""pgw#763 stage 4, GPU arm: is a CUDA context actually re-established?

A poisoned or dead CUDA context is the case that most justifies the
control/compute split — a sticky CUDA error fails every later CUDA call in that
process no matter what Python catches, and only a fresh process gets a clean
context. Stage 1 asserted respawn re-establishes it. This asks the silicon.

The witness is `os.getpid()` returned from inside the handler, alongside a real
matmul result. Two probes bracketing one death must report DIFFERENT pids on the
SAME pod, both with a functioning device: same pod proves nothing was rebought,
different pid proves the process is new, and a correct matmul proves the new
process has a WORKING context rather than merely an initialized one.
"""

import os
from typing import List

import msgspec
from gen_worker import NoWarmup, RequestContext, Resources, ValidationError, endpoint


class ProbeInput(msgspec.Struct):
    text: str = ""


class ProbeOutput(msgspec.Struct):
    pid: int
    device_name: str
    device_count: int
    # A real reduction over a device-side matmul. A context that is merely
    # initialized cannot produce this; a poisoned one raises instead.
    matmul_trace: float
    vram_allocated_bytes: int
    cuda_context_ok: bool


class OOMOutput(msgspec.Struct):
    response: str


# Kept resident for the lifetime of the process so the OOM happens with real
# weights-shaped bytes in VRAM — te#138's actual shape, not a bare allocation.
_RESIDENT: List[object] = []


def _cuda_or_refuse():
    import torch

    if not torch.cuda.is_available():
        raise ValidationError("procsplit-cuda-probe requires a CUDA device")
    return torch


@endpoint(
    # The FUNCTION's declared resources are what the hub's compute-class
    # resolver reads (from the discovery manifest) — endpoint.toml's
    # `[resources]` is only the endpoint default, and the mutable
    # release-level patch cannot promote a function that declared no GPU.
    # Omitting this bought a CPU pod for a CUDA endpoint on the first GPU-arm
    # attempt: `[compute-class] resolved function=cuda-probe class=cpu`, then
    # the handler correctly refused with "requires a CUDA device".
    resources=Resources(gpu=True),
    # The platform BOOT-WARMS every declared function, and it warmed
    # `cuda_oom` — which allocated host RAM until the kernel killed the
    # child before a single request was served. GEN_WORKER_OOM_PROBE gates
    # TENANT traffic; it does not gate the platform's own warmup, so an
    # env gate alone cannot keep a deliberately-fatal probe safe.
    warmup=NoWarmup(reason="pgw#763 resilience probe: cuda_oom deliberately dies; warming it kills the pod at boot"),
    # th#1087 declaration gate — see marco-polo's note: GEN_WORKER_PROCESS_SPLIT
    # should be platform-reserved, not tenant-declared.
    env=["GEN_WORKER_OOM_PROBE", "GEN_WORKER_PROCESS_SPLIT"],
)
class ProcSplitCudaProbe:
    def cuda_probe(self, ctx: RequestContext, data: ProbeInput) -> ProbeOutput:
        """Establish (or re-establish) a CUDA context and prove it computes."""
        torch = _cuda_or_refuse()
        ctx.raise_if_cancelled()
        dev = torch.device("cuda:0")
        if not _RESIDENT:
            # ~512 MiB of resident VRAM: the "loaded model" this pod is holding.
            _RESIDENT.append(torch.empty(128 * 1024 * 1024, dtype=torch.float32, device=dev))
        a = torch.randn(1024, 1024, device=dev, dtype=torch.float32)
        trace = float((a @ a.T).diagonal().sum().item())
        torch.cuda.synchronize()
        out = ProbeOutput(
            pid=os.getpid(),
            device_name=torch.cuda.get_device_name(0),
            device_count=torch.cuda.device_count(),
            matmul_trace=trace,
            vram_allocated_bytes=int(torch.cuda.memory_allocated(0)),
            cuda_context_ok=True,
        )
        ctx.log(
            f"pgw#763 cuda probe: pid={out.pid} device={out.device_name} "
            f"vram_allocated={out.vram_allocated_bytes}"
        )
        return out

    def cuda_oom(self, ctx: RequestContext, data: ProbeInput) -> OOMOutput:
        """Hold VRAM, then exhaust HOST RAM until the cgroup killer fires.

        te#138's exact shape: a pipeline resident on the card while the
        container's host-RAM budget is blown. Uncatchable by construction, so
        the surviving control parent is the only thing that can report which
        request died and keep the pod serving.
        """
        if os.environ.get("GEN_WORKER_OOM_PROBE", "").strip() != "1":
            raise ValidationError(
                "cuda-oom is a pgw#763 resilience probe; it is refused unless "
                "the endpoint version sets GEN_WORKER_OOM_PROBE=1")
        torch = _cuda_or_refuse()
        dev = torch.device("cuda:0")
        if not _RESIDENT:
            _RESIDENT.append(torch.empty(128 * 1024 * 1024, dtype=torch.float32, device=dev))
        ctx.log("pgw#763 cuda oom probe: VRAM resident, now exhausting host RAM")
        chunks = []
        for i in range(4096):  # 1 TiB ceiling; the cgroup lands long before
            chunks.append(bytearray(256 * 1024 * 1024))
            chunks[-1][::4096] = b"\x01" * (len(chunks[-1]) // 4096)  # touch
            if i % 4 == 0:
                ctx.log(f"pgw#763 cuda oom probe: {(i + 1) * 256} MB host resident")
        return OOMOutput(response="unreachable: the cgroup never fired")
