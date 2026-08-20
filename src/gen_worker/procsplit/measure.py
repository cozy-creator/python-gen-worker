"""Pre-import hardware + canary measurement for the control parent (delta 2).

A handful of floats a worker reports about itself become FLEET-WIDE verdicts:
``HardwareUnsuitable`` fences a machine; ``HostCanary`` condemns a SKU on the
SPFabricLedger; the reported ``gpu_name`` chooses which verdict key gets written.
Measured inside the worker process — which has already imported tenant endpoint
code — any of them could be replaced by a handler, a module import side effect,
or a monkeypatched ``torch``. The hub-side corroboration gate contains that; this
removes the ability.

The control parent must stay torch-free (it is the process that survives a CUDA
death), so it cannot measure the silicon itself. It runs THIS module as a
short-lived subprocess instead, before and independently of any compute child:

    python -m gen_worker.procsplit.measure

It imports gen_worker and torch and nothing of the tenant's — no endpoint
module is ever named — measures, prints one JSON object, and exits. The parent
keeps the numbers and stamps them onto every Hello it relays. There is no
window in which tenant code has run and the measurement has not.
"""

from __future__ import annotations

import json
import os
import logging
import sys
import time
from typing import TYPE_CHECKING, Any, Dict, Tuple

if TYPE_CHECKING:  # heavy edges stay off this module's import scope
    from ..hostfacts import HostFacts

logger = logging.getLogger(__name__)


def measure() -> Dict[str, Any]:
    """Hardware facts, the boot host canary, and the build identity."""
    out: Dict[str, Any] = {"hardware": {}, "canary": None, "gen_worker_version": ""}
    # pgw#1414: RETRY AN EMPTY CENSUS WHEN THE CONTAINER HOLDS GPU DEVICE
    # NODES, and if it stays empty say so LOUDLY instead of shipping a clean
    # cpu-class answer. Measured on a rented 4090 (pod 3ntpe1zwbksuwo): the
    # census read `driver="" gpu="" count=0`, the worker registered
    # `class=cpu gpu=0`, and the hub then declined placement with
    # `compute_class_mismatch` 703+ times in a loop with NO terminal state
    # while the pod billed. Zero errors anywhere — because a swallowed census
    # and a genuinely cardless box produce byte-identical output.
    #
    # This census runs in the PARENT before any endpoint import; the child's
    # CUDA probe runs later. A driver mount landing between them gives exactly
    # that incident — a cpu Hello from a pod whose probe then passes — which is
    # why the retry is likely to fix it outright.
    #
    # ⚠️ pgw#1436: THE IN-PROCESS RETRY BELOW CANNOT RECOVER A CAPABILITY GAP,
    # and that is measured, not theorised. `torch.cuda.is_available()`
    # initialises CUDA lazily and ONCE PER PROCESS; a first call made before the
    # runtime is ready freezes False for the life of the process, so every later
    # attempt in this loop re-asks a question torch has already answered
    # permanently. Three rented pods across two machine classes and three driver
    # versions retried 4x with backoff and recovered on NONE of them — which a
    # genuine driver-mount race could never explain.
    #
    # The loop is KEPT because it does work for a `device` gap (NVML has no such
    # cache), and it is now bounded by `_CENSUS_RETRIES_INPROC`. The gap set is
    # REPORTED to the parent, which re-spawns this module in a FRESH process —
    # the only thing that can clear a frozen CUDA init. See
    # `parent._measure_host`.
    attempts = _CENSUS_RETRIES if gpu_devices_present() else 1
    for attempt in range(attempts):
        try:
            facts = probe_hardware()
        except Exception as exc:  # never fatal: an unmeasured axis is a zero
            out["hardware_error"] = f"{type(exc).__name__}: {exc}"
            break
        out.pop("hardware_error", None)
        out["hardware"] = facts.as_dict()
        gaps = _census_gaps(facts)
        # pgw#1436: the parent re-spawns on this. Reported every time, so an
        # incomplete census is machine-visible one level up instead of being
        # re-derived there from field emptiness.
        out["census_gaps"] = list(gaps)
        if not gaps or attempt == attempts - 1:
            break
        # Progressive: the CUDA runtime takes longer to come up than the
        # driver does, so the attempt that closes a `capability` gap is
        # usually later than the one that closes a `device` gap.
        backoff = _CENSUS_BACKOFF_S * (attempt + 1)
        logger.warning(
            "host census incomplete on attempt %d/%d (missing: %s) while %s "
            "exists — a GPU was assigned to this container, so this is "
            "UNREADABLE, not absent; retrying in %.1fs (pgw#1414/#1417)",
            attempt + 1, attempts, ", ".join(gaps),
            next((n for n in _GPU_DEVICE_NODES if os.path.exists(n)), "?"),
            backoff,
        )
        time.sleep(backoff)

    if "hardware_error" not in out and gpu_devices_present():
        facts_dict = out.get("hardware") or {}
        if (facts_dict.get("gpu_count") or facts_dict.get("gpu_name")
                or facts_dict.get("driver_version")) and not facts_dict.get("gpu_sm"):
            # pgw#1417: the card came back, its COMPUTE CAPABILITY did not.
            # Distinct from `census_unreadable` and distinctly worse to leave
            # silent: this host registers as `class=gpu` and looks healthy,
            # then refuses EVERY request with `gpu_capability_incompatible`,
            # because pgw#984 derives `min_sm` on every v2 release. Not
            # reporting an SM is not the same as not having one.
            # pgw#1436: ASK THE QUESTION THAT HAS AN ANSWER.
            #
            # Everything above this line restates the SYMPTOM ("gpu_sm is
            # empty"). `gpu_sm` is empty because `device_identity()` gates on
            # `cuda_ready()` — a bare `torch.cuda.is_available()` that swallows
            # every exception and returns False — so the reason is discarded at
            # the point of use. `cuda_state()` is the three-valued verdict built
            # on a real allocate/op/synchronize probe, and `cuda_ready`'s own
            # docstring says anything REPORTING to the fleet must call it
            # instead, "because this predicate cannot express it". The census is
            # exactly that caller and never called it.
            #
            # Vocabulary is HardwareUnsuitable's, deliberately: the same
            # reason_class/detail pair the FAILURE carrier already ships
            # (torch_unavailable | cuda_unavailable | driver_too_old |
            # cuda_error | unknown). One classification, two carriers.
            reason_class, detail = _capability_reason()
            out["capability_reason_class"] = reason_class
            out["capability_detail"] = detail
            out["capability_unreadable"] = (
                f"GPU {facts_dict.get('gpu_name') or '(unnamed)'} "
                f"(driver {facts_dict.get('driver_version') or '?'}) was read, "
                f"but its compute capability was not, across {attempts} "
                f"attempt(s). `gpu_sm` is empty, so every request carrying a "
                f"derived min_sm will refuse gpu_capability_incompatible. The "
                f"CUDA RUNTIME, not the driver, answers this — the driver is "
                f"clearly up. Probe says {reason_class}: {detail}"
            )
            logger.error(
                "capability_unreadable: %s (pgw#1417/#1436)",
                out["capability_unreadable"],
            )
        elif not (facts_dict.get("gpu_count") or facts_dict.get("gpu_name")
                  or facts_dict.get("driver_version")):
            # THE TYPED STATE. Not `hardware_error` — nothing raised — and not
            # silence, which is what let a cpu-class Hello leave this host.
            out["census_unreadable"] = (
                f"GPU device nodes exist "
                f"({', '.join(n for n in _GPU_DEVICE_NODES if os.path.exists(n))}) "
                f"but {attempts} census attempt(s) read no driver, no device "
                f"and no name. A card may be present and not answering — this "
                f"host must NOT be registered as cpu-class."
            )
            logger.error(
                "census_unreadable: %s (pgw#1414)", out["census_unreadable"]
            )
    try:
        from ..host_canary import get_host_canary

        c = get_host_canary()
        out["canary"] = {
            "memcpy_gbps": c.memcpy_gbps,
            "d2h_gbps": c.d2h_gbps,
            "pinned_alloc_ok": c.pinned_alloc_ok,
            "cpu_single_mbps": c.cpu_single_mbps,
            "cpu_multi_mbps": c.cpu_multi_mbps,
            "vcpus": c.vcpus,
            "ram_total_gb": c.ram_total_gb,
            "duration_ms": c.duration_ms,
            "interconnect": c.interconnect,
            "peer_gbps": c.peer_gbps,
            "peer_access": c.peer_access,
            "topo_link": c.topo_link,
        }
    except Exception as exc:
        out["canary_error"] = f"{type(exc).__name__}: {exc}"
    try:
        from ..toolchain import gen_worker_version

        out["gen_worker_version"] = gen_worker_version()
    except Exception:
        pass
    return out


def main() -> int:
    # One JSON object on stdout and nothing else. Every log line the imports
    # emit goes to stderr, where the parent forwards it to the pod log.
    sys.stdout.write(json.dumps(measure()))
    sys.stdout.flush()
    return 0


# ---------------------------------------------------------------------------
# pgw#1373: `probe_hardware` LIVES HERE NOW. It came from the deleted
# `lifecycle.py`, and its obvious home looked like `hostfacts` — it returns a
# `HostFacts`. That was wrong, and the serve-role closure walk said so
# immediately: the probe needs `topology` and `models.hub_policy` — NAMED FROM
# THE PACKAGE ROOT, i.e. `gen_worker.topology` and `gen_worker.models.hub_policy`,
# which from inside this package are `..topology` and `..models.hub_policy`.
# Spelling them bare here is how a single dot got typed at the import below and
# cost the fleet its `gpu_sm` (pgw#1438); the two dots are load-bearing. Putting
# it in `hostfacts` put that edge on the MODEL-FREE serve surface, dragging
# `models.residency -> models.memory -> structure_only -> diffusers` onto a
# path whose whole point is that it never imports a model library. `hostfacts`
# says "import this module freely" in its own header and has to keep meaning it.
#
# This module is the only caller and it runs in a DEDICATED measurement
# subprocess, so weight here costs nothing.
#
# Leaving the function deleted was a SILENT degradation, which is why it is
# restored rather than dropped: the `except Exception` above turns an
# ImportError into `hardware_error` plus an EMPTY hardware dict, so the parent
# shipped a measurement with no gpu_name, no gpu_count and no torch version,
# and every consumer read those zeros as a CPU-only box instead of as a missing
# measurement.



#: The container's own evidence that a GPU was ASSIGNED to it, independent of
#: whether the driver or CUDA runtime is answering yet. The NVIDIA container
#: runtime creates these nodes at container start; `privdrop.py` already
#: relaxes their modes for the dropped child, so the repo knows them.
_GPU_DEVICE_NODES = ("/dev/nvidiactl", "/dev/nvidia0", "/dev/nvidia-uvm")

#: pgw#1414: a cold-start driver mount can lose a race with this census.
#: Bounded, and short: the parent's whole measurement runs under
#: `_MEASURE_TIMEOUT_S`, and a pod waiting here is a pod not serving.
_CENSUS_RETRIES = 4
_CENSUS_BACKOFF_S = 1.5


def _capability_reason() -> Tuple[str, str]:
    """Why the CUDA runtime would not answer — `(reason_class, detail)`.

    pgw#1436. `cuda_state()` allocates, so it is never on a hot path; this is
    called at most once per census, only when a card was read and its
    capability was not. Vocabulary matches `HardwareUnsuitable.reason_class`
    so the hub classifies a LIVE worker's degradation with the same words it
    already uses for a dead one.

    Never raises: a diagnostic that fails must not cost the census its
    measurement, which is the whole failure mode this issue exists to end.
    """
    try:
        from ..hostfacts import cuda_state

        state = cuda_state()
        klass = (getattr(state, "probe_class", "") or "").strip() or "unknown"
        detail = (getattr(state, "detail", "") or "").strip()
        return klass, detail or "(probe returned no detail)"
    except Exception as exc:  # noqa: BLE001 — a probe never changes an outcome
        return "unknown", f"cuda_state() itself failed: {type(exc).__name__}: {exc}"


def gpu_devices_present() -> bool:
    """Whether this container was handed GPU device nodes.

    THE DISCRIMINATOR pgw#1414 turns on: "this host has no GPU" and "this host
    has a GPU I could not read" need OPPOSITE behaviour, and every other signal
    conflates them. `nvidia-smi` missing, NVML failing and `device_count() == 0`
    look identical on a cardless box and on a 4090 whose driver mount has not
    landed yet — which is exactly the pod that billed while the scheduler
    declined it 703 times.

    Device nodes are the one fact that does not depend on the driver working:
    the runtime creates them because a card was ASSIGNED, so their presence
    beside an empty census means "unreadable", never "absent".
    """
    return any(os.path.exists(node) for node in _GPU_DEVICE_NODES)


#: What a host holding GPU device nodes still OWES after a census attempt.
#: pgw#1417: this was `_census_is_empty`, and "empty" was the wrong question.
#: It asked *did we learn anything* when a GPU worker must answer *did we learn
#: everything a GPU worker has to report*. Round 4 of the rental proof recovered
#: `driver="580.173.02" gpu="NVIDIA GeForce RTX 4090" count=1` on the retry and
#: BROKE OUT — because that satisfied "not empty" — while `gpu_sm` was still 0.
#: The pod then registered `class=gpu` with no SM and every request refused
#: `gpu_capability_incompatible`, since pgw#984 derives `min_sm` on EVERY v2
#: release. A card with no capability is not a partial success; it is a worker
#: that cannot be given work.
#:
#: The two gaps have DIFFERENT causes and appear at different times, which is
#: why they are named separately: `device` is the driver mount (nvidia-smi/NVML)
#: and `capability` is the CUDA RUNTIME (`cuda_ready()` +
#: `torch.cuda.get_device_capability()`, `models/hub_policy.py:71`). The runtime
#: comes up AFTER the driver, so a census can legitimately see the card one
#: attempt before it can see the capability — which is exactly what round 4
#: measured.
def _census_gaps(facts: "HostFacts") -> Tuple[str, ...]:
    """`()` when this census is complete enough to register a GPU worker."""
    if not (facts.gpu_count or facts.gpu_name or facts.driver_version):
        return ("device",)
    if not facts.gpu_sm:
        return ("capability",)
    return ()


def probe_hardware() -> "HostFacts":
    """Measure this host ONCE into one immutable :class:`HostFacts`.

    The single producer of the facts the fleet acts on. ``vram_free_bytes`` is
    the input `gate_functions` turns into `unavailable` + `serve_plans` for
    EVERY function, so on a wide pod it must be the free pool of the group with
    the LEAST room, not card 0's. The MIN is the honest single scalar until the
    fit ladder is per-rank: it never promises a group room it does not have.
    """
    from ..hostfacts import (
        HostFacts, device_count, device_identity, free_vram_bytes,
        total_vram_bytes,
    )
    from ..topology import TopologyError as _TopologyError

    gpu_count = 0
    vram_total = 0
    vram_free = 0
    gpu_name = ""
    gpu_sm = ""
    torch_version = ""
    cuda_version = ""
    driver_version = ""
    installed_libs: tuple[str, ...] = ()
    # The HOST driver, read from NVML rather than torch: it must stay readable
    # when the CUDA runtime is not. A cu130 build on a 570.x host imports fine,
    # reports every version string correctly, and dies on its first allocation.
    try:
        from ..hardware_report import _nvidia_smi_driver_and_gpu

        driver_version, gpu_name = _nvidia_smi_driver_and_gpu()
    except Exception:
        pass
    try:
        count = device_count()
        if count:
            gpu_count = count
            gpu_name = device_identity(0)[0] or gpu_name
            vram_total = total_vram_bytes(0) or 0
            card0_free = free_vram_bytes(0) or 0
            vram_free = card0_free
            worst = _worst_group_free_vram_bytes()
            if worst and worst != card0_free:
                logger.info(
                    "fit inputs: gating on the least-free group (%d bytes) "
                    "instead of card 0 (%d bytes) — pgw#776",
                    int(worst), int(card0_free))
                vram_free = int(worst)
    except _TopologyError:
        # A WEDGED fabric is the one topology fault that must reach the caller:
        # `delivered_topology` raises it so the hub re-packs instead of this
        # worker booting and hanging every collective. Swallowed here it read
        # as "card 0, fine" and the pod went on to serve.
        raise
    except Exception:
        pass
    try:
        # 🔻 `..models`, NOT `.models`, and the defect was exactly this LEVEL —
        # nothing else about the call was ever wrong. The probe came from
        # `gen_worker/lifecycle.py` (pgw#1373), where ONE dot reached
        # `gen_worker.models`. From `gen_worker/procsplit/` one dot reaches
        # `gen_worker.procsplit.models`, which does not exist among this
        # package's modules — and the `except Exception: pass` below swallowed
        # the ModuleNotFoundError, so EVERY worker on EVERY pod reported an
        # empty `gpu_sm`/`torch_version`/`cuda_version` and then refused every
        # request carrying a pgw#984-derived `min_sm` (pgw#1417/#1436). It was
        # unconditional, which is why sdxl reproduced it six times out of six.
        #
        # WHERE THE WRONG DOT CAME FROM, so the next reader does not retype it:
        # the block comment at the top of this section says the probe "needs
        # `topology` and `models.hub_policy`" — bare names, written from the
        # PACKAGE ROOT's perspective, because that is where the probe lived when
        # the sentence was written. Read from inside `procsplit/`, `models.…`
        # transcribes to one dot. Every other relative import in this module is
        # `..`; this was the only single dot in the file.
        from ..models.hub_policy import detect_worker_capabilities

        caps = detect_worker_capabilities()
        installed_libs = tuple(str(x) for x in (caps.installed_libs or []))
        torch_version = str(caps.torch_version or "")
        cuda_version = str(caps.cuda_version or "")
        if caps.gpu_sm:
            gpu_sm = str(int(caps.gpu_sm))
    except Exception:
        pass
    return HostFacts(
        gpu_count=gpu_count,
        vram_total_bytes=vram_total,
        vram_free_bytes=vram_free,
        gpu_name=gpu_name,
        gpu_sm=gpu_sm,
        torch_version=torch_version,
        cuda_version=cuda_version,
        driver_version=driver_version,
        installed_libs=installed_libs,
    )

def _worst_group_free_vram_bytes() -> int:
    """The free pool of the least-roomy execution group, or 0 if unknowable.

    Reads the DELIVERED topology: on a pod this worker itself demoted for a
    non-NVLink fabric, ``from_env`` describes a packing that is not being
    served, so reporting from it makes reported and served topologies disagree.
    """

    from ..topology import delivered_topology

    groups = delivered_topology().all_groups()
    return int(min((g.free_vram_bytes() for g in groups), default=0))


# ---------------------------------------------------------------------------
# LAST IN THE FILE, DELIBERATELY.
#
# pgw#1414: this block used to sit above `probe_hardware` and its helpers, and
# `python -m gen_worker.procsplit.measure` — which is how the control parent
# actually runs this module — executes top to bottom, so `main()` ran BEFORE
# the definitions below were bound. The census died with
# `NameError: name 'gpu_devices_present' is not defined`, printed nothing on
# stdout, and the parent read an EMPTY measurement: the very cpu-class Hello
# this issue exists to prevent, caused by the fix for it.
#
# It survived an `import`-based check because importing binds every definition
# before anything calls them. Only the `-m` path — the production one — fails.
if __name__ == "__main__":
    sys.exit(main())
