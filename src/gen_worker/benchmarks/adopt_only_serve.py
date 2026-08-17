"""pgw#1328's acceptance bar, on a real GPU: boot → adopt → arm → serve, in a
process where importing the mint lane RAISES.

The claim is not "the adopt-only code path runs". It is that the split is a
PROCESS BOUNDARY — §4.28's *"minting is a side effect of serving"* on one side,
§4.29's pull-by-key on the other — and a boundary that only exists inside one
interpreter is not one. So this probe is deliberately TWO processes on one pod,
and the second cannot import what the first used:

* **phase 1, eager-capable** (this process): mint one small graph through the
  sealed ``Engine`` — a real ``torch.export`` + AOTInductor compile, exactly as
  a serving pod mints — write its constants to a store, and arm it once so the
  parent holds the reference bytes an ordinary pod would serve.
* **phase 2, adopt-only** (a CHILD, ``--serve``): declare the role, install the
  import blocker, and only THEN import the serving modules. It arms the same
  key from the same store with no ``nn.Module`` anywhere on the path
  (pgw#1329), serves the same inputs, refuses a call outside the envelope with
  the typed refusal, and refuses to mint.

The verdicts, and each one can fail on its own:

1. every one of the nine declared mint modules raises ``MintMachineryUnavailable``
   in the child, and none of them is resident afterwards;
2. the child's served output is **bitwise identical** to the parent's — the
   role removes a capability, not a numeric;
3. a call outside the declared envelope produces a typed refusal naming the
   key, the tcg#37 selection outcome and the ranked candidate that missed —
   never an eager fallback, which the child has no module to perform;
4. asking the seam to mint produces ``mint_forbidden``, not an ``ImportError``
   somebody has to attribute.

Run on the pod, on the fleet line (research/RIG-ENV.md: torch 2.13.0 / CUDA
13.0, upstream `pytorch/pytorch:2.13.0-cuda13.0-cudnn9-runtime` on RunPod)::

    apt-get install -y build-essential      # AOTI links a real .so
    python -m gen_worker.cuda_root          # the runtime base ships no CUDA tree
    python -m gen_worker.benchmarks.adopt_only_serve --output row.json

Both setup lines are load-bearing and were each learned by a failed run: the
runtime bases carry no `g++` and no CUDA root at all — `cuda_root`'s own
docstring is the record of why, and this probe is exactly the "PAID failure"
it describes if you skip it.

Bitwise means bitwise: ``.view(torch.uint8)``, not ``allclose`` and not
``torch.equal`` (which compares VALUES, so a ``-0.0``/``0.0`` pair or two NaN
payloads would pass).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, cast

import torch
from safetensors.torch import save_file
from torch import nn

from gen_worker import hostfacts

FAMILY = "adopt-only-1328"
GRAPH_CLASS = "probe-block"
TARGET = "denoiser"
WIDTH = 64
WEIGHT_SET = "probe://adopt-only-1328/v1"


class ProbeBlock(nn.Module):
    """Small, and with one non-persistent buffer (pgw#825) so the store arm is
    the real one rather than the easy half."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(WIDTH, WIDTH)
        self.register_buffer("scale", torch.linspace(0.5, 1.5, WIDTH))
        self.register_buffer(
            "bias_branch", torch.linspace(-0.25, 0.25, WIDTH), persistent=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        hidden = torch.nn.functional.gelu(self.lin(value))
        scale = cast(torch.Tensor, self.scale)
        branch = cast(torch.Tensor, self.bias_branch)
        return hidden * scale + branch


def _toolchain() -> Dict[str, str]:
    return {
        "torch": str(torch.__version__),
        "python": platform.python_version(),
        "cuda": str(torch.version.cuda or ""),
    }


def _raw(tensor: torch.Tensor) -> str:
    return tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes().hex()


def _inputs(device: str) -> List[torch.Tensor]:
    return [
        torch.linspace(-3.0, 3.0, WIDTH, device=device).reshape(1, WIDTH),
        torch.linspace(-1.0, 1.0, WIDTH, device=device).reshape(1, WIDTH),
        torch.zeros(1, WIDTH, device=device),
    ]


# ── phase 2: the ADOPT-ONLY child ────────────────────────────────────────


def serve(plan: Dict[str, Any]) -> Dict[str, Any]:
    """Everything an adopt-only pod does, in a process that cannot mint.

    The ORDER of the first three statements is the whole proof: role, blocker,
    and only then the serving imports. A child that imported first would have
    the machinery resident and would be measuring nothing.
    """
    from gen_worker.serve import guard, role

    role.declare(role.ServeRole.ADOPT_ONLY)
    guard.install()

    blocked: Dict[str, str] = {}
    for name in role.MINT_MACHINERY:
        try:
            __import__(name)
        except guard.MintMachineryUnavailable as exc:
            blocked[name] = exc.blocked
        else:  # pragma: no cover — the failure this probe exists to catch
            raise SystemExit(f"the adopt-only child IMPORTED {name}")

    from gen_worker import aot_constants, aot_serve
    from gen_worker.serve import mint_seam, refusal

    device = "cuda"
    cfg = SimpleNamespace(family=FAMILY, lora_bucket=0)
    store = aot_constants.SafetensorsConstantStore.for_component(
        Path(plan["component"]), weight_set=WEIGHT_SET,
        why="pgw#1328 adopt-only serve")
    armed = aot_serve.arm_compiled_graph_from_store(
        cfg, plan["key"], store, device=device, cache_dir=Path(plan["cas"]))
    with torch.no_grad():
        served = [_raw(armed(sample)) for sample in _inputs(device)]

    # The MISS. A rank-2 call the declared class does not admit: on an
    # eager-capable pod this is "serve this request eager"; here there is no
    # module to serve it with, so it must be an answer.
    miss: Dict[str, Any] = {}
    try:
        armed(torch.zeros(2, WIDTH, device=device))
    except refusal.AdoptOnlyRefused as exc:
        refused = exc.refusal
        miss = {
            "kind": refused.kind.value,
            "disposition": refused.disposition.value,
            "key": refused.compiled_graph_key,
            "selection": refused.selection.value if refused.selection else "",
            "candidates": [
                {"graph_class": str(row.graph_class),  # tcg-vocab: a column of this probe's own result row, read off CandidateMiss
                 "distance": list(row.distance),
                 "misses": [n.render() for n in row.misses]}
                for row in refused.candidates],
            "reported": refused.reported,
            "wire_detail": refused.wire_detail(),
        }
    else:  # pragma: no cover
        raise SystemExit("a call outside the envelope was SERVED, not refused")

    # The MINT, refused as a decision rather than as an import error.
    mint: Dict[str, Any] = {}
    try:
        mint_seam.supervision().may_delegate()
    except refusal.AdoptOnlyRefused as exc:
        mint = {"reason": exc.reason,
                "disposition": exc.refusal.disposition.value}
    else:  # pragma: no cover
        raise SystemExit("the adopt-only seam permitted a mint")

    return {
        "blocked": blocked,
        "resident_after": [n for n in role.MINT_MACHINERY if n in sys.modules],
        "supervision": type(mint_seam.supervision()).__name__,
        "served": served,
        "graph_class": armed.graph_class,  # tcg-vocab: own result column, off StoreArmedGraph
        "weight_set": str(armed.weight_set),
        "store_sourced_constants": len(armed.constants),
        "miss": miss,
        "mint": mint,
    }


# ── phase 1: the EAGER-CAPABLE parent ────────────────────────────────────


def run(output: Path, target: str) -> Dict[str, Any]:
    from gen_worker import aot_constants, aot_serve
    from gen_worker._vendor.tensorfs import LocalCAS
    from gen_worker._vendor.torchcg import (
        Engine, GraphClassSpec, RuntimeCompatibility, build_call_ingress)

    device_name, host_sm = hostfacts.device_identity()
    if not hostfacts.cuda_ready() or not host_sm:
        raise SystemExit("this probe is the GPU bar; it must not run on CPU")
    if target != host_sm:
        raise SystemExit(
            f"--target {target} but this card is {host_sm} ({device_name})")
    device = "cuda"
    torch.manual_seed(1328)

    module = ProbeBlock().to(device).eval()
    example = _inputs(device)[0]
    with torch.no_grad():
        program = torch.export.export(module, (example,))
    ingress = build_call_ingress(program, ("value",), (example,), {})
    spec = GraphClassSpec(
        GRAPH_CLASS, TARGET, program,
        {"v": 3, "lifted_inputs": [],
         "pytree": {"in": "leaf", "out": "leaf", "ingress": ingress.as_dict()},
         "specialization": {}})

    scratch = Path(tempfile.mkdtemp(prefix="pgw1328-"))
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(scratch / "inductor-cache")
    cas_root = scratch / "cas"
    engine = Engine(LocalCAS(cas_root))
    minted = engine.compile(
        spec, RuntimeCompatibility(target, toolchain=_toolchain()),
        scratch / "minted")
    key = str(minted.compiled_graph.key)

    component = scratch / "store" / TARGET
    component.mkdir(parents=True)
    state: Dict[str, torch.Tensor] = {
        name: tensor.detach().cpu().contiguous()
        for name, tensor in module.state_dict().items()}
    for name, buffer in module.named_buffers():
        if buffer is not None:
            state.setdefault(str(name), buffer.detach().cpu().contiguous())
    save_file(state, str(component / "model.safetensors"))

    # The REFERENCE bytes: the same store arm an ordinary eager-capable pod
    # performs. Comparing the child against THIS rather than against eager is
    # deliberate — eager-vs-compiled is close and not bitwise (§4.28's accepted
    # trade), so an eager reference would make the comparison unfalsifiable.
    cfg = SimpleNamespace(family=FAMILY, lora_bucket=0)
    store = aot_constants.SafetensorsConstantStore.for_component(
        component, weight_set=WEIGHT_SET, why="pgw#1328 reference arm")
    reference_arm = aot_serve.arm_compiled_graph_from_store(
        cfg, key, store, device=device, cache_dir=cas_root)
    with torch.no_grad():
        reference = [_raw(reference_arm(sample)) for sample in _inputs(device)]

    plan = {"key": key, "cas": str(cas_root), "component": str(component)}
    child = subprocess.run(
        [sys.executable, "-m", __spec__.name if __spec__ else
         "gen_worker.benchmarks.adopt_only_serve",
         "--serve", json.dumps(plan)],
        capture_output=True, text=True, check=False)
    if child.returncode != 0:
        raise SystemExit(
            f"the adopt-only child failed:\n{child.stdout}\n{child.stderr}")
    answer = json.loads(child.stdout.strip().splitlines()[-1])

    row: Dict[str, Any] = {
        "issue": "pgw#1328",
        "key": key,
        "target": target,
        "device_name": device_name,
        "toolchain": _toolchain(),
        "reference_graph_class": reference_arm.graph_class,  # tcg-vocab: own result column, off StoreArmedGraph
        "child": answer,
        "guard_blocked_every_mint_module": (
            sorted(answer["blocked"]) == sorted(
                __import__("gen_worker.serve.role", fromlist=["role"]
                           ).MINT_MACHINERY)),
        "no_mint_module_resident_in_child": not answer["resident_after"],
        "supervision_is_no_mint": answer["supervision"] == "NoMint",
        "served_bitwise_equal": answer["served"] == reference,
        "miss_refused_not_served": bool(answer["miss"]),
        "miss_names_the_key": answer["miss"].get("key") == key,
        "miss_carries_candidates": bool(answer["miss"].get("candidates")),
        "mint_forbidden": answer["mint"].get("reason") == "mint_forbidden",
    }
    output.write_text(json.dumps(row, indent=2) + "\n")
    print(json.dumps(row, indent=2), flush=True)

    for verdict in (
        "guard_blocked_every_mint_module", "no_mint_module_resident_in_child",
        "supervision_is_no_mint", "served_bitwise_equal",
        "miss_refused_not_served", "miss_names_the_key",
        "miss_carries_candidates", "mint_forbidden",
    ):
        if not row[verdict]:
            raise SystemExit(f"pgw#1328 BAR FAILED: {verdict}")
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("row.json"))
    parser.add_argument("--target", default="", help="concrete sm_NN")
    parser.add_argument("--serve", default="", help="internal: the adopt-only child")
    arguments = parser.parse_args()
    if arguments.serve:
        print(json.dumps(serve(json.loads(arguments.serve))), flush=True)
        return
    target = arguments.target or hostfacts.device_identity()[1]
    if not target:
        raise SystemExit("no CUDA device to derive a target from")
    run(arguments.output, target)


if __name__ == "__main__":
    main()
