"""pgw#1329's acceptance bar, on a real GPU: two arms, one bit pattern.

The claim under test is not "the store path runs". It is that a compiled
graph armed from STORE bytes by manifest FQN produces output **bitwise
identical** to the same graph armed from a resident eager module — because if
it does not, "loading the pipeline is a policy choice" is false and adopt-only
serving (pgw#1328) changes what a pod answers.

One pod does the whole thing, because the proof is arm-vs-arm on ONE host:
portability is tcg#4's question, not this one.

    python -m gen_worker.benchmarks.store_arm_parity --output row.json

Steps, in order:

1. build a small module with real named parameters AND a non-persistent
   buffer, on the GPU;
2. mint it through the sealed ``Engine`` — a real ``torch.export`` +
   AOTInductor compile, exactly as a serving pod mints (§4.28/§4.30);
3. write the module's ``state_dict`` to safetensors — the STORE;
4. arm A: ``arm_compiled_graph`` (module-sourced, today's path);
5. arm B: ``arm_compiled_graph_from_store`` (store-sourced, no module),
   with ``diffusers`` fenced unimportable and ``nn.Module.__init__``
   poisoned for the duration;
6. run both over the same pinned inputs and compare RAW BYTES;
7. arm a CONTROL from a store whose `lin.weight` differs by one ULP and
   require the output to move on exactly the inputs that can carry the
   perturbation — an equality nothing can falsify is not a proof that the
   store fed the graph.

Bitwise means bitwise: the comparison is over ``.view(torch.uint8)``, not
``allclose``, and not ``torch.equal`` (which compares values, so two NaN
payloads or a -0.0/0.0 pair would pass).
"""

from __future__ import annotations

import argparse
import builtins
import json
import os
import platform
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, cast

import torch
from safetensors.torch import save_file
from torch import nn

from gen_worker import aot_constants, aot_serve, hostfacts
from gen_worker._vendor.tensorfs import LocalCAS
from gen_worker._vendor.torchcg import (
    Engine,
    GraphClassSpec,
    RuntimeCompatibility,
    build_call_ingress,
)

FAMILY = "store-arm-parity"
GRAPH_CLASS = "probe-block"
TARGET = "denoiser"
WIDTH = 64
WEIGHT_SET = "probe://store-arm-parity/v1"


class ProbeBlock(nn.Module):
    """Small, but every constant KIND the bind has to handle.

    Two parameters, one persistent buffer and one NON-persistent buffer.
    The last is the pgw#825 shape: ``state_dict()`` omits it while
    ``torch.export`` lifts it as a ``ConstantType::Buffer`` under its real
    FQN, so the module arm reaches it only through ``named_buffers`` — and
    the store arm must reach it through the checkpoint, or fail closed
    naming it. A probe without one would prove the easy half.
    """

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(WIDTH, WIDTH)
        self.register_buffer("scale", torch.linspace(0.5, 1.5, WIDTH))
        self.register_buffer(
            "bias_branch", torch.linspace(-0.25, 0.25, WIDTH), persistent=False
        )

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


def _store_state(module: nn.Module) -> Dict[str, torch.Tensor]:
    """Exactly what ``resident_constants`` sees, as a checkpoint would hold it.

    ``state_dict()`` plus the non-persistent buffers — the same union, so the
    store and the module are offering the identical set and any output
    difference is attributable to the PATH, not to a different table.
    """

    state: Dict[str, torch.Tensor] = {
        name: tensor.detach().cpu().contiguous()
        for name, tensor in module.state_dict().items()
    }
    for name, buffer in module.named_buffers():
        if buffer is not None:
            state.setdefault(str(name), buffer.detach().cpu().contiguous())
    return state


def _raw(tensor: torch.Tensor) -> bytes:
    return tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()


class _NoDiffusers:
    """``diffusers`` unimportable and ``nn.Module`` unconstructible."""

    def __enter__(self) -> "_NoDiffusers":
        self._import: Callable[..., Any] = builtins.__import__
        self._module_init: Callable[..., Any] = nn.Module.__init__

        def fenced(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "diffusers" or name.startswith("diffusers."):
                raise AssertionError(f"the store-sourced arm imported {name!r}")
            return self._import(name, *args, **kwargs)

        def poisoned(_self: Any, *args: Any, **kwargs: Any) -> None:
            raise AssertionError("the store-sourced arm constructed an nn.Module")

        builtins.__import__ = fenced
        setattr(nn.Module, "__init__", poisoned)
        return self

    def __exit__(self, *_exc: object) -> None:
        builtins.__import__ = self._import
        setattr(nn.Module, "__init__", self._module_init)


def run(output: Path, target: str) -> Dict[str, Any]:
    # `hostfacts` is the ONE home for these three questions (pgw#896). Asking
    # torch directly here would be the 75th raw `is_available()` call site, and
    # it would also format `sm_NN` a second way — the axis the artifact is
    # keyed on must not have two spellings in the one program that proves the
    # artifact.
    device_name, host_sm = hostfacts.device_identity()
    if not hostfacts.cuda_ready() or not host_sm:
        raise SystemExit("this probe is the GPU bar; it must not run on CPU")
    if target != host_sm:
        # An artifact minted for one sm and executed on another is the exact
        # confusion `cg-key-v1` exists to make impossible; a probe that allowed
        # it could report a bitwise verdict about a pairing no pod can have.
        raise SystemExit(
            f"--target {target} but this card is {host_sm} ({device_name})"
        )
    device = "cuda"
    torch.manual_seed(1329)

    module = ProbeBlock().to(device).eval()
    example = torch.linspace(-3.0, 3.0, WIDTH, device=device).reshape(1, WIDTH)
    with torch.no_grad():
        program = torch.export.export(module, (example,))
    ingress = build_call_ingress(program, ("value",), (example,), {})
    spec = GraphClassSpec(
        GRAPH_CLASS,
        TARGET,
        program,
        {
            "v": 3,
            "lifted_inputs": [],
            "pytree": {"in": "leaf", "out": "leaf", "ingress": ingress.as_dict()},
            "specialization": {},
        },
    )

    scratch = Path(tempfile.mkdtemp(prefix="pgw1329-"))
    # The probe owns its Inductor cache outright rather than deferring to an
    # inherited one: a warm cache from an earlier arm would make "both arms
    # compiled the same graph" an assumption instead of a fact.
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(scratch / "inductor-cache")
    cas_root = scratch / "cas"
    engine = Engine(LocalCAS(cas_root))
    minted = engine.compile(
        spec, RuntimeCompatibility(target, toolchain=_toolchain()), scratch / "minted"
    )
    key = str(minted.compiled_graph.key)

    component = scratch / "store" / TARGET
    component.mkdir(parents=True)
    state = _store_state(module)
    save_file(state, str(component / "model.safetensors"))

    cfg = SimpleNamespace(family=FAMILY, lora_bucket=0)
    inputs: List[torch.Tensor] = [
        example,
        torch.linspace(-1.0, 1.0, WIDTH, device=device).reshape(1, WIDTH),
        torch.zeros(1, WIDTH, device=device),
    ]

    # --- arm A: today's path, the eager module IS the constant source ------
    pipeline = SimpleNamespace(**{TARGET: module})
    aot_serve.arm_compiled_graph(pipeline, cfg, key, cas_root)
    with torch.no_grad():
        module_out = [module(sample) for sample in inputs]

    # --- arm B: the store, with no module reachable at all -----------------
    store = aot_constants.SafetensorsConstantStore.for_component(
        component, weight_set=WEIGHT_SET, why="pgw#1329 store-sourced arm"
    )
    with _NoDiffusers():
        armed = aot_serve.arm_compiled_graph_from_store(
            cfg, key, store, device=device, cache_dir=cas_root
        )
        with torch.no_grad():
            store_out = [armed(sample) for sample in inputs]

    # --- the NON-VACUITY control ------------------------------------------
    # An equality proof is worth nothing until the same rig can produce an
    # INequality. A store whose `lin.weight` differs by one ULP must move the
    # output; if it does not, the store arm is not the thing feeding the
    # graph and every `bitwise_equal` above is an artefact of the rig.
    control_dir = scratch / "store-control" / TARGET
    control_dir.mkdir(parents=True)
    perturbed = dict(state)
    perturbed["lin.weight"] = torch.nextafter(
        state["lin.weight"], torch.full_like(state["lin.weight"], float("inf"))
    )
    save_file(perturbed, str(control_dir / "model.safetensors"))
    control_store = aot_constants.SafetensorsConstantStore.for_component(
        control_dir, weight_set=WEIGHT_SET + "-control", why="pgw#1329 control"
    )
    with _NoDiffusers():
        control = aot_serve.arm_compiled_graph_from_store(
            cfg, key, control_store, device=device, cache_dir=cas_root
        )
        with torch.no_grad():
            control_out = [control(sample) for sample in inputs]
    # A perturbed WEIGHT can only reach the output through a non-zero input:
    # the all-zeros call is `lin(0) = bias`, where the weight is arithmetically
    # unreachable. So the control's expectation is per input, and it binds in
    # BOTH directions — a zero-input call that moved would mean the
    # perturbation leaked somewhere it cannot legitimately reach.
    control_moved = [
        _raw(left) != _raw(right) for left, right in zip(store_out, control_out)
    ]
    control_expected = [bool(sample.abs().sum().item() > 0) for sample in inputs]

    rows = [
        {
            "call": index,
            "bitwise_equal": _raw(left) == _raw(right),
            "control_differs": moved,
            "control_expected_to_differ": expected,
            "control_as_expected": moved == expected,
            "dtype": str(left.dtype),
            "shape": list(left.shape),
        }
        for index, (left, right, moved, expected) in enumerate(
            zip(module_out, store_out, control_moved, control_expected)
        )
    ]
    row = {
        "issue": "pgw#1329",
        "key": key,
        "target": target,
        # a column of THIS probe's result row, off StoreArmedGraph — not a
        # read of TCG's metadata block.
        "graph_class": armed.graph_class,  # tcg-vocab: own result column
        "weight_set": str(armed.weight_set),
        "control_weight_set": str(control.weight_set),
        "declared_constants": len(armed.runner.declared_fqns()),
        "store_held_tensors": len(store.describe()),
        "store_sourced_constants": len(armed.constants),
        "distinct_runners": armed.runner is not control.runner,
        "toolchain": _toolchain(),
        "device_name": device_name,
        "calls": rows,
        "bitwise_equal": all(entry["bitwise_equal"] for entry in rows),
        "control_non_vacuous": any(control_moved)
        and control_moved == control_expected,
        "constant_source": armed.meta.get("constant_source"),
    }
    output.write_text(json.dumps(row, indent=2) + "\n")
    print(json.dumps(row, indent=2), flush=True)
    if not row["bitwise_equal"]:
        raise SystemExit("STORE-SOURCED ARM IS NOT BITWISE EQUAL — the bar failed")
    if not row["control_non_vacuous"]:
        raise SystemExit(
            "the 1-ULP control did not move the output where it must (or moved "
            "it where it cannot) — the equality above is vacuous and proves "
            "nothing about the store arm"
        )
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("row.json"))
    parser.add_argument("--target", default="", help="concrete sm_NN; derived when empty")
    arguments = parser.parse_args()
    target = arguments.target or hostfacts.device_identity()[1]
    if not target:
        raise SystemExit("no CUDA device to derive a target from")
    run(arguments.output, target)


if __name__ == "__main__":
    main()
