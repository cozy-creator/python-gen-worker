"""pgw#1001 — a bucket-bearing compiled graph can be adopted by the runtime that minted it.

Three defects, each of which ALONE made that impossible, each RED at HEAD.
Found in ~20 s a cycle on the pgw#997 micro family's `micro-lora` vehicle,
after pgw#999 made the refusal say its own name. sdxl minted `w8a8-lora64`;
this is that lane's LoRA half.

The third is the important one, and it is not a probe bug. `install_lifted_
lora_forward` replaces `model.forward` WHOLESALE, so a plain call that worked
the instant before the install raised the instant after it. On an armed pod a
branchless request falling back to eager hits that in PRODUCTION. It surfaced
as `numerics_refused` only because the numerics gate is the thing that probes
the branchless axis — so "fix the probe" would have masked a serving break.

    THE INVARIANT: arming a bucket must not alter the semantics of calls
    that do not use the bucket.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import pytest
import torch
from torch import nn

from gen_worker import compile_cache as cc
from gen_worker.compiled_graph_adopt import AdoptOutcome
from gen_worker.models import lora_lifted, provision

BUCKET = 64


@dataclass
class _Cfg:
    family: str = "pgw1001"
    lora_bucket: int = BUCKET
    shapes: Tuple[Tuple[int, int], ...] = ((256, 256),)
    targets: Tuple[str, ...] = ("transformer", "decoder")
    text_lens: Tuple[int, ...] = (16,)
    guidance_scales: Tuple[float, ...] = ()
    regional: bool = False


class _Denoiser(nn.Module):
    """Branch-capable: plain `nn.Linear`s, which gw#558 covers."""

    def __init__(self) -> None:
        super().__init__()
        self.proj_in = nn.Linear(8, 16)
        self.proj_out = nn.Linear(16, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj_out(torch.nn.functional.silu(self.proj_in(x)))


class _Decoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj_in = nn.Linear(8, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj_in(x)


class _Pipe:
    """A pipeline naming its denoiser from the SDK's own vocabulary."""

    def __init__(self) -> None:
        self.transformer = _Denoiser().eval()
        self.decoder = _Decoder().eval()


def _tcg_metadata() -> Dict[str, Any]:
    """One TCG graph specialization naming its own target, with no worker package map."""
    return {
        "graph_specialization": {
            "name": "transformer/adapter=true,cfg=true",
            "target": "transformer",
        },
    }


@pytest.fixture()
def armed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Drive the arm through one valid TCG graph-specialization metadata row."""
    from gen_worker import aot_serve

    monkeypatch.setattr(
        provision, "_compiled_graph_metadata", lambda p: _tcg_metadata())
    monkeypatch.setattr(provision, "arm_route", lambda mode: object())
    monkeypatch.setattr(
        aot_serve, "enable", lambda *a, **k: AdoptOutcome.hit("armed"))
    monkeypatch.setattr(provision, "gate_compiled_graph_numerics", lambda p, c: True)


# ---------------------------------------------------------------------------
# Defect 1 — the top-level `targets`/`module` do not exist
# ---------------------------------------------------------------------------


def test_the_lifted_target_is_resolved_from_the_per_entry_targets(
    armed: None, tmp_path: Path,
) -> None:
    """RED at HEAD: `targets` came only from `meta["targets"]`, which a packed
    compiled graph does not carry, so `module_name` was "" and the install was SILENTLY
    SKIPPED — leaving `aot_serve.enable` to refuse the artifact it had just
    been handed with `lifted_inputs_unbindable`.

    pgw#1176 CONSOLIDATED defect 2 into this row. It read: *"`decoder` sorts
    first among the entry names, and installing a lifted forward on a module
    with no branch container fails by name"* — a premise that needed ONE
    artifact carrying SEVERAL entries so there was a set to sort and a wrong
    first element to pick. An artifact now carries one graph specialization, so that
    shape is unconstructible and a row that built it would assert against
    nothing real. What survives of it is the second assert below: the install
    lands on the branch-capable denoiser and nowhere else.
    """
    pipe = _Pipe()
    cc.apply_lora_execution_lane(pipe, BUCKET)
    artifact = tmp_path / "graph.tar.gz"
    artifact.write_bytes(b"compiled graph")

    outcome = provision.arm_aot(pipe, _Cfg(), None, artifact, BUCKET)

    assert outcome.armed
    assert lora_lifted.lifted_binding(pipe.transformer) is not None, (
        "the lifted binding was never installed — the artifact's target lives "
        "in its ENTRY block and this call site only looked at the top level")
    assert lora_lifted.lifted_binding(pipe.decoder) is None, (
        "the lifted forward landed on the DECODER — `branch_targets` is the "
        "authority on which module is the denoiser")


# ---------------------------------------------------------------------------
# Defect 3 — THE INVARIANT. This one is a serving bug, not a probe bug.
# ---------------------------------------------------------------------------


def _plain_feed() -> torch.Tensor:
    return torch.randn(4, 8, generator=torch.Generator().manual_seed(1001))


def test_arming_a_bucket_does_not_change_a_call_that_does_not_use_it() -> None:
    """THE invariant, and the whole reason defect 3 is not a probe fix.

    RED at HEAD: the identical feed returned a tensor before
    `install_lifted_lora_forward` and raised `ValidationError: the lifted
    LoRA argument 'lora_a' is missing` after it. An armed pod's eager
    fallback for a branchless request would raise in PRODUCTION.
    """
    pipe = _Pipe()
    x = _plain_feed()
    with torch.no_grad():
        before = pipe.transformer(x).clone()

    cc.apply_lora_execution_lane(pipe, BUCKET)
    lora_lifted.install_lifted_lora_forward(pipe.transformer, BUCKET)

    with torch.no_grad():
        after = pipe.transformer(x)

    assert torch.equal(before, after), (
        "arming the bucket changed a call that does not use the bucket")


def test_a_half_supplied_adapter_pair_is_still_refused() -> None:
    """The fallthrough must not become a swallow: one operand without the
    other is a caller error and stays one."""
    pipe = _Pipe()
    cc.apply_lora_execution_lane(pipe, BUCKET)
    lora_lifted.install_lifted_lora_forward(pipe.transformer, BUCKET)

    with pytest.raises(Exception) as excinfo:
        pipe.transformer(_plain_feed(), lora_a=torch.zeros(4))
    assert "lora" in str(excinfo.value).lower()


def test_the_branchless_arm_is_TRACEABLE_without_operands() -> None:
    """The flip side of the invariant, and the reason the old refusal had to
    go rather than be gated: an `adapter=false` entry IS the branchless graph,
    so capturing it with no operands is CORRECT, not a substitution. Before
    this change that capture was impossible — the wrapper refused it.
    """
    pipe = _Pipe()
    cc.apply_lora_execution_lane(pipe, BUCKET)
    lora_lifted.install_lifted_lora_forward(pipe.transformer, BUCKET)

    exported = torch.export.export(
        pipe.transformer, (_plain_feed(),), strict=True)
    assert exported is not None
