#!/usr/bin/env python3
"""pgw#1606 GPU leg — the ladder against a REAL card, and the confession it prints.

Everything else about this issue proves on CPU against fabricated cards, which
is correct: a decision does not need a GPU to be wrong. Exactly two facts
cannot be fabricated, and this leg is for those two only:

  1. `host_card_facts()` reads a real sm and a real VRAM figure, not sm0.
  2. `w8a8_gemm_mode()` is a LIVE micro-benchmark — one 16x16 `_scaled_mm`
     call, then a 4096-cubed GEMM median-of-ten that must clear 1.10x over
     bf16. Its verdict is the fp8 rung's gate, and on a CPU box it is
     unaskable. What it answers on THIS card is a measurement, and the whole
     point of the ladder consuming it rather than assuming it.

Then one real materialization on the card, so the `ctx.load_pipeline` path is
exercised with CUDA present rather than only on CPU tensors.

Short by construction. Run it, read the four blocks, tear down.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def main() -> int:
    import torch

    from gen_worker._vendor.tensorfs import contracts
    from gen_worker.serving import lane_ladder as L
    from gen_worker.serving import lane_materialize as M
    from gen_worker.serving.lane_host import HostKernelGates, host_card_facts

    if not torch.cuda.is_available():
        print("REFUSING: no CUDA. This leg exists only to read facts a CPU "
              "box cannot answer; running it without a card would produce a "
              "green result that proves nothing.")
        return 2

    print("=" * 72)
    print("1. THE CARD, MEASURED")
    print("=" * 72)
    card = host_card_facts()
    print(f"   {card}")
    print(f"   label -> {card.label}")
    assert card.sm > 0, "a real card must not read as sm0"

    print()
    print("=" * 72)
    print("2. THE KERNEL GATE — a benchmark, not a table lookup")
    print("=" * 72)
    gates = HostKernelGates()
    t0 = time.time()
    w8a8 = gates.w8a8_mode()
    t1 = time.time()
    w4a4 = gates.w4a4_mode()
    t2 = time.time()
    print(f"   w8a8_gemm_mode() -> {w8a8!r}   ({t1 - t0:.2f}s to answer)")
    print(f"   w4a4_gemm_mode() -> {w4a4!r}   ({t2 - t1:.2f}s to answer)")
    print("   (an empty string is a REAL veto: the fp8/fp4 GEMM did not clear")
    print("    1.10x over bf16 on this card, and the rung is rejected by name)")

    print()
    print("=" * 72)
    print("3. THE LADDER, ON THIS CARD, OVER TWO REAL SDXL CONTRACTS")
    print("=" * 72)
    from gen_worker.models import SDXL
    from gen_worker.serving.loader import LoadedEndpoint
    from gen_worker.serving.model import Model

    BF16 = contracts.SDXL_DIFFUSERS_BF16
    FP8 = contracts.SDXL_DIFFUSERS_FP8_ROWWISE

    class TwoLane(Model[SDXL], lanes={BF16: "vram7g", FP8: "vram5g"}):
        def load(self, ctx) -> None:  # pragma: no cover
            self.pipe = ctx.load_pipeline(object)

    class Staged:
        """Both trees staged, fp8 half the bytes — the shape the upcast rung
        exists for. Sizes are the only fabricated thing here, and they are
        fabricated because staging two real SDXL trees is a download, not a
        decision."""

        def verdict(self, contract_id: str) -> str:
            return L.VERDICT_SATISFIES

        def transfer_bytes(self, contract_id: str) -> int:
            return {BF16.stamp: 6_900_000_000, FP8.stamp: 3_500_000_000}.get(
                contract_id, 0)

    loaded = LoadedEndpoint(module_name="gpu-leg", entrypoints={},
                            models=(TwoLane,))
    resolved = loaded.resolve(TwoLane, card=card, verdicts=Staged(),
                              gates=gates)
    print()
    print("   THE CONFESSION:")
    print(f"   {resolved.confession()}")
    print()
    print(f"   chosen body      : {resolved.body}")
    print(f"   chosen contract  : {resolved.contract_id}")
    print(f"   reason           : {resolved.reason}")
    for rung in resolved.rejected:
        print(f"   rejected         : {rung.line()}")
    if resolved.upcast:
        print(f"   fetching instead : {resolved.fetch_contract} "
              f"(saved {resolved.transfer_saved_bytes / 1e9:.2f} GB on the wire)")

    print()
    print("=" * 72)
    print("4. A REAL MATERIALIZATION WITH CUDA PRESENT")
    print("=" * 72)
    import tempfile

    from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel

    from gen_worker.serving.context import DeployBinding, LoadContext

    unet = UNet2DModel(
        sample_size=8, in_channels=3, out_channels=3,
        block_out_channels=(32, 32), layers_per_block=1,
        down_block_types=("DownBlock2D", "AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D", "UpBlock2D"), norm_num_groups=8,
    ).eval()
    with tempfile.TemporaryDirectory() as tmp:
        tree = Path(tmp) / "tiny"
        DDPMPipeline(unet=unet,
                     scheduler=DDPMScheduler(num_train_timesteps=10)
                     ).save_pretrained(tree)
        ctx: LoadContext = LoadContext(
            binding=DeployBinding(checkpoint_ref="gpu-leg@1",
                                  checkpoint_dir=tree),
            resolved=resolved if L.is_baseline(resolved.body) else
            L.ResolvedLane(declared=resolved.declared, body="bf16-w16a16",
                           reason=L.CHOSE_BASELINE, card=card),
            device="cuda",
        )
        pipe = ctx.load_pipeline(DDPMPipeline)
        census = M.lane_census(type("P", (), {"unet": pipe.unet})())
        dev = next(pipe.unet.parameters()).device
        print(f"   built on         : {dev}")
        print(f"   census           : {census}")
        print(f"   observed lane    : {M.lane_of(type('P', (), {'unet': pipe.unet})())}")
        assert dev.type == "cuda", (
            "the leg must place on the card, or it proved nothing a CPU run "
            "would not have")
        del pipe
    torch.cuda.empty_cache()

    print()
    print("=" * 72)
    print("DONE. Card released.")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
