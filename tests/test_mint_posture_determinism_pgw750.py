"""pgw#750 (task 1): the resident placement refinement (off vs vae_only)
must be DETERMINISTIC per SKU — never a function of live free VRAM.

Live evidence (0.2.14 four-leg proof): an identical L4 fleet split 6/13
into off/vae_only cohorts because the off-headroom branch keyed on
marginal live free VRAM. The two modes trace DIFFERENT decode graph
classes, so the compiled object set a mint proves — and whether the proof
passes at all — was per-pod roulette (~6 pods burned 915s each into
``self_mint_abort phase=proof_failed``). Same-SKU adoption requires every
pod of one SKU to mint the same recipe.

Red-verified: on the pre-fix tree the determinism test fails (the two
free-VRAM postures pick different modes)."""

from __future__ import annotations

import pytest

from gen_worker.models import memory


class _Pipe:
    pass


def _mode(avail: float, total: float = 24.0, model: float = 8.0) -> str:
    return memory.select_auto_mode(
        pipeline=_Pipe(),
        available_vram_gb=avail,
        model_size_gb=model,
        total_vram_gb=total,
    )


def test_resident_refinement_is_free_vram_independent() -> None:
    """The pgw#750 live shape: same SKU (24 GB L4), same model, two pods
    whose free VRAM straddles the off-headroom threshold — the resident
    mode must be IDENTICAL."""
    # Pre-fix: 23.5 free -> usable 21.5, 21.5-8 >= 8 -> "off";
    #          17.0 free -> usable 15.0, 15.0-8 <  8 -> "vae_only".
    roomy = _mode(avail=23.5)
    tight = _mode(avail=17.0)
    assert roomy == tight, (
        f"resident refinement split by live free VRAM ({roomy!r} vs "
        f"{tight!r}) — the pgw#750 mint-posture roulette"
    )


def test_refinement_keys_on_total_capacity_per_sku() -> None:
    # 24 GB SKU, 8 GB model: sku_usable = 24-1-2 = 21, 21-8 >= 8 -> off.
    assert _mode(avail=17.0, total=24.0, model=8.0) == "off"
    # 16 GB SKU, 8 GB model: sku_usable = 13, 13-8 < 8 -> vae_only.
    assert _mode(avail=15.0, total=16.0, model=8.0) == "vae_only"


def test_fit_decisions_stay_free_vram_based() -> None:
    """Safety is untouched: a card whose FREE VRAM cannot hold the model
    still takes the offload rungs, whatever its total capacity."""
    assert _mode(avail=5.0, total=24.0, model=8.0) == "group_offload"
    assert _mode(avail=7.5, total=24.0, model=8.0) == "model_offload"


def test_no_total_probe_falls_back_to_free(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without a total-capacity probe (no CUDA) the old free-based
    refinement is the only input left — degrade, never crash."""
    monkeypatch.setattr(memory, "get_total_vram_gb", lambda *a, **k: 0.0)
    mode = memory.select_auto_mode(
        pipeline=_Pipe(), available_vram_gb=23.5, model_size_gb=8.0)
    assert mode in ("off", "vae_only")
