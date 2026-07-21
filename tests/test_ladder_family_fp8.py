"""th#964 family lane policy — twin of tensorhub's precision family policy
(internal/orchestrator/precision family_policy_th964_test.go): conv-UNet
roots (sd1/sd2/sdxl) AUTO-pick the scale-free #fp8 sibling on sm_89+, w8a8
rows are AUTO-ineligible, bf16 stays the sub-floor default."""

from __future__ import annotations

import pytest

from gen_worker.api.binding import HF, Hub
from gen_worker.models import gguf_local, ladder
from gen_worker.models.hub_client import WorkerResolvedFlavor, WorkerResolvedRepo, WorkerResolvedRepoFile


@pytest.mark.parametrize(("family", "root"), [
    ("sdxl", "sdxl"),
    ("sdxl-illustrious", "sdxl"),
    ("sdxl-pony", "sdxl"),
    ("sdxl-turbo", "sdxl"),
    ("sd15", "sd1"),
    ("sd14", "sd1"),
    ("SD 1.5", "sd1"),
    ("sd2", "sd2"),
    ("sd35-large-turbo", "sd35-large"),
    ("flux1-dev", "flux1"),
    ("flux.2-klein-4b", "flux2-klein-4b"),
    ("z-image-turbo", "z-image"),
    ("qwen-image", "qwen-image"),
    ("something-new", "something-new"),
    ("", ""),
])
def test_family_root(family: str, root: str) -> None:
    assert ladder.family_root(family) == root


def test_w8a8_excluded_roots_match_hub_table() -> None:
    assert ladder.CONV_UNET_W8A8_EXCLUDED_ROOTS == {"sd1", "sd2", "sdxl"}
    for fam in ("sdxl", "sdxl-illustrious", "sd15", "sd2", "SDXL-Pony"):
        assert ladder.w8a8_excluded_for_family(fam)
    for fam in ("flux1-dev", "qwen-image", "ltx-2", "", "sdxl-distilled"):
        assert not ladder.w8a8_excluded_for_family(fam)


def _rows(*pairs: tuple[str, int]) -> list[WorkerResolvedFlavor]:
    return [WorkerResolvedFlavor(flavor=f, size_bytes=s) for f, s in pairs]


SDXL_ROWS = _rows(("", 6_941_377_969), ("fp8", 4_382_791_561), ("fp8-w8a8", 4_722_020_263))


def test_pick_sdxl_sm89_prefers_scale_free_fp8() -> None:
    for sm in (89, 90, 120):
        assert ladder.pick_family_fp8_flavor(
            SDXL_ROWS, model_family="sdxl-illustrious", gpu_sm=sm, free_vram_gb=24.0,
        ) == "fp8"


def test_pick_below_sm89_keeps_bf16_subfloor_default() -> None:
    assert ladder.pick_family_fp8_flavor(
        SDXL_ROWS, model_family="sdxl", gpu_sm=86, free_vram_gb=24.0,
    ) == ""


def test_pick_w8a8_only_never_auto_selected() -> None:
    rows = _rows(("", 7_000_000_000), ("fp8-w8a8", 4_722_020_263))
    assert ladder.pick_family_fp8_flavor(
        rows, model_family="sdxl", gpu_sm=89, free_vram_gb=24.0,
    ) == ""


def test_pick_non_excluded_family_is_policy_off() -> None:
    assert ladder.pick_family_fp8_flavor(
        SDXL_ROWS, model_family="flux1-dev", gpu_sm=90, free_vram_gb=80.0,
    ) == ""
    assert ladder.pick_family_fp8_flavor(
        SDXL_ROWS, model_family="", gpu_sm=90, free_vram_gb=80.0,
    ) == ""


def test_pick_gates_best_row_on_fit() -> None:
    # 4.38 GB row vs 4.0 GB free: hub walk falls to bf16, local keeps declared.
    assert ladder.pick_family_fp8_flavor(
        SDXL_ROWS, model_family="sdxl", gpu_sm=89, free_vram_gb=4.0,
    ) == ""


def test_pick_tiebreak_smallest_token() -> None:
    rows = _rows(("fp8-e4m3", 4_000_000_000), ("fp8", 4_400_000_000))
    assert ladder.pick_family_fp8_flavor(
        rows, model_family="sdxl", gpu_sm=89, free_vram_gb=24.0,
    ) == "fp8"


def _resolved(
    rows: list[WorkerResolvedFlavor], family: str = "",
    size_bytes: int = 6_941_377_969,
) -> WorkerResolvedRepo:
    files = [WorkerResolvedRepoFile(
        path="model_index.json", size_bytes=1, blake3="ab", url="http://x/f",
    )]
    return WorkerResolvedRepo(
        snapshot_digest="d" * 8, files=files, size_bytes=size_bytes,
        sibling_flavors=rows, model_family=family,
    )


def test_rebind_bare_sdxl_binding_to_fp8() -> None:
    binding = Hub("acme/wai-illustrious")
    out = ladder.maybe_rebind_family_fp8(
        binding, resolved=_resolved(SDXL_ROWS, "sdxl-illustrious"),
        gpu_sm=89, free_vram_gb=24.0,
    )
    assert out.flavor == "fp8"
    assert out.path == binding.path


def test_rebind_slot_family_fallback_when_resolve_has_none() -> None:
    binding = Hub("acme/wai-illustrious")
    out = ladder.maybe_rebind_family_fp8(
        binding, resolved=_resolved(SDXL_ROWS, ""), slot_family="sdxl",
        gpu_sm=89, free_vram_gb=24.0,
    )
    assert out.flavor == "fp8"


def test_rebind_resolve_family_wins_over_slot_family() -> None:
    binding = Hub("acme/some-model")
    out = ladder.maybe_rebind_family_fp8(
        binding, resolved=_resolved(SDXL_ROWS, "flux1-dev"), slot_family="sdxl",
        gpu_sm=89, free_vram_gb=24.0,
    )
    assert out is binding


def test_rebind_fails_open_on_unfoldable_binding() -> None:
    binding = HF("acme/model")
    out = ladder.maybe_rebind_family_fp8(
        binding, resolved=_resolved(SDXL_ROWS, "sdxl"), gpu_sm=89, free_vram_gb=24.0,
    )
    assert out is binding


def test_select_gguf_ignores_w8a8_row_for_excluded_family() -> None:
    # Base 12 GB on 6 GB free: neither resident nor 0.55-cast fits. The only
    # non-gguf sibling is a FITTING w8a8 row — AUTO-ineligible for sdxl, so
    # the GGUF pick must proceed.
    rows = _rows(("fp8-w8a8", 4_722_020_263), ("gguf-q4_k_m", 3_000_000_000))
    pick = gguf_local.select_gguf(
        _resolved(rows, "sdxl", size_bytes=12_000_000_000), gpu_sm=86, free_vram_gb=6.0,
    )
    assert pick is not None and pick.flavor == "gguf-q4_k_m"
    # Same rows without the family classification: w8a8 counts as a fitting
    # native rung and suppresses the pick (pre-th#964 behavior preserved).
    assert gguf_local.select_gguf(
        _resolved(rows, "", size_bytes=12_000_000_000), gpu_sm=86, free_vram_gb=6.0,
    ) is None
