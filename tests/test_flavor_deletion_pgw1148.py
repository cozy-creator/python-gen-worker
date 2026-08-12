"""pgw#1148 / DESIGN-RULINGS §1.32(d): the flavor is dead as a WEIGHT ADDRESS,
and alive as the COMPILE CELL fragment.

Both halves are asserted here because the whole risk of this deletion is
mixing them up: `grep -rn flavor src/` is a four-way homonym, and the compile
cache's cell keys are `#`-shaped. The hub made exactly this split (th#1803
kept `release.ParseCanonicalRef`'s `#` tail for the cache and refuses the
selector at its request surfaces with `flavor_selection_removed` /
`binding_flavor_removed`, both 400); the SDK mirrors it client-side so a
caller is told at the site they wrote instead of by a server error.
"""

from __future__ import annotations

import pytest

from gen_worker import compile_cache as cc
from gen_worker.api.binding import HF, Hub, ModelRef, rebind_pick, wire_ref
from gen_worker.models import refs
from gen_worker.models.refs import (
    FlavorSelectorRemoved,
    HuggingFaceRef,
    parse_model_ref,
    refuse_flavor_selector,
)

CELL_REF = "root/family-sdxl:cells#ck1-4f2a9b"


# --------------------------------------------------------------------------
# The refusal
# --------------------------------------------------------------------------

@pytest.mark.parametrize("ref", [
    "owner/repo#fp8",
    "owner/repo#fp8-w8a8",
    "owner/repo:latest#bf16",
    "owner/repo#svdq-int4-r128",
    "owner/repo#gguf-q4_k_m",
])
def test_hub_binding_refuses_a_flavor_selector(ref: str) -> None:
    with pytest.raises(FlavorSelectorRemoved) as e:
        Hub(ref)
    msg = str(e.value)
    assert "REMOVED" in msg
    # The refusal must say what to write instead, or it is just a wall.
    assert "Slot(layouts=" in msg
    assert "@sha256:" in msg


def test_hf_binding_refuses_a_flavor_selector() -> None:
    with pytest.raises(FlavorSelectorRemoved):
        HF("black-forest-labs/FLUX.1-dev#bf16")


def test_raw_modelref_construction_refuses_too() -> None:
    """The factories are sugar; the struct is the boundary (pgw#511). A
    `msgspec.structs.replace` or a direct construction must not slip past."""
    with pytest.raises(FlavorSelectorRemoved):
        ModelRef(source="tensorhub", path="owner/repo#fp8")


def test_hf_parse_refuses_rather_than_silently_stripping() -> None:
    """The old parser STRIPPED the `#` tail off an HF ref and carried it as
    `HuggingFaceRef.flavor`. Silently dropping it would resolve a DIFFERENT
    repo than the caller named, so it refuses."""
    with pytest.raises(FlavorSelectorRemoved):
        parse_model_ref("owner/repo#bf16", provider="hf")


def test_a_hub_resolution_carrying_a_flavor_is_refused() -> None:
    """th#1803 makes the ladder's pick a DIGEST. A resolution that still
    carries a `#flavor` is a hub the worker must not silently obey."""
    with pytest.raises(FlavorSelectorRemoved):
        rebind_pick(Hub("owner/repo"), resolved_ref="owner/repo#fp8-w8a8")


def test_refuse_flavor_selector_is_a_valueerror() -> None:
    """Callers that already catch ValueError on ref grammar keep working."""
    assert issubclass(FlavorSelectorRemoved, ValueError)
    refuse_flavor_selector("owner/repo:prod")  # no `#`: silent
    refuse_flavor_selector("")


# --------------------------------------------------------------------------
# The deletion
# --------------------------------------------------------------------------

def test_the_flavor_axis_is_gone_from_the_binding() -> None:
    assert not hasattr(Hub("owner/repo"), "flavor")
    with pytest.raises(TypeError):
        Hub("owner/repo", flavor="fp8")  # type: ignore[call-arg]


def test_hf_refs_have_no_flavor_and_no_cache_key_fold() -> None:
    """The fold gave two "flavors" of one HF repo two residency entries."""
    assert not hasattr(HuggingFaceRef("owner/repo"), "flavor")
    assert HuggingFaceRef("owner/repo", "main").canonical() == "owner/repo@main"


def test_wire_ref_can_no_longer_mint_a_flavored_ref() -> None:
    assert wire_ref(Hub("owner/repo")) == "owner/repo"
    assert wire_ref(Hub("owner/repo", tag="latest")) == "owner/repo:latest"


def test_deleted_names_are_deleted_not_aliased() -> None:
    """§1.32(d) is DELETE, not alias (pre-launch)."""
    for name in ("flavor_token",):
        assert not hasattr(refs, name), f"refs.{name} survived the deletion"
    from gen_worker.models import gguf_local, ladder
    for mod, name in (
        (ladder, "pick_family_fp8_flavor"),
        (ladder, "maybe_rebind_family_fp8"),
        (gguf_local, "select_gguf"),
        (gguf_local, "maybe_rebind_gguf"),
        (gguf_local, "fetch_gguf_snapshot"),
    ):
        assert not hasattr(mod, name), f"{mod.__name__}.{name} survived"


def test_fold_ref_has_no_flavor_overlay() -> None:
    assert refs.fold_ref("owner/repo", tag="canary") == "owner/repo:canary"
    with pytest.raises(TypeError):
        refs.fold_ref("owner/repo", flavor="fp8")  # type: ignore[call-arg]


def test_the_hub_resolve_no_longer_sends_or_parses_a_flavor() -> None:
    """th#1803 dropped `?flavor=` AND `sibling_flavors` from the hub. Parsing
    a field the server cannot send is a wire that lies about its contract."""
    from gen_worker.models import hub_client
    assert not hasattr(hub_client, "WorkerResolvedFlavor")
    assert not hasattr(hub_client.WorkerResolvedRepo(snapshot_digest="d", files=[]),
                       "sibling_flavors")


# --------------------------------------------------------------------------
# The half that must NOT move: compile cells are `#`-shaped
# --------------------------------------------------------------------------

def test_the_cell_fragment_still_parses() -> None:
    th = parse_model_ref(CELL_REF).tensorhub
    assert th is not None
    assert (th.owner, th.repo, th.tag, th.flavor) == (
        "root", "family-sdxl", "cells", "ck1-4f2a9b")


def test_parse_cell_ref_is_unchanged() -> None:
    assert cc.parse_cell_ref(CELL_REF) == ("sdxl", "ck1-4f2a9b")
    assert cc.family_from_ref(CELL_REF) == "sdxl"


def test_a_cell_ref_round_trips_through_the_normal_form() -> None:
    assert refs.normalize_model_ref(CELL_REF) == CELL_REF


def test_the_trt_cell_predicate_still_reads_its_fragment() -> None:
    from gen_worker import trt_engine
    assert trt_engine.is_engine_ref(
        "root/family-sdxl:cells#trt-rtx-4090-trt10.16-fp16")
    assert not trt_engine.is_engine_ref("owner/repo")


def test_the_compile_cache_modules_are_byte_untouched_by_this_deletion() -> None:
    """The GPU/compile-cell homonym is NOT this ruling's subject. Asserted as
    a fact about the diff, not a promise in a comment: every module that owns
    a cell KEY path must be identical to `origin/master`."""
    import subprocess
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    fenced = [
        "src/gen_worker/compile_cache.py",
        "src/gen_worker/cell_key.py",
        "src/gen_worker/aot_mint.py",
        "src/gen_worker/aot_serve.py",
        "src/gen_worker/trt_engine.py",
        "src/gen_worker/fleet_cells.py",
        "src/gen_worker/local_cells.py",
        "src/gen_worker/mint_budget.py",
    ]
    base = subprocess.run(
        ["git", "merge-base", "HEAD", "origin/master"],
        cwd=root, capture_output=True, text=True)
    if base.returncode != 0:
        pytest.skip("no origin/master to diff against")
    changed = subprocess.run(
        ["git", "diff", "--name-only", base.stdout.strip(), "--", *fenced],
        cwd=root, capture_output=True, text=True)
    assert changed.returncode == 0
    assert changed.stdout.strip() == "", (
        "pgw#1148 must not touch a compile-cell module; changed: "
        + changed.stdout)
