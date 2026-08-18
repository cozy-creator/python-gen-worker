"""pgw#772: the serving lane is deterministic per (release x declared config).

Probing LIVE free VRAM to choose a lane makes `lane` — the only GPU-dependent
axis of the ten in the compiled graph key — a function of the individual card's headroom.
An RTX 4090's ~1.5 GiB surplus over an L4 (same release, same image, both
sm_89) then flips its base lane to "", a lane nothing mints for, so the better
card misses every published checkpoint INCLUDING its own same-SKU compiled graph and
serves eager for life. The tax that probe dodges is +1.9%, for the structural
fp8 storage lane.

Red-verified with the probe restored: the high-headroom load lands base lane ""
while the low-headroom load lands "fp8-hooks", and the two requested compiled graph keys
diverge on the `lane` axis.

Standing rule (the pgw#765 `sku` guard, applied to `lane`): no adoption or
identity path may take a live device measurement as a hash input. A card
that CANNOT run the declared lane is different — the fit ladder's rungs are
declared, typed, reported transitions, and the second test pins that they
still engage.
"""

from gen_worker.models import loading



# pgw#1373: the two key-determinism cases and their `_arm` helper went with
# `fleet_compiled_graphs.arm_identity` and the v1 compiled-graph key they
# measured. The standing guard below is NOT about that key — it fences
# `models/loading.py`, which survives — so it stays.


def test_voluntary_upcast_probe_stays_removed() -> None:
    """Standing guard: the free-VRAM upcast probe must not come back. A
    reintroduction under the old names trips here by name; one under a new
    name trips the headline test by behavior."""
    assert not hasattr(loading, "bf16_resident_fits")
    assert not hasattr(loading, "BF16_RESIDENT_MARGIN_GB")
    import inspect

    src = inspect.getsource(loading.load_from_pretrained)
    assert "bf16_resident" not in src
