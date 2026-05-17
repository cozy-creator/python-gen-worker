"""Issue #20 fix 2: HFRepo carries `.dtype()` instead of leaking
tensorhub's `#flavor` convention through the HF API surface.

- ``HFRepo("acme/x").dtype("bf16")`` sets ``_dtype="bf16"`` (and NOT
  ``_flavor="bf16"`` which is tensorhub-only).
- ``HFRepo.flavor()`` is a deprecation shim for one release: maps to
  ``.dtype()`` with a ``DeprecationWarning``. Drops in 0.8.x.
- ``Repo`` (tensorhub) is unchanged: ``.flavor()`` still sets ``_flavor``.
- Wire / manifest serialization emits ``dtype`` for HF bindings.
"""

from __future__ import annotations

import warnings

import pytest

from gen_worker import HFRepo, Repo


# ---------------------------------------------------------------------------
# .dtype() — the new chainable.
# ---------------------------------------------------------------------------


def test_hfrepo_dtype_sets_dtype_field() -> None:
    r = HFRepo("acme/x").dtype("fp16")
    assert r._dtype == "fp16"
    # Critically NOT _flavor — flavor is tensorhub-only.
    assert r._flavor == ""


def test_hfrepo_dtype_is_immutable() -> None:
    r1 = HFRepo("acme/x")
    r2 = r1.dtype("bf16")
    assert r1 is not r2
    assert r1._dtype == ""
    assert r2._dtype == "bf16"
    assert isinstance(r2, HFRepo)


def test_hfrepo_dtype_rejects_empty() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        HFRepo("acme/x").dtype("")
    with pytest.raises(ValueError, match="non-empty"):
        HFRepo("acme/x").dtype("   ")


def test_hfrepo_dtype_chainable_with_revision_and_allow_override() -> None:
    class _Pipe:
        pass

    r = HFRepo("acme/x").dtype("bf16").revision("a1b2c3d").allow_override(_Pipe)
    assert r._dtype == "bf16"
    assert r._revision == "a1b2c3d"
    assert r._allow_override is True


# ---------------------------------------------------------------------------
# .flavor() — deprecation shim that maps to .dtype().
# ---------------------------------------------------------------------------


def test_hfrepo_flavor_emits_deprecation_and_maps_to_dtype() -> None:
    r1 = HFRepo("acme/x")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        r2 = r1.flavor("bf16")
    msgs = [str(w.message) for w in caught if issubclass(w.category, DeprecationWarning)]
    assert any("HFRepo.flavor()" in m for m in msgs), (
        f"expected DeprecationWarning about HFRepo.flavor(); got {msgs}"
    )
    # The shim sets _dtype, NOT _flavor.
    assert r2._dtype == "bf16"
    assert r2._flavor == ""


# ---------------------------------------------------------------------------
# Tensorhub Repo.flavor() is unchanged.
# ---------------------------------------------------------------------------


def test_tensorhub_repo_flavor_unchanged() -> None:
    """Fix 2 is HF-specific. Tensorhub Repo.flavor() must still set
    _flavor and emit no warning."""
    r1 = Repo("acme/y")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        r2 = r1.flavor("nf4")
    assert r2._flavor == "nf4"
    assert not any(
        issubclass(w.category, DeprecationWarning) for w in caught
    ), "tensorhub Repo.flavor() must not be deprecated"


# ---------------------------------------------------------------------------
# Wire format: HF bindings emit `dtype`, not `flavor`.
# ---------------------------------------------------------------------------


def test_wire_format_hf_binding_carries_dtype() -> None:
    from gen_worker.worker import _binding_to_wire

    out = _binding_to_wire("pipeline", str, HFRepo("acme/x").dtype("fp16"))
    binding = out["binding"]
    assert binding["provider"] == "hf"
    assert binding["dtype"] == "fp16"
    # Flavor field is still emitted (empty) for back-compat with the
    # consumer side, but it must NOT carry the dtype.
    assert binding["flavor"] == ""


def test_wire_format_hf_binding_no_dtype_field_when_unset() -> None:
    """When HFRepo has no dtype, the wire output omits the dtype field."""
    from gen_worker.worker import _binding_to_wire

    out = _binding_to_wire("pipeline", str, HFRepo("acme/x"))
    binding = out["binding"]
    assert "dtype" not in binding


def test_wire_format_tensorhub_binding_has_no_dtype_field() -> None:
    """Tensorhub bindings never carry a dtype field — dtype is HF-only."""
    from gen_worker.worker import _binding_to_wire

    out = _binding_to_wire("pipeline", str, Repo("acme/y").flavor("nf4"))
    binding = out["binding"]
    assert "dtype" not in binding
    assert binding["flavor"] == "nf4"
