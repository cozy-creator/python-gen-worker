"""Typed-provider Repo subclass tests — HFRepo, CivitaiRepo.

Validates the provider-typed binding surface promised by `progress.json`
issue #10 (typed-provider-repo-classes):

- ``HFRepo`` / ``CivitaiRepo`` construction (happy path + invalid input).
- Immutability of modifiers: each modifier returns a NEW instance,
  the original is unchanged.
- ``provider`` property returns the right value per subclass.
- Wire serialization (``_binding_to_wire`` / ``_wire_ref``) round-trips
  the provider field and emits the expected ref shape:
  * tensorhub (``cozy``) refs go on the wire bare.
  * HF / civitai refs get a ``<provider>:`` prefix for legacy
    resolvers, alongside the explicit ``provider`` field.
- ``isinstance(HFRepo(...), Repo)`` is True — subclass relationship
  preserved so existing ``isinstance(b, Repo)`` checks still match.
- Defensive: ``HFRepo("noslash")`` raises (HF refs require ``owner/repo``).
- Defensive: unknown kwargs to the dataclass constructor raise.
"""

from __future__ import annotations

import pytest

from gen_worker import CivitaiRepo, HFRepo, Repo
from gen_worker.worker import _binding_to_wire, _wire_ref


# ---------------------------------------------------------------------------
# Provider property — one source of truth per subclass.
# ---------------------------------------------------------------------------


def test_repo_provider_is_cozy() -> None:
    assert Repo("acme/myrepo").provider == "cozy"


def test_hfrepo_provider_is_hf() -> None:
    assert HFRepo("Qwen/Qwen2.5-1.5B-Instruct").provider == "hf"


def test_civitairepo_provider_is_civitai() -> None:
    assert CivitaiRepo("123456").provider == "civitai"


# ---------------------------------------------------------------------------
# Subclass relationship — existing ``isinstance(b, Repo)`` checks must hold.
# ---------------------------------------------------------------------------


def test_hfrepo_is_repo_subclass() -> None:
    r = HFRepo("Qwen/Qwen2.5-1.5B-Instruct")
    assert isinstance(r, Repo)
    assert isinstance(r, HFRepo)


def test_civitairepo_is_repo_subclass() -> None:
    r = CivitaiRepo("123456")
    assert isinstance(r, Repo)
    assert isinstance(r, CivitaiRepo)


# ---------------------------------------------------------------------------
# Construction — happy path + invalid input.
# ---------------------------------------------------------------------------


def test_hfrepo_happy_path() -> None:
    r = HFRepo("Qwen/Qwen2.5-1.5B-Instruct")
    assert r.ref == "Qwen/Qwen2.5-1.5B-Instruct"
    assert r._flavor == ""
    assert r._tag == "prod"
    assert r._revision == ""


def test_hfrepo_strips_whitespace() -> None:
    assert HFRepo("  Qwen/Qwen2.5-1.5B-Instruct  ").ref == "Qwen/Qwen2.5-1.5B-Instruct"


def test_hfrepo_requires_slash_in_ref() -> None:
    with pytest.raises(ValueError, match="owner/repo"):
        HFRepo("noslash")


def test_hfrepo_rejects_empty_ref() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        HFRepo("")


def test_hfrepo_rejects_whitespace_only_ref() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        HFRepo("   ")


def test_civitairepo_happy_path() -> None:
    r = CivitaiRepo("123456")
    assert r.ref == "123456"
    assert r._version_id == ""


def test_civitairepo_rejects_empty_ref() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        CivitaiRepo("")


def test_constructor_rejects_unknown_kwargs() -> None:
    """Defensive: a typo in a kwarg should fail loud, not silently no-op."""
    with pytest.raises(TypeError):
        HFRepo("Qwen/Q", flavor="nf4")  # type: ignore[call-arg]  # use .flavor(), not kwarg
    with pytest.raises(TypeError):
        CivitaiRepo("123", versionn="2")  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        Repo("acme/r", revision="abc")  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# Modifier immutability — every modifier returns a new instance.
# ---------------------------------------------------------------------------


def test_hfrepo_flavor_is_immutable() -> None:
    r1 = HFRepo("Qwen/Q")
    r2 = r1.flavor("nf4")
    assert r1 is not r2
    assert r1._flavor == ""
    assert r2._flavor == "nf4"
    # Provider survives the modifier.
    assert r2.provider == "hf"
    assert isinstance(r2, HFRepo)


def test_hfrepo_tag_is_immutable() -> None:
    r1 = HFRepo("Qwen/Q")
    r2 = r1.tag("canary")
    assert r1 is not r2
    assert r1._tag == "prod"
    assert r2._tag == "canary"
    assert isinstance(r2, HFRepo)
    assert r2.provider == "hf"


def test_hfrepo_allow_override_is_immutable() -> None:
    class _Pipe:
        pass

    r1 = HFRepo("Qwen/Q")
    r2 = r1.allow_override(_Pipe)
    assert r1 is not r2
    assert r1._allow_override is False
    assert r2._allow_override is True
    assert r2._pipeline_classes == ("test_api_binding.test_hfrepo_allow_override_is_immutable.<locals>._Pipe",)
    assert isinstance(r2, HFRepo)


def test_hfrepo_revision_is_immutable() -> None:
    r1 = HFRepo("Qwen/Q")
    r2 = r1.revision("a1b2c3d")
    assert r1 is not r2
    assert r1._revision == ""
    assert r2._revision == "a1b2c3d"
    assert isinstance(r2, HFRepo)


def test_hfrepo_revision_rejects_empty() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        HFRepo("Qwen/Q").revision("")


def test_civitairepo_version_is_immutable() -> None:
    r1 = CivitaiRepo("123456")
    r2 = r1.version("789012")
    assert r1 is not r2
    assert r1._version_id == ""
    assert r2._version_id == "789012"
    assert isinstance(r2, CivitaiRepo)
    assert r2.provider == "civitai"


def test_civitairepo_version_rejects_empty() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        CivitaiRepo("123").version("")


def test_civitairepo_modifiers_preserve_subclass() -> None:
    r = CivitaiRepo("123456").flavor("fp16").tag("canary").version("789")
    assert isinstance(r, CivitaiRepo)
    assert r.provider == "civitai"
    assert r._flavor == "fp16"
    assert r._tag == "canary"
    assert r._version_id == "789"


# ---------------------------------------------------------------------------
# Wire serialization — `_wire_ref` + `_binding_to_wire` round-trip.
# ---------------------------------------------------------------------------


def test_wire_ref_cozy_is_bare() -> None:
    """Tensorhub refs go on the wire without a prefix."""
    assert _wire_ref(Repo("acme/myrepo")) == "acme/myrepo"


def test_wire_ref_hf_has_prefix() -> None:
    """HF refs get an `hf:` prefix for legacy resolver compatibility."""
    assert _wire_ref(HFRepo("Qwen/Qwen2.5-1.5B-Instruct")) == "hf:Qwen/Qwen2.5-1.5B-Instruct"


def test_wire_ref_civitai_has_prefix() -> None:
    """Civitai refs get a `civitai:` prefix for legacy resolver compatibility."""
    assert _wire_ref(CivitaiRepo("123456")) == "civitai:123456"


def test_binding_to_wire_cozy_fixed() -> None:
    """Fixed tensorhub binding round-trips: bare ref + provider=cozy."""
    out = _binding_to_wire("pipeline", str, Repo("acme/myrepo").flavor("nf4"))
    assert out["param"] == "pipeline"
    binding = out["binding"]
    assert binding["kind"] == "fixed"
    assert binding["ref"] == "acme/myrepo"
    assert binding["provider"] == "cozy"
    assert binding["flavor"] == "nf4"
    assert binding["tag"] == "prod"
    assert binding["allow_override"] is False
    assert binding["pipeline_classes"] == []


def test_binding_to_wire_hf_fixed() -> None:
    """Fixed HF binding round-trips: prefixed ref + provider=hf."""
    out = _binding_to_wire("pipeline", str, HFRepo("Qwen/Qwen2.5-1.5B-Instruct"))
    binding = out["binding"]
    assert binding["kind"] == "fixed"
    assert binding["ref"] == "hf:Qwen/Qwen2.5-1.5B-Instruct"
    assert binding["provider"] == "hf"


def test_binding_to_wire_civitai_fixed() -> None:
    """Fixed civitai binding round-trips: prefixed ref + provider=civitai."""
    out = _binding_to_wire("pipeline", str, CivitaiRepo("123456"))
    binding = out["binding"]
    assert binding["kind"] == "fixed"
    assert binding["ref"] == "civitai:123456"
    assert binding["provider"] == "civitai"


def test_binding_to_wire_dispatch_mixed_providers() -> None:
    """Dispatch table with mixed providers: each entry stamps its own provider."""
    from gen_worker import dispatch

    d = dispatch(
        field="variant",
        table={
            "cozy": Repo("acme/r").flavor("nf4"),
            "hf": HFRepo("Qwen/Q").flavor("bf16"),
            "civitai": CivitaiRepo("123").flavor("fp16"),
        },
    )
    out = _binding_to_wire("pipeline", str, d)
    binding = out["binding"]
    assert binding["kind"] == "dispatch"
    assert binding["field"] == "variant"
    table = binding["table"]
    assert table["cozy"]["ref"] == "acme/r"
    assert table["cozy"]["provider"] == "cozy"
    assert table["hf"]["ref"] == "hf:Qwen/Q"
    assert table["hf"]["provider"] == "hf"
    assert table["civitai"]["ref"] == "civitai:123"
    assert table["civitai"]["provider"] == "civitai"
