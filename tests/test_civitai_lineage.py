"""Issue #71: civitai branch records the same structured base-model
lineage attributes the HF clone path does, derived from the civitai
`baseModel` enum + (when present) kohya `__metadata__` block.

These tests exercise the family-mapping and the attribute-synthesis
shape — the full ingest path requires a live civitai API and a real
.safetensors file with metadata, both out of scope for unit tests."""
from __future__ import annotations

import pytest

from gen_worker.conversion.base_model_families import (
    CANONICAL_FAMILIES,
    civitai_to_family,
    is_canonical_family,
    kohya_to_family,
    repo_to_family,
)


# -------------------------------------------------------------------
# civitai_to_family — the family resolver for civitai-mirrored uploads
# -------------------------------------------------------------------

@pytest.mark.parametrize(
    "civitai_value,expected_family",
    [
        ("SDXL 1.0", "sdxl"),
        ("SDXL Turbo", "sdxl-turbo"),
        ("SDXL Lightning", "sdxl-lightning"),
        ("SDXL Hyper", "sdxl-hyper"),
        ("Pony", "sdxl-pony"),
        ("SD 1.5", "sd15"),
        ("SD 1.4", "sd14"),
        ("SD 2.0", "sd2"),
        ("SD 2.1", "sd2"),
        ("Flux.1 D", "flux1-dev"),
        ("Flux.1 S", "flux1-schnell"),
        ("Illustrious", "sdxl-illustrious"),
    ],
)
def test_civitai_to_family_known_values(civitai_value: str, expected_family: str) -> None:
    """Each Civitai `baseModel` enum value maps to a canonical family
    slug. The resolver is the single source of truth for civitai
    ingestion attributing a `base_model_family` to the destination
    checkpoint."""
    got = civitai_to_family(civitai_value)
    assert got == expected_family, f"civitai_to_family({civitai_value!r}) = {got!r}, want {expected_family!r}"
    assert is_canonical_family(got), f"resolved family {got!r} not in CANONICAL_FAMILIES"


def test_civitai_to_family_unknown_returns_none() -> None:
    """Unrecognized values surface as None so the caller records
    `lineage_source=civitai_baseModel` but `base_model_family=""`. The
    caller's responsibility, not the resolver's, to fall back to
    `unknown` when needed."""
    assert civitai_to_family("UnreleasedFutureModel") is None
    assert civitai_to_family("") is None
    assert civitai_to_family(None) is None  # type: ignore[arg-type]


def test_civitai_to_family_whitespace_normalized() -> None:
    assert civitai_to_family("  SDXL 1.0  ") == "sdxl"
    assert civitai_to_family("Pony") == "sdxl-pony"


# -------------------------------------------------------------------
# Cross-resolver invariants
# -------------------------------------------------------------------

def test_canonical_families_set_is_finite_and_lowercase() -> None:
    """The family enum must be lowercase and finite — tensorhub mirrors
    the set on the server side for validation, so drift breaks
    classifier→server roundtrips."""
    assert len(CANONICAL_FAMILIES) > 0
    for slug in CANONICAL_FAMILIES:
        assert slug == slug.lower(), f"family slug {slug!r} not lowercase"
        # Slugs use kebab-case; underscores are forbidden so search-axis
        # consumers don't have to worry about underscore-vs-dash drift.
        assert "_" not in slug, f"family slug {slug!r} contains underscore"


def test_repo_to_family_known_repos() -> None:
    """A handful of canonical HF repos should resolve to expected
    families. This guards against an accidental table drop."""
    assert repo_to_family("stabilityai/stable-diffusion-xl-base-1.0") == "sdxl"
    assert repo_to_family("black-forest-labs/FLUX.1-dev") == "flux1-dev"


def test_kohya_to_family_known_versions() -> None:
    """Kohya `ss_base_model_version` strings → canonical families."""
    assert kohya_to_family("stable-diffusion-xl-1.0") == "sdxl"
    assert kohya_to_family("stable-diffusion-v1-5") == "sd15"
    assert kohya_to_family("flux1") == "flux1-dev"


def test_unknown_inputs_return_none_or_heuristic() -> None:
    """Empty inputs surface None across all three resolvers — caller
    falls back to `lineage_source=unknown`."""
    assert kohya_to_family("") is None
    assert kohya_to_family(None) is None  # type: ignore[arg-type]
    assert repo_to_family("") is None
    assert repo_to_family(None) is None  # type: ignore[arg-type]
