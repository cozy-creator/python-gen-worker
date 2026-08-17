"""pgw#1313 — ONE requirement vocabulary, at TWO levels, at THREE scopes.

Charter: `research/machine-compatibility-design.md` (Paul, 2026-08-17).
*"Requirements are minimum supported and recommended… the endpoint author
already specifies the (GPU, lane) rungs, meaning the author is the one
benchmarking and figuring this out; leave it to them to configure."*

The properties this file fences, each because absence of it has cost us
something measurable before:

  1. an unknown term is refused BY NAME, per term and per level — an ignored
     requirement is one that silently does not hold;
  2. `recommended` below `minimum` is a contradiction, refused at declaration;
  3. the COMPACT form is the MINIMUM, and the fleet's existing declarations
     emit a byte-identical manifest row — repinning this wheel cannot silently
     drop a floor the hub is already enforcing (th#2030's ingest reads
     `min_sm` flat, and th#2072 is what grows the reader for `recommended`);
  4. the FUNCTION scope exists at all, because training-endpoints has ZERO
     `Slot(...)` model slots (te#209) and its four endpoints cannot express a
     floor any other way;
  5. every string the fleet declares TODAY, and every string ie#740/te#224
     will write, parses — probed as the literal text, not as a synthetic.
"""

from __future__ import annotations

import itertools
import textwrap
from pathlib import Path

import pytest

from gen_worker import LayoutRequirements, RequirementTerms, Resources
from gen_worker.models.tensor_layout_contract import (
    KNOWN_REQUIREMENT_TERMS,
    LayoutDeclarationError,
    REQUIREMENT_LEVELS,
    parse_layout_requirements,
    parse_requirement_terms,
)


def _requirement(resources: Resources) -> LayoutRequirements:
    """The parsed function-scope requirement, asserted present."""
    parsed = resources.requirement()
    assert parsed is not None
    return parsed


# ---------------------------------------------------------------------------
# 1. the term bag
# ---------------------------------------------------------------------------


def test_the_vocabulary_is_the_five_ruled_terms() -> None:
    assert KNOWN_REQUIREMENT_TERMS == (
        "min_sm", "min_vram_gb", "min_host_ram_gb", "min_cuda", "min_torch")
    assert REQUIREMENT_LEVELS == ("minimum", "recommended")


@pytest.mark.parametrize("compact,expected", [
    ("sm100+", {"min_sm": 100}),
    ("vram80g", {"min_vram_gb": 80.0}),
    ("cuda12.8+", {"min_cuda": "12.8"}),
    ("torch2.9+", {"min_torch": "2.9"}),
    ("sm100+, vram80g, cuda12.8+, torch2.9+", {
        "min_sm": 100, "min_vram_gb": 80.0,
        "min_cuda": "12.8", "min_torch": "2.9"}),
])
def test_every_term_has_a_compact_spelling_that_round_trips(
    compact: str, expected: dict,
) -> None:
    terms = parse_requirement_terms(compact, where="t")
    assert terms.declared_terms() == expected
    assert parse_requirement_terms(terms.render(), where="t") == terms
    # ...and the structured spelling is the SAME declaration.
    assert parse_requirement_terms(expected, where="t") == terms
    assert parse_requirement_terms(RequirementTerms(**expected), where="t") == terms


@pytest.mark.parametrize("bad", [
    "kernels", "kernels=sa2", "flash3", "sm100", "sm_100+", "sm89", "8.9",
    "vram80", "vram80gb", "ram64", "cuda12.8", "torch>=2.9", "torch2.9",
    "", "  ", "sm100+,", ",sm100+", 100, 8.9, None, ["sm100+"],
])
def test_an_unknown_or_unbuilt_term_is_refused_by_name(bad: object) -> None:
    """`kernels` is the term the ruling NAMES and this issue deliberately does
    not BUILD: there is no runtime kernel-capability probe in this worker, so
    it would be a floor with no fact behind it."""
    with pytest.raises(LayoutDeclarationError):
        parse_layout_requirements(bad, where="t")


@pytest.mark.parametrize("bad,offender", [
    ("sm100+, kernels", "kernels"),
    ("sm100+, flash3", "flash3"),
    ("vram80g, disk200g", "disk200g"),
    ("sm100+, torch>=2.9", "torch>=2.9"),
])
def test_an_unknown_term_BESIDE_a_known_one_is_refused_by_name(
    bad: str, offender: str,
) -> None:
    """The instrument that cannot pass vacuously: a declaration whose OTHER
    terms are valid cannot fall through to the empty-declaration refusal, so
    this is the arm that proves the unknown term itself is what refused, and
    that the message names it."""
    with pytest.raises(LayoutDeclarationError) as excinfo:
        parse_layout_requirements(bad, where="t")
    assert offender in str(excinfo.value)


def test_the_unbuilt_kernels_term_says_why_when_named_structurally() -> None:
    with pytest.raises(LayoutDeclarationError) as excinfo:
        parse_layout_requirements({"kernels": ["sa2"]}, where="t")
    message = str(excinfo.value)
    assert "kernels" in message and "not" in message and "built" in message


@pytest.mark.parametrize("bad", [
    "sm89+, sm100+", "vram24g, vram80g", "cuda12.8+, cuda13.0+",
])
def test_a_term_declared_twice_is_refused(bad: str) -> None:
    with pytest.raises(LayoutDeclarationError, match="twice"):
        parse_layout_requirements(bad, where="t")


@pytest.mark.parametrize("bad", [
    {}, LayoutRequirements(), RequirementTerms(),
    LayoutRequirements(minimum={}), {"minimum": {}, "recommended": {}},
])
def test_an_empty_declaration_is_refused_never_defaulted(bad: object) -> None:
    """A requirement that requires nothing is not a declaration. `0`/absent is
    the axis NOBODY ANSWERED, never 'runs anywhere'."""
    with pytest.raises(LayoutDeclarationError):
        parse_layout_requirements(bad, where="t")


# ---------------------------------------------------------------------------
# 2. the two levels
# ---------------------------------------------------------------------------


def test_the_compact_form_is_the_minimum() -> None:
    compact = parse_layout_requirements("sm100+, vram24g", where="t")
    assert compact.min_terms().declared_terms() == {
        "min_sm": 100, "min_vram_gb": 24.0}
    assert not compact.recommended_terms().declared()
    assert compact == parse_layout_requirements(
        LayoutRequirements(minimum="sm100+, vram24g"), where="t")


def test_recommended_is_additive_and_does_not_touch_the_minimum() -> None:
    pair = parse_layout_requirements(LayoutRequirements(
        minimum="sm80+, vram48g", recommended="sm90+, vram80g"), where="t")
    assert pair.min_terms().declared_terms() == {
        "min_sm": 80, "min_vram_gb": 48.0}
    assert pair.recommended_terms().declared_terms() == {
        "min_sm": 90, "min_vram_gb": 80.0}
    # A level may be declared alone: a recommendation with no floor is a
    # legitimate statement ("this is what it wants"), and a floor with no
    # recommendation is the fleet's current shape.
    assert parse_layout_requirements(
        LayoutRequirements(recommended="sm90+"), where="t"
    ).min_terms().declared() is False


@pytest.mark.parametrize("minimum,recommended", [
    ("sm90+", "sm80+"),
    ("vram80g", "vram48g"),
    ("cuda12.8+", "cuda12.4+"),
    ("torch2.9+", "torch2.8+"),
    ("sm90+, vram80g", "sm100+, vram48g"),
])
def test_a_recommendation_below_the_floor_is_refused(
    minimum: str, recommended: str,
) -> None:
    """A recommendation below the floor is a contradiction, not a preference —
    and it is caught per TERM, so one good term cannot mask a bad one."""
    with pytest.raises(LayoutDeclarationError, match="below minimum"):
        parse_layout_requirements(
            LayoutRequirements(minimum=minimum, recommended=recommended),
            where="t")


def test_version_terms_compare_component_wise_not_lexically() -> None:
    """`"2.9" < "2.13"` lexically and `2.9 < 2.13` numerically point OPPOSITE
    ways; a dotted version is neither a string nor a float."""
    ok = parse_layout_requirements(
        LayoutRequirements(minimum="torch2.9+", recommended="torch2.13+"),
        where="t")
    assert ok.recommended_terms().min_torch == "2.13"
    with pytest.raises(LayoutDeclarationError, match="below minimum"):
        parse_layout_requirements(
            LayoutRequirements(minimum="torch2.13+", recommended="torch2.9+"),
            where="t")


def test_the_levels_and_the_terms_never_share_one_mapping() -> None:
    with pytest.raises(LayoutDeclarationError, match="minimum"):
        parse_layout_requirements(
            {"minimum": "sm80+", "min_sm": 90}, where="t")


# ---------------------------------------------------------------------------
# 3. host RAM: recommended-only
# ---------------------------------------------------------------------------


def test_host_ram_is_refused_as_a_minimum_at_every_scope() -> None:
    """Paul, 2026-07-11: RunPod GPU pods cannot select or guarantee host RAM,
    so a declared minimum was unenforceable theater and the standing
    instruction is not to rebuild a boot-time RAM gate. The refusal names the
    move rather than dropping the term."""
    for declaration in ("ram64g", "sm89+, ram64g",
                        {"min_host_ram_gb": 64},
                        LayoutRequirements(minimum="ram64g")):
        with pytest.raises(LayoutDeclarationError, match="RECOMMENDED only"):
            parse_layout_requirements(declaration, where="t")
    rec = parse_layout_requirements(
        LayoutRequirements(recommended="ram64g"), where="t")
    assert rec.recommended_terms().min_host_ram_gb == 64.0


# ---------------------------------------------------------------------------
# 4. the manifest row — declared axes only, per term AND per level
# ---------------------------------------------------------------------------


def test_the_fleets_existing_declarations_emit_a_byte_identical_row() -> None:
    """THE ordering property. th#2030's ingest reads `min_sm` flat from this
    row TODAY. If the two levels re-shaped it, every one of the fleet's 18
    declarations would silently lose its floor the moment an endpoint repinned
    this wheel — the exact defect class this program exists to end."""
    for compact, row in (("sm89+", {"min_sm": 89}),
                         ("sm100+", {"min_sm": 100})):
        assert parse_layout_requirements(
            compact, where="t").manifest_row() == row


def test_only_declared_axes_and_declared_levels_reach_the_manifest() -> None:
    pair = parse_layout_requirements(LayoutRequirements(
        minimum="sm80+", recommended="sm90+, vram80g, ram64g"), where="t")
    assert pair.manifest_row() == {
        "min_sm": 80,
        "recommended": {
            "min_sm": 90, "min_vram_gb": 80.0, "min_host_ram_gb": 64.0},
    }
    # No zeros, no empty strings, no `recommended: {}` — an undeclared axis
    # must not arrive at the hub as a value a reader can mistake for a floor.
    assert parse_layout_requirements("sm80+", where="t").manifest_row() == {
        "min_sm": 80}


# ---------------------------------------------------------------------------
# 5. the FUNCTION scope, through the real discovery manifest
# ---------------------------------------------------------------------------


_NEXT_PKG = itertools.count()


def _function_manifest(
    tmp_path: Path, resources: str, monkeypatch: pytest.MonkeyPatch,
) -> dict:
    from gen_worker.discovery.discover import discover_manifest

    pkg = f"ep1313_{next(_NEXT_PKG)}"
    (tmp_path / "pyproject.toml").write_text(textwrap.dedent(f"""
        [project]
        name = "{pkg}"

        [tool.gen_worker]
        main = "{pkg}.main"
    """))
    src = tmp_path / pkg
    src.mkdir()
    (src / "__init__.py").write_text("")
    (src / "main.py").write_text(textwrap.dedent(f"""
        import msgspec
        from gen_worker import (
            LayoutRequirements, RequestContext, Resources, endpoint)

        class In_(msgspec.Struct):
            prompt: str = ""

        class Out_(msgspec.Struct):
            y: str = ""

        @endpoint(kind="training", resources={resources})
        class Trainer:
            def train(self, ctx: RequestContext, data: In_) -> Out_:
                return Out_()
    """))
    monkeypatch.syspath_prepend(str(tmp_path))
    return discover_manifest(tmp_path)


def test_the_function_scope_round_trips_through_a_real_discover_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """te#209's named hole: training-endpoints has ZERO `Slot(...)` model
    slots, so its four `vram_gb = 80` endpoints have nothing to hang a
    per-(slot, handle) requirement on. This is the scope that carries them."""
    manifest = _function_manifest(
        tmp_path,
        'Resources(gpu=True, vcpus=16, requires=LayoutRequirements('
        'minimum="sm80+, vram80g", recommended="sm90+, ram96g"))',
        monkeypatch)
    resources = manifest["functions"][0]["resources"]
    assert resources["requires"] == {
        "min_sm": 80, "min_vram_gb": 80.0,
        "recommended": {"min_sm": 90, "min_host_ram_gb": 96.0},
    }
    # The floor travels under ONE key. The `compute_capability` back-projection
    # is deleted: th#2072 landed, the hub prefers `requires` wherever present,
    # and this projection was only ever emitted when `min_sm` was declared —
    # i.e. only when `requires` had already answered.
    assert "compute_capability" not in resources
    assert "vram_gb" not in resources and "min_vram_gb" not in resources
    assert "ram_gb" not in resources


def test_manifest_dict_and_the_declaration_agree() -> None:
    declared = Resources(requires="sm100+, vram24g", vcpus=8)
    assert declared.manifest_dict() == {
        "gpu": True,
        "vcpus": 8,
        "requires": {"min_sm": 100, "min_vram_gb": 24.0},
    }
    assert _requirement(declared).manifest_row() == declared.manifest_dict()[
        "requires"]


# ---------------------------------------------------------------------------
# 6. the fleet, probed as literal text
# ---------------------------------------------------------------------------

#: Every requirement string DECLARED in the fleet today (ie, 2026-08-17):
#: 15 x fp8-rowwise on sm89 plus qwen-image-svdq-bench's three.
_FLEET_TODAY = ("sm89+", "sm100+")

#: The successor spellings ie#740 / te#224 will write. te's four endpoints
#: carry `vram_gb = 80` in endpoint.toml with no slot to hang it on; ie's
#: `ram_gb_hint` values are 64.0 (ltx-video-2.3, wan-2.2 a14b), 32.0 (wan-2.2
#: ti2v) and 96.0 (minimax-h3), which become RECOMMENDATIONS.
_FLEET_SUCCESSORS = (
    "vram80g",
    "sm89+, vram80g",
    "sm100+, vram80g",
)
_FLEET_RECOMMENDATIONS = ("ram64g", "ram32g", "ram96g", "sm90+, ram96g")


@pytest.mark.parametrize("declaration", _FLEET_TODAY + _FLEET_SUCCESSORS)
def test_every_fleet_minimum_parses(declaration: str) -> None:
    assert parse_layout_requirements(declaration, where="t").declared()
    # ...and at the function scope, which is where te#224 writes them.
    assert Resources(requires=declaration).requirement() is not None


@pytest.mark.parametrize("declaration", _FLEET_RECOMMENDATIONS)
def test_every_fleet_recommendation_parses(declaration: str) -> None:
    parsed = Resources(requires=LayoutRequirements(recommended=declaration))
    assert _requirement(parsed).recommended_terms().declared()
