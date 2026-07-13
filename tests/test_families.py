"""gen_worker.families — the pgw#520 per-family inference-defaults
vocabulary: registration, JSON-schema export (golden file), and the
`gen-worker families export-schemas` CLI hook."""

from __future__ import annotations

import json
from pathlib import Path

import msgspec
import pytest

from gen_worker.families import (
    FamilyDefaults,
    SdxlDefaults,
    export_all_schemas,
    export_json_schema,
    family,
    family_for,
    family_registry,
)

TESTDATA = Path(__file__).parent / "testdata"


def test_sdxl_registers_and_round_trips() -> None:
    assert family_for("sdxl") is SdxlDefaults
    d = SdxlDefaults(steps=30, guidance=7.0)
    assert d.family == "sdxl"
    assert d.schema_version == 1
    # frozen: no post-construction mutation.
    with pytest.raises(AttributeError):
        d.steps = 1  # type: ignore[misc]


def test_family_registry_contains_shipped_families() -> None:
    reg = family_registry()
    assert reg["sdxl"] is SdxlDefaults


def test_positional_construction_works_kw_only_does_not_leak_to_subclass(
) -> None:
    """pgw#524 item 2: FamilyDefaults's own `schema_version` field is
    kw_only=True, but msgspec's kw_only only affects fields declared on the
    class where it's set — it does NOT propagate to a subclass's own
    fields. A copy-pasted positional preset row (declaration order) must
    keep working, not TypeError."""
    d = SdxlDefaults("euler_a", 28, 6.0)
    assert d.scheduler == "euler_a"
    assert d.steps == 28
    assert d.guidance == 6.0
    assert d.schema_version == 1  # base's kw_only field: unaffected default


def test_duplicate_family_name_raises() -> None:
    with pytest.raises(ValueError, match="already registered"):
        @family("sdxl")
        class Dupe(FamilyDefaults, frozen=True):
            pass


def test_family_decorator_rejects_non_family_defaults_subclass() -> None:
    with pytest.raises(TypeError, match="FamilyDefaults"):
        @family("not-a-family")
        class NotAFamily:
            pass


def test_unregistered_family_lookup_returns_none() -> None:
    assert family_for("does-not-exist") is None
    with pytest.raises(KeyError):
        export_json_schema("does-not-exist")


def test_family_forbids_unknown_fields() -> None:
    with pytest.raises(msgspec.ValidationError):
        msgspec.json.decode(b'{"unknown_field": 1}', type=SdxlDefaults)


def test_sdxl_schema_export_matches_golden_file() -> None:
    """Golden-file test: the exported schema is a stable, reviewable
    contract — a diff here is a deliberate vocabulary change, not an
    accident. Regenerate with:
        uv run python -c "from gen_worker.families import export_json_schema; \\
            import json; print(json.dumps(export_json_schema('sdxl'), indent=2, sort_keys=True))"
    """
    golden = json.loads((TESTDATA / "sdxl.schema.json").read_text())
    assert export_json_schema("sdxl") == golden


def test_sdxl_schema_is_closed_draft_2020_12() -> None:
    schema = export_json_schema("sdxl")
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["additionalProperties"] is False
    assert schema["type"] == "object"


def test_export_all_schemas_keys_by_family_name() -> None:
    schemas = export_all_schemas()
    assert "sdxl" in schemas
    assert schemas["sdxl"]["title"] == "SdxlDefaults"


def test_max_guidance_is_optional_clamp_field() -> None:
    """max_guidance is a constraint the repo may clamp — never a required
    field (pgw#520: repo metadata may clamp within the contract, never
    reshape it)."""
    assert "max_guidance" in SdxlDefaults.__struct_fields__
    assert SdxlDefaults().max_guidance is None


def test_cli_export_schemas_writes_one_file_per_family(tmp_path: Path) -> None:
    from gen_worker.cli import main

    out_dir = tmp_path / "schemas"
    rc = main(["families", "export-schemas", str(out_dir)])
    assert rc == 0
    written = sorted(p.name for p in out_dir.glob("*.schema.json"))
    assert "sdxl.schema.json" in written
    body = json.loads((out_dir / "sdxl.schema.json").read_text())
    assert body["title"] == "SdxlDefaults"
