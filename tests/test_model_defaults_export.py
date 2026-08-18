"""``gen-worker models export`` — the mechanical ``{names, schemas}`` artifact.

Acceptance: the document carries all five family names + the launch LoRA
overlay names; every schema is a valid draft-2020-12 document; platform
values and real hub rows round-trip through jsonschema validation; and the
validator is never stricter than the decoder (partial rows and unknown
fields pass). Exercised through the real CLI dispatcher, not the function
alone.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jsonschema
import msgspec
import pytest

from gen_worker.cli import main as cli_main
from gen_worker.models import defaults_vocabularies, export_document


def _validator(schema: dict[str, object]) -> jsonschema.Draft202012Validator:
    jsonschema.Draft202012Validator.check_schema(schema)
    return jsonschema.Draft202012Validator(schema)


def test_the_document_names_the_launch_set() -> None:
    doc = export_document()
    assert doc["names"] == [
        "sdxl", "sd15", "sd2", "hidream-o1", "wan22", "sdxl.lora", "sd15.lora",
    ]
    schemas = doc["schemas"]
    assert isinstance(schemas, dict)
    assert set(schemas) == set(doc["names"])  # type: ignore[arg-type]


def test_every_schema_is_valid_2020_12_and_round_trips_platform_values() -> None:
    doc = export_document()
    schemas = doc["schemas"]
    assert isinstance(schemas, dict)
    for name, cls in defaults_vocabularies().items():
        schema = schemas[name]
        assert isinstance(schema, dict)
        assert schema["title"] == name
        validator = _validator(schema)
        # The zero-arg platform opinion validates against its own schema
        # (through JSON, the shape a hub row actually takes).
        full_row = msgspec.json.decode(msgspec.json.encode(cls()))
        validator.validate(full_row)
        # The hub's JSONB is PARTIAL: the empty object must validate too.
        validator.validate({})


def test_the_validator_is_never_stricter_than_the_decoder() -> None:
    schemas = export_document()["schemas"]
    assert isinstance(schemas, dict)
    sdxl = schemas["sdxl"]
    assert isinstance(sdxl, dict)
    validator = _validator(sdxl)
    # Partial knob objects and unknown (newer-schema) fields both pass —
    # the evolution rule says the decoder ignores/fills them.
    validator.validate({"steps": {"default": 8}, "cfg": False})
    validator.validate({"a_future_field": 123})


def test_an_ill_typed_row_fails_schema_validation() -> None:
    schemas = export_document()["schemas"]
    assert isinstance(schemas, dict)
    sdxl = schemas["sdxl"]
    assert isinstance(sdxl, dict)
    validator = _validator(sdxl)
    with pytest.raises(jsonschema.ValidationError):
        validator.validate({"cfg": "yes"})
    with pytest.raises(jsonschema.ValidationError):
        validator.validate({"steps": {"default": "fast"}})


def test_the_document_is_json_stable(tmp_path: Path) -> None:
    """Same structs in, same bytes out — the artifact is mechanical."""
    a = json.dumps(export_document(), sort_keys=True)
    b = json.dumps(export_document(), sort_keys=True)
    assert a == b
    # And JSON-safe end to end: no Python-native residue survives dumps/loads.
    assert json.loads(a) == export_document()


def test_the_cli_emits_the_document(
    tmp_path: Path, capsys: "pytest.CaptureFixture[str]"
) -> None:
    assert cli_main(["models", "export"]) == 0
    out = capsys.readouterr().out
    assert json.loads(out) == export_document()

    dest = tmp_path / "nested" / "defaults.json"
    assert cli_main(["models", "export", "--out", str(dest)]) == 0
    written: Any = json.loads(dest.read_text())
    assert written == export_document()


def test_the_cli_classifies_contract_stamps(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert cli_main(["models", "classify", "sdxl.clip-g-fused-qkv@1"]) == 0
    assert capsys.readouterr().out.strip() == "sdxl"
    # Unrecognized = unclassified, visibly — the row stays NULL.
    assert cli_main(["models", "classify", "minimax.h3-dit-native@1"]) == 1
    assert capsys.readouterr().out.strip() == "unclassified"


def test_the_cli_decodes_a_row_with_the_worker_verdict(
    capsys: pytest.CaptureFixture[str],
) -> None:
    row = '{"steps": {"default": 8}, "cfg": false}'
    assert cli_main(["models", "decode", "sdxl", "--row", row]) == 0
    decoded: Any = json.loads(capsys.readouterr().out)
    assert decoded["steps"]["default"] == 8
    assert decoded["cfg"] is False
    assert decoded["positive_preamble"] == "masterpiece, best quality"

    # The LoRA overlay decodes through the same surface.
    assert cli_main(["models", "decode", "sdxl.lora", "--row", '{"sampler": "lcm"}']) == 0
    assert json.loads(capsys.readouterr().out)["sampler"] == "lcm"

    # A typed refusal names the field and exits 1.
    assert cli_main(["models", "decode", "sdxl", "--row", '{"steps": "fast"}']) == 1
    assert "steps" in capsys.readouterr().err

    # An unrecognized name is a usage error naming the recognized set.
    assert cli_main(["models", "decode", "flux"]) == 2
    assert "sdxl.lora" in capsys.readouterr().err
