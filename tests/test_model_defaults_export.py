"""``gen-worker models export`` — the mechanical ``{names, schemas}`` artifact."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import jsonschema
import msgspec
import pytest

from gen_worker.cli import main as cli_main
from gen_worker.models import defaults_vocabularies, export_document


def _validator(schema: dict[str, object]) -> jsonschema.Draft202012Validator:
    jsonschema.Draft202012Validator.check_schema(schema)
    return jsonschema.Draft202012Validator(schema)


LAUNCH_SET_NAMES: list[str] = [
    "sdxl", "sd15", "sd2", "hidream-o1", "wan22", "minimax-h3", "rife",
    "qwen3.6-27b-mtp", "qwen3.6-35b-a3b",
    "flux1", "flux2-klein",
    "qwen-image", "z-image", "sensenova-u1",
    "krea-2", "anima", "ernie",
    "stable-audio", "musicgen",
    "ltx-2", "ltx-2-upsampler",
    "internvl-u",
    "trellis2", "hunyuan3d",
    "joycaption",
    "sdxl.lora", "sd15.lora",
]


def test_the_document_names_the_launch_set() -> None:
    doc = export_document()
    assert doc["names"] == LAUNCH_SET_NAMES
    schemas = doc["schemas"]
    assert isinstance(schemas, dict)
    assert set(schemas) == set(doc["names"])  # type: ignore[arg-type]


def test_the_pinned_list_above_cannot_go_STALE_against_the_registry() -> None:

    from gen_worker.models.model_types import LORA_OVERLAYS, MODEL_TYPES

    registry = {mt.name for mt in MODEL_TYPES} | {ov.name for ov in LORA_OVERLAYS}
    pinned = set(LAUNCH_SET_NAMES)

    missing = sorted(registry - pinned)
    assert not missing, (
        f"{missing} are registered in MODEL_TYPES/LORA_OVERLAYS but absent from "
        "LAUNCH_SET_NAMES — the pinned export list is stale and the document would "
        "ship without them. Add them to LAUNCH_SET_NAMES."
    )
    stale = sorted(pinned - registry)
    assert not stale, (
        f"{stale} are pinned in LAUNCH_SET_NAMES but no longer in "
        "MODEL_TYPES/LORA_OVERLAYS — a deleted family left its name behind."
    )

    emitted = cast("list[str]", export_document()["names"])
    if emitted != LAUNCH_SET_NAMES:
        first = next(
            i for i, (a, b) in enumerate(zip(emitted, LAUNCH_SET_NAMES)) if a != b
        )
        raise AssertionError(
            f"LAUNCH_SET_NAMES is in the wrong ORDER (membership is fine): at "
            f"index {first} the export emits {emitted[first]!r} but the pinned "
            f"list has {LAUNCH_SET_NAMES[first]!r}. The order is "
            f"`defaults_vocabularies()` insertion order, NOT the MODEL_TYPES "
            f"tuple order — match the mapping, not the tuple."
        )


def test_every_schema_is_valid_2020_12_and_round_trips_platform_values() -> None:
    doc = export_document()
    schemas = doc["schemas"]
    assert isinstance(schemas, dict)
    for name, cls in defaults_vocabularies().items():
        schema = schemas[name]
        assert isinstance(schema, dict)
        assert schema["title"] == name
        validator = _validator(schema)
        full_row = msgspec.json.decode(msgspec.json.encode(cls()))
        validator.validate(full_row)
        validator.validate({})


def test_the_validator_is_never_stricter_than_the_decoder() -> None:
    schemas = export_document()["schemas"]
    assert isinstance(schemas, dict)
    sdxl = schemas["sdxl"]
    assert isinstance(sdxl, dict)
    validator = _validator(sdxl)
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
    # A bare TOPOLOGY handle classifies (pgw#1621: the fingerprints are v2
    # topology spellings now — `sdxl.clip-g-fused@1`, not the v1
    # `sdxl.clip-g-fused-qkv@1`).
    assert cli_main(["models", "classify", "sdxl.clip-g-fused@1"]) == 0
    assert capsys.readouterr().out.strip() == "sdxl"
    # ...and so does the whole v2 STAMP PAIR, because `model_type_for_contract`
    # matches the TOPOLOGY HALF only: which architecture a checkpoint is, is a
    # fact about which tensors it has and never about how they are quantized.
    # One topology, three quants, one answer.
    for quant in ("plain.bf16@1", "cozy.fp8-rowwise@1", "cozy.nvfp4-flat@1"):
        assert cli_main(["models", "classify", f"sdxl.diffusers@1+{quant}"]) == 0
        assert capsys.readouterr().out.strip() == "sdxl"
    # Both packaged H3 topologies classify to the one H3 vocabulary — its
    # fingerprint is `minimax-h3.*`, which `test_model_defaults.py` has
    # asserted since pgw#1386. This file still expected the pre-#1386
    # "unclassified" answer, so the two contradicted each other and this
    # assertion was red on master; corrected in passing (pgw#1393).
    assert cli_main(["models", "classify", "minimax-h3.native@1"]) == 0
    assert capsys.readouterr().out.strip() == "minimax-h3"
    assert cli_main(["models", "classify", "minimax-h3.diffusers@1+plain.bf16@1"]) == 0
    assert capsys.readouterr().out.strip() == "minimax-h3"
    assert cli_main(["models", "classify", "dit.blocks-fused-qkv@1"]) == 1
    assert capsys.readouterr().out.strip() == "unclassified"
    # And the RETIRED v1 handle is the other kind: v2 moved `h3-dit` out of the
    # format half and into the PRODUCER (`minimax` -> `minimax-h3`), so
    # `minimax.h3-dit-native@1` matches no fingerprint. It says so rather than
    # being coerced back — a display name names no topology (pgw#1621).
    assert cli_main(["models", "classify", "minimax.h3-dit-native@1"]) == 1
    assert capsys.readouterr().out.strip() == "unclassified"
    assert cli_main(["models", "classify", "flux1.diffusers@1"]) == 0
    assert capsys.readouterr().out.strip() == "flux1"


def test_the_cli_decodes_a_row_with_the_worker_verdict(
    capsys: pytest.CaptureFixture[str],
) -> None:
    row = '{"steps": {"default": 8}, "cfg": false}'
    assert cli_main(["models", "decode", "sdxl", "--row", row]) == 0
    decoded: Any = json.loads(capsys.readouterr().out)
    assert decoded["steps"]["default"] == 8
    assert decoded["cfg"] is False
    assert decoded["positive_preamble"] == "masterpiece, best quality"

    assert cli_main(["models", "decode", "sdxl.lora", "--row", '{"scheduler": "lcm"}']) == 0
    assert json.loads(capsys.readouterr().out)["scheduler"] == "lcm"

    assert cli_main(["models", "decode", "sdxl", "--row", '{"steps": "fast"}']) == 1
    assert "steps" in capsys.readouterr().err

    assert cli_main(["models", "decode", "flux"]) == 2
    assert "sdxl.lora" in capsys.readouterr().err
