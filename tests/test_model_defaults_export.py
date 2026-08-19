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
from typing import Any, cast

import jsonschema
import msgspec
import pytest

from gen_worker.cli import main as cli_main
from gen_worker.models import defaults_vocabularies, export_document


def _validator(schema: dict[str, object]) -> jsonschema.Draft202012Validator:
    jsonschema.Draft202012Validator.check_schema(schema)
    return jsonschema.Draft202012Validator(schema)


def test_the_document_names_exactly_the_registry() -> None:
    """pgw#1445 — DERIVED from the registry, not pinned by hand.

    This used to be a hand-maintained ``LAUNCH_SET_NAMES`` list, and the list was
    a SERIALIZATION POINT: every ModelType PR had to edit it, so two could never
    be in flight concurrently without one going stale. It went stale for four
    lanes in a row (audio, LTX-2, pgw#1432, and master itself), and it failed in
    the worst available place — the full ``tests`` job, which is red for
    unrelated reasons, so a lane that diffed the failure COUNT against the known
    baseline concluded "not mine" and enqueued. Paul's minimum-hand-specification
    ruling: derive what can be derived.

    WHAT IS ASSERTED HERE IS NOT CIRCULAR, and the distinction is the whole
    design. ``export_document()`` builds its names as
    ``list(defaults_vocabularies())`` (``defaults_export.py:58``), so comparing
    the export against that mapping would test the emitter against its own
    input and prove nothing. ``MODEL_TYPES`` and ``LORA_OVERLAYS`` are a
    DIFFERENT source — the declaration tuples — so the export is checked against
    them instead.

    WHAT IS DELIBERATELY *NOT* ASSERTED: the exact sequence within the base
    block. That order is ``defaults_vocabularies()`` insertion order, and no
    independent source defines it — the mapping's order genuinely differs from
    the ``MODEL_TYPES`` tuple's today (the tuple runs ``... Flux2Klein, Krea2,
    Anima, Ernie, QwenImage, ZImage ...`` while the emitter runs
    ``... flux2-klein, qwen-image, z-image, krea-2, anima, ernie ...``). Pinning
    it again would recreate exactly the hand-maintained list this issue deletes.
    The ORDERING PROPERTY the docstring actually promises — base types first,
    then the overlays — IS checked, because that one is structural rather than
    incidental.
    """

    from gen_worker.models.model_types import LORA_OVERLAYS, MODEL_TYPES

    base = [mt.name for mt in MODEL_TYPES]
    overlays = [ov.name for ov in LORA_OVERLAYS]
    names = cast("list[str]", export_document()["names"])

    # NON-VACUITY FIRST: set equality against an empty registry is trivially
    # true, so an import that silently produced no types would PASS every
    # assertion below. Anchors that have been in the launch set since it
    # existed, and a floor, so "the registry went empty" fails here loudly
    # instead of passing quietly.
    assert len(base) >= 7, f"the model-type registry collapsed to {base}"
    assert "sdxl" in base, "sdxl vanished from MODEL_TYPES — the registry is wrong"
    assert "sdxl.lora" in overlays, "the LoRA overlays vanished from LORA_OVERLAYS"

    registry = set(base) | set(overlays)
    emitted = set(names)

    missing = sorted(registry - emitted)
    assert not missing, (
        f"{missing} are registered in MODEL_TYPES/LORA_OVERLAYS but the export "
        "document does not name them — they would ship with NO SCHEMA. Register "
        "them in `defaults_vocabularies()`, which is what the emitter reads."
    )
    extra = sorted(emitted - registry)
    assert not extra, (
        f"{extra} are named by the export document but are in neither "
        "MODEL_TYPES nor LORA_OVERLAYS — a deleted family left its vocabulary "
        "behind in `defaults_vocabularies()`."
    )

    duplicates = sorted({n for n in names if names.count(n) > 1})
    assert not duplicates, (
        f"{duplicates} appear more than once in the export document; "
        "`defaults_vocabularies()` is a mapping, so a repeat means two families "
        "share a name."
    )

    # The ordering PROPERTY the docstring promises: "base types first, then the
    # LoRA overlays". Structural, and checkable without pinning a sequence.
    overlay_positions = [names.index(n) for n in overlays]
    base_positions = [names.index(n) for n in base]
    assert min(overlay_positions) > max(base_positions), (
        "the export interleaves LoRA overlays with base types; "
        "`export_document` documents base types first, then the overlays"
    )

    schemas = export_document()["schemas"]
    assert isinstance(schemas, dict)
    assert set(schemas) == emitted, (
        "every named family must carry a schema — "
        f"named-without-schema: {sorted(emitted - set(schemas))}, "
        f"schema-without-name: {sorted(set(schemas) - emitted)}"
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
    # Both packaged H3 layouts classify to the one H3 vocabulary — its
    # fingerprint is `minimax.h3-*`, which `test_model_defaults.py` has
    # asserted since pgw#1386. This file still expected the pre-#1386
    # "unclassified" answer, so the two contradicted each other and this
    # assertion was red on master; corrected in passing (pgw#1393).
    assert cli_main(["models", "classify", "minimax.h3-dit-native@1"]) == 0
    assert capsys.readouterr().out.strip() == "minimax-h3"
    # Unrecognized = unclassified, visibly — the row stays NULL. The SHARED
    # flux/timm block-spelling fragment is deliberately one of these: it
    # names no single family, so it classifies nothing (pgw#1393).
    assert cli_main(["models", "classify", "dit.blocks-fused-qkv@1"]) == 1
    assert capsys.readouterr().out.strip() == "unclassified"
    assert cli_main(["models", "classify", "flux1.diffusers-bf16@1"]) == 0
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

    # The LoRA overlay decodes through the same surface.
    assert cli_main(["models", "decode", "sdxl.lora", "--row", '{"scheduler": "lcm"}']) == 0
    assert json.loads(capsys.readouterr().out)["scheduler"] == "lcm"

    # A typed refusal names the field and exits 1.
    assert cli_main(["models", "decode", "sdxl", "--row", '{"steps": "fast"}']) == 1
    assert "steps" in capsys.readouterr().err

    # An unrecognized name is a usage error naming the recognized set.
    assert cli_main(["models", "decode", "flux"]) == 2
    assert "sdxl.lora" in capsys.readouterr().err
