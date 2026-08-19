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


#: THE PINNED LAUNCH SET. Hoisted out of the test body (pgw#1432) so a FENCE can
#: read it: it is the THIRD registration site a new ModelType owes, beside the
#: `MODEL_TYPES` tuple and `defaults_vocabularies()`, and it was the only one of
#: the three with nothing asserting it against the registry.
LAUNCH_SET_NAMES: list[str] = [
    "sdxl", "sd15", "sd2", "hidream-o1", "wan22", "minimax-h3", "rife",
    # pgw#1422: the two qwen3.6 LLM roots — the first NON-DIFFUSION
    # vocabularies in the export (max_tokens/temperature/top_p, no steps
    # and no guidance).
    "qwen3.6-27b-mtp", "qwen3.6-35b-a3b",
    # pgw#1393: FLUX.1 (dev/schnell/Flex.2) and FLUX.2 Klein (4b/9b).
    "flux1", "flux2-klein",
    "qwen-image", "z-image",
    # pgw#1427 (se#769 wave 3).
    "krea-2", "anima", "ernie",
    # pgw#1430 (se#769 audio lane): stable-audio covers stable-audio-open
    # AND foundation-1 (one root, two checkpoint rows); musicgen is its own
    # root — transformers, autoregressive, no scheduler.
    "stable-audio", "musicgen",
    # pgw#1420 (se#769): LTX-2 and its 2x spatial latent upsampler — TWO roots,
    # their headers sharing ZERO keys. Added here by the pgw#1432 fence below,
    # which named them on the first rebase that brought them in; the pinned
    # list had gone stale exactly as predicted, one lane later.
    "ltx-2", "ltx-2-upsampler",
    # pgw#1432 (se#769 vlm lane): InternVL-U is a unified VLM serving
    # TEXT-TO-IMAGE, so it is an ordinary diffusion vocabulary here despite
    # the name — one sourced `steps` knob and nothing else, the Rife shape.
    "internvl-u",
    # pgw#1424 (se#769 3D lane): the two 3D roots, declaring OPPOSITE lane
    # facts — trellis2 carries a real lane document (trellis2.dit-bf16@1),
    # hunyuan3d deliberately carries NONE and no MissingContract sentinel
    # either, because every core model in its repo is a pickle and no
    # safetensors-shaped document can ever describe one.
    "trellis2", "hunyuan3d",
    "sdxl.lora", "sd15.lora",
]


def test_the_document_names_the_launch_set() -> None:
    doc = export_document()
    assert doc["names"] == LAUNCH_SET_NAMES
    schemas = doc["schemas"]
    assert isinstance(schemas, dict)
    assert set(schemas) == set(doc["names"])  # type: ignore[arg-type]


def test_the_pinned_list_above_cannot_go_STALE_against_the_registry() -> None:
    """pgw#1432 — the FENCE for the list in the test above.

    A new ModelType has THREE registration sites: the ``MODEL_TYPES`` tuple,
    ``defaults_vocabularies()`` (what the export emitter actually reads), and
    the pinned ``names`` list in the test above. The first two are fenced —
    pgw#1001 asserts ``MODEL_TYPES`` ⊆ ``defaults_vocabularies()`` — and the
    third was not, so a type registered in both fenced sites still shipped a
    stale export list.

    THAT GAP IS EXPENSIVE OUT OF PROPORTION TO ITS SIZE, because of WHERE it
    fails. The pinned list lives in the full ``tests`` job, which is red on
    master for unrelated reasons; a lane that adds a ModelType, sees ``tests``
    red, and diffs the COUNT against the known baseline concludes "not mine"
    and enqueues. Only diffing failure NAMES catches it, and nothing forced
    anyone to.

    So this asserts the pinned list against the registry rather than against a
    human's memory, and it fails by NAMING the missing family instead of
    printing a list diff.
    """

    from gen_worker.models.model_types import LORA_OVERLAYS, MODEL_TYPES

    registry = {mt.name for mt in MODEL_TYPES} | {ov.name for ov in LORA_OVERLAYS}
    # Deliberately the CONSTANT, not `export_document()["names"]`. The document
    # is built from `defaults_vocabularies()`, so comparing against it would
    # only re-assert pgw#1001's fence and leave the pinned list unguarded —
    # which is the exact hole this closes.
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

    # MEMBERSHIP IS NOT ENOUGH: the sibling assertion compares LISTS, and the
    # emitted order is `defaults_vocabularies()` INSERTION order — which is not
    # the `MODEL_TYPES` tuple order and genuinely differs from it today
    # (the tuple runs ... Flux2Klein, Krea2, Anima, Ernie, QwenImage, ZImage ...
    # while the emitter runs ... flux2-klein, qwen-image, z-image, krea-2 ...).
    # So a set-equal fence stays green while the sibling stays red on a
    # reordering, which is the same "green fence, red job" split this whole
    # test exists to close. Checked here so the reordering case ALSO fails by
    # naming the position rather than printing two long lists.
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
