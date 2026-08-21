from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, cast

import msgspec

from ..models.defaults_decode import (
    CarriesDefaults,
    DefaultsDecodeError,
    decode_model_defaults,
)
from ..models.defaults_export import export_document
from ..models.model_types import (
    LORA_OVERLAYS,
    defaults_vocabularies,
    model_type_by_name,
    model_type_for_contract,
)


def add_subparser(sub: "argparse._SubParsersAction[Any]") -> None:
    p = sub.add_parser(
        "models",
        help="Model-typed serving defaults (the pgw#1376 vocabulary).",
        description=(
            "Operate on the gen_worker.models model-type vocabulary: names, "
            "Defaults structs, and their exported JSON Schemas."
        ),
    )
    models_sub = p.add_subparsers(dest="models_command", metavar="<command>")
    models_sub.required = True

    export = models_sub.add_parser(
        "export",
        help="Emit the {names, schemas} JSON document from the Defaults structs.",
    )
    export.add_argument(
        "--out",
        default="-",
        help="Destination file, or '-' for stdout (default).",
    )
    export.set_defaults(_handler=_handle_export)

    classify = models_sub.add_parser(
        "classify",
        help="Map a recorded tensorfs contract stamp (<name>@<version>) to a model name.",
    )
    classify.add_argument("stamp", help="Layout stamp, e.g. sdxl.diffusers@1+plain.bf16@1 (the topology half is what classifies).")
    classify.set_defaults(_handler=_handle_classify)

    decode = models_sub.add_parser(
        "decode",
        help="Decode a candidate hub defaults row as the named model type (typed refusal on garbage).",
    )
    decode.add_argument("name", help="Model name (the hub `model` column value).")
    decode.add_argument(
        "--row",
        default="{}",
        help="The defaults JSONB object: inline JSON, @file, or '-' for stdin. Default {}.",
    )
    decode.set_defaults(_handler=_handle_decode)


def _handle_export(args: argparse.Namespace) -> int:
    text = json.dumps(export_document(), indent=2, sort_keys=True) + "\n"
    out = str(args.out or "-")
    if out == "-":
        sys.stdout.write(text)
        return 0
    dest = Path(out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(text)
    sys.stderr.write(f"wrote {dest}\n")
    return 0


def _handle_classify(args: argparse.Namespace) -> int:
    mt = model_type_for_contract(str(args.stamp))
    if mt is None:
        sys.stdout.write("unclassified\n")
        return 1
    sys.stdout.write(f"{mt.name}\n")
    return 0


def _carrier(name: str) -> "CarriesDefaults[msgspec.Struct] | None":
    mt = model_type_by_name(name)
    if mt is not None:
        return cast("CarriesDefaults[msgspec.Struct]", mt)
    for overlay in LORA_OVERLAYS:
        if overlay.name == name:
            return cast("CarriesDefaults[msgspec.Struct]", overlay)
    return None


def _handle_decode(args: argparse.Namespace) -> int:
    name = str(args.name)
    model_type = _carrier(name)
    if model_type is None:
        sys.stderr.write(
            f"gen-worker models decode: unrecognized model {name!r}; "
            f"recognized: {', '.join(defaults_vocabularies())}\n"
        )
        return 2
    raw = str(args.row or "{}")
    try:
        if raw == "-":
            text = sys.stdin.read()
        elif raw.startswith("@"):
            text = Path(raw[1:]).read_text()
        else:
            text = raw
        row = json.loads(text or "{}")
    except (OSError, json.JSONDecodeError) as e:
        sys.stderr.write(f"gen-worker models decode: unreadable row: {e}\n")
        return 2
    if not isinstance(row, dict):
        sys.stderr.write("gen-worker models decode: the row must be a JSON object\n")
        return 2
    try:
        decoded = decode_model_defaults(model_type, model=name, defaults=row)
    except DefaultsDecodeError as e:
        sys.stderr.write(f"{e}\n")
        return 1
    sys.stdout.write(
        json.dumps(msgspec.json.decode(msgspec.json.encode(decoded)), indent=2,
                   sort_keys=True) + "\n"
    )
    return 0
