#!/usr/bin/env python3
"""The registry CONTRACT of the shipped artifact.

The SDK deliberately ships every family registry EMPTY: vocabularies are
declared by the endpoint that owns them, because a vocabulary in the library
needs a wheel release to change. "Registries are populated" is
therefore the WRONG assertion for a bare wheel. The right one — asserted here —
is the contract:

  A. bare import exposes working registry mechanisms in their DOCUMENTED bare
     state (family/convert/export/layout-conversion registries empty, generic
     component vocabulary present, unknown families refused BY NAME with the
     fix in the message);
  B. a representative consumer's declarations, imported via the documented
     ``load_declaration_module`` path, become visible in ALL six registries
     and resolve (aliases, layout hints, foreign-catalog adapter, defaults
     class, export declaration, layout conversion edge).

Run with ``--installed`` (the publish workflow does, against the freshly built
wheel in a clean venv) to also refuse a ``gen_worker`` that resolves from this
checkout instead of site-packages. A failure names the registry that broke.

If a future release deliberately changes the bare-state contract (e.g. ships
builtin declarations), update phase A in the same commit — that is the point:
registry state changes must be deliberate, in either direction.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_FAILURES: list[str] = []


def _check(registry: str, ok: bool, detail: str) -> None:
    tag = "ok" if ok else "FAIL"
    print(f"[{tag}] {registry}: {detail}")
    if not ok:
        _FAILURES.append(f"{registry}: {detail}")


def phase_installed() -> None:
    import gen_worker

    loc = Path(gen_worker.__file__).resolve()
    repo_src = (_SCRIPT_DIR.parent / "src").resolve()
    _check(
        "installed",
        repo_src not in loc.parents,
        f"gen_worker resolves from {loc} (must NOT be the checkout's src/)",
    )


def phase_bare() -> None:
    from gen_worker.api.export_contract import registered_export_families
    from gen_worker.component_vocab import component_vocabulary
    from gen_worker.convert import (
        UnknownFamilyError,
        registered_layout_conversions,
        registered_layout_productions,
        registered_layouts,
        registered_repackage_families,
        require_repackage_family,
    )
    from gen_worker.families import family_registry

    _check("families", family_registry() == {}, "bare registry empty-by-design")
    _check("repackage", registered_repackage_families() == (), "bare registry empty-by-design")
    _check("layouts", registered_layouts() == (), "bare registry empty-by-design")
    _check("export", registered_export_families() == (), "bare registry empty-by-design")
    _check(
        "layout-conversions",
        registered_layout_conversions() == () and registered_layout_productions() == (),
        "bare registry empty-by-design (§1.33: edges are declared by whoever "
        "owns the format, never shipped in the wheel)",
    )
    vocab = component_vocabulary()
    _check(
        "component_vocab",
        bool(vocab.denoisers and vocab.text_encoders and vocab.vaes),
        "generic diffusers vocabulary ships in the wheel",
    )
    try:
        require_repackage_family("no-such-family")
    except UnknownFamilyError as exc:
        _check(
            "repackage-refusal",
            "register_repackage_family" in str(exc),
            "unknown family refused by name, message carries the fix",
        )
    else:
        _check("repackage-refusal", False, "unknown family was NOT refused")


def phase_consumer() -> None:
    from gen_worker.api.export_contract import registered_export_families
    from gen_worker.convert import (
        civitai_to_family,
        load_declaration_module,
        normalize_family,
        registered_layout_conversions,
        registered_layouts,
        registered_repackage_families,
    )
    from gen_worker.convert.layout import infer_model_family_from_hint
    from gen_worker.families import family_for

    sys.path.insert(0, str(_SCRIPT_DIR))
    load_declaration_module("_registry_contract_consumer")
    fam = "contractcheck"

    _check("families", family_for(fam) is not None, "@family class registered + resolvable")
    _check(
        "repackage",
        fam in registered_repackage_families(),
        "RepackageFamily registered",
    )
    _check(
        "repackage-alias",
        normalize_family("contractcheck-finetune-v2") == fam and normalize_family("cc") == fam,
        "alias + lineage-prefix normalization resolves",
    )
    _check(
        "layouts",
        any(d.family == fam for d in registered_layouts()),
        "LayoutDeclaration registered",
    )
    _check(
        "layouts-detect",
        infer_model_family_from_hint(f"someone/{fam}-v1") == fam,
        "hint detection routes through the declared matcher",
    )
    _check(
        "civitai",
        civitai_to_family("ContractCheck 1.0") == fam,
        "foreign-catalog adapter maps the injected enum",
    )
    _check(
        "export",
        fam in registered_export_families(),
        "class-bearing Compile declaration registered",
    )
    edges = registered_layout_conversions()
    _check(
        "layout-conversions",
        {(e.from_id, e.to_id) for e in edges}
        == {("comfy.splitfiles@1", "diffusers.multifile@1"),
            ("diffusers.multifile@1", "comfy.splitfiles@1")},
        "TopologyConversion registered — BOTH directions, because the "
        "round-trip admission proof passed on both",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--installed",
        action="store_true",
        help="require gen_worker to resolve from site-packages, not this checkout",
    )
    args = parser.parse_args()

    if args.installed:
        phase_installed()
    phase_bare()
    phase_consumer()

    if _FAILURES:
        print(f"\nregistry contract BROKEN ({len(_FAILURES)} failure(s)):")
        for line in _FAILURES:
            print(f"  - {line}")
        return 1
    print("\nregistry contract holds")
    return 0


if __name__ == "__main__":
    sys.exit(main())
