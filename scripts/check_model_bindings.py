#!/usr/bin/env python3
"""pgw#1332 — every committed family binding is what its export implies.

The catalog commits TWO halves per family: ``<family>.export.json`` (the
declaration-time fake-tensor export) and ``<family>.py`` (the typed bindings
generated from it). The failure this guard exists for is that they drift — an
export regenerated and committed while the binding beside it was not, or a
binding hand-edited to fix something the declaration should have fixed. Either
way the type an endpoint compiles against stops describing the classes the
fleet will actually arm, and nothing else notices until a pod does.

It is a BYTE COMPARISON and it is cheap, because codegen is a pure function of
the document: no torch, no diffusers, no GPU, no network. That is what lets it
live in the two-minute required job rather than behind the fifteen-minute one.

    python scripts/check_model_bindings.py

Exit 0 when every pair is consistent; 1 with the exact regeneration command
otherwise.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from gen_worker.model.codegen import render_module  # noqa: E402
from gen_worker.model.snapshot import ModelExport  # noqa: E402

CATALOG = REPO / "src" / "gen_worker" / "model" / "catalog"
GENERATED = CATALOG / "_generated"

#: family handle -> the declaration its bindings lazily import. Kept HERE and
#: not derived from the directory, so a family whose declaration module was
#: renamed or deleted fails this guard rather than regenerating against
#: whatever is still importable.
SPECS = {
    "flux1_dev": "gen_worker.model.catalog.flux1_dev:FLUX1_DEV",
    "sdxl": "gen_worker.model.catalog.sdxl:SDXL",
}


def main() -> int:
    documents = sorted(GENERATED.glob("*.export.json"))
    found = {path.name.removesuffix(".export.json") for path in documents}
    problems: list[str] = []

    unlisted = sorted(found - set(SPECS))
    if unlisted:
        problems.append(
            f"{unlisted[0]}.export.json is committed but scripts/check_model_bindings.py "
            "does not list it; add it to SPECS so the pair is fenced"
        )
    absent = sorted(set(SPECS) - found)
    if absent:
        problems.append(
            f"SPECS lists {absent[0]!r} but no {absent[0]}.export.json is committed"
        )

    for family in sorted(found & set(SPECS)):
        document = GENERATED / f"{family}.export.json"
        binding = GENERATED / f"{family}.py"
        raw = document.read_text()
        try:
            export = ModelExport.loads(raw)
        except Exception as exc:  # noqa: BLE001 - report, never traceback
            problems.append(f"{document.name} does not decode: {exc}")
            continue
        if str(export.family) != family:
            problems.append(
                f"{document.name} declares family {str(export.family)!r}, not {family!r}"
            )
            continue
        if raw != export.dumps():
            problems.append(
                f"{document.name} is not canonical; rewrite it with "
                f"`gen-worker model export {SPECS[family]}`"
            )
        module, _, attr = SPECS[family].partition(":")
        expected = render_module(export, spec_module=module, spec_attr=attr)
        actual = binding.read_text() if binding.is_file() else ""
        if actual != expected:
            problems.append(
                f"{binding.name} is not what {document.name} implies (digest "
                f"{export.digest()}). Regenerate:\n"
                f"    gen-worker model generate "
                f"src/gen_worker/model/catalog/_generated/{family}.export.json "
                f"--spec {SPECS[family]}"
            )

    if problems:
        print("family bindings: NOT current", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    print(f"family bindings: {len(found)} pair(s) current")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
