#!/usr/bin/env python3
"""pgw#1332 — every committed family binding is what its export implies.

The catalog commits TWO halves per family: a declaration-time document and
``<family>.py`` (the typed bindings generated from it). Which document depends
on the tier — ``<family>.export.json`` (``family_export_v1``, a fake-tensor
export) for a graph model, ``<family>.eager.json`` (``eager_model_v1``, a
projection of the declaration, pgw#1346 B5) for an eager one — and both tiers
are fenced here by the same rule. The failure this guard exists for is that
the two halves drift — an
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
from collections.abc import Callable
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from gen_worker.model.codegen import render_eager_module, render_module  # noqa: E402
from gen_worker.model.snapshot import EagerExport, ModelExport  # noqa: E402

CATALOG = REPO / "src" / "gen_worker" / "model" / "catalog"
GENERATED = CATALOG / "_generated"

#: family handle -> the declaration its bindings lazily import. Kept HERE and
#: not derived from the directory, so a family whose declaration module was
#: renamed or deleted fails this guard rather than regenerating against
#: whatever is still importable.
SPECS = {
    "flux1_dev": "gen_worker.model.catalog.flux1_dev:FLUX1_DEV",
    "flux2_klein_4b": "gen_worker.model.catalog.flux2_klein_4b:FLUX2_KLEIN_4B",
    "flux2_klein_9b": "gen_worker.model.catalog.flux2_klein_9b:FLUX2_KLEIN_9B",
    "sd2": "gen_worker.model.catalog.sd15:SD2",
    "sd15": "gen_worker.model.catalog.sd15:SD15",
    "sdxl": "gen_worker.model.catalog.sdxl:SDXL",
}

#: The same map for the EAGER tier (pgw#1346 B5), kept separate rather than
#: merged with a suffix column: the two tiers commit DIFFERENT documents
#: (``eager_model_v1`` vs ``family_export_v1``) read by different decoders, and
#: a single map keyed only by family would let a graph model silently be fenced
#: as an eager one — which would pass while generating bindings with no runner
#: callables at all.
EAGER_SPECS = {
    "chatterbox": "gen_worker.model.catalog.boundary_audio:CHATTERBOX",
    "flex2_preview": "gen_worker.model.catalog.flex2_preview:FLEX2_PREVIEW",
    "foundation_1": "gen_worker.model.catalog.boundary_audio:FOUNDATION_1",
    "hunyuan3d": "gen_worker.model.catalog.boundary_3d:HUNYUAN3D",
    "internvl_u": "gen_worker.model.catalog.boundary_llm:INTERNVL_U",
    "joycaption": "gen_worker.model.catalog.boundary_llm:JOYCAPTION",
    "musicgen": "gen_worker.model.catalog.boundary_audio:MUSICGEN",
    "qwen36_27b_mtp": "gen_worker.model.catalog.boundary_llm:QWEN36_27B_MTP",
    "qwen36_35b_a3b": "gen_worker.model.catalog.boundary_llm:QWEN36_35B_A3B",
    "stable_audio_open": "gen_worker.model.catalog.boundary_audio:STABLE_AUDIO_OPEN",
    "trellis_3d": "gen_worker.model.catalog.boundary_3d:TRELLIS_3D",
}


def _check(
    specs: dict[str, str],
    suffix: str,
    decode: Callable[[str], Any],
    render: Callable[..., str],
) -> tuple[int, list[str]]:
    """One tier's committed (document, binding) pairs, byte-compared."""

    documents = sorted(GENERATED.glob(f"*.{suffix}"))
    found = {path.name.removesuffix(f".{suffix}") for path in documents}
    problems: list[str] = []

    unlisted = sorted(found - set(specs))
    if unlisted:
        problems.append(
            f"{unlisted[0]}.{suffix} is committed but scripts/check_model_bindings.py "
            "does not list it; add it so the pair is fenced"
        )
    absent = sorted(set(specs) - found)
    if absent:
        problems.append(f"SPECS lists {absent[0]!r} but no {absent[0]}.{suffix} is committed")

    for family in sorted(found & set(specs)):
        document = GENERATED / f"{family}.{suffix}"
        binding = GENERATED / f"{family}.py"
        raw = document.read_text()
        try:
            export = decode(raw)
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
                f"`gen-worker model export {specs[family]}`"
            )
        module, _, attr = specs[family].partition(":")
        expected = render(export, spec_module=module, spec_attr=attr)
        actual = binding.read_text() if binding.is_file() else ""
        if actual != expected:
            problems.append(
                f"{binding.name} is not what {document.name} implies (digest "
                f"{export.digest()}). Regenerate:\n"
                f"    gen-worker model generate "
                f"src/gen_worker/model/catalog/_generated/{family}.{suffix} "
                f"--spec {specs[family]}"
            )
    return len(found), problems


def main() -> int:
    graph_pairs, problems = _check(SPECS, "export.json", ModelExport.loads, render_module)
    eager_pairs, eager_problems = _check(
        EAGER_SPECS, "eager.json", EagerExport.loads, render_eager_module
    )
    problems.extend(eager_problems)
    # A family cannot be both tiers. Two documents under one handle would make
    # the catalog index ambiguous and the winner would be whichever renderer
    # wrote `<family>.py` last.
    both = sorted(set(SPECS) & set(EAGER_SPECS))
    if both:
        problems.append(
            f"{both[0]!r} is listed as BOTH a graph and an eager model; one family, "
            "one tier, one committed document"
        )

    if problems:
        print("family bindings: NOT current", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    print(f"family bindings: {graph_pairs} graph + {eager_pairs} eager pair(s) current")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
