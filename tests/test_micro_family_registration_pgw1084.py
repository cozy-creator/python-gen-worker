"""pgw#1084 §8.4.1 — the worker half of the micro-family cross-repo fence.

THE DEFECT CLASS, three times in three weeks. `examples/micro-diffusion` gains
a member; tensorhub's `internal/modelfamily` does not learn about it; and
because build-time discovery walks the WHOLE package
(`top_level = main_module.split(".", 1)[0]`, discovery/discover.py), the
unregistered family fails EVERY build that carries the tarball — not only its
own function. Each occurrence was found by a lane that had already paid for the
build:

    micro-escape        pgw#1068   worked around by excluding *_escape.py
    micro-conv          pgw#1073   same shape, recorded in modelfamily.go
    micro-pad32(+…)     pgw#1079   found by pgw#1084 §8.4.1, mid-campaign:
                                   `invalid discovery manifest: family
                                    "micro-pad32" is not a known architecture
                                    family`

The refusal is already TYPED and already names its remedy (tensorhub
`modelfamily.HowToRegister`). That was not enough, three times over, because it
can only fire against a live hub with a tarball already uploaded. Neither repo's
CI can reach the other's registry, so the fence is two halves over one shared
constant, `examples/micro-diffusion/FAMILIES`:

    this file          the FAMILIES list equals what discovery ACTUALLY finds
    tensorhub          the same list, vendored + peer-fetched, is registered
                       (internal/modelfamily/MICRO_EXAMPLE_FAMILIES,
                        scripts/micro-family-drift.sh)

RED-PROVEN: adding a member declaring a family absent from FAMILIES fails
`test_families_declared_equal_the_shared_constant`, naming the family, the file
and the tensorhub commit it also owes.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Set

import pytest

from gen_worker.discovery.discover import discover_manifest

REPO = Path(__file__).resolve().parents[1]
MICRO = REPO / "examples" / "micro-diffusion"
FAMILIES_FILE = MICRO / "FAMILIES"

#: Where the peer half lives. Quoted in every failure below, because the whole
#: point of this fence is that the author is in the WRONG REPOSITORY to fix it.
PEER_REGISTRY = "tensorhub internal/modelfamily/modelfamily.go"
PEER_VENDORED = "tensorhub internal/modelfamily/MICRO_EXAMPLE_FAMILIES"


def _shared_constant() -> List[str]:
    lines = []
    for raw in FAMILIES_FILE.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            lines.append(line)
    return lines


def _manifest_families(manifest: Dict[str, Any]) -> Set[str]:
    """Every family spelling tensorhub's manifest gates will read.

    The hub applies `modelfamily.IsKnown` in three places on this document —
    `slots[].family` and `bindings[].family` (manifest_contract.go, the walk
    attempt-27 actually hit) and `compile.family` (th#1310) — so the fence
    collects all of them plus `config_family`, rather than guessing which gate
    a future member will trip first.
    """
    out: Set[str] = set()

    def add(value: Any) -> None:
        if isinstance(value, str) and value.strip():
            out.add(value.strip())

    for fn in manifest.get("functions") or []:
        add((fn.get("compile") or {}).get("family"))
        add(fn.get("config_family"))
        for slot in fn.get("slots") or []:
            add(slot.get("family"))
        for binding in (fn.get("bindings") or {}).values():
            if isinstance(binding, dict):
                add(binding.get("family"))
    return out


@pytest.fixture(scope="module")
def declared() -> Set[str]:
    """The REAL build-time scan, not a grep over the source.

    `discover_manifest` is the exact entry point the image's build step runs
    (`python -m gen_worker.discover`), so a member this fence cannot see is a
    member the hub cannot see either. It needs no GPU and no network.
    """
    return _manifest_families(discover_manifest(MICRO))


def test_families_declared_equal_the_shared_constant(declared: Set[str]) -> None:
    listed = set(_shared_constant())
    unlisted = sorted(declared - listed)
    dead = sorted(listed - declared)
    assert not unlisted, (
        f"{unlisted} reach the discovery manifest but are not in "
        f"{FAMILIES_FILE.relative_to(REPO)}.\n"
        f"A family is not usable until it exists in BOTH repositories, and "
        f"discovery scans the whole package — so until then EVERY build "
        f"carrying this example fails, not just the new function.\n"
        f"  1. add it to {FAMILIES_FILE.relative_to(REPO)}\n"
        f"  2. register it in {PEER_REGISTRY} (canonicalFamilies, plus "
        f"rootOverrides if it shares micro-diffusion's weight envelope) and "
        f"update {PEER_VENDORED} in the SAME commit — there is no runtime "
        f"registration path, it takes effect at the next hub deploy\n"
        f"  3. seed the catalog binding on the stack (examples/micro-diffusion/"
        f"README.md step 0)")
    assert not dead, (
        f"{dead} are listed in {FAMILIES_FILE.relative_to(REPO)} but no longer "
        f"reach the discovery manifest. Delete the line here and the matching "
        f"entry in {PEER_REGISTRY}, or the hub keeps a family nothing declares.")


def test_the_shared_constant_is_sorted_and_unique() -> None:
    """It is diffed byte-for-byte against a vendored copy in another repo."""
    listed = _shared_constant()
    assert listed == sorted(set(listed)), (
        "FAMILIES must be sorted with no duplicates: tensorhub's "
        "scripts/micro-family-drift.sh compares it to a vendored copy, and an "
        "ordering difference there is indistinguishable from a real one")
    for name in listed:
        assert re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", name), (
            f"{name!r} is not a normalized family spelling; tensorhub's "
            f"modelfamily.Normalize lowercases and strips dots, so a name that "
            f"needs normalizing is registered under a DIFFERENT key than the "
            f"one written here")


def test_every_declared_family_is_reachable_from_the_package(declared: Set[str]) -> None:
    """Guards the fence's own premise: the scan sees more than `main`.

    If `discover_manifest` ever stopped walking submodules this test would
    still pass trivially against a shrunken FAMILIES file, so assert the
    property that makes the fence meaningful — the scan finds families declared
    in modules `endpoint.toml`'s `main` never imports.
    """
    assert "micro-pad32-branchy" in declared, (
        "the discovery scan no longer reaches main_pad32_branchy.py — either a "
        "member was deleted (update FAMILIES) or the walk stopped covering the "
        "package, which would make this whole fence vacuous")
    assert len(declared) >= 6
