"""pgw#1621 — the v2 stamp-pair surface accepts every migrated endpoint's REAL
declaration, read out of the sibling `serverless-endpoints` checkout.

**WHY THIS FILE EXISTS AND WHY IT IS NOT A UNIT TEST.** The declaration surface
is exercised everywhere in this repo against declarations this repo wrote. That
proves the surface accepts what we thought we would write. It does not prove it
accepts what the fleet ACTUALLY wrote, and the difference was not academic: the
handle grammar here required a hyphen-free producer segment
(``^[a-z0-9]+\\.``), which is correct for every v1 QUANT handle — hyphens only
ever fell in the FORMAT segment — and wrong for half the v2 TOPOLOGY corpus.
``flux2-klein.diffusers@1``, ``hidream-o1.diffusers@1``,
``minimax-h3.diffusers@1``, ``z-image.diffusers@1``,
``ltx2-upsampler.diffusers@1``, ``sdxl-inpainting.diffusers@1`` and
``qwen3-6-35b-a3b.transformers@1`` all refused as "not a handle". **Five
migrated endpoints could not have declared a lane at all**, and every test in
this repo was green.

It was caught by running the real `_parse_lanes` over real `main.py` files.
That is the standard this file keeps: **verify the artifact, not a
reconstruction.**

TWO DELIBERATE CHOICES, both of which the first cut of this probe got wrong:

* **AST, never a regex.** A regex for ``contracts.<CONST>`` over the source
  matches inside DOCSTRINGS. The first cut reported hunyuan3d as blocked on a
  constant named ``X`` that exists only in prose explaining why that endpoint
  has no lane. The walk below looks at ``lanes=`` keywords on class headers and
  nothing else.
* **`_parse_lanes`, never a module import.** Importing an endpoint's `main.py`
  needs its own third-party deps (diffsynth, cozy_rife, ...) which are not
  installed here, so five endpoints could never be answered that way — and a
  test that silently skips five of the things it exists to check is the
  failure mode, not the check. `_parse_lanes` IS the surface under test: it is
  what `__init_subclass__` calls.

**A MISSING SIBLING SKIPS; AN EMPTY SCAN FAILS.** The sibling checkout is not
present in every environment, so absence is an honest skip. What is never
allowed is a scan that found the checkout, matched nothing, and reported
green — `test_the_fleet_scan_is_not_vacuous` is that guard, and it asserts a
COUNT.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from gen_worker.demand import GiB, const
from gen_worker.models.tensor_layout_contract import display_names
from gen_worker.serving.lane_spec import lane
from gen_worker.serving.model import _parse_lanes

#: Endpoints whose declared lane has NO ratified (topology, quant) pair yet.
#: These are a CORPUS gap, not a surface defect: a v2 topology is extracted
#: mechanically from a reference checkpoint's banked headers, and these five
#: have not been banked. Listed by name so the day one lands, this file goes
#: RED and the name is removed — the same self-deleting shape pgw#1606's dtype
#: waiver had. An entry that cannot expire is a permanent hole nobody re-reads.
CORPUS_GAP: dict[str, str] = {
    "FLUX1_DIFFUSERS_BF16": "flux1.diffusers — `flux1` is declared in tensorfs "
                            "spec/v2/headers/SOURCES.tsv and was never banked",
    "STABLE_AUDIO_DIFFUSERS_FP16": "stable-audio.diffusers — likewise declared "
                                   "in SOURCES.tsv and never banked",
    "TRELLIS2_DIT_BF16": "trellis2.dit — no source row at all",
    "QWEN_IMAGE_DIFFUSERS_BF16": "qwen-image.diffusers — no source row at all",
    "INTERNVL_U_DIFFUSERS_BF16": "internvl-u.diffusers — no source row at all",
}


def _sibling() -> Path | None:
    """The `serverless-endpoints` checkout beside this one, or None.

    Walks this repo's parents rather than taking a configured path: the
    workspace layout (`~/cozy/<repo>`) is the fact, and a configured path is a
    second one that can be stale.
    """
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "serverless-endpoints"
        if (candidate / ".git").exists():
            return candidate
    return None


def _pairs() -> dict[str, tuple[str, str]]:
    """v1 constant name -> the (topology, quant) pair it is the display name of.

    Built by INVERTING the ratified table, never by string surgery on the v1
    spelling: `musicgen`'s pair is `plain.f16@1` while its display name says
    `-fp16@1`, so a derivation would be wrong for it.
    """
    out: dict[str, tuple[str, str]] = {}
    for pair, v1 in display_names().items():
        constant = v1.split("@", 1)[0].replace(".", "_").replace("-", "_").upper()
        topology, _, quant = pair.partition("+")
        out[constant] = (topology, quant)
    return out


def _lane_key_constants(main_py: Path) -> list[str]:
    """Every `contracts.<CONST>` used as a `lanes=` KEY on a class header."""
    tree = ast.parse(main_py.read_text(encoding="utf-8"), filename=str(main_py))
    found: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for keyword in node.keywords:
            if keyword.arg != "lanes" or not isinstance(keyword.value, ast.Dict):
                continue
            for key in keyword.value.keys:
                if (isinstance(key, ast.Attribute)
                        and isinstance(key.value, ast.Name)
                        and key.value.id == "contracts"):
                    found.append(key.attr)
    return found


def _declaring_endpoints() -> list[tuple[str, Path, list[str]]]:
    root = _sibling()
    if root is None:
        return []
    rows = []
    for pyproject in sorted(root.glob("*/pyproject.toml")):
        for main_py in sorted(pyproject.parent.glob("src/*/main.py")):
            consts = _lane_key_constants(main_py)
            if consts:
                rows.append((pyproject.parent.name, main_py, consts))
    return rows


needs_sibling = pytest.mark.skipif(
    _sibling() is None,
    reason="no sibling serverless-endpoints checkout; the fleet cannot be read "
           "here, and inventing one would prove nothing",
)


@needs_sibling
def test_the_fleet_scan_is_not_vacuous() -> None:
    """Read the COUNT, not the verdict.

    A glob that matched nothing makes every assertion below pass. That is the
    exact failure this whole file exists to prevent, so it is asserted first.
    """
    rows = _declaring_endpoints()
    assert len(rows) >= 10, (
        f"only {len(rows)} endpoint(s) with a `lanes=` declaration found under "
        f"{_sibling()} — the scan broke or the fleet moved. Every assertion "
        f"below passes vacuously on an empty scan."
    )


@needs_sibling
def test_every_declared_fleet_lane_resolves_to_a_ratified_pair() -> None:
    """Every v1 lane handle the fleet declares is either a ratified pair or a
    NAMED corpus gap. A third outcome does not exist."""
    pairs = _pairs()
    unaccounted: dict[str, list[str]] = {}
    for name, _main, consts in _declaring_endpoints():
        missing = sorted(c for c in consts
                         if c not in pairs and c not in CORPUS_GAP)
        if missing:
            unaccounted[name] = missing
    assert not unaccounted, (
        f"these endpoints declare a lane with no ratified (topology, quant) "
        f"pair and no CORPUS_GAP entry: {unaccounted}. Either the pair exists "
        f"and `display-names.json` is missing its row, or the topology has not "
        f"been banked — and in the second case it belongs in CORPUS_GAP WITH "
        f"the reason, so that the day it lands this test goes red and the "
        f"entry is removed."
    )


@needs_sibling
def test_the_declaration_surface_accepts_every_migrated_endpoint() -> None:
    """The acceptance, through the REAL `_parse_lanes` — the same function
    `Model.__init_subclass__` calls, over the fleet's own declarations."""
    pairs = _pairs()
    refused: dict[str, str] = {}
    accepted: dict[str, list[str]] = {}
    for name, _main, consts in _declaring_endpoints():
        if any(c in CORPUS_GAP for c in consts):
            continue
        declaration = {pairs[c]: lane(request=const(GiB(1))) for c in consts}
        holder = type(f"Probe_{name}", (), {})
        try:
            declared = _parse_lanes(holder, declaration)
        except Exception as exc:  # noqa: BLE001 - the verdict under test
            refused[name] = f"{type(exc).__name__}: {exc}"
            continue
        accepted[name] = [row.contract_id for row in declared]
    assert not refused, (
        f"the v2 declaration surface REFUSED a real fleet declaration: "
        f"{refused}. This is the assertion that caught the hyphen-free "
        f"producer grammar; a refusal here means the fleet cannot declare "
        f"what the corpus ratifies."
    )
    assert len(accepted) >= 5, (
        f"only {len(accepted)} endpoint(s) reached the surface at all "
        f"({sorted(accepted)}); the rest were skipped as corpus gaps. If "
        f"CORPUS_GAP has grown to swallow the fleet, this test is measuring "
        f"nothing."
    )


@needs_sibling
def test_a_corpus_gap_entry_deletes_itself_when_its_topology_lands() -> None:
    """A CORPUS_GAP entry is a fact about the vendored corpus, never a
    preference, so it cannot outlive its reason.

    The day a banked topology gives one of these a ratified pair, this goes red
    naming the entry — and removing it is then a one-line change that makes the
    endpoint declarable. That is the same self-deleting shape pgw#1606's dtype
    waiver had, and it is why that waiver is gone rather than forgotten.
    """
    pairs = _pairs()
    landed = sorted(c for c in CORPUS_GAP if c in pairs)
    assert not landed, (
        f"these CORPUS_GAP entries now HAVE a ratified pair: {landed}. The "
        f"topology was banked and the waiver has outlived its reason — delete "
        f"the entries; the endpoints become declarable with no other change."
    )
