"""pgw#988 — the publish DECLARE and the discovery FILTER are one contract.

The defect this file exists to prevent: `entries` was added to
`fleet_cells._UNBOUNDED_ENVELOPE_BLOCKS` (correctly — it is 98% of a real
cell's envelope and it put the declare over the hub's 32 MiB route cap), while
`aot_cells._discover_inner` went on verifying the FULL contract against that
same declare. The two halves of one contract disagreed, so 100% of cells
published from `c2e52f5f` on were undiscoverable, on every pod, forever — and
a pod that finds no cell mints its own, so it presented as COST, not as an
error.

Every test here drives the real `control_plane_metadata` into the real
`_candidates`. That seam is the whole point: either side alone can be
"correct" while the pair is broken.

RED-verified against a pristine `v0.93.1` tree: 6 of these 8 fail, and the
headline row rejects with the defect's exact wire signature —

    {'verify:malformed declared contract: metadata declares no entries map': 1}
"""

from __future__ import annotations

import platform
from typing import Any, Dict

import pytest

from gen_worker import aot_cells, aot_serve, fleet_cells

FAMILY = "toyfam"


@pytest.fixture()
def stub_runtime(monkeypatch: pytest.MonkeyPatch) -> Dict[str, str]:
    rt = {"sku": "l40s", "sm": "sm_89", "torch": "2.13.0+cu130",
          "cuda": "13.0"}
    monkeypatch.setattr(aot_serve, "runtime_key", lambda: dict(rt))
    return rt


def _entry() -> Dict[str, Any]:
    block: Dict[str, Any] = {
        "target": "unet",
        "inputs": [{"name": "sample", "shape": [1, 4, 64, 64],
                    "dtype": "bfloat16"}],
        "symbols": {},
        "constants": [],
    }
    block["range_digest"] = aot_serve.range_digest(block)
    block["class_hash"] = aot_serve.class_hash(block, strict=True,
                                               lora_bucket=0)
    return block


def _full_envelope(**over: Any) -> Dict[str, Any]:
    """A complete, self-consistent AOT cell envelope — what the mint stamps
    and what rides INSIDE the artifact."""
    entries = {"generate": _entry()}
    meta: Dict[str, Any] = {
        "format": aot_serve.ARTIFACT_FORMAT,
        "kind": aot_serve.ARTIFACT_KIND,
        "family": FAMILY,
        "sm": "sm_89", "torch": "2.13.0+cu130", "cuda": "13.0",
        "sku": "l40s",
        "cell_key": "ck6-" + "a" * 56,
        "entries": entries,
        "strict_export": True,
        "lora_bucket": 0,
        "weight_lane": "",
        "package_constants_in_so": False,
        "source_ref": "", "source_digest": "",
        "host_isa": {"machine": platform.machine(), "march": "",
                     "simdlen": 0, "level": ""},
        # The block that made the declare grow with the artifact.
        "guard_manifest": {"x": "y" * 64},
    }
    meta["combined_graph_hash"] = aot_serve.combined_graph_hash(
        str(b["class_hash"]) for b in entries.values())
    meta.update(over)
    return meta


def _listing(meta: Dict[str, Any]) -> list:
    return [{"checkpoint_id": "ck-1", "updated_at": "2026-08-06T00:00:00Z",
             "metadata": meta}]


def test_a_cell_published_through_the_real_declare_is_discoverable(
    stub_runtime: Dict[str, str],
) -> None:
    """THE regression row. Publisher -> declare -> consumer, no fixtures in
    between. This failed on `origin/master` with
    `verify:malformed declared contract: metadata declares no entries map`."""
    declare = fleet_cells.control_plane_metadata(_full_envelope())
    rejected: Dict[str, int] = {}
    rows = aot_cells._candidates(_listing(declare), FAMILY, "", rejected)
    assert [r[1] for r in rows] == ["ck-1"], rejected


def test_the_declare_drops_the_map_and_keeps_a_bounded_summary() -> None:
    """The size property th#1645 bought must survive the fix: the map is
    still gone, and what replaces it is two scalars."""
    declare = fleet_cells.control_plane_metadata(_full_envelope())
    assert "entries" not in declare
    assert "guard_manifest" not in declare
    assert declare[aot_serve.ENTRIES_COUNT_KEY] == 1
    assert len(declare[aot_serve.ENTRIES_DIGEST_KEY]) == 16


def test_the_summary_is_computed_from_the_map_it_replaces() -> None:
    """Two different entry maps must not summarize identically — otherwise
    the stand-in states nothing."""
    a = fleet_cells.control_plane_metadata(_full_envelope())
    other = _full_envelope()
    other["entries"]["generate"]["target"] = "transformer"
    b = fleet_cells.control_plane_metadata(other)
    assert a[aot_serve.ENTRIES_DIGEST_KEY] != b[aot_serve.ENTRIES_DIGEST_KEY]


def test_a_pre_pgw988_declare_still_passes_the_prefilter(
    stub_runtime: Dict[str, str],
) -> None:
    """The prefilter fails OPEN on a silent summary, on purpose.

    A cell published before this fix carries no summary. Refusing it here
    would be pgw#988 again with a different message — the prefilter's job is
    to avoid paying for a download, and the GATE is the artifact-level verify.
    """
    declare = fleet_cells.control_plane_metadata(_full_envelope())
    declare.pop(aot_serve.ENTRIES_COUNT_KEY)
    declare.pop(aot_serve.ENTRIES_DIGEST_KEY)
    rejected: Dict[str, int] = {}
    rows = aot_cells._candidates(_listing(declare), FAMILY, "", rejected)
    assert [r[1] for r in rows] == ["ck-1"], rejected


def test_a_summary_that_is_present_and_malformed_is_refused_by_name(
    stub_runtime: Dict[str, str],
) -> None:
    declare = fleet_cells.control_plane_metadata(_full_envelope())
    declare[aot_serve.ENTRIES_COUNT_KEY] = 0
    rejected: Dict[str, int] = {}
    rows = aot_cells._candidates(_listing(declare), FAMILY, "", rejected)
    assert rows == []
    assert any(cls.startswith("verify:") and "entries_count" in cls
               for cls in rejected), rejected


def test_the_runtime_axes_are_still_refused_strictly_at_the_prefilter(
    stub_runtime: Dict[str, str],
) -> None:
    """Splitting the contract must not soften the axes a declare CAN state.
    An sm mismatch is still a pre-download refusal, by name."""
    declare = fleet_cells.control_plane_metadata(_full_envelope(sm="sm_120"))
    rejected: Dict[str, int] = {}
    rows = aot_cells._candidates(_listing(declare), FAMILY, "", rejected)
    assert rows == []
    assert any("sm" in cls for cls in rejected), rejected


def test_the_full_contract_is_verified_against_the_artifact_not_the_declare(
    stub_runtime: Dict[str, str],
) -> None:
    """The other half of the split, asserted directly: `verify` still fails
    closed on a broken entry — it just runs where the entries actually are."""
    envelope = _full_envelope()
    assert aot_serve.verify(dict(envelope), family=FAMILY) == ""
    envelope["entries"]["generate"]["class_hash"] = "0" * 16
    assert "class_hash" in aot_serve.verify(dict(envelope), family=FAMILY)


def test_a_declare_alone_could_never_pass_the_artifact_level_gate() -> None:
    """Why the split is not a loosening: the bounded declare CANNOT satisfy
    `verify`, so nothing can be armed on the strength of a declare."""
    declare = fleet_cells.control_plane_metadata(_full_envelope())
    assert aot_serve.verify(dict(declare), family=FAMILY) != ""
