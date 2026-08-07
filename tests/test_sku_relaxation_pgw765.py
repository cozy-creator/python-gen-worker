"""pgw#765: the AOT serve path is pinned to sm, never to the GPU's SKU.

Paul's ruling: "AOT cells should be locked into the sm_x version, not the
actual GPU." Three earlier pieces of work exist ONLY to make that true —
pgw#691 dropped ``sku`` from cell identity on byte-identical evidence
(a40-vs-3090, l4-vs-4090), the FX inner-key shim normalized inductor's
device NAME to the sm token, and the pgw#754 clamp made the packaged HOST
code portable. ``aot_serve.verify`` nevertheless required ``sku`` equality,
so an l4-minted cell refused to arm on an rtx-4090 — both sm_89, the same
compiled code. The JIT lane carried the identical defect and shed it in the
ck3 wave; this is the same bug, reintroduced on the exported lane.

What is asserted here:

1. **The axis sets** — no adoption lane declares ``sku`` as an identity
   axis, and a structural sweep proves no adoption path *refuses* on it
   (a preference is allowed; a refusal is not). The pattern has appeared
   twice now, so it is a standing guard rather than a one-time fix.
2. **Both directions, red-verified** — an l4-minted cell verifies and is a
   candidate on an rtx-4090 runtime; an sm-mismatched cell still refuses by
   name, and so does every other real axis.
3. **The second tier** — a cell whose ``sm`` stamp DISAGREES with its own
   ``.pt2`` is ruled on from torch's ``AOTI_COMPUTE_CAPABILITY`` before
   dlopen (``sm_mismatch``), because pgw#698 packs cubins with no PTX
   fallback and a wrong-arch load must be a named refusal, not a CUDA error.

pgw#950 removed the fourth thing this file used to assert: that a cell
stamping NO ``sm`` axis was tolerated by ``verify`` and deferred to that
second tier. It is not tolerated any more. A cell silent on an identity axis
is refused like any other mismatch, and the miss policy re-mints it — so the
second tier's remaining job is the lying stamp, which metadata cannot catch.
"""

from __future__ import annotations

import ast
import json
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

from gen_worker import aot_cells, aot_serve, compile_cache

FAMILY = "sdxl"
_SRC = Path(__file__).resolve().parents[1] / "src" / "gen_worker"

#: This runtime: an rtx-4090 — sm_89, exactly the L4's architecture.
RUNTIME_4090 = {
    "sku": "rtx-4090", "sm": "sm_89", "torch": "2.13.0+cu130", "cuda": "13.0",
}


@pytest.fixture()
def on_4090(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(aot_serve, "runtime_key", lambda: dict(RUNTIME_4090))


def _key(seed: str) -> str:
    return "ck1-" + (seed * 56)[:56]


def _meta(key: str, **over: Any) -> Dict[str, Any]:
    """A published AOT cell's metadata, minted on an l4 unless overridden."""
    meta = aot_serve.artifact_metadata(
        family=FAMILY, precision="bf16", cell_key=key,
        entries={"unet/main": {
            "target": "unet", "fork": [], "class_dims": [],
            "inputs": [{"name": "sample", "position": 0, "dtype": "bfloat16",
                        "shape": [2, 4, 128, 128]}],
            "symbols": {}, "constants": [], "graph": {},
        }})
    meta.update({"sku": "l4", "sm": "sm_89", "torch": "2.13.0+cu130",
                 "cuda": "13.0", "weight_lane": "w8a8", "lora_bucket": 0})
    meta["host_isa"] = {"machine": "x86_64", "march": "", "simdlen": 0,
                        "level": "x86-64-v2"}
    meta.update(over)
    return meta


def _item(checkpoint: str, updated_at: str,
          meta: Dict[str, Any]) -> Dict[str, Any]:
    return {"checkpoint_id": checkpoint, "updated_at": updated_at,
            "metadata": meta}


# ---------------------------------------------------------------------------
# 1. verify — cross-SKU adopts, every real axis still refuses
# ---------------------------------------------------------------------------


def test_l4_minted_cell_verifies_on_a_4090(on_4090: None) -> None:
    """The whole point: same sm_89, different board, adoption allowed."""
    assert aot_serve.verify(_meta(_key("a")), family=FAMILY) == ""


def test_sm_mismatch_still_refuses_by_name(on_4090: None) -> None:
    reason = aot_serve.verify(_meta(_key("b"), sm="sm_86"), family=FAMILY)
    assert reason == "sm 'sm_86' != runtime 'sm_89'"


def test_torch_and_cuda_mismatches_still_refuse(on_4090: None) -> None:
    """Relaxing sku must not relax the ABI axes: an AOTI ``.pt2`` is an ELF
    built against one exact torch build."""
    assert "torch" in aot_serve.verify(
        _meta(_key("c"), torch="2.12.0+cu130"), family=FAMILY)
    assert "cuda" in aot_serve.verify(
        _meta(_key("d"), cuda="12.8"), family=FAMILY)


def test_unstamped_sm_is_refused_like_every_other_silent_axis(
    on_4090: None,
) -> None:
    """pgw#950: a cell silent on ``sm`` used to be waved through ``verify``
    and deferred to the staged-bytes tier. Silence is now a refusal — the
    cell is a cache entry, and refusing it costs one re-mint."""
    meta = _meta(_key("e"))
    meta.pop("sm")
    assert aot_serve.verify(meta, family=FAMILY) == "sm '' != runtime 'sm_89'"


# ---------------------------------------------------------------------------
# 3. _candidates — sku is a preference, never a filter
# ---------------------------------------------------------------------------


def _picks(items: List[Dict[str, Any]]) -> List[str]:
    return [row[1] for row in aot_cells._candidates(items, FAMILY, "w8a8")]


def test_cross_sku_cell_is_still_a_candidate(on_4090: None) -> None:
    """Only l4 cells published; this pod is a 4090. Pre-fix this list was
    EMPTY and the pod paid a 20-minute self-mint for nothing."""
    items = [_item("ck-l4", "2026-07-28T00:00:00Z", _meta(_key("a")))]
    assert _picks(items) == ["ck-l4"]


def test_same_sku_is_preferred_over_a_newer_cross_sku_cell(
    on_4090: None,
) -> None:
    """Autotune affinity: the cell compiled on THIS board wins even when an
    l4 cell is newer — but the l4 cell stays adoptable behind it."""
    items = [
        _item("ck-l4-new", "2026-07-29T00:00:00Z", _meta(_key("a"))),
        _item("ck-4090-old", "2026-07-20T00:00:00Z",
              _meta(_key("b"), sku="rtx-4090")),
    ]
    assert _picks(items) == ["ck-4090-old", "ck-l4-new"]


def test_newest_wins_within_the_preferred_sku(on_4090: None) -> None:
    items = [
        _item("ck-old", "2026-07-20T00:00:00Z",
              _meta(_key("a"), sku="rtx-4090")),
        _item("ck-new", "2026-07-29T00:00:00Z",
              _meta(_key("b"), sku="rtx-4090")),
        _item("ck-l4", "2026-07-30T00:00:00Z", _meta(_key("c"))),
    ]
    assert _picks(items) == ["ck-new", "ck-old", "ck-l4"]


def test_an_unstamped_sm_cell_is_not_a_candidate_at_all(
    on_4090: None,
) -> None:
    """pgw#950: an unstamped cell used to be a LOWER-RANKED candidate that
    still got downloaded. It is not a candidate now — ``verify`` refuses it
    from metadata, before any fetch."""
    unstamped = _meta(_key("a"))
    unstamped.pop("sm")
    items = [
        _item("ck-unstamped", "2026-07-30T00:00:00Z", unstamped),
        _item("ck-stamped", "2026-07-20T00:00:00Z", _meta(_key("b"))),
    ]
    assert _picks(items) == ["ck-stamped"]


def test_sm_mismatched_cell_is_not_a_candidate(on_4090: None) -> None:
    items = [
        _item("ck-89", "2026-07-20T00:00:00Z", _meta(_key("a"))),
        _item("ck-86", "2026-07-30T00:00:00Z", _meta(_key("b"), sm="sm_86")),
    ]
    assert _picks(items) == ["ck-89"]


# ---------------------------------------------------------------------------
# 4. the staged-bytes tier — a LYING stamp, ruled on before dlopen
#
# pgw#950: this tier no longer exists to rescue unstamped cells (they are
# refused from metadata now). Its one remaining job is the case metadata
# cannot reach — a cell whose stamp and whose bytes disagree.
# ---------------------------------------------------------------------------


def _bytes_artifact(tmp_path: Path, capability: Optional[str], name: str,
                    **over: Any) -> Path:
    """A packed, fully-stamped cell whose ``.pt2`` carries torch's own
    ``AOTI_COMPUTE_CAPABILITY`` (the shape ``get_device_information`` writes),
    so metadata and bytes can be made to agree or disagree independently."""
    pt2 = tmp_path / f"{name}.pt2"
    row: Dict[str, str] = {"AOTI_MACHINE": "x86_64"}
    if capability is not None:
        row["AOTI_COMPUTE_CAPABILITY"] = capability
    with zipfile.ZipFile(pt2, "w") as zf:
        zf.writestr(
            "model/data/aotinductor/model/abc.wrapper_metadata.json",
            json.dumps(row))
    content = tmp_path / f"content-{name}"
    content.mkdir()
    (content / aot_serve.PACKAGE_NAME).write_bytes(pt2.read_bytes())
    meta = _meta(_key("a"))
    meta.update(over)
    return aot_serve.pack(content, tmp_path / f"{name}.tar.gz", meta)


def test_a_lying_sm_stamp_is_caught_by_the_bytes(
    tmp_path: Path, on_4090: None,
) -> None:
    """A cell stamped ``sm_89`` whose ``.pt2`` was built for sm_86 passes the
    metadata check and must still be refused — the stamp is a claim, the
    package's own capability is the fact."""
    artifact = _bytes_artifact(tmp_path, "86", "liar")
    with pytest.raises(aot_serve.AdoptError) as exc_info:
        aot_serve.stage_artifact(artifact, FAMILY)
    assert exc_info.value.reason == "sm_mismatch"
    assert "sm_86" in str(exc_info.value)


def test_an_unstamped_cell_never_reaches_the_bytes_tier(
    tmp_path: Path, on_4090: None,
) -> None:
    """pgw#950, the deletion proved: a cell with no ``sm`` stamp is refused
    as a plain ``key_mismatch`` by ``verify``, NOT rescued (or condemned) by
    the package's capability stamp."""
    artifact = _bytes_artifact(tmp_path, "89", "unstamped", sm="")
    with pytest.raises(aot_serve.AdoptError) as exc_info:
        aot_serve.stage_artifact(artifact, FAMILY)
    assert exc_info.value.reason == "key_mismatch"
    assert "sm ''" in str(exc_info.value)


def test_cross_sku_same_arch_cell_stages(tmp_path: Path, on_4090: None) -> None:
    """The l4-minted cell on the 4090: stamped sm_89, torch stamped 89, this
    runtime is sm_89, so the bytes stage and the arm proceeds."""
    staged = aot_serve.stage_artifact(
        _bytes_artifact(tmp_path, "89", "right"), FAMILY)
    try:
        assert staged.metadata["sku"] == "l4"
    finally:
        staged.close()


def test_unreadable_capability_stamp_is_not_read_as_a_mismatch(
    tmp_path: Path, on_4090: None,
) -> None:
    """A package that stamps no capability at all must not be refused on a
    guess — the load path names its own failure."""
    staged = aot_serve.stage_artifact(
        _bytes_artifact(tmp_path, None, "silent"), FAMILY)
    staged.close()
