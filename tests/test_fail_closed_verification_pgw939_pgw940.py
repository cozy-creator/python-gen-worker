"""pgw#939 + pgw#940 — missing evidence is a refusal, never a skipped check.

**pgw#939 (supply chain).** `if expected { compare }` — so an absent expected
value ADMITS. The artifact in every case is bytes that will subsequently be
safetensors-loaded or armed as a compiled graph, and `DESIGN-RULINGS.md` §1.22
decides all of them the same way: missing evidence is an integrity verdict,
not a disabled check.

**pgw#940 (measurement).** One question — *how much VRAM is free?* — answered
four times, with `0` meaning both "no card" and "the probe raised", and the
code reading the shared zero as the permissive case. §1.22 again, and the
asymmetry decides it: admitting on an unreadable measurement OOMs paid tenant
work, refusing costs a rung of performance.

TCG now owns compiled-graph declaration and admission, so the first rows use
its closed public declaration instead of the deleted worker verifier. The
remaining rows preserve the independent download and measurement checks.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from gen_worker._vendor.torchcg import (
    CallIngress,
    CallInput,
    DeclarationError,
    GraphSpecializationDeclaration,
)


def _declaration(
    *,
    range_digest: str | None = None,
    graph_witness: str = "a" * 16,
) -> GraphSpecializationDeclaration:
    ingress = CallIngress(
        parameters=("sample",),
        flat_arity=1,
        inputs=(CallInput(
            "sample", 0, "sample", 0, (), "sample", "bfloat16",
            (1, 4, "s0", 64),
        ),),
        symbols=(("s0", (16, 160)),),
    )
    return GraphSpecializationDeclaration(
        name="unet/h=64",
        target="unet",
        graph={
            "v": 4,
            "constant_fqns": ["w.weight"],
            "ingress": ingress.as_dict(),
        },
        graph_witness=graph_witness,
        range_digest=ingress.digest() if range_digest is None else range_digest,
        specialization_dims=(("h", 64),),
    )


def test_tcg_declaration_is_the_closed_identity_control() -> None:
    declaration = _declaration()

    assert declaration.range_digest == CallIngress.from_graph(
        declaration.graph
    ).digest()
    assert len(declaration.specialization_hash) == 16
    assert "family" not in declaration.facts()


def test_tcg_refuses_an_absent_range_digest() -> None:
    with pytest.raises(DeclarationError, match="range_digest"):
        _declaration(range_digest="")


def test_tcg_refuses_a_range_digest_that_does_not_restate_ingress() -> None:
    with pytest.raises(DeclarationError, match="does not restate"):
        _declaration(range_digest="0" * 32)


def test_tcg_refuses_an_absent_graph_witness() -> None:
    with pytest.raises(DeclarationError, match="graph_witness"):
        _declaration(graph_witness="")


# ---------------------------------------------------------------------------
# pgw#939 "also in scope" — the cubin arch gate vanished per kernel
# ---------------------------------------------------------------------------


# pgw#1181 REMOVED the four `_ptx_jit_gaps` rows (pgw#939 site 2: an unreadable
# cubin must be a GAP, not a skip). The gate they cover ran inside
# `compile_cache.pack`, over the inductor/triton capture a
# `torch-inductor-cache` compiled graph was built from — a completeness check on kernels
# packed into that artifact. The format has had no writer since pgw#1178 and is
# deleted here, so there is no capture to walk and no pack to refuse.
#
# The pgw#939 PRINCIPLE this file exists for is untouched and the other 19 rows
# still assert it on live sites: absent evidence is an integrity verdict, never
# a disabled check. On the exported lane the same principle is enforced by the
# key — a compiled graph that cannot state an axis has no identity and never resolves.


# ---------------------------------------------------------------------------
# pgw#939 §1-2 — civitai downloads
# ---------------------------------------------------------------------------


def test_an_unhashed_download_is_distinguishable_from_a_verified_one(
    tmp_path: Path,
) -> None:
    """Civitai publishes `AutoV2`/`CRC32`/`BLAKE3` without `SHA256` for many
    large/GGUF files. The manifest recorded `"sha256": ""` for those, with no
    marker separating "hash matched" from "no hash was available" — so every
    downstream reader saw a completed, verified download.

    Refusing is not available (ingesting those files is what the lane is for),
    so the acceptance is that the two states stop looking alike.
    """
    from gen_worker.models import download

    verified = download._civitai_adoptable(
        _sized(tmp_path / "a.bin", 10),
        {"name": "a.bin", "size_bytes": 10, "sha256": "ab" * 32}, None,
    )
    assert verified is not None and verified["sha256_source"] == "civitai"

    unhashed = download._civitai_adoptable(
        _sized(tmp_path / "b.bin", 10),
        {"name": "b.bin", "size_bytes": 10, "sha256": ""}, None,
    )
    assert unhashed is not None and unhashed["sha256_source"] == "unverified"
    assert verified["sha256_source"] != unhashed["sha256_source"]


def _sized(path: Path, n: int) -> Path:
    path.write_bytes(b"\0" * n)
    return path


def test_an_undeclared_size_no_longer_adopts_whatever_is_at_the_path(
    tmp_path: Path,
) -> None:
    """`if dst.exists() and (not f["size_bytes"] or st_size == size)`. The
    left half of the `or` short-circuits: with no declared size ANY file at
    that path was adopted as a complete download — no size, no hash, no read.
    A truncated prior attempt is exactly that state."""
    from gen_worker.models import download

    dst = _sized(tmp_path / "big.gguf", 3)  # a stub, not the model
    assert download._civitai_adoptable(
        dst, {"name": "big.gguf", "size_bytes": 0, "sha256": ""}, None,
    ) is None


def test_an_undeclared_size_adopts_against_our_own_completed_manifest(
    tmp_path: Path,
) -> None:
    """The fast path survives where there IS evidence: this directory's own
    manifest, which is written only after every file completed."""
    from gen_worker.models import download

    dst = _sized(tmp_path / "big.gguf", 3)
    prior = {"name": "big.gguf", "size_bytes": 3, "sha256": "cd" * 32,
             "sha256_source": "observed"}
    row = download._civitai_adoptable(
        dst, {"name": "big.gguf", "size_bytes": 0, "sha256": ""}, prior,
    )
    assert row is not None and row["sha256"] == "cd" * 32
    assert row["sha256_source"] == "observed"


def test_a_prior_manifest_that_disagrees_with_the_disk_forces_a_redownload(
    tmp_path: Path,
) -> None:
    from gen_worker.models import download

    dst = _sized(tmp_path / "big.gguf", 3)
    prior = {"name": "big.gguf", "size_bytes": 999, "sha256": "cd" * 32}
    assert download._civitai_adoptable(
        dst, {"name": "big.gguf", "size_bytes": 0, "sha256": ""}, prior,
    ) is None


def test_a_declared_size_mismatch_still_redownloads(tmp_path: Path) -> None:
    from gen_worker.models import download

    dst = _sized(tmp_path / "m.safetensors", 5)
    assert download._civitai_adoptable(
        dst, {"name": "m.safetensors", "size_bytes": 10, "sha256": ""}, None,
    ) is None


def test_a_torn_prior_manifest_is_no_evidence(tmp_path: Path) -> None:
    from gen_worker.models import download

    m = tmp_path / ".civitai.json"
    m.write_text("{not json", encoding="utf-8")
    assert download._civitai_prior_manifest(m) == {}
    m.write_text(json.dumps({"files": "nope"}), encoding="utf-8")
    assert download._civitai_prior_manifest(m) == {}


# ---------------------------------------------------------------------------
# pgw#940 §1 — DELETED WITH ITS SUBJECT
# ---------------------------------------------------------------------------
#
# pgw#940 §1 covered `entry_workers`' DEVICE bound: an unreadable card used to
# share the "0 free VRAM" branch with an absent one and licensed 8 concurrent
# compile children on a card nobody could read. The fix was right and the rows
# were real; §4.33 then deleted the bound they guarded. K is f(cores, one
# measured child RSS) — free VRAM is not divided, sampled, or read at all, so
# there is no zero left to misread and `DeviceProbeError`, `device_facts` and
# `CardCensus` are gone with it.
#
# The behaviour §1 protected is covered by its ABSENCE: `entry_workers` takes
# no device reading, and `test_aot_compile_pool_pgw809
# .test_the_width_and_its_inputs_ride_the_telemetry` fails if a device term
# reappears in the width record. pgw#940 §2 below is a DIFFERENT site (the
# offload-rung selector in `models.memory`) and is untouched.

_GIB = 1024 ** 3


# ---------------------------------------------------------------------------
# pgw#940 §2 — zero free VRAM selected the most memory-hungry rung
# ---------------------------------------------------------------------------


def test_no_cuda_still_selects_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """"off" on a CPU host is not a placement claim — there is no card to
    offload FROM. Unchanged."""
    from gen_worker.models import memory

    monkeypatch.setattr(
        memory, "available_vram",
        lambda *a, **k: memory.VramReading(0.0, memory.VRAM_NO_CUDA))
    assert memory.select_auto_mode(pipeline=object()) == "off"


def test_an_unreadable_card_does_not_select_full_residency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`"off"` means fully resident, no offload at all — the single most
    memory-hungry rung. On a GPU host whose probe failed, a pipeline that
    needed `group_offload` was loaded fully resident and OOMed during load.
    The unknown-model-size branch a few lines below has always descended to
    `group_offload`; only the unknown-free-VRAM branch opened."""
    from gen_worker.models import memory

    monkeypatch.setattr(
        memory, "available_vram",
        lambda *a, **k: memory.VramReading(0.0, memory.VRAM_UNREADABLE))
    assert memory.select_auto_mode(pipeline=object()) == "group_offload"


def test_the_reading_carries_its_zero_cause(monkeypatch: pytest.MonkeyPatch) -> None:
    from gen_worker.models import memory

    class _NoCuda:
        @staticmethod
        def is_available() -> bool:
            return False

    import sys
    import types

    fake = types.SimpleNamespace(cuda=_NoCuda())
    monkeypatch.setitem(sys.modules, "torch", fake)
    assert memory.available_vram().reason == memory.VRAM_NO_CUDA

    class _Broken:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def mem_get_info(_i: int) -> tuple:
            raise RuntimeError("CUDA context lost")

    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(cuda=_Broken()))
    reading = memory.available_vram()
    assert reading.reason == memory.VRAM_UNREADABLE and not reading.measured
    # The reporting shape keeps collapsing them; that is what it is for.
    assert memory.get_available_vram_gb() == 0.0


# ---------------------------------------------------------------------------
# pgw#940 §3 — a saturated card read as an empty one
# ---------------------------------------------------------------------------


# pgw#1373: `test_a_saturated_card_no_longer_reads_as_empty` deleted with the
# module it drove (executor). `_Unsizable` below is NOT part of that case — it
# is the shared fixture the two host-move-guard tests use, and it sat between
# them.


class _Unsizable:
    """The shape the guard no-opped on: `parameters()` raises, as it does for
    meta-device modules, accelerate-hooked modules mid-dispatch, and custom
    `nn.Module` subclasses that override `parameters`."""

    def parameters(self, recurse: bool = True) -> Any:
        raise RuntimeError("meta device: parameters are not materialized")

    def buffers(self, recurse: bool = True) -> Any:
        raise RuntimeError("meta device")


def test_an_unmeasurable_module_no_longer_skips_the_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`except Exception: return 0` -> `0 < 1 GiB` -> early return -> the move
    proceeds. The guard exists solely to refuse a `.to("cpu")` that gets the
    container cgroup-SIGKILLed: "no exception, no finally, process gone
    mid-instruction"."""
    from gen_worker import host_move_guard as guard

    seen: list[int] = []

    def _record(incoming: int, **_k: Any) -> None:
        seen.append(incoming)

    monkeypatch.setattr(guard, "_refuse_if_over_budget", _record)
    guard.check_host_ram_move(_Unsizable())
    assert seen == [guard._MIN_GUARDED_GIB * guard._GIB], (
        "an unsizable module returned without the budget ever being consulted")


def test_a_small_measurable_move_still_short_circuits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The guard is not turned into a per-move probe: a module it CAN size
    and that is below the floor still returns without touching the budget."""
    torch = pytest.importorskip("torch")
    from gen_worker import host_move_guard as guard

    called: list[int] = []
    monkeypatch.setattr(
        guard, "_refuse_if_over_budget",
        lambda incoming, **_k: called.append(incoming))
    guard.check_host_ram_move(torch.nn.Linear(4, 4))
    assert called == []
