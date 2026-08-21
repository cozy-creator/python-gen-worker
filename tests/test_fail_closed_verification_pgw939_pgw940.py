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


def test_an_unhashed_download_is_distinguishable_from_a_verified_one(
    tmp_path: Path,
) -> None:
    """Civitai publishes `AutoV2`/`CRC32`/`BLAKE3` without `SHA256` for many large/GGUF files."""
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
    """`if dst.exists() and (not f["size_bytes"] or st_size == size)`."""
    from gen_worker.models import download

    dst = _sized(tmp_path / "big.gguf", 3)
    assert download._civitai_adoptable(
        dst, {"name": "big.gguf", "size_bytes": 0, "sha256": ""}, None,
    ) is None


def test_an_undeclared_size_adopts_against_our_own_completed_manifest(
    tmp_path: Path,
) -> None:
    """The fast path survives where there IS evidence: this directory's own manifest, which is written only after every file completed."""
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


_GIB = 1024 ** 3


def test_no_cuda_still_selects_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """"off" on a CPU host is not a placement claim — there is no card to offload FROM."""
    from gen_worker.models import memory

    monkeypatch.setattr(
        memory, "available_vram",
        lambda *a, **k: memory.VramReading(0.0, memory.VRAM_NO_CUDA))
    assert memory.select_auto_mode(pipeline=object()) == "off"


def test_an_unreadable_card_does_not_select_full_residency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`"off"` means fully resident, no offload at all — the single most memory-hungry rung."""
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
    assert memory.get_available_vram_gb() == 0.0


class _Unsizable:

    def parameters(self, recurse: bool = True) -> Any:
        raise RuntimeError("meta device: parameters are not materialized")

    def buffers(self, recurse: bool = True) -> Any:
        raise RuntimeError("meta device")


def test_an_unmeasurable_module_no_longer_skips_the_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`except Exception: return 0` -> `0 < 1 GiB` -> early return -> the move proceeds."""
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
    """The guard is not turned into a per-move probe: a module it CAN size and that is below the floor still returns without touching the budget."""
    torch = pytest.importorskip("torch")
    from gen_worker import host_move_guard as guard

    called: list[int] = []
    monkeypatch.setattr(
        guard, "_refuse_if_over_budget",
        lambda incoming, **_k: called.append(incoming))
    guard.check_host_ram_move(torch.nn.Linear(4, 4))
    assert called == []
