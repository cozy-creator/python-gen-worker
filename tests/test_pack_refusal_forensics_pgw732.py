"""Every pack refusal must be self-diagnosing (pgw#732).

The first real-GPU mint ran its whole plan and was refused AT PACK for both
functions, and the verbatim reason was LOST — the hub persists no typed worker
events, so a refusal costs one pod run to see. These tests pin that each pack
refusal carries the facts that discriminate its candidate causes, so ONE run
settles which gate fired: where inductor was actually pointed, what DID land in
the capture, whether the fx cache was disabled/bypassed/bundled, and whether
this process compiled at all or reused in-process code (pgw#604's class).
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from gen_worker import compile_cache as cc  # noqa: E402

# The forensics report inductor's cache flags only when inductor is already
# imported (it must never trigger a fresh import — see _inductor_cache_config).
# On a real pod a compile just ran; here we import it explicitly, once.
importlib.import_module("torch._inductor.config")


class _Cfg:
    shapes = [(64, 64)]
    targets = ["transformer"]
    guidance_scales = ()
    regional = False
    lora_bucket = 0


def _capture(tmp_path: Path, *, inductor_files=(), fx_entries=0) -> Path:
    capture = tmp_path / "capture"
    (capture / "inductor").mkdir(parents=True)
    (capture / "triton").mkdir(parents=True)
    for name in inductor_files:
        target = capture / "inductor" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("x")
    for index in range(fx_entries):
        entry = capture / "inductor" / "fxgraph" / f"e{index}"
        entry.mkdir(parents=True)
        (entry / "entry").write_bytes(b"fx")
    return capture


def _assert_discriminating(message: str) -> None:
    """The four facts that separate the candidate causes."""
    for fact in ("capture=", "latched=", "tree=", "inductor="):
        assert fact in message, f"refusal cannot be diagnosed without {fact}: {message}"
    assert "fx_graph_cache=" in message
    assert "force_disable_caches=" in message
    assert "bundle_triton_into_fx_graph_cache=" in message


def test_empty_capture_refusal_names_where_the_compile_went(tmp_path, monkeypatch):
    elsewhere = tmp_path / "elsewhere"
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(elsewhere / "inductor"))
    monkeypatch.setenv("TRITON_CACHE_DIR", str(elsewhere / "triton"))
    capture = _capture(tmp_path)
    with pytest.raises(RuntimeError) as exc:
        cc.finish_fleet_mint(
            object(), _Cfg(), "toyfam", tmp_path / "cell.tar.gz", capture)
    message = str(exc.value)
    assert "captured nothing" in message
    _assert_discriminating(message)
    # The gw#608 latch question is now ANSWERED by evidence, not asked.
    assert "latched=NO" in message
    assert str(elsewhere / "inductor") in message


def test_missing_fx_entries_refusal_counts_what_did_land(tmp_path, monkeypatch):
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(tmp_path / "capture" / "inductor"))
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "capture" / "triton"))
    capture = _capture(tmp_path, inductor_files=("aotautograd/a", "codecache/b"))
    with pytest.raises(RuntimeError) as exc:
        cc.finish_fleet_mint(
            object(), _Cfg(), "toyfam", tmp_path / "cell.tar.gz", capture)
    message = str(exc.value)
    assert "NO FX-graph cache entries" in message
    # "compiled but bypassed the fx cache" vs "never re-compiled" is the whole
    # question, so the census of what DID land is part of the refusal.
    assert "wrote 2 other file(s)" in message
    assert "aotautograd:1" in message and "codecache:1" in message
    _assert_discriminating(message)
    assert "latched=yes" in message


def test_partial_capture_refusal_carries_forensics(tmp_path):
    capture = _capture(tmp_path, fx_entries=1)
    with pytest.raises(RuntimeError) as exc:
        cc.finish_fleet_mint(
            object(), _Cfg(), "toyfam", tmp_path / "cell.tar.gz", capture,
            expected_graphs=4)
    message = str(exc.value)
    assert "holds 1 FX-graph entrie(s)" in message and "compiled 4 graph(s)" in message
    _assert_discriminating(message)


def test_cubin_gate_refusal_carries_its_census(tmp_path):
    root = tmp_path / "cap"
    kernels = root / "triton" / "kern"
    kernels.mkdir(parents=True)
    (kernels / "k1.ptx").write_text("ptx")          # PTX only -> a real gap
    (kernels / "k2.ptx").write_text("ptx")
    (kernels / "k2.cubin").write_bytes(b"\x7fELF")  # complete
    meta = cc.artifact_metadata(
        family="toyfam", source_ref="self-mint", shapes=[(64, 64)],
        targets=["transformer"])
    meta["sm"] = "sm_89"
    with pytest.raises(RuntimeError) as exc:
        cc.pack(root, tmp_path / "cell.tar.gz", meta)
    message = str(exc.value)
    assert "cubin-completeness gate (pgw#698)" in message
    assert "k1.ptx" in message
    # A real PTX exposure and a false gap (cubins bundled into the fx entry
    # instead of written beside the ptx) are indistinguishable without this.
    assert "census: 2 ptx, 1 cubin" in message
    assert "sm=sm_89" in message
    assert "bundle_triton=" in message


def test_forensics_never_raises_on_a_hostile_capture(tmp_path):
    """A diagnostic that can fail is a second lost refusal."""
    assert "capture=" in cc._capture_forensics(tmp_path / "does-not-exist")
    assert "capture=" in cc._capture_forensics(tmp_path, pipe=object())
