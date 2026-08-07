"""Every pack refusal must be self-diagnosing (pgw#733).

The first real-GPU mint ran its whole plan and was refused AT PACK for both
functions, and the verbatim reason was LOST — the hub persists no typed worker
events, so a refusal costs one pod run to see.

pgw#1010 deleted three of the five refusals with the thing they diagnosed: the
inductor CAPTURE. `finish_fleet_mint` packed a dynamo cell and its
"captured nothing" / "no fx entries" / "partial capture" gates (and the
`_capture_forensics` report they carried) went with it, because no path packs
one any more. The cubin-completeness gate below is on `pack`, which the local
cell store still uses, and it keeps its census for exactly the original reason.
"""

from __future__ import annotations

import importlib

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
