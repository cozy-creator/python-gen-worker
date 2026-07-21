"""gw#611: mint -> pack -> FRESH-PROCESS adopt -> warmup must count >=1 hit.

Real cross-process portability over the real primitives (capture_env /
torch.compile / pack / seed_artifact) with real CPU inductor compiles in
subprocesses — the same FxGraphCache lookup surface the fleet uses (apply()
only adds the CUDA-side guard wrapper around the identical compile call).

Live failures this pins (worker logs unreachable on RunPod; these are the
only local reproductions):

- th#954 SDXL second boot: `compile_cell_failed warmups=1, cache_hits=0,
  cache_misses=0` on a cell that SERVED. Measured mechanism (torch 2.13):
  with the AOT autograd cache in bundled mode, an AOT hit loads the
  compiled artifact and the fxgraph counters stay COMPLETELY silent — a
  healthy serving boot reads 0/0 to a proof watching only fxgraph_*, and
  fail-closed bricks the release. The proof now credits `aot_cache_hit`
  as serving evidence.
- B200 0.40.5 (gw#608 residual): AOT enabled on the compile thread with
  ASLR-embedded CUDA-path keys -> 0 hits / 8 real misses. gw#608+dbd7d6b
  pin the AOT layer off fleet-wide; the main test here proves the FX lane
  contract end-to-end and turns red if that regresses.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import threading
from pathlib import Path

import pytest

from gen_worker import compile_cache as cc

pytestmark = pytest.mark.skipif(
    not cc.toolchain_present(), reason="CPU inductor compile needs a C toolchain"
)

_CHILD = textwrap.dedent(
    """
    import json, sys
    from pathlib import Path

    role, workdir, mode = sys.argv[1], Path(sys.argv[2]), sys.argv[3]

    import gen_worker.compile_cache as cc
    import torch

    FAMILY = "gw611-portability"

    def enable_bundled_aot():
        # The pre-gw#608 consumer shape that produced th#954's 0/0: AOT
        # autograd cache live (seed_env/capture_env disabled it; re-enable)
        # in bundled mode, where an AOT hit never consults FxGraphCache.
        import torch._functorch.config as fconf

        fconf.enable_autograd_cache = True
        try:
            fconf._config["enable_autograd_cache"].env_value_force = True
        except Exception:
            pass
        try:
            fconf.bundled_autograd_cache = True
        except Exception:
            pass

    def compile_once():
        torch.manual_seed(0)
        m = torch.nn.Sequential(torch.nn.Linear(16, 16), torch.nn.ReLU())
        m.eval()
        compiled = torch.compile(m, dynamic=False)
        before = cc.inductor_counters()
        with torch.no_grad():
            compiled(torch.ones(2, 16))
        return cc.counters_delta(before, cc.inductor_counters())

    def compile_two_lanes():
        # gw#614: two distinct graphs sharing ONE capture — the t2i lane and
        # the (synthesized-warmup-exercised) edit lane of a family cell.
        torch.manual_seed(0)
        t2i = torch.nn.Sequential(torch.nn.Linear(16, 16), torch.nn.ReLU())
        edit = torch.nn.Sequential(torch.nn.Linear(32, 8), torch.nn.Tanh())
        t2i.eval(); edit.eval()
        ct, ce = torch.compile(t2i, dynamic=False), torch.compile(edit, dynamic=False)
        before = cc.inductor_counters()
        with torch.no_grad():
            ct(torch.ones(2, 16))
            ce(torch.ones(2, 32))
        return cc.counters_delta(before, cc.inductor_counters())

    if role == "mint":
        cap = workdir / "capture"
        cc.capture_env(cap)
        if mode == "bundled_aot":
            enable_bundled_aot()
        delta = compile_two_lanes() if mode == "twolane" else compile_once()
        entries = [p for p in (cap / "inductor").rglob("*") if p.is_file()]
        meta = cc.artifact_metadata(
            family=FAMILY, shapes=((16, 16),), targets=("0",))
        cc.pack(cap, workdir / "cell.tar.gz", meta)
        print(json.dumps({"delta": delta, "fx_entries": len(entries)}))
    else:
        meta = cc.seed_artifact(
            workdir / "cell.tar.gz", FAMILY, cache_dir=workdir / "adopt-cache")
        if mode == "bundled_aot":
            enable_bundled_aot()
        delta = compile_two_lanes() if mode == "twolane" else compile_once()
        print(json.dumps({"delta": delta, "family": str(meta.get("family"))}))
    """
)


def _run_child(tmp_path: Path, role: str, *, mode: str = "prod") -> dict:
    script = tmp_path / "child.py"
    script.write_text(_CHILD)
    env = dict(os.environ)
    env.pop("TORCHINDUCTOR_CACHE_DIR", None)
    env.pop("TRITON_CACHE_DIR", None)
    if mode == "bundled_aot":
        env.pop("TORCHINDUCTOR_AUTOGRAD_CACHE", None)
    else:
        # The production entrypoint contract (gw#608/dbd7d6b): set before any
        # torch import so every thread and compile subprocess sees it.
        env["TORCHINDUCTOR_AUTOGRAD_CACHE"] = "0"
    proc = subprocess.run(
        [sys.executable, str(script), role, str(tmp_path), mode],
        capture_output=True, text=True, timeout=600, env=env,
        cwd=str(Path(__file__).resolve().parents[1]),
    )
    assert proc.returncode == 0, (
        f"{role}(mode={mode}) child failed:\n{proc.stdout}\n{proc.stderr}"
    )
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_fresh_process_adopt_serves_from_the_packed_cell(tmp_path):
    """The gw#611 task-list repro: a packed cell must serve the adopting
    process's first compile as FX cache hits — zero recompiles."""
    mint = _run_child(tmp_path, "mint")
    assert mint["fx_entries"] > 0, "mint capture must hold FX entries"
    assert mint["delta"].get("fxgraph_cache_miss", 0) >= 1, (
        "the mint compile is cold by construction"
    )

    adopt = _run_child(tmp_path, "adopt")
    hits = adopt["delta"].get("fxgraph_cache_hit", 0)
    misses = adopt["delta"].get("fxgraph_cache_miss", 0)
    assert hits >= 1 and misses == 0, (
        f"fresh-process adopt must SERVE from the cell (delta={adopt['delta']})"
        " — the gw#611 release-bricking shape is hits=0"
    )


def test_union_cell_serves_both_lanes_in_a_fresh_process(tmp_path):
    """gw#614: a family cell minted with BOTH lanes' graphs (the synthesized
    media-variant warmup exercises the sibling lane, so the union publishes)
    must serve BOTH lanes as FX hits in a fresh adopting process — zero
    recompiles. Pre-gw#614 mints packed lane-1-only cells; this is the
    complete-cell counterpart of the gw#611 single-lane contract."""
    mint = _run_child(tmp_path, "mint", mode="twolane")
    assert mint["delta"].get("fxgraph_cache_miss", 0) >= 2, (
        "the two-lane mint cold-compiles one graph per lane"
    )
    assert mint["fx_entries"] > 0

    adopt = _run_child(tmp_path, "adopt", mode="twolane")
    hits = adopt["delta"].get("fxgraph_cache_hit", 0)
    misses = adopt["delta"].get("fxgraph_cache_miss", 0)
    assert hits >= 2 and misses == 0, (
        f"the union cell must serve BOTH lanes cross-process "
        f"(delta={adopt['delta']}) — one lane missing is the gw#611 qwen "
        "hits=1/misses=1 release-bricker"
    )


def test_bundled_aot_serving_is_visible_to_the_proof(tmp_path):
    """The measured th#954 0/0 mechanism: bundled AOT serves the adopt with
    fxgraph counters fully silent. The counters surface must report the AOT
    hit so the warmup proof credits a SERVING boot instead of bricking the
    release (red before the gw#611 aot_cache_hit accounting)."""
    _run_child(tmp_path, "mint", mode="bundled_aot")
    adopt = _run_child(tmp_path, "adopt", mode="bundled_aot")
    delta = adopt["delta"]
    assert delta.get("fxgraph_cache_hit", 0) == 0, (
        f"expected the bundled-AOT consumer to bypass FxGraphCache entirely "
        f"(the 0/0 shape); got {delta} — torch behavior changed, re-derive "
        "the gw#611 mechanism"
    )
    assert delta.get("aot_cache_hit", 0) >= 1, (
        f"a bundled-AOT-served adopt must be visible as aot_cache_hit "
        f"(delta={delta}); invisible serving is the th#954 release-bricker"
    )


def test_guard_wrapper_credits_aot_layer_hits(monkeypatch):
    """Unit revert-turns-red for the accounting fix: a wrapped call served
    by the AOT layer (fxgraph silent) must count as a cache hit on the
    guard signal — the executor's proof reads exactly this counter."""
    ticks = iter([
        {"fxgraph_cache_hit": 0, "fxgraph_cache_miss": 0,
         "fxgraph_cache_bypass": 0, "aot_cache_hit": 0, "aot_cache_miss": 0},
        {"fxgraph_cache_hit": 0, "fxgraph_cache_miss": 0,
         "fxgraph_cache_bypass": 0, "aot_cache_hit": 1, "aot_cache_miss": 0},
    ])
    monkeypatch.setattr(cc, "inductor_counters", lambda: next(ticks))
    signal = {
        "callback": None, "lock": threading.Lock(),
        "successful_calls": 0, "cache_hits": 0, "cache_misses": 0,
    }
    wrapped = cc._guarded(
        lambda: "eager", lambda: "compiled", "transformer",
        failure_signal=signal,
    )
    assert wrapped() == "compiled"
    assert signal["successful_calls"] == 1
    assert signal["cache_hits"] == 1, (
        "AOT-layer-served call must prove (th#954: a serving cell read as "
        "hits=0 and fail-closed the release)"
    )
    assert signal["cache_misses"] == 0
