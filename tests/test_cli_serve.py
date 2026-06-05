"""``gen-worker serve`` + ``invoke`` — collapsed integration suite (floor 15-18).

  15. REAL unix-socket round-trip via a `serve` SUBPROCESS + the `invoke` client,
      SIGINT teardown removes the socket, AND `serve </dev/null` (non-TTY stdin)
      stays alive (the stdin-EOF regression).
  16. `serve --function NAME` boots ONLY the hosting class; `--list-functions`
      lists routable names WITHOUT booting (no setup / no model load).
  17. Lazy setup: serve sets up a class on FIRST invoke (not at boot), only the
      invoked class's siblings stay cold; `--eager` sets up at boot.
  18. `invoke` payload forms (inline / @file / - stdin) + the no-serve-running
      error.

The socket transport runs `serve` as a real subprocess (it blocks + owns SIGINT)
against the on-disk marco-polo example. The boot/dispatch + filter tests drive
the real `_Endpoint` / candidate-filter helpers directly.
"""

from __future__ import annotations

import io
import json
import signal
import subprocess
import sys
import time
import types
from pathlib import Path

import msgspec
import pytest

import gen_worker.cli as cli
import gen_worker.cli.serve as serve_mod
from gen_worker import RequestContext, inference, invocable
from gen_worker.cli import run as run_mod
from gen_worker.models.cache import ModelCache


class _In(msgspec.Struct):
    text: str = ""


class _Out(msgspec.Struct):
    response: str
    setup_calls: int = 0


_EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "examples" / "marco-polo"


# --------------------------------------------------------------------------- #
# 17. Lazy vs eager setup + dispatch correctness (boot the real _Endpoint)      #
# --------------------------------------------------------------------------- #


def _multi_class_module(name: str, setups: dict) -> types.ModuleType:
    mod = types.ModuleType(name)

    @inference()
    class Alpha:
        def setup(self) -> None:
            setups["alpha"] = setups.get("alpha", 0) + 1

        @invocable(name="do_alpha")
        def go(self, ctx: RequestContext, data: _In) -> _Out:
            return _Out(response="a", setup_calls=setups.get("alpha", 0))

    @inference()
    class Beta:
        def setup(self) -> None:
            setups["beta"] = setups.get("beta", 0) + 1

        @invocable(name="do_beta")
        def go(self, ctx: RequestContext, data: _In) -> _Out:
            return _Out(response="b")

    Alpha.__module__ = name
    Beta.__module__ = name
    mod.Alpha = Alpha
    mod.Beta = Beta
    sys.modules[name] = mod
    return mod


def test_lazy_setup_on_first_invoke_only_invoked_class_and_eager_at_boot() -> None:
    setups: dict = {}
    mod = _multi_class_module("_test_serve_lazy", setups)
    candidates = run_mod.discover_candidates(mod)

    ep = serve_mod._Endpoint(offline=False, allow_publish=False)
    ep.boot(candidates)
    assert sorted(ep.function_names()) == ["do_alpha", "do_beta"]
    assert setups == {}  # LAZY: boot indexes but does NOT setup

    # First invoke sets up ONLY Alpha; Beta (never called) stays cold. Warm after.
    for _ in range(2):
        env = ep.dispatch("do_alpha", {"text": "x"})
        assert env["ok"] is True
        result = [e for e in env["events"] if e["event"] == "result"][0]
        assert result["value"]["response"] == "a" and result["value"]["setup_calls"] == 1
    assert setups == {"alpha": 1}  # warm: setup ran once; Beta never loaded

    # Unknown function -> not_found envelope.
    assert ep.dispatch("nope", {})["error"]["kind"] == "not_found"
    ep.shutdown()

    # --eager: setup runs at boot, before any dispatch.
    setups2: dict = {}
    mod2 = _multi_class_module("_test_serve_eager", setups2)
    ep2 = serve_mod._Endpoint(offline=False, allow_publish=False)
    ep2.boot(run_mod.discover_candidates(mod2), eager=True)
    assert setups2 == {"alpha": 1, "beta": 1}
    ep2.shutdown()


# --------------------------------------------------------------------------- #
# ModelCache-backed residency: serve drives the PRODUCTION ModelCache for       #
# model placement (demote/promote/evict), not a parallel warm-only path.        #
# Uses fake pipeline-shaped objects + a tiny-VRAM ModelCache so it runs on CPU.  #
# --------------------------------------------------------------------------- #


class _FakePipeline:
    """A diffusers-pipeline duck type: ``.to()`` records device moves and it
    exposes a ``unet`` slot so ``_find_pipeline_object`` recognizes it."""

    def __init__(self, size_gb: float) -> None:
        self.size_gb = size_gb
        self.device = "cpu"
        self.moves: list = []
        self.unet = object()  # makes _looks_like_pipeline() true

    def to(self, device: str):  # noqa: D401
        self.device = device
        self.moves.append(device)
        return self


def _two_model_module(name: str, setups: dict, sizes: dict) -> types.ModuleType:
    mod = types.ModuleType(name)

    @inference()
    class Big:
        def setup(self) -> None:
            setups["big"] = setups.get("big", 0) + 1
            self.pipeline = _FakePipeline(sizes["big"]).to("cuda")

        @invocable(name="gen_big")
        def go(self, ctx: RequestContext, data: _In) -> _Out:
            return _Out(response="big", setup_calls=setups.get("big", 0))

    @inference()
    class Small:
        def setup(self) -> None:
            setups["small"] = setups.get("small", 0) + 1
            self.pipeline = _FakePipeline(sizes["small"]).to("cuda")

        @invocable(name="gen_small")
        def go(self, ctx: RequestContext, data: _In) -> _Out:
            return _Out(response="small", setup_calls=setups.get("small", 0))

    Big.__module__ = name
    Small.__module__ = name
    mod.Big = Big
    mod.Small = Small
    sys.modules[name] = mod
    return mod


def test_serve_drives_modelcache_demote_promote_evict(monkeypatch) -> None:
    """Over-subscribe a tiny-VRAM ModelCache with two models, then invoke
    big -> small -> big and assert the cache demoted/promoted them (setup runs
    ONCE per model; the cache moves them VRAM<->CPU thereafter)."""
    setups: dict = {}
    sizes = {"big": 6.0, "small": 2.0}  # 8 > 5GB budget => can't both reside
    mod = _two_model_module("_test_serve_cache", setups, sizes)
    candidates = run_mod.discover_candidates(mod)

    # estimate_pipeline_size_gb has no torch here — feed it the fake size.
    monkeypatch.setattr(
        serve_mod, "_estimate_size_gb",
        lambda pipe: float(getattr(pipe, "size_gb", 5.0)),
    )

    ep = serve_mod._Endpoint(offline=False, allow_publish=False)
    ep.boot(candidates)
    # Tiny VRAM budget so the two models over-subscribe (forces eviction).
    ep._model_cache = ModelCache(max_vram_gb=5.0, model_cache_dir="/tmp/_gw_cache_test")
    cache = ep._model_cache
    big_id = ep._model_id_by_inst[id(ep.functions["gen_big"].instance)]
    small_id = ep._model_id_by_inst[id(ep.functions["gen_small"].instance)]

    # 1) big: cold setup + registered VRAM-resident.
    assert ep.dispatch("gen_big", {"text": "x"})["ok"] is True
    assert cache.residency_tier(big_id) == "VRAM"
    assert cache.get_vram_models() == [big_id]
    assert setups == {"big": 1}

    # 2) small: cold setup; cache makes room => big DEMOTED to the CPU-RAM tier.
    assert ep.dispatch("gen_small", {"text": "x"})["ok"] is True
    assert cache.residency_tier(small_id) == "VRAM"
    assert cache.residency_tier(big_id) == "RAM"   # demoted, NOT re-setup
    big_pipe = ep._pipeline_by_inst[id(ep.functions["gen_big"].instance)]
    assert big_pipe.moves[-1] == "cpu"             # real .to("cpu") demote
    assert setups == {"big": 1, "small": 1}

    # 3) big AGAIN: promoted RAM->VRAM (small demoted), setup NOT re-run.
    assert ep.dispatch("gen_big", {"text": "x"})["ok"] is True
    assert cache.residency_tier(big_id) == "VRAM"  # promoted back
    assert cache.residency_tier(small_id) == "RAM"  # small demoted to make room
    assert big_pipe.moves[-1] == "cuda"            # PCIe swap-in, not reload
    assert setups == {"big": 1, "small": 1}        # setup() ran ONCE per model
    ep.shutdown()


# --------------------------------------------------------------------------- #
# 16. --function boots only the hosting class; --list-functions never boots     #
# --------------------------------------------------------------------------- #


def test_filter_by_function_and_list_functions_without_booting(capsys) -> None:
    setups: dict = {}
    name = "_test_serve_filter"
    mod = types.ModuleType(name)

    @inference()
    class Alpha:
        def setup(self) -> None:
            setups["Alpha"] = setups.get("Alpha", 0) + 1

        @inference.function(name="alpha_one")
        def alpha_one(self, ctx: RequestContext, data: _In) -> _Out:
            return _Out(response="a1")

        @inference.function(name="alpha_two")
        def alpha_two(self, ctx: RequestContext, data: _In) -> _Out:
            return _Out(response="a2")

    @inference()
    class Beta:
        def setup(self) -> None:
            setups["Beta"] = setups.get("Beta", 0) + 1

        @inference.function(name="beta_one")
        def beta_one(self, ctx: RequestContext, data: _In) -> _Out:
            return _Out(response="b1")

    Alpha.__module__ = name
    Beta.__module__ = name
    mod.Alpha = Alpha
    mod.Beta = Beta
    sys.modules[name] = mod
    candidates = run_mod.discover_candidates(mod)

    # --function alpha_one keeps the WHOLE Alpha class (both fns), drops Beta.
    filtered = serve_mod._filter_candidates_by_function(candidates, ["alpha_one"])
    assert sorted(c.fn_name for c in filtered) == ["alpha_one", "alpha_two"]
    assert all(c.cls.__name__ == "Alpha" for c in filtered)
    ep = serve_mod._Endpoint(offline=False, allow_publish=False)
    ep.boot(filtered)
    assert ep.function_names() == ["alpha_one", "alpha_two"]
    ep.dispatch("alpha_one", {"text": "x"})
    assert setups == {"Alpha": 1}  # only Alpha; Beta never set up
    ep.shutdown()

    # Unknown --function name errors.
    with pytest.raises(run_mod._UsageError, match="not found"):
        serve_mod._filter_candidates_by_function(candidates, ["nope"])

    # --list-functions prints names + hosting class and boots NOTHING.
    setups.clear()
    rc = cli.main(["serve", "--list-functions", "--module", name])
    assert rc == 0
    out = capsys.readouterr().out
    assert "alpha_one" in out and "beta_one" in out and "Alpha" in out and "Beta" in out
    assert setups == {}


# --------------------------------------------------------------------------- #
# 18. invoke payload forms + no-serve error                                    #
# --------------------------------------------------------------------------- #


def test_invoke_payload_forms_and_no_serve_error(tmp_path, monkeypatch, capsys) -> None:
    from gen_worker.cli import invoke as invoke_mod

    assert invoke_mod._read_payload('{"text":"marco"}') == {"text": "marco"}  # inline
    p = tmp_path / "req.json"
    p.write_text(json.dumps({"text": "marco"}))
    assert invoke_mod._read_payload(f"@{p}") == {"text": "marco"}  # @file
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps({"text": "marco"})))
    assert invoke_mod._read_payload("-") == {"text": "marco"}  # - stdin
    assert invoke_mod._read_payload("") == {}  # empty -> {}

    # invoke with no serve running -> clear error + exit 2.
    rc = cli.main(["invoke", "marco_polo", "{}", "--socket", str(tmp_path / "absent.sock")])
    assert rc == 2
    assert "serve" in capsys.readouterr().err.lower()


# --------------------------------------------------------------------------- #
# 15. REAL socket round-trip + SIGINT teardown + non-TTY stdin stays alive      #
# --------------------------------------------------------------------------- #


def _wait_for_socket(sock: Path, proc: subprocess.Popen, timeout: float = 10.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if sock.exists():
            return
        if proc.poll() is not None:
            raise AssertionError(
                f"serve exited early (rc={proc.returncode}): "
                f"{proc.stderr.read() if proc.stderr else ''}"
            )
        time.sleep(0.05)
    raise AssertionError("serve never created the socket")


def _stop_serve(proc: subprocess.Popen) -> None:
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5.0)


@pytest.mark.skipif(
    not (_EXAMPLE_DIR / "endpoint.toml").exists(),
    reason="marco-polo example not present",
)
@pytest.mark.parametrize("stdin_mode", ["no-stdin", "devnull"])
def test_socket_roundtrip_sigint_teardown_and_nontty_stdin(tmp_path, capsys, stdin_mode) -> None:
    """A `serve` subprocess (real unix socket) answers an `invoke`, then SIGINT
    tears it down + removes the socket. The `devnull` case is the stdin-EOF
    regression: `serve </dev/null` (non-TTY, immediate EOF, NO --no-stdin) must
    keep serving instead of dying before any invoke can connect."""
    sock = tmp_path / "rt.sock"
    argv = [sys.executable, "-m", "gen_worker.cli", "serve", "--socket", str(sock)]
    kwargs: dict = dict(cwd=str(_EXAMPLE_DIR), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if stdin_mode == "no-stdin":
        argv.append("--no-stdin")
    else:
        kwargs["stdin"] = subprocess.DEVNULL  # immediate EOF — the footgun trigger

    proc = subprocess.Popen(argv, **kwargs)
    try:
        _wait_for_socket(sock, proc)       # would fail if serve died on stdin EOF
        assert proc.poll() is None         # still alive
        rc = cli.main(["invoke", "marco_polo", json.dumps({"text": "marco"}), "--socket", str(sock)])
        assert rc == 0
        assert json.loads(capsys.readouterr().out.strip())["response"] == "polo"
    finally:
        _stop_serve(proc)
    assert proc.returncode == 0
    assert not sock.exists()  # SIGINT teardown removed the socket


# --------------------------------------------------------------------------- #
# serve --vram-budget sizes the in-process ModelCache to a host allotment (#347)
# --------------------------------------------------------------------------- #


def test_vram_budget_sizes_model_cache(monkeypatch) -> None:
    captured = {}

    class _FakeCache:
        def __init__(self, max_vram_gb=None, **kw):
            captured["max_vram_gb"] = max_vram_gb

    monkeypatch.setattr(serve_mod, "ModelCache", _FakeCache)
    # Explicit budget -> ModelCache sized to it, regardless of host VRAM.
    c = serve_mod._build_model_cache(vram_budget_gb=4.0)
    assert isinstance(c, _FakeCache)
    assert captured["max_vram_gb"] == 4.0

    # _Endpoint threads the budget through.
    captured.clear()
    serve_mod._Endpoint(offline=False, allow_publish=False, vram_budget_gb=6.0)
    assert captured["max_vram_gb"] == 6.0
