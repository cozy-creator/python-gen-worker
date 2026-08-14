"""pgw#622: eager-while-compiling with hot-swap.

A novel input signature serves eager immediately, warms the compiled path
in a background thread, and atomically hot-swaps; tight headroom does NOT
degrade to a sequential compile (pgw#1215 step 4 deleted the ungated arm that
could, and the warm thread ensures headroom inside its own turn); a
preexisting warm signature (cell/boot-warmed) short-circuits everything; the
completed warm republishes the grown cell.
"""

from __future__ import annotations

import contextlib
import tarfile
import threading
import time

import pytest
import torch

from gen_worker import compile_cache as cc
from gen_worker import fleet_cells, hot_swap


def _gate(kind: str):
    """The executor's background turn, as a test double.

    pgw#1215 step 4: an ungated Router is no longer a mode — `Router.enable`
    and every warm-job enqueue refuse it typed — so a test that wants
    concurrent routing wires the gate exactly as `Executor._wire_turn_gate`
    does in production.
    """
    return contextlib.nullcontext()


def _router(*, fail_closed: bool = False) -> "hot_swap.Router":
    router = hot_swap.Router(fail_closed=fail_closed)
    router.set_turn_gate(_gate)
    return router


def _wait(predicate, timeout=30.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def _signal(router):
    return {
        "callback": None,
        "lock": threading.Lock(),
        "successful_calls": 0,
        "cache_hits": 0,
        "cache_misses": 0,
        "router": router,
    }


def _wrapper(original, compiled, router, label="toy"):
    return cc._guarded(
        original, compiled, label, failure_signal=_signal(router))


# ---------------------------------------------------------------------------
# Signatures
# ---------------------------------------------------------------------------


def test_signature_keys_on_tensor_shape_dtype_and_scalars():
    a = hot_swap.signature((torch.zeros(2, 8),), {"flag": True})
    b = hot_swap.signature((torch.ones(2, 8),), {"flag": True})
    c = hot_swap.signature((torch.zeros(3, 8),), {"flag": True})
    d = hot_swap.signature((torch.zeros(2, 8),), {"flag": False})
    e = hot_swap.signature((torch.zeros(2, 8, dtype=torch.float64),), {"flag": True})
    assert a == b  # values never enter tensor identity
    assert a != c and a != d and a != e


def test_signature_recurses_containers():
    a = hot_swap.signature((), {"cond": {"ids": torch.zeros(1, 6)}})
    b = hot_swap.signature((), {"cond": {"ids": torch.zeros(1, 7)}})
    assert a != b


def test_dummy_batch_is_zero_filled_and_structure_preserving():
    args = (torch.full((2, 3), 7.0), [torch.ones(4)], "keep")
    dummy = hot_swap._dummy_value(args)
    assert torch.equal(dummy[0], torch.zeros(2, 3))
    assert torch.equal(dummy[1][0], torch.zeros(4))
    assert dummy[2] == "keep"
    assert hot_swap.signature(args, {}) == hot_swap.signature(dummy, {})


# ---------------------------------------------------------------------------
# Routing: miss -> eager + background compile + atomic swap
# ---------------------------------------------------------------------------


def test_miss_serves_eager_then_hot_swaps():
    release = threading.Event()
    bg_calls = []

    def original(x):
        return "eager"

    def compiled(x):
        bg_calls.append(x)
        assert release.wait(30)
        return "compiled"

    router = _router()
    assert router.enable()
    wrapper = _wrapper(original, compiled, router)

    x = torch.full((2, 8), 3.0)
    assert wrapper(x) == "eager"          # never blocks on the compile
    assert wrapper(x) == "eager"          # followups at the shape stay eager
    assert _wait(lambda: bg_calls)
    assert torch.equal(bg_calls[0], torch.zeros(2, 8))  # dummy, not the batch
    release.set()
    sig = ("toy", hot_swap.signature((x,), {}))
    assert _wait(lambda: sig in router.warm)
    assert wrapper(x) == "compiled"       # hot-swapped (real batch, inline)
    zero_warms = [c for c in bg_calls if torch.equal(c, torch.zeros(2, 8))]
    assert len(zero_warms) == 1           # exactly one dummy warm per signature


def test_swap_is_race_free_vs_in_flight_eager_batch():
    eager_gate = threading.Event()
    in_eager = threading.Event()

    def original(x):
        in_eager.set()
        assert eager_gate.wait(30)
        return "eager"

    def compiled(x):
        return "compiled"

    router = _router()
    router.enable()
    wrapper = _wrapper(original, compiled, router)
    x = torch.zeros(2, 8)

    results = []
    t = threading.Thread(target=lambda: results.append(wrapper(x)))
    t.start()
    assert in_eager.wait(30)              # eager batch in flight
    sig = ("toy", hot_swap.signature((x,), {}))
    assert _wait(lambda: sig in router.warm)  # bg warm completes mid-flight
    assert wrapper(x) == "compiled"       # new calls take the compiled path
    eager_gate.set()
    t.join(30)
    assert results == ["eager"]           # in-flight batch finished eagerly


def test_tight_headroom_does_not_degrade_a_gated_router(monkeypatch):
    """Tight VRAM headroom is NOT a reason to pay an inline compile.

    pgw#1215 step 4 rewrites what this row asserts. It used to pin the
    degrade-to-sequential fallback — a branch that only an UNGATED router
    could reach, kept alive for "kill-switch parity" with
    ``GEN_WORKER_BG_YIELD``, a switch pgw#995 deleted and which
    ``test_no_env_restores_the_pre_pgw677_tree`` asserts cannot exist. The
    real contract is the one pgw#677 shipped: the warm thread owns the device
    inside its exclusive turn and ensures its own headroom there
    (``_ensure_headroom``), so a novel signature serves EAGER and warms in the
    background no matter how tight the headroom is.

    This row is a CONTRACT assertion, not the deletion's red-proof — a gated
    router already behaved this way. Its predecessor asserted the opposite on
    a BARE (ungated) router, which is the only state that could degrade; the
    red-proof for the deletion is
    ``test_an_ungated_router_cannot_route_concurrently`` below.
    """
    monkeypatch.setattr(hot_swap, "_headroom_ok", lambda device: False)
    calls = []

    def compiled(x):
        calls.append("inline")
        return "compiled"

    router = _router()
    router.enable()
    wrapper = _wrapper(lambda x: "eager", compiled, router)
    x = torch.zeros(2, 8)
    assert wrapper(x) == "eager"          # serves eager, never inline
    assert calls == []                    # the serving thread compiled nothing
    sig = ("toy", hot_swap.signature((x,), {}))
    assert _wait(lambda: sig in router.warm)  # the background turn did it


def test_an_ungated_router_cannot_route_concurrently(monkeypatch):
    """The "ungated legacy" mode is DELETED, not merely unused.

    RED before this change: an ungated Router enabled fine and its
    tight-headroom route returned COMPILED — the pre-pgw#677 degrade, kept
    for parity with a kill switch that no longer exists.
    """
    monkeypatch.setattr(hot_swap, "_headroom_ok", lambda device: False)
    ungated = hot_swap.Router()
    with pytest.raises(hot_swap.RouterNotGated):
        ungated.enable()
    assert not ungated.concurrent
    # ...and the gate cannot be un-wired back into that state either.
    ungated.set_turn_gate(_gate)
    with pytest.raises(hot_swap.RouterNotGated):
        ungated.set_turn_gate(None)
    assert ungated.enable()


def test_prewarmed_signature_short_circuits():
    """Boot-warmed / cell-covered shapes never enter the eager+bg path."""
    router = _router()
    wrapper = _wrapper(lambda x: "eager", lambda x: "compiled", router)
    x = torch.zeros(2, 8)
    assert wrapper(x) == "compiled"       # sequential window (pre-enable)
    router.enable()
    assert wrapper(x) == "compiled"       # warm: straight to compiled
    assert not router.pending


def test_fail_closed_execution_lane_never_enables():
    router = _router(fail_closed=True)
    assert not router.enable()
    wrapper = _wrapper(lambda x: "eager", lambda x: "compiled", router)
    assert wrapper(torch.zeros(2, 8)) == "compiled"


def test_background_failure_is_contained_per_signature():
    def compiled(x):
        if x.shape[0] == 9:
            raise RuntimeError("boom")
        return "compiled"

    router = _router()
    router.enable()
    wrapper = _wrapper(lambda x: "eager", compiled, router)
    bad = torch.zeros(9, 8)
    assert wrapper(bad) == "eager"
    bad_sig = ("toy", hot_swap.signature((bad,), {}))
    assert _wait(lambda: bad_sig in router.bg_failed)
    assert wrapper(bad) == "eager"        # that signature stays eager
    good = torch.zeros(2, 8)
    assert wrapper(good) == "eager"       # other signatures still route
    assert _wait(lambda: ("toy", hot_swap.signature((good,), {})) in router.warm)
    assert wrapper(good) == "compiled"


def test_signature_explosion_disables_concurrency():
    router = _router()
    router.enable()
    with router.lock:
        router.warm.update(("toy", ("pad", i)) for i in range(hot_swap._MAX_SIGS))
    wrapper = _wrapper(lambda x: "eager", lambda x: "compiled", router)
    assert wrapper(torch.zeros(2, 8)) == "compiled"
    assert not router.concurrent


def test_unwrap_closes_router():
    class Pipe:
        pass

    pipe = Pipe()
    router = _router()
    setattr(pipe, cc._MARKER_ATTR, {
        "targets": ["toy"], "shapes": [], "cache": True, "originals": [],
        "regional_mods": [], "failure_signal": _signal(router),
    })
    assert hot_swap.router_of(pipe) is router
    cc.unwrap(pipe)
    assert router.closed
    assert hot_swap.router_of(pipe) is None


def test_enable_noop_without_router():
    class Pipe:
        pass

    assert not hot_swap.enable(Pipe())



# ---------------------------------------------------------------------------
# Real torch.compile integration (CPU toy module)
# ---------------------------------------------------------------------------


def test_real_compile_eager_while_compiling_roundtrip():
    torch._dynamo.reset()

    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.randn(8, 8))

        def forward(self, x):
            return torch.nn.functional.relu(x @ self.w) + 1.0

    m = Toy()
    original = m.forward
    compiled = torch.compile(m.forward, dynamic=False)
    router = _router()
    wrapper = _wrapper(original, compiled, router, label="toy.forward")

    # boot warmup window: sequential compile marks the declared shape warm
    warm_x = torch.randn(2, 8)
    assert torch.allclose(wrapper(warm_x), original(warm_x))
    assert ("toy.forward", hot_swap.signature((warm_x,), {})) in router.warm

    router.enable()
    novel = torch.randn(5, 8)
    t0 = time.monotonic()
    out = wrapper(novel)                  # novel shape: served eagerly
    first_latency = time.monotonic() - t0
    assert torch.allclose(out, original(novel))
    sig = ("toy.forward", hot_swap.signature((novel,), {}))
    assert _wait(lambda: sig in router.warm, timeout=120)
    compile_wall = time.monotonic() - t0
    assert torch.allclose(wrapper(novel), original(novel))  # compiled path
    # The eager serve never waited on the background dynamo+inductor compile.
    # Asserted as a share of the compile this run actually paid, not as an
    # absolute 5s: a slow runner slows both, and the property is the ratio
    # .
    assert first_latency < compile_wall / 5.0, (
        f"the eager serve took {first_latency:.2f}s of a {compile_wall:.2f}s "
        f"background compile — it is waiting on the compile, not dodging it"
    )


# ---------------------------------------------------------------------------
# Cell republish after a background warm
# ---------------------------------------------------------------------------


class _Publisher:
    def __init__(self, enabled=True):
        self._enabled = enabled
        self.published = []

    def enabled(self):
        return self._enabled

    def publish(self, family, artifact, meta, mint_duration_ms=0):
        with tarfile.open(artifact) as tar:
            names = sorted(tar.getnames())
        self.published.append((family, names, dict(meta)))
        return "ckpt-1"


class _Cfg:
    family = "toyfam"
    shapes = ((512, 512),)
    targets = ("transformer",)
    guidance_scales = ()
    regional = False
    lora_bucket = 0


def _pin_identity(monkeypatch):
    key = {
        "sku": "rtx-4090", "sm": "sm_89", "torch": str(torch.__version__),
        "triton": "3.0.0", "cuda": "13.0", "cuda_driver": "13000",
        "image_digest": "sha256:feed",
    }
    monkeypatch.setattr(cc, "runtime_key", lambda: dict(key))
    monkeypatch.setattr(cc, "gen_worker_version", lambda: "0.0.0-test")
    # pgw#681 gate at its torch boundary, simmed: these republish rigs
    # never compile through dynamo, so extraction would honestly report
    # closure unprovable and refuse the republish.
    # The pgw#681 mint gate this simmed is deleted.
    # `guard_closure.closure_manifest` classified every compiled graph at
    # the MINT and wrote the result into the cell's metadata; it went with
    # the `torch-inductor-cache` format that carried it, so a rig whose
    # compiles never touch dynamo has no gate left to satisfy.


def test_the_shape_warm_has_no_republish_backend_since_pgw1010() -> None:
    """A background novel-shape warm still GROWS this pod's live cache — the
    three tests that used to stand here proved it then republished the grown
    cell for the fleet. That cell was a dynamo artifact with no consumer
    (`aot_cells` adopts `aot-inductor` only), so pgw#1010 deleted the
    republish outright rather than leaving it to ship bytes nothing adopts.
    The growth itself is unchanged; what is gone is the shipping.
    """
    assert not hasattr(fleet_cells, "republish_after_shape_warm")


# ---------------------------------------------------------------------------
# apply() wires the router into consumer guards
# ---------------------------------------------------------------------------


def test_apply_installs_router_for_guarded_arms(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    class Pipe:
        pass

    class Cfg:
        family = "toyfam"
        shapes = ((64, 64),)
        targets = ("transformer",)
        guidance_scales = ()
        regional = False
        lora_bucket = 0

    pipe = Pipe()
    pipe.transformer = torch.nn.Linear(4, 4)
    assert cc.apply(pipe, Cfg(), cache_ready=True)
    router = hot_swap.router_of(pipe)
    assert isinstance(router, hot_swap.Router)
    assert not router.concurrent          # sequential until executor enables
    # pgw#1215 step 4: `compile_cache.apply` installs the router; the EXECUTOR
    # wires the turn gate (`_wire_turn_gate`) before it enables concurrency.
    router.set_turn_gate(_gate)
    assert hot_swap.enable(pipe)
    assert router.concurrent
    cc.unwrap(pipe)
    assert router.closed

    producer = Pipe()
    producer.transformer = torch.nn.Linear(4, 4)
    assert cc.apply(producer, Cfg(), cache_ready=True, guard=False)
    assert hot_swap.router_of(producer) is None  # producer arms never route
    cc.unwrap(producer)
