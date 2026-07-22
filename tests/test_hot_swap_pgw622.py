"""pgw#622: eager-while-compiling with hot-swap.

A novel input signature serves eager immediately, warms the compiled path
in a background thread, and atomically hot-swaps; tight headroom degrades
to the sequential compile-then-serve; a preexisting warm signature (cell/
boot-warmed) short-circuits everything; the completed warm republishes the
grown cell.
"""

from __future__ import annotations

import tarfile
import threading
import time

import torch

from gen_worker import compile_cache as cc
from gen_worker import fleet_cells, hot_swap


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

    router = hot_swap.Router()
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

    router = hot_swap.Router()
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


def test_tight_headroom_degrades_to_sequential(monkeypatch):
    monkeypatch.setattr(hot_swap, "_first_cuda_device", lambda a, k: 0)
    monkeypatch.setattr(hot_swap, "_headroom_ok", lambda device: False)
    calls = []

    def compiled(x):
        calls.append("inline")
        return "compiled"

    router = hot_swap.Router()
    router.enable()
    wrapper = _wrapper(lambda x: "eager", compiled, router)
    x = torch.zeros(2, 8)
    assert wrapper(x) == "compiled"       # today's compile-then-serve
    assert calls == ["inline"]            # no background job
    sig = ("toy", hot_swap.signature((x,), {}))
    assert sig in router.warm             # inline success still marks warm


def test_prewarmed_signature_short_circuits():
    """Boot-warmed / cell-covered shapes never enter the eager+bg path."""
    router = hot_swap.Router()
    wrapper = _wrapper(lambda x: "eager", lambda x: "compiled", router)
    x = torch.zeros(2, 8)
    assert wrapper(x) == "compiled"       # sequential window (pre-enable)
    router.enable()
    assert wrapper(x) == "compiled"       # warm: straight to compiled
    assert not router.pending


def test_fail_closed_lane_never_enables():
    router = hot_swap.Router(fail_closed=True)
    assert not router.enable()
    wrapper = _wrapper(lambda x: "eager", lambda x: "compiled", router)
    assert wrapper(torch.zeros(2, 8)) == "compiled"


def test_background_failure_is_contained_per_signature():
    def compiled(x):
        if x.shape[0] == 9:
            raise RuntimeError("boom")
        return "compiled"

    router = hot_swap.Router()
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
    router = hot_swap.Router()
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
    router = hot_swap.Router()
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


def test_on_warmed_fires_and_debounce_coalesces():
    warmed = threading.Event()
    runs = []
    gate = threading.Event()

    def publish():
        runs.append(1)
        assert gate.wait(30)

    debounced = hot_swap.Debounce(publish)

    router = hot_swap.Router()
    router.enable(on_warmed=lambda: (debounced(), warmed.set()))
    wrapper = _wrapper(lambda x: "eager", lambda x: "compiled", router)
    assert wrapper(torch.zeros(2, 8)) == "eager"
    assert warmed.wait(30)
    # a burst while the first publish is in flight coalesces to ONE rerun
    debounced()
    debounced()
    debounced()
    gate.set()
    assert _wait(lambda: len(runs) == 2 and not debounced._running)
    time.sleep(0.05)
    assert len(runs) == 2


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
    router = hot_swap.Router()
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
    assert torch.allclose(wrapper(novel), original(novel))  # compiled path
    # the eager serve never waited on the background dynamo+inductor compile
    assert first_latency < 5.0


# ---------------------------------------------------------------------------
# Cell republish after a background warm
# ---------------------------------------------------------------------------


class _Publisher:
    def __init__(self, enabled=True):
        self._enabled = enabled
        self.published = []

    def enabled(self):
        return self._enabled

    def publish(self, family, artifact, meta):
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


def test_republish_packs_live_root_under_same_key(monkeypatch, tmp_path):
    _pin_identity(monkeypatch)
    live_root = tmp_path / "compile-cache"
    (live_root / "inductor" / "fxgraph" / "aa" / "bb").mkdir(parents=True)
    (live_root / "inductor" / "fxgraph" / "aa" / "bb" / "entry").write_bytes(b"g1")
    (live_root / "triton").mkdir()

    class Pipe:
        pass

    publisher = _Publisher()
    assert fleet_cells.republish_after_shape_warm(
        Pipe(), _Cfg(), "toyfam", publisher, live_root)
    family, names, meta = publisher.published[0]
    assert family == "toyfam"
    assert "inductor/fxgraph/aa/bb/entry" in names
    assert meta["family"] == "toyfam"
    from gen_worker import cell_key

    assert meta["cell_key"] == cell_key.from_artifact_metadata(meta).digest


def test_republish_without_sink_is_a_loud_noop(tmp_path, caplog):
    live_root = tmp_path / "compile-cache"
    (live_root / "inductor").mkdir(parents=True)

    class Pipe:
        pass

    assert not fleet_cells.republish_after_shape_warm(
        Pipe(), _Cfg(), "toyfam", _Publisher(enabled=False), live_root)
    assert any(
        "SHAPE_WARM_WITHOUT_PUBLISH_SINK" in r.message for r in caplog.records)


def test_republish_failure_never_raises(monkeypatch, tmp_path):
    _pin_identity(monkeypatch)
    live_root = tmp_path / "compile-cache"
    (live_root / "inductor").mkdir(parents=True)

    class Boom(_Publisher):
        def publish(self, family, artifact, meta):
            raise RuntimeError("wire down")

    class Pipe:
        pass

    assert not fleet_cells.republish_after_shape_warm(
        Pipe(), _Cfg(), "toyfam", Boom(), live_root)


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
    assert hot_swap.enable(pipe)
    assert router.concurrent
    cc.unwrap(pipe)
    assert router.closed

    producer = Pipe()
    producer.transformer = torch.nn.Linear(4, 4)
    assert cc.apply(producer, Cfg(), cache_ready=True, guard=False)
    assert hot_swap.router_of(producer) is None  # producer arms never route
    cc.unwrap(producer)
