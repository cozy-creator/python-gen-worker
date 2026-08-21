"""pgw#1421, the WIDENING: the declaration binds to the runtime.

The isolated arms (`test_engine_runtime.py`) prove the supervisor.
These prove the two boundaries it has to cross for a real endpoint:

* **DISCOVERY** reads an author's `ctx.engine(LlamaServer(...))` statically —
  no author code runs — and the endpoint.lock states the engine BY NAME. That
  is what makes "which binary does this pod start" answerable from the release
  instead of from the source.
* **THE HOST** stops every engine a load context started, after the author's
  `unload` and regardless of what it did. An external engine is invisible to
  torch's allocator, so a stranded one is VRAM nothing can see or reclaim.

The fixture is written as the qwen3.6 pair will be, so what these arms bind
to is the real migration shape rather than a shape invented for a test.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, cast

import pytest

from gen_worker.discovery.entrypoints_v2 import (
    EntrypointDiscoveryError,
    _engine_runtime,
    _model_slot,
    discover_entrypoints,
    lift_engine_runtimes,
)
from gen_worker import LlamaServer, lane
from gen_worker._vendor.tensorfs import contracts as _tfs_contracts
from gen_worker.demand import GiB, const
from gen_worker.models import SDXL
from gen_worker.serving import (
    DeployBinding,
    EndpointHost,
    LoadContext,
    Model,
    load_endpoint,
)
from gen_worker.serving.engine_runtime import ENGINE_SPEC_RUNTIMES

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "serving_engine_endpoint"
HOST_FIXTURE_DIR = (
    Path(__file__).parent / "fixtures" / "serving_engine_host_endpoint"
)

# Same stand-in as the isolated file; duplicated rather than imported so this
# file states its own substrate (a stand-in that drifts silently is worse than
# one written twice).
_STAND_IN = textwrap.dedent(
    '''
    import http.server, sys, threading, time
    port, warmup, chatter, die = (int(sys.argv[1]), float(sys.argv[2]),
                                  float(sys.argv[3]), int(sys.argv[4]))
    if die >= 0:
        sys.exit(die)
    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            body = b"READY"
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        def log_message(self, *a):
            pass
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", port), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print("stand-in: ready", flush=True)
    while True:
        time.sleep(3600)
    '''
)


@pytest.fixture
def declarations() -> Any:
    """The DECLARATION fixture: three model classes, no process ever started
    (discovery reads the AST — it runs no author code)."""
    load_endpoint(FIXTURE_DIR)
    import serving_engine_fixture.main as main

    return main


@pytest.fixture
def host_fixture(tmp_path: Path) -> Any:
    """The SUPERVISION fixture: one model whose engine really boots."""
    script = tmp_path / "stand_in.py"
    script.write_text(_STAND_IN)
    loaded = load_endpoint(HOST_FIXTURE_DIR)
    import serving_engine_host_fixture.main as main

    main.STAND_IN_SCRIPT = str(script)
    main.ORDER.clear()
    return loaded, main


# --------------------------------------------------------------------------
# 1. the static read: a declaration -> an engine NAME, importing nothing
# --------------------------------------------------------------------------


def test_the_declaration_names_its_engine(declarations: Any) -> None:
    main = declarations
    assert _engine_runtime(main.GgufModel) == "llama-server"
    assert _engine_runtime(main.VllmModel) == "vllm"
    # THE CONTROL, and it is the load-bearing half: a plain pytorch model
    # must read as hosting NO engine. Without this arm a `_engine_runtime`
    # that returned a constant would pass every assertion above.
    assert _engine_runtime(main.PytorchModel) == ""


def test_the_registry_and_the_classes_cannot_drift() -> None:
    """The static fallback resolves a name off the AST when the class object
    is not there to ask, so its table has to agree with the classes. Two
    producers of one fact is how they stop agreeing."""
    from gen_worker.serving.engine_runtime import LlamaServer, VllmServer

    assert ENGINE_SPEC_RUNTIMES == {
        "VllmServer": VllmServer.runtime,
        "LlamaServer": LlamaServer.runtime,
    }
    assert sorted(ENGINE_SPEC_RUNTIMES.values()) == ["llama-server", "vllm"]


# --------------------------------------------------------------------------
# 2. the lock: the census names the engine, and the CONTROL is absent from it
# --------------------------------------------------------------------------


def test_the_lock_census_names_the_engine_runtime(declarations: Any) -> None:
    # The REAL discovery walk, end to end — the marker landing (pgw#1431) is
    # what lets this run to completion instead of refusing at the first
    # engine-hosted slot.
    rows = discover_entrypoints("serving_engine_fixture.main")
    census = lift_engine_runtimes(rows)

    assert [(r["entrypoint"], r["runtime"]) for r in census] == [
        ("chat", "llama-server"),
        ("complete", "vllm"),
    ]
    gguf = census[0]
    assert gguf["slot"] == "model"
    assert gguf["model_class"].endswith("main.GgufModel")
    # `draw` is the pytorch CONTROL: no row, so a lock that carries this block
    # at all IS an engine-hosted endpoint.
    assert all(r["entrypoint"] != "draw" for r in census)

    # LIFTED, not copied: the two keys are gone from the slot rows, so the
    # `entrypoints[]` block the hub decodes is byte-identical to what it was
    # before this landed.
    for row in rows:
        for slot in row["slots"]:
            assert "engine_runtime" not in slot
            assert "model_class" not in slot


# --------------------------------------------------------------------------
# 3. the SEAM with pgw#1431: an UNMARKED engine-hosted slot is told what to
#    write, in words it can follow
# --------------------------------------------------------------------------


class UnmarkedGgufModel(
    Model[SDXL],
    lanes={_tfs_contracts.SDXL_DIFFUSERS_BF16: lane(request=const(GiB(1)))},
):
    """Engine-hosted and NOT marked — the mistake a migrating author makes.

    An engine-hosted model is self-loading by construction, so this class is
    a contradiction the author has not noticed yet. What it must not get is
    the generic refusal's advice: `ctx.load(StableDiffusionXLPipeline)` is a
    load the streaming engine refuses BY DESIGN for a block-quantized
    container, so telling this author to write one sends them into a wall.

    pgw#1599: this class used to carry `eager_only=`, which is deleted. It
    declares a real lane like every model class does, and it calls no
    `ctx.compile` — the ABSENT MARK is the entire eager statement, and it is
    orthogonal to `self_loading=`, which is what this test is about.
    """

    def load(self, ctx: LoadContext[SDXL]) -> None:
        self.engine = ctx.engine(LlamaServer(extra_args=["-ngl", "99"]))


def test_an_unmarked_engine_hosted_slot_is_told_what_to_write() -> None:
    from gen_worker.discovery.entrypoints_v2 import _pipeline_class_or_refuse
    from gen_worker.serving.entrypoints import SlotSpec

    slot = SlotSpec(name="model", kind="model", annotation=UnmarkedGgufModel)
    emitted = _model_slot(slot)
    row: Dict[str, Any] = {"name": "chat", "slots": [emitted]}
    # The static read still worked — discovery KNOWS what this boots, which is
    # exactly why "could not read the pipeline class" would be a false
    # sentence here.
    assert emitted["engine_runtime"] == "llama-server"
    assert emitted["pipeline_class"] == ""

    with pytest.raises(EntrypointDiscoveryError) as excinfo:
        _pipeline_class_or_refuse([row])
    message = str(excinfo.value)
    assert "ENGINE-HOSTED (llama-server)" in message
    assert "self_loading=" in message
    # And it must NOT hand out the advice that cannot work here.
    assert "ctx.load(StableDiffusionXLPipeline)" not in message


def test_a_marked_engine_hosted_slot_publishes(declarations: Any) -> None:
    """The green path: `self_loading=` (pgw#1431, landed) is what carries an
    engine-hosted slot through the gate the hub requires — this lane adds no
    surface of its own for it, and the fixture proves the inheritance."""
    rows = discover_entrypoints("serving_engine_fixture.main")
    by_name = {row["name"]: row for row in rows}
    gguf_slot = by_name["chat"]["slots"][0]
    assert "pipeline_class" not in gguf_slot
    assert "llama-server" in gguf_slot["self_loading"]


# --------------------------------------------------------------------------
# 4. the host: teardown is STRUCTURAL, even when the author's unload explodes
# --------------------------------------------------------------------------


def _dead(proc: subprocess.Popen, *, within_s: float = 10.0) -> bool:
    deadline = time.monotonic() + within_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return True
        time.sleep(0.05)
    return False


def test_the_host_reaps_the_engine_after_a_raising_unload(
    host_fixture: Any, tmp_path: Path
) -> None:
    loaded, main = host_fixture
    host = EndpointHost(
        loaded,
        DeployBinding(checkpoint_ref="ckpt:qwen@1", checkpoint_dir=tmp_path),
        output_dir=tmp_path / "out",
    )
    host.setup()
    instance = host.instances[main.StandInModel]
    # The author's own attribute — `Model` is a typed skeleton, so the engine
    # handle lives on the author's class, not on the base.
    handle = cast(Any, instance.model).engine
    assert handle.alive
    # The engine really serves: the entrypoint's dispatch target answers.
    with urllib.request.urlopen(handle.base_url + "/health", timeout=5) as r:
        assert r.status == 200
    # The load context tracks it, which is what makes the stop structural
    # rather than a thing `unload` has to remember.
    assert instance.load_context.engines == (handle,)

    host.evict(main.StandInModel)

    assert main.ORDER == ["author_unload"]  # the author's unload DID run
    assert not handle.alive
    assert _dead(handle.process)
    assert len(instance.load_context.engines) == 0
    assert main.StandInModel not in host.instances


def test_stop_engines_is_idempotent_and_survives_a_raising_stop(
    tmp_path: Path,
) -> None:
    """The host calls `stop_engines` unconditionally, so one bad handle must
    not strand the ones behind it — that is the whole reason the loop
    swallows."""

    class Boom:
        runtime = "boom"

        def stop(self) -> None:
            raise RuntimeError("stop exploded")

    class Counted:
        runtime = "counted"

        def __init__(self) -> None:
            self.stops = 0

        def stop(self) -> None:
            self.stops += 1

    ctx: LoadContext[Any] = LoadContext(
        binding=DeployBinding(checkpoint_ref="c", checkpoint_dir=tmp_path)
    )
    counted = Counted()
    ctx._engines.extend([counted, Boom()])
    ctx.stop_engines()
    # Newest-first: Boom raised, and the one behind it was still stopped.
    assert counted.stops == 1
    assert len(ctx.engines) == 0
    ctx.stop_engines()  # idempotent: nothing left, nothing raised
    assert counted.stops == 1
