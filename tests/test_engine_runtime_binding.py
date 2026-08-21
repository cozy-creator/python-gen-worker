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

#: pgw#1621: a lane is the `(topology, quant)` stamp pair. The v1 Contract
#: object (`contracts.SDXL_DIFFUSERS_BF16`) is deleted with the v1 corpus.
_SDXL_BF16 = ("sdxl.diffusers@1", "plain.bf16@1")

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "serving_engine_endpoint"
HOST_FIXTURE_DIR = (
    Path(__file__).parent / "fixtures" / "serving_engine_host_endpoint"
)

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
    """The DECLARATION fixture: three model classes, no process ever started (discovery reads the AST — it runs no author code)."""
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


def test_the_declaration_names_its_engine(declarations: Any) -> None:
    main = declarations
    assert _engine_runtime(main.GgufModel) == "llama-server"
    assert _engine_runtime(main.VllmModel) == "vllm"
    assert _engine_runtime(main.PytorchModel) == ""


def test_the_registry_and_the_classes_cannot_drift() -> None:
    """The static fallback resolves a name off the AST when the class object is not there to ask, so its table has to agree with the classes."""
    from gen_worker.serving.engine_runtime import LlamaServer, VllmServer

    assert ENGINE_SPEC_RUNTIMES == {
        "VllmServer": VllmServer.runtime,
        "LlamaServer": LlamaServer.runtime,
    }
    assert sorted(ENGINE_SPEC_RUNTIMES.values()) == ["llama-server", "vllm"]


def test_the_lock_census_names_the_engine_runtime(declarations: Any) -> None:
    rows = discover_entrypoints("serving_engine_fixture.main")
    census = lift_engine_runtimes(rows)

    assert [(r["entrypoint"], r["runtime"]) for r in census] == [
        ("chat", "llama-server"),
        ("complete", "vllm"),
    ]
    gguf = census[0]
    assert gguf["slot"] == "model"
    assert gguf["model_class"].endswith("main.GgufModel")
    assert all(r["entrypoint"] != "draw" for r in census)

    for row in rows:
        for slot in row["slots"]:
            assert "engine_runtime" not in slot
            assert "model_class" not in slot


class UnmarkedGgufModel(
    Model[SDXL],
    lanes={_SDXL_BF16: lane(request=const(GiB(1)))},
):
    """Engine-hosted and NOT marked — the mistake a migrating author makes."""

    def load(self, ctx: LoadContext[SDXL]) -> None:
        self.engine = ctx.engine(LlamaServer(extra_args=["-ngl", "99"]))


def test_an_unmarked_engine_hosted_slot_is_told_what_to_write() -> None:
    from gen_worker.discovery.entrypoints_v2 import _pipeline_class_or_refuse
    from gen_worker.serving.entrypoints import SlotSpec

    slot = SlotSpec(name="model", kind="model", annotation=UnmarkedGgufModel)
    emitted = _model_slot(slot)
    row: Dict[str, Any] = {"name": "chat", "slots": [emitted]}
    assert emitted["engine_runtime"] == "llama-server"
    assert emitted["pipeline_class"] == ""

    with pytest.raises(EntrypointDiscoveryError) as excinfo:
        _pipeline_class_or_refuse([row])
    message = str(excinfo.value)
    assert "ENGINE-HOSTED (llama-server)" in message
    assert "self_loading=" in message
    assert "ctx.load(StableDiffusionXLPipeline)" not in message


def test_a_marked_engine_hosted_slot_publishes(declarations: Any) -> None:
    rows = discover_entrypoints("serving_engine_fixture.main")
    by_name = {row["name"]: row for row in rows}
    gguf_slot = by_name["chat"]["slots"][0]
    assert "pipeline_class" not in gguf_slot
    assert "llama-server" in gguf_slot["self_loading"]


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
    handle = cast(Any, instance.model).engine
    assert handle.alive
    with urllib.request.urlopen(handle.base_url + "/health", timeout=5) as r:
        assert r.status == 200
    assert instance.load_context.engines == (handle,)

    host.evict(main.StandInModel)

    assert main.ORDER == ["author_unload"]
    assert not handle.alive
    assert _dead(handle.process)
    assert len(instance.load_context.engines) == 0
    assert main.StandInModel not in host.instances


def test_stop_engines_is_idempotent_and_survives_a_raising_stop(
    tmp_path: Path,
) -> None:
    """The host calls `stop_engines` unconditionally, so one bad handle must not strand the ones behind it — that is the whole reason the loop swallows."""

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
    assert counted.stops == 1
    assert len(ctx.engines) == 0
    ctx.stop_engines()
    assert counted.stops == 1
