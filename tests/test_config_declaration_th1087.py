"""th#1087 stage B: declared config parameters + declared envs.

Proven over the real codepaths:
  1. Discovery manifest emits ``config_params`` + ``env`` blocks (hub parse
     input for write-time validation) and omits them when undeclared.
  2. A dispatched job reads the declared defaults through ``ctx.config``
     (typed access, hub-double gRPC wire).
  3. Decoration-time validation rejects malformed declarations with
     helpful errors.
"""

from __future__ import annotations

import msgspec
import pytest

from gen_worker import ConfigParam, endpoint
from gen_worker.pb import worker_scheduler_pb2 as pb

from harness.hub_double import hub_double, is_ready, is_result_for
from harness.toy_endpoints import EchoIn


class _In(msgspec.Struct):
    text: str = ""


class _Out(msgspec.Struct):
    ok: bool = True


def test_manifest_emits_config_and_env_blocks() -> None:
    from gen_worker.discovery.discover import _extract_entries
    from harness import toy_endpoints

    (fn,) = _extract_entries(
        toy_endpoints.ConfigKnobsEndpoint, "harness.toy_endpoints"
    )
    assert fn["config_params"] == [
        {
            "name": "scheduler",
            "type": "string",
            "default": "ddim",
            "choices": ["ddim", "euler_a"],
        },
        {"name": "default_steps", "type": "int", "default": 30, "ge": 1, "le": 150},
    ]
    assert fn["env"] == ["TOY_API_BASE"]
    plain = _extract_entries(toy_endpoints.Basics, "harness.toy_endpoints")
    assert all("config_params" not in f and "env" not in f for f in plain)


def test_ctx_config_typed_access_over_the_wire() -> None:
    with hub_double() as (scheduler, _harness):
        conn = scheduler.wait_connection(0)
        conn.wait_for(is_ready)
        conn.send(run_job=pb.RunJob(
            request_id="r-cfg", attempt=1, function_name="config-echo",
            input_payload=msgspec.msgpack.encode(EchoIn(text="x"))))
        res = conn.wait_for(is_result_for("r-cfg")).job_result
        assert res.status == pb.JOB_STATUS_OK, res.safe_message
        assert msgspec.msgpack.decode(res.inline)["response"] == "ddim:30"


def test_decoration_validation_errors() -> None:
    with pytest.raises(TypeError, match="not int"):
        ConfigParam("steps", int, "thirty")
    with pytest.raises(ValueError, match="not in choices"):
        ConfigParam("sched", str, default="x", choices=["a", "b"])
    with pytest.raises(ValueError, match="ge/le apply to int/float"):
        ConfigParam("sched", str, default="a", ge=1)
    with pytest.raises(ValueError, match="does not match regex"):
        ConfigParam("slug", str, default="BAD", regex="[a-z]+")
    with pytest.raises(ValueError, match="< ge"):
        ConfigParam("steps", int, 0, ge=1, le=150)
    with pytest.raises(TypeError, match="one of"):
        ConfigParam("blob", dict, {})
    with pytest.raises(ValueError, match="identifier"):
        ConfigParam("bad name", str, "a")

    def _decl(**kwargs):
        @endpoint(**kwargs)
        class E:
            def go(self, ctx, payload: _In) -> _Out:
                return _Out()
        return E

    with pytest.raises(TypeError, match="config= entries must be ConfigParam"):
        _decl(config=["not-a-param"])
    with pytest.raises(ValueError, match="repeats 'steps'"):
        _decl(config=[ConfigParam("steps", int, 1), ConfigParam("steps", int, 2)])
    with pytest.raises(ValueError, match="not a valid"):
        _decl(env=["BAD NAME"])
    with pytest.raises(ValueError, match="not a valid"):
        _decl(env=["lowercase"])
    with pytest.raises(ValueError, match="repeats"):
        _decl(env=["A_TOKEN", "A_TOKEN"])
    with pytest.raises(TypeError, match="env= must be a list/tuple"):
        _decl(env="A_TOKEN")
