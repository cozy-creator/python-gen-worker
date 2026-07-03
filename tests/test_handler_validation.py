"""Decoration-time validation of per-request model slots vs handler signatures.

Exercises ``_validate_handler_injection`` through the real ``@inference``
decorator (not a stub), so it asserts the contract end-to-end: a correctly
shaped endpoint decorates cleanly; a mismatched one raises at IMPORT/decoration
time with a message that names the offending parameter and states the rule.

The contract (#337): static (Repo) bindings are injected into ``setup()``;
dispatch bindings are injected into the handler per request. Therefore every
extra handler parameter must map to a dispatch slot, and every dispatch slot
must be consumed by some handler.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

import msgspec
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gen_worker import (  # noqa: E402
    HFRepo,
    RequestContext,
    dispatch,
    inference,
    invocable,
)


class _In(msgspec.Struct):
    prompt: str = "x"
    model: Literal["a", "b"] = "a"


class _NoField(msgspec.Struct):
    prompt: str = "x"


class _StrField(msgspec.Struct):
    prompt: str = "x"
    model: str = "a"


_TABLE = {"a": HFRepo("acme/a"), "b": HFRepo("acme/b")}


def test_dispatch_slot_consumed_by_handler_is_valid() -> None:
    @inference(models={"pipeline": dispatch(field="model", table=_TABLE)})
    class Ok:
        def setup(self) -> None:
            ...

        @invocable(name="gen_ok")
        def generate(self, ctx: RequestContext, payload: _In, pipeline: object):
            return None

    assert getattr(Ok, "__gen_worker_endpoint_spec__", None) is not None


def test_fixed_slot_in_setup_with_plain_handler_is_valid() -> None:
    @inference(models={"pipeline": HFRepo("acme/fixed")})
    class Ok2:
        def setup(self, pipeline: object) -> None:
            self.p = pipeline

        @invocable(name="gen_ok2")
        def generate(self, ctx: RequestContext, payload: _In):
            return None

    assert getattr(Ok2, "__gen_worker_endpoint_spec__", None) is not None


def test_handler_with_var_kwargs_is_not_rejected() -> None:
    @inference(models={"pipeline": dispatch(field="model", table=_TABLE)})
    class Ok3:
        def setup(self) -> None:
            ...

        @invocable(name="gen_ok3")
        def generate(self, ctx: RequestContext, payload: _In, **kwargs):
            return None

    assert getattr(Ok3, "__gen_worker_endpoint_spec__", None) is not None


def test_handler_param_matching_no_slot_is_rejected() -> None:
    with pytest.raises(ValueError) as ei:
        @inference(models={"pipeline": dispatch(field="model", table=_TABLE)})
        class Bad:
            def setup(self) -> None:
                ...

            @invocable(name="gen_bad")
            def generate(self, ctx: RequestContext, payload: _In, pipline: object):
                return None

    msg = str(ei.value)
    assert "pipline" in msg
    assert "injected into the handler per request" in msg


def test_fixed_slot_named_on_handler_is_rejected() -> None:
    with pytest.raises(ValueError) as ei:
        @inference(models={"vae": HFRepo("acme/vae")})
        class Bad2:
            def setup(self, vae: object) -> None:
                self.v = vae

            @invocable(name="gen_bad2")
            def generate(self, ctx: RequestContext, payload: _In, vae: object):
                return None

    assert "vae" in str(ei.value)


def test_dispatch_slot_not_consumed_by_any_handler_is_rejected() -> None:
    with pytest.raises(ValueError) as ei:
        @inference(models={"pipeline": dispatch(field="model", table=_TABLE)})
        class Bad3:
            def setup(self) -> None:
                ...

            @invocable(name="gen_bad3")
            def generate(self, ctx: RequestContext, payload: _In):
                return None

    msg = str(ei.value)
    assert "pipeline" in msg
    assert "no handler accepts" in msg


def test_error_messages_state_the_two_site_rule() -> None:
    with pytest.raises(ValueError) as ei:
        @inference(models={"pipeline": dispatch(field="model", table=_TABLE)})
        class Bad4:
            def setup(self) -> None:
                ...

            @invocable(name="gen_bad4")
            def generate(self, ctx: RequestContext, payload: _In, wrong: object):
                return None

    assert "setup()" in str(ei.value)
    assert "per request" in str(ei.value)

# --------------------------------------------------------------------------- #
# dispatch() Literal contract (binding.py) — wired validators                  #
# --------------------------------------------------------------------------- #


def test_dispatch_field_missing_from_payload_is_rejected() -> None:
    with pytest.raises(ValueError, match="does not exist"):
        @inference(models={"pipeline": dispatch(field="model", table=_TABLE)})
        class Bad:
            def setup(self) -> None:
                ...

            @invocable(name="gen_nofield")
            def generate(self, ctx: RequestContext, payload: _NoField, pipeline: object):
                return None


def test_dispatch_field_not_literal_is_rejected() -> None:
    with pytest.raises(ValueError, match="Literal"):
        @inference(models={"pipeline": dispatch(field="model", table=_TABLE)})
        class Bad2:
            def setup(self) -> None:
                ...

            @invocable(name="gen_strfield")
            def generate(self, ctx: RequestContext, payload: _StrField, pipeline: object):
                return None


def test_dispatch_table_key_outside_literal_is_rejected() -> None:
    with pytest.raises(ValueError, match="not members of Literal"):
        @inference(models={"pipeline": dispatch(field="model", table={**_TABLE, "c": HFRepo("acme/c")})})
        class Bad3:
            def setup(self) -> None:
                ...

            @invocable(name="gen_extrakey")
            def generate(self, ctx: RequestContext, payload: _In, pipeline: object):
                return None
