"""pgw#520 hub-less resolution: ``cozy run`` / ``gen-worker run`` only ever
runs a Slot's ``default=`` ref (no hub to resolve a curated/BYOM pick
against); a payload that NAMES a model via the ``selected_by`` field fails
clearly instead of silently running the default. Drives the real
``cli.main(["run", ...])`` with ``--offline`` (deterministic cache-miss,
matching the existing exit-code-matrix test's technique) — no network mock
needed since the clear-error path fires before any resolve attempt."""

from __future__ import annotations

import json
import sys
import types

import msgspec
import pytest

import gen_worker.cli as cli
import gen_worker.cli.run as run_mod
from gen_worker import Hub, RequestContext, Slot, endpoint
from gen_worker.families import SdxlDefaults


class _In(msgspec.Struct):
    prompt: str = ""
    model: str = ""


class _Out(msgspec.Struct):
    y: str


def _slot_module(name: str) -> types.ModuleType:
    mod = types.ModuleType(name)

    @endpoint(model=Slot(
        object, selected_by="model",
        default=Hub("test-org/test-repo", flavor="bf16"),
        fallback=SdxlDefaults(steps=28),
    ))
    class Gen:
        def setup(self, model: object) -> None:
            self.model = model

        def generate(self, ctx: RequestContext, data: _In) -> _Out:
            return _Out(y="ok")

    Gen.__module__ = name
    mod.Gen = Gen
    sys.modules[name] = mod
    return mod


@pytest.fixture(autouse=True)
def _force_cold(monkeypatch) -> None:
    monkeypatch.setattr(run_mod, "_warm_serve_socket", lambda: None)


def test_hubless_default_only_run_hits_normal_offline_cache_miss(tmp_path, capsys, monkeypatch) -> None:
    """No selected_by value in the payload -> the CLI attempts the Slot's
    default= ref exactly like a bare binding would (proves the default path
    is reached, not short-circuited)."""
    _slot_module("_test_slot_default")
    monkeypatch.setenv("TENSORHUB_CAS_DIR", str(tmp_path / "empty-cas"))
    rc = cli.main([
        "run", "--module", "_test_slot_default", "--offline",
        "--payload", json.dumps({"prompt": "x"}),
    ])
    assert rc == run_mod.EXIT_MODEL_RESOLUTION
    assert "model resolution failed" in capsys.readouterr().err


def test_hubless_named_model_pick_fails_clearly(tmp_path, capsys, monkeypatch) -> None:
    """A payload that NAMES a model via selected_by, with no hub configured,
    must fail with the specific "no hub is configured" message — not the
    generic offline-cache-miss message, and not a silent default run."""
    _slot_module("_test_slot_named")
    monkeypatch.setenv("TENSORHUB_CAS_DIR", str(tmp_path / "empty-cas"))
    rc = cli.main([
        "run", "--module", "_test_slot_named", "--offline",
        "--payload", json.dumps({"prompt": "x", "model": "some-curated-id"}),
    ])
    assert rc == run_mod.EXIT_MODEL_RESOLUTION
    err = capsys.readouterr().err
    assert "no hub is configured" in err
    assert "some-curated-id" in err
