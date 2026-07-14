"""#377: binding files/provider apply to bare-ref residency downloads."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict, List

import msgspec

import gen_worker.executor as executor_mod
from gen_worker.api.binding import HF
from gen_worker.executor import Executor
from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.registry import EndpointSpec


class _In(msgspec.Struct):
    x: str


class _Out(msgspec.Struct):
    y: str


_BINDING = HF(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    dtype="fp16",
    files=("*.json", "*.txt", "*.fp16.safetensors"),
)
_REF = "stable-diffusion-v1-5/stable-diffusion-v1-5"


def _spec() -> EndpointSpec:
    class Endpoint:
        def setup(self, model: str) -> None:  # pragma: no cover
            pass

        def run(self, ctx, payload: _In) -> _Out:  # pragma: no cover
            return _Out(y=payload.x)

    return EndpointSpec(
        name="sd15", method=Endpoint.run, kind="inference",
        payload_type=_In, output_mode="single", cls=Endpoint,
        attr_name="run", models={"model": _BINDING},
    )


async def _noop_send(msg: pb.WorkerMessage) -> None:
    pass


def _capture_download(monkeypatch, tmp_path: Path) -> List[Dict[str, Any]]:
    calls: List[Dict[str, Any]] = []

    async def _fake_ensure_local(ref: str, **kwargs: Any) -> Path:
        calls.append({"ref": ref, **kwargs})
        return tmp_path

    monkeypatch.setattr(executor_mod, "ensure_local", _fake_ensure_local)
    return calls


def test_bare_ref_store_ensure_local_applies_binding_files(monkeypatch, tmp_path) -> None:
    # DesiredResidency carries a ref and snapshot, not the declared binding.
    calls = _capture_download(monkeypatch, tmp_path)

    async def _run() -> None:
        ex = Executor([_spec()], _noop_send)
        await ex.store.ensure_local(_REF)

    asyncio.run(_run())
    assert calls[0]["provider"] == "huggingface"
    assert calls[0]["allow_patterns"] == _BINDING.files


def test_explicit_binding_still_wins(monkeypatch, tmp_path) -> None:
    calls = _capture_download(monkeypatch, tmp_path)
    override = HF(_REF, files=("only-this.safetensors",))

    async def _run() -> None:
        ex = Executor([_spec()], _noop_send)
        await ex.store.ensure_local(_REF, binding=override)

    asyncio.run(_run())
    assert calls[0]["allow_patterns"] == ("only-this.safetensors",)


def test_unknown_ref_downloads_without_patterns(monkeypatch, tmp_path) -> None:
    calls = _capture_download(monkeypatch, tmp_path)

    async def _run() -> None:
        ex = Executor([_spec()], _noop_send)
        await ex.store.ensure_local("acme/unbound")

    asyncio.run(_run())
    assert calls[0]["provider"] is None
    assert calls[0]["allow_patterns"] == ()
