from __future__ import annotations

import os
from pathlib import Path

import requests
import pytest
import gen_worker.request_context as rc
from gen_worker.api.types import Tensors
from gen_worker.worker import RequestContext


def test_action_context_local_output_backend(tmp_path: Path) -> None:
    ctx = RequestContext(
        "rid1",
        local_output_dir=str(tmp_path),
        owner="o1",
        invoker_id="u1",
    )
    ref = "runs/rid1/outputs/hello.bin"
    asset = ctx.save_bytes(ref, b"abc")
    assert asset.ref == ref
    assert asset.local_path is not None
    p = Path(asset.local_path)
    assert p.exists()
    assert p.read_bytes() == b"abc"


def test_action_context_save_checkpoint_local_output(tmp_path: Path) -> None:
    ctx = RequestContext(
        "rid2",
        local_output_dir=str(tmp_path),
        owner="o1",
        invoker_id="u1",
    )
    local = tmp_path / "converted.safetensors"
    local.write_bytes(b"weights")
    tensors = ctx.save_checkpoint("runs/rid2/outputs/final.safetensors", str(local))
    assert isinstance(tensors, Tensors)
    assert tensors.ref == "runs/rid2/outputs/final.safetensors"
    assert tensors.local_path is not None
    assert tensors.format == "safetensors"
    assert tensors.sha256


def test_action_context_open_output_stream_local_output(tmp_path: Path) -> None:
    ctx = RequestContext(
        "rid3",
        local_output_dir=str(tmp_path),
        owner="o1",
        invoker_id="u1",
    )
    with ctx.open_output_stream("runs/rid3/outputs/chunked.bin") as stream:
        stream.write(b"ab")
        stream.write(b"cd")
        assert stream.bytes_written == 4
        asset = stream.finalize()
    assert asset.ref == "runs/rid3/outputs/chunked.bin"
    assert asset.local_path is not None
    assert Path(asset.local_path).read_bytes() == b"abcd"


def test_action_context_open_checkpoint_stream_local_output(tmp_path: Path) -> None:
    ctx = RequestContext(
        "rid4",
        local_output_dir=str(tmp_path),
        owner="o1",
        invoker_id="u1",
    )
    with ctx.open_checkpoint_stream("runs/rid4/outputs/final.safetensors") as stream:
        stream.write(b"we")
        stream.write(b"ights")
        tensors = stream.finalize()
    assert isinstance(tensors, Tensors)
    assert tensors.format == "safetensors"
    assert tensors.local_path is not None
    assert Path(tensors.local_path).read_bytes() == b"weights"


def test_action_context_stream_checkpoint_matches_save_checkpoint(tmp_path: Path) -> None:
    ctx = RequestContext(
        "rid5",
        local_output_dir=str(tmp_path),
        owner="o1",
        invoker_id="u1",
    )
    source = tmp_path / "source.bin"
    source.write_bytes(b"abcdefgh")

    saved_file = ctx.save_checkpoint("runs/rid5/outputs/by-file.safetensors", str(source))
    with ctx.open_checkpoint_stream("runs/rid5/outputs/by-stream.safetensors", format="safetensors") as stream:
        stream.write(b"abcd")
        stream.write(b"efgh")
        saved_stream = stream.finalize()

    assert saved_file.local_path is not None
    assert saved_stream.local_path is not None
    assert Path(saved_file.local_path).read_bytes() == Path(saved_stream.local_path).read_bytes()


def test_action_context_stream_finalize_failure_cleans_temp_file(tmp_path: Path, monkeypatch) -> None:
    stream_tmp = tmp_path / "stream-tmp.bin"

    def _mkstemp(*, prefix: str, suffix: str) -> tuple[int, str]:
        _ = prefix
        _ = suffix
        fd = os.open(stream_tmp, os.O_CREAT | os.O_TRUNC | os.O_RDWR, 0o600)
        return fd, str(stream_tmp)

    monkeypatch.setattr(rc.tempfile, "mkstemp", _mkstemp)

    ctx = RequestContext("rid6", local_output_dir=str(tmp_path), owner="o1", invoker_id="u1")

    def _fail_save_checkpoint(ref: str, local_path: str, format: str | None = None) -> Tensors:
        _ = ref
        _ = local_path
        _ = format
        raise RuntimeError("boom")

    monkeypatch.setattr(ctx, "save_checkpoint", _fail_save_checkpoint)

    writer = ctx.open_checkpoint_stream("runs/rid6/outputs/fail.safetensors", format="safetensors")
    writer.write(b"weights")
    with pytest.raises(RuntimeError, match="boom"):
        writer.finalize()
    assert not stream_tmp.exists()


def test_action_context_save_file_network_error_is_deterministic(tmp_path: Path, monkeypatch) -> None:
    src = tmp_path / "upload.bin"
    src.write_bytes(b"abc")
    ctx = RequestContext(
        "rid7",
        file_api_base_url="https://files.example.test",
        file_api_token="token",
        owner="o1",
    )

    def _fail_put(*args, **kwargs):  # type: ignore[no-untyped-def]
        _ = args
        _ = kwargs
        raise requests.ConnectionError("network down")

    monkeypatch.setattr(requests, "put", _fail_put)

    with pytest.raises(RuntimeError, match="file save failed \\(network_error\\)"):
        ctx.save_file("runs/rid7/outputs/upload.bin", str(src))
