"""gw#402: llama.cpp/GGUF serving runtime.

Unit: GGUF resolution, fit planning, stream accumulation, command shapes.
Integration (real llama-server + the VENDORED tiny stories260K gguf, CPU): boot,
stream, terminal output, token counts. No network: the model is
`tests/testdata/stories260K.gguf` (pgw#966 — it used to be an hf_hub_download on
every CI run, and HF's rate limiter reddened PR #483). Binary resolution:
$GEN_WORKER_LLAMA_SERVER_BIN or PATH; tests skip when absent — the binary is
still a third-party fetch, so it degrades to a counted skip and never a red.
GPU smoke is GEN_WORKER_GPU_SMOKE-gated.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import shutil
from pathlib import Path
from typing import Iterator, List

import msgspec
import pytest

from gen_worker.api.errors import FatalError
from gen_worker.api.streaming import (
    BatchItemDelta,
    IncrementalTokenDelta,
    StreamAccumulator,
    StreamResult,
    TokenUsage,
)
from gen_worker.runtimes.llama import (
    GGUFInfo,
    chat_deltas,
    completion_deltas,
    plan_fit,
    read_gguf_info,
    resolve_gguf,
)
from gen_worker.runtimes.server import DegradingBoot, ServerProcess, llama_server

# ---------------------------------------------------------------------------
# resolve_gguf
# ---------------------------------------------------------------------------


def test_resolve_gguf_file_passthrough() -> None:
    assert resolve_gguf("/models/x.gguf") == Path("/models/x.gguf")


def test_resolve_gguf_single_in_dir(tmp_path) -> None:
    (tmp_path / "sub").mkdir()
    target = tmp_path / "sub" / "model-q4_k_m.gguf"
    target.write_bytes(b"x")
    assert resolve_gguf(tmp_path) == target


def test_resolve_gguf_split_shards_pick_first(tmp_path) -> None:
    for i in (2, 1, 3):
        (tmp_path / f"big-q4-0000{i}-of-00003.gguf").write_bytes(b"x")
    assert resolve_gguf(tmp_path).name == "big-q4-00001-of-00003.gguf"


def test_resolve_gguf_multiple_models_fail_closed(tmp_path) -> None:
    (tmp_path / "a-q4_k_m.gguf").write_bytes(b"x")
    (tmp_path / "a-q8_0.gguf").write_bytes(b"x")
    with pytest.raises(FatalError, match="pin the flavor"):
        resolve_gguf(tmp_path)


def test_resolve_gguf_empty_dir_fails(tmp_path) -> None:
    with pytest.raises(FatalError, match="no .gguf"):
        resolve_gguf(tmp_path)


# ---------------------------------------------------------------------------
# plan_fit — degraded is fewer layers, never a crash
# ---------------------------------------------------------------------------

_INFO = GGUFInfo(
    architecture="llama", n_layers=32, n_ctx_train=4096,
    n_embd=4096, n_head=32, n_head_kv=8, size_bytes=4 * 1024**3,
)


def test_plan_full_offload_when_everything_fits() -> None:
    plan = plan_fit(_INFO, free_vram_gb=80.0)
    assert plan.n_gpu_layers == 33  # +1 output layer
    assert not plan.degraded
    assert plan.n_ctx == 4096


def test_plan_partial_offload_degrades() -> None:
    plan = plan_fit(_INFO, free_vram_gb=3.0)
    assert 0 < plan.n_gpu_layers < 33
    assert plan.degraded


def test_plan_monotonic_in_budget() -> None:
    layers = [plan_fit(_INFO, free_vram_gb=g).n_gpu_layers for g in (0.0, 2.0, 3.0, 5.0, 80.0)]
    assert layers == sorted(layers)
    assert layers[0] == 0 and layers[-1] == 33


def test_plan_no_budget_is_cpu_not_crash() -> None:
    for gb in (0.0, -5.0, 0.5):
        plan = plan_fit(_INFO, free_vram_gb=gb)
        assert plan.n_gpu_layers == 0


def test_plan_unknown_geometry_is_cpu_not_crash() -> None:
    plan = plan_fit(GGUFInfo(), free_vram_gb=24.0)
    assert plan.n_gpu_layers == 0
    assert plan.n_ctx == 4096


def test_plan_ctx_clamped_to_trained() -> None:
    plan = plan_fit(_INFO, free_vram_gb=80.0, n_ctx=1_000_000)
    assert plan.n_ctx == 4096


# ---------------------------------------------------------------------------
# llama_server command shapes
# ---------------------------------------------------------------------------


def test_caller_pinned_ngl_respected() -> None:
    proc = llama_server("/models/x.gguf", port=4321, extra_args=["-ngl", "99", "-c", "8192"])
    assert isinstance(proc, ServerProcess)
    assert proc.command[:3] == ["llama-server", "-m", "/models/x.gguf"]
    assert proc.command.count("-ngl") == 1


def test_unreadable_header_boots_with_defaults() -> None:
    proc = llama_server("/models/x.gguf", port=4321)
    assert isinstance(proc, ServerProcess)
    assert "-ngl" not in proc.command


def test_auto_fit_builds_degrade_ladder(tmp_path, monkeypatch) -> None:
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"GGUF")
    import gen_worker.runtimes.llama as llama_mod

    monkeypatch.setattr(llama_mod, "read_gguf_info", lambda p: _INFO)
    monkeypatch.setattr(llama_mod, "free_vram_gb", lambda: 3.0)
    boot = llama_server(str(gguf))
    assert isinstance(boot, DegradingBoot)
    ngls = [c.command[c.command.index("-ngl") + 1] for c in boot.candidates]
    assert ngls[-1] == "0" and len(ngls) == 3
    assert int(ngls[0]) > int(ngls[1]) > 0


# ---------------------------------------------------------------------------
# StreamAccumulator / terminal StreamResult
# ---------------------------------------------------------------------------


def test_accumulator_concatenates_tokens_and_usage() -> None:
    acc = StreamAccumulator()
    for t in ("Hel", "lo ", "world"):
        acc.add(IncrementalTokenDelta(text=t))
    acc.add(TokenUsage(prompt_tokens=5, completion_tokens=3, tokens_per_second=42.0))
    out = acc.result()
    assert out.text == "Hello world"
    assert out.usage is not None and out.usage.completion_tokens == 3
    assert not out.truncated


def test_accumulator_multiplexes_item_ids() -> None:
    acc = StreamAccumulator()
    acc.add(IncrementalTokenDelta(text="a", item_id="x"))
    acc.add(IncrementalTokenDelta(text="b", item_id="y"))
    out = acc.result()
    assert out.texts == {"x": "a", "y": "b"}
    assert out.text == ""


def test_accumulator_batch_items_with_chunks_and_errors() -> None:
    acc = StreamAccumulator()
    acc.add(BatchItemDelta(index=0, item_id="a", chunk=b"cap", content_type="text/plain"))
    acc.add(BatchItemDelta(index=0, item_id="a", chunk=b"tion", finished=True,
                           content_type="text/plain"))
    acc.add(BatchItemDelta(index=1, item_id="b", error="boom", finished=True))
    out = acc.result()
    assert out is not None and len(out.items) == 2
    assert out.items[0].content == "caption" and out.items[0].content_type == "text/plain"
    assert out.items[1].error == "boom"


def test_accumulator_caps_content_not_metadata() -> None:
    acc = StreamAccumulator(max_bytes=4)
    acc.add(BatchItemDelta(index=0, chunk=b"12345678", finished=True))
    out = acc.result()
    assert out is not None and out.truncated
    assert out.items[0].content == b""  # metadata survives, content dropped


def test_accumulator_empty_stream_returns_none() -> None:
    assert StreamAccumulator().result() is None


def test_stream_mode_terminal_output_retrievable() -> None:
    """gw#475: a completed stream request's JobResult carries the aggregate."""
    from gen_worker.executor import Executor
    from gen_worker.pb import worker_scheduler_pb2 as pb
    from gen_worker.registry import EndpointSpec

    class _In(msgspec.Struct):
        prompt: str = "hi"

    def _gen(ctx, payload: _In) -> Iterator[IncrementalTokenDelta]:
        yield IncrementalTokenDelta(text="to")
        yield IncrementalTokenDelta(text="ken")
        yield TokenUsage(prompt_tokens=1, completion_tokens=2, tokens_per_second=9.0)

    async def _go() -> List:
        sent: List = []

        async def _send(msg) -> None:
            sent.append(msg)

        spec = EndpointSpec(name="fn", method=_gen, kind="inference",
                            payload_type=_In, output_mode="stream")
        ex = Executor([spec], _send)
        await ex.handle_run_job(pb.RunJob(
            request_id="r1", attempt=1, function_name="fn",
            input_payload=msgspec.msgpack.encode(_In())))
        job = ex.jobs[("r1", 1)]
        assert job.task is not None
        await job.task
        return sent

    sent = asyncio.run(_go())
    chunks = [m.job_progress for m in sent if m.WhichOneof("msg") == "job_progress"]
    assert [c.data for c in chunks if c.content_type == "text/plain"] == [b"to", b"ken"]
    results = [m.job_result for m in sent if m.WhichOneof("msg") == "job_result"]
    assert results and results[-1].status == pb.JOB_STATUS_OK
    out = msgspec.msgpack.decode(results[-1].inline, type=StreamResult)
    assert out.text == "token"
    assert out.usage is not None
    assert out.usage.prompt_tokens == 1 and out.usage.completion_tokens == 2


# ---------------------------------------------------------------------------
# Integration: real llama-server, tiny real GGUF, CPU inference.
# ---------------------------------------------------------------------------

# The model is VENDORED, not fetched.
#
# Until this issue this fixture ran `hf_hub_download("ggml-org/models-moved",
# "tinyllamas/stories15M-q4_0.gguf")` on every CI run, and PR #483's `tests` job
# went red on `429 Too Many Requests` from huggingface.co. A required check that
# can go red because somebody else's rate limiter had a bad minute is not a
# check — the same property a silently-skipped row breaks, from the other side.
#
# Why vendoring and not a cache: an `actions/cache` restore only sees caches
# From the run's own ref or the DEFAULT branch. This repo's PRs target
# `dev`, which is neither, so the cache would never once be warm without a
# scheduled master-side seeding job — machinery, for an immutable 1.1 MB file.
# Our own object storage means credentials in a public repo's CI; a release
# asset means out-of-band state to maintain. 1,185,376 bytes in git, once,
# beats all three.
#
# Why stories260K rather than the stories15M-q4_0 this replaced: same repo, same
# `llama` architecture, same role (it is llama.cpp's own CI model) — and 1.1 MB
# against 19.1 MB, which is the difference between a fixture and a 20% bump in
# clone size for every checkout forever. Nothing here measures the WEIGHTS; what
# is under test is our llama-server wrapper, and a real server serving a real
# GGUF is the production path either way.
_TINY_GGUF = Path(__file__).resolve().parent / "testdata" / "stories260K.gguf"
_TINY_GGUF_SHA256 = "270cba1bd5109f42d03350f60406024560464db173c0e387d91f0426d3bd256d"


def _llama_bin() -> str | None:
    env = os.environ.get("GEN_WORKER_LLAMA_SERVER_BIN", "")
    if env and Path(env).is_file():
        return env
    return shutil.which("llama-server")


needs_llama = pytest.mark.skipif(
    _llama_bin() is None,
    reason="llama-server binary not found (PATH or GEN_WORKER_LLAMA_SERVER_BIN)",
)


@pytest.fixture(scope="session")
def tiny_gguf_dir(tmp_path_factory) -> Path:
    """A directory holding the vendored GGUF — `resolve_gguf` takes a dir.

    The digest is checked rather than trusted: a truncated checkout (git-lfs not
    installed, a partial clone) would otherwise surface as an inscrutable
    llama-server boot failure instead of the one-line cause."""
    got = hashlib.sha256(_TINY_GGUF.read_bytes()).hexdigest()
    assert got == _TINY_GGUF_SHA256, (
        f"{_TINY_GGUF} is not the vendored model (sha256 {got}, expected "
        f"{_TINY_GGUF_SHA256})")
    snap = tmp_path_factory.mktemp("gguf-snap")
    (snap / _TINY_GGUF.name).symlink_to(_TINY_GGUF)
    return snap


@pytest.fixture()
def llama_path(monkeypatch) -> None:
    bin_path = _llama_bin()
    assert bin_path is not None
    monkeypatch.setenv("PATH", f"{Path(bin_path).parent}:{os.environ.get('PATH', '')}")


@needs_llama
@pytest.mark.integration
def test_real_server_stream_and_usage(tiny_gguf_dir, llama_path) -> None:
    handle = llama_server(str(tiny_gguf_dir)).start()
    try:
        deltas = list(completion_deltas(
            handle, "Once upon a time", max_tokens=16, temperature=0.0))
        text = "".join(d.text for d in deltas if isinstance(d, IncrementalTokenDelta))
        assert text.strip()
        usages = [d for d in deltas if isinstance(d, TokenUsage)]
        assert len(usages) == 1
        u = usages[0]
        assert u.prompt_tokens > 0
        assert 0 < u.completion_tokens <= 20
        assert u.tokens_per_second > 0

        chat = list(chat_deltas(
            handle, [{"role": "user", "content": "Tell me a story."}],
            max_tokens=8, temperature=0.0))
        assert any(isinstance(d, IncrementalTokenDelta) and d.text for d in chat)
        assert isinstance(chat[-1], TokenUsage)
    finally:
        handle.stop()
    assert not handle.alive


@needs_llama
@pytest.mark.integration
def test_real_server_terminal_output_via_executor(tiny_gguf_dir, llama_path) -> None:
    """Full pump path against a live llama-server: live chunks AND a
    retrievable terminal StreamResult with sane token counts."""
    from gen_worker.executor import Executor
    from gen_worker.pb import worker_scheduler_pb2 as pb
    from gen_worker.registry import EndpointSpec

    handle = llama_server(str(tiny_gguf_dir)).start()

    class _In(msgspec.Struct):
        prompt: str = "Once upon a time"

    def _complete(ctx, payload: _In) -> Iterator[IncrementalTokenDelta]:
        yield from completion_deltas(
            handle, payload.prompt, max_tokens=12, temperature=0.0,
            cancelled=lambda: ctx.cancelled)

    async def _go() -> List:
        sent: List = []

        async def _send(msg) -> None:
            sent.append(msg)

        spec = EndpointSpec(name="complete", method=_complete, kind="inference",
                            payload_type=_In, output_mode="stream")
        ex = Executor([spec], _send)
        await ex.handle_run_job(pb.RunJob(
            request_id="r1", attempt=1, function_name="complete",
            input_payload=msgspec.msgpack.encode(_In())))
        job = ex.jobs[("r1", 1)]
        assert job.task is not None
        await job.task
        return sent

    try:
        sent = asyncio.run(_go())
    finally:
        handle.stop()

    live = b"".join(
        m.job_progress.data for m in sent
        if m.WhichOneof("msg") == "job_progress"
        and m.job_progress.content_type == "text/plain")
    results = [m.job_result for m in sent if m.WhichOneof("msg") == "job_result"]
    assert results and results[-1].status == pb.JOB_STATUS_OK
    out = msgspec.msgpack.decode(results[-1].inline, type=StreamResult)
    assert out.text and out.text.encode() == live  # terminal == what streamed
    assert out.usage is not None and out.usage.completion_tokens > 0


@needs_llama
@pytest.mark.integration
def test_real_gguf_header_info(tiny_gguf_dir) -> None:
    info = read_gguf_info(resolve_gguf(tiny_gguf_dir))
    assert info.architecture == "llama"
    # `size_bytes` sums the shard files; the bound only has to separate "read a
    # real model" from "read a header and stopped" — the vendored file is
    # 1,185,376 bytes, so do not raise this to something it cannot clear.
    assert info.n_layers > 0 and info.size_bytes > 100_000
    plan = plan_fit(info, free_vram_gb=80.0)
    assert plan.n_gpu_layers == info.n_layers + 1


# ---------------------------------------------------------------------------
# GPU smoke (nightly lane): CUDA llama-server build required.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(os.environ.get("GEN_WORKER_GPU_SMOKE") != "1",
                    reason="GPU smoke runs on the nightly GPU CI lane")
@needs_llama
@pytest.mark.integration
def test_gpu_smoke_offload_and_measured_vram(tiny_gguf_dir, llama_path) -> None:
    from gen_worker.runtimes.server import process_vram_bytes

    boot = llama_server(str(tiny_gguf_dir))
    handle = boot.start()
    try:
        deltas = list(completion_deltas(handle, "Once", max_tokens=4, temperature=0.0))
        assert any(isinstance(d, IncrementalTokenDelta) and d.text for d in deltas)
        assert process_vram_bytes(handle.process.pid) > 0
    finally:
        handle.stop()
