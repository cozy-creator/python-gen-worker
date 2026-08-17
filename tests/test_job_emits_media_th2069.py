"""``@job(emits_media=True)`` — the media sibling of ``publishes``.

The hub mints a job's ``upload_media`` grant off the release's declaration
(th#2069), so an undeclared job's token cannot upload media. This fences the
pgw half: the declaration exists on the decorator, reaches the manifest row the
hub reads, and an undeclared media write is a typed refusal at the call site
instead of a hub 403 after the bytes moved.

Endpoints are unaffected by construction — media IS an endpoint's product, so
the authority comes with the kind and there is nothing to declare.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import msgspec
import pytest

from gen_worker import JobContext, job
from gen_worker.api.errors import MediaNotDeclaredError
from gen_worker.jobs import JobDispatch, execute_job
from gen_worker.registry import extract_job_spec


class In(msgspec.Struct):
    pass


class Out(msgspec.Struct):
    declared: bool = False


@job(emits_media=True, name="declared-media")
def declared_media(ctx: JobContext, spec: In) -> Out:
    ctx.save_bytes("outputs/report.json", b"{}")
    return Out(declared=ctx.emits_media)


@job(name="silent-media")
def silent_media(ctx: JobContext, spec: In) -> Out:
    ctx.save_bytes("outputs/report.json", b"{}")
    return Out(declared=ctx.emits_media)


def _ctx(tmp_path: Path, *, emits_media: Any = None) -> JobContext:
    return JobContext(
        request_id="r-2069", job_id="j-2069",
        emits_media=emits_media,
        local_output_dir=str(tmp_path),
        execution_hints={"kind": "job"},
    )


def _run(fn: Any, ctx: JobContext) -> Any:
    spec = extract_job_spec(fn)
    assert spec is not None
    return execute_job(
        JobDispatch(job_name=spec.name, payload=msgspec.msgpack.encode(In())),
        jobs={spec.name: spec}, ctx=ctx, reraise=True,
    )


# ---- the declaration surface ---------------------------------------------


def test_the_declaration_reaches_the_spec_and_defaults_off() -> None:
    assert extract_job_spec(declared_media).emits_media is True  # type: ignore[union-attr]
    assert extract_job_spec(silent_media).emits_media is False  # type: ignore[union-attr]


def test_it_is_independent_of_publishes() -> None:
    """An eval job writes media and NO repo; a quality matrix writes both.
    Reading one off the other is the mistake this sibling exists to prevent."""

    @job(emits_media=True, publishes=False, name="eval-only")
    def eval_only(ctx: JobContext, spec: In) -> Out:
        return Out()

    @job(emits_media=True, publishes=True, name="matrix")
    def matrix(ctx: JobContext, spec: In) -> Out:
        return Out()

    assert (extract_job_spec(eval_only).emits_media,  # type: ignore[union-attr]
            extract_job_spec(eval_only).publishes) == (True, False)  # type: ignore[union-attr]
    assert (extract_job_spec(matrix).emits_media,  # type: ignore[union-attr]
            extract_job_spec(matrix).publishes) == (True, True)  # type: ignore[union-attr]


def test_the_manifest_row_carries_it_because_that_is_what_the_hub_reads() -> None:
    from gen_worker.discovery.discover import _job_entry

    row = _job_entry(extract_job_spec(declared_media), Path.cwd())
    assert row["emits_media"] is True
    assert _job_entry(extract_job_spec(silent_media), Path.cwd())["emits_media"] is False


# ---- the fence, both directions ------------------------------------------


def test_an_undeclared_job_is_refused_before_a_byte_moves(tmp_path: Path) -> None:
    with pytest.raises(MediaNotDeclaredError) as caught:
        _run(silent_media, _ctx(tmp_path))
    assert caught.value.surface == "save_bytes"
    assert "emits_media=True" in str(caught.value)
    assert not list(tmp_path.rglob("report.json"))


def test_a_declared_job_mints_the_media(tmp_path: Path) -> None:
    outcome = _run(declared_media, _ctx(tmp_path))
    assert msgspec.msgpack.decode(outcome.result)["declared"] is True
    assert [p.name for p in tmp_path.rglob("report.json")] == ["report.json"]


def test_save_file_is_fenced_too(tmp_path: Path) -> None:
    src = tmp_path / "src.bin"
    src.write_bytes(b"\x00" * 8)
    ctx = _ctx(tmp_path / "out", emits_media=False)
    with pytest.raises(MediaNotDeclaredError) as caught:
        ctx.save_file("outputs/copy.bin", src)
    assert caught.value.surface == "save_file"


def test_an_endpoint_declares_nothing_and_writes_media(tmp_path: Path) -> None:
    """The gate is JOBS-ONLY. An undeclared context (`None`) is an endpoint,
    whose product IS media — refusing there would break every endpoint."""
    ctx = _ctx(tmp_path)
    assert ctx.emits_media is True
    ctx.save_bytes("outputs/image.webp", b"\x00" * 4)


def test_the_result_envelope_is_not_media(tmp_path: Path) -> None:
    """Worker->orchestrator transport rides no media grant, so an undeclared
    job still returns its result struct rather than failing to report."""
    ctx = _ctx(tmp_path, emits_media=False)
    ctx._save_result_envelope("results/r-2069.msgpack", b"\x00" * 4)


# ---- one home for the declaration ----------------------------------------


def test_execute_job_stamps_it_so_a_dispatch_head_cannot_forget(
    tmp_path: Path,
) -> None:
    ctx = _ctx(tmp_path)              # head passed nothing
    _run(declared_media, ctx)
    assert ctx.emits_media is True


def test_a_caller_may_not_grant_media_authority_the_release_never_declared(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="declaration is the release's"):
        _run(silent_media, _ctx(tmp_path, emits_media=True))
