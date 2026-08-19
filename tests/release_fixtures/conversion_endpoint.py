"""A CONVERSION producer — the pgw#1475 shape, verbatim.

`cast-dtype` and 24 of its siblings enter through `conversion/_common.py`'s
`source_from_ctx`, whose first line raises when `ctx.source_path` is falsy.
The raise site below is that one, character for character, so a green run
proves the PLATFORM materialized the reserved `source` and not that the check
was removed.

`ingest` is the OTHER half of the population: `clone-huggingface` names an
upstream URL and no reserved `source`, and it SUCCEEDED on the same release
in the same minute `cast-dtype` failed. Keeping both here means the fixture
partitions exactly where the defect did.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import msgspec

from gen_worker import RequestContext, ValidationError, entrypoint


class SourceRepo(msgspec.Struct, forbid_unknown_fields=True):
    """The reserved `source` struct the hub resolves and pins."""

    ref: str
    attributes: dict[str, str] = {}


class DestinationRepo(msgspec.Struct, forbid_unknown_fields=True):
    repo: str = ""
    release: str = ""


class CastInput(msgspec.Struct, forbid_unknown_fields=True):
    source: SourceRepo
    dtypes: list[str] = []
    destination: Optional[DestinationRepo] = None


class CastOutput(msgspec.Struct):
    source_path: str
    source_ref: str
    source_attributes: dict[str, str]
    tensor_bytes: int
    destination_repo: str


class IngestInput(msgspec.Struct, forbid_unknown_fields=True):
    upstream_url: str


class IngestOutput(msgspec.Struct):
    upstream_url: str
    source_path_is_none: bool


def _source_from_ctx(ctx: Any) -> tuple[Path, dict[str, Any]]:
    """`conversion/_common.source_from_ctx`, carried over unchanged — THE
    raise site the production failure came from, and the door all 25
    `source`-taking producers enter through."""
    if not ctx.source_path:
        raise ValidationError(
            "this function requires the reserved `source` payload field; "
            "the worker materializes it and populates ctx.source_path"
        )
    return Path(ctx.source_path), ctx.source or {}


@entrypoint(kind="conversion", publishes=True)
def cast_dtype(ctx: RequestContext, payload: CastInput) -> CastOutput:
    path, info = _source_from_ctx(ctx)
    # jobs#298's half: the materialized tree is PROJECTED — every
    # `*.safetensors` in it is a ~128 B `TFSSTUB1` pointer — and the producer
    # asks tier 3 for real bytes because it is about to hand the path to a
    # safetensors reader. The platform hands over the RAW tree; this call is
    # the consumer's, and it must still work on what the platform handed over.
    from gen_worker.models.materialized_view import third_party_dir

    real = Path(third_party_dir(path, why="pgw#1475 fixture reads tensors"))
    total = sum(
        p.stat().st_size for p in real.rglob("*")
        if p.is_file() and p.suffix == ".safetensors"
    )
    return CastOutput(
        source_path=str(path),
        source_ref=str(info.get("ref") or ""),
        source_attributes=dict(info.get("attributes") or {}),
        tensor_bytes=total,
        destination_repo=str((ctx.destination or {}).get("repo") or ""),
    )


@entrypoint(kind="conversion", publishes=True)
def ingest(ctx: RequestContext, payload: IngestInput) -> IngestOutput:
    """The `clone-huggingface` shape: no reserved `source` at all."""
    return IngestOutput(
        upstream_url=payload.upstream_url,
        source_path_is_none=ctx.source_path is None,
    )
