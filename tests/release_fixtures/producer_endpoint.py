"""The PRODUCER plane, v2-spelled (pgw#1406 / th#2173).

pgw#983 deleted ``@job``, and every one of the 27 conversion producers in
``cozy-creator/jobs`` is a ``@job``. Their whole declaration set is three
kwargs — ``publishes=`` (22 of 27), ``env=`` (3), ``emits_media=`` (4) — so
this fixture is those three shapes, spelled on ``@entrypoint``:

* ``cast_dtype`` — ``publishes=True``, weightless, CPU. The real
  ``transform.py:137`` shape: it reads the reserved ``source`` contract and
  writes a checkpoint.
* ``clone_repo`` — ``publishes=True`` + ``env=("HF_TOKEN",
  "CIVITAI_API_KEY")``. The ``mirror.py`` shape.
* ``quality_matrix`` — ``publishes=True, emits_media=True``, GPU.
* ``score_bench`` — ``emits_media=True`` and NO ``publishes``: an eval that
  writes a report and no repo. It must be REFUSED at the publisher surface.
* ``describe`` — declares NOTHING. Byte-identical to a pre-pgw#1406 row, and
  refused at the publisher surface.
"""

from __future__ import annotations

import msgspec

from gen_worker import RequestContext, Resources, entrypoint


class SourceRepo(msgspec.Struct, forbid_unknown_fields=True):
    ref: str


class CastDtypeInput(msgspec.Struct, forbid_unknown_fields=True):
    source: SourceRepo
    destination_repo: str
    dtypes: list[str]


class PublishResult(msgspec.Struct):
    published: list[str]


class CloneInput(msgspec.Struct, forbid_unknown_fields=True):
    source: SourceRepo
    destination_repo: str


class MatrixInput(msgspec.Struct, forbid_unknown_fields=True):
    source: SourceRepo
    prompts: list[str]


class MatrixResult(msgspec.Struct):
    rows: int


class BenchInput(msgspec.Struct, forbid_unknown_fields=True):
    prompts: list[str]


class BenchResult(msgspec.Struct):
    score: float


class DescribeInput(msgspec.Struct, forbid_unknown_fields=True):
    ref: str


class DescribeResult(msgspec.Struct):
    summary: str


@entrypoint(publishes=True)
def cast_dtype(ctx: RequestContext, payload: CastDtypeInput) -> PublishResult:
    """`conversion/src/conversion/transform.py:137`, re-decorated."""
    return PublishResult(published=list(payload.dtypes))


@entrypoint(publishes=True, env=("HF_TOKEN", "CIVITAI_API_KEY"))
def clone_repo(ctx: RequestContext, payload: CloneInput) -> PublishResult:
    """`conversion/src/conversion/mirror.py:194`, re-decorated."""
    return PublishResult(published=[payload.destination_repo])


@entrypoint(
    resources=Resources(gpu=True), publishes=True, emits_media=True
)
def quality_matrix(ctx: RequestContext, payload: MatrixInput) -> MatrixResult:
    """`conversion/src/conversion/quality_matrix.py:389`, re-decorated."""
    return MatrixResult(rows=len(payload.prompts))


@entrypoint(resources=Resources(gpu=True, vcpus=8), emits_media=True)
def score_bench(ctx: RequestContext, payload: BenchInput) -> BenchResult:
    """`conversion/src/conversion/score_bench.py:383`: media, NO repo."""
    return BenchResult(score=float(len(payload.prompts)))


@entrypoint
def describe(ctx: RequestContext, payload: DescribeInput) -> DescribeResult:
    """Declares nothing — the undeclared control."""
    return DescribeResult(summary=payload.ref)
