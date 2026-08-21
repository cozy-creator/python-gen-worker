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


@entrypoint(kind="conversion", publishes=True)
def cast_dtype(ctx: RequestContext, payload: CastDtypeInput) -> PublishResult:
    """`conversion/src/conversion/transform.py:137`, re-decorated."""
    return PublishResult(published=list(payload.dtypes))


@entrypoint(kind="conversion", publishes=True,
            env=("HF_TOKEN", "CIVITAI_API_KEY"))
def clone_repo(ctx: RequestContext, payload: CloneInput) -> PublishResult:
    """`conversion/src/conversion/mirror.py:194`, re-decorated."""
    return PublishResult(published=[payload.destination_repo])


@entrypoint(
    kind="conversion", resources=Resources(gpu=True),
    publishes=True, emits_media=True,
)
def quality_matrix(ctx: RequestContext, payload: MatrixInput) -> MatrixResult:
    """`conversion/src/conversion/quality_matrix.py:389`, re-decorated."""
    return MatrixResult(rows=len(payload.prompts))


@entrypoint(kind="eval", resources=Resources(gpu=True, vcpus=8),
            emits_media=True)
def score_bench(ctx: RequestContext, payload: BenchInput) -> BenchResult:
    """`conversion/src/conversion/score_bench.py:383`: media, NO repo."""
    return BenchResult(score=float(len(payload.prompts)))


@entrypoint
def describe(ctx: RequestContext, payload: DescribeInput) -> DescribeResult:
    """Declares nothing — the undeclared control."""
    return DescribeResult(summary=payload.ref)
