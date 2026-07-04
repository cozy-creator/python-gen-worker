"""Registry spec names are canonical slugs — the ONE wire/dispatch vocabulary.

The orchestrator's canonical function name is the slug (`_` -> `-`, tensorhub
builder.NormalizeSlug); the discovery manifest publishes the same slug. The
worker must advertise and match RunJob.function_name on it, so EndpointSpec
names slugify at extraction time.
"""

import msgspec

from gen_worker import RequestContext, endpoint
from gen_worker.registry import extract_specs


class _In(msgspec.Struct):
    text: str = ""


class _Out(msgspec.Struct):
    response: str


@endpoint
class SnakeCase:
    def marco_polo(self, ctx: RequestContext, data: _In) -> _Out:
        return _Out(response="polo")

    def marco_polo_slow(self, ctx: RequestContext, data: _In) -> _Out:
        return _Out(response="polo")


def test_spec_names_are_slugs() -> None:
    names = sorted(s.name for s in extract_specs(SnakeCase))
    assert names == ["marco-polo", "marco-polo-slow"]
    # python attr names survive separately for manifest python_name.
    attrs = sorted(s.attr_name for s in extract_specs(SnakeCase))
    assert attrs == ["marco_polo", "marco_polo_slow"]
