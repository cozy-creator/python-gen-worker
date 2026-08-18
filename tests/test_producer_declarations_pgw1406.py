"""The producer plane's declarations on ``@entrypoint`` (pgw#1406 / th#2173).

pgw#983 deleted ``@job``. All 27 conversion producers in ``cozy-creator/jobs``
are ``@job`` functions and NOTHING there is written against ``@entrypoint``;
the plane keeps running only because ``conversion/pyproject.toml:198`` pins
the pre-#983 SDK. These tests pin the successor: the three declarations, what
they emit, and what they refuse.

The kwargs' semantics are v1's (``git show 722e30b9^:src/gen_worker/api/jobs.py``)
with two deliberate deltas, both named at their assertion:

1. ``emits_media`` is TRI-STATE here and a bool on ``@job``. Undeclared must
   stay the inference default, because on the request path the hub grants
   ``upload_media`` unconditionally.
2. ``emits_media`` is NOT emitted on the manifest row. Measured at tensorhub
   master: ``manifestFunction`` has no decoder for it, and
   ``capability_subject_th2068.go:124`` sets ``UploadsMedia: true`` for every
   request. A key on the wire that nothing reads is th#2087's own defect.

``publishes`` and ``env`` need NO hub change at all: the hub already decodes
``functions[].publishes`` / ``functions[].env``, and ``entrypoints[]`` folds
into ``Functions`` at ParseManifest's one decode site (th#2146).
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType
from typing import Iterator

import pytest

from gen_worker.api.errors import MediaNotDeclaredError, PublishNotDeclaredError
from gen_worker.discovery.entrypoints_v2 import discover_entrypoints
from gen_worker.serving.context import RequestContext
from gen_worker.serving.entrypoints import ENTRYPOINT_ATTR, EntrypointDeclarationError

FIXTURES = Path(__file__).resolve().parent / "release_fixtures"
MODULE = "producer_endpoint"


@pytest.fixture(scope="module")
def producers() -> Iterator[ModuleType]:
    sys.path.insert(0, str(FIXTURES))
    try:
        yield importlib.import_module(MODULE)
    finally:
        sys.path.remove(str(FIXTURES))


@pytest.fixture(scope="module")
def rows(producers: ModuleType) -> dict[str, dict]:
    return {row["name"]: row for row in discover_entrypoints(MODULE)}


def spec(fn: object) -> object:
    return getattr(fn, ENTRYPOINT_ATTR)


# -- the declarations reach the spec ----------------------------------------


def test_publishes_reaches_the_spec(producers: ModuleType) -> None:
    assert spec(producers.cast_dtype).publishes is True
    assert spec(producers.clone_repo).publishes is True
    assert spec(producers.quality_matrix).publishes is True
    # score_bench emits media and writes NO repo — the two are independent in
    # both directions, exactly as they were on `@job`.
    assert spec(producers.score_bench).publishes is False
    assert spec(producers.describe).publishes is False


def test_env_is_normalized_to_a_tuple(producers: ModuleType) -> None:
    assert spec(producers.clone_repo).env == ("HF_TOKEN", "CIVITAI_API_KEY")
    assert spec(producers.cast_dtype).env == ()


def test_kind_reaches_the_spec_and_the_row(
    producers: ModuleType, rows: dict[str, dict]
) -> None:
    """The fourth gap, and the one the three ruled kwargs do not cover.

    A producer that leaves `kind` at `inference` gets
    `NamesReservedRefs(kind) == false` hub-side, so the reserved `source`
    payload field it depends on is never resolved and never granted — the body
    raises on its own first line. The vocabulary is copied verbatim from
    `internal/functionkind.All`."""
    assert spec(producers.cast_dtype).kind == "conversion"
    assert spec(producers.score_bench).kind == "eval"
    assert spec(producers.describe).kind == ""

    assert rows["cast_dtype"]["kind"] == "conversion"
    assert rows["score_bench"]["kind"] == "eval"
    # Undeclared is the hub's own default, unchanged.
    assert rows["describe"]["kind"] == "inference"


def test_an_unknown_kind_is_refused() -> None:
    """A value the hub does not normalize would silently become `inference`
    — the exact silence this kwarg exists to end."""
    with pytest.raises(EntrypointDeclarationError, match="kind= must be one of"):
        _probe(
            "import msgspec\n"
            "from gen_worker import RequestContext, entrypoint\n"
            "class I(msgspec.Struct): pass\n"
            "class O(msgspec.Struct): pass\n"
            "@entrypoint(kind='quantization')\n"
            "def f(ctx: RequestContext, payload: I) -> O: ...\n",
            "pgw1406_bad_kind",
        )


def test_emits_media_is_tri_state(producers: ModuleType) -> None:
    """DELIBERATE DELTA 1 from `@job`, which defaulted to False.

    On the request path the hub grants `upload_media` unconditionally
    (`capability_subject_th2068.go:124`), so an UNDECLARED entrypoint must
    read as "not job-scoped" — None — and keep today's behaviour. Only an
    explicit False opts out."""
    assert spec(producers.quality_matrix).emits_media is True
    assert spec(producers.score_bench).emits_media is True
    assert spec(producers.describe).emits_media is None
    assert spec(producers.cast_dtype).emits_media is None


# -- what the manifest carries ----------------------------------------------


def test_the_manifest_row_carries_publishes_and_env(rows: dict[str, dict]) -> None:
    assert rows["cast_dtype"]["publishes"] is True
    assert rows["clone_repo"]["publishes"] is True
    assert rows["clone_repo"]["env"] == ["HF_TOKEN", "CIVITAI_API_KEY"]
    assert rows["quality_matrix"]["publishes"] is True


def test_a_non_publishing_row_omits_the_key(rows: dict[str, dict]) -> None:
    """`omitempty` on both sides: absent IS "did not declare", which is the
    fail-closed reading the hub's capability minter already uses."""
    assert "publishes" not in rows["score_bench"]
    assert "publishes" not in rows["describe"]
    assert "env" not in rows["describe"]


def test_emits_media_is_not_on_the_wire(rows: dict[str, dict]) -> None:
    """DELIBERATE DELTA 2, and the reason is a measurement at tensorhub master.

    `manifestFunction` decodes `publishes` and `env`; it has NO `emits_media`
    field, and the request path's `UploadsMedia` is an unconditional `true`.
    Emitting the key would be a mirror with no reader — th#2087's fence exists
    to catch exactly that. The declaration is enforced SDK-side instead (see
    `test_an_explicit_emits_media_false_refuses_media`)."""
    for row in rows.values():
        assert "emits_media" not in row


def test_the_undeclared_row_is_byte_identical(rows: dict[str, dict]) -> None:
    """pgw#1406 must be invisible to every endpoint that does not use it."""
    assert set(rows["describe"]) == {
        "name", "python_name", "module", "declared_module", "class_name",
        "kind", "input_schema", "payload_schema_sha256", "output_schema",
        "output_schema_sha256", "incremental_output", "slots",
    }


# -- what the declarations REFUSE -------------------------------------------


def _probe(source: str, name: str) -> None:
    module = type(sys)(name)
    module.__dict__["__name__"] = name
    sys.modules[name] = module
    try:
        exec(compile(source, name, "exec"), module.__dict__)
    finally:
        sys.modules.pop(name, None)


def test_a_bare_string_env_is_refused() -> None:
    """v1's rule, kept: a string would iterate into characters."""
    with pytest.raises(EntrypointDeclarationError, match="iterate into"):
        _probe(
            "import msgspec\n"
            "from gen_worker import RequestContext, entrypoint\n"
            "class I(msgspec.Struct): pass\n"
            "class O(msgspec.Struct): pass\n"
            "@entrypoint(env='HF_TOKEN')\n"
            "def f(ctx: RequestContext, payload: I) -> O: ...\n",
            "pgw1406_env_str",
        )


@pytest.mark.parametrize(
    "bad, match",
    [("hf_token", "not a valid"), ("9LIVES", "not a valid"), ("A-B", "not a valid")],
)
def test_an_invalid_env_name_is_refused(bad: str, match: str) -> None:
    with pytest.raises(EntrypointDeclarationError, match=match):
        _probe(
            "import msgspec\n"
            "from gen_worker import RequestContext, entrypoint\n"
            "class I(msgspec.Struct): pass\n"
            "class O(msgspec.Struct): pass\n"
            f"@entrypoint(env=({bad!r},))\n"
            "def f(ctx: RequestContext, payload: I) -> O: ...\n",
            f"pgw1406_env_{abs(hash(bad))}",
        )


def test_a_repeated_env_name_is_refused() -> None:
    with pytest.raises(EntrypointDeclarationError, match="repeats"):
        _probe(
            "import msgspec\n"
            "from gen_worker import RequestContext, entrypoint\n"
            "class I(msgspec.Struct): pass\n"
            "class O(msgspec.Struct): pass\n"
            "@entrypoint(env=('HF_TOKEN', 'HF_TOKEN'))\n"
            "def f(ctx: RequestContext, payload: I) -> O: ...\n",
            "pgw1406_env_dupe",
        )


def test_publishes_must_be_a_bool() -> None:
    with pytest.raises(EntrypointDeclarationError, match="publishes= is a bool"):
        _probe(
            "import msgspec\n"
            "from gen_worker import RequestContext, entrypoint\n"
            "class I(msgspec.Struct): pass\n"
            "class O(msgspec.Struct): pass\n"
            "@entrypoint(publishes='yes')\n"
            "def f(ctx: RequestContext, payload: I) -> O: ...\n",
            "pgw1406_pub_str",
        )


# -- the SDK half of "one fact, two enforcers" ------------------------------


def test_an_undeclared_function_is_refused_the_publisher_surface() -> None:
    """The hub mints no repo-write grant for an undeclared function, so the
    refusal arrives at the call site instead of after an upload."""
    ctx: RequestContext = RequestContext("req-undeclared", publishes=False)
    with pytest.raises(PublishNotDeclaredError):
        ctx._require_publish_declaration("save_checkpoint")


def test_a_declared_producer_reaches_the_publisher_surface() -> None:
    ctx: RequestContext = RequestContext("req-declared", publishes=True)
    ctx._require_publish_declaration("save_checkpoint")  # does not raise


def test_an_explicit_emits_media_false_refuses_media() -> None:
    ctx: RequestContext = RequestContext("req-nomedia", emits_media=False)
    with pytest.raises(MediaNotDeclaredError):
        ctx._require_media_declaration("save_file")


def test_an_undeclared_context_keeps_intrinsic_media_authority() -> None:
    """Every endpoint that never says the word is unchanged: media IS the
    product of a request, and the hub grants it unconditionally."""
    ctx: RequestContext = RequestContext("req-plain")
    ctx._require_media_declaration("save_file")  # does not raise
    assert ctx.emits_media is True


# -- the serve loop stamps the declaration onto the body's context ----------


def test_the_serve_loop_stamps_the_declaration(producers: ModuleType) -> None:
    """The declaration is useless if the running body never sees it. This is
    the wire from the spec to `ctx.publishes` the hub's grant mirrors."""
    from gen_worker.serving.serve_loop import ServeLoop

    host = object.__new__(ServeLoop)
    host._context_kwargs = {}
    host._output_dir = None

    producer = host._make_context("r1", None, spec(producers.cast_dtype))
    assert producer.publishes is True

    control = host._make_context("r2", None, spec(producers.describe))
    assert control.publishes is False
    assert control.emits_media is True  # undeclared -> intrinsic


def test_the_producer_context_carries_the_producer_surface() -> None:
    """ONE context class (pgw#1294/pgw#1306): `mktemp` and the reserved
    source/destination contract are on the class every entrypoint receives,
    and the DECLARATION — not the class — decides what may be written."""
    ctx: RequestContext = RequestContext(
        "req-producer", publishes=True, source_info={"ref": "org/model:main"}
    )
    scratch = ctx.mktemp()
    assert scratch.is_dir()
    assert ctx.mktemp() != scratch
    assert ctx.source == {"ref": "org/model:main"}
    assert ctx.source_path is None
    ctx._set_source_path(str(scratch))
    assert ctx.source_path == str(scratch)
