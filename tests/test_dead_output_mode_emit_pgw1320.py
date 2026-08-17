"""pgw#1320 — the manifest carries ONE spelling of output cardinality.

`discover.py` used to emit `output_mode: "incremental"|"single"` beside
`incremental_output: bool`. The hub has never decoded the first one:
`output_mode` appears nowhere in tensorhub's `internal/builder`, which
decodes `incremental_output` (`builder/domain.go`,
`builder/manifest_contract.go`) and republishes it under that name.

Three spellings existed for one bit — `EndpointSpec.output_mode`
("single"|"stream") in process, `output_mode` ("incremental"|"single") on
the manifest, `incremental_output` (bool) on the wire. The middle one was
read by nobody and is gone.

The fence runs `discover_manifest` — the exact entry point the image's
build step runs — so it is a property of the manifest a real build emits,
not of a fixture somebody kept in sync by hand.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

# The one spelling the hub decodes, and the one it never has.
LIVE_KEY = "incremental_output"
DEAD_KEY = "output_mode"


def _endpoint_tree(root: Path) -> None:
    """A toy endpoint with BOTH cardinalities — a struct-returning function
    and an Iterator-returning one, which is what `_inspect_return` turns
    into "single" vs "stream"."""
    (root / "pyproject.toml").write_text(textwrap.dedent("""
        [project]
        name = "ep1320"

        [tool.gen_worker]
        main = "ep1320.main"
    """))
    src = root / "ep1320"
    src.mkdir()
    (src / "__init__.py").write_text("")
    (src / "main.py").write_text(textwrap.dedent("""
        from typing import Iterator

        import msgspec
        from gen_worker import RequestContext, Resources, endpoint

        class In_(msgspec.Struct):
            prompt: str = ""

        class Out_(msgspec.Struct):
            y: str = ""

        @endpoint(resources=Resources(gpu=False))
        class Both:
            def setup(self) -> None: ...

            def whole(self, ctx: RequestContext, data: In_) -> Out_:
                return Out_()

            def streamed(self, ctx: RequestContext, data: In_) -> Iterator[Out_]:
                yield Out_()
    """))


@pytest.fixture()
def functions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    from gen_worker.discovery.discover import discover_manifest

    _endpoint_tree(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    return list(discover_manifest(tmp_path)["functions"])


def test_the_walk_is_not_vacuous(functions: list[dict]) -> None:
    """The control for the absence assertion below. An absent-key fence
    passes just as well when the discovery walk found nothing, so prove
    the walk produced BOTH cardinalities before believing a "not in"."""
    by_name = {fn["name"]: fn for fn in functions}
    assert set(by_name) == {"whole", "streamed"}, by_name.keys()
    assert by_name["whole"][LIVE_KEY] is False
    assert by_name["streamed"][LIVE_KEY] is True


def test_the_manifest_carries_no_output_mode(functions: list[dict]) -> None:
    """pgw#1320: the dead key is gone from a REAL discovery manifest.

    Restoring the emit fails here by name — and the hub could not tell,
    because it has no decoder for it to fail in."""
    carriers = [fn["name"] for fn in functions if DEAD_KEY in fn]
    assert carriers == [], (
        f"{DEAD_KEY!r} is back on {carriers} — the hub has never decoded it "
        f"({DEAD_KEY} appears nowhere in tensorhub/internal/builder); "
        f"{LIVE_KEY} is the fact and the only spelling the hub reads"
    )


def test_no_manifest_block_reintroduces_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Scoped to the function rows above, the fence would miss the key
    reappearing on a sibling manifest block (jobs, lanes, decode set).
    Assert it against the whole document instead."""
    import json

    from gen_worker.discovery.discover import discover_manifest

    _endpoint_tree(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    doc = json.dumps(discover_manifest(tmp_path))

    assert LIVE_KEY in doc, "the walk emitted no cardinality at all"
    assert f'"{DEAD_KEY}"' not in doc, (
        f"{DEAD_KEY!r} reappeared somewhere in the manifest document"
    )
