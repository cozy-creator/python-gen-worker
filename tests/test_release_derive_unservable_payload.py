"""pgw#1527: an unservable payload costs ONE payload, not the document.

A single member of a 7-payload sweep that the endpoint cannot serve used to
cost every graph the other six would produce — against the enumeration's own
"pre-warming completeness aid, never a correctness gate" doctrine and against
pgw#1449's skip-and-count precedent for entrypoints.

**The cut is deliberately narrow and the narrowness IS the fix.** Catch-and-
count only where the DEEPEST traceback frame is ENDPOINT-owned. An SDK-frame
exception stays fatal, because walls 1-8 were every one of them an SDK defect
surfacing exactly this way (the hollow session, the trace context, the
provenance walk, the output-integrity floor) and a blanket catch would have
turned all eight into quiet coverage gaps instead of the refusals that got
them fixed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("torch")
pytest.importorskip("diffusers")
pytest.importorskip("transformers")
import gen_worker._vendor.torchcg  # noqa: E402,F401

from gen_worker.release.derive import (  # noqa: E402
    deepest_endpoint_frame,
    endpoint_source_root,
)

FIXTURES = Path(__file__).resolve().parent / "release_fixtures"

LOCK = (
    "version = 1\n"
    '\n[[package]]\nname = "torch"\nversion = "2.13.0"\n'
    '\n[[package]]\nname = "triton"\nversion = "3.7.1"\n'
    '\n[[package]]\nname = "nvidia-cublas"\nversion = "13.1.1.3"\n'
    '\n[[package]]\nname = "diffusers"\nversion = "0.39.0"\n'
)


@pytest.fixture(scope="module")
def tree(tmp_path_factory: pytest.TempPathFactory) -> Path:
    sys.path.insert(0, str(FIXTURES))
    try:
        import tiny_tree
    finally:
        sys.path.remove(str(FIXTURES))
    return tiny_tree.save_config_only(tmp_path_factory.mktemp("unservable-configs"))


def _derive(tree: Path, out: Path, module: str) -> int:
    from gen_worker.cli import main

    lockfile = out.parent / "uv.lock"
    lockfile.write_text(LOCK)
    return main([
        "release", "derive",
        "--dir", str(FIXTURES),
        "--module", module,
        "--checkpoint", str(tree),
        "--lockfile", str(lockfile),
        "--out", str(out),
    ])


def test_the_servable_payload_still_produces_its_graphs(
    tree: Path, tmp_path: Path
) -> None:
    """Two payloads, one unservable: the document exists and carries the other."""

    out = tmp_path / "release.json"
    assert _derive(tree, out, "unservable_payload_endpoint") == 0

    (lane,) = json.loads(out.read_bytes())["graphs"]["lanes"]
    assert lane["unobserved_targets"] == []
    assert len(lane["graphs"]) >= 1


def test_the_skipped_payload_is_STATED_with_the_author_frame(
    tree: Path, tmp_path: Path
) -> None:
    """Louder than a missing one: index, frame and exception, per entrypoint."""

    out = tmp_path / "release.json"
    assert _derive(tree, out, "unservable_payload_endpoint") == 0

    row = json.loads(out.read_bytes())["entrypoints"]["generate"]["unservable"]
    assert len(row) == 1
    (skipped,) = row
    assert skipped["payload"] == 1
    assert "cannot serve the large bucket" in skipped["error"]
    assert skipped["error"].startswith("ValueError:")
    # The line that has to change, in the AUTHOR's file.
    assert skipped["frame"].startswith("unservable_payload_endpoint.py:")
    assert "_side_for" in skipped["frame"]


def test_an_endpoint_with_NOTHING_skipped_carries_no_such_key(
    tree: Path, tmp_path: Path
) -> None:
    """Absent, not empty — so existing documents cannot move."""

    out = tmp_path / "release.json"
    assert _derive(tree, out, "tiny_endpoint") == 0

    for entry in json.loads(out.read_bytes())["entrypoints"].values():
        assert "unservable" not in entry


# ---------------------------------------------------------------------------
# The predicate itself. These are the assertions that keep the cut narrow.


def _raise_from(where: Any) -> BaseException:
    try:
        where()
    except Exception as exc:  # noqa: BLE001 - the subject
        return exc
    raise AssertionError("expected a raise")


def test_an_SDK_frame_is_NEVER_endpoint_owned() -> None:
    """Walls 1-8 all looked like this, and all of them must stay fatal."""

    from gen_worker.release.derive import _auto_payloads

    class NotAStruct:
        pass

    # A REAL raise from inside gen_worker: the deepest frame is derive.py's
    # own line, not this file's. (Raising an SDK-defined exception FROM here
    # would prove nothing — the frame is what is classified, not the class.)
    import traceback

    import gen_worker

    exc = _raise_from(lambda: _auto_payloads("@entrypoint x", NotAStruct))
    sdk = Path(gen_worker.__file__).resolve().parent
    # Pin the PREMISE, or the assertions below could pass for the wrong
    # reason: the deepest frame really is inside the SDK.
    deepest = Path(traceback.extract_tb(exc.__traceback__)[-1].filename).resolve()
    assert deepest.is_relative_to(sdk), deepest

    # Even with the endpoint root set to the SDK's OWN directory — the most
    # permissive root that could possibly be passed — an SDK frame is refused.
    assert deepest_endpoint_frame(exc, sdk) is None
    assert deepest_endpoint_frame(exc, Path(__file__).resolve().parent) is None


def test_a_frame_OUTSIDE_the_endpoint_root_is_not_claimed() -> None:
    """torch/diffusers/stdlib raise from their own files: still fatal.

    Conservative on purpose: author code that trips a shape error inside torch
    reports a TORCH frame, and this cannot tell that apart from an SDK-induced
    one without guessing.
    """

    exc = _raise_from(lambda: json.loads("{bad"))
    assert deepest_endpoint_frame(exc, FIXTURES) is None


def test_an_ENDPOINT_frame_IS_claimed() -> None:
    sys.path.insert(0, str(FIXTURES))
    try:
        import unservable_payload_endpoint as ep
    finally:
        sys.path.remove(str(FIXTURES))

    exc = _raise_from(lambda: ep._side_for(ep.Size.LARGE))
    frame = deepest_endpoint_frame(exc, endpoint_source_root(ep))
    assert frame is not None
    assert frame.name == "_side_for"
    assert Path(frame.filename).name == "unservable_payload_endpoint.py"


def test_an_UNKNOWABLE_root_means_fatal() -> None:
    """"Unsure" must always resolve to "fatal", never to "skip"."""

    exc = _raise_from(lambda: 1 / 0)
    assert deepest_endpoint_frame(exc, None) is None


def test_the_endpoint_root_is_the_TOP_LEVEL_package_directory() -> None:
    """A package endpoint's helpers count as its own code.

    h3 raises from `long_video.py`, not from `main.py`, so a root narrowed to
    the main module's file would miss every real case.
    """

    sys.path.insert(0, str(FIXTURES))
    try:
        import unservable_payload_endpoint as ep
    finally:
        sys.path.remove(str(FIXTURES))

    assert endpoint_source_root(ep) == FIXTURES
