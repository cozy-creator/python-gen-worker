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
    _third_party_root,
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

    import traceback

    import gen_worker

    exc = _raise_from(lambda: _auto_payloads("@entrypoint x", NotAStruct))
    sdk = Path(gen_worker.__file__).resolve().parent
    deepest = Path(traceback.extract_tb(exc.__traceback__)[-1].filename).resolve()
    assert deepest.is_relative_to(sdk), deepest

    assert deepest_endpoint_frame(exc, sdk) is None
    assert deepest_endpoint_frame(exc, Path(__file__).resolve().parent) is None


def test_a_frame_OUTSIDE_the_endpoint_root_is_not_claimed() -> None:
    """torch/diffusers/stdlib raise from their own files: still fatal."""

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
    """A package endpoint's helpers count as its own code."""

    sys.path.insert(0, str(FIXTURES))
    try:
        import unservable_payload_endpoint as ep
    finally:
        sys.path.remove(str(FIXTURES))

    assert endpoint_source_root(ep) == FIXTURES


def test_a_SIBLING_top_level_module_is_endpoint_owned(tmp_path: Path) -> None:
    """h3's actual shape, and the hole that killed its run."""

    sys.path.insert(0, str(FIXTURES))
    try:
        import sibling_helper
        import tiny_endpoint
    finally:
        sys.path.remove(str(FIXTURES))

    root = endpoint_source_root(tiny_endpoint)
    exc = _raise_from(lambda: sibling_helper.refuse_unservable(60))
    frame = deepest_endpoint_frame(exc, root)

    assert frame is not None, "a sibling author module must be endpoint-owned"
    assert Path(frame.filename).name == "sibling_helper.py"
    assert frame.name == "refuse_unservable"


def test_a_PACKAGE_endpoints_root_is_the_directory_ABOVE_the_package() -> None:
    """That is what makes a sibling reachable at all."""

    sys.path.insert(0, str(FIXTURES))
    try:
        import modular_tiny_endpoint
    finally:
        sys.path.remove(str(FIXTURES))

    assert endpoint_source_root(modular_tiny_endpoint) == FIXTURES


def test_the_SUBTRACTION_keeps_an_SDK_frame_fatal_even_under_a_WIDE_root() -> None:
    """The original concern, tested at its worst case."""

    import gen_worker
    from gen_worker.release.derive import _auto_payloads

    class NotAStruct:
        pass

    exc = _raise_from(lambda: _auto_payloads("@entrypoint x", NotAStruct))
    sdk = Path(gen_worker.__file__).resolve().parent
    assert deepest_endpoint_frame(exc, Path(sdk.anchor)) is None
    assert deepest_endpoint_frame(exc, sdk) is None
    assert deepest_endpoint_frame(exc, sdk.parent) is None


def test_the_SUBTRACTION_keeps_a_THIRD_PARTY_frame_fatal_under_a_wide_root() -> None:
    """torch/diffusers live in site-packages, and a wide root would swallow it."""

    import traceback

    from diffusers import AutoencoderKL

    exc = _raise_from(lambda: AutoencoderKL.load_config("/nonexistent-tree"))
    deepest = Path(traceback.extract_tb(exc.__traceback__)[-1].filename).resolve()
    assert _third_party_root(deepest), deepest

    assert deepest_endpoint_frame(exc, Path(deepest.anchor)) is None


def test_an_endpoint_INSTALLED_into_site_packages_claims_nothing() -> None:
    """The degenerate case resolves the SAFE way."""

    import traceback

    from diffusers import AutoencoderKL

    exc = _raise_from(lambda: AutoencoderKL.load_config("/nonexistent-tree"))
    installed = Path(traceback.extract_tb(exc.__traceback__)[-1].filename).resolve()
    assert _third_party_root(installed)

    assert deepest_endpoint_frame(exc, installed.parent) is None
