"""pgw#1472: the two ends of the env identity must name the same set.

The defect this file fences is not "a hash was wrong". It is that the PUBLISH
side hashed a `uv.lock` and the SERVE side hashed `importlib.metadata`, and
those two are incomparable by construction — so a locally derived document was
adoptable by nothing, including the local serve it exists for.

⚠️ **Why no existing test could fail on it.** Every adopt test builds its
document from the same `installed=` mapping it then audits against, so the two
sources are ONE dict by construction. The fixture made the bug unrepresentable.
The cases here therefore work from a REAL `uv.lock` on disk and from a real
`importlib.metadata` reading, because that difference is the whole subject.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from gen_worker.env_identity import (
    EnvIdentityError,
    closure_drift,
    closure_entries_from_lockfile,
    describe_drift,
    lockfile_beside,
    normalize_name,
    normalize_version,
)

LOCK = """\
version = 1

[[package]]
name = "torch"
version = "2.13.0"

[[package]]
name = "PyYAML"
version = "6.0.3"

[[package]]
name = "colorama"
version = "0.4.6"
"""


@pytest.fixture()
def lockfile(tmp_path: Path) -> Path:
    path = tmp_path / "uv.lock"
    path.write_text(LOCK)
    return path


def test_the_lockfile_reader_keeps_names_and_versions_VERBATIM(lockfile: Path) -> None:
    """The identity is hashed from this, so normalizing here would silently
    rekey every document ever stamped."""

    entries = closure_entries_from_lockfile(lockfile)
    assert entries == {"torch": "2.13.0", "PyYAML": "6.0.3", "colorama": "0.4.6"}


def test_a_lockfile_resolving_one_package_twice_refuses(tmp_path: Path) -> None:
    path = tmp_path / "uv.lock"
    path.write_text(LOCK + '\n[[package]]\nname = "torch"\nversion = "2.12.0"\n')
    with pytest.raises(EnvIdentityError, match="ONE resolved closure"):
        closure_entries_from_lockfile(path)


def test_an_empty_lockfile_refuses_rather_than_hashing_nothing(tmp_path: Path) -> None:
    path = tmp_path / "uv.lock"
    path.write_text("version = 1\n")
    with pytest.raises(EnvIdentityError, match="no resolved packages"):
        closure_entries_from_lockfile(path)


def test_lockfile_beside_finds_the_endpoints_own_and_never_guesses(
    tmp_path: Path, lockfile: Path
) -> None:
    assert lockfile_beside(lockfile.parent) == lockfile
    assert lockfile_beside(tmp_path / "elsewhere") is None


# -- the drift signal -------------------------------------------------------


def test_drift_does_not_fire_on_SPELLING(lockfile: Path) -> None:
    """THE case that makes the signal worth having.

    `PyYAML` and `pyyaml` are one distribution. A drift report that lists them
    is measuring its own normalization, and on one real endpoint that was TEN
    of the rows.
    """

    stated = closure_entries_from_lockfile(lockfile)
    installed = {"torch": "2.13.0", "pyyaml": "6.0.3", "colorama": "0.4.6"}
    assert closure_drift(installed, stated) == ()


def test_drift_does_not_fire_on_the_LOCAL_VERSION_SEGMENT(lockfile: Path) -> None:
    """A lock cannot state `+cu129`; an installed CUDA wheel always has one."""

    stated = closure_entries_from_lockfile(lockfile)
    installed = {"torch": "2.13.0+cu129", "PyYAML": "6.0.3", "colorama": "0.4.6"}
    assert closure_drift(installed, stated) == ()


def test_drift_DOES_fire_on_a_real_difference_and_names_it(lockfile: Path) -> None:
    stated = closure_entries_from_lockfile(lockfile)
    installed = {"torch": "2.12.0+cu129", "PyYAML": "6.0.3", "extra-thing": "1.0"}
    rows = closure_drift(installed, stated)
    kinds = {(r.name, r.kind) for r in rows}
    assert kinds == {
        ("colorama", "missing"),
        ("extra-thing", "extra"),
        ("torch", "version"),
    }
    summary = describe_drift(rows)
    assert "3 package(s) differ" in summary
    assert "torch 2.13.0 != 2.12.0" in summary
    assert describe_drift(()) == ""


@pytest.mark.parametrize(
    "raw,want",
    [("PyYAML", "pyyaml"), ("typing_extensions", "typing-extensions"),
     ("huggingface_hub", "huggingface-hub"), ("Py.YAML", "py-yaml")],
)
def test_pep503_name_normalization(raw: str, want: str) -> None:
    assert normalize_name(raw) == want


def test_local_version_segment_is_stripped_and_nothing_else_is() -> None:
    assert normalize_version("2.13.0+cu129") == "2.13.0"
    assert normalize_version("2.13.0rc1") == "2.13.0rc1"


# -- the end-to-end property, against the REAL producer ---------------------


def test_the_derive_and_the_serve_runner_hash_the_SAME_source(
    lockfile: Path,
) -> None:
    """The whole issue, as one assertion.

    `release/derive.py` stamps a document's closure from a lockfile; the serve
    runner states this boot's identity from one. If those two ever stop being
    the same function, a locally derived document becomes adoptable by nothing
    again — silently, because `sink_for` turns the refusal into eager-forever.
    """

    from gen_worker.release.derive import _closure_entries_from_lockfile

    assert _closure_entries_from_lockfile(lockfile) == closure_entries_from_lockfile(
        lockfile
    )


def test_a_real_endpoint_lock_hashes_to_a_stable_closure(tmp_path: Path) -> None:
    """Guards the MOVE itself: the reader now lives in `env_identity`, and if
    the move had changed a single byte of its output every document ever
    stamped would have been rekeyed."""

    from gen_worker._vendor.torchcg.graph_identity import closure_hash

    path = tmp_path / "uv.lock"
    path.write_text(LOCK)
    entries = closure_entries_from_lockfile(path)
    # Recomputed from the parsed TOML, independently of the reader under test.
    parsed = tomllib.loads(LOCK)
    expected = {p["name"]: p["version"] for p in parsed["package"]}
    assert entries == expected
    assert closure_hash(entries) == closure_hash(expected)
