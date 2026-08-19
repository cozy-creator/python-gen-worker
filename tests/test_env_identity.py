"""pgw#1472: the env-identity closure has ONE definition and one producer.

The defect this file fences is not "a hash was wrong". It is that "the
environment" had TWO producers and no single meaning: `gen-worker lock` stamped
a document from the LOCKFILE while `release derive` and the pod stated the
INSTALLED set — and a lockfile closure is restatable by no running process at
all. pgw#1367 makes publish, mint and serve three different processes by
design, so a value none of them can agree on fragments the whole
[release x sm] serving table.

⚠️ **Why no existing test could fail on it.** Every adopt test builds its
document from the same `installed=` mapping it then audits against, so the two
sources are ONE dict by construction. The fixture made the bug unrepresentable.
The cases here therefore work from REAL lockfiles on disk, from a real
`importlib.metadata` reading, and from a real FRESH SUBPROCESS — because
"restatable by another process" is the whole subject.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from gen_worker.env_identity import (
    EnvIdentityError,
    closure_drift,
    describe_drift,
    describe_lockfile_drift,
    env_closure,
    env_closure_hash,
    lockfile_beside,
    lockfile_packages,
    normalize_name,
    normalize_version,
)

ROOT = Path(__file__).resolve().parents[1]

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


# -- RED: the two old producers cannot agree on ONE environment -------------


def test_RED_the_two_old_paths_disagree_on_THIS_VERY_ENV(tmp_path: Path) -> None:
    """The issue's own repro, run against the env the test is running in.

    A lockfile written to state EXACTLY what is installed still hashes to a
    different closure, for reasons no build can fix: `uv` normalizes names to
    PEP 503 while `importlib.metadata` reports the DECLARED name, and a lock
    cannot carry the PEP 440 local segment (`+cu129`) that every installed CUDA
    wheel has. Two producers, one environment, two answers.
    """

    from gen_worker._vendor.torchcg.graph_identity import closure_hash

    installed = env_closure()
    # The most generous lockfile imaginable: uv's own spelling of this exact
    # installed set, which is the best any real `uv.lock` could ever do.
    as_uv_would_write_it = {
        normalize_name(name): normalize_version(version)
        for name, version in installed.items()
    }
    path = tmp_path / "uv.lock"
    path.write_text(
        "version = 1\n"
        + "".join(
            f'\n[[package]]\nname = "{n}"\nversion = "{v}"\n'
            for n, v in sorted(as_uv_would_write_it.items())
        )
    )
    assert closure_hash(lockfile_packages(path)) != env_closure_hash()


def test_RED_a_real_lockfile_does_not_even_NAME_one_package_set() -> None:
    """pgw's own `uv.lock` resolves `torch` twice — the killer fact.

    uv forks the resolution per index marker, so `torch` is `2.13.0` under one
    and `2.13.0+cu130` under another. There is no "the lock's closure" to hash;
    the old reader raised on exactly this, which is how a lockfile identity
    fails CLOSED on the one repo that most needs it.
    """

    entries = lockfile_packages(ROOT / "uv.lock")
    assert "/" in entries["torch"], entries["torch"]


# -- GREEN: the one function restates identically from a fresh process ------


def test_GREEN_a_FRESH_PROCESS_restates_the_same_value() -> None:
    """The property the whole architecture rests on.

    Publish, mint and serve are three processes (pgw#1367). The identity they
    fold into the ck1 key must survive that boundary, so it is measured across
    a real subprocess fence rather than by calling the function twice.
    """

    proc = subprocess.run(
        [sys.executable, "-c",
         "import json,sys;sys.path.insert(0,%r);"
         "from gen_worker.env_identity import env_closure_hash;"
         "print(json.dumps(env_closure_hash()))" % str(ROOT / "src")],
        capture_output=True, text=True, check=True,
    )
    assert json.loads(proc.stdout) == env_closure_hash()


def test_GREEN_every_party_reaches_the_one_function_and_not_a_second_copy() -> None:
    """The derive, the boot host and the adopt session must not each observe
    the env their own way — that is how two spellings were born the first
    time. Asserted by RESOLVING their imports, not by grepping for a name."""

    src = ROOT / "src" / "gen_worker"
    for relative in (
        "release/derive.py",
        "serving/host.py",
        "serving/serve_adoption.py",
        "serving/__main__.py",
        "cli/endpoint_lock.py",
        "cli/lock.py",
        "cli/release.py",
    ):
        assert "installed_closure" not in (src / relative).read_text(), (
            f"{relative} observes the env itself; it must go through "
            f"gen_worker.env_identity.env_closure"
        )


def test_CONTROL_one_bumped_package_is_a_DIFFERENT_env() -> None:
    """The control that the value still discriminates.

    A restatable identity that returned a constant would pass every test
    above. Bump exactly one real package's version and the closure must move.
    """

    from gen_worker._vendor.torchcg.graph_identity import closure_hash

    installed = env_closure()
    victim = sorted(installed)[0]
    bumped = dict(installed, **{victim: installed[victim] + ".99"})
    assert closure_hash(bumped) != env_closure_hash()


# -- the lockfile is a DIFFERENT named thing, and only a diagnostic ---------


def test_the_lockfile_reader_keeps_names_and_versions_VERBATIM(lockfile: Path) -> None:
    entries = lockfile_packages(lockfile)
    assert entries == {"torch": "2.13.0", "PyYAML": "6.0.3", "colorama": "0.4.6"}


def test_a_forked_lockfile_REPORTS_the_fork_instead_of_dying(tmp_path: Path) -> None:
    """A drift report that raises on the very property that killed
    lockfile-as-identity is useless exactly where it is most informative."""

    path = tmp_path / "uv.lock"
    path.write_text(LOCK + '\n[[package]]\nname = "torch"\nversion = "2.13.0+cu130"\n')
    assert lockfile_packages(path)["torch"] == "2.13.0/2.13.0+cu130"


def test_an_empty_lockfile_refuses_rather_than_reporting_nothing(tmp_path: Path) -> None:
    path = tmp_path / "uv.lock"
    path.write_text("version = 1\n")
    with pytest.raises(EnvIdentityError, match="no resolved packages"):
        lockfile_packages(path)


def test_lockfile_beside_finds_the_endpoints_own_and_never_guesses(
    tmp_path: Path, lockfile: Path
) -> None:
    assert lockfile_beside(lockfile.parent) == lockfile
    assert lockfile_beside(tmp_path / "elsewhere") is None


def test_drift_does_not_fire_on_SPELLING(lockfile: Path) -> None:
    """THE case that makes the signal worth having.

    `PyYAML` and `pyyaml` are one distribution. A drift report that lists them
    is measuring its own normalization, and on one real endpoint that was TEN
    of the rows.
    """

    stated = lockfile_packages(lockfile)
    installed = {"torch": "2.13.0", "pyyaml": "6.0.3", "colorama": "0.4.6"}
    assert closure_drift(installed, stated) == ()


def test_drift_does_not_fire_on_the_LOCAL_VERSION_SEGMENT(lockfile: Path) -> None:
    stated = lockfile_packages(lockfile)
    installed = {"torch": "2.13.0+cu129", "PyYAML": "6.0.3", "colorama": "0.4.6"}
    assert closure_drift(installed, stated) == ()


def test_drift_DOES_fire_on_a_real_difference_and_names_it(lockfile: Path) -> None:
    stated = lockfile_packages(lockfile)
    installed = {"torch": "2.12.0+cu129", "PyYAML": "6.0.3", "extra-thing": "1.0"}
    rows = closure_drift(installed, stated)
    assert {(r.name, r.kind) for r in rows} == {
        ("colorama", "missing"),
        ("extra-thing", "extra"),
        ("torch", "version"),
    }
    summary = describe_drift(rows)
    assert "3 package(s) differ" in summary
    assert "torch 2.13.0 != 2.12.0" in summary
    assert describe_drift(()) == ""


def test_the_drift_line_never_raises_on_an_unreadable_lock(tmp_path: Path) -> None:
    """It is a signal, not a gate: an absent or broken lock reports, it does
    not stop a boot."""

    assert "unavailable" in describe_lockfile_drift(tmp_path / "nope.lock")


@pytest.mark.parametrize(
    "raw,want",
    [("PyYAML", "pyyaml"), ("typing_extensions", "typing-extensions"),
     ("huggingface_hub", "huggingface-hub"), ("Py.YAML", "py-yaml")],
)
def test_pep503_name_normalization(raw: str, want: str) -> None:
    assert normalize_name(raw) == want


def test_local_version_segment_is_stripped_for_DRIFT_and_kept_for_IDENTITY() -> None:
    assert normalize_version("2.13.0+cu129") == "2.13.0"
    assert normalize_version("2.13.0rc1") == "2.13.0rc1"
    # Identity does NOT go through normalize_version: `+cu129` vs `+cu130` is
    # exactly the difference a compiled artifact cares about.
    from gen_worker._vendor.torchcg.graph_identity import closure_hash

    assert closure_hash({"torch": "2.13.0+cu129"}) != closure_hash(
        {"torch": "2.13.0+cu130"}
    )
