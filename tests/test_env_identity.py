"""pgw#1489: the env half of an artifact key is the COMPILE STACK, and nothing else.

The defect this file fences is not "a hash was wrong". It is that the key was
a hash of the ENTIRE resolved package set — a second representation of what
the endpoint's `uv.lock` already pins, structurally able to disagree with
itself (pgw#1472's measurement: 43-package diffs between envs that serve
identically), and able to split the artifact pool on a docs extra.

⚠️ **Why no earlier test could fail on it.** Every adopt test built its
document from the same mapping it then audited against, so the two sources
were ONE dict by construction and irrelevant drift was unrepresentable. The
cases here therefore work from REAL lockfile bytes with a real irrelevant
diff in them, because that difference is the whole subject.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gen_worker._vendor.torchcg.graph_identity import EnvIdentity
from gen_worker.env_identity import (
    EnvIdentityError,
    compile_stack_from_lockfile,
    cuda_bucket,
    cuda_buckets,
    installed_stack_drift,
    lock_entries,
    lockfile_beside,
)


def _lock(**versions: str) -> str:
    rows = {
        "torch": "2.13.0",
        "triton": "3.6.0",
        "nvidia-cublas-cu12": "12.8.4.1",
        "nvidia-cuda-runtime-cu12": "12.8.90",
        "diffusers": "0.36.0",
        "pillow": "11.1.0",
        "colorama": "0.4.6",
        "rich": "13.9.4",
    }
    rows.update(versions)
    return "version = 1\n" + "".join(
        f'\n[[package]]\nname = "{name}"\nversion = "{version}"\n'
        for name, version in rows.items()
    )


@pytest.fixture()
def lockfile(tmp_path: Path) -> Path:
    path = tmp_path / "uv.lock"
    path.write_text(_lock())
    return path


def test_the_stack_is_the_compiler_and_nothing_else(lockfile: Path) -> None:
    assert compile_stack_from_lockfile(lockfile) == (
        ("nvidia-cublas-cu12", "12.8.4.1"),
        ("nvidia-cuda-runtime-cu12", "12.8.90"),
        ("torch", "2.13.0"),
        ("triton", "3.6.0"),
    )
    # And the reader still sees the whole lock — the selection is the key
    # input, not the reading.
    assert set(lock_entries(lockfile)) > set(dict(compile_stack_from_lockfile(lockfile)))


def test_IRRELEVANT_DRIFT_DOES_NOT_MOVE_THE_KEY(tmp_path: Path) -> None:
    """THE pgw#1489 green arm, at the key level.

    Two lockfiles that differ in every package that cannot reach the compiler
    — a pillow bump, a dropped extra, a new dev tool — are ONE artifact env.
    Under the closure key these were two pools and every artifact in the
    second one had to be re-minted.
    """

    mine = tmp_path / "mine.lock"
    mine.write_text(_lock())
    theirs = tmp_path / "theirs.lock"
    theirs.write_text(
        _lock(pillow="12.0.0", colorama="0.4.7", rich="14.1.0", diffusers="0.37.0")
        + '\n[[package]]\nname = "pytest"\nversion = "9.0.0"\n'
    )

    assert lock_entries(mine) != lock_entries(theirs)
    assert compile_stack_from_lockfile(mine) == compile_stack_from_lockfile(theirs)
    assert (
        EnvIdentity(stack=compile_stack_from_lockfile(mine), sm="sm_89").value
        == EnvIdentity(stack=compile_stack_from_lockfile(theirs), sm="sm_89").value
    )


def test_a_compiler_bump_DOES_move_the_key(tmp_path: Path) -> None:
    base = tmp_path / "a.lock"
    base.write_text(_lock())
    for bumped in ({"torch": "2.14.0"}, {"triton": "3.7.0"},
                   {"nvidia-cublas-cu12": "12.9.0.1"}):
        other = tmp_path / "b.lock"
        other.write_text(_lock(**bumped))
        assert (
            EnvIdentity(stack=compile_stack_from_lockfile(base), sm="sm_89").value
            != EnvIdentity(stack=compile_stack_from_lockfile(other), sm="sm_89").value
        ), bumped


def test_sm_is_the_other_free_variable(lockfile: Path) -> None:
    stack = compile_stack_from_lockfile(lockfile)
    assert (
        EnvIdentity(stack=stack, sm="sm_89").value
        != EnvIdentity(stack=stack, sm="sm_120").value
    )


def test_a_lock_with_no_torch_refuses_by_name(tmp_path: Path) -> None:
    path = tmp_path / "uv.lock"
    path.write_text('version = 1\n\n[[package]]\nname = "rich"\nversion = "13.9.4"\n')
    with pytest.raises(EnvIdentityError, match="states torch"):
        compile_stack_from_lockfile(path)


def test_the_reader_refuses_an_unreadable_or_empty_lock(tmp_path: Path) -> None:
    with pytest.raises(EnvIdentityError, match="cannot read lockfile"):
        lock_entries(tmp_path / "absent.lock")
    empty = tmp_path / "uv.lock"
    empty.write_text("version = 1\n")
    with pytest.raises(EnvIdentityError, match="no resolved packages"):
        lock_entries(empty)


def test_lockfile_beside_is_the_endpoint_s_own(tmp_path: Path, lockfile: Path) -> None:
    assert lockfile_beside(lockfile.parent) == lockfile
    assert lockfile_beside(tmp_path / "elsewhere") is None


def test_the_derive_and_the_serve_runner_READ_THE_SAME_SOURCE(lockfile: Path) -> None:
    """The whole issue, as one assertion.

    `release/derive.py` stamps a document's stack from a lockfile; the serve
    runner states this boot's stack from one. If those two ever stop being the
    same function, a locally derived document becomes adoptable by nothing
    again — silently, because the refusal turns into eager-forever.
    """

    from gen_worker.release.derive import _compile_stack_from_lockfile

    assert _compile_stack_from_lockfile(lockfile) == compile_stack_from_lockfile(
        lockfile
    )


def test_a_derive_without_a_lockfile_refuses_instead_of_restating_the_env() -> None:
    """No installed-set fallback survives: it was the second representation."""

    import inspect

    from gen_worker.release import derive

    source = inspect.getsource(derive.derive_release)
    assert "installed_closure" not in source
    assert "pass `lockfile=`" in source


def test_the_drift_report_is_DIAGNOSTIC_and_reads_the_stack_only(
    lockfile: Path,
) -> None:
    """It may exist; it may not gate. Rows are strings for a log line."""

    rows = installed_stack_drift(dict(compile_stack_from_lockfile(lockfile)))
    assert all(isinstance(row, str) for row in rows)
    # The fixture's nvidia pin is not installed here, so the diagnostic names
    # it by package — and nothing anywhere acts on that.
    assert any(row.startswith("nvidia-cublas-cu12 ") for row in rows)
    # The `+cu130` local segment a lockfile cannot express is NOT drift.
    assert not any(row.startswith("torch ") for row in
                   installed_stack_drift({"torch": "2.13.0"}))
    # A package that cannot reach the compiler cannot appear in it at all.
    assert not any("pillow" in row or "colorama" in row for row in rows)


#: A real `uv lock` output shape, minimized: ONE lock, two CUDA buckets as
#: conflicting extras (the mechanism Paul verified 2026-08-19). `cudnn` is
#: locked ONCE and shared, and its own edge names the OTHER bucket's cublas —
#: which is exactly the trap a depth-first read falls into.
FLAVOR_LOCK = """\
version = 1

[[package]]
name = "endpoint"
version = "0.1.0"
source = { virtual = "." }

[package.optional-dependencies]
cu126 = [
    { name = "torch", version = "2.8.0+cu126", source = { registry = "https://download.pytorch.org/whl/cu126" } },
]
cu128 = [
    { name = "torch", version = "2.8.0+cu128", source = { registry = "https://download.pytorch.org/whl/cu128" } },
]

[[package]]
name = "torch"
version = "2.8.0+cu126"
source = { registry = "https://download.pytorch.org/whl/cu126" }
dependencies = [
    { name = "nvidia-cublas-cu12", version = "12.6.4.1" },
    { name = "nvidia-cudnn-cu12" },
    { name = "pillow", version = "11.1.0" },
    { name = "triton", version = "3.4.0" },
]

[[package]]
name = "torch"
version = "2.8.0+cu128"
source = { registry = "https://download.pytorch.org/whl/cu128" }
dependencies = [
    { name = "nvidia-cublas-cu12", version = "12.8.4.1" },
    { name = "nvidia-cudnn-cu12" },
    { name = "pillow", version = "11.1.0" },
    { name = "triton", version = "3.4.0" },
]

[[package]]
name = "nvidia-cublas-cu12"
version = "12.6.4.1"

[[package]]
name = "nvidia-cublas-cu12"
version = "12.8.4.1"

[[package]]
name = "nvidia-cudnn-cu12"
version = "9.10.2.21"
dependencies = [
    { name = "nvidia-cublas-cu12", version = "12.8.4.1" },
]

[[package]]
name = "triton"
version = "3.4.0"

[[package]]
name = "pillow"
version = "11.1.0"
"""


@pytest.fixture()
def flavor_lock(tmp_path: Path) -> Path:
    path = tmp_path / "flavor" / "uv.lock"
    path.parent.mkdir()
    path.write_text(FLAVOR_LOCK)
    return path


def test_one_lock_states_every_bucket_and_each_keys_apart(flavor_lock: Path) -> None:
    assert cuda_buckets(flavor_lock) == ("cu126", "cu128")
    early = dict(compile_stack_from_lockfile(flavor_lock, bucket="cu126"))
    late = dict(compile_stack_from_lockfile(flavor_lock, bucket="cu128"))
    assert early["torch"] == "2.8.0+cu126" and late["torch"] == "2.8.0+cu128"
    # THE TRAP: cudnn is shared and its own edge names cu128's cublas. The
    # bucket's answer is the one ITS torch states, not the deepest edge found.
    assert early["nvidia-cublas-cu12"] == "12.6.4.1"
    assert late["nvidia-cublas-cu12"] == "12.8.4.1"
    # Shared, and therefore in both.
    assert early["nvidia-cudnn-cu12"] == late["nvidia-cudnn-cu12"] == "9.10.2.21"
    # A package that cannot reach the compiler is in neither.
    assert "pillow" not in early and "pillow" not in late
    assert (
        EnvIdentity(stack=tuple(sorted(early.items())), sm="sm_89").value
        != EnvIdentity(stack=tuple(sorted(late.items())), sm="sm_89").value
    )


def test_a_multi_bucket_lock_refuses_to_guess(flavor_lock: Path) -> None:
    with pytest.raises(EnvIdentityError, match="locks 2 CUDA buckets"):
        compile_stack_from_lockfile(flavor_lock)
    with pytest.raises(EnvIdentityError, match="which its author never locked"):
        compile_stack_from_lockfile(flavor_lock, bucket="cu999")
    # The raw reader resolves a fork by bucket too, and refuses to guess when
    # it cannot attribute one — the same rule, one level down.
    with pytest.raises(EnvIdentityError, match=r"resolves 'torch' 2 ways"):
        lock_entries(flavor_lock)


def test_a_single_resolution_lock_ignores_the_host_bucket(lockfile: Path) -> None:
    """Every endpoint today. The bucket is a question only a flavored lock asks."""

    assert cuda_buckets(lockfile) == ()
    assert compile_stack_from_lockfile(
        lockfile, bucket="cu130"
    ) == compile_stack_from_lockfile(lockfile)


def test_the_host_reports_its_bucket_and_resolves_nothing() -> None:
    bucket = cuda_bucket()
    assert bucket == "" or (bucket.startswith("cu") and bucket[2:].isdigit())


def test_a_FORKED_lock_is_read_by_the_host_s_bucket(tmp_path: Path) -> None:
    """pgw#1472 measured this and concluded a lock cannot be an identity.

    uv forks a resolution per index marker, so a lock legitimately states
    `torch` at both `2.13.0` and `2.13.0+cu130` — pgw's own does. The fork is a
    CUDA fork: its branches differ by the PEP 440 local segment, which IS the
    bucket, so the host's bucket picks its branch. A reader that raised here
    failed closed on the repo that most needs it.
    """

    path = tmp_path / "uv.lock"
    path.write_text(
        _lock()
        + '\n[[package]]\nname = "torch"\nversion = "2.13.0+cu130"\n'
    )
    assert dict(compile_stack_from_lockfile(path, bucket="cu130"))["torch"] == (
        "2.13.0+cu130"
    )
    with pytest.raises(EnvIdentityError, match="resolves 'torch' 2 ways"):
        compile_stack_from_lockfile(path)
    with pytest.raises(EnvIdentityError, match="matches none of them"):
        compile_stack_from_lockfile(path, bucket="cu126")


def test_pgws_OWN_lockfile_reads_cleanly() -> None:
    """The measured case, on the real file: 17 rows and a flavored torch."""

    repo = Path(__file__).resolve().parents[1] / "uv.lock"
    stack = dict(compile_stack_from_lockfile(repo, bucket=cuda_bucket() or "cu130"))
    assert stack["torch"].startswith("2.") and "+cu" in stack["torch"]
    assert len(stack) > 5 and "pytest" not in stack


def test_an_extra_is_a_BUCKET_only_when_it_is_named_for_one() -> None:
    """pgw's own pyproject has an extra called `torch`. It is not a CUDA line."""

    repo = Path(__file__).resolve().parents[1] / "uv.lock"
    assert cuda_buckets(repo) == ()
