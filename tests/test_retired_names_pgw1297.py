"""pgw#1297: the rename is complete, and it moved no key.

Three properties, and the first is the one with money attached.

`boot_key._tcg_version()` is a component of `closure_digest`, the compile
memo's key. It used to read the compiled-graph library's distribution version
and now reads `vendored_rev(<package>)` — so the VENDORED.toml table key is an
ARGUMENT to a key derivation, and pgw#1297 renamed that table key. If the
derivation had folded the NAME rather than the REV, every cached compiled graph
on the fleet would miss once and re-mint: exactly the silent-money failure
pgw#1299's fence exists for. It folds the rev, and this pins that.
"""

from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
VENDORED = REPO / "src" / "gen_worker" / "_vendor" / "VENDORED.toml"

#: The value `_tcg_version()` returned BEFORE the table key was renamed, read
#: off `origin/master` at 12316afd. Hard-coded on purpose: reading it from
#: VENDORED.toml would make this test agree with any rename of the thing it
#: exists to hold still.
TCG_REV_BEFORE_THE_RENAME = "ad5f4cb9f89bbe91d2a30ca218f70d5326630368"


def test_the_vendored_table_rename_did_not_move_the_compile_memo_key() -> None:
    from gen_worker.boot_key import _tcg_version

    assert _tcg_version() == TCG_REV_BEFORE_THE_RENAME, (
        "the TCG component of `closure_digest` moved. The pgw#1297 rename was "
        "supposed to change a lookup key and nothing else — a moved component "
        "invalidates every boot memo on the fleet and re-mints every family."
    )


def test_the_vendored_packages_are_spelled_the_way_upstream_spells_them() -> None:
    """Upstream is `tensorfs` and `torchcg`; so is `_vendor/`.

    Not cosmetic: pgw#1295 deletes `_vendor/` and installs the real
    distributions, and matching names make that a change of import PREFIX only.
    """
    packages = tomllib.loads(VENDORED.read_text())["packages"]
    assert set(packages) == {"tensorfs", "torchcg"}
    for name in packages:
        assert (VENDORED.parent / name / "__init__.py").is_file()
    assert not (VENDORED.parent / "torch_compiled_graphs").exists()  # retired-name: the old directory must be GONE, so this names it

    import gen_worker._vendor.tensorfs  # noqa: F401
    import gen_worker._vendor.torchcg  # noqa: F401

    with pytest.raises(ModuleNotFoundError):
        __import__("gen_worker._vendor.torch_compiled_graphs")  # retired-name: the old import path must be dead, so this names it


#: Each case is one spelling the fence must catch. They have to be written out
#: literally — a red proof that spells its subject indirectly proves nothing.
REINTRODUCED = [
    ("from gen_worker._vendor.torch_compiled_graphs import X\n", "module"),  # retired-name: red proof
    ('DEPS = ["hashrepo>=0.3"]\n', "requirement"),  # retired-name: red proof
    ('"""See :mod:`torch-compiled-graphs` for policy."""\n', "prose"),  # retired-name: red proof
    ('"""The worker\'s one HashRepo store."""\n', "prose, wrong case"),  # retired-name: red proof
]


@pytest.mark.parametrize("body, spelling", REINTRODUCED)
def test_the_fence_catches_a_reintroduced_old_name(
    tmp_path: Path, body: str, spelling: str
) -> None:
    """The red proof, by adding one — a sweep that only ever prints clean
    guards nothing. The wrong-case arm is the one every previous attempt at
    this rename could not see, because each swept with a case-sensitive grep.
    """
    (tmp_path / "reintroduced.py").write_text(body)
    run = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "lint_retired_package_names.py"),
         str(tmp_path)],
        capture_output=True, text=True,
    )
    assert run.returncode == 1, f"{spelling}: the fence stayed green\n{run.stdout}"
    assert "is retired" in run.stderr


def test_the_fence_is_clean_on_this_tree() -> None:
    run = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "lint_retired_package_names.py")],
        capture_output=True, text=True,
    )
    assert run.returncode == 0, run.stderr
