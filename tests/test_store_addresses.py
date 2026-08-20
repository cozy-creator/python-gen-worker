"""Where the box's machine-scoped stores live, and who is allowed to answer.

# pgw#1526: `artifacts_dir` defaulted to `Path(".compiled-graphs")` — relative
# to whatever cwd the daemon started in — so running `up` or `compile` inside an
# endpoint deposited a machine-scoped AOTInductor store into a source tree.
# pgw#1513 in a different path: one store whose address was decided by an
# accident of cwd instead of stated once by the layer that owns it.

ASSERTED AT THE STORE/PATH LEVEL, DELIBERATELY, and the reason is recorded
because it changes what a green run here proves. The peer #1532 lane reports
adoption currently enumerating ZERO records over a full store
(`adopted=0, holes=0`) for a cause one layer down that is theirs and unfixed. An
adoption-based acceptance of THIS move would therefore read empty for their
reason rather than mine — a pass and a failure that look identical. So these
assert what this diff actually changed: the location is answered by
`cli/workspace.py`, `compile` builds there, the serve side adopts from the same
address, and the cwd-relative default is gone from the tree.
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

import pytest

from gen_worker import config
from gen_worker.cli import daemon, up, workspace

SRC = Path(workspace.__file__).resolve().parents[1]

#: The retired spelling. A default anywhere in `src/` is the defect; the
#: `publish` exclusion set names it on purpose (a floor, not a default).
RETIRED = ".compiled-graphs"


@pytest.fixture()
def zero_config(monkeypatch):
    """No Settings installed — the state a bare `gen-worker` invocation is in."""
    monkeypatch.setattr(config, "current_or", lambda fallback: fallback)
    return None


# ---------------------------------------------------------------------------
# one home answers it
# ---------------------------------------------------------------------------

def test_the_box_cache_is_the_default_and_workspace_states_it(zero_config):
    root = workspace.artifacts_root()
    assert root == workspace.DEFAULT_ARTIFACTS
    assert root.is_absolute(), root
    assert root == Path.home() / ".cache" / "cozy" / "compiled-graphs"


def test_the_address_does_not_depend_on_where_you_cd(zero_config, tmp_path,
                                                     monkeypatch):
    """The whole defect, stated as a test.

    Two cwds, one answer. Under the old default these differed, and the
    difference was invisible: each run wrote a real store, just not the same
    one — and one of them was inside somebody's source tree.
    """
    monkeypatch.chdir(tmp_path)
    from_a = workspace.artifacts_root()
    other = tmp_path / "some" / "endpoint"
    other.mkdir(parents=True)
    monkeypatch.chdir(other)
    from_b = workspace.artifacts_root()

    assert from_a == from_b
    assert tmp_path not in from_a.parents


def test_config_overrides_it_through_the_pipeline_not_an_env_read(monkeypatch):
    """`COZY_ARTIFACTS` is a Settings field, like the other two store roots.

    Through the real loader map, not by patching the function: a module reading
    `os.environ` itself would be a second loader that can disagree with the
    first (§1.18), and that is the failure this field exists to avoid.
    """
    from gen_worker.config import loader

    assert loader._ENV_TO_FIELD["COZY_ARTIFACTS"] == "artifacts_root"
    stated = Path("/srv/artifacts")
    monkeypatch.setattr(
        config, "current_or",
        lambda fallback: config.Settings(artifacts_root=str(stated)))
    assert workspace.artifacts_root() == stated


# ---------------------------------------------------------------------------
# every consumer resolves to that one answer
# ---------------------------------------------------------------------------

def test_a_bootspec_with_no_artifacts_dir_lands_in_the_box_cache(zero_config):
    assert daemon.BootSpec(endpoint_dir=Path("/tmp/ep")).artifacts_dir == \
        workspace.artifacts_root()


def test_the_bootspec_default_is_read_when_the_spec_is_built_not_at_import(
    monkeypatch,
):
    """A `default_factory`, not a module constant.

    The CLI installs Settings at process entry, which happens AFTER this module
    is imported. A default frozen at import would keep the pre-config answer
    forever and no test that never installs Settings could see it.
    """
    stated = Path("/srv/late-config")
    monkeypatch.setattr(
        config, "current_or",
        lambda fallback: config.Settings(artifacts_root=str(stated)))
    assert daemon.BootSpec(endpoint_dir=Path("/tmp/ep")).artifacts_dir == stated


def test_up_honors_an_explicit_artifacts_dir_and_resolves_it(tmp_path):
    """`--artifacts-dir` still wins. The default died; the flag did not."""
    args = argparse.Namespace(artifacts_dir=str(tmp_path / "mine"))
    assert up._artifacts_dir(args) == (tmp_path / "mine").resolve()


def test_up_falls_back_to_the_box_cache_when_unstated(zero_config):
    assert up._artifacts_dir(argparse.Namespace(artifacts_dir="")) == \
        workspace.artifacts_root()


def test_the_detached_child_is_handed_an_absolute_path(zero_config, tmp_path):
    """`up -d` re-execs with a different cwd.

    A relative value — or one the child re-defaults for itself — would put the
    child's artifacts somewhere the parent never looks. That is a build under
    one address and a lookup under another, which reads as an endless cache
    miss rather than as an error, so it is asserted rather than assumed.
    """
    args = argparse.Namespace(artifacts_dir="relative-dir")
    resolved = up._artifacts_dir(args)
    assert resolved.is_absolute(), resolved


def test_compile_and_the_serve_side_mint_resolve_the_same_address(zero_config):
    """`compile` WRITES and the background mint WRITES; they share a ledger.

    Two spellings would make each skip work the other did not do.
    """
    from gen_worker.serving.self_mint import SelfMint

    assert SelfMint().artifacts_dir == workspace.artifacts_root()


def test_the_standalone_serve_entry_resolves_the_same_address(zero_config,
                                                              tmp_path):
    from gen_worker.serving import __main__ as serve_main

    assert serve_main._artifacts_dir(argparse.Namespace(artifacts_dir="")) == \
        workspace.artifacts_root()
    assert serve_main._artifacts_dir(
        argparse.Namespace(artifacts_dir=str(tmp_path))) == tmp_path.resolve()


# ---------------------------------------------------------------------------
# the cwd-relative default is DEAD — no fallback, no alias
# ---------------------------------------------------------------------------

def test_no_cwd_relative_compiled_graphs_default_survives_in_src():
    """Structural, because a reintroduced default is silent.

    Nothing goes red when a store writes to the wrong directory: it writes a
    real store there. Only a scan notices, which is why this is a scan and not
    a promise.
    """
    offenders: list[str] = []
    for path in sorted(SRC.rglob("*.py")):
        if "_vendor" in path.parts or "pb" in path.parts:
            continue
        rel = path.relative_to(SRC.parent)
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            # A bare string CONSTANT is fine (`publish`'s exclusion set names
            # the directory to refuse it). What must not come back is the
            # spelling used as a PATH: `Path(".compiled-graphs")`.
            if not (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "Path"):
                continue
            for arg in node.args:
                if isinstance(arg, ast.Constant) and arg.value == RETIRED:
                    offenders.append(f"{rel}:{node.lineno}")
    assert not offenders, (
        "pgw#1526: a cwd-relative `.compiled-graphs` path is back at "
        f"{offenders}. The address is answered ONCE by "
        "`cli/workspace.artifacts_root()`; a default that depends on where you "
        "happened to `cd` is not a default."
    )


def test_publish_still_refuses_the_directory_as_a_floor():
    """The exclusion is a FLOOR and outlives the default it was written for.

    `--artifacts-dir .compiled-graphs` is still legal, so a tree can still
    contain one — and it still must never be uploaded.
    """
    from gen_worker.cli.publish import _EXCLUDE_DIRS

    assert RETIRED in _EXCLUDE_DIRS
