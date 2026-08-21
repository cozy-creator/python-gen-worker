"""Where the box's machine-scoped stores live, and who is allowed to answer."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

import pytest

from gen_worker import config
from gen_worker.cli import daemon, up, workspace

SRC = Path(workspace.__file__).resolve().parents[1]

RETIRED = ".compiled-graphs"


@pytest.fixture()
def zero_config(monkeypatch):
    """No Settings installed — the state a bare `gen-worker` invocation is in."""
    monkeypatch.setattr(config, "current_or", lambda fallback: fallback)
    return None


def test_the_box_cache_is_the_default_and_workspace_states_it(zero_config):
    root = workspace.artifacts_root()
    assert root == workspace.DEFAULT_ARTIFACTS
    assert root.is_absolute(), root
    assert root == Path.home() / ".cache" / "cozy" / "compiled-graphs"


def test_the_address_does_not_depend_on_where_you_cd(zero_config, tmp_path,
                                                     monkeypatch):
    """The whole defect, stated as a test."""
    monkeypatch.chdir(tmp_path)
    from_a = workspace.artifacts_root()
    other = tmp_path / "some" / "endpoint"
    other.mkdir(parents=True)
    monkeypatch.chdir(other)
    from_b = workspace.artifacts_root()

    assert from_a == from_b
    assert tmp_path not in from_a.parents


def test_config_overrides_it_through_the_pipeline_not_an_env_read(monkeypatch):
    """`COZY_ARTIFACTS` is a Settings field, like the other two store roots."""
    from gen_worker.config import loader

    assert loader._ENV_TO_FIELD["COZY_ARTIFACTS"] == "artifacts_root"
    stated = Path("/srv/artifacts")
    monkeypatch.setattr(
        config, "current_or",
        lambda fallback: config.Settings(artifacts_root=str(stated)))
    assert workspace.artifacts_root() == stated


def test_a_bootspec_with_no_artifacts_dir_lands_in_the_box_cache(zero_config):
    assert daemon.BootSpec(endpoint_dir=Path("/tmp/ep")).artifacts_dir == \
        workspace.artifacts_root()


def test_the_bootspec_default_is_read_when_the_spec_is_built_not_at_import(
    monkeypatch,
):
    """A `default_factory`, not a module constant."""
    stated = Path("/srv/late-config")
    monkeypatch.setattr(
        config, "current_or",
        lambda fallback: config.Settings(artifacts_root=str(stated)))
    assert daemon.BootSpec(endpoint_dir=Path("/tmp/ep")).artifacts_dir == stated


def test_up_honors_an_explicit_artifacts_dir_and_resolves_it(tmp_path):
    """`--artifacts-dir` still wins."""
    args = argparse.Namespace(artifacts_dir=str(tmp_path / "mine"))
    assert up._artifacts_dir(args) == (tmp_path / "mine").resolve()


def test_up_falls_back_to_the_box_cache_when_unstated(zero_config):
    assert up._artifacts_dir(argparse.Namespace(artifacts_dir="")) == \
        workspace.artifacts_root()


def test_the_detached_child_is_handed_an_absolute_path(zero_config, tmp_path):
    """`up -d` re-execs with a different cwd."""
    args = argparse.Namespace(artifacts_dir="relative-dir")
    resolved = up._artifacts_dir(args)
    assert resolved.is_absolute(), resolved


def test_compile_and_the_serve_side_mint_resolve_the_same_address(zero_config):
    """`compile` WRITES and the background mint WRITES; they share a ledger."""
    from gen_worker.serving.self_mint import SelfMint

    assert SelfMint().artifacts_dir == workspace.artifacts_root()


def test_the_standalone_serve_entry_resolves_the_same_address(zero_config,
                                                              tmp_path):
    from gen_worker.serving import __main__ as serve_main

    assert serve_main._artifacts_dir(argparse.Namespace(artifacts_dir="")) == \
        workspace.artifacts_root()
    assert serve_main._artifacts_dir(
        argparse.Namespace(artifacts_dir=str(tmp_path))) == tmp_path.resolve()


def test_no_cwd_relative_compiled_graphs_default_survives_in_src():
    """Structural, because a reintroduced default is silent."""
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
    """The exclusion is a FLOOR and outlives the default it was written for."""
    from gen_worker.cli.publish import _EXCLUDE_DIRS

    assert RETIRED in _EXCLUDE_DIRS
