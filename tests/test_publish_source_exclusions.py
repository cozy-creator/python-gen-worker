"""What a source publish must never upload.

Subject-named, not incident-named (pgw#1362 / 4.34b); lineage in comments.
"""

from __future__ import annotations

from pathlib import Path

from gen_worker.cli.publish import source_files


def _tree(root: Path, files: dict[str, str]) -> None:
    for name, body in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")


def test_credential_directories_never_ship(tmp_path):
    """# pgw#1529: a secret-leak vector, not bloat.

    `publish` packages the WHOLE TREE when the endpoint is not a git work tree
    (no commit to take a manifest from). A private key sitting beside an
    endpoint would then go to the hub with it. cozy-local's archiver has
    excluded these since it was written and asserts it; this client did not,
    and both publish the same trees.
    """
    _tree(tmp_path, {
        "endpoint.toml": "main = 'x.main'",
        ".ssh/id_ed25519": "PRIVATE KEY",
        ".aws/credentials": "[default]",
        ".gnupg/secring.gpg": "secret",
        ".kube/config": "token: x",
        ".secrets/token": "s",
        ".azure/accessTokens.json": "{}",
    })
    shipped = {p.relative_to(tmp_path).as_posix() for p in source_files(tmp_path)}

    assert shipped == {"endpoint.toml"}, (
        f"a credential file was about to be uploaded: {sorted(shipped)}"
    )


def test_a_local_runs_output_never_ships(tmp_path):
    """# pgw#1529 / cl#88: cwd-relative defaults leave these in the tree.

    Asserted as DIRECTORIES: the artifact store holds `<graph>.so.unpacked/`
    trees whose bytes are mostly model.pt2 and metadata, so a `.so` suffix
    filter would ship the bulk and look fixed.
    """
    _tree(tmp_path, {
        "endpoint.toml": "main = 'x.main'",
        ".compiled-graphs/env/g.so.unpacked/model.pt2": "aoti",
        ".compiled-graphs/env/g.so.unpacked/metadata.json": "{}",
        "outputs/run-0/image.webp": "image",
    })
    shipped = {p.relative_to(tmp_path).as_posix() for p in source_files(tmp_path)}

    assert shipped == {"endpoint.toml"}, sorted(shipped)


def test_tool_caches_never_ship(tmp_path):
    _tree(tmp_path, {
        "endpoint.toml": "main = 'x.main'",
        ".tox/py312/x": "t",
        ".ruff_cache/c": "r",
        "node_modules/pkg/index.js": "js",
        ".mypy_cache/3.12/x.json": "{}",
    })
    shipped = {p.relative_to(tmp_path).as_posix() for p in source_files(tmp_path)}

    assert shipped == {"endpoint.toml"}, sorted(shipped)


def test_the_endpoints_own_source_still_ships(tmp_path):
    """The exclusions must not be so broad they eat the endpoint.

    A guard that ships nothing is as broken as one that ships everything, and
    the failure looks like success right up to the hub's build.
    """
    _tree(tmp_path, {
        "endpoint.toml": "main = 'x.main'",
        "pyproject.toml": "[project]",
        "src/x/main.py": "print('ok')",
        "README.md": "# x",
    })
    shipped = {p.relative_to(tmp_path).as_posix() for p in source_files(tmp_path)}

    assert shipped == {
        "endpoint.toml", "pyproject.toml", "src/x/main.py", "README.md",
    }, sorted(shipped)
