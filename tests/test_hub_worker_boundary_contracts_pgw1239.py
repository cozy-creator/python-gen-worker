"""Bind the public Hub-worker boundary corpus to python-gen-worker sources."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest


_ROOT = Path(__file__).parents[1]
_DEFAULT_CORPUS = Path(__file__).parent / "testdata" / "hub_worker_boundary_contracts.json"
_DEFAULT_DIGEST = Path(__file__).parent / "testdata" / "HUB_WORKER_BOUNDARY_CONTRACTS_DIGEST"
_CORPUS = Path(os.environ.get("HUB_WORKER_BOUNDARY_CONTRACT_FILE", _DEFAULT_CORPUS))

_SOURCE_PATHS = {
    "loader": _ROOT / "src" / "gen_worker" / "config" / "loader.py",
    "settings": _ROOT / "src" / "gen_worker" / "config" / "settings.py",
    "c2pa": _ROOT / "src" / "gen_worker" / "content_credentials.py",
    "topology": _ROOT / "src" / "gen_worker" / "topology.py",
    "procsplit": _ROOT / "src" / "gen_worker" / "procsplit" / "__init__.py",
    "discovery": _ROOT / "src" / "gen_worker" / "discovery" / "discover.py",
    # pgw#1331: `runtime_key()` moved to `compile_facts`, the READ half of the
    # compile cache — `compile_cache` re-exports it. The raw launch-value read
    # is checked where it lives, not where it is re-exported from.
    "compile_facts": _ROOT / "src" / "gen_worker" / "compile_facts.py",
    "model_store": _ROOT / "src" / "gen_worker" / "models" / "store.py",
    "provision": _ROOT / "src" / "gen_worker" / "models" / "provision.py",
}
_SOURCE_ENV = {name: f"HUB_WORKER_BOUNDARY_{name.upper()}_SOURCE" for name in _SOURCE_PATHS}


def _document() -> dict[str, Any]:
    document = json.loads(_CORPUS.read_text(encoding="utf-8"))
    assert document["schema"] == "hub-worker-boundary-contracts-v1"
    return document


def _sources() -> dict[str, str]:
    return {
        name: Path(os.environ.get(_SOURCE_ENV[name], path)).read_text(encoding="utf-8")
        for name, path in _SOURCE_PATHS.items()
    }


def _assignment(tree: ast.Module, name: str) -> ast.expr:
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            return node.value
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == name
            and node.value is not None
        ):
            return node.value
    raise AssertionError(f"source assignment {name} is missing")


def _literal_assignment(tree: ast.Module, name: str) -> Any:
    return ast.literal_eval(_assignment(tree, name))


def _strings_in_assignment(tree: ast.Module, name: str) -> set[str]:
    return {
        node.value
        for node in ast.walk(_assignment(tree, name))
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }


def _settings_fields(tree: ast.Module) -> set[str]:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "Settings":
            return {
                item.target.id
                for item in node.body
                if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name)
            }
    raise AssertionError("Settings class is missing")


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"source function {name} is missing")


def _method(tree: ast.Module, class_name: str, name: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == name:
                    return item
    raise AssertionError(f"source method {class_name}.{name} is missing")


def _prints_name(node: ast.AST, name: str) -> bool:
    return any(
        isinstance(item, ast.Call)
        and isinstance(item.func, ast.Name)
        and item.func.id == "print"
        and any(isinstance(arg, ast.Name) and arg.id == name for arg in item.args)
        for item in ast.walk(node)
    )


def _calls_name(node: ast.AST, name: str) -> bool:
    return any(
        isinstance(item, ast.Call) and isinstance(item.func, ast.Name) and item.func.id == name
        for item in ast.walk(node)
    )


def _raw_environ_gets(tree: ast.AST) -> set[str]:
    values: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        func = node.func
        if not (
            isinstance(func, ast.Attribute)
            and func.attr == "get"
            and isinstance(func.value, ast.Attribute)
            and func.value.attr == "environ"
            and isinstance(func.value.value, ast.Name)
            and func.value.value.id == "os"
        ):
            continue
        key = node.args[0]
        if isinstance(key, ast.Constant) and isinstance(key.value, str):
            values.add(key.value)
    return values


def _attribute_names(tree: ast.AST) -> set[str]:
    return {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}


def _assert_contracts(document: dict[str, Any], sources: dict[str, str]) -> None:
    contracts = document["contracts"]
    trees = {name: ast.parse(source) for name, source in sources.items()}

    loader_map = _literal_assignment(trees["loader"], "_ENV_TO_FIELD")
    settings_fields = _settings_fields(trees["settings"])

    active = {row["env"]: row["field"] for row in contracts["active_launch_settings"]}
    assert len(active) == 6, "active_launch_settings must contain exactly six rows"
    for env_name, field_name in active.items():
        assert loader_map.get(env_name) == field_name, env_name
        assert field_name in settings_fields, field_name
    assert "WORKER_IMAGE_DIGEST" in _raw_environ_gets(trees["compile_facts"]), (
        "compile_facts must consume the raw WORKER_IMAGE_DIGEST launch value"
    )

    external_secret_fields = {
        "CIVITAI_API_KEY": "civitai_api_key",
        "HF_TOKEN": "hf_token",
    }
    external_secret_names = contracts["external_secret_env_names"]
    assert external_secret_names == sorted(external_secret_names)
    assert external_secret_names == sorted(external_secret_fields)
    provision_fields = _attribute_names(trees["provision"])
    for env_name, field_name in external_secret_fields.items():
        assert loader_map.get(env_name) == field_name, env_name
        assert field_name in settings_fields, field_name
        assert field_name in provision_fields, (
            f"models.provision no longer consumes Settings.{field_name}"
        )

    c2pa = contracts["c2pa"]
    supplied = {row["env"]: row["field"] for row in c2pa["supplied"]}
    assert len(supplied) == 3, "C2PA supplied must contain exactly three rows"
    for env_name, field_name in supplied.items():
        assert loader_map.get(env_name) == field_name, env_name
        assert field_name in settings_fields, field_name

    forbidden = set(c2pa["forbidden"])
    loader_forbidden = set(_literal_assignment(trees["loader"], "REFUSED_KEY_MATERIAL"))
    runtime_forbidden = set(_literal_assignment(trees["c2pa"], "_REFUSED_KEY_ENVS"))
    assert forbidden == loader_forbidden, "loader C2PA forbidden set drifted"
    assert forbidden == runtime_forbidden, "runtime C2PA forbidden set drifted"
    assert not {name.lower().removeprefix("gen_worker_") for name in forbidden} & settings_fields, (
        "forbidden C2PA key material became a Settings field"
    )

    fill = contracts["managed_fill_source"]
    assert loader_map.get(fill["env"]) == fill["field"]
    assert fill["field"] in settings_fields
    store_init = _method(trees["model_store"], "ModelStore", "__init__")
    assert _calls_name(store_init, "tensorhub_fill_source_dir"), (
        "ModelStore no longer resolves the managed fill source"
    )
    assert fill["env"] in _raw_environ_gets(store_init), (
        "ModelStore no longer names the managed fill env in its boot diagnosis"
    )

    topology_env = contracts["execution_topology"]["env"]
    assert _literal_assignment(trees["topology"], "ENV_VAR") == topology_env
    assert _literal_assignment(trees["procsplit"], "ENV_TOPOLOGY") == topology_env
    assert topology_env in _strings_in_assignment(trees["loader"], "_OWNED_NON_SETTINGS")

    marker = contracts["build_input_failure"]["marker"]
    assert _literal_assignment(trees["discovery"], "BUILD_INPUT_FAILURE_MARKER") == marker
    assert _prints_name(
        _function(trees["discovery"], "_fail_build_input"),
        "BUILD_INPUT_FAILURE_MARKER",
    ), "discovery no longer emits its build-input failure marker"


def test_hub_worker_boundary_contracts_match_pgw1239() -> None:
    _assert_contracts(_document(), _sources())


def test_hub_worker_boundary_digest_matches_pgw1239() -> None:
    active = [
        line.strip().split()[0]
        for line in _DEFAULT_DIGEST.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert len(active) == 1
    assert active[0] == hashlib.sha256(_DEFAULT_CORPUS.read_bytes()).hexdigest()


def test_drift_script_carries_every_private_consumer_path_pgw1239() -> None:
    script = (_ROOT / "scripts" / "hub-worker-boundary-drift.sh").read_text(encoding="utf-8")
    assert 'hub_rel="internal/wirecontract/testdata"' in script
    assert 'trainer_rel="image_lora_finetuner/tests/testdata"' in script
    assert 'if [ "$side" = "pgw" ] && [ -z "$peer_dir" ]; then' in script


def _run_contract_test(
    *, corpus: Path | None = None, source_override: tuple[str, Path] | None = None
) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    if corpus is not None:
        env["HUB_WORKER_BOUNDARY_CONTRACT_FILE"] = os.fspath(corpus)
    if source_override is not None:
        name, path = source_override
        env[_SOURCE_ENV[name]] = os.fspath(path)
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            os.fspath(Path(__file__)),
            "-k",
            "test_hub_worker_boundary_contracts_match_pgw1239",
        ],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize(
    "contract_class",
    [
        "active_launch_settings",
        "external_secret_env_names",
        "c2pa",
        "managed_fill_source",
        "execution_topology",
        "build_input_failure",
    ],
)
def test_each_contract_class_has_semantic_red_pgw1239(tmp_path: Path, contract_class: str) -> None:
    document = json.loads(_DEFAULT_CORPUS.read_text(encoding="utf-8"))
    contracts = document["contracts"]
    if contract_class == "active_launch_settings":
        contracts[contract_class][0]["field"] = "broken_active_launch_field"
    elif contract_class == "external_secret_env_names":
        contracts[contract_class][0] = "BROKEN_EXTERNAL_SECRET_ENV"
    elif contract_class == "c2pa":
        contracts[contract_class]["supplied"][0]["field"] = "broken_c2pa_field"
    elif contract_class == "managed_fill_source":
        contracts[contract_class]["field"] = "broken_fill_field"
    elif contract_class == "execution_topology":
        contracts[contract_class]["env"] = "BROKEN_EXECUTION_TOPOLOGY"
    else:
        contracts[contract_class]["marker"] = "BROKEN_BUILD_INPUT_FAILURE"

    corpus = tmp_path / _DEFAULT_CORPUS.name
    corpus.write_text(json.dumps(document), encoding="utf-8")
    got = _run_contract_test(corpus=corpus)
    assert got.returncode == 1, got.stdout + got.stderr


@pytest.mark.parametrize(
    ("source_name", "old", "new"),
    [
        ("loader", '"WORKER_ID": "worker_id"', '"WORKER_ID": "broken_worker_id"'),
        ("settings", "worker_id: str =", "broken_worker_id: str ="),
        ("loader", '"HF_TOKEN": "hf_token"', '"HF_TOKEN": "broken_hf_token"'),
        (
            "provision",
            "current_or(_STANDALONE).civitai_api_key",
            "current_or(_STANDALONE).hf_home",
        ),
        (
            "loader",
            '"GEN_WORKER_C2PA_KEY_PEM": (',
            '"GEN_WORKER_C2PA_KEY_PEM_BROKEN": (',
        ),
        (
            "c2pa",
            '"GEN_WORKER_C2PA_KEY_PEM",',
            '"GEN_WORKER_C2PA_KEY_PEM_BROKEN",',
        ),
        (
            "model_store",
            "fill_source_dir or tensorhub_fill_source_dir()",
            "fill_source_dir or tensorhub_cas_dir()",
        ),
        (
            "topology",
            'ENV_VAR = "WORKER_EXECUTION_TOPOLOGY"',
            'ENV_VAR = "BROKEN_EXECUTION_TOPOLOGY"',
        ),
        (
            "procsplit",
            'ENV_TOPOLOGY = "WORKER_EXECUTION_TOPOLOGY"',
            'ENV_TOPOLOGY = "BROKEN_EXECUTION_TOPOLOGY"',
        ),
        (
            "loader",
            '    "WORKER_EXECUTION_TOPOLOGY",',
            '    "BROKEN_EXECUTION_TOPOLOGY",',
        ),
        (
            "discovery",
            'BUILD_INPUT_FAILURE_MARKER = "TENSORHUB_BUILD_INPUT_FAILURE:discovery"',
            'BUILD_INPUT_FAILURE_MARKER = "BROKEN_BUILD_INPUT_FAILURE"',
        ),
        (
            "compile_facts",
            'os.environ.get("WORKER_IMAGE_DIGEST", "")',
            'os.environ.get("BROKEN_WORKER_IMAGE_DIGEST", "")',
        ),
    ],
)
def test_each_source_binding_has_red_pgw1239(
    tmp_path: Path, source_name: str, old: str, new: str
) -> None:
    source = _SOURCE_PATHS[source_name].read_text(encoding="utf-8")
    assert source.count(old) == 1, (source_name, old, source.count(old))
    candidate = tmp_path / f"{source_name}.py"
    candidate.write_text(source.replace(old, new, 1), encoding="utf-8")
    got = _run_contract_test(source_override=(source_name, candidate))
    assert got.returncode == 1, got.stdout + got.stderr


def test_digest_gate_can_go_red_pgw1239(tmp_path: Path) -> None:
    corpus = tmp_path / _DEFAULT_CORPUS.name
    digest = tmp_path / _DEFAULT_DIGEST.name
    corpus.write_bytes(_DEFAULT_CORPUS.read_bytes() + b"\n")
    digest.write_bytes(_DEFAULT_DIGEST.read_bytes())
    got = subprocess.run(
        [
            sys.executable,
            os.fspath(_ROOT / "scripts" / "check_hub_worker_boundary_contracts_digest.py"),
            "--corpus",
            os.fspath(corpus),
            "--digest",
            os.fspath(digest),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert got.returncode == 1
    assert "changed without its digest" in got.stdout


def test_redigested_peer_mutation_reaches_byte_comparison_pgw1239(
    tmp_path: Path,
) -> None:
    peer = tmp_path / "peer"
    peer.mkdir()
    document = json.loads(_DEFAULT_CORPUS.read_text(encoding="utf-8"))
    document["contracts"]["active_launch_settings"][0]["field"] = "peer_mutation"
    peer_corpus = peer / _DEFAULT_CORPUS.name
    peer_corpus.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    peer_digest = hashlib.sha256(peer_corpus.read_bytes()).hexdigest()
    (peer / _DEFAULT_DIGEST.name).write_text(
        f"{peer_digest}  {_DEFAULT_CORPUS.name}\n", encoding="utf-8"
    )

    got = subprocess.run(
        [os.fspath(_ROOT / "scripts" / "hub-worker-boundary-drift.sh")],
        env={**os.environ, "HUB_WORKER_BOUNDARY_PEER_DIR": os.fspath(peer)},
        check=False,
        capture_output=True,
        text=True,
    )
    assert got.returncode == 1
    assert "differs from python-gen-worker" in got.stderr
    assert "peer corpus does not match its digest" not in got.stderr
