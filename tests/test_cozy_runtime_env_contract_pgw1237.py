"""Bind the shared Cozy runtime-env corpus to the worker's real seams."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import pytest

from gen_worker import aot_delivery, config
from gen_worker.cli.local_context import _save_local_bytes
from gen_worker.models.cache_paths import (
    open_worker_cas,
    tensorhub_cache_dir,
    tensorhub_cas_dir,
)

_CONTRACTS = Path(__file__).parents[1] / "src" / "gen_worker" / "contracts"
_DEFAULT_CORPUS = _CONTRACTS / "cozy_runtime_env_vectors.json"
_DEFAULT_DIGEST = _CONTRACTS / "COZY_RUNTIME_ENV_DIGEST"
_CORPUS = Path(os.environ.get("COZY_RUNTIME_ENV_CORPUS", _DEFAULT_CORPUS))


def _document() -> dict[str, Any]:
    return json.loads(_CORPUS.read_text(encoding="utf-8"))


def _variables() -> dict[str, dict[str, Any]]:
    document = _document()
    assert document["schema"] == "cozy-runtime-env-v1"
    rows = document["variables"]
    assert isinstance(rows, list)
    return {row["role"]: row for row in rows}


def test_runtime_env_semantics_match_pgw1237(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    rows = _variables()
    assert set(rows) == {
        "cache_root",
        "asset_ref_root",
        "hub_url",
        "bearer_token",
    }

    cache = rows["cache_root"]
    cache_root = tmp_path / "cache-root"
    monkeypatch.setenv(cache["name"], os.fspath(cache_root))
    config.reload_for_test()
    try:
        assert not hasattr(config.Settings(), "tensorhub_cas_dir")
        assert tensorhub_cache_dir() == cache_root
        assert tensorhub_cas_dir() == cache_root / cache["consumer_relative_path"]
        assert open_worker_cas().root == tensorhub_cas_dir()
    finally:
        config.reset_for_test()

    output = rows["asset_ref_root"]
    assert output["name"] == "GEN_WORKER_LOCAL_OUTPUT_DIR"
    output_root = tmp_path / "output-root"
    monkeypatch.setenv(output["name"], os.fspath(output_root))
    asset = _save_local_bytes(output["sample_asset_ref"], b"pgw#1237")
    expected = output_root / output["sample_asset_ref"]
    assert asset.local_path is not None
    assert Path(asset.local_path) == expected
    assert expected.read_bytes() == b"pgw#1237"

    hub_url = rows["hub_url"]
    assert hub_url == {
        "name": "TENSORHUB_URL",
        "role": "hub_url",
        "config_field": "tensorhub_url",
    }
    monkeypatch.setenv(hub_url["name"], "https://hub.invalid")
    settings = config.reload_for_test()
    try:
        assert getattr(settings, hub_url["config_field"]) == "https://hub.invalid"
    finally:
        config.reset_for_test()

    bearer = rows["bearer_token"]
    assert bearer["name"] == "TENSORHUB_TOKEN"
    assert bearer["config_field"] == "tensorhub_token"
    assert bearer["optional"] is True
    assert bearer["secret"] is True
    assert "value" not in bearer and "sample" not in bearer
    monkeypatch.setenv(bearer["name"], "redacted-test-sentinel")
    settings = config.reload_for_test()
    try:
        assert getattr(settings, bearer["config_field"]) == "redacted-test-sentinel"
    finally:
        config.reset_for_test()


def test_aot_default_uses_the_canonical_worker_cas(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("TENSORHUB_CACHE_DIR", os.fspath(tmp_path / "cache-root"))
    config.reload_for_test()
    seen: list[Path | None] = []
    real_open = open_worker_cas

    def _open(root: Path | None = None) -> Any:
        seen.append(root)
        return real_open(root)

    monkeypatch.setattr(aot_delivery, "open_worker_cas", _open)
    try:
        with pytest.raises(aot_delivery.NamedArtifactUnavailable, match="no transport"):
            aot_delivery._materialize_named_artifact(
                "cg-key-v1-" + "a" * 56,
                "micro-diffusion",
                "compiled-graph-v1",
                "sha256:" + "0" * 64,
                None,
                receipt=object(),  # type: ignore[arg-type]
                cache_dir=None,
                what="root convergence test",
            )
        assert seen == [None]
        assert open_worker_cas().root == tensorhub_cas_dir()
    finally:
        config.reset_for_test()


def test_runtime_env_corpus_digest_matches_pgw1237() -> None:
    active = [
        line.strip().split()[0]
        for line in _DEFAULT_DIGEST.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert len(active) == 1
    assert active[0] == hashlib.sha256(_DEFAULT_CORPUS.read_bytes()).hexdigest()


def test_removal_blocked_projection_matches_package_authority() -> None:
    projection = Path(__file__).parent / "testdata"
    assert (projection / _DEFAULT_CORPUS.name).read_bytes() == _DEFAULT_CORPUS.read_bytes()
    assert (projection / _DEFAULT_DIGEST.name).read_bytes() == _DEFAULT_DIGEST.read_bytes()


def test_runtime_env_digest_gate_can_go_red_pgw1237(tmp_path: Path) -> None:
    corpus = tmp_path / "cozy_runtime_env_vectors.json"
    digest = tmp_path / "COZY_RUNTIME_ENV_DIGEST"
    corpus.write_bytes(_DEFAULT_CORPUS.read_bytes() + b"\n")
    digest.write_bytes(_DEFAULT_DIGEST.read_bytes())
    got = subprocess.run(
        [
            os.fspath(Path(__file__).parents[1] / "scripts" / "check_cozy_runtime_env_digest.py"),
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


@pytest.mark.parametrize(
    ("role", "field", "mutation"),
    [
        ("cache_root", "consumer_relative_path", "not-cas"),
        ("asset_ref_root", "name", "GEN_WORKER_LOCAL_OUTPUT_DIR_BROKEN"),
        ("hub_url", "name", "TENSORHUB_URL_BROKEN"),
        ("bearer_token", "name", "TENSORHUB_TOKEN_BROKEN"),
    ],
)
def test_runtime_env_semantic_fence_can_go_red_pgw1237(
    tmp_path: Path, role: str, field: str, mutation: str
) -> None:
    document = json.loads(_DEFAULT_CORPUS.read_text(encoding="utf-8"))
    row = next(row for row in document["variables"] if row["role"] == role)
    row[field] = mutation
    corpus = tmp_path / "cozy_runtime_env_vectors.json"
    corpus.write_text(json.dumps(document), encoding="utf-8")

    got = subprocess.run(
        [
            "uv",
            "run",
            "pytest",
            "-q",
            os.fspath(Path(__file__)),
            "-k",
            "test_runtime_env_semantics_match_pgw1237",
        ],
        env={**os.environ, "COZY_RUNTIME_ENV_CORPUS": os.fspath(corpus)},
        check=False,
        capture_output=True,
        text=True,
    )
    assert got.returncode == 1
    assert mutation in got.stdout


def test_runtime_env_peer_gate_can_go_red_pgw1237(tmp_path: Path) -> None:
    peer = tmp_path / "peer"
    peer.mkdir()
    for source in (_DEFAULT_CORPUS, _DEFAULT_DIGEST):
        (peer / source.name).write_bytes(source.read_bytes())
    (peer / _DEFAULT_CORPUS.name).write_bytes(_DEFAULT_CORPUS.read_bytes() + b"\n")
    peer_digest = hashlib.sha256((peer / _DEFAULT_CORPUS.name).read_bytes()).hexdigest()
    (peer / _DEFAULT_DIGEST.name).write_text(peer_digest + "\n", encoding="utf-8")

    got = subprocess.run(
        ["bash", os.fspath(Path(__file__).parents[1] / "scripts" / "cozy-runtime-env-drift.sh")],
        env={"PATH": "/usr/bin:/bin", "COZY_RUNTIME_ENV_PEER_DIR": os.fspath(peer)},
        check=False,
        capture_output=True,
        text=True,
    )
    assert got.returncode == 1
    assert "cozy_runtime_env_vectors.json differs" in got.stderr
