"""Bind the shared Cozy runtime-env corpus to the worker's real seams."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from gen_worker import config
from gen_worker.cli.local_context import _save_local_bytes
from gen_worker.models.cache_paths import tensorhub_cache_dir, tensorhub_cas_dir


_DEFAULT_CORPUS = Path(__file__).parent / "testdata" / "cozy_runtime_env_vectors.json"
_DEFAULT_DIGEST = Path(__file__).parent / "testdata" / "COZY_RUNTIME_ENV_DIGEST"
_CORPUS = Path(os.environ.get("COZY_RUNTIME_ENV_CORPUS", _DEFAULT_CORPUS))


def _document() -> dict[str, Any]:
    return json.loads(_CORPUS.read_text(encoding="utf-8"))


def _variables() -> dict[str, dict[str, str]]:
    document = _document()
    assert document["schema"] == "cozy-runtime-env-v1"
    rows = document["variables"]
    assert isinstance(rows, list)
    return {row["role"]: row for row in rows}


def test_runtime_env_semantics_match_pgw1237(monkeypatch, tmp_path: Path) -> None:
    rows = _variables()
    assert set(rows) == {"cache_root", "asset_ref_root"}

    cache = rows["cache_root"]
    cache_root = tmp_path / "cache-root"
    monkeypatch.setenv(cache["name"], os.fspath(cache_root))
    config.reload_for_test()
    try:
        assert tensorhub_cache_dir() == cache_root
        assert tensorhub_cas_dir() == cache_root / cache["consumer_relative_path"]
    finally:
        config.reset_for_test()

    output = rows["asset_ref_root"]
    output_root = tmp_path / "output-root"
    monkeypatch.setenv(output["name"], os.fspath(output_root))
    asset = _save_local_bytes(output["sample_asset_ref"], b"pgw#1237")
    expected = output_root / output["sample_asset_ref"]
    assert Path(asset.local_path) == expected
    assert expected.read_bytes() == b"pgw#1237"


def test_runtime_env_corpus_digest_matches_pgw1237() -> None:
    active = [
        line.strip().split()[0]
        for line in _DEFAULT_DIGEST.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert len(active) == 1
    assert active[0] == hashlib.sha256(_DEFAULT_CORPUS.read_bytes()).hexdigest()


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


def test_runtime_env_semantic_fence_can_go_red_pgw1237(tmp_path: Path) -> None:
    document = json.loads(_DEFAULT_CORPUS.read_text(encoding="utf-8"))
    document["variables"][0]["consumer_relative_path"] = "not-cas"
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
    assert "not-cas" in got.stdout


def test_runtime_env_peer_gate_can_go_red_pgw1237(tmp_path: Path) -> None:
    peer = tmp_path / "peer"
    peer.mkdir()
    for source in (_DEFAULT_CORPUS, _DEFAULT_DIGEST):
        (peer / source.name).write_bytes(source.read_bytes())
    (peer / _DEFAULT_CORPUS.name).write_bytes(_DEFAULT_CORPUS.read_bytes() + b"\n")

    got = subprocess.run(
        ["bash", os.fspath(Path(__file__).parents[1] / "scripts" / "cozy-runtime-env-drift.sh")],
        env={"PATH": "/usr/bin:/bin", "COZY_RUNTIME_ENV_PEER_DIR": os.fspath(peer)},
        check=False,
        capture_output=True,
        text=True,
    )
    assert got.returncode == 1
    assert "cozy_runtime_env_vectors.json differs" in got.stderr
