"""Bind the public worker-value corpus to python-gen-worker's live values."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from gen_worker import activity, cell_resolve, compile_cache, host_canary
from gen_worker.convert.layout_converters import derived_artifact_identity
from gen_worker.local_cell_store import UNTRUSTED_REFUSAL_CODE
from gen_worker.models import execution_lanes
from gen_worker.models.cozy_snapshot import PICKLE_WEIGHT_EXTENSIONS
from gen_worker.models.tensor_layout_contract import (
    LayoutDeclarationError,
    LayoutId,
    parse_layout_id,
)
from gen_worker.models.w8a8_lora import RANK_BUCKETS


_ROOT = Path(__file__).parents[1]
_DEFAULT_CORPUS = Path(__file__).parent / "testdata" / "worker_value_contracts.json"
_DEFAULT_DIGEST = Path(__file__).parent / "testdata" / "WORKER_VALUE_CONTRACTS_DIGEST"
_CORPUS = Path(os.environ.get("WORKER_VALUE_CONTRACT_FILE", _DEFAULT_CORPUS))


def _document() -> dict[str, Any]:
    document = json.loads(_CORPUS.read_text(encoding="utf-8"))
    assert document["schema"] == "worker-value-contracts-v1"
    return document


def _lane_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for body in execution_lanes.known_execution_lane_bodies():
        supported: list[str] = []
        family = ""
        for mode in (execution_lanes.EXEC_COMPILED, execution_lanes.EXEC_EAGER):
            try:
                lane = execution_lanes.parse_execution_lane(f"{body}+{mode}")
            except ValueError:
                continue
            if execution_lanes.valid_execution_lane(lane):
                supported.append(mode)
                family = execution_lanes.family_of(lane)
        rows.append({"body": body, "family": family, "execution": supported})
    return rows


def test_exact_worker_values_match_pgw1237() -> None:
    exact = _document()["exact"]

    assert exact["activity_decision_kinds"] == [
        activity.KIND_SELF_MINT_COMPILE,
        activity.KIND_WARMUP,
        activity.KIND_AOT_MINT,
    ]
    assert exact["activity_duration_rollups"] == [
        {"kind": activity.KIND_AOT_MINT, "phase": activity.PHASE_MINTED},
        {"kind": activity.KIND_JIT_COMPILE, "phase": activity.PHASE_MINTED},
    ]
    assert set(exact["cell_resolve_hub_refusal_codes"]) == set(
        cell_resolve.REFUSAL_CODES
    )
    assert exact["cell_publish_untrusted_refusal_code"] == UNTRUSTED_REFUSAL_CODE
    assert exact["compilecache_rank_buckets"] == list(RANK_BUCKETS)
    assert exact["execution_lane_bodies"] == _lane_rows()
    assert exact["pickle_weight_extensions"] == list(PICKLE_WEIGHT_EXTENSIONS)


def test_bounded_worker_relations_match_pgw1237() -> None:
    relations = _document()["relations"]

    assert relations["sku_slug"]["domain"] == "ASCII GPU identity strings"
    for case in relations["sku_slug"]["cases"]:
        assert compile_cache.sku_slug(case["input"]) == case["result"]

    assert relations["system_repo"]["domain"] == "non-empty family names"
    for case in relations["system_repo"]["cases"]:
        assert compile_cache.system_repo(case["family"]) == case["result"]

    rank_buckets = {0, *RANK_BUCKETS}
    lane_relation = relations["execution_lane_with_bucket"]
    for case in lane_relation["cases"]:
        assert case["bucket"] in rank_buckets
        assert compile_cache.execution_lane_label(
            case["base"], case["bucket"]
        ) == case["result"]

    layout_relation = relations["layout_id"]
    for case in layout_relation["cases"]:
        got = parse_layout_id(case["input"], where="pgw#1237 corpus")
        assert got.topology == case["topology"]
        assert got.quant == case["quant"]
        assert got.render() == case["render"]
    for value in layout_relation["refusals"]:
        with pytest.raises(LayoutDeclarationError):
            parse_layout_id(value, where="pgw#1237 corpus")

    for case in relations["derived_artifact_identity"]["cases"]:
        target = LayoutId(**case["target"])
        assert derived_artifact_identity(
            case["source_digest"], case["chain_digests"], target
        ) == case["result"]

    fabric = relations["sp_fabric"]
    assert fabric["domain"] == "canonical worker-emitted interconnect spellings"
    assert host_canary.SP_MIN_PEER_GBPS == fabric["min_peer_gbps"]
    for case in fabric["cases"]:
        assert host_canary.sp_admits(
            case["interconnect"], case["peer_gbps"]
        ) is case["admits"]
        assert host_canary.is_fabric_wedge(
            case["peer_access"], case["peer_gbps"]
        ) is case["wedge"]


def test_worker_value_corpus_digest_matches_pgw1237() -> None:
    active = [
        line.strip().split()[0]
        for line in _DEFAULT_DIGEST.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert len(active) == 1
    assert active[0] == hashlib.sha256(_DEFAULT_CORPUS.read_bytes()).hexdigest()


def test_worker_value_digest_gate_can_go_red_pgw1237(tmp_path: Path) -> None:
    corpus = tmp_path / _DEFAULT_CORPUS.name
    digest = tmp_path / _DEFAULT_DIGEST.name
    corpus.write_bytes(_DEFAULT_CORPUS.read_bytes() + b"\n")
    digest.write_bytes(_DEFAULT_DIGEST.read_bytes())
    got = subprocess.run(
        [
            sys.executable,
            os.fspath(_ROOT / "scripts" / "check_worker_value_contracts_digest.py"),
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


def test_worker_value_semantic_fence_can_go_red_pgw1237(tmp_path: Path) -> None:
    document = json.loads(_DEFAULT_CORPUS.read_text(encoding="utf-8"))
    document["exact"]["compilecache_rank_buckets"][0] = 17
    corpus = tmp_path / _DEFAULT_CORPUS.name
    corpus.write_text(json.dumps(document), encoding="utf-8")

    got = subprocess.run(
        [
            "uv",
            "run",
            "pytest",
            "-q",
            os.fspath(Path(__file__)),
            "-k",
            "test_exact_worker_values_match_pgw1237",
        ],
        env={**os.environ, "WORKER_VALUE_CONTRACT_FILE": os.fspath(corpus)},
        check=False,
        capture_output=True,
        text=True,
    )
    assert got.returncode == 1
    assert "compilecache_rank_buckets" in got.stdout


def test_worker_value_relation_fence_can_go_red_pgw1237(tmp_path: Path) -> None:
    document = json.loads(_DEFAULT_CORPUS.read_text(encoding="utf-8"))
    document["relations"]["sku_slug"]["cases"][0]["result"] = "not-rtx-4090"
    corpus = tmp_path / _DEFAULT_CORPUS.name
    corpus.write_text(json.dumps(document), encoding="utf-8")

    got = subprocess.run(
        [
            "uv",
            "run",
            "pytest",
            "-q",
            os.fspath(Path(__file__)),
            "-k",
            "test_bounded_worker_relations_match_pgw1237",
        ],
        env={**os.environ, "WORKER_VALUE_CONTRACT_FILE": os.fspath(corpus)},
        check=False,
        capture_output=True,
        text=True,
    )
    assert got.returncode == 1
    assert "not-rtx-4090" in got.stdout


def test_worker_value_peer_gate_can_go_red_pgw1237(tmp_path: Path) -> None:
    peer = tmp_path / "peer"
    peer.mkdir()
    for source in (_DEFAULT_CORPUS, _DEFAULT_DIGEST):
        (peer / source.name).write_bytes(source.read_bytes())
    (peer / _DEFAULT_CORPUS.name).write_bytes(_DEFAULT_CORPUS.read_bytes() + b"\n")
    peer_digest = hashlib.sha256((peer / _DEFAULT_CORPUS.name).read_bytes()).hexdigest()
    (peer / _DEFAULT_DIGEST.name).write_text(peer_digest + "\n", encoding="utf-8")

    got = subprocess.run(
        ["bash", os.fspath(_ROOT / "scripts" / "worker-value-contract-drift.sh")],
        env={
            "PATH": "/usr/bin:/bin",
            "WORKER_VALUE_CONTRACT_PEER_DIR": os.fspath(peer),
        },
        check=False,
        capture_output=True,
        text=True,
    )
    assert got.returncode == 1
    assert "worker_value_contracts.json differs from python-gen-worker" in got.stderr
