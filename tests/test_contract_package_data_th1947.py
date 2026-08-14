"""Package-data authority and temporary-projection red proofs for th#1947."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from importlib.resources import files
from pathlib import Path

from gen_worker.contracts import (
    WORKER_VALUE_CONTRACT_NAME,
    worker_value_contract_bytes,
    worker_value_contract_sha256,
)


ROOT = Path(__file__).parents[1]
PACKAGE = files("gen_worker.contracts")


def _recorded_digest() -> str:
    for line in PACKAGE.joinpath("WORKER_VALUE_CONTRACTS_DIGEST").read_text().splitlines():
        value = line.strip()
        if value and not value.startswith("#"):
            return value
    return ""


def test_importable_corpus_carries_explicit_axes_and_emitters() -> None:
    payload = worker_value_contract_bytes()
    assert payload == PACKAGE.joinpath(WORKER_VALUE_CONTRACT_NAME).read_bytes()
    document = json.loads(payload)
    axes = document["compiled_graph_runtime"]["axes"]
    assert set(axes) == {
        "hub_http_refusal_codes",
        "per_answer_statuses",
        "worker_only_verdicts",
        "envelope",
    }
    assert axes["hub_http_refusal_codes"]["emitter"] == "tensorhub"
    assert axes["per_answer_statuses"]["emitter"] == "tensorhub"
    assert axes["worker_only_verdicts"]["emitter"] == "python-gen-worker"
    assert all(row["emitter"] for row in axes["envelope"]["values"])
    assert "cell_resolve_ambiguous" not in payload.decode()
    assert "cell_publish_untrusted_tier" not in payload.decode()


def test_packaged_corpus_matches_its_packaged_digest() -> None:
    assert worker_value_contract_sha256() == _recorded_digest()
    assert worker_value_contract_sha256() == hashlib.sha256(
        worker_value_contract_bytes()
    ).hexdigest()


def test_temporary_testdata_projection_is_byte_identical() -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "project_worker_value_contracts.py")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_projection_gate_goes_red_for_independently_edited_bytes(
    tmp_path: Path,
) -> None:
    for name in (WORKER_VALUE_CONTRACT_NAME, "WORKER_VALUE_CONTRACTS_DIGEST"):
        (tmp_path / name).write_bytes(PACKAGE.joinpath(name).read_bytes())
    (tmp_path / WORKER_VALUE_CONTRACT_NAME).write_bytes(b"{}\n")
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "project_worker_value_contracts.py"),
            "--projection-dir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert WORKER_VALUE_CONTRACT_NAME in result.stdout
