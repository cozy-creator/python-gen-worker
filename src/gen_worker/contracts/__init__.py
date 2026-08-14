"""Importable machine-readable contracts owned by gen-worker."""

from __future__ import annotations

import hashlib
from importlib.resources import files


_PACKAGE = "gen_worker.contracts"
WORKER_VALUE_CONTRACT_NAME = "worker_value_contracts.json"
WORKER_VALUE_DIGEST_NAME = "WORKER_VALUE_CONTRACTS_DIGEST"


def worker_value_contract_bytes() -> bytes:
    """Return the canonical worker-value corpus exactly as packaged."""

    return files(_PACKAGE).joinpath(WORKER_VALUE_CONTRACT_NAME).read_bytes()


def worker_value_contract_sha256() -> str:
    """Return the digest of the exact packaged corpus bytes."""

    return hashlib.sha256(worker_value_contract_bytes()).hexdigest()


__all__ = [
    "WORKER_VALUE_CONTRACT_NAME",
    "WORKER_VALUE_DIGEST_NAME",
    "worker_value_contract_bytes",
    "worker_value_contract_sha256",
]
