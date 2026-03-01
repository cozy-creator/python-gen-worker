from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class TrainerSubprocessContractV1:
    """Versioned process-boundary contract between runtime and trainer subprocess."""

    contract_version: str = "v1"
    trainer_plugin: str = ""
    job_spec_path: str = ""
    arrow_ipc_path: str = ""
    checkpoints_dir: str = "/tmp/training/checkpoints"
    samples_dir: str = "/tmp/training/samples"
    events_path: str | None = None
    extra_env: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.contract_version != "v1":
            raise ValueError("unsupported contract_version")
        if not self.trainer_plugin:
            raise ValueError("trainer_plugin is required")
        if not self.job_spec_path:
            raise ValueError("job_spec_path is required")

    def to_json(self) -> dict[str, Any]:
        return {
            "contract_version": self.contract_version,
            "trainer_plugin": self.trainer_plugin,
            "job_spec_path": self.job_spec_path,
            "arrow_ipc_path": self.arrow_ipc_path,
            "checkpoints_dir": self.checkpoints_dir,
            "samples_dir": self.samples_dir,
            "events_path": self.events_path,
            "extra_env": dict(self.extra_env),
        }

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> "TrainerSubprocessContractV1":
        return cls(
            contract_version=str(payload.get("contract_version") or "v1"),
            trainer_plugin=str(payload.get("trainer_plugin") or ""),
            job_spec_path=str(payload.get("job_spec_path") or ""),
            arrow_ipc_path=str(payload.get("arrow_ipc_path") or ""),
            checkpoints_dir=str(payload.get("checkpoints_dir") or "/tmp/training/checkpoints"),
            samples_dir=str(payload.get("samples_dir") or "/tmp/training/samples"),
            events_path=str(payload.get("events_path") or "") or None,
            extra_env={str(k): str(v) for (k, v) in (payload.get("extra_env") or {}).items()},
        )


def write_subprocess_contract(path: str, contract: TrainerSubprocessContractV1) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(contract.to_json(), f, separators=(",", ":"), sort_keys=True)


def read_subprocess_contract(path: str) -> TrainerSubprocessContractV1:
    p = Path(path)
    with open(p, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError("subprocess contract must be a JSON object")
    return TrainerSubprocessContractV1.from_json(payload)


__all__ = [
    "TrainerSubprocessContractV1",
    "read_subprocess_contract",
    "write_subprocess_contract",
]
