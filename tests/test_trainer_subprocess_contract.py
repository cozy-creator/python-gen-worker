from __future__ import annotations

from pathlib import Path

from gen_worker.trainer import TrainerSubprocessContractV1, read_subprocess_contract, write_subprocess_contract


def test_subprocess_contract_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "contract.json"
    c1 = TrainerSubprocessContractV1(
        trainer_plugin="my_pkg.train:MyTrainer",
        job_spec_path="/tmp/job.json",
        arrow_ipc_path="/tmp/batches.arrow",
        events_path="/tmp/events.jsonl",
        extra_env={"A": "1"},
    )
    write_subprocess_contract(str(path), c1)
    c2 = read_subprocess_contract(str(path))
    assert c2.contract_version == "v1"
    assert c2.trainer_plugin == "my_pkg.train:MyTrainer"
    assert c2.arrow_ipc_path == "/tmp/batches.arrow"
    assert c2.extra_env["A"] == "1"
