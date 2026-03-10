from __future__ import annotations

import sys
from pathlib import Path

from gen_worker.worker import Worker


def _make_worker(tmp_path: Path, **kwargs) -> Worker:
    mod_dir = tmp_path / "mod"
    mod_dir.mkdir(parents=True, exist_ok=True)
    (mod_dir / "hello_mod.py").write_text(
        """
from __future__ import annotations

import msgspec
from gen_worker import RequestContext, ResourceRequirements, worker_function

class Input(msgspec.Struct):
    name: str

class Output(msgspec.Struct):
    message: str

@worker_function(ResourceRequirements())
def hello(ctx: RequestContext, payload: Input) -> Output:
    return Output(message=f"hello {payload.name}")
""".lstrip(),
        encoding="utf-8",
    )
    sys.path.insert(0, str(mod_dir))
    return Worker(
        scheduler_addr="127.0.0.1:50051",
        user_module_names=["hello_mod"],
        worker_jwt="jwt-test",
        **kwargs,
    )


def test_lb_only_retries_keeps_seed_list_stable(tmp_path: Path) -> None:
    w = _make_worker(tmp_path, scheduler_addrs=["orchestrator-lb:50051"], lb_only_retries=True)
    before = list(w.scheduler_addrs)
    w._set_scheduler_addr("orch-direct:50051")
    assert w.scheduler_addr == "orch-direct:50051"
    assert w.scheduler_addrs == before


def test_reconnect_delay_is_exponential_and_capped(tmp_path: Path) -> None:
    w = _make_worker(tmp_path, reconnect_delay=0.1, lb_only_retries=True)
    w._reconnect_jitter_seconds = 0.0
    w._reconnect_delay_max = 1.0
    assert w._next_reconnect_delay(1) == 0.1
    assert w._next_reconnect_delay(2) == 0.2
    assert w._next_reconnect_delay(3) == 0.4
    assert w._next_reconnect_delay(4) == 0.8
    assert w._next_reconnect_delay(5) == 1.0
    assert w._next_reconnect_delay(10) == 1.0
