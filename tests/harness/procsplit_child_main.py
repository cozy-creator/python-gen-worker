from __future__ import annotations

import os
import sys


def main() -> int:
    from gen_worker import postmortem
    from gen_worker.config import load_settings
    from gen_worker.procsplit.oom_rank import raise_own_oom_score_adj
    from gen_worker.worker import Worker

    raise_own_oom_score_adj()

    postmortem.enable_fault_dump()

    settings = load_settings(
        orchestrator_public_addr="127.0.0.1:1",
        worker_id=os.environ.get("PGW763_WORKER_ID", "split-child"),
        tensorhub_cache_dir=os.environ.get("TENSORHUB_CACHE_DIR", ""),
    )
    modules = [
        m for m in os.environ.get(
            "PGW763_CHILD_MODULES", "harness.procsplit_endpoints"
        ).split(",") if m
    ]
    worker = Worker(settings, modules)
    return worker.run()


if __name__ == "__main__":
    sys.exit(main())
