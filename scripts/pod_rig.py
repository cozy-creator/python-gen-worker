#!/usr/bin/env python3
"""pgw#1347 — the pod mint-rig, runnable from anywhere.

    scripts/pod_rig.py mint --gpu a40 --rail 2.00 --lane pgw1331-clip \
        --target gen_worker.model.catalog.flux1_dev:FLUX1_DEV --runner clip
    scripts/pod_rig.py sweep
    scripts/pod_rig.py terminate --pod <id>

The package lives in `scripts/mint_rig/` and is deliberately NOT in
`src/gen_worker`: no pod needs the thing that rents pods, and shipping a control
plane inside the worker wheel is how a rented machine ends up holding an API key
it has no use for. This file is the entry point that makes `scripts/` importable
without a PYTHONPATH the operator has to remember — which matters because the
kill-set records this exact command as the way to stop a pod.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mint_rig.__main__ import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
