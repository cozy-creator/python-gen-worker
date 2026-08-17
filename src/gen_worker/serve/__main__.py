"""pgw#1328: ``python -m gen_worker.serve`` — the ADOPT-ONLY worker entry.

The whole entry point is four statements, and their ORDER is the guarantee:

1. declare the role, so every later ``serve.role.adopt_only()`` read — the key
   deriver, the mint seam, the ingress fallback — answers the same way;
2. install the import blocker, so a module that names the mint lane raises
   here rather than at a call an hour into the pod's life;
3. import the ordinary worker entry, which is now free of the lane at module
   scope (it registers §4.28's mint supervision only in the other role);
4. run it.

**There is no flag, no env and no config.** §1.17 and Paul's standing rule keep
a DECISION out of the environment, and pgw#1327 already refused a
``GEN_WORKER_ADOPT_ONLY`` knob for this exact question. A pod is adopt-only
because of the command that started it, which is a property of the image, and
that is the only way to say it.

This is deliberately the SAME worker after step 3 — same hub protocol, same
dispatch, same residency, same telemetry. An adopt-only pod is not a different
program; it is this program with one capability removed and every miss it
would have absorbed turned into an answer.
"""

from __future__ import annotations

import sys


def main() -> int:
    from . import guard, role

    role.declare(role.ServeRole.ADOPT_ONLY)
    guard.install()

    from ..entrypoint import _run_main

    return int(_run_main())


if __name__ == "__main__":
    sys.exit(main())
