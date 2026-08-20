"""gen-worker CLI — top-level argparse dispatcher. THE endpoint interface.

pgw#1491, Paul's ruling 2026-08-19: gen-worker IS the endpoint CLI. Every
endpoint-scoped verb lives here because every one of them must run INSIDE the
endpoint's venv against its pinned torch — ``lock`` needs its tracer,
``compile`` its inductor, ``up`` its runtime. An external tool could only ever
shell in and invoke this.

The surface, in the order a user meets it::

    login       authenticate to the hub (credentials USER-GLOBAL, shared by
                every endpoint venv on this machine)
    lock        AUTHOR-time: trace the endpoint, write endpoint.lock (committed)
    download    fetch a checkpoint onto this machine
    compile     pre-warm compiled graphs for this card: fetch else build
    up [-d]     bring the endpoint up resident (foreground; -d detaches)
    run         send one request to the endpoint that is up
    down        stop it
    publish     push this endpoint's source to the hub; the hub builds

plus ``models export`` and ``release derive``, which are machinery rather than
user surface.

HARDCUT (pgw#1491): ``sync``, ``serve`` (as a verb) and ``gen`` are DELETED
spellings, not aliases. ``sync`` was incoherent once checkpoints became runtime
config — deps are ``uv sync``, artifacts are ``compile``, weights are
``download``. ``serve`` split into ``up``/``down`` so that starting an endpoint
and sending it work are different acts. Pre-launch, a compatibility alias would
only preserve the confusion it renames.

Exit codes (mirrored by every subcommand):

- ``0``    success
- ``1``    the command failed (message to stderr)
- ``2``    CLI usage / validation error (help / message to stderr)
- ``3``    model resolution failure (tensorhub unreachable AND cache miss)
- ``130``  SIGINT (Ctrl-C; standard shell convention)

"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from .. import config


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gen-worker",
        description=(
            "The endpoint CLI. Download a checkpoint, compile for this card, "
            "bring the endpoint up, run requests against it, publish it."
        ),
    )
    sub = parser.add_subparsers(dest="command", metavar="<command>")
    sub.required = False  # so `gen-worker --help` works without a sub

    # Lazy-imported so the cli stays cheap for `gen-worker --help`: `up` pulls
    # in torch, and a user asking for help must not pay for it.
    from . import models_export as _models_mod
    _models_mod.add_subparser(sub)

    from . import release as _release_mod
    _release_mod.add_subparser(sub)

    # The endpoint surface (pgw#1491). Every one of these is a front door onto
    # ONE serving core — `run` in particular executes nothing itself.
    from . import lock as _lock_mod
    _lock_mod.add_subparser(sub)

    from . import download as _download_mod
    _download_mod.add_subparser(sub)

    from . import compile as _compile_mod
    _compile_mod.add_subparser(sub)

    from . import up as _up_mod
    _up_mod.add_subparser(sub)

    from . import run as _run_mod
    _run_mod.add_subparser(sub)

    from . import login as _login_mod
    _login_mod.add_subparser(sub)

    from . import publish as _publish_mod
    _publish_mod.add_subparser(sub)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point for the ``gen-worker`` console_script.

    Returns the integer exit code so tests can call ``main([...])`` directly
    without spawning a subprocess. The console_script wrapper turns the
    returned int into the process exit code via ``sys.exit(main())``.
    """
    # the bootstrap-owned load for the STANDALONE CLI process entry — one load
    # PER PROCESS ENTRY, and this is the third (after `entrypoint._run_main`
    # and `procsplit.parent.run_parent`). Without it the `models/` helpers fall
    # back to their zero-value standalone defaults and a `gen-worker run`
    # silently loses TENSORHUB_URL / _TOKEN / _CACHE_DIR.
    config.install(config.load_settings())
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help(sys.stderr)
        return 2
    handler = getattr(args, "_handler", None)
    if handler is None:  # pragma: no cover - argparse guards this
        parser.print_help(sys.stderr)
        return 2
    try:
        return int(handler(args) or 0)
    except SystemExit as e:
        return int(e.code or 0)


if __name__ == "__main__":  # pragma: no cover - module-run convenience
    sys.exit(main())
