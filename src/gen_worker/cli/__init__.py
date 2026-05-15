"""gen-worker CLI — top-level argparse dispatcher.

First subcommand: ``run`` — invoke an endpoint method in the local Python
interpreter against a JSON payload. See ``docs/local-dev.md``.

Exit codes (mirrored by every subcommand):

- ``0``    success
- ``1``    user-code exception (traceback to stderr)
- ``2``    CLI usage / validation error (help / message to stderr)
- ``3``    model resolution failure (tensorhub unreachable AND cache miss,
           or ``--offline`` with cache miss)
- ``130``  SIGINT (Ctrl-C; standard shell convention)
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gen-worker",
        description=(
            "Develop and dogfood gen-worker endpoints locally. "
            "Sub-command 'run' executes one endpoint method against a JSON payload "
            "in the local Python interpreter, mirroring production behavior for "
            "model resolution, payload validation, and context wiring."
        ),
    )
    sub = parser.add_subparsers(dest="command", metavar="<command>")
    sub.required = False  # so `gen-worker --help` works without a sub

    # Lazy-import the run subcommand wiring so the cli stays cheap to import
    # for `gen-worker --help` in CI.
    from . import run as _run_mod
    _run_mod.add_subparser(sub)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point for the ``gen-worker`` console_script.

    Returns the integer exit code so tests can call ``main([...])`` directly
    without spawning a subprocess. The console_script wrapper turns the
    returned int into the process exit code via ``sys.exit(main())``.
    """
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
