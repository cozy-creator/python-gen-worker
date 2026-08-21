"""gen-worker CLI — top-level argparse dispatcher. THE FIRST STATEMENT IS THE SETTINGS AUTHORITY, exactly as in `gen_worker/entrypoint.py`: this package is the OTHER front door (`gen-worker up` is what cozy-local and every local box boot), and `PYTORCH_CUDA_ALLOC_CONF` is read ONCE when the CUDA caching allocator initializes, so the declared env has to be in `os.environ` before anything that can reach torch is imported — a later imposition is a silent no-op (pgw#1640)."""

from __future__ import annotations

from ..settings_authority import impose_process_env  # noqa: E402

impose_process_env()

import argparse  # noqa: E402
import sys  # noqa: E402
from typing import List, Optional  # noqa: E402

from .. import config  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gen-worker",
        description=(
            "The endpoint CLI. Download a checkpoint, compile for this card, "
            "bring the endpoint up, run requests against it, publish it."
        ),
    )
    sub = parser.add_subparsers(dest="command", metavar="<command>")
    sub.required = False

    from . import models_export as _models_mod
    _models_mod.add_subparser(sub)

    from . import release as _release_mod
    _release_mod.add_subparser(sub)

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
    """Entry point for the ``gen-worker`` console_script."""
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
