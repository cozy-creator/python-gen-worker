"""Enable ``python -m gen_worker.cli`` as an alias for the ``gen-worker``
console script."""

from __future__ import annotations

import sys

from . import main

if __name__ == "__main__":
    sys.exit(main())
