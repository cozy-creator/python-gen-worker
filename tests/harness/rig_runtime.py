"""pgw#997: the rig's synthetic-runtime hook, as a MODULE of its own.

The micro-diffusion endpoint under ``examples/`` is a REAL org worker and must
carry no rig-only code — a fixture that edits the thing it measures is not a
fixture. But the mint child is a fresh interpreter whose only hook is
``MintRequest.modules``, and on a cardless box the seal needs an ``sm``
(pgw#983).

So the rig lists this module FIRST in ``modules`` and the endpoint SECOND.
Discovery imports both, finds its endpoints in the second, and the probe
patch has already happened. Nothing in ``micro_diffusion`` knows the rig
exists.
"""

from __future__ import annotations

from harness.tiny_diffusion import (
    SYNTHETIC_RUNTIME,
    SYNTHETIC_RUNTIME_ENV,
    install_synthetic_runtime_if_asked,
)

SYNTHETIC_RUNTIME_INSTALLED = install_synthetic_runtime_if_asked()

__all__ = [
    "SYNTHETIC_RUNTIME",
    "SYNTHETIC_RUNTIME_ENV",
    "SYNTHETIC_RUNTIME_INSTALLED",
]
