"""Shared test infrastructure for the th#960/pgw#609 greenfield suite (P1-P10).

The only double in this package is ``hub_double`` — it stands at the true
process boundary pgw does not own (the hub scheduler's gRPC service). Every
other module here (``blob_host``, ``toy_endpoints``, ``subprocess_runner``)
drives REAL gen_worker code paths: real TCP, real blake3-verified downloads,
real subprocess boot. See ~/cozy/cozy-creator-tracker/TEST-SYSTEM-DESIGN.md.
"""

from __future__ import annotations
