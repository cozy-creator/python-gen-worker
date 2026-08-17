"""The tensorhub grant wire: opaque presigned URLs in, CAS objects out.

This was vendored from tensorfs and is now FIRST-PARTY, and the move is a
correction rather than a convenience. `TransferGrant.from_wire` parses
tensorhub's grant JSON; the retry ladder, the parallel fan-out and the
per-object progress emission are this worker's policy about this worker's
uplink. None of it is storage, and upstream tensorfs agrees -- its own
transfer plane is now the Rust `sync.rs` pack/blob client, which a pure-Python
wheel cannot carry (pgw#1310). So the two modules were pinned to a lineage
upstream had ABANDONED, and a digest fence over them certified fidelity to a
rev nobody would ever fix.

The split is by ownership, not by expedience:

* **tensorfs** owns the object store, the chunk manifest, the projected
  snapshot tree and the direct tensor reader -- everything true about bytes at
  rest;
* **this package** owns how those bytes arrive from tensorhub:
  :mod:`~gen_worker.transfer.grants` (the wire) and
  :mod:`~gen_worker.transfer.journal` (resuming an interrupted publish).

The pgw#1287 fix lives here now: progress is emitted as EACH object lands, not
after the batch drains. A 31.6 GB snapshot reporting `0 / total` for its whole
duration is what got correctly-downloading pods condemned by the hub's
6-minute freshness window, and a test asserts it by making the second object
wait on the first object's callback.

**This module deliberately re-exports NOTHING.** A convenience facade here
would not be free: `scripts/lint_unreached_surface.py` resolves a use to its
DEFINING module, so surface reached only through a re-export reads to that
guard as dead code. Measured, not assumed -- adding the facade made
`TransferJournal` report as unreached while `hubio/client.py` was constructing
one. Import from `.grants` or `.journal`.
"""

from __future__ import annotations

__all__: list[str] = []
