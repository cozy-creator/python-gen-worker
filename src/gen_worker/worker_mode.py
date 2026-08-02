"""th#1359 Part 2: the worker's OPERATING MODE — serve, or forge.

Paul's ratified mechanism, verbatim: *"the forge is just running endpoint code
on a datacenter machine we trust, and then having it compile as needed,
without serving traffic. It's sort of like, a 'mint only mode' for the
endpoint, as opposed to being generally available for traffic."*

So a forge worker is the SAME release image, booted with ``WORKER_MODE=forge``.
It is a MODE, not a fork — the identity axes a cell is keyed on (family, lane,
sm, cuda/torch/triton/gen_worker/diffusers/transformers, image_digest,
``env_seal``) are properties of the image and of the hub-resolved lane, and the
mode touches none of them. That is the whole reason the mode exists rather than
a dedicated forge image: **a forge-minted cell must be the same artifact a
serving pod would have produced** (pgw#846's gate — process may change, product
may not).

Load-bearing non-consequences, stated so nobody re-derives them wrong:

* **Not a sealed knob.** ``WORKER_MODE`` is deliberately absent from
  :mod:`gen_worker.env_seal`'s ``CANONICAL_CONFIG``. A sealed knob re-digests
  the ``env_seal`` key axis, which would give a forge-minted cell a key no
  serving pod can ever compute — i.e. it would break the single property the
  forge exists to have. Same reasoning as ``GEN_WORKER_PREFER_AOT``
  (``config.settings``: "must never join the sealed config table").
* **Not a graph knob.** Nothing here may reach the traced graph, kernel
  selection, shape specialization or layout. The forge changes WHERE a mint
  runs and WHAT ELSE is resident on the card, never what is compiled.

What the mode DOES change, and why (two blockers measured on real pods):

1. **No resident serving model.** pgw#846 attempts fourteen/fifteen died with
   the mint child and the serving pipeline both resident; the compile pool's
   width term is bound by free VRAM and came out K=1 on a host that could run
   127 CPU-side. A forge pod releases the serving instance before the mint
   child starts, so the pool gets the card.
2. **Never in the serving rotation.** pgw#846 attempt sixteen ran 29 minutes
   and was abandoned by a DRAIN — ``executor.shutdown_instances()`` abandons
   every background mint unconditionally, and its only caller is the drain
   path. A pod that receives no tenant dispatch is not a scale-down candidate
   and is not drained out from under its own mint.
"""

from __future__ import annotations

from .config import get_settings

#: The default. A normal serving worker; every existing behaviour.
MODE_SERVE = "serve"

#: Mint-only. Registers with the hub, receives no tenant dispatch, holds no
#: resident serving model, runs the mint(s) it was scheduled for, publishes,
#: and retires.
MODE_FORGE = "forge"

#: The env name the hub injects at pod launch. `WORKER_*` is the hub-injected
#: namespace (WORKER_ID / WORKER_JWT / WORKER_RELEASE_ID / WORKER_IMAGE_DIGEST
#: / WORKER_CONFIG_GENERATION), and this is the same kind of fact: what the
#: hub bought this pod for.
ENV_WORKER_MODE = "WORKER_MODE"

_KNOWN = (MODE_SERVE, MODE_FORGE)


def mode() -> str:
    """This worker's operating mode; :data:`MODE_SERVE` unless set to a known
    other value.

    An UNKNOWN spelling reads as ``serve`` rather than raising. A worker that
    refuses to boot because the hub sent it a mode from a newer vocabulary is
    a worker that turns a forward-compatibility gap into a dead pod; serving
    is the safe reading, and the mode is echoed on Hello so the hub can see
    the disagreement.
    """
    raw = str(getattr(get_settings(), "worker_mode", "") or "").strip().lower()
    return raw if raw in _KNOWN else MODE_SERVE


def is_forge() -> bool:
    """Whether this worker is a mint-only forge pod."""
    return mode() == MODE_FORGE


def declared() -> str:
    """The RAW declared mode, for reporting.

    :func:`mode` normalises an unknown value to ``serve``; Hello carries what
    was actually declared, so an unrecognised mode is visible at the hub as a
    disagreement instead of silently becoming a serving pod.
    """
    return str(getattr(get_settings(), "worker_mode", "") or "").strip().lower()
