"""The ONE meta-instantiation seam, owned and proven in-process.

Weight-free instantiation is not an optimization — it is step 1 of every
boot-time adopt (§4.27) and of the zero-download forge. The key a pod derives,
and therefore whether it can ASK the hub for a cell at all, begins with
building a compile target from code + config alone.

WHY THIS IS NOT ``accelerate.init_empty_weights``
-------------------------------------------------
Importing ``accelerate`` without anything declaring it means an endpoint that
deliberately ships no diffusers stack cannot derive a key, never issues a
resolve, and self-mints forever.

Declaring the dependency instead is rejected on two facts:

* ``accelerate`` hard-requires ``torch>=2.0.0``, and this package's base wheel
  deliberately carries no torch ("the base wheel stays lean for plain-Python
  (API-proxy) endpoints", ``pyproject.toml``). A core dependency would put a
  multi-GB torch behind ``pip install gen-worker``, and into the exported
  requirement set every endpoint Dockerfile installs over its base image's own
  torch. Putting it in the ``torch`` extra costs nothing but reaches nobody:
  no fleet family requests that extra.
* What we actually need is ~20 lines with a contract this tree already states
  as a REQUIREMENT — parameters on meta, **buffers real** — and the upstream
  version of it reads ``ACCELERATE_INIT_INCLUDE_BUFFERS`` from the ambient
  environment. An ambient third-party env var that can flip a structure
  invariant a cell key is derived from is not a dependency, it is a hazard.

``accelerate`` remains declared in the ``torch`` extra for the real-weight
quantized loaders (:mod:`.w8a8`, :mod:`.w4a4`, :mod:`.svdq_native`) that build
their skeletons with it. Those run only on artifacts that only exist in
families which ship it; the boot-key path no longer depends on any of that.

WHY THE CAPABILITY IS PROVEN AND NOT ASSUMED
--------------------------------------------
:func:`require_meta_init` builds a probe module and checks the invariant on the
result. A meta-init that silently stopped moving parameters would trace real
weights; one that started moving BUFFERS would make ``aot_package``'s literal
constants unpackable, much later and much less legibly. Either way the answer
is a typed :class:`MetaInitUnavailable` naming the capability, which
``structure_only`` turns into a refusal a boot-adopt event can distinguish from
a family that is genuinely stranded.
"""

from __future__ import annotations

import contextlib
from typing import Any, Iterator

from ..api.errors import WorkerError

#: The capability this module provides, named so a refusal can say what is
#: missing rather than that "something" is.
CAPABILITY = "meta-instantiation (parameters on meta, buffers real)"


class MetaInitUnavailable(WorkerError):
    """This PROCESS cannot meta-instantiate — a broken image, never normal.

    Distinct in kind from "this family has no structure-only build": that is a
    correct, permanent property of some trees, while this is an install that
    cannot do what every install is expected to do.
    """

    def __init__(self, *, capability: str, lacks: str) -> None:
        self.capability = str(capability or CAPABILITY)
        self.lacks = str(lacks or "")
        super().__init__(
            f"this process cannot provide {self.capability}: {self.lacks}. "
            f"Weight-free instantiation is how a compile target is built from "
            f"CODE + CONFIG, so without it no boot key can be derived, no "
            f"cell can be asked for, and this pod will self-mint on every "
            f"boot")


@contextlib.contextmanager
def init_empty_weights() -> Iterator[None]:
    """Instantiate modules with their PARAMETERS on ``meta`` and no storage.

    Buffers are left alone on purpose — they are config-derived, KB-to-MB
    scale, and they are what a literal-bearing cell packs.
    """
    import torch
    from torch import nn

    original = nn.Module.register_parameter
    meta = torch.device("meta")

    def register_empty_parameter(
        module: Any, name: str, param: Any,
    ) -> None:
        original(module, name, param)
        held = module._parameters.get(name)
        if held is None:
            return
        cls = type(held)
        kwargs = dict(held.__dict__)
        kwargs["requires_grad"] = held.requires_grad
        module._parameters[name] = cls(held.to(meta), **kwargs)

    setattr(nn.Module, "register_parameter", register_empty_parameter)
    try:
        yield
    finally:
        setattr(nn.Module, "register_parameter", original)


def require_meta_init() -> None:
    """Prove the capability on THIS process, or refuse naming it."""
    try:
        import torch
        from torch import nn
    except Exception as exc:  # noqa: BLE001
        raise MetaInitUnavailable(
            capability=CAPABILITY,
            lacks=f"`torch` is not importable ({exc})") from exc

    class _Probe(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.empty(2, 2))
            self.register_buffer("table", torch.zeros(2))

    try:
        with init_empty_weights():
            probe = _Probe()
    except Exception as exc:  # noqa: BLE001
        raise MetaInitUnavailable(
            capability=CAPABILITY,
            lacks=f"building a probe module refused ({exc!r})") from exc

    if probe.weight.device.type != "meta":
        raise MetaInitUnavailable(
            capability=CAPABILITY,
            lacks=(f"a parameter built under the context manager landed on "
                   f"{probe.weight.device!r}, not on meta — this build would "
                   f"hold real weights"))
    buffer = probe.get_buffer("table")
    if buffer.device.type == "meta":
        raise MetaInitUnavailable(
            capability=CAPABILITY,
            lacks=("a BUFFER built under the context manager landed on meta; "
                   "buffers are config-derived values a literal-bearing cell "
                   "packs, and a fake one makes the package unbuildable"))


__all__ = [
    "CAPABILITY", "MetaInitUnavailable", "init_empty_weights",
    "require_meta_init",
]
