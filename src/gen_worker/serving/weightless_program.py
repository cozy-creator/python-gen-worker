"""Load a WEIGHTS-FREE exported program — the one the derive actually writes.

pgw#1468, closing the gap pgw#1465 opened. The derive publishes a graph blob
that carries structure and no weights; ``mint_child`` then calls
``torch.export.load`` on it. Those two statements are incompatible, and the
incompatibility is silent until a real model hits it.

**What the derive writes.** Saving a program whose parameters are FakeTensors
produces a pt2 archive with a full ``model_weights_config.json`` — shape, dtype,
device, strides, storage offset, per parameter — and a payload file of **zero
bytes**. torch's saver dedups payload files by storage identity, and every fake
tensor presents the same degenerate storage, so all of them collapse onto one
entry. Measured on sd1.5's UNet: 686 entries, every one ``is_param: true``,
every one ``path_name: "weight_0"``, every one ``storage_offset: 0``,
**1,719,041,928 bytes declared against a 0-byte payload**.

**Why loading it fails.** ``_load_state_dict`` maps every FQN onto that single
payload and sizes the storage from it, then ``as_strided``s each parameter into
it. The first parameter fits by construction; the rest are out of bounds::

    RuntimeError: setStorage: sizes [1280, 320], strides [320, 1], storage
    offset 0, and itemsize 2 requiring a storage size of 819200 are out of
    bounds for storage of size 23040

23040 B is ``conv_in.weight`` — 11,520 elements, the FIRST config entry — and
819200 B is a later, larger one.

**Why nobody caught it.** A program whose parameters are all the SAME shape
cannot go out of bounds, so it "loads". It is still wrong — every parameter
aliases one buffer — but nothing raises. Uniform-shape programs are exactly what
a hand-written round-trip check produces, which is why the pgw#1465 evidence
read green while no real model could load at all.

**The fix.** Rebuild the state dict from the metadata instead of from a payload
that was never written. The archive already states shape, dtype and device for
every parameter, which is the whole of what AOTI needs — a graph artifact
carries structure, and asking torch to materialize weights out of it is the
category error. Tensors come back zero-filled, on the recorded device, each with
its OWN storage.

This is deliberately narrow: it engages only when the payload is empty. An
archive that really carries weights is left to torch untouched, so nothing about
the weight-bearing path changes.
"""

from __future__ import annotations

import json
import os
from typing import Any, Callable, Mapping

#: Inside the pt2 archive, the directory holding weight payloads.
WEIGHTS_DIR = "weights"

#: The single payload every fake parameter dedups onto. Named because the
#: emptiness of THIS file is the whole signal.
SHARED_PAYLOAD = "weight_0"


def is_weightless(archive_reader: Any, model_name: str) -> bool:
    """True when this archive declares parameters but carries no payload bytes.

    The discriminator is the payload's SIZE, not the presence of the config or
    the count of entries: a weight-bearing archive has the same config shape and
    must keep torch's own path.
    """
    # Real attribute; torch just does not list it in `__all__`.
    from torch.export.pt2_archive._package import (  # type: ignore[attr-defined]
        WEIGHTS_CONFIG_FILENAME_FORMAT,
    )

    config_file = WEIGHTS_CONFIG_FILENAME_FORMAT.format(model_name)
    if config_file not in archive_reader.get_file_names():
        return False
    try:
        payload = archive_reader.read_bytes(os.path.join(WEIGHTS_DIR, SHARED_PAYLOAD))
    except Exception:  # noqa: BLE001 - an unreadable payload is an empty one
        return True
    return not payload


def _tensor_from_meta(tensor_meta: Mapping[str, Any]) -> Any:
    """One zero-filled tensor matching the recorded shape/dtype/device.

    Real storage rather than a fake tensor: a fake here would relocate the same
    "no bytes" problem into inductor, which is where it would be hardest to read.
    """
    import torch
    from torch._export.serde.serialize import _SERIALIZE_TO_TORCH_DTYPE

    dtype = _SERIALIZE_TO_TORCH_DTYPE[int(tensor_meta["dtype"])]
    sizes = [int(s["as_int"]) for s in tensor_meta["sizes"]]
    device_meta = tensor_meta.get("device") or {}
    device_type = str(device_meta.get("type") or "cpu")
    if device_type == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device(device_type, int(device_meta.get("index") or 0))
    return torch.zeros(sizes, dtype=dtype, device=device)


def state_dict_from_config(archive_reader: Any, model_name: str) -> dict[str, Any]:
    """The state dict this archive DESCRIBES, built from its own metadata."""
    import torch
    # Real attribute; torch just does not list it in `__all__`.
    from torch.export.pt2_archive._package import (  # type: ignore[attr-defined]
        WEIGHTS_CONFIG_FILENAME_FORMAT,
    )

    config_file = WEIGHTS_CONFIG_FILENAME_FORMAT.format(model_name)
    config = json.loads(archive_reader.read_bytes(config_file))["config"]

    rebuilt: dict[str, Any] = {}
    for fqn, entry in config.items():
        tensor = _tensor_from_meta(entry["tensor_meta"])
        # The export verifier checks parameter-ness BY TYPE, not by this flag,
        # so a plain tensor for an `is_param` entry passes every other step and
        # then fails `_verify_exported_program_signature` at the very end.
        if entry.get("is_param"):
            tensor = torch.nn.Parameter(tensor, requires_grad=False)
        rebuilt[fqn] = tensor
    return rebuilt


def install() -> None:
    """Teach ``torch.export.load`` to read a weights-free archive. Idempotent.

    A monkeypatch, and it should be read as a deliberate one: the alternative is
    reimplementing ``load_pt2`` to reach the same object, which would fork far
    more of torch's surface than this replaces. Scoped to the empty-payload case
    and delegating everything else to the original keeps the blast radius at
    "archives our own derive wrote".
    """
    from torch.export.pt2_archive import _package

    if getattr(_package._load_state_dict, "_pgw1468", False):
        return

    original: Callable[..., Any] = _package._load_state_dict

    def _load_state_dict(archive_reader: Any, model_name: str) -> Any:
        if not is_weightless(archive_reader, model_name):
            return original(archive_reader, model_name)
        return state_dict_from_config(archive_reader, model_name)

    # The marker IS the idempotency check, and tests read it to assert which
    # loader is in place.
    _load_state_dict._pgw1468 = True  # type: ignore[attr-defined]
    _package._load_state_dict = _load_state_dict


def uninstall() -> None:
    """Restore torch's own loader. Idempotent; the companion to ``install()``.

    pgw#1485: ``install()`` is PROCESS-WIDE, so any test asserting the stock
    path is really measuring whichever test ran before it. The suite's
    snapshot-and-restore fixture cannot fix that — it snapshots a state that is
    already wrong, so the red arm silently stops being red. That is exactly what
    happened when an unrelated change shifted the xdist distribution: two guards
    went from passing to failing with no code of theirs involved. A guard whose
    verdict depends on scheduling is not a guard.
    """
    from torch.export.pt2_archive import _package

    patched = _package._load_state_dict
    if not getattr(patched, "_pgw1468", False):
        return
    for slot in getattr(patched, "__closure__", None) or ():
        candidate = slot.cell_contents
        if callable(candidate) and not getattr(candidate, "_pgw1468", False):
            _package._load_state_dict = candidate
            return


__all__ = [
    "SHARED_PAYLOAD",
    "WEIGHTS_DIR",
    "install",
    "is_weightless",
    "state_dict_from_config",
    "uninstall",
]
