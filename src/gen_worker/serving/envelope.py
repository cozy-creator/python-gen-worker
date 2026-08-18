"""The request envelope: SIGNATURE-DERIVED, decoded per entrypoint (pgw#1372).

Paul's envelope ruling: the checkpoint pick leaves the payload entirely —
the platform wire shape is derived from the entrypoint's own signature,
never declared twice::

    {"model": "org/ckpt@3",                    # ONE model slot
     "adapters": {"turbo": {...row...},        # per adapter-slot picks,
                  "loras": [{...}, {...}]},    #   keyed by PARAMETER NAME
     "input": {...the payload struct...}}

    {"models": {"base": "...", "refiner": "..."},   # >1 model slots:
     "input": {...}}                                #   keyed by slot name

``input`` is the entrypoint's payload schema (typed 400s stay msgspec's).
Adapter values are the HUB'S resolution output riding the request — fully
pinned refs, local paths, decoded defaults, platform-clamped scale; the
worker never resolves picks itself. A ``DistillationAdapter`` slot only
accepts a distillation-marked row (the typed takeover guard, enforced here
again even though the hub refuses first — a misdeploy must not seize the
serving config).

Everything unknown refuses loudly: an unknown top-level key, a pick for an
undeclared slot, ``"model"`` against a multi-model signature (or
``"models"`` against a single-model one), a list where the slot is single.
Silence is never an outcome — a dropped pick would serve the wrong bytes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import msgspec

from .context import Adapter, DistillationAdapter
from .entrypoints import EntrypointSpec, SlotSpec


class EnvelopeError(ValueError):
    """The request envelope does not fit the entrypoint's signature."""


class AdapterRow(msgspec.Struct, forbid_unknown_fields=True):
    """One hub-resolved adapter as it rides the wire.

    ``distillation`` is the hub's kind marker (release metadata records
    which slots demand it); ``defaults`` is the decoded overlay row the
    model type's schema gives meaning to."""

    ref: str
    path: str
    name: str = ""
    defaults: Optional[Dict[str, Any]] = None
    scale: float = 1.0
    distillation: bool = False

    def materialize(self, defaults_schema: Optional[type]) -> Adapter:
        """Build the typed slot value. ``defaults_schema`` is the model
        type's adapter-defaults struct (``SDXL.Lora.Defaults`` — pgw#1377's
        adapter-of-base scoping); the wire row's overlay decodes into it,
        an absent overlay takes the schema's zero-arg servable values, and
        an ill-typed one refuses naming the adapter."""
        cls = DistillationAdapter if self.distillation else Adapter
        defaults: Any = self.defaults
        if defaults_schema is not None:
            try:
                defaults = (
                    msgspec.convert(self.defaults, type=defaults_schema)
                    if self.defaults is not None
                    else defaults_schema()
                )
            except msgspec.ValidationError as exc:
                raise EnvelopeError(
                    f"adapter {self.ref!r} defaults do not fit "
                    f"{defaults_schema.__name__}: {exc}"
                ) from None
        return cls(
            name=self.name or self.ref,
            path=Path(self.path),
            defaults=defaults,
            scale=self.scale,
            ref=self.ref,
        )


class _Envelope(msgspec.Struct, forbid_unknown_fields=True):
    """The raw wire shape; slot-level validity is per-entrypoint, below."""

    input: msgspec.Raw
    model: Optional[str] = None
    models: Optional[Dict[str, str]] = None
    adapters: Optional[Dict[str, msgspec.Raw]] = None


@dataclass(frozen=True, slots=True)
class DecodedRequest:
    """One envelope, decoded against one entrypoint's signature."""

    payload: Any
    #: Checkpoint pick per MODEL slot name (every declared slot present —
    #: deploy defaults are the caller's to fill before decode refuses).
    model_picks: Tuple[Tuple[str, str], ...]
    #: Materialized value per ADAPTER slot name: Adapter | None for single
    #: slots, list[Adapter] for plural ones.
    adapter_values: Tuple[Tuple[str, Any], ...]


def _decode_adapter_value(
    slot: SlotSpec, raw: msgspec.Raw, defaults_schema: Optional[type]
) -> Any:
    if slot.kind == "adapters":
        try:
            rows = msgspec.json.decode(raw, type=List[AdapterRow])
        except msgspec.ValidationError as exc:
            raise EnvelopeError(
                f"adapters[{slot.name!r}] must be a list of adapter rows: {exc}"
            ) from None
        return [row.materialize(defaults_schema) for row in rows]
    try:
        row = msgspec.json.decode(raw, type=Optional[AdapterRow])
    except msgspec.ValidationError as exc:
        raise EnvelopeError(
            f"adapters[{slot.name!r}] must be one adapter row: {exc}"
        ) from None
    if row is None:
        if slot.required:
            raise EnvelopeError(
                f"adapter slot {slot.name!r} is required and the envelope "
                f"sends null"
            )
        return None
    value = row.materialize(defaults_schema)
    if slot.annotation is DistillationAdapter and not isinstance(
        value, DistillationAdapter
    ):
        # The typed takeover guard, worker-side: a style LoRA cannot seize a
        # distillation slot even when a misconfigured hub let it through.
        raise EnvelopeError(
            f"adapter slot {slot.name!r} takes a distillation adapter; "
            f"{row.ref!r} is not distillation-marked (a misdeploy the hub "
            f"should have refused)"
        )
    return value


def decode_envelope(
    spec: EntrypointSpec,
    envelope: Any,
    *,
    default_picks: Optional[Mapping[str, str]] = None,
    adapter_defaults_schema: Optional[type] = None,
) -> DecodedRequest:
    """Decode one wire envelope against one entrypoint's signature.

    ``default_picks`` are the DEPLOYMENT's per-slot checkpoint defaults
    (slot name -> ref); an envelope that names no pick for a slot falls back
    there, and a slot with neither is a typed refusal — the worker never
    guesses which bytes to serve.
    """

    if isinstance(envelope, (bytes, bytearray)):
        raw_bytes = bytes(envelope)
    else:
        raw_bytes = msgspec.json.encode(envelope)
    try:
        parsed = msgspec.json.decode(raw_bytes, type=_Envelope)
    except msgspec.ValidationError as exc:
        raise EnvelopeError(f"malformed request envelope: {exc}") from None

    model_slots = dict(spec.model_params)
    defaults = dict(default_picks or {})

    # -- model picks: "model" xor "models", derived from the signature ------
    if parsed.model is not None and parsed.models is not None:
        raise EnvelopeError('an envelope carries "model" or "models", never both')
    picks: Dict[str, str] = {}
    if not model_slots:
        # pgw#1392: a WEIGHTLESS entrypoint. Its envelope has NO model field
        # — not a null one, not an empty one — so a pick is a typed refusal
        # naming the real reason instead of advising the other spelling.
        if parsed.model is not None or parsed.models is not None:
            raise EnvelopeError(
                f"{spec.name!r} declares no model slot; its envelope carries "
                f'no "model"/"models" key at all (nothing is resident for a '
                f"weightless entrypoint)"
            )
    elif len(model_slots) == 1:
        (only_slot,) = model_slots
        if parsed.models is not None:
            raise EnvelopeError(
                f'{spec.name!r} declares one model slot ({only_slot!r}); the '
                f'envelope key is "model", not "models"'
            )
        if parsed.model is not None:
            picks[only_slot] = parsed.model
    else:
        if parsed.model is not None:
            raise EnvelopeError(
                f'{spec.name!r} declares {len(model_slots)} model slots '
                f'({sorted(model_slots)}); the envelope key is "models" '
                f'keyed by slot name, not "model"'
            )
        for slot_name, ref in (parsed.models or {}).items():
            if slot_name not in model_slots:
                raise EnvelopeError(
                    f'"models" names unknown slot {slot_name!r} '
                    f"(declared: {sorted(model_slots)})"
                )
            picks[slot_name] = ref
    for slot_name in model_slots:
        if not picks.get(slot_name):
            fallback = defaults.get(slot_name, "")
            if not fallback:
                raise EnvelopeError(
                    f"model slot {slot_name!r} has no envelope pick and no "
                    f"deployment default; the worker never guesses which "
                    f"bytes to serve"
                )
            picks[slot_name] = fallback

    # -- adapter picks: keyed by slot name, declared slots only -------------
    adapter_slots = {slot.name: slot for slot in spec.slots if slot.kind != "model"}
    sent = dict(parsed.adapters or {})
    unknown = sorted(set(sent) - set(adapter_slots))
    if unknown:
        raise EnvelopeError(
            f'"adapters" names undeclared slot(s) {unknown} '
            f"(declared: {sorted(adapter_slots)})"
        )
    values: Dict[str, Any] = {}
    for name, slot in adapter_slots.items():
        if name in sent:
            values[name] = _decode_adapter_value(
                slot, sent[name], adapter_defaults_schema)
        elif slot.kind == "adapters":
            values[name] = []
        elif slot.required:
            raise EnvelopeError(
                f"adapter slot {name!r} is required and the envelope sends "
                f"no pick for it"
            )
        else:
            values[name] = None

    # -- the payload, last: its schema is the entrypoint's own --------------
    payload = msgspec.json.decode(parsed.input, type=spec.payload_type)

    return DecodedRequest(
        payload=payload,
        model_picks=tuple(sorted(picks.items())),
        adapter_values=tuple(sorted(values.items())),
    )


__all__ = ["AdapterRow", "DecodedRequest", "EnvelopeError", "decode_envelope"]
