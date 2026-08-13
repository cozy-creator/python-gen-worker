"""pgw#1217: a CLIENT-SUPPLIED ref is a residency key, so it must be normalized
where it enters — or one model mints TWO residency identities.

`payload.source` (and `text_encoder`/`candidate`) and every `lora overlay.ref`
arrive as free-form strings from the job payload. Before this fix the executor
took them verbatim and used each for TWO things that both require normal form:

  1. `snapshots.get(ref)` — a lookup into the hub-resolved snapshot map, which
     is keyed by NORMALIZED refs (it is built from `binding_wire_refs` /
     `wire_ref`, i.e. `models.refs` normal form); and
  2. `store.ensure_local(ref, ...)` — the residency key itself.

So a non-normal spelling of the same model MISSED its snapshot and then minted
a second residency identity for the same weights: a redundant multi-GB
download, a second disk-GC entry, and two cache identities where the hub
believes there is one. That is the th#736 mechanic `api.binding.rebind_pick`'s
own docstring warns about — *"a pick the rebound binding cannot re-mint would
split the slot into two residency identities"*.

`acme/repo:prod` is the cheapest witness: `prod` is `DEFAULT_REF_TAG`, so the
grammar ELIDES it and both spellings are the same model by definition
(`models/refs.py::TensorhubRef.canonical`). Nothing about that is exotic — a
client that echoes back the tag it was given produces it.

REVERT-TURNS-RED: every test below fails on the parent commit — the recorded
residency key is `acme/repo:prod`, the snapshot lookup returns None, and the
two spellings produce two distinct keys.

Sibling context: gw#491 fixed exactly this invariant one level down, for the
DIGEST spelling, three lines from the adapter site here (*"one adapter must
never mint two cache identities"*). It left the REF spelling open; this closes
it.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

from gen_worker import dispatch
from gen_worker.api.errors import ValidationError
from gen_worker.executor import Executor
from gen_worker.pb import worker_scheduler_pb2 as pb

#: The same model, spelled two legitimate ways. `prod` is the grammar default,
#: so `canonical()` elides it — these are one model, not two.
NORMAL = "acme/repo"
NON_NORMAL = "acme/repo:prod"


class _Ctx:
    """Only the surface `_materialize_source` touches."""

    def __init__(self) -> None:
        self.source_path = ""

    def _set_source_path(self, p: str) -> None:
        self.source_path = p


def _executor(tmp_path: Path) -> Executor:
    async def _send(_msg: pb.WorkerMessage) -> None:
        return None

    return Executor([], _send)


def _recording_store(
    ex: Executor, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> Tuple[List[str], List[Optional[pb.Snapshot]]]:
    """Record the residency key and the snapshot each call actually resolved."""
    keys: List[str] = []
    snaps: List[Optional[pb.Snapshot]] = []

    async def _ensure_local(
        ref: str, snapshot: Optional[pb.Snapshot] = None, **_kw: Any,
    ) -> Path:
        keys.append(ref)
        snaps.append(snapshot)
        p = tmp_path / ref.replace("/", "_").replace(":", "_")
        p.mkdir(parents=True, exist_ok=True)
        return p

    monkeypatch.setattr(ex.store, "ensure_local", _ensure_local)
    return keys, snaps


def _snapshots() -> Dict[str, pb.Snapshot]:
    """The hub's map, keyed in NORMAL form — which is how it is really built."""
    return {
        NORMAL: pb.Snapshot(
            digest="blake3:" + "a" * 64,
            files=[pb.SnapshotFile(
                path="model.safetensors", size_bytes=5, blake3="cd" * 32,
                url="http://r2.invalid/p")],
        )
    }


# --- payload.source ---------------------------------------------------------


def test_a_non_normal_source_ref_resolves_to_the_normalized_residency_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """THE defect. `acme/repo:prod` must reach the store as `acme/repo`."""
    ex = _executor(tmp_path)
    keys, _snaps = _recording_store(ex, tmp_path, monkeypatch)

    asyncio.run(ex._materialize_source(
        _Ctx(), {"ref": NON_NORMAL}, _snapshots()))

    assert keys == [NORMAL], (
        f"the residency key must be normal form; got {keys!r}. A non-normal "
        f"spelling here mints a SECOND residency identity for one model."
    )


def test_a_non_normal_source_ref_still_finds_its_hub_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The consequence that costs money: the hub's snapshot map is keyed in
    normal form, so an unnormalized lookup MISSES and the worker re-downloads
    weights it was already handed the manifest for."""
    ex = _executor(tmp_path)
    _keys, snaps = _recording_store(ex, tmp_path, monkeypatch)

    asyncio.run(ex._materialize_source(
        _Ctx(), {"ref": NON_NORMAL}, _snapshots()))

    assert snaps and snaps[0] is not None, (
        "the snapshot lookup missed: the map is keyed in normal form, so an "
        "unnormalized ref re-downloads weights whose manifest is in hand"
    )
    assert snaps[0].digest == "blake3:" + "a" * 64


def test_both_spellings_of_one_model_are_ONE_residency_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The invariant, stated directly (th#736): one model, one identity."""
    ex = _executor(tmp_path)
    keys, _snaps = _recording_store(ex, tmp_path, monkeypatch)

    asyncio.run(ex._materialize_source(_Ctx(), {"ref": NORMAL}, _snapshots()))
    asyncio.run(ex._materialize_source(
        _Ctx(), {"ref": NON_NORMAL}, _snapshots()))

    assert len(set(keys)) == 1, (
        f"two spellings of ONE model produced {len(set(keys))} residency "
        f"identities: {keys!r}"
    )


def test_a_malformed_source_ref_is_a_named_refusal_not_a_bogus_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Normalizing at the boundary means the grammar is enforced there. A ref
    that cannot be parsed must be REFUSED by name — never carried on to become
    a residency key nothing can resolve."""
    ex = _executor(tmp_path)
    keys, _snaps = _recording_store(ex, tmp_path, monkeypatch)

    with pytest.raises(ValidationError) as exc:
        asyncio.run(ex._materialize_source(
            _Ctx(), {"ref": "no-owner-segment"}, _snapshots()))

    assert "source" in str(exc.value)
    assert keys == [], "a refused ref must never reach the store"


def test_the_refusal_names_the_field_it_came_from(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`_materialize_source` serves `source`, `text_encoder` and `candidate`;
    the refusal has to say which one, or an operator cannot find it."""
    ex = _executor(tmp_path)
    _recording_store(ex, tmp_path, monkeypatch)

    with pytest.raises(ValidationError) as exc:
        asyncio.run(ex._materialize_source(
            _Ctx(), {"ref": "no-owner-segment"}, _snapshots(),
            field_name="text_encoder"))

    assert "text_encoder" in str(exc.value)


# --- lora overlay.ref -------------------------------------------------------


def _spec_with_slot() -> Any:
    class _Spec:
        models = {"pipeline": object()}
    return _Spec()


def test_a_non_normal_lora_ref_resolves_to_the_normalized_residency_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The second inlet, same defect. gw#491 fixed the DIGEST spelling three
    lines from here and left the REF spelling open."""
    ex = _executor(tmp_path)
    keys, snaps = _recording_store(ex, tmp_path, monkeypatch)

    monkeypatch.setattr(
        "gen_worker.utils.lora.parse_adapter",
        lambda *_a, **_kw: {}, raising=False)

    async def _go() -> None:
        await ex._prepare_adapters(
            {"pipeline": (dispatch.AdapterOrder(ref=NON_NORMAL, weight=1.0),)},
            _spec_with_slot(),
            _snapshots(),
        )

    try:
        asyncio.run(_go())
    except Exception:
        # Parsing a real adapter is not what this asserts; the residency key is
        # recorded before any parse can fail.
        pass

    assert keys == [NORMAL], (
        f"the lora residency key must be normal form; got {keys!r}"
    )
    assert snaps and snaps[0] is not None, (
        "the lora snapshot lookup missed for a non-normal spelling"
    )


def test_a_malformed_lora_ref_is_a_named_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    ex = _executor(tmp_path)
    keys, _snaps = _recording_store(ex, tmp_path, monkeypatch)

    async def _go() -> None:
        await ex._prepare_adapters(
            {"pipeline": (dispatch.AdapterOrder(ref="no-owner-segment"),)},
            _spec_with_slot(),
            _snapshots(),
        )

    with pytest.raises(ValidationError) as exc:
        asyncio.run(_go())

    assert "lora" in str(exc.value).lower()
    assert keys == [], "a refused lora ref must never reach the store"
