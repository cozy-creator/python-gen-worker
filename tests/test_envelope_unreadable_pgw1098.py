"""An UNREADABLE cell envelope must refuse BY NAME, not vanish.

The failure mode: `artifact_meta.MAX_METADATA_BYTES` (16 MiB) refuses to READ a
36-entry envelope, `try_read_metadata` turns that refusal into `None`, and
`None` reads as "this cell states no facts" at two consecutive call sites, each
silently skipping the work it owed:

  1. `adopt_delegated_mint` skips the key-axis divergence check;
  2. `arm_aot` skips the lifted-binding install, because an envelope that states
     no targets names no module to install onto.

`aot_serve.enable`, reading the SAME member through a SECOND unbounded reader,
then finds the artifact declaring `lora_a`/`lora_b` against an unlifted module
and refuses `lifted_inputs_unbindable`. The gate that noticed gets named; the
read that failed does not — a 92-minute 36/36 compile publishes nothing and the
only trace of the real cause on the wire is the word `unreadable` in one event's
`compiled_graph_key=` field.
"""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path
from typing import Any, Dict

import pytest

from gen_worker import artifact_meta
from gen_worker.cell_adopt import AdoptOutcome


def _cell(path: Path, meta: Dict[str, Any], *, pad_to: int = 0) -> Path:
    """A tarball carrying `metadata.json` at the root, optionally padded so the
    member's DECLARED size crosses a bound. The padding is a real JSON field —
    an under-declaring header is a different threat and is already bounded."""
    if pad_to:
        meta = dict(meta)
        blob = json.dumps(meta).encode()
        meta["_pad"] = "x" * max(0, pad_to - len(blob) - 16)
    payload = json.dumps(meta).encode()
    with tarfile.open(path, mode="w:gz") as tar:
        info = tarfile.TarInfo(artifact_meta.METADATA_NAME)
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))
    return path


#: The shape row 7's envelope had: entries whose `target` is the denoiser, and
#: a stamped key. Everything the two skipped gates needed was right here.
#:
#: DELIBERATELY PRE-pgw#1176, and left that way: `format: 2`, an `entries` MAP
#: and a `ck1` key. This fixture's job is to reproduce a REAL artifact from the
#: row 7 incident so the metadata SIZE bound is measured against bytes that
#: actually existed. Migrating it to a one-entry `cg-key-v1` envelope would shrink
#: the very thing under test — a 36-entry cell is what made the envelope
#: exceed 16 MiB — and would assert the bound against a shape that never
#: overflowed it.
_ROW7_META: Dict[str, Any] = {
    "format": 2,
    "kind": "aot-inductor",
    "family": "sdxl",
    "compiled_graph_key": "ck1-" + "a" * 56,
    "lora_bucket": 64,
    "entries": {
        "unet/adapter=true,cfg=true/B=2,H_lat=128,T_txt=77,W_lat=128": {
            "target": "unet",
        },
    },
}


# ---------------------------------------------------------------------------
# 1. The bound itself: a real sdxl envelope must be READABLE.
# ---------------------------------------------------------------------------


def test_a_36_entry_sdxl_scale_envelope_is_readable(tmp_path: Path) -> None:
    """RED pre-fix: 16 MiB refused row 7's envelope.

    The size is not invented. `fleet_cells._UNBOUNDED_ENVELOPE_BLOCKS` records
    a MEASURED sdxl cell whose metadata is 13,377,167 bytes on a 69 MB
    artifact, and states that it grows with the artifact. Row 7's artifact was
    ~141 MB with 36 AOT entries carrying per-class contracts and constant
    manifests, so its envelope sits above 16 MiB — which is why a 20 MiB
    envelope is the honest regression vehicle here.
    """
    artifact = _cell(tmp_path / "cell.tar.gz", _ROW7_META, pad_to=20 << 20)

    meta = artifact_meta.read_metadata(artifact)

    assert meta["compiled_graph_key"] == _ROW7_META["compiled_graph_key"]
    assert meta["entries"]["unet/adapter=true,cfg=true/B=2,H_lat=128,"
                           "T_txt=77,W_lat=128"]["target"] == "unet"


def test_the_bound_is_a_memory_bound_not_the_declare_bound() -> None:
    """The derivation, pinned. `_UNBOUNDED_ENVELOPE_BLOCKS` are STRIPPED from
    the declare precisely because they belong in the artifact — so sizing the
    artifact-plane read off `CELL_DECLARE_MAX_BYTES` refuses the shape the
    design requires. Any future edit that re-couples them fails here."""
    from gen_worker import fleet_cells

    assert artifact_meta.MAX_METADATA_BYTES >= 16 * fleet_cells.CELL_DECLARE_MAX_BYTES
    # Still bounded, and still well under a decompression bomb's scale:
    # pgw#1013's OOM threat is real and must not be reopened.
    assert artifact_meta.MAX_METADATA_BYTES < (128 << 20)


def test_an_oversized_envelope_still_refuses_before_decompressing(
    tmp_path: Path,
) -> None:
    """The threat pgw#1013 closed stays closed, and names its bound."""
    artifact = _cell(
        tmp_path / "huge.tar.gz", _ROW7_META,
        pad_to=artifact_meta.MAX_METADATA_BYTES + (1 << 20))

    with pytest.raises(artifact_meta.ArtifactMetadataError) as excinfo:
        artifact_meta.read_metadata(artifact)

    assert str(artifact_meta.MAX_METADATA_BYTES) in str(excinfo.value)


# ---------------------------------------------------------------------------
# 2. One reader. The asymmetry is what made the failure invisible.
# ---------------------------------------------------------------------------


def test_there_is_ONE_envelope_reader_and_it_is_BOUNDED(tmp_path: Path) -> None:
    """RED pre-fix: `aot_serve.unpack_metadata` kept its own UNBOUNDED scan, so
    on row 7's cell the bounded reader refused and this one succeeded. That
    disagreement is the whole mechanism — `arm_aot` got `None` and skipped its
    install while `enable`, reading through the second reader, saw a lifted
    artifact and refused it.

    pgw#1270 removed the second reader outright: TCG's Engine owns artifact
    import and this repo has one bounded envelope reader left. So the property
    is asserted the strong way — the reader is bounded in both directions, and
    `aot_serve` exposes no reader of its own for the asymmetry to come back on.
    """
    from gen_worker import aot_serve

    artifact = _cell(tmp_path / "cell.tar.gz", _ROW7_META, pad_to=20 << 20)
    assert artifact_meta.read_metadata(artifact)["family"] == _ROW7_META["family"]

    over = _cell(
        tmp_path / "over.tar.gz", _ROW7_META,
        pad_to=artifact_meta.MAX_METADATA_BYTES + (1 << 20))
    # It refuses, and it does not answer "there are no facts here".
    with pytest.raises(artifact_meta.ArtifactMetadataError):
        artifact_meta.read_metadata(over)

    assert not hasattr(aot_serve, "unpack_metadata"), (
        "a second envelope reader is back; pgw#1098 is that asymmetry")


# ---------------------------------------------------------------------------
# 3. Unreadable is not absent — the two silent skips.
# ---------------------------------------------------------------------------


class _Cfg:
    family = "sdxl"
    lora_bucket = 64
    targets = ("unet",)


def test_arm_aot_refuses_a_declared_bucket_it_cannot_resolve_a_target_for(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RED pre-fix: this returned `lifted_inputs_unbindable` with NO root.

    An unreadable envelope names no targets, so `module_name` was `""`,
    `lifted_target` was `None`, and the install branch was simply not entered
    — no exception, therefore no `lifted_install_error`, therefore a refusal
    that blames the downstream contract gate. Row 7's event, verbatim.
    """
    from gen_worker import aot_serve
    from gen_worker.models import lora_lifted, provision

    class _Pipe:
        def __init__(self) -> None:
            self.unet = object()

    monkeypatch.setattr(provision, "arm_route", lambda mode: object())
    monkeypatch.setattr(lora_lifted, "branch_targets", lambda p: {"unet": p.unet})
    monkeypatch.setattr(
        aot_serve, "enable",
        lambda *a, **k: AdoptOutcome.miss(
            "lifted_inputs_unbindable",
            "artifact declares lifted adapter input(s) ['lora_a', 'lora_b'] "
            "but the module has no lifted binding to supply them"))

    artifact = tmp_path / "cell.tar.gz"
    artifact.write_bytes(b"not a tarball")   # => metadata unreadable, meta=None

    outcome = provision.arm_aot(_Pipe(), _Cfg(), None, artifact, 64, None)

    assert not outcome.armed
    # The gate that noticed is still named — it really did refuse...
    assert outcome.reason == "lifted_inputs_unbindable"
    # ...but the refusal now carries WHY the binding was never installed.
    assert "root:" in outcome.detail
    assert "no lifted target resolved" in outcome.detail
    assert "unreadable" in outcome.detail


def test_adopt_delegated_mint_refuses_an_unreadable_envelope_by_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RED pre-fix: `meta=None` flowed past the pgw#1042 divergence check into
    an arm that could not succeed, and the wire blamed the LoRA contract.

    The refusal must (a) be its own class, (b) come BEFORE any arm, and
    (c) carry the reader's own message, so the next reader of this event knows
    a bound refused an envelope rather than that a cell was malformed."""
    from gen_worker import fleet_cells

    armed_calls: list = []
    monkeypatch.setattr(
        fleet_cells.provision, "arm_aot",
        lambda *a, **k: armed_calls.append(a) or AdoptOutcome.miss("x", "y"))

    target = tmp_path / "adopted.tar.gz"
    mint_root = tmp_path / "mint-root"
    mint_root.mkdir()
    produced = tmp_path / "cell.tar.gz"
    # Over the bound: readable bytes, refused envelope. The distinction the
    # pre-fix tree could not express.
    _cell(produced, _ROW7_META,
          pad_to=artifact_meta.MAX_METADATA_BYTES + (1 << 20))

    pending = fleet_cells.PendingSelfMint(
        family="sdxl", arm_token="arm1-" + "b" * 40,
        ref="repo#arm1", cfg=_Cfg(), target=target, mint_root=mint_root,
        publisher=None, cache_dir=tmp_path / "cache", arm_key=None)

    minted = fleet_cells.adopt_delegated_mint(object(), pending, [produced])

    assert minted is None
    reason, why = fleet_cells.adopt_refusal(pending)
    assert reason == "compiled_graph_envelope_unreadable"
    assert artifact_meta.METADATA_NAME in why
    # (b): refused BEFORE the arm, so no gate downstream of the read can be
    # blamed for a fact it was never given.
    assert armed_calls == []
