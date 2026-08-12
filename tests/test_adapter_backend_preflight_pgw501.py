"""pgw#501 — an endpoint that declares adapter serving proves it can, AT BUILD.

The measured failure this closes (qwen-image-edit BYOM serve, paid A100): the
image booted, served base edits fine, and died on the FIRST `_models` overlay
with *"adapter failed to load onto base pipeline: PEFT backend is required for
this method."* `peft` is not a gen-worker dependency and never should be — an
endpoint that serves adapters declares that in ITS image. What was missing is
the platform PROVING the declared capability exists, fail-closed and by name,
before a pod is ever bought.

`lora_bucket > 0` is the declaration (`allow_lora` has not existed since
pgw#523). These rows ride the same `aot_preconditions` block
`discovery.validation` already turns into a build error, so a refusal stops
the build rather than shipping an endpoint that dies on request one.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gen_worker import aot_preconditions as pre
from gen_worker.discovery.validation import validate_endpoint_lock

FAMILY = "sdxl-501"


def _lock(rows) -> dict:
    return {"functions": [], "aot_preconditions": [r.manifest_row() for r in rows]}


def test_an_adapter_endpoint_without_peft_REFUSES_AT_BUILD() -> None:
    """RED before pgw#501: this build passed and the pod died later."""
    rows = pre.adapter_backend_preconditions({FAMILY: 64}, present=False)

    (row,) = rows
    assert row.check == pre.CHECK_ADAPTER_BACKEND
    assert row.verdict == pre.REFUSED
    assert row.family == FAMILY
    # named, and it names the FIX — an author reading this must not have to
    # go and find out what a "PEFT backend" is.
    assert "peft" in row.detail and "lora_bucket=64" in row.detail
    assert "dependencies" in row.detail

    result = validate_endpoint_lock(_lock(rows))
    assert result.ok is False
    assert any("adapter_backend" in e and "peft" in e for e in result.errors)


def test_the_same_endpoint_WITH_peft_passes() -> None:
    rows = pre.adapter_backend_preconditions({FAMILY: 64}, present=True)

    (row,) = rows
    assert row.verdict == pre.OK
    result = validate_endpoint_lock(_lock(rows))
    assert result.ok is True
    assert not result.errors


def test_an_endpoint_that_declares_NO_bucket_owes_nothing() -> None:
    """The capability is owed by the DECLARATION, not by every image. A
    base-only endpoint must not be forced to carry an adapter backend."""
    assert pre.adapter_backend_preconditions({FAMILY: 0}, present=False) == ()
    assert pre.adapter_backend_preconditions({}, present=False) == ()


def test_one_row_per_declaring_family_and_the_bucket_is_reported() -> None:
    rows = pre.adapter_backend_preconditions(
        {"a": 32, "b": 0, "c": 128}, present=False)

    assert [r.family for r in rows] == ["a", "c"]
    assert "lora_bucket=32" in rows[0].detail
    assert "lora_bucket=128" in rows[1].detail


def test_the_probe_is_find_spec_not_an_import() -> None:
    """Discovery must not pay a heavy import to answer a yes/no, and `peft` is
    deliberately NOT a `heavy_deps` stubbed root, so the probe stays honest
    inside a torch-less manifest build (a stubbed root would answer `True` for
    a package that is not there)."""
    from gen_worker.discovery import heavy_deps

    assert pre.ADAPTER_BACKEND_DIST not in heavy_deps.DEFAULT_HEAVY_ROOTS
    assert isinstance(pre.adapter_backend_present(), bool)
    source = Path(pre.__file__).read_text()
    body = source[source.index("def adapter_backend_present"):]
    body = body[:body.index("def adapter_backend_preconditions")]
    code = body[body.index('"""', body.index('"""') + 3) + 3:]
    assert "find_spec" in code
    assert "import peft" not in code


def test_discovery_wires_the_check_and_not_only_the_AOT_ones() -> None:
    """An adapter-serving endpoint owes this whether or not it compiles, so
    the call must sit OUTSIDE `static_mint_preconditions`' export-declaration
    filter — a family with no registered declaration produces no AOT row at
    all, and used to produce no adapter row either."""
    from gen_worker.discovery import discover

    source = Path(discover.__file__).read_text()
    assert "adapter_backend_preconditions(" in source
    # and the declaration it keys on is the live one
    assert "allow_lora" not in source


@pytest.mark.parametrize("bucket", [1, 64, 128])
def test_any_positive_bucket_declares_the_capability(bucket: int) -> None:
    rows = pre.adapter_backend_preconditions({FAMILY: bucket}, present=False)
    assert len(rows) == 1 and rows[0].verdict == pre.REFUSED
