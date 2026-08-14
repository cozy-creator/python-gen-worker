"""th#1937 / th#1932 item 7 — ONE ``file_layout`` vocabulary, enforced here too.

Four spellings of one axis were live across two repositories and none was read
back.  tensorhub now refuses a dead spelling at DECLARE with
``file_layout_unknown_token``; these pin that this repo cannot emit one.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gen_worker.convert.clone import OutputSpec, normalize_outputs
from gen_worker.models.file_layout import (
    KNOWN_FILE_LAYOUTS,
    MULTI_FILE,
    NOT_APPLICABLE,
    SINGLE_FILE,
    validate_file_layout,
)


@pytest.mark.parametrize(
    "dead", ["singlefile", "diffusers", "multifile", "transformers", "single_file", "multi_file"]
)
def test_dead_spellings_are_refused_and_named(dead: str) -> None:
    """RED-VERIFIED: every pre-ruling spelling raises, and the message says what
    to write instead.  A validator that accepted everything would pass a test
    that only asserted the good tokens."""
    with pytest.raises(ValueError) as excinfo:
        validate_file_layout(dead)
    message = str(excinfo.value)
    assert "file_layout_unknown_token" in message
    assert SINGLE_FILE in message or MULTI_FILE in message


@pytest.mark.parametrize("live", [SINGLE_FILE, MULTI_FILE, NOT_APPLICABLE, "  "])
def test_ruled_tokens_and_the_absent_value_pass(live: str) -> None:
    assert validate_file_layout(live) in KNOWN_FILE_LAYOUTS | {NOT_APPLICABLE}


def test_normalize_outputs_speaks_only_the_ruled_vocabulary() -> None:
    specs = normalize_outputs(
        [{"dtype": "bf16", "file_layout": MULTI_FILE, "file_type": "safetensors"}]
    )
    assert specs == [OutputSpec(dtype="bf16", file_layout=MULTI_FILE, file_type="safetensors")]

    # The default when the caller states no layout is still the ruled token.
    assert normalize_outputs(None)[0].file_layout in KNOWN_FILE_LAYOUTS

    with pytest.raises(ValueError, match="unsupported output.file_layout"):
        normalize_outputs([{"dtype": "bf16", "file_layout": "diffusers"}])


def test_source_detection_returns_ruled_tokens(tmp_path: Path) -> None:
    from gen_worker.convert.source import _detect_file_layout

    (tmp_path / "model_index.json").write_text("{}", encoding="utf-8")
    assert _detect_file_layout(tmp_path) == MULTI_FILE

    flat = tmp_path / "flat"
    flat.mkdir()
    assert _detect_file_layout(flat) == SINGLE_FILE
