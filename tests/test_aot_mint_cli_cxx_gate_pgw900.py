"""pgw#900: the AOTI mint operator CLI must gate on the C++ linker, not any C
compiler.

An AOTI mint links a real ``.so`` through inductor's C++ wrapper. A C-only
image (``cc``/``gcc`` present, no working C++) satisfies ``toolchain_present()``
— the dynamo-lane predicate — but the mint then loads the model, exports both
graph classes, and only discovers the missing C++ compiler 336 s later at the
linker (``InvalidCxxCompiler``, measured on a real L4 at 0.84.0). The CLI must
refuse at second zero on the SAME predicate the mint child already uses,
``cxx_toolchain_present``.

These tests drive ``aot_mint.main`` only as far as the toolchain gate — the
declaration load is stubbed and the request carries no model, so no compile is
ever reached (CPU-minimal / no-local-mint safe). RED on master: with a C
compiler present and no C++, master's ``toolchain_present`` gate PASSES and
``main`` falls through to the no-model BAD REQUEST (rc 3) instead of refusing.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from gen_worker import aot_mint


def _request(tmp_path: Path) -> Path:
    # A valid family so `_load_spec` succeeds, but NO source_ref and (below) no
    # --model, so a run that passes the toolchain gate stops at BAD REQUEST
    # (rc 3) BEFORE any model load or compile.
    req = tmp_path / "mint_request.json"
    req.write_text(json.dumps({"family": "sdxl"}))
    return req


@pytest.fixture(autouse=True)
def _stub_declaration(monkeypatch: pytest.MonkeyPatch) -> None:
    # The gate sits after `aot_declaration.load_declaration`; stub it so no real
    # family module (or compile) is needed to reach the gate.
    monkeypatch.setattr(
        "gen_worker.aot_declaration.load_declaration",
        lambda body, request_path=None: None,
        raising=True,
    )


def test_c_only_image_is_refused_at_the_gate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # C compiler present, C++ linker absent: the exact L4 shape.
    monkeypatch.setattr(aot_mint, "cxx_toolchain_present", lambda: False,
                        raising=False)
    monkeypatch.setattr(aot_mint, "toolchain_present", lambda: True,
                        raising=False)  # master's predicate — deliberately True

    rc = aot_mint.main([str(_request(tmp_path)), "--out", str(tmp_path / "out")])

    # Branch: refused at the gate (rc 2). Master: gate passes on the C-only
    # image, falls through to no-model BAD REQUEST (rc 3) — the fail-open.
    assert rc == 2


def test_full_cxx_toolchain_passes_the_gate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Control: a working C++ image must NOT be refused. It passes the gate and
    # stops at the no-model BAD REQUEST, proving the gate does not over-refuse.
    monkeypatch.setattr(aot_mint, "cxx_toolchain_present", lambda: True,
                        raising=False)
    monkeypatch.setattr(aot_mint, "toolchain_present", lambda: True,
                        raising=False)

    rc = aot_mint.main([str(_request(tmp_path)), "--out", str(tmp_path / "out")])

    assert rc == 3  # BAD REQUEST: no --model and no source_ref
