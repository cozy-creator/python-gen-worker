"""pgw#846 retirement semantics: regional compiled graphs are RETIRED, and a compiled graph whose
metadata still says ``mode='regional'`` is declined BY NAME — never handed to
the whole-graph arm (whose denoiser-scope bind table it cannot use, pgw#827)
and never silently defaulted. The pipeline stays eager.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from gen_worker.models import provision


def test_arm_route_serves_only_the_whole_graph_mode() -> None:
    assert provision.arm_route("") == "aot_serve.enable"
    assert provision.arm_route("regional") is None
    assert provision.arm_route("some-future-recipe") is None


@pytest.mark.parametrize("mode", ["regional", "some-future-recipe"])
def test_a_compiled_graph_whose_mode_has_no_arm_is_declined_by_name_and_stays_eager(
    monkeypatch: pytest.MonkeyPatch, mode: str,
) -> None:
    from gen_worker import aot_serve

    def _never(*_a: Any, **_k: Any) -> bool:  # pragma: no cover - the defect
        raise AssertionError(
            f"a mode={mode!r} compiled_graph must never reach the whole-graph arm")

    monkeypatch.setattr(aot_serve, "enable", _never)

    class _Pipe:
        pass

    outcome = provision.arm_aot(
        _Pipe(), object(), None, Path("/nonexistent/compiled_graph.tar.gz"),
        0, {"mode": mode})
    assert outcome.armed is False
    # The decline is BY NAME (pgw#827/pgw#923), not a bare False.
    assert outcome.reason == "no_arm_for_mode"
