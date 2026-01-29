from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from gen_worker.hf_selection import HFSelectionPolicy, finalize_diffusers_download, plan_diffusers_download


def _load_fixture(name: str) -> dict[str, Any]:
    p = Path(__file__).parent / "fixtures" / "hf_selection" / name
    return json.loads(p.read_text("utf-8"))


@pytest.mark.parametrize(
    "fixture_name",
    [
        "sd15_minimal_fp16.json",
        "sd15_sharded_unet_fp16.json",
    ],
)
def test_hf_selection_fixtures(fixture_name: str) -> None:
    fx = _load_fixture(fixture_name)

    policy = HFSelectionPolicy(**fx["policy"])
    repo_files = list(fx["repo_files"])
    plan = plan_diffusers_download(model_index=fx["model_index"], repo_files=repo_files, policy=policy)
    selected = finalize_diffusers_download(
        plan=plan,
        repo_files=repo_files,
        weight_index_json_by_file=fx["weight_index_json_by_file"],
    )

    assert selected == set(fx["expected_files"])

