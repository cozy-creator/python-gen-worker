from __future__ import annotations

import hashlib
import json
from typing import Any

import pytest

from gen_worker import bind_contract
from gen_worker.serving.streaming.census import CensusMismatch, I5_TOTALITY


def _document() -> tuple[str, bytes]:
    raw = json.dumps(
        {
            "v": 1,
            "kind": "tensorhub.bind-contract@1",
            "identity": {
                "release_id": "release-a",
                "derive_image_digest": "sha256:image",
                "config_digest": "sha256:config",
            },
            "construction_census": {
                "v": 1,
                "kind": "gen-worker.construction-census@1",
                "pipeline_class": "TinyPipeline",
                "components": {
                    "unet": {
                        "component": "unet",
                        "class": "TinyUNet",
                        "tensors": [],
                        "eval_mode": True,
                    }
                },
            },
            "env_compile_stack": [],
            "lanes": [{"stamp": "tiny@1"}],
            "graphs": [],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return "sha256:" + hashlib.sha256(raw).hexdigest(), raw


def test_fetch_verifies_the_CAS_address_before_decoding() -> None:
    digest, raw = _document()

    class Response:
        status = 200

        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self, _limit: int) -> bytes:
            return raw

    got = bind_contract.fetch(digest, "https://objects.test/bind", opener=lambda *_a, **_k: Response())
    assert got.identity.config_digest == "sha256:config"
    assert got.census.pipeline_class == "TinyPipeline"

    wrong = "sha256:" + ("0" * 64)
    with pytest.raises(bind_contract.BindContractError, match="fetched bytes hashing"):
        bind_contract.fetch(wrong, "https://objects.test/bind", opener=lambda *_a, **_k: Response())


def test_mismatch_report_is_attributed_to_the_bind_not_the_pod() -> None:
    digest, raw = _document()
    contract = bind_contract.decode(raw, digest=digest)
    mismatch = CensusMismatch(
        I5_TOTALITY, "unet", "weight", "shape moved", where="serve Tiny"
    )
    payload = json.loads(bind_contract.refusal_payload(contract, mismatch))
    assert payload == {
        "release_id": "release-a",
        "derive_image_digest": "sha256:image",
        "config_digest": "sha256:config",
        "bind_contract_digest": digest,
        "code": "bind_contract_census_mismatch",
        "invariant": "I5_TOTALITY",
        "component": "unet",
        "tensor": "weight",
        "detail": str(mismatch),
    }
    assert "pod" not in payload and "worker" not in payload
