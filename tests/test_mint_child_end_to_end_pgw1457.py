"""pgw#1457: the mint child compiles a REAL exported program, end to end.

This is the test pgw#1456 named and did not have. The v2 mint path had never
compiled a single graph from any caller, and nothing noticed, because every
test around it stopped at a hand-built request or a hand-seeded artifact. Two
separate defects lived in that gap:

* pgw#1456 -- ``mint_child`` hand-built a two-field graph interface that
  ``GraphSpecializationDeclaration`` refused by name before any compilation started.
  Its first fix hand-built a FIVE-field one. tcg#55 deleted the parameter
  entirely, so there is nothing left to hand-build.
* pgw#1458 -- the derive stamped a device the mint could not compile for, and
  the failure surfaced from inside AOTInductor minutes in.

So the drive here is deliberately the REAL one: a real ``torch.export``
program, serialized to a real blob, handed to ``compile_one`` as the real JSON
request the parent writes, producing a real AOTI package in a real
``LocalCAS``. Nothing is mocked, monkeypatched, or seeded. It runs on CPU --
which is the only substrate this repo has (pgw has no GPU CI lane, pgw#953
parked), and which is why the cuda leg of the same path is a developer-box and
pod concern, stated rather than pretended.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

torch: Any = pytest.importorskip("torch")

from gen_worker._vendor.torchcg import build_call_ingress  # noqa: E402
from gen_worker.serving.mint_child import compile_one, contract_digest  # noqa: E402


class Denoiser(torch.nn.Module):  # type: ignore[misc]
    def __init__(self) -> None:
        super().__init__()
        self.projection = torch.nn.Linear(8, 8)

    def forward(self, value: Any) -> Any:
        return self.projection(value).relu()


def _request(tmp_path: Path) -> dict[str, Any]:
    """The parent's real request, built the way `mint.py` builds it."""

    example = torch.zeros((2, 8), dtype=torch.float32)
    program = torch.export.export(Denoiser(), (example,))
    ingress = build_call_ingress(program, ("value",), (example,), {})

    blob = tmp_path / "graph.pt2"
    torch.export.save(program, blob)
    return {
        "blob": str(blob),
        "graph": "denoiser/h=8",
        "target": "projection",
        # `mint.py` sends `record.ingress.as_dict()`; the child decodes it back
        # into a typed CallIngress. That round trip IS the contract now.
        "ingress": ingress.as_dict(),
        "target_arch": "cpu",
        "toolchain": {"torch": str(torch.__version__)},
        "cas": str(tmp_path / "cas"),
        "destination": str(tmp_path / "compiled"),
        "result": str(tmp_path / "artifact.txt"),
        "contract": contract_digest(),
    }


@pytest.mark.slow
def test_the_mint_child_compiles_a_real_exported_program(tmp_path: Path) -> None:
    """The whole child, on the real path, from blob to unpacked artifact."""

    request = _request(tmp_path)
    Path(request["cas"]).mkdir(parents=True, exist_ok=True)

    destination = compile_one(request)

    assert destination == Path(request["destination"])
    assert destination.is_dir(), "the child must leave the resolved artifact unpacked"
    payload = sorted(p.name for p in destination.rglob("*") if p.is_file())
    assert payload, f"nothing was materialized into {destination}"
    # A real AOTI package, not a placeholder: the store is populated too.
    assert any(Path(request["cas"]).rglob("*")), "the compiled graph never reached the CAS"


@pytest.mark.slow
def test_the_child_request_carries_no_graph_interface_to_get_wrong(
    tmp_path: Path,
) -> None:
    """The pgw#1456 defect is UNREPRESENTABLE, not caught.

    The request the parent writes has no place to put a graph interface, and
    the child has no parameter to pass one to. A stub cannot be built, so it
    cannot be wrong -- which is a stronger property than any assertion about a
    stub's contents, and it is the whole point of tcg#55.
    """

    request = _request(tmp_path)
    assert "graph_interface" not in request
    assert set(request["ingress"]) == {
        "v",
        "parameters",
        "flat_arity",
        "inputs",
        "symbols",
        "excluded_inputs",
    }

    # It survives the parent's actual serialization, which is JSON on argv.
    round_tripped = json.loads(json.dumps(request))
    Path(round_tripped["cas"]).mkdir(parents=True, exist_ok=True)
    assert compile_one(round_tripped).is_dir()


def test_the_child_refuses_a_retired_v3_interface_by_name(tmp_path: Path) -> None:
    """Old bytes name themselves rather than being coerced into a v4 key.

    Graph specializations are content addressed, so a v3 document has nothing to
    migrate -- it re-derives. What must never happen is one being reshaped
    into something no producer ever derived.
    """

    from gen_worker._vendor.torchcg import CallIngress, GraphSpecializationDeclaration
    from gen_worker._vendor.torchcg.declaration import RetiredGraphInterface

    request = _request(tmp_path)
    ingress = CallIngress.decode(request["ingress"])
    retired = {
        "v": 3,
        "constant_fqns": [],
        "lifted_inputs": [],
        "pytree": {"in": "leaf", "out": "leaf", "ingress": request["ingress"]},
        "specialization": {},
    }
    with pytest.raises(RetiredGraphInterface, match="RETIRED v3 shape"):
        GraphSpecializationDeclaration(
            "denoiser/h=8", "projection", retired, "0" * 16, ingress.digest()
        )
