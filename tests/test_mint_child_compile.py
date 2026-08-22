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

    example = torch.zeros((2, 8), dtype=torch.float32)
    program = torch.export.export(Denoiser(), (example,))
    ingress = build_call_ingress(program, ("value",), (example,), {})

    blob = tmp_path / "graph.pt2"
    torch.export.save(program, blob)
    return {
        "blob": str(blob),
        "graph": "denoiser/h=8",
        "target": "projection",
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
    assert any(Path(request["cas"]).rglob("*")), "the compiled graph never reached the CAS"


@pytest.mark.slow
def test_the_child_binds_a_symbolic_parent_for_a_static_record(
    tmp_path: Path,
) -> None:
    """pgw#1603: the store banks the symbolic PARENT under a static record's
    identity — a stripped parent blob, a static ingress, and the record's
    exact graph hash must still compile end to end (bind happens at the
    engine's plan seam and refuses any identity drift)."""

    from gen_worker._vendor.torchcg.mint import strip_diagnostics
    from gen_worker._vendor.torchcg.identity import graph_hash

    module = Denoiser()
    height = torch.export.Dim("height", min=2, max=16)
    example = torch.zeros((8, 8), dtype=torch.float32)
    parent = torch.export.export(
        module, (example,), dynamic_shapes={"value": {0: height}}, strict=False
    )
    concrete = torch.zeros((4, 8), dtype=torch.float32)
    bound = torch.export.export(module, (concrete,), strict=False)
    ingress = build_call_ingress(bound, ("value",), (concrete,), {})
    graph = graph_hash(bound, ingress)

    strip_diagnostics(parent)
    blob = tmp_path / "parent.pt2"
    torch.export.save(parent, blob)

    request = _request(tmp_path)
    request["blob"] = str(blob)
    request["graph"] = graph
    request["ingress"] = ingress.as_dict()
    Path(request["cas"]).mkdir(parents=True, exist_ok=True)

    destination = compile_one(request)
    assert destination.is_dir()
    assert any(Path(request["cas"]).rglob("*"))


@pytest.mark.slow
def test_the_child_request_carries_no_graph_interface_to_get_wrong(
    tmp_path: Path,
) -> None:

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

    round_tripped = json.loads(json.dumps(request))
    Path(round_tripped["cas"]).mkdir(parents=True, exist_ok=True)
    assert compile_one(round_tripped).is_dir()


def test_the_retired_v3_INTERFACE_can_no_longer_be_expressed(tmp_path: Path) -> None:
    """This used to assert that a v3 graph-interface object refuses BY NAME
    rather than being coerced into a v4 key.

    tcg#90 deleted the graph-interface object itself — there is no
    `{v, constant_fqns, ingress}` document any more, and no `declaration.py` to
    hold its version. The ingress travels directly into the graph hash, so
    there is no versioned wrapper for old bytes to be shaped like and nothing
    to coerce. The refusal is not weaker; the thing it refused cannot be built.

    What replaces it is the ONE decode that still guards this seam: a request's
    ingress must decode to the exact closed field set, or the child never mints.
    """

    from gen_worker._vendor.torchcg import CallIngress
    from gen_worker._vendor.torchcg.refuse import IngressError

    request = _request(tmp_path)
    assert CallIngress.decode(request["ingress"])

    retired = dict(request["ingress"])
    retired["pytree"] = {"in": "leaf", "out": "leaf"}
    with pytest.raises(IngressError, match="wrong field set"):
        CallIngress.decode(retired)

    absent = {k: v for k, v in request["ingress"].items() if k != "symbols"}
    with pytest.raises(IngressError, match="wrong field set"):
        CallIngress.decode(absent)
