"""TCG 0.3 is the compile child's sole identity/compile/package interior."""

from __future__ import annotations

import ast
import inspect
import json
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from torch_compiled_graphs import CallIngress, CallInput, build_call_ingress, spans

from gen_worker import aot_compile_child as child
from gen_worker import aot_compile_pool as pool
from gen_worker import aot_mint
from gen_worker.aot_compile_pool import EntryJob

torch = pytest.importorskip("torch")


class _Traced:
    def __init__(self, name: str = "denoiser/b=1") -> None:
        self.name = name
        self.program = object()
        self.nodes = 7
        self.declared = 1
        self.timings = {"export_s": 0.25}
        self.releases = 0
        ingress = CallIngress(
            parameters=("sample",),
            flat_arity=1,
            inputs=(CallInput(
                name="sample",
                position=0,
                param="sample",
                param_position=0,
                path=(),
                exported_name="sample",
                dtype="float32",
                shape=(1, 4),
            ),),
        )
        self.block = {
            "target": "denoiser",
            "fork": [["adapter", False]],
            "class_dims": [["batch", 1]],
            "graph": {
                "v": 3,
                "lifted_inputs": [],
                "pytree": {
                    "user_inputs": ["sample"],
                    "in_spec": "leaf",
                    "out_spec": "leaf",
                    "ingress": ingress.as_dict(),
                },
                "specialization": {},
            },
        }

    def release(self) -> None:
        self.releases += 1
        self.program = None


def _export_spec() -> Any:
    return SimpleNamespace(strict=True, lora_bucket=0)


class _Engine:
    def __init__(self, tmp_path: Path, *, recorded_key: str | None = None,
                 outcome: str = "minted", refusal: BaseException | None = None) -> None:
        self.tmp_path = tmp_path
        self.recorded_key = recorded_key
        self.outcome = outcome
        self.refusal = refusal
        self.compile_calls = 0
        self.export_calls = 0

    def compile(self, spec: Any, runtime: Any, destination: Path) -> Any:
        self.compile_calls += 1
        if self.refusal is not None:
            raise self.refusal
        key = "cg-key-v1-" + "a" * 56
        metadata = {
            "compiled_graph_key": self.recorded_key or key,
            "graph_class": {"name": spec.graph_class},
            "compiled_graph_format": 1,
        }
        return SimpleNamespace(
            outcome=SimpleNamespace(value=self.outcome),
            compiled_graph=SimpleNamespace(key=key, metadata=metadata),
        )

    def export_artifact(self, key: str, destination: Path) -> Path:
        self.export_calls += 1
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"tcg-artifact")
        return destination


def test_exported_row_maps_to_one_tcg_graph_class_spec() -> None:
    traced = _Traced()

    spec = child._graph_class_spec(traced, _export_spec())

    assert spec.graph_class == traced.name
    assert spec.target == "denoiser"
    assert spec.program is traced.program
    assert spec.graph == traced.block["graph"]
    assert spec.fork == (("adapter", False),)
    assert spec.class_dims == (("batch", 1),)
    assert len(CallIngress.from_graph(spec.graph).digest()) == 32
    assert not hasattr(spec, "range_digest")
    assert spec.strict is True
    assert spec.lora_bucket == 0


def test_real_nested_call_uses_one_tcg_ingress_and_rekeys_on_change() -> None:
    class Nested(torch.nn.Module):
        def forward(self, sample: Any, cond: dict[str, Any]) -> Any:
            return sample + cond["bias"]

    module = Nested().eval()
    args = (torch.randn(2, 4), {"bias": torch.randn(2, 4)})
    program = torch.export.export(module, args, strict=True)
    ingress = build_call_ingress(program, ("sample", "cond"), args, {})
    export_spec = aot_mint.ExportSpec(family="micro", target="denoiser")
    block = aot_mint.keying_block(program, ingress, export_spec)
    traced = aot_mint.TracedClass(
        name="denoiser/b=2",
        block=block,
        nodes=1,
        program=program,
    )

    declared = child._graph_class_spec(traced, export_spec).declare()

    assert block["graph"]["pytree"]["ingress"] == ingress.as_dict()
    assert not ({"inputs", "symbols", "range_digest"} & set(block))
    assert declared.range_digest == ingress.digest()

    changed = CallIngress(
        ingress.parameters,
        ingress.flat_arity,
        ingress.inputs,
        ingress.symbols,
        ("unused",),
    )
    changed_block = aot_mint.keying_block(program, changed, export_spec)
    changed_spec = child._graph_class_spec(
        aot_mint.TracedClass(
            name=traced.name,
            block=changed_block,
            nodes=1,
            program=program,
        ),
        export_spec,
    ).declare()

    assert changed_spec.range_digest != declared.range_digest
    assert changed_spec.class_hash != declared.class_hash


def test_worker_carries_no_second_package_or_ingress_implementation() -> None:
    package = Path(aot_mint.__file__).resolve().parent
    removed = ("aot_package", "aot_flatten", "aot_contract")

    assert all(not (package / f"{name}.py").exists() for name in removed)
    production = "\n".join(
        path.read_text() for path in package.rglob("*.py")
    )
    for name in removed:
        assert f"from . import {name}" not in production
        assert f"from .{name} import" not in production


def test_tcg_result_is_the_unchanged_minimal_pool_wire(tmp_path: Path) -> None:
    traced = _Traced()
    engine = _Engine(tmp_path)

    result = child._compile_traced_class(
        traced,
        _export_spec(),
        engine,
        object(),
        work=tmp_path / "work",
        out_dir=tmp_path / "artifacts",
    )

    assert engine.compile_calls == 1
    assert engine.export_calls == 1
    assert result.packed.name == traced.name
    assert result.packed.key == "cg-key-v1-" + "a" * 56
    assert Path(result.packed.artifact).read_bytes() == b"tcg-artifact"
    assert json.loads(result.packed.metadata) == {
        "compiled_graph_format": 1,
        "compiled_graph_key": result.packed.key,
        "graph_class": {"name": traced.name},
    }
    assert result.compile_s >= 0
    assert result.reuse_s == 0


def test_wrong_tcg_metadata_ref_refuses_before_artifact_export(tmp_path: Path) -> None:
    engine = _Engine(tmp_path, recorded_key="cg-key-v1-" + "b" * 56)

    with pytest.raises(ValueError, match="not selected ref"):
        child._compile_traced_class(
            _Traced(),
            _export_spec(),
            engine,
            object(),
            work=tmp_path / "work",
            out_dir=tmp_path / "artifacts",
        )

    assert engine.compile_calls == 1
    assert engine.export_calls == 0


def test_every_in_flight_row_releases_on_refusal(tmp_path: Path) -> None:
    traced = _Traced()
    engine = _Engine(tmp_path, refusal=RuntimeError("compile refused"))

    with pytest.raises(RuntimeError, match="compile refused"):
        child.compile_traced_class(
            traced,
            _export_spec(),
            engine,
            object(),
            work=tmp_path / "work",
            out_dir=tmp_path / "artifacts",
        )

    assert traced.releases == 1
    assert traced.program is None


def test_tcg_reuse_reports_zero_compile_time(tmp_path: Path) -> None:
    engine = _Engine(tmp_path, outcome="reused")

    result = child.compile_traced_class(
        _Traced(),
        _export_spec(),
        engine,
        object(),
        work=tmp_path / "work",
        out_dir=tmp_path / "artifacts",
    )

    assert result.compile_s == 0
    assert result.reuse_s >= 0


def test_retry_filters_held_classes_before_export(monkeypatch: pytest.MonkeyPatch) -> None:
    plan = object()
    monkeypatch.setattr(aot_mint, "declared_class_rows", lambda *args: [(plan, True)])
    monkeypatch.setattr(aot_mint._decl, "plan_entry_name", lambda value: "held/class")
    monkeypatch.setattr(aot_mint, "_arm_branches", lambda *args: None)
    monkeypatch.setattr(aot_mint, "_export_entry", lambda *args, **kwargs: pytest.fail(
        "a held class reached export"))
    job = EntryJob(have_classes=("held/class",), share_count=1)

    rows = list(child._trace_share(
        aot_mint,
        object(),
        SimpleNamespace(lora_bucket=0),
        object(),
        job,
    ))

    assert rows == []


def test_runtime_uses_the_canonical_worker_cas(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch_compiled_graphs

    from gen_worker import compile_cache
    from gen_worker.models import cache_paths

    cas = object()
    captured: dict[str, Any] = {}

    class _Runtime:
        def __init__(self, target: str, *, toolchain: dict[str, str]) -> None:
            captured["target"] = target
            captured["toolchain"] = toolchain

    class _TCGEngine:
        def __init__(self, value: Any) -> None:
            captured["cas"] = value

    monkeypatch.setattr(torch_compiled_graphs, "RuntimeCompatibility", _Runtime)
    monkeypatch.setattr(torch_compiled_graphs, "Engine", _TCGEngine)
    monkeypatch.setattr(cache_paths, "open_worker_cas", lambda: cas)
    monkeypatch.setattr(compile_cache, "runtime_key", lambda: {"sm": "sm_86"})
    monkeypatch.setattr(
        compile_cache,
        "toolchain_digest",
        lambda: (("torch", "exact"), ("triton", "exact")),
    )

    engine, runtime = child._tcg_runtime()

    assert isinstance(engine, _TCGEngine)
    assert isinstance(runtime, _Runtime)
    assert captured == {
        "cas": cas,
        "target": "sm_86",
        "toolchain": {"torch": "exact", "triton": "exact"},
    }


def test_pool_has_no_exported_program_resume_bank(tmp_path: Path) -> None:
    width = pool.entry_workers(
        1,
        limit=1,
        vcpus=16,
        available_bytes=64 * 1024**3,
        device_lock=True,
    )
    box = pool.EntryCompilePool(tmp_path / "pool", width=width)

    assert not hasattr(box, "bank")
    assert box.cache_dir == str(tmp_path / "pool" / "inductor-cache")
    assert not any(key.startswith("resume_") for key in box.ledger.facts())
    assert "aot_resume" not in inspect.getsource(pool)


def test_child_runs_the_span_closure_check_nonfatally(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    seen: list[dict[str, float]] = []

    def record(spans: Mapping[str, float]) -> list[str]:
        seen.append(dict(spans))
        return ["forced partition defect"]

    monkeypatch.setattr(spans, "check", record)
    ledger = spans.SpanLedger()

    fields = child._span_fields(ledger, {}, {}, {})

    assert seen
    assert "child_wall_s" in seen[0]
    assert fields["spans"]
    assert "forced partition defect" in caplog.text


def test_tcg_compile_stays_in_the_compile_child_and_fence_is_green() -> None:
    tree = ast.parse(inspect.getsource(child._compile_traced_class))
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "compile"
    ]
    assert len(calls) == 1

    repo = Path(__file__).resolve().parent.parent
    completed = subprocess.run(
        [sys.executable, str(repo / "scripts/lint_serving_process_compiles.py")],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
