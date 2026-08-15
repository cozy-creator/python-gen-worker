"""Delivery-to-TCG ownership at the worker runtime boundary."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from gen_worker import aot_delivery, aot_serve, receipts
from torch_compiled_graphs import StoreOutcome

FAMILY = "runtime-arm"
KEY = "cg-key-v1-" + "a" * 56
REF = f"root/family-{FAMILY}#{KEY}"


def test_delivery_checks_receipt_before_importing_exact_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = tmp_path / "compiled-graph.tar.gz"
    artifact.write_bytes(b"verified transport bytes")
    order: list[tuple[str, object]] = []

    class Engine:
        def import_artifact(self, key: str, path: Path) -> object:
            order.append(("import", (key, Path(path))))
            return SimpleNamespace(outcome=StoreOutcome.STORED)

    def _receipt(path: Path, family: str) -> bool:
        order.append(("receipt", (Path(path), family)))
        return True

    monkeypatch.setattr(receipts, "gate_delivered_artifact", _receipt)
    monkeypatch.setattr(
        aot_delivery, "open_worker_engine", lambda _root=None: Engine()
    )

    aot_delivery._import_verified_artifact(
        artifact, cell_ref=REF, cache_dir=tmp_path / "cas", what="test"
    )

    assert order == [
        ("receipt", (artifact, FAMILY)),
        ("import", (KEY, artifact)),
    ]


def test_delivery_refuses_divergent_local_key_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = tmp_path / "compiled-graph.tar.gz"
    artifact.write_bytes(b"different admitted bytes")

    class Engine:
        def import_artifact(self, _key: str, _path: Path) -> object:
            return SimpleNamespace(outcome=StoreOutcome.DIVERGENT)

    monkeypatch.setattr(
        receipts, "gate_delivered_artifact", lambda *_args: True
    )
    monkeypatch.setattr(
        aot_delivery, "open_worker_engine", lambda _root=None: Engine()
    )

    with pytest.raises(aot_delivery.NamedArtifactUnavailable) as excinfo:
        aot_delivery._import_verified_artifact(
            artifact, cell_ref=REF, cache_dir=tmp_path / "cas", what="test"
        )

    assert excinfo.value.reason == "artifact_divergent"


def test_delivery_never_imports_an_unkeyed_ref(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = tmp_path / "compiled-graph.tar.gz"
    artifact.write_bytes(b"bytes")
    monkeypatch.setattr(
        receipts,
        "gate_delivered_artifact",
        lambda *_args: pytest.fail("receipt gate reached for an unkeyed ref"),
    )

    with pytest.raises(aot_delivery.NamedArtifactUnavailable) as excinfo:
        aot_delivery._import_verified_artifact(
            artifact,
            cell_ref=f"root/family-{FAMILY}#old-cell-key",
            cache_dir=tmp_path / "cas",
            what="test",
        )

    assert excinfo.value.reason == "artifact_unpinned"


def test_serve_handoff_removes_only_delivery_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from torch_compiled_graphs import artifact as artifact_mod

    incoming = tmp_path / "compiled-graph-transfer" / ".incoming"
    incoming.mkdir(parents=True)
    artifact = incoming / "graph.tar.gz"
    artifact.write_bytes(b"staged")

    class Engine:
        def import_artifact(self, _key: str, _path: Path) -> object:
            return SimpleNamespace(outcome=StoreOutcome.PRESENT)

    monkeypatch.setattr(
        artifact_mod, "read_metadata", lambda _path: {"compiled_graph_key": KEY}
    )
    monkeypatch.setattr(aot_serve, "open_worker_engine", lambda _root=None: Engine())
    monkeypatch.setattr(
        aot_serve,
        "arm_compiled_graph",
        lambda *_args, **_kwargs: {"compiled_graph_key": KEY},
    )

    result = aot_serve._import_and_arm(
        object(), object(), artifact, tmp_path / "cas", expected=None, declared=()
    )

    assert result == {"compiled_graph_key": KEY}
    assert not artifact.exists()
