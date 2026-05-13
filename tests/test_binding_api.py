"""Repo / Dispatch binding API surface tests.

Validates the chainable modifier behavior promised by `progress.json` issue
#9 (decorator-table-model-bindings):

- `Repo(ref)` is a frozen dataclass and a usable binding with defaults.
- Modifier methods (`.flavor`, `.tag`, `.allow_override`) return new
  immutable instances; chain order is commutative.
- `.allow_override(*classes)` accepts class objects OR string FQNs.
- Zero-arg `.allow_override()` raises ValueError at decoration time.
- `dispatch(field, table)` constructs a Dispatch with the same invariants.
"""

from __future__ import annotations

import pytest

from gen_worker import Dispatch, Repo, dispatch


class _Pipe:
    pass


class _OtherPipe:
    pass


def test_repo_bare_is_binding_with_defaults() -> None:
    r = Repo("acme/myrepo")
    assert r.ref == "acme/myrepo"
    assert r._flavor == ""
    assert r._tag == "prod"
    assert r._allow_override is False
    assert r._pipeline_classes == ()


def test_repo_rejects_empty_ref() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        Repo("")
    with pytest.raises(ValueError, match="non-empty"):
        Repo("   ")


def test_repo_flavor_returns_new_instance() -> None:
    r1 = Repo("acme/myrepo")
    r2 = r1.flavor("nf4")
    assert r1 is not r2
    assert r1._flavor == ""
    assert r2._flavor == "nf4"
    # Frozen — can't mutate.
    with pytest.raises((AttributeError, Exception)):
        r1.ref = "other/repo"  # type: ignore[misc]


def test_repo_tag_returns_new_instance() -> None:
    r = Repo("acme/myrepo")
    r2 = r.tag("canary")
    assert r2._tag == "canary"
    assert r._tag == "prod"


def test_repo_chain_order_is_commutative() -> None:
    a = Repo("acme/r").flavor("nf4").tag("canary").allow_override(_Pipe)
    b = Repo("acme/r").tag("canary").allow_override(_Pipe).flavor("nf4")
    c = Repo("acme/r").allow_override(_Pipe).flavor("nf4").tag("canary")
    assert a == b == c


def test_allow_override_accepts_class_object() -> None:
    r = Repo("acme/r").allow_override(_Pipe)
    assert r._allow_override is True
    assert r._pipeline_classes == (
        "test_binding_api._Pipe",
    )


def test_allow_override_accepts_string_fqn() -> None:
    r = Repo("acme/r").allow_override("pkg.mod.MyPipeline")
    assert r._pipeline_classes == ("pkg.mod.MyPipeline",)


def test_allow_override_accepts_multiple_classes() -> None:
    r = Repo("acme/r").allow_override(_Pipe, _OtherPipe)
    assert len(r._pipeline_classes) == 2
    assert "test_binding_api._Pipe" in r._pipeline_classes
    assert "test_binding_api._OtherPipe" in r._pipeline_classes


def test_allow_override_deduplicates() -> None:
    r = Repo("acme/r").allow_override(_Pipe, _Pipe, "test_binding_api._Pipe")
    assert r._pipeline_classes == ("test_binding_api._Pipe",)


def test_allow_override_zero_arg_raises_value_error() -> None:
    with pytest.raises(ValueError, match="at least one"):
        Repo("acme/r").allow_override()


def test_allow_override_rejects_non_class_non_string() -> None:
    with pytest.raises(TypeError):
        Repo("acme/r").allow_override(42)  # type: ignore[arg-type]


def test_allow_override_rejects_empty_string() -> None:
    with pytest.raises(ValueError):
        Repo("acme/r").allow_override("")


def test_dispatch_basic_shape() -> None:
    d = dispatch(
        field="variant",
        table={
            "nf4": Repo("acme/r").flavor("nf4"),
            "int8": Repo("acme/r").flavor("int8"),
        },
    )
    assert isinstance(d, Dispatch)
    assert d.field == "variant"
    assert set(d.table.keys()) == {"nf4", "int8"}


def test_dispatch_rejects_empty_table() -> None:
    with pytest.raises(ValueError, match="non-empty table"):
        dispatch(field="variant", table={})


def test_dispatch_rejects_non_repo_value() -> None:
    with pytest.raises(TypeError, match="Repo instances"):
        dispatch(field="variant", table={"a": "not-a-repo"})  # type: ignore[dict-item]


def test_dispatch_allow_override_returns_new_instance() -> None:
    d1 = dispatch(field="v", table={"a": Repo("x/y")})
    d2 = d1.allow_override(_Pipe)
    assert d1 is not d2
    assert d1._allow_override is False
    assert d2._allow_override is True
    assert d2._pipeline_classes == ("test_binding_api._Pipe",)


def test_dispatch_allow_override_zero_arg_raises() -> None:
    d = dispatch(field="v", table={"a": Repo("x/y")})
    with pytest.raises(ValueError, match="at least one"):
        d.allow_override()
