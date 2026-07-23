"""th#1051 RuntimeFormula: parse, canonical term keys (must byte-match the Go
side), payload validation, worker-side term evaluation."""

import msgspec
import pytest

from gen_worker.api.formula import RuntimeFormula


class Payload(msgspec.Struct):
    prompt: str = ""
    num_inference_steps: int = 28
    megapixels: float = 1.0
    width: int = 1024
    height: int = 1024
    hires: bool = False


def test_terms_and_keys():
    rf = RuntimeFormula(
        "a + b*num_inference_steps + c*num_inference_steps*megapixels"
    )
    assert [t.key for t in rf.terms] == [
        "1", "num_inference_steps", "num_inference_steps*megapixels",
    ]
    assert rf.fields == ("megapixels", "num_inference_steps")


def test_composite_key_matches_go_canonical():
    # Go: "steps*(width*height/1000000)" -> "steps*width*height/1000000"
    rf = RuntimeFormula("a + b*num_inference_steps*(width*height/1000000)")
    assert rf.terms[1].key == "num_inference_steps*width*height/1000000"


def test_validate_for_payload():
    rf = RuntimeFormula("a + b*num_inference_steps + c*num_inference_steps*megapixels")
    rf.validate_for_payload(Payload, "gen")  # ok

    with pytest.raises(ValueError, match="not a payload field"):
        RuntimeFormula("a + b*frames").validate_for_payload(Payload, "gen")
    with pytest.raises(ValueError, match="collides"):
        RuntimeFormula("megapixels + b*num_inference_steps").validate_for_payload(
            Payload, "gen"
        )

    class NoDefault(msgspec.Struct):
        steps: int

    with pytest.raises(ValueError, match="default"):
        RuntimeFormula("a + b*steps").validate_for_payload(NoDefault, "gen")


def test_rejects():
    for src in [
        "a - b*num_inference_steps",   # top-level minus
        "2 + b*num_inference_steps",   # term must start with a constant
        "a + a*num_inference_steps",   # duplicate constant
        "a + b/num_inference_steps",   # constant must multiply
        "a + b*max(x, y)",             # calls
        "a + b*3",                     # factor with no payload field
    ]:
        with pytest.raises(ValueError):
            RuntimeFormula(src)


def test_term_values_from_struct():
    rf = RuntimeFormula("a + b*num_inference_steps + c*num_inference_steps*megapixels")
    got = rf.term_values_from_struct(Payload(num_inference_steps=50, megapixels=2.0))
    assert got == {
        "1": 1.0,
        "num_inference_steps": 50.0,
        "num_inference_steps*megapixels": 100.0,
    }
    # Bool fields coerce 0/1.
    rf2 = RuntimeFormula("a + b*hires")
    assert rf2.term_values_from_struct(Payload(hires=True)) == {"1": 1.0, "hires": 1.0}


def test_manifest_emits_runtime_formula(tmp_path, monkeypatch):
    import textwrap

    monkeypatch.syspath_prepend(str(tmp_path))
    pkg = tmp_path / "ep_th1051"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "main.py").write_text(textwrap.dedent("""
        import msgspec
        from gen_worker import RequestContext, RuntimeFormula, endpoint

        class In_(msgspec.Struct):
            prompt: str = ""
            num_inference_steps: int = 28
            megapixels: float = 1.0

        class Out_(msgspec.Struct):
            y: str

        @endpoint(runtime=RuntimeFormula(
            "a + b*num_inference_steps + c*num_inference_steps*megapixels"
        ))
        def generate(ctx: RequestContext, data: In_) -> Out_:
            return Out_(y="ok")
    """))

    from gen_worker.discovery.discover import discover_functions

    (fn,) = discover_functions(tmp_path, main_module="ep_th1051.main")
    assert fn["runtime_formula"] == (
        "a + b*num_inference_steps + c*num_inference_steps*megapixels"
    )
