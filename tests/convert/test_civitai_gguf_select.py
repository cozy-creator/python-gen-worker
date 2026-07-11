"""th#611: civitai gguf-only versions select exactly one gguf file.

Safetensors-bearing versions keep the existing behavior (safetensors only,
primary first). GGUF-only versions pick ONE gguf — civitai reuses a single
filename across quantType variants, so plural downloads would collide.
"""

from __future__ import annotations

import pytest

from gen_worker.models.download import _civitai_select_files


def _f(name, *, id=1, primary=False, quant=None, size_kb=1000):
    meta = {"format": "GGUF" if name.endswith(".gguf") else "SafeTensor"}
    if quant:
        meta["quantType"] = quant
    return {
        "id": id,
        "name": name,
        "downloadUrl": f"https://civitai.example/{id}",
        "sizeKB": size_kb,
        "primary": primary,
        "metadata": meta,
        "hashes": {"SHA256": "ab" * 32},
    }


def test_safetensors_win_over_gguf():
    files = _civitai_select_files({"files": [
        _f("model.gguf", id=1, quant="Q5_K_M"),
        _f("model.safetensors", id=2, primary=True),
    ]})
    assert [f["name"] for f in files] == ["model.safetensors"]


def test_gguf_only_picks_one_by_preference():
    files = _civitai_select_files({"files": [
        _f("m.gguf", id=1, quant="Q5_K_M"),
        _f("m.gguf", id=2, quant="Q8_0"),
    ]})
    assert len(files) == 1
    assert files[0]["quant_type"] == "q8_0"  # preference head


def test_gguf_only_explicit_quant_pick():
    files = _civitai_select_files({"files": [
        _f("m.gguf", id=1, quant="Q5_K_M"),
        _f("m.gguf", id=2, quant="Q8_0"),
    ]}, gguf_quant="q5_k_m")
    assert len(files) == 1
    assert files[0]["quant_type"] == "q5_k_m"


def test_gguf_only_quant_from_filename():
    files = _civitai_select_files({"files": [
        _f("model-Q4_K_M.gguf", id=1),
        _f("model-Q8_0.gguf", id=2),
    ]}, gguf_quant="q4_k_m")
    assert len(files) == 1
    assert files[0]["name"] == "model-Q4_K_M.gguf"


def test_gguf_quant_not_found_raises():
    with pytest.raises(ValueError, match="civitai_gguf_quant_not_found"):
        _civitai_select_files({"files": [
            _f("m.gguf", id=1, quant="Q5_K_M"),
        ]}, gguf_quant="q4_k_m")


def test_no_weights_empty():
    assert _civitai_select_files({"files": [_f("notes.zip", id=1)]}) == []


def test_gguf_quant_from_download_url():
    # Real civitai shape (Fascium mv3006357): one filename for all quants,
    # no metadata.quantType; the non-primary file's URL carries it.
    files = _civitai_select_files({"files": [
        {"id": 1, "name": "m.gguf", "sizeKB": 6331446,
         "downloadUrl": "https://civitai.com/api/download/models/3006357",
         "metadata": {"format": "GGUF"}},
        {"id": 2, "name": "m.gguf", "sizeKB": 9514038,
         "downloadUrl": "https://civitai.com/api/download/models/3006357?type=Model&format=GGUF&quantType=Q8_0",
         "metadata": {"format": "GGUF"}},
    ]}, gguf_quant="q8_0")
    assert len(files) == 1
    assert files[0]["id"] == 2


def test_ingest_civitai_gguf_classification(tmp_path, monkeypatch):
    """gguf-only civitai trees classify strategy=gguf (as-is publish)."""
    import struct
    from gen_worker.convert import ingest as ing

    # Minimal real gguf header: magic/v3, 0 tensors, 1 KV (general.file_type=17).
    buf = b"GGUF" + struct.pack("<IQQ", 3, 0, 1)
    key = b"general.file_type"
    buf += struct.pack("<Q", len(key)) + key + struct.pack("<I", 4) + struct.pack("<I", 17)
    (tmp_path / "model.gguf").write_bytes(buf)

    monkeypatch.setattr(
        "gen_worker.models.download.fetch_civitai_model_version",
        lambda vid, api_key="": {"baseModel": "Flux.2 Klein 9B", "modelId": 1,
                                 "model": {"type": "Checkpoint"}},
    )
    monkeypatch.setattr(
        "gen_worker.models.download.download_civitai",
        lambda vid, dest, api_key="", progress=None, gguf_quant=None: dest,
    )
    src = ing.ingest_civitai(123, tmp_path, civitai_api_key="", gguf_quant="q5_k_m")
    assert src.classification is not None
    assert src.classification.strategy == "gguf"
    assert src.attrs["dtype"] == "gguf:q5_k_m"
    assert src.attrs["file_type"] == "gguf"
