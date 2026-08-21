"""Dataset — handle to a dataset snapshot on local disk."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator
import random

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

_BYTES_KEY = "__bytes_b64__"


class Dataset:
    """Handle to a dataset snapshot on local disk."""

    def __init__(
        self,
        *,
        ref: str,
        split: str,
        path: Path,
        attributes: dict | None = None,
    ) -> None:
        self._ref = ref
        self._split = split
        self._path = Path(path)
        self._attributes = dict(attributes or {})
        self._info_cache: dict | None | str = "unloaded"

    @property
    def ref(self) -> str:
        return self._ref

    @property
    def split(self) -> str:
        return self._split

    @property
    def path(self) -> Path:
        return self._path

    @property
    def attributes(self) -> dict:
        return self._attributes

    def iter_examples(self) -> Iterator[dict]:
        """Yield raw dataset rows as dicts."""
        import datasets as hf_datasets
        ds = hf_datasets.load_from_disk(str(self._path))
        if isinstance(ds, hf_datasets.DatasetDict):
            ds = ds[self._split]
        for row in ds:
            yield row

    def as_dataloader(
        self,
        *,
        tokenizer: Any,
        n: int,
        seq_length: int,
        batch_size: int = 1,
        seed: int = 42,
    ) -> "DataLoader":
        """Return a torch DataLoader of tokenized batches for calibration."""

        import torch
        from torch.utils.data import DataLoader, TensorDataset

        rng = random.Random(seed)
        examples: list[dict] = list(self.iter_examples())
        if not examples:
            raise ValueError(f"dataset {self._ref}:{self._split} is empty")
        rng.shuffle(examples)
        sampled = examples[:n]
        text_field = _guess_text_field(sampled[0])
        tokenized_ids: list[torch.Tensor] = []
        attention_masks: list[torch.Tensor] = []
        for ex in sampled:
            enc = tokenizer(
                ex[text_field],
                truncation=True, max_length=seq_length, padding="max_length",
                return_tensors="pt",
            )
            tokenized_ids.append(enc["input_ids"].squeeze(0))
            attention_masks.append(enc["attention_mask"].squeeze(0))
        ds = TensorDataset(torch.stack(tokenized_ids), torch.stack(attention_masks))
        return DataLoader(ds, batch_size=batch_size, shuffle=False)

    def dataset_info(self) -> dict:
        """Return the parsed ``dataset_info.json`` at ``path`` or ``{}``."""
        if self._info_cache == "unloaded":
            self._info_cache = _load_dataset_info(self._path)
        if isinstance(self._info_cache, dict):
            return self._info_cache
        return {}

    @property
    def kind(self) -> str:
        """One of ``"prompt_corpus"``, ``"eval_set"``, or ``""`` (unknown)."""
        return str(self.dataset_info().get("kind") or "")

    def is_prompt_corpus(self) -> bool:
        """True iff this artifact is a prompt corpus."""
        return self.kind == "prompt_corpus"

    def shards(self) -> list[Path]:
        """Return the list of jsonl shard files under ``data/``."""
        data_dir = self._path / "data"
        if not data_dir.is_dir():
            return []
        return sorted(data_dir.glob("*.jsonl"))

    def iter_prompts(self) -> Iterator[dict]:
        """Yield prompt rows — ``{prompt, category, length_bucket, seed}``."""
        shards = self.shards()
        if not shards:
            raise FileNotFoundError(
                f"dataset {self._ref!r} has no prompts — expected jsonl "
                f"shards under {self._path}/data/"
            )
        yield from _iter_jsonl_prompt_columns(shards)

    def iter_rows(self) -> Iterator[dict]:
        """Yield ALL rows (including image columns on eval sets)."""
        shards = self.shards()
        if not shards:
            yield from self.iter_prompts()
            return
        for shard in shards:
            with open(shard) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    yield _decode_row(json.loads(line))


def write_jsonl_shard(
    rows: list[dict], out_dir: Path, *, name: str = "train-00000.jsonl",
) -> Path:
    """Write ``rows`` as a jsonl shard at ``<out_dir>/data/<name>``."""
    data_dir = out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / name
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, default=_json_row_default))
            f.write("\n")
    return path


def _json_row_default(obj: Any) -> Any:
    if isinstance(obj, (bytes, bytearray)):
        return {_BYTES_KEY: base64.b64encode(bytes(obj)).decode("ascii")}
    raise TypeError(f"object of type {type(obj).__name__} is not JSON serializable")


def _decode_row(row: dict) -> dict:
    return {
        k: (base64.b64decode(v[_BYTES_KEY])
            if isinstance(v, dict) and set(v) == {_BYTES_KEY} else v)
        for k, v in row.items()
    }


def _load_dataset_info(path: Path) -> dict | None:
    info_path = path / "dataset_info.json"
    if not info_path.exists():
        return None
    try:
        with open(info_path) as f:
            data = json.load(f)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


_WANTED_PROMPT_COLS = ("prompt", "category", "length_bucket", "seed")


def _iter_jsonl_prompt_columns(shards: list[Path]) -> Iterator[dict]:
    for shard in shards:
        with open(shard) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                raw = json.loads(line)
                if "prompt" not in raw:
                    raise ValueError(
                        f"jsonl shard {shard} has a row missing the required "
                        f"'prompt' field (have: {sorted(raw)})"
                    )
                yield {k: raw[k] for k in _WANTED_PROMPT_COLS if k in raw}


def _guess_text_field(example: dict) -> str:
    preferred = ("text", "content", "prompt", "input", "sentence", "article")
    for name in preferred:
        if name in example and isinstance(example[name], str):
            return name
    for name, val in example.items():
        if isinstance(val, str):
            return name
    raise ValueError(
        f"dataset example has no string field to tokenize; columns: {list(example)}"
    )


__all__ = ["Dataset", "write_jsonl_shard"]
