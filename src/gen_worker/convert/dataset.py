"""Dataset — handle to a dataset snapshot on local disk.

Parallels Source but for datasets. Constructed from a local snapshot path
(e.g. one returned by ``ctx.resolve_dataset``).

Used by calibration-based quant (GPTQ/AWQ/modelopt-ptq-static), pruning with
gradient scoring, distillation (both teacher/student training loops), and
fine-tuning (LoRA / full-parameter / continued pretraining).

## Supported artifact shapes

Three shapes are recognized; all three live on local disk — tenants see a
unified interface.

### 1. HF-datasets layout (``load_from_disk``-compatible)

Typical for LLM calibration (wikitext / c4) and fine-tuning (image+caption /
instruction data). Exposed via ``iter_examples`` / ``as_dataloader``.

### 2. Prompt corpus

```
<root>/
├── dataset_info.json      # {kind: "prompt_corpus", features, num_rows, ...}
└── data/
    └── train-00000.jsonl     # one row per line: prompt, category, length_bucket, seed
```

One corpus can be reused across source models and calibrated-quant recipes.
Exposed via ``is_prompt_corpus`` / ``dataset_info`` / ``iter_prompts``.

### 3. Eval set

```
<root>/
├── dataset_info.json      # {kind: "eval_set", features, variants, prompt_corpus, ...}
└── data/
    └── train-00000.jsonl     # one row per line: prompt, seed, category, image_<variant>
```

Source+variant-specific eval sets can hold side-by-side renders for A/B eval.
Exposed via ``dataset_info`` / ``iter_rows``.

## Detection

``is_prompt_corpus`` / ``Dataset.kind`` feature-detect via
``dataset_info.json`` at the snapshot root.

## JSONL row encoding

Rows are one JSON object per line. ``bytes``/``bytearray`` values (e.g. an
``image`` column on an edit corpus) are base64-wrapped as
``{"__bytes_b64__": "..."}`` on write and transparently decoded back to
``bytes`` on read — see ``write_jsonl_shard`` / ``_decode_row``.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

_BYTES_KEY = "__bytes_b64__"


class Dataset:
    """Handle to a dataset snapshot on local disk.

    Public surface:
      ref, split, attributes, path   -- simple accessors
      iter_examples()                -- yield raw dataset rows (HF layout)
      as_dataloader(...)             -- torch DataLoader of tokenized batches
                                        (for modelopt / GPTQ / AWQ LLM
                                         calibration)
    Prompt-corpus + eval-set shapes:

      dataset_info()                 -- parsed dataset_info.json
      kind                           -- "prompt_corpus" | "eval_set" | ""
      is_prompt_corpus()             -- True if kind == "prompt_corpus"
      iter_prompts()                 -- yield prompt rows from the jsonl
                                        data/ shard
      iter_rows()                    -- yield full rows with image columns
                                        (for eval sets)
      shards()                       -- list[Path] of jsonl files under data/
    """

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
        """Yield raw dataset rows as dicts.

        Uses the HF ``datasets`` library for reading; the materialized snapshot
        must be in a format ``datasets.load_from_disk`` or ``datasets.Dataset.from_*``
        understands. Library materialization ensures this.
        """
        import datasets as hf_datasets
        ds = hf_datasets.load_from_disk(str(self._path))
        # Some materializations store as DatasetDict keyed by split name
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
        """Return a torch DataLoader of tokenized batches for calibration.

        Tokenizes the first ``n`` examples to ``seq_length`` tokens each,
        batches at ``batch_size``. Deterministic seeded shuffle. Suitable
        input for modelopt's ``forward_loop`` and GPTQ/AWQ calibration.
        """
        import random

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

    # ---- prompt-corpus + eval-set ------------------------------

    def dataset_info(self) -> dict:
        """Return the parsed ``dataset_info.json`` at ``path`` or ``{}``.

        The layout writes a ``dataset_info.json`` at the snapshot
        root with ``{kind, features, num_rows, ...}``. HF-datasets
        snapshots and LLM-text datasets don't have this file —
        ``dataset_info()`` returns ``{}`` for them so callers can use
        ``dataset_info().get("kind")`` as a feature test.
        """
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
        """Return the list of jsonl shard files under ``data/``.

        Convention: ``<root>/data/train-*.jsonl``. Returns sorted shard
        paths so iteration is deterministic. Empty list if the snapshot
        isn't in the prompt-corpus / eval-set layout.
        """
        data_dir = self._path / "data"
        if not data_dir.is_dir():
            return []
        return sorted(data_dir.glob("*.jsonl"))

    def iter_prompts(self) -> Iterator[dict]:
        """Yield prompt rows — ``{prompt, category, length_bucket, seed}``.

        Reads the jsonl shard(s) under ``data/`` one row at a time, keeping
        only the prompt + metadata fields. Image columns on eval sets are
        dropped per row rather than materialized — calibration reads only
        the text it needs.

        Raises ``FileNotFoundError`` if no jsonl shards are present —
        callers should guard with ``is_prompt_corpus()`` or
        ``kind == "eval_set"``.
        """
        shards = self.shards()
        if not shards:
            raise FileNotFoundError(
                f"dataset {self._ref!r} has no prompts — expected jsonl "
                f"shards under {self._path}/data/"
            )
        yield from _iter_jsonl_prompt_columns(shards)

    def iter_rows(self) -> Iterator[dict]:
        """Yield ALL rows (including image columns on eval sets).

        Unlike ``iter_prompts``, this decodes every field — image bytes
        are base64-decoded for each row. Suitable for eval-set consumers
        (human review UI, VLM eval jobs) that need the rendered outputs.

        For prompt-corpus datasets this is equivalent to ``iter_prompts``
        since there are no image columns.
        """
        shards = self.shards()
        if not shards:
            yield from self.iter_prompts()  # raises FileNotFoundError
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
    """Write ``rows`` as a jsonl shard at ``<out_dir>/data/<name>``.

    One JSON object per line. ``bytes``/``bytearray`` values (e.g. a PNG
    ``image`` column on an edit corpus) are base64-wrapped transparently —
    see ``_decode_row`` for the read side.
    """
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
    """Read ``dataset_info.json`` at ``path``.

    Returns ``None`` when the file doesn't exist or is unparseable — the
    caller treats that as "not a prompt-corpus / eval-set artifact."
    """
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
    """Yield rows from jsonl shards, keeping only the prompt + metadata
    fields (``prompt`` / ``category`` / ``length_bucket`` / ``seed``).

    Image columns — which on an eval set can be tens of MB per row — are
    dropped immediately after each row is parsed, not materialized further.
    """
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
    """Pick the most likely text column from a dataset row.

    Preference order: 'text' > 'content' > 'prompt' > 'input' > first string
    field. Raises if no string field is found.
    """
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
