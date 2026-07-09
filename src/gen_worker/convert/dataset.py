"""Dataset — library-constructed handle to a materialized dataset snapshot.

Parallels Source but for datasets. Constructed from ``payload.datasets[i]``
entries by the library before invoking the tenant function. Tenants access
via the reserved ``datasets: list[Dataset]`` parameter.

Used by calibration-based quant (GPTQ/AWQ/modelopt-ptq-static), pruning with
gradient scoring, distillation (both teacher/student training loops), and
fine-tuning (LoRA / full-parameter / continued pretraining).

## Supported artifact shapes

Three shapes are recognized. All three live on local disk after the library
has materialized the ``DatasetRef`` — tenants see a unified interface.

### 1. HF-datasets layout (``load_from_disk``-compatible)

Typical for LLM calibration (wikitext / c4) and fine-tuning (image+caption /
instruction data). Exposed via ``iter_examples`` / ``as_dataloader`` /
``as_hf_dataset``.

### 2. Prompt corpus

```
<root>/
├── dataset_info.json      # {kind: "prompt_corpus", features, num_rows, ...}
└── data/
    └── train-00000.parquet     # columns: prompt, category, length_bucket, seed
```

One corpus can be reused across source models and calibrated-quant recipes.
Exposed via ``is_prompt_corpus`` / ``dataset_info`` / ``iter_prompts``.

### 3. Eval set

```
<root>/
├── dataset_info.json      # {kind: "eval_set", features, variants, prompt_corpus, ...}
└── data/
    └── train-00000.parquet     # columns: prompt, seed, category, image_<variant>: Image
```

Source+variant-specific eval sets can hold side-by-side renders for A/B eval.
Exposed via ``is_eval_set`` / ``dataset_info`` / ``iter_rows``.

## Detection

``is_prompt_corpus`` / ``is_eval_set`` feature-detect via
``dataset_info.json`` at the snapshot root. Earlier revisions (#41 shipped
with ``manifest.json`` + ``prompts.jsonl``) are handled transparently for
readback compat — we do NOT require callers to migrate existing artifacts,
but new artifacts write ``dataset_info.json`` + parquet.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

if TYPE_CHECKING:
    from torch.utils.data import DataLoader
    import datasets as hf_datasets


class Dataset:
    """Handle to a materialized dataset snapshot.

    Constructed by the library from ``payload.datasets[i]`` (a ``DatasetRef``)
    + the resolved dataset-variant snapshot on local disk.

    Public surface:
      ref, split, attributes, path   -- simple accessors
      iter_examples()                -- yield raw dataset rows (HF layout)
      as_dataloader(...)             -- torch DataLoader of tokenized batches
                                        (for modelopt / GPTQ / AWQ LLM
                                         calibration)
      as_hf_dataset(...)             -- tokenized datasets.Dataset
                                        (for transformers.Trainer / peft /
                                         accelerate training flows)

    Prompt-corpus + eval-set shapes:

      dataset_info()                 -- parsed dataset_info.json (the new
                                        format) or legacy manifest.json
      kind                           -- "prompt_corpus" | "eval_set" | ""
      is_prompt_corpus()             -- True if kind == "prompt_corpus"
      is_eval_set()                  -- True if kind == "eval_set"
      iter_prompts()                 -- yield prompt rows from the parquet
                                        data/ shard (or legacy prompts.jsonl)
      iter_rows()                    -- yield full rows with image columns
                                        (for eval sets)
      parquet_shards()               -- list[Path] of parquet files under data/
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

    def as_hf_dataset(
        self,
        *,
        tokenizer: Any | None = None,
        max_seq_length: int | None = None,
    ) -> "hf_datasets.Dataset":
        """Return a ``datasets.Dataset`` for the declared split.

        If ``tokenizer`` + ``max_seq_length`` are passed, eagerly tokenize the
        text column and add ``input_ids`` + ``attention_mask`` columns (the
        shape ``transformers.Trainer`` expects). If omitted, return the raw
        dataset with no tokenization — for tenants that handle tokenization
        themselves or use trl.SFTTrainer (which tokenizes internally).
        """
        import datasets as hf_datasets

        ds = hf_datasets.load_from_disk(str(self._path))
        if isinstance(ds, hf_datasets.DatasetDict):
            ds = ds[self._split]
        if tokenizer is None or max_seq_length is None:
            return ds
        text_field = _guess_text_field(ds[0])

        def _tok(batch: dict) -> dict:
            return tokenizer(
                batch[text_field],
                truncation=True, max_length=max_seq_length, padding="max_length",
            )

        return ds.map(_tok, batched=True, remove_columns=[text_field])

    # ---- prompt-corpus + eval-set ------------------------------

    def dataset_info(self) -> dict:
        """Return the parsed ``dataset_info.json`` at ``path`` or ``{}``.

        The layout writes a ``dataset_info.json`` at the snapshot
        root with ``{kind, features, num_rows, ...}``. HF-datasets
        snapshots and LLM-text datasets don't have this file —
        ``dataset_info()`` returns ``{}`` for them so callers can use
        ``dataset_info().get("kind")`` as a feature test.

        Readback compat: if ``dataset_info.json`` is absent but the legacy
        ``manifest.json`` from the shipped-then-rolled-back #41 format is
        present, return that. ``kind`` is normalized to values
        (``calibration_dataset`` → ``prompt_corpus``).
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

    def is_eval_set(self) -> bool:
        """True iff this artifact is an eval set."""
        return self.kind == "eval_set"

    def parquet_shards(self) -> list[Path]:
        """Return the list of parquet shard files under ``data/``.

        Convention: ``<root>/data/train-*.parquet``. Returns sorted shard
        paths so iteration is deterministic. Empty list if the snapshot
        isn't in the parquet layout (e.g. legacy jsonl shape).
        """
        data_dir = self._path / "data"
        if not data_dir.is_dir():
            return []
        return sorted(data_dir.glob("*.parquet"))

    def iter_prompts(self) -> Iterator[dict]:
        """Yield prompt rows — ``{prompt, category, length_bucket, seed}``.

        Primary path: read the parquet shard(s) under ``data/`` using
        pyarrow, pushing down the prompt + metadata columns only. Image
        columns on eval sets are skipped automatically — calibration
        reads only the text it needs.

        Fallback: if the snapshot is in the legacy shape from the
        rolled-back #41 format (``prompts.jsonl`` at the root), iterate
        that directly. Kept for forward-compat on any dataset that might
        still be in flight.

        Raises ``FileNotFoundError`` if neither shape is present —
        callers should guard with ``is_prompt_corpus() or is_eval_set()``.
        """
        shards = self.parquet_shards()
        if shards:
            yield from _iter_parquet_prompt_columns(shards)
            return
        legacy = self._path / "prompts.jsonl"
        if legacy.exists():
            with open(legacy) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    yield json.loads(line)
            return
        raise FileNotFoundError(
            f"dataset {self._ref!r} has no prompts — expected parquet "
            f"shards under {self._path}/data/ or legacy prompts.jsonl"
        )

    def iter_rows(self) -> Iterator[dict]:
        """Yield ALL rows (including image columns on eval sets).

        Unlike ``iter_prompts``, this does not push down columns — image
        bytes are decoded for each row. Suitable for eval-set consumers
        (human review UI, VLM eval jobs) that need the rendered outputs.

        For prompt-corpus datasets this is equivalent to ``iter_prompts``
        since there are no image columns.
        """
        shards = self.parquet_shards()
        if not shards:
            # Fall back to iter_prompts for legacy jsonl.
            yield from self.iter_prompts()
            return
        import pyarrow.parquet as pq
        for shard in shards:
            table = pq.read_table(str(shard))
            for row in table.to_pylist():
                yield row

    # ---- legacy aliases ------------------------------------------

    def manifest(self) -> dict:
        """Back-compat alias: parsed ``dataset_info.json`` or legacy
        ``manifest.json`` at ``path``, whichever exists.

        ``dataset_info.json`` is the canonical name; ``manifest()`` now
        returns the same dict as ``dataset_info()``. Kept as an alias for
        older callers.
        """
        return self.dataset_info()

    def is_calibration_dataset(self) -> bool:
        """Deprecated alias for ``is_prompt_corpus()``.

        New code should use ``is_prompt_corpus()``.
        """
        return self.is_prompt_corpus()


def _load_dataset_info(path: Path) -> dict | None:
    """Read ``dataset_info.json`` at ``path``, falling back to
    ``manifest.json`` (legacy #41 shape) with field normalization.

    Returns ``None`` when neither file exists or either is unparseable —
    the caller treats that as "not a prompt-corpus / eval-set artifact."
    """
    info_path = path / "dataset_info.json"
    if info_path.exists():
        try:
            with open(info_path) as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            return None

    # Legacy shape: manifest.json with kind=calibration_dataset. Normalize
    # the kind to the current vocabulary so downstream branching still fires.
    legacy = path / "manifest.json"
    if legacy.exists():
        try:
            with open(legacy) as f:
                data = json.load(f)
            if isinstance(data, dict):
                kind = str(data.get("kind") or "")
                if kind == "calibration_dataset":
                    data = dict(data)
                    data["kind"] = "prompt_corpus"
                return data
        except json.JSONDecodeError:
            return None
    return None


def _iter_parquet_prompt_columns(shards: list[Path]) -> Iterator[dict]:
    """Yield rows from parquet shards using the pyarrow column pushdown.

    Reads only the lightweight text/metadata columns present in the schema
    (prompt / category / length_bucket / seed). Image columns — which on
    an eval set can be tens of MB per row — are NOT materialized. This
    keeps the calibration-read cost linear in prompt count, not total
    snapshot size.
    """
    import pyarrow.parquet as pq

    wanted_cols = ("prompt", "category", "length_bucket", "seed")
    for shard in shards:
        pf = pq.ParquetFile(str(shard))
        available = set(pf.schema_arrow.names)
        cols = [c for c in wanted_cols if c in available]
        if "prompt" not in cols:
            raise ValueError(
                f"parquet shard {shard} is missing the required 'prompt' "
                f"column (have: {sorted(available)})"
            )
        for batch in pf.iter_batches(columns=cols):
            for row in batch.to_pylist():
                yield row


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


__all__ = ["Dataset"]
