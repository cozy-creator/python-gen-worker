"""Dataset — library-constructed handle to a materialized dataset snapshot.

Parallels Source but for datasets. Constructed from ``payload.datasets[i]``
entries by the library before invoking the tenant function. Tenants access
via the reserved ``datasets: list[Dataset]`` parameter.

Used by calibration-based quant (GPTQ/AWQ/modelopt-ptq-static), pruning with
gradient scoring, distillation (both teacher/student training loops), and
fine-tuning (LoRA / full-parameter / continued pretraining).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Iterator, Optional

if TYPE_CHECKING:
    from torch.utils.data import DataLoader
    import datasets as hf_datasets


class Dataset:
    """Handle to a materialized dataset snapshot.

    Constructed by the library from ``payload.datasets[i]`` (a ``DatasetRef``)
    + the resolved dataset-variant snapshot on local disk.

    Public surface:
      ref, split, attributes, path  -- simple accessors
      iter_examples()               -- yield raw dataset rows
      as_dataloader(...)            -- torch DataLoader of tokenized batches
                                       (for modelopt / GPTQ / AWQ calibration)
      as_hf_dataset(...)            -- tokenized datasets.Dataset
                                       (for transformers.Trainer / peft /
                                        accelerate training flows)
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
