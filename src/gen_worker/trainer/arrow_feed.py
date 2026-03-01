from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Protocol


@dataclass(frozen=True)
class ArrowFeedConfig:
    batch_size: int = 32
    readahead: int = 2
    columns: tuple[str, ...] = ()


class ArrowBatchFeeder(Protocol):
    def iter_batches(self) -> Iterable[Any]:
        ...


class ParquetArrowBatchFeeder:
    """Parquet-first batch feeder using PyArrow scanner batches."""

    def __init__(self, parquet_paths: list[str], config: ArrowFeedConfig | None = None) -> None:
        self.parquet_paths = [str(Path(p)) for p in parquet_paths]
        self.config = config or ArrowFeedConfig()

    def iter_batches(self) -> Iterator[Any]:
        try:
            import pyarrow.dataset as ds
        except Exception as e:  # pragma: no cover - requires optional dependency
            raise RuntimeError(
                "pyarrow is required for parquet trainer mode; install with training extras"
            ) from e

        scanner = ds.dataset(self.parquet_paths, format="parquet").scanner(
            columns=list(self.config.columns) or None,
            batch_size=self.config.batch_size,
            batch_readahead=self.config.readahead,
        )
        yield from scanner.to_batches()


__all__ = ["ArrowBatchFeeder", "ArrowFeedConfig", "ParquetArrowBatchFeeder"]
