"""Datasets and dataloader helpers for token sequence parquet files."""

from __future__ import annotations

import json
import logging
import platform
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import (
    ConcatDataset,
    DataLoader,
    Dataset,
    IterableDataset,
    get_worker_info,
    random_split,
)

from heptokens.data.sequence import (
    LABELS_KEY,
    MASK_KEY,
    TOKENS_KEY,
    TYPE_IDS_KEY,
    first_existing_column,
)
from heptokens.data.atlas_mappable import BaseMapModule
from heptokens.data.collation import collate_and_transform

log = logging.getLogger(__name__)
NUM_WORKERS = 0 if platform.system() == "Darwin" else 2


class TokenParquetDataset(Dataset):
    """Load tokenized sequences from parquet files.

    The dataset normalizes parquet column names to a generic sequence batch:
    ``tokens``, ``mask``, optional ``type_ids``, and optional ``labels``.
    Preferred parquet columns are ``tokens``, ``mask``, and optional ``type_ids``.
    Legacy columns from the old standalone tokenizer are also accepted:
    ``input_ids``, ``attention_mask``, and ``token_type_ids``.
    """

    def __init__(
        self,
        parquet_path: str,
        label: int | None = None,
        *,
        token_column: str | None = None,
        mask_column: str | None = None,
        type_column: str | None = None,
        label_column: str | None = None,
    ) -> None:
        import pyarrow.parquet as pq

        log.info("Loading %s%s...", parquet_path, "" if label is None else f" (label={label})")
        t0 = time.time()
        table = pq.read_table(parquet_path)
        self.n = len(table)
        token_column = token_column or first_existing_column(
            table.column_names,
            ["tokens", "input_ids"],
        )
        mask_column = mask_column or first_existing_column(
            table.column_names,
            ["mask", "attention_mask"],
        )
        type_column = type_column or first_existing_column(
            table.column_names,
            ["type_ids", "token_type_ids"],
            required=False,
        )

        self.tokens = np.array(table[token_column].to_pylist(), dtype=np.int64)
        self.mask = np.array(table[mask_column].to_pylist(), dtype=bool)

        if type_column and type_column in table.column_names:
            self.type_ids = np.array(table[type_column].to_pylist(), dtype=np.int64)
        else:
            self.type_ids = np.zeros_like(self.tokens, dtype=np.int64)

        if label is not None:
            self.labels = np.full(self.n, label, dtype=np.int64)
        elif label_column and label_column in table.column_names:
            self.labels = np.array(table[label_column].to_pylist(), dtype=np.int64)
        else:
            self.labels = None

        log.info("  Loaded %s sequences in %.1fs", self.n, time.time() - t0)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> dict:
        batch = {
            TOKENS_KEY: torch.tensor(self.tokens[idx], dtype=torch.long),
            MASK_KEY: torch.tensor(self.mask[idx], dtype=torch.bool),
            TYPE_IDS_KEY: torch.tensor(self.type_ids[idx], dtype=torch.long),
        }
        if self.labels is not None:
            batch[LABELS_KEY] = torch.tensor(self.labels[idx], dtype=torch.long)
        return batch


@dataclass(frozen=True)
class _ParquetFileSpec:
    path: str
    num_rows: int
    row_group_rows: tuple[int, ...]
    row_group_offsets: tuple[int, ...]
    token_column: str
    mask_column: str
    type_column: str | None
    global_offset: int


def _split_hash(indices: np.ndarray, seed: int) -> np.ndarray:
    """Map row indices to deterministic, well-mixed uint64 values."""
    values = indices.astype(np.uint64, copy=False) + np.uint64(seed)
    values ^= values >> np.uint64(30)
    values *= np.uint64(0xBF58476D1CE4E5B9)
    values ^= values >> np.uint64(27)
    values *= np.uint64(0x94D049BB133111EB)
    values ^= values >> np.uint64(31)
    return values


def _sequence_array_to_numpy(array, dtype: np.dtype) -> np.ndarray:
    """Convert fixed-size Arrow sequence arrays without constructing Python lists."""
    import pyarrow as pa

    if pa.types.is_fixed_size_list(array.type):
        values = np.asarray(array.values.to_numpy(zero_copy_only=False), dtype=dtype)
        return values.reshape(len(array), array.type.list_size)
    return np.asarray(array.to_pylist(), dtype=dtype)


class StreamingTokenParquetDataset(IterableDataset):
    """Stream token sequences from Parquet row groups with bounded memory.

    Rows are assigned to train/validation splits by a deterministic hash of
    their global row index. Workers and distributed ranks shard row groups, so
    no worker loads or yields another worker's row-group data.
    """

    def __init__(
        self,
        parquet_files: list[str],
        *,
        split_start: float,
        split_end: float,
        seed: int = 42,
        max_rows: int | None = None,
        stream_batch_size: int = 4096,
        shuffle: bool = False,
        reshuffle_each_iteration: bool = True,
        token_column: str | None = None,
        mask_column: str | None = None,
        type_column: str | None = None,
    ) -> None:
        super().__init__()
        import pyarrow.parquet as pq

        if not parquet_files:
            raise ValueError("At least one parquet file is required")
        if not 0.0 <= split_start < split_end <= 1.0:
            raise ValueError("Split bounds must satisfy 0 <= start < end <= 1")
        if stream_batch_size <= 0:
            raise ValueError("stream_batch_size must be positive")

        specs = []
        global_offset = 0
        for path in parquet_files:
            parquet = pq.ParquetFile(path)
            names = parquet.schema_arrow.names
            resolved_tokens = token_column or first_existing_column(
                names, ["tokens", "input_ids"]
            )
            resolved_mask = mask_column or first_existing_column(
                names, ["mask", "attention_mask"]
            )
            resolved_types = type_column or first_existing_column(
                names, ["type_ids", "token_type_ids"], required=False
            )
            row_group_rows = tuple(
                parquet.metadata.row_group(index).num_rows
                for index in range(parquet.metadata.num_row_groups)
            )
            offsets = []
            offset = 0
            for rows in row_group_rows:
                offsets.append(offset)
                offset += rows
            specs.append(
                _ParquetFileSpec(
                    path=str(path),
                    num_rows=parquet.metadata.num_rows,
                    row_group_rows=row_group_rows,
                    row_group_offsets=tuple(offsets),
                    token_column=resolved_tokens,
                    mask_column=resolved_mask,
                    type_column=resolved_types,
                    global_offset=global_offset,
                )
            )
            global_offset += parquet.metadata.num_rows

        self.specs = tuple(specs)
        self.total_rows = global_offset
        self.split_start = split_start
        self.split_end = split_end
        self.seed = seed
        self.stream_batch_size = stream_batch_size
        self.shuffle = shuffle
        self.reshuffle_each_iteration = reshuffle_each_iteration
        self._iteration = 0
        expected_rows = int(round(self.total_rows * (split_end - split_start)))
        self.max_rows = min(max_rows, expected_rows) if max_rows is not None else None
        self.expected_rows = self.max_rows if self.max_rows is not None else expected_rows

        log.info(
            "Streaming %d parquet rows from %d files for split [%.3f, %.3f); expected rows=%d",
            self.total_rows,
            len(self.specs),
            self.split_start,
            self.split_end,
            self.expected_rows,
        )

    def __len__(self) -> int:
        return self.expected_rows

    def _shard(self) -> tuple[int, int]:
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        num_workers = worker.num_workers if worker is not None else 1
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            rank = 0
            world_size = 1
        return rank * num_workers + worker_id, world_size * num_workers

    def _worker_limit(self, shard_id: int, num_shards: int) -> int | None:
        if self.max_rows is None:
            return None
        base, remainder = divmod(self.max_rows, num_shards)
        return base + int(shard_id < remainder)

    def __iter__(self):
        import pyarrow.parquet as pq

        shard_id, num_shards = self._shard()
        worker_limit = self._worker_limit(shard_id, num_shards)
        if worker_limit == 0:
            return

        work = [
            (file_index, row_group_index)
            for file_index, spec in enumerate(self.specs)
            for row_group_index in range(len(spec.row_group_rows))
        ]
        worker = get_worker_info()
        if not self.reshuffle_each_iteration:
            iteration_seed = 0
        elif worker is not None:
            iteration_seed = torch.initial_seed()
        else:
            iteration_seed = self._iteration
            self._iteration += 1
        rng = np.random.default_rng(
            (self.seed + shard_id + iteration_seed) % np.iinfo(np.uint64).max
        )
        if self.shuffle:
            rng.shuffle(work)
        work = work[shard_id::num_shards]

        yielded = 0
        denominator = 2**64
        lower = (
            np.uint64(int(self.split_start * denominator))
            if self.split_start > 0.0
            else None
        )
        upper = (
            np.uint64(int(self.split_end * denominator))
            if self.split_end < 1.0
            else None
        )

        for file_index, row_group_index in work:
            spec = self.specs[file_index]
            parquet = pq.ParquetFile(spec.path)
            columns = [spec.token_column, spec.mask_column]
            if spec.type_column is not None:
                columns.append(spec.type_column)

            batch_offset = 0
            for batch in parquet.iter_batches(
                batch_size=self.stream_batch_size,
                row_groups=[row_group_index],
                columns=columns,
            ):
                row_start = (
                    spec.global_offset
                    + spec.row_group_offsets[row_group_index]
                    + batch_offset
                )
                global_indices = np.arange(
                    row_start,
                    row_start + batch.num_rows,
                    dtype=np.uint64,
                )
                hashes = _split_hash(global_indices, self.seed)
                in_split = np.ones(batch.num_rows, dtype=bool)
                if lower is not None:
                    in_split &= hashes >= lower
                if upper is not None:
                    in_split &= hashes < upper
                selected = np.flatnonzero(in_split)
                batch_offset += batch.num_rows
                if len(selected) == 0:
                    continue
                if self.shuffle:
                    rng.shuffle(selected)

                token_index = batch.schema.get_field_index(spec.token_column)
                mask_index = batch.schema.get_field_index(spec.mask_column)
                tokens = _sequence_array_to_numpy(
                    batch.column(token_index), np.dtype(np.int64)
                )
                masks = _sequence_array_to_numpy(
                    batch.column(mask_index), np.dtype(bool)
                )
                if spec.type_column is not None:
                    type_index = batch.schema.get_field_index(spec.type_column)
                    type_ids = _sequence_array_to_numpy(
                        batch.column(type_index), np.dtype(np.int64)
                    )
                else:
                    type_ids = np.zeros_like(tokens, dtype=np.int64)

                for index in selected:
                    yield {
                        TOKENS_KEY: torch.from_numpy(tokens[index]),
                        MASK_KEY: torch.from_numpy(masks[index]),
                        TYPE_IDS_KEY: torch.from_numpy(type_ids[index]),
                    }
                    yielded += 1
                    if worker_limit is not None and yielded >= worker_limit:
                        return


def make_classification_loaders(
    signal_parquets: list[str],
    background_parquets: list[str],
    *,
    batch_size: int = 256,
    max_sequences: int | None = None,
    seed: int = 42,
    token_column: str | None = None,
    mask_column: str | None = None,
    type_column: str | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    dataset_kwargs = {
        "token_column": token_column,
        "mask_column": mask_column,
        "type_column": type_column,
    }
    datasets = [TokenParquetDataset(fp, label=1, **dataset_kwargs) for fp in signal_parquets]
    datasets.extend(
        TokenParquetDataset(fp, label=0, **dataset_kwargs) for fp in background_parquets
    )
    full = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]

    if max_sequences and len(full) > max_sequences:
        full, _ = random_split(
            full,
            [max_sequences, len(full) - max_sequences],
            generator=torch.Generator().manual_seed(99),
        )

    total = len(full)
    train_size = int(0.7 * total)
    val_size = int(0.15 * total)
    test_size = total - train_size - val_size
    train_set, val_set, test_set = random_split(
        full,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed),
    )
    kwargs = {"batch_size": batch_size, "num_workers": NUM_WORKERS}
    return (
        DataLoader(train_set, shuffle=True, **kwargs),
        DataLoader(val_set, shuffle=False, **kwargs),
        DataLoader(test_set, shuffle=False, **kwargs),
    )


def make_pretrain_loaders(
    parquet_files: list[str],
    *,
    batch_size: int = 256,
    max_sequences: int | None = None,
    seed: int = 42,
    token_column: str | None = None,
    mask_column: str | None = None,
    type_column: str | None = None,
    stream_batch_size: int = 4096,
) -> tuple[DataLoader, DataLoader]:
    total_limit = max_sequences if max_sequences and max_sequences > 0 else None
    train_limit = int(0.9 * total_limit) if total_limit is not None else None
    val_limit = total_limit - train_limit if total_limit is not None else None
    dataset_kwargs = dict(
        parquet_files=parquet_files,
        seed=seed,
        stream_batch_size=stream_batch_size,
        token_column=token_column,
        mask_column=mask_column,
        type_column=type_column,
    )
    train_set = StreamingTokenParquetDataset(
        split_start=0.0,
        split_end=0.9,
        max_rows=train_limit,
        shuffle=True,
        **dataset_kwargs,
    )
    val_set = StreamingTokenParquetDataset(
        split_start=0.9,
        split_end=1.0,
        max_rows=val_limit,
        shuffle=True,
        reshuffle_each_iteration=False,
        **dataset_kwargs,
    )
    kwargs = {"batch_size": batch_size, "num_workers": NUM_WORKERS}
    return (
        DataLoader(train_set, **kwargs),
        DataLoader(val_set, **kwargs),
    )


class TokenParquetClassificationModule(BaseMapModule):
    """DataModule for labelled signal/background token parquet files."""

    def __init__(
        self,
        *,
        signal_parquets: list[str],
        background_parquets: list[str],
        train_frac: float = 0.7,
        val_frac: float = 0.15,
        test_frac: float = 0.15,
        seed: int = 42,
        max_sequences: int | None = None,
        token_column: str | None = None,
        mask_column: str | None = None,
        type_column: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if not abs(train_frac + val_frac + test_frac - 1.0) < 1e-6:
            raise ValueError("train_frac + val_frac + test_frac must sum to 1.0")

        dataset_kwargs = {
            "token_column": token_column,
            "mask_column": mask_column,
            "type_column": type_column,
        }
        datasets = [TokenParquetDataset(fp, label=1, **dataset_kwargs) for fp in signal_parquets]
        datasets.extend(
            TokenParquetDataset(fp, label=0, **dataset_kwargs) for fp in background_parquets
        )
        full = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]
        if max_sequences and len(full) > max_sequences:
            full, _ = random_split(
                full,
                [max_sequences, len(full) - max_sequences],
                generator=torch.Generator().manual_seed(seed),
            )

        total = len(full)
        train_size = int(train_frac * total)
        val_size = int(val_frac * total)
        test_size = total - train_size - val_size
        self.train_set, self.valid_set, self.test_set = random_split(
            full,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(seed),
        )

    def setup(self, stage: str) -> None:
        pass


class TokenParquetPretrainModule(BaseMapModule):
    """Stream prepared, pre-split Parquet shards for masked pretraining."""

    def __init__(
        self,
        *,
        prepared_dir: str | None = None,
        train_parquet_files: list[str] | None = None,
        val_parquet_files: list[str] | None = None,
        parquet_files: list[str] | None = None,
        train_frac: float = 0.9,
        val_frac: float = 0.1,
        seed: int = 42,
        max_sequences: int | None = None,
        token_column: str | None = None,
        mask_column: str | None = None,
        type_column: str | None = None,
        stream_batch_size: int = 4096,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if not abs(train_frac + val_frac - 1.0) < 1e-6:
            raise ValueError("train_frac + val_frac must sum to 1.0")

        manifest = None
        if prepared_dir is not None:
            prepared_path = Path(prepared_dir)
            manifest_path = prepared_path / "manifest.json"
            if not manifest_path.is_file():
                raise FileNotFoundError(
                    f"Prepared pretraining manifest is missing: {manifest_path}"
                )
            manifest = json.loads(manifest_path.read_text())
            train_parquet_files = sorted(
                str(path) for path in (prepared_path / "train").glob("*.parquet")
            )
            val_parquet_files = sorted(
                str(path) for path in (prepared_path / "val").glob("*.parquet")
            )
        if not train_parquet_files or not val_parquet_files:
            if parquet_files:
                raise ValueError(
                    "Raw parquet_files are no longer accepted for production pretraining. "
                    "Run scripts/prepare_token_parquet_pretrain_shards.py, then set "
                    "datamodule.prepared_dir or explicit train_parquet_files and "
                    "val_parquet_files."
                )
            raise ValueError(
                "Provide prepared_dir or both train_parquet_files and val_parquet_files"
            )

        total_limit = max_sequences if max_sequences and max_sequences > 0 else None
        if manifest is not None and manifest.get("total_rows"):
            limit_train_frac = manifest["train_rows"] / manifest["total_rows"]
        else:
            limit_train_frac = train_frac
        train_limit = (
            int(total_limit * limit_train_frac) if total_limit is not None else None
        )
        val_limit = total_limit - train_limit if total_limit is not None else None
        common_kwargs = dict(
            seed=seed,
            stream_batch_size=stream_batch_size,
            token_column=token_column,
            mask_column=mask_column,
            type_column=type_column,
        )
        self.train_set = StreamingTokenParquetDataset(
            parquet_files=[str(path) for path in train_parquet_files],
            split_start=0.0,
            split_end=1.0,
            max_rows=train_limit,
            shuffle=True,
            **common_kwargs,
        )
        self.valid_set = StreamingTokenParquetDataset(
            parquet_files=[str(path) for path in val_parquet_files],
            split_start=0.0,
            split_end=1.0,
            max_rows=val_limit,
            shuffle=True,
            reshuffle_each_iteration=False,
            **common_kwargs,
        )
        self.test_set = self.valid_set
        if manifest is not None:
            if self.train_set.total_rows != manifest.get("train_rows"):
                raise ValueError(
                    "Prepared train shard rows do not match manifest: "
                    f"{self.train_set.total_rows} != {manifest.get('train_rows')}"
                )
            if self.valid_set.total_rows != manifest.get("val_rows"):
                raise ValueError(
                    "Prepared validation shard rows do not match manifest: "
                    f"{self.valid_set.total_rows} != {manifest.get('val_rows')}"
                )
            log.info(
                "Prepared token dataset verified: train=%d val=%d sources=%d",
                self.train_set.total_rows,
                self.valid_set.total_rows,
                len(manifest.get("source_counts", {})),
            )

    def setup(self, stage: str) -> None:
        pass

    def _get_dataloader(
        self,
        dataset: IterableDataset,
        shuffle: bool,
        drop_last: bool,
        sampler=None,
    ) -> DataLoader:
        collate_fn = None
        if self.transforms is not None:
            collate_fn = partial(collate_and_transform, transforms=self.transforms)

        dataloader_kwargs = {}
        if self.num_workers > 0:
            dataloader_kwargs["persistent_workers"] = self.persistent_workers
            if self.multiprocessing_context is not None:
                dataloader_kwargs["multiprocessing_context"] = self.multiprocessing_context
        if sampler is not None:
            log.warning("Ignoring sampler for streaming token parquet dataloader")

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=drop_last,
            collate_fn=collate_fn,
            **dataloader_kwargs,
        )

    def get_data_sample(self) -> dict:
        for dataset in (self.valid_set, self.train_set):
            try:
                return next(iter(dataset))
            except StopIteration:
                continue
        raise RuntimeError("No token sequences available in the parquet files")
