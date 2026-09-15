"""Datasets and dataloader helpers for token sequence parquet files."""

from __future__ import annotations

import json
import logging
import os
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
TOKEN_VOCABULARY_METADATA_KEY = b"heptokens_token_vocabulary"


def _read_token_vocabulary(parquet_path: str) -> dict | None:
    """Read the grouped-token vocabulary stored in Parquet schema metadata."""
    import pyarrow.parquet as pq

    metadata = pq.ParquetFile(parquet_path).schema_arrow.metadata or {}
    encoded = metadata.get(TOKEN_VOCABULARY_METADATA_KEY)
    return json.loads(encoded) if encoded is not None else None


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
    label_column: str | None
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
        if pa.types.is_list(array.type.value_type) or pa.types.is_fixed_size_list(
            array.type.value_type
        ):
            return np.asarray(array.to_pylist(), dtype=dtype)
        values = np.asarray(array.values.to_numpy(zero_copy_only=False), dtype=dtype)
        return values.reshape(len(array), array.type.list_size)
    return np.asarray(array.to_pylist(), dtype=dtype)


def _bounded_shuffle(items, *, buffer_size: int, rng: np.random.Generator):
    """Yield every item once in bounded-memory, approximately shuffled order."""
    if buffer_size <= 1:
        yield from items
        return

    buffer = []
    for item in items:
        if len(buffer) < buffer_size:
            buffer.append(item)
            continue
        index = int(rng.integers(buffer_size))
        outgoing = buffer[index]
        buffer[index] = item
        yield outgoing

    rng.shuffle(buffer)
    yield from buffer


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
        shuffle_buffer_size: int = 8192,
        shuffle: bool = False,
        reshuffle_each_iteration: bool = True,
        token_column: str | None = None,
        mask_column: str | None = None,
        type_column: str | None = None,
        label_column: str | None = None,
        require_labels: bool = False,
        loader_num_workers: int = 0,
        distributed_batch_size: int | None = None,
        distributed_drop_last: bool = False,
    ) -> None:
        super().__init__()
        import pyarrow.parquet as pq

        if not parquet_files:
            raise ValueError("At least one parquet file is required")
        if not 0.0 <= split_start < split_end <= 1.0:
            raise ValueError("Split bounds must satisfy 0 <= start < end <= 1")
        if stream_batch_size <= 0:
            raise ValueError("stream_batch_size must be positive")
        if shuffle_buffer_size < 0:
            raise ValueError("shuffle_buffer_size must be non-negative")

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
            resolved_labels = label_column or first_existing_column(
                names, ["label", "labels"], required=False
            )
            if require_labels and (
                resolved_labels is None or resolved_labels not in names
            ):
                raise KeyError(f"Expected a label column in {path}, found {names}")
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
                    label_column=resolved_labels,
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
        self.shuffle_buffer_size = shuffle_buffer_size
        self.shuffle = shuffle
        self.reshuffle_each_iteration = reshuffle_each_iteration
        self.loader_num_workers = max(1, loader_num_workers)
        self.distributed_batch_size = distributed_batch_size
        self.distributed_drop_last = distributed_drop_last
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
        rank, world_size = self._distributed_context()
        if world_size <= 1:
            return self.expected_rows
        limits = self._distributed_worker_limits(world_size, self.loader_num_workers)
        return sum(
            limits[worker_id * world_size + rank]
            for worker_id in range(self.loader_num_workers)
        )

    @staticmethod
    def _distributed_context() -> tuple[int, int]:
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank(), dist.get_world_size()
        return int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))

    def _assignment_work(self) -> list[tuple[int, int]]:
        work = [
            (file_index, row_group_index)
            for file_index, spec in enumerate(self.specs)
            for row_group_index in range(len(spec.row_group_rows))
        ]
        if self.shuffle:
            assignment_rng = np.random.default_rng(self.seed)
            assignment_rng.shuffle(work)
        return work

    def _distributed_worker_limits(
        self,
        world_size: int,
        num_workers: int,
    ) -> tuple[int, ...]:
        """Return worker limits that give every DDP rank an equal epoch size."""
        num_shards = world_size * num_workers
        work = self._assignment_work()
        assigned_rows = []
        for shard_id in range(num_shards):
            rows = sum(
                self.specs[file_index].row_group_rows[row_group_index]
                for file_index, row_group_index in work[shard_id::num_shards]
            )
            assigned_rows.append(rows)

        if self.distributed_drop_last:
            if not self.distributed_batch_size:
                raise ValueError(
                    "distributed_batch_size is required when distributed_drop_last=True"
                )
            unit = self.distributed_batch_size
            assigned_units = [rows // unit for rows in assigned_rows]
        else:
            unit = 1
            assigned_units = assigned_rows

        rank_units = [
            sum(
                assigned_units[worker_id * world_size + rank]
                for worker_id in range(num_workers)
            )
            for rank in range(world_size)
        ]
        rank_limit = min(rank_units)
        if self.max_rows is not None:
            rank_limit = min(rank_limit, self.max_rows // world_size // unit)

        limits = [0] * num_shards
        for rank in range(world_size):
            remaining = rank_limit
            for worker_id in range(num_workers):
                shard_id = worker_id * world_size + rank
                worker_units = min(assigned_units[shard_id], remaining)
                limits[shard_id] = worker_units * unit
                remaining -= worker_units
            if remaining != 0:
                raise RuntimeError(
                    f"Could not allocate the common DDP epoch size for rank {rank}: "
                    f"{remaining} rows remain"
                )
        return tuple(limits)

    def _shard(self) -> tuple[int, int]:
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        num_workers = worker.num_workers if worker is not None else 1
        rank, world_size = self._distributed_context()
        # Interleave ranks before workers so a small number of row groups is
        # distributed across ranks instead of all landing on rank zero.
        return worker_id * world_size + rank, world_size * num_workers

    def _worker_limit(
        self,
        shard_id: int,
        num_shards: int,
        num_workers: int,
    ) -> int | None:
        world_size, remainder = divmod(num_shards, num_workers)
        if remainder:
            raise RuntimeError(
                f"Cannot divide {num_shards} distributed workers into "
                f"groups of {num_workers}"
            )
        if world_size > 1:
            return self._distributed_worker_limits(
                world_size,
                num_workers,
            )[shard_id]
        if self.max_rows is None:
            return None
        base, remainder = divmod(self.max_rows, num_shards)
        return base + int(shard_id < remainder)

    def __iter__(self):
        import pyarrow.parquet as pq

        shard_id, num_shards = self._shard()
        worker = get_worker_info()
        num_workers = worker.num_workers if worker is not None else 1
        worker_limit = self._worker_limit(shard_id, num_shards, num_workers)
        if worker_limit == 0:
            return

        work = self._assignment_work()
        if not self.reshuffle_each_iteration:
            iteration_seed = 0
        elif worker is not None:
            # The DataLoader base seed changes when non-persistent workers are
            # recreated; the local counter also advances persistent workers.
            iteration_seed = torch.initial_seed() + self._iteration
            self._iteration += 1
        else:
            iteration_seed = self._iteration
            self._iteration += 1
        # Give every rank/worker a disjoint row-group assignment before applying
        # any worker-specific epoch shuffle.  Previously each worker shuffled the
        # complete list with a different seed and only then took a strided slice;
        # those different permutations could assign one row group to multiple
        # workers while omitting another row group entirely.
        work = work[shard_id::num_shards]
        if self.shuffle:
            iteration_rng = np.random.default_rng(
                (self.seed + shard_id + iteration_seed) % np.iinfo(np.uint64).max
            )
            iteration_rng.shuffle(work)

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

        use_shuffle_buffer = self.shuffle and self.shuffle_buffer_size > 1

        def assigned_samples():
            for file_index, row_group_index in work:
                spec = self.specs[file_index]
                parquet = pq.ParquetFile(spec.path)
                columns = [spec.token_column, spec.mask_column]
                if spec.type_column is not None:
                    columns.append(spec.type_column)
                if spec.label_column is not None:
                    columns.append(spec.label_column)

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
                        iteration_rng.shuffle(selected)

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
                        type_ids = np.zeros_like(masks, dtype=np.int64)
                    labels = None
                    if spec.label_column is not None:
                        label_index = batch.schema.get_field_index(spec.label_column)
                        labels = np.asarray(
                            batch.column(label_index).to_numpy(zero_copy_only=False),
                            dtype=np.int64,
                        )

                    for index in selected:
                        sample = {
                            TOKENS_KEY: torch.from_numpy(tokens[index].copy()),
                            MASK_KEY: torch.from_numpy(masks[index].copy()),
                            TYPE_IDS_KEY: torch.from_numpy(type_ids[index].copy()),
                        }
                        if labels is not None:
                            sample[LABELS_KEY] = torch.tensor(
                                labels[index], dtype=torch.long
                            )
                        if use_shuffle_buffer:
                            # A tensor view would keep its complete Arrow batch alive.
                            # Clone buffered samples so memory is proportional to the
                            # configured number of examples, not to old row groups.
                            sample = {key: value.clone() for key, value in sample.items()}
                        yield sample

        samples = assigned_samples()
        if use_shuffle_buffer:
            samples = _bounded_shuffle(
                samples,
                buffer_size=self.shuffle_buffer_size,
                rng=iteration_rng,
            )

        yielded = 0
        for sample in samples:
            yield sample
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


class GroupedTokenParquetClassificationModule(BaseMapModule):
    """Stream a prepared grouped-token classification dataset."""

    def __init__(
        self,
        *,
        prepared_dir: str,
        seed: int = 42,
        max_sequences: int | None = None,
        token_column: str | None = None,
        mask_column: str | None = None,
        type_column: str | None = None,
        label_column: str | None = None,
        stream_batch_size: int = 4096,
        shuffle_buffer_size: int = 8192,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        prepared_path = Path(prepared_dir)
        manifest_path = prepared_path / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Prepared classification manifest is missing: {manifest_path}"
            )
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("identity_overlap_detected") is not False:
            raise ValueError("Prepared classification manifest does not verify disjoint splits")

        total_limit = max_sequences if max_sequences and max_sequences > 0 else None
        split_datasets = {}
        for split_name in ("train", "val", "test"):
            class_files = {
                class_name: sorted(
                    str(path)
                    for path in (prepared_path / split_name / class_name).glob(
                        "*.parquet"
                    )
                )
                for class_name in ("signal", "background")
            }
            if not class_files["signal"] or not class_files["background"]:
                raise FileNotFoundError(
                    f"Prepared {split_name} split must contain signal and background shards"
                )

            expected = manifest.get("split_counts", {}).get(split_name, {})
            if expected.get("signal") != expected.get("background"):
                raise ValueError(f"Prepared {split_name} split is not class balanced")
            split_total = expected.get("total")
            split_limit = None
            if total_limit is not None:
                dataset_total = sum(
                    counts.get("total", 0)
                    for counts in manifest.get("split_counts", {}).values()
                )
                split_limit = int(total_limit * split_total / dataset_total)

            split_datasets[split_name] = StreamingTokenParquetDataset(
                parquet_files=class_files["signal"] + class_files["background"],
                split_start=0.0,
                split_end=1.0,
                seed=seed,
                max_rows=split_limit,
                stream_batch_size=stream_batch_size,
                shuffle_buffer_size=shuffle_buffer_size,
                shuffle=True,
                reshuffle_each_iteration=split_name == "train",
                token_column=token_column,
                mask_column=mask_column,
                type_column=type_column,
                label_column=label_column,
                require_labels=True,
                loader_num_workers=self.num_workers,
                distributed_batch_size=self.batch_size,
                distributed_drop_last=split_name == "train",
            )
            if split_total is not None and split_limit is None:
                observed = split_datasets[split_name].total_rows
                if observed != split_total:
                    raise ValueError(
                        f"Prepared {split_name} rows do not match manifest: "
                        f"{observed} != {split_total}"
                    )

        self.train_set = split_datasets["train"]
        self.valid_set = split_datasets["val"]
        self.test_set = split_datasets["test"]
        self.manifest = manifest
        log.info(
            "Prepared grouped classification dataset verified: train=%d val=%d test=%d",
            self.train_set.total_rows,
            self.valid_set.total_rows,
            self.test_set.total_rows,
        )

    def setup(self, stage: str) -> None:
        pass

    def train_dataloader(self) -> DataLoader:
        # IterableDataset performs row-group and bounded-buffer shuffling itself.
        return self._get_dataloader(
            self.train_set,
            shuffle=False,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return self._get_dataloader(
            self.valid_set,
            shuffle=False,
            drop_last=False,
        )

    def test_dataloader(self) -> DataLoader:
        return self._get_dataloader(
            self.test_set,
            shuffle=False,
            drop_last=False,
        )


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
        shuffle_buffer_size: int = 8192,
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

        self.token_vocabulary = _read_token_vocabulary(str(train_parquet_files[0]))
        validation_vocabulary = _read_token_vocabulary(str(val_parquet_files[0]))
        if self.token_vocabulary != validation_vocabulary:
            raise ValueError(
                "Prepared train and validation shards contain different token vocabularies"
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
            shuffle_buffer_size=shuffle_buffer_size,
            token_column=token_column,
            mask_column=mask_column,
            type_column=type_column,
            loader_num_workers=self.num_workers,
            distributed_batch_size=self.batch_size,
        )
        self.train_set = StreamingTokenParquetDataset(
            parquet_files=[str(path) for path in train_parquet_files],
            split_start=0.0,
            split_end=1.0,
            max_rows=train_limit,
            shuffle=True,
            distributed_drop_last=True,
            **common_kwargs,
        )
        self.valid_set = StreamingTokenParquetDataset(
            parquet_files=[str(path) for path in val_parquet_files],
            split_start=0.0,
            split_end=1.0,
            max_rows=val_limit,
            shuffle=True,
            reshuffle_each_iteration=False,
            distributed_drop_last=True,
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

    def get_token_vocabulary(self) -> dict | None:
        """Return schema vocabulary metadata for model output-head construction."""
        return self.token_vocabulary
