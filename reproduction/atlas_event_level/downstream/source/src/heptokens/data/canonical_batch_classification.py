"""Opt-in identity-scheduled classification, independent of Parquet grouping.

Only training is replaced. Validation/test loaders, model inputs, class weights
and the existing classifier implementations are unchanged. The production
contract is a single process/GPU with no loader workers or gradient accumulation.
"""
from __future__ import annotations

from collections import OrderedDict
from functools import partial
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from lightning.pytorch.callbacks import Callback
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from heptokens.data.canonical_batch_schedule import (
    event_identity_digest, load_index, load_schedule,
)
from heptokens.data.collation import collate_and_transform
from heptokens.data.token_parquet import (
    GroupedTokenParquetClassificationModule, _sequence_array_to_numpy,
)

CANONICAL_ROW = "canonical_event_row"
CANONICAL_SOURCE = "canonical_source_id"
CANONICAL_EVENT = "canonical_event_index"


def _sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _identity_keys(source, event, label):
    result = np.empty(len(event), dtype=[("source", "<u4"), ("event", "<i8"), ("label", "u1")])
    result["source"], result["event"], result["label"] = source, event, label
    return result


def _json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


class CanonicalTrainingDataset(IterableDataset):
    """Read precomputed ordered batches, verifying each selected identity.

    Row-group arrays are cached and converted once, not once per event. Input
    views must have groups of at most 128 rows to bound random-access overhead.
    Canonical-to-physical row mapping is derived from the actual identity columns,
    not from assumptions about filenames or offsets across representations.
    """

    def __init__(self, original, index_dir, schedule_dir, *, seed, batch_size,
                 cache_bytes=512 * 1024 * 1024, maximum_row_group=128, production=False):
        super().__init__()
        import pyarrow.parquet as pq

        self.index_dir = Path(index_dir).resolve()
        self.index = load_index(self.index_dir)
        self.identities = self.index.splits["train"]
        self.sources = tuple(self.index.sources)
        self.source_ids = {name: i for i, name in enumerate(self.sources)}
        self.schedule_dir = Path(schedule_dir).resolve()
        self.seed, self.batch_size = int(seed), int(batch_size)
        self.production = bool(production)
        self.specs = tuple(original.specs)
        self.include_tokens = original.include_tokens
        self.total_rows = self.expected_rows = len(self.identities.event_index)
        if self.total_rows != original.total_rows:
            raise ValueError("Prepared train rows differ from canonical identity index")
        if cache_bytes < 1 or maximum_row_group < 1:
            raise ValueError("Cache and maximum row-group sizes must be positive")
        self.cache_limit = int(cache_bytes)
        self.cache, self.cache_size, self.row_group_reads = OrderedDict(), 0, 0
        self.parquet_handles = {}
        self.file_stats = {}
        self.locations = np.full((self.total_rows, 3), -1, dtype=np.int32)
        expected = _identity_keys(self.identities.source_id, self.identities.event_index, self.identities.labels)
        ordered = np.argsort(expected, order=("source", "event", "label"))
        sorted_keys = expected[ordered]
        if len(sorted_keys) > 1 and np.any(sorted_keys[1:] == sorted_keys[:-1]):
            raise ValueError("Duplicate canonical training identity")
        seen = np.zeros(self.total_rows, dtype=bool)
        for file_id, spec in enumerate(self.specs):
            handle = pq.ParquetFile(spec.path)
            self.parquet_handles[file_id] = handle
            status = Path(spec.path).stat()
            self.file_stats[file_id] = (status.st_size, status.st_mtime_ns)
            required = {"source_file", "event_index", spec.label_column}
            if None in required or not required.issubset(handle.schema_arrow.names):
                raise ValueError("Scheduled training requires source_file, event_index and label columns")
            for group_id in range(handle.num_row_groups):
                rows = handle.metadata.row_group(group_id).num_rows
                if rows > maximum_row_group:
                    raise ValueError("Normalize classification training views to at most 128-row groups before scheduling")
                table = handle.read_row_group(group_id, columns=list(required))
                source, event, label = self._read_identities(table, spec)
                keys = _identity_keys(source, event, label)
                positions = np.searchsorted(sorted_keys, keys)
                if np.any(positions >= len(sorted_keys)) or not np.array_equal(sorted_keys[positions], keys):
                    raise ValueError("Prepared event identity or label differs from canonical index")
                canonical = ordered[positions]
                if len(np.unique(canonical)) != len(canonical) or seen[canonical].any():
                    raise ValueError("Duplicate prepared event identity")
                seen[canonical] = True
                self.locations[canonical] = np.column_stack((np.full(rows, file_id), np.full(rows, group_id), np.arange(rows)))
        if not seen.all():
            raise ValueError("Prepared training view is missing canonical identities")
        self.epoch = None
        self.schedule = None
        self.schedule_path = None
        self.maximum_batches = None
        self.issued_batches = 0
        self.started_epochs = set()

    def _read_identities(self, table, spec):
        import pyarrow as pa
        for column in ("event_index", spec.label_column):
            if not pa.types.is_integer(table[column].type) or table[column].null_count:
                raise ValueError("Saved event index and label must be non-null integers")
        try:
            source = np.asarray([self.source_ids[str(value)] for value in table["source_file"].to_pylist()], dtype=np.uint32)
        except KeyError as error:
            raise ValueError("Unknown source_file in prepared identity columns") from error
        event = np.asarray(table["event_index"].to_numpy(), dtype=np.int64)
        label = np.asarray(table[spec.label_column].to_numpy(), dtype=np.int64)
        if np.any(event < 0) or np.any(label < 0) or np.any(label > 255):
            raise ValueError("Invalid event index or classification label")
        return source, event, label.astype(np.uint8)

    def arm_epoch(self, epoch, maximum_batches=None):
        epoch = int(epoch)
        if self.epoch == epoch:
            if maximum_batches is not None and maximum_batches != self.maximum_batches:
                raise ValueError("Cannot change the batch budget inside an armed epoch")
            return
        if epoch != (0 if self.epoch is None else self.epoch + 1):
            raise ValueError("Canonical epochs cannot be skipped, repeated or resumed")
        if self.epoch is not None and self.issued_batches != self.maximum_batches:
            raise ValueError("Previous scheduled epoch was not fully consumed")
        path = self.schedule_dir / f"train_seed{self.seed}_epoch{epoch:02d}.npz"
        schedule = load_schedule(path, index_dir=self.index)
        meta = schedule.metadata
        for key, expected in (("seed", self.seed), ("epoch", epoch), ("batch_size", self.batch_size), ("split", "train")):
            if meta.get(key) != expected:
                raise ValueError("Canonical schedule setting differs: " + key)
        if self.production:
            for key, expected in (("virtual_chunk_size", 128), ("shuffle_buffer_size", 8192),
                                  ("drop_last", True), ("shuffle", True), ("full_rows", self.total_rows)):
                if meta.get(key) != expected:
                    raise ValueError("Production canonical schedule differs: " + key)
        if len(schedule.order) % self.batch_size or len(schedule.batch_hashes) != len(schedule.order) // self.batch_size:
            raise ValueError("Incomplete scheduled batch boundaries")
        full_batches = len(schedule.batch_hashes)
        maximum_batches = full_batches if maximum_batches is None else int(maximum_batches)
        if not 0 < maximum_batches <= full_batches:
            raise ValueError("Requested batch budget exceeds the canonical schedule")
        self.epoch, self.schedule, self.schedule_path = epoch, schedule, path
        self.maximum_batches, self.issued_batches = maximum_batches, 0

    def __len__(self):
        if self.maximum_batches is not None:
            return self.maximum_batches
        return self.total_rows // self.batch_size

    def _group(self, file_id, group_id):
        import pyarrow.parquet as pq
        key = (int(file_id), int(group_id))
        status = Path(self.specs[key[0]].path).stat()
        if (status.st_size, status.st_mtime_ns) != self.file_stats[key[0]]:
            raise ValueError("Prepared training file changed after the identity scan")
        if key in self.cache:
            value = self.cache.pop(key); self.cache[key] = value
            return value[0]
        spec = self.specs[key[0]]
        columns = [spec.mask_column, spec.type_column, spec.label_column, "source_file", "event_index"]
        if self.include_tokens:
            columns.append(spec.token_column)
        columns += [spec.continuous_feature_column, spec.continuous_feature_mask_column, spec.position_role_column]
        columns = list(dict.fromkeys(name for name in columns if name is not None))
        handle = self.parquet_handles.get(key[0])
        if handle is None:
            handle = self.parquet_handles[key[0]] = pq.ParquetFile(spec.path)
        table = handle.read_row_group(key[1], columns=columns)
        source, event, label = self._read_identities(table, spec)
        data = {CANONICAL_SOURCE: source.astype(np.int64), CANONICAL_EVENT: event,
                "labels": label.astype(np.int64)}
        pairs = [("mask", spec.mask_column, bool), ("type_ids", spec.type_column, np.int64)]
        if self.include_tokens:
            pairs.append(("tokens", spec.token_column, np.int64))
        pairs += [("continuous_features", spec.continuous_feature_column, np.float32),
                  ("continuous_feature_mask", spec.continuous_feature_mask_column, bool),
                  ("position_role_ids", spec.position_role_column, np.int64)]
        for name, column, dtype in pairs:
            if column is not None:
                data[name] = _sequence_array_to_numpy(table[column].combine_chunks(), np.dtype(dtype))
        if "type_ids" not in data:
            data["type_ids"] = np.zeros(data["mask"].shape, dtype=np.int64)
        size = sum(value.nbytes for value in data.values())
        while self.cache and self.cache_size + size > self.cache_limit:
            _, (_, old_size) = self.cache.popitem(last=False); self.cache_size -= old_size
        if size <= self.cache_limit:
            self.cache[key] = (data, size); self.cache_size += size
        self.row_group_reads += 1
        return data

    def __iter__(self):
        if get_worker_info() is not None:
            raise ValueError("Canonical scheduled training requires num_workers=0")
        if self.epoch is None or self.schedule is None:
            raise ValueError("Arm an explicit canonical epoch before iterating training data")
        if self.epoch in self.started_epochs:
            raise ValueError("Extra training iterator would repeat a canonical epoch")
        self.started_epochs.add(self.epoch)
        for batch_id in range(self.maximum_batches):
            positions = np.asarray(self.schedule.order[batch_id * self.batch_size:(batch_id + 1) * self.batch_size], dtype=np.int64)
            locations = self.locations[positions]
            groups, inverse = np.unique(locations[:, :2], axis=0, return_inverse=True)
            arrays = None
            for group_index, (file_id, group_id) in enumerate(groups):
                selected = np.flatnonzero(inverse == group_index)
                local = locations[selected, 2]
                values = self._group(file_id, group_id)
                if arrays is None:
                    arrays = {name: np.empty((self.batch_size,) + value.shape[1:], dtype=value.dtype)
                              for name, value in values.items()}
                if set(arrays) != set(values):
                    raise ValueError("Mixed input schemas in scheduled training view")
                for name, value in values.items():
                    arrays[name][selected] = value[local]
            assert arrays is not None
            if (not np.array_equal(arrays[CANONICAL_SOURCE], self.identities.source_id[positions])
                    or not np.array_equal(arrays[CANONICAL_EVENT], self.identities.event_index[positions])
                    or not np.array_equal(arrays["labels"], self.identities.labels[positions])):
                raise ValueError("Actual loaded row identity differs from the scheduled event")
            observed = event_identity_digest(arrays[CANONICAL_SOURCE], arrays[CANONICAL_EVENT], arrays["labels"], sources=self.sources)
            if observed != str(self.schedule.batch_hashes[batch_id]):
                raise ValueError("Actual loaded ordered batch hash differs from canonical schedule")
            arrays[CANONICAL_ROW] = positions
            self.issued_batches = batch_id + 1
            yield {name: torch.from_numpy(value.copy()) for name, value in arrays.items()}


class CanonicalBatchClassificationModule(GroupedTokenParquetClassificationModule):
    """Opt-in training-only adapter for the existing five-class datamodule."""

    def __init__(self, *, canonical_index_dir, canonical_schedule_dir=None,
                 canonical_cache_bytes=512 * 1024 * 1024, production=True, **kwargs):
        super().__init__(**kwargs)
        if self.num_workers != 0:
            raise ValueError("Canonical scheduled training requires num_workers=0")
        if production and (self.batch_size != 256 or self.n_classes != 5 or kwargs.get("seed", 42) not in (42, 43, 44)):
            raise ValueError("Production canonical batches require five classes, batch256 and seed42/43/44")
        if kwargs.get("max_sequences") not in (None, 0):
            raise ValueError("Canonical scheduled training cannot silently truncate the source dataset")
        self.production = bool(production)
        self.canonical_index_dir = Path(canonical_index_dir).resolve()
        self.canonical_schedule_dir = Path(canonical_schedule_dir or self.canonical_index_dir / "schedules").resolve()
        self.original_train_set = self.train_set
        self.train_set = CanonicalTrainingDataset(self.original_train_set, self.canonical_index_dir,
            self.canonical_schedule_dir, seed=kwargs.get("seed", 42), batch_size=self.batch_size,
            cache_bytes=canonical_cache_bytes, production=self.production)
        if list(self.train_set.index.manifest["class_names"]) != self.class_names:
            raise ValueError("Prepared class names/order differ from canonical index")
        if production and self.train_set.total_rows != 1700673:
            raise ValueError("Production canonical training membership must contain 1700673 events")

    def set_canonical_epoch(self, epoch, maximum_batches=None):
        self.train_set.arm_epoch(epoch, maximum_batches)

    def _training_budget(self, trainer):
        limit = trainer.limit_train_batches
        if self.production and not ((trainer.max_epochs == 1 and type(limit) is int and limit == 200)
                or (trainer.max_epochs == 10 and type(limit) is float and limit == 1.0)):
            raise ValueError("Production canonical fit is either the 200-step smoke or a full ten-epoch fit")
        if isinstance(limit, int):
            if self.production and limit != 200:
                raise ValueError("Production integer training limit is reserved for the 200-step smoke")
            return int(limit)
        if float(limit) != 1.0:
            raise ValueError("Canonical full fit requires limit_train_batches=1.0")
        return self.train_set.total_rows // self.batch_size

    def train_dataloader(self):
        trainer = getattr(self, "trainer", None)
        if trainer is not None:
            if trainer.world_size != 1 or trainer.num_nodes != 1 or trainer.accumulate_grad_batches != 1:
                raise ValueError("Canonical classification currently requires one GPU/process and no gradient accumulation")
            if self.production and not any(isinstance(callback, CanonicalBatchAuditCallback) for callback in trainer.callbacks):
                raise ValueError("Production fit requires CanonicalBatchAuditCallback")
            self.set_canonical_epoch(trainer.current_epoch, self._training_budget(trainer))
        elif self.train_set.epoch is None:
            raise ValueError("Offline iteration requires an explicitly selected canonical epoch")
        collate = partial(collate_and_transform, do_default_collate=False, transforms=self.transforms) if self.transforms is not None else None
        return DataLoader(self.train_set, batch_size=None, num_workers=0, pin_memory=self.pin_memory, collate_fn=collate)


class CanonicalBatchAuditCallback(Callback):
    """Fail before optimizer updates if any actual ordered event batch differs."""

    def __init__(self):
        super().__init__()
        self.epochs = []
        self.observed_hashes = []
        self.manifest_sha256 = None

    def on_fit_start(self, trainer, pl_module):
        if not isinstance(trainer.datamodule, CanonicalBatchClassificationModule):
            raise ValueError("Canonical audit requires the opt-in canonical datamodule")
        if getattr(trainer, "ckpt_path", None) is not None or trainer.current_epoch != 0 or trainer.global_step != 0:
            raise ValueError("Canonical classifier audit requires a fresh fit, not a resumed classifier")
        self.manifest_sha256 = _sha(trainer.datamodule.canonical_index_dir / "index_manifest.json")
        self.epochs, self.observed_hashes = [], []
        self._write(trainer, passed=False)

    def on_train_epoch_start(self, trainer, pl_module):
        data = trainer.datamodule
        data.set_canonical_epoch(trainer.current_epoch, data._training_budget(trainer))
        if trainer.current_epoch != len(self.epochs):
            raise ValueError("Canonical batch audit cannot skip or repeat an epoch")
        if _sha(data.canonical_index_dir / "index_manifest.json") != self.manifest_sha256:
            raise ValueError("Canonical identity manifest changed during training")
        self.schedule_sha256 = _sha(data.train_set.schedule_path)
        self.observed_hashes = []

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        data = trainer.datamodule.train_set
        if batch_idx != len(self.observed_hashes) or batch_idx >= data.maximum_batches:
            raise ValueError("Canonical batch index repeated, skipped or exceeded")
        required = (CANONICAL_ROW, CANONICAL_SOURCE, CANONICAL_EVENT, "labels")
        if any(name not in batch for name in required):
            raise ValueError("Training batch lost canonical identity metadata")
        arrays = {name: batch[name].detach().cpu().numpy() for name in required}
        expected_rows = data.schedule.order[batch_idx * data.batch_size:(batch_idx + 1) * data.batch_size]
        if arrays[CANONICAL_ROW].shape != (data.batch_size,) or not np.array_equal(arrays[CANONICAL_ROW], expected_rows):
            raise ValueError("Actual training batch has a different ordered canonical row list")
        actual = event_identity_digest(arrays[CANONICAL_SOURCE], arrays[CANONICAL_EVENT], arrays["labels"], sources=data.sources)
        if actual != str(data.schedule.batch_hashes[batch_idx]):
            raise ValueError("Actual training event IDs/order/labels differ before optimizer update")
        self.observed_hashes.append(actual)

    def on_train_epoch_end(self, trainer, pl_module):
        data = trainer.datamodule.train_set
        if len(self.observed_hashes) != data.maximum_batches or data.issued_batches != data.maximum_batches:
            raise ValueError("Incomplete canonical epoch batch audit")
        expected_steps = sum(entry["batches"] for entry in self.epochs) + data.maximum_batches
        if trainer.global_step != expected_steps:
            raise ValueError("Optimizer-step count differs from the verified canonical batch stream")
        if _sha(data.schedule_path) != self.schedule_sha256:
            raise ValueError("Canonical schedule changed during the epoch")
        rolling = hashlib.sha256("".join(self.observed_hashes).encode("ascii")).hexdigest()
        if data.maximum_batches == len(data.schedule.batch_hashes) and rolling != data.schedule.metadata["batch_hashes_sha256"]:
            raise ValueError("Complete ordered-batch audit digest differs from canonical schedule")
        entry = dict(epoch=int(trainer.current_epoch), seed=data.seed, batches=data.maximum_batches,
            batch_size=data.batch_size, used_rows=data.maximum_batches * data.batch_size,
            schedule_sha256=self.schedule_sha256, ordered_batch_hashes_sha256=rolling,
            schedule_metadata=data.schedule.metadata, batch_hashes=list(self.observed_hashes))
        self.epochs.append(entry)
        self._write(trainer, passed=False)

    def on_fit_end(self, trainer, pl_module):
        if len(self.epochs) != trainer.max_epochs:
            raise ValueError("Canonical audit did not verify every requested training epoch")
        self._write(trainer, passed=True)

    def _write(self, trainer, passed):
        if getattr(trainer, "is_global_zero", True):
            _json(Path(trainer.default_root_dir) / "canonical_batch_audit.json", dict(
                passed=passed, contract="exact ordered training event identities and labels per batch",
                index_manifest_sha256=self.manifest_sha256, seed=trainer.datamodule.train_set.seed,
                validation_test="unchanged existing loaders; this audit controls training batches only",
                epochs=self.epochs))
