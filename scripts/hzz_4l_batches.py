"""Read the sealed four-lepton shards and ordered training schedules."""

from collections import OrderedDict
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, IterableDataset

from heptokens.data.token_parquet import (
    StreamingTokenParquetDataset,
    _read_continuous_schema,
    _read_token_vocabulary,
    _sequence_array_to_numpy,
)

SPLITS = ("train", "val", "test")
IDENTITY = ("source_file", "event_index", "label")
RECORD = np.dtype([("source_id", "<u4"), ("event_index", "<i8"), ("label", "u1")])


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def pinned(path, digest):
    if sha256(path) != digest:
        raise ValueError(f"SHA256 differs for {path}")
    return path


def read_json(path):
    return json.loads(Path(path).read_text())


def batch_digest(namespace, identities):
    return hashlib.sha256(b"canonical-ordered-events-v1\0" + namespace + identities.tobytes()).hexdigest()


class StudyData:
    """Verify the original selection, indices, schedules and chosen shard view."""

    def __init__(self, root, cfg, kind, seed):
        self.kind, self.seed = kind, seed
        self.batch_size = cfg["protocol"]["batch_size"]
        self.steps = cfg["protocol"]["updates_per_epoch"]
        self.class_names = cfg["class_names"]
        for relative, digest in cfg["dataset"]["manifest_sha256"].items():
            if relative.split("/")[0] in ("selection", "canonical_index", kind):
                pinned(root / relative, digest)
        self.receipt = read_json(root / "selection/selection_receipt.json")
        self.view = read_json(root / kind / "selection_view.json")
        manifest = read_json(root / kind / "manifest.json")
        index_path = root / "canonical_index"
        self.index = read_json(index_path / "index_manifest.json")
        campaign = read_json(index_path / "schedule_campaign.json")
        if not (self.view["passed"] and campaign["passed"]
                and cfg["selection_id"] == self.receipt["selection_id"] == self.view["selection_id"]
                == campaign["selection_id"]
                and self.view["representation"] == kind
                and self.index["class_names"] == self.class_names
                and campaign["index_identity_sha256"] == self.index["index_identity_sha256"]):
            raise ValueError("Selected dataset provenance differs")
        self.sources = self.index["sources"]
        self.source_ids = {name: i for i, name in enumerate(self.sources)}
        self.namespace = bytes.fromhex(self.index["source_dictionary_sha256"])
        source_remap = np.asarray([self.receipt["sources"].index(s) for s in self.sources])
        self.identities, self.shards = {}, {split: [] for split in SPLITS}
        self.columns = {"mask": bool, "type_ids": np.int64}
        if kind == "continuous":
            self.columns.update(continuous_features=np.float32, continuous_feature_mask=bool,
                                position_role_ids=np.int64)
        else:
            self.columns["tokens"] = np.int64
        self.schema = self.vocabulary = None
        for relative, spec in sorted(self.view["shards"].items()):
            split, name, basename = Path(relative).parts
            if split not in SPLITS or name not in self.class_names or not basename.endswith(".parquet"):
                raise ValueError(f"Unexpected selected shard name: {relative}")
            path = pinned(root / kind / relative, spec["sha256"])
            handle = pq.ParquetFile(path)
            if handle.metadata.num_rows != spec["rows"] or not {*IDENTITY, *self.columns}.issubset(handle.schema_arrow.names):
                raise ValueError(f"Unexpected row count or fields: {relative}")
            self.shards[split].append((self.class_names.index(name), path))
            if kind == "continuous":
                value = _read_continuous_schema(str(path))
                if value is None or self.schema not in (None, value):
                    raise ValueError(f"Continuous input schema differs: {relative}")
                self.schema = value
            else:
                value = _read_token_vocabulary(str(path))
                if value is None or self.vocabulary not in (None, value):
                    raise ValueError(f"Q8 token vocabulary differs: {relative}")
                self.vocabulary = value
        for shards in self.shards.values():
            shards.sort(key=lambda item: (item[0], item[1]))
        with np.load(root / "selection/membership.npz", allow_pickle=False) as membership:
            for split in SPLITS:
                spec = self.index["splits"][split]
                with np.load(pinned(index_path / spec["file"], spec["file_sha256"]), allow_pickle=False) as values:
                    records = np.empty(spec["rows"], dtype=RECORD)
                    for field, key in (("source_id", "source_id"), ("event_index", "event_index"), ("label", "labels")):
                        actual = values[key]
                        records[field] = actual
                        if field == "source_id":
                            actual = source_remap[actual]
                        if not np.array_equal(actual, membership[f"{split}_{key}"]):
                            raise ValueError(f"Wrong canonical identities: {split}/{key}")
                counts = cfg["split_counts"][split]
                if (batch_digest(self.namespace, records) != spec["identity_sha256"]
                        or np.bincount(records["label"], minlength=len(self.class_names)).tolist() != counts
                        or manifest["split_counts"][split] != dict(zip(self.class_names, counts), total=sum(counts))):
                    raise ValueError(f"Wrong selected split: {split}")
                self.identities[split] = records
        counts = np.asarray(cfg["split_counts"]["train"])
        self.weights = (sum(counts) / (len(counts) * counts)).tolist()
        if not np.allclose([manifest["recommended_cross_entropy_weights"][name] for name in self.class_names],
                           self.weights, rtol=1e-10, atol=0.0):
            raise ValueError("Selected training weights differ")
        self.orders, self.batch_hashes = [], []
        schedule_specs = {(spec["seed"], spec["epoch"]): spec for spec in campaign["schedules"]}
        for epoch in range(cfg["protocol"]["epochs"]):
            spec = schedule_specs[seed, epoch]
            with np.load(pinned(index_path / "schedules" / spec["file"], spec["archive_sha256"]),
                         allow_pickle=False) as archive:
                order, hashes = archive["order"], archive["batch_hashes"]
                if (json.loads(str(archive["metadata_json"].item()))
                        != {k: v for k, v in spec.items() if k not in {"file", "archive_sha256"}}
                        or order.shape != (self.steps * self.batch_size,) or hashes.shape != (self.steps,)
                        or spec["dropped_rows"] != cfg["protocol"]["dropped_rows_per_epoch"]
                        or spec["index_identity_sha256"] != self.index["index_identity_sha256"]
                        or not np.array_equal(np.sort(np.concatenate((order, archive["dropped_order"]))),
                                              np.arange(len(self.identities["train"])))):
                    raise ValueError(f"Wrong canonical schedule: {spec['file']}")
            self.orders.append(order)
            self.batch_hashes.append(hashes)
        self.train = ScheduledBatches(self)

    def read_ids(self, table):
        result = np.empty(len(table), dtype=RECORD)
        result["source_id"] = [self.source_ids[value] for value in table["source_file"].to_pylist()]
        result["event_index"] = table["event_index"].to_numpy()
        result["label"] = table["label"].to_numpy()
        return result

    def read_inputs(self, table):
        return {name: _sequence_array_to_numpy(table[name].combine_chunks(), np.dtype(dtype))
                for name, dtype in self.columns.items()}


class ScheduledBatches(IterableDataset):
    """Join canonical IDs to physical row groups; cache decoded groups, not epochs."""

    def __init__(self, data, cache_bytes=512 * 1024 * 1024):
        self.data, self.cache_bytes = data, cache_bytes
        self.cache, self.cache_size = OrderedDict(), 0
        self.handles, self.file_stats = [], []
        self.epoch, self.consumed = None, {}
        canonical = data.identities["train"]
        permutation = np.argsort(canonical)
        ordered = canonical[permutation]
        self.locations = np.full((len(canonical), 3), -1, dtype=np.int32)
        for file_id, (label, path) in enumerate(data.shards["train"]):
            handle = pq.ParquetFile(path)
            self.handles.append(handle)
            self.file_stats.append(self.stat(path))
            for group in range(handle.num_row_groups):
                ids = data.read_ids(handle.read_row_group(group, columns=list(IDENTITY)))
                indices = np.searchsorted(ordered, ids)
                if (len(ids) > 128 or not np.all(ids["label"] == label) or np.any(indices >= len(ordered))
                        or not np.array_equal(ids, ordered[indices]) or len(indices) != len(np.unique(indices))
                        or np.any(self.locations[permutation[indices], 0] >= 0)):
                    raise ValueError(f"Invalid or duplicate canonical training identities in {path}:{group}")
                self.locations[permutation[indices]] = np.column_stack((
                    np.full(len(ids), file_id), np.full(len(ids), group), np.arange(len(ids))))
        if np.any(self.locations < 0):
            raise ValueError("Selected training view omits canonical identities")

    @staticmethod
    def stat(path):
        status = path.stat()
        return status.st_size, status.st_mtime_ns

    def __len__(self):
        return self.data.steps

    def group(self, file_id, group):
        key = int(file_id), int(group)
        if self.stat(self.data.shards["train"][file_id][1]) != self.file_stats[file_id]:
            raise ValueError("Training shard changed after verification")
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        table = self.handles[file_id].read_row_group(group, columns=[*IDENTITY, *self.data.columns])
        ids, values = self.data.read_ids(table), self.data.read_inputs(table)
        size = ids.nbytes + sum(arr.nbytes for arr in values.values())
        while self.cache and self.cache_size + size > self.cache_bytes:
            self.cache_size -= self.cache.popitem(last=False)[1][2]
        if size <= self.cache_bytes:
            self.cache[key] = ids, values, size
            self.cache_size += size
        return ids, values, size

    def __iter__(self):
        if torch.utils.data.get_worker_info() is not None or self.epoch is None or self.epoch in self.consumed:
            raise ValueError("Every canonical epoch requires one explicitly armed, single-worker iterator")
        epoch = self.epoch
        self.consumed[epoch] = 0
        for idx, positions in enumerate(self.data.orders[epoch].reshape(-1, self.data.batch_size)):
            locations = self.locations[positions]
            groups, inverse = np.unique(locations[:, :2], axis=0, return_inverse=True)
            ids = np.empty(len(positions), dtype=RECORD)
            result = None
            for i, (file_id, group) in enumerate(groups):
                selected = np.flatnonzero(inverse == i)
                locals_ = locations[selected, 2]
                actual, decoded, _ = self.group(file_id, group)
                ids[selected] = actual[locals_]
                if result is None:
                    result = {name: np.empty((len(positions), *arr.shape[1:]), dtype=arr.dtype)
                              for name, arr in decoded.items()}
                for name, arr in decoded.items():
                    result[name][selected] = arr[locals_]
            if (not np.array_equal(ids, self.data.identities["train"][positions])
                    or batch_digest(self.data.namespace, ids) != self.data.batch_hashes[epoch][idx]):
                raise ValueError(f"Canonical batch identity differs at epoch {epoch}, batch {idx}")
            result["labels"] = ids["label"].astype(np.int64)
            self.consumed[epoch] += 1
            yield {name: torch.from_numpy(arr) for name, arr in result.items()}


class StudyDataModule(LightningDataModule):
    def __init__(self, data):
        super().__init__()
        self.data = data
        # Keep the archived validation stream, including its fixed bounded shuffle.
        self.val = StreamingTokenParquetDataset(
            parquet_files=[str(path) for _, path in data.shards["val"]],
            split_start=0.0, split_end=1.0, seed=data.seed, stream_batch_size=4096,
            shuffle_buffer_size=8192, shuffle=True, reshuffle_each_iteration=False,
            label_column="label", require_labels=True, require_continuous=data.kind == "continuous",
            continuous_feature_column="continuous_features" if data.kind == "continuous" else None,
            include_tokens=data.kind != "continuous", loader_num_workers=0,
            distributed_batch_size=data.batch_size, distributed_drop_last=False,
        )
        if self.val.total_rows != len(data.identities["val"]):
            raise ValueError("Validation stream count differs from the selected split")

    def train_dataloader(self):
        self.data.train.epoch = self.trainer.current_epoch
        return DataLoader(self.data.train, batch_size=None, num_workers=0, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.data.batch_size, num_workers=0, pin_memory=True)
