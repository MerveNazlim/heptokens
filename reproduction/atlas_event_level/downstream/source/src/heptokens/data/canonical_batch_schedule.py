"""Immutable, representation-independent event indices and batch schedules.

The v1 schedule deliberately reproduces the historical 128-row *logical*
classification stream. Its chunks come from this canonical event index, never
from a representation's physical Parquet row groups. This permits a complete
identity-stream audit of existing 128-row runs before any result is reused.

Only the single-process, single-GPU classification protocol is represented.
No model, tokenizer, feature, label, split, weight or optimizer is changed.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

FORMAT_VERSION = 1
ALGORITHM = "canonical-logical-chunks-pcg64-bounded-shuffle-v1"
INDEX_NAME = "index_manifest.json"
SPLITS = ("train", "val", "test")
_IDENTITY_DTYPE = np.dtype([("source_id", "<u4"), ("event_index", "<i8"), ("label", "u1")])
_ARRAY_DTYPES = {"source_id": "<u4", "event_index": "<i8", "labels": "u1",
                 "shard_id": "<u4", "local_row": "<u4"}


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _file_sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def source_dictionary_digest(sources: Sequence[str]) -> str:
    sources = tuple(sources)
    if any(not isinstance(s, str) or not s for s in sources) or len(sources) != len(set(sources)):
        raise ValueError("Canonical source dictionary must contain distinct nonempty strings")
    return _sha(b"canonical-source-dictionary-v1\0" + _json_bytes(sources))


def _identity_records(source_ids, event_indices, labels, sources):
    ids, events, targets = np.asarray(source_ids), np.asarray(event_indices), np.asarray(labels)
    if ids.ndim != 1 or events.shape != ids.shape or targets.shape != ids.shape:
        raise ValueError("Ordered event identity arrays must be aligned vectors")
    for value in (ids, events, targets):
        if not np.issubdtype(value.dtype, np.integer):
            raise ValueError("Event identity arrays must contain integers")
    if (np.any(ids < 0) or np.any(ids >= len(sources)) or np.any(events < 0)
            or np.any(targets < 0) or np.any(targets > 255)):
        raise ValueError("Invalid canonical source ID, event index or class label")
    records = np.empty(len(ids), dtype=_IDENTITY_DTYPE)
    records["source_id"], records["event_index"], records["label"] = ids, events, targets
    return records


def event_identity_digest(source_ids, event_indices, labels, *, sources: Sequence[str]) -> str:
    """Hash ordered actual event IDs/labels in the pinned source dictionary.

    The dictionary digest makes numerical source IDs unambiguous. Packed records
    are little-endian and unpadded, so the digest is independent of host byte
    order and of physical Parquet layout. Batch boundaries are hashed separately.
    """
    namespace = bytes.fromhex(source_dictionary_digest(sources))
    records = _identity_records(source_ids, event_indices, labels, sources)
    return _sha(b"canonical-ordered-events-v1\0" + namespace + records.tobytes())


@dataclass(frozen=True)
class SplitIndex:
    source_id: np.ndarray
    event_index: np.ndarray
    labels: np.ndarray
    shard_id: np.ndarray
    local_row: np.ndarray
    shards: tuple[dict, ...]

    @property
    def label(self):
        return self.labels

    def __len__(self):
        return len(self.labels)


@dataclass(frozen=True)
class CanonicalIndex:
    root: Path
    manifest: dict
    sources: tuple[str, ...]
    splits: Mapping[str, SplitIndex]

    @property
    def identity_sha256(self):
        return self.manifest["index_identity_sha256"]


@dataclass(frozen=True)
class CanonicalSchedule:
    order: np.ndarray
    dropped_order: np.ndarray
    batch_hashes: np.ndarray
    metadata: dict
    path: Path | None = None


def event_batch_hash(index: CanonicalIndex, positions, split="train") -> str:
    data = index.splits[split]
    positions = np.asarray(positions)
    if positions.ndim != 1 or not np.issubdtype(positions.dtype, np.integer):
        raise ValueError("Batch positions must be an integer vector")
    if np.any(positions < 0) or np.any(positions >= len(data)):
        raise ValueError("Batch contains an out-of-range canonical row")
    return event_identity_digest(data.source_id[positions], data.event_index[positions],
                                 data.labels[positions], sources=index.sources)


def _ordered_class_names(manifest):
    classes = sorted(manifest.get("classes", []), key=lambda item: int(item["label"]))
    if not classes or [int(c["label"]) for c in classes] != list(range(len(classes))):
        raise ValueError("Canonical manifest needs contiguous classes in label order")
    names = [str(c["name"]) for c in classes]
    if len(set(names)) != len(names) or len(names) > 256:
        raise ValueError("Invalid canonical class names")
    if manifest.get("n_classes", len(names)) != len(names):
        raise ValueError("Canonical manifest n_classes differs from its class definitions")
    return names


def _split_identity(data, sources):
    return event_identity_digest(data.source_id, data.event_index, data.labels, sources=sources)


def _index_identity(manifest):
    payload = {key: manifest[key] for key in ("format_version", "source_dictionary_sha256", "class_names")}
    payload["splits"] = {name: {key: spec[key] for key in ("rows", "identity_sha256", "shards")}
                         for name, spec in manifest["splits"].items()}
    return _sha(b"canonical-event-index-v1\0" + _json_bytes(payload))


def _validate_membership(splits):
    identities = np.concatenate([np.rec.fromarrays((s.source_id, s.event_index),
                    names=("source_id", "event_index")) for s in splits.values()])
    order = np.lexsort((identities["event_index"], identities["source_id"]))
    identities = identities[order]
    if len(identities) > 1 and np.any((identities["source_id"][1:] == identities["source_id"][:-1])
            & (identities["event_index"][1:] == identities["event_index"][:-1])):
        raise ValueError("Duplicate event identity within a split or overlapping train/val/test membership")


def build_index(canonical_dir, output_dir, *, expected_manifest_sha256=None):
    """Build once from canonical identity columns; never overwrite an index.

    Canonical row order is manifest class-label order, sorted files within class,
    then original row order. It matches the original classifier construction;
    globally sorting relative paths would incorrectly move ttH ahead of VBF.
    """
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    canonical_dir, output_dir = Path(canonical_dir), Path(output_dir)
    if output_dir.exists():
        raise FileExistsError("Refusing to overwrite canonical index: " + str(output_dir))
    raw = (canonical_dir / "manifest.json").read_bytes()
    manifest_sha = _sha(raw)
    if expected_manifest_sha256 is not None and manifest_sha != expected_manifest_sha256:
        raise ValueError("Canonical source manifest SHA256 differs")
    source_manifest = json.loads(raw)
    class_names = _ordered_class_names(source_manifest)
    sources, source_codes, split_data = [], {}, {}
    for split in SPLITS:
        parts = {key: [] for key in _ARRAY_DTYPES}
        shards, selected_paths, offset = [], set(), 0
        counts = source_manifest.get("split_counts", {}).get(split, {})
        for label, class_name in enumerate(class_names):
            paths = sorted((canonical_dir / split / class_name).glob("*.parquet"))
            if not paths:
                raise ValueError(f"No canonical shards for {split}/{class_name}")
            class_rows = 0
            for path in paths:
                selected_paths.add(path.resolve())
                parquet, local_offset = pq.ParquetFile(path), 0
                if parquet.metadata.num_rows > np.iinfo(np.uint32).max:
                    raise ValueError("Canonical shard exceeds uint32 row address space")
                shard_id = len(shards)
                for batch in parquet.iter_batches(batch_size=65536, columns=["source_file", "event_index", "label"]):
                    encoded = pc.dictionary_encode(batch.column(batch.schema.get_field_index("source_file")))
                    if encoded.null_count or encoded.dictionary.null_count:
                        raise ValueError("Canonical event source cannot be null")
                    dictionary = encoded.dictionary.to_pylist()
                    for name in dictionary:
                        if not isinstance(name, str) or not name:
                            raise ValueError("Canonical event source must be a nonempty string")
                        if name not in source_codes:
                            source_codes[name] = len(sources); sources.append(name)
                    mapping = np.asarray([source_codes[name] for name in dictionary], dtype=np.uint32)
                    ids = mapping[np.asarray(encoded.indices.to_numpy(), dtype=np.int64)]
                    event_column = batch.column(batch.schema.get_field_index("event_index"))
                    label_column = batch.column(batch.schema.get_field_index("label"))
                    if event_column.null_count or label_column.null_count:
                        raise ValueError("Canonical event index/label cannot be null")
                    events = np.asarray(event_column.to_numpy())
                    labels = np.asarray(label_column.to_numpy())
                    _identity_records(ids, events, labels, sources)
                    if not np.all(labels == label):
                        raise ValueError("Canonical class path and saved labels disagree")
                    n = batch.num_rows
                    values = dict(source_id=ids, event_index=events, labels=labels,
                        shard_id=np.full(n, shard_id, dtype=np.uint32),
                        local_row=np.arange(local_offset, local_offset + n, dtype=np.uint32))
                    for key, value in values.items():
                        parts[key].append(np.asarray(value, dtype=_ARRAY_DTYPES[key]))
                    local_offset += n
                if local_offset != parquet.metadata.num_rows:
                    raise ValueError("Canonical shard identity scan incomplete")
                shards.append(dict(relative_path=path.relative_to(canonical_dir).as_posix(), rows=local_offset,
                    offset=offset, class_name=class_name, label=label))
                offset += local_offset; class_rows += local_offset
            if class_rows != counts.get(class_name):
                raise ValueError(f"Canonical manifest class count differs: {split}/{class_name}")
        actual_paths = {p.resolve() for p in (canonical_dir / split).rglob("*.parquet")}
        if selected_paths != actual_paths or offset != counts.get("total"):
            raise ValueError("Canonical shard inventory or split total differs")
        if offset > np.iinfo(np.uint32).max:
            raise ValueError("Canonical split exceeds uint32 schedule address space")
        arrays = {key: np.concatenate(value) for key, value in parts.items()}
        split_data[split] = SplitIndex(**arrays, shards=tuple(shards))
    _validate_membership(split_data)
    index_manifest = dict(format_version=FORMAT_VERSION, purpose="canonical_classification_event_identity_index",
        class_names=class_names, sources=sources, source_dictionary_sha256=source_dictionary_digest(sources),
        source_manifest_sha256=manifest_sha, splits={})
    for split, data in split_data.items():
        index_manifest["splits"][split] = dict(rows=len(data), identity_sha256=_split_identity(data, sources),
            shards=list(data.shards), file=split + ".npz")
    index_manifest["index_identity_sha256"] = _index_identity(index_manifest)
    output_dir.mkdir(parents=True, exist_ok=False)
    for split, data in split_data.items():
        path = output_dir / (split + ".npz")
        with path.open("xb") as stream:
            np.savez_compressed(stream, **{key: getattr(data, key) for key in _ARRAY_DTYPES})
        index_manifest["splits"][split]["file_sha256"] = _file_sha(path)
    (output_dir / INDEX_NAME).write_bytes(_json_bytes(index_manifest) + b"\n")
    return index_manifest


def load_index(index_dir):
    root = Path(index_dir)
    manifest = json.loads((root / INDEX_NAME).read_text())
    if manifest.get("format_version") != FORMAT_VERSION or set(manifest.get("splits", {})) != set(SPLITS):
        raise ValueError("Unsupported or incomplete canonical event index")
    sources = tuple(manifest["sources"])
    if source_dictionary_digest(sources) != manifest["source_dictionary_sha256"]:
        raise ValueError("Canonical source dictionary digest differs")
    splits = {}
    for name in SPLITS:
        spec = manifest["splits"][name]
        if spec["file"] != name + ".npz":
            raise ValueError("Invalid canonical split filename")
        path = root / spec["file"]
        if _file_sha(path) != spec["file_sha256"]:
            raise ValueError("Canonical split archive SHA256 differs")
        with np.load(path, allow_pickle=False) as saved:
            if set(saved.files) != set(_ARRAY_DTYPES):
                raise ValueError("Unexpected canonical split fields")
            arrays = {key: np.asarray(saved[key]) for key in _ARRAY_DTYPES}
        for key, dtype in _ARRAY_DTYPES.items():
            if arrays[key].dtype != np.dtype(dtype) or arrays[key].shape != (spec["rows"],):
                raise ValueError("Invalid canonical split array: " + key)
        data = SplitIndex(**arrays, shards=tuple(spec["shards"]))
        if _split_identity(data, sources) != spec["identity_sha256"]:
            raise ValueError("Canonical split identity digest differs")
        offset = 0
        for shard_id, shard in enumerate(data.shards):
            rows = shard["rows"]
            if shard["offset"] != offset or not isinstance(rows, int) or rows < 0:
                raise ValueError("Invalid canonical shard offsets")
            section = slice(offset, offset + rows)
            if not np.all(data.shard_id[section] == shard_id) or not np.array_equal(data.local_row[section], np.arange(rows)):
                raise ValueError("Canonical row-to-shard mapping differs")
            if not np.all(data.labels[section] == shard["label"]):
                raise ValueError("Canonical shard label differs")
            offset += rows
        if offset != len(data):
            raise ValueError("Incomplete canonical shard mapping")
        for value in arrays.values():
            value.flags.writeable = False
        splits[name] = data
    if _index_identity(manifest) != manifest["index_identity_sha256"]:
        raise ValueError("Canonical event index logical digest differs")
    _validate_membership(splits)
    return CanonicalIndex(root=root, manifest=manifest, sources=sources, splits=splits)


def logical_row_order(index, seed, epoch, *, split="train", virtual_chunk_size=128,
                      shuffle_buffer_size=8192, shuffle=True):
    """Generate the explicit identity order without reading feature Parquets.

    The RNG calls below mirror frozen StreamingTokenParquetDataset's single-
    process path, including lazy within-chunk shuffles interleaved with bounded
    buffer selection. Counter state is an explicit epoch, never iterator count.
    """
    if type(seed) is not int or seed < 0 or type(epoch) is not int or epoch < 0:
        raise ValueError("Schedule seed and epoch must be nonnegative integers")
    if split not in index.splits or type(virtual_chunk_size) is not int or virtual_chunk_size <= 0:
        raise ValueError("Invalid schedule split/chunk size")
    if type(shuffle_buffer_size) is not int or shuffle_buffer_size < 0:
        raise ValueError("Invalid schedule shuffle buffer")
    data = index.splits[split]
    if not shuffle:
        return np.arange(len(data), dtype=np.uint32)
    work = [(shard["offset"] + start, min(virtual_chunk_size, shard["rows"] - start))
            for shard in data.shards for start in range(0, shard["rows"], virtual_chunk_size)]
    assignment_rng = np.random.Generator(np.random.PCG64(seed))
    assignment_rng.shuffle(work)
    iteration_rng = np.random.Generator(np.random.PCG64(
        (seed + epoch) % np.iinfo(np.uint64).max))
    iteration_rng.shuffle(work)

    def assigned():
        for start, count in work:
            selected = np.arange(count)
            iteration_rng.shuffle(selected)
            for position in selected:
                yield start + int(position)

    def bounded():
        if shuffle_buffer_size <= 1:
            yield from assigned(); return
        buffer = []
        for item in assigned():
            if len(buffer) < shuffle_buffer_size:
                buffer.append(item); continue
            choice = int(iteration_rng.integers(shuffle_buffer_size))
            outgoing = buffer[choice]; buffer[choice] = item
            yield outgoing
        iteration_rng.shuffle(buffer)
        yield from buffer

    return np.fromiter(bounded(), dtype=np.uint32, count=len(data))


def _schedule_identity(metadata):
    return _sha(b"canonical-batch-schedule-v1\0" + _json_bytes(
        {key: value for key, value in metadata.items() if key != "schedule_sha256"}))


def build_schedule(index_dir, seed, epoch, *, split="train", batch_size=256,
                   virtual_chunk_size=128, shuffle_buffer_size=8192,
                   drop_last=None, shuffle=True, output_dir=None):
    index = index_dir if isinstance(index_dir, CanonicalIndex) else load_index(index_dir)
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("Batch size must be a positive integer")
    if drop_last is None:
        drop_last = split == "train"
    if type(drop_last) is not bool or type(shuffle) is not bool:
        raise ValueError("drop_last/shuffle must be explicit booleans")
    directory = Path(output_dir) if output_dir is not None else index.root / "schedules"
    path = directory / f"{split}_seed{seed}_epoch{epoch:02d}.npz"
    expected = dict(seed=seed, epoch=epoch, split=split, batch_size=batch_size,
        virtual_chunk_size=virtual_chunk_size, shuffle_buffer_size=shuffle_buffer_size,
        drop_last=drop_last, shuffle=shuffle, index_identity_sha256=index.identity_sha256)
    if path.exists():
        schedule = load_schedule(path, index_dir=index)
        if any(schedule.metadata.get(key) != value for key, value in expected.items()):
            raise ValueError("Existing immutable schedule has different requested settings")
        return schedule
    full_order = logical_row_order(index, seed, epoch, split=split,
        virtual_chunk_size=virtual_chunk_size, shuffle_buffer_size=shuffle_buffer_size, shuffle=shuffle)
    used = len(full_order) // batch_size * batch_size if drop_last else len(full_order)
    order, dropped = full_order[:used], full_order[used:]
    hashes = np.asarray([event_batch_hash(index, order[start:start + batch_size], split)
                        for start in range(0, used, batch_size)], dtype="U64")
    metadata = dict(format_version=FORMAT_VERSION, algorithm=ALGORITHM,
        rng="numpy.random.PCG64", numpy_version=np.__version__, **expected,
        source_dictionary_sha256=index.manifest["source_dictionary_sha256"],
        split_identity_sha256=index.manifest["splits"][split]["identity_sha256"],
        full_rows=len(full_order), used_rows=used, dropped_rows=len(dropped), batches=len(hashes),
        order_sha256=_sha(order.astype("<u4", copy=False).tobytes()),
        dropped_order_sha256=_sha(dropped.astype("<u4", copy=False).tobytes()),
        batch_hashes_sha256=_sha("".join(hashes).encode("ascii")),
        stream_sha256=event_batch_hash(index, order, split))
    metadata["schedule_sha256"] = _schedule_identity(metadata)
    directory.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez_compressed(stream, order=order, dropped_order=dropped, batch_hashes=hashes,
            metadata_json=np.asarray(_json_bytes(metadata).decode("utf-8")))
    path.with_suffix(".json").write_bytes(_json_bytes(dict(metadata,
        archive_sha256=_file_sha(path))) + b"\n")
    return load_schedule(path, index_dir=index)


def load_schedule(path, index_dir=None):
    path = Path(path)
    sidecar = json.loads(path.with_suffix(".json").read_text())
    if sidecar.pop("archive_sha256", None) != _file_sha(path):
        raise ValueError("Canonical schedule archive SHA256 differs")
    with np.load(path, allow_pickle=False) as saved:
        if set(saved.files) != {"order", "dropped_order", "batch_hashes", "metadata_json"}:
            raise ValueError("Unexpected canonical schedule fields")
        order, dropped, hashes = (np.asarray(saved[key]) for key in ("order", "dropped_order", "batch_hashes"))
        metadata = json.loads(str(saved["metadata_json"].item()))
    if metadata != sidecar or metadata.get("format_version") != FORMAT_VERSION or metadata.get("algorithm") != ALGORITHM:
        raise ValueError("Canonical schedule metadata differs")
    if _schedule_identity(metadata) != metadata.get("schedule_sha256"):
        raise ValueError("Canonical schedule logical digest differs")
    for array in (order, dropped):
        if array.dtype != np.dtype("uint32") or array.ndim != 1:
            raise ValueError("Canonical schedule order must be uint32 vectors")
    if hashes.dtype.kind != "U" or hashes.ndim != 1 or any(len(str(value)) != 64 for value in hashes):
        raise ValueError("Canonical schedule batch hashes are invalid")
    expected = dict(order_sha256=_sha(order.astype("<u4", copy=False).tobytes()),
        dropped_order_sha256=_sha(dropped.astype("<u4", copy=False).tobytes()),
        batch_hashes_sha256=_sha("".join(hashes).encode("ascii")), used_rows=len(order),
        dropped_rows=len(dropped), batches=len(hashes), full_rows=len(order) + len(dropped))
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError("Canonical schedule arrays differ from their metadata")
    complete = np.concatenate((order, dropped))
    if not np.array_equal(np.sort(complete), np.arange(len(complete), dtype=np.uint32)):
        raise ValueError("Canonical schedule repeats or omits event rows")
    batch_size = metadata["batch_size"]
    if len(hashes) != (len(order) + batch_size - 1) // batch_size:
        raise ValueError("Canonical schedule batch boundaries differ")
    if metadata["drop_last"] and (len(order) % batch_size or len(dropped) >= batch_size):
        raise ValueError("Canonical schedule drop-last budget differs")
    if not metadata["drop_last"] and len(dropped):
        raise ValueError("Non-dropping schedule cannot omit events")
    if index_dir is not None:
        index = index_dir if isinstance(index_dir, CanonicalIndex) else load_index(index_dir)
        split = metadata["split"]
        if (metadata["index_identity_sha256"] != index.identity_sha256
                or metadata["split_identity_sha256"] != index.manifest["splits"][split]["identity_sha256"]
                or len(complete) != len(index.splits[split])):
            raise ValueError("Canonical schedule belongs to another event index")
        actual = [event_batch_hash(index, order[start:start + batch_size], split)
                  for start in range(0, len(order), batch_size)]
        if not np.array_equal(hashes, actual) or metadata["stream_sha256"] != event_batch_hash(index, order, split):
            raise ValueError("Canonical schedule ordered event-ID hashes differ")
    for value in (order, dropped, hashes):
        value.flags.writeable = False
    return CanonicalSchedule(order=order, dropped_order=dropped, batch_hashes=hashes, metadata=metadata, path=path)
