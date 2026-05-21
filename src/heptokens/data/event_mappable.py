"""For loading events from event-level HDF files and creating a mappable dataset."""

# TODO(event-tokenization):
# 1. Add an optional event schema preset, e.g. configs/event_schema/atlas.yaml,
#    that can generate object_collections without forcing every config to repeat
#    long HDF5 paths.
# 2. Add per-object mask paths to the default configs once the canonical ATLAS
#    event HDF5 structure is stable.
# 3. Make combined-mode type IDs configurable instead of deriving them from
#    collection order. This will matter for modality embeddings.
# 4. Add support for object boundary metadata/separator-token studies. This is
#    needed for comparing separator tokens, modality embeddings, or both.
# 5. Eventually return event-level inputs as batch["event"] for models that
#    understand event-level context directly. For now they are also routed to
#    batch["jets"] to stay compatible with existing heptokens tokenizer models.
# 6. EventSingleFileMapModule now accepts data_paths as well as data_path. The
#    class name is kept for backward compatibility, but it should eventually be
#    renamed once downstream configs have migrated.
# 7. EventStreamMapModule provides chunked HDF5 reads for larger samples. The
#    mappable module remains useful for small prototype/debug runs.

from collections.abc import Mapping
from functools import partial
import logging

import h5py
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import ConcatDataset, DataLoader, Dataset, IterableDataset, random_split

from heptokens.data.atlas_mappable import BaseMapModule
from heptokens.data.collation import collate_and_transform

log = logging.getLogger(__name__)


def _as_list(value) -> list:
    if value is None:
        return []
    return list(value)


def _as_dict(value) -> dict:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    return dict(value)


def _resolve_h5_path(handle: h5py.File, h5_path: str) -> np.ndarray:
    """Read a dataset or structured-array field via a slash-separated path.

    The path walks h5py groups until it reaches either a plain Dataset or a
    structured Dataset field, e.g. ``"common/jets/pt"`` resolves to
    ``handle["common/jets"]["pt"]`` if ``common/jets`` is a structured array.
    """
    parts = h5_path.strip("/").split("/")
    node = handle
    for i, part in enumerate(parts):
        if isinstance(node, h5py.Dataset):
            field = "/".join(parts[i:])
            return node[field][:]
        node = node[part]
    if isinstance(node, h5py.Dataset):
        return node[:]
    raise ValueError(f"Path {h5_path!r} did not resolve to an HDF5 dataset")


def _resolve_h5_node(handle: h5py.File, h5_path: str) -> tuple[h5py.Dataset, str | None]:
    """Resolve an HDF5 path without reading the dataset into memory."""
    parts = h5_path.strip("/").split("/")
    node = handle
    for i, part in enumerate(parts):
        if isinstance(node, h5py.Dataset):
            return node, "/".join(parts[i:])
        node = node[part]
    if isinstance(node, h5py.Dataset):
        return node, None
    raise ValueError(f"Path {h5_path!r} did not resolve to an HDF5 dataset")


def _read_h5_slice(
    handle: h5py.File,
    h5_path: str,
    event_slice: slice,
    num_objects: int | None = None,
) -> np.ndarray:
    dataset, field = _resolve_h5_node(handle, h5_path)
    array = dataset[event_slice] if field is None else dataset[event_slice][field]
    if array.ndim == 2 and num_objects is not None:
        array = array[:, :num_objects]
    return array


def _infer_total_events(
    handle: h5py.File,
    event_inputs: list[str],
    object_collections: list[dict],
) -> int:
    if "metadata" in handle and "n_events" in handle["metadata"].attrs:
        return int(handle["metadata"].attrs["n_events"])
    if "n_events" in handle.attrs:
        return int(handle.attrs["n_events"])
    if object_collections:
        dataset, _ = _resolve_h5_node(handle, object_collections[0]["inputs"][0])
        return int(dataset.shape[0])
    if event_inputs:
        dataset, _ = _resolve_h5_node(handle, event_inputs[0])
        return int(dataset.shape[0])
    raise ValueError("At least one event input or object collection is required")


def _slice_events(array: np.ndarray, n_events: int, n_objects: int | None = None) -> np.ndarray:
    if array.ndim == 1 or n_objects is None:
        return array[:n_events]
    return array[:n_events, :n_objects]


def _find_collection(object_collections: list[dict], object_type: str) -> dict:
    for collection in object_collections:
        if collection.get("object_name") == object_type:
            return collection
    available = ", ".join(str(c.get("object_name")) for c in object_collections)
    raise ValueError(f"Object collection {object_type!r} is not configured. Available: {available}")


def _infer_dimensions(
    handle: h5py.File,
    event_inputs: list[str],
    object_collections: list[dict],
    num_events: int | None,
) -> tuple[int, int]:
    if object_collections:
        first_input = object_collections[0]["inputs"][0]
        probe = _resolve_h5_path(handle, first_input)
        total_events = probe.shape[0]
    elif event_inputs:
        probe = _resolve_h5_path(handle, event_inputs[0])
        total_events = probe.shape[0]
    else:
        raise ValueError("At least one event input or object collection is required")

    n_events = min(total_events, num_events) if num_events else total_events
    return total_events, n_events


def _read_event_inputs(
    handle: h5py.File,
    event_inputs: list[str],
    n_events: int,
    fallback: np.ndarray,
) -> np.ndarray:
    if not event_inputs:
        return fallback
    event_arrays = []
    for inp in event_inputs:
        arr = _resolve_h5_path(handle, inp)[:n_events].astype(np.float32)
        if arr.ndim > 1:
            arr = arr.reshape(n_events, -1)
        else:
            arr = arr[:, None]
        event_arrays.append(arr)
    return np.concatenate(event_arrays, axis=-1)


def _read_mask(
    handle: h5py.File,
    mask_input: str | None,
    n_events: int,
    n_objects: int,
) -> np.ndarray:
    if mask_input is None:
        return np.ones((n_events, n_objects), dtype=bool)
    mask = _resolve_h5_path(handle, mask_input)
    if mask.ndim == 1:
        mask = mask[:, None]
    return mask[:n_events, :n_objects].astype(bool)


def _read_collection(
    handle: h5py.File,
    collection: dict,
    n_events: int,
    num_objects: int | None,
    default_mask_input: str | None,
) -> tuple[np.ndarray, np.ndarray]:
    inputs = _as_list(collection.get("inputs"))
    if not inputs:
        raise ValueError(f"Object collection {collection.get('object_name')!r} has no inputs")

    feature_arrays = []
    n_objects = None
    for inp in inputs:
        arr = _resolve_h5_path(handle, inp).astype(np.float32)
        if arr.ndim == 1:
            arr = arr[:, None]
        if n_objects is None:
            total_objects = arr.shape[1]
            n_objects = min(total_objects, num_objects) if num_objects else total_objects
        feature_arrays.append(_slice_events(arr, n_events, n_objects))

    csts = np.stack(feature_arrays, axis=-1)
    csts = np.nan_to_num(csts, nan=0.0, posinf=0.0, neginf=0.0)
    csts = np.clip(csts, -1e4, 1e4)

    mask_input = collection.get("mask_input", default_mask_input)
    mask = _read_mask(handle, mask_input, n_events, csts.shape[1])
    return csts, mask


def _load_labels(
    handle: h5py.File,
    label_key: str | None,
    n_events: int,
) -> np.ndarray:
    if label_key is None:
        return np.zeros(n_events, dtype=np.int64)
    if label_key in handle:
        return handle[label_key][:n_events].astype(np.int64)
    if (
        "events" in handle
        and handle["events"].dtype.names
        and label_key in handle["events"].dtype.names
    ):
        return handle["events"][label_key][:n_events].astype(np.int64)
    log.warning("Label key %s not found; using zero labels", label_key)
    return np.zeros(n_events, dtype=np.int64)


def _read_event_inputs_slice(
    handle: h5py.File,
    event_inputs: list[str],
    event_slice: slice,
    fallback: np.ndarray,
) -> np.ndarray:
    if not event_inputs:
        return fallback
    n_events = len(range(*event_slice.indices(_infer_total_events(handle, event_inputs, []))))
    event_arrays = []
    for inp in event_inputs:
        arr = _read_h5_slice(handle, inp, event_slice).astype(np.float32)
        if arr.ndim > 1:
            arr = arr.reshape(n_events, -1)
        else:
            arr = arr[:, None]
        event_arrays.append(arr)
    return np.concatenate(event_arrays, axis=-1)


def _read_mask_slice(
    handle: h5py.File,
    mask_input: str | None,
    event_slice: slice,
    n_objects: int,
    n_events: int,
) -> np.ndarray:
    if mask_input is None:
        return np.ones((n_events, n_objects), dtype=bool)
    mask = _read_h5_slice(handle, mask_input, event_slice, n_objects)
    if mask.ndim == 1:
        mask = mask[:, None]
    return mask.astype(bool)


def _read_collection_slice(
    handle: h5py.File,
    collection: dict,
    event_slice: slice,
    num_objects: int | None,
    default_mask_input: str | None,
) -> tuple[np.ndarray, np.ndarray]:
    inputs = _as_list(collection.get("inputs"))
    if not inputs:
        raise ValueError(f"Object collection {collection.get('object_name')!r} has no inputs")

    feature_arrays = []
    n_objects = None
    for inp in inputs:
        arr = _read_h5_slice(handle, inp, event_slice).astype(np.float32)
        if arr.ndim == 1:
            arr = arr[:, None]
        if n_objects is None:
            total_objects = arr.shape[1]
            n_objects = min(total_objects, num_objects) if num_objects else total_objects
        feature_arrays.append(arr[:, :n_objects])

    csts = np.stack(feature_arrays, axis=-1)
    csts = np.nan_to_num(csts, nan=0.0, posinf=0.0, neginf=0.0)
    csts = np.clip(csts, -1e4, 1e4)

    mask_input = collection.get("mask_input", default_mask_input)
    mask = _read_mask_slice(handle, mask_input, event_slice, csts.shape[1], csts.shape[0])
    return csts, mask


def _load_labels_slice(
    handle: h5py.File,
    label_key: str | None,
    event_slice: slice,
    n_events: int,
) -> np.ndarray:
    if label_key is None:
        return np.zeros(n_events, dtype=np.int64)
    if label_key in handle:
        return handle[label_key][event_slice].astype(np.int64)
    if (
        "events" in handle
        and handle["events"].dtype.names
        and label_key in handle["events"].dtype.names
    ):
        return handle["events"][event_slice][label_key].astype(np.int64)
    log.warning("Label key %s not found; using zero labels", label_key)
    return np.zeros(n_events, dtype=np.int64)


def _load_chunk(
    handle: h5py.File,
    event_slice: slice,
    *,
    event_inputs: list[str],
    object_collections: list[dict],
    output_mode: str,
    object_type: str | None,
    mask_input: str | None,
    label_key: str | None,
    num_objects: int | None,
    max_objects: dict,
) -> dict:
    n_events = event_slice.stop - event_slice.start
    if output_mode == "combined":
        data = EventMapDataset._load_combined(
            handle,
            event_inputs,
            object_collections,
            mask_input,
            n_events,
            num_objects,
            max_objects,
            event_slice=event_slice,
        )
    elif output_mode == "object":
        if object_type is None:
            raise ValueError("output_mode='object' requires object_type")
        data = EventMapDataset._load_object(
            handle,
            event_inputs,
            object_collections,
            object_type,
            mask_input,
            n_events,
            num_objects,
            max_objects,
            event_slice=event_slice,
        )
    elif output_mode == "separate":
        data = EventMapDataset._load_separate(
            handle,
            event_inputs,
            object_collections,
            mask_input,
            n_events,
            num_objects,
            max_objects,
            event_slice=event_slice,
        )
    else:
        raise ValueError("output_mode must be one of 'combined', 'object', or 'separate'")

    data["labels"] = _load_labels_slice(handle, label_key, event_slice, n_events)
    return data


def _split_bounds(
    total_size: int,
    split: str,
    train_frac: float,
    val_frac: float,
) -> tuple[int, int]:
    train_size = int(total_size * train_frac)
    val_size = int(total_size * val_frac)
    if split in {"train", "fit"}:
        return 0, train_size
    if split in {"valid", "val", "validate"}:
        return train_size, train_size + val_size
    if split in {"test", "predict"}:
        return train_size + val_size, total_size
    raise ValueError("split must be one of 'train', 'valid', or 'test'")


def _distributed_shard(chunks: list[tuple[str, int, int]]) -> list[tuple[str, int, int]]:
    worker_info = torch.utils.data.get_worker_info()
    worker_id = worker_info.id if worker_info is not None else 0
    num_workers = worker_info.num_workers if worker_info is not None else 1

    rank = 0
    world_size = 1
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()

    shard_id = rank * num_workers + worker_id
    num_shards = world_size * num_workers
    return chunks[shard_id::num_shards]


class EventStreamDataset(IterableDataset):
    """Stream event-level HDF5 samples in chunks instead of loading all arrays.

    The output contract matches :class:`EventMapDataset`; this only changes how
    data are read. Splits are contiguous within each file so chunked reads stay
    efficient for large HDF5 samples.
    """

    def __init__(
        self,
        *,
        file_paths: list[str],
        split: str,
        event_inputs: list[str] | None = None,
        object_collections: list[dict] | None = None,
        output_mode: str = "combined",
        object_type: str | None = None,
        mask_input: str | None = None,
        label_key: str | None = None,
        num_events: int | None = None,
        num_objects: int | None = None,
        max_objects: dict | None = None,
        train_frac: float = 0.7,
        val_frac: float = 0.15,
        chunk_size: int = 1000,
        seed: int = 42,
        shuffle_chunks: bool = True,
    ) -> None:
        super().__init__()
        self.file_paths = list(file_paths)
        self.split = split
        self.event_inputs = _as_list(event_inputs)
        self.object_collections = [
            _as_dict(collection) for collection in _as_list(object_collections)
        ]
        self.output_mode = output_mode
        self.object_type = object_type
        self.mask_input = mask_input
        self.label_key = label_key
        self.num_events = num_events
        self.num_objects = num_objects
        self.max_objects = _as_dict(max_objects)
        self.train_frac = train_frac
        self.val_frac = val_frac
        self.chunk_size = chunk_size
        self.seed = seed
        self.shuffle_chunks = shuffle_chunks
        self.segments = self._build_segments()
        self.num_samples = sum(end - start for _, start, end in self.segments)
        log.info("Prepared %s streamed events for %s split", self.num_samples, split)

    def _file_event_count(self, path: str) -> int:
        with h5py.File(path, mode="r") as handle:
            total_events = _infer_total_events(handle, self.event_inputs, self.object_collections)
        return min(total_events, self.num_events) if self.num_events else total_events

    def _build_segments(self) -> list[tuple[str, int, int]]:
        segments = []
        for path in self.file_paths:
            total_events = self._file_event_count(path)
            start, end = _split_bounds(
                total_events,
                self.split,
                self.train_frac,
                self.val_frac,
            )
            if end > start:
                segments.append((path, start, end))
        return segments

    def _build_chunks(self) -> list[tuple[str, int, int]]:
        chunks = []
        for path, start, end in self.segments:
            for chunk_start in range(start, end, self.chunk_size):
                chunk_end = min(chunk_start + self.chunk_size, end)
                chunks.append((path, chunk_start, chunk_end))
        if self.shuffle_chunks and self.split in {"train", "fit"}:
            rng = np.random.default_rng(self.seed)
            rng.shuffle(chunks)
        return chunks

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self):
        chunks = _distributed_shard(self._build_chunks())
        current_path = None
        handle = None
        try:
            for path, chunk_start, chunk_end in chunks:
                if path != current_path:
                    if handle is not None:
                        handle.close()
                    handle = h5py.File(path, mode="r")
                    current_path = path

                chunk = _load_chunk(
                    handle,
                    slice(chunk_start, chunk_end),
                    event_inputs=self.event_inputs,
                    object_collections=self.object_collections,
                    output_mode=self.output_mode,
                    object_type=self.object_type,
                    mask_input=self.mask_input,
                    label_key=self.label_key,
                    num_objects=self.num_objects,
                    max_objects=self.max_objects,
                )

                for idx in range(len(chunk["labels"])):
                    yield {key: value[idx] for key, value in chunk.items()}
        finally:
            if handle is not None:
                handle.close()


class EventMapDataset(Dataset):
    """Loads event-level data from an HDF5 file using explicit path configuration.

    ``output_mode="combined"`` returns one object sequence under ``csts`` /
    ``mask``. This is the single-tokenizer-across-object-types mode.

    ``output_mode="object"`` returns one configured object collection under
    ``csts`` / ``mask``. This is for one tokenizer per object type.

    ``output_mode="separate"`` returns one tensor per object collection, e.g.
    ``jets_csts`` / ``jets_mask`` and ``electrons_csts`` / ``electrons_mask``.
    This is for future multimodal or joint event tokenizer studies.
    """

    def __init__(
        self,
        file_path: str,
        event_inputs: list[str] | None = None,
        object_collections: list[dict] | None = None,
        output_mode: str = "combined",
        object_type: str | None = None,
        mask_input: str | None = None,
        label_key: str | None = None,
        num_events: int | None = None,
        num_objects: int | None = None,
        max_objects: dict | None = None,
    ) -> None:
        super().__init__()
        event_inputs = _as_list(event_inputs)
        object_collections = [_as_dict(collection) for collection in _as_list(object_collections)]
        max_objects = _as_dict(max_objects)

        self.data_dict = {}
        with h5py.File(file_path, mode="r") as handle:
            _, n_events = _infer_dimensions(handle, event_inputs, object_collections, num_events)

            if output_mode == "combined":
                self.data_dict.update(
                    self._load_combined(
                        handle,
                        event_inputs,
                        object_collections,
                        mask_input,
                        n_events,
                        num_objects,
                        max_objects,
                    )
                )
            elif output_mode == "object":
                if object_type is None:
                    raise ValueError("output_mode='object' requires object_type")
                self.data_dict.update(
                    self._load_object(
                        handle,
                        event_inputs,
                        object_collections,
                        object_type,
                        mask_input,
                        n_events,
                        num_objects,
                        max_objects,
                    )
                )
            elif output_mode == "separate":
                self.data_dict.update(
                    self._load_separate(
                        handle,
                        event_inputs,
                        object_collections,
                        mask_input,
                        n_events,
                        num_objects,
                        max_objects,
                    )
                )
            else:
                raise ValueError(
                    "output_mode must be one of 'combined', 'object', or 'separate'"
                )

            self.data_dict["labels"] = _load_labels(handle, label_key, n_events)

        self.num_events = len(self.data_dict["labels"])
        log.info("Loaded %s events in %s mode from %s", self.num_events, output_mode, file_path)

    @staticmethod
    def _load_combined(
        handle: h5py.File,
        event_inputs: list[str],
        object_collections: list[dict],
        mask_input: str | None,
        n_events: int,
        num_objects: int | None,
        max_objects: dict,
        event_slice: slice | None = None,
    ) -> dict:
        cst_parts = []
        mask_parts = []
        type_parts = []
        n_features = None
        for type_id, collection in enumerate(object_collections, start=1):
            name = collection.get("object_name", f"object_{type_id}")
            if event_slice is None:
                csts, mask = _read_collection(
                    handle,
                    collection,
                    n_events,
                    max_objects.get(name, num_objects),
                    mask_input,
                )
            else:
                csts, mask = _read_collection_slice(
                    handle,
                    collection,
                    event_slice,
                    max_objects.get(name, num_objects),
                    mask_input,
                )
            if n_features is None:
                n_features = csts.shape[-1]
            elif csts.shape[-1] != n_features:
                raise ValueError(
                    "Combined mode requires each object collection to have the same "
                    f"number of inputs. {name!r} has {csts.shape[-1]}, expected {n_features}."
                )
            cst_parts.append(csts)
            mask_parts.append(mask)
            type_parts.append(np.full(mask.shape, type_id, dtype=np.int64))

        if cst_parts:
            csts = np.concatenate(cst_parts, axis=1)
            mask = np.concatenate(mask_parts, axis=1)
            type_ids = np.concatenate(type_parts, axis=1)
            fallback_jets = mask.sum(axis=1).astype(np.float32).reshape(-1, 1)
        else:
            csts = np.zeros((n_events, 0, 0), dtype=np.float32)
            mask = np.zeros((n_events, 0), dtype=bool)
            type_ids = np.zeros((n_events, 0), dtype=np.int64)
            fallback_jets = np.zeros((n_events, 1), dtype=np.float32)

        return {
            "csts": csts,
            "mask": mask,
            "type_ids": type_ids,
            # TODO(event-tokenization): keep this compatibility shim until
            # tokenizer models consume batch["event"] directly.
            "jets": (
                _read_event_inputs(handle, event_inputs, n_events, fallback_jets)
                if event_slice is None
                else _read_event_inputs_slice(handle, event_inputs, event_slice, fallback_jets)
            ),
        }

    @staticmethod
    def _load_object(
        handle: h5py.File,
        event_inputs: list[str],
        object_collections: list[dict],
        object_type: str,
        mask_input: str | None,
        n_events: int,
        num_objects: int | None,
        max_objects: dict,
        event_slice: slice | None = None,
    ) -> dict:
        collection = _find_collection(object_collections, object_type)
        if event_slice is None:
            csts, mask = _read_collection(
                handle,
                collection,
                n_events,
                max_objects.get(object_type, num_objects),
                mask_input,
            )
        else:
            csts, mask = _read_collection_slice(
                handle,
                collection,
                event_slice,
                max_objects.get(object_type, num_objects),
                mask_input,
            )
        fallback_jets = mask.sum(axis=1).astype(np.float32).reshape(-1, 1)
        return {
            "csts": csts,
            "mask": mask,
            # TODO(event-tokenization): keep this compatibility shim until
            # tokenizer models consume batch["event"] directly.
            "jets": (
                _read_event_inputs(handle, event_inputs, n_events, fallback_jets)
                if event_slice is None
                else _read_event_inputs_slice(handle, event_inputs, event_slice, fallback_jets)
            ),
        }

    @staticmethod
    def _load_separate(
        handle: h5py.File,
        event_inputs: list[str],
        object_collections: list[dict],
        mask_input: str | None,
        n_events: int,
        num_objects: int | None,
        max_objects: dict,
        event_slice: slice | None = None,
    ) -> dict:
        data = {}
        for collection in object_collections:
            name = collection.get("object_name")
            if name is None:
                raise ValueError("Each object collection needs object_name in separate mode")
            if event_slice is None:
                csts, mask = _read_collection(
                    handle,
                    collection,
                    n_events,
                    max_objects.get(name, num_objects),
                    mask_input,
                )
            else:
                csts, mask = _read_collection_slice(
                    handle,
                    collection,
                    event_slice,
                    max_objects.get(name, num_objects),
                    mask_input,
                )
            data[f"{name}_csts"] = csts
            data[f"{name}_mask"] = mask

        fallback_event = np.zeros((n_events, 0), dtype=np.float32)
        event = (
            _read_event_inputs(handle, event_inputs, n_events, fallback_event)
            if event_slice is None
            else _read_event_inputs_slice(handle, event_inputs, event_slice, fallback_event)
        )
        data["event"] = event
        # TODO(event-tokenization): this mirrors event inputs into "jets" only
        # for compatibility with current heptokens tokenizer interfaces.
        data["jets"] = event
        return data

    def __len__(self) -> int:
        return self.num_events

    def __getitem__(self, idx: int) -> dict:
        return {k: v[idx] for k, v in self.data_dict.items()}


class EventSingleFileMapModule(BaseMapModule):
    """DataModule that loads one or more event-level HDF5 files and splits them."""

    def __init__(
        self,
        *,
        data_path: str | None = None,
        data_paths: list[str] | None = None,
        train_frac: float = 0.7,
        val_frac: float = 0.15,
        test_frac: float = 0.15,
        seed: int = 42,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        dataset_config = dict(self.data_config)
        data_path = data_path or dataset_config.pop("data_path", None)
        data_paths = data_paths or dataset_config.pop("data_paths", None)
        self.data_config = dataset_config

        if not abs(train_frac + val_frac + test_frac - 1.0) < 1e-6:
            raise ValueError("train_frac + val_frac + test_frac must sum to 1.0")
        if data_paths:
            paths = list(data_paths)
        elif data_path is not None:
            paths = [data_path]
        else:
            raise ValueError("Either data_path or data_paths must be provided")

        self.data_path = paths[0]
        self.data_paths = paths
        self.train_frac = train_frac
        self.val_frac = val_frac
        self.test_frac = test_frac
        self.seed = seed

        datasets = [EventMapDataset(path, **dataset_config) for path in paths]
        full_dataset = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]
        total_size = len(full_dataset)
        train_size = int(total_size * train_frac)
        val_size = int(total_size * val_frac)
        test_size = total_size - train_size - val_size

        generator = torch.Generator().manual_seed(seed)
        self.train_set, self.valid_set, self.test_set = random_split(
            full_dataset, [train_size, val_size, test_size], generator=generator
        )

    def setup(self, stage: str) -> None:
        """Datasets are already split in __init__."""


class EventStreamMapModule(BaseMapModule):
    """DataModule that streams one or more event-level HDF5 files in chunks."""

    def __init__(
        self,
        *,
        data_path: str | None = None,
        data_paths: list[str] | None = None,
        train_frac: float = 0.7,
        val_frac: float = 0.15,
        test_frac: float = 0.15,
        seed: int = 42,
        chunk_size: int = 1000,
        shuffle_chunks: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        dataset_config = dict(self.data_config)
        data_path = data_path or dataset_config.pop("data_path", None)
        data_paths = data_paths or dataset_config.pop("data_paths", None)
        self.data_config = dataset_config

        if not abs(train_frac + val_frac + test_frac - 1.0) < 1e-6:
            raise ValueError("train_frac + val_frac + test_frac must sum to 1.0")
        if data_paths:
            paths = list(data_paths)
        elif data_path is not None:
            paths = [data_path]
        else:
            raise ValueError("Either data_path or data_paths must be provided")

        self.data_path = paths[0]
        self.data_paths = paths
        self.train_frac = train_frac
        self.val_frac = val_frac
        self.test_frac = test_frac
        self.seed = seed
        self.chunk_size = chunk_size
        self.shuffle_chunks = shuffle_chunks

        stream_dataset_config = {
            **dataset_config,
            "file_paths": paths,
            "train_frac": train_frac,
            "val_frac": val_frac,
            "chunk_size": chunk_size,
            "seed": seed,
            "shuffle_chunks": shuffle_chunks,
        }
        self.train_set = EventStreamDataset(split="train", **stream_dataset_config)
        self.valid_set = EventStreamDataset(split="valid", **stream_dataset_config)
        self.test_set = EventStreamDataset(split="test", **stream_dataset_config)

    def setup(self, stage: str) -> None:
        """Datasets are already configured in __init__."""

    def _get_dataloader(
        self,
        dataset: IterableDataset,
        shuffle: bool,
        drop_last: bool,
    ) -> DataLoader:
        """Create a dataloader for iterable datasets.

        ``shuffle`` is intentionally ignored; chunk shuffling happens in
        :class:`EventStreamDataset`.
        """
        collate_fn = None
        if self.transforms is not None:
            collate_fn = partial(collate_and_transform, transforms=self.transforms)

        dataloader_kwargs = {}
        if self.num_workers > 0:
            dataloader_kwargs["persistent_workers"] = self.persistent_workers
            if self.multiprocessing_context is not None:
                dataloader_kwargs["multiprocessing_context"] = self.multiprocessing_context

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=drop_last,
            collate_fn=collate_fn,
            **dataloader_kwargs,
        )
