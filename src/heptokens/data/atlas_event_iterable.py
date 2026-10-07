"""Iterable ATLAS event HDF5 datasets for memory-light tokenizer training."""

from __future__ import annotations

import hashlib
import logging
from collections import defaultdict
from contextlib import ExitStack
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset

from heptokens.data.atlas_event_mappable import AtlasEventMapDataset, _as_dict, _as_list
from heptokens.data.atlas_mappable import BaseMapModule
from heptokens.data.collation import collate_and_transform

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class _FileSpec:
    path: str
    n_events: int
    domain: str
    split_ids: np.ndarray


def _resolve_h5_slice(
    handle: h5py.File,
    h5_path: str,
    event_slice: slice,
    n_objects: int | None = None,
) -> np.ndarray:
    """Read only an event slice from a dataset or compound-dataset field."""
    parts = h5_path.strip("/").split("/")
    node = handle
    field: str | None = None
    for index, part in enumerate(parts):
        if isinstance(node, h5py.Dataset):
            field = "/".join(parts[index:])
            break
        node = node[part]

    if not isinstance(node, h5py.Dataset):
        raise ValueError(f"Path {h5_path!r} did not resolve to an HDF5 dataset")

    dataset = node.fields(field) if field is not None else node
    if n_objects is None or len(node.shape) < 2:
        return dataset[event_slice]
    return dataset[event_slice, :n_objects]


def _infer_domain(path: str, explicit_domain: str | None = None) -> str:
    if explicit_domain is not None:
        return str(explicit_domain)
    # The immediate parent is the staged sample category (MC, data, or
    # realdata). Looking at every path component would misclassify paths below
    # a generic directory such as /home/zephyr/Data/ as real data.
    parent = Path(path).parent.name.lower()
    return "data" if parent in {"data", "realdata"} else "mc"


def _assign_global_event_splits(
    specs: list[tuple[str, int, str]],
    train_frac: float,
    val_frac: float,
    seed: int,
) -> list[_FileSpec]:
    """Reproduce the eager loader's seeded global ``random_split`` membership.

    Only a uint8 split id is retained per event. Feature arrays remain on disk.
    """
    total_events = sum(n_events for _, n_events, _ in specs)
    train_size = int(total_events * train_frac)
    val_size = int(total_events * val_frac)

    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(total_events, generator=generator).numpy()
    split_ids = np.full(total_events, 2, dtype=np.uint8)
    split_ids[permutation[:train_size]] = 0
    split_ids[permutation[train_size : train_size + val_size]] = 1
    del permutation

    assigned: list[_FileSpec] = []
    offset = 0
    for path, n_events, domain in specs:
        assigned.append(
            _FileSpec(
                path=path,
                n_events=n_events,
                domain=domain,
                split_ids=split_ids[offset : offset + n_events],
            )
        )
        offset += n_events
    return assigned


class AtlasEventObjectIterableDataset(IterableDataset):
    """Stream H5 chunks with legacy ordering or bounded cross-file mixing."""

    def __init__(
        self,
        file_specs: list[_FileSpec],
        *,
        split_id: int,
        seed: int,
        shuffle: bool,
        event_inputs: list[str] | None = None,
        object_collections: list[dict] | None = None,
        object_type: str,
        mask_input: str | None = None,
        label_key: str | None = None,
        num_objects: int | None = None,
        max_objects: dict | None = None,
        chunk_size: int = 4096,
        shuffle_mode: str = "legacy",
        shuffle_buffer_size: int = 16384,
        shuffle_file_window_size: int = 16,
    ) -> None:
        super().__init__()
        self.file_specs = list(file_specs)
        self.split_id = int(split_id)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self._iteration = 0
        self.event_inputs = _as_list(event_inputs)
        self.object_collections = [
            _as_dict(collection) for collection in _as_list(object_collections)
        ]
        self.object_type = object_type
        self.mask_input = mask_input
        self.label_key = label_key
        self.num_objects = num_objects
        self.max_objects = _as_dict(max_objects)
        self.chunk_size = int(chunk_size)
        self.shuffle_mode = shuffle_mode
        self.shuffle_buffer_size = int(shuffle_buffer_size)
        self.shuffle_file_window_size = int(shuffle_file_window_size)
        self.num_events = int(
            sum(np.count_nonzero(spec.split_ids == self.split_id) for spec in self.file_specs)
        )

        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.shuffle_mode not in {"legacy", "buffered"}:
            raise ValueError("shuffle_mode must be 'legacy' or 'buffered'")
        if self.shuffle_mode == "buffered":
            if self.shuffle_buffer_size <= 0:
                raise ValueError("shuffle_buffer_size must be positive")
            if self.shuffle_file_window_size <= 0:
                raise ValueError("shuffle_file_window_size must be positive")
        if not self.file_specs:
            log.warning("AtlasEventObjectIterableDataset created with no files")

    def __len__(self) -> int:
        return self.num_events

    def _iteration_rng(self) -> np.random.Generator:
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            # Every worker must derive the same file permutation before taking
            # its disjoint stride. PyTorch assigns base_seed + worker_id.
            iteration_seed = worker_info.seed - worker_info.id + self._iteration
        else:
            iteration_seed = self.seed + self._iteration
        self._iteration += 1
        return np.random.default_rng(iteration_seed % (2**63 - 1))

    def _worker_file_specs(self, rng: np.random.Generator) -> list[_FileSpec]:
        specs = list(self.file_specs)
        if self.shuffle:
            rng.shuffle(specs)
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            return specs
        return specs[worker_info.id :: worker_info.num_workers]

    def __iter__(self):
        rng = self._iteration_rng()
        collection = AtlasEventMapDataset._find_collection(
            self.object_collections,
            self.object_type,
        )
        if self.shuffle_mode == "buffered" and self.shuffle:
            samples = self._iter_buffered_samples(rng, collection)
            yield from self._bounded_shuffle(samples, rng)
        else:
            # Keep the original Google iteration and RNG calls unchanged.
            # Validation/test always follow this unshuffled path.
            yield from self._iter_legacy_samples(rng, collection)

    def _iter_legacy_samples(self, rng: np.random.Generator, collection: dict):
        for spec in self._worker_file_specs(rng):
            chunk_starts = list(range(0, spec.n_events, self.chunk_size))
            if self.shuffle:
                rng.shuffle(chunk_starts)
            with h5py.File(spec.path, mode="r") as handle:
                for start in chunk_starts:
                    end = min(start + self.chunk_size, spec.n_events)
                    selected = spec.split_ids[start:end] == self.split_id
                    if not np.any(selected):
                        continue
                    yield from self._read_chunk(
                        handle,
                        collection,
                        slice(start, end),
                        selected,
                        rng,
                    )

    def _iter_buffered_samples(self, rng: np.random.Generator, collection: dict):
        """Interleave chunks from a bounded window, opening each file once."""
        specs = self._worker_file_specs(rng)
        for offset in range(0, len(specs), self.shuffle_file_window_size):
            window = specs[offset : offset + self.shuffle_file_window_size]
            chunk_tasks = [
                (spec, start)
                for spec in window
                for start in range(0, spec.n_events, self.chunk_size)
            ]
            rng.shuffle(chunk_tasks)
            with ExitStack() as stack:
                handles = {
                    spec.path: stack.enter_context(h5py.File(spec.path, mode="r"))
                    for spec in window
                }
                for spec, start in chunk_tasks:
                    end = min(start + self.chunk_size, spec.n_events)
                    selected = spec.split_ids[start:end] == self.split_id
                    if np.any(selected):
                        yield from self._read_chunk(
                            handles[spec.path], collection, slice(start, end), selected, rng
                        )

    def _bounded_shuffle(self, samples, rng: np.random.Generator):
        """Mix at most the configured number of owned event arrays per worker."""
        buffer = []
        for sample in samples:
            # Retaining a view would keep its entire H5 chunk alive.
            sample = {
                key: value.copy() if isinstance(value, np.ndarray) else value
                for key, value in sample.items()
            }
            if len(buffer) < self.shuffle_buffer_size:
                buffer.append(sample)
                continue
            index = int(rng.integers(len(buffer)))
            yield buffer[index]
            buffer[index] = sample
        rng.shuffle(buffer)
        yield from buffer

    def _read_chunk(
        self,
        handle: h5py.File,
        collection: dict,
        event_slice: slice,
        selected: np.ndarray,
        rng: np.random.Generator,
    ):
        inputs = _as_list(collection.get("inputs"))
        if not inputs:
            raise ValueError(f"Object collection {self.object_type!r} has no inputs")

        n_objects = self._chunk_num_objects(handle, inputs[0])
        feature_arrays = []
        for h5_path in inputs:
            array = _resolve_h5_slice(handle, h5_path, event_slice, n_objects).astype(
                np.float32
            )
            if array.ndim == 1:
                array = array[:, None]
            feature_arrays.append(array)

        csts = np.stack(feature_arrays, axis=-1)
        csts = np.nan_to_num(csts, nan=0.0, posinf=0.0, neginf=0.0)
        csts = np.clip(csts, -1e4, 1e4)

        mask_path = collection.get("mask_input", self.mask_input)
        if mask_path is None:
            mask = np.ones(csts.shape[:2], dtype=bool)
        else:
            mask = _resolve_h5_slice(handle, mask_path, event_slice, csts.shape[1])
            if mask.ndim == 1:
                mask = mask[:, None]
            mask = mask.astype(bool)

        fallback_jets = mask.sum(axis=1).astype(np.float32).reshape(-1, 1)
        jets = self._read_event_inputs(handle, event_slice, fallback_jets)
        labels = self._read_labels(handle, event_slice, len(csts))

        selected_indices = np.flatnonzero(selected)
        if self.shuffle:
            rng.shuffle(selected_indices)
        for local_index in selected_indices:
            yield {
                "csts": csts[local_index],
                "mask": mask[local_index],
                "jets": jets[local_index],
                "labels": labels[local_index],
            }

    def _chunk_num_objects(self, handle: h5py.File, first_input: str) -> int | None:
        if self.object_type in self.max_objects:
            return int(self.max_objects[self.object_type])
        if self.num_objects is not None:
            return int(self.num_objects)

        parts = first_input.strip("/").split("/")
        node = handle
        for part in parts:
            if isinstance(node, h5py.Dataset):
                break
            node = node[part]
        if not isinstance(node, h5py.Dataset) or len(node.shape) < 2:
            return None
        return int(node.shape[1])

    def _read_event_inputs(
        self,
        handle: h5py.File,
        event_slice: slice,
        fallback: np.ndarray,
    ) -> np.ndarray:
        if not self.event_inputs:
            return fallback

        n_events = len(fallback)
        event_arrays = []
        for h5_path in self.event_inputs:
            array = _resolve_h5_slice(handle, h5_path, event_slice).astype(np.float32)
            if array.ndim > 1:
                array = array.reshape(n_events, -1)
            else:
                array = array[:, None]
            event_arrays.append(array)
        return np.concatenate(event_arrays, axis=-1)

    def _read_labels(
        self,
        handle: h5py.File,
        event_slice: slice,
        n_events: int,
    ) -> np.ndarray:
        if self.label_key is None:
            return np.zeros(n_events, dtype=np.int64)
        if self.label_key in handle:
            return handle[self.label_key][event_slice].astype(np.int64)
        if (
            "events" in handle
            and handle["events"].dtype.names
            and self.label_key in handle["events"].dtype.names
        ):
            return handle["events"].fields(self.label_key)[event_slice].astype(np.int64)
        return np.zeros(n_events, dtype=np.int64)


class AtlasEventObjectIterableModule(BaseMapModule):
    """Stream event HDF5 files while preserving eager global event splits."""

    def __init__(
        self,
        *,
        data_path: str | None = None,
        data_paths: list[str] | None = None,
        data_domains: list[str] | None = None,
        train_frac: float = 0.7,
        val_frac: float = 0.15,
        test_frac: float = 0.15,
        seed: int = 42,
        output_mode: str = "object",
        object_type: str | None = None,
        chunk_size: int = 4096,
        shuffle_mode: str = "legacy",
        shuffle_buffer_size: int = 16384,
        shuffle_file_window_size: int = 16,
        split_by_domain: bool = True,
        sampling_domain_fractions: dict[str, float] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if output_mode != "object":
            raise ValueError("AtlasEventObjectIterableModule only supports output_mode='object'")
        if object_type is None:
            raise ValueError("AtlasEventObjectIterableModule requires object_type")
        if not abs(train_frac + val_frac + test_frac - 1.0) < 1e-6:
            raise ValueError("train_frac + val_frac + test_frac must sum to 1.0")
        if shuffle_mode not in {"legacy", "buffered"}:
            raise ValueError("shuffle_mode must be 'legacy' or 'buffered'")
        if split_by_domain:
            log.info(
                "split_by_domain is retained for config compatibility; the iterable "
                "loader now reproduces the eager global event-level split"
            )
        if sampling_domain_fractions:
            log.warning(
                "sampling_domain_fractions is not yet supported by the iterable loader"
            )

        dataset_config = dict(self.data_config)
        data_path = data_path or dataset_config.pop("data_path", None)
        data_paths = data_paths or dataset_config.pop("data_paths", None)
        self.data_config = dataset_config

        if data_paths:
            paths = [str(path) for path in data_paths]
        elif data_path is not None:
            paths = [str(data_path)]
        else:
            raise ValueError("Either data_path or data_paths must be provided")
        if data_domains is not None and len(data_domains) != len(paths):
            raise ValueError(
                "data_domains must contain one domain label per data path: "
                f"got {len(data_domains)} labels for {len(paths)} paths"
            )

        raw_specs = self._build_file_specs(paths, data_domains)
        specs = _assign_global_event_splits(
            raw_specs,
            train_frac,
            val_frac,
            seed,
        )

        common_dataset_kwargs = dict(
            event_inputs=dataset_config.get("event_inputs"),
            object_collections=dataset_config.get("object_collections"),
            object_type=object_type,
            mask_input=dataset_config.get("mask_input"),
            label_key=dataset_config.get("label_key"),
            num_objects=dataset_config.get("num_objects"),
            max_objects=dataset_config.get("max_objects"),
            chunk_size=chunk_size,
            shuffle_mode=shuffle_mode,
            shuffle_buffer_size=shuffle_buffer_size,
            shuffle_file_window_size=shuffle_file_window_size,
        )
        self.train_set = AtlasEventObjectIterableDataset(
            specs,
            split_id=0,
            seed=seed,
            shuffle=True,
            **common_dataset_kwargs,
        )
        self.valid_set = AtlasEventObjectIterableDataset(
            specs,
            split_id=1,
            seed=seed,
            shuffle=False,
            **common_dataset_kwargs,
        )
        self.test_set = AtlasEventObjectIterableDataset(
            specs,
            split_id=2,
            seed=seed,
            shuffle=False,
            **common_dataset_kwargs,
        )
        self._train_loader_generation = 0

        log.info(
            "Iterable event-object split: train=%d events val=%d test=%d",
            len(self.train_set),
            len(self.valid_set),
            len(self.test_set),
        )
        fingerprint = hashlib.sha256()
        for spec in specs:
            fingerprint.update(Path(spec.path).name.encode())
            fingerprint.update(spec.n_events.to_bytes(8, "little"))
            fingerprint.update(spec.split_ids.tobytes())
        log.info("Iterable split fingerprint: %s", fingerprint.hexdigest())
        self._report_split_statistics(specs, object_type, chunk_size)

    def _build_file_specs(
        self,
        paths: list[str],
        data_domains: list[str] | None,
    ) -> list[tuple[str, int, str]]:
        specs = []
        num_events = self.data_config.get("num_events")
        event_inputs = self.data_config.get("event_inputs")
        object_collections = self.data_config.get("object_collections")
        for index, path in enumerate(paths):
            explicit_domain = data_domains[index] if data_domains is not None else None
            with h5py.File(path, mode="r") as handle:
                _, n_events = AtlasEventMapDataset._infer_dimensions(
                    handle,
                    _as_list(event_inputs),
                    [_as_dict(collection) for collection in _as_list(object_collections)],
                    num_events,
                )
            specs.append((path, int(n_events), _infer_domain(path, explicit_domain)))
        return specs

    def _report_split_statistics(
        self,
        specs: list[_FileSpec],
        object_type: str,
        chunk_size: int,
    ) -> None:
        """Report exact event and valid-object composition without loading features."""
        collection = AtlasEventMapDataset._find_collection(
            [_as_dict(item) for item in _as_list(self.data_config.get("object_collections"))],
            object_type,
        )
        inputs = _as_list(collection.get("inputs"))
        mask_path = collection.get("mask_input", self.data_config.get("mask_input"))
        num_objects = _as_dict(self.data_config.get("max_objects")).get(
            object_type,
            self.data_config.get("num_objects"),
        )
        stats: dict[tuple[int, str], list[int]] = defaultdict(lambda: [0, 0])

        for spec in specs:
            for split_id in (0, 1, 2):
                stats[(split_id, spec.domain)][0] += int(
                    np.count_nonzero(spec.split_ids == split_id)
                )

            with h5py.File(spec.path, mode="r") as handle:
                if num_objects is None and inputs:
                    node = handle
                    for part in inputs[0].strip("/").split("/"):
                        if isinstance(node, h5py.Dataset):
                            break
                        node = node[part]
                    n_objects = int(node.shape[1]) if len(node.shape) >= 2 else None
                else:
                    n_objects = int(num_objects) if num_objects is not None else None

                for start in range(0, spec.n_events, chunk_size):
                    end = min(start + chunk_size, spec.n_events)
                    if mask_path is None:
                        valid_per_event = np.full(
                            end - start,
                            n_objects or 1,
                            dtype=np.int64,
                        )
                    else:
                        mask = _resolve_h5_slice(
                            handle,
                            mask_path,
                            slice(start, end),
                            n_objects,
                        ).astype(bool)
                        if mask.ndim == 1:
                            mask = mask[:, None]
                        valid_per_event = mask.sum(axis=1, dtype=np.int64)
                    chunk_splits = spec.split_ids[start:end]
                    for split_id in (0, 1, 2):
                        stats[(split_id, spec.domain)][1] += int(
                            valid_per_event[chunk_splits == split_id].sum()
                        )

        split_names = ("train", "val", "test")
        for split_id, split_name in enumerate(split_names):
            for domain in sorted({spec.domain for spec in specs}):
                events, valid_objects = stats[(split_id, domain)]
                log.info(
                    "Iterable split composition %s/%s: events=%d valid_objects=%d",
                    split_name,
                    domain,
                    events,
                    valid_objects,
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
        if dataset.shuffle_mode == "buffered":
            if shuffle:
                loader_seed = dataset.seed + self._train_loader_generation
                self._train_loader_generation += 1
            else:
                loader_seed = dataset.seed + 100_000 + dataset.split_id
            dataloader_kwargs["generator"] = torch.Generator().manual_seed(loader_seed)
        if self.num_workers > 0:
            dataloader_kwargs["persistent_workers"] = self.persistent_workers
            if self.multiprocessing_context is not None:
                dataloader_kwargs["multiprocessing_context"] = self.multiprocessing_context

        if sampler is not None:
            log.warning("Ignoring sampler for iterable event-object dataloader")
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
        for dataset in (self.valid_set, self.train_set, self.test_set):
            try:
                return next(iter(dataset))
            except StopIteration:
                continue
        raise RuntimeError("No events available in any iterable split")
