"""Iterable ATLAS event HDF5 datasets for memory-light tokenizer training."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from functools import partial
import logging
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
    parts = set(Path(path).parts)
    return "data" if "realdata" in parts else "mc"


def _split_specs_by_domain(
    specs: list[_FileSpec],
    train_frac: float,
    val_frac: float,
    seed: int,
    split_by_domain: bool,
) -> tuple[list[_FileSpec], list[_FileSpec], list[_FileSpec]]:
    groups: dict[str, list[_FileSpec]] = defaultdict(list)
    if split_by_domain:
        for spec in specs:
            groups[spec.domain].append(spec)
    else:
        groups["all"] = list(specs)

    train: list[_FileSpec] = []
    val: list[_FileSpec] = []
    test: list[_FileSpec] = []
    rng = np.random.default_rng(seed)

    for domain, domain_specs in sorted(groups.items()):
        order = rng.permutation(len(domain_specs))
        shuffled = [domain_specs[int(i)] for i in order]
        n_files = len(shuffled)
        n_train = int(n_files * train_frac)
        n_val = int(n_files * val_frac)
        train.extend(shuffled[:n_train])
        val.extend(shuffled[n_train : n_train + n_val])
        test.extend(shuffled[n_train + n_val :])
        log.info(
            "File split %s: train=%d val=%d test=%d",
            domain,
            n_train,
            n_val,
            n_files - n_train - n_val,
        )

    for split_specs in (train, val, test):
        rng.shuffle(split_specs)
    return train, val, test


class AtlasEventObjectIterableDataset(IterableDataset):
    """Stream one object collection from event HDF5 files in chunks."""

    def __init__(
        self,
        file_specs: list[_FileSpec],
        *,
        event_inputs: list[str] | None = None,
        object_collections: list[dict] | None = None,
        object_type: str,
        mask_input: str | None = None,
        label_key: str | None = None,
        num_objects: int | None = None,
        max_objects: dict | None = None,
        chunk_size: int = 4096,
    ) -> None:
        super().__init__()
        self.file_specs = list(file_specs)
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
        self.num_events = int(sum(spec.n_events for spec in self.file_specs))

        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if not self.file_specs:
            log.warning("AtlasEventObjectIterableDataset created with no files")

    def __len__(self) -> int:
        return self.num_events

    def _worker_file_specs(self) -> list[_FileSpec]:
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            return self.file_specs
        return [
            spec
            for index, spec in enumerate(self.file_specs)
            if index % worker_info.num_workers == worker_info.id
        ]

    def __iter__(self):
        collection = AtlasEventMapDataset._find_collection(
            self.object_collections,
            self.object_type,
        )
        for spec in self._worker_file_specs():
            with h5py.File(spec.path, mode="r") as handle:
                for start in range(0, spec.n_events, self.chunk_size):
                    end = min(start + self.chunk_size, spec.n_events)
                    yield from self._read_chunk(handle, collection, slice(start, end))

    def _read_chunk(
        self,
        handle: h5py.File,
        collection: dict,
        event_slice: slice,
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

        for local_index in range(len(csts)):
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
    """DataModule that streams event HDF5 files with random file-level splits."""

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
        if sampling_domain_fractions:
            log.warning(
                "sampling_domain_fractions is ignored by the iterable file-split loader"
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

        specs = self._build_file_specs(paths, data_domains)
        train_specs, val_specs, test_specs = _split_specs_by_domain(
            specs,
            train_frac,
            val_frac,
            seed,
            split_by_domain,
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
        )
        self.train_set = AtlasEventObjectIterableDataset(
            train_specs,
            **common_dataset_kwargs,
        )
        self.valid_set = AtlasEventObjectIterableDataset(
            val_specs,
            **common_dataset_kwargs,
        )
        self.test_set = AtlasEventObjectIterableDataset(
            test_specs,
            **common_dataset_kwargs,
        )

        log.info(
            "Iterable event-object split: train=%d events val=%d test=%d",
            len(self.train_set),
            len(self.valid_set),
            len(self.test_set),
        )

    def _build_file_specs(
        self,
        paths: list[str],
        data_domains: list[str] | None,
    ) -> list[_FileSpec]:
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
            specs.append(
                _FileSpec(
                    path=path,
                    n_events=int(n_events),
                    domain=_infer_domain(path, explicit_domain),
                )
            )
        return specs

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
