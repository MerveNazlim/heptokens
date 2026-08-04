"""ATLAS-style event HDF5 mappable datasets for tokenizer training.

The default configs use ATLAS event HDF5 layout conventions, but feature paths
are provided by YAML config. A config can therefore choose ``common/...``,
``atlas/...``, or another namespace without changing this module.
"""

from collections.abc import Mapping
import logging

import h5py
import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset, Sampler, random_split

from heptokens.data.atlas_mappable import BaseMapModule

log = logging.getLogger(__name__)


class DomainBalancedSampler(Sampler[int]):
    """Sample a domain first and then an event within that domain.

    ``torch.multinomial`` cannot sample from more than 2^24 categories, which
    makes ``WeightedRandomSampler`` unsuitable for very large event datasets.
    This sampler only draws from the small set of domains, then uniformly draws
    an event index from the selected domain.
    """

    def __init__(
        self,
        domain_indices: list[np.ndarray],
        domain_probabilities: list[float],
        num_samples: int,
        seed: int,
        chunk_size: int = 65536,
    ) -> None:
        if not domain_indices or len(domain_indices) != len(domain_probabilities):
            raise ValueError("domain_indices and domain_probabilities must align")
        if any(len(indices) == 0 for indices in domain_indices):
            raise ValueError("Every sampled domain must contain at least one event")

        probabilities = np.asarray(domain_probabilities, dtype=np.float64)
        if np.any(probabilities < 0) or probabilities.sum() <= 0:
            raise ValueError("Domain probabilities must be non-negative and non-zero")

        self.domain_indices = [
            np.asarray(indices, dtype=np.int64) for indices in domain_indices
        ]
        self.domain_probabilities = probabilities / probabilities.sum()
        self.num_samples = int(num_samples)
        self.chunk_size = int(chunk_size)
        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        remaining = self.num_samples
        while remaining > 0:
            size = min(self.chunk_size, remaining)
            selected_domains = self.rng.choice(
                len(self.domain_indices),
                size=size,
                replace=True,
                p=self.domain_probabilities,
            )
            sampled_indices = np.empty(size, dtype=np.int64)
            for domain_idx, indices in enumerate(self.domain_indices):
                positions = np.flatnonzero(selected_domains == domain_idx)
                if len(positions) == 0:
                    continue
                draws = self.rng.integers(0, len(indices), size=len(positions))
                sampled_indices[positions] = indices[draws]
            yield from sampled_indices.tolist()
            remaining -= size

    def __len__(self) -> int:
        return self.num_samples


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
    """Read a dataset or structured-array field via a slash-separated path."""
    # Follow the path from the config and read the dataset or field.
    parts = h5_path.strip("/").split("/")
    node = handle
    for index, part in enumerate(parts):
        if isinstance(node, h5py.Dataset):
            field = "/".join(parts[index:])
            return node[field][:]
        node = node[part]
    if isinstance(node, h5py.Dataset):
        return node[:]
    raise ValueError(f"Path {h5_path!r} did not resolve to an HDF5 dataset")


def _resolve_h5_node(handle: h5py.File, h5_path: str) -> h5py.Dataset:
    """Resolve an HDF5 path to the backing dataset without reading it."""
    # Used when we only need the shape, not the full array.
    parts = h5_path.strip("/").split("/")
    node = handle
    for part in parts:
        if isinstance(node, h5py.Dataset):
            return node
        node = node[part]
    if isinstance(node, h5py.Dataset):
        return node
    raise ValueError(f"Path {h5_path!r} did not resolve to an HDF5 dataset")


class AtlasEventMapDataset(Dataset):
    """Load ATLAS event HDF5 objects using explicit YAML path configuration.

    ``output_mode="object"`` returns one configured object collection under
    ``csts`` and ``mask``. This is the mode used to train one tokenizer per
    object type.

    ``output_mode="combined"`` concatenates configured object collections under
    ``csts`` and ``mask``. All collections must use the same number of inputs.
    """

    @staticmethod
    def _infer_dimensions(
        handle: h5py.File,
        event_inputs: list[str],
        object_collections: list[dict],
        num_events: int | None,
    ) -> tuple[int, int]:
        # First try metadata. If it is missing, use the first configured input.
        if "metadata" in handle and "n_events" in handle["metadata"].attrs:
            total_events = int(handle["metadata"].attrs["n_events"])
        elif "n_events" in handle.attrs:
            total_events = int(handle.attrs["n_events"])
        elif object_collections:
            dataset = _resolve_h5_node(handle, object_collections[0]["inputs"][0])
            total_events = int(dataset.shape[0])
        elif event_inputs:
            total_events = int(_resolve_h5_node(handle, event_inputs[0]).shape[0])
        else:
            raise ValueError("At least one event input or object collection is required")

        n_events = min(total_events, num_events) if num_events else total_events
        return total_events, n_events

    @staticmethod
    def _slice_events(
        array: np.ndarray,
        n_events: int,
        n_objects: int | None = None,
    ) -> np.ndarray:
        if array.ndim == 1 or n_objects is None:
            return array[:n_events]
        return array[:n_events, :n_objects]

    @staticmethod
    def _find_collection(object_collections: list[dict], object_type: str) -> dict:
        for collection in object_collections:
            if collection.get("object_name") == object_type:
                return collection
        available = ", ".join(
            str(collection.get("object_name")) for collection in object_collections
        )
        raise ValueError(
            f"Object collection {object_type!r} is not configured. Available: {available}"
        )

    @staticmethod
    def _read_event_inputs(
        handle: h5py.File,
        event_inputs: list[str],
        n_events: int,
        fallback: np.ndarray,
    ) -> np.ndarray:
        if not event_inputs:
            return fallback

        # Load event-level inputs, for example pileup or primary vertex.
        event_arrays = []
        for h5_path in event_inputs:
            array = _resolve_h5_path(handle, h5_path)[:n_events].astype(np.float32)
            if array.ndim > 1:
                array = array.reshape(n_events, -1)
            else:
                array = array[:, None]
            event_arrays.append(array)
        return np.concatenate(event_arrays, axis=-1)

    @staticmethod
    def _read_mask(
        handle: h5py.File,
        mask_input: str | None,
        n_events: int,
        n_objects: int,
    ) -> np.ndarray:
        # Load the object mask. If there is no mask, keep all object slots.
        if mask_input is None:
            return np.ones((n_events, n_objects), dtype=bool)

        mask = _resolve_h5_path(handle, mask_input)
        if mask.ndim == 1:
            mask = mask[:, None]
        return mask[:n_events, :n_objects].astype(bool)

    @staticmethod
    def _read_collection(
        handle: h5py.File,
        collection: dict,
        n_events: int,
        num_objects: int | None,
        default_mask_input: str | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        # Load one object collection, for example jets or electrons.
        inputs = _as_list(collection.get("inputs"))
        if not inputs:
            raise ValueError(f"Object collection {collection.get('object_name')!r} has no inputs")

        # Load each feature and stack them as (events, objects, features).
        feature_arrays = []
        n_objects = None
        for h5_path in inputs:
            array = _resolve_h5_path(handle, h5_path).astype(np.float32)
            if array.ndim == 1:
                array = array[:, None]
            if n_objects is None:
                total_objects = array.shape[1]
                n_objects = min(total_objects, num_objects) if num_objects else total_objects
            feature_arrays.append(AtlasEventMapDataset._slice_events(array, n_events, n_objects))

        # Replace bad values before sending features to the tokenizer.
        csts = np.stack(feature_arrays, axis=-1)
        csts = np.nan_to_num(csts, nan=0.0, posinf=0.0, neginf=0.0)
        csts = np.clip(csts, -1e4, 1e4)

        mask_input = collection.get("mask_input", default_mask_input)
        mask = AtlasEventMapDataset._read_mask(handle, mask_input, n_events, csts.shape[1])
        return csts, mask

    @staticmethod
    def _load_labels(
        handle: h5py.File,
        label_key: str | None,
        n_events: int,
    ) -> np.ndarray:
        # Tokenizer training does not use labels, so use zeros by default.
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

    def __init__(
        self,
        file_path: str,
        event_inputs: list[str] | None = None,
        object_collections: list[dict] | None = None,
        output_mode: str = "object",
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
            # Decide how many events to read from this file.
            _, n_events = self._infer_dimensions(
                handle,
                event_inputs,
                object_collections,
                num_events,
            )

            if output_mode == "object":
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
            elif output_mode == "combined":
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
            else:
                raise ValueError("output_mode must be 'object' or 'combined'")

            self.data_dict["labels"] = self._load_labels(handle, label_key, n_events)

        self.num_events = len(self.data_dict["labels"])
        log.info("Loaded %s events in %s mode from %s", self.num_events, output_mode, file_path)

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
    ) -> dict:
        # Load only one object type. This is used for per-object tokenizers.
        collection = AtlasEventMapDataset._find_collection(object_collections, object_type)
        csts, mask = AtlasEventMapDataset._read_collection(
            handle,
            collection,
            n_events,
            max_objects.get(object_type, num_objects),
            mask_input,
        )
        fallback_jets = mask.sum(axis=1).astype(np.float32).reshape(-1, 1)
        return {
            "csts": csts,
            "mask": mask,
            # Existing VQ-VAE code expects event inputs under "jets".
            # The object features are returned under "csts".
            "jets": AtlasEventMapDataset._read_event_inputs(
                handle,
                event_inputs,
                n_events,
                fallback_jets,
            ),
        }

    @staticmethod
    def _load_combined(
        handle: h5py.File,
        event_inputs: list[str],
        object_collections: list[dict],
        mask_input: str | None,
        n_events: int,
        num_objects: int | None,
        max_objects: dict,
    ) -> dict:
        # Join several object types into one object axis.
        cst_parts = []
        mask_parts = []
        type_parts = []
        n_features = None

        for type_id, collection in enumerate(object_collections, start=1):
            name = collection.get("object_name", f"object_{type_id}")
            csts, mask = AtlasEventMapDataset._read_collection(
                handle,
                collection,
                n_events,
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
            "jets": AtlasEventMapDataset._read_event_inputs(
                handle,
                event_inputs,
                n_events,
                fallback_jets,
            ),
        }

    def __len__(self) -> int:
        return self.num_events

    def __getitem__(self, idx: int) -> dict:
        return {key: value[idx] for key, value in self.data_dict.items()}


class AtlasEventMapModule(BaseMapModule):
    """DataModule that loads one or more ATLAS event HDF5 files and splits them."""

    def __init__(
        self,
        *,
        data_path: str | None = None,
        data_paths: list[str] | None = None,
        data_domains: list[str] | None = None,
        sampling_domain_fractions: dict[str, float] | None = None,
        sampling_balance_by: str = "valid_objects",
        sampling_num_samples: int | None = None,
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
        if data_domains is not None and len(data_domains) != len(paths):
            raise ValueError(
                "data_domains must contain one domain label per data path: "
                f"got {len(data_domains)} labels for {len(paths)} paths"
            )
        self.data_domains = list(data_domains) if data_domains is not None else None
        self.sampling_domain_fractions = (
            dict(sampling_domain_fractions) if sampling_domain_fractions else None
        )
        self.sampling_balance_by = sampling_balance_by
        self.sampling_num_samples = sampling_num_samples
        self.train_frac = train_frac
        self.val_frac = val_frac
        self.test_frac = test_frac
        self.seed = seed

        datasets = [AtlasEventMapDataset(path, **dataset_config) for path in paths]
        full_dataset = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]
        total_size = len(full_dataset)
        train_size = int(total_size * train_frac)
        val_size = int(total_size * val_frac)
        test_size = total_size - train_size - val_size

        generator = torch.Generator().manual_seed(seed)
        self.train_set, self.valid_set, self.test_set = random_split(
            full_dataset, [train_size, val_size, test_size], generator=generator
        )
        self.train_sampler = self._build_train_sampler(datasets)

    def _build_train_sampler(
        self,
        datasets: list[AtlasEventMapDataset],
    ) -> DomainBalancedSampler | None:
        """Build a sampler targeting domain fractions in valid-object loss input.

        The default behavior remains unchanged. When fractions are configured,
        events are sampled with replacement. For ``valid_objects`` balancing,
        domain event probabilities are corrected for each domain's mean number
        of valid objects per event, so the expected valid-object contribution
        follows the requested fractions.
        """
        targets = self.sampling_domain_fractions
        if self.data_domains is None:
            if targets is None:
                return None
            raise ValueError(
                "sampling_domain_fractions requires data_domains to identify each input file"
            )

        event_domains = np.concatenate(
            [
                np.full(len(dataset), domain, dtype=object)
                for dataset, domain in zip(datasets, self.data_domains)
            ]
        )
        valid_objects = np.concatenate(
            [
                np.asarray(dataset.data_dict["mask"], dtype=bool).sum(axis=1)
                for dataset in datasets
            ]
        ).astype(np.float64)
        train_indices = np.asarray(self.train_set.indices, dtype=np.int64)
        train_domains = event_domains[train_indices]
        train_valid_objects = valid_objects[train_indices]

        available = set(str(domain) for domain in np.unique(train_domains))
        total_events = len(train_domains)
        total_valid_objects = float(train_valid_objects.sum())
        for domain in sorted(available):
            domain_mask = train_domains == domain
            domain_events = int(domain_mask.sum())
            domain_valid_objects = float(train_valid_objects[domain_mask].sum())
            log.info(
                "Natural training composition %s: events=%d (%0.3f), "
                "valid_objects=%0.0f (%0.3f)",
                domain,
                domain_events,
                domain_events / total_events,
                domain_valid_objects,
                domain_valid_objects / total_valid_objects if total_valid_objects > 0 else 0.0,
            )

        if targets is None:
            return None
        if self.sampling_balance_by not in {"events", "valid_objects"}:
            raise ValueError("sampling_balance_by must be 'events' or 'valid_objects'")

        targets = {str(name): float(value) for name, value in targets.items()}
        if any(value < 0 for value in targets.values()) or sum(targets.values()) <= 0:
            raise ValueError("sampling_domain_fractions must contain non-negative weights")
        target_total = sum(targets.values())
        targets = {name: value / target_total for name, value in targets.items()}

        missing = sorted(set(targets) - available)
        if missing:
            raise ValueError(f"Requested sampling domains are absent from training data: {missing}")

        raw_event_probabilities: dict[str, float] = {}
        domain_stats: dict[str, tuple[int, float]] = {}
        for domain, target in targets.items():
            domain_mask = train_domains == domain
            n_events = int(domain_mask.sum())
            mean_valid = float(train_valid_objects[domain_mask].mean())
            domain_stats[domain] = (n_events, mean_valid)
            if self.sampling_balance_by == "valid_objects":
                if mean_valid <= 0:
                    raise ValueError(
                        f"Domain {domain!r} has no valid objects and cannot be object-balanced"
                    )
                raw_event_probabilities[domain] = target / mean_valid
            else:
                raw_event_probabilities[domain] = target

        probability_total = sum(raw_event_probabilities.values())
        event_probabilities = {
            domain: value / probability_total
            for domain, value in raw_event_probabilities.items()
        }

        sampled_domain_indices = []
        sampled_domain_probabilities = []
        for domain, event_probability in event_probabilities.items():
            domain_mask = train_domains == domain
            n_events, mean_valid = domain_stats[domain]
            sampled_domain_indices.append(np.flatnonzero(domain_mask))
            sampled_domain_probabilities.append(event_probability)
            log.info(
                "Sampling domain %s: target=%0.3f, train_events=%d, "
                "mean_valid_objects=%0.4f, event_probability=%0.3f",
                domain,
                targets[domain],
                n_events,
                mean_valid,
                event_probability,
            )

        unconfigured = sorted(available - set(targets))
        if unconfigured:
            log.warning("Training domains assigned zero sampling weight: %s", unconfigured)

        num_samples = self.sampling_num_samples or len(self.train_set)
        log.info(
            "Enabled domain-aware training sampler: balance_by=%s, num_samples=%d",
            self.sampling_balance_by,
            num_samples,
        )
        return DomainBalancedSampler(
            domain_indices=sampled_domain_indices,
            domain_probabilities=sampled_domain_probabilities,
            num_samples=int(num_samples),
            seed=self.seed + 1,
        )

    def setup(self, stage: str) -> None:
        """Datasets are already split in ``__init__``."""
