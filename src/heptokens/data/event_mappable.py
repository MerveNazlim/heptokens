"""For loading events from event-level HDF files and creating a mappable dataset."""

import logging

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, random_split

from heptokens.data.atlas_mappable import BaseMapModule

log = logging.getLogger(__name__)


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
            # Remaining parts are field names inside a structured array
            field = "/".join(parts[i:])
            return node[field][:]
        node = node[part]
    if isinstance(node, h5py.Dataset):
        return node[:]
    raise ValueError(f"Path {h5_path!r} did not resolve to an HDF5 dataset")


class EventMapDataset(Dataset):
    """Loads event-level data from an HDF5 file using explicit path configuration.

    Parameters
    ----------
    file_path:
        Path to the HDF5 file.
    event_inputs:
        List of HDF5 paths for event-level scalar features, e.g.
        ``["common/event/mu"]``.  Each path must resolve to a 1-D array of
        length n_events.  The resulting array is stored under ``"jets"`` with
        shape ``[n_events, len(event_inputs)]``.
    object_collections:
        List of dicts, each with keys ``"object_name"`` and ``"inputs"``.
        ``"object_name"`` is the HDF5 group/dataset path (used only for logging).
        ``"inputs"`` is a list of HDF5 paths for object features; each must
        resolve to a 2-D array of shape ``[n_events, max_objects]``.  All
        collections are concatenated along the feature axis and stored under
        ``"csts"`` with shape ``[n_events, n_objects, n_features]``.
        The first input of the first collection is used to determine n_events
        and n_objects.
    mask_input:
        Optional HDF5 path for a boolean validity mask of shape
        ``[n_events, max_objects]``.  When omitted, all objects up to
        ``n_objects`` are treated as valid.
    num_events:
        If set, cap the number of events loaded.
    num_objects:
        If set, cap the number of objects per event loaded.
    """

    def __init__(
        self,
        file_path: str,
        event_inputs: list[str],
        object_collections: list[dict],
        mask_input: str | None = None,
        num_events: int | None = None,
        num_objects: int | None = None,
    ) -> None:
        super().__init__()

        self.data_dict = {}
        with h5py.File(file_path, mode="r") as handle:
            # Determine dimensions from the first object input
            first_input = object_collections[0]["inputs"][0]
            probe = _resolve_h5_path(handle, first_input)
            total_events, total_objects = probe.shape
            n_events = min(total_events, num_events) if num_events else total_events
            n_objects = min(total_objects, num_objects) if num_objects else total_objects

            # Build csts: [n_events, n_objects, n_features]
            feature_arrays = []
            for collection in object_collections:
                for inp in collection["inputs"]:
                    arr = _resolve_h5_path(handle, inp)[:n_events, :n_objects].astype(np.float32)
                    feature_arrays.append(arr)
            self.data_dict["csts"] = np.stack(feature_arrays, axis=-1)

            # Build event-level feature array: [n_events, n_event_features]
            event_arrays = []
            for inp in event_inputs:
                arr = _resolve_h5_path(handle, inp)[:n_events].astype(np.float32)
                event_arrays.append(arr)
            self.data_dict["jets"] = np.stack(event_arrays, axis=-1)

            # Validity mask: explicit path takes priority; fall back to all-valid.
            if mask_input is not None:
                self.data_dict["mask"] = _resolve_h5_path(handle, mask_input)[
                    :n_events, :n_objects
                ].astype(bool)
            else:
                self.data_dict["mask"] = np.ones((n_events, n_objects), dtype=bool)

            # Placeholder labels
            self.data_dict["labels"] = np.zeros(n_events, dtype=np.int64)

        self.num_events = self.data_dict["csts"].shape[0]
        self.num_objects = self.data_dict["csts"].shape[1]
        n_features = self.data_dict["csts"].shape[2]
        log.info(
            f"Loaded {self.num_events} events with up to {self.num_objects} objects "
            f"({n_features} features) from {file_path}"
        )

    def __len__(self) -> int:
        return self.num_events

    def __getitem__(self, idx: int) -> dict:
        return {k: v[idx] for k, v in self.data_dict.items()}


class EventSingleFileMapModule(BaseMapModule):
    """DataModule that loads an event-level HDF5 file and splits into train/val/test."""

    def __init__(
        self,
        *,
        data_path: str,
        train_frac: float = 0.7,
        val_frac: float = 0.15,
        test_frac: float = 0.15,
        seed: int = 42,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if not abs(train_frac + val_frac + test_frac - 1.0) < 1e-6:
            raise ValueError("train_frac + val_frac + test_frac must sum to 1.0")

        self.data_path = data_path
        self.train_frac = train_frac
        self.val_frac = val_frac
        self.test_frac = test_frac
        self.seed = seed

        full_dataset = EventMapDataset(self.data_path, **self.data_config)
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
