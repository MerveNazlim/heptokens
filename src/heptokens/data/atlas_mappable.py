"""For loading in jets from HDF files and creating a mappable dataset."""

import logging
from abc import ABC, abstractmethod
from functools import partial

import h5py
import numpy as np
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, Sampler, random_split

from heptokens.data.collation import collate_and_transform
from heptokens.utils.plot_physics import JET_FEATURES, TRACK_FEATURES, load_jets_table

log = logging.getLogger(__name__)


class MapDataset(Dataset):
    """Loads a collection of jets from HDF files and creates a mappable dataset."""

    def __init__(
        self,
        file_path: str,
        jet_features: list | None = None,
        cst_features: list | None = None,
        label_key: str = "HadronConeExclTruthLabelID",
        num_jets: int | None = None,
        num_csts: int | None = None,
        max_jet_pt: float | None = None,
        max_cst_pt: float | None = None,
    ) -> None:
        super().__init__()
        if jet_features is None:
            jet_features = JET_FEATURES
        if cst_features is None:
            cst_features = TRACK_FEATURES
        self.jet_features = jet_features
        self.cst_features = cst_features

        # Load data from HDF5 file
        self.data_dict = {}
        # Load jet-level data from the jets table
        jets_table = load_jets_table(file_path)
        self.data_dict["jets"] = jets_table[self.jet_features].to_numpy()[:num_jets]
        # Set the truth labels as targets
        labels = jets_table[label_key].to_numpy()[:num_jets]
        # Convert labels based on unique values to a contiguous range starting at 0
        # TODO: is this the correct thing to do?
        unique_labels = np.unique(labels)
        label_map = {label: i for i, label in enumerate(unique_labels)}
        self.data_dict["labels"] = np.array([label_map[label] for label in labels])
        # Load constituent-level data from the tracks table
        with h5py.File(file_path, mode="r") as handle:
            tracks_ds = handle["tracks"]
            # Load the slice once
            tracks_slice = tracks_ds[:num_jets, :num_csts]
            # Pre-allocate and fill constituent features array
            num_features = len(cst_features)
            self.data_dict["csts"] = np.empty(
                (tracks_slice.shape[0], tracks_slice.shape[1], num_features), dtype=np.float32
            )
            # Fill constituent features
            for i, key in enumerate(cst_features):
                self.data_dict["csts"][:, :, i] = tracks_slice[key]
            # Load validity mask
            self.data_dict["mask"] = tracks_slice["valid"]

        # Apply jet pT upper cut if requested
        if max_jet_pt is not None:
            pt_col = self.jet_features.index("pt")
            keep = self.data_dict["jets"][:, pt_col] <= max_jet_pt
            n_before = len(self.data_dict["jets"])
            for key in self.data_dict:
                self.data_dict[key] = self.data_dict[key][keep]
            n_after = len(self.data_dict["jets"])
            log.info(f"Removed {n_before - n_after} jets with pt > {max_jet_pt:.0f} MeV")

        # Remove jets with any constituent pt above threshold
        if max_cst_pt is not None:
            pt_col = self.cst_features.index("pt")
            cst_pts = self.data_dict["csts"][:, :, pt_col]
            max_per_jet = np.where(self.data_dict["mask"], cst_pts, 0).max(axis=1)
            keep = max_per_jet <= max_cst_pt
            n_before = len(self.data_dict["jets"])
            for key in self.data_dict:
                self.data_dict[key] = self.data_dict[key][keep]
            n_after = len(self.data_dict["jets"])
            log.info(f"Removed {n_before - n_after} jets with constituent pt > {max_cst_pt:.0f} MeV")

        self.num_jets = self._get_num_jets(num_jets)
        self.num_csts = self._get_num_csts(num_csts)
        log.info(f"Loaded {self.num_jets} jets with {self.num_csts} constituents from {file_path}")

    def __len__(self) -> int:
        return self.num_jets

    def __getitem__(self, idx: int) -> tuple:
        return {k: v[idx] for k, v in self.data_dict.items()}

    def _get_num_jets(self, num_jets: int | None) -> int:
        file_len = self.data_dict["jets"].shape[0]
        if num_jets is None:
            return file_len
        return min(file_len, num_jets)

    def _get_num_csts(self, num_csts: int | None) -> int:
        file_len = self.data_dict["csts"].shape[1]
        if num_csts is None:
            return file_len
        return min(file_len, num_csts)


class BaseMapModule(LightningDataModule, ABC):
    """Base class for MapDataModules with shared dataloader logic."""

    def __init__(
        self,
        *,
        n_classes: int,
        num_workers: int = 6,
        batch_size: int = 1000,
        pin_memory: bool = True,
        persistent_workers: bool | None = None,
        multiprocessing_context: str | None = None,
        transforms: dict | None = None,
        **data_config,
    ) -> None:
        super().__init__()
        self.n_classes = n_classes
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.persistent_workers = (
            num_workers > 0 if persistent_workers is None else persistent_workers
        )
        self.multiprocessing_context = multiprocessing_context
        self.transforms = transforms
        self.data_config = data_config

        # These will be set by subclasses
        self.train_set: Dataset
        self.valid_set: Dataset
        self.test_set: Dataset

    @abstractmethod
    def setup(self, stage: str) -> None:
        """Subclasses must implement dataset setup."""
        pass

    def _get_dataloader(
        self,
        dataset: Dataset,
        shuffle: bool,
        drop_last: bool,
        sampler: Sampler | None = None,
    ) -> DataLoader:
        """Internal helper to create dataloaders."""
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
            shuffle=shuffle if sampler is None else False,
            sampler=sampler,
            drop_last=drop_last,
            collate_fn=collate_fn,
            **dataloader_kwargs,
        )

    def train_dataloader(self) -> DataLoader:
        return self._get_dataloader(
            self.train_set,
            shuffle=True,
            drop_last=True,
            sampler=getattr(self, "train_sampler", None),
        )

    def val_dataloader(self) -> DataLoader:
        return self._get_dataloader(self.valid_set, shuffle=False, drop_last=False)

    def test_dataloader(self) -> DataLoader:
        return self._get_dataloader(self.test_set, shuffle=False, drop_last=False)

    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()

    def get_data_sample(self) -> tuple:
        """Get a data sample to initialise the network with the right dimensions."""
        return next(iter(self.valid_set))

    def get_n_classes(self) -> int:
        """Get the number of classes in the dataset."""
        return self.n_classes


class PresplitMapModule(BaseMapModule):
    """DataModule for pre-split train/val/test datasets."""

    def __init__(
        self,
        *,
        train_path: str,
        val_path: str,
        test_path: str,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path

        # Initialize validation set early to get a sample
        self.valid_set = MapDataset(self.val_path, **self.data_config)

    def setup(self, stage: str) -> None:
        """Sets up the relevant datasets."""
        if stage in {"fit", "train"}:
            self.train_set = MapDataset(self.train_path, **self.data_config)
        if stage in {"predict", "test"}:
            self.test_set = MapDataset(self.test_path, **self.data_config)


class SingleFileMapModule(BaseMapModule):
    """DataModule that splits a single dataset file into train/val/test."""

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

        # Load full dataset and split it
        full_dataset = MapDataset(self.data_path, **self.data_config)
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
        pass
