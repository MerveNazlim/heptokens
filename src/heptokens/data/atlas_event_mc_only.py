"""Read only MC features while retaining an original mixed-input event split."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Subset

from heptokens.data.atlas_event_mappable import AtlasEventMapDataset
from heptokens.data.atlas_mappable import BaseMapModule


def mc_split_indices(reference_paths, reference_counts, mc_paths, train_frac, val_frac, seed):
    """Filter the original global permutation into compact, MC-only indices."""
    if not (0 < train_frac < 1 and 0 < val_frac < 1 and train_frac + val_frac < 1):
        raise ValueError("Each original split fraction must be positive")
    if len(reference_paths) != len(reference_counts) or not reference_paths:
        raise ValueError("Original paths and event counts must align")
    if len(set(reference_paths)) != len(reference_paths) or len(set(mc_paths)) != len(mc_paths):
        raise ValueError("Duplicate input paths")
    if any(type(n) is not int or n < 0 for n in reference_counts):
        raise ValueError("Invalid original event counts")
    selected = set(mc_paths)
    if not selected or [p for p in reference_paths if p in selected] != list(mc_paths):
        raise ValueError("MC paths must be an ordered subsequence of the original inputs")
    counts = np.asarray(reference_counts, dtype=np.int64)
    ends = np.cumsum(counts)
    starts = ends - counts
    is_mc = np.asarray([p in selected for p in reference_paths], dtype=bool)
    mc_counts = counts * is_mc
    mc_starts = np.cumsum(mc_counts) - mc_counts
    total = int(ends[-1])
    train_size, val_size = int(total * train_frac), int(total * val_frac)
    permutation = torch.randperm(total, generator=torch.Generator().manual_seed(seed)).numpy()
    result = []
    for begin, end in (
        (0, train_size),
        (train_size, train_size + val_size),
        (train_size + val_size, total),
    ):
        chunks = []
        for offset in range(begin, end, 1_000_000):
            original = permutation[offset : min(offset + 1_000_000, end)]
            files = np.searchsorted(ends, original, side="right")
            keep = is_mc[files]
            chosen_files = files[keep]
            chunks.append(original[keep] - starts[chosen_files] + mc_starts[chosen_files])
        indices = np.concatenate(chunks) if chunks else np.empty(0, dtype=np.int64)
        if not len(indices):
            raise ValueError("Original partition contains no MC events")
        result.append(indices)
    return result


class AtlasEventMCOnlyMapModule(BaseMapModule):
    """Use the MC portion of each original train/validation/test partition.

    Collision-data event counts participate in split reconstruction, but their
    feature arrays are never loaded. Sampling is without replacement per epoch.
    """

    def __init__(
        self,
        *,
        data_paths,
        reference_data_paths,
        reference_event_counts,
        split_audit_path,
        preprocessing_audit_path=None,
        data_path=None,
        data_domains=None,
        sampling_domain_fractions=None,
        sampling_balance_by="valid_objects",
        sampling_num_samples=None,
        train_frac=0.7,
        val_frac=0.15,
        test_frac=0.15,
        seed=42,
        **kwargs,
    ):
        if (
            data_domains is not None
            or sampling_domain_fractions
            or sampling_num_samples is not None
        ):
            raise ValueError("This pilot uses the original MC membership without resampling")
        if abs(train_frac + val_frac + test_frac - 1) > 1e-6:
            raise ValueError("Split fractions must sum to one")
        if kwargs.get("output_mode") != "object" or kwargs.get("object_type") != "muons":
            raise ValueError("This pilot is muon-only")
        super().__init__(**kwargs)
        self.data_paths = list(data_paths)
        self.train_frac, self.val_frac, self.test_frac = train_frac, val_frac, test_frac
        self.seed = seed
        indices = mc_split_indices(
            list(reference_data_paths),
            list(reference_event_counts),
            self.data_paths,
            train_frac,
            val_frac,
            seed,
        )
        self.data_path = self.data_paths[0]
        counts_by_path = dict(zip(reference_data_paths, reference_event_counts))
        datasets = [AtlasEventMapDataset(path, **self.data_config) for path in self.data_paths]
        for path, dataset in zip(self.data_paths, datasets):
            if len(dataset) != counts_by_path[path]:
                raise ValueError(f"MC event count changed since preparation: {path}")
        full_dataset = ConcatDataset(datasets)
        self.train_set, self.valid_set, self.test_set = [Subset(full_dataset, i) for i in indices]
        if len(self.train_set) < self.batch_size:
            raise ValueError("MC training partition is smaller than one full batch")
        masks = [dataset.data_dict["mask"] for dataset in datasets]
        valid_per_event = np.concatenate([m.astype(bool).sum(axis=1) for m in masks])
        audit = {
            "mc_files": self.data_paths,
            "original_input_files": len(reference_data_paths),
            "original_events": sum(reference_event_counts),
            "mc_events": len(full_dataset),
            "seed": seed,
            "split_fractions": [train_frac, val_frac, test_frac],
            "partitions": {
                name: {
                    "events": len(rows),
                    "valid_objects": int(valid_per_event[rows].sum()),
                    "ordered_compact_indices_sha256": hashlib.sha256(rows.tobytes()).hexdigest(),
                }
                for name, rows in zip(("train", "val", "test"), indices)
            },
            "steps_per_epoch": len(self.train_set) // self.batch_size,
            "dropped_training_events_per_epoch": len(self.train_set) % self.batch_size,
            "note": "MC membership filtered from the original global event permutation. "
            "No MC-only resplit or replacement sampling. Historical membership assumes unchanged "
            "inputs/split implementation. See separate preprocessing-fit provenance.",
        }
        if preprocessing_audit_path is not None:
            fit = json.loads(Path(preprocessing_audit_path).read_text())
            if (
                fit["fit_domain"] != "mc"
                or fit["fit_partition"] != "train"
                or fit["h5_files"] != self.data_paths
                or fit["train_events"] != len(indices[0])
                or fit["n_objects_fit"] != audit["partitions"]["train"]["valid_objects"]
                or fit["ordered_compact_train_indices_sha256"]
                != audit["partitions"]["train"]["ordered_compact_indices_sha256"]
            ):
                raise ValueError("Preprocessing was not fitted on this MC training partition")
            audit["preprocessing_fit"] = str(Path(preprocessing_audit_path).resolve())
        path = Path(split_audit_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(audit, indent=2) + "\n")

    def setup(self, stage):
        """Membership and MC datasets are constructed in __init__."""
