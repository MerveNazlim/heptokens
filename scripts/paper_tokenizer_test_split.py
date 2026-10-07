"""Reconstruct saved global test membership before selecting evaluation files.

This is model-test evaluation, NOT preprocessing-disjoint file holdout. It
assumes the saved input files and split implementation are unchanged since
training; no historical event-membership manifest is supplied to this workflow.
"""

from __future__ import annotations

import hashlib
import logging
from functools import partial
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import IterableDataset

import paper_tokenizer_capacity as capacity

log = logging.getLogger(__name__)
TARGET = "heptokens.data.atlas_event_mappable.AtlasEventMapModule"
CAVEAT = (
    "Model-test events reconstructed from the saved ordered full input list, "
    "event counts, seed and global random_split fractions. Preprocessing was fitted "
    "before the event split; these events are NOT guaranteed held out from preprocessing. "
    "Reconstruction assumes unchanged source files and the training split implementation; "
    "no historical membership manifest was checked. This is not file-disjoint holdout or "
    "an event-ID deduplication audit. Evaluation streams the frozen selected files in "
    "file-list order and ascending event row, not the eager loader's permutation order."
)


def global_split_ids(total, train_frac, val_frac, seed):
    """Match AtlasEventMapModule's CPU random_split without materializing rows."""
    train_size = int(total * train_frac)
    val_size = int(total * val_frac)
    permutation = torch.randperm(total, generator=torch.Generator().manual_seed(seed)).numpy()
    split_ids = np.full(total, 2, dtype=np.uint8)
    split_ids[permutation[:train_size]] = 0
    split_ids[permutation[train_size : train_size + val_size]] = 1
    return split_ids


def read_h5_slice(handle, path, event_slice, n_objects=None):
    """Slice an H5 dataset or compound field before reading it into memory."""
    import h5py

    node = handle
    field = None
    parts = path.strip("/").split("/")
    for index, part in enumerate(parts):
        if isinstance(node, h5py.Dataset):
            field = "/".join(parts[index:])
            break
        node = node[part]
    if not isinstance(node, h5py.Dataset):
        raise ValueError(f"Path {path!r} did not resolve to an HDF5 dataset")
    dataset = node.fields(field) if field is not None else node
    if n_objects is None or len(node.shape) < 2:
        return dataset[event_slice]
    return dataset[event_slice, :n_objects]


class SavedTestEventDataset(IterableDataset):
    """Read frozen test rows in bounded chunks, using the eager loader's conventions.

    Kept with the plotting script so deployment does not require replacing the
    training streaming module or relying on its private, version-specific API.
    """

    def __init__(self, entries, dm, chunk_size=4096):
        super().__init__()
        if dm.get("output_mode", "object") != "object":
            raise ValueError("Saved-test plotting requires output_mode='object'")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        self.entries = entries
        self.dm = dm
        self.chunk_size = chunk_size

    def __len__(self):
        return sum(len(entry["indices"]) for entry in self.entries)

    def __iter__(self):
        import h5py
        from heptokens.data.atlas_event_mappable import AtlasEventMapDataset

        if torch.utils.data.get_worker_info() is not None:
            raise ValueError("Frozen saved-test evaluation requires num_workers=0")
        dm = self.dm
        collection = AtlasEventMapDataset._find_collection(
            dm.get("object_collections") or [], dm["object_type"]
        )
        inputs = collection.get("inputs") or []
        if not inputs:
            raise ValueError(f"Object collection {dm['object_type']!r} has no inputs")
        limit = (dm.get("max_objects") or {}).get(dm["object_type"], dm.get("num_objects"))
        mask_path = collection.get("mask_input", dm.get("mask_input"))
        for entry in self.entries:
            with h5py.File(entry["path"], "r") as handle:
                for start in range(0, entry["n_events"], self.chunk_size):
                    end = min(start + self.chunk_size, entry["n_events"])
                    left, right = np.searchsorted(entry["indices"], [start, end])
                    selected = entry["indices"][left:right] - start
                    if len(selected) == 0:
                        continue
                    event_slice = slice(start, end)
                    features, n_objects = [], None
                    for path in inputs:
                        array = read_h5_slice(handle, path, event_slice, n_objects or limit or None)
                        array = array.astype(np.float32)
                        if array.ndim == 1:
                            array = array[:, None]
                        if n_objects is None:
                            n_objects = min(array.shape[1], limit) if limit else array.shape[1]
                        features.append(array[:, :n_objects])
                    csts = np.stack(features, axis=-1)
                    csts = np.clip(np.nan_to_num(csts, nan=0.0, posinf=0.0, neginf=0.0), -1e4, 1e4)
                    if mask_path is None:
                        mask = np.ones(csts.shape[:2], dtype=bool)
                    else:
                        mask = read_h5_slice(handle, mask_path, event_slice, n_objects)
                        if mask.ndim == 1:
                            mask = mask[:, None]
                        mask = mask[:, :n_objects].astype(bool)
                    event_arrays = [
                        read_h5_slice(handle, path, event_slice)
                        .astype(np.float32)
                        .reshape(end - start, -1)
                        for path in dm.get("event_inputs") or []
                    ]
                    jets = (
                        np.concatenate(event_arrays, axis=-1)
                        if event_arrays
                        else mask.sum(axis=1).astype(np.float32).reshape(-1, 1)
                    )
                    label_key = dm.get("label_key")
                    if label_key is not None and label_key in handle:
                        labels = handle[label_key][event_slice].astype(np.int64)
                    elif (
                        label_key is not None
                        and "events" in handle
                        and handle["events"].dtype.names
                        and label_key in handle["events"].dtype.names
                    ):
                        labels = handle["events"].fields(label_key)[event_slice].astype(np.int64)
                    else:
                        labels = np.zeros(end - start, dtype=np.int64)
                    for row in selected:
                        yield {
                            "csts": csts[row],
                            "mask": mask[row],
                            "jets": jets[row],
                            "labels": labels[row],
                        }


def saved_settings(run):
    capacity.verify_file(run["config"])
    cfg = OmegaConf.load(run["config"]["path"])
    dm = OmegaConf.to_container(cfg.datamodule, resolve=True)
    if dm.get("_target_") != TARGET:
        raise ValueError(f"Saved-test reconstruction only supports {TARGET}: {run['object']}")
    paths = dm.get("data_paths") or [dm.get("data_path")]
    capacity.h5_names(paths)
    paths = [str((Path(run["run_dir"]) / p).resolve()) for p in paths]
    if len(set(paths)) != len(paths):
        raise ValueError(f"Repeated resolved H5 paths in full inputs: {run['object']}")
    fractions = [dm.get(key) for key in ("train_frac", "val_frac", "test_frac")]
    if any(v is None or not 0 <= v <= 1 for v in fractions) or abs(sum(fractions) - 1) >= 1e-6:
        raise ValueError(f"Missing/invalid saved split fractions: {run['object']}")
    if fractions[2] <= 0 or dm.get("seed") is None:
        raise ValueError(f"No saved test split/seed for {run['object']}")
    return dm, {
        "ordered_paths": paths,
        "seed": int(dm["seed"]),
        "train_frac": fractions[0],
        "val_frac": fractions[1],
        "test_frac": fractions[2],
        "num_events": dm.get("num_events"),
    }


def freeze_partition(runs, samples, output_dir):
    import h5py
    import heptokens.data.atlas_event_mappable as eager
    import heptokens.data.collation as collation

    configs = []
    reference = None
    for run in runs:
        dm, settings = saved_settings(run)
        if reference is not None and settings != reference:
            changed = [k for k in settings if settings[k] != reference[k]]
            raise ValueError(f"Saved global split differs for {run['object']}: {changed}")
        configs.append(dm)
        reference = settings
    wanted = {}
    for sample, records in samples.items():
        if not records:
            raise ValueError(f"No selected {sample} files")
        for record in records:
            path = record["path"]
            if path in wanted:
                raise ValueError(f"Duplicate/overlapping selected file: {path}")
            wanted[path] = sample
    absent = set(wanted) - set(reference["ordered_paths"])
    if absent:
        raise ValueError(f"Selected files are not in the saved full input list: {sorted(absent)}")
    counts, inputs = [], []
    for i, path in enumerate(reference["ordered_paths"]):
        record = capacity.file_record(path)
        if path in wanted:
            selected_record = next(r for r in samples[wanted[path]] if r["path"] == path)
            if selected_record != record:
                raise ValueError(f"Selected file changed since sample audit: {path}")
        with h5py.File(path, "r") as handle:
            per_run = [
                int(
                    eager.AtlasEventMapDataset._infer_dimensions(
                        handle,
                        dm.get("event_inputs") or [],
                        dm.get("object_collections") or [],
                        dm.get("num_events"),
                    )[1]
                )
                for dm in configs
            ]
        if len(set(per_run)) != 1 or per_run[0] < 0:
            raise ValueError(
                f"Invalid/inconsistent event counts across full runs: {path}: {per_run}"
            )
        counts.append((path, per_run[0], wanted.get(path, "not_selected")))
        inputs.append({"file": record, "n_events": per_run[0]})
        if (i + 1) % 100 == 0:
            log.info(
                "Read event-count metadata: %d/%d full inputs",
                i + 1,
                len(reference["ordered_paths"]),
            )
    total = sum(n for _, n, _ in counts)
    log.info(
        "Reconstructing global membership for %s events over %d files, seed=%d; no inference",
        f"{total:,}",
        len(counts),
        reference["seed"],
    )
    # This is the same seeded CPU randperm and integer split rounding as the
    # eager training loader. Restrict files ONLY after the global assignment.
    split_ids = global_split_ids(
        total, reference["train_frac"], reference["val_frac"], reference["seed"]
    )
    split_digest = hashlib.sha256(split_ids).hexdigest()
    by_path = {}
    offset = 0
    for path, count, _ in counts:
        if path in wanted:
            by_path[path] = np.flatnonzero(split_ids[offset : offset + count] == 2)
        offset += count
    del split_ids
    selected, arrays = [], {}
    counts_by_path = {p: n for p, n, _ in counts}
    for sample, records in samples.items():
        for record in records:
            key = f"file_{len(selected):04d}"
            path = record["path"]
            indices = by_path[path]
            arrays[key] = indices
            selected.append(
                {
                    "sample": sample,
                    "path": path,
                    "key": key,
                    "n_events": counts_by_path[path],
                    "test_events": len(indices),
                }
            )
        n = sum(s["test_events"] for s in selected if s["sample"] == sample)
        if n == 0:
            raise ValueError(f"No original test events in selected {sample} inputs")
        log.info(
            "%s: %s original test events in %d frozen files (before object cap)",
            sample,
            f"{n:,}",
            len(records),
        )
    # Recheck all source identities before publishing the membership artifact.
    for item in inputs:
        capacity.verify_file(item["file"])
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "test_membership.npz"
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)
    train_size = int(total * reference["train_frac"])
    val_size = int(total * reference["val_frac"])
    return {
        "settings": reference,
        "inputs": inputs,
        "selected": selected,
        "membership": capacity.file_record(path, content_hash=True),
        "global_split_sha256": split_digest,
        "counts": {
            "total": total,
            "train": train_size,
            "val": val_size,
            "test": total - train_size - val_size,
        },
        "torch_version": str(torch.__version__),
        "implementation": [
            capacity.file_record(path, content_hash=True)
            for path in (__file__, eager.__file__, collation.__file__)
        ],
        "historical_membership_verified": False,
        "preprocessing_disjoint": False,
    }


def verify_partition(plan):
    partition = plan["test_partition"]
    for record in partition["implementation"]:
        capacity.verify_file(record)
    for item in partition["inputs"]:
        capacity.verify_file(item["file"])
    capacity.verify_file(partition["membership"])


def test_loader(cfg, plan, sample, transforms):
    from heptokens.data.collation import collate_and_transform

    partition = plan["test_partition"]
    capacity.verify_file(partition["membership"])
    entries = []
    with np.load(partition["membership"]["path"], allow_pickle=False) as archive:
        for item in partition["selected"]:
            if item["sample"] != sample:
                continue
            indices = archive[item["key"]]
            if (
                indices.ndim != 1
                or len(indices) != item["test_events"]
                or indices.dtype.kind not in "iu"
                or np.any(indices < 0)
                or np.any(indices >= item["n_events"])
                or np.any(np.diff(indices) <= 0)
            ):
                raise ValueError(f"Invalid saved test indices for {item['path']}")
            entries.append({**item, "indices": indices})
    if [item["path"] for item in entries] != [r["path"] for r in plan["samples"][sample]]:
        raise ValueError("Saved test membership does not match the frozen sample file order")
    dm = OmegaConf.to_container(cfg.datamodule, resolve=True)
    dataset = SavedTestEventDataset(entries, dm)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=plan["evaluation"]["batch_size"],
        num_workers=0,
        collate_fn=partial(collate_and_transform, transforms=transforms),
        drop_last=False,
    )
