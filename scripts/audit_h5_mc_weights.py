"""Audit H5 inputs for event weights and compare with AtlasOpenMagic metadata."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import h5py
import numpy as np


WEIGHT_PATTERN = re.compile(
    r"weight|pileup|pile_up|puweight|prw|sumofweights|sum_of_weights",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file-lists",
        nargs="+",
        type=Path,
        required=True,
        help="Text files containing one H5 path per line.",
    )
    parser.add_argument(
        "--files-per-list",
        type=int,
        default=2,
        help="Representative H5 files to inspect from each list.",
    )
    parser.add_argument("--sample-values", type=int, default=5)
    parser.add_argument("--atlasopenmagic-release", default="2024r-pp")
    return parser.parse_args()


def scalar(value):
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    if isinstance(value, np.generic):
        return value.item()
    return value


def read_paths(path: Path, limit: int) -> list[Path]:
    paths = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        paths.append(Path(line))
        if len(paths) >= limit:
            break
    return paths


def find_dsid(handle: h5py.File) -> int:
    if "metadata" in handle:
        value = scalar(handle["metadata"].attrs.get("dsid", 0))
        try:
            if int(value) > 0:
                return int(value)
        except (TypeError, ValueError):
            pass

    path = "atlas/event/mcChannelNumber"
    if path in handle:
        values = np.asarray(handle[path][:])
        unique = np.unique(values[values > 0])
        if len(unique) == 1:
            return int(unique[0])
    return 0


def sample_dataset(dataset: h5py.Dataset, count: int) -> str:
    if dataset.size == 0:
        return "[]"
    try:
        if dataset.ndim == 0:
            values = np.asarray([dataset[()]])
        else:
            selection = (slice(0, min(count, dataset.shape[0])),) + (
                (slice(None),) * (dataset.ndim - 1)
            )
            values = np.asarray(dataset[selection])
        return np.array2string(values, threshold=20, edgeitems=3)
    except Exception as exc:  # Some compound or variable-length datasets reject slicing.
        return f"<could not sample: {exc}>"


def inspect_file(path: Path, sample_values: int) -> int:
    print(f"\nH5 file: {path}")
    if not path.exists():
        print("  ERROR: file does not exist")
        return 0

    with h5py.File(path, "r") as handle:
        dsid = find_dsid(handle)
        print(f"  DSID: {dsid}")
        matches = []

        def visitor(name, item):
            for attr_name, attr_value in item.attrs.items():
                full_name = f"{name or '/'}@{attr_name}"
                if WEIGHT_PATTERN.search(full_name):
                    matches.append(("attribute", full_name, None, None, scalar(attr_value)))
            if isinstance(item, h5py.Dataset) and WEIGHT_PATTERN.search(name):
                matches.append(
                    (
                        "dataset",
                        name,
                        item.shape,
                        str(item.dtype),
                        sample_dataset(item, sample_values),
                    )
                )

        handle.visititems(visitor)
        for attr_name, attr_value in handle.attrs.items():
            if WEIGHT_PATTERN.search(attr_name):
                matches.append(("attribute", f"/@{attr_name}", None, None, scalar(attr_value)))

        if not matches:
            print("  No weight-like datasets or attributes found")
        else:
            print("  Weight-like fields:")
            for kind, name, shape, dtype, value in matches:
                if kind == "dataset":
                    print(f"    dataset {name}: shape={shape} dtype={dtype}")
                    print(f"      sample={value}")
                else:
                    print(f"    attribute {name}={value!r}")
        return dsid


def print_atlasopenmagic(dsids: set[int], release: str) -> None:
    if not dsids:
        return
    try:
        import atlasopenmagic as atom
    except ImportError:
        print("\natlasopenmagic is not installed; skipping official metadata lookup")
        return

    if release:
        atom.set_release(release)
    print("\nAtlasOpenMagic metadata:")
    for dsid in sorted(dsids):
        metadata = atom.get_metadata(str(dsid))
        print(f"  DSID {dsid}")
        for key in (
            "cross_section_pb",
            "genFiltEff",
            "kFactor",
            "nEvents",
            "sumOfWeights",
            "sumOfWeightsSquared",
        ):
            print(f"    {key}: {metadata.get(key)!r}")


def main() -> None:
    args = parse_args()
    dsids = set()
    for file_list in args.file_lists:
        print(f"\n=== {file_list} ===")
        for path in read_paths(file_list, args.files_per_list):
            dsid = inspect_file(path, args.sample_values)
            if dsid > 0:
                dsids.add(dsid)
    print_atlasopenmagic(dsids, args.atlasopenmagic_release)


if __name__ == "__main__":
    main()
