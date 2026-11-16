#!/usr/bin/env python3
"""Lossless, unselected CMS NanoAOD branch transport for the audited MC workflow.

Each object collection is stored flat with original ROOT counts and event
offsets. This is a separate schema from the older padded CMS review H5s.
"""

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import zlib

import awkward as ak
import h5py
import numpy as np
import uproot

from audit_cms_native_root import FEATURES
from cms_quality_selection import policy


FORMAT = "cms-native-nanoaod-mc-v1"
PROFILES = ("mc", "mc-with-ancestry")
LEGACY_PROFILES = {"tokenizer": "mc", "hzz": "mc", "ttw": "mc",
                   "tth-signal": "mc-with-ancestry"}
OBJECTS = {obj: obj.lower() + "s" for obj in FEATURES}
KINEMATICS = {"pt", "eta", "phi", "mass", "charge"}
SCALARS = ("run", "luminosityBlock", "event")
TREE = "Events"


def dumps(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def native_profile(profile=None, *, keep_gen_ancestry=False):
    if profile is None:
        return "mc-with-ancestry" if keep_gen_ancestry else "mc"
    profile = LEGACY_PROFILES.get(profile, profile)
    if profile not in PROFILES:
        raise ValueError(f"Unknown native CMS H5 profile: {profile!r}")
    if keep_gen_ancestry and profile != "mc-with-ancestry":
        raise ValueError("Generator ancestry conflicts with the selected export")
    return profile


def branch_paths(profile):
    profile = native_profile(profile)
    cfg = policy("mc")
    branches = {name: (f"cms/event/{name}", None) for name in SCALARS}
    branches["genWeight"] = ("weights/genWeight", None)
    branches.update({name: (f"cms/event_filters/{name}", None) for name in cfg["event_filters"]})
    for obj, name in OBJECTS.items():
        common, native = f"common/{name}", f"cms/{name}"
        branches["n" + obj] = (f"{common}/n", None)
        fields = set(FEATURES[obj])
        fields.update(rule["branch"].removeprefix(obj + "_") for rule in cfg["object_cuts"][name])
        for field in sorted(fields):
            group = common if field in KINEMATICS else native
            branches[f"{obj}_{field}"] = (f"{group}/{field}", obj)
    if profile == "mc-with-ancestry":
        branches["nGenPart"] = ("truth/GenPart/n", None)
        for field in ("pdgId", "genPartIdxMother"):
            branches[f"GenPart_{field}"] = (f"truth/GenPart/{field}", "GenPart")
    return branches


def count_path(obj):
    return "truth/GenPart/n" if obj == "GenPart" else f"common/{OBJECTS[obj]}/n"


def offset_path(obj):
    return count_path(obj).rsplit("/", 1)[0] + "/offsets"


def file_fingerprint(path):
    digest, adler, size = hashlib.sha256(), 1, 0
    with path.open("rb") as handle:
        if handle.read(4) != b"root":
            raise ValueError("Input does not have a ROOT file signature")
        handle.seek(0)
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
            adler = zlib.adler32(block, adler)
            size += len(block)
    return {"root_size_bytes": size, "root_adler32": f"adler32:{adler:08x}", "root_sha256": digest.hexdigest()}


def check_expected(provenance, expected_size=None, expected_checksum=None, expected_uri=None):
    if expected_size is not None and provenance["root_size_bytes"] != expected_size:
        raise ValueError("ROOT input size differs from source manifest")
    if expected_checksum is not None and provenance["root_adler32"] != expected_checksum:
        raise ValueError("ROOT input Adler-32 differs from source manifest")
    if expected_uri is not None and provenance["source_uri"] != expected_uri:
        raise ValueError("ROOT source URI differs from source manifest")


def primitive(array, branch):
    flat = ak.to_numpy(array)
    if np.ma.isMaskedArray(flat):
        if np.any(np.ma.getmaskarray(flat)):
            raise ValueError(f"{branch}: missing values cannot be represented by a physical zero")
        flat = flat.data
    value = np.asarray(flat)
    if value.ndim != 1 or value.dtype.kind not in "biuf":
        raise ValueError(f"{branch}: expected a one-dimensional primitive NanoAOD leaf")
    return value


def append(dataset, values):
    if not len(values):
        return
    end = len(dataset)
    dataset.resize((end + len(values),))
    dataset[end:] = values


def make_dataset(handle, name, dtype):
    return handle.create_dataset(name, shape=(0,), maxshape=(None,), dtype=dtype, chunks=(65536,),
                                 compression="gzip", compression_opts=1, shuffle=True, fletcher32=True)


def stat_key(path):
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def convert(source, output, profile=None, *, keep_gen_ancestry=False, source_uri=None,
            expected_size=None, expected_checksum=None, step_size="32 MB", record_id=None):
    """Atomically write a new H5 containing the study's NanoAOD branches."""
    source, output = Path(source), Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Refusing to overwrite: {output}")
    if record_id is not None and (type(record_id) is not int or record_id < 1):
        raise ValueError("record_id must be a positive integer")
    profile = native_profile(profile, keep_gen_ancestry=keep_gen_ancestry)
    mapping = branch_paths(profile)
    initial = stat_key(source)
    provenance = file_fingerprint(source)
    check_expected(provenance, expected_size, expected_checksum)
    provenance.update(source_uri=str(source) if source_uri is None else source_uri, profile=profile,
                      record_id=record_id, tree_name=TREE)
    if not isinstance(provenance["source_uri"], str) or not provenance["source_uri"]:
        raise ValueError("source_uri must be nonempty")

    with uproot.open(source) as root:
        tree = root[TREE]
        available = tree.keys()
        missing = sorted(set(mapping) - set(available))
        if missing:
            raise ValueError(f"Missing required native CMS ROOT branches: {', '.join(missing)}")
        names = [name for name in available if name in mapping]
        if len(names) != len(mapping):
            raise ValueError("Duplicate required native branch names")
        provenance["root_entries"] = int(tree.num_entries)
        empty = tree.arrays(names, entry_start=0, entry_stop=0, library="ak", how=dict)
        groups = tuple(dict.fromkeys(group for _, group in mapping.values() if group is not None))
        registry = {}
        with tempfile.NamedTemporaryFile(dir=output.parent, prefix="." + output.name + ".", suffix=".tmp") as temp:
            temporary = Path(temp.name)
            with h5py.File(temporary, "w") as out:
                metadata = out.create_group("metadata")
                metadata.attrs["format"] = FORMAT
                metadata.attrs["provenance_json"] = dumps(provenance)
                metadata.attrs["selection"] = "none; original ROOT events and objects in source order"
                metadata.attrs["units"] = "unmodified CMS NanoAOD; pT/mass GeV, eta dimensionless, phi radians"
                metadata.attrs["object_layout"] = "flat primitive arrays with common collection offsets; no padding"
                metadata.attrs["status"] = "writing"
                for name in names:
                    path, group = mapping[name]
                    arr = empty[name] if group is None else ak.flatten(empty[name], axis=1)
                    dtype = primitive(arr, name).dtype
                    ds = make_dataset(out, path, dtype)
                    branch = tree[name]
                    ds.attrs["source_branch"] = name
                    ds.attrs["root_typename"] = branch.typename
                    registry[name] = {"path": path, "collection": group, "dtype": dtype.str,
                                      "typename": branch.typename, "title": getattr(branch, "title", "")}
                for group in groups:
                    ds = make_dataset(out, offset_path(group), np.int64)
                    append(ds, np.array([0], dtype=np.int64))
                    ds.attrs["note"] = "One monotonically nondecreasing offset per event boundary"
                ends = dict.fromkeys(groups, 0)
                seen = 0
                for batch in tree.iterate(names, step_size=step_size, library="ak", how=dict):
                    n = len(batch["event"])
                    counts = {}
                    for group in groups:
                        name = "n" + group
                        raw = primitive(batch[name], name)
                        if raw.shape != (n,) or raw.dtype.kind not in "iu" or np.any(raw < 0) or np.any(
                            raw > np.iinfo(np.int64).max
                        ):
                            raise ValueError(f"{name}: invalid source multiplicities")
                        counts[group] = raw.astype(np.int64, copy=False)
                    for name in names:
                        path, group = mapping[name]
                        if group is None:
                            values = primitive(batch[name], name)
                            if values.shape != (n,):
                                raise ValueError(f"{name}: scalar length differs from event count")
                        else:
                            leaf = batch[name]
                            if not np.array_equal(primitive(ak.num(leaf, axis=1), name), counts[group]):
                                raise ValueError(f"{name}: object lengths disagree with n{group}")
                            values = primitive(ak.flatten(leaf, axis=1), name)
                        if values.dtype.str != registry[name]["dtype"]:
                            raise ValueError(f"{name}: ROOT leaf dtype changed during iteration")
                        append(out[path], values)
                    for group, count in counts.items():
                        offsets = ends[group] + np.cumsum(count, dtype=np.int64)
                        if len(offsets):
                            if offsets[-1] < ends[group]:
                                raise ValueError(f"{group}: object offset overflow")
                            append(out[offset_path(group)], offsets)
                            ends[group] = int(offsets[-1])
                    seen += n
                if seen != tree.num_entries or stat_key(source) != initial:
                    raise ValueError("ROOT source changed or was not fully copied")
                for name, (path, group) in mapping.items():
                    if len(out[path]) != (seen if group is None else ends[group]):
                        raise ValueError(f"{name}: stored value count differs from source")
                metadata.attrs["branches_json"] = json.dumps(registry, separators=(",", ":"), allow_nan=False)
                metadata.attrs.modify("status", "complete")
                out.flush()
            # link() publishes only when output does not exist, even if raced.
            os.link(temporary, output)
    return provenance


@dataclass(frozen=True)
class NativeBranch:
    typename: str
    title: str


def approx_entries(step_size, num_entries, datasets):
    if type(step_size) is int and step_size > 0:
        return step_size
    if isinstance(step_size, str):
        match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([kMGT]?i?B)\s*", step_size, re.IGNORECASE)
        if match:
            magnitude, unit = match.groups()
            unit = unit.upper()
            power = {"": 0, "K": 1, "M": 2, "G": 3, "T": 4}.get(unit[0] if unit != "B" else "")
            if power is not None:
                budget = float(magnitude) * (1024 if "I" in unit else 1000) ** power
                if 0 < budget < float("inf"):
                    bytes_per_row = max(1, sum(ds.size * ds.dtype.itemsize for ds in datasets) / max(1, num_entries))
                    return max(1, int(budget / bytes_per_row))
    raise ValueError("step_size must be positive event count or a byte size such as '32 MB'")


class NativeH5Tree:
    def __init__(self, handle):
        self._file = handle
        if "metadata" not in handle:
            raise ValueError("Not a completed strict native CMS H5 file")
        meta = handle["metadata"]
        if meta.attrs.get("format") != FORMAT or meta.attrs.get("status") != "complete":
            raise ValueError("Not a completed strict native CMS H5 file")
        provenance = self.provenance = json.loads(meta.attrs["provenance_json"])
        self.num_entries = n = provenance["root_entries"]
        if (
            type(n) is not int or n < 0 or provenance["tree_name"] != TREE
            or type(provenance["root_size_bytes"]) is not int or provenance["root_size_bytes"] < 4
            or not isinstance(provenance["source_uri"], str) or not provenance["source_uri"]
            or not re.fullmatch(r"adler32:[0-9a-f]{8}", provenance["root_adler32"])
            or not re.fullmatch(r"[0-9a-f]{64}", provenance["root_sha256"])
            or (provenance["record_id"] is not None and
                (type(provenance["record_id"]) is not int or provenance["record_id"] < 1))
        ):
            raise ValueError("Invalid recorded native CMS ROOT provenance")
        self._registry = registry = json.loads(meta.attrs["branches_json"])
        mapping = branch_paths(provenance["profile"])
        if set(registry) != set(mapping):
            raise ValueError("Native CMS H5 branch registry is incomplete")
        self._groups = {}
        for name, entry in registry.items():
            path, group = mapping[name]
            if (entry["path"], entry["collection"]) != (path, group):
                raise ValueError(f"{name}: invalid native branch path")
            ds = handle[path]
            if (ds.ndim != 1 or ds.dtype.str != entry["dtype"] or not ds.fletcher32
                    or ds.attrs.get("source_branch") != name
                    or ds.attrs.get("root_typename") != entry["typename"]):
                raise ValueError(f"{name}: invalid native branch dataset")
            if group is None:
                if ds.shape != (n,):
                    raise ValueError(f"{name}: invalid native scalar length")
                continue
            if group not in self._groups:
                offsets = handle[offset_path(group)]
                counts = handle[count_path(group)]
                if (offsets.shape != (n + 1,) or offsets.dtype != np.dtype(np.int64) or not offsets.fletcher32
                        or offsets[0] != 0 or offsets[-1] < 0 or counts.dtype.kind not in "iu"):
                    raise ValueError(f"{group}: invalid native collection offsets/counts")
                self._groups[group] = offsets, counts, int(offsets[-1])
            if len(ds) != self._groups[group][2]:
                raise ValueError(f"{name}: value length disagrees with final native offset")

    def keys(self):
        return list(self._registry)

    def __getitem__(self, name):
        entry = self._registry[name]
        return NativeBranch(entry["typename"], entry["title"])

    def iterate(self, expressions=None, *, step_size="32 MB", entry_start=0, entry_stop=None,
                library="ak", how=dict):
        """Read the exact requested primitive Awkward columns in source order."""
        if library != "ak" or how is not dict:
            raise ValueError("Native H5 event reader requires library='ak' and how=dict")
        names = self.keys() if expressions is None else ([expressions] if isinstance(expressions, str) else list(expressions))
        entries = {name: self._registry[name] for name in names}
        begin = 0 if entry_start is None else entry_start
        stop_at = self.num_entries if entry_stop is None else entry_stop
        if type(begin) is not int or type(stop_at) is not int or min(begin, stop_at) < 0:
            raise ValueError("Native H5 entry bounds must be nonnegative integers")
        end = min(stop_at, self.num_entries)
        datasets = {name: self._file[entry["path"]] for name, entry in entries.items()}
        requested_groups = {entry["collection"] for entry in entries.values() if entry["collection"] is not None}
        stride = approx_entries(step_size, self.num_entries,
                                [*datasets.values(), *(self._groups[group][0] for group in requested_groups)])
        for start in range(begin, end, stride):
            stop = min(start + stride, end)
            ranges = {}
            for group in requested_groups:
                offsets, counts, total = self._groups[group]
                local = offsets[start:stop + 1]
                length = np.diff(local)
                if local[0] < 0 or local[-1] > total or np.any(length < 0) or not np.array_equal(
                    length, counts[start:stop]
                ):
                    raise ValueError(f"{group}: native collection offsets disagree with counts")
                ranges[group] = int(local[0]), int(local[-1]), length
            batch = {}
            for name, dataset in datasets.items():
                group = entries[name]["collection"]
                if group is None:
                    batch[name] = ak.Array(dataset[start:stop])
                else:
                    a, b, length = ranges[group]
                    batch[name] = ak.unflatten(dataset[a:b], length)
            yield batch


class NativeH5File:
    def __init__(self, path, *, expected_size=None, expected_checksum=None, expected_uri=None):
        self._handle = h5py.File(path, "r")
        try:
            self._tree = NativeH5Tree(self._handle)
            check_expected(self._tree.provenance, expected_size, expected_checksum, expected_uri)
        except BaseException:
            self._handle.close()
            raise

    def __getitem__(self, name):
        if name != TREE:
            raise KeyError(name)
        return self._tree

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._handle.close()


def is_native_h5(path):
    """Detect an HDF5 container (legacy H5 is rejected by the strict reader)."""
    path = Path(path)
    return path.suffix.lower() in (".h5", ".hdf5") or h5py.is_hdf5(path)


@contextmanager
def open_native_h5(path, *, expected_size=None, expected_checksum=None, expected_uri=None):
    with NativeH5File(path, expected_size=expected_size, expected_checksum=expected_checksum,
                      expected_uri=expected_uri) as handle:
        yield handle[TREE]


@contextmanager
def open_event_source(path):
    if is_native_h5(path):
        with NativeH5File(path) as handle:
            yield handle
    else:
        with uproot.open(path) as handle:
            yield handle


def verify_root_identity(path, size, checksum, source_uri=None):
    if is_native_h5(path):
        with open_native_h5(path, expected_size=size, expected_checksum=checksum, expected_uri=source_uri) as tree:
            return tree.provenance
    provenance = file_fingerprint(Path(path))
    check_expected(provenance, size, checksum)
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--keep-gen-ancestry", action="store_true")
    parser.add_argument("--profile", choices=(*PROFILES, *LEGACY_PROFILES), help=argparse.SUPPRESS)
    parser.add_argument("--source-uri")
    parser.add_argument("--expected-size", type=int)
    parser.add_argument("--expected-checksum")
    parser.add_argument("--record-id", type=int)
    parser.add_argument("--step-size", default="32 MB", help="Event count or byte budget, e.g. '32 MB'")
    args = parser.parse_args()
    step_size = int(args.step_size) if args.step_size.isdecimal() else args.step_size
    result = convert(args.source, args.output, args.profile, source_uri=args.source_uri,
                     keep_gen_ancestry=args.keep_gen_ancestry,
                     expected_size=args.expected_size, expected_checksum=args.expected_checksum,
                     record_id=args.record_id, step_size=step_size)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
