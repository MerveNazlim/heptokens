#!/usr/bin/env python3
"""Extract native CMS MC pilot tokenizer objects; never starts training."""

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path

import awkward as ak
import numpy as np
import uproot

from audit_cms_native_root import FEATURES, MCPilotCuts, PT_BINS, check_values


EVENT_FIELDS = ("run", "luminosityBlock", "event")
SPLITS = ("train", "validation", "test")
SEED = 42
DTYPES = {
    "run": np.uint32, "luminosityBlock": np.uint32, "event": np.uint64,
    "split": np.uint8, "object_index": np.uint32,
}


@contextmanager
def open_cms_events(source, input_format="root", *, expected_size=None, expected_checksum=None,
                    expected_uri=None, expected_record_id=None, expected_h5_sha256=None):
    """Yield the common event iterator and any opt-in HDF5 lineage.

    ROOT callers retain their existing verification and receipts. For HDF5,
    the converter's recorded ROOT provenance is compared with the caller's
    official inputs before processing; the actual HDF5 bytes get their own hash.
    """
    if input_format == "root":
        if expected_h5_sha256 is not None:
            raise ValueError("An expected HDF5 SHA-256 requires --input-format hdf5")
        with uproot.open(source) as root:
            yield root["Events"], {}
        return
    if input_format != "hdf5":
        raise ValueError(f"Unsupported input format: {input_format}")
    from cms_native_h5 import open_native_h5

    source = Path(source)
    with open_native_h5(source, expected_size=expected_size, expected_checksum=expected_checksum,
                        expected_uri=expected_uri) as tree:
        provenance = tree.provenance
        record_id = provenance.get("record_id")
        if expected_record_id is not None and record_id not in (None, expected_record_id):
            raise ValueError("HDF5 source record ID differs from the requested record")
        digest = hashlib.sha256()
        with source.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 << 20), b""):
                digest.update(chunk)
        h5_sha256 = digest.hexdigest()
        if expected_h5_sha256 is not None and h5_sha256 != expected_h5_sha256:
            raise ValueError("HDF5 input checksum differs from its pinned SHA-256")
        yield tree, {"input_format": "hdf5", "input_h5_sha256": h5_sha256,
                     "native_h5_source": dict(provenance)}


def add_h5_options(parser):
    parser.add_argument("--input-format", choices=("root", "hdf5"), default="root",
                        help="Use the native CMS HDF5 branch-preserving export instead of ROOT")
    parser.add_argument("--expected-h5-sha256",
                        help="Optional SHA-256 pin for the HDF5 file bytes")


def open_campaign_events(source, sample, input_format, expected_h5_sha256=None):
    return open_cms_events(source, input_format, expected_size=sample["size"],
                           expected_checksum=sample["checksum"], expected_uri=sample["uri"],
                           expected_record_id=sample["record_id"], expected_h5_sha256=expected_h5_sha256)


def event_split(record_id, run, lumi, event):
    key = f"{SEED}|mc:{record_id}|{int(run)}|{int(lumi)}|{int(event)}"
    bucket = int.from_bytes(hashlib.blake2b(key.encode(), digest_size=8).digest(), "big") % 10000
    return 0 if bucket < 9000 else 1 if bucket < 9500 else 2


def vector(array, dtype):
    if np.ma.isMaskedArray(array) and np.any(np.ma.getmaskarray(array)):
        raise ValueError("Null event identifier, object index, or selected field")
    return np.asarray(array, dtype=dtype)


def pt_summary(pt):
    if not len(pt):
        return None
    return {
        **dict(zip(("min", "p05", "p50", "p95", "p99", "max"),
                   map(float, np.quantile(pt, [0, .05, .50, .95, .99, 1])))),
        "hist_counts": np.histogram(pt, bins=PT_BINS + [np.inf])[0].tolist(),
    }


def native_values(obj, field, jagged, sizes):
    name = f"{obj}_{field}"
    if not np.array_equal(ak.to_numpy(ak.num(jagged, axis=1)), sizes):
        raise ValueError(f"{name}: jagged lengths differ from n{obj}")
    flat = ak.to_numpy(ak.flatten(jagged, axis=1))
    values = np.asarray(flat, dtype=np.float32)
    if np.ma.isMaskedArray(flat):
        values = np.where(np.ma.getmaskarray(flat), np.nan, values)
    finite, _, suspect, invalid_category = check_values(field, values)
    if field == "cutBased":
        invalid_category |= finite & ((values < 0) | (values > (4 if obj == "Electron" else 3)))
    elif field == "decayMode":
        invalid_category |= finite & ((values < 0) | (values > 255))
    elif field.startswith("idDeepTau"):
        # UL2016 v2p1 nested working-point bit masks. Zero is valid.
        valid_codes = [0, 1, 3, 7, 15] + ([31, 63, 127, 255] if field.endswith("VSe") else [])
        invalid_category |= finite & ~np.isin(values, valid_codes)
    elif field == "nConstituents":
        invalid_category |= finite & ((values < 0) | (values != np.trunc(values)))
    return values, finite & ~suspect & ~invalid_category


def extract(source, record_id, max_events=None, *, input_format="root", expected_size=None,
            expected_checksum=None, expected_uri=None, expected_h5_sha256=None):
    pieces = {"event_" + f: [] for f in (*EVENT_FIELDS, "split")}
    for obj in FEATURES:
        for field in (*EVENT_FIELDS, "split", "object_index", "features"):
            pieces[f"{obj}_{field}"] = []
    cut_counts = {obj: np.zeros(3, dtype=np.int64) for obj in FEATURES}
    input_counts = np.zeros(3, dtype=np.int64)
    rejected_fields = {obj: {f: [0, 0, 0] for f in fields} for obj, fields in FEATURES.items()}
    if input_format == "root" and any(x is not None for x in (expected_size, expected_checksum, expected_uri)):
        raise ValueError("Explicit source provenance options are only supported with --input-format hdf5")
    with open_cms_events(source, input_format, expected_size=expected_size, expected_checksum=expected_checksum,
                         expected_uri=expected_uri, expected_record_id=record_id,
                         expected_h5_sha256=expected_h5_sha256) as (tree, input_lineage):
        cuts = MCPilotCuts(set(tree.keys()))
        required = set(EVENT_FIELDS) | cuts.required | {"n" + obj for obj in FEATURES}
        required.update(f"{obj}_{f}" for obj, fields in FEATURES.items() for f in fields)
        missing = sorted(required - set(tree.keys()))
        if missing:
            raise ValueError(f"Missing required ROOT branches: {', '.join(missing)}")
        expected = tree.num_entries if max_events is None else min(max_events, tree.num_entries)
        for chunk in tree.iterate(sorted(required), entry_stop=expected, step_size="64 MB",
                                  library="ak", how=dict):
            ids = {f: vector(ak.to_numpy(chunk[f]), DTYPES[f]) for f in EVENT_FIELDS}
            n = len(ids["event"])
            split = np.fromiter((event_split(record_id, *t) for t in zip(*ids.values())),
                                dtype=np.uint8, count=n)
            input_counts += np.bincount(split, minlength=3)
            event_keep = cuts.event_mask(chunk, n)
            for field, values in dict(ids, split=split).items():
                pieces[f"event_{field}"].append(values[event_keep])
            for obj, fields in FEATURES.items():
                sizes = vector(ak.to_numpy(chunk["n" + obj]), np.int64)
                if np.any(sizes < 0):
                    raise ValueError(f"Negative n{obj}")
                keep = vector(ak.to_numpy(ak.flatten(cuts.object_mask(obj, chunk, sizes, event_keep), axis=1)), bool)
                row_split = np.repeat(split, sizes)
                cut_counts[obj] += np.bincount(row_split[keep], minlength=3)
                usable = keep.copy()
                columns = []
                for field in fields:
                    values, valid = native_values(obj, field, chunk[f"{obj}_{field}"], sizes)
                    columns.append(values)
                    for code, count in enumerate(np.bincount(row_split[keep & ~valid], minlength=3)):
                        rejected_fields[obj][field][code] += int(count)
                    usable &= valid
                pieces[f"{obj}_features"].append(np.column_stack([v[usable] for v in columns]))
                pieces[f"{obj}_split"].append(row_split[usable])
                for field, values in ids.items():
                    pieces[f"{obj}_{field}"].append(np.repeat(values, sizes)[usable])
                index = ak.to_numpy(ak.flatten(ak.local_index(chunk[f"{obj}_pt"], axis=1), axis=1))
                pieces[f"{obj}_object_index"].append(vector(index[usable], np.uint32))
    if int(input_counts.sum()) != expected:
        raise RuntimeError(f"Expected {expected} ROOT entries; scanned {int(input_counts.sum())}")

    arrays = {"record_id": np.asarray(record_id, dtype=np.int32)}
    for key, chunks in pieces.items():
        if chunks:
            arrays[key] = np.concatenate(chunks)
        else:
            obj, field = key.split("_", 1)
            arrays[key] = (np.empty((0, len(FEATURES[obj])), dtype=np.float32) if field == "features"
                           else np.empty(0, dtype=DTYPES[field]))
    report = {
        **input_lineage,
        "record_id": record_id,
        "events_scanned": expected,
        "split_policy": "BLAKE2b(seed=42, record_id, run, luminosityBlock, event); 90/5/5",
        "cut_policy": cuts.policy["name"],
        "feature_names": FEATURES,
        "pt_hist_lower_edges_gev": PT_BINS,
        "events": {
            name: {"scanned": int(input_counts[code]),
                   "after_event_filters": int(np.count_nonzero(arrays["event_split"] == code))}
            for code, name in enumerate(SPLITS)
        },
        "objects": {},
    }
    for obj in FEATURES:
        report["objects"][obj] = {}
        for code, name in enumerate(SPLITS):
            pt = arrays[f"{obj}_features"][arrays[f"{obj}_split"] == code, 0]
            report["objects"][obj][name] = {
                "objects_after_cuts": int(cut_counts[obj][code]),
                "usable_objects": len(pt),
                "pt_gev": pt_summary(pt),
                "invalid_selected_by_feature": {
                    f: counts[code] for f, counts in rejected_fields[obj].items() if counts[code]
                },
            }
    return arrays, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--record-id", type=int, required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--max-events", type=int)
    add_h5_options(parser)
    parser.add_argument("--source-uri", help="Optionally pin the recorded original ROOT URI for HDF5 input")
    parser.add_argument("--expected-size", type=int, help="Optionally pin the recorded original ROOT size in bytes")
    parser.add_argument("--expected-checksum", help="Optionally pin the recorded original ROOT adler32:... checksum")
    args = parser.parse_args()
    if args.record_id <= 0 or (args.max_events is not None and args.max_events <= 0):
        parser.error("record ID and max events must be positive")
    if args.output_npz.exists() or args.output_json.exists() or args.output_npz == args.output_json:
        parser.error("both output paths must be new and distinct")
    arrays, report = extract(args.source, args.record_id, args.max_events, input_format=args.input_format,
                             expected_size=args.expected_size, expected_checksum=args.expected_checksum,
                             expected_uri=args.source_uri, expected_h5_sha256=args.expected_h5_sha256)
    with args.output_npz.open("xb") as handle:
        np.savez_compressed(handle, **arrays)
    with args.output_json.open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    source_label = "HDF5" if args.input_format == "hdf5" else "ROOT"
    print(f"Extracted five collections from {report['events_scanned']} {source_label} entries for {args.record_id}")


if __name__ == "__main__":
    main()
