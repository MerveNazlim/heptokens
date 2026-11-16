#!/usr/bin/env python3
"""Freeze and materialize one reconstructed e/mu >=4 five-class cohort.

The original Q8 train/validation/test membership is authoritative. A selected
view copies complete Parquet rows by actual event identity, never reassigns a
split, and never fits preprocessing, tokenizers or models.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from heptokens.data.canonical_batch_schedule import (
    build_index, build_schedule, event_identity_digest, load_index, source_dictionary_digest,
)

NAMES = ("ggf", "vbf", "vh", "tth", "zz_continuum")
SPLITS = ("train", "val", "test")
SEEDS = (42, 43, 44)
SOURCE_SHA = "19cdf0a7afe372d9269df1213474e5945196c63c34b8b864b447b60197f8ba3a"
LEGACY_Q1_SOURCE_SHA = "277440adf9881f2949b36d9bc1c47466f85dbe6bc9700129f7f0ea2e5b3dbab3"
SELECTION_ID = "hzz_4l_active_e_mu_min4_v1"
PREDICATE = dict(minimum=4, object_types=["electrons", "muons"],
    type_ids=dict(electrons=4, muons=5), require_active_mask=True,
    no_tau_candidates=True, no_mass_or_pairing_cut=True)
SELECTION_RECEIPT = "selection_receipt.json"
MEMBERSHIP = "membership.npz"
VIEW_RECEIPT = "selection_view.json"
IDENTITY_DTYPE = np.dtype([("source", "<u4"), ("event", "<i8")])
FIELDS = dict(source_id="<u4", event_index="<i8", labels="u1", dsids="<i8", layout="S32")
IDENTITY_COLUMNS = ("source_file", "event_index", "label", "dsid", "mask", "type_ids")
REPRESENTATIONS = {"q1": (1, 98820, "q1_reencoding"), "q4": (4, 262660, "q4_reencoding"),
    "q8": (8, 131588, None), "continuous": (8, 131588, "continuous_preparation")}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def new_output(destination, *protected):
    target = Path(destination).resolve()
    partial = target.with_name(target.name + ".partial")
    if target.exists() or partial.exists() or any(
            p == target or p in target.parents for p in map(lambda x: Path(x).resolve(), protected)):
        raise FileExistsError(f"Use a new output directory outside the protected inputs: {target}")
    partial.mkdir(parents=True)
    return target, partial


def read_manifest(root, expected_sha=None):
    path = Path(root) / "manifest.json"
    digest = sha256(path)
    if expected_sha is not None and digest != expected_sha:
        raise ValueError("Original representation manifest SHA256 differs")
    manifest = json.loads(path.read_text())
    definitions = sorted(manifest["classes"], key=lambda entry: entry["label"])
    if (tuple(entry["name"] for entry in definitions) != NAMES
            or [entry["label"] for entry in definitions] != list(range(5))
            or manifest.get("n_classes") != 5 or manifest.get("identity_overlap_detected") is not False):
        raise ValueError("Require the five original classes and disjoint splits")
    return manifest, digest, definitions


def inventory(root):
    root = Path(root)
    by_split = {split: [(label, name, path) for label, name in enumerate(NAMES)
        for path in sorted((root / split / name).glob("*.parquet"))] for split in SPLITS}
    for split, items in by_split.items():
        if ({p for _, _, p in items} != set((root / split).rglob("*.parquet"))
                or set(name for _, name, _ in items) != set(NAMES)):
            raise ValueError(f"Unexpected/missing Parquet shards for {split}")
    return by_split


def metadata(handle, representation):
    q, vocab, _ = REPRESENTATIONS[representation]
    schema = handle.schema_arrow
    if not set(IDENTITY_COLUMNS).issubset(schema.names):
        raise ValueError("Classification shard lacks canonical IDs, labels or object masks")
    meta = schema.metadata or {}
    v = json.loads(meta[b"heptokens_token_vocabulary"])
    if (v.get("max_quantizers"), v.get("vocab_size")) != (q, vocab):
        raise ValueError(f"Wrong fixed tokenizer vocabulary for {representation}")
    for name, type_id in PREDICATE["type_ids"].items():
        if v["objects"][name]["type_id"] != type_id:
            raise ValueError("Selected e/mu type IDs differ from original Q8 vocabulary")
    if representation == "continuous" and (
            b"heptokens_continuous_schema" not in meta or not {
                "continuous_features", "continuous_feature_mask", "position_role_ids"
            }.issubset(schema.names)):
        raise ValueError("Continuous view lacks the original H5-derived feature/role schema")


def column(batch, name):
    return batch.column(batch.schema.get_field_index(name))


def codes(batch, names, mapping, *, extend=False):
    encoded = pc.dictionary_encode(column(batch, "source_file"))
    if encoded.null_count or encoded.dictionary.null_count:
        raise ValueError("Null event source")
    dictionary = encoded.dictionary.to_pylist()
    for name in dictionary:
        if not isinstance(name, str) or not name:
            raise ValueError("Invalid event source")
        if name not in mapping:
            if not extend:
                raise ValueError(f"Unexpected source file: {name}")
            mapping[name] = len(names); names.append(name)
    numeric = np.asarray([mapping[name] for name in dictionary], dtype="<u4")
    return numeric[np.asarray(encoded.indices.to_numpy(), dtype=np.int64)]


def identities(batch, expected_label, allowed_dsids):
    values = []
    for name in ("event_index", "label", "dsid"):
        arr = column(batch, name)
        if arr.null_count or not pa.types.is_integer(arr.type):
            raise ValueError(f"Require non-null integer {name}")
        values.append(np.asarray(arr.to_numpy(), dtype=np.int64))
    event, label, dsid = values
    if np.any(event < 0) or not np.all(label == expected_label) or not np.isin(dsid, allowed_dsids).all():
        raise ValueError("Wrong event index, class label or DSID")
    return event, label.astype("u1"), dsid


def arrays(batch, length):
    mask, types = column(batch, "mask"), column(batch, "type_ids")
    if (not pa.types.is_fixed_size_list(mask.type) or not pa.types.is_fixed_size_list(types.type)
            or mask.type.list_size != length or types.type.list_size != length
            or not pa.types.is_boolean(mask.type.value_type) or not pa.types.is_integer(types.type.value_type)
            or mask.null_count or types.null_count or mask.flatten().null_count or types.flatten().null_count):
        raise ValueError("Invalid mask/type sequence layout")
    active = np.asarray(mask.flatten().to_numpy(zero_copy_only=False)).reshape(-1, length)
    types = np.asarray(types.flatten().to_numpy()).reshape(-1, length)
    if np.any(types < 0) or np.any(types > 255):
        raise ValueError("Out-of-range position type ID")
    return active, types.astype("u1")


def layout_hashes(active, types):
    values = np.concatenate((active.astype("u1"), types), axis=1)
    return np.asarray([hashlib.sha256(row).digest() for row in values], dtype="S32")


def keys(source, event):
    output = np.empty(len(source), dtype=IDENTITY_DTYPE)
    output["source"], output["event"] = source, event
    return output


def require_unique(value):
    ordered = np.sort(value)
    if len(ordered) > 1 and np.any(ordered[1:] == ordered[:-1]):
        raise ValueError("Duplicate event ID within a split or across train/val/test")


def counts(labels):
    return {**{name: int(np.sum(labels == index)) for index, name in enumerate(NAMES)},
            "total": len(labels)}


def weights(split_counts):
    train = split_counts["train"]
    return {name: train["total"] / (5 * train[name]) for name in NAMES}


def digest_membership(data, sources, *, sorted_ids=False):
    order = np.argsort(keys(data["source_id"], data["event_index"])) if sorted_ids else slice(None)
    return event_identity_digest(data["source_id"][order], data["event_index"][order],
                                 data["labels"][order], sources=sources)


def select(source_dir, output_dir, *, expected_source_sha=SOURCE_SHA, original_index_dir=None):
    source = Path(source_dir).resolve()
    manifest, source_sha, classes = read_manifest(source, expected_source_sha)
    length, q = manifest["token_shape"]
    if q != 8:
        raise ValueError("The authoritative selection must use original Q8")
    original = load_index(original_index_dir) if original_index_dir is not None else None
    if original is not None and original.manifest["source_manifest_sha256"] != source_sha:
        raise ValueError("Original index identifies a different source manifest")
    target, partial = new_output(output_dir, source)
    sources, mapping, full, selected, shards, selected_counts = [], {}, [], {}, {}, {}
    for split, items in inventory(source).items():
        parts = {key: [] for key in FIELDS}
        seen = {name: 0 for name in NAMES}; offset = 0
        for label, name, path in items:
            handle = pq.ParquetFile(path); metadata(handle, "q8")
            original_rows, kept = 0, 0
            for batch in handle.iter_batches(batch_size=4096, columns=IDENTITY_COLUMNS):
                source_ids = codes(batch, sources, mapping, extend=True)
                events, labels, dsids = identities(batch, label, classes[label]["dsids"])
                active, types = arrays(batch, length)
                choose = np.sum(active & np.isin(types, (4, 5)), axis=1) >= 4
                if original is not None:
                    saved = original.splits[split]; section = slice(offset, offset + len(events))
                    saved_sources = np.asarray([original.sources[int(i)] for i in saved.source_id[section]])
                    actual_sources = np.asarray([sources[int(i)] for i in source_ids])
                    if not (np.array_equal(saved_sources, actual_sources)
                            and np.array_equal(saved.event_index[section], events)
                            and np.array_equal(saved.labels[section], labels)):
                        raise ValueError("Original Q8 IDs/labels differ from the authoritative index")
                else:
                    full.append(keys(source_ids, events))
                fields = dict(source_id=source_ids[choose], event_index=events[choose], labels=labels[choose],
                    dsids=dsids[choose], layout=layout_hashes(active[choose], types[choose]))
                for key, value in fields.items():
                    parts[key].append(np.asarray(value, dtype=FIELDS[key]))
                original_rows += len(events); offset += len(events); kept += int(choose.sum())
            relative = path.relative_to(source).as_posix()
            if original_rows != handle.metadata.num_rows:
                raise ValueError(f"Incomplete Q8 scan: {relative}")
            shards[relative] = dict(rows=original_rows, selected_rows=kept, sha256=sha256(path))
            seen[name] += original_rows
        seen["total"] = offset
        if seen != manifest["split_counts"][split] or original is not None and offset != len(original.splits[split]):
            raise ValueError(f"Original split/count mismatch: {split}")
        selected[split] = {key: np.concatenate(value) for key, value in parts.items()}
        selected_counts[split] = counts(selected[split]["labels"])
        if min(selected_counts[split][name] for name in NAMES) == 0:
            raise ValueError(f"Selection lost an entire class in {split}")
        print(f"PASS selected {selected_counts[split]['total']:,} / {offset:,} {split} events", flush=True)
    if full:
        require_unique(np.concatenate(full))
    with (partial / MEMBERSHIP).open("xb") as stream:
        np.savez_compressed(stream, **{f"{split}_{key}": value for split, data in selected.items()
                                      for key, value in data.items()})
    (partial / "source_manifest.json").write_bytes((source / "manifest.json").read_bytes())
    receipt = dict(format_version=1, passed=True, selection_id=SELECTION_ID, predicate=PREDICATE,
        source_manifest_sha256=source_sha, source_split_counts=manifest["split_counts"],
        source_shards=shards, class_names=list(NAMES), split_counts=selected_counts,
        recommended_cross_entropy_weights=weights(selected_counts), sources=sources,
        source_dictionary_sha256=source_dictionary_digest(sources), membership_sha256=sha256(partial / MEMBERSHIP),
        original_index_identity_sha256=original.identity_sha256 if original is not None else None,
        sequence_length=length,
        ordered_identity_sha256={s: digest_membership(d, sources) for s, d in selected.items()},
        membership_identity_sha256={s: digest_membership(d, sources, sorted_ids=True) for s, d in selected.items()})
    write_json(partial / SELECTION_RECEIPT, receipt)
    partial.rename(target)
    return receipt


def load_selection(root):
    root = Path(root)
    receipt = json.loads((root / SELECTION_RECEIPT).read_text())
    if (receipt.get("format_version") != 1 or receipt.get("passed") is not True
            or receipt.get("selection_id") != SELECTION_ID or receipt.get("predicate") != PREDICATE
            or receipt.get("class_names") != list(NAMES)
            or sha256(root / MEMBERSHIP) != receipt.get("membership_sha256")
            or sha256(root / "source_manifest.json") != receipt.get("source_manifest_sha256")
            or source_dictionary_digest(receipt["sources"]) != receipt.get("source_dictionary_sha256")):
        raise ValueError("Invalid or altered four-lepton selection receipt")
    data = {}
    with np.load(root / MEMBERSHIP, allow_pickle=False) as saved:
        if set(saved.files) != {f"{split}_{key}" for split in SPLITS for key in FIELDS}:
            raise ValueError("Invalid membership archive fields")
        for split in SPLITS:
            rows = {key: saved[f"{split}_{key}"] for key in FIELDS}
            if any(rows[key].shape != (receipt["split_counts"][split]["total"],)
                   or rows[key].dtype != np.dtype(dtype) for key, dtype in FIELDS.items()):
                raise ValueError("Invalid membership vector types/shapes")
            if (counts(rows["labels"]) != receipt["split_counts"][split]
                    or digest_membership(rows, receipt["sources"]) != receipt["ordered_identity_sha256"][split]
                    or digest_membership(rows, receipt["sources"], sorted_ids=True) != receipt["membership_identity_sha256"][split]):
                raise ValueError(f"Membership count or identity digest differs: {split}")
            data[split] = rows
    require_unique(np.concatenate([keys(d["source_id"], d["event_index"]) for d in data.values()]))
    if weights(receipt["split_counts"]) != receipt["recommended_cross_entropy_weights"]:
        raise ValueError("Four-lepton weights differ from selected training counts")
    return receipt, data


def filter_view(selection_dir, source_dir, output_dir, representation, *, expected_parent_sha=None):
    selection_dir, source = Path(selection_dir).resolve(), Path(source_dir).resolve()
    receipt, data = load_selection(selection_dir)
    parent, parent_sha, classes = read_manifest(source, expected_parent_sha)
    original = json.loads((selection_dir / "source_manifest.json").read_text())
    q, _, provenance_key = REPRESENTATIONS[representation]
    provenance = parent.get(provenance_key, {}) if provenance_key else {}
    legacy_q1 = (representation == "q1" and not provenance and parent_sha == LEGACY_Q1_SOURCE_SHA
                 and expected_parent_sha == LEGACY_Q1_SOURCE_SHA
                 and receipt["source_manifest_sha256"] == SOURCE_SHA)
    # This independently prepared Zephyr Q1 parent has representation-specific
    # input paths and predates the H5 reencoding receipts. Its explicit pin,
    # independent Q1 vocabulary, full cohort and selected layout are required.
    class_fields = ("name", "label", "dsids", "available_rows", "dsid_rows")
    same_class_defs = ([{k: c[k] for k in class_fields} for c in parent["classes"]]
                       == [{k: c[k] for k in class_fields} for c in original["classes"]])
    if (parent["token_shape"] != [receipt["sequence_length"], q]
            or parent["split_counts"] != receipt["source_split_counts"] or not same_class_defs
            or not legacy_q1 and (provenance.get("source_manifest_sha256") if provenance_key else parent_sha)
               != receipt["source_manifest_sha256"]):
        raise ValueError("Representation does not derive from the original unsliced classification cohort")
    target, partial = new_output(output_dir, source, selection_dir)
    sources = receipt["sources"]; mapping = {name: i for i, name in enumerate(sources)}
    rows = {key: np.concatenate([data[s][key] for s in SPLITS]) for key in FIELDS}
    all_keys = keys(rows["source_id"], rows["event_index"])
    order = np.argsort(all_keys); sorted_keys = all_keys[order]
    split_ids = np.concatenate([np.full(len(data[s]["labels"]), i, dtype="u1") for i, s in enumerate(SPLITS)])
    seen, shards = np.zeros(len(all_keys), dtype=bool), {}
    source_paths = set(receipt["source_shards"])
    files = inventory(source)
    if {p.relative_to(source).as_posix() for items in files.values() for _, _, p in items} != source_paths:
        raise ValueError("Prepared representation shard inventory differs from the original")
    for split_id, (split, items) in enumerate(files.items()):
        for label, name, path in items:
            relative = path.relative_to(source).as_posix()
            source_spec = receipt["source_shards"][relative]
            parent_file_sha = sha256(path)
            if provenance_key and not legacy_q1:
                expected = provenance.get("shards", {}).get(relative, {})
                if expected.get("sha256") != parent_file_sha:
                    raise ValueError(f"Prepared input shard checksum differs: {relative}")
                if representation in ("q4", "continuous") and expected.get("source_sha256") != source_spec["sha256"]:
                    raise ValueError(f"Prepared shard's original Q8 checksum differs: {relative}")
            elif representation == "q8" and source_spec["sha256"] != parent_file_sha:
                raise ValueError(f"Original Q8 shard changed after selection: {relative}")
            handle = pq.ParquetFile(path); metadata(handle, representation)
            if handle.metadata.num_rows != source_spec["rows"]:
                raise ValueError(f"Prepared original shard count differs: {relative}")
            out = partial / relative; writer = None; kept = 0
            try:
                for batch in handle.iter_batches(batch_size=2048):
                    source_ids = codes(batch, sources, mapping)
                    events, labels, dsids = identities(batch, label, classes[label]["dsids"])
                    lookup = keys(source_ids, events); positions = np.searchsorted(sorted_keys, lookup)
                    in_range = positions < len(sorted_keys)
                    found = np.zeros(len(positions), dtype=bool)
                    found[in_range] = sorted_keys[positions[in_range]] == lookup[in_range]
                    selected_positions = order[positions[found]]
                    active, types = arrays(batch, receipt["sequence_length"])
                    if not np.array_equal(np.sum(active & np.isin(types, (4, 5)), axis=1) >= 4, found):
                        raise ValueError(f"Selected event membership differs from original Q8 predicate: {relative}")
                    if not found.any():
                        continue
                    if (np.any(split_ids[selected_positions] != split_id)
                            or not np.array_equal(rows["labels"][selected_positions], labels[found])
                            or not np.array_equal(rows["dsids"][selected_positions], dsids[found])):
                        raise ValueError(f"Selected event changed its split/label/DSID: {relative}")
                    if len(np.unique(selected_positions)) != len(selected_positions) or seen[selected_positions].any():
                        raise ValueError(f"Selected event repeated: {relative}")
                    if not np.array_equal(rows["layout"][selected_positions], layout_hashes(active[found], types[found])):
                        raise ValueError(f"Selected event mask/type layout changed: {relative}")
                    seen[selected_positions] = True
                    filtered = pa.Table.from_batches([batch.filter(pa.array(found))])
                    if writer is None:
                        out.parent.mkdir(parents=True, exist_ok=True)
                        writer = pq.ParquetWriter(out, handle.schema_arrow, compression="zstd")
                    writer.write_table(filtered, row_group_size=128)
                    kept += filtered.num_rows
            finally:
                if writer is not None:
                    writer.close()
            if representation == "q8" and kept != source_spec["selected_rows"]:
                raise ValueError(f"Frozen Q8 shard selection changed: {relative}")
            output_sha = None
            if kept:
                check = pq.ParquetFile(out)
                if (not check.schema_arrow.equals(handle.schema_arrow, check_metadata=True)
                        or check.metadata.num_rows != kept or any(check.metadata.row_group(i).num_rows > 128
                             for i in range(check.metadata.num_row_groups))):
                    raise ValueError(f"Filtered Parquet schema or row-group contract differs: {relative}")
                output_sha = sha256(out)
            shards[relative] = dict(parent_rows=source_spec["rows"], rows=kept,
                                    parent_sha256=parent_file_sha, sha256=output_sha)
        print(f"PASS {representation} {split}: exact selected row identities and original payloads", flush=True)
    if not seen.all():
        raise ValueError(f"Missing {int((~seen).sum())} selected event IDs")
    manifest = deepcopy(parent)
    manifest["classes"] = deepcopy(original["classes"])
    # Original provenance is kept verbatim in the companion parent manifest;
    # never describe its unfiltered per-shard receipts as selected artifacts.
    for name in ("q4_reencoding", "q1_reencoding", "continuous_preparation", "shards", "files", "outputs"):
        manifest.pop(name, None)
    manifest.update(split_counts=receipt["split_counts"], recommended_cross_entropy_weights=receipt["recommended_cross_entropy_weights"],
        selected_identity_count=len(seen), selection="at least four active reconstructed electrons/muons; original splits preserved",
        split_method="original exact seeded split, then frozen event-ID selection", compression="zstd",
        four_lepton_selection=dict(selection_id=SELECTION_ID, original_q8_manifest_sha256=receipt["source_manifest_sha256"],
            parent_manifest_sha256=parent_sha, selection_receipt_sha256=sha256(selection_dir / SELECTION_RECEIPT),
            membership_sha256=receipt["membership_sha256"], membership_identity_sha256=receipt["membership_identity_sha256"],
            representation=representation, predicate=PREDICATE, original_split_counts=receipt["source_split_counts"],
            parent_provenance="pinned_zephyr_q1" if legacy_q1 else "verified_canonical_parent"))
    manifest["notes"] = ["Selected payload rows are unchanged. Original class/DSID provenance is in parent_manifest.json.",
        "No resplitting, re-encoding or preprocessing fit. Inverse-frequency weights use selected train rows only."]
    dsid_counts = {}
    for label, entry in enumerate(manifest["classes"]):
        subset = rows["dsids"][rows["labels"] == label]
        entry["available_rows"] = len(subset)
        entry["dsid_rows"] = {str(dsid): int(np.sum(subset == dsid)) for dsid in entry["dsids"]}
        dsid_counts.update(entry["dsid_rows"])
    if "dsid_counts" in manifest:
        manifest["dsid_counts"] = dsid_counts
    (partial / "parent_manifest.json").write_bytes((source / "manifest.json").read_bytes())
    write_json(partial / "manifest.json", manifest)
    view = dict(format_version=1, passed=True, **manifest["four_lepton_selection"],
        manifest_sha256=sha256(partial / "manifest.json"), split_counts=receipt["split_counts"],
        recommended_cross_entropy_weights=receipt["recommended_cross_entropy_weights"], shards=shards)
    write_json(partial / VIEW_RECEIPT, view)
    partial.rename(target)
    return view


def verify_view(source_dir, selection_dir, *, representation=None):
    """Verify staged filtered bytes and their common selected-cohort provenance."""
    source, selection = Path(source_dir), Path(selection_dir)
    receipt, _ = load_selection(selection)
    view = json.loads((source / VIEW_RECEIPT).read_text())
    manifest, manifest_sha, _ = read_manifest(source, view.get("manifest_sha256"))
    bound = manifest.get("four_lepton_selection", {})
    if (view.get("passed") is not True or view.get("format_version") != 1
            or view.get("manifest_sha256") != manifest_sha or not bound
            or any(view.get(key) != value for key, value in bound.items())
            or bound.get("selection_receipt_sha256") != sha256(selection / SELECTION_RECEIPT)
            or bound.get("membership_sha256") != receipt["membership_sha256"]
            or bound.get("membership_identity_sha256") != receipt["membership_identity_sha256"]
            or sha256(source / "parent_manifest.json") != bound.get("parent_manifest_sha256")
            or any(view.get(key) != receipt[key] or manifest.get(key) != receipt[key]
                   for key in ("split_counts", "recommended_cross_entropy_weights"))
            or representation is not None and bound.get("representation") != representation):
        raise ValueError("Filtered view manifest differs from the frozen selection")
    if set(view["shards"]) != set(receipt["source_shards"]):
        raise ValueError("Filtered view original shard inventory differs")
    files = inventory(source)
    actual = {path.relative_to(source).as_posix() for items in files.values() for _, _, path in items}
    if actual != {path for path, item in view["shards"].items() if item["rows"] > 0}:
        raise ValueError("Filtered view has missing or unexpected Parquet shards")
    for relative in actual:
        path = source / relative; entry = view["shards"][relative]
        if sha256(path) != entry["sha256"] or pq.ParquetFile(path).metadata.num_rows != entry["rows"]:
            raise ValueError(f"Filtered shard hash or count changed: {relative}")
    return view


def build_selected_index(source_dir, selection_dir, index_dir, *, batch_size=256, epochs=10):
    source, selection = Path(source_dir).resolve(), Path(selection_dir).resolve()
    receipt, selected = load_selection(selection)
    view = verify_view(source, selection, representation="q8")
    target, partial = new_output(index_dir, source, selection)
    nested = partial / "index"
    build_index(source, nested, expected_manifest_sha256=view["manifest_sha256"])
    index = load_index(nested)
    mapping = {name: i for i, name in enumerate(receipt["sources"])}
    remap = np.asarray([mapping[name] for name in index.sources], dtype="<u4")
    for split in SPLITS:
        actual, expected = index.splits[split], selected[split]
        if not (np.array_equal(remap[actual.source_id], expected["source_id"])
                and np.array_equal(actual.event_index, expected["event_index"])
                and np.array_equal(actual.labels, expected["labels"])):
            raise ValueError(f"Canonical index differs from frozen ordered selected IDs: {split}")
    schedules = []
    for seed in SEEDS:
        for epoch in range(epochs):
            schedule = build_schedule(index, seed, epoch, batch_size=batch_size)
            schedules.append(dict(schedule.metadata, file=schedule.path.name, archive_sha256=sha256(schedule.path)))
        print(f"PASS {epochs} selected canonical schedules, seed {seed}", flush=True)
    updates, dropped = divmod(receipt["split_counts"]["train"]["total"], batch_size)
    if any(s["batches"] != updates or s["dropped_rows"] != dropped for s in schedules):
        raise ValueError("Selected schedule update/drop-last budget differs")
    report = dict(format_version=1, passed=True, selection_id=SELECTION_ID, split_counts=receipt["split_counts"],
        selection_receipt_sha256=sha256(selection / SELECTION_RECEIPT), membership_sha256=receipt["membership_sha256"],
        index_identity_sha256=index.identity_sha256, index_manifest_sha256=sha256(nested / "index_manifest.json"),
        selected_q8_manifest_sha256=view["manifest_sha256"],
        settings=dict(seeds=list(SEEDS), epochs=epochs, batch_size=batch_size, logical_chunk_size=128,
                      shuffle_buffer_size=8192, drop_last=True),
        updates_per_epoch=updates, dropped_rows_per_epoch=dropped, full_optimizer_steps=epochs * updates,
        schedules=schedules)
    write_json(nested / "schedule_campaign.json", report)
    nested.rename(target); partial.rmdir()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    choose = sub.add_parser("select")
    choose.add_argument("--expected-source-manifest-sha256", default=SOURCE_SHA)
    choose.add_argument("--original-index-dir", type=Path)
    view = sub.add_parser("filter")
    view.add_argument("--selection-dir", type=Path, required=True)
    view.add_argument("--representation", choices=tuple(REPRESENTATIONS), required=True)
    view.add_argument("--expected-parent-manifest-sha256")
    index = sub.add_parser("build-index")
    index.add_argument("--selection-dir", type=Path, required=True)
    index.add_argument("--index-dir", type=Path, required=True)
    verify = sub.add_parser("verify-view")
    verify.add_argument("--selection-dir", type=Path, required=True)
    verify.add_argument("--representation", choices=tuple(REPRESENTATIONS))
    for command in (choose, view, index, verify):
        command.add_argument("--source-dir", type=Path, required=True)
    for command in (choose, view):
        command.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "select":
        report = select(args.source_dir, args.output_dir, expected_source_sha=args.expected_source_manifest_sha256,
                        original_index_dir=args.original_index_dir)
    elif args.command == "filter":
        report = filter_view(args.selection_dir, args.source_dir, args.output_dir, args.representation,
                             expected_parent_sha=args.expected_parent_manifest_sha256)
    elif args.command == "verify-view":
        report = verify_view(args.source_dir, args.selection_dir, representation=args.representation)
    else:
        report = build_selected_index(args.source_dir, args.selection_dir, args.index_dir)
    print(json.dumps({k: report[k] for k in ("selection_id", "split_counts", "membership_sha256",
        "manifest_sha256", "index_identity_sha256", "full_optimizer_steps") if k in report}, indent=2))
    return report


if __name__ == "__main__":
    main()
