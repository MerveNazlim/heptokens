#!/usr/bin/env python3
"""Prepare the isolated ATLAS five-class Paper0-core cohort and exact views."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import prepare_hzz_4l_multiclass as base
import prepare_paper0_atlas_hzz_selected_q1 as paper0
import hzz_4l_core_cohort as cohort


EXPECTED_CORE_COUNTS = {s: dict(zip(base.NAMES, n, strict=True)) | {"total": sum(n)} for s, n in cohort.COUNTS.items()}
MIN4_ID = "hzz_4l_active_e_mu_min4_v1"
MIN4_PREDICATE = dict(minimum=4, object_types=["electrons", "muons"], type_ids=dict(electrons=4, muons=5),
                      require_active_mask=True, no_tau_candidates=True, no_mass_or_pairing_cut=True)


def audit_q8(source, inventory, original):
    """Independently align every original Q8 row and pre-stage its loose candidates."""
    manifest, sha, classes = base.read_manifest(source, base.SOURCE_SHA)
    if (inventory["source_manifest_sha256"] != sha or inventory["classes"] != manifest["classes"]
            or inventory["split_counts"] != manifest["split_counts"]
            or original.manifest["source_manifest_sha256"] != sha or manifest["token_shape"] != [256, 8]):
        raise ValueError("Original H5 inventory/index not bound to the exact Q8 cohort")
    sources = original.sources
    mapping = {name: i for i, name in enumerate(sources)}
    if set(sources) != set(inventory["h5_sources"]) or any(Path(s).name != s for s in sources):
        raise ValueError("Original H5 and Q8 source inventory differs")
    offsets = np.full(len(sources), np.iinfo("i8").max, dtype="i8")
    for data in original.splits.values():
        np.minimum.at(offsets, data.source_id, data.event_index)
    infos = []
    for i, name in enumerate(sources):
        rows = inventory["h5_sources"][name]["selected_events"]
        infos.append(dict(rows=rows, offset=int(offsets[i]), split=np.full(rows, 255, dtype="u1"),
                          multiplicity=np.zeros((rows, 2), dtype="u1"), labels=set(), dsids=set()))
    shard_hashes = {item["path"]: item["sha256"] for item in inventory["q8_shards"]}
    shards, loose = {}, {}
    for split_id, (split, items) in enumerate(base.inventory(source).items()):
        data = original.splits[split]; offset = 0
        parts = {key: [] for key in (*base.FIELDS, "shard_id")}
        for shard_id, (label, name, path) in enumerate(items):
            handle = pq.ParquetFile(path)
            base.metadata(handle, "q8")
            relative = path.relative_to(source).as_posix()
            digest = base.sha256(path)
            shard = data.shards[shard_id]
            if (digest != shard_hashes.get(relative) or shard["relative_path"] != relative or shard["label"] != label
                    or shard["rows"] != handle.metadata.num_rows or shard["offset"] != offset):
                raise ValueError(f"Original Q8 shard differs from the pinned inventory/index: {relative}")
            for batch in handle.iter_batches(batch_size=4096, columns=base.IDENTITY_COLUMNS):
                ids = base.codes(batch, sources, mapping)
                events, labels, dsids = base.identities(batch, label, classes[label]["dsids"])
                section = slice(offset, offset + len(events)); offset += len(events)
                if not all(np.array_equal(actual, expected[section]) for actual, expected in (
                    (ids, data.source_id), (events, data.event_index), (labels, data.labels))):
                    raise ValueError(f"Original Q8 identity/order differs from the canonical index: {relative}")
                active, types = base.arrays(batch, 256)
                mult = np.column_stack([(active & (types == kind)).sum(axis=1) for kind in (4, 5)])
                for src in np.unique(ids):
                    take = ids == src
                    info = infos[src]; local = events[take] - info["offset"]
                    if (np.any(local < 0) or np.any(local >= info["rows"]) or np.any(info["split"][local] != 255)
                            or len(np.unique(local)) != len(local)):
                        raise ValueError(f"Non-contiguous/repeated raw H5 export coordinates: {sources[src]}")
                    info["split"][local], info["multiplicity"][local] = split_id, mult[take]
                    info["labels"].add(label); info["dsids"].update(map(int, dsids[take]))
                kept = mult.sum(axis=1) >= 4
                fields = (ids[kept], events[kept], labels[kept], dsids[kept],
                          base.layout_hashes(active[kept], types[kept]), data.shard_id[section][kept])
                for key, val in zip(parts, fields, strict=True):
                    parts[key].append(val)
            shards[relative] = dict(rows=shard["rows"], sha256=digest)
        if offset != len(data) or len(data.shards) != len(items):
            raise ValueError(f"Incomplete original Q8 split: {split}")
        loose[split] = {key: np.concatenate(parts[key]).astype(base.FIELDS.get(key, "<u4"), copy=False) for key in parts}
        print(f"PASS original Q8 IDs, split, source offset and e/mu counts: {split}", flush=True)
    if (set(shards) != set(shard_hashes) or any(np.any(info["split"] == 255)
            or len(info["labels"]) != 1 or len(info["dsids"]) != 1 for info in infos)):
        raise ValueError("Incomplete Q8/H5 event bridge")
    return manifest, infos, loose, shards


def select_exact_h5(h5_dir, source_inventory, original, infos):
    results, provenance, cutflow = [], {}, defaultdict(Counter)
    for name, info in zip(original.sources, infos, strict=True):
        label, = info["labels"]
        dsid, = info["dsids"]
        class_name = base.NAMES[label]
        spec = source_inventory["h5_sources"][name]
        # select_h5 is the exact validated Paper0 implementation. Its original
        # binary class lookup is extended per DSID, never by multiclass label.
        alias = f"hzz_core_dsid_{dsid}"
        if alias in paper0.CLASSES:
            raise ValueError(f"Reserved transient DSID name: {alias}")
        paper0.CLASSES[alias] = (label, dsid, class_name, label)
        try:
            rows, flow, digest = paper0.select_h5(h5_dir / name, dict(info, name=alias), spec)
        finally:
            del paper0.CLASSES[alias]
        keep = np.zeros(info["rows"], dtype=bool)
        keep[rows["h5_row"]] = True
        if int(keep.sum()) != flow["hzz_core"] or not np.array_equal(info["split"][rows["h5_row"]], rows["split"]):
            raise ValueError(f"Exact H5 core selection repeats or changes an event: {name}")
        results.append(keep)
        cutflow[class_name].update(flow)
        provenance[name] = dict(**spec, class_name=class_name, dsid=dsid, cutflow=flow,
            original_event_index_offset=info["offset"],
            selected_splits={s: int(np.sum(rows["split"] == i)) for i, s in enumerate(base.SPLITS)},
            selected_channels={ch: int(np.sum(rows["channel"] == ch)) for ch in paper0.CHANNELS})
        assert digest == spec["sha256"]
        print(f"PASS exact Paper0 core: {class_name} {name} {flow['hzz_core']:,} / {flow['h5_entries']:,}", flush=True)
    return results, provenance, {key: dict(val) for key, val in cutflow.items()}


def select(q8, h5_dir, inventory_path, original_index, selection):
    q8, h5_dir, inventory_path, original_index, selection = map(Path, (q8, h5_dir, inventory_path, original_index, selection))
    cohort.activate()
    if base.sha256(Path(__file__).with_name("audit_cms_hzz_transfer_v1.py")) != cohort.PREDICATE["candidate_code_sha256"]:
        raise ValueError("Exact Paper0 candidate implementation SHA differs")
    original = base.load_index(original_index)
    inventory = json.loads(inventory_path.read_text())
    manifest, infos, loose, shards = audit_q8(q8, inventory, original)
    keep_by_source, provenance, cutflow = select_exact_h5(h5_dir, inventory, original, infos)
    selected, split_counts = {}, {}
    for split, rows in loose.items():
        keep = np.zeros(len(rows["labels"]), dtype=bool)
        for i, info in enumerate(infos):
            use = rows["source_id"] == i
            keep[use] = keep_by_source[i][rows["event_index"][use] - info["offset"]]
        selected[split] = {key: rows[key][keep] for key in base.FIELDS}
        split_counts[split] = base.counts(selected[split]["labels"])
        retained = np.bincount(rows["shard_id"][keep], minlength=len(original.splits[split].shards))
        for shard, n in zip(original.splits[split].shards, retained, strict=True):
            shards[shard["relative_path"]]["selected_rows"] = int(n)
    if (sum(x["total"] for x in split_counts.values()) != sum(int(k.sum()) for k in keep_by_source)
            or split_counts != EXPECTED_CORE_COUNTS):
        raise ValueError(f"Paper0-core counts differ from the independently verified baseline: {split_counts}")
    target, partial = base.new_output(selection, q8, h5_dir)
    with (partial / base.MEMBERSHIP).open("xb") as stream:
        np.savez_compressed(stream, **{f"{split}_{key}": value for split, data in selected.items() for key, value in data.items()})
    (partial / "source_manifest.json").write_bytes((q8 / "manifest.json").read_bytes())
    receipt = dict(format_version=1, passed=True, selection_id=cohort.SELECTION_ID, predicate=cohort.PREDICATE,
        source_manifest_sha256=base.SOURCE_SHA, source_split_counts=manifest["split_counts"], source_shards=shards,
        class_names=list(base.NAMES), split_counts=split_counts, recommended_cross_entropy_weights=base.weights(split_counts),
        sources=list(original.sources), source_dictionary_sha256=base.source_dictionary_digest(original.sources),
        membership_sha256=base.sha256(partial / base.MEMBERSHIP), original_index_identity_sha256=original.identity_sha256,
        sequence_length=256, ordered_identity_sha256={s: base.digest_membership(d, original.sources) for s, d in selected.items()},
        membership_identity_sha256={s: base.digest_membership(d, original.sources, sorted_ids=True) for s, d in selected.items()},
        paper0_core_provenance=dict(input_inventory_sha256=base.sha256(inventory_path), h5_sources=provenance, cutflow=cutflow,
            selection_implementation=paper0.SELECTION, source_q8_loose_counts={s: base.counts(d["labels"]) for s, d in loose.items()}))
    base.write_json(partial / base.SELECTION_RECEIPT, receipt)
    base.load_selection(partial)
    partial.rename(target)
    print(f"PASS frozen Paper0-core membership: {receipt['membership_sha256']}", flush=True)
    return receipt


def verify_parent(source, representation, receipt, selection, expected_parent_sha, superset_selection):
    parent, parent_sha, classes = base.read_manifest(source, expected_parent_sha)
    original = json.loads((selection / "source_manifest.json").read_text())
    q, _, provenance_key = base.REPRESENTATIONS[representation]
    if parent["token_shape"] != [receipt["sequence_length"], q]:
        raise ValueError("Source token shape differs from the representation")
    actual_paths = {path.relative_to(source).as_posix() for items in base.inventory(source).values() for _, _, path in items}
    if superset_selection:
        try:
            base.SELECTION_ID, base.PREDICATE = MIN4_ID, MIN4_PREDICATE
            superset, _ = base.load_selection(superset_selection)
            old_view = base.verify_view(source, superset_selection, representation=representation)
        finally:
            cohort.activate()
        if (superset["source_manifest_sha256"] != receipt["source_manifest_sha256"]
                or superset["source_split_counts"] != receipt["source_split_counts"]
                or set(superset["source_shards"]) != set(receipt["source_shards"])
                or any((spec["rows"], spec["sha256"]) != (receipt["source_shards"][rel]["rows"], receipt["source_shards"][rel]["sha256"])
                       for rel, spec in superset["source_shards"].items())):
            raise ValueError("Loose superset and Paper0 core are not from the same original Q8 bytes")
        return parent, parent_sha, classes, original, "verified_min4_superset", old_view["shards"], actual_paths
    fields = ("name", "label", "dsids", "available_rows", "dsid_rows")
    if (parent["split_counts"] != receipt["source_split_counts"] or set(receipt["source_shards"]) != actual_paths
            or [{k: c[k] for k in fields} for c in parent["classes"]] != [{k: c[k] for k in fields} for c in original["classes"]]):
        raise ValueError("Input is not the canonical original unsliced cohort")
    provenance = parent.get(provenance_key, {}) if provenance_key else {}
    legacy_q1 = (representation == "q1" and not provenance and parent_sha == base.LEGACY_Q1_SOURCE_SHA
                 and expected_parent_sha == base.LEGACY_Q1_SOURCE_SHA
                 and receipt["source_manifest_sha256"] == base.SOURCE_SHA)
    if not legacy_q1 and (provenance.get("source_manifest_sha256") if provenance_key else parent_sha) != base.SOURCE_SHA:
        raise ValueError("Original source manifest SHA differs")
    parent_specs = {}
    for rel, spec in receipt["source_shards"].items():
        sha = None if legacy_q1 else spec["sha256"]
        if provenance_key and not legacy_q1:
            expected = provenance.get("shards", {}).get(rel, {})
            sha = expected.get("sha256")
            if not sha or representation in ("q4", "continuous") and expected.get("source_sha256") != spec["sha256"]:
                raise ValueError(f"Full source shard is not bound to the original Q8 bytes: {rel}")
        parent_specs[rel] = dict(rows=spec["rows"], sha256=sha)
    return parent, parent_sha, classes, original, "pinned_zephyr_q1" if legacy_q1 else "verified_canonical_parent", parent_specs, actual_paths


def filter_view(selection, source, output, representation, *, expected_parent_sha=None, superset_selection=None):
    selection, source = Path(selection).resolve(), Path(source).resolve()
    receipt, data = cohort.validate_selection(selection)
    parent, parent_sha, classes, original, parent_kind, parent_specs, actual_paths = verify_parent(
        source, representation, receipt, selection, expected_parent_sha, superset_selection)
    target, partial = base.new_output(output, source, selection)
    sources = receipt["sources"]; mapping = {name: i for i, name in enumerate(sources)}
    rows = {key: np.concatenate([data[s][key] for s in base.SPLITS]) for key in base.FIELDS}
    all_keys = base.keys(rows["source_id"], rows["event_index"])
    order = np.argsort(all_keys); sorted_keys = all_keys[order]
    splits = np.concatenate([np.full(len(data[s]["labels"]), i, dtype="u1") for i, s in enumerate(base.SPLITS)])
    seen, shards = np.zeros(len(order), dtype=bool), {}
    for relative, spec in sorted(parent_specs.items()):
        split, name, _ = Path(relative).parts
        split_id, label = base.SPLITS.index(split), base.NAMES.index(name)
        path, out = source / relative, partial / relative
        kept, output_sha, parent_file_sha = 0, None, None
        if relative in actual_paths:
            parent_file_sha = base.sha256(path)
            handle = pq.ParquetFile(path)
            base.metadata(handle, representation)
            if (spec.get("sha256") is not None and spec["sha256"] != parent_file_sha
                    or handle.metadata.num_rows != spec["rows"]):
                raise ValueError(f"Parent row count or SHA differs: {relative}")
            writer = None
            try:
                for batch in handle.iter_batches(batch_size=2048):
                    ids = base.codes(batch, sources, mapping)
                    events, labels, dsids = base.identities(batch, label, classes[label]["dsids"])
                    lookup = base.keys(ids, events); positions = np.searchsorted(sorted_keys, lookup)
                    in_range = positions < len(sorted_keys)
                    found = np.zeros(len(batch), dtype=bool)
                    found[in_range] = sorted_keys[positions[in_range]] == lookup[in_range]
                    if not found.any():
                        continue
                    idx = order[positions[found]]
                    active, types = base.arrays(batch, receipt["sequence_length"])
                    if (np.any(np.sum(active[found] & np.isin(types[found], (4, 5)), axis=1) < 4)
                            or np.any(splits[idx] != split_id) or not np.array_equal(rows["labels"][idx], labels[found])
                            or not np.array_equal(rows["dsids"][idx], dsids[found])
                            or not np.array_equal(rows["layout"][idx], base.layout_hashes(active[found], types[found]))
                            or len(np.unique(idx)) != len(idx) or seen[idx].any()):
                        raise ValueError(f"Paper0-core event payload/identity/split differs or repeats: {relative}")
                    seen[idx] = True
                    table = pa.Table.from_batches([batch.filter(pa.array(found))])
                    if writer is None:
                        out.parent.mkdir(parents=True, exist_ok=True)
                        writer = pq.ParquetWriter(out, handle.schema_arrow, compression="zstd")
                    writer.write_table(table, row_group_size=128)
                    kept += table.num_rows
            finally:
                if writer is not None:
                    writer.close()
            if kept:
                check = pq.ParquetFile(out)
                if (not check.schema_arrow.equals(handle.schema_arrow, check_metadata=True)
                        or check.metadata.num_rows != kept or any(check.metadata.row_group(i).num_rows > 128
                             for i in range(check.metadata.num_row_groups))):
                    raise ValueError(f"Selected Parquet row count/schema changed: {relative}")
                output_sha = base.sha256(out)
        elif spec["rows"] != 0:
            raise ValueError(f"Parent shard missing: {relative}")
        if representation == "q8" and kept != receipt["source_shards"][relative]["selected_rows"]:
            raise ValueError(f"Q8 Paper0-core per-shard membership changed: {relative}")
        shards[relative] = dict(parent_rows=spec["rows"], rows=kept, parent_sha256=parent_file_sha, sha256=output_sha)
    if not seen.all():
        raise ValueError(f"Missing {int((~seen).sum())} frozen Paper0-core events")
    manifest = deepcopy(parent)
    manifest["classes"] = deepcopy(original["classes"])
    for field in ("q4_reencoding", "q1_reencoding", "continuous_preparation", "shards", "files", "outputs"):
        manifest.pop(field, None)
    bound = dict(selection_id=cohort.SELECTION_ID, original_q8_manifest_sha256=receipt["source_manifest_sha256"],
        parent_manifest_sha256=parent_sha, selection_receipt_sha256=base.sha256(selection / base.SELECTION_RECEIPT),
        membership_sha256=receipt["membership_sha256"], membership_identity_sha256=receipt["membership_identity_sha256"],
        representation=representation, predicate=cohort.PREDICATE, original_split_counts=receipt["source_split_counts"],
        parent_provenance=parent_kind)
    manifest.update(split_counts=receipt["split_counts"], recommended_cross_entropy_weights=receipt["recommended_cross_entropy_weights"],
        selected_identity_count=len(order), selection="Exact reconstructed Paper0 CMS-inspired HZZ candidate core; original splits preserved",
        split_method="original exact seeded split, then frozen event-ID selection", compression="zstd", four_lepton_selection=bound)
    manifest["notes"] = ["Selected rows retain their original payload. Parent manifest retained unchanged.",
        "No resplitting, re-encoding or preprocessing refit. Class weights use selected train rows only."]
    dsid_counts = {}
    for label, entry in enumerate(manifest["classes"]):
        dsids = rows["dsids"][rows["labels"] == label]
        entry["available_rows"] = len(dsids)
        entry["dsid_rows"] = {str(dsid): int(np.sum(dsids == dsid)) for dsid in entry["dsids"]}
        dsid_counts.update(entry["dsid_rows"])
    if "dsid_counts" in manifest:
        manifest["dsid_counts"] = dsid_counts
    (partial / "parent_manifest.json").write_bytes((source / "manifest.json").read_bytes())
    base.write_json(partial / "manifest.json", manifest)
    view = dict(format_version=1, passed=True, **bound, manifest_sha256=base.sha256(partial / "manifest.json"),
        split_counts=receipt["split_counts"], recommended_cross_entropy_weights=receipt["recommended_cross_entropy_weights"], shards=shards)
    base.write_json(partial / base.VIEW_RECEIPT, view)
    base.verify_view(partial, selection, representation=representation)
    partial.rename(target)
    print(f"PASS exact {representation} Paper0-core view: {view['manifest_sha256']}", flush=True)
    return view


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    choose = sub.add_parser("select")
    for key in ("q8-dir", "h5-dir", "input-inventory", "original-index-dir", "selection-dir"):
        choose.add_argument(f"--{key}", type=Path, required=True)
    view = sub.add_parser("filter")
    for key in ("selection-dir", "source-dir", "output-dir"):
        view.add_argument(f"--{key}", type=Path, required=True)
    view.add_argument("--representation", choices=tuple(base.REPRESENTATIONS), required=True)
    view.add_argument("--superset-selection-dir", type=Path)
    view.add_argument("--expected-parent-manifest-sha256")
    index = sub.add_parser("build-index")
    for key in ("selection-dir", "source-dir", "index-dir"):
        index.add_argument(f"--{key}", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "select":
        report = select(args.q8_dir, args.h5_dir, args.input_inventory, args.original_index_dir, args.selection_dir)
    elif args.command == "filter":
        report = filter_view(args.selection_dir, args.source_dir, args.output_dir, args.representation,
                             expected_parent_sha=args.expected_parent_manifest_sha256, superset_selection=args.superset_selection_dir)
    else:
        cohort.validate_selection(args.selection_dir)
        report = base.build_selected_index(args.source_dir, args.selection_dir, args.index_dir)
    print(json.dumps({key: report[key] for key in ("selection_id", "split_counts", "membership_sha256", "manifest_sha256",
                                                  "index_identity_sha256", "full_optimizer_steps") if key in report}, indent=2))


if __name__ == "__main__":
    main()
