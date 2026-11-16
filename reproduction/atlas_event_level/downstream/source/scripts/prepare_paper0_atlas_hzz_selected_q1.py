#!/usr/bin/env python3
"""Apply the CMS HZZ candidate core to the original ATLAS H5 and join Q1 rows.

Use the full Q1 parent split assignments, not the earlier downsampled binary
sample. Require a checksum-pinned H5 inventory and verify each H5 row's join
against the contiguous per-file Q1 index and its electron/muon multiplicities.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack
from itertools import groupby
import json
from pathlib import Path

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from prepare_hzz_4l_multiclass import (
    LEGACY_Q1_SOURCE_SHA, SOURCE_SHA, SPLITS, arrays, metadata, new_output, read_manifest, sha256, write_json,
)

SELECTION_ID = "paper0_atlas_hzz_cms_core_q1_v1"
CLASSES = {"background": (0, 700600, "zz_continuum", 4), "signal": (1, 345060, "ggf", 0)}
LEPTONS = {"Electron": ("electrons", 4, 2.47), "Muon": ("muons", 5, 2.5)}
FIELDS = ("pt", "eta", "phi", "charge")
CHANNELS = ("4e", "2e2mu", "4mu")
BASE_FLOW = ("h5_entries", "at_least_four_active_h5_e_mu", "at_least_four_usable_e_mu",
             "two_disjoint_sfos", "two_disjoint_sfos_both_mll_gt5", "hzz_core")
SELECTION = {
    "candidate_objects": "Active original ATLAS reconstructed H5 e/mu, finite pT/eta/phi, unit charge, pT>10 GeV, "
                         "electron |eta|<2.47 and muon |eta|<2.5. No additional ID/isolation.",
    "candidate_core": "Exact CMS transfer candidate function: two disjoint SFOS pairs; leading/subleading pT>20/10 GeV; "
                      "40<mZ1<120 and 12<mZ2<120 GeV; all pair dR>0.02; all OS pair masses>4 GeV; m4l>70 GeV.",
    "ranking": "Z1 closest to 91.1876 GeV, then largest Z2 scalar lepton pT sum, then source-index order.",
    "mass_window": "No additional mass window; narrower windows in the cutflow are diagnostic only.",
    "input_payload": "Unchanged full ATLAS Q1 event sequences, including taus, tracks, and all original context. "
                     "No additional jet cleaning, retokenization, candidate-only projection or preprocessing fit.",
    "scope": "CMS-inspired candidate selection on the original ATLAS object masks, not the official ATLAS or CMS analysis.",
}


def q1_input(q1, expected_sha):
    manifest, digest, _ = read_manifest(q1, expected_sha)
    if manifest.get("token_shape") != [256, 1]:
        raise ValueError("Expected original full Q1 [256,1] classification inputs")
    paths = {(split, name): sorted((q1 / split / parent).glob("*.parquet"))
             for split in SPLITS for name, (_, _, parent, _) in CLASSES.items()}
    for (split, name), shards in paths.items():
        parent = CLASSES[name][2]
        if not shards or sum(pq.ParquetFile(p).metadata.num_rows for p in shards) != manifest["split_counts"][split][parent]:
            raise ValueError(f"Original full Q1 shard inventory differs for {split}/{name}")
    return paths, digest


def q1_index(q1, paths):
    """Resolve each historical Q1 global exporter offset without assuming file order."""
    parts, shard_sha = defaultdict(list), {}
    for (split, name), shards in paths.items():
        _, expected_dsid, _, old_label = CLASSES[name]
        for path in shards:
            shard_sha[path.relative_to(q1).as_posix()] = sha256(path)
            handle = pq.ParquetFile(path)
            metadata(handle, "q1")
            for b in handle.iter_batches(batch_size=4096, columns=[
                "source_file", "source_file_events", "event_index", "dsid", "label", "mask", "type_ids",
            ]):
                source, n, event, dsid, label, _, _ = b.columns
                names = np.asarray(source.to_pylist())
                n, event, dsid, label = [c.to_numpy() for c in (n, event, dsid, label)]
                if source.null_count or np.any(n <= 0) or np.any(event < 0) or np.any(dsid != expected_dsid) or np.any(label != old_label):
                    raise ValueError(f"Invalid original Q1 class/identity: {path}")
                active, kinds = arrays(b, 256)
                mult = np.column_stack([(active & (kinds == typ)).sum(axis=1) for _, typ, _ in LEPTONS.values()])
                for source_name in np.unique(names):
                    use = names == source_name
                    parts[str(source_name)].append((name, split, n[use], event[use], mult[use]))
    result = {}
    for source, blocks in parts.items():
        names = {b[0] for b in blocks}
        size = np.concatenate([b[2] for b in blocks])
        event = np.concatenate([b[3] for b in blocks])
        order = np.argsort(event)
        if (len(names) != 1 or np.any(size != len(event))
                or np.any(np.diff(event[order]) != 1)):
            raise ValueError(f"Original Q1 does not cover each source row exactly once: {source}")
        result[source] = dict(name=names.pop(), offset=int(event[order[0]]), rows=len(event),
            split=np.concatenate([np.full(len(b[3]), SPLITS.index(b[1]), dtype="u1") for b in blocks])[order],
            multiplicity=np.concatenate([b[4] for b in blocks])[order])
    return result, shard_sha


def select_h5(path, info, expected, *, chunk_size=8192):
    # The production CMS functions are used directly; the ATLAS masks replace
    # CMS-specific quality/ID decisions. Its ROOT audit imports uproot.
    from audit_cms_hzz_transfer_v1 import WINDOWS, find_candidates

    digest = sha256(path)
    if path.stat().st_size != expected["bytes"] or digest != expected["sha256"] or expected["selected_events"] != info["rows"]:
        raise ValueError(f"Original H5 input differs from the verified inventory: {path.name}")
    n = info["rows"]
    _, dsid, _, _ = CLASSES[info["name"]]
    flow = Counter(dict.fromkeys((*BASE_FLOW, *(f"m4l_{lo}_{hi}" for lo, hi in WINDOWS)), 0))
    flow["h5_entries"] = n
    selected = defaultdict(list)
    with h5py.File(path, "r") as h5:
        attrs = h5["metadata"].attrs
        if (attrs["units_pt"] != "GeV" or attrs["units_angles"] != "radians"
                or int(attrs["n_events"]) != n or int(attrs["dsid"]) != dsid):
            raise ValueError(f"H5 units/rows/DSID differ: {path.name}")
        event = h5["atlas/event"]
        if any(event[key].shape != (n,) for key in ("mcChannelNumber", "run_number", "event_number")):
            raise ValueError(f"H5 event identity shape differs: {path.name}")
        for start in range(0, n, chunk_size):
            slc = slice(start, min(start + chunk_size, n))
            run, number = (np.asarray(event[k][slc], dtype=np.int64) for k in ("run_number", "event_number"))
            if np.any(event["mcChannelNumber"][slc] != dsid) or np.any(run < 0) or np.any(number < 0):
                raise ValueError(f"H5 event identity invalid: {path.name}")
            active_counts, usable_counts, objects = [], [], {}
            for obj, (collection, _, eta_max) in LEPTONS.items():
                h = h5[f"common/{collection}"]
                active = np.asarray(h["mask"][slc], dtype=bool)
                pt, eta, phi, charge = (np.asarray(h[key][slc], dtype=np.float64) for key in FIELDS)
                if active.ndim != 2 or any(x.shape != active.shape for x in (pt, eta, phi, charge)):
                    raise ValueError(f"H5 object layout invalid: {path.name}/{collection}")
                if not np.array_equal(active.sum(axis=1), h["n"][slc]):
                    raise ValueError(f"H5 mask multiplicity disagrees with saved count: {path.name}/{collection}")
                usable = (active & np.isfinite(pt) & np.isfinite(eta) & np.isfinite(phi)
                          & (pt > 10) & (np.abs(eta) < eta_max) & (np.abs(charge) == 1))
                active_counts.append(active.sum(axis=1)); usable_counts.append(usable.sum(axis=1))
                objects[obj] = (usable, pt, eta, phi, charge)
            if not np.array_equal(np.column_stack(active_counts), info["multiplicity"][slc]):
                raise ValueError(f"H5 raw row order does not reproduce original Q1 e/mu mask counts: {path.name}")
            flow["at_least_four_active_h5_e_mu"] += int(np.count_nonzero(sum(active_counts) >= 4))
            candidates = np.flatnonzero(sum(usable_counts) >= 4)
            flow["at_least_four_usable_e_mu"] += len(candidates)
            for i in candidates:
                leptons = [(float(pt[i, j]), float(eta[i, j]), float(phi[i, j]), int(charge[i, j]), obj)
                           for obj, (usable, pt, eta, phi, charge) in objects.items() for j in np.flatnonzero(usable[i])]
                loose, mass5, best = find_candidates(leptons)
                flow["two_disjoint_sfos"] += int(loose)
                flow["two_disjoint_sfos_both_mll_gt5"] += int(mass5)
                if best is None:
                    continue
                _, channel, m4l, _, _ = best
                for key, value in zip(("h5_row", "run_number", "event_number", "m4l", "channel"),
                                      (start + i, run[i], number[i], m4l, channel), strict=True):
                    selected[key].append(value)
                flow["hzz_core"] += 1
                for lo, hi in WINDOWS:
                    flow[f"m4l_{lo}_{hi}"] += int(lo < m4l < hi)
    cols = {name: np.asarray(selected[name], dtype=dtype) for name, dtype in {
        "h5_row": "<i8", "run_number": "<i8", "event_number": "<i8", "m4l": "<f8", "channel": "U5",
    }.items()}
    cols["event_index"] = cols["h5_row"] + info["offset"]
    cols["split"] = info["split"][cols["h5_row"]]
    return cols, dict(flow), digest


def audit_selected_event_numbers(h5_dir, index, selected):
    """Reject repeated raw lepton payloads when ATLAS MC run/event numbers collide."""
    counts = {}
    with ExitStack() as stack:
        opened = {}

        def fingerprint(source, row):
            if source not in opened:
                opened[source] = stack.enter_context(h5py.File(h5_dir / source, "r"))
            h5 = opened[source]
            return tuple(np.asarray(h5[f"common/{collection}/{key}"][row]).tobytes()
                         for collection, _, _ in LEPTONS.values() for key in (*FIELDS, "mask"))

        for name in CLASSES:
            stats = Counter(dict.fromkeys(("colliding_keys", "colliding_rows", "cross_split_keys"), 0))
            ids = np.concatenate([
                np.rec.fromarrays([rows["run_number"], rows["event_number"],
                    np.full(len(rows["h5_row"]), source), rows["h5_row"], rows["split"]],
                    names="run,event,source,row,split")
                for source, rows in selected.items() if index[source]["name"] == name])
            ids.sort(order=("run", "event"))
            repeated = (ids["run"][:-1] == ids["run"][1:]) & (ids["event"][:-1] == ids["event"][1:])
            collided = ids[np.r_[False, repeated] | np.r_[repeated, False]] if len(ids) else ids
            for _, block in groupby(collided, key=lambda r: (r["run"], r["event"])):
                group = list(block)
                stats["colliding_keys"] += 1
                stats["colliding_rows"] += len(group)
                stats["cross_split_keys"] += len({r["split"] for r in group}) > 1
                if len({fingerprint(r["source"], r["row"]) for r in group}) != len(group):
                    raise ValueError(f"Run/event-number collision has repeated raw H5 lepton payload in {name}")
            counts[name] = dict(stats)
    return counts


def verify_unique_q1_source_ranges(index):
    # Class Parquets have independent global exporter offsets; offsets need
    # only be unique within each input class, not across signal/background.
    for name in CLASSES:
        ranges = sorted((record["offset"], record["offset"] + record["rows"])
                        for record in index.values() if record["name"] == name)
        if not ranges or any(prev[1] > curr[0] for prev, curr in zip(ranges, ranges[1:])):
            raise ValueError(f"Historical Q1 exporter source offsets overlap in {name}")


def materialize(paths, partial, index, selected):
    seen = {source: np.zeros(len(rows["h5_row"]), dtype=bool) for source, rows in selected.items()}
    split_counts, shard_sha = {}, {}
    for (split, name), shards in paths.items():
        label = CLASSES[name][0]
        written = 0
        for path in shards:
            handle = pq.ParquetFile(path)
            relative = (Path(split) / name / path.name).as_posix()
            out = partial / relative
            writer = None
            try:
                for batch in handle.iter_batches(batch_size=4096):
                    sources = np.asarray(batch["source_file"].to_pylist())
                    events = batch["event_index"].to_numpy()
                    keep = np.zeros(len(events), dtype=bool)
                    extras = {key: np.empty(len(events), dtype=dtype) for key, dtype in {
                        "h5_row": "<i8", "run_number": "<i8", "event_number": "<i8", "m4l": "<f8", "channel": "U5",
                    }.items()}
                    for source in np.unique(sources):
                        if index[source]["name"] != name:
                            raise ValueError(f"Source changed class: {source}")
                        pos = np.flatnonzero(sources == source)
                        rows = selected[source]
                        local = np.searchsorted(rows["event_index"], events[pos])
                        in_range = local < len(rows["event_index"])
                        pos, local = pos[in_range], local[in_range]
                        match = rows["event_index"][local] == events[pos]
                        pos, local = pos[match], local[match]
                        if (np.any(rows["split"][local] != SPLITS.index(split)) or len(np.unique(local)) != len(local)
                                or np.any(seen[source][local])):
                            raise ValueError(f"Selected Q1 identity repeated or split changed: {source}")
                        seen[source][local] = True
                        keep[pos] = True
                        for key, value in extras.items():
                            value[pos] = rows[key][local]
                    if not keep.any():
                        continue
                    filtered = pa.Table.from_batches([batch.filter(pa.array(keep))])
                    col = filtered.schema.get_field_index("label")
                    filtered = filtered.set_column(col, filtered.schema.field(col),
                        pa.array(np.full(filtered.num_rows, label, dtype=np.int64)))
                    for key, values in extras.items():
                        if key in filtered.schema.names:
                            raise ValueError(f"Candidate metadata would overwrite input column {key}")
                        filtered = filtered.append_column(key, pa.array(values[keep]))
                    if writer is None:
                        out.parent.mkdir(parents=True, exist_ok=True)
                        writer = pq.ParquetWriter(out, filtered.schema, compression="zstd")
                    writer.write_table(filtered, row_group_size=128)
                    written += filtered.num_rows
            finally:
                if writer is not None:
                    writer.close()
            if out.exists():
                shard_sha[relative] = sha256(out)
        split_counts.setdefault(split, {})[name] = written
    if not all(s.all() for s in seen.values()):
        raise ValueError("Some H5-selected events are missing from original Q1 materialization")
    if any(not all(c.values()) for c in split_counts.values()):
        raise ValueError("Selected Q1 requires both classes in every split")
    for c in split_counts.values():
        c["total"] = sum(c.values())
    return split_counts, shard_sha


def prepare(h5_dir, inventory_path, q1_dir, output_dir, *, expected_q1_sha=LEGACY_Q1_SOURCE_SHA,
            expected_original_sha=SOURCE_SHA):
    h5_dir, inventory_path, q1, output_dir = [Path(p).resolve() for p in (h5_dir, inventory_path, q1_dir, output_dir)]
    inventory = json.loads(inventory_path.read_text())
    if inventory["source_manifest_sha256"] != expected_original_sha:
        raise ValueError("Original H5 inventory is not bound to the original canonical Q8 cohort")
    paths, q1_sha = q1_input(q1, expected_q1_sha)
    index, q1_shards = q1_index(q1, paths)
    verify_unique_q1_source_ranges(index)
    selected, source_records, totals = {}, {}, defaultdict(Counter)
    for name, info in sorted(index.items()):
        if Path(name).name != name or name not in inventory["h5_sources"]:
            raise ValueError(f"Unexpected historical Q1 H5 source: {name}")
        rows, flow, _ = select_h5(h5_dir / name, info, inventory["h5_sources"][name])
        selected[name] = rows
        totals[info["name"]].update(flow)
        source_records[name] = dict(**inventory["h5_sources"][name],
            original_q1_event_index_offset=info["offset"], class_name=info["name"], cutflow=flow,
            selected_splits={split: int(np.count_nonzero(rows["split"] == i)) for i, split in enumerate(SPLITS)},
            channels={channel: int(np.count_nonzero(rows["channel"] == channel)) for channel in CHANNELS})
        print(f"{name}: {flow['hzz_core']:,} / {flow['h5_entries']:,} {info['name']} events", flush=True)
    event_number_collisions = audit_selected_event_numbers(h5_dir, index, selected)
    target, partial = new_output(output_dir, h5_dir, q1)
    counts, shards = materialize(paths, partial, index, selected)
    files_by_split = {split: sorted(name for name, rec in source_records.items() if rec["selected_splits"][split]) for split in SPLITS}
    file_overlap = sum(map(len, files_by_split.values())) != len(set().union(*map(set, files_by_split.values())))
    classes = [dict(name=name, label=label, dsids=[dsid], available_rows=totals[name]["h5_entries"],
                    selected_rows=totals[name]["hzz_core"]) for name, (label, dsid, _, _) in CLASSES.items()]
    manifest = dict(format_version=1, task="ATLAS ggF HZZ vs continuum ZZ, reconstructed CMS-inspired HZZ core",
        selection_id=SELECTION_ID, selection=SELECTION, split_method="inherited from full unbalanced original Q1 parent",
        n_classes=2, classes=classes, token_shape=[256, 1], split_counts=counts, seed=42,
        identity_overlap_detected=False, identity_columns=["source_file", "event_index"],
        selected_event_number_collision_audit=event_number_collisions,
        file_overlap_detected=file_overlap, source_files_by_split=files_by_split,
        selected_identity_count=sum(c["total"] for c in counts.values()), no_class_undersampling=True,
        recommended_cross_entropy_weights={name: counts["train"]["total"] / (2 * counts["train"][name]) for name in CLASSES},
        source_q1_manifest_sha256=q1_sha, source_h5_inventory_sha256=sha256(inventory_path),
        original_q8_manifest_sha256=expected_original_sha, original_q1_shards=q1_shards,
        candidate_code_sha256=sha256(Path(__file__).with_name("audit_cms_hzz_transfer_v1.py")),
        candidate_metadata=["h5_row", "run_number", "event_number", "m4l", "channel"],
        cutflow={name: dict(total) for name, total in totals.items()}, h5_sources=source_records, shards=shards)
    names = sorted(selected)
    fields = ("h5_row", "run_number", "event_number", "event_index", "m4l", "channel", "split")
    with (partial / "membership.npz").open("xb") as f:
        np.savez_compressed(f, sources=np.asarray(names),
            source_id=np.repeat(np.arange(len(names), dtype="<u4"), [len(selected[name]["h5_row"]) for name in names]),
            labels=np.concatenate([np.full(len(selected[name]["h5_row"]), CLASSES[index[name]["name"]][0], dtype="u1") for name in names]),
            **{key: np.concatenate([selected[name][key] for name in names]) for key in fields})
    manifest["membership_sha256"] = sha256(partial / "membership.npz")
    write_json(partial / "manifest.json", manifest)
    partial.rename(target)
    return manifest


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("h5-dir", "input-inventory", "q1-dir", "output-dir"):
        p.add_argument(f"--{key}", type=Path, required=True)
    a = p.parse_args(argv)
    manifest = prepare(a.h5_dir, a.input_inventory, a.q1_dir, a.output_dir)
    print(json.dumps({key: manifest[key] for key in ("selection_id", "split_counts", "cutflow", "membership_sha256")}, indent=2))


if __name__ == "__main__":
    main()
