#!/usr/bin/env python3
"""Dedicated HZZ four-lepton mass triptych, using full-Q8 saved-test events.

The inclusive event-observable plan/caches are never modified. Signal files are
identified by DSID within the full training inventory, not by filename prefixes.
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

import paper_event_reconstruction as event
import paper_tokenizer_test_split as split
from paper_plot_style import paper_style, save_figure

reco = event.reco
log = logging.getLogger(__name__)
VERSION = "paper-full-q8-hzz-mass-v1"
OBJECTS = ("electrons", "muons")


def signal_files(inputs, requested):
    """Inspect metadata, validating available per-event DSIDs for selected files."""
    import h5py

    found, unknown = [], []
    for item in inputs:
        record = item["file"]
        reco.capacity.verify_file(record)
        with h5py.File(record["path"], "r") as handle:
            attrs = handle["metadata"].attrs if "metadata" in handle else {}
            raw = np.asarray(attrs.get("dsid", 0)).reshape(-1)
            if raw.size != 1:
                raise ValueError(f"Non-scalar DSID metadata: {record['path']}")
            dsid = int(raw[0])
            if dsid > 0 and dsid not in requested:
                continue
            node = handle.get("atlas/event")
            has_channel = (isinstance(node, h5py.Group) and "mcChannelNumber" in node) or (
                isinstance(node, h5py.Dataset) and "mcChannelNumber" in (node.dtype.names or ())
            )
            if has_channel:
                channels = set()
                for start in range(0, item["n_events"], 65536):
                    values = split.read_h5_slice(
                        handle,
                        "atlas/event/mcChannelNumber",
                        slice(start, min(start + 65536, item["n_events"])),
                    )
                    channels.update(np.unique(values).tolist())
                if dsid in requested or channels.intersection(requested):
                    if len(channels) != 1 or (dsid > 0 and channels != {dsid}):
                        raise ValueError(f"Mixed/inconsistent signal DSIDs: {record['path']}")
                    value = next(iter(channels))
                    if not np.isfinite(value) or value != int(value):
                        raise ValueError(f"Invalid signal DSID: {record['path']}")
                    dsid = int(value)
            if dsid in requested:
                process = attrs.get("process", "")
                if isinstance(process, bytes):
                    process = process.decode("utf-8")
                found.append({"file": record, "dsid": dsid, "process": str(process)})
            elif dsid <= 0 and not has_channel:
                unknown.append(record["path"])
    missing = set(requested) - {row["dsid"] for row in found}
    if missing:
        raise ValueError(
            f"No verified signal files for DSIDs {sorted(missing)} in the saved full inventory. "
            "No external files or training-event fallback selected."
        )
    return found, unknown


def eligible_selection(plan, max_events):
    """Apply the four-lepton requirement to test rows BEFORE the event cap."""
    configs, columns = {}, {}
    for run in plan["runs"]:
        obj = run["object"]
        configs[obj] = OmegaConf.to_container(
            OmegaConf.load(run["config"]["path"]).datamodule, resolve=True
        )
        columns[obj] = [run["feature_names"].index(k) for k in ("pt", "eta", "phi")]
    ids, audit = [], []
    partition = plan["test_partition"]
    with np.load(partition["membership"]["path"], allow_pickle=False) as archive:
        for file_index, entry in enumerate(partition["selected"]):
            rows = archive[entry["key"]]
            readers = [
                iter(split.SavedTestEventDataset([{**entry, "indices": rows}], configs[obj]))
                for obj in OBJECTS
            ]
            examined = accepted = 0
            try:
                for row, *objects in zip(rows, *readers, strict=True):
                    examined += 1
                    count = 0
                    for obj, values in zip(OBJECTS, objects, strict=True):
                        kinematics = values["csts"][:, columns[obj]]
                        count += int(
                            np.count_nonzero(
                                values["mask"]
                                & np.isfinite(kinematics).all(axis=1)
                                & (kinematics[:, 0] > 0)
                            )
                        )
                    if count >= 4:
                        ids.append((file_index, int(row)))
                        accepted += 1
                    if len(ids) == max_events:
                        break
            finally:
                for reader in readers:
                    reader.close()
            audit.append(
                {
                    "path": entry["path"],
                    "test_events_available": len(rows),
                    "test_events_examined": examined,
                    "four_lepton_events_selected": accepted,
                }
            )
            log.info(
                "%s: examined %s test events, selected %s with >=4 leptons",
                Path(entry["path"]).name,
                f"{examined:,}",
                f"{accepted:,}",
            )
            if len(ids) == max_events:
                break
    if not ids:
        raise ValueError("No eligible four-lepton signal test events; no plan/inference produced")
    return np.asarray(ids, dtype=np.int64), audit


def build_plan(args):
    if (args.output_dir / "plan.json").exists():
        raise FileExistsError("HZZ plan already frozen; use evaluate/summarize/plot")
    parent = reco.load_plan(args.reconstruction_dir)
    if parent.get("evaluation_protocol") != "saved-test":
        raise ValueError("A saved-test full-Q8 reconstruction plan is required")
    split.verify_partition(parent)
    runs = []
    for obj in OBJECTS:
        matches = [r for r in parent["runs"] if r["object"] == obj]
        if len(matches) != 1:
            raise ValueError(f"Missing/ambiguous {obj} run")
        run = matches[0]
        if (run["q"], run["k"], run["d"]) != reco.capacity.SELECTED_CONFIGURATIONS[obj]:
            raise ValueError(f"Wrong selected full tokenizer: {obj}")
        for key in ("config", "completion", "checkpoint", "preprocessor", "preprocessor_metadata"):
            reco.capacity.verify_file(run[key])
        if any(run["feature_names"].count(k) != 1 for k in ("pt", "eta", "phi")):
            raise ValueError(f"Need exactly one pt/eta/phi in saved {obj} features")
        runs.append(run)
    files, unknown = signal_files(parent["test_partition"]["inputs"], args.signal_dsids)
    for dsid in args.signal_dsids:
        log.info(
            "Signal DSID %d: %d files in full inventory",
            dsid,
            sum(f["dsid"] == dsid for f in files),
        )
    samples = {"mc": [f["file"] for f in files]}
    partition = split.freeze_partition(runs, samples, args.output_dir)
    if (
        partition["settings"] != parent["test_partition"]["settings"]
        or partition["global_split_sha256"] != parent["test_partition"]["global_split_sha256"]
    ):
        raise ValueError("Global test membership differs from parent audit; refusing signal plan")
    plan = {
        "version": VERSION,
        "source_plan": reco.capacity.file_record(
            args.reconstruction_dir / "plan.json", content_hash=True
        ),
        "runs": runs,
        "test_partition": partition,
        "samples": samples,
        "signal_files": files,
        "unidentified_files": unknown,
        "settings": {
            "signal_dsids": args.signal_dsids,
            "max_four_lepton_events": args.max_events,
            "batch_size": parent["evaluation"]["batch_size"],
            "input_momentum_unit": parent["input_momentum_unit"],
            "checkpoint_name": parent["checkpoint_name"],
        },
        "statistics": parent["statistics"],
        "holdout_note": parent["holdout_note"],
        "implementation": [
            reco.capacity.file_record(p, content_hash=True)
            for p in (__file__, event.__file__, event.physics.__file__)
        ],
    }
    ids, audit = eligible_selection(plan, args.max_events)
    membership = args.output_dir / "event_membership.npz"
    event.save_npz(membership, mc=ids)
    plan.update(
        {
            "membership": reco.capacity.file_record(membership, content_hash=True),
            "counts": {"mc": len(ids)},
            "selection_audit": audit,
        }
    )
    reco.write_json(args.output_dir / "plan.json", reco.seal(plan, "plan_id"))
    log.info(
        "Frozen %s HZZ four-lepton test events; no inference. Inclusive HT/Delta R unchanged.",
        f"{len(ids):,}",
    )


def load_plan(directory):
    plan = reco.read_sealed(directory / "plan.json", "plan_id")
    if plan["version"] != VERSION:
        raise ValueError("Not a dedicated HZZ mass plan")
    return plan


def evaluate(args):
    plan = load_plan(args.output_dir)
    reco.capacity.verify_file(plan["source_plan"])
    split.verify_partition(plan)
    for record in plan["implementation"]:
        reco.capacity.verify_file(record)
    ids = event.membership_ids(plan, "mc")
    for run in plan["runs"]:
        for key in ("config", "completion", "checkpoint", "preprocessor", "preprocessor_metadata"):
            reco.capacity.verify_file(run[key])
        path = event.cache_path(args.output_dir, plan, run["object"], "mc")
        if path.exists() or path.with_suffix(".json").exists():
            event.read_cache(path, plan, run, "mc", ids)
            log.info("Reusing verified HZZ cache: %s", path)
            continue
        if args.cache_only:
            raise FileNotFoundError(f"Missing cache: {path}; inference forbidden")
        arrays = event.infer(run, plan, plan, "mc", ids, args.device)
        event.save_npz(path, **arrays)
        reco.write_json(
            path.with_suffix(".json"),
            reco.seal(
                {
                    "plan_id": plan["plan_id"],
                    "object": run["object"],
                    "sample": "mc",
                    "checkpoint": run["checkpoint"],
                    "arrays": reco.capacity.file_record(path, content_hash=True),
                },
                "receipt_id",
            ),
        )


def mass_pairs(items, ids):
    for obj in OBJECTS:
        event.validate_arrays(items[obj], ids)
    pairs = np.full((len(ids), 2), np.nan)
    for index in range(len(ids)):
        objects = {"jets": (np.empty((0, 3)), np.empty((0, 3)))}
        for obj in OBJECTS:
            arrays = items[obj]
            first, last = arrays["offsets"][index : index + 2]
            objects[obj] = (arrays["original"][first:last], arrays["reconstruction"][first:last])
        pairs[index] = event.physics.paired_observables(objects)["m4l"]
    return pairs


def load_summary(directory, plan):
    summary = reco.read_sealed(directory / "hzz_summary.json", "summary_id")
    if summary["plan_id"] != plan["plan_id"]:
        raise ValueError("Wrong HZZ summary provenance")
    return summary


def summarize(args):
    plan = load_plan(args.output_dir)
    destination = args.output_dir / "hzz_summary.json"
    if destination.exists():
        for source in load_summary(args.output_dir, plan)["sources"]:
            reco.capacity.verify_file(source)
        log.info("Reusing HZZ summary")
        return
    ids = event.membership_ids(plan, "mc")
    items, sources = {}, []
    for run in plan["runs"]:
        path = event.cache_path(args.output_dir, plan, run["object"], "mc")
        items[run["object"]], receipt = event.read_cache(path, plan, run, "mc", ids)
        sources.extend(
            [
                receipt["arrays"],
                reco.capacity.file_record(path.with_suffix(".json"), content_hash=True),
            ]
        )
    x, y = mass_pairs(items, ids).T
    settings = plan["statistics"]
    edges = reco.bin_edges(x, settings["n_bins"], settings["percentiles"])
    residual_edges = reco.bin_edges(y - x, settings["n_bins"], settings["percentiles"])
    row = reco.feature_summary(x, y, edges, residual_edges, settings["min_ratio_count"], "m4l")
    row.update(
        {
            "total_events": len(x),
            "original_eligible": int(np.isfinite(x).sum()),
            "decoded_invalid": int((np.isfinite(x) & ~np.isfinite(y)).sum()),
        }
    )
    log.info(
        "HZZ mass: %d selected, %d eligible, %d paired, %d decoded-invalid events",
        len(x),
        row["original_eligible"],
        row["finite_pairs"],
        row["decoded_invalid"],
    )
    if row["finite_pairs"] == 0:
        raise ValueError("No finite paired HZZ masses; refusing to produce an empty mass plot")
    reco.write_json(
        destination,
        reco.seal(
            {
                "plan_id": plan["plan_id"],
                "sources": sources,
                "features": [
                    {
                        "name": "m4l",
                        "label": r"$m_{4\ell}$",
                        "unit": "GeV",
                        "edges": edges.tolist(),
                        "residual_edges": residual_edges.tolist(),
                        "domains": {"mc": row},
                    }
                ],
            },
            "summary_id",
        ),
    )


def caption(plan, row):
    return (
        "# HZZ four-lepton mass reconstruction\n\n"
        f"Simulation only; signal DSIDs: {plan['settings']['signal_dsids']}. "
        "DSIDs select signal processes, not generator-matched leptons. "
        "Full Q8/cb2048/dim8 electron and muon tokenizers, unchanged joblibs and "
        f"{plan['settings']['checkpoint_name']}. {plan['holdout_note']}\n\n"
        "Signal files are selected from the saved full input inventory. Test membership "
        "is assigned globally before signal selection and its digest must match the parent audit. "
        "The event cap is applied after requiring >=4 valid original electrons/muons, "
        "in file/row order; this is not a uniform or process-balanced sample. "
        "Saved object masks/caps and cleaning are retained. The mass uses the four highest "
        "original-pT leptons and the same identities after decoding, with fixed lepton rest masses. "
        "No SFOS pairing, ID/isolation, mass window or other HZZ analysis cuts are added. "
        "Original is detector-reconstructed, not generator truth. Invalid decoded selected "
        "leptons are not replaced. Inclusive HT and Delta R samples are unchanged.\n\n"
        f"Selected: {row['total_events']:,}; eligible: {row['original_eligible']:,}; "
        f"finite pairs: {row['finite_pairs']:,}; decoded-invalid: {row['decoded_invalid']:,}. "
        "The triptych shows distributions with a decoded/original bin-count ratio, "
        "decoded-minus-original mass residual in GeV, and paired density with y=x. "
        f"Display ranges use {plan['statistics']['percentiles']} percentiles; "
        "histograms are normalized by all finite paired events, including those outside "
        "the display window. "
        f"Ratio bins require {plan['statistics']['min_ratio_count']} original entries. "
        "No uncertainty bands are inferred. Metrics include out-of-window counts. "
        "This tests object-tokenizer closure, not downstream event-model performance.\n"
    )


def plot(args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plan = load_plan(args.output_dir)
    summary = load_summary(args.output_dir, plan)
    destination = args.output_dir / "figures"
    with paper_style():
        fig = reco.render_triptych_page(
            summary["features"],
            "event",
            "mc",
            1,
            1,
            log_y=args.y_scale == "log",
            title=r"$H\to ZZ^{(*)}\to4\ell$",
            count_label="events",
        )
        save_figure(fig, destination / f"hzz_m4l_{args.y_scale}")
        plt.close(fig)
    row = summary["features"][0]["domains"]["mc"]
    metrics = {k: v for k, v in row.items() if not isinstance(v, list) and k != "total_objects"}
    with (destination / "hzz_m4l_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics))
        writer.writeheader()
        writer.writerow(metrics)
    (destination / "hzz_m4l_caption.md").write_text(caption(plan, row))
    log.info("Saved HZZ mass triptych: %s; no inference", destination)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("plan")
    audit.add_argument("--reconstruction-dir", type=Path, required=True)
    audit.add_argument(
        "--signal-dsids",
        nargs="+",
        type=int,
        default=[345060],
        help="HZZ MC DSIDs; default is the existing ggF signal selection",
    )
    audit.add_argument(
        "--max-events",
        type=int,
        default=50000,
        help="Cap after requiring four original leptons in signal TEST events",
    )
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--device", default="cpu")
    evaluation.add_argument("--cache-only", action="store_true")
    summary = commands.add_parser("summarize")
    plotting = commands.add_parser("plot")
    plotting.add_argument("--y-scale", choices=("log", "linear"), default="log")
    for command in (audit, evaluation, summary, plotting):
        command.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "plan" and (
        args.max_events <= 0
        or any(d <= 0 for d in args.signal_dsids)
        or len(set(args.signal_dsids)) != len(args.signal_dsids)
    ):
        parser.error("Need a positive event cap and unique positive signal DSIDs")
    return args


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    {"plan": build_plan, "evaluate": evaluate, "summarize": summarize, "plot": plot}[args.command](
        args
    )


if __name__ == "__main__":
    main()
