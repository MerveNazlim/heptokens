#!/usr/bin/env python3
"""Paired full-Q8 event closure with explicit file, event-row and object-slot IDs.

The per-object reconstruction caches are intentionally not joined: they do not
store event boundaries. This workflow freezes a common test-event subset, makes
separate aligned caches, then renders the same paper triptychs without inference.
"""

from __future__ import annotations

import argparse
import csv
import logging
import time
from functools import partial
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

import paper_tokenizer_reconstruction as reco
from paper_plot_style import paper_style, save_figure
from heptokens.utils import event_observables as physics

log = logging.getLogger(__name__)
VERSION = "paper-full-q8-event-closure-v1"
OBJECTS = ("electrons", "muons", "jets")
OBSERVABLES = (
    ("m4l", r"$m_{4\ell}$", "GeV"),
    ("delta_r_lj", r"$\Delta R(\ell_1,j_1)$", ""),
    ("HT", r"$H_T$", "GeV"),
)


def load_plan(directory):
    plan = reco.read_sealed(directory / "plan.json", "plan_id")
    if plan["version"] != VERSION:
        raise ValueError(
            "Not an event-closure plan; do not use the object-reconstruction output directory"
        )
    return plan


def source_plan(plan):
    reco.capacity.verify_file(plan["source_plan"])
    return reco.load_plan(Path(plan["source_plan"]["path"]).parent)


def event_selection(parent, sample, max_events):
    partition = parent["test_partition"]
    entries = [entry for entry in partition["selected"] if entry["sample"] == sample]
    if [entry["path"] for entry in entries] != [r["path"] for r in parent["samples"][sample]]:
        raise ValueError("Frozen test file order does not match the parent plan")
    ids = []
    remaining = max_events
    with np.load(partition["membership"]["path"], allow_pickle=False) as archive:
        for file_index, entry in enumerate(entries):
            rows = archive[entry["key"]]
            if (
                rows.ndim != 1
                or rows.dtype.kind not in "iu"
                or len(rows) != entry["test_events"]
                or np.any(rows < 0)
                or np.any(rows >= entry["n_events"])
                or np.any(np.diff(rows) <= 0)
            ):
                raise ValueError(f"Invalid parent test membership: {entry['path']}")
            rows = rows[:remaining]
            ids.append(np.column_stack((np.full(len(rows), file_index, dtype=np.int64), rows)))
            remaining -= len(rows)
            if remaining == 0:
                break
    result = np.concatenate(ids).astype(np.int64)
    if len(result) == 0:
        raise ValueError(f"No test events available for {sample}")
    return result


def save_npz(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def build_plan(args):
    if (args.output_dir / "plan.json").exists():
        raise FileExistsError("Event plan already frozen; use evaluate/summarize/plot")
    parent = reco.load_plan(args.reconstruction_dir)
    if parent.get("evaluation_protocol") != "saved-test":
        raise ValueError(
            "Event closure currently requires the explicit saved-test reconstruction plan"
        )
    from paper_tokenizer_test_split import verify_partition

    verify_partition(parent)
    runs = []
    for obj in OBJECTS:
        matches = [run for run in parent["runs"] if run["object"] == obj]
        if len(matches) != 1:
            raise ValueError(f"Missing/ambiguous {obj} run")
        run = matches[0]
        if (run["q"], run["k"], run["d"]) != reco.capacity.SELECTED_CONFIGURATIONS[obj]:
            raise ValueError(f"Wrong selected full tokenizer: {obj}")
        for key in ("config", "completion", "checkpoint", "preprocessor", "preprocessor_metadata"):
            reco.capacity.verify_file(run[key])
        if any(run["feature_names"].count(name) != 1 for name in ("pt", "eta", "phi")):
            raise ValueError(f"Need exactly one pt/eta/phi feature in saved {obj} configuration")
        runs.append(run)
    ids = {sample: event_selection(parent, sample, args.max_events) for sample in reco.DOMAINS}
    membership = args.output_dir / "event_membership.npz"
    save_npz(membership, **ids)
    plan = reco.seal(
        {
            "version": VERSION,
            "source_plan": reco.capacity.file_record(
                args.reconstruction_dir / "plan.json", content_hash=True
            ),
            "membership": reco.capacity.file_record(membership, content_hash=True),
            "runs": runs,
            "counts": {sample: len(rows) for sample, rows in ids.items()},
            "settings": {
                "max_events_per_domain": args.max_events,
                "batch_size": parent["evaluation"]["batch_size"],
                "jet_pt_threshold_gev": args.jet_pt_threshold,
                "m4l_selection": "four_leading_original_electrons_or_muons",
                "delta_r_pairing": "leading_original_lepton_and_jet",
                "input_momentum_unit": parent["input_momentum_unit"],
                "checkpoint_name": parent["checkpoint_name"],
            },
            "statistics": parent["statistics"],
            "holdout_note": parent["holdout_note"],
            "implementation": [
                reco.capacity.file_record(p, content_hash=True)
                for p in (__file__, physics.__file__)
            ],
        },
        "plan_id",
    )
    reco.write_json(args.output_dir / "plan.json", plan)
    for sample, rows in ids.items():
        log.info(
            "%s: %s common test events for electrons, muons and jets", sample, f"{len(rows):,}"
        )
    log.info("Frozen event audit: %s; no inference", args.output_dir / "plan.json")
    log.info(
        "m4l: four leading original e/mu; Delta R: original leading lepton/jet; jet threshold: %g GeV",
        args.jet_pt_threshold,
    )


def membership_ids(plan, sample):
    reco.capacity.verify_file(plan["membership"])
    with np.load(plan["membership"]["path"], allow_pickle=False) as archive:
        ids = archive[sample]
    if ids.shape != (plan["counts"][sample], 2) or ids.dtype.kind not in "iu":
        raise ValueError("Malformed event membership")
    return ids


def identified_loader(cfg, parent, ids, sample, transforms, batch_size):
    import torch
    from paper_tokenizer_test_split import SavedTestEventDataset
    from heptokens.data.collation import collate_and_transform

    dm = OmegaConf.to_container(cfg.datamodule, resolve=True)
    entries = [e for e in parent["test_partition"]["selected"] if e["sample"] == sample]

    class IdentifiedEvents(torch.utils.data.IterableDataset):
        def __iter__(self):
            for file_index, entry in enumerate(entries):
                rows = ids[ids[:, 0] == file_index, 1]
                if not len(rows):
                    continue
                dataset = SavedTestEventDataset([{**entry, "indices": rows}], dm)
                for row, values in zip(rows, dataset, strict=True):
                    yield {
                        **values,
                        "source_file_index": np.int64(file_index),
                        "event_row": np.int64(row),
                    }

    return torch.utils.data.DataLoader(
        IdentifiedEvents(),
        batch_size=batch_size,
        num_workers=0,
        drop_last=False,
        collate_fn=partial(collate_and_transform, transforms=transforms),
    )


def cache_path(directory, plan, obj, sample):
    return directory / "cache" / plan["plan_id"] / obj / f"{sample}.npz"


def validate_arrays(arrays, expected_ids):
    if not np.array_equal(arrays["event_ids"], expected_ids):
        raise ValueError("Event identity/order mismatch; refusing to combine object caches")
    offsets, slots = arrays["offsets"], arrays["slot_ids"]
    x, y = arrays["original"], arrays["reconstruction"]
    if (
        x.ndim != 2
        or x.shape[1] != 3
        or x.shape != y.shape
        or offsets.dtype.kind not in "iu"
        or offsets.shape != (len(expected_ids) + 1,)
        or offsets[0] != 0
        or offsets[-1] != len(x)
        or np.any(np.diff(offsets) < 0)
        or slots.dtype.kind not in "iu"
        or slots.shape != (len(x),)
        or np.any(slots < 0)
    ):
        raise ValueError("Invalid packed event/object shapes")
    for first, last in zip(offsets[:-1], offsets[1:]):
        if np.any(np.diff(slots[first:last]) <= 0):
            raise ValueError("Duplicate/unordered object slots within event")


def read_cache(path, plan, run, sample, ids):
    receipt = reco.read_sealed(path.with_suffix(".json"), "receipt_id")
    if (
        receipt["plan_id"] != plan["plan_id"]
        or receipt["object"] != run["object"]
        or receipt["sample"] != sample
        or receipt["checkpoint"] != run["checkpoint"]
        or receipt["arrays"]["path"] != str(path.resolve())
    ):
        raise ValueError("Wrong event-cache provenance")
    reco.capacity.verify_file(receipt["arrays"])
    with np.load(path, allow_pickle=False) as archive:
        arrays = {
            key: archive[key]
            for key in ("event_ids", "offsets", "slot_ids", "original", "reconstruction")
        }
    validate_arrays(arrays, ids)
    return arrays, receipt


def infer(run, parent, plan, sample, ids, device):
    import torch
    import analyze_vqvae_tokenizer as diagnostics

    reco.capacity.validate_checkpoint(run)
    cfg = OmegaConf.load(run["config"]["path"])
    OmegaConf.update(
        cfg, "datamodule.transforms.preprocess.cst_fn.filename", run["preprocessor"]["path"]
    )
    transforms, inverse = diagnostics.transform_list_and_cst_fn_from_cfg(cfg)
    if inverse is None:
        raise ValueError("Missing inverse preprocessor")
    loader = identified_loader(cfg, parent, ids, sample, transforms, plan["settings"]["batch_size"])
    device = torch.device(device)
    model = diagnostics.load_analysis_model(Path(run["run_dir"]), run["checkpoint"]["path"], device)
    columns = [run["feature_names"].index(name) for name in ("pt", "eta", "phi")]
    scale = 0.001 if plan["settings"]["input_momentum_unit"] == "MeV" else 1.0
    originals, decoded, slots, counts = [], [], [], []
    seen = 0
    started = last_log = time.monotonic()
    try:
        with torch.no_grad():
            for batch in loader:
                actual_ids = np.column_stack(
                    (batch["source_file_index"].numpy(), batch["event_row"].numpy())
                )
                n = len(actual_ids)
                if not np.array_equal(actual_ids, ids[seen : seen + n]):
                    raise ValueError("Loader changed event ordering; refusing inference")
                batch = diagnostics.to_device(batch, device)
                mask = batch["mask"].bool()
                counts.append(mask.sum(dim=1).cpu().numpy())
                slots.append(torch.nonzero(mask, as_tuple=True)[1].cpu().numpy())
                if bool(mask.any()):
                    z_q, _, _ = model.encode(batch)
                    prediction = model.decode(z_q, batch)
                    x = inverse.inverse_transform(batch["csts"][mask].cpu().float().numpy())[
                        :, columns
                    ]
                    y = inverse.inverse_transform(prediction[mask].cpu().float().numpy())[
                        :, columns
                    ]
                    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
                    x[:, 0] *= scale
                    y[:, 0] *= scale
                    originals.append(x)
                    decoded.append(y)
                seen += n
                now = time.monotonic()
                if seen == n or now - last_log >= 30 or seen == len(ids):
                    log.info(
                        "%s/%s: %s/%s events, %.1f events/s",
                        run["object"],
                        sample,
                        f"{seen:,}",
                        f"{len(ids):,}",
                        seen / max(now - started, 1e-6),
                    )
                    last_log = now
        if seen != len(ids):
            raise ValueError("Loader ended before the frozen event selection")
        arrays = {
            "event_ids": ids,
            "offsets": np.r_[0, np.cumsum(np.concatenate(counts))].astype(np.int64),
            "slot_ids": np.concatenate(slots).astype(np.int64),
            "original": np.concatenate(originals) if originals else np.empty((0, 3)),
            "reconstruction": np.concatenate(decoded) if decoded else np.empty((0, 3)),
        }
        validate_arrays(arrays, ids)
        return arrays
    finally:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


def evaluate(args):
    from paper_tokenizer_test_split import verify_partition

    plan = load_plan(args.output_dir)
    parent = source_plan(plan)
    verify_partition(parent)
    for record in plan["implementation"]:
        reco.capacity.verify_file(record)
    for run in plan["runs"]:
        if run["object"] not in args.objects:
            continue
        for key in ("config", "completion", "checkpoint", "preprocessor", "preprocessor_metadata"):
            reco.capacity.verify_file(run[key])
        for sample in args.samples:
            ids = membership_ids(plan, sample)
            path = cache_path(args.output_dir, plan, run["object"], sample)
            if path.exists() or path.with_suffix(".json").exists():
                read_cache(path, plan, run, sample, ids)
                log.info("Reusing verified event cache: %s", path)
                continue
            if args.cache_only:
                raise FileNotFoundError(f"Missing event cache: {path}; inference forbidden")
            arrays = infer(run, parent, plan, sample, ids, args.device)
            save_npz(path, **arrays)
            reco.write_json(
                path.with_suffix(".json"),
                reco.seal(
                    {
                        "plan_id": plan["plan_id"],
                        "object": run["object"],
                        "sample": sample,
                        "checkpoint": run["checkpoint"],
                        "arrays": reco.capacity.file_record(path, content_hash=True),
                    },
                    "receipt_id",
                ),
            )
            log.info("Saved event-aligned cache: %s", path)


def derive(items, ids, threshold):
    for arrays in items.values():
        validate_arrays(arrays, ids)
    pairs = {name: np.full((len(ids), 2), np.nan) for name, _, _ in OBSERVABLES}
    for index in range(len(ids)):
        objects = {}
        for obj in OBJECTS:
            arrays = items[obj]
            first, last = arrays["offsets"][index : index + 2]
            objects[obj] = (arrays["original"][first:last], arrays["reconstruction"][first:last])
        for name, pair in physics.paired_observables(objects, threshold).items():
            pairs[name][index] = pair
    return pairs


def summarize(args):
    plan = load_plan(args.output_dir)
    destination = args.output_dir / "event_summary.json"
    if destination.exists():
        summary = load_summary(args.output_dir, plan)
        for source in summary["sources"]:
            reco.capacity.verify_file(source)
        log.info("Reusing event-observable summary")
        return
    missing = [
        str(path)
        for run in plan["runs"]
        for sample in reco.DOMAINS
        for path in (
            cache_path(args.output_dir, plan, run["object"], sample),
            cache_path(args.output_dir, plan, run["object"], sample).with_suffix(".json"),
        )
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing event-aligned caches; no inference performed:\n" + "\n".join(missing)
        )
    pairs, sources = {}, []
    for sample in reco.DOMAINS:
        ids = membership_ids(plan, sample)
        items = {}
        for run in plan["runs"]:
            path = cache_path(args.output_dir, plan, run["object"], sample)
            items[run["object"]], receipt = read_cache(path, plan, run, sample, ids)
            sources.extend(
                [
                    receipt["arrays"],
                    reco.capacity.file_record(path.with_suffix(".json"), content_hash=True),
                ]
            )
        pairs[sample] = derive(items, ids, plan["settings"]["jet_pt_threshold_gev"])
        del items
    settings = plan["statistics"]
    features = []
    for name, label, unit in OBSERVABLES:
        pooled = np.concatenate([p[name][:, 0] for p in pairs.values()])
        residuals = np.concatenate([p[name][:, 1] - p[name][:, 0] for p in pairs.values()])
        edges = reco.bin_edges(pooled, settings["n_bins"], settings["percentiles"])
        residual_edges = reco.bin_edges(residuals, settings["n_bins"], settings["percentiles"])
        domains = {}
        for sample, values in pairs.items():
            x, y = values[name].T
            row = reco.feature_summary(
                x, y, edges, residual_edges, settings["min_ratio_count"], name
            )
            row.update(
                {
                    "total_events": len(x),
                    "original_eligible": int(np.isfinite(x).sum()),
                    "decoded_invalid": int((np.isfinite(x) & ~np.isfinite(y)).sum()),
                }
            )
            domains[sample] = row
            log.info(
                "%s/%s: %d eligible, %d paired, %d decoded-invalid events",
                sample,
                name,
                row["original_eligible"],
                row["finite_pairs"],
                row["decoded_invalid"],
            )
        features.append(
            {
                "name": name,
                "label": label,
                "unit": unit,
                "edges": edges.tolist(),
                "residual_edges": residual_edges.tolist(),
                "domains": domains,
            }
        )
    reco.write_json(
        destination,
        reco.seal(
            {"plan_id": plan["plan_id"], "features": features, "sources": sources}, "summary_id"
        ),
    )


def load_summary(directory, plan):
    summary = reco.read_sealed(directory / "event_summary.json", "summary_id")
    if summary["plan_id"] != plan["plan_id"] or [f["name"] for f in summary["features"]] != [
        o[0] for o in OBSERVABLES
    ]:
        raise ValueError("Wrong event summary provenance")
    return summary


def caption(plan):
    return (
        "# Reconstruction of derived event observables\n\n"
        f"Full Q8/cb2048/dim8 electron, muon and jet tokenizers; {plan['settings']['checkpoint_name']}. "
        f"{plan['holdout_note']}\n\n"
        f"Evaluated events: simulation {plan['counts']['mc']:,}; collision data {plan['counts']['data']:,}. "
        "This is the file/row-ordered prefix of the frozen selected test files, capped in events, "
        "not a uniform sample of the complete test partition. The same (source file, event row) "
        "is used across all three objects, retaining original object slots and saved masks/caps. "
        "This tests closure through object tokenizers before event-sequence truncation; "
        "it is not a validation of an existing Parquet's vocabulary or sequence assembly.\n\n"
        "m4l: invariant mass of the four highest-original-pT valid electrons/muons, using "
        "fixed electron/muon rest masses and identical object identities after decoding. "
        "No opposite-sign/same-flavour pairing, lepton ID/isolation or HZZ analysis cuts are added. "
        "Delta R(l1,j1): the highest-original-pT valid electron/muon and highest-original-pT "
        f"valid jet with original pT > {plan['settings']['jet_pt_threshold_gev']:g} GeV. "
        "The same pair is retained after decoding, without re-ranking or reapplying the jet cut; "
        "Delta R = sqrt((Delta eta)^2 + (wrapped Delta phi)^2), with Delta phi in [-pi,pi]. "
        f"HT: sum of valid jet pT above {plan['settings']['jet_pt_threshold_gev']:g} GeV, "
        "with the threshold applied independently to original and decoded jets, including "
        "threshold migration. Zero-jet events have HT=0. Events without four leptons or a "
        "lepton/jet pair are ineligible for those observables, not assigned zero. "
        "Invalid decoded selected objects are not replaced by other objects; excluded pair "
        "counts are exported. Original means detector-reconstructed input, not generator truth.\n\n"
        "Each row contains an original/decoded distribution with a decoded/original bin-count "
        "ratio, a decoded-minus-original residual, and paired density with y=x and equal axes. "
        "Mass and HT are in GeV; Delta R is dimensionless. Histograms use common MC/data edges "
        "from original 0.5--99.5 percentiles; residual ranges use paired residual percentiles. "
        "Histograms are normalized by the full finite paired-event count, including events "
        "outside displayed ranges, not independently within the window. "
        f"Ratio bins require {plan['statistics']['min_ratio_count']} original entries. "
        "CSV reports eligible/paired/decoded-invalid events, out-of-window counts, median "
        "residual, median absolute residual and residual IQR. No uncertainties are inferred.\n"
    )


def plot(args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plan = load_plan(args.output_dir)
    summary = load_summary(args.output_dir, plan)
    destination = args.output_dir / "figures"
    with paper_style():
        for sample in reco.DOMAINS:
            fig = reco.render_triptych_page(
                summary["features"],
                "event",
                sample,
                1,
                1,
                log_y=args.y_scale == "log",
                title="Event observables",
                count_label="events",
            )
            save_figure(fig, destination / f"event_observables_{sample}_{args.y_scale}")
            plt.close(fig)
    rows = []
    for feature in summary["features"]:
        for sample, values in feature["domains"].items():
            rows.append(
                {
                    "observable": feature["name"],
                    "sample": sample,
                    "unit": feature["unit"],
                    **{
                        key: value
                        for key, value in values.items()
                        if not isinstance(value, list) and key != "total_objects"
                    },
                }
            )
    with (destination / "event_observables_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (destination / "event_observables_caption.md").write_text(caption(plan))
    log.info("Saved event triptychs: %s; summaries only, no inference", destination)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("plan")
    audit.add_argument("--reconstruction-dir", type=Path, required=True)
    audit.add_argument(
        "--max-events",
        type=int,
        default=50000,
        help="Per MC/data domain, before observable eligibility cuts",
    )
    audit.add_argument(
        "--jet-pt-threshold",
        type=float,
        default=20.0,
        help="GeV; original Delta R jet selection and independent HT thresholds",
    )
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--objects", choices=OBJECTS, nargs="+", default=list(OBJECTS))
    evaluation.add_argument(
        "--samples", choices=tuple(reco.DOMAINS), nargs="+", default=list(reco.DOMAINS)
    )
    evaluation.add_argument("--device", default="cpu")
    evaluation.add_argument("--cache-only", action="store_true")
    summary = commands.add_parser("summarize")
    plotting = commands.add_parser("plot")
    plotting.add_argument("--y-scale", choices=("log", "linear"), default="log")
    for cmd in (audit, evaluation, summary, plotting):
        cmd.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "plan" and (
        args.max_events <= 0 or not np.isfinite(args.jet_pt_threshold) or args.jet_pt_threshold < 0
    ):
        parser.error("Need a positive event cap and finite nonnegative jet threshold")
    return args


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    {"plan": build_plan, "evaluate": evaluate, "summarize": summarize, "plot": plot}[args.command](
        args
    )


if __name__ == "__main__":
    main()
