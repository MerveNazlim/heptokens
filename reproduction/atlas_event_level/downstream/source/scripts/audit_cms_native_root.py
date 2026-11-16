#!/usr/bin/env python3
"""Read-only audit of proposed native CMS inputs and optional MC pilot cutflow."""

import argparse
import json
from collections import Counter
from pathlib import Path

import awkward as ak
import numpy as np
import uproot


FEATURES = {
    "Electron": (
        "pt", "eta", "phi", "charge", "mvaFall17V2Iso", "cutBased",
        "miniPFRelIso_chg", "pfRelIso03_all",
    ),
    "Muon": (
        "pt", "eta", "phi", "charge", "mediumId", "tightId",
        "miniPFRelIso_chg", "pfRelIso03_all",
    ),
    "Photon": (
        "pt", "eta", "phi", "cutBased", "mvaID", "pfRelIso03_chg", "pfRelIso03_all",
    ),
    "Tau": (
        "pt", "eta", "phi", "charge", "decayMode", "rawDeepTau2017v2p1VSjet",
        "idDeepTau2017v2p1VSe", "idDeepTau2017v2p1VSmu",
    ),
    "Jet": (
        "pt", "eta", "phi", "mass", "nConstituents", "btagDeepFlavB",
        "btagDeepFlavCvL", "btagDeepFlavCvB", "btagDeepFlavQG",
    ),
}
CATEGORIES = {"charge", "cutBased", "mediumId", "tightId", "decayMode"}
# Numerical fiducials from ATLAS_data_DAOD_PHYSLITE.37019990._000001.pool.root.1.h5
# in the saved 20261012-064959-2272289 review; not a detector-equivalence claim.
MC_FIDUCIALS = {"Electron": (10, 2.47), "Muon": (10, 2.5), "Photon": (15, 2.37),
                "Tau": (20, 2.5), "Jet": (20, 4.5)}
PT_BINS = [0, 10, 15, 20, 30, 50, 75, 100, 150, 200, 300, 500, 1000]


class MCPilotCuts:
    def __init__(self, available):
        from cms_quality_selection import policy, quality_mask
        self.mask = quality_mask
        self.policy = policy("mc")
        self.rules = {obj: self.policy["object_cuts"][obj.lower() + "s"] for obj in FEATURES}
        self.required = set(self.policy["event_filters"]) | {"Tau_charge"}
        for obj, rules in self.rules.items():
            self.required.update((f"{obj}_pt", f"{obj}_eta", *(r["branch"] for r in rules)))
        self.missing = sorted(self.required - available)
        self.report = {
            "status": "missing_selection_branches" if self.missing else "complete",
            "missing_selection_branches": self.missing,
            "policy": self.policy["name"],
            "event_filters": self.policy["event_filters"],
            "object_id_rules": self.rules,
            "fiducial_pt_min_gev_and_abs_eta_max": MC_FIDUCIALS,
            "additional_fiducial_rules": "Exclude photon [1.37,1.52] and tau (1.37,1.52) abs-eta gaps; require abs(Tau_charge)==1.",
            "scope": "Pilot fiducial and quality cuts; no object cap, overlap removal, deduplication or split.",
            "events_after_event_filters": None if self.missing else 0,
            "pt_hist_lower_edges_gev": PT_BINS,
            "collections": {},
        }
        self.pt = {obj: [] for obj in FEATURES}
        for obj in FEATURES:
            self.report["collections"][obj] = {
                "objects_after_event_filters": None if self.missing else 0,
                "objects_after_event_and_object_cuts": None if self.missing else 0,
                "events_with_object_after_cuts": None if self.missing else 0,
                "objects_passing_cuts_and_all_provisional_feature_checks": None,
                "pt_after_cuts_gev": None,
            }

    def event_mask(self, chunk, size):
        if self.missing:
            return None
        keep = np.ones(size, dtype=bool)
        for branch in self.policy["event_filters"]:
            keep &= self.mask(ak.to_numpy(chunk[branch]), dict(branch=branch, op="eq", value=1, maximum=1))
        self.report["events_after_event_filters"] += int(keep.sum())
        return keep

    def object_mask(self, obj, chunk, sizes, event_keep):
        pt, eta = chunk[f"{obj}_pt"], chunk[f"{obj}_eta"]
        pt_min, eta_max = MC_FIDUCIALS[obj]
        abs_eta = abs(eta)
        keep = np.isfinite(pt) & (pt > pt_min) & np.isfinite(eta) & (abs_eta < eta_max)
        if obj == "Photon":
            keep = keep & ((abs_eta < 1.37) | (abs_eta > 1.52))
        elif obj == "Tau":
            keep = keep & ~((abs_eta > 1.37) & (abs_eta < 1.52)) & (abs(chunk["Tau_charge"]) == 1)
        for rule in self.rules[obj]:
            if not np.array_equal(ak.to_numpy(ak.num(chunk[rule["branch"]], axis=1)), sizes):
                raise ValueError(f"{rule['branch']}: selection branch jagged lengths differ from object counts")
            keep = keep & self.mask(chunk[rule["branch"]], rule)
        return ak.fill_none(keep & ak.Array(event_keep), False)

    def collect(self, obj, chunk, sizes, event_keep, valid):
        if event_keep is None:
            return
        keep = self.object_mask(obj, chunk, sizes, event_keep)
        after = self.report["collections"][obj]
        after["objects_after_event_filters"] += int(sizes[event_keep].sum())
        after["objects_after_event_and_object_cuts"] += int(ak.sum(keep))
        after["events_with_object_after_cuts"] += int(ak.sum(ak.any(keep, axis=1)))
        if valid is not None:
            selected_valid = ak.to_numpy(ak.flatten(keep, axis=1)) & valid
            key = "objects_passing_cuts_and_all_provisional_feature_checks"
            after[key] = (after[key] or 0) + int(np.count_nonzero(selected_valid))
        self.pt[obj].append(ak.to_numpy(ak.flatten(chunk[f"{obj}_pt"][keep], axis=1)))

    def finalize(self):
        for obj, chunks in self.pt.items():
            if not chunks:
                continue
            values = np.concatenate(chunks)
            if not len(values):
                continue
            quantiles = np.quantile(values, [0., .05, .50, .95, .99, 1.])
            self.report["collections"][obj]["pt_after_cuts_gev"] = {
                **dict(zip(["min", "p05", "p50", "p95", "p99", "max"], map(float, quantiles))),
                "hist_counts": np.histogram(values, bins=PT_BINS + [np.inf])[0].tolist(),
            }
        return self.report


def categorical(field):
    return field in CATEGORIES or field.startswith("idDeepTau")


def check_values(field, values):
    finite = np.isfinite(values)
    negative = finite & (values < 0)
    suspicious = np.zeros(len(values), dtype=bool)
    if "RelIso" in field or field.startswith("btagDeepFlav"):
        suspicious = negative
    invalid_category = np.zeros(len(values), dtype=bool)
    if categorical(field):
        if field in {"mediumId", "tightId"}:
            invalid_category = finite & ~np.isin(values, [0, 1])
        else:
            invalid_category[finite] = values[finite] != np.trunc(values[finite])
    return finite, negative, suspicious, invalid_category


def audit(source, max_events=None, step_size="64 MB", mc_pilot_cuts=False):
    with uproot.open(source) as root:
        tree = root["Events"]
        available = set(tree.keys())
        entries = min(tree.num_entries, max_events) if max_events is not None else tree.num_entries
        report = {
            "source": source,
            "tree": "Events",
            "events_in_file": tree.num_entries,
            "events_scanned": 0,
            "scope": "raw ROOT; no selection, deduplication, split assignment or export",
            "validity": (
                "Provisional checks: finite, integer-like categories, boolean IDs in {0,1}, "
                "and no negative relative isolation or DeepFlavour values. Negative values "
                "are candidates for source-specific sentinel review, not established sentinels. "
                "No full era-specific semantic validation was applied."
            ),
            "collections": {},
        }
        wanted = []
        for obj, fields in FEATURES.items():
            count = "n" + obj
            wanted.extend([count, *(f"{obj}_{field}" for field in fields)])
            report["collections"][obj] = coll = {
                "count_branch": count if count in available else None,
                "raw_objects": 0 if count in available or f"{obj}_pt" in available else None,
                "events_with_objects": 0 if count in available or f"{obj}_pt" in available else None,
                "objects_passing_all_provisional_checks": 0,
                "events_with_object_passing_all_provisional_checks": 0,
                "features": {},
            }
            for field in fields:
                name = f"{obj}_{field}"
                if name not in available:
                    coll["features"][name] = {"present": False}
                    coll["objects_passing_all_provisional_checks"] = None
                    coll["events_with_object_passing_all_provisional_checks"] = None
                    continue
                coll["features"][name] = {
                    "present": True, "root_typename": tree[name].typename,
                    "values": 0, "nonfinite_or_null": 0, "negative_finite": 0,
                    "valid_zero": 0, "suspect_negative": 0, "invalid_category": 0,
                    "finite_min": None, "finite_max": None,
                }
                if categorical(field):
                    coll["features"][name]["observed_categories"] = Counter()
        cuts = MCPilotCuts(available) if mc_pilot_cuts else None
        if cuts and not cuts.missing:
            wanted.extend(sorted(cuts.required))
        expressions = list(dict.fromkeys(name for name in wanted if name in available))
        chunks = tree.iterate(
            expressions, entry_stop=entries, step_size=step_size, library="ak", how=dict
        ) if expressions else ()
        if not expressions:
            report["events_scanned"] = entries
        for chunk in chunks:
            n_events = len(next(iter(chunk.values())))
            report["events_scanned"] += n_events
            event_keep = cuts.event_mask(chunk, n_events) if cuts else None
            for obj, fields in FEATURES.items():
                coll = report["collections"][obj]
                count = coll["count_branch"]
                if count is not None:
                    sizes = ak.to_numpy(chunk[count])
                elif f"{obj}_pt" in chunk:
                    sizes = ak.to_numpy(ak.num(chunk[f"{obj}_pt"], axis=1))
                else:
                    if any(f"{obj}_{field}" in chunk for field in fields):
                        raise ValueError(f"{obj}: branches present but no {count or 'n' + obj} or pt branch")
                    continue
                if np.any(sizes < 0):
                    raise ValueError(f"{obj}: negative object multiplicity")
                n_objects = int(sizes.sum())
                coll["raw_objects"] += n_objects
                coll["events_with_objects"] += int(np.count_nonzero(sizes))
                valid = np.ones(n_objects, dtype=bool)
                for field in fields:
                    name = f"{obj}_{field}"
                    if name not in chunk:
                        continue
                    jagged = chunk[name]
                    if not np.array_equal(ak.to_numpy(ak.num(jagged, axis=1)), sizes):
                        raise ValueError(f"{name}: jagged lengths differ from {count or obj + '_pt'}")
                    flat = ak.to_numpy(ak.flatten(jagged, axis=1))
                    values = np.asarray(flat, dtype=float)
                    if np.ma.isMaskedArray(flat):
                        values = np.where(np.ma.getmaskarray(flat), np.nan, values)
                    finite, negative, suspect, invalid_category = check_values(field, values)
                    good = finite & ~suspect & ~invalid_category
                    valid &= good
                    stats = coll["features"][name]
                    stats["values"] += n_objects
                    stats["nonfinite_or_null"] += int(np.count_nonzero(~finite))
                    stats["negative_finite"] += int(np.count_nonzero(negative))
                    stats["suspect_negative"] += int(np.count_nonzero(suspect))
                    stats["invalid_category"] += int(np.count_nonzero(invalid_category))
                    stats["valid_zero"] += int(np.count_nonzero(good & (values == 0)))
                    if np.any(finite):
                        minimum, maximum = float(values[finite].min()), float(values[finite].max())
                        previous_min, previous_max = stats["finite_min"], stats["finite_max"]
                        stats["finite_min"] = minimum if previous_min is None else min(minimum, previous_min)
                        stats["finite_max"] = maximum if previous_max is None else max(maximum, previous_max)
                    if categorical(field):
                        labels, counts = np.unique(values[finite], return_counts=True)
                        stats["observed_categories"].update(
                            {format(float(label), "g"): int(n) for label, n in zip(labels, counts)}
                        )
                if coll["objects_passing_all_provisional_checks"] is not None:
                    coll["objects_passing_all_provisional_checks"] += int(np.count_nonzero(valid))
                    coll["events_with_object_passing_all_provisional_checks"] += int(
                        ak.sum(ak.any(ak.unflatten(valid, sizes.astype(np.int64)), axis=1))
                    )
                if cuts:
                    cuts.collect(obj, chunk, sizes, event_keep, valid
                                 if coll["objects_passing_all_provisional_checks"] is not None else None)
        if report["events_scanned"] != entries:
            raise RuntimeError(f"expected {entries} scanned entries, saw {report['events_scanned']}")
        report["ready_for_export"] = all(
            stats["present"] and not any(
                stats[key] for key in ("nonfinite_or_null", "suspect_negative", "invalid_category")
            )
            for coll in report["collections"].values() for stats in coll["features"].values()
        )
        if cuts:
            report["mc_pilot_cuts"] = cuts.finalize()
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Local ROOT file or uproot-compatible URL")
    parser.add_argument("--output", required=True, type=Path, help="New JSON report path")
    parser.add_argument("--max-events", type=int)
    parser.add_argument("--mc-pilot-cuts", action="store_true")
    args = parser.parse_args()
    if args.max_events is not None and args.max_events <= 0:
        parser.error("--max-events must be positive")
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    report = audit(args.source, args.max_events, mc_pilot_cuts=args.mc_pilot_cuts)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Scanned {report['events_scanned']} ROOT events; ready_for_export={report['ready_for_export']}")


if __name__ == "__main__":
    main()
