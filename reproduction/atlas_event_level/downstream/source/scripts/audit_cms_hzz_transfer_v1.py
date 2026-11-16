#!/usr/bin/env python3
"""Read-only, checksum-pinned four-lepton pilot on CMS NanoAOD simulation."""

import argparse
from collections import Counter
from itertools import combinations
import json
import math
from pathlib import Path

import awkward as ak
import numpy as np
import uproot

from audit_cms_native_root import FEATURES, MCPilotCuts
from audit_cms_tth_ttw import usable, verify_root


RECORDS = {37728: "ggF_H125_to_ZZstar_to_4l", 75589: "continuum_ZZ_to_4l"}
Z_MASS = 91.1876
LEPTON_MASS = {"Electron": 0.000510999, "Muon": 0.1056584}
CHANNELS = {4: "4e", 2: "2e2mu", 0: "4mu"}
WINDOWS = ((105, 140), (115, 135), (120, 130))
MASS_EDGES = (0, 70, 105, 115, 120, 125, 130, 135, 140, 180, 250, 500, 1000)


def p4(pt, eta, phi, obj):
    px, py, pz = pt * math.cos(phi), pt * math.sin(phi), pt * math.sinh(eta)
    return math.hypot(pt * math.cosh(eta), LEPTON_MASS[obj]), px, py, pz


def mass(vectors):
    energy, px, py, pz = map(sum, zip(*vectors))
    return math.sqrt(max(energy * energy - px * px - py * py - pz * pz, 0.0))


def find_candidates(leptons):
    """Return loose cut stages and a deterministic eligible Z1/Z2 pair."""
    n = len(leptons)
    vectors = [p4(pt, eta, phi, obj) for pt, eta, phi, _, obj in leptons]
    pairs = {}
    sfos = []
    for i, j in combinations(range(n), 2):
        pt_i, eta_i, phi_i, q_i, obj_i = leptons[i]
        pt_j, eta_j, phi_j, q_j, obj_j = leptons[j]
        os = q_i == -q_j
        mll = mass((vectors[i], vectors[j]))
        dr = math.hypot(eta_i - eta_j, math.remainder(phi_i - phi_j, 2 * math.pi))
        pairs[(i, j)] = (mll, dr, os)
        if os and obj_i == obj_j:
            sfos.append((i, j))
    loose, mass5 = False, False
    best = None
    for a, b in combinations(sfos, 2):
        if set(a) & set(b):
            continue
        loose = True
        z1, z2 = sorted((a, b), key=lambda ij: (abs(pairs[ij][0] - Z_MASS), ij))
        m1, m2 = pairs[z1][0], pairs[z2][0]
        mass5 |= min(m1, m2) > 5
        if not (40 < m1 < 120 and 12 < m2 < 120):
            continue
        ids = sorted(a + b)
        pts = sorted((leptons[i][0] for i in ids), reverse=True)
        if not (pts[0] > 20 and pts[1] > 10):
            continue
        if any(dr <= 0.02 or (os and mll <= 4)
               for mll, dr, os in (pairs[pair] for pair in combinations(ids, 2))):
            continue
        m4l = mass([vectors[i] for i in ids])
        if m4l <= 70:
            continue
        rank = (abs(m1 - Z_MASS), -sum(leptons[i][0] for i in z2), z1, z2)
        if best is None or rank < best[0]:
            n_e = sum(leptons[i][4] == "Electron" for i in ids)
            best = rank, CHANNELS[n_e], m4l, m1, m2
    return loose, mass5, best


def audit(source, record, source_uri, size, checksum):
    verify_root(source, size, checksum)
    report = {
        "record_id": record, "dataset": RECORDS[record], "source_uri": source_uri,
        "official_size_bytes": size, "official_checksum": checksum,
        "selection": {
            "objects": "Existing CMS native validity and MC object/quality cuts (e/mu pT>10 GeV).",
            "candidate": "At least four usable e/mu; two disjoint SFOS pairs, Z1 closest to 91.1876 GeV.",
            "core": "Leading/subleading pT>20/10 GeV, 40<mZ1<120, 12<mZ2<120 GeV, all pair dR>0.02, "
                    "all OS pair masses>4 GeV and m4l>70 GeV. Rank passing candidates by closest mZ1, "
                    "then largest scalar Z2 lepton pT sum, then source order.",
            "scope": "CMS-inspired feasibility audit, not the official analysis. No trigger, FSR recovery, "
                     "matrix-element ranking, additional isolation, generator-level lepton cuts or fake estimate.",
            "mass_windows": [list(w) for w in WINDOWS],
        },
    }
    with uproot.open(source) as root:
        tree = root["Events"]
        available = set(tree.keys())
        cuts = MCPilotCuts(available)
        expected = {"run", "luminosityBlock", "event", "genWeight", *cuts.required}
        expected.update("n" + obj for obj in FEATURES)
        expected.update(f"{obj}_{field}" for obj, fields in FEATURES.items() for field in fields)
        missing = sorted(expected - available)
        report.update(root_entries=tree.num_entries, missing_branches=missing)
        if missing:
            return {**report, "status": "missing_branches"}

        counts = Counter(dict.fromkeys(("root_entries", "event_quality", "at_least_four_usable_e_mu",
                                        "two_disjoint_sfos", "two_disjoint_sfos_both_mll_gt5", "hzz_core"), 0))
        channels = {channel: Counter({"hzz_core": 0, **{f"m4l_{lo}_{hi}": 0 for lo, hi in WINDOWS}})
                    for channel in CHANNELS.values()}
        histograms = {channel: np.zeros(len(MASS_EDGES), dtype=np.int64) for channel in channels}
        objects = {stage: Counter(dict.fromkeys(FEATURES, 0)) for stage in ("hzz_core", "m4l_105_140")}
        weights = {stage: {"sumw": 0.0, "sumw2": 0.0, "negative": 0} for stage in objects}
        for chunk in tree.iterate(sorted(expected), library="ak", how=dict, step_size="32 MB"):
            n = len(chunk["event"])
            counts["root_entries"] += n
            quality = ak.Array(cuts.event_mask(chunk, n))
            counts["event_quality"] += int(ak.sum(quality))
            collections = {obj: usable(chunk, obj, cuts, quality) for obj in FEATURES}
            size_by_obj = {obj: ak.to_numpy(ak.num(rows, axis=1)) for obj, rows in collections.items()}
            candidates = ak.to_numpy(quality) & (size_by_obj["Electron"] + size_by_obj["Muon"] >= 4)
            counts["at_least_four_usable_e_mu"] += int(candidates.sum())
            leptons = {}
            for obj in LEPTON_MASS:
                kept = collections[obj][candidates]
                pts = chunk[f"{obj}_pt"][candidates][kept.index]
                leptons[obj] = ak.to_list(ak.zip(
                    {"pt": pts, "eta": kept.eta, "phi": kept.phi, "charge": kept.charge}
                ))
            gen_weight = ak.to_numpy(chunk["genWeight"])
            for i, source_row in enumerate(np.flatnonzero(candidates)):
                event = [
                    (float(r["pt"]), float(r["eta"]), float(r["phi"]), int(r["charge"]), obj)
                    for obj in LEPTON_MASS for r in leptons[obj][i]
                ]
                loose, mass5, best = find_candidates(event)
                counts["two_disjoint_sfos"] += loose
                counts["two_disjoint_sfos_both_mll_gt5"] += mass5
                if best is None:
                    continue
                _, channel, m4l, _, _ = best
                counts["hzz_core"] += 1
                channels[channel]["hzz_core"] += 1
                histograms[channel][np.searchsorted(MASS_EDGES, m4l, side="right") - 1] += 1
                stages = ["hzz_core"]
                for lo, hi in WINDOWS:
                    if lo < m4l < hi:
                        label = f"m4l_{lo}_{hi}"
                        counts[label] += 1
                        channels[channel][label] += 1
                        if label in objects:
                            stages.append(label)
                w = float(gen_weight[source_row])
                if not math.isfinite(w):
                    raise ValueError("Nonfinite generator weight in selected event")
                for stage in stages:
                    for obj in objects[stage]:
                        objects[stage][obj] += int(size_by_obj[obj][source_row])
                    weights[stage]["sumw"] += w
                    weights[stage]["sumw2"] += w * w
                    weights[stage]["negative"] += int(w < 0)
        for lo, hi in WINDOWS:
            counts.setdefault(f"m4l_{lo}_{hi}", 0)
        if counts["root_entries"] != tree.num_entries or any(
            counts[key] != sum(x[key] for x in channels.values())
            for key in channels["4e"]
        ):
            raise RuntimeError("Incomplete ROOT audit or inconsistent channel totals")
    return {
        **report, "status": "complete", "cutflow": dict(counts), "channels": channels,
        "object_counts_before_overlap_cleaning": objects, "gen_weights_diagnostics": weights,
        "m4l_lower_edges_gev_last_bin_overflow": MASS_EDGES,
        "m4l_counts": {key: values.tolist() for key, values in histograms.items()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--record-id", type=int, choices=RECORDS, required=True)
    parser.add_argument("--source-uri", required=True)
    parser.add_argument("--expected-size", type=int, required=True)
    parser.add_argument("--expected-checksum", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.expected_size < 4:
        parser.error("Require a fresh output and valid ROOT file size")
    report = audit(args.source, args.record_id, args.source_uri, args.expected_size, args.expected_checksum)
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report["status"] == "complete" else 3


if __name__ == "__main__":
    raise SystemExit(main())
