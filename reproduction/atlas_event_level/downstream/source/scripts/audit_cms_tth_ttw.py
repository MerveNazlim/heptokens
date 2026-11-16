#!/usr/bin/env python3
"""CPU-only pilot cutflow for CMS ttH(non-bb) and ttW in same-sign dileptons."""
import argparse
from collections import Counter
import json
from pathlib import Path
import zlib

import awkward as ak
import numpy as np
import uproot

from audit_cms_native_root import FEATURES, MCPilotCuts
from cms_mc_pilot_extract import native_values


TTH, TTW = 67691, 68073
GEN = ("GenPart_pdgId", "GenPart_genPartIdxMother")
DEEPJET_MEDIUM = 0.2489


def verify_root(path, size, checksum):
    if path.stat().st_size != size:
        raise ValueError("ROOT input size differs from official file index")
    with path.open("rb") as stream:
        if stream.read(4) != b"root":
            raise ValueError("Input does not have ROOT file signature")
        stream.seek(0)
        adler = 1
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            adler = zlib.adler32(chunk, adler)
    if f"adler32:{adler:08x}" != checksum:
        raise ValueError("ROOT input checksum differs from official file index")


def hww_from_ancestry(pdg, mother):
    """Require opposite-sign W daughters of the same recorded Higgs."""
    safe = (mother >= 0) & (mother < ak.num(pdg, axis=1))
    mom_pdg = ak.fill_none(pdg[ak.mask(mother, safe)], 0)
    plus = mother[(pdg == 24) & (mom_pdg == 25)]
    minus = mother[(pdg == -24) & (mom_pdg == 25)]
    partners = ak.cartesian((plus, minus), axis=1)
    p, m = ak.unzip(partners)
    return ak.any(p == m, axis=1)


def usable(chunk, obj, cuts, event_ok):
    counts = ak.to_numpy(chunk["n" + obj])
    selected = cuts.object_mask(obj, chunk, counts, event_ok)
    valid = np.ones(int(counts.sum()), dtype=bool)
    for field in FEATURES[obj]:
        _, ok = native_values(obj, field, chunk[f"{obj}_{field}"], counts)
        valid &= ok
    selected = selected & ak.unflatten(valid, counts)
    if obj == "Jet":
        selected = selected & (chunk["Jet_mass"] >= 0)
    elif obj in ("Electron", "Muon"):
        selected = selected & (abs(chunk[f"{obj}_charge"]) == 1)
    fields = ("eta", "phi", "btagDeepFlavB") if obj == "Jet" else ("eta", "phi")
    if obj in ("Electron", "Muon"):
        fields += ("charge",)
    return ak.zip({**{f: chunk[f"{obj}_{f}"][selected] for f in fields},
                   "index": ak.local_index(chunk[f"{obj}_pt"], axis=1)[selected]})


def selection_masks(chunk, cuts, signal):
    """Return the audited event masks and selected original object indices."""
    n = len(chunk["nElectron"])
    event_ok = ak.Array(cuts.event_mask(chunk, n))
    electron, muon, jet = [usable(chunk, obj, cuts, event_ok) for obj in ("Electron", "Muon", "Jet")]
    lep = ak.concatenate((electron, muon), axis=1)
    exactly_two = ak.num(lep, axis=1) == 2
    charge = ak.sum(lep.charge, axis=1)
    ss = exactly_two & (abs(charge) == 2)
    # Clean against the selected e/μ, using wrapped angular difference.
    combinations = ak.cartesian({"j": jet, "l": lep}, axis=1, nested=True)
    dphi = (combinations.j.phi - combinations.l.phi + np.pi) % (2 * np.pi) - np.pi
    dr2 = (combinations.j.eta - combinations.l.eta)**2 + dphi**2
    clean = jet[~ak.any(dr2 < 0.4**2, axis=2)]
    raw_nj, nj = ak.num(jet, axis=1), ak.num(clean, axis=1)
    btag = (abs(clean.eta) < 2.4) & (clean.btagDeepFlavB > DEEPJET_MEDIUM)
    raw_btag = (abs(jet.eta) < 2.4) & (jet.btagDeepFlavB > DEEPJET_MEDIUM)
    nb = ak.sum(btag, axis=1)
    selected = ss & (nj >= 4) & (nb >= 1)
    masks = {
        "root_entries": ak.Array(np.ones(n, dtype=bool)),
        "event_quality": event_ok,
        "exactly_two_e_or_mu": event_ok & exactly_two,
        "same_sign": ss,
        "same_sign_ge4_jets_before_overlap_removal": ss & (raw_nj >= 4),
        "same_sign_ge4_jets_ge1_btag_before_overlap_removal": ss & (raw_nj >= 4) & ak.any(raw_btag, axis=1),
        "same_sign_ge4_jets": ss & (nj >= 4),
        "same_sign_ge4_jets_ge1_btag": selected,
        "final_plus_plus": selected & (charge == 2),
        "final_minus_minus": selected & (charge == -2),
        "final_ee": selected & (ak.num(electron, axis=1) == 2),
        "final_emu": selected & (ak.num(electron, axis=1) == 1),
        "final_mumu": selected & (ak.num(electron, axis=1) == 0),
    }
    scopes = {"inclusive": masks}
    if signal:
        hww = hww_from_ancestry(chunk[GEN[0]], chunk[GEN[1]])
        scopes["hww_tagged"] = {name: mask & hww for name, mask in masks.items()}
    return scopes, {"Electron": electron.index, "Muon": muon.index, "Jet": clean.index}


def cutflow(chunk, cuts, signal):
    scopes, _ = selection_masks(chunk, cuts, signal)
    return {
        scope: {name: int(ak.sum(mask)) for name, mask in entries.items()}
        for scope, entries in scopes.items()
    }


def audit(source, record, source_uri, expected_size, expected_checksum, max_events=None):
    verify_root(source, expected_size, expected_checksum)
    with uproot.open(source) as root:
        tree = root["Events"]
        available = set(tree.keys())
        cuts = MCPilotCuts(available)
        expected_fields = {"run", "luminosityBlock", "event", "genWeight", *cuts.required}
        expected_fields.update("n" + obj for obj in FEATURES)
        expected_fields.update(f"{obj}_{field}" for obj, fields in FEATURES.items() for field in fields)
        if record == TTH:
            expected_fields.update(GEN)
        missing = sorted(expected_fields - available)
        result = {
            "record_id": record,
            "dataset": "ttH_nonbb" if record == TTH else "ttW_leptonic",
            "source": source_uri,
            "official_size_bytes": expected_size,
            "official_checksum": expected_checksum,
            "root_entries": tree.num_entries,
            "missing_branches": missing,
            "met_pt_phi_available": all(f"MET_{f}" in available for f in ("pt", "phi")),
            "status": "missing_branches" if missing else "complete",
            "selection": {
                "policy": cuts.policy["name"],
                "leptons": "Exactly two usable electrons/muons with charge ±1, of the same sign; existing pilot pT/eta/loose ID and native feature validity.",
                "jets": "Existing pilot jet fiducials and tight ID; native feature validity and mass >= 0; remove jets within deltaR < 0.4 of selected leptons for final counts.",
                "btag": "At least one selected jet with |eta| < 2.4 and Jet_btagDeepFlavB > 0.2489 (UL16 postVFP DeepJet medium).",
                "scope": "Unweighted one-file cutflow, no trigger, MET cut, train/test split or scale factors.",
                "truth": "HWW tag requires W+ and W- with the same immediate Higgs mother in GenPart. It is a signal decay label only.",
            },
        }
        if missing:
            return result
        needed = set(cuts.policy["event_filters"])
        for obj in ("Electron", "Muon", "Jet"):
            needed.update(("n" + obj, *(f"{obj}_{f}" for f in FEATURES[obj]),
                           *(r["branch"] for r in cuts.rules[obj])))
        if record == TTH:
            needed.update(GEN)
        inclusive = Counter()
        hww = Counter()
        stop = min(max_events, tree.num_entries) if max_events is not None else tree.num_entries
        for chunk in tree.iterate(sorted(needed), library="ak", how=dict, step_size="32 MB", entry_stop=stop):
            flow = cutflow(chunk, cuts, record == TTH)
            inclusive.update(flow["inclusive"])
            if record == TTH:
                hww.update(flow["hww_tagged"])
        if inclusive["root_entries"] != stop or inclusive["event_quality"] != cuts.report["events_after_event_filters"]:
            raise RuntimeError("Not all requested events were counted")
        for counts in (inclusive, hww) if record == TTH else (inclusive,):
            final = counts["same_sign_ge4_jets_ge1_btag"]
            if final != counts["final_plus_plus"] + counts["final_minus_minus"] or final != sum(
                    counts[f"final_{c}"] for c in ("ee", "emu", "mumu")):
                raise RuntimeError("Inconsistent final category counts")
        result.update({"scanned": stop, "inclusive_cutflow": dict(inclusive),
                       "hww_tagged_cutflow": dict(hww) if record == TTH else None})
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--record-id", type=int, choices=(TTH, TTW), required=True)
    parser.add_argument("--source-uri", required=True)
    parser.add_argument("--expected-size", type=int, required=True)
    parser.add_argument("--expected-checksum", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-events", type=int)
    args = parser.parse_args()
    if args.output.exists() or args.expected_size < 4 or (args.max_events is not None and args.max_events < 1):
        parser.error("Require a new report, valid size and positive optional max-events")
    result = audit(args.source, args.record_id, args.source_uri, args.expected_size,
                   args.expected_checksum, args.max_events)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "complete" else 3


if __name__ == "__main__":
    raise SystemExit(main())
