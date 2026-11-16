"""Explicit CMS UL2016 data pilot policy, not an ATLAS quality-ID translation."""

from copy import deepcopy

import awkward as ak
import numpy as np


PROFILE_NAME = "atlas_common_cms_ul2016_data_quality_v1"
MC_PROFILE_NAME = "atlas_common_cms_ul2016_mc_quality_v1"
SCHEMA_URL = "https://opendata.cern.ch/eos/opendata/cms/dataset-semantics/derived-data/PFNano/31308/JetHT_doc.html"
PREFIXES = {"electrons": "Electron", "muons": "Muon", "photons": "Photon",
            "taus": "Tau", "jets": "Jet"}
# These working points are analysis choices for the pilot, not equal-efficiency
# replacements for LHLoose, IsEMLoose, RNNJetScore or ATLAS event cleaning.
OBJECT_CUTS = {
    "electrons": [dict(name="cms_loose_noiso", branch="Electron_mvaFall17V2noIso_WPL",
                       op="eq", value=1, maximum=1)],
    "muons": [dict(name="cms_loose_id", branch="Muon_looseId", op="eq", value=1, maximum=1)],
    "photons": [dict(name="cms_loose_id", branch="Photon_cutBased", op="ge", value=1, maximum=3),
                dict(name="cms_electron_veto", branch="Photon_electronVeto", op="eq", value=1, maximum=1)],
    "taus": [dict(name="cms_deeptau_vsjet_loose", branch="Tau_idDeepTau2017v2p1VSjet",
                  op="bits", value=8, maximum=255)],
    "jets": [dict(name="cms_tight_id", branch="Jet_jetId", op="bits", value=2, maximum=7)],
}
EVENT_FILTERS = [
    "Flag_goodVertices", "Flag_globalSuperTightHalo2016Filter",
    "Flag_HBHENoiseFilter", "Flag_HBHENoiseIsoFilter",
    "Flag_EcalDeadCellTriggerPrimitiveFilter", "Flag_BadPFMuonFilter",
    "Flag_BadPFMuonDzFilter", "Flag_hfNoisyHitsFilter", "Flag_eeBadScFilter",
]
POLICY_NOTES = [
    "Explicit pilot choices for UL2016 DATA; not a universal CMS prescription or an MC policy.",
    "Electrons use a no-isolation loose MVA ID; no extra lepton isolation threshold is applied.",
    "ATLAS author/OQ and impact-parameter requirements remain unresolved, not replaced by these IDs.",
    "Photon cutBased is ordinal 0/1/2/3 here, NOT the separate Fall17V1 bitmap branch.",
    "Photon cut-based ID includes isolation criteria; this differs from ATLAS IsEMLoose.",
    "Tau VSjet uses bit 8, not score >= 8; no additional VSe/VSmu or prong cut is implied.",
    "Tight jet ID is bit 2, not jetId >= 2. No jet puId/JVT substitute is applied.",
    "ATLAS fiducial gaps are retained as the previous comparison choice, NOT CMS ECAL geometry.",
    "No new trigger, overlap-removal, event-multiplicity or impact-parameter cuts are applied.",
    "The JetHT input remains trigger-selected; this does not produce an unbiased inclusive sample.",
]
FEATURE_CONTRACT = {
    "status": "pilot_candidate_not_an_atlas_tokenizer_adapter",
    "object_features": {
        "electrons": ["pt", "eta", "phi", "charge"],
        "muons": ["pt", "eta", "phi", "charge"],
        "photons": ["pt", "eta", "phi"],
        "taus": ["pt", "eta", "phi", "charge"],
        "jets": ["pt", "eta", "phi", "mass"],
    },
    "event_features": ["common/met/pt", "common/met/phi"],
    "excluded_until_validated": ["trk_iso03", "event/mu", "jets/n_trk", "tracks", "met/sumet"],
    "notes": "Common names and units do not equate detector response. Keep masks; refit common-feature "
             "tokenizers/preprocessing on training data only. Do not reuse full-feature ATLAS tokenizers.",
}


def policy(sample_type="data"):
    if sample_type not in {"data", "mc"}:
        raise ValueError("Expected data or mc policy")
    result = dict(name="cms_ul2016_data_pilot_v1", sample_type="data", run_range=[271036, 284044],
                object_cuts=deepcopy(OBJECT_CUTS), event_filters=list(EVENT_FILTERS),
                sources=[SCHEMA_URL,
                         "https://github.com/cms-sw/cmssw/blob/CMSSW_10_6_X/PhysicsTools/NanoAOD/python/jets_cff.py",
                         "https://github.com/cms-sw/cmssw/blob/CMSSW_10_6_X/PhysicsTools/NanoAOD/python/photons_cff.py"],
                notes=list(POLICY_NOTES))
    if sample_type == "mc":
        result.update(name="cms_ul2016_mc_pilot_v1", sample_type="mc", run_range=None)
        result["event_filters"].remove("Flag_eeBadScFilter")
        result["notes"][0] = "UL2016 NanoAODv9 MC pilot: same object IDs, no data certification or data run range."
        result["notes"][-1] = "MC is not selected by data stream triggers or luminosity-normalized."
        result["notes"].append("eeBadScFilter is data-only and is not required for MC; all other named filters are required.")
    return result


def add_cms_quality(profile, sample_type="data"):
    if profile["name"] != "atlas_common_fiducial_v1":
        raise ValueError("CMS quality must be added to the original ATLAS-fiducial profile")
    result = deepcopy(profile)
    result["cms_quality"] = policy(sample_type)
    result["name"] = PROFILE_NAME if sample_type == "data" else MC_PROFILE_NAME
    result["shared_feature_contract"] = deepcopy(FEATURE_CONTRACT)
    return result


def validate_quality(profile):
    if profile.get("name") in {PROFILE_NAME, MC_PROFILE_NAME}:
        sample_type = "data" if profile["name"] == PROFILE_NAME else "mc"
        if profile.get("cms_quality") != policy(sample_type):
            raise ValueError("CMS quality policy differs from the versioned UL2016 pilot policy")
        if profile.get("shared_feature_contract") != FEATURE_CONTRACT:
            raise ValueError("Shared feature contract differs from the versioned pilot contract")
    elif "cms_quality" in profile:
        raise ValueError("CMS quality block requires a CMS quality profile name")


def rules_for(profile, name):
    return profile.get("cms_quality", {}).get("object_cuts", {}).get(name, []) if profile else []


def event_filters(profile, extra=()):
    native = profile.get("cms_quality", {}).get("event_filters", []) if profile else []
    return list(dict.fromkeys([*native, *extra]))


def audit_quality(tree, profile, sample_type):
    if not profile or "cms_quality" not in profile:
        return None
    validate_quality(profile)
    if sample_type != profile["cms_quality"]["sample_type"]:
        raise ValueError("CMS quality profile sample-type mismatch (data profile is data only)")
    required = required_branches(profile)
    names = set(tree.keys())
    missing = sorted(required - names)
    if missing:
        raise ValueError(f"Missing required CMS quality branches (no cuts were skipped): {missing}")
    return dict(status="required_branches_present", policy=profile["cms_quality"]["name"],
                required_branches={name: dict(typename=tree[name].typename, title=tree[name].title)
                                   for name in sorted(required)},
                semantics="Exact named UL2016 branches; dtype/value ranges checked when read. "
                          "Other eras or producer semantics require a new profile.")


def required_branches(profile):
    if not profile or "cms_quality" not in profile:
        return set()
    required = set(event_filters(profile))
    for name, prefix in PREFIXES.items():
        required.add(prefix + "_pt")
        required.update(rule["branch"] for rule in rules_for(profile, name))
    return required


def quality_mask(values, rule):
    """Works on original jagged branches and on padded H5 arrays for verification."""
    flat = ak.to_numpy(ak.flatten(values, axis=None)) if isinstance(values, ak.Array) else np.asarray(values).ravel()
    if flat.size and (flat.dtype.kind not in "bui" or np.any(flat < 0) or np.any(flat > rule["maximum"])):
        raise ValueError(f"Invalid ID encoding for {rule['branch']}; expected integer/bool 0..{rule['maximum']}")
    if not flat.size:
        return ak.zeros_like(values, dtype=bool) if isinstance(values, ak.Array) else np.zeros_like(values, dtype=bool)
    if rule["op"] == "eq":
        return values == rule["value"]
    if rule["op"] == "ge":
        return values >= rule["value"]
    if rule["op"] == "bits":
        return (values & rule["value"]) == rule["value"]
    raise ValueError(f"Unsupported quality operator: {rule['op']}")
