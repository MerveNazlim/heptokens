"""Paired event observables, keeping original object identities for closure."""

from __future__ import annotations

import math
import numpy as np

ELECTRON_MASS_GEV = 0.000511
MUON_MASS_GEV = 0.10566


def four_vector(pt: float, eta: float, phi: float, mass: float) -> np.ndarray:
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    energy = np.sqrt(max(px * px + py * py + pz * pz + mass * mass, 0.0))
    return np.array([energy, px, py, pz], dtype=np.float64)


def invariant_mass(vectors: list[np.ndarray]) -> float:
    if not vectors:
        return math.nan
    total = np.sum(vectors, axis=0)
    m2 = total[0] ** 2 - total[1] ** 2 - total[2] ** 2 - total[3] ** 2
    return float(np.sqrt(max(m2, 0.0)))


def delta_r(a, b):
    """Eta/phi distance with the azimuthal difference wrapped to [-pi, pi]."""
    dphi = np.arctan2(np.sin(a[2] - b[2]), np.cos(a[2] - b[2]))
    return float(np.hypot(a[1] - b[1], dphi))


def valid_kinematics(row):
    return bool(np.all(np.isfinite(row)) and row[0] > 0)


def paired_observables(objects, jet_pt_threshold=20.0):
    """Inputs: matched original/decoded (pT [GeV], eta, phi) rows per object.

    Four-lepton mass uses the four leading original electrons/muons, as in the
    legacy event diagnostic. No SFOS pairing, isolation or HZZ analysis cuts are
    implied. Delta R uses the leading original lepton and leading original jet
    above the jet threshold. Invalid decoded selections stay NaN, not replaced.
    HT reapplies the same threshold independently to original and decoded jets.
    """
    result = {name: (math.nan, math.nan) for name in ("m4l", "delta_r_lj", "HT")}
    leptons = []
    for obj, mass in (("electrons", ELECTRON_MASS_GEV), ("muons", MUON_MASS_GEV)):
        original, decoded = objects[obj]
        for index, row in enumerate(original):
            if valid_kinematics(row):
                leptons.append((row, decoded[index], mass))
    leptons.sort(key=lambda item: item[0][0], reverse=True)
    if len(leptons) >= 4:
        selected = leptons[:4]
        with np.errstate(over="ignore", invalid="ignore"):
            original_mass = invariant_mass([four_vector(*a, mass) for a, _, mass in selected])
            decoded_mass = (
                invariant_mass([four_vector(*b, mass) for _, b, mass in selected])
                if all(valid_kinematics(b) for _, b, _ in selected)
                else math.nan
            )
        result["m4l"] = (original_mass, decoded_mass)
    jets, decoded_jets = objects["jets"]
    candidates = [
        i for i, row in enumerate(jets) if valid_kinematics(row) and row[0] > jet_pt_threshold
    ]
    if leptons and candidates:
        index = max(candidates, key=lambda i: jets[i, 0])
        a, b, _ = leptons[0]
        result["delta_r_lj"] = (
            delta_r(a, jets[index]),
            (
                delta_r(b, decoded_jets[index])
                if valid_kinematics(b) and valid_kinematics(decoded_jets[index])
                else math.nan
            ),
        )
    ht = []
    for rows in (jets, decoded_jets):
        pts = rows[:, 0]
        ht.append(float(pts[pts > jet_pt_threshold].sum()) if np.isfinite(pts).all() else math.nan)
    result["HT"] = tuple(ht)
    return result
