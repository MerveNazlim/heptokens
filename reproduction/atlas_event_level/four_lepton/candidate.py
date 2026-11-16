"""Four-lepton candidate kernel from the recorded ATLAS selection helper.

Input leptons already pass the upstream object cuts (see ../README.md).
"""

from itertools import combinations
import math

Z_MASS = 91.1876
LEPTON_MASS = {"Electron": 0.000510999, "Muon": 0.1056584}
CHANNELS = {4: "4e", 2: "2e2mu", 0: "4mu"}


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
