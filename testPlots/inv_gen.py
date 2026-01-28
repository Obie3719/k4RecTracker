#!/usr/bin/env python3
import math
from itertools import combinations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from podio.reading import get_reader
import uproot

INPUT_FILE = "/ceph/omunkombwe/fitter_idea_mumu50k.root"
CROSS_SECTION_PB = 2011.09   # pb; set to ee->Z->mumu value
LUMINOSITY_FB = 150     # fb^-1; set to target lumi
PT_MIN = 20.0            # GeV
ETA_MAX = 2.4
ISO_DR = 0.3
MUON_MASS = 0.105658     # GeV
MASS_BINS = np.linspace(60.0, 120.0, 120)

def count_events(path):
    with uproot.open(path) as f:
        events_keys = [k for k in f.keys() if k.startswith("events;")]
        events_key = max(events_keys, key=lambda n: int(n.split(";")[1]))
        return f[events_key].num_entries

def event_weight():
    n = count_events(INPUT_FILE)
    lumi_pb = LUMINOSITY_FB * 1e3  # fb^-1 -> pb^-1
    return n, (CROSS_SECTION_PB * lumi_pb) / n

def eta(px, py, pz):
    p = math.sqrt(px*px + py*py + pz*pz)
    if p == abs(pz):
        return float("inf") * (1 if pz >= 0 else -1)
    return 0.5 * math.log((p+pz)/(p-pz))

def delta_phi(a, b):
    d = a - b
    while d > math.pi:
        d -= 2*math.pi
    while d <= -math.pi:
        d += 2*math.pi
    return d

def delta_r(eta1, phi1, eta2, phi2):
    return math.hypot(eta1 - eta2, delta_phi(phi1, phi2))

def build_charged_particles(event):
    out = []
    for mc in event.get("MCParticles"):
        if abs(mc.getCharge()) < 1e-9 or mc.getGeneratorStatus() != 1:
            continue
        mom = mc.getMomentum()
        px, py, pz = mom.x, mom.y, mom.z
        pt = math.hypot(px, py)
        out.append({
            "pdg": mc.getPDG(),
            "charge": mc.getCharge(),
            "px": px,
            "py": py,
            "pz": pz,
            "pt": pt,
            "eta": eta(px, py, pz),
            "phi": math.atan2(py, px),
            "energy": math.sqrt(px*px + py*py + pz*pz + (MUON_MASS if abs(mc.getPDG()) == 13 else 0.0)**2),
        })
    return out

def is_isolated(mu, charged):
    for other in charged:
        if other is mu:
            continue
        if delta_r(mu["eta"], mu["phi"], other["eta"], other["phi"]) < ISO_DR:
            return False
    return True

def select_muons(event):
    charged = build_charged_particles(event)
    muons = []
    for p in charged:
        if abs(p["pdg"]) != 13:
            continue
        if p["pt"] < PT_MIN or abs(p["eta"]) > ETA_MAX:
            continue
        if not is_isolated(p, charged):
            continue
        muons.append(p)
    return muons

def invariant_mass(v1, v2):
    e = v1["energy"] + v2["energy"]
    px = v1["px"] + v2["px"]
    py = v1["py"] + v2["py"]
    pz = v1["pz"] + v2["pz"]
    m2 = e*e - (px*px + py*py + pz*pz)
    return math.sqrt(max(m2, 0.0))

def main():
    n_events, w = event_weight()
    print(f"Events: {n_events}, per-event weight: {w:.6g}")
    reader = get_reader(INPUT_FILE)
    events = reader.get("events")

    masses, weights = [], []
    for event in events:
        muons = select_muons(event)
        for mu1, mu2 in combinations(muons, 2):
            if mu1["charge"] * mu2["charge"] >= 0:
                continue  # require opposite charge
            masses.append(invariant_mass(mu1, mu2))
            weights.append(w)

    masses = np.asarray(masses)
    weights = np.asarray(weights)
    plt.figure(figsize=(6, 4))
    plt.hist(masses, bins=MASS_BINS, weights=weights, histtype="stepfilled",
             alpha=0.75, color="C1", edgecolor="black",
             label="opposite-charge isolated muons")
    plt.xlabel(r"$m_{\mu\mu}$ [GeV]")
    plt.ylabel("Events (weighted)")
    plt.title("ee -> Z -> mu mu invariant mass")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("mumu_invariant_mass.png", dpi=150)
    print("Saved mumu_invariant_mass.png")

if __name__ == "__main__":
    main()
