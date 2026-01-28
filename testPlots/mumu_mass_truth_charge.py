#!/usr/bin/env python3
# mumu_mass_reco_fixed_charge.py
# Fixed by flipping omega sign for charge assignment

import math
from itertools import combinations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from podio.reading import get_reader
import uproot

# Analysis configuration
INPUT_FILE = "/ceph/omunkombwe/fitter_idea_mumu50k.root"
CROSS_SECTION_PB = 2011.9
LUMINOSITY_FB = 150
PT_MIN_GEV = 20.0
ETA_MAX = 2.4
ISO_DR = 0.3
MASS_BINS = np.linspace(60.0, 120.0, 120)
MUON_MASS_GEV = 0.105658
REQUIRE_OPPOSITE_CHARGE = False

OMEGA_MODE = "curvature_mm"
B_FIELD_TESLA = 2.0

def count_events(path):
    with uproot.open(path) as f:
        event_keys = [k for k in f.keys() if k.startswith("events;")]
        events_key = max(event_keys, key=lambda name: int(name.split(";")[1]))
        return f[events_key].num_entries


def compute_event_weight(cross_section_pb, luminosity_fb, n_events):
    lumi_pb = luminosity_fb * 1e3
    return (cross_section_pb * lumi_pb) / float(n_events)


def pseudorapidity(px, py, pz):
    p = math.sqrt(px**2 + py**2 + pz**2)
    if p == abs(pz):
        return float("inf") * (1 if pz >= 0 else -1)
    return 0.5 * math.log((p + pz) / (p - pz))


def delta_phi(phi1, phi2):
    dphi = phi1 - phi2
    while dphi > math.pi:
        dphi -= 2 * math.pi
    while dphi <= -math.pi:
        dphi += 2 * math.pi
    return dphi


def delta_r(eta1, phi1, eta2, phi2):
    return math.hypot(eta1 - eta2, delta_phi(phi1, phi2))


def pt_from_omega(omega):
    if omega == 0:
        return 0.0
    if OMEGA_MODE == "q_over_pt":
        return abs(1.0 / omega)
    return abs(0.299792458 * B_FIELD_TESLA / (omega * 1000.0))


def trackstate_to_vec(state):
    omega = state.omega
    pt = pt_from_omega(omega)
    phi = state.phi
    tan_lambda = state.tanLambda

    px = pt * math.cos(phi)
    py = pt * math.sin(phi)
    pz = pt * tan_lambda
    eta = pseudorapidity(px, py, pz)
    energy = math.sqrt(px**2 + py**2 + pz**2 + MUON_MASS_GEV**2)
    
    # FLIPPED SIGN — this fixes the all-negative charge bug
    charge = -1 if omega > 0 else 1 if omega < 0 else 0

    return {
        "pt": pt,
        "eta": eta,
        "phi": phi,
        "px": px,
        "py": py,
        "pz": pz,
        "energy": energy,
        "charge": charge,
        "omega": omega,
    }


def track_charge(track, state):
    # Try explicit charge first
    for getter in ("getCharge", "charge"):
        try:
            attr = getattr(track, getter)
            q = attr() if callable(attr) else attr
            if q is not None:
                return int(q)
        except:
            pass
    # Fallback to flipped omega sign
    if state and state.omega != 0:
        return -1 if state.omega > 0 else 1
    return 0


def select_muons(event):
    muons = []
    tracks = event.get("Fitted_tracks_muon")
    for track in tracks:
        if track.trackStates_size() == 0:
            continue
        state = track.getTrackStates(0)
        vec = trackstate_to_vec(state)
        vec["charge"] = track_charge(track, state)
        if vec["pt"] < PT_MIN_GEV or abs(vec["eta"]) > ETA_MAX:
            continue
        muons.append(vec)

    # Isolation
    isolated = []
    for i, mu in enumerate(muons):
        if all(delta_r(mu["eta"], mu["phi"], other["eta"], other["phi"]) >= ISO_DR 
               for j, other in enumerate(muons) if i != j):
            isolated.append(mu)
    return isolated

def invariant_mass(v1, v2):
    e = v1["energy"] + v2["energy"]
    px = v1["px"] + v2["px"]
    py = v1["py"] + v2["py"]
    pz = v1["pz"] + v2["pz"]
    return math.sqrt(max(e**2 - (px**2 + py**2 + pz**2), 0.0))

def main():
    n_events = count_events(INPUT_FILE)
    w = compute_event_weight(CROSS_SECTION_PB, LUMINOSITY_FB, n_events)
    print(f"Events: {n_events}, weight: {w:.4f}")

    reader = get_reader(INPUT_FILE)
    events = reader.get("events")

    masses = []
    weights = []

    for event in events:
        muons = select_muons(event)
        for mu1, mu2 in combinations(muons, 2):
            if REQUIRE_OPPOSITE_CHARGE and mu1["charge"] * mu2["charge"] >= 0:
                continue
            masses.append(invariant_mass(mu1, mu2))
            weights.append(w)

    print(f"Found {len(masses)} opposite-charge pairs")

    plt.figure(figsize=(8, 6))
    plt.hist(masses, bins=MASS_BINS, weights=weights, histtype='step', linewidth=2, color='navy')
    plt.xlabel(r"$m_{\mu\mu}$ [GeV]")
    plt.ylabel("Events")
    plt.title("Z → μ⁺μ⁻ Invariant Mass (reconstructed)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("mumu_mass_fixed.png", dpi=200)
    plt.close()

    print("Plot saved: mumu_mass_fixed.png")

if __name__ == "__main__":
    main()