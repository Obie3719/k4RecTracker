#!/usr/bin/env python3
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
CROSS_SECTION_PB = 2011.9   # ee->Z->mumu cross section from Sample_z_mumu (pb)
LUMINOSITY_FB = 150         # set to target luminosity in fb^-1
PT_MIN_GEV = 5.0
ETA_MAX = 2.4
ISO_DR = 0.3
D0_MAX_MM = 0.5
Z0_MAX_MM = 0.5
CHI2_NDF_MAX = 5.0
MIN_TRACKSTATES = 1
MASS_BINS = np.linspace(60.0, 120.0, 150)
MUON_MASS_GEV = 0.105658
DEBUG_EVENTS = 0          # set >0 to print a few events' track params


B_FIELD_TESLA = 2.0


def count_events(path):
    with uproot.open(path) as f:
        event_keys = [k for k in f.keys() if k.startswith("events;")]
        events_key = max(event_keys, key=lambda name: int(name.split(";")[1]))
        return f[events_key].num_entries


def compute_event_weight(cross_section_pb, luminosity_fb, n_events):
    lumi_pb = luminosity_fb * 1e3  # fb^-1 to pb^-1
    return (cross_section_pb * lumi_pb) / float(n_events)


def pseudorapidity(px, py, pz):
    p = math.sqrt(px * px + py * py + pz * pz)
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
    energy = math.sqrt(px * px + py * py + pz * pz + MUON_MASS_GEV * MUON_MASS_GEV)

    return {
        "pt": pt,
        "eta": eta,
        "phi": phi,
        "px": px,
        "py": py,
        "pz": pz,
        "energy": energy,
        "omega": omega,
    }


def _get_track_attr(track, name, default=None):
    try:
        attr = getattr(track, name)
    except Exception:
        return default
    try:
        return attr() if callable(attr) else attr
    except Exception:
        return default


def select_muons(event, evt_index=None):
    muons = []
    tracks = event.get("Fitted_tracks_muon")
    for it, track in enumerate(tracks):
        if track.trackStates_size() < MIN_TRACKSTATES:
            continue
        state = track.getTrackStates(0)
        if abs(state.D0) > D0_MAX_MM:
            continue
        if abs(state.Z0) > Z0_MAX_MM:
            continue
        chi2 = _get_track_attr(track, "getChi2")
        ndf = _get_track_attr(track, "getNdf")
        if chi2 is None:
            chi2 = _get_track_attr(track, "chi2")
        if ndf is None:
            ndf = _get_track_attr(track, "ndf")
        if chi2 is not None and ndf is not None and ndf > 0:
            if (chi2 / ndf) > CHI2_NDF_MAX:
                continue
        vec = trackstate_to_vec(state)
        if vec["pt"] < PT_MIN_GEV:
            continue
        if abs(vec["eta"]) > ETA_MAX:
            continue
        muons.append(vec)

    isolated = []
    for i, mu in enumerate(muons):
        iso = True
        for j, other in enumerate(muons):
            if i == j:
                continue
            if delta_r(mu["eta"], mu["phi"], other["eta"], other["phi"]) < ISO_DR:
                iso = False
                break
        if iso:
            isolated.append(mu)

    if DEBUG_EVENTS and DEBUG_EVENTS > 0:
        if evt_index is None or evt_index < DEBUG_EVENTS:
            print(f"Event {evt_index if evt_index is not None else '?'} debug:")
            for m in muons[:6]:
                print(f"  mu cand pt={m['pt']:.2f} eta={m['eta']:.2f} phi={m['phi']:.2f} omega={m['omega']}")
    return isolated


def invariant_mass(v1, v2):
    e = v1["energy"] + v2["energy"]
    px = v1["px"] + v2["px"]
    py = v1["py"] + v2["py"]
    pz = v1["pz"] + v2["pz"]
    m2 = e * e - (px * px + py * py + pz * pz)
    return math.sqrt(max(m2, 0.0))


def accumulate_masses(events, event_weight):
    masses = []
    weights = []
    total_muons = 0
    total_pairs = 0
    for ievt, event in enumerate(events):
        muons = select_muons(event, evt_index=ievt)
        total_muons += len(muons)
        for mu1, mu2 in combinations(muons, 2):
            masses.append(invariant_mass(mu1, mu2))
            weights.append(event_weight)
            total_pairs += 1
    print(f"Selected muons: {total_muons}, pairs: {total_pairs}")
    return np.asarray(masses), np.asarray(weights)


def plot_masses(masses, weights, bins=MASS_BINS, out="mumu_invariant_mass_reco.png"):
    plt.figure(figsize=(6, 4))
    plt.hist(
        masses,
        bins=bins,
        weights=weights,
        histtype="stepfilled",
        alpha=0.75,
        color="C1",
        edgecolor="black",
        label="isolated muons (reco)",
    )
    #plt.yscale("log")
    plt.xlabel(r"$m_{\mu\mu}$ [GeV]")
    plt.ylabel("Events (weighted)")
    plt.title("invariant mass (reconstructed)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")


def main():
    n_events = count_events(INPUT_FILE)
    w = compute_event_weight(CROSS_SECTION_PB, LUMINOSITY_FB, n_events)
    print(f"Events: {n_events}, per-event weight: {w:.6g}")

    reader = get_reader(INPUT_FILE)
    events = reader.get("events")

    masses, weights = accumulate_masses(events, w)
    plot_masses(masses, weights)


if __name__ == "__main__":
    main()
