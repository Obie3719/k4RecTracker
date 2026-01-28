#!/usr/bin/env python3
import math
from collections import Counter, defaultdict

import numpy as np
import uproot
from podio.reading import get_reader

INPUT_FILE = "/ceph/omunkombwe/fitter_idea_mumu50k.root"
TRACK_COLLECTION = "Fitted_tracks_muon"
MUON_MASS_GEV = 0.105658
PURITY_MIN = 0.75
ETA_MAX = 2.4
PT_MIN = 5.0
B_FIELD_TESLA = 2.0
OMEGA_MODE = "curvature_mm"  # {"curvature_mm", "curvature_m", "q_over_pt"}


def count_events(path):
    with uproot.open(path) as f:
        event_keys = [k for k in f.keys() if k.startswith("events;")]
        events_key = max(event_keys, key=lambda name: int(name.split(";")[1]))
        return f[events_key].num_entries


def iter_events(events, max_events=None, log_every=500):
    for ievt, event in enumerate(events):
        if max_events is not None and ievt >= max_events:
            break
        if log_every and ievt and ievt % log_every == 0:
            print(f"Processed {ievt} events...")
        yield ievt, event


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


def pt_from_omega(omega, bfield):
    if omega == 0:
        return 0.0
    if OMEGA_MODE == "q_over_pt":
        return abs(1.0 / omega)
    if OMEGA_MODE == "curvature_m":
        return abs(0.299792458 * bfield / omega)
    return abs(0.299792458 * bfield / (omega * 1000.0))


def trackstate_to_vec(state, bfield):
    omega = state.omega
    pt = pt_from_omega(omega, bfield)
    phi = state.phi
    tan_lambda = state.tanLambda
    px = pt * math.cos(phi)
    py = pt * math.sin(phi)
    pz = pt * tan_lambda
    p = math.sqrt(px * px + py * py + pz * pz)
    eta = pseudorapidity(px, py, pz)
    theta = (math.pi / 2.0) - math.atan(tan_lambda)
    energy = math.sqrt(p * p + MUON_MASS_GEV * MUON_MASS_GEV)
    charge = -1 if omega > 0 else 1 if omega < 0 else 0
    return {
        "pt": pt,
        "eta": eta,
        "phi": phi,
        "theta": theta,
        "p": p,
        "px": px,
        "py": py,
        "pz": pz,
        "energy": energy,
        "charge": charge,
    }


def mc_particle_to_vec(mc):
    mom = mc.getMomentum()
    px, py, pz = mom.x, mom.y, mom.z
    p = math.sqrt(px * px + py * py + pz * pz)
    pt = math.hypot(px, py)
    eta = pseudorapidity(px, py, pz)
    phi = math.atan2(py, px)
    theta = math.acos(pz / p) if p > 0 else 0.0
    energy = math.sqrt(p * p + mc.getMass() * mc.getMass())
    vtx = mc.getVertex()
    return {
        "pt": pt,
        "eta": eta,
        "phi": phi,
        "theta": theta,
        "p": p,
        "px": px,
        "py": py,
        "pz": pz,
        "energy": energy,
        "vx": vtx.x,
        "vy": vtx.y,
        "vz": vtx.z,
    }


def compute_true_d0(phi, vx, vy):
    return -vx * math.sin(phi) + vy * math.cos(phi)


def invariant_mass(v1, v2):
    e = v1["energy"] + v2["energy"]
    px = v1["px"] + v2["px"]
    py = v1["py"] + v2["py"]
    pz = v1["pz"] + v2["pz"]
    return math.sqrt(max(e * e - (px * px + py * py + pz * pz), 0.0))


def is_truth_muon(mc):
    try:
        return abs(mc.getPDG()) == 13 and mc.getGeneratorStatus() == 1
    except Exception:
        return False


def oid(obj):
    obj_id = obj.getObjectID()
    return (obj_id.collectionID, obj_id.index)


def build_sim_to_truth(sim_hits):
    sim_to_truth = {}
    for sim in sim_hits:
        mc_part = sim.getParticle()
        sim_to_truth[oid(sim)] = mc_part.getObjectID().index
    return sim_to_truth


def build_digi_truth_map(associations, sim_to_truth):
    digi_truth = defaultdict(set)
    for assoc in associations:
        digi_oid = oid(assoc.getFrom())
        sim_oid = oid(assoc.getTo())
        truth_idx = sim_to_truth.get(sim_oid)
        if truth_idx is not None:
            digi_truth[digi_oid].add(truth_idx)
    return digi_truth


def match_tracks(event, track_collection, digi_truth):
    matches = []
    tracks = event.get(track_collection)
    for track in tracks:
        counts = Counter()
        for ihit in range(track.trackerHits_size()):
            digi = track.getTrackerHits(ihit)
            for truth_idx in digi_truth.get(oid(digi), ()):
                counts[truth_idx] += 1
        if counts:
            best_idx, shared_hits = counts.most_common(1)[0]
            purity = shared_hits / track.trackerHits_size() if track.trackerHits_size() else 0.0
        else:
            best_idx = None
            purity = 0.0
        matches.append({"track": track, "truth_index": best_idx, "purity": purity})
    return matches


def collect_truth_muons(event):
    muons = []
    mc_particles = event.get("MCParticles")
    for idx, mc in enumerate(mc_particles):
        if not is_truth_muon(mc):
            continue
        muons.append({"index": idx, "vec": mc_particle_to_vec(mc)})
    return muons


def collect_matched_muons(
    event, track_collection, bfield, purity_min, pt_min, eta_max
):
    sim_to_truth = build_sim_to_truth(event.get("DCHCollection"))
    digi_truth = build_digi_truth_map(event.get("DCH_DigiSimAssociationCollection"), sim_to_truth)
    matches = match_tracks(event, track_collection, digi_truth)
    mc_particles = event.get("MCParticles")

    best_by_truth = {}
    for match in matches:
        tidx = match["truth_index"]
        if tidx is None or tidx >= len(mc_particles):
            continue
        if not is_truth_muon(mc_particles[tidx]):
            continue
        prev = best_by_truth.get(tidx)
        if prev is None or match["purity"] > prev["purity"]:
            best_by_truth[tidx] = match

    truth_muons = collect_truth_muons(event)
    matched = []
    for t in truth_muons:
        match = best_by_truth.get(t["index"])
        if match is None or match["purity"] < purity_min:
            continue
        track = match["track"]
        if track.trackStates_size() == 0:
            continue
        state = track.getTrackStates(0)
        reco = trackstate_to_vec(state, bfield)
        if reco["pt"] < pt_min or abs(reco["eta"]) > eta_max:
            continue
        matched.append(
            {
                "truth": t["vec"],
                "reco": reco,
                "state": state,
                "purity": match["purity"],
            }
        )
    return truth_muons, matched
