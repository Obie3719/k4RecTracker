#!/usr/bin/env python3
import argparse
import math
import os
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from podio.reading import get_reader

DEFAULT_INPUT = "/ceph/omunkombwe/fitter_idea_mumu50k.root"
DEFAULT_TRACK_COLLECTION = "CDCHTracks"
DEFAULT_MC_COLLECTION = "MCParticles"

SIM_HITS_NAMES = [
    "DCHCollection",
    "VertexBarrelCollection",
    "VertexEndcapCollection",
    "SiWrBCollection",
    "SiWrDCollection",
]

ASSOC_NAMES = [
    "DCH_DigiSimAssociationCollection",
    "VTXBSimDigiLinks",
    "VTXDSimDigiLinks",
    "SiWrBSimDigiLinks",
    "SiWrDSimDigiLinks",
]

# omega is curvature [1/mm]
B_FIELD_TESLA = 2.0

DEFAULT_PT_MIN = 0.0
DEFAULT_PT_MAX = 100.0
DEFAULT_ETA_MAX = 2.4
DEFAULT_PURITY_MIN = 0.90

# From your dump: loc=1 is clearly the IP/perigee state
DEFAULT_STATE_LOCATION = 1


def parse_pdg_arg(value):
    text = value.strip().lower()
    if text in ("all", "any", "charged"):
        return None
    pdgs = set()
    for part in value.replace(" ", ",").split(","):
        part = part.strip()
        if not part:
            continue
        pdgs.add(int(part))
    return pdgs or None


def parse_args():
    p = argparse.ArgumentParser(
        description="Truth vs reco pT scatter plot (truth-matched), robust TrackState selection + Δφ cut."
    )
    p.add_argument("-i", "--input", default=DEFAULT_INPUT, help="Input PODIO root file.")
    p.add_argument("--tracks", default=DEFAULT_TRACK_COLLECTION, help="Track collection name.")
    p.add_argument("--mc", default=DEFAULT_MC_COLLECTION, help="MCParticles collection.")
    p.add_argument("--bfield", type=float, default=B_FIELD_TESLA, help="B field [T].")

    p.add_argument("--purity-min", type=float, default=DEFAULT_PURITY_MIN,
                   help="Minimum shared-hit purity for a match.")
    p.add_argument("--pt-min", type=float, default=DEFAULT_PT_MIN, help="Truth pT cut [GeV].")
    p.add_argument("--pt-max", type=float, default=DEFAULT_PT_MAX, help="Max pT for axes [GeV].")
    p.add_argument("--eta-max", type=float, default=DEFAULT_ETA_MAX, help="Truth |eta| cut.")

    p.add_argument("--pdg", type=parse_pdg_arg, default=parse_pdg_arg("13"),
                   help="comma-separated PDG codes, or 'all' for any charged particle")
    p.add_argument("--gen-status", type=int, default=1, help="Generator status (use -1 to disable).")
    p.add_argument("--max-events", type=int, default=None, help="Process at most N events.")

    # Key fixes / debug knobs
    p.add_argument("--state-location", type=int, default=DEFAULT_STATE_LOCATION,
                   help="TrackState.location to use (from your dump: 1=AtIP/perigee).")
    p.add_argument("--dphi-max", type=float, default=0.20,
                   help="Max |Δφ| between reco state.phi and truth phi (radians).")
    p.add_argument("--min-hits", type=int, default=10,
                   help="Require at least N tracker hits per track.")
    p.add_argument("--omega-max", type=float, default=None,
                   help="Optional sanity cut: require |omega| < omega-max [1/mm].")
    p.add_argument("--dump-bad", action="store_true",
                   help="Print a few pathological matches for debugging.")

    p.add_argument("-o", "--output", default="reco_vs_truth_pt.png", help="Output plot filename.")
    p.add_argument("--output-dir", default="validation_plots", help="Directory for the output plot.")
    return p.parse_args()


def resolve_output_path(output_dir, output):
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, output)


def oid(obj):
    obj_id = obj.getObjectID()
    return (obj_id.collectionID, obj_id.index)


def build_sim_to_truth(sim_hits_list):
    sim_to_truth = {}
    for sim_hits in sim_hits_list:
        for sim in sim_hits:
            mc_part = sim.getParticle()
            sim_to_truth[oid(sim)] = mc_part.getObjectID().index
    return sim_to_truth


def build_digi_truth_map(associations_list, sim_to_truth):
    digi_truth = defaultdict(set)
    for associations in associations_list:
        for assoc in associations:
            digi_oid = oid(assoc.getFrom())
            sim_oid = oid(assoc.getTo())
            truth_idx = sim_to_truth.get(sim_oid)
            if truth_idx is not None:
                digi_truth[digi_oid].add(truth_idx)
    return digi_truth


def match_tracks(tracks, digi_truth):
    matches = []
    for track in tracks:
        nhits = track.trackerHits_size()
        counts = Counter()
        for ihit in range(nhits):
            digi = track.getTrackerHits(ihit)
            for truth_idx in digi_truth.get(oid(digi), ()):
                counts[truth_idx] += 1
        if counts and nhits:
            best_idx, shared_hits = counts.most_common(1)[0]
            purity = shared_hits / nhits
        else:
            best_idx = None
            purity = 0.0
        matches.append({"track": track, "truth_index": best_idx, "purity": purity})
    return matches


def pseudorapidity(px, py, pz):
    p = math.sqrt(px * px + py * py + pz * pz)
    if p == abs(pz):
        return float("inf") * (1 if pz >= 0 else -1)
    return 0.5 * math.log((p + pz) / (p - pz))


def pt_from_curvature_mm(omega_mm, bfield):
    # omega in 1/mm -> curvature in 1/m is omega_mm * 1000
    if omega_mm == 0:
        return 0.0
    return abs(0.299792458 * bfield / (omega_mm * 1000.0))


def trackstate_to_vec(state, bfield):
    pt = pt_from_curvature_mm(state.omega, bfield)
    phi = state.phi
    tan_lambda = state.tanLambda
    px = pt * math.cos(phi)
    py = pt * math.sin(phi)
    pz = pt * tan_lambda
    eta = pseudorapidity(px, py, pz)
    return {"pt": pt, "eta": eta, "phi": phi}


def mc_particle_to_vec(mc):
    mom = mc.getMomentum()
    px, py, pz = mom.x, mom.y, mom.z
    pt = math.hypot(px, py)
    eta = pseudorapidity(px, py, pz)
    phi = math.atan2(py, px)
    return {"pt": pt, "eta": eta, "phi": phi}


def is_truth_selected(mc, pdg_allow, gen_status):
    try:
        if abs(mc.getCharge()) <= 0:
            return False
        if gen_status is not None and mc.getGeneratorStatus() != gen_status:
            return False
        if pdg_allow is None:
            return True
        return abs(mc.getPDG()) in pdg_allow
    except Exception:
        return False


def delta_phi(a, b):
    d = a - b
    while d > math.pi:
        d -= 2 * math.pi
    while d < -math.pi:
        d += 2 * math.pi
    return d


def pick_state_by_location(track, loc_code):
    n = track.trackStates_size()
    for i in range(n):
        st = track.getTrackStates(i)
        if st.location == loc_code:
            return st
    return None


def main():
    args = parse_args()
    out_path = resolve_output_path(args.output_dir, args.output)

    gen_status = None if args.gen_status == -1 else args.gen_status

    reader = get_reader(args.input)
    events = reader.get("events")

    pts_true = []
    pts_reco = []

    bad_printed = 0

    for iev, event in enumerate(events):
        if args.max_events is not None and iev >= args.max_events:
            break

        sim_hits_list = [event.get(name) for name in SIM_HITS_NAMES]
        associations_list = [event.get(name) for name in ASSOC_NAMES]
        sim_to_truth = build_sim_to_truth(sim_hits_list)
        digi_truth = build_digi_truth_map(associations_list, sim_to_truth)

        tracks = event.get(args.tracks)
        matches = match_tracks(tracks, digi_truth)
        mc_particles = event.get(args.mc)

        # best match per truth particle
        best_by_truth = {}
        for match in matches:
            track = match["track"]
            if track.trackerHits_size() < args.min_hits:
                continue
            if match["purity"] < args.purity_min:
                continue

            tidx = match["truth_index"]
            if tidx is None or tidx >= len(mc_particles):
                continue

            mc = mc_particles[tidx]
            if not is_truth_selected(mc, args.pdg, gen_status):
                continue

            prev = best_by_truth.get(tidx)
            if prev is None or match["purity"] > prev["purity"]:
                best_by_truth[tidx] = match

        for tidx, match in best_by_truth.items():
            track = match["track"]
            state = pick_state_by_location(track, args.state_location)
            if state is None:
                continue

            if args.omega_max is not None and abs(state.omega) > args.omega_max:
                continue

            reco = trackstate_to_vec(state, args.bfield)
            truth = mc_particle_to_vec(mc_particles[tidx])

            if truth["pt"] < args.pt_min:
                continue
            if abs(truth["eta"]) > args.eta_max:
                continue
            if reco["pt"] <= 0 or not np.isfinite(reco["pt"]):
                continue

            # Angular consistency cut: kills wrong matches
            dphi = abs(delta_phi(reco["phi"], truth["phi"]))
            if dphi > args.dphi_max:
                if args.dump_bad and bad_printed < 20 and truth["pt"] > 40:
                    rp = state.referencePoint
                    print("REJECT dphi",
                          "pt_true", f"{truth['pt']:.2f}",
                          "pt_reco", f"{reco['pt']:.2f}",
                          "dphi", f"{dphi:.3f}",
                          "omega", f"{state.omega:.3e}",
                          "loc", state.location,
                          "ref", (rp.x, rp.y, rp.z),
                          "nhits", track.trackerHits_size(),
                          "purity", f"{match['purity']:.2f}")
                    bad_printed += 1
                continue

            pts_true.append(truth["pt"])
            pts_reco.append(reco["pt"])

    if not pts_true:
        print("No matched tracks found after cuts.")
        return

    pts_true = np.array(pts_true)
    pts_reco = np.array(pts_reco)

    plt.figure(figsize=(7, 6))
    plt.scatter(pts_true, pts_reco, s=10, alpha=0.30, color="black", label="matched tracks")
    plt.plot([0.0, args.pt_max], [0.0, args.pt_max], ls="--", color="red", lw=1.5, label="Ideal")
    plt.xlim(0.0, args.pt_max)
    plt.ylim(0.0, args.pt_max)
    plt.xlabel(r"Truth $p_{T}$ [GeV]")
    plt.ylabel(r"Reconstructed $p_{T}$ [GeV]")
    plt.title("Truth vs reconstructed $p_{T}$")
    plt.grid(True, alpha=0.35, linestyle=":")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
