import argparse
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from podio.reading import get_reader
import edm4hep


def oid(obj):
    obj_id = obj.getObjectID()
    return (obj_id.collectionID, obj_id.index) # unique identifier tuple


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


# TrackState interpretation. EDM4hep/LCIO stores omega as curvature (q/R) in 1/mm.
OMEGA_MODE = "curvature_mm"  # options: "curvature_mm", "curvature_m", "q_over_pt"
B_FIELD_TESLA = 2.0

DEFAULT_PT_MIN = 0.0
DEFAULT_PT_MAX = 46.0
DEFAULT_PT_BINS = 23


def get_collection(event, name):
    try:
        return event.get(name)
    except Exception as exc:
        raise KeyError(f"Collection '{name}' not found in input file") from exc


def parse_pdg_arg(value):
    text = value.strip().lower()
    if text in ("all", "any", "charged"):
        return None
    pdgs = set()
    for part in value.replace(" ", ",").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            pdgs.add(int(part))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid PDG code: {part}") from exc
    return pdgs or None


def build_sim_to_truth(sim_hits_list):
    sim_to_truth = {}
    for sim_hits in sim_hits_list:
        for sim in sim_hits:
            mc_part = sim.getParticle()
            sim_to_truth[oid(sim)] = mc_part.getObjectID().index
    return sim_to_truth # map from sim hit OID to MC truth index


def build_digi_truth_map(associations_list, sim_to_truth):
    digi_truth = defaultdict(set)
    for associations in associations_list:
        for assoc in associations:
            digi_oid = oid(assoc.getFrom())         
            sim_oid  = oid(assoc.getTo())           
            truth_idx = sim_to_truth.get(sim_oid)
            if truth_idx is not None:
                digi_truth[digi_oid].add(truth_idx) # each truth particle counted once per digi
    return digi_truth # map from digi OID to set of MC truth indices


def match_tracks(event, tracks_name, sim_hits_names, assoc_names): # match reconstructed tracks to MC truth particles
    tracks = get_collection(event, tracks_name)

    sim_hits_list = [get_collection(event, name) for name in sim_hits_names]
    associations_list = [get_collection(event, name) for name in assoc_names]

    sim_to_truth = build_sim_to_truth(sim_hits_list)
    digi_truth   = build_digi_truth_map(associations_list, sim_to_truth)

    matches = []
    for _, track in enumerate(tracks):
        counts = Counter()
        for ihit in range(track.trackerHits_size()):
            digi = track.getTrackerHits(ihit)
            for truth_idx in digi_truth.get(oid(digi), ()):
                counts[truth_idx] += 1           # count shared hits

        if counts:
            best_idx, shared_hits = counts.most_common(1)[0]
        else:
            best_idx = None
            shared_hits = 0

        matches.append({
            "truth_index": best_idx, 
            "purity": (shared_hits / track.trackerHits_size()) if track.trackerHits_size() else 0.0, 
        })
    return matches 


def pt_from_omega(omega):
    if omega == 0:
        return None
    if OMEGA_MODE == "q_over_pt":
        return abs(1.0 / omega)
    if OMEGA_MODE == "curvature_m":
        return abs(0.299792458 * B_FIELD_TESLA / omega)
    return abs(0.299792458 * B_FIELD_TESLA / (omega * 1000.0))


def track_pt(track):
    states = track.getTrackStates()
    if len(states) == 0:
        return None
    return pt_from_omega(states[0].omega)


def is_reconstructable(mc, pdg_allow, gen_status): # define reconstructable particles
    if abs(mc.getCharge()) <= 0:
        return False
    if gen_status is not None and mc.getGeneratorStatus() != gen_status:
        return False
    if pdg_allow is None:
        return True
    return abs(mc.getPDG()) in pdg_allow


def collect_reconstructable_indices(mc_particles, pdg_allow, gen_status):
    indices = set()
    for idx, mc in enumerate(mc_particles):
        if is_reconstructable(mc, pdg_allow, gen_status):
            indices.add(idx)
    return indices


def accumulate_efficiency_reco(
    events,
    pt_bins,
    tracks_name,
    sim_hits_names,
    assoc_names,
    mc_name,
    pdg_allow,
    gen_status,
    min_purity,
    max_events,
):
    totals = np.zeros(len(pt_bins) - 1)
    selected  = np.zeros(len(pt_bins) - 1)
    
    for iev, event in enumerate(events):
        if max_events is not None and iev >= max_events:
            break

        matches = match_tracks(event, tracks_name, sim_hits_names, assoc_names)
        mc_particles = get_collection(event, mc_name)
        reconstructable = collect_reconstructable_indices(mc_particles, pdg_allow, gen_status)
        tracks = get_collection(event, tracks_name)

        for track, match in zip(tracks, matches):
            pt = track_pt(track)
            if pt is None:
                continue
            bin_idx = np.digitize(pt, pt_bins) - 1
            if bin_idx < 0 or bin_idx >= len(totals):
                continue  # pT outside defined bins
            totals[bin_idx] += 1
            if (
                match["truth_index"] is not None
                and match["truth_index"] in reconstructable
                and match["purity"] >= min_purity
            ):
                selected[bin_idx] += 1
    return totals, selected


def wilson_interval(selected, totals, z=1.96):
    eff = np.zeros_like(selected)
    err = np.zeros_like(selected)
    
    for i, (k, n) in enumerate(zip(selected, totals)):
        if n == 0:
            eff[i] = 0.0
            err[i] = 0.0
        else:
            p = k / n
            denom = 1 + (z * z) / n
            center = (p + (z * z) / (2 * n)) / denom
            half = (z * math.sqrt((p * (1 - p) / n) + (z * z) / (4 * n * n))) / denom
            eff[i] = center
            err[i] = half
    return eff, err


def plot_eff_vs_pt(pt_bins, totals, selected, out, title):
    
    eff, err = wilson_interval(selected, totals, z=1.96)  # 95% CL (1.96 sigma)
    centers = 0.5 * (pt_bins[:-1] + pt_bins[1:])
    
    plt.errorbar(centers, eff, yerr=err, fmt="o", color="C0", label="efficiency")
    plt.title(title)
    plt.xlabel("pT [GeV] (reco)")
    plt.ylabel("Tracking efficiency")
    plt.ylim(0.6, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tracking efficiency vs reco pT from EDM4hep ROOT files")
    parser.add_argument("--input", default="/ceph/omunkombwe/fitter_idea_mumu50k.root",
                        help="EDM4hep ROOT file path")
    parser.add_argument("--tracks", default="Fitted_tracks_muon",
                        help="track collection name")
    parser.add_argument("--mc", default="MCParticles",
                        help="MCParticles collection name")
    parser.add_argument("--pdg", type=parse_pdg_arg, default=parse_pdg_arg("13"),
                        help="comma-separated PDG codes, or 'all' for any charged particle")
    parser.add_argument("--gen-status", type=int, default=1,
                        help="generator status to select (use -1 to disable)")
    parser.add_argument("--min-purity", type=float, default=0.75,
                        help="minimum shared-hit purity to count as reconstructed")
    parser.add_argument("--pt-min", type=float, default=DEFAULT_PT_MIN,
                        help="minimum pT for bins (GeV)")
    parser.add_argument("--pt-max", type=float, default=DEFAULT_PT_MAX,
                        help="maximum pT for bins (GeV)")
    parser.add_argument("--pt-bins", type=int, default=DEFAULT_PT_BINS,
                        help="number of pT bins")
    parser.add_argument("--max-events", type=int, default=None,
                        help="maximum number of events to process")
    parser.add_argument("--out", default="eff_vs_pt_reco.png",
                        help="output plot filename")
    args = parser.parse_args()

    gen_status = None if args.gen_status == -1 else args.gen_status
    pt_bins = np.linspace(args.pt_min, args.pt_max, args.pt_bins + 1)

    if args.pdg is None:
        title = "Tracking Efficiency vs pT (reco, all charged)"
    else:
        pdg_list = ",".join(str(p) for p in sorted(args.pdg))
        title = f"Tracking Efficiency vs pT (reco, PDG: {pdg_list})"

    reader = get_reader(args.input)
    events = reader.get("events")
    totals, selected = accumulate_efficiency_reco(
        events,
        pt_bins,
        args.tracks,
        SIM_HITS_NAMES,
        ASSOC_NAMES,
        args.mc,
        args.pdg,
        gen_status,
        args.min_purity,
        args.max_events,
    )
    plot_eff_vs_pt(pt_bins, totals, selected, out=args.out, title=title)
