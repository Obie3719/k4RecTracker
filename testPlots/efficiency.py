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
        
        #if track.trackerHits_size() < 4:
        #    continue
        
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



DEFAULT_PT_MIN = 0.20
DEFAULT_PT_MAX = 200.0
DEFAULT_PT_BINS = 60

def mc_pt(mc):
    mom = mc.getMomentum()
    return math.hypot(mom.x, mom.y)



def is_reconstructable(mc, pdg_allow, gen_status): # define reconstructable particles
    if abs(mc.getCharge()) <= 0:
        return False
    if gen_status is not None and mc.getGeneratorStatus() != gen_status:
        return False
    if pdg_allow is None:
        return True
    return abs(mc.getPDG()) in pdg_allow




def best_purity_by_truth(matches): # get best purity per truth particle
    best = {}
    for m in matches:
        tidx = m["truth_index"]
        if tidx is None:
            continue
        purity = m["purity"]
        if purity > best.get(tidx, 0.0):
            best[tidx] = purity 
    return best



def accumulate_efficiency(
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
        best_purity = best_purity_by_truth(matches)
        mc_particles = get_collection(event, mc_name)

        for idx, mc in enumerate(mc_particles): #
            if not is_reconstructable(mc, pdg_allow, gen_status):
                continue
            pt = mc_pt(mc)
            bin_idx = np.digitize(pt, pt_bins) - 1
            if bin_idx < 0 or bin_idx >= len(totals):
                continue  # pT outside defined bins
            totals[bin_idx] += 1
            if best_purity.get(idx, 0.0) >= min_purity:
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
    plt.xlabel("pT [GeV]")
    plt.ylabel("Tracking efficiency")
    plt.ylim(0.0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")

def resolve_pt_bins(pt_bins_arg, pt_min, pt_max):
    raw = np.asarray(pt_bins_arg)
    if raw.ndim == 0:
        n_bins = int(raw.item())
        if n_bins <= 0:
            raise ValueError("--pt-bins must be a positive integer")
        return np.linspace(pt_min, pt_max, n_bins + 1)

    # Backward-compatible path if a bin-edge array is passed as default.
    pt_bins = raw.astype(float).ravel()
    if pt_bins.size < 2:
        raise ValueError("pt bin-edge array must have at least 2 entries")
    if not np.all(np.diff(pt_bins) > 0):
        raise ValueError("pt bin edges must be strictly increasing")
    return pt_bins


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tracking efficiency vs pT from EDM4hep ROOT files")
    parser.add_argument("--input", default="/ceph/omunkombwe/pi_Barrel2/deg60/fit_pi_P0.2-200.0GeV_theta60deg_nev100000_merged.root",
                        help="EDM4hep ROOT file path")
    parser.add_argument("--tracks", default="Fitted_tracks",
                        help="track collection name")
    parser.add_argument("--mc", default="MCParticles",
                        help="MCParticles collection name")
    parser.add_argument("--pdg", type=parse_pdg_arg, default=parse_pdg_arg("211"),
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
    parser.add_argument("--out", default="eff_vs_pt_pion1.png",
                        help="output plot filename")
    args = parser.parse_args()

    gen_status = None if args.gen_status == -1 else args.gen_status
    pt_bins = resolve_pt_bins(args.pt_bins, args.pt_min, args.pt_max)

    if args.pdg is None:
        title = "Tracking Efficiency vs pT (all charged)"
    else:
        pdg_list = ",".join(str(p) for p in sorted(args.pdg))
        title = f"Tracking Efficiency vs pT (PDG: {pdg_list})"

    reader = get_reader(args.input)
    events = reader.get("events")
    totals, selected = accumulate_efficiency(
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
