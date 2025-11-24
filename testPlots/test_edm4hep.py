import math
import numpy as np
import matplotlib
matplotlib.use("Agg")  # must come before importing pyplot
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from podio.reading import get_reader
import edm4hep


def oid(obj):
    obj_id = obj.getObjectID()
    return (obj_id.collectionID, obj_id.index) # unique identifier tuple



def build_sim_to_truth(sim_hits):
    sim_to_truth = {}
    for sim in sim_hits:
        mc_part = sim.getParticle()
        sim_to_truth[oid(sim)] = mc_part.getObjectID().index
    return sim_to_truth # map from sim hit OID to MC truth index



def build_digi_truth_map(associations, sim_to_truth):
    digi_truth = defaultdict(set)
    for assoc in associations:
        digi_oid = oid(assoc.getFrom())         
        sim_oid  = oid(assoc.getTo())           
        truth_idx = sim_to_truth.get(sim_oid)
        if truth_idx is not None:
            digi_truth[digi_oid].add(truth_idx) # each truth particle counted once per digi
    return digi_truth # map from digi OID to set of MC truth indices



def match_tracks(event): # match reconstructed tracks to MC truth particles
    tracks        = event.get("CDCHTracks")
    sim_hits      = event.get("DCHCollection")
    associations  = event.get("DCH_DigiSimAssociationCollection")

    sim_to_truth = build_sim_to_truth(sim_hits)
    digi_truth   = build_digi_truth_map(associations, sim_to_truth)

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



#PT_BINS = np.array([0.5, 0.8, 1.2, 1.6, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])  # GeV
PT_BINS = np.linspace(0.5, 4.5, 25)  # GeV

def mc_pt(mc):
    mom = mc.getMomentum()
    return math.hypot(mom.x, mom.y)



def is_reconstructable(mc): # define reconstructable particles
    return (abs(mc.getCharge()) > 0) and (mc.getGeneratorStatus() == 1)  and (mc.getPDG() == 13)




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



def accumulate_efficiency(events, pt_bins): 
    totals = np.zeros(len(pt_bins) - 1)
    selected  = np.zeros(len(pt_bins) - 1)
    
    for event in events:
        matches = match_tracks(event)
        best_purity = best_purity_by_truth(matches)
        mc_particles = event.get("MCParticles")

        for idx, mc in enumerate(mc_particles): #
            if not is_reconstructable(mc):
                continue
            pt = mc_pt(mc)
            bin_idx = np.digitize(pt, pt_bins) - 1
            if bin_idx < 0 or bin_idx >= len(totals):
                continue  # pT outside defined bins
            totals[bin_idx] += 1
            if best_purity.get(idx, 0.0) >= 0.75:
                selected[bin_idx] += 1
    return totals, selected



def wald_interval(selected, totals, z=1):
    eff = np.zeros_like(selected)
    err = np.zeros_like(selected)
    
    for i, (k, n) in enumerate(zip(selected, totals)):
        if n == 0:
            eff[i] = 0.0
            err[i] = 0.0
        else:
            p = k / n
            eff[i] = p
            err[i] = z * math.sqrt(p * (1 - p) / n)
    return eff, err



def plot_eff_vs_pt(pt_bins, totals, selected, out="eff_vs_pt.png"):
    
    eff, err = wald_interval(selected, totals, z=1.0)  # 68% CL (1 sigma)
    centers = 0.5 * (pt_bins[:-1] + pt_bins[1:])
    
    
    plt.errorbar(centers, eff, yerr=err, fmt="o", color="C0", label="efficiency")
    plt.xlabel("pT [GeV]")
    plt.ylabel("Tracking efficiency")
    plt.ylim(0.9, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    reader = get_reader("/ceph/omunkombwe/fit__idea_o1_v03_60.root")
    events = reader.get("events")
    totals, selected = accumulate_efficiency(events, PT_BINS)
    plot_eff_vs_pt(PT_BINS, totals, selected)

