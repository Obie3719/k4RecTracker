#!/usr/bin/env python3
"""
Truth-match IDEA CDCH tracks to MCParticles using uproot + awkward + numpy.

Fix: read branches without `how="zip"` so keys stay exactly as in the TTree
(e.g. "CDCHTracks/CDCHTracks.trackerHits_begin").
"""
import argparse
import numpy as np
import awkward as ak
import uproot
from collections import defaultdict, Counter

# --- Helpers -----------------------------------------------------------------

def load_event(tree, entry):
    """Load all needed branches for a single event as a flat dict of Awkward arrays."""
    branches = [
        # Tracks -> trackerHits relations
        "CDCHTracks/CDCHTracks.trackerHits_begin",
        "CDCHTracks/CDCHTracks.trackerHits_end",
        "_CDCHTracks_trackerHits/_CDCHTracks_trackerHits.collectionID",
        "_CDCHTracks_trackerHits/_CDCHTracks_trackerHits.index",
        # Associations RecoHit <-> SimHit
        "DCH_DigiSimAssociationCollection/DCH_DigiSimAssociationCollection.weight",
        "_DCH_DigiSimAssociationCollection_from/_DCH_DigiSimAssociationCollection_from.collectionID",
        "_DCH_DigiSimAssociationCollection_from/_DCH_DigiSimAssociationCollection_from.index",
        "_DCH_DigiSimAssociationCollection_to/_DCH_DigiSimAssociationCollection_to.collectionID",
        "_DCH_DigiSimAssociationCollection_to/_DCH_DigiSimAssociationCollection_to.index",
        # SimHit -> MC
        "_DCHCollection_particle/_DCHCollection_particle.collectionID",
        "_DCHCollection_particle/_DCHCollection_particle.index",
    ]
    arrays = tree.arrays(branches, entry_start=entry, entry_stop=entry+1)  # flat dict
    # unwrap single-event arrays to first element
    ev = {}
    for k, arr in arrays.items():
        # Each is a length-1 Awkward array; take [0] to get the per-event content
        ev[k] = arr[0]
    return ev

def build_track_hit_oids(ev):
    """Return list of lists of RecoHit OIDs per track: [ [(collID, idx), ...], ... ]"""
    begins = ev["CDCHTracks/CDCHTracks.trackerHits_begin"]
    ends   = ev["CDCHTracks/CDCHTracks.trackerHits_end"]
    rel_c  = ev["_CDCHTracks_trackerHits/_CDCHTracks_trackerHits.collectionID"]
    rel_i  = ev["_CDCHTracks_trackerHits/_CDCHTracks_trackerHits.index"]
    ntrk = len(begins)
    tracks_hits = []
    for i in range(ntrk):
        s, t = int(begins[i]), int(ends[i])
        oids = [(int(rel_c[j]), int(rel_i[j])) for j in range(s, t)]
        tracks_hits.append(oids)
    return tracks_hits

def build_assoc_index(ev):
    """Return dict: RecoHitOID -> list of (SimHitOID, weight)."""
    w   = ev["DCH_DigiSimAssociationCollection/DCH_DigiSimAssociationCollection.weight"]
    fc  = ev["_DCH_DigiSimAssociationCollection_from/_DCH_DigiSimAssociationCollection_from.collectionID"]
    fi  = ev["_DCH_DigiSimAssociationCollection_from/_DCH_DigiSimAssociationCollection_from.index"]
    tc  = ev["_DCH_DigiSimAssociationCollection_to/_DCH_DigiSimAssociationCollection_to.collectionID"]
    ti  = ev["_DCH_DigiSimAssociationCollection_to/_DCH_DigiSimAssociationCollection_to.index"]
    assoc = defaultdict(list)
    n = len(w)
    for k in range(n):
        assoc[(int(fc[k]), int(fi[k]))].append(((int(tc[k]), int(ti[k])), float(w[k])))
    return assoc

def match_tracks(ev, purity_cut=0.7):
    tracks_hits = build_track_hit_oids(ev)
    assoc_idx   = build_assoc_index(ev)
    mc_c = ev["_DCHCollection_particle/_DCHCollection_particle.collectionID"]
    mc_i = ev["_DCHCollection_particle/_DCHCollection_particle.index"]

    results = []
    for it, hit_oids in enumerate(tracks_hits):
        w_by_mc = Counter()
        total_w = 0.0
        n_used  = 0
        for rh_oid in hit_oids:
            for (sh_oid, w) in assoc_idx.get(rh_oid, ()):  # sh_oid = (sim_cid, sim_idx)
                _, sim_idx = sh_oid
                # Guard: sim_idx must be within DCHCollection range
                if sim_idx < 0 or sim_idx >= len(mc_i):
                    continue
                # Map SimHit -> MC using index-aligned arrays
                mc_oid = (int(mc_c[sim_idx]), int(mc_i[sim_idx]))
                w_by_mc[mc_oid] += w
                total_w += w
                n_used += 1
        if total_w <= 0 or not w_by_mc:
            results.append((-1, 0.0, 0.0, 0))
            continue
        best_mc, best_w = max(w_by_mc.items(), key=lambda kv: kv[1])
        purity = best_w / total_w if total_w else 0.0
        if purity < purity_cut:
            results.append((-1, purity, total_w, n_used))
        else:
            # Return only MC index; collectionID is available in best_mc[0] if needed
            results.append((best_mc[1], purity, total_w, n_used))
    return results

# --- CLI ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Truth-match CDCH tracks to MCParticles (uproot)")
    ap.add_argument("-i", "--input", required=True, help="EDM4hep ROOT file")
    ap.add_argument("--tree", default="events", help="TTree name (default: events)")
    ap.add_argument("--purity-cut", type=float, default=0.7, help="Minimum purity to accept a match")
    args = ap.parse_args()

    with uproot.open(args.input) as f:
        tree = f[args.tree]
        nentries = tree.num_entries
        print("# event, track_index, best_mc_index, purity, total_weight, n_hits_used")
        for entry in range(nentries):
            ev = load_event(tree, entry)
            results = match_tracks(ev, purity_cut=args.purity_cut)
            for it, (mc_idx, purity, totw, nused) in enumerate(results):
                print(f"{entry}, {it}, {mc_idx}, {purity:.3f}, {totw:.3f}, {nused}")

if __name__ == "__main__":
    main()