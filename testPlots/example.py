from collections import defaultdict, Counter
from podio.reading import get_reader
import edm4hep



def oid(obj):
    obj_id = obj.getObjectID()
    return (obj_id.collectionID, obj_id.index)



def build_sim_to_truth(sim_hits):
    sim_to_truth = {}
    for sim in sim_hits:
        part = sim.getParticle()
        sim_to_truth[oid(sim)] = part.getObjectID().index
    return sim_to_truth



def build_digi_truth_map(associations, sim_to_truth):
    digi_truth = defaultdict(set)
    for assoc in associations:
        digi_oid = oid(assoc.getFrom())         # DCH_DigiCollection 
        sim_oid  = oid(assoc.getTo())           # DCHCollection 
        truth_idx = sim_to_truth.get(sim_oid)
        if truth_idx is not None:
            digi_truth[digi_oid].add(truth_idx) # each truth particle counted once per digi
    return digi_truth



def match_tracks(event):
    tracks        = event.get("CDCHTracks")
    sim_hits      = event.get("DCHCollection")
    associations  = event.get("DCH_DigiSimAssociationCollection")
    mc_particles  = event.get("MCParticles")

    sim_to_truth = build_sim_to_truth(sim_hits)
    digi_truth   = build_digi_truth_map(associations, sim_to_truth)

    matches = []
    for itrack, track in enumerate(tracks):
        
        #if track.trackerHits_size() < 4:
        #    continue
        
        counts = Counter()
        
        for ihit in range(track.trackerHits_size()):
            digi = track.getTrackerHits(ihit)
            for truth_idx in digi_truth.get(oid(digi), ()):
                counts[truth_idx] += 1           # count shared hits, no weights


        if counts:
            best_idx, shared_hits = counts.most_common(1)[0]
            best_truth = mc_particles[best_idx]
        else:
            best_idx = None
            best_truth = None
            shared_hits = 0


        matches.append({
            "track_index": itrack,
            "track_id": oid(track),
            "truth_index": best_idx,
            "truth_pdg": best_truth.getPDG() if best_truth else None,
            "shared_hits": shared_hits,
            "purity": (shared_hits / track.trackerHits_size()) if track.trackerHits_size() else 0.0, 
        })
    return matches



if __name__ == "__main__":
    reader = get_reader("/ceph/omunkombwe/eezmu_fitter_edm4hep.root")
    events = reader.get("events")
    for iev, event in enumerate(events):
        for match in match_tracks(event):
            print(f"event {iev}", match)
        if iev == 10:
            break
