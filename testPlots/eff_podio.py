import sys, ROOT, podio, edm4hep
from podio.reading import get_reader
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, namedtuple


reader = get_reader("/ceph/omunkombwe/fit__idea_o1_v03_60.root")
events = reader.get("events")



for iev, event in enumerate(events):
    if iev >= 5:
        break
    print(f"Event {iev}")
    for assoc in event.get("DCH_DigiSimAssociationCollection"):
        d_digi = assoc.getFrom().getObjectID()
        s_hit = assoc.getTo().getObjectID()
        print(f"  digi {d_digi.index} (coll {d_digi.collectionID}) --> sim {s_hit.index} (coll {s_hit.collectionID})")

    tracks = event.get("CDCHTracks")
    for itrk, track in enumerate(tracks):
        dch_hits = [hit.getObjectID().index
                    for hit in track.getTrackerHits()
                    if hit.getObjectID().collectionID == 3486206992]  # DCH_DigiCollection ID
        print(f"  track {itrk}: {len(dch_hits)} DCH digis used -> {dch_hits}")


#collections = [
   # ("DCHCollection", "DCH hit"),
   # ("MCParticles", "MCParticle"),
   # ("DCH_DigiSimAssociationCollection", "DCH assoc"),
    #("CDCHTracks", "Track"),
#]

#for iev, event in enumerate(events):
   # if iev >= 5:
       # break
    #print(f"Event {iev}")
    #for coll_name, label in collections:
        #coll = event.get(coll_name)
        #print(f"  {coll_name} (n={len(coll)})")
        #for obj in coll:
            #oid = obj.getObjectID()
            #print(f"    {label}: index={oid.index} collectionID={oid.collectionID}")
    #print()




#for iev, event in enumerate(events):
   # print(f"Event {iev}")
    #for sim in event.get("DCHCollection"):
      #  sim_id = sim.getObjectID()
        #print("sim hit", sim_id.index, sim_id.collectionID)
        
        
    #for part in event.get("MCParticles"):
    #    part_id = part.getObjectID()
        #print("MC", part_id.index, part_id.collectionID)
       
    #for assoc in event.get("DCH_DigiSimAssociationCollection"):
    #    from_idx = assoc.getFrom().getObjectID()
    #    to_idx = assoc.getTo().getObjectID()
        #print("Assoc", from_idx.index, from_idx.collectionID,
        #     "->", to_idx.index, to_idx.collectionID)
          
    #for track in event.get("CDCHTracks"):
    #    track_id = track.getObjectID()
    
        #print("Track", track_id.index, track_id.collectionID)
