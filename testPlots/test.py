import math
import numpy as np
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from podio.reading import get_reader
import edm4hep

reader = get_reader("/ceph/omunkombwe/fit__idea_o1_v03_60.root")
events = reader.get("events")


for iev, event in enumerate(events):
    if iev >= 6:
        break
    print(f"Event {iev}")
    for assoc in event.get("DCH_DigiSimAssociationCollection"):
        digi_hit = assoc.getFrom().getObjectID()
        sim_hit = assoc.getTo().getObjectID()
        print(f"  digi_idx {digi_hit.index} (coll {digi_hit.collectionID}) --> sim_idx {sim_hit.index} (coll {sim_hit.collectionID})")

    tracks = event.get("CDCHTracks")
    for itrk, track in enumerate(tracks):
        dch_hits = [hit.getObjectID().index
                    for hit in track.getTrackerHits()
                    if hit.getObjectID().collectionID == 3486206992]  # DCH_DigiCollection ID
        print(f"  track {itrk}: {len(dch_hits)} DCH digis used -> {dch_hits}")
