import math
from itertools import combinations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from podio.reading import get_reader
import uproot
from podio.reading import get_reader
r=get_reader("/ceph/omunkombwe/fitter_idea_mumu50k.root")
events=r.get("events")
with_hits=with_mc=0
for trk in events[0].get("Fitted_tracks_muon"):
    nh = trk.trackerHits_size()
    with_hits += nh>0
    mc_link = False
    for i in range(nh):
        hit = trk.getTrackerHits(i)
        mc = None
        for g in ("getParticle","getMCParticle","particle","MCParticle"):
            try:
                attr=getattr(hit,g); mc=attr() if callable(attr) else attr
                if mc is not None: mc_link=True; break
            except Exception: pass
    with_mc += mc_link
print("tracks:", len(events[0].get('Fitted_tracks_muon')),
      "with hits:", with_hits, "with MC link:", with_mc)

