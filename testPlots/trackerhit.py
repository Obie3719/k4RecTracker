#!/usr/bin/env python3
# trackerhit_sources_pie.py — works everywhere!

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter

# Try both possible podio APIs
try:
    from podio.reading import get_reader
    USE_OLD_API = True
    print("Using old podio API (get_reader)")
except ImportError:
    from podio import root_io
    USE_OLD_API = False
    print("Using new podio API (root_io.Reader)")

INPUT_FILE = "/ceph/omunkombwe/fit__idea_o1_v03_60.root"
MAX_EVENTS = 100
OUTPUT_FILE = "CDCHTracks_hit_sources.png"

# Open file with the correct API
if USE_OLD_API:
    reader = get_reader(INPUT_FILE)
    events = reader.get("events")
else:
    reader = root_io.Reader(INPUT_FILE)
    events = reader.get_entries("events")   # ← correct for new API

hit_sources = {}      # collectionID → name
all_coll_ids = []
total_hits = 0

print("Analyzing CDCHTracks hit sources...")
for iev, event in enumerate(events):
    if iev >= MAX_EVENTS:
        break
    if iev % 20 == 0:
        print(f"  event {iev}")

    # Build collection ID → name map (once)
    if not hit_sources:
        if USE_OLD_API:
            names = event.getCollectionNames()
        else:
            names = event.get_available_collections()
        for name in names:
            coll = event.get(name)
            if hasattr(coll, "getID"):
                hit_sources[coll.getID()] = name

    try:
        tracks = event.get("CDCHTracks")
    except:
        continue

    for track in tracks:
        for hit in track.getTrackerHits():
            oid = hit.getObjectID()
            coll_id = oid.collectionID
            all_coll_ids.append(coll_id)
            total_hits += 1

# Count and plot
count = Counter(all_coll_ids)

labels = []
sizes = []
for coll_id, n_hits in count.most_common():
    name = hit_sources.get(coll_id, f"ID={coll_id}")
    labels.append(f"{name}\n({n_hits:,} hits)")
    sizes.append(n_hits)

plt.figure(figsize=(10, 8))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, textprops={'fontsize': 13})

verdict = ("YES — CDCHTracks use ONLY DCH hits!" 
           if len(count) == 1 and any("DCH" in name for name in hit_sources.values()) 
           else "Multiple sub-detectors contribute!")

plt.title(f"Source of TrackerHits in CDCHTracks\n"
          f"{total_hits:,} hits from {min(iev+1, MAX_EVENTS)} events\n\n"
          f"{verdict}", 
          fontsize=14, pad=20)

plt.tight_layout()
plt.savefig(OUTPUT_FILE, dpi=200, bbox_inches='tight')
plt.close()

print(f"\nPlot saved: {OUTPUT_FILE}")
print(verdict)