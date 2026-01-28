#!/usr/bin/env python3

import math
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from podio.reading import get_reader
import edm4hep


OUTPUT_DIR = "./event_displays/"
os.makedirs(OUTPUT_DIR, exist_ok=True)


reader = get_reader("/ceph/omunkombwe/fit__idea_o1_v03_60.root")
events = reader.get("events")  

for iev, event in enumerate(events):
    if iev >= 6:
        break

    print(f"\n=== Event {iev} ===")
    
    digi_coll = event.get("DCH_DigiCollection")

    x = []
    y = []
    z = []
    oid_to_idx = {}  

    for i, hit in enumerate(digi_coll):
        pos = hit.getPosition()
        x.append(pos.x)
        y.append(pos.y)
        z.append(pos.z)
        oid = (hit.getObjectID().collectionID, hit.getObjectID().index)
        oid_to_idx[oid] = i

    x = np.array(x)
    y = np.array(y)
    z = np.array(z)

    print(f"  Total DCH digi hits: {len(digi_coll)}")

    tracks = event.get("CDCHTracks")
    print(f"  Total tracks: {len(tracks)}")

    plt.figure(figsize=(14, 6))

    #  XY view 
    ax1 = plt.subplot(1, 2, 1)
    ax1.scatter(x, y, c='royalblue', s=12, alpha=0.7, label='All VTX digi hits')

    colors = plt.cm.tab10(np.linspace(0, 1, len(tracks)))

    for itrk, track in enumerate(tracks):
        color = colors[itrk % 10]

        tx, ty = [], []
        for hit in track.getTrackerHits():
            oid = (hit.getObjectID().collectionID, hit.getObjectID().index)
            if oid in oid_to_idx:
                idx = oid_to_idx[oid]
                tx.append(x[idx])
                ty.append(y[idx])

        if len(tx) == 0:
            continue

        tx = np.array(tx)
        ty = np.array(ty)

        # Plot hits on this track
        ax1.scatter(tx, ty, c=[color], s=50, edgecolors='black', linewidth=0.5,
                    label=f'Track {itrk} ({len(tx)} hits)')

    
        #if len(tx) > 3:
        r = np.sqrt(tx**2 + ty**2)
        order = np.argsort(r)
        ax1.plot(tx[order], ty[order], color=color, lw=2.5, alpha=0.9)

    ax1.set_xlabel("x [mm]")
    ax1.set_ylabel("y [mm]")
    ax1.set_title(f"Event {iev} – DCH (Old-digitizer) (XY)")
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9, loc="upper right")

    # RZ view 
    ax2 = plt.subplot(1, 2, 2)
    r = np.sqrt(x**2 + y**2)
    ax2.scatter(z, r, c='royalblue', s=12, alpha=0.7)

    for itrk, track in enumerate(tracks):
        color = colors[itrk % 10]
        tz, tr = [], []
        for hit in track.getTrackerHits():
            oid = (hit.getObjectID().collectionID, hit.getObjectID().index)
            if oid in oid_to_idx:
                idx = oid_to_idx[oid]
                tz.append(z[idx])
                tr.append(r[idx])

        if len(tz) == 0:
            continue

        tz = np.array(tz)
        tr = np.array(tr)

        ax2.scatter(tz, tr, c=[color], s=50, edgecolors='black', linewidth=0.5)
        #if len(tz) > 3:
        order = np.argsort(tz)
        ax2.plot(tz[order], tr[order], color=color, lw=2.5)

    ax2.set_xlabel("z [mm]")
    ax2.set_ylabel("R [mm]")
    ax2.set_title("RZ view")
    ax2.grid(True, alpha=0.3)

    plt.suptitle(f"Event {iev} – {len(tracks)} reco track(s)", fontsize=14)
    plt.tight_layout()

    outfile = os.path.join(OUTPUT_DIR, f"event_{iev:04d}.png")
    plt.savefig(outfile, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"  → saved {outfile}")

print(f"\nAll done! Check folder: {OUTPUT_DIR}")