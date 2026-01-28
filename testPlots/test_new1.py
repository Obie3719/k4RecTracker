#!/usr/bin/

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from podio.reading import get_reader
import edm4hep


OUTPUT_DIR = "./tracking_display_sample/"
os.makedirs(OUTPUT_DIR, exist_ok=True)


SUBDETECTORS = {
    "VTX Barrel":       "VTXBDigis",
    "Drift Chamber":    "DCH_DigiCollection",
    "SiWrapper Barrel": "SiWrBDigis",
}


SUBDET_COLORS = {
    "VTX Barrel":       "red",
    "Drift Chamber":    "black",
    "SiWrapper Barrel": "green",
}


def oid(hit):
    oid = hit.getObjectID()
    return (oid.collectionID, oid.index)


reader = get_reader("/ceph/omunkombwe/eezmu_fitter_edm4hep.root")
events = reader.get("events")


for iev, event in enumerate(events):
    if iev >= 10:
        break


    print(f"\n=== Event {iev} ===")


    all_hits = {}
    oid_to_info = {}
    subdet_hit_count = {}


    for name, coll_name in SUBDETECTORS.items():
        coll = event.get(coll_name)
        x = [h.getPosition().x for h in coll]
        y = [h.getPosition().y for h in coll]
        z = [h.getPosition().z for h in coll]
        all_hits[name] = (np.array(x), np.array(y), np.array(z))
        subdet_hit_count[name] = len(coll)


        for i, h in enumerate(coll):
            oid_to_info[oid(h)] = (x[i], y[i], z[i], name)

        print(f"  {name}: {len(coll)} hits")


    tracks = event.get("CDCHTracks")
    #print(f"  Total tracks: {len(tracks)}")

    plt.figure(figsize=(16, 7))

    ax_xy = plt.subplot(1, 2, 1)
    ax_xy.set_aspect('equal')
    ax_xy.set_xlabel("x [mm]"); ax_xy.set_ylabel("y [mm]")
    ax_xy.set_title(f"Event {iev} – Full IDEA Tracker (XY)")

    ax_rz = plt.subplot(1, 2, 2)
    ax_rz.set_xlabel("z [mm]"); ax_rz.set_ylabel("R [mm]")
    ax_rz.set_title("RZ view")

    # Background hits
    for name, (x, y, z) in all_hits.items():
        r = np.sqrt(x**2 + y**2)
        ax_xy.scatter(x, y, c=SUBDET_COLORS[name], s=10, alpha=0.6)
        ax_rz.scatter(z, r, c=SUBDET_COLORS[name], s=10, alpha=0.6)

    
    track_colors = plt.cm.tab20(np.linspace(0, 1, len(tracks)))  
    legend_elements = []

    
    for name in SUBDETECTORS:
        legend_elements.append(
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=SUBDET_COLORS[name],
                       markersize=10, label=f"{name}: {subdet_hit_count[name]} hits")
        )

    
    
    for itrk, track in enumerate(tracks):
        color = track_colors[itrk]  

        hit_coords = []
        for hit in track.getTrackerHits():
            key = oid(hit)
            if key in oid_to_info:
                x, y, z, _ = oid_to_info[key]
                hit_coords.append((x, y, z))

        if not hit_coords:
            continue

        xs, ys, zs = zip(*hit_coords)
        xs = np.array(xs); ys = np.array(ys); zs = np.array(zs)
        rs = np.sqrt(xs**2 + ys**2)

        order_xy = np.argsort(rs)
        order_rz = np.argsort(zs)

        # Plot with exact color
        ax_xy.plot(xs[order_xy], ys[order_xy], color=color, lw=3, alpha=0.9)
        ax_rz.plot(zs[order_rz], rs[order_rz], color=color, lw=3, alpha=0.9)

        # Legend entry with EXACT same color
        legend_elements.append(
            plt.Line2D([0], [0], color=color, lw=6,
                       label=f"Track {itrk}: {len(hit_coords)} hits")
        )



    ax_xy.legend(handles=legend_elements, loc="upper right", fontsize=10, framealpha=0.95, ncol=2)
    ax_xy.grid(True, alpha=0.3); ax_rz.grid(True, alpha=0.3)


    plt.suptitle(f"Event {iev} – Full IDEA Tracker (New-digitizer)", fontsize=16)
    plt.tight_layout()


    outfile = os.path.join(OUTPUT_DIR, f"event_{iev:04d}_perfect.png")
    plt.savefig(outfile, dpi=220, bbox_inches='tight')
    plt.close()
    print(f"  → saved {outfile}")

print(f"\nDone! Check folder: {OUTPUT_DIR}")