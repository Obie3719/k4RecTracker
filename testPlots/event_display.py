#!/usr/bin/env python3
# event_display_digi_vs_sim.py
# Shows digitized hits (gray) and their corresponding simulated hits (red + lines)

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from podio.reading import get_reader
import edm4hep


MAX_EVENTS = 10
OUTPUT_DIR = "./event_displays_digi_vs_sim/"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def oid(obj):
    obj_id = obj.getObjectID()
    return (obj_id.collectionID, obj_id.index)


reader = get_reader("/ceph/omunkombwe/eezmu_fitter_edm4hep.root")
events = reader.get("events")


for iev, event in enumerate(events):
    if iev >= MAX_EVENTS:
        break

    print(f"\n=== Event {iev} ===")

    # Get digitized hits
    digi_coll = event.get("DCH_DigiCollection")

    # Get digitized hit positions
    digi_x = []
    digi_y = []
    digi_z = []
    digi_oid_to_idx = {}

    for i, hit in enumerate(digi_coll):
        pos = hit.getPosition()
        digi_x.append(pos.x)
        digi_y.append(pos.y)
        digi_z.append(pos.z)
        digi_oid_to_idx[oid(hit)] = i

    digi_x = np.array(digi_x)
    digi_y = np.array(digi_y)
    digi_z = np.array(digi_z)

    # 2. Get simulated hits
    sim_coll = event.get("DCHCollection")
    sim_x = []
    sim_y = []
    sim_z = []
    sim_oid_to_idx = {}

    for i, hit in enumerate(sim_coll):
        pos = hit.getPosition()
        sim_x.append(pos.x)
        sim_y.append(pos.y)
        sim_z.append(pos.z)
        sim_oid_to_idx[oid(hit)] = i


    sim_x = np.array(sim_x)
    sim_y = np.array(sim_y)
    sim_z = np.array(sim_z)


    # Get associations
    assoc_coll = event.get("DCH_DigiSimAssociationCollection")


    # Store pairs: (digi_index, sim_index)
    pairs = []
    for assoc in assoc_coll:
        digi_oid = oid(assoc.getFrom())   # digitized hit
        sim_oid  = oid(assoc.getTo())     # simulated hit
        if digi_oid in digi_oid_to_idx and sim_oid in sim_oid_to_idx:
            digi_idx = digi_oid_to_idx[digi_oid]
            sim_idx  = sim_oid_to_idx[sim_oid]
            pairs.append((digi_idx, sim_idx))



    print(f"  {len(digi_coll)} digitized hits")
    print(f"  {len(sim_coll)} simulated hits")
    print(f"  {len(pairs)} associations found")

       
       
    # ------------------- Plotting -------------------
    plt.figure(figsize=(14, 6))

    # XY view
    ax1 = plt.subplot(1, 2, 1)
    ax1.scatter(digi_x, digi_y, c='royalblue', s=25, alpha=0.85,
                label=f'Digi-hits : {len(digi_coll)}', zorder=1)
    ax1.scatter(sim_x,  sim_y,  c='red',       s=30, alpha=0.9,
                label=f'Sim-hits : {len(sim_coll)}',  zorder=3)



    # Draw connections
    for digi_idx, sim_idx in pairs:
        ax1.plot([digi_x[digi_idx], sim_x[sim_idx]],
                 [digi_y[digi_idx], sim_y[sim_idx]],
                 color='red', alpha=0.4, lw=0.8, zorder=2)



    ax1.set_xlabel("x [mm]")
    ax1.set_ylabel("y [mm]")
    ax1.set_title(f"Event {iev} – Digitized vs Simulated Hits (XY)")
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)


    # RZ view
    ax2 = plt.subplot(1, 2, 2)
    digi_r = np.sqrt(digi_x**2 + digi_y**2)
    sim_r  = np.sqrt(sim_x**2  + sim_y**2)



    ax2.scatter(digi_z, digi_r, c='royalblue', s=25, alpha=0.85)
    ax2.scatter(sim_z,  sim_r,  c='red',       s=30, alpha=0.9)


    # Draw connections
    for digi_idx, sim_idx in pairs:
        ax2.plot([digi_z[digi_idx], sim_z[sim_idx]],
                 [digi_r[digi_idx], sim_r[sim_idx]],
                 color='red', alpha=0.4, lw=0.8)



    ax2.set_xlabel("z [mm]")
    ax2.set_ylabel("R [mm]")
    ax2.set_title("RZ view")
    ax2.grid(True, alpha=0.3)

    
    legend_text = (
        f"Assoc   : {len(pairs)}"
    )
    
    
    ax1.legend(title=legend_text, title_fontsize=11, fontsize=10,
               loc="upper right", framealpha=0.95, fancybox=True, shadow=True)



    plt.suptitle(f"Event {iev} – DCH: Digitized vs Simulated Hits (Old-digitizer)", fontsize=14)
    plt.tight_layout()
    
    

    outfile = os.path.join(OUTPUT_DIR, f"event_{iev:04d}_digi_vs_sim.png")
    plt.savefig(outfile, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  → saved {outfile}")