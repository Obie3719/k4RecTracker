#!/usr/bin/env python3
# residuals.py — Compute and plot residuals between DCH digitized hits and simulated hits

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from podio.reading import get_reader
import edm4hep

MAX_EVENTS   = 5000
OUTPUT_FILE  = "dch_residuals.png"


def oid(obj):
    obj_id = obj.getObjectID()
    return (obj_id.collectionID, obj_id.index)


reader = get_reader("/ceph/omunkombwe/fit__idea_o1_v03_60.root")
events = reader.get("events")


delta_x = []
delta_y = []
delta_z = []
delta_r = []
n_associations = 0


for iev, event in enumerate(events):
    if iev >= MAX_EVENTS:
        break
    if iev % 500 == 0:
        print(f"  event {iev}")

    digi_coll = event.get("DCH_DigiCollection")
    sim_coll  = event.get("DCHCollection")
    assoc_coll = event.get("DCH_DigiSimAssociationCollection")

    digi_pos = {oid(h): h.getPosition() for h in digi_coll}
    sim_pos  = {oid(h): h.getPosition() for h in sim_coll}

    for assoc in assoc_coll:
        digi_oid = oid(assoc.getFrom())
        sim_oid  = oid(assoc.getTo())
        if digi_oid not in digi_pos or sim_oid not in sim_pos:
            continue

        pd = digi_pos[digi_oid]
        ps = sim_pos[sim_oid]

        dx = pd.x - ps.x
        dy = pd.y - ps.y
        dz = pd.z - ps.z
        dr = np.sqrt(dx**2 + dy**2)

        delta_x.append(dx)
        delta_y.append(dy)
        delta_z.append(dz)
        delta_r.append(dr)
        
        n_associations += 1


delta_x = np.array(delta_x)
delta_y = np.array(delta_y)
delta_z = np.array(delta_z)
delta_r = np.array(delta_r)

print(f"\nDone! {n_associations} associated hits collected.")

# Plotting
plt.figure(figsize=(12, 9))


bins_x = np.linspace(-20, 20, 120)
bins_y = np.linspace(-20, 20, 120)
bins_z = np.linspace(-100, 100, 150)
bins_r = np.linspace(0, 20, 100)   


def plot_hist(ax, data, bins, title, xlabel):
    ax.hist(data, bins=bins) #histtype='step', linewidth=2.2, color='navy'
    ax.set_title(title, fontsize=14, pad=12)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Number of hits", fontsize=12)
    ax.grid(True, alpha=0.3)



ax1 = plt.subplot(2, 2, 1)
plot_hist(ax1, delta_x, bins_x, "Residual in x (digi − sim)", "Δx [mm]")

ax2 = plt.subplot(2, 2, 2)
plot_hist(ax2, delta_y, bins_y, "Residual in y (digi − sim)", "Δy [mm]")

ax3 = plt.subplot(2, 2, 3)
plot_hist(ax3, delta_z, bins_z, "Residual in z (digi − sim)", "Δz [mm]")

ax4 = plt.subplot(2, 2, 4)
plot_hist(ax4, delta_r, bins_r, "Transverse residual ΔR", "ΔR = √(Δx² + Δy²) [mm]")



plt.suptitle(f"DCH Digitization Residuals(Old-digitizer)\n"
             f"{n_associations:,} associated hits from {min(MAX_EVENTS, iev):,} events",
             fontsize=16, y=0.98)



plt.tight_layout(rect=[0, 0.02, 1, 0.95])
plt.savefig(OUTPUT_FILE, dpi=250, bbox_inches='tight')
plt.close()

print(f"Plot saved → {OUTPUT_FILE}")