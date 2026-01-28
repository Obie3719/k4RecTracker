#!/usr/bin/env python3
# fitted_tracks_display.py — FINAL BULLETPROOF VERSION

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from podio.reading import get_reader
import edm4hep

INPUT_FILE = "/ceph/omunkombwe/fit_idea_o1_v03_new.root"
MAX_EVENTS = 10
OUTPUT_DIR = "./fitted_tracks_display_new/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_track_parameters(track):
    """Safely extract parameters from Fitted_tracks_muon"""
    states = track.getTrackStates()
    if len(states) == 0:
        return None

    ts = states[0]  # first state (usually at IP or beginning)

    # Extract track state parameters
    d0       = float(ts.D0)
    z0       = float(ts.Z0)
    phi      = float(ts.phi)
    omega    = float(ts.omega)
    tanl     = float(ts.tanLambda)

    # Extract chi2 and ndf safely (cppyy returns overload objects)
    chi2 = float(track.getChi2())
    ndf  = int(track.getNdf())        # ← force int conversion
    chi2ndf = chi2 / ndf if ndf > 0 else 999.0

    pT = abs(1.0 / omega) if abs(omega) > 1e-10 else float('inf')

    return {
        'd0': d0,
        'z0': z0,
        'phi': phi,
        'tanLambda': tanl,
        'omega': omega,
        'pT': pT,
        'chi2ndf': chi2ndf,
        'ndf': ndf
    }

def helix_point(params, z):
    """Return (x, y) at given z using track parameters"""
    d0, phi0, omega, tanl, z0 = params['d0'], params['phi'], params['omega'], params['tanLambda'], params['z0']
    dz = z - z0
    if abs(omega) < 1e-10:
        return d0 * np.cos(phi0), d0 * np.sin(phi0)
    theta = omega * dz / tanl
    x = d0 * np.cos(phi0) + (1/omega) * (np.sin(phi0 + theta) - np.sin(phi0))
    y = d0 * np.sin(phi0) - (1/omega) * (np.cos(phi0 + theta) - np.cos(phi0))
    return x, y

reader = get_reader(INPUT_FILE)
events = reader.get("events")

for iev, event in enumerate(events):
    if iev >= MAX_EVENTS:
        break

    print(f"\n=== Event {iev} ===")
    try:
        fitted_tracks = event.get("Fitted_tracks_muon")
    except:
        print("  No Fitted_tracks_muon → skip")
        continue

    if len(fitted_tracks) == 0:
        continue

    print(f"  Found {len(fitted_tracks)} fitted track(s)")

    # Background hits
    digi = event.get("DCH_DigiCollection")
    digi_x = np.array([h.getPosition().x for h in digi])
    digi_y = np.array([h.getPosition().y for h in digi])
    digi_z = np.array([h.getPosition().z for h in digi])
    digi_r = np.sqrt(digi_x**2 + digi_y**2)

    plt.figure(figsize=(14, 6))

    ax_xy = plt.subplot(1, 2, 1)
    ax_xy.scatter(digi_x, digi_y, c='lightgray', s=8, alpha=0.6)

    ax_rz = plt.subplot(1, 2, 2)
    ax_rz.scatter(digi_z, digi_r, c='lightgray', s=8, alpha=0.6)

    colors = plt.cm.viridis(np.linspace(0, 1, len(fitted_tracks)))
    legend_entries = []

    for i, trk in enumerate(fitted_tracks):
        params = get_track_parameters(trk)
        if params is None:
            continue

        color = colors[i]
        z_vals = np.linspace(digi_z.min(), digi_z.max(), 300)
        x_vals, y_vals, r_vals = [], [], []

        for z in z_vals:
            x, y = helix_point(params, z)
            x_vals.append(x)
            y_vals.append(y)
            r_vals.append(np.sqrt(x**2 + y**2))

        x_vals = np.array(x_vals)
        y_vals = np.array(y_vals)
        r_vals = np.array(r_vals)

        ax_xy.plot(x_vals, y_vals, color=color, lw=3, alpha=0.9)
        ax_rz.plot(z_vals, r_vals, color=color, lw=3, alpha=0.9)

        label = (f"Track {i}\n"
                 f"pT = {params['pT']:.2f} GeV\n"
                 f"d₀ = {params['d0']:+.2f} mm\n"
                 f"χ²/ndf = {params['chi2ndf']:.2f}")
        legend_entries.append(plt.Line2D([0], [0], color=color, lw=3, label=label))

    ax_xy.set_xlabel("x [mm]"); ax_xy.set_ylabel("y [mm]"); ax_xy.set_aspect('equal'); ax_xy.grid(alpha=0.3)
    ax_rz.set_xlabel("z [mm]"); ax_rz.set_ylabel("R [mm]"); ax_rz.grid(alpha=0.3)

    ax_xy.legend(handles=legend_entries, loc="upper right", fontsize=10, framealpha=0.95)

    plt.suptitle(f"Event {iev} – {len(legend_entries)} Fitted Track(s)", fontsize=14)
    plt.tight_layout()

    outfile = os.path.join(OUTPUT_DIR, f"fitted_event_{iev:04d}.png")
    plt.savefig(outfile, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  → saved {outfile}")

print(f"\nDone! Check folder: {OUTPUT_DIR}")