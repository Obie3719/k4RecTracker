#!/usr/bin/env python3
# fitted_tracks_display.py
# Shows Fitted_tracks_muon — the final result after track fitting

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from podio.reading import get_reader
import edm4hep

# --------------------------- CONFIG ---------------------------
INPUT_FILE = "/ceph/omunkombwe/fit__idea_o1_v03_60.root"   # or your new model
MAX_EVENTS = 10
OUTPUT_DIR = "./fitted_tracks_display/"
os.makedirs(OUTPUT_DIR, exist_ok=True)
# --------------------------------------------------------------

def get_track_parameters(track):
    """Extract track parameters from trackStatesAtBP (at beginning of plane)"""
    # Get the first track state (usually at BP - beginning of plane)
    track_states = track.getTrackStates()
    if len(track_states) == 0:
        return None
    
    ts = track_states[0]  # Assuming first is at BP
    d0 = ts.D0
    z0 = ts.Z0
    phi = ts.phi
    tanLambda = ts.tanLambda
    omega = ts.omega  # curvature = q / p
    chi2 = track.chi2
    ndf = track.ndf
    pT = abs(1 / omega) if omega != 0 else float('inf')
    chi2ndf = chi2 / ndf if ndf > 0 else 999
    
    return {
        'd0': d0,
        'z0': z0,
        'phi': phi,
        'tanLambda': tanLambda,
        'omega': omega,
        'pT': pT,
        'chi2ndf': chi2ndf,
        'ndf': ndf
    }

def get_position_at_z(params, z_target=0.0):
    """Extrapolate track to z = z_target using track parameters"""
    if params is None:
        return None
    
    d0 = params['d0']
    phi0 = params['phi']
    omega = params['omega']
    tanl = params['tanLambda']
    z0 = params['z0']

    if omega == 0:
        # Straight line approximation
        x = d0 * np.cos(phi0)
        y = d0 * np.sin(phi0)
        return x, y

    # Parametric helix equations
    dz = z_target - z0
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
        print("  No Fitted_tracks_muon collection → skip")
        continue

    if len(fitted_tracks) == 0:
        print("  No fitted tracks in this event")
        continue

    print(f"  Found {len(fitted_tracks)} fitted track(s)")

    # Get all digitized hits for background
    digi_coll = event.get("DCH_DigiCollection")
    digi_x = np.array([h.getPosition().x for h in digi_coll])
    digi_y = np.array([h.getPosition().y for h in digi_coll])
    digi_z = np.array([h.getPosition().z for h in digi_coll])
    digi_r = np.sqrt(digi_x**2 + digi_y**2)

    plt.figure(figsize=(14, 6))

    # --- XY view ---
    ax1 = plt.subplot(1, 2, 1)
    ax1.scatter(digi_x, digi_y, c='lightgray', s=8, alpha=0.6, label='All DCH digi hits')

    # --- RZ view ---
    ax2 = plt.subplot(1, 2, 2)
    ax2.scatter(digi_z, digi_r, c='lightgray', s=8, alpha=0.6)

    colors = plt.cm.viridis(np.linspace(0, 1, len(fitted_tracks)))

    legend_lines = []

    for i, track in enumerate(fitted_tracks):
        color = colors[i]

        # Get parameters
        params = get_track_parameters(track)
        if params is None:
            print(f"  Track {i}: No track states → skip")
            continue

        d0 = params['d0']
        z0 = params['z0']
        phi0 = params['phi']
        omega = params['omega']
        tanl = params['tanLambda']
        pT = params['pT']
        chi2ndf = params['chi2ndf']
        ndf = params['ndf']

        # Generate smooth helix for plotting
        z_vals = np.linspace(min(digi_z), max(digi_z), 200)
        x_vals = []
        y_vals = []
        r_vals = []

        for z in z_vals:
            x, y = get_position_at_z(params, z)
            if x is not None:
                x_vals.append(x)
                y_vals.append(y)
                r_vals.append(np.sqrt(x**2 + y**2))

        if len(x_vals) > 0:
            x_vals = np.array(x_vals)
            y_vals = np.array(y_vals)
            r_vals = np.array(r_vals)

            # Plot smooth fitted track
            ax1.plot(x_vals, y_vals, color=color, lw=3, alpha=0.9)
            ax2.plot(z_vals, r_vals, color=color, lw=3, alpha=0.9)

            # Add to legend
            label = (f"Track {i}\n"
                     f"pT = {pT:.2f} GeV\n"
                     f"d₀ = {d0:+.2f} mm\n"
                     f"χ²/ndf = {chi2ndf:.2f} ({ndf} dof)")
            legend_lines.append(plt.Line2D([0], [0], color=color, lw=3, label=label))

    ax1.set_xlabel("x [mm]")
    ax1.set_ylabel("y [mm]")
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.set_title(f"Event {iev} – Fitted Tracks (XY)")

    ax2.set_xlabel("z [mm]")
    ax2.set_ylabel("R [mm]")
    ax2.grid(True, alpha=0.3)
    ax2.set_title("RZ view")

    # Beautiful multi-line legend
    if legend_lines:
        ax1.legend(handles=legend_lines, loc="upper right", fontsize=9, framealpha=0.95)

    plt.suptitle(f"Event {iev} – {len([p for p in [get_track_parameters(t) for t in fitted_tracks] if p])} Fitted Track(s)", fontsize=14)
    plt.tight_layout()

    outfile = os.path.join(OUTPUT_DIR, f"fitted_event_{iev:04d}.png")
    plt.savefig(outfile, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  → saved {outfile}")

print(f"\nAll done! Check: {OUTPUT_DIR}")