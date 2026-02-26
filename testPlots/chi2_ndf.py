#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Overlay chi2/ndf distributions of reconstructed tracks per theta."""

import os
import re
import glob
from importlib import util

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =========================
# CONFIG
# =========================
BASE_DIR = "/ceph/omunkombwe/pi_Barrel_10/"
SUBDIR_GLOB = "deg*"
FIT_GLOB = "fit_*_merged.root"
FILENAME_RE = re.compile(r"fit_.*_theta(?P<theta>\d+)deg_.*_merged\.root")

TRACK_FILTER_MODE = "off"  # "off" | "inclusive" | "exclusive"
ACTIVE_SUBDETECTORS = ["DCH", "VTXB", "VTXD", "SIWRB", "SIWRD"]

X_MIN = 0.0
X_MAX = 5.0
N_BINS = 200

OUT_PNG = "chi2_ndf_reco_tracks_pi_barrel.png"

# Largest angle -> blue, second -> red, third -> black, smallest -> green.
ANGLE_PALETTE_DESC = ["blue", "red", "black", "limegreen"]
FALLBACK_COLORS = ["tab:purple", "tab:orange", "tab:brown", "tab:cyan", "tab:pink", "tab:gray"]


def load_mom_reso_module(script_dir):
    path = os.path.join(script_dir, "mom_reso.py")
    spec = util.spec_from_file_location("mom_reso_mod", path)
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_color_map(thetas):
    color_map = {}
    ordered = sorted(thetas, reverse=True)
    for idx, theta in enumerate(ordered):
        if idx < len(ANGLE_PALETTE_DESC):
            color_map[theta] = ANGLE_PALETTE_DESC[idx]
        else:
            color_map[theta] = FALLBACK_COLORS[(idx - len(ANGLE_PALETTE_DESC)) % len(FALLBACK_COLORS)]
    return color_map


def collect_chi2_ndf_values(m, filename, active_digi_names):
    reader = m.get_reader(filename)
    events = reader.get("events")

    values = []
    summary = {
        "events": 0,
        "events_with_tracks": 0,
        "tracks_before_filter": 0,
        "tracks_after_filter": 0,
        "tracks_valid_chi2ndf": 0,
    }

    for event in events:
        summary["events"] += 1

        tracks = m.get_tracks_from_event(event)
        if len(tracks) == 0:
            continue
        summary["events_with_tracks"] += 1
        summary["tracks_before_filter"] += len(tracks)

        tracks = m.filter_tracks_by_active_digis(
            event=event,
            tracks=tracks,
            active_digi_names=active_digi_names,
            mode=TRACK_FILTER_MODE,
        )
        summary["tracks_after_filter"] += len(tracks)

        for trk in tracks:
            chi2_ndf = m.track_chi2ndf(trk)
            if chi2_ndf is None:
                continue
            values.append(chi2_ndf)
            summary["tracks_valid_chi2ndf"] += 1

    return np.asarray(values, dtype=float), summary


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    m = load_mom_reso_module(here)

    sim_hits_names, assoc_names, active_digi_names = m.active_collection_names(ACTIVE_SUBDETECTORS)
    _ = (sim_hits_names, assoc_names)  # kept for consistency/debugging

    print("Active subdetectors:", ACTIVE_SUBDETECTORS)
    print("Track filter mode:", TRACK_FILTER_MODE)
    print("Track-filter digi collections:", active_digi_names)

    vals_by_theta = {}
    stats_by_theta = {}

    subdirs = sorted(glob.glob(os.path.join(BASE_DIR, SUBDIR_GLOB)))
    if not subdirs:
        raise FileNotFoundError(f"No subdirectories found under {BASE_DIR} with pattern {SUBDIR_GLOB}")

    for subdir in subdirs:
        fit_files = sorted(glob.glob(os.path.join(subdir, FIT_GLOB)))
        if not fit_files:
            print(f"Warning: no {FIT_GLOB} files in {subdir}")
            continue

        for file_path in fit_files:
            fname = os.path.basename(file_path)
            match = FILENAME_RE.match(fname)
            if not match:
                print(f"Warning: unrecognized filename format: {file_path}")
                continue

            theta = int(match.group("theta"))
            print(f"\nAnalyzing {file_path} (theta={theta} deg)")

            vals, summary = collect_chi2_ndf_values(m, file_path, active_digi_names)
            if vals.size == 0:
                print(f"  -> no valid chi2/ndf values in {fname}")
                continue

            vals_by_theta.setdefault(theta, []).append(vals)
            agg = stats_by_theta.setdefault(theta, {k: 0 for k in summary.keys()})
            for key, val in summary.items():
                agg[key] += int(val)

            in_range = int(np.count_nonzero((vals >= X_MIN) & (vals <= X_MAX)))
            print(
                "  summary: "
                f"events={summary['events']}, "
                f"tracks_before_filter={summary['tracks_before_filter']}, "
                f"tracks_after_filter={summary['tracks_after_filter']}, "
                f"valid_chi2ndf={summary['tracks_valid_chi2ndf']}, "
                f"in_plot_range={in_range}"
            )

    if not vals_by_theta:
        raise RuntimeError("No chi2/ndf values collected. Check inputs/collections/cuts.")

    theta_list = sorted(vals_by_theta.keys(), reverse=True)
    color_map = build_color_map(theta_list)
    bins = np.linspace(X_MIN, X_MAX, N_BINS + 1)

    plt.figure(figsize=(10, 7))
    for theta in theta_list:
        vals = np.concatenate(vals_by_theta[theta])
        if vals.size == 0:
            continue
        plt.hist(
            vals,
            bins=bins,
            histtype="step",
            linewidth=1.6,
            color=color_map[theta],
            label=f"{theta} deg",
        )

    plt.title("Pion Reconstructed Tracks Chi2 over nDof", fontsize=14)
    plt.xlabel(r"$\chi^2/\mathrm{ndf}$", fontsize=12)
    plt.ylabel("Entries", fontsize=12)
    plt.xlim(X_MIN, X_MAX)
    plt.grid(True, linestyle="--", alpha=0.45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=300)
    plt.close()
    print(f"\nSaved {OUT_PNG}")

    print("\nPer-theta totals:")
    for theta in theta_list:
        s = stats_by_theta[theta]
        total_vals = int(np.concatenate(vals_by_theta[theta]).size)
        print(
            f"  theta={theta:>3}°: "
            f"events={s['events']}, "
            f"tracks_before_filter={s['tracks_before_filter']}, "
            f"tracks_after_filter={s['tracks_after_filter']}, "
            f"valid_chi2ndf={total_vals}"
        )


if __name__ == "__main__":
    main()
