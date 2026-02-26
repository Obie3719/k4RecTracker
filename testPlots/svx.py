#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Overlay: number of good SVX hits per fitted track, per theta."""

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
BASE_DIR = "/ceph/omunkombwe/pi_Grand/barrel"
SUBDIR_GLOB = "deg*"
FIT_GLOB = "fit_*_merged.root"
FILENAME_RE = re.compile(r"fit_.*_theta(?P<theta>\d+)deg_.*_merged\.root")

TRACK_FILTER_MODE = "off"  # "off" | "inclusive" | "exclusive"
ACTIVE_SUBDETECTORS = ["DCH", "VTXB", "VTXD", "SIWRB", "SIWRD"]

# SVX ~= vertex silicon in this setup
SVX_DIGI_COLLECTIONS = ["VTXBDigis", "VTXDDigis"]
KEEP_ONLY_TRACKS_WITH_SVX = False

X_MIN = 0.0
X_MAX = 16.0
N_BINS = 64

OUT_PNG = "svx_hits_per_track.png"

# Match reference palette convention.
ANGLE_COLOR_FIXED = {
    90: "blue",
    75: "red",
    60: "black",
    45: "limegreen",
    40: "blue",
    30: "red",
    20: "black",
    10: "limegreen",
}
FALLBACK_COLORS = ["tab:purple", "tab:orange", "tab:brown", "tab:cyan", "tab:pink", "tab:gray"]


def load_mom_reso_module(script_dir):
    path = os.path.join(script_dir, "mom_reso.py")
    spec = util.spec_from_file_location("mom_reso_mod", path)
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_color_map(thetas):
    color_map = {}
    fallback_idx = 0
    for theta in sorted(thetas, reverse=True):
        if theta in ANGLE_COLOR_FIXED:
            color_map[theta] = ANGLE_COLOR_FIXED[theta]
        else:
            color_map[theta] = FALLBACK_COLORS[fallback_idx % len(FALLBACK_COLORS)]
            fallback_idx += 1
    return color_map


def build_oid_set(m, event, collection_names):
    oids = set()
    for name in collection_names:
        coll = m.get_collection(event, name)
        if coll is None:
            continue
        for hit in coll:
            oids.add(m.oid(hit))
    return oids


def collect_hits_per_track(m, filename, active_digi_names):
    reader = m.get_reader(filename)
    events = reader.get("events")

    counts = []
    summary = {
        "events": 0,
        "events_with_tracks": 0,
        "tracks_before_filter": 0,
        "tracks_after_filter": 0,
        "tracks_with_svx": 0,
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

        svx_oids = build_oid_set(m, event, SVX_DIGI_COLLECTIONS)

        for trk in tracks:
            n_svx = 0
            for hit in m.iter_tracker_hits(trk):
                if m.oid(hit) in svx_oids:
                    n_svx += 1

            if KEEP_ONLY_TRACKS_WITH_SVX and n_svx <= 0:
                continue

            counts.append(n_svx)
            if n_svx > 0:
                summary["tracks_with_svx"] += 1

    return np.asarray(counts, dtype=float), summary


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    m = load_mom_reso_module(here)

    sim_hits_names, assoc_names, active_digi_names = m.active_collection_names(ACTIVE_SUBDETECTORS)
    _ = (sim_hits_names, assoc_names)

    print("Active subdetectors:", ACTIVE_SUBDETECTORS)
    print("Track filter mode:", TRACK_FILTER_MODE)
    print("Track-filter digi collections:", active_digi_names)
    print("SVX digi collections:", SVX_DIGI_COLLECTIONS)
    print("Keep only tracks with SVX hits:", KEEP_ONLY_TRACKS_WITH_SVX)

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

            vals, summary = collect_hits_per_track(m, file_path, active_digi_names)
            if vals.size == 0:
                print(f"  -> no SVX-hit track values in {fname}")
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
                f"tracks_with_svx={summary['tracks_with_svx']}, "
                f"in_plot_range={in_range}"
            )

    if not vals_by_theta:
        raise RuntimeError("No SVX-hit values collected. Check inputs/collections/cuts.")

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

    plt.title("Number of good svx hits per fitted tracks", fontsize=14)
    plt.xlabel("Number of svx hits per tracks", fontsize=12)
    plt.ylabel("entries", fontsize=12)
    plt.xlim(X_MIN, X_MAX)
    plt.ticklabel_format(axis="y", style="sci", scilimits=(3, 3), useMathText=True)
    plt.grid(True, linestyle="--", alpha=0.45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=300)
    plt.close()
    print(f"\nSaved {OUT_PNG}")


if __name__ == "__main__":
    main()

