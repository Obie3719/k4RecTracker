#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Plot truth-matched impact-parameter resolutions vs generated momentum.

This script intentionally reuses the same matching/filtering/binning logic from
mom_reso.py so momentum/angle/impact-parameter studies stay comparable.
"""

import os
import re
import glob
import math
import csv
from importlib import util

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =========================
# PHYSICS / ANALYSIS CONFIG
# =========================

# Input discovery
BASE_DIR = "/ceph/omunkombwe/pi_Barrel_10"  # Base directory containing subdirs with fit_*.root files
SUBDIR_GLOB = "deg*"
FIT_GLOB = "fit_*_merged.root"
FILENAME_RE = re.compile(r"fit_.*_theta(?P<theta>\d+)deg_.*_merged\.root")

# Truth and track selection
PDG_ABS = 211          # pion PDG ID    
GEN_STATUS = 1
ATIP_LOCATION = 1
MIN_PURITY = 0.75
MIN_NDF = 5
MAX_CHI2NDF = 10.0   # set to None to disable chi2/ndf upper cut

# Binning and stats
P_MIN_GEV = 0.2
P_MAX_GEV = 10.0
P_BINS = 60
MIN_TRACKS_PER_BIN = 120
SIGMA_METHOD = "q68"       # "q68" or "std"
BOOTSTRAP_ITERS = 80

TRACK_COLLECTION_CANDIDATES = ["Fitted_tracks"]

# For prompt gun samples, truth d0/z0 at IP are expected to be 0.
ASSUME_PROMPT_TRUTH = True
MM_TO_UM = 1000.0

# ===========================================================
# Subdetector config (keep aligned with mom_reso.py)
# ===========================================================
SUBDETECTORS = {
    "DCH": {
        "sim_hits": "DCHCollection",
        "assoc": "DCH_DigiSimAssociationCollection",
        "digi": "DCH_DigiCollection",
    },
    "VTXB": {
        "sim_hits": "VertexBarrelCollection",
        "assoc": "VTXBSimDigiLinks",
        "digi": "VTXBDigis",
    },
    "VTXD": {
        "sim_hits": "VertexEndcapCollection",
        "assoc": "VTXDSimDigiLinks",
        "digi": "VTXDDigis",
    },
    "SIWRB": {
        "sim_hits": "SiWrBCollection",
        "assoc": "SiWrBSimDigiLinks",
        "digi": "SiWrBDigis",
    },
    "SIWRD": {
        "sim_hits": "SiWrDCollection",
        "assoc": "SiWrDSimDigiLinks",
        "digi": "SiWrDDigis",
    },
}

# Default = all five combined
ACTIVE_SUBDETECTORS = ["DCH", "VTXB", "VTXD", "SIWRB", "SIWRD"]  # Only DCH for this analysis

# Track filter w.r.t ACTIVE_SUBDETECTORS digi collections:
#   "off"       : no track filtering by digi collection membership
#   "inclusive" : keep track if >=1 hit belongs to active digi collections
#   "exclusive" : keep track only if ALL its hits belong to active digi collections
TRACK_FILTER_MODE = "off"

# Output
OUT_D0 = "pi_D0_Resolution_Barrel.png"
OUT_Z0 = "pi_Z0_Resolution_Barrel.png"
OUT_D0_PT = "pi_D0_Resolution_vsPt_Barrel.png"
OUT_Z0_PT = "pi_Z0_Resolution_vsPt_Barrel.png"
OUT_D0_RESIDUAL = "pi_D0_Residuals_Barrel.png"
OUT_Z0_RESIDUAL = "pi_Z0_Residuals_Barrel.png"

CSV_D0 = "pi_idea_truthMatched_binned_metrics_d0.csv"
CSV_Z0 = "pi_idea_truthMatched_binned_metrics_z0.csv"
CSV_D0_PT = "pi_idea_truthMatched_binned_metrics_d0_pt.csv"
CSV_Z0_PT = "pi_idea_truthMatched_binned_metrics_z0_pt.csv"

# Largest angle -> blue, second -> red, third -> black, smallest -> green.
ANGLE_PALETTE_DESC = ["blue", "red", "black", "limegreen"]
FALLBACK_COLORS = ["tab:purple", "tab:orange", "tab:brown", "tab:cyan", "tab:pink", "tab:gray"]


def build_angle_color_map(thetas):
    color_map = {}
    ordered = sorted(thetas, reverse=True)
    for idx, theta in enumerate(ordered):
        if idx < len(ANGLE_PALETTE_DESC):
            color_map[theta] = ANGLE_PALETTE_DESC[idx]
        else:
            color_map[theta] = FALLBACK_COLORS[(idx - len(ANGLE_PALETTE_DESC)) % len(FALLBACK_COLORS)]
    return color_map


def load_mom_reso_module(script_dir):
    path = os.path.join(script_dir, "mom_reso.py")
    spec = util.spec_from_file_location("mom_reso_mod", path)
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pick_track_state_d0_z0(m, track, atip_location):
    states = m.get_track_states(track)
    if len(states) == 0:
        return None

    chosen = None
    for st in states:
        loc = m.get_value(st, "location", "getLocation")
        if loc == atip_location:
            chosen = st
            break
    if chosen is None:
        chosen = states[0]

    d0 = m.get_value(chosen, "D0", "getD0")
    z0 = m.get_value(chosen, "Z0", "getZ0")
    if d0 is None or z0 is None:
        return None

    return float(d0), float(z0)


def calculate_event_ip_residuals(m, filename, sim_hits_names, assoc_names, active_digi_names):
    reader = m.get_reader(filename)
    events = reader.get("events")

    p_true_vals = []
    pt_true_vals = []
    residuals_d0 = []
    residuals_z0 = []

    n_events = 0
    n_has_mc = 0
    n_has_tracks = 0
    n_subdet_pass = 0
    n_matched_raw = 0
    n_matched_quality = 0
    n_quality_rejected = 0

    for event in events:
        n_events += 1

        mc_particles = m.get_collection(event, "MCParticles")
        if mc_particles is None or len(mc_particles) == 0:
            continue
        n_has_mc += 1

        tracks = m.get_tracks_from_event(event)
        if len(tracks) == 0:
            continue
        n_has_tracks += 1

        tracks = m.filter_tracks_by_active_digis(
            event=event,
            tracks=tracks,
            active_digi_names=active_digi_names,
            mode=TRACK_FILTER_MODE,
        )
        if len(tracks) == 0:
            continue
        n_subdet_pass += 1

        pair_any, pair_quality, _quality_stats = m.select_best_truth_matched_pair(
            event=event,
            tracks=tracks,
            mc_particles=mc_particles,
            sim_hits_names=sim_hits_names,
            assoc_names=assoc_names,
            min_ndf=MIN_NDF,
            max_chi2ndf=MAX_CHI2NDF,
        )
        if pair_any is None:
            continue
        n_matched_raw += 1

        if pair_quality is None:
            n_quality_rejected += 1
            continue
        n_matched_quality += 1

        mc_truth, best_track = pair_quality

        mom = m.get_vector3(m.get_value(mc_truth, "momentum", "getMomentum"))
        if mom is None:
            continue
        px, py, pz = mom
        p_true = math.sqrt(px * px + py * py + pz * pz)
        pt_true = math.hypot(px, py)
        if p_true <= 0 or pt_true <= 0:
            continue

        state = pick_track_state_d0_z0(m, best_track, ATIP_LOCATION)
        if state is None:
            continue
        d0_rec_mm, z0_rec_mm = state

        if ASSUME_PROMPT_TRUTH:
            d0_res_mm = d0_rec_mm
            z0_res_mm = z0_rec_mm
        else:
            # Placeholder for non-prompt truth handling.
            d0_res_mm = d0_rec_mm
            z0_res_mm = z0_rec_mm

        p_true_vals.append(p_true)
        pt_true_vals.append(pt_true)
        residuals_d0.append(d0_res_mm * MM_TO_UM)
        residuals_z0.append(z0_res_mm * MM_TO_UM)

    if len(p_true_vals) == 0:
        return None

    summary = {
        "total": n_events,
        "with_mc": n_has_mc,
        "with_tracks": n_has_tracks,
        "subdet_pass": n_subdet_pass,
        "matched_raw": n_matched_raw,
        "matched_quality": n_matched_quality,
        "quality_rejected": n_quality_rejected,
        "matched": n_matched_quality,
        "used": len(p_true_vals),
    }

    return (
        np.asarray(p_true_vals, dtype=float),
        np.asarray(pt_true_vals, dtype=float),
        np.asarray(residuals_d0, dtype=float),
        np.asarray(residuals_z0, dtype=float),
        summary,
    )


def write_metrics_csv(path, theta_to_rows, quantity_name):
    fieldnames = [
        "theta_deg",
        "quantity",
        "bin_lo",
        "bin_hi",
        "bin_center",
        "n",
        "q16",
        "q84",
        "sigma68",
        "std",
        "sigma",
        "sigma_err",
    ]

    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for theta in sorted(theta_to_rows.keys()):
            for r in theta_to_rows[theta]:
                w.writerow(
                    {
                        "theta_deg": theta,
                        "quantity": quantity_name,
                        "bin_lo": r["lo"],
                        "bin_hi": r["hi"],
                        "bin_center": r["center"],
                        "n": r["n"],
                        "q16": r["q16"],
                        "q84": r["q84"],
                        "sigma68": r["sigma68"],
                        "std": r["std"],
                        "sigma": r["sigma"],
                        "sigma_err": r["sigma_err"],
                    }
                )


def plot_residual_overlay(outpath, title, xlabel, residuals_by_theta):
    all_vals = []
    for theta in residuals_by_theta:
        vals = np.asarray(residuals_by_theta[theta], dtype=float)
        if vals.size > 0:
            all_vals.append(vals)

    if not all_vals:
        print(f"Skip {outpath}: no residuals")
        return

    arr_all = np.concatenate(all_vals)
    lo, hi = np.percentile(arr_all, [0.5, 99.5])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = -1.0, 1.0
    pad = 0.06 * (hi - lo)
    bins = np.linspace(lo - pad, hi + pad, 140)

    theta_list = sorted(residuals_by_theta.keys(), reverse=True)
    color_map = build_angle_color_map(theta_list)

    plt.figure(figsize=(10, 7))
    for theta in theta_list:
        vals = np.asarray(residuals_by_theta[theta], dtype=float)
        if vals.size == 0:
            continue
        plt.hist(
            vals,
            bins=bins,
            histtype="step",
            density=True,
            linewidth=1.8,
            color=color_map[theta],
            label=f"Theta = {theta}°",
        )

    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel("Normalized entries", fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()
    print(f"Saved {outpath}")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    m = load_mom_reso_module(here)

    # Keep mom_reso helpers configured consistently.
    m.PDG_ABS = PDG_ABS
    m.GEN_STATUS = GEN_STATUS
    m.MIN_PURITY = MIN_PURITY
    m.MIN_NDF = MIN_NDF
    m.MAX_CHI2NDF = MAX_CHI2NDF
    m.TRACK_COLLECTION_CANDIDATES = TRACK_COLLECTION_CANDIDATES
    m.SUBDETECTORS = SUBDETECTORS

    sim_hits_names, assoc_names, active_digi_names = m.active_collection_names(ACTIVE_SUBDETECTORS)

    print("Active subdetectors:", ACTIVE_SUBDETECTORS)
    print("Track filter mode:", TRACK_FILTER_MODE)
    print("Fit-quality cuts:", f"MIN_NDF={MIN_NDF}, MAX_CHI2NDF={MAX_CHI2NDF}")
    print("Truth-matching sim-hit collections:", sim_hits_names)
    print("Truth-matching association collections:", assoc_names)
    print("Track-filter digi collections:", active_digi_names)
    print("Assume prompt truth d0/z0 = 0:", ASSUME_PROMPT_TRUTH)

    results_by_theta = {}

    subdirs = sorted(glob.glob(os.path.join(BASE_DIR, SUBDIR_GLOB)))
    if not subdirs:
        raise FileNotFoundError(
            f"No subdirectories found under {BASE_DIR} with pattern {SUBDIR_GLOB}"
        )

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

            res = calculate_event_ip_residuals(
                m=m,
                filename=file_path,
                sim_hits_names=sim_hits_names,
                assoc_names=assoc_names,
                active_digi_names=active_digi_names,
            )
            if res is None:
                print(f"  -> no usable matched events in {fname}")
                continue

            p_true, pt_true, residuals_d0, residuals_z0, summary = res
            print(
                "  summary: "
                f"total={summary['total']}, "
                f"with_mc={summary['with_mc']}, "
                f"with_tracks={summary['with_tracks']}, "
                f"subdet_pass={summary['subdet_pass']}, "
                f"matched_raw={summary['matched_raw']}, "
                f"matched_quality={summary['matched_quality']}, "
                f"quality_rejected={summary['quality_rejected']}, "
                f"used={summary['used']}"
            )

            store = results_by_theta.setdefault(
                theta, {"p_true": [], "pt_true": [], "res_d0": [], "res_z0": []}
            )
            store["p_true"].append(p_true)
            store["pt_true"].append(pt_true)
            store["res_d0"].append(residuals_d0)
            store["res_z0"].append(residuals_z0)

    if not results_by_theta:
        raise RuntimeError("No results accumulated. Check input files/collections/cuts.")

    bin_edges = np.logspace(np.log10(P_MIN_GEV), np.log10(P_MAX_GEV), P_BINS + 1)

    rows_d0_by_theta = {}
    rows_z0_by_theta = {}
    rows_d0_pt_by_theta = {}
    rows_z0_pt_by_theta = {}
    curve_d0 = {}
    err_d0 = {}
    curve_z0 = {}
    err_z0 = {}
    curve_d0_pt = {}
    err_d0_pt = {}
    curve_z0_pt = {}
    err_z0_pt = {}
    residuals_d0_by_theta = {}
    residuals_z0_by_theta = {}

    for theta, vals in sorted(results_by_theta.items()):
        p_true = np.concatenate(vals["p_true"]) if vals["p_true"] else np.asarray([], dtype=float)
        pt_true = np.concatenate(vals["pt_true"]) if vals["pt_true"] else np.asarray([], dtype=float)
        res_d0 = np.concatenate(vals["res_d0"]) if vals["res_d0"] else np.asarray([], dtype=float)
        res_z0 = np.concatenate(vals["res_z0"]) if vals["res_z0"] else np.asarray([], dtype=float)

        if p_true.size == 0:
            continue

        rows_d0 = m.binned_metrics(
            p_true,
            res_d0,
            bin_edges,
            MIN_TRACKS_PER_BIN,
            method=SIGMA_METHOD,
            n_bootstrap=BOOTSTRAP_ITERS,
            seed=6000 + theta,
        )
        rows_z0 = m.binned_metrics(
            p_true,
            res_z0,
            bin_edges,
            MIN_TRACKS_PER_BIN,
            method=SIGMA_METHOD,
            n_bootstrap=BOOTSTRAP_ITERS,
            seed=7000 + theta,
        )
        s = math.sin(math.radians(theta))
        if s > 0.0:
            pt_edges = bin_edges * s
            rows_d0_pt = m.binned_metrics(
                pt_true,
                res_d0,
                pt_edges,
                MIN_TRACKS_PER_BIN,
                method=SIGMA_METHOD,
                n_bootstrap=BOOTSTRAP_ITERS,
                seed=8000 + theta,
            )
            rows_z0_pt = m.binned_metrics(
                pt_true,
                res_z0,
                pt_edges,
                MIN_TRACKS_PER_BIN,
                method=SIGMA_METHOD,
                n_bootstrap=BOOTSTRAP_ITERS,
                seed=9000 + theta,
            )
        else:
            rows_d0_pt = []
            rows_z0_pt = []

        rows_d0_by_theta[theta] = rows_d0
        rows_z0_by_theta[theta] = rows_z0
        rows_d0_pt_by_theta[theta] = rows_d0_pt
        rows_z0_pt_by_theta[theta] = rows_z0_pt
        residuals_d0_by_theta[theta] = res_d0
        residuals_z0_by_theta[theta] = res_z0

        x, y, ye = m.rows_to_curve(rows_d0)
        curve_d0[theta] = (x, y)
        err_d0[theta] = ye

        x, y, ye = m.rows_to_curve(rows_z0)
        curve_z0[theta] = (x, y)
        err_z0[theta] = ye

        x, y, ye = m.rows_to_curve(rows_d0_pt)
        curve_d0_pt[theta] = (x, y)
        err_d0_pt[theta] = ye

        x, y, ye = m.rows_to_curve(rows_z0_pt)
        curve_z0_pt[theta] = (x, y)
        err_z0_pt[theta] = ye

        print(
            f"Theta {theta:>3}°: bins(d0,p)={len(rows_d0)}, bins(z0,p)={len(rows_z0)}, "
            f"bins(d0,pT)={len(rows_d0_pt)}, bins(z0,pT)={len(rows_z0_pt)}, "
            f"entries={len(p_true)}"
        )

    write_metrics_csv(CSV_D0, rows_d0_by_theta, "delta_d0_um")
    write_metrics_csv(CSV_Z0, rows_z0_by_theta, "delta_z0_um")
    write_metrics_csv(CSV_D0_PT, rows_d0_pt_by_theta, "delta_d0_um_vs_pt")
    write_metrics_csv(CSV_Z0_PT, rows_z0_pt_by_theta, "delta_z0_um_vs_pt")
    print(f"Saved {CSV_D0}")
    print(f"Saved {CSV_Z0}")
    print(f"Saved {CSV_D0_PT}")
    print(f"Saved {CSV_Z0_PT}")

    m.plot_curves(
        outpath=OUT_D0,
        title=r"Impact Parameter Resolution ($d_0$)",
        xlabel=r"Generated Momentum $p$ [GeV/c]",
        ylabel=r"$\sigma_{d_0}$ [$\mu$m]",
        curve_by_theta=curve_d0,
        yerr_by_theta=err_d0,
    )
    m.plot_curves(
        outpath=OUT_Z0,
        title=r"Impact Parameter Resolution ($z_0$)",
        xlabel=r"Generated Momentum $p$ [GeV/c]",
        ylabel=r"$\sigma_{z_0}$ [$\mu$m]",
        curve_by_theta=curve_z0,
        yerr_by_theta=err_z0,
    )
    m.plot_curves(
        outpath=OUT_D0_PT,
        title=r"Impact Parameter Resolution ($d_0$)",
        xlabel=r"Generated Transverse Momentum $p_T$ [GeV/c]",
        ylabel=r"$\sigma_{d_0}$ [$\mu$m]",
        curve_by_theta=curve_d0_pt,
        yerr_by_theta=err_d0_pt,
    )
    m.plot_curves(
        outpath=OUT_Z0_PT,
        title=r"Impact Parameter Resolution ($z_0$)",
        xlabel=r"Generated Transverse Momentum $p_T$ [GeV/c]",
        ylabel=r"$\sigma_{z_0}$ [$\mu$m]",
        curve_by_theta=curve_z0_pt,
        yerr_by_theta=err_z0_pt,
    )
    plot_residual_overlay(
        outpath=OUT_D0_RESIDUAL,
        title=r"$d_0$ Residual Distributions",
        xlabel=r"$\Delta d_0$ [$\mu$m]",
        residuals_by_theta=residuals_d0_by_theta,
    )
    plot_residual_overlay(
        outpath=OUT_Z0_RESIDUAL,
        title=r"$z_0$ Residual Distributions",
        xlabel=r"$\Delta z_0$ [$\mu$m]",
        residuals_by_theta=residuals_z0_by_theta,
    )


if __name__ == "__main__":
    main()
