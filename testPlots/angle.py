#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Plot truth-matched angular resolutions (phi/theta) vs generated momentum.

This script intentionally reuses the same matching/filtering/binning logic from
mom_reso.py so momentum and angle studies are directly comparable.
"""

import os
import re
import glob
import math
import csv
from importlib import util

import numpy as np


# =========================
# PHYSICS / ANALYSIS CONFIG
# =========================

# Input discovery
BASE_DIR = "/ceph/omunkombwe/pi_Barrel_10"  # Base directory containing subdirs with fit_*.root files
SUBDIR_GLOB = "deg*"
FIT_GLOB = "fit_*_merged.root"
FILENAME_RE = re.compile(r"fit_.*_theta(?P<theta>\d+)deg_.*_merged\.root")

# Truth and track selection
PDG_ABS = 211           # pion PDG ID
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
ACTIVE_SUBDETECTORS = ["DCH", "VTXB", "VTXD", "SIWRB", "SIWRD"]

# Track filter w.r.t ACTIVE_SUBDETECTORS digi collections:
#   "off"       : no track filtering by digi collection membership
#   "inclusive" : keep track if >=1 hit belongs to active digi collections
#   "exclusive" : keep track only if ALL its hits belong to active digi collections
TRACK_FILTER_MODE = "off"

# Output
OUT_THETA = "pi_Theta_Resolution_Barrel.png"
OUT_PHI = "pi_Phi_Resolution_Barrel.png"
OUT_PHI_MODPI = "pi_PhiMod_Resolution_Barrel.png"

CSV_THETA = "pi_idea_truthMatched_binned_metrics_theta.csv"
CSV_PHI = "pi_idea_truthMatched_binned_metrics_phi.csv"
CSV_PHI_MODPI = "pi_idea_truthMatched_binned_metrics_phi_modpi.csv"


def load_mom_reso_module(script_dir):
    path = os.path.join(script_dir, "mom_reso.py")
    spec = util.spec_from_file_location("mom_reso_mod", path)
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def wrap_phi(dphi):
    while dphi > math.pi:
        dphi -= 2.0 * math.pi
    while dphi < -math.pi:
        dphi += 2.0 * math.pi
    return dphi


def wrap_phi_mod_pi(dphi):
    # Treat phi and phi±pi as equivalent direction hypotheses.
    while dphi > 0.5 * math.pi:
        dphi -= math.pi
    while dphi < -0.5 * math.pi:
        dphi += math.pi
    return dphi


def pick_track_state_phi_tanlambda(m, track, atip_location):
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

    phi = m.get_value(chosen, "phi", "getPhi")
    tan_lambda = m.get_value(chosen, "tanLambda", "getTanLambda")
    if phi is None or tan_lambda is None:
        return None

    return float(phi), float(tan_lambda)


def calculate_event_angular_residuals(m, filename, sim_hits_names, assoc_names, active_digi_names):
    reader = m.get_reader(filename)
    events = reader.get("events")

    p_true_vals = []
    residuals_phi = []
    residuals_phi_modpi = []
    residuals_theta = []

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
        if p_true <= 0:
            continue

        phi_true = math.atan2(py, px)
        theta_true = math.atan2(math.hypot(px, py), pz)

        state = pick_track_state_phi_tanlambda(m, best_track, ATIP_LOCATION)
        if state is None:
            continue

        phi_rec, tan_lambda = state
        theta_rec = math.atan2(1.0, tan_lambda)

        dphi = wrap_phi(phi_rec - phi_true)

        p_true_vals.append(p_true)
        residuals_phi.append(dphi)
        residuals_phi_modpi.append(wrap_phi_mod_pi(dphi))
        residuals_theta.append(theta_rec - theta_true)

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
        np.asarray(residuals_phi, dtype=float),
        np.asarray(residuals_phi_modpi, dtype=float),
        np.asarray(residuals_theta, dtype=float),
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

    # Keep local and module subdetector setup aligned.
    m.SUBDETECTORS = SUBDETECTORS
    sim_hits_names, assoc_names, active_digi_names = m.active_collection_names(ACTIVE_SUBDETECTORS)

    print("Active subdetectors:", ACTIVE_SUBDETECTORS)
    print("Track filter mode:", TRACK_FILTER_MODE)
    print("Fit-quality cuts:", f"MIN_NDF={MIN_NDF}, MAX_CHI2NDF={MAX_CHI2NDF}")
    print("Truth-matching sim-hit collections:", sim_hits_names)
    print("Truth-matching association collections:", assoc_names)
    print("Track-filter digi collections:", active_digi_names)
    print("Color coding:", "largest angle -> blue, second -> red, third -> black, smallest -> green")

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

            res = calculate_event_angular_residuals(
                m=m,
                filename=file_path,
                sim_hits_names=sim_hits_names,
                assoc_names=assoc_names,
                active_digi_names=active_digi_names,
            )
            if res is None:
                print(f"  -> no usable matched events in {fname}")
                continue

            p_true, residuals_phi, residuals_phi_modpi, residuals_theta, summary = res
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
                theta, {"p_true": [], "res_phi": [], "res_phi_modpi": [], "res_theta": []}
            )
            store["p_true"].append(p_true)
            store["res_phi"].append(residuals_phi)
            store["res_phi_modpi"].append(residuals_phi_modpi)
            store["res_theta"].append(residuals_theta)

    if not results_by_theta:
        raise RuntimeError("No results accumulated. Check input files/collections/cuts.")

    bin_edges = np.logspace(np.log10(P_MIN_GEV), np.log10(P_MAX_GEV), P_BINS + 1)

    rows_phi_by_theta = {}
    rows_phi_modpi_by_theta = {}
    rows_theta_by_theta = {}
    curve_phi = {}
    curve_phi_modpi = {}
    err_phi = {}
    err_phi_modpi = {}
    curve_theta = {}
    err_theta = {}

    for theta, vals in sorted(results_by_theta.items()):
        p_true = np.concatenate(vals["p_true"]) if vals["p_true"] else np.asarray([], dtype=float)
        res_phi = np.concatenate(vals["res_phi"]) if vals["res_phi"] else np.asarray([], dtype=float)
        res_phi_modpi = np.concatenate(vals["res_phi_modpi"]) if vals["res_phi_modpi"] else np.asarray([], dtype=float)
        res_theta = np.concatenate(vals["res_theta"]) if vals["res_theta"] else np.asarray([], dtype=float)

        if p_true.size == 0:
            continue

        rows_phi = m.binned_metrics(
            p_true,
            res_phi,
            bin_edges,
            MIN_TRACKS_PER_BIN,
            method=SIGMA_METHOD,
            n_bootstrap=BOOTSTRAP_ITERS,
            seed=4000 + theta,
        )
        rows_phi_modpi = m.binned_metrics(
            p_true,
            res_phi_modpi,
            bin_edges,
            MIN_TRACKS_PER_BIN,
            method=SIGMA_METHOD,
            n_bootstrap=BOOTSTRAP_ITERS,
            seed=4500 + theta,
        )
        rows_theta = m.binned_metrics(
            p_true,
            res_theta,
            bin_edges,
            MIN_TRACKS_PER_BIN,
            method=SIGMA_METHOD,
            n_bootstrap=BOOTSTRAP_ITERS,
            seed=5000 + theta,
        )

        rows_phi_by_theta[theta] = rows_phi
        rows_phi_modpi_by_theta[theta] = rows_phi_modpi
        rows_theta_by_theta[theta] = rows_theta

        x, y, ye = m.rows_to_curve(rows_phi)
        curve_phi[theta] = (x, y)
        err_phi[theta] = ye

        x, y, ye = m.rows_to_curve(rows_phi_modpi)
        curve_phi_modpi[theta] = (x, y)
        err_phi_modpi[theta] = ye

        x, y, ye = m.rows_to_curve(rows_theta)
        curve_theta[theta] = (x, y)
        err_theta[theta] = ye

        print(
            f"Theta {theta:>3}°: bins(phi)={len(rows_phi)}, bins(phi_modpi)={len(rows_phi_modpi)}, "
            f"bins(theta)={len(rows_theta)}, "
            f"entries={len(p_true)}"
        )

    write_metrics_csv(CSV_PHI, rows_phi_by_theta, "delta_phi_rad")
    write_metrics_csv(CSV_PHI_MODPI, rows_phi_modpi_by_theta, "delta_phi_mod_pi_rad")
    write_metrics_csv(CSV_THETA, rows_theta_by_theta, "delta_theta_rad")
    print(f"Saved {CSV_PHI}")
    print(f"Saved {CSV_PHI_MODPI}")
    print(f"Saved {CSV_THETA}")

    m.plot_curves(
        outpath=OUT_PHI,
        title="Pion Phi Resolution",
        xlabel=r"Generated Momentum $p$ [GeV/c]",
        ylabel=r"$\sigma_\phi$ [rad]",
        curve_by_theta=curve_phi,
        yerr_by_theta=err_phi,
    )
    m.plot_curves(
        outpath=OUT_PHI_MODPI,
        title="Pion Phi Mod Pi Resolution",
        xlabel=r"Generated Momentum $p$ [GeV/c]",
        ylabel=r"$\sigma_{\phi}$ [rad]",
        curve_by_theta=curve_phi_modpi,
        yerr_by_theta=err_phi_modpi,
    )
    m.plot_curves(
        outpath=OUT_THETA,
        title="Pion Theta Resolution",
        xlabel=r"Generated Momentum $p$ [GeV/c]",
        ylabel=r"$\sigma_\theta$ [rad]",
        curve_by_theta=curve_theta,
        yerr_by_theta=err_theta,
    )


if __name__ == "__main__":
    main()
