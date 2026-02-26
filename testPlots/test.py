#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import glob
import math
from collections import defaultdict, Counter

import ROOT  # noqa: F401
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from podio.reading import get_reader
import edm4hep  # noqa: F401


# =========================
# PHYSICS / ANALYSIS CONFIG
# =========================

# Detector / conversion
B_FIELD = 2.0  # Tesla (IDEA central solenoid)
A_CONST = 2.99792458e-4  # pT[GeV] = A_CONST * B[T] / |omega[1/mm]|

# Input discovery
BASE_DIR = "/ceph/omunkombwe/pi_Barrel"
SUBDIR_GLOB = "deg*"
FIT_GLOB = "fit_*_merged.root"
FILENAME_RE = re.compile(r"fit_.*_theta(?P<theta>\d+)deg_.*_merged\.root")

# Truth + track selection
PDG_ABS = 211         # pion PDG
GEN_STATUS = 1
ATIP_LOCATION = 1
MIN_PURITY = 0.75

# Binning/stats
P_MIN_GEV = 0.2
P_MAX_GEV = 10.0
P_BINS = 100
MIN_TRACKS_PER_BIN = 80
SIGMA_METHOD = "q68"  # "q68" or "std"

TRACK_COLLECTION_CANDIDATES = ["Fitted_tracks"]

# ===============================================
# Subdetector setup (EDIT THIS to scan subsystems)
# ===============================================
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

# Default: combine all 5 subdetectors
ACTIVE_SUBDETECTORS = ["DCH", "VTXB", "VTXD", "SIWRB", "SIWRD"]

# Track filter mode w.r.t ACTIVE_SUBDETECTORS digi hits:
#   "off"       : no filtering
#   "inclusive" : keep track if >=1 hit belongs to active digi collections
#   "exclusive" : keep track only if ALL hits belong to active digi collections
TRACK_FILTER_MODE = "inclusive"

# Output files
OUT_P = "pi_Momentum_Resolution_truthMatched.png"
OUT_PT = "pi_Pt_Resolution_truthMatched.png"

OUT_RESID_P_HIST = "pi_residual_p_hist_by_theta.png"
OUT_RESID_PT_HIST = "pi_residual_pt_hist_by_theta.png"
OUT_RESID_P_2D = "pi_residual_p_vs_ptrue_by_theta.png"
OUT_RESID_PT_2D = "pi_residual_pt_vs_pttrue_by_theta.png"

RESID_HIST_BINS = 140
RESID_RANGE_PERCENTILES = (0.5, 99.5)


# =========================
# Helpers
# =========================

def get_value(obj, attr, getter=None, alt_getters=None):
    if obj is None:
        return None

    if getter and hasattr(obj, getter):
        try:
            return getattr(obj, getter)()
        except Exception:
            pass

    if alt_getters:
        for alt in alt_getters:
            if hasattr(obj, alt):
                try:
                    return getattr(obj, alt)()
                except Exception:
                    pass

    if hasattr(obj, attr):
        val = getattr(obj, attr)
        return val() if callable(val) else val

    return None


def oid(obj):
    obj_id = obj.getObjectID()
    return (obj_id.collectionID, obj_id.index)


def get_collection(event, name):
    try:
        return event.get(name)
    except Exception:
        return None


def get_vector3(vec):
    if vec is None:
        return None
    x = get_value(vec, "x", "getX")
    y = get_value(vec, "y", "getY")
    z = get_value(vec, "z", "getZ")
    if x is None or y is None or z is None:
        return None
    return float(x), float(y), float(z)


def iter_tracker_hits(track):
    if hasattr(track, "trackerHits_size") and hasattr(track, "getTrackerHits"):
        n_hits = track.trackerHits_size()
        for i in range(n_hits):
            yield track.getTrackerHits(i)
        return

    hits = get_value(track, "trackerHits", "getTrackerHits")
    if hits is None:
        return
    for hit in hits:
        yield hit


def get_track_states(track):
    states = get_value(track, "trackStates", "getTrackStates")
    if states is None:
        return []
    try:
        return list(states)
    except Exception:
        out = []
        try:
            n = len(states)
            for i in range(n):
                out.append(states[i])
        except Exception:
            pass
        return out


def get_tracks_from_event(event):
    for coll_name in TRACK_COLLECTION_CANDIDATES:
        tracks = get_collection(event, coll_name)
        if tracks is not None:
            return list(tracks)
    return []


def track_chi2ndf(track):
    chi2 = get_value(track, "chi2", "getChi2")
    ndf = get_value(track, "ndf", "getNdf")
    if chi2 is None or ndf is None or chi2 <= 0 or ndf <= 0:
        return None
    return float(chi2) / float(ndf)


def pick_track_state(track, atip_location):
    states = get_track_states(track)
    if len(states) == 0:
        return None

    chosen = None
    for st in states:
        loc = get_value(st, "location", "getLocation")
        if loc == atip_location:
            chosen = st
            break
    if chosen is None:
        chosen = states[0]

    omega = get_value(chosen, "omega", "getOmega")
    tan_lambda = get_value(chosen, "tanLambda", "getTanLambda")
    if omega is None or tan_lambda is None:
        return None

    return float(omega), float(tan_lambda)


# =========================
# Subdetector selection helpers
# =========================

def active_collection_names(active_subdetectors):
    sim_hits_names = []
    assoc_names = []
    digi_names = []
    for key in active_subdetectors:
        if key not in SUBDETECTORS:
            raise ValueError(f"Unknown subdetector '{key}'. Valid keys: {list(SUBDETECTORS.keys())}")
        sim_hits_names.append(SUBDETECTORS[key]["sim_hits"])
        assoc_names.append(SUBDETECTORS[key]["assoc"])
        digi_names.append(SUBDETECTORS[key]["digi"])
    return sim_hits_names, assoc_names, digi_names


def build_hit_oid_set(event, collection_names):
    oids = set()
    for name in collection_names:
        coll = get_collection(event, name)
        if coll is None:
            continue
        for hit in coll:
            oids.add(oid(hit))
    return oids


def filter_tracks_by_active_digis(event, tracks, active_digi_names, mode):
    if mode == "off":
        return tracks

    selected_oids = build_hit_oid_set(event, active_digi_names)
    if len(selected_oids) == 0:
        return []

    kept = []
    for trk in tracks:
        hits = list(iter_tracker_hits(trk))
        if len(hits) == 0:
            continue

        n_in = 0
        for h in hits:
            if oid(h) in selected_oids:
                n_in += 1

        if mode == "inclusive":
            if n_in > 0:
                kept.append(trk)
        elif mode == "exclusive":
            if n_in == len(hits):
                kept.append(trk)
        else:
            raise ValueError(f"Unknown TRACK_FILTER_MODE='{mode}'")

    return kept


# =========================
# Truth matching
# =========================

def build_sim_to_truth(event, sim_hits_names):
    """
    Map sim-hit OID -> MC truth objectID index.
    """
    sim_to_truth = {}
    for name in sim_hits_names:
        sim_hits = get_collection(event, name)
        if sim_hits is None:
            continue
        for sim in sim_hits:
            mc_part = get_value(sim, "particle", "getParticle")
            if mc_part is None:
                continue
            sim_to_truth[oid(sim)] = mc_part.getObjectID().index
    return sim_to_truth


def build_digi_truth_map(event, assoc_names, sim_to_truth):
    digi_truth = defaultdict(set)
    for name in assoc_names:
        associations = get_collection(event, name)
        if associations is None:
            continue
        for assoc in associations:
            digi = get_value(assoc, "from", "getFrom")
            sim = get_value(assoc, "to", "getTo")
            if digi is None or sim is None:
                continue
            truth_idx = sim_to_truth.get(oid(sim))
            if truth_idx is not None:
                digi_truth[oid(digi)].add(truth_idx)
    return digi_truth


def match_tracks_to_truth(event, tracks, sim_hits_names, assoc_names):
    sim_to_truth = build_sim_to_truth(event, sim_hits_names)
    digi_truth = build_digi_truth_map(event, assoc_names, sim_to_truth)

    matches = []
    for track in tracks:
        counts = Counter()
        nhits = 0
        for digi in iter_tracker_hits(track):
            nhits += 1
            for tidx in digi_truth.get(oid(digi), ()):
                counts[tidx] += 1

        if counts:
            best_idx, shared_hits = counts.most_common(1)[0]
        else:
            best_idx, shared_hits = None, 0

        purity = (shared_hits / nhits) if nhits > 0 else 0.0
        matches.append({
            "track": track,
            "truth_index": best_idx,
            "purity": purity,
            "shared_hits": shared_hits,
            "nhits": nhits,
        })
    return matches


def select_best_truth_matched_pair(event, tracks, mc_particles, sim_hits_names, assoc_names):
    matches = match_tracks_to_truth(event, tracks, sim_hits_names, assoc_names)

    mc_by_oid_index = {mc.getObjectID().index: mc for mc in mc_particles}

    best_pair = None
    best_score = None

    for m in matches:
        tidx = m["truth_index"]
        if tidx is None:
            continue
        if m["purity"] < MIN_PURITY:
            continue

        mc = mc_by_oid_index.get(tidx)
        if mc is None:
            continue

        if GEN_STATUS is not None:
            gen = get_value(mc, "generatorStatus", "getGeneratorStatus")
            if gen != GEN_STATUS:
                continue

        pdg = get_value(mc, "PDG", "getPDG", ["getPdg"])
        if pdg is None:
            pdg = get_value(mc, "pdg", None)
        if PDG_ABS is not None:
            if pdg is None or abs(int(pdg)) != PDG_ABS:
                continue

        mom = get_vector3(get_value(mc, "momentum", "getMomentum"))
        if mom is None:
            continue

        chi2_ndf = track_chi2ndf(m["track"])
        if chi2_ndf is None:
            chi2_ndf = 1e9

        # max purity, max shared_hits, min chi2/ndf
        score = (m["purity"], m["shared_hits"], -chi2_ndf)

        if best_score is None or score > best_score:
            best_score = score
            best_pair = (mc, m["track"])

    return best_pair


# =========================
# Event loop
# =========================

def calculate_event_residuals(filename, sim_hits_names, assoc_names, active_digi_names):
    reader = get_reader(filename)
    events = reader.get("events")

    p_true_vals = []
    pt_true_vals = []
    residuals_p = []
    residuals_pt = []

    n_events = 0
    n_has_mc = 0
    n_has_tracks = 0
    n_subdet_pass = 0
    n_matched = 0

    for event in events:
        n_events += 1

        mc_particles = get_collection(event, "MCParticles")
        if mc_particles is None or len(mc_particles) == 0:
            continue
        n_has_mc += 1

        tracks = get_tracks_from_event(event)
        if len(tracks) == 0:
            continue
        n_has_tracks += 1

        # Apply subdetector-based track filtering
        tracks = filter_tracks_by_active_digis(
            event=event,
            tracks=tracks,
            active_digi_names=active_digi_names,
            mode=TRACK_FILTER_MODE,
        )
        if len(tracks) == 0:
            continue
        n_subdet_pass += 1

        pair = select_best_truth_matched_pair(
            event=event,
            tracks=tracks,
            mc_particles=mc_particles,
            sim_hits_names=sim_hits_names,
            assoc_names=assoc_names,
        )
        if pair is None:
            continue

        mc_truth, best_track = pair
        n_matched += 1

        mom = get_vector3(get_value(mc_truth, "momentum", "getMomentum"))
        if mom is None:
            continue

        px, py, pz = mom
        p_true = math.sqrt(px * px + py * py + pz * pz)
        pt_true = math.hypot(px, py)
        if p_true <= 0 or pt_true <= 0:
            continue

        state = pick_track_state(best_track, ATIP_LOCATION)
        if state is None:
            continue

        omega_mm, tan_lambda = state
        if omega_mm == 0:
            continue

        pt_rec = A_CONST * B_FIELD / abs(omega_mm)
        p_rec = pt_rec * math.sqrt(1.0 + tan_lambda * tan_lambda)

        p_true_vals.append(p_true)
        pt_true_vals.append(pt_true)
        residuals_p.append((p_rec - p_true) / p_true)
        residuals_pt.append((pt_rec - pt_true) / pt_true)

    if len(residuals_p) == 0 or len(residuals_pt) == 0:
        return None

    summary = {
        "total": n_events,
        "with_mc": n_has_mc,
        "with_tracks": n_has_tracks,
        "subdet_pass": n_subdet_pass,
        "matched": n_matched,
        "used": len(residuals_p),
    }

    return (
        np.asarray(p_true_vals, dtype=float),
        np.asarray(pt_true_vals, dtype=float),
        np.asarray(residuals_p, dtype=float),
        np.asarray(residuals_pt, dtype=float),
        summary,
    )


# =========================
# Binning / resolution
# =========================

def bin_sigma(values):
    if len(values) < 2:
        return None

    if SIGMA_METHOD.lower() == "std":
        return float(np.std(values, ddof=1))

    q16, q84 = np.percentile(values, [16.0, 84.0])
    return float(0.5 * (q84 - q16))


def binned_resolution(xvals, residuals, bin_edges, min_tracks):
    centers = []
    sigmas = []

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (xvals >= lo) & (xvals < hi)
        n = int(np.count_nonzero(mask))
        if n < min_tracks:
            continue

        vals = residuals[mask]
        sig = bin_sigma(vals)
        if sig is None or not np.isfinite(sig):
            continue

        centers.append(math.sqrt(lo * hi))
        sigmas.append(sig)

    return centers, sigmas


# =========================
# Residual diagnostic plots
# =========================

def concat_theta_array(results_by_theta, theta, key):
    arrs = results_by_theta.get(theta, {}).get(key, [])
    if not arrs:
        return np.asarray([], dtype=float)
    return np.concatenate(arrs).astype(float)


def plot_residual_histograms_by_theta(results_by_theta, key, outname, xlabel, title):
    thetas = sorted(results_by_theta.keys())

    all_vals = []
    for th in thetas:
        vals = concat_theta_array(results_by_theta, th, key)
        vals = vals[np.isfinite(vals)]
        if vals.size > 0:
            all_vals.append(vals)

    if not all_vals:
        print(f"Warning: no data for {key}, skip {outname}")
        return

    all_concat = np.concatenate(all_vals)
    qlo, qhi = np.percentile(all_concat, RESID_RANGE_PERCENTILES)
    if (not np.isfinite(qlo)) or (not np.isfinite(qhi)) or (qhi <= qlo):
        qlo, qhi = float(np.min(all_concat)), float(np.max(all_concat))
    pad = 0.08 * (qhi - qlo) if qhi > qlo else 1e-6
    xr = (qlo - pad, qhi + pad)

    plt.figure(figsize=(10, 7))
    for th in thetas:
        vals = concat_theta_array(results_by_theta, th, key)
        vals = vals[np.isfinite(vals)]
        if vals.size < 5:
            continue
        plt.hist(
            vals,
            bins=RESID_HIST_BINS,
            range=xr,
            density=True,
            histtype="step",
            linewidth=1.8,
            label=f"Theta = {th}° (N={vals.size})",
        )

    plt.axvline(0.0, color="k", linestyle="--", linewidth=1.2, alpha=0.8)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel("Normalized entries", fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(outname, dpi=300)
    plt.close()
    print(f"Saved {outname}")


def plot_residual_vs_true_grid(results_by_theta, x_key, y_key, outname, xlabel, ylabel, title):
    thetas = sorted(results_by_theta.keys())
    if len(thetas) == 0:
        print(f"Warning: no theta data for {outname}")
        return

    # Global y-limits for easier panel-to-panel comparison
    all_y = []
    for th in thetas:
        y = concat_theta_array(results_by_theta, th, y_key)
        y = y[np.isfinite(y)]
        if y.size > 0:
            all_y.append(y)
    if all_y:
        all_y = np.concatenate(all_y)
        y_lo, y_hi = np.percentile(all_y, RESID_RANGE_PERCENTILES)
        if not np.isfinite(y_lo) or not np.isfinite(y_hi) or y_hi <= y_lo:
            y_lo, y_hi = float(np.min(all_y)), float(np.max(all_y))
        y_pad = 0.08 * (y_hi - y_lo) if y_hi > y_lo else 1e-6
        y_lim = (y_lo - y_pad, y_hi + y_pad)
    else:
        y_lim = None

    n = len(thetas)
    ncols = 2
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4.5 * nrows), sharey=True)

    # Normalize axes object to 2D array-like indexing
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = np.array([axes])
    elif ncols == 1:
        axes = np.array([[ax] for ax in axes])

    hb_for_cbar = None
    for i, th in enumerate(thetas):
        r = i // ncols
        c = i % ncols
        ax = axes[r, c]

        x = concat_theta_array(results_by_theta, th, x_key)
        y = concat_theta_array(results_by_theta, th, y_key)
        m = np.isfinite(x) & np.isfinite(y) & (x > 0)
        x = x[m]
        y = y[m]

        if x.size == 0:
            ax.set_title(f"Theta = {th}° (no data)")
            ax.axis("off")
            continue

        hb_for_cbar = ax.hexbin(x, y, gridsize=55, bins="log", mincnt=1)
        ax.set_xscale("log")
        ax.axhline(0.0, color="k", linestyle="--", linewidth=1.0, alpha=0.8)
        if y_lim is not None:
            ax.set_ylim(*y_lim)

        ax.set_title(f"Theta = {th}° (N={x.size})", fontsize=11)
        ax.grid(True, which="both", linestyle="--", alpha=0.25)

        if r == nrows - 1:
            ax.set_xlabel(xlabel, fontsize=11)
        if c == 0:
            ax.set_ylabel(ylabel, fontsize=11)

    # Hide unused panels
    for j in range(n, nrows * ncols):
        r = j // ncols
        c = j % ncols
        axes[r, c].axis("off")

    if hb_for_cbar is not None:
        cbar = fig.colorbar(hb_for_cbar, ax=axes, shrink=0.92)
        cbar.set_label("log10(counts)", fontsize=10)

    fig.suptitle(title, fontsize=15, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(outname, dpi=300)
    plt.close(fig)
    print(f"Saved {outname}")


# =========================
# Main
# =========================

def main():
    if len(ACTIVE_SUBDETECTORS) == 0:
        raise ValueError("ACTIVE_SUBDETECTORS is empty. Add at least one subdetector key.")

    sim_hits_names, assoc_names, active_digi_names = active_collection_names(ACTIVE_SUBDETECTORS)

    results_by_theta = {}

    subdirs = sorted(glob.glob(os.path.join(BASE_DIR, SUBDIR_GLOB)))
   
    for subdir in subdirs:
        fit_files = sorted(glob.glob(os.path.join(subdir, FIT_GLOB)))
        if not fit_files:
            print(f"Warning: no {FIT_GLOB} files in {subdir}")
            continue

        for file_path in fit_files:
            fname = os.path.basename(file_path)
            m = FILENAME_RE.match(fname)
            if not m:
                print(f"Warning: unrecognized filename format: {file_path}")
                continue

            theta = int(m.group("theta"))
            print(f"\nAnalyzing {file_path} (theta={theta} deg)")

            res = calculate_event_residuals(
                filename=file_path,
                sim_hits_names=sim_hits_names,
                assoc_names=assoc_names,
                active_digi_names=active_digi_names,
            )

            if res is None:
                print(f"  -> no usable matched events in {fname}")
                continue

            p_true, pt_true, residuals_p, residuals_pt, summary = res

            print(
                "  summary: "
                f"total={summary['total']}, "
                f"with_mc={summary['with_mc']}, "
                f"with_tracks={summary['with_tracks']}, "
                f"subdet_pass={summary['subdet_pass']}, "
                f"matched={summary['matched']}, "
                f"used={summary['used']}"
            )

            slot = results_by_theta.setdefault(
                theta, {"p_true": [], "pt_true": [], "res_p": [], "res_pt": []}
            )
            slot["p_true"].append(p_true)
            slot["pt_true"].append(pt_true)
            slot["res_p"].append(residuals_p)
            slot["res_pt"].append(residuals_pt)

    if not results_by_theta:
        raise RuntimeError("No results accumulated. Check input files/collections/cuts.")

    # -------------------------
    # Residual diagnostic plots
    # -------------------------
    plot_residual_histograms_by_theta(
        results_by_theta=results_by_theta,
        key="res_p",
        outname=OUT_RESID_P_HIST,
        xlabel=r"$(p_{\mathrm{rec}}-p_{\mathrm{true}})/p_{\mathrm{true}}$",
        title="Muon Momentum Residual Distribution by Angle",
    )

    plot_residual_histograms_by_theta(
        results_by_theta=results_by_theta,
        key="res_pt",
        outname=OUT_RESID_PT_HIST,
        xlabel=r"$(p_{T,\mathrm{rec}}-p_{T,\mathrm{true}})/p_{T,\mathrm{true}}$",
        title=r"Muon $p_T$ Residual Distribution by Angle",
    )

    plot_residual_vs_true_grid(
        results_by_theta=results_by_theta,
        x_key="p_true",
        y_key="res_p",
        outname=OUT_RESID_P_2D,
        xlabel=r"True momentum $p$ [GeV/c]",
        ylabel=r"$(p_{\mathrm{rec}}-p_{\mathrm{true}})/p_{\mathrm{true}}$",
        title="Residual Behavior vs True Momentum (per angle)",
    )

    plot_residual_vs_true_grid(
        results_by_theta=results_by_theta,
        x_key="pt_true",
        y_key="res_pt",
        outname=OUT_RESID_PT_2D,
        xlabel=r"True transverse momentum $p_T$ [GeV/c]",
        ylabel=r"$(p_{T,\mathrm{rec}}-p_{T,\mathrm{true}})/p_{T,\mathrm{true}}$",
        title=r"Residual Behavior vs True $p_T$ (per angle)",
    )

    # -------------------------
    # Resolution vs momentum
    # -------------------------
    bin_edges = np.logspace(np.log10(P_MIN_GEV), np.log10(P_MAX_GEV), P_BINS + 1)

    final_p = {}
    x_p = {}
    final_pt = {}
    x_pt = {}

    for theta, vals in sorted(results_by_theta.items()):
        p_true = np.concatenate(vals["p_true"]) if vals["p_true"] else np.asarray([], dtype=float)
        pt_true = np.concatenate(vals["pt_true"]) if vals["pt_true"] else np.asarray([], dtype=float)
        res_p = np.concatenate(vals["res_p"]) if vals["res_p"] else np.asarray([], dtype=float)
        res_pt = np.concatenate(vals["res_pt"]) if vals["res_pt"] else np.asarray([], dtype=float)

        if p_true.size == 0:
            continue

        centers_p, sigmas_p = binned_resolution(p_true, res_p, bin_edges, MIN_TRACKS_PER_BIN)

        s = math.sin(math.radians(theta))
        if s > 0.0:
            pt_edges = bin_edges * s
            centers_pt, sigmas_pt = binned_resolution(pt_true, res_pt, pt_edges, MIN_TRACKS_PER_BIN)
        else:
            centers_pt, sigmas_pt = [], []

        x_p[theta] = centers_p
        final_p[theta] = sigmas_p
        x_pt[theta] = centers_pt
        final_pt[theta] = sigmas_pt

        print(
            f"Theta {theta:>3}°: bins(p)={len(sigmas_p)}, bins(pT)={len(sigmas_pt)}, "
            f"entries={len(p_true)}"
        )

    # Plot sigma_p/p vs p
    plt.figure(figsize=(10, 7))
    for theta in sorted(final_p.keys()):
        if len(final_p[theta]) == 0:
            continue
        plt.plot(x_p[theta], final_p[theta], "o--", label=f"Theta = {theta}°")

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Generated Momentum $p$ [GeV/c]", fontsize=12)
    plt.ylabel(r"Momentum Resolution $\sigma_p / p$", fontsize=12)
    #title_method = "68% half-width" if SIGMA_METHOD.lower() == "q68" else "sample std"
    plt.title(f"Pion Momentum Resolution", fontsize=14)
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_P, dpi=300)
    plt.close()
    print(f"Saved {OUT_P}")

    # Plot sigma_pT/pT vs pT
    plt.figure(figsize=(10, 7))
    for theta in sorted(final_pt.keys()):
        if len(final_pt[theta]) == 0:
            continue
        plt.plot(x_pt[theta], final_pt[theta], "o--", label=f"Theta = {theta}°")

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Generated Transverse Momentum $p_T$ [GeV/c]", fontsize=12)
    plt.ylabel(r"Transverse Momentum Resolution $\sigma_{p_T} / p_T$", fontsize=12)
    plt.title(f"Pion $p_T$ Resolution", fontsize=14)
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_PT, dpi=300)
    plt.close()
    print(f"Saved {OUT_PT}")


if __name__ == "__main__":
    main()