#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import csv
import glob
import math
from collections import defaultdict, Counter

import ROOT  # noqa: F401  (kept because many EDM4hep setups expect ROOT import side-effects)
import numpy as np
import matplotlib.pyplot as plt
from podio.reading import get_reader
import edm4hep  # noqa: F401  (required for podio edm4hep types)

# =========================
# PHYSICS / ANALYSIS CONFIG
# =========================

# Detector / conversion
B_FIELD = 2.0  # Tesla (IDEA central solenoid)
# For this fitted-track output, omega follows omega = q * B / pT, so pT[GeV] = B[T] / |omega|
A_CONST = 1.0

# Input discovery
BASE_DIR = "/ceph/omunkombwe/pi_forward"  # Base directory containing subdirs with fit_*.root files
SUBDIR_GLOB = "deg*"
FIT_GLOB = "fit_*_merged.root"
FILENAME_RE = re.compile(r"fit_.*_theta(?P<theta>\d+)deg_.*_merged\.root")

# Truth and track selection
PDG_ABS = 211         # pion PDG ID
GEN_STATUS = 1
ATIP_LOCATION = 1
MIN_PURITY = 0.75
MIN_NDF = 5
MAX_CHI2NDF = 10.0   # set to None to disable chi2/ndf upper cut

# Binning and stats
P_MIN_GEV = 0.1
P_MAX_GEV = 10.0
P_BINS = 60
MIN_TRACKS_PER_BIN = 120
SIGMA_METHOD = "q68"       # "q68" (robust) or "std"
BOOTSTRAP_ITERS = 80       # used for q68 uncertainty estimate

TRACK_COLLECTION_CANDIDATES = ["Fitted_tracks"]

# ===========================================================
# Subdetector config (EDIT THIS LIST to study combinations)
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
OUT_P = "pi_Momentum_Resolution_forward.png"
OUT_PT = "pi_Pt_Resolution_forward.png"
OUT_INVPT = "pi_InvPt_Resolution_forward.png"
OUT_PT2 = "pi_Pt2_Resolution_forward.png"

CSV_P = "pi_idea_forward_binned_metrics_p.csv"
CSV_PT = "pi_idea_forward_binned_metrics_pt.csv"
CSV_INVPT = "pi_idea_forward_binned_metrics_invpt.csv"
CSV_PT2 = "pi_idea_forward_binned_metrics_pt2.csv"

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
    # EDM4hep style
    if hasattr(track, "trackerHits_size") and hasattr(track, "getTrackerHits"):
        n_hits = track.trackerHits_size()
        for i in range(n_hits):
            yield track.getTrackerHits(i)
        return

    # Alternative style
    hits = get_value(track, "trackerHits", "getTrackerHits")
    if hits is None:
        return
    for hit in hits:
        yield hit


def get_track_states(track):
    """
    Return list of track states robustly.
    """
    states = get_value(track, "trackStates", "getTrackStates")
    if states is None:
        return []
    try:
        return list(states)
    except Exception:
        # fallback if states behaves like indexable object
        out = []
        try:
            n = len(states)
            for i in range(n):
                out.append(states[i])
        except Exception:
            pass
        return out


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


def build_sim_to_truth(event, sim_hits_names):
   
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
            
            digi = get_value(assoc, "getFrom")
            sim = get_value(assoc, "getTo")
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
        matches.append(
            {
                "track": track,
                "truth_index": best_idx,
                "purity": purity,
                "shared_hits": shared_hits,
                "nhits": nhits,
            }
        )
    return matches


def track_chi2ndf(track):
    chi2 = get_value(track, "chi2", "getChi2")
    ndf = get_value(track, "ndf", "getNdf")
    if chi2 is None or ndf is None or chi2 <= 0 or ndf <= 0:
        return None
    return float(chi2) / float(ndf)


def passes_track_quality(track, min_ndf, max_chi2ndf):
    ndf = get_value(track, "ndf", "getNdf")
    chi2_ndf = track_chi2ndf(track)

    if min_ndf is None and max_chi2ndf is None:
        # quality cuts disabled
        return True, float(chi2_ndf) if chi2_ndf is not None else 1e9, None

    if ndf is None or chi2_ndf is None:
        return False, None, "missing_fit"

    ndf = float(ndf)
    if min_ndf is not None and ndf < float(min_ndf):
        return False, float(chi2_ndf), "low_ndf"

    if max_chi2ndf is not None and chi2_ndf > float(max_chi2ndf):
        return False, float(chi2_ndf), "high_chi2ndf"

    return True, float(chi2_ndf), None


def select_best_truth_matched_pair(
    event,
    tracks,
    mc_particles,
    sim_hits_names,
    assoc_names,
    min_ndf,
    max_chi2ndf,
):
    matches = match_tracks_to_truth(event, tracks, sim_hits_names, assoc_names)

    mc_by_oid_index = {}
    for mc in mc_particles:
        mc_by_oid_index[mc.getObjectID().index] = mc

    best_pair_any = None
    best_score_any = None
    best_pair_quality = None
    best_score_quality = None

    quality_stats = {
        "candidates": 0,
        "quality_pass": 0,
        "quality_fail_missing": 0,
        "quality_fail_low_ndf": 0,
        "quality_fail_high_chi2ndf": 0,
    }

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

        quality_stats["candidates"] += 1

        # tie-break quality in the "any matched" branch
        chi2_ndf = track_chi2ndf(m["track"])
        if chi2_ndf is None:
            chi2_ndf = 1e9

        # max purity, then min chi2/ndf, then max shared_hits
        score_any = (m["purity"], -chi2_ndf, m["shared_hits"])
        if best_score_any is None or score_any > best_score_any:
            best_score_any = score_any
            best_pair_any = (mc, m["track"])

        passed_quality, chi2_q, fail_reason = passes_track_quality(
            m["track"],
            min_ndf=min_ndf,
            max_chi2ndf=max_chi2ndf,
        )
        if not passed_quality:
            if fail_reason == "missing_fit":
                quality_stats["quality_fail_missing"] += 1
            elif fail_reason == "low_ndf":
                quality_stats["quality_fail_low_ndf"] += 1
            elif fail_reason == "high_chi2ndf":
                quality_stats["quality_fail_high_chi2ndf"] += 1
            continue

        quality_stats["quality_pass"] += 1

        score_quality = (m["purity"], -chi2_q, m["shared_hits"])
        if best_score_quality is None or score_quality > best_score_quality:
            best_score_quality = score_quality
            best_pair_quality = (mc, m["track"])

    return best_pair_any, best_pair_quality, quality_stats


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


def get_tracks_from_event(event):
    for coll_name in TRACK_COLLECTION_CANDIDATES:
        tracks = get_collection(event, coll_name)
        if tracks is not None:
            return list(tracks)
    return []


# =========================
def calculate_event_residuals(filename, sim_hits_names, assoc_names, active_digi_names):
    reader = get_reader(filename)
    events = reader.get("events")

    p_true_vals = []
    pt_true_vals = []
    residuals_p = []
    residuals_pt = []
    residuals_invpt = []

    n_events = 0
    n_has_mc = 0
    n_has_tracks = 0
    n_subdet_pass = 0
    n_matched_raw = 0
    n_matched_quality = 0
    n_quality_rejected = 0
    n_quality_fail_missing = 0
    n_quality_fail_low_ndf = 0
    n_quality_fail_high_chi2ndf = 0

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

        # Track filtering based on ACTIVE_SUBDETECTORS digi collections
        tracks = filter_tracks_by_active_digis(
            event=event,
            tracks=tracks,
            active_digi_names=active_digi_names,
            mode=TRACK_FILTER_MODE,
        )
        if len(tracks) == 0:
            continue
        n_subdet_pass += 1

        pair_any, pair_quality, quality_stats = select_best_truth_matched_pair(
            event=event,
            tracks=tracks,
            mc_particles=mc_particles,
            sim_hits_names=sim_hits_names,
            assoc_names=assoc_names,
            min_ndf=MIN_NDF,
            max_chi2ndf=MAX_CHI2NDF,
        )
        n_quality_fail_missing += quality_stats["quality_fail_missing"]
        n_quality_fail_low_ndf += quality_stats["quality_fail_low_ndf"]
        n_quality_fail_high_chi2ndf += quality_stats["quality_fail_high_chi2ndf"]

        if pair_any is None:
            continue
        n_matched_raw += 1

        if pair_quality is None:
            n_quality_rejected += 1
            continue
        n_matched_quality += 1

        mc_truth, best_track = pair_quality

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
        residuals_invpt.append((1.0 / pt_rec) - (1.0 / pt_true))

    if len(residuals_p) == 0 or len(residuals_pt) == 0 or len(residuals_invpt) == 0:
        return None

    summary = {
        "total": n_events,
        "with_mc": n_has_mc,
        "with_tracks": n_has_tracks,
        "subdet_pass": n_subdet_pass,
        "matched_raw": n_matched_raw,
        "matched_quality": n_matched_quality,
        "quality_rejected": n_quality_rejected,
        "quality_fail_missing": n_quality_fail_missing,
        "quality_fail_low_ndf": n_quality_fail_low_ndf,
        "quality_fail_high_chi2ndf": n_quality_fail_high_chi2ndf,
        "matched": n_matched_quality,
        "used": len(residuals_p),
    }

    return (
        np.asarray(p_true_vals, dtype=float),
        np.asarray(pt_true_vals, dtype=float),
        np.asarray(residuals_p, dtype=float),
        np.asarray(residuals_pt, dtype=float),
        np.asarray(residuals_invpt, dtype=float),
        summary,
    )


# =========================
# Binning / resolution
# =========================

def bootstrap_sigma68_err(values, n_iter, rng):
    n = int(values.size)
    if n < 2 or n_iter <= 1:
        return float("nan")

    out = np.empty(n_iter, dtype=float)
    for i in range(n_iter):
        idx = rng.integers(0, n, size=n)
        sample = values[idx]
        q16, q84 = np.percentile(sample, [16.0, 84.0])
        out[i] = 0.5 * (q84 - q16)

    return float(np.std(out, ddof=1)) if out.size > 1 else float("nan")


def sigma_and_error(values, method, n_bootstrap, rng):
    if values.size < 2:
        return None

    q16, q84 = np.percentile(values, [16.0, 84.0])
    sigma68 = float(0.5 * (q84 - q16))
    std = float(np.std(values, ddof=1))

    if method.lower() == "std":
        sigma = std
        sigma_err = float(sigma / math.sqrt(2.0 * (values.size - 1))) if values.size > 2 else float("nan")
    else:
        sigma = sigma68
        sigma_err = bootstrap_sigma68_err(values, n_bootstrap, rng)

    return {
        "q16": float(q16),
        "q84": float(q84),
        "sigma68": sigma68,
        "std": std,
        "sigma": float(sigma),
        "sigma_err": float(sigma_err),
    }


def binned_metrics(xvals, residuals, bin_edges, min_tracks, method, n_bootstrap, seed):
    rows = []
    rng = np.random.default_rng(seed)

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (xvals >= lo) & (xvals < hi)
        n = int(np.count_nonzero(mask))
        if n < min_tracks:
            continue

        vals = residuals[mask]
        stats = sigma_and_error(vals, method=method, n_bootstrap=n_bootstrap, rng=rng)
        if stats is None:
            continue
        if not np.isfinite(stats["sigma"]) or stats["sigma"] <= 0:
            continue

        rows.append(
            {
                "lo": float(lo),
                "hi": float(hi),
                "center": float(math.sqrt(lo * hi)),
                "n": n,
                "q16": stats["q16"],
                "q84": stats["q84"],
                "sigma68": stats["sigma68"],
                "std": stats["std"],
                "sigma": stats["sigma"],
                "sigma_err": stats["sigma_err"],
            }
        )

    return rows


def rows_to_curve(rows, y_key="sigma", yerr_key="sigma_err"):
    x = np.asarray([r["center"] for r in rows], dtype=float)
    y = np.asarray([r[y_key] for r in rows], dtype=float)
    ye = np.asarray([r.get(yerr_key, float("nan")) for r in rows], dtype=float)

    mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    return x[mask], y[mask], ye[mask]


def rows_pt_to_pt2(rows_pt):
    rows_pt2 = []
    for r in rows_pt:
        if r["center"] <= 0:
            continue
        scale = 1.0 / r["center"]
        rr = dict(r)
        rr["q16"] = r["q16"] * scale
        rr["q84"] = r["q84"] * scale
        rr["sigma68"] = r["sigma68"] * scale
        rr["std"] = r["std"] * scale
        rr["sigma"] = r["sigma"] * scale
        rr["sigma_err"] = r["sigma_err"] * scale if np.isfinite(r["sigma_err"]) else float("nan")
        rows_pt2.append(rr)
    return rows_pt2


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


def plot_curves(outpath, title, xlabel, ylabel, curve_by_theta, yerr_by_theta):
    plt.figure(figsize=(10, 7))
    theta_list = sorted(curve_by_theta.keys(), reverse=True)
    color_map = build_angle_color_map(theta_list)
    for theta in theta_list:
        x, y = curve_by_theta[theta]
        if len(x) == 0:
            continue
        ye = yerr_by_theta.get(theta, np.asarray([], dtype=float))
        if len(ye) == len(y) and np.any(np.isfinite(ye)):
            plt.errorbar(
                x,
                y,
                yerr=ye,
                fmt="+--",
                ms=5,
                lw=1.5,
                capsize=2,
                color=color_map[theta],
                label=f"Theta = {theta}°",
            )
        else:
            plt.plot(x, y, "+--", ms=5, lw=1.8, color=color_map[theta], label=f"Theta = {theta}°")

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()
    print(f"Saved {outpath}")


def main():
    if len(ACTIVE_SUBDETECTORS) == 0:
        raise ValueError("ACTIVE_SUBDETECTORS is empty. Add at least one subdetector key.")

    sim_hits_names, assoc_names, active_digi_names = active_collection_names(ACTIVE_SUBDETECTORS)

    print("Active subdetectors:", ACTIVE_SUBDETECTORS)
    print("Track filter mode:", TRACK_FILTER_MODE)
    print("Fit-quality cuts:", f"MIN_NDF={MIN_NDF}, MAX_CHI2NDF={MAX_CHI2NDF}")
    print("Truth-matching sim-hit collections:", sim_hits_names)
    print("Truth-matching association collections:", assoc_names)
    print("Track-filter digi collections:", active_digi_names)

    results_by_theta = {}
    quality_summary_by_theta = {}

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

            p_true, pt_true, residuals_p, residuals_pt, residuals_invpt, summary = res

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
            print(
                "           "
                f"quality_failures(track-candidate): missing_fit={summary['quality_fail_missing']}, "
                f"low_ndf={summary['quality_fail_low_ndf']}, "
                f"high_chi2ndf={summary['quality_fail_high_chi2ndf']}"
            )

            qsum = quality_summary_by_theta.setdefault(
                theta,
                {
                    "total": 0,
                    "with_mc": 0,
                    "with_tracks": 0,
                    "subdet_pass": 0,
                    "matched_raw": 0,
                    "matched_quality": 0,
                    "quality_rejected": 0,
                    "quality_fail_missing": 0,
                    "quality_fail_low_ndf": 0,
                    "quality_fail_high_chi2ndf": 0,
                    "used": 0,
                },
            )
            for key, val in summary.items():
                if key in qsum:
                    qsum[key] += int(val)

            store = results_by_theta.setdefault(
                theta, {"p_true": [], "pt_true": [], "res_p": [], "res_pt": [], "res_invpt": []}
            )
            store["p_true"].append(p_true)
            store["pt_true"].append(pt_true)
            store["res_p"].append(residuals_p)
            store["res_pt"].append(residuals_pt)
            store["res_invpt"].append(residuals_invpt)

    if not results_by_theta:
        raise RuntimeError("No results accumulated. Check input files/collections/cuts.")

    print("\nPer-theta fit-quality cutflow:")

    def pct(num, den):
        return (100.0 * float(num) / float(den)) if den > 0 else 0.0

    for theta in sorted(quality_summary_by_theta.keys()):
        s = quality_summary_by_theta[theta]
        pass_frac = pct(s["matched_quality"], s["matched_raw"])
        with_mc_frac = pct(s["with_mc"], s["total"])
        track_find_frac = pct(s["with_tracks"], s["total"])
        subdet_pass_frac = pct(s["subdet_pass"], s["with_tracks"])
        match_raw_frac = pct(s["matched_raw"], s["subdet_pass"])
        used_frac = pct(s["used"], s["total"])
        print(
            f"  theta={theta:>3}°: "
            f"matched_raw={s['matched_raw']}, "
            f"matched_quality={s['matched_quality']} ({pass_frac:.1f}%), "
            f"quality_rejected={s['quality_rejected']}, "
            f"used={s['used']}, "
            f"track_candidate_fails: missing={s['quality_fail_missing']}, "
            f"low_ndf={s['quality_fail_low_ndf']}, "
            f"high_chi2ndf={s['quality_fail_high_chi2ndf']}"
        )
        print(
            "             "
            f"stage_eff(%): with_mc/total={with_mc_frac:.1f}, "
            f"with_tracks/total={track_find_frac:.1f}, "
            f"subdet_pass/with_tracks={subdet_pass_frac:.1f}, "
            f"matched_raw/subdet_pass={match_raw_frac:.1f}, "
            f"matched_quality/matched_raw={pass_frac:.1f}, "
            f"used/total={used_frac:.1f}"
        )

    # Log-spaced bins in p
    bin_edges = np.logspace(np.log10(P_MIN_GEV), np.log10(P_MAX_GEV), P_BINS + 1)

    rows_p_by_theta = {}
    rows_pt_by_theta = {}
    rows_invpt_by_theta = {}
    rows_pt2_by_theta = {}

    curve_p = {}
    err_p = {}
    curve_pt = {}
    err_pt = {}
    curve_invpt = {}
    err_invpt = {}
    curve_pt2 = {}
    err_pt2 = {}

    for theta, vals in sorted(results_by_theta.items()):
        p_true = np.concatenate(vals["p_true"]) if vals["p_true"] else np.asarray([], dtype=float)
        pt_true = np.concatenate(vals["pt_true"]) if vals["pt_true"] else np.asarray([], dtype=float)
        res_p = np.concatenate(vals["res_p"]) if vals["res_p"] else np.asarray([], dtype=float)
        res_pt = np.concatenate(vals["res_pt"]) if vals["res_pt"] else np.asarray([], dtype=float)
        res_invpt = np.concatenate(vals["res_invpt"]) if vals["res_invpt"] else np.asarray([], dtype=float)

        if p_true.size == 0:
            continue

        rows_p = binned_metrics(
            p_true,
            res_p,
            bin_edges,
            MIN_TRACKS_PER_BIN,
            method=SIGMA_METHOD,
            n_bootstrap=BOOTSTRAP_ITERS,
            seed=1000 + theta,
        )

        # For fixed-theta gun samples: pT = p * sin(theta)
        s = math.sin(math.radians(theta))
        if s > 0.0:
            pt_edges = bin_edges * s
            rows_pt = binned_metrics(
                pt_true,
                res_pt,
                pt_edges,
                MIN_TRACKS_PER_BIN,
                method=SIGMA_METHOD,
                n_bootstrap=BOOTSTRAP_ITERS,
                seed=2000 + theta,
            )
            rows_invpt = binned_metrics(
                pt_true,
                res_invpt,
                pt_edges,
                MIN_TRACKS_PER_BIN,
                method=SIGMA_METHOD,
                n_bootstrap=BOOTSTRAP_ITERS,
                seed=3000 + theta,
            )
            rows_pt2 = rows_pt_to_pt2(rows_pt)
        else:
            rows_pt = []
            rows_invpt = []
            rows_pt2 = []

        rows_p_by_theta[theta] = rows_p
        rows_pt_by_theta[theta] = rows_pt
        rows_invpt_by_theta[theta] = rows_invpt
        rows_pt2_by_theta[theta] = rows_pt2

        x, y, ye = rows_to_curve(rows_p)
        curve_p[theta] = (x, y)
        err_p[theta] = ye

        x, y, ye = rows_to_curve(rows_pt)
        curve_pt[theta] = (x, y)
        err_pt[theta] = ye

        x, y, ye = rows_to_curve(rows_invpt)
        curve_invpt[theta] = (x, y)
        err_invpt[theta] = ye

        x, y, ye = rows_to_curve(rows_pt2)
        curve_pt2[theta] = (x, y)
        err_pt2[theta] = ye

        print(
            f"Theta {theta:>3}°: bins(p)={len(rows_p)}, bins(pT)={len(rows_pt)}, "
            f"bins(invpT)={len(rows_invpt)}, bins(pT2)={len(rows_pt2)}, "
            f"entries={len(p_true)}"
        )

    write_metrics_csv(CSV_P, rows_p_by_theta, "delta_p_over_p")
    write_metrics_csv(CSV_PT, rows_pt_by_theta, "delta_pt_over_pt")
    write_metrics_csv(CSV_INVPT, rows_invpt_by_theta, "delta_invpt")
    write_metrics_csv(CSV_PT2, rows_pt2_by_theta, "delta_pt_over_pt2_from_ratio")

    print(f"Saved {CSV_P}")
    print(f"Saved {CSV_PT}")
    print(f"Saved {CSV_INVPT}")
    print(f"Saved {CSV_PT2}")

    plot_curves(
        outpath=OUT_P,
        title="Pion Momentum Resolution",
        xlabel=r"Generated Momentum $p$ [GeV/c]",
        ylabel=r"Momentum Resolution $\sigma_p / p$",
        curve_by_theta=curve_p,
        yerr_by_theta=err_p,
    )
    plot_curves(
        outpath=OUT_PT,
        title=r"Pion $p_T$ Resolution",
        xlabel=r"Generated Transverse Momentum $p_T$ [GeV/c]",
        ylabel=r"Transverse Momentum Resolution $\sigma_{p_T} / p_T$",
        curve_by_theta=curve_pt,
        yerr_by_theta=err_pt,
    )
    plot_curves(
        outpath=OUT_INVPT,
        title=r"Pion Curvature Resolution",
        xlabel=r"Generated Transverse Momentum $p_T$ [GeV/c]",
        ylabel=r"$\sigma(1/p_T)$ [GeV$^{-1}$]",
        curve_by_theta=curve_invpt,
        yerr_by_theta=err_invpt,
    )
    plot_curves(
        outpath=OUT_PT2,
        title=r"Pion $p_T$ Resolution",
        xlabel=r"Generated Transverse Momentum $p_T$ [GeV/c]",
        ylabel=r"$\sigma_{p_T} / p_T^2$ [GeV$^{-1}$]",
        curve_by_theta=curve_pt2,
        yerr_by_theta=err_pt2,
    )


if __name__ == "__main__":
    main()
