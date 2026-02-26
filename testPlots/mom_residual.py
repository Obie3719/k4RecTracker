#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
momentum_residuals_by_theta.py

Compute momentum residual histograms per angle:
  r_p  = (p_rec  - p_true ) / p_true
  r_pt = (pt_rec - pt_true) / pt_true

Truth matching:
  track -> digi hits -> associated sim hits -> MC truth index
(best match by purity/shared hits/chi2-ndf, with PDG/gen-status filters)

Outputs:
  - mom_residuals_theta{theta}.png
  - mom_residuals_overlay_p.png
  - mom_residuals_overlay_pt.png
"""

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
# CONFIG
# =========================

# Input discovery
BASE_DIR = "/ceph/omunkombwe/pi_forward"
SUBDIR_GLOB = "deg*"
FIT_GLOB = "fit_*_merged.root"
FILENAME_RE = re.compile(r"fit_.*_theta(?P<theta>\d+)deg_.*_merged\.root")

MAX_EVENTS_PER_FILE = 100000  # set None for all events

# Physics / selection
B_FIELD = 2.0
A_CONST = 1.0  # pT[GeV] = A_CONST * B[T] / |omega[1/mm]|
ATIP_LOCATION = 1
PDG_ABS = 211
GEN_STATUS = 1
MIN_PURITY = 0.75

TRACK_COLLECTION_CANDIDATES = ["Fitted_tracks"]

# Subdetector setup (edit to scan)
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

# default: all 5 combined
ACTIVE_SUBDETECTORS = ["DCH", "VTXB", "VTXD", "SIWRB", "SIWRD"]

# Track filtering with active digi collections
# "off"       : no filter
# "inclusive" : keep if >=1 hit in active digi collections
# "exclusive" : keep if all hits in active digi collections
TRACK_FILTER_MODE = "off"

# Histogram settings
NBINS_RES = 140
RANGE_QUANTILES = (0.5, 99.5)  # robust global range
PAD_FRAC = 0.08

# Outputs
OUT_PER_THETA_FMT = "pi_residuals_theta{theta}.png"
OUT_OVERLAY_P = "pi_residuals_overlay_p.png"
OUT_OVERLAY_PT = "pi_residuals_overlay_pt.png"


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
        for g in alt_getters:
            if hasattr(obj, g):
                try:
                    return getattr(obj, g)()
                except Exception:
                    pass

    if hasattr(obj, attr):
        v = getattr(obj, attr)
        return v() if callable(v) else v

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
        n = track.trackerHits_size()
        for i in range(n):
            yield track.getTrackerHits(i)
        return

    hits = get_value(track, "trackerHits", "getTrackerHits")
    if hits is None:
        return
    for h in hits:
        yield h


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
    for name in TRACK_COLLECTION_CANDIDATES:
        coll = get_collection(event, name)
        if coll is not None:
            return list(coll)
    return []


def track_chi2ndf(track):
    chi2 = get_value(track, "chi2", "getChi2")
    ndf = get_value(track, "ndf", "getNdf")
    if chi2 is None or ndf is None:
        return None
    if chi2 <= 0 or ndf <= 0:
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
    sim_to_truth = {}
    for name in sim_hits_names:
        sim_hits = get_collection(event, name)
        if sim_hits is None:
            continue
        for sim in sim_hits:
            mc = get_value(sim, "particle", "getParticle")
            if mc is None:
                continue
            sim_to_truth[oid(sim)] = mc.getObjectID().index
    return sim_to_truth


def build_digi_truth_map(event, assoc_names, sim_to_truth):
    digi_truth = defaultdict(set)
    for name in assoc_names:
        assocs = get_collection(event, name)
        if assocs is None:
            continue
        for a in assocs:
            digi = get_value(a, "from", "getFrom")
            sim = get_value(a, "to", "getTo")
            if digi is None or sim is None:
                continue
            tidx = sim_to_truth.get(oid(sim))
            if tidx is not None:
                digi_truth[oid(digi)].add(tidx)
    return digi_truth


def match_tracks_to_truth(event, tracks, sim_hits_names, assoc_names):
    sim_to_truth = build_sim_to_truth(event, sim_hits_names)
    digi_truth = build_digi_truth_map(event, assoc_names, sim_to_truth)

    matches = []
    for trk in tracks:
        counts = Counter()
        nhits = 0
        for digi in iter_tracker_hits(trk):
            nhits += 1
            for tidx in digi_truth.get(oid(digi), ()):
                counts[tidx] += 1

        if counts:
            best_idx, shared_hits = counts.most_common(1)[0]
        else:
            best_idx, shared_hits = None, 0

        purity = (shared_hits / nhits) if nhits > 0 else 0.0
        matches.append({
            "track": trk,
            "truth_index": best_idx,
            "purity": purity,
            "shared_hits": shared_hits,
            "nhits": nhits,
        })
    return matches


def select_best_truth_matched_pair(event, tracks, mc_particles, sim_hits_names, assoc_names):
    matches = match_tracks_to_truth(event, tracks, sim_hits_names, assoc_names)
    mc_by_idx = {mc.getObjectID().index: mc for mc in mc_particles}

    best_pair = None
    best_score = None

    for m in matches:
        tidx = m["truth_index"]
        if tidx is None:
            continue
        if m["purity"] < MIN_PURITY:
            continue

        mc = mc_by_idx.get(tidx)
        if mc is None:
            continue

        if GEN_STATUS is not None:
            gs = get_value(mc, "generatorStatus", "getGeneratorStatus")
            if gs != GEN_STATUS:
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

        c2 = track_chi2ndf(m["track"])
        if c2 is None:
            c2 = 1e9

        # maximize purity/shared hits, then minimize chi2/ndf
        score = (m["purity"], m["shared_hits"], -c2)

        if best_score is None or score > best_score:
            best_score = score
            best_pair = (mc, m["track"])

    return best_pair


# =========================
# Residual collection
# =========================

def calculate_momentum_residuals_file(filepath, sim_hits_names, assoc_names, active_digi_names):
    reader = get_reader(filepath)
    events = reader.get("events")

    res_p = []
    res_pt = []

    cutflow = {
        "total": 0,
        "with_mc": 0,
        "with_tracks": 0,
        "subdet_pass": 0,
        "matched": 0,
        "used": 0,
    }

    for iev, event in enumerate(events):
        if MAX_EVENTS_PER_FILE is not None and iev >= MAX_EVENTS_PER_FILE:
            break

        cutflow["total"] += 1

        mc_particles = get_collection(event, "MCParticles")
        if mc_particles is None or len(mc_particles) == 0:
            continue
        cutflow["with_mc"] += 1

        tracks = get_tracks_from_event(event)
        if len(tracks) == 0:
            continue
        cutflow["with_tracks"] += 1

        tracks = filter_tracks_by_active_digis(
            event=event,
            tracks=tracks,
            active_digi_names=active_digi_names,
            mode=TRACK_FILTER_MODE,
        )
        if len(tracks) == 0:
            continue
        cutflow["subdet_pass"] += 1

        pair = select_best_truth_matched_pair(
            event=event,
            tracks=tracks,
            mc_particles=mc_particles,
            sim_hits_names=sim_hits_names,
            assoc_names=assoc_names,
        )
        if pair is None:
            continue
        cutflow["matched"] += 1

        mc_truth, track = pair

        mom = get_vector3(get_value(mc_truth, "momentum", "getMomentum"))
        if mom is None:
            continue
        px, py, pz = mom
        p_true = math.sqrt(px * px + py * py + pz * pz)
        pt_true = math.hypot(px, py)
        if p_true <= 0 or pt_true <= 0:
            continue

        st = pick_track_state(track, ATIP_LOCATION)
        if st is None:
            continue
        omega_mm, tan_lambda = st
        if omega_mm == 0:
            continue

        pt_rec = A_CONST * B_FIELD / abs(omega_mm)
        p_rec = pt_rec * math.sqrt(1.0 + tan_lambda * tan_lambda)

        res_p.append(p_rec - p_true)
        res_pt.append(pt_rec - pt_true)
        cutflow["used"] += 1

    return np.asarray(res_p, dtype=float), np.asarray(res_pt, dtype=float), cutflow


# =========================         
# Plotting
# =========================

def robust_range(values, qlo=0.5, qhi=99.5, pad_frac=0.08, fallback=(-0.05, 0.05)):
    if values.size == 0:
        return fallback
    lo, hi = np.percentile(values, [qlo, qhi])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.min(values))
        hi = float(np.max(values))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return fallback
    pad = pad_frac * (hi - lo)
    return (lo - pad, hi + pad)


def summary_text(arr):
    if arr.size == 0:
        return "N=0"
    q16, q50, q84 = np.percentile(arr, [16, 50, 84])
    mean = np.mean(arr)
    std = np.std(arr, ddof=1) if arr.size > 1 else 0.0
    return f"N={arr.size}, mean={mean:.3e}, std={std:.3e}, median={q50:.3e}, q16={q16:.3e}, q84={q84:.3e}"


def plot_per_theta(theta, rp, rpt, range_p, range_pt):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    axes[0].hist(rp, bins=NBINS_RES, range=range_p)
    axes[0].axvline(0.0, color="k", linestyle="--", linewidth=1.0)
    axes[0].set_title(rf"$\Delta p/p$ (theta={theta}°)")
    axes[0].set_xlabel(r"$(p_{rec}-p_{true})/p_{true}$")
    axes[0].set_ylabel("Entries")
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(rpt, bins=NBINS_RES, range=range_pt)
    axes[1].axvline(0.0, color="k", linestyle="--", linewidth=1.0)
    axes[1].set_title(rf"$\Delta p_T/p_T$ (theta={theta}°)")
    axes[1].set_xlabel(r"$(p_{T,rec}-p_{T,true})/p_{T,true}$")
    axes[1].set_ylabel("Entries")
    axes[1].grid(True, alpha=0.3)

    txt = f"Δp/p: {summary_text(rp)}\nΔpT/pT: {summary_text(rpt)}"
    fig.suptitle(f"Momentum Residuals by Angle\n{txt}", fontsize=12, y=1.02)

    plt.tight_layout()
    outname = OUT_PER_THETA_FMT.format(theta=theta)
    plt.savefig(outname, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {outname}")


def plot_overlay(by_theta, key, outname, xlabel):
    plt.figure(figsize=(10, 7))
    for theta in sorted(by_theta.keys()):
        arr = by_theta[theta][key]
        if arr.size < 10:
            continue
        # normalized overlay for shape comparison
        plt.hist(
            arr,
            bins=NBINS_RES,
            density=True,
            histtype="step",
            linewidth=1.8,
            label=f"Theta = {theta}° (N={arr.size})",
        )

    plt.axvline(0.0, color="k", linestyle="--", linewidth=1.0)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel("Normalized entries", fontsize=12)
    plt.title("Residual Shape Comparison by Angle", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(outname, dpi=250)
    plt.close()
    print(f"Saved {outname}")


# =========================
# Main
# =========================

def main():
    if len(ACTIVE_SUBDETECTORS) == 0:
        raise ValueError("ACTIVE_SUBDETECTORS is empty.")

    sim_hits_names, assoc_names, active_digi_names = active_collection_names(ACTIVE_SUBDETECTORS)

    print("Active subdetectors:", ACTIVE_SUBDETECTORS)
    print("Track filter mode:", TRACK_FILTER_MODE)
    print(f"Truth filters: PDG_ABS={PDG_ABS}, GEN_STATUS={GEN_STATUS}, MIN_PURITY={MIN_PURITY}")
    print("Sim-hit collections:", sim_hits_names)
    print("Association collections:", assoc_names)
    print("Digi collections:", active_digi_names)

    subdirs = sorted(glob.glob(os.path.join(BASE_DIR, SUBDIR_GLOB)))
    if not subdirs:
        raise FileNotFoundError(f"No subdirs under {BASE_DIR} with pattern {SUBDIR_GLOB}")

    # aggregate arrays by theta
    temp = defaultdict(lambda: {"res_p": [], "res_pt": []})

    for subdir in subdirs:
        files = sorted(glob.glob(os.path.join(subdir, FIT_GLOB)))
        if not files:
            print(f"Warning: no {FIT_GLOB} in {subdir}")
            continue

        for fp in files:
            fname = os.path.basename(fp)
            m = FILENAME_RE.match(fname)
            if not m:
                print(f"Warning: skip unrecognized file name: {fname}")
                continue
            theta = int(m.group("theta"))

            print(f"\nAnalyzing {fp} (theta={theta} deg)")
            rp, rpt, cf = calculate_momentum_residuals_file(
                fp, sim_hits_names, assoc_names, active_digi_names
            )

            print(
                "  summary: "
                f"total={cf['total']}, with_mc={cf['with_mc']}, with_tracks={cf['with_tracks']}, "
                f"subdet_pass={cf['subdet_pass']}, matched={cf['matched']}, used={cf['used']}"
            )

            if rp.size == 0 or rpt.size == 0:
                print("  -> no residual entries from this file")
                continue

            temp[theta]["res_p"].append(rp)
            temp[theta]["res_pt"].append(rpt)

    if not temp:
        raise RuntimeError(
            "No residual data collected. Check collections/cuts. "
            f"(PDG_ABS={PDG_ABS}, GEN_STATUS={GEN_STATUS}, MIN_PURITY={MIN_PURITY})"
        )

    # finalize by theta
    by_theta = {}
    for th in sorted(temp.keys()):
        rp = np.concatenate(temp[th]["res_p"]) if temp[th]["res_p"] else np.asarray([], dtype=float)
        rpt = np.concatenate(temp[th]["res_pt"]) if temp[th]["res_pt"] else np.asarray([], dtype=float)
        by_theta[th] = {"res_p": rp, "res_pt": rpt}
        print(f"Theta {th:>3}°: N(Δp/p)={rp.size}, N(ΔpT/pT)={rpt.size}")

    # global ranges for consistent per-theta comparison
    all_rp = np.concatenate([by_theta[t]["res_p"] for t in by_theta if by_theta[t]["res_p"].size > 0])
    all_rpt = np.concatenate([by_theta[t]["res_pt"] for t in by_theta if by_theta[t]["res_pt"].size > 0])

    rp_range = robust_range(
        all_rp,
        qlo=RANGE_QUANTILES[0],
        qhi=RANGE_QUANTILES[1],
        pad_frac=PAD_FRAC,
        fallback=(-0.05, 0.05),
    )
    rpt_range = robust_range(
        all_rpt,
        qlo=RANGE_QUANTILES[0],
        qhi=RANGE_QUANTILES[1],
        pad_frac=PAD_FRAC,
        fallback=(-0.05, 0.05),
    )

    print(f"Global histogram range Δp/p: {rp_range}")
    print(f"Global histogram range ΔpT/pT: {rpt_range}")

    # per-theta hist plots (entries vs residual)
    for theta in sorted(by_theta.keys()):
        rp = by_theta[theta]["res_p"]
        rpt = by_theta[theta]["res_pt"]
        if rp.size < 10 or rpt.size < 10:
            continue
        plot_per_theta(theta, rp, rpt, rp_range, rpt_range)

    # overlay plots (normalized)
    plot_overlay(
        by_theta,
        key="res_p",
        outname=OUT_OVERLAY_P,
        xlabel=r"$(p_{rec}-p_{true})/p_{true}$",
    )
    plot_overlay(
        by_theta,
        key="res_pt",
        outname=OUT_OVERLAY_PT,
        xlabel=r"$(p_{T,rec}-p_{T,true})/p_{T,true}$",
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
