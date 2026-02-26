#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import glob
import math
import csv
import argparse
from collections import defaultdict, Counter

import ROOT  # noqa: F401  (kept for EDM4hep/ROOT runtime setups)
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from podio.reading import get_reader
import edm4hep  # noqa: F401


# -----------------------------
# Default collection names
# -----------------------------
DEFAULT_SIM_HITS_NAMES = [
    "DCHCollection",
    "VertexBarrelCollection",
    "VertexEndcapCollection",
    "SiWrBCollection",
    "SiWrDCollection",
]

DEFAULT_ASSOC_NAMES = [
    "DCH_DigiSimAssociationCollection",
    "VTXBSimDigiLinks",
    "VTXDSimDigiLinks",
    "SiWrBSimDigiLinks",
    "SiWrDSimDigiLinks",
]

DEFAULT_BARREL_DIGI_COLLECTIONS = [
    "DCH_DigiCollection",
    "VTXBDigis",
    "SiWrBDigis",
]

DEFAULT_ENDCAP_DIGI_COLLECTIONS = [
    "VTXDDigis",
    "SiWrDDigis",
]

A_CONST = 2.99792458e-4  # 0.299792458 * 1e-3 ; pT[GeV] = A_CONST * B[T] / |omega[1/mm]|


# -----------------------------
# Generic helpers
# -----------------------------
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
    # edm4hep python binding style
    if hasattr(track, "trackerHits_size") and hasattr(track, "getTrackerHits"):
        n_hits = track.trackerHits_size()
        for i in range(n_hits):
            yield track.getTrackerHits(i)
        return

    # fallback style
    hits = get_value(track, "trackerHits", "getTrackerHits")
    if hits is None:
        return
    for h in hits:
        yield h


def track_nhits(track):
    if hasattr(track, "trackerHits_size"):
        try:
            return int(track.trackerHits_size())
        except Exception:
            pass
    return sum(1 for _ in iter_tracker_hits(track))


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


def get_tracks_from_event(event, track_collections):
    for name in track_collections:
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


# -----------------------------
# Region filtering helpers
# -----------------------------
def build_hit_oid_set(event, collection_names):
    out = set()
    for name in collection_names:
        coll = get_collection(event, name)
        if coll is None:
            continue
        for hit in coll:
            out.add(oid(hit))
    return out


def is_barrel_track(track, barrel_oids, endcap_oids):
    hits = list(iter_tracker_hits(track))
    if not hits:
        return False

    b = 0
    e = 0
    for h in hits:
        h_id = oid(h)
        if h_id in barrel_oids:
            b += 1
        elif h_id in endcap_oids:
            e += 1
    return e == 0 and b > 0


def is_endcap_track(track, barrel_oids, endcap_oids):
    hits = list(iter_tracker_hits(track))
    if not hits:
        return False

    b = 0
    e = 0
    for h in hits:
        h_id = oid(h)
        if h_id in barrel_oids:
            b += 1
        elif h_id in endcap_oids:
            e += 1
    return b == 0 and e > 0


# -----------------------------
# Truth matching (hit association)
# -----------------------------
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
        associations = get_collection(event, name)
        if associations is None:
            continue
        for assoc in associations:
            digi = get_value(assoc, "from", "getFrom")
            sim = get_value(assoc, "to", "getTo")
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
            "shared_hits": int(shared_hits),
            "nhits": int(nhits),
        })
    return matches


def pass_truth_particle_filters(mc, pdg_abs, gen_status, require_charged):
    if mc is None:
        return False

    if require_charged:
        q = get_value(mc, "charge", "getCharge")
        if q is not None and abs(float(q)) <= 0:
            return False

    if gen_status is not None:
        gs = get_value(mc, "generatorStatus", "getGeneratorStatus")
        if gs != gen_status:
            return False

    if pdg_abs is not None:
        pdg = get_value(mc, "PDG", "getPDG", ["getPdg"])
        if pdg is None:
            pdg = get_value(mc, "pdg", None)
        if pdg is None:
            return False
        if abs(int(pdg)) != pdg_abs:
            return False

    mom = get_vector3(get_value(mc, "momentum", "getMomentum"))
    if mom is None:
        return False

    return True


def select_best_truth_matched_pair(
    event,
    tracks,
    mc_particles,
    sim_hits_names,
    assoc_names,
    pdg_abs,
    gen_status,
    require_charged,
    min_purity,
    min_hits_track,
    min_shared_hits,
):
    matches = match_tracks_to_truth(event, tracks, sim_hits_names, assoc_names)

    # truth index key used by associations
    mc_by_oid_index = {mc.getObjectID().index: mc for mc in mc_particles}

    best = None
    best_score = None

    for m in matches:
        tidx = m["truth_index"]
        if tidx is None:
            continue
        if m["purity"] < min_purity:
            continue
        if m["nhits"] < min_hits_track:
            continue
        if m["shared_hits"] < min_shared_hits:
            continue

        mc = mc_by_oid_index.get(tidx)
        if not pass_truth_particle_filters(mc, pdg_abs, gen_status, require_charged):
            continue

        c2 = track_chi2ndf(m["track"])
        if c2 is None:
            c2 = 1e9

        # maximize purity/shared hits/nhits, then minimize chi2/ndf
        score = (m["purity"], m["shared_hits"], m["nhits"], -c2)
        if best_score is None or score > best_score:
            best_score = score
            best = (mc, m["track"], m)

    return best


# -----------------------------
# Per-file event loop
# -----------------------------
def calculate_event_residuals(
    filename,
    track_collections,
    mc_collection,
    sim_hits_names,
    assoc_names,
    barrel_digi_collections,
    endcap_digi_collections,
    region,
    pdg_abs,
    gen_status,
    require_charged,
    min_purity,
    min_hits_track,
    min_shared_hits,
    atip_location,
    b_field_tesla,
):
    reader = get_reader(filename)
    events = reader.get("events")

    p_true_vals = []
    pt_true_vals = []
    res_p = []
    res_pt = []
    res_invpt = []  # (1/pT_rec - 1/pT_true)

    n_events = 0
    n_has_mc = 0
    n_has_tracks = 0
    n_region_pass = 0
    n_matched = 0
    n_used = 0

    for event in events:
        n_events += 1

        mc_particles = get_collection(event, mc_collection)
        if mc_particles is None or len(mc_particles) == 0:
            continue
        n_has_mc += 1

        tracks = get_tracks_from_event(event, track_collections)
        if len(tracks) == 0:
            continue
        n_has_tracks += 1

        # region filtering
        if region in ("barrel", "endcap"):
            barrel_oids = build_hit_oid_set(event, barrel_digi_collections)
            endcap_oids = build_hit_oid_set(event, endcap_digi_collections)

            if region == "barrel":
                tracks = [t for t in tracks if is_barrel_track(t, barrel_oids, endcap_oids)]
            else:
                tracks = [t for t in tracks if is_endcap_track(t, barrel_oids, endcap_oids)]

            if len(tracks) == 0:
                continue
        n_region_pass += 1

        match = select_best_truth_matched_pair(
            event=event,
            tracks=tracks,
            mc_particles=mc_particles,
            sim_hits_names=sim_hits_names,
            assoc_names=assoc_names,
            pdg_abs=pdg_abs,
            gen_status=gen_status,
            require_charged=require_charged,
            min_purity=min_purity,
            min_hits_track=min_hits_track,
            min_shared_hits=min_shared_hits,
        )
        if match is None:
            continue

        mc_truth, best_track, _m = match
        n_matched += 1

        mom = get_vector3(get_value(mc_truth, "momentum", "getMomentum"))
        if mom is None:
            continue

        px, py, pz = mom
        p_true = math.sqrt(px * px + py * py + pz * pz)
        pt_true = math.hypot(px, py)
        if p_true <= 0 or pt_true <= 0:
            continue

        state = pick_track_state(best_track, atip_location)
        if state is None:
            continue

        omega_mm, tan_lambda = state
        if omega_mm == 0:
            continue

        pt_rec = A_CONST * b_field_tesla / abs(omega_mm)
        if pt_rec <= 0:
            continue
        p_rec = pt_rec * math.sqrt(1.0 + tan_lambda * tan_lambda)

        p_true_vals.append(p_true)
        pt_true_vals.append(pt_true)
        res_p.append((p_rec - p_true) / p_true)
        res_pt.append((pt_rec - pt_true) / pt_true)
        res_invpt.append((1.0 / pt_rec) - (1.0 / pt_true))

        n_used += 1

    summary = {
        "events_total": n_events,
        "events_with_mc": n_has_mc,
        "events_with_tracks": n_has_tracks,
        "events_region_pass": n_region_pass,
        "events_matched": n_matched,
        "events_used": n_used,
    }

    if n_used == 0:
        return None, summary

    data = {
        "p_true": np.asarray(p_true_vals, dtype=float),
        "pt_true": np.asarray(pt_true_vals, dtype=float),
        "res_p": np.asarray(res_p, dtype=float),
        "res_pt": np.asarray(res_pt, dtype=float),
        "res_invpt": np.asarray(res_invpt, dtype=float),
    }
    return data, summary


# -----------------------------
# Bin stats / resolution
# -----------------------------
def bootstrap_sigma68_err(values, n_iter=0, rng=None):
    if n_iter <= 1:
        return float("nan")
    n = len(values)
    if n < 5:
        return float("nan")

    if rng is None:
        rng = np.random.default_rng(12345)

    out = np.empty(n_iter, dtype=float)
    for i in range(n_iter):
        idx = rng.integers(0, n, size=n)
        sample = values[idx]
        q16, q84 = np.percentile(sample, [16.0, 84.0])
        out[i] = 0.5 * (q84 - q16)

    return float(np.std(out, ddof=1))


def binned_metrics(xvals, residuals, bin_edges, min_tracks, bootstrap_iters=0, seed=12345):
    """
    Returns list of dict rows with:
      lo, hi, center, n, median, q16, q84, sigma68, std, sigma68_err, q05, q95
    """
    rng = np.random.default_rng(seed)
    rows = []

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (xvals >= lo) & (xvals < hi)
        vals = residuals[mask]
        n = int(vals.size)
        if n < min_tracks:
            continue

        q05, q16, q50, q84, q95 = np.percentile(vals, [5.0, 16.0, 50.0, 84.0, 95.0])
        sigma68 = 0.5 * (q84 - q16)
        std = float(np.std(vals, ddof=1)) if n > 1 else float("nan")
        sigma68_err = bootstrap_sigma68_err(vals, n_iter=bootstrap_iters, rng=rng)

        rows.append({
            "lo": float(lo),
            "hi": float(hi),
            "center": float(math.sqrt(lo * hi)),  # geometric center for log bins
            "n": n,
            "median": float(q50),
            "q16": float(q16),
            "q84": float(q84),
            "q05": float(q05),
            "q95": float(q95),
            "sigma68": float(sigma68),
            "std": float(std),
            "sigma68_err": float(sigma68_err),
        })

    return rows


def rows_to_curve(rows, method="q68"):
    x = np.asarray([r["center"] for r in rows], dtype=float)
    if method == "std":
        y = np.asarray([r["std"] for r in rows], dtype=float)
        yerr = np.full_like(y, np.nan, dtype=float)
    else:
        y = np.asarray([r["sigma68"] for r in rows], dtype=float)
        yerr = np.asarray([r["sigma68_err"] for r in rows], dtype=float)
    return x, y, yerr


def fit_high_pt_sigma_pt_over_pt2_constant(x_pt, y_sigma_over_pt, xmin=5.0):
    """
    In high pT region, often y = sigma(pT)/pT ~ c * pT, where c ~ sigma(1/pT).
    Fit y = c*x through origin for x>=xmin.
    """
    x = np.asarray(x_pt, dtype=float)
    y = np.asarray(y_sigma_over_pt, dtype=float)
    m = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0) & (x >= xmin)
    if np.count_nonzero(m) < 3:
        return None
    xx = x[m]
    yy = y[m]
    denom = np.sum(xx * xx)
    if denom <= 0:
        return None
    c = np.sum(xx * yy) / denom
    return float(c)


# -----------------------------
# Output helpers
# -----------------------------
def write_metrics_csv(path, theta_to_rows, quantity_name):
    fieldnames = [
        "theta_deg", "quantity",
        "bin_lo", "bin_hi", "bin_center",
        "n",
        "median",
        "q05", "q16", "q84", "q95",
        "sigma68", "std", "sigma68_err",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for theta in sorted(theta_to_rows.keys()):
            for r in theta_to_rows[theta]:
                w.writerow({
                    "theta_deg": theta,
                    "quantity": quantity_name,
                    "bin_lo": r["lo"],
                    "bin_hi": r["hi"],
                    "bin_center": r["center"],
                    "n": r["n"],
                    "median": r["median"],
                    "q05": r["q05"],
                    "q16": r["q16"],
                    "q84": r["q84"],
                    "q95": r["q95"],
                    "sigma68": r["sigma68"],
                    "std": r["std"],
                    "sigma68_err": r["sigma68_err"],
                })


def plot_curves(
    outpath,
    title,
    xlabel,
    ylabel,
    curve_by_theta,
    yerr_by_theta=None,
    xlog=True,
    ylog=True,
    show_errorbars=True,
):
    plt.figure(figsize=(10, 7))

    for theta in sorted(curve_by_theta.keys()):
        x, y = curve_by_theta[theta]
        if len(x) == 0:
            continue

        if yerr_by_theta is not None:
            ye = yerr_by_theta.get(theta, None)
        else:
            ye = None

        if show_errorbars and ye is not None and np.any(np.isfinite(ye)):
            plt.errorbar(
                x, y, yerr=ye, fmt="o--", ms=5, lw=1.5, capsize=2,
                label=f"Theta = {theta}°"
            )
        else:
            plt.plot(x, y, "o--", ms=5, lw=1.8, label=f"Theta = {theta}°")

    if xlog:
        plt.xscale("log")
    if ylog:
        plt.yscale("log")

    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=15)
    plt.grid(True, which="both", linestyle="--", alpha=0.45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()
    print(f"Saved {outpath}")


# -----------------------------
# Argument parsing
# -----------------------------
def parse_csv_list(text):
    return [x.strip() for x in text.split(",") if x.strip()]


def parse_args():
    p = argparse.ArgumentParser(
        description="IDEA tracking resolution vs momentum with truth matching and robust bin metrics."
    )

    # Input discovery
    p.add_argument("--base-dir", default="/ceph/omunkombwe/pi_Barrel")
    p.add_argument("--subdir-glob", default="deg*")
    p.add_argument("--file-glob", default="fit_*_merged.root")
    p.add_argument("--theta-regex", default=r"fit_.*_theta(?P<theta>\d+)deg_.*_merged\.root")

    # Physics
    p.add_argument("--b-field", type=float, default=2.0, help="Tesla")
    p.add_argument("--pdg-abs", type=int, default=211, help="Use -1 for no PDG filter")
    p.add_argument("--gen-status", type=int, default=1, help="Use -1 to disable")
    p.add_argument("--require-charged", action="store_true", default=False)

    # Collections
    p.add_argument("--tracks", default="Fitted_tracks", help="comma-separated track collections")
    p.add_argument("--mc", default="MCParticles")
    p.add_argument("--sim-hits", default=",".join(DEFAULT_SIM_HITS_NAMES))
    p.add_argument("--assocs", default=",".join(DEFAULT_ASSOC_NAMES))
    p.add_argument("--barrel-digis", default=",".join(DEFAULT_BARREL_DIGI_COLLECTIONS))
    p.add_argument("--endcap-digis", default=",".join(DEFAULT_ENDCAP_DIGI_COLLECTIONS))

    # Region
    p.add_argument("--region", choices=["barrel", "endcap", "all"], default="barrel")

    # Matching / quality cuts
    p.add_argument("--min-purity", type=float, default=0.75)
    p.add_argument("--min-hits-track", type=int, default=8)
    p.add_argument("--min-shared-hits", type=int, default=6)
    p.add_argument("--atip-location", type=int, default=1)

    # Binning / stats
    p.add_argument("--p-min", type=float, default=0.2)
    p.add_argument("--p-max", type=float, default=10.0)
    p.add_argument("--p-bins", type=int, default=20)
    p.add_argument("--min-tracks-bin", type=int, default=50)
    p.add_argument("--sigma-method", choices=["q68", "std"], default="q68")
    p.add_argument("--bootstrap-iters", type=int, default=100, help="0 to disable bootstrap errors")

    # Output / plotting
    p.add_argument("--out-prefix", default="pi_idea_truthMatched")
    p.add_argument("--show-errorbars", action="store_true", default=False)
    p.add_argument("--high-pt-fit-min", type=float, default=5.0, help="GeV for sigma(pT)/pT^2 fit printout")

    return p.parse_args()


# -----------------------------
# Main
# -----------------------------
def main():
    args = parse_args()

    pdg_abs = None if args.pdg_abs == -1 else abs(args.pdg_abs)
    gen_status = None if args.gen_status == -1 else args.gen_status

    track_collections = parse_csv_list(args.tracks)
    sim_hits_names = parse_csv_list(args.sim_hits)
    assoc_names = parse_csv_list(args.assocs)
    barrel_digi_collections = parse_csv_list(args.barrel_digis)
    endcap_digi_collections = parse_csv_list(args.endcap_digis)

    theta_re = re.compile(args.theta_regex)

    subdirs = sorted(glob.glob(os.path.join(args.base_dir, args.subdir_glob)))
    if not subdirs:
        raise FileNotFoundError(
            f"No subdirectories found under {args.base_dir} with pattern {args.subdir_glob}"
        )

    # Accumulate arrays per theta
    by_theta = {}

    n_files_seen = 0
    n_files_used = 0

    for subdir in subdirs:
        fit_files = sorted(glob.glob(os.path.join(subdir, args.file_glob)))
        if not fit_files:
            print(f"Warning: no {args.file_glob} files in {subdir}")
            continue

        for fpath in fit_files:
            n_files_seen += 1
            fname = os.path.basename(fpath)
            m = theta_re.match(fname)
            if not m:
                print(f"Warning: filename did not match theta regex: {fpath}")
                continue

            theta = int(m.group("theta"))
            print(f"\nAnalyzing {fpath} (theta={theta} deg)")

            data, summary = calculate_event_residuals(
                filename=fpath,
                track_collections=track_collections,
                mc_collection=args.mc,
                sim_hits_names=sim_hits_names,
                assoc_names=assoc_names,
                barrel_digi_collections=barrel_digi_collections,
                endcap_digi_collections=endcap_digi_collections,
                region=args.region,
                pdg_abs=pdg_abs,
                gen_status=gen_status,
                require_charged=args.require_charged,
                min_purity=args.min_purity,
                min_hits_track=args.min_hits_track,
                min_shared_hits=args.min_shared_hits,
                atip_location=args.atip_location,
                b_field_tesla=args.b_field,
            )

            print(
                "  summary:"
                f" total={summary['events_total']},"
                f" with_mc={summary['events_with_mc']},"
                f" with_tracks={summary['events_with_tracks']},"
                f" region_pass={summary['events_region_pass']},"
                f" matched={summary['events_matched']},"
                f" used={summary['events_used']}"
            )

            if data is None:
                print("  -> no usable events in this file")
                continue

            n_files_used += 1
            slot = by_theta.setdefault(theta, {
                "p_true": [],
                "pt_true": [],
                "res_p": [],
                "res_pt": [],
                "res_invpt": [],
            })
            slot["p_true"].append(data["p_true"])
            slot["pt_true"].append(data["pt_true"])
            slot["res_p"].append(data["res_p"])
            slot["res_pt"].append(data["res_pt"])
            slot["res_invpt"].append(data["res_invpt"])

    if n_files_used == 0 or len(by_theta) == 0:
        raise RuntimeError("No usable files/events after selection.")

    print(f"\nFiles seen: {n_files_seen}, files used: {n_files_used}")
    print(f"Thetas found: {sorted(by_theta.keys())}")

    # Common p-bin edges
    p_edges = np.logspace(np.log10(args.p_min), np.log10(args.p_max), args.p_bins + 1)

    # Per-theta metrics
    rows_p_by_theta = {}
    rows_pt_by_theta = {}
    rows_invpt_by_theta = {}

    # Curves for plotting
    curve_p = {}
    curve_pt = {}
    curve_invpt = {}
    err_p = {}
    err_pt = {}
    err_invpt = {}

    for theta in sorted(by_theta.keys()):
        p_true = np.concatenate(by_theta[theta]["p_true"]) if by_theta[theta]["p_true"] else np.asarray([], dtype=float)
        pt_true = np.concatenate(by_theta[theta]["pt_true"]) if by_theta[theta]["pt_true"] else np.asarray([], dtype=float)
        res_p = np.concatenate(by_theta[theta]["res_p"]) if by_theta[theta]["res_p"] else np.asarray([], dtype=float)
        res_pt = np.concatenate(by_theta[theta]["res_pt"]) if by_theta[theta]["res_pt"] else np.asarray([], dtype=float)
        res_invpt = np.concatenate(by_theta[theta]["res_invpt"]) if by_theta[theta]["res_invpt"] else np.asarray([], dtype=float)

        if p_true.size == 0:
            continue

        # sigma_p/p vs p
        rows_p = binned_metrics(
            xvals=p_true,
            residuals=res_p,
            bin_edges=p_edges,
            min_tracks=args.min_tracks_bin,
            bootstrap_iters=args.bootstrap_iters,
            seed=1000 + theta,
        )

        # sigma_pT/pT vs pT (theta-scaled bins for gun-like samples)
        s = math.sin(math.radians(theta))
        if s > 0:
            pt_edges = p_edges * s
            rows_pt = binned_metrics(
                xvals=pt_true,
                residuals=res_pt,
                bin_edges=pt_edges,
                min_tracks=args.min_tracks_bin,
                bootstrap_iters=args.bootstrap_iters,
                seed=2000 + theta,
            )
            rows_invpt = binned_metrics(
                xvals=pt_true,
                residuals=res_invpt,  # absolute residual in 1/GeV
                bin_edges=pt_edges,
                min_tracks=args.min_tracks_bin,
                bootstrap_iters=args.bootstrap_iters,
                seed=3000 + theta,
            )
        else:
            rows_pt = []
            rows_invpt = []

        rows_p_by_theta[theta] = rows_p
        rows_pt_by_theta[theta] = rows_pt
        rows_invpt_by_theta[theta] = rows_invpt

        x, y, ye = rows_to_curve(rows_p, args.sigma_method)
        curve_p[theta] = (x, y)
        err_p[theta] = ye

        x, y, ye = rows_to_curve(rows_pt, args.sigma_method)
        curve_pt[theta] = (x, y)
        err_pt[theta] = ye

        # For invpt, always use q68 (or std fallback if requested)
        x, y, ye = rows_to_curve(rows_invpt, args.sigma_method)
        curve_invpt[theta] = (x, y)
        err_invpt[theta] = ye

        # high-pT constant printout: sigma(pT)/pT^2 from y = sigma(pT)/pT
        if len(rows_pt) > 0:
            x_fit = np.asarray([r["center"] for r in rows_pt], dtype=float)
            y_fit = np.asarray(
                [r["sigma68"] if args.sigma_method == "q68" else r["std"] for r in rows_pt],
                dtype=float
            )
            c = fit_high_pt_sigma_pt_over_pt2_constant(
                x_pt=x_fit,
                y_sigma_over_pt=y_fit,
                xmin=args.high_pt_fit_min
            )
            if c is not None:
                print(f"Theta {theta:>3} deg: high-pT approx sigma(pT)/pT^2 ≈ {c:.3e} GeV^-1")

    # Write CSV tables
    csv_p = f"{args.out_prefix}_binned_metrics_p.csv"
    csv_pt = f"{args.out_prefix}_binned_metrics_pt.csv"
    csv_invpt = f"{args.out_prefix}_binned_metrics_invpt.csv"

    write_metrics_csv(csv_p, rows_p_by_theta, "delta_p_over_p")
    write_metrics_csv(csv_pt, rows_pt_by_theta, "delta_pt_over_pt")
    write_metrics_csv(csv_invpt, rows_invpt_by_theta, "delta_invpt")

    print(f"\nSaved {csv_p}")
    print(f"Saved {csv_pt}")
    print(f"Saved {csv_invpt}")

    # Labels / title
    method_label = "68% half-width" if args.sigma_method == "q68" else "std dev"
    region_label = args.region.capitalize()

    # Plot sigma_p/p vs p
    plot_curves(
        outpath=f"{args.out_prefix}_resolution_p.png",
        title=f"Pion Momentum Resolution ({region_label}, truth-matched, {method_label})",
        xlabel=r"Generated Momentum $p$ [GeV/$c$]",
        ylabel=r"Momentum Resolution $\sigma_p/p$",
        curve_by_theta=curve_p,
        yerr_by_theta=err_p,
        xlog=True,
        ylog=True,
        show_errorbars=args.show_errorbars and args.sigma_method == "q68",
    )

    # Plot sigma_pT/pT vs pT
    plot_curves(
        outpath=f"{args.out_prefix}_resolution_pt.png",
        title=f"Pion $p_T$ Resolution ({region_label}, truth-matched, {method_label})",
        xlabel=r"Generated Transverse Momentum $p_T$ [GeV/$c$]",
        ylabel=r"Transverse Momentum Resolution $\sigma_{p_T}/p_T$",
        curve_by_theta=curve_pt,
        yerr_by_theta=err_pt,
        xlog=True,
        ylog=True,
        show_errorbars=args.show_errorbars and args.sigma_method == "q68",
    )

    # Plot sigma(1/pT) vs pT
    plot_curves(
        outpath=f"{args.out_prefix}_resolution_invpt.png",
        title=f"Pion Curvature-like Resolution ({region_label}, truth-matched, {method_label})",
        xlabel=r"Generated Transverse Momentum $p_T$ [GeV/$c$]",
        ylabel=r"$\sigma(1/p_T)$ [GeV$^{-1}$]",
        curve_by_theta=curve_invpt,
        yerr_by_theta=err_invpt,
        xlog=True,
        ylog=True,
        show_errorbars=args.show_errorbars and args.sigma_method == "q68",
    )

    print("\nDone.")


if __name__ == "__main__":
    main()