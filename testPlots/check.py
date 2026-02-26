#!/usr/bin/env python3
"""Quick omega-unit consistency check for fitted EDM4hep tracks.

This script computes
    k = median(pt_true * |omega| / B)
from truth-matched tracks and compares k against common conventions:
  - k ~ 1.0           -> pt = B / |omega|
  - k ~ 0.299792458   -> pt = 0.299792458 * B / |omega|
  - k ~ 2.99792458e-4 -> pt = 2.99792458e-4 * B / |omega|

Usage examples:
  python3 check.py
  python3 check.py --base-dir /ceph/omunkombwe/pi_Barrel
  python3 check.py --files "/ceph/omunkombwe/pi_Barrel/deg*/fit_*_merged.root"
"""

import argparse
import glob
import math
import os
from collections import defaultdict
from importlib import util

import numpy as np


def load_mom_reso_module(script_dir):
    path = os.path.join(script_dir, "mom_reso.py")
    spec = util.spec_from_file_location("mom_reso_mod", path)
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_csv_list(text):
    return [x.strip() for x in text.split(",") if x.strip()]


def discover_files(args):
    files = []
    if args.files:
        for pattern in args.files:
            files.extend(glob.glob(pattern))
    else:
        subdirs = sorted(glob.glob(os.path.join(args.base_dir, args.subdir_glob)))
        for subdir in subdirs:
            files.extend(sorted(glob.glob(os.path.join(subdir, args.file_glob))))

    uniq = sorted(set(files))
    if args.max_files is not None:
        uniq = uniq[: args.max_files]
    return uniq


def sigma68(values):
    q16, q84 = np.percentile(values, [16.0, 84.0])
    return 0.5 * (q84 - q16)


def summarize(values):
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return None
    q16, q50, q84 = np.percentile(arr, [16.0, 50.0, 84.0])
    return {
        "n": int(arr.size),
        "q16": float(q16),
        "median": float(q50),
        "q84": float(q84),
        "sigma68": float(0.5 * (q84 - q16)),
        "mean": float(np.mean(arr)),
    }


def select_recommendation(k_median):
    candidates = {
        "pt = B/|omega| (A_CONST=1.0)": 1.0,
        "pt = 0.299792458*B/|omega| (A_CONST=0.299792458)": 0.299792458,
        "pt = 2.99792458e-4*B/|omega| (A_CONST=2.99792458e-4)": 2.99792458e-4,
    }

    best_name = None
    best_scale = None
    best_rel = None
    for name, scale in candidates.items():
        rel = abs(k_median / scale - 1.0)
        if best_rel is None or rel < best_rel:
            best_rel = rel
            best_name = name
            best_scale = scale

    return best_name, best_scale, best_rel


def run(args):
    here = os.path.dirname(os.path.abspath(__file__))
    m = load_mom_reso_module(here)

    # Reuse mom_reso matching/filter helpers, but with CLI overrides.
    m.PDG_ABS = None if args.pdg_abs < 0 else abs(args.pdg_abs)
    m.GEN_STATUS = None if args.gen_status < 0 else args.gen_status
    m.MIN_PURITY = args.min_purity

    active_subdetectors = parse_csv_list(args.active_subdetectors)
    sim_hits_names, assoc_names, active_digi_names = m.active_collection_names(active_subdetectors)

    files = discover_files(args)
    if not files:
        raise FileNotFoundError("No input files found. Check --files or --base-dir patterns.")

    print("Using files:")
    for f in files:
        print(f"  {f}")

    k_values = []
    resid_by_const = defaultdict(list)
    per_file_k = {}

    max_events = args.max_events
    total_events_seen = 0

    constants = [1.0, 0.299792458, 2.99792458e-4]

    for path in files:
        reader = m.get_reader(path)
        events = reader.get("events")

        this_file_k = []

        for event in events:
            total_events_seen += 1
            if max_events is not None and total_events_seen > max_events:
                break

            mc_particles = m.get_collection(event, "MCParticles")
            if mc_particles is None or len(mc_particles) == 0:
                continue

            tracks = m.get_tracks_from_event(event)
            if len(tracks) == 0:
                continue

            tracks = m.filter_tracks_by_active_digis(
                event=event,
                tracks=tracks,
                active_digi_names=active_digi_names,
                mode=args.track_filter_mode,
            )
            if len(tracks) == 0:
                continue

            pair = m.select_best_truth_matched_pair(
                event=event,
                tracks=tracks,
                mc_particles=mc_particles,
                sim_hits_names=sim_hits_names,
                assoc_names=assoc_names,
            )
            if pair is None:
                continue

            mc_truth, best_track = pair
            mom = m.get_vector3(m.get_value(mc_truth, "momentum", "getMomentum"))
            if mom is None:
                continue

            px, py, _ = mom
            pt_true = math.hypot(px, py)
            if pt_true <= 0:
                continue

            state = m.pick_track_state(best_track, args.atip_location)
            if state is None:
                continue

            omega, _ = state
            if omega == 0:
                continue

            kval = pt_true * abs(omega) / args.b_field
            k_values.append(kval)
            this_file_k.append(kval)

            for c in constants:
                pt_rec = c * args.b_field / abs(omega)
                resid_by_const[c].append((pt_rec - pt_true) / pt_true)

        if this_file_k:
            per_file_k[path] = summarize(this_file_k)

        if max_events is not None and total_events_seen > max_events:
            break

    summary_k = summarize(k_values)
    if summary_k is None:
        raise RuntimeError("No matched tracks were selected; cannot estimate omega convention.")

    best_name, best_scale, best_rel = select_recommendation(summary_k["median"])

    print("\n=== Omega Convention Check ===")
    print(f"B field used: {args.b_field} T")
    print(f"Matched tracks used: {summary_k['n']}")
    print(f"k = median(pt_true*|omega|/B) = {summary_k['median']:.9g}")
    print(f"k 16/84 percentiles: {summary_k['q16']:.9g} / {summary_k['q84']:.9g}")
    print(f"k sigma68: {summary_k['sigma68']:.9g}")
    print(f"Recommended convention: {best_name}")
    print(f"Relative distance to recommendation: {best_rel:.3%}")

    print("\nResidual medians by candidate constant:")
    for c in constants:
        s = summarize(resid_by_const[c])
        if s is None:
            continue
        print(
            f"  A_CONST={c:.9g}: median(delta_pt/pt)={s['median']:.6g}, "
            f"sigma68={s['sigma68']:.6g}, n={s['n']}"
        )

    if args.show_per_file:
        print("\nPer-file k medians:")
        for f in files:
            s = per_file_k.get(f)
            if s is None:
                print(f"  {f}: no selected tracks")
            else:
                print(f"  {f}: k_median={s['median']:.9g}, n={s['n']}")


def build_parser():
    p = argparse.ArgumentParser(description="Quick omega-unit consistency check from truth-matched tracks.")

    p.add_argument("--files", nargs="*", default=None, help="Input files or glob patterns.")
    p.add_argument("--base-dir", default="/ceph/omunkombwe/pi_Barrel", help="Base directory for discovery mode.")
    p.add_argument("--subdir-glob", default="deg*", help="Subdir glob under base-dir.")
    p.add_argument("--file-glob", default="fit_*_merged.root", help="File glob inside subdirs.")

    p.add_argument("--max-files", type=int, default=None, help="Max files to process.")
    p.add_argument("--max-events", type=int, default=None, help="Max events across all files.")

    p.add_argument("--b-field", type=float, default=2.0, help="Magnetic field in Tesla.")
    p.add_argument("--atip-location", type=int, default=1, help="TrackState location code (AtIP=1).")

    p.add_argument("--pdg-abs", type=int, default=211, help="Abs(PDG). Use -1 to disable.")
    p.add_argument("--gen-status", type=int, default=1, help="Generator status. Use -1 to disable.")
    p.add_argument("--min-purity", type=float, default=0.75, help="Minimum truth-match purity.")

    p.add_argument(
        "--active-subdetectors",
        default="DCH,VTXB,VTXD,SIWRB,SIWRD",
        help="Comma-separated subdetectors used for truth matching/filtering.",
    )
    p.add_argument(
        "--track-filter-mode",
        choices=["off", "inclusive", "exclusive"],
        default="inclusive",
        help="Track filter mode w.r.t active digi collections.",
    )

    p.add_argument("--show-per-file", action="store_true", help="Print per-file k medians.")

    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    run(args)
