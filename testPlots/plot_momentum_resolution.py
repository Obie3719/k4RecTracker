#!/usr/bin/env python3

import argparse
import glob
import math
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from podio.reading import get_reader
import edm4hep


C_LIGHT = 2.99792458e8
A_CONST = C_LIGHT * 1e3 * 1e-15  # 2.99792458e-4


def parse_nominal_p(filename: str) -> Optional[float]:
    match = re.search(r"P([0-9.]+)-([0-9.]+)GeV", os.path.basename(filename))
    if not match:
        return None
    p_min = float(match.group(1))
    p_max = float(match.group(2))
    return 0.5 * (p_min + p_max)


def parse_theta_deg(filename: str) -> Optional[int]:
    match = re.search(r"theta([0-9]+)deg", os.path.basename(filename))
    if not match:
        return None
    return int(match.group(1))


def get_value(obj, attr: str, getter: Optional[str] = None, alt_getters: Optional[List[str]] = None):
    if getter and hasattr(obj, getter):
        return getattr(obj, getter)()
    if alt_getters:
        for alt in alt_getters:
            if hasattr(obj, alt):
                return getattr(obj, alt)()
    if hasattr(obj, attr):
        val = getattr(obj, attr)
        return val() if callable(val) else val
    return None


def get_vector3(vec) -> Optional[Tuple[float, float, float]]:
    if vec is None:
        return None
    x = get_value(vec, "x", "getX")
    y = get_value(vec, "y", "getY")
    z = get_value(vec, "z", "getZ")
    if x is None or y is None or z is None:
        return None
    return float(x), float(y), float(z)


def select_truth_momentum(mc_particles, pdg_abs: Optional[int]) -> Optional[Tuple[float, float, float]]:
    best = None
    best_p2 = -1.0

    for particle in mc_particles:
        gen = get_value(particle, "generatorStatus", "getGeneratorStatus")
        if gen is None or gen != 1:
            continue
        pdg = get_value(particle, "PDG", "getPDG", ["getPdg"])
        if pdg is None:
            pdg = get_value(particle, "pdg", None)
        if pdg_abs is not None and pdg is not None and abs(int(pdg)) != pdg_abs:
            continue

        mom = get_vector3(get_value(particle, "momentum", "getMomentum"))
        if mom is None:
            continue
        p2 = mom[0] ** 2 + mom[1] ** 2 + mom[2] ** 2
        if p2 > best_p2:
            best_p2 = p2
            best = mom

    if best is not None:
        return best

    for particle in mc_particles:
        mom = get_vector3(get_value(particle, "momentum", "getMomentum"))
        if mom is None:
            continue
        p2 = mom[0] ** 2 + mom[1] ** 2 + mom[2] ** 2
        if p2 > best_p2:
            best_p2 = p2
            best = mom

    return best


def choose_best_track(tracks) -> Optional[object]:
    best_track = None
    best_val = None

    for track in tracks:
        chi2 = get_value(track, "chi2", "getChi2")
        ndf = get_value(track, "ndf", "getNdf")
        if chi2 is None or ndf is None:
            continue
        if chi2 <= 0 or ndf <= 0:
            continue

        states = get_value(track, "trackStates", "getTrackStates")
        if states is None or len(states) == 0:
            continue

        val = float(chi2) / float(ndf)
        if best_val is None or val < best_val:
            best_val = val
            best_track = track

    return best_track


def pick_track_state(track, atip_location: int) -> Optional[Tuple[float, float]]:
    states = get_value(track, "trackStates", "getTrackStates")
    if states is None or len(states) == 0:
        return None

    chosen = None
    for state in states:
        loc = get_value(state, "location", "getLocation")
        if loc == atip_location:
            chosen = state
            break

    if chosen is None:
        chosen = states[0]

    omega = get_value(chosen, "omega", "getOmega")
    tan_lambda = get_value(chosen, "tanLambda", "getTanLambda")
    if omega is None or tan_lambda is None:
        return None

    return float(omega), float(tan_lambda)


def compute_resolutions(
    filename: str,
    track_collection: str,
    bfield: float,
    atip_location: int,
    pdg_abs: Optional[int],
    min_tracks: int,
    max_events: Optional[int],
) -> Optional[Dict[str, float]]:
    reader = get_reader(filename)
    events = reader.get("events")

    res_p = []
    res_pt = []
    truth_p = []
    truth_pt = []

    for iev, event in enumerate(events):
        if max_events is not None and iev >= max_events:
            break

        mc_particles = event.get("MCParticles")
        if len(mc_particles) == 0:
            continue

        truth_mom = select_truth_momentum(mc_particles, pdg_abs)
        if truth_mom is None:
            continue

        px, py, pz = truth_mom
        p_true = math.sqrt(px * px + py * py + pz * pz)
        pt_true = math.sqrt(px * px + py * py)
        if p_true <= 0 or pt_true <= 0:
            continue

        try:
            tracks = event.get(track_collection)
        except KeyError as exc:
            raise RuntimeError(f"Collection '{track_collection}' not found in {filename}") from exc

        if len(tracks) == 0:
            continue

        best_track = choose_best_track(tracks)
        if best_track is None:
            continue

        state = pick_track_state(best_track, atip_location)
        if state is None:
            continue

        omega_val, tan_val = state
        if omega_val == 0:
            continue

        pt_reco = A_CONST * bfield / abs(omega_val)
        p_reco = pt_reco * math.sqrt(1.0 + tan_val * tan_val)

        res_p.append((p_reco - p_true) / p_true)
        res_pt.append((pt_reco - pt_true) / pt_true)
        truth_p.append(p_true)
        truth_pt.append(pt_true)

    if len(res_p) < min_tracks:
        return None

    res_p = np.asarray(res_p)
    res_pt = np.asarray(res_pt)
    truth_p = np.asarray(truth_p)
    truth_pt = np.asarray(truth_pt)

    return {
        "p_mean": float(np.mean(truth_p)),
        "pt_mean": float(np.mean(truth_pt)),
        "sigma_p_over_p": float(np.std(res_p)),
        "sigma_pt_over_pt": float(np.std(res_pt)),
        "n_tracks": float(len(res_p)),
    }


def plot_dataset(
    datasets: Dict[str, List[Dict[str, float]]],
    x_key: str,
    y_key: str,
    xlabel: str,
    ylabel: str,
    title: str,
    outpath: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.2))

    for label, points in datasets.items():
        if not points:
            continue
        xs = [p[x_key] for p in points]
        ys = [p[y_key] for p in points]
        ax.plot(xs, ys, marker="o", lw=2.0, label=label)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot momentum and pT resolution from fit_*.root files")
    parser.add_argument(
        "--base-dir",
        default="/ceph/omunkombwe",
        help="Base directory containing fit_*.root files",
    )
    parser.add_argument(
        "--track-collection",
        default="Fitted_tracks_pion",
        help="Track collection name (default: Fitted_tracks_pion)",
    )
    parser.add_argument(
        "--bfield",
        type=float,
        default=2.0,
        help="Magnetic field Bz in Tesla (default: 2.0)",
    )
    parser.add_argument(
        "--atip-location",
        type=int,
        default=1,
        help="TrackState.location code for AtIP (default: 1)",
    )
    parser.add_argument(
        "--pdg-abs",
        type=int,
        default=211,
        help="Absolute PDG id to select truth particle (default: 211 for pion)",
    )
    parser.add_argument(
        "--min-tracks",
        type=int,
        default=50,
        help="Minimum tracks required to keep a momentum point",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=None,
        help="Limit the number of events per file (for quick tests)",
    )
    parser.add_argument(
        "--output-dir",
        default="./momentum_resolution_plots1",
        help="Output directory for plots",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    datasets: Dict[str, List[Dict[str, float]]] = {}

    files = sorted(glob.glob(os.path.join(args.base_dir, "fit_*.root")))
    if not files:
        print(f"No fit_*.root files found in {args.base_dir}")
    else:
        grouped: Dict[str, List[Dict[str, float]]] = {}
        for fname in files:
            nominal = parse_nominal_p(fname)
            theta = parse_theta_deg(fname)
            label = f"{theta} deg" if theta is not None else "unknown theta"
            result = compute_resolutions(
                fname,
                args.track_collection,
                args.bfield,
                args.atip_location,
                args.pdg_abs,
                args.min_tracks,
                args.max_events,
            )
            if result is None:
                print(f"Skipping {os.path.basename(fname)} (not enough tracks)")
                continue
            result["p_nominal"] = nominal if nominal is not None else result["p_mean"]
            grouped.setdefault(label, []).append(result)
            print(
                f"{os.path.basename(fname)}: p~{result['p_mean']:.3f} GeV, "
                f"sigma_p/p={result['sigma_p_over_p']:.4e}, n={int(result['n_tracks'])}"
            )

        for label, points in grouped.items():
            if not points:
                continue
            points.sort(key=lambda p: p["p_mean"])
            datasets[label] = points

    if not datasets:
        print("No datasets built; nothing to plot")
        return

    plot_dataset(
        datasets,
        x_key="p_mean",
        y_key="sigma_p_over_p",
        xlabel="Momentum [GeV]",
        ylabel=r"$\sigma_p / p$",
        title="Momentum Resolution",
        outpath=os.path.join(args.output_dir, "momentum_resolution.png"),
    )

    plot_dataset(
        datasets,
        x_key="pt_mean",
        y_key="sigma_pt_over_pt",
        xlabel="Transverse momentum [GeV]",
        ylabel=r"$\sigma_{p_T} / p_T$",
        title="Transverse Momentum Resolution",
        outpath=os.path.join(args.output_dir, "pt_resolution.png"),
    )

    print(f"Saved plots to {args.output_dir}")


if __name__ == "__main__":
    main()
