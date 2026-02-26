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
import edm4hep  # noqa: F401  (required for podio edm4hep types)


FILENAME_RE = re.compile(
    r"fit_.*_P(?P<p_low>[\d.]+)-(?P<p_high>[\d.]+)GeV_theta(?P<theta>\d+)deg_.*\.root"
)

BARREL_DIGI_COLLECTIONS = [
    "DCH_DigiCollection",
    "VTXBDigis",
    "SiWrBDigis",
]
ENDCAP_DIGI_COLLECTIONS = [
    "VTXDDigis",
    "SiWrDDigis",
]


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


def oid(obj) -> Tuple[int, int]:
    obj_id = obj.getObjectID()
    return (obj_id.collectionID, obj_id.index)


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


def build_hit_oid_set(event, collection_names: List[str]) -> set:
    oids = set()
    for name in collection_names:
        try:
            coll = event.get(name)
        except Exception:
            continue
        for hit in coll:
            oids.add(oid(hit))
    return oids


def is_barrel_track(track, barrel_oids: set, endcap_oids: set) -> bool:
    hits = list(iter_tracker_hits(track))
    if not hits:
        return False

    barrel_count = 0
    endcap_count = 0
    for hit in hits:
        h_oid = oid(hit)
        if h_oid in barrel_oids:
            barrel_count += 1
        elif h_oid in endcap_oids:
            endcap_count += 1

    return endcap_count == 0 and barrel_count > 0


def is_endcap_track(track, barrel_oids: set, endcap_oids: set) -> bool:
    hits = list(iter_tracker_hits(track))
    if not hits:
        return False

    barrel_count = 0
    endcap_count = 0
    for hit in hits:
        h_oid = oid(hit)
        if h_oid in barrel_oids:
            barrel_count += 1
        elif h_oid in endcap_oids:
            endcap_count += 1

    return barrel_count == 0 and endcap_count > 0


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

    phi = get_value(chosen, "phi", "getPhi")
    tan_lambda = get_value(chosen, "tanLambda", "getTanLambda")
    if phi is None or tan_lambda is None:
        return None

    return float(phi), float(tan_lambda)


def wrap_phi(dphi: float) -> float:
    while dphi > math.pi:
        dphi -= 2.0 * math.pi
    while dphi < -math.pi:
        dphi += 2.0 * math.pi
    return dphi


def compute_theta_phi_resolution(
    filename: str,
    track_collection: str,
    atip_location: int,
    pdg_abs: Optional[int],
    min_tracks: int,
    barrel_only: bool,
    endcap_only: bool,
) -> Optional[Dict[str, float]]:
    reader = get_reader(filename)
    events = reader.get("events")

    residuals_theta: List[float] = []
    residuals_phi: List[float] = []

    for event in events:
        mc_particles = event.get("MCParticles")
        if len(mc_particles) == 0:
            continue

        truth_mom = select_truth_momentum(mc_particles, pdg_abs)
        if truth_mom is None:
            continue

        px, py, pz = truth_mom
        p_true = math.sqrt(px * px + py * py + pz * pz)
        if p_true <= 0:
            continue

        phi_true = math.atan2(py, px)
        theta_true = math.atan2(math.hypot(px, py), pz)

        try:
            tracks = event.get(track_collection)
        except Exception:
            continue

        if len(tracks) == 0:
            continue

        if barrel_only or endcap_only:
            barrel_oids = build_hit_oid_set(event, BARREL_DIGI_COLLECTIONS)
            endcap_oids = build_hit_oid_set(event, ENDCAP_DIGI_COLLECTIONS)
            if barrel_only and not endcap_only:
                tracks = [t for t in tracks if is_barrel_track(t, barrel_oids, endcap_oids)]
            elif endcap_only and not barrel_only:
                tracks = [t for t in tracks if is_endcap_track(t, barrel_oids, endcap_oids)]
            if len(tracks) == 0:
                continue

        best_track = choose_best_track(tracks)
        if best_track is None:
            continue

        state = pick_track_state(best_track, atip_location)
        if state is None:
            continue

        phi_rec, tan_lambda = state
        theta_rec = math.atan2(1.0, tan_lambda)

        residuals_phi.append(wrap_phi(phi_rec - phi_true))
        residuals_theta.append(theta_rec - theta_true)

    n_tracks = len(residuals_theta)
    if n_tracks < min_tracks:
        return None

    res_theta = np.asarray(residuals_theta, dtype=float)
    res_phi = np.asarray(residuals_phi, dtype=float)

    return {
        "sigma_theta": float(np.std(res_theta)),
        "sigma_phi": float(np.std(res_phi)),
        "n_tracks": float(n_tracks),
    }


def aggregate_sigma(values: List[Dict[str, float]], key: str) -> Tuple[float, float]:
    sigmas = np.asarray([v[key] for v in values], dtype=float)
    if sigmas.size == 0:
        return 0.0, 0.0
    mean = float(np.mean(sigmas))
    if sigmas.size > 1:
        err = float(np.std(sigmas, ddof=1) / math.sqrt(sigmas.size))
    else:
        n = int(values[0]["n_tracks"])
        err = float(mean / math.sqrt(2.0 * (n - 1))) if n > 1 else 0.0
    return mean, err


def plot_resolution(
    datasets: Dict[int, Dict[float, List[Dict[str, float]]]],
    sigma_key: str,
    ylabel: str,
    title: str,
    outpath: str,
    show_errors: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.2))

    for theta in sorted(datasets.keys()):
        p_map = datasets[theta]
        momenta = sorted(p_map.keys())
        ys = []
        yerr = []
        for p in momenta:
            mean, err = aggregate_sigma(p_map[p], sigma_key)
            ys.append(mean)
            yerr.append(err)
        if show_errors:
            ax.errorbar(
                momenta,
                ys,
                yerr=yerr,
                fmt="o-",
                lw=1.6,
                ms=4,
                capsize=2,
                label=f"{theta} deg",
            )
        else:
            ax.plot(momenta, ys, "o-", lw=1.6, ms=4, label=f"{theta} deg")

    ax.set_xlabel("Momentum, GeV/c")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot theta/phi resolution vs momentum from fit_*.root files")
    parser.add_argument("--base-dir", default="/ceph/omunkombwe/idea_o1_v03_gun", help="Base directory with deg* subdirs")
    parser.add_argument("--subdir-glob", default="deg*", help="Subdirectory glob pattern")
    parser.add_argument("--fit-glob", default="fit_*.root", help="Fit file glob within subdirectories")
    parser.add_argument("--track-collection", default="Fitted_tracks_pion", help="Track collection name")
    parser.add_argument("--atip-location", type=int, default=1, help="TrackState location (AtIP=1)")
    parser.add_argument("--pdg-abs", type=int, default=211, help="Absolute PDG id for truth particle")
    parser.add_argument("--min-tracks", type=int, default=2, help="Minimum tracks to accept a point")
    parser.add_argument("--barrel-only", action="store_true", help="Use only barrel tracks")
    parser.add_argument("--endcap-only", action="store_true", help="Use only endcap tracks")
    parser.add_argument("--theta-keep", default="45,60,75,90", help="Comma-separated theta values to keep (empty=all)")
    parser.add_argument("--output-dir", default=".", help="Output directory for plots")
    parser.add_argument("--no-errorbars", action="store_true", help="Disable error bars")

    args = parser.parse_args()

    theta_keep: Optional[set] = None
    if args.theta_keep.strip():
        theta_keep = {int(t.strip()) for t in args.theta_keep.split(",") if t.strip()}

    if args.barrel_only and args.endcap_only:
        print("Both --barrel-only and --endcap-only set; using all tracks.")
        args.barrel_only = False
        args.endcap_only = False

    os.makedirs(args.output_dir, exist_ok=True)

    results_by_theta: Dict[int, Dict[float, List[Dict[str, float]]]] = {}

    subdirs = sorted(glob.glob(os.path.join(args.base_dir, args.subdir_glob)))
    if not subdirs:
        raise FileNotFoundError(f"No subdirectories found under {args.base_dir} with pattern {args.subdir_glob}")

    for subdir in subdirs:
        fit_files = sorted(glob.glob(os.path.join(subdir, args.fit_glob)))
        if not fit_files:
            continue

        for file_path in fit_files:
            fname = os.path.basename(file_path)
            match = FILENAME_RE.match(fname)
            if not match:
                continue

            p_low = float(match.group("p_low"))
            p_high = float(match.group("p_high"))
            if abs(p_low - p_high) > 1e-6:
                continue

            theta = int(match.group("theta"))
            if theta_keep is not None and theta not in theta_keep:
                continue

            p_gen = p_low
            result = compute_theta_phi_resolution(
                file_path,
                args.track_collection,
                args.atip_location,
                args.pdg_abs,
                args.min_tracks,
                args.barrel_only,
                args.endcap_only,
            )
            if result is None:
                continue

            results_by_theta.setdefault(theta, {}).setdefault(p_gen, []).append(result)

    if not results_by_theta:
        print("No datasets built; nothing to plot.")
        return

    theta_out = os.path.join(args.output_dir, "Theta_Resolution.png")
    phi_out = os.path.join(args.output_dir, "Phi_Resolution.png")

    plot_resolution(
        results_by_theta,
        sigma_key="sigma_theta",
        ylabel=r"$\sigma_\theta$ [rad]",
        title="Theta resolution",
        outpath=theta_out,
        show_errors=not args.no_errorbars,
    )
    plot_resolution(
        results_by_theta,
        sigma_key="sigma_phi",
        ylabel=r"$\sigma_\phi$ [rad]",
        title="Phi Resolution",
        outpath=phi_out,
        show_errors=not args.no_errorbars,
    )

    print(f"Saved: {theta_out}")
    print(f"Saved: {phi_out}")


if __name__ == "__main__":
    main()
