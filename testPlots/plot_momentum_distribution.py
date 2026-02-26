import argparse
import math
import os
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from podio.reading import get_reader
import edm4hep  # noqa: F401  (required for podio edm4hep types)


def get_collection(event, name):
    try:
        return event.get(name)
    except Exception as exc:
        raise KeyError(f"Collection '{name}' not found in input file") from exc


def momentum_values(events, collection, max_events, gen_status, charged_only, pdg_abs):
    values = []
    for iev, event in enumerate(events):
        if max_events is not None and iev >= max_events:
            break
        particles = get_collection(event, collection)
        for mc in particles:
            if gen_status is not None and mc.getGeneratorStatus() != gen_status:
                continue
            if charged_only and abs(mc.getCharge()) <= 0:
                continue
            if pdg_abs is not None and abs(mc.getPDG()) != pdg_abs:
                continue
            mom = mc.getMomentum()
            p = math.sqrt(mom.x * mom.x + mom.y * mom.y + mom.z * mom.z)
            values.append(p)
    return np.asarray(values, dtype=float)

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


def select_truth_momentum(mc_particles, pdg_abs: Optional[int], gen_status: Optional[int]) -> Optional[Tuple[float, float, float]]:
    best = None
    best_p2 = -1.0

    for particle in mc_particles:
        gen = get_value(particle, "generatorStatus", "getGeneratorStatus")
        if gen_status is not None and gen is not None and gen != gen_status:
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


def momentum_resolution_vs_p(
    events,
    track_collection: str,
    mc_collection: str,
    gen_status: Optional[int],
    pdg_abs: Optional[int],
    bfield: float,
    atip_location: int,
    max_events: Optional[int],
):
    c_light = 2.99792458e8
    a_const = c_light * 1e3 * 1e-15  # 2.99792458e-4

    p_true_list = []
    res_list = []

    for iev, event in enumerate(events):
        if max_events is not None and iev >= max_events:
            break

        mc_particles = get_collection(event, mc_collection)
        truth_mom = select_truth_momentum(mc_particles, pdg_abs, gen_status)
        if truth_mom is None:
            continue

        px, py, pz = truth_mom
        p_true = math.sqrt(px * px + py * py + pz * pz)
        if p_true <= 0:
            continue

        try:
            tracks = event.get(track_collection)
        except KeyError as exc:
            raise RuntimeError(f"Collection '{track_collection}' not found in input file") from exc

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

        pt_reco = a_const * bfield / abs(omega_val)
        p_reco = pt_reco * math.sqrt(1.0 + tan_val * tan_val)
        res = (p_reco - p_true) / p_true

        p_true_list.append(p_true)
        res_list.append(res)

    return np.asarray(p_true_list, dtype=float), np.asarray(res_list, dtype=float)


def main():
    parser = argparse.ArgumentParser(description="Plot momentum distribution from MCParticles")
    parser.add_argument(
        "-i",
        "--input",
        default="/ceph/omunkombwe/files/deg60/fit_pi_P0.2-10.0GeV_theta60deg_nev1000_merged.root",
        help="Input EDM4hep ROOT file",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="momentum_distribution.png",
        help="Output PNG file",
    )
    parser.add_argument(
        "--collection",
        default="MCParticles",
        help="MCParticles collection name",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=None,
        help="Limit number of events (default: all)",
    )
    parser.add_argument(
        "--gen-status",
        type=int,
        default=1,
        help="Only include particles with this generator status",
    )
    parser.add_argument(
        "--charged-only",
        action="store_true",
        help="Only include charged particles",
    )
    parser.add_argument(
        "--pdg",
        type=int,
        default=211,
        help="Only include particles with this absolute PDG code",
    )
    parser.add_argument(
        "--track-collection",
        default="Fitted_tracks_pion",
        help="Track collection name for resolution (default: Fitted_tracks_pion)",
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
        "--min-tracks",
        type=int,
        default=50,
        help="Minimum entries per momentum bin for resolution plot",
    )
    parser.add_argument(
        "--output-resolution",
        default="momentum_resolution_vs_p.png",
        help="Output PNG for momentum resolution vs p",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=60,
        help="Number of histogram bins",
    )
    parser.add_argument(
        "--pmin",
        type=float,
        default=None,
        help="Min momentum for histogram range",
    )
    parser.add_argument(
        "--pmax",
        type=float,
        default=None,
        help="Max momentum for histogram range",
    )

    args = parser.parse_args()

    reader = get_reader(args.input)
    events = reader.get("events")

    pvals = momentum_values(
        events,
        args.collection,
        args.max_events,
        args.gen_status,
        args.charged_only,
        args.pdg,
    )

    if pvals.size == 0:
        raise RuntimeError("No particles selected; check collection name and filters")

    if args.pmin is None:
        pmin = float(np.min(pvals))
    else:
        pmin = args.pmin

    if args.pmax is None:
        pmax = float(np.max(pvals))
    else:
        pmax = args.pmax

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(pvals, bins=args.bins, range=(pmin, pmax), histtype="step", linewidth=2)
    ax.set_xlabel("p [GeV]")
    ax.set_ylabel("Entries")
    ax.set_title("Momentum distribution")
    ax.grid(True, alpha=0.3)

    outdir = os.path.dirname(args.output)
    if outdir:
        os.makedirs(outdir, exist_ok=True)

    fig.tight_layout()
    fig.savefig(args.output, dpi=200)
    plt.close(fig)

    p_true, res = momentum_resolution_vs_p(
        events,
        args.track_collection,
        args.collection,
        args.gen_status,
        args.pdg,
        args.bfield,
        args.atip_location,
        args.max_events,
    )

    fig_res = plt.figure(figsize=(8, 6))
    axr = fig_res.add_subplot(1, 1, 1)
    axr.set_xlabel("p [GeV]")
    axr.set_ylabel("sigma(p)/p")
    axr.set_title("Momentum resolution vs p")
    axr.grid(True, alpha=0.3)

    if p_true.size > 0:
        if args.pmin is None:
            pmin_res = float(np.min(p_true))
        else:
            pmin_res = args.pmin

        if args.pmax is None:
            pmax_res = float(np.max(p_true))
        else:
            pmax_res = args.pmax

        bins = np.linspace(pmin_res, pmax_res, args.bins + 1)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        sigma = []
        centers = []

        for i in range(len(bins) - 1):
            mask = (p_true >= bins[i]) & (p_true < bins[i + 1])
            if np.count_nonzero(mask) < args.min_tracks:
                continue
            sigma.append(float(np.std(res[mask])))
            centers.append(float(bin_centers[i]))

        if centers:
            axr.plot(centers, sigma, marker="o", linestyle="-")
        else:
            axr.text(0.5, 0.5, "No bins pass min-tracks", ha="center", va="center", transform=axr.transAxes)
    else:
        axr.text(0.5, 0.5, "No tracks / truth matched", ha="center", va="center", transform=axr.transAxes)

    outdir = os.path.dirname(args.output_resolution)
    if outdir:
        os.makedirs(outdir, exist_ok=True)

    fig_res.tight_layout()
    fig_res.savefig(args.output_resolution, dpi=200)
    plt.close(fig_res)


if __name__ == "__main__":
    main()
