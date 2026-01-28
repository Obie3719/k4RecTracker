#!/usr/bin/env python3
import argparse
import os
import sys
from itertools import combinations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from podio.reading import get_reader

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
import muon_validation_common as common

ETA_BINS_MASS = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.4])


def parse_args():
    parser = argparse.ArgumentParser(description="Z mass width vs |eta|.")
    parser.add_argument("-i", "--input", default=common.INPUT_FILE, help="Input PODIO root file.")
    parser.add_argument(
        "-o", "--output", default="mass_width_vs_eta.png", help="Output plot filename."
    )
    parser.add_argument(
        "--output-dir", default="validation_plots", help="Directory for the output plot."
    )
    parser.add_argument("--max-events", type=int, default=None, help="Process at most N events.")
    parser.add_argument(
        "--track-collection",
        default=common.TRACK_COLLECTION,
        help="Reco track collection to use.",
    )
    parser.add_argument("--bfield", type=float, default=common.B_FIELD_TESLA, help="B field [T].")
    parser.add_argument(
        "--purity-min",
        type=float,
        default=common.PURITY_MIN,
        help="Minimum shared-hit purity for a match.",
    )
    parser.add_argument("--pt-min", type=float, default=common.PT_MIN, help="Reco pT cut [GeV].")
    parser.add_argument(
        "--eta-max", type=float, default=common.ETA_MAX, help="Reco |eta| cut."
    )
    return parser.parse_args()


def resolve_output_path(output_dir, output):
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, output)


def voigt_profile(x, mu, sigma, gamma, norm):
    from scipy.special import wofz

    z = ((x - mu) + 1j * gamma) / (sigma * np.sqrt(2.0))
    return norm * np.real(wofz(z)) / (sigma * np.sqrt(2.0 * np.pi))


def fit_voigt_sigma(masses):
    try:
        from scipy.optimize import curve_fit
    except Exception:
        return np.std(masses), False

    if len(masses) < 50:
        return np.std(masses), False

    hist, edges = np.histogram(masses, bins=60, range=(60.0, 120.0))
    centers = 0.5 * (edges[:-1] + edges[1:])
    p0 = (91.2, 2.0, 2.5, float(np.max(hist)))
    try:
        popt, _ = curve_fit(voigt_profile, centers, hist, p0=p0, maxfev=20000)
        sigma = abs(popt[1])
        return sigma, True
    except Exception:
        return np.std(masses), False


def main():
    args = parse_args()
    out_path = resolve_output_path(args.output_dir, args.output)

    n_events = common.count_events(args.input)
    print(f"Input: {args.input}")
    print(f"Events in file: {n_events}")

    reader = get_reader(args.input)
    events = reader.get("events")

    masses_by_eta_bin = [[] for _ in range(len(ETA_BINS_MASS) - 1)]

    for _, event in common.iter_events(events, args.max_events):
        _, matched = common.collect_matched_muons(
            event,
            args.track_collection,
            args.bfield,
            args.purity_min,
            args.pt_min,
            args.eta_max,
        )
        if len(matched) < 2:
            continue
        for mu1, mu2 in combinations(matched, 2):
            mass = common.invariant_mass(mu1["reco"], mu2["reco"])
            eta_pair = max(abs(mu1["truth"]["eta"]), abs(mu2["truth"]["eta"]))
            ib = np.digitize(eta_pair, ETA_BINS_MASS) - 1
            if 0 <= ib < len(masses_by_eta_bin):
                masses_by_eta_bin[ib].append(mass)

    centers = 0.5 * (ETA_BINS_MASS[:-1] + ETA_BINS_MASS[1:])
    widths = []
    used_voigt = []
    for masses in masses_by_eta_bin:
        if not masses:
            widths.append(np.nan)
            used_voigt.append(False)
            continue
        sigma, ok = fit_voigt_sigma(np.array(masses))
        widths.append(sigma)
        used_voigt.append(ok)

    if any(used_voigt):
        print("Voigt fit used where possible; RMS used as fallback.")
    else:
        print("Voigt fit not available; RMS widths plotted.")

    plt.figure(figsize=(8, 5))
    plt.plot(centers, widths, "o-", color="darkred")
    plt.xlabel(r"$|\eta|$ bin center")
    plt.ylabel(r"$\sigma(m_{\mu\mu})$ [GeV]")
    plt.title("Z mass width vs |eta|")
    plt.grid(True, alpha=0.35)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
