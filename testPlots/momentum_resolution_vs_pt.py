#!/usr/bin/env python3
import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from podio.reading import get_reader

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
import muon_validation_common as common

PT_BINS_RES = np.linspace(0.0, 60.0, 28)


def parse_args():
    parser = argparse.ArgumentParser(description="Muon momentum resolution vs pT.")
    parser.add_argument("-i", "--input", default=common.INPUT_FILE, help="Input PODIO root file.")
    parser.add_argument(
        "-o",
        "--output",
        default="momentum_resolution_vs_pt.png",
        help="Output plot filename.",
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


def binned_stats(x, y, bins):
    centers = 0.5 * (bins[:-1] + bins[1:])
    mean = np.full_like(centers, np.nan, dtype=float)
    std = np.full_like(centers, np.nan, dtype=float)
    counts = np.zeros_like(centers, dtype=int)
    idx = np.digitize(x, bins) - 1
    for i in range(len(centers)):
        sel = y[idx == i]
        counts[i] = sel.size
        if sel.size:
            mean[i] = np.mean(sel)
            std[i] = np.std(sel)
    return centers, mean, std, counts


def main():
    args = parse_args()
    out_path = resolve_output_path(args.output_dir, args.output)

    n_events = common.count_events(args.input)
    print(f"Input: {args.input}")
    print(f"Events in file: {n_events}")

    reader = get_reader(args.input)
    events = reader.get("events")

    residuals = []
    for _, event in common.iter_events(events, args.max_events):
        _, matched = common.collect_matched_muons(
            event,
            args.track_collection,
            args.bfield,
            args.purity_min,
            args.pt_min,
            args.eta_max,
        )
        for mu in matched:
            truth = mu["truth"]
            reco = mu["reco"]
            if truth["p"] == 0:
                continue
            res = (reco["p"] - truth["p"]) / truth["p"]
            residuals.append((truth["pt"], res))

    if not residuals:
        print("No matched muons found for momentum resolution.")
        return

    pts = np.array([r[0] for r in residuals])
    res = np.array([r[1] for r in residuals])
    centers, mean, std, _ = binned_stats(pts, res, PT_BINS_RES)

    plt.figure(figsize=(8, 6))
    plt.scatter(pts, res, s=8, alpha=0.3, label="per-track")
    plt.errorbar(
        centers, mean, yerr=std, fmt="o", color="firebrick", lw=2, label="mean +/- sigma"
    )
    plt.xlabel(r"$p_{T}^{\text{true}}$ [GeV]")
    plt.ylabel(r"$(p_{\text{reco}}-p_{\text{true}})/p_{\text{true}}$")
    plt.title("Momentum resolution vs pT")
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
