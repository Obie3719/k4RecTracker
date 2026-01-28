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

PT_BINS_EFF = np.linspace(0.0, 60.0, 25)


def parse_args():
    parser = argparse.ArgumentParser(description="Muon reconstruction efficiency vs pT.")
    parser.add_argument("-i", "--input", default=common.INPUT_FILE, help="Input PODIO root file.")
    parser.add_argument(
        "-o", "--output", default="efficiency_vs_pt.png", help="Output plot filename."
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
    parser.add_argument("--pt-min", type=float, default=common.PT_MIN, help="Truth pT cut [GeV].")
    parser.add_argument(
        "--eta-max", type=float, default=common.ETA_MAX, help="Truth |eta| cut."
    )
    return parser.parse_args()


def resolve_output_path(output_dir, output):
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, output)


def wald_interval(selected, totals, z=1.0):
    eff = np.zeros_like(selected, dtype=float)
    err = np.zeros_like(selected, dtype=float)
    for i, (k, n) in enumerate(zip(selected, totals)):
        if n == 0:
            eff[i] = 0.0
            err[i] = 0.0
        else:
            p = k / n
            eff[i] = p
            err[i] = z * np.sqrt(p * (1.0 - p) / n)
    return eff, err


def main():
    args = parse_args()
    out_path = resolve_output_path(args.output_dir, args.output)

    n_events = common.count_events(args.input)
    print(f"Input: {args.input}")
    print(f"Events in file: {n_events}")

    reader = get_reader(args.input)
    events = reader.get("events")

    totals = np.zeros(len(PT_BINS_EFF) - 1)
    selected = np.zeros(len(PT_BINS_EFF) - 1)

    for _, event in common.iter_events(events, args.max_events):
        truth_muons = common.collect_truth_muons(event)
        for t in truth_muons:
            vec = t["vec"]
            if vec["pt"] < args.pt_min or abs(vec["eta"]) > args.eta_max:
                continue
            ib = np.digitize(vec["pt"], PT_BINS_EFF) - 1
            if 0 <= ib < len(totals):
                totals[ib] += 1

        _, matched = common.collect_matched_muons(
            event,
            args.track_collection,
            args.bfield,
            args.purity_min,
            args.pt_min,
            args.eta_max,
        )
        for mu in matched:
            pt = mu["truth"]["pt"]
            ib = np.digitize(pt, PT_BINS_EFF) - 1
            if 0 <= ib < len(selected):
                selected[ib] += 1

    eff, err = wald_interval(selected, totals, z=1.0)
    centers = 0.5 * (PT_BINS_EFF[:-1] + PT_BINS_EFF[1:])

    plt.figure(figsize=(7, 5))
    plt.errorbar(centers, eff, yerr=err, fmt="o", color="navy")
    plt.ylim(0.0, 1.05)
    plt.xlabel(r"$p_{T}^{\text{true}}$ [GeV]")
    plt.ylabel("Efficiency (reco / gen)")
    plt.title("Muon reconstruction efficiency vs pT")
    plt.grid(True, alpha=0.35)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
