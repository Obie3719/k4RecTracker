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


def parse_args():
    parser = argparse.ArgumentParser(description="Muon eta and phi distributions.")
    parser.add_argument("-i", "--input", default=common.INPUT_FILE, help="Input PODIO root file.")
    parser.add_argument(
        "-o",
        "--output",
        default="eta_phi_distributions.png",
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


def main():
    args = parse_args()
    out_path = resolve_output_path(args.output_dir, args.output)

    n_events = common.count_events(args.input)
    print(f"Input: {args.input}")
    print(f"Events in file: {n_events}")

    reader = get_reader(args.input)
    events = reader.get("events")

    eta_truth = []
    eta_reco = []
    phi_truth = []
    phi_reco = []

    for _, event in common.iter_events(events, args.max_events):
        truth_muons = common.collect_truth_muons(event)
        for t in truth_muons:
            vec = t["vec"]
            if vec["pt"] < args.pt_min or abs(vec["eta"]) > args.eta_max:
                continue
            eta_truth.append(vec["eta"])
            phi_truth.append(vec["phi"])

        _, matched = common.collect_matched_muons(
            event,
            args.track_collection,
            args.bfield,
            args.purity_min,
            args.pt_min,
            args.eta_max,
        )
        for mu in matched:
            reco = mu["reco"]
            eta_reco.append(reco["eta"])
            phi_reco.append(reco["phi"])

    if not eta_truth:
        print("No muons found for eta/phi plot.")
        return

    plt.figure(figsize=(10, 8))
    ax1 = plt.subplot(2, 1, 1)
    bins_eta = np.linspace(-2.5, 2.5, 60)
    ax1.hist(eta_truth, bins=bins_eta, histtype="step", lw=2, label="truth")
    ax1.hist(eta_reco, bins=bins_eta, histtype="stepfilled", alpha=0.35, label="reco")
    ax1.set_xlabel(r"$\eta$")
    ax1.set_ylabel("Muons")
    ax1.set_title("Pseudorapidity")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2 = plt.subplot(2, 1, 2)
    bins_phi = np.linspace(-np.pi, np.pi, 60)
    ax2.hist(phi_truth, bins=bins_phi, histtype="step", lw=2, label="truth")
    ax2.hist(phi_reco, bins=bins_phi, histtype="stepfilled", alpha=0.35, label="reco")
    ax2.set_xlabel(r"$\phi$ [rad]")
    ax2.set_ylabel("Muons")
    ax2.set_title("Azimuthal angle")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
