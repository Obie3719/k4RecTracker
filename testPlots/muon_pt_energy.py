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
    parser = argparse.ArgumentParser(description="Muon pT and energy spectra.")
    parser.add_argument("-i", "--input", default=common.INPUT_FILE, help="Input PODIO root file.")
    parser.add_argument(
        "-o", "--output", default="muon_pt_energy.png", help="Output plot filename."
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

    pt_truth = []
    pt_reco = []
    energy_truth = []
    energy_reco = []

    for _, event in common.iter_events(events, args.max_events):
        truth_muons = common.collect_truth_muons(event)
        for t in truth_muons:
            vec = t["vec"]
            if vec["pt"] < args.pt_min or abs(vec["eta"]) > args.eta_max:
                continue
            pt_truth.append(vec["pt"])
            energy_truth.append(vec["energy"])

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
            pt_reco.append(reco["pt"])
            energy_reco.append(reco["energy"])

    if not pt_truth:
        print("No muons found for pT/energy plot.")
        return

    plt.figure(figsize=(10, 8))
    ax1 = plt.subplot(2, 1, 1)
    bins_pt = np.linspace(0.0, 60.0, 60)
    ax1.hist(pt_truth, bins=bins_pt, histtype="step", lw=2, label="truth")
    ax1.hist(pt_reco, bins=bins_pt, histtype="stepfilled", alpha=0.35, label="reco (matched)")
    ax1.set_xlabel(r"$p_{T}$ [GeV]")
    ax1.set_ylabel("Muons")
    ax1.set_title("Muon pT spectrum")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2 = plt.subplot(2, 1, 2)
    bins_e = np.linspace(30.0, 55.0, 60)
    ax2.hist(energy_truth, bins=bins_e, histtype="step", lw=2, label="truth")
    ax2.hist(
        energy_reco, bins=bins_e, histtype="stepfilled", alpha=0.35, label="reco (matched)"
    )
    ax2.axvline(45.59, color="firebrick", ls="--", label="45.59 GeV")
    ax2.set_xlabel("Energy [GeV]")
    ax2.set_ylabel("Muons")
    ax2.set_title("Muon energy spectrum")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
