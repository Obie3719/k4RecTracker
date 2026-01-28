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
    parser = argparse.ArgumentParser(description="Muon impact parameter resolution (d0, z0).")
    parser.add_argument("-i", "--input", default=common.INPUT_FILE, help="Input PODIO root file.")
    parser.add_argument(
        "-o",
        "--output",
        default="impact_parameter_resolution.png",
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

    d0_res = []
    z0_res = []
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
            state = mu["state"]
            true_d0 = common.compute_true_d0(truth["phi"], truth["vx"], truth["vy"])
            d0_res.append(state.D0 - true_d0)
            z0_res.append(state.Z0 - truth["vz"])

    if not d0_res:
        print("No matched muons found for impact parameter resolution.")
        return

    plt.figure(figsize=(10, 4))
    ax1 = plt.subplot(1, 2, 1)
    ax1.hist(d0_res, bins=np.linspace(-1.0, 1.0, 80), histtype="step", lw=2)
    ax1.set_xlabel(r"$d_{0}^{\text{reco}}-d_{0}^{\text{true}}$ [mm]")
    ax1.set_ylabel("Tracks")
    ax1.set_title("Transverse impact parameter resolution")
    ax1.grid(True, alpha=0.3)
    ax1.text(
        0.98,
        0.85,
        fr"$\sigma$={np.std(d0_res):.3e}",
        transform=ax1.transAxes,
        ha="right",
        fontsize=10,
    )

    ax2 = plt.subplot(1, 2, 2)
    ax2.hist(z0_res, bins=np.linspace(-1.5, 1.5, 80), histtype="step", lw=2, color="maroon")
    ax2.set_xlabel(r"$z_{0}^{\text{reco}}-z_{0}^{\text{true}}$ [mm]")
    ax2.set_ylabel("Tracks")
    ax2.set_title("Longitudinal impact parameter resolution")
    ax2.grid(True, alpha=0.3)
    ax2.text(
        0.98,
        0.85,
        fr"$\sigma$={np.std(z0_res):.3e}",
        transform=ax2.transAxes,
        ha="right",
        fontsize=10,
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
