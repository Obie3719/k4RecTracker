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

DEFAULT_CROSS_SECTION_PB = 2011.9
DEFAULT_LUMINOSITY_FB = 150.0
DEFAULT_PT_MAX = 60.0
DEFAULT_PT_BINS = 60


def compute_event_weight(cross_section_pb, luminosity_fb, n_events):
    lumi_pb = luminosity_fb * 1e3
    return (cross_section_pb * lumi_pb) / float(n_events)


def parse_args():
    parser = argparse.ArgumentParser(description="Weighted reco muon pT distribution.")
    parser.add_argument("-i", "--input", default=common.INPUT_FILE, help="Input PODIO root file.")
    parser.add_argument(
        "--track-collection",
        default=common.TRACK_COLLECTION,
        help="Reco track collection to use.",
    )
    parser.add_argument("--bfield", type=float, default=common.B_FIELD_TESLA, help="B field [T].")
    parser.add_argument(
        "--pt-min", type=float, default=common.PT_MIN, help="Reco pT cut [GeV]."
    )
    parser.add_argument(
        "--eta-max", type=float, default=common.ETA_MAX, help="Reco |eta| cut."
    )
    parser.add_argument(
        "--cross-section-pb",
        type=float,
        default=DEFAULT_CROSS_SECTION_PB,
        help="Cross section in pb (for event weights).",
    )
    parser.add_argument(
        "--luminosity-fb",
        type=float,
        default=DEFAULT_LUMINOSITY_FB,
        help="Luminosity in fb^-1 (for event weights).",
    )
    parser.add_argument(
        "--pt-max", type=float, default=DEFAULT_PT_MAX, help="Max pT for histogram [GeV]."
    )
    parser.add_argument(
        "--pt-bins", type=int, default=DEFAULT_PT_BINS, help="Number of histogram bins."
    )
    parser.add_argument("--max-events", type=int, default=None, help="Process at most N events.")
    parser.add_argument(
        "-o", "--output", default="reco_pt_weighted.png", help="Output plot filename."
    )
    parser.add_argument(
        "--output-dir", default="validation_plots", help="Directory for the output plot."
    )
    return parser.parse_args()


def resolve_output_path(output_dir, output):
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, output)


def main():
    args = parse_args()
    out_path = resolve_output_path(args.output_dir, args.output)

    n_events = common.count_events(args.input)
    w = compute_event_weight(args.cross_section_pb, args.luminosity_fb, n_events)
    print(f"Input: {args.input}")
    print(f"Events in file: {n_events}")
    print(f"Per-event weight: {w:.6g}")

    reader = get_reader(args.input)
    events = reader.get("events")

    pts = []
    weights = []
    for iev, event in common.iter_events(events, args.max_events):
        tracks = event.get(args.track_collection)
        for track in tracks:
            if track.trackStates_size() == 0:
                continue
            state = track.getTrackStates(0)
            reco = common.trackstate_to_vec(state, args.bfield)
            pt = reco["pt"]
            if pt < args.pt_min or abs(reco["eta"]) > args.eta_max:
                continue
            pts.append(pt)
            weights.append(w)

    if not pts:
        print("No reco muons found for pT distribution.")
        return

    bins = np.linspace(0.0, args.pt_max, args.pt_bins + 1)
    plt.figure(figsize=(7, 5))
    plt.hist(
        pts,
        bins=bins,
        weights=weights,
        histtype="stepfilled",
        alpha=0.75,
        color="C0",
        edgecolor="black",
        label="reco muons",
    )
    plt.xlabel(r"$p_{T}^{reco}$ [GeV]")
    plt.ylabel("Events (weighted)")
    plt.title("Reco muon pT distribution (weighted)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
