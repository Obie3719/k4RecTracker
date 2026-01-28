#!/usr/bin/env python3
import argparse
import math
import os
from collections import Counter, defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from podio.reading import get_reader

DEFAULT_INPUT = "/ceph/omunkombwe/fitter_idea_mumu50k.root"
DEFAULT_TRACK_COLLECTION = "Fitted_tracks_muon"
DEFAULT_MC_COLLECTION = "MCParticles"

SIM_HITS_NAMES = [
    "DCHCollection",
    "VertexBarrelCollection",
    "VertexEndcapCollection",
    "SiWrBCollection",
    "SiWrDCollection",
]

ASSOC_NAMES = [
    "DCH_DigiSimAssociationCollection",
    "VTXBSimDigiLinks",
    "VTXDSimDigiLinks",
    "SiWrBSimDigiLinks",
    "SiWrDSimDigiLinks",
]

OMEGA_MODE = "curvature_mm"
B_FIELD_TESLA = 2.0

DEFAULT_PT_MIN = 5.0
DEFAULT_PT_MAX = 60.0
DEFAULT_PT_BINS = 30
DEFAULT_ETA_MAX = 2.4
DEFAULT_ETA_BINS = 24
DEFAULT_PURITY_MIN = 0.75


def parse_pdg_arg(value):
    text = value.strip().lower()
    if text in ("all", "any", "charged"):
        return None
    pdgs = set()
    for part in value.replace(" ", ",").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            pdgs.add(int(part))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid PDG code: {part}") from exc
    return pdgs or None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Track pT residuals vs true pT and eta (truth-matched)."
    )
    parser.add_argument("-i", "--input", default=DEFAULT_INPUT, help="Input PODIO root file.")
    parser.add_argument(
        "--tracks", default=DEFAULT_TRACK_COLLECTION, help="Track collection name."
    )
    parser.add_argument("--mc", default=DEFAULT_MC_COLLECTION, help="MCParticles collection.")
    parser.add_argument("--bfield", type=float, default=B_FIELD_TESLA, help="B field [T].")
    parser.add_argument(
        "--purity-min",
        type=float,
        default=DEFAULT_PURITY_MIN,
        help="Minimum shared-hit purity for a match.",
    )
    parser.add_argument("--pt-min", type=float, default=DEFAULT_PT_MIN, help="Truth pT cut [GeV].")
    parser.add_argument(
        "--pt-max", type=float, default=DEFAULT_PT_MAX, help="Max pT for bins [GeV]."
    )
    parser.add_argument(
        "--pt-bins", type=int, default=DEFAULT_PT_BINS, help="Number of pT bins."
    )
    parser.add_argument(
        "--eta-max",
        type=float,
        default=DEFAULT_ETA_MAX,
        help="Truth |eta| cut and eta range.",
    )
    parser.add_argument(
        "--eta-bins", type=int, default=DEFAULT_ETA_BINS, help="Number of eta bins."
    )
    parser.add_argument(
        "--pdg",
        type=parse_pdg_arg,
        default=parse_pdg_arg("13"),
        help="comma-separated PDG codes, or 'all' for any charged particle",
    )
    parser.add_argument("--gen-status", type=int, default=1, help="Generator status (use -1 to disable).")
    parser.add_argument("--max-events", type=int, default=None, help="Process at most N events.")
    parser.add_argument(
        "-o",
        "--output",
        default="pt_residuals_vs_pt_eta.png",
        help="Output plot filename.",
    )
    parser.add_argument(
        "--output-dir", default="validation_plots", help="Directory for the output plot."
    )
    return parser.parse_args()


def resolve_output_path(output_dir, output):
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, output)


def oid(obj):
    obj_id = obj.getObjectID()
    return (obj_id.collectionID, obj_id.index)


def build_sim_to_truth(sim_hits_list):
    sim_to_truth = {}
    for sim_hits in sim_hits_list:
        for sim in sim_hits:
            mc_part = sim.getParticle()
            sim_to_truth[oid(sim)] = mc_part.getObjectID().index
    return sim_to_truth


def build_digi_truth_map(associations_list, sim_to_truth):
    digi_truth = defaultdict(set)
    for associations in associations_list:
        for assoc in associations:
            digi_oid = oid(assoc.getFrom())
            sim_oid = oid(assoc.getTo())
            truth_idx = sim_to_truth.get(sim_oid)
            if truth_idx is not None:
                digi_truth[digi_oid].add(truth_idx)
    return digi_truth


def match_tracks(tracks, digi_truth):
    matches = []
    for track in tracks:
        counts = Counter()
        for ihit in range(track.trackerHits_size()):
            digi = track.getTrackerHits(ihit)
            for truth_idx in digi_truth.get(oid(digi), ()):
                counts[truth_idx] += 1
        if counts:
            best_idx, shared_hits = counts.most_common(1)[0]
            purity = shared_hits / track.trackerHits_size() if track.trackerHits_size() else 0.0
        else:
            best_idx = None
            purity = 0.0
        matches.append({"track": track, "truth_index": best_idx, "purity": purity})
    return matches


def pseudorapidity(px, py, pz):
    p = math.sqrt(px * px + py * py + pz * pz)
    if p == abs(pz):
        return float("inf") * (1 if pz >= 0 else -1)
    return 0.5 * math.log((p + pz) / (p - pz))


def pt_from_omega(omega, bfield):
    if omega == 0:
        return 0.0
    if OMEGA_MODE == "q_over_pt":
        return abs(1.0 / omega)
    if OMEGA_MODE == "curvature_m":
        return abs(0.299792458 * bfield / omega)
    return abs(0.299792458 * bfield / (omega * 1000.0))


def trackstate_to_vec(state, bfield):
    omega = state.omega
    pt = pt_from_omega(omega, bfield)
    phi = state.phi
    tan_lambda = state.tanLambda
    px = pt * math.cos(phi)
    py = pt * math.sin(phi)
    pz = pt * tan_lambda
    eta = pseudorapidity(px, py, pz)
    return {"pt": pt, "eta": eta, "phi": phi, "px": px, "py": py, "pz": pz}


def mc_particle_to_vec(mc):
    mom = mc.getMomentum()
    px, py, pz = mom.x, mom.y, mom.z
    pt = math.hypot(px, py)
    eta = pseudorapidity(px, py, pz)
    return {"pt": pt, "eta": eta, "px": px, "py": py, "pz": pz}


def is_truth_selected(mc, pdg_allow, gen_status):
    try:
        if abs(mc.getCharge()) <= 0:
            return False
        if gen_status is not None and mc.getGeneratorStatus() != gen_status:
            return False
        if pdg_allow is None:
            return True
        return abs(mc.getPDG()) in pdg_allow
    except Exception:
        return False


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

    gen_status = None if args.gen_status == -1 else args.gen_status
    pt_bins = np.linspace(args.pt_min, args.pt_max, args.pt_bins + 1)
    eta_bins = np.linspace(-args.eta_max, args.eta_max, args.eta_bins + 1)

    reader = get_reader(args.input)
    events = reader.get("events")

    pts = []
    etas = []
    residuals = []

    for iev, event in enumerate(events):
        if args.max_events is not None and iev >= args.max_events:
            break
        if iev and iev % 500 == 0:
            print(f"Processed {iev} events...")

        sim_hits_list = [event.get(name) for name in SIM_HITS_NAMES]
        associations_list = [event.get(name) for name in ASSOC_NAMES]
        sim_to_truth = build_sim_to_truth(sim_hits_list)
        digi_truth = build_digi_truth_map(associations_list, sim_to_truth)

        tracks = event.get(args.tracks)
        matches = match_tracks(tracks, digi_truth)
        mc_particles = event.get(args.mc)

        best_by_truth = {}
        for match in matches:
            tidx = match["truth_index"]
            if tidx is None or tidx >= len(mc_particles):
                continue
            mc = mc_particles[tidx]
            if not is_truth_selected(mc, args.pdg, gen_status):
                continue
            if match["purity"] < args.purity_min:
                continue
            prev = best_by_truth.get(tidx)
            if prev is None or match["purity"] > prev["purity"]:
                best_by_truth[tidx] = match

        for tidx, match in best_by_truth.items():
            track = match["track"]
            if track.trackStates_size() == 0:
                continue
            state = track.getTrackStates(0)
            reco = trackstate_to_vec(state, args.bfield)

            mc = mc_particles[tidx]
            truth = mc_particle_to_vec(mc)
            if truth["pt"] < args.pt_min or abs(truth["eta"]) > args.eta_max:
                continue
            if reco["pt"] <= 0:
                continue

            residual = (1.0 / reco["pt"]) - (1.0 / truth["pt"])
            pts.append(truth["pt"])
            etas.append(truth["eta"])
            residuals.append(residual)

    if not residuals:
        print("No matched tracks found for residuals.")
        return

    pts = np.array(pts)
    etas = np.array(etas)
    residuals = np.array(residuals)

    pt_centers, pt_mean, pt_std, _ = binned_stats(pts, residuals, pt_bins)
    eta_centers, eta_mean, eta_std, _ = binned_stats(etas, residuals, eta_bins)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(pts, residuals, s=8, alpha=0.25, label="per-track")
    axes[0].errorbar(
        pt_centers, pt_mean, yerr=pt_std, fmt="o", color="firebrick", label="mean +/- sigma"
    )
    axes[0].set_xlabel(r"$p_{T}^{true}$ [GeV]")
    axes[0].set_ylabel(r"$1/p_{T}^{reco}-1/p_{T}^{true}$ [1/GeV]")
    axes[0].set_title("1/pT residual vs pT")
    axes[0].grid(True, alpha=0.35)
    axes[0].legend()

    axes[1].scatter(etas, residuals, s=8, alpha=0.25, label="per-track")
    axes[1].errorbar(
        eta_centers, eta_mean, yerr=eta_std, fmt="o", color="firebrick", label="mean +/- sigma"
    )
    axes[1].set_xlabel(r"$\eta^{true}$")
    axes[1].set_ylabel(r"$1/p_{T}^{reco}-1/p_{T}^{true}$ [1/GeV]")
    axes[1].set_title("1/pT residual vs eta")
    axes[1].grid(True, alpha=0.35)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
