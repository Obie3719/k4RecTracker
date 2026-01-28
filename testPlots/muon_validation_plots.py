#!/usr/bin/env python3
"""
Standalone muon performance validation.

It produces:
- Momentum resolution ( (p_reco - p_true)/p_true ) vs pT
- Angular resolution (Δθ, Δφ)
- Impact parameter resolution (d0, z0)
- Muon pT and energy spectra
- η and φ distributions for uniformity checks
- Reconstruction efficiency vs η and pT
- Z mass width vs |η| bin
"""

import argparse
import math
import os
from collections import Counter, defaultdict
from itertools import combinations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import uproot
from podio.reading import get_reader

# ------------------------------------------------------------
# Configuration defaults
# ------------------------------------------------------------
INPUT_FILE = "/ceph/omunkombwe/fitter_idea_mumu50k.root"
TRACK_COLLECTION = "Fitted_tracks_muon"
MUON_MASS_GEV = 0.105658
PURITY_MIN = 0.75  # min shared-hit purity to accept a match
ETA_MAX = 2.4
PT_MIN = 5.0
B_FIELD_TESLA = 2.0
OMEGA_MODE = "curvature_mm"  # {"curvature_mm", "curvature_m", "q_over_pt"}

PT_BINS_EFF = np.linspace(0.0, 60.0, 25)
ETA_BINS_EFF = np.linspace(-2.5, 2.5, 25)
PT_BINS_RES = np.linspace(0.0, 60.0, 28)
ETA_BINS_MASS = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.4])


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Muon validation plots.")
    parser.add_argument("-i", "--input", default=INPUT_FILE, help="Input PODIO root file.")
    parser.add_argument(
        "-o", "--output-dir", default="validation_plots", help="Directory for all output PNGs."
    )
    parser.add_argument(
        "--max-events", type=int, default=None, help="Process at most this many events."
    )
    parser.add_argument(
        "--track-collection",
        default=TRACK_COLLECTION,
        help="Reco track collection to use (default: Fitted_tracks_muon).",
    )
    parser.add_argument("--bfield", type=float, default=B_FIELD_TESLA, help="B field in Tesla.")
    parser.add_argument(
        "--purity-min",
        type=float,
        default=PURITY_MIN,
        help="Minimum shared-hit purity for a reco→truth match.",
    )
    return parser.parse_args()


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def count_events(path):
    with uproot.open(path) as f:
        event_keys = [k for k in f.keys() if k.startswith("events;")]
        events_key = max(event_keys, key=lambda name: int(name.split(";")[1]))
        return f[events_key].num_entries


def pseudorapidity(px, py, pz):
    p = math.sqrt(px * px + py * py + pz * pz)
    if p == abs(pz):
        return float("inf") * (1 if pz >= 0 else -1)
    return 0.5 * math.log((p + pz) / (p - pz))


def delta_phi(phi1, phi2):
    dphi = phi1 - phi2
    while dphi > math.pi:
        dphi -= 2 * math.pi
    while dphi <= -math.pi:
        dphi += 2 * math.pi
    return dphi


def pt_from_omega(omega, bfield):
    if omega == 0:
        return 0.0
    if OMEGA_MODE == "q_over_pt":
        return abs(1.0 / omega)
    if OMEGA_MODE == "curvature_m":
        return abs(0.299792458 * bfield / omega)
    return abs(0.299792458 * bfield / (omega * 1000.0))  # omega in 1/mm


def oid(obj):
    obj_id = obj.getObjectID()
    return (obj_id.collectionID, obj_id.index)


def build_sim_to_truth(sim_hits):
    sim_to_truth = {}
    for sim in sim_hits:
        mc_part = sim.getParticle()
        sim_to_truth[oid(sim)] = mc_part.getObjectID().index
    return sim_to_truth


def build_digi_truth_map(associations, sim_to_truth):
    digi_truth = defaultdict(set)
    for assoc in associations:
        digi_oid = oid(assoc.getFrom())
        sim_oid = oid(assoc.getTo())
        truth_idx = sim_to_truth.get(sim_oid)
        if truth_idx is not None:
            digi_truth[digi_oid].add(truth_idx)
    return digi_truth


def match_tracks(event, track_collection, digi_truth):
    matches = []
    tracks = event.get(track_collection)
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


def mc_particle_to_vec(mc):
    mom = mc.getMomentum()
    px, py, pz = mom.x, mom.y, mom.z
    p = math.sqrt(px * px + py * py + pz * pz)
    pt = math.hypot(px, py)
    eta = pseudorapidity(px, py, pz)
    phi = math.atan2(py, px)
    theta = math.acos(pz / p) if p > 0 else 0.0
    energy = math.sqrt(p * p + mc.getMass() * mc.getMass())
    vtx = mc.getVertex()
    return {
        "pt": pt,
        "eta": eta,
        "phi": phi,
        "theta": theta,
        "p": p,
        "energy": energy,
        "vx": vtx.x,
        "vy": vtx.y,
        "vz": vtx.z,
    }


def trackstate_to_vec(state, bfield):
    omega = state.omega
    pt = pt_from_omega(omega, bfield)
    phi = state.phi
    tan_lambda = state.tanLambda
    pz = pt * tan_lambda
    p = math.sqrt(pt * pt + pz * pz)
    eta = pseudorapidity(pt * math.cos(phi), pt * math.sin(phi), pz)
    theta = (math.pi / 2.0) - math.atan(tan_lambda)
    energy = math.sqrt(p * p + MUON_MASS_GEV * MUON_MASS_GEV)
    # flip sign (omega sign bug was observed in earlier studies)
    charge = -1 if omega > 0 else 1 if omega < 0 else 0
    return {
        "pt": pt,
        "eta": eta,
        "phi": phi,
        "theta": theta,
        "p": p,
        "pz": pz,
        "energy": energy,
        "charge": charge,
    }


def is_truth_muon(mc):
    try:
        return abs(mc.getPDG()) == 13 and mc.getGeneratorStatus() == 1
    except Exception:
        return False


def invariant_mass(v1, v2):
    e = v1["energy"] + v2["energy"]
    px = v1["pt"] * math.cos(v1["phi"]) + v2["pt"] * math.cos(v2["phi"])
    py = v1["pt"] * math.sin(v1["phi"]) + v2["pt"] * math.sin(v2["phi"])
    pz = v1["pz"] + v2["pz"]
    return math.sqrt(max(e * e - (px * px + py * py + pz * pz), 0.0))


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
            err[i] = z * math.sqrt(p * (1.0 - p) / n)
    return eff, err


def compute_true_d0(phi, vx, vy):
    return -vx * math.sin(phi) + vy * math.cos(phi)


# ------------------------------------------------------------
# Plotters
# ------------------------------------------------------------
def plot_momentum_resolution(residuals, pt_bins, out_path):
    if not residuals:
        print("No matched tracks for momentum resolution plot.")
        return
    pts = np.array([r[0] for r in residuals])
    res = np.array([r[1] for r in residuals])
    centers, mean, std, counts = binned_stats(pts, res, pt_bins)

    plt.figure(figsize=(8, 6))
    plt.scatter(res, pts, s=8, alpha=0.3, label="per-track")
    plt.errorbar(
        centers, mean, yerr=std, fmt="o", color="firebrick", lw=2, label="mean +/- sigma"
    )
    plt.xlabel(r"$(p_{\text{reco}}-p_{\text{true}})/p_{\text{true}}$")
    plt.ylabel(r"$p_{T}^{\text{true}}$ [GeV]")
    plt.title("Momentum resolution vs pT")
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_angle_resolution(dtheta, dphi, out_path):
    plt.figure(figsize=(10, 4))
    ax1 = plt.subplot(1, 2, 1)
    ax1.hist(dtheta, bins=np.linspace(-0.01, 0.01, 80), histtype="step", lw=2)
    ax1.set_xlabel(r"$\Delta\theta$ [rad]")
    ax1.set_ylabel("Tracks")
    ax1.set_title(r"Angle resolution $\Delta\theta$")
    ax1.grid(True, alpha=0.3)
    ax1.text(
        0.98,
        0.85,
        fr"$\sigma$={np.std(dtheta):.3e}",
        transform=ax1.transAxes,
        ha="right",
        fontsize=10,
    )

    ax2 = plt.subplot(1, 2, 2)
    ax2.hist(dphi, bins=np.linspace(-0.01, 0.01, 80), histtype="step", lw=2, color="darkgreen")
    ax2.set_xlabel(r"$\Delta\phi$ [rad]")
    ax2.set_ylabel("Tracks")
    ax2.set_title(r"Angle resolution $\Delta\phi$")
    ax2.grid(True, alpha=0.3)
    ax2.text(
        0.98,
        0.85,
        fr"$\sigma$={np.std(dphi):.3e}",
        transform=ax2.transAxes,
        ha="right",
        fontsize=10,
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_impact_parameters(d0_res, z0_res, out_path):
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


def plot_pt_energy(pt_truth, pt_reco, e_truth, e_reco, out_path):
    plt.figure(figsize=(10, 8))
    ax1 = plt.subplot(2, 1, 1)
    bins_pt = np.linspace(0.0, 47.0, 60)
    ax1.hist(pt_truth, bins=bins_pt, histtype="step", lw=2, label="truth")
    ax1.hist(pt_reco, bins=bins_pt, histtype="stepfilled", alpha=0.35, label="reco (matched)")
    ax1.set_xlabel(r"$p_{T}$ [GeV]")
    ax1.set_ylabel("Muons")
    ax1.set_title("Muon $p_{T}$ spectrum")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2 = plt.subplot(2, 1, 2)
    bins_e = np.linspace(30.0, 55.0, 60)
    ax2.hist(e_truth, bins=bins_e, histtype="step", lw=2, label="truth")
    ax2.hist(e_reco, bins=bins_e, histtype="stepfilled", alpha=0.35, label="reco (matched)")
    ax2.axvline(45.59, color="firebrick", ls="--", label="45.59 GeV")
    ax2.set_xlabel("Energy [GeV]")
    ax2.set_ylabel("Muons")
    ax2.set_title("Muon energy spectrum")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_eta_phi(truth_eta, reco_eta, truth_phi, reco_phi, out_path):
    plt.figure(figsize=(10, 8))
    ax1 = plt.subplot(2, 1, 1)
    bins_eta = np.linspace(-2, 2, 60)
    ax1.hist(truth_eta, bins=bins_eta, histtype="step", lw=2, label="truth")
    ax1.hist(reco_eta, bins=bins_eta, histtype="stepfilled", alpha=0.35, label="reco (matched)")
    ax1.set_xlabel(r"$\eta$")
    ax1.set_ylabel("Muons")
    ax1.set_title("Pseudorapidity")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2 = plt.subplot(2, 1, 2)
    bins_phi = np.linspace(-math.pi, math.pi, 60)
    ax2.hist(truth_phi, bins=bins_phi, histtype="step", lw=2, label="truth")
    ax2.hist(reco_phi, bins=bins_phi, histtype="stepfilled", alpha=0.35, label="reco (matched)")
    ax2.set_xlabel(r"$\phi$ [rad]")
    ax2.set_ylabel("Muons")
    ax2.set_title("Azimuthal angle")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_efficiency(bins, totals, selected, xlabel, title, out_path):
    eff, err = wald_interval(selected, totals, z=1.0)
    centers = 0.5 * (bins[:-1] + bins[1:])
    plt.figure(figsize=(7, 5))
    plt.errorbar(centers, eff, yerr=err, fmt="o", color="navy")
    plt.ylim(0.0, 1.05)
    plt.xlabel(xlabel)
    plt.ylabel("Efficiency (reco / gen)")
    plt.title(title)
    plt.grid(True, alpha=0.35)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_mass_width(eta_bins, masses_by_bin, out_path):
    centers = 0.5 * (eta_bins[:-1] + eta_bins[1:])
    widths = []
    counts = []
    means = []
    for arr in masses_by_bin:
        arr = np.array(arr)
        counts.append(arr.size)
        if arr.size:
            means.append(np.mean(arr))
            widths.append(np.std(arr))
        else:
            means.append(np.nan)
            widths.append(np.nan)

    plt.figure(figsize=(8, 5))
    plt.plot(centers, widths, "o-", color="darkred", label="RMS width")
    plt.xlabel(r"$|\eta|$ bin center")
    plt.ylabel(r"$\sigma(m_{\mu\mu})$ [GeV]")
    plt.title("Z mass width vs |η|")
    plt.grid(True, alpha=0.35)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.bar(centers, counts, width=np.diff(eta_bins), align="center", alpha=0.6, color="gray")
    plt.xlabel(r"$|\eta|$ bin center")
    plt.ylabel("Pairs in bin")
    plt.title("Statistics per |η| bin (mass plot)")
    plt.tight_layout()
    plt.savefig(out_path.replace(".png", "_counts.png"), dpi=200)
    plt.close()


# ------------------------------------------------------------
# Main loop
# ------------------------------------------------------------
def main():
    args = parse_args()
    ensure_dir(args.output_dir)

    n_events = count_events(args.input)
    print(f"Input: {args.input}")
    print(f"Events in file: {n_events}")

    reader = get_reader(args.input)
    events = reader.get("events")

    momentum_residuals = []
    dtheta_res = []
    dphi_res = []
    d0_res = []
    z0_res = []
    pt_truth = []
    pt_reco = []
    energy_truth = []
    energy_reco = []
    eta_truth = []
    eta_reco = []
    phi_truth = []
    phi_reco = []

    totals_pt = np.zeros(len(PT_BINS_EFF) - 1)
    selected_pt = np.zeros(len(PT_BINS_EFF) - 1)
    totals_eta = np.zeros(len(ETA_BINS_EFF) - 1)
    selected_eta = np.zeros(len(ETA_BINS_EFF) - 1)
    masses_by_eta_bin = [[] for _ in range(len(ETA_BINS_MASS) - 1)]

    for ievt, event in enumerate(events):
        if args.max_events is not None and ievt >= args.max_events:
            break
        if ievt and ievt % 500 == 0:
            print(f"Processed {ievt} events...")

        sim_to_truth = build_sim_to_truth(event.get("DCHCollection"))
        digi_truth = build_digi_truth_map(event.get("DCH_DigiSimAssociationCollection"), sim_to_truth)
        matches = match_tracks(event, args.track_collection, digi_truth)
        mc_particles = event.get("MCParticles")

        # best match per truth muon
        best_by_truth = {}
        for match in matches:
            tidx = match["truth_index"]
            if tidx is None or tidx >= len(mc_particles):
                continue
            if not is_truth_muon(mc_particles[tidx]):
                continue
            prev = best_by_truth.get(tidx)
            if prev is None or match["purity"] > prev["purity"]:
                best_by_truth[tidx] = match

        # truth muons and efficiency bookkeeping
        truth_muons = []
        for idx, mc in enumerate(mc_particles):
            if not is_truth_muon(mc):
                continue
            truth_vec = mc_particle_to_vec(mc)
            truth_muons.append((idx, mc, truth_vec))

            pt_truth.append(truth_vec["pt"])
            eta_truth.append(truth_vec["eta"])
            phi_truth.append(truth_vec["phi"])
            energy_truth.append(truth_vec["energy"])

            ib_pt = np.digitize(truth_vec["pt"], PT_BINS_EFF) - 1
            ib_eta = np.digitize(truth_vec["eta"], ETA_BINS_EFF) - 1
            if 0 <= ib_pt < len(totals_pt):
                totals_pt[ib_pt] += 1
            if 0 <= ib_eta < len(totals_eta):
                totals_eta[ib_eta] += 1

        # matched reco muons
        matched_muons = []
        for tidx, mc, truth_vec in truth_muons:
            match = best_by_truth.get(tidx)
            if match is None or match["purity"] < args.purity_min:
                continue
            track = match["track"]
            if track.trackStates_size() == 0:
                continue
            state = track.getTrackStates(0)
            reco_vec = trackstate_to_vec(state, args.bfield)
            if reco_vec["pt"] < PT_MIN or abs(reco_vec["eta"]) > ETA_MAX:
                continue

            matched_muons.append({"truth": truth_vec, "reco": reco_vec, "state": state})

            pt_reco.append(reco_vec["pt"])
            eta_reco.append(reco_vec["eta"])
            phi_reco.append(reco_vec["phi"])
            energy_reco.append(reco_vec["energy"])

            mom_res = (reco_vec["p"] - truth_vec["p"]) / truth_vec["p"] if truth_vec["p"] else 0.0
            momentum_residuals.append((truth_vec["pt"], mom_res))
            dtheta_res.append(reco_vec["theta"] - truth_vec["theta"])
            dphi_res.append(delta_phi(reco_vec["phi"], truth_vec["phi"]))

            true_d0 = compute_true_d0(truth_vec["phi"], truth_vec["vx"], truth_vec["vy"])
            d0_res.append(state.D0 - true_d0)
            z0_res.append(state.Z0 - truth_vec["vz"])

            ib_pt = np.digitize(truth_vec["pt"], PT_BINS_EFF) - 1
            ib_eta = np.digitize(truth_vec["eta"], ETA_BINS_EFF) - 1
            if 0 <= ib_pt < len(selected_pt):
                selected_pt[ib_pt] += 1
            if 0 <= ib_eta < len(selected_eta):
                selected_eta[ib_eta] += 1

        # Z mass by |eta| bin
        if len(matched_muons) >= 2:
            for mu1, mu2 in combinations(matched_muons, 2):
                mass = invariant_mass(mu1["reco"], mu2["reco"])
                eta_pair = max(abs(mu1["truth"]["eta"]), abs(mu2["truth"]["eta"]))
                ib = np.digitize(eta_pair, ETA_BINS_MASS) - 1
                if 0 <= ib < len(masses_by_eta_bin):
                    masses_by_eta_bin[ib].append(mass)

    # ------------------------------------------------------------
    # Produce plots
    # ------------------------------------------------------------
    plot_momentum_resolution(
        momentum_residuals,
        PT_BINS_RES,
        os.path.join(args.output_dir, "momentum_resolution_vs_pt.png"),
    )
    plot_angle_resolution(
        dtheta_res, dphi_res, os.path.join(args.output_dir, "angular_resolution.png")
    )
    plot_impact_parameters(
        d0_res, z0_res, os.path.join(args.output_dir, "impact_parameter_resolution.png")
    )
    plot_pt_energy(
        pt_truth,
        pt_reco,
        energy_truth,
        energy_reco,
        os.path.join(args.output_dir, "muon_pt_energy.png"),
    )
    plot_eta_phi(
        eta_truth,
        eta_reco,
        phi_truth,
        phi_reco,
        os.path.join(args.output_dir, "eta_phi_distributions.png"),
    )
    plot_efficiency(
        PT_BINS_EFF,
        totals_pt,
        selected_pt,
        xlabel=r"$p_{T}^{\text{true}}$ [GeV]",
        title="Muon reconstruction efficiency vs pT",
        out_path=os.path.join(args.output_dir, "efficiency_vs_pt.png"),
    )
    plot_efficiency(
        ETA_BINS_EFF,
        totals_eta,
        selected_eta,
        xlabel=r"$\eta^{\text{true}}$",
        title="Muon reconstruction efficiency vs η",
        out_path=os.path.join(args.output_dir, "efficiency_vs_eta.png"),
    )
    plot_mass_width(
        ETA_BINS_MASS,
        masses_by_eta_bin,
        os.path.join(args.output_dir, "mass_width_vs_eta.png"),
    )
    print(f"Done. Plots written to {args.output_dir}/")


if __name__ == "__main__":
    main()
